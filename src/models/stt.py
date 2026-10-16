"""``viam:openvino-genai:speech-to-text`` - Whisper speech recognition on Intel CPU/GPU/NPU.

Two ways in:
* DoCommand {"command": "transcribe", ...} with an audio file (base64 or a path on the machine) or raw pcm16.
* Optional streaming: with ``mic`` (an audio_in component) and ``transcript_target`` set, the service listens to the
  mic, cuts the stream at the empty-chunk segment markers a wake-word filter emits, transcribes each segment and
  calls ``transcript_target.DoCommand({"command": "deliverTranscript", ...})``. This is the same contract Viam's
  ``viam:speech-to-text`` cloud STT models use, so the two are interchangeable in a voice pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.audio_in import AudioIn
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.generic import Generic
from viam.utils import ValueTypes, struct_to_dict

from . import audio
from .config import PipelineConfig, _opt_num, _opt_str
from .loader import PipelineLoader

LOGGER = getLogger(__name__)
SOURCE = "openvino-whisper"


def _whisper_language(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.strip()
    return lang if lang.startswith("<|") else f"<|{lang.lower()}|>"


class SpeechConfig:
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        self.pipeline = PipelineConfig.from_attributes(attrs)
        self.language = _opt_str(attrs, "language")
        task = attrs.get("task", "transcribe")
        if task not in ("transcribe", "translate"):
            raise ValueError("task must be 'transcribe' or 'translate'")
        self.task = task
        self.mic = _opt_str(attrs, "mic")
        self.transcript_target = _opt_str(attrs, "transcript_target")
        if self.transcript_target and not self.mic:
            raise ValueError("transcript_target needs mic: transcripts are only delivered for audio streamed from a mic")
        self.max_segment_seconds = _opt_num(attrs, "max_segment_seconds", 1, 600) or 30.0

    def dependencies(self) -> List[str]:
        return [n for n in (self.mic, self.transcript_target) if n]


class OpenVINOSpeechToText(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model.from_string("viam:openvino-genai:speech-to-text")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[SpeechConfig] = None
        self.loader: Optional[PipelineLoader] = None
        self.mic: Optional[AudioIn] = None
        self.target: Optional[ResourceBase] = None
        self._listener: Optional[asyncio.Task] = None
        self.last_transcript = ""
        self.segments = 0

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        cfg = SpeechConfig(struct_to_dict(config.attributes))
        return cfg.dependencies(), []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = SpeechConfig(struct_to_dict(config.attributes))
        self._stop_listener()
        old = self.loader
        if old is None or old.cfg != cfg.pipeline:
            import openvino_genai as ov_genai

            self.loader = PipelineLoader(self.name, cfg.pipeline, ov_genai.WhisperPipeline).start()
            if old is not None:
                old.close()
        self.cfg = cfg
        by_name = {rn.name: res for rn, res in dependencies.items()}
        self.mic = by_name.get(cfg.mic) if cfg.mic else None  # type: ignore[assignment]
        self.target = by_name.get(cfg.transcript_target) if cfg.transcript_target else None
        if self.mic is not None:
            self._listener = asyncio.get_running_loop().create_task(self._listen())

    # -- transcription -------------------------------------------------------------------------------------------

    def _transcribe_sync(self, samples, request: Mapping[str, Any]) -> Dict[str, Any]:
        assert self.cfg is not None and self.loader is not None
        language = _whisper_language(request.get("language") or self.cfg.language)  # type: ignore[arg-type]
        task = request.get("task") or self.cfg.task
        timestamps = bool(request.get("return_timestamps", False))

        def go(pipe):
            gc = pipe.get_generation_config()
            if language:
                gc.language = language
            gc.task = task
            gc.return_timestamps = timestamps
            t0 = time.perf_counter()
            res = pipe.generate(samples.tolist(), gc)
            return res, (time.perf_counter() - t0) * 1000

        res, ms = self.loader.run(go)
        out: Dict[str, Any] = {
            "text": res.texts[0].strip() if res.texts else "",
            "audio_seconds": round(audio.duration_seconds(samples), 3),
            "inference_ms": round(ms, 1),
            "device": self.loader.device or "",
        }
        if timestamps and getattr(res, "chunks", None):
            out["chunks"] = [{"start": c.start_ts, "end": c.end_ts, "text": c.text.strip()} for c in res.chunks]
        return out

    async def transcribe(self, samples, request: Mapping[str, Any]) -> Dict[str, Any]:
        if audio.duration_seconds(samples) == 0:
            raise ValueError("audio is empty")
        return await asyncio.to_thread(self._transcribe_sync, samples, request)

    # -- mic streaming -------------------------------------------------------------------------------------------

    async def _flush(self, chunks: List[bytes], sample_rate: int, channels: int) -> None:
        if not chunks:
            return
        samples = audio.decode_pcm16(b"".join(chunks), sample_rate, channels)
        result = await self.transcribe(samples, {})
        text = result["text"]
        self.segments += 1
        self.last_transcript = text
        LOGGER.info(f"{self.name}: transcript ({result['audio_seconds']}s audio, {result['inference_ms']} ms): {text!r}")
        if self.target is not None and text:
            await self.target.do_command(  # type: ignore[attr-defined]
                {"command": "deliverTranscript", "transcript": text, "is_final": True, "confidence": 1.0, "source": SOURCE}
            )

    async def _listen(self) -> None:
        assert self.cfg is not None
        backoff = 1.0
        while True:
            chunks: List[bytes] = []
            seconds, sample_rate, channels = 0.0, 16000, 1
            try:
                stream = await self.mic.get_audio(codec="pcm16", duration_seconds=0, previous_timestamp_ns=0)  # type: ignore[union-attr]
                backoff = 1.0
                async for resp in stream:
                    chunk = resp.audio
                    if not chunk.audio_data:  # segment-end marker from a wake-word filter
                        await self._flush(chunks, sample_rate, channels)
                        chunks, seconds = [], 0.0
                        continue
                    if chunk.HasField("audio_info"):
                        sample_rate = chunk.audio_info.sample_rate_hz or sample_rate
                        channels = chunk.audio_info.num_channels or channels
                    chunks.append(chunk.audio_data)
                    seconds += len(chunk.audio_data) / (2 * channels * sample_rate)
                    if seconds >= self.cfg.max_segment_seconds:
                        await self._flush(chunks, sample_rate, channels)
                        chunks, seconds = [], 0.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep listening through mic restarts and transient errors
                LOGGER.warning(f"{self.name}: mic stream error, retrying in {backoff:.0f}s: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _stop_listener(self) -> None:
        if self._listener is not None:
            self._listener.cancel()
            self._listener = None

    # -- API -----------------------------------------------------------------------------------------------------

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        assert self.loader is not None and self.cfg is not None
        cmd = command.get("command")
        if cmd == "status":
            return {
                **self.loader.status(),
                "listening": self._listener is not None and not self._listener.done(),
                "mic": self.cfg.mic or "",
                "transcript_target": self.cfg.transcript_target or "",
                "segments": self.segments,
                "last_transcript": self.last_transcript,
            }
        if cmd == "transcribe":
            return await self.transcribe(_samples_from_request(command), command)
        raise ValueError(f"unknown command {cmd!r}; supported: transcribe, status")

    async def close(self) -> None:
        self._stop_listener()
        if self.loader is not None:
            self.loader.close()


def _samples_from_request(command: Mapping[str, Any]):
    given = [k for k in ("audio_b64", "audio_path", "pcm16_b64") if command.get(k)]
    if len(given) != 1:
        raise ValueError("transcribe needs exactly one of audio_b64 (an audio file), audio_path (a file on the machine) or pcm16_b64")
    if given[0] == "audio_path":
        return audio.decode_file(str(command["audio_path"]))
    data = base64.b64decode(str(command[given[0]]))
    if given[0] == "audio_b64":
        return audio.decode_file_bytes(data)
    sr = command.get("sample_rate")
    if not isinstance(sr, (int, float)):
        raise ValueError("pcm16_b64 needs sample_rate")
    return audio.decode_pcm16(data, int(sr), int(command.get("num_channels", 1)))  # type: ignore[arg-type]
