import asyncio
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from viam.media.video import NamedImage
from viam.proto.common import ResourceName

from conftest import make_config, wait_ready
from models.llm import OpenVINOLLM
from models.loader import PipelineLoader
from models.config import PipelineConfig
from models.stt import OpenVINOSpeechToText, _samples_from_request
from models.vlm import OpenVINOVLM


def _rn(name, subtype="camera"):
    return ResourceName(namespace="rdk", type="component", subtype=subtype, name=name)


async def test_loader_falls_back_to_cpu(fake_genai):
    fake_genai.cls.fail_on = ("GPU",)
    loader = PipelineLoader("x", PipelineConfig.from_attributes({"model_path": fake_genai.model_path}), fake_genai.cls).start()
    assert wait_ready(loader) == "ready"
    assert loader.status()["device"] == "CPU"
    assert "CACHE_DIR" in loader._pipeline.props


async def test_loader_reports_missing_model(fake_genai):
    loader = PipelineLoader("x", PipelineConfig.from_attributes({"model_path": "/nope"}), fake_genai.cls).start()
    assert wait_ready(loader) == "error"
    with pytest.raises(RuntimeError, match="not available"):
        loader.run(lambda p: p)


async def test_loader_rejects_requests_while_loading(fake_genai):
    loader = PipelineLoader("x", PipelineConfig.from_attributes({"model_path": fake_genai.model_path}), fake_genai.cls)
    with pytest.raises(RuntimeError, match="still loading"):
        loader.run(lambda p: p)


async def test_llm_sessions_keep_history(fake_genai):
    llm = OpenVINOLLM.new(make_config("llm", {"model_path": fake_genai.model_path, "system_prompt": "be brief"}), {})
    wait_ready(llm.loader)
    fake_genai.cls.reply = "hello"
    out = await llm.do_command({"command": "generate", "prompt": "hi", "session": "s1", "max_new_tokens": 5})
    assert out["text"] == "hello" and out["history_messages"] == 2
    msgs = llm.build_messages("again", None, "s1")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    await llm.do_command({"command": "reset", "session": "s1"})
    assert (await llm.do_command({"command": "status"}))["sessions"] == []
    with pytest.raises(ValueError, match="prompt"):
        await llm.do_command({"command": "generate", "prompt": " "})
    with pytest.raises(ValueError, match="unknown command"):
        await llm.do_command({"command": "nope"})


async def test_llm_history_is_trimmed(fake_genai):
    llm = OpenVINOLLM.new(make_config("llm", {"model_path": fake_genai.model_path, "max_history_messages": 4}), {})
    wait_ready(llm.loader)
    for i in range(5):
        await llm.do_command({"command": "generate", "prompt": f"p{i}", "session": "s"})
    assert len(llm.sessions["s"]) == 4 and llm.sessions["s"][0]["content"] == "p3"


class FakeRGBDCamera:
    """Returns depth first, like an RGBD camera may, to check the VLM picks the color frame."""

    def __init__(self):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (255, 0, 0)).save(buf, format="JPEG")
        self.jpeg = buf.getvalue()

    async def get_images(self, **kwargs):
        return [NamedImage("depth", b"\x00" * 10, "image/vnd.viam.dep"), NamedImage("color", self.jpeg, "image/jpeg")], None


async def test_vlm_ask_uses_color_frame_and_classifications(fake_genai):
    fake_genai.cls.reply = "a red square"
    cam = FakeRGBDCamera()
    vlm = OpenVINOVLM.new(make_config("vlm", {"model_path": fake_genai.model_path, "camera_names": ["rs"]}), {_rn("rs"): cam})
    wait_ready(vlm.loader)
    out = await vlm.do_command({"command": "ask", "question": "what color?"})
    assert out["answer"] == "a red square"
    tensor = vlm.loader._pipeline.calls[-1][1] if vlm.loader._pipeline.calls[-1][1] else vlm.loader._pipeline.calls[-1][2]["images"]
    assert tensor  # an image was passed
    cls = await vlm.get_classifications_from_camera("rs", 1)
    assert cls[0].class_name == "a red square" and cls[0].confidence == 1.0
    gc = vlm.loader._pipeline.calls[-1][2]["generation_config"]
    assert gc.repetition_penalty == 1.1 and gc.max_new_tokens == 128
    with pytest.raises(ValueError, match="not configured"):
        await vlm.do_command({"command": "ask", "question": "q", "camera_name": "other"})


async def test_vlm_ask_with_image_b64(fake_genai):
    vlm = OpenVINOVLM.new(make_config("vlm", {"model_path": fake_genai.model_path}), {})
    wait_ready(vlm.loader)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    out = await vlm.do_command({"command": "ask", "question": "q", "image_b64": base64.b64encode(buf.getvalue()).decode(), "max_new_tokens": 7})
    assert out["answer"] == fake_genai.cls.reply
    assert vlm.loader._pipeline.calls[-1][2]["generation_config"].max_new_tokens == 7


def _wav_b64(seconds=0.5, sr=8000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_audio_inputs_resampled_to_16k():
    assert len(_samples_from_request({"audio_b64": _wav_b64(0.5, 8000)})) == 8000
    pcm = (np.zeros(4800, dtype="<i2")).tobytes()  # 0.1 s stereo at 24 kHz
    assert len(_samples_from_request({"pcm16_b64": base64.b64encode(pcm).decode(), "sample_rate": 24000, "num_channels": 2})) == 1600
    with pytest.raises(ValueError, match="exactly one"):
        _samples_from_request({})
    with pytest.raises(ValueError, match="sample_rate"):
        _samples_from_request({"pcm16_b64": "AAAA"})
    with pytest.raises(ValueError, match="decode"):
        _samples_from_request({"audio_b64": base64.b64encode(b"not audio").decode()})


async def test_stt_transcribe_command(fake_genai):
    fake_genai.cls.reply = " hello world "
    stt = OpenVINOSpeechToText.new(make_config("stt", {"model_path": fake_genai.model_path, "language": "en"}), {})
    wait_ready(stt.loader)
    out = await stt.do_command({"command": "transcribe", "audio_b64": _wav_b64()})
    assert out["text"] == "hello world" and out["audio_seconds"] == 0.5


class FakeMic:
    def __init__(self, segments):
        self.segments = segments

    async def get_audio(self, **kwargs):
        segs = self.segments

        async def gen():
            info = SimpleNamespace(sample_rate_hz=16000, num_channels=1)
            for seg in segs:
                for data in seg + [b""]:
                    chunk = SimpleNamespace(audio_data=data, audio_info=info, HasField=lambda f: True)
                    yield SimpleNamespace(audio=chunk)
            await asyncio.sleep(3600)

        return gen()


class FakeTarget:
    def __init__(self):
        self.got = []

    async def do_command(self, cmd, **kwargs):
        self.got.append(cmd)
        return {}


async def test_stt_mic_segments_delivered(fake_genai):
    fake_genai.cls.reply = "turn on the lights"
    one_tenth = np.zeros(1600, dtype="<i2").tobytes()
    mic, target = FakeMic([[one_tenth, one_tenth], [one_tenth]]), FakeTarget()
    cfg = make_config("stt", {"model_path": fake_genai.model_path, "mic": "mic", "transcript_target": "router"})
    stt = OpenVINOSpeechToText.new(cfg, {_rn("mic", "audio_in"): mic, ResourceName(namespace="rdk", type="service", subtype="generic", name="router"): target})
    wait_ready(stt.loader)
    for _ in range(100):
        if len(target.got) == 2:
            break
        await asyncio.sleep(0.02)
    assert [g["transcript"] for g in target.got] == ["turn on the lights"] * 2
    assert target.got[0]["command"] == "deliverTranscript" and target.got[0]["is_final"] is True
    status = await stt.do_command({"command": "status"})
    assert status["segments"] == 2 and status["listening"] is True
    await stt.close()
