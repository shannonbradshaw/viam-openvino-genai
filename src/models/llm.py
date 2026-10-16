"""``viam:openvino-genai:llm`` - a local language model (Qwen, Llama, Phi, ...) as a generic service.

DoCommand {"command": "generate", "prompt": "..."} returns the model's reply. With "session": "<id>" the service
keeps that conversation's history so follow-up prompts have context; {"command": "reset", "session": "<id>"}
clears it. Sessions live in memory and are lost when the module restarts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.generic import Generic
from viam.utils import ValueTypes, struct_to_dict

from .config import GenerationOptions, PipelineConfig, _opt_num, _opt_str
from .loader import PipelineLoader

LOGGER = getLogger(__name__)

DEFAULT_GENERATION = {"max_new_tokens": 256, "repetition_penalty": 1.1}


class LLMConfig:
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        self.pipeline = PipelineConfig.from_attributes(attrs)
        self.generation = GenerationOptions.from_attributes(attrs, DEFAULT_GENERATION)
        self.system_prompt = _opt_str(attrs, "system_prompt")
        self.max_history_messages = int(_opt_num(attrs, "max_history_messages", 2, 1000, integer=True) or 20)


class OpenVINOLLM(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model.from_string("viam:openvino-genai:llm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[LLMConfig] = None
        self.loader: Optional[PipelineLoader] = None
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        LLMConfig(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = LLMConfig(struct_to_dict(config.attributes))
        old = self.loader
        if old is None or old.cfg != cfg.pipeline:
            import openvino_genai as ov_genai

            self.loader = PipelineLoader(self.name, cfg.pipeline, ov_genai.LLMPipeline).start()
            if old is not None:
                old.close()
            self.sessions.clear()
        self.cfg = cfg

    def build_messages(self, prompt: str, system: Optional[str], session: Optional[str]) -> List[Dict[str, str]]:
        assert self.cfg is not None
        messages: List[Dict[str, str]] = []
        sys_prompt = system or self.cfg.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if session:
            messages.extend(self.sessions.get(session, []))
        messages.append({"role": "user", "content": prompt})
        return messages

    def remember(self, session: str, prompt: str, reply: str) -> None:
        assert self.cfg is not None
        history = self.sessions.setdefault(session, [])
        history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        del history[: max(0, len(history) - self.cfg.max_history_messages)]

    def _generate_sync(self, messages: List[Dict[str, str]], gen: GenerationOptions) -> Dict[str, Any]:
        assert self.loader is not None
        import openvino_genai as ov_genai

        def go(pipe):
            gc = gen.apply(pipe.get_generation_config())
            t0 = time.perf_counter()
            res = pipe.generate(ov_genai.ChatHistory(messages), generation_config=gc)
            return res, time.perf_counter() - t0

        res, secs = self.loader.run(go)
        m = res.perf_metrics
        return {
            "text": res.texts[0].strip() if res.texts else "",
            "seconds": round(secs, 3),
            "ttft_ms": round(m.get_ttft().mean, 1),
            "tokens_per_second": round(m.get_throughput().mean, 1),
            "device": self.loader.device or "",
        }

    async def generate(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        assert self.cfg is not None
        prompt = command.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("generate needs a non-empty prompt")
        session = command.get("session")
        if session is not None and (not isinstance(session, str) or not session):
            raise ValueError("session must be a non-empty string")
        system = command.get("system")
        messages = self.build_messages(prompt, system if isinstance(system, str) else None, session)
        out = await asyncio.to_thread(self._generate_sync, messages, self.cfg.generation.overridden(command))
        if session:
            self.remember(session, prompt, out["text"])
            out["session"] = session
            out["history_messages"] = len(self.sessions[session])
        return out

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        assert self.loader is not None
        cmd = command.get("command")
        if cmd == "status":
            return {**self.loader.status(), "sessions": sorted(self.sessions)}
        if cmd == "generate":
            return await self.generate(command)
        if cmd == "reset":
            session = command.get("session")
            if session:
                self.sessions.pop(str(session), None)
            else:
                self.sessions.clear()
            return {"reset": str(session or "all")}
        raise ValueError(f"unknown command {cmd!r}; supported: generate, reset, status")

    async def close(self) -> None:
        if self.loader is not None:
            self.loader.close()
