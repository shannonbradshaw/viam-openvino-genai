import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viam.proto.app.robot import ComponentConfig  # noqa: E402
from viam.utils import dict_to_struct  # noqa: E402


def make_config(name: str, attrs: Mapping[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(dict(attrs)))


class _Metric:
    def __init__(self, v):
        self.mean = v


class FakeResult:
    def __init__(self, text: str):
        self.texts = [text]
        self.chunks = []
        self.perf_metrics = SimpleNamespace(get_ttft=lambda: _Metric(12.0), get_throughput=lambda: _Metric(50.0))


class FakePipeline:
    """Stands in for openvino_genai.{Whisper,VLM,LLM}Pipeline. Records calls; fails on devices in ``fail_on``."""

    fail_on: tuple = ()
    reply = "ok"

    def __init__(self, model_path: str, device: str, **props):
        if device in self.fail_on:
            raise RuntimeError(f"{device} unavailable")
        self.model_path, self.device, self.props = model_path, device, props
        self.calls: list = []

    def get_generation_config(self):
        return SimpleNamespace()

    def generate(self, inputs, *args, **kwargs):
        self.calls.append((inputs, args, kwargs))
        return FakeResult(self.reply)


@pytest.fixture
def fake_genai(monkeypatch, tmp_path):
    import openvino_genai

    class Fake(FakePipeline):
        pass

    for cls in ("WhisperPipeline", "VLMPipeline", "LLMPipeline"):
        monkeypatch.setattr(openvino_genai, cls, Fake)
    monkeypatch.setattr("models.loader.available_devices", lambda: ["CPU", "GPU"])
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return SimpleNamespace(cls=Fake, model_path=str(model_dir))


def wait_ready(loader, timeout=5.0):
    t0 = time.time()
    while loader.state == "loading" and time.time() - t0 < timeout:
        time.sleep(0.01)
    return loader.state
