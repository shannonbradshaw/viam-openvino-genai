import pytest

from models.config import GenerationOptions, PipelineConfig, camera_names
from models.llm import LLMConfig
from models.stt import SpeechConfig
from models.vlm import VLMConfig


def test_model_path_required():
    with pytest.raises(ValueError, match="model_path"):
        PipelineConfig.from_attributes({})


def test_device_defaults_to_gpu_with_cpu_fallback():
    cfg = PipelineConfig.from_attributes({"model_path": "/m"})
    assert (cfg.device, cfg.fallback_device) == ("GPU", "CPU")


def test_fallback_disabled_when_same_or_empty():
    assert PipelineConfig.from_attributes({"model_path": "/m", "device": "cpu"}).fallback_device is None
    assert PipelineConfig.from_attributes({"model_path": "/m", "fallback_device": ""}).fallback_device is None


@pytest.mark.parametrize("bad", ["TPU", 3, "cuda"])
def test_bad_device(bad):
    with pytest.raises(ValueError, match="device"):
        PipelineConfig.from_attributes({"model_path": "/m", "device": bad})


def test_generation_ranges():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationOptions.from_attributes({"max_new_tokens": 0})
    with pytest.raises(ValueError, match="top_p"):
        GenerationOptions.from_attributes({"top_p": 2})
    with pytest.raises(ValueError, match="do_sample"):
        GenerationOptions.from_attributes({"do_sample": "yes"})


def test_generation_overrides_validated_and_merged():
    base = GenerationOptions.from_attributes({}, {"max_new_tokens": 128, "repetition_penalty": 1.1})
    o = base.overridden({"max_new_tokens": 10, "prompt": "ignored"})
    assert (o.max_new_tokens, o.repetition_penalty) == (10, 1.1)
    with pytest.raises(ValueError):
        base.overridden({"temperature": -1})


def test_vlm_defaults_include_repetition_penalty():
    cfg = VLMConfig({"model_path": "/m"})
    assert cfg.generation.repetition_penalty == 1.1 and cfg.generation.max_new_tokens == 128


def test_camera_names_merge_and_validate():
    assert camera_names({"camera_name": "a", "camera_names": ["b", "a"]}) == ["a", "b"]
    with pytest.raises(ValueError, match="camera_names"):
        camera_names({"camera_names": "a"})


def test_speech_target_requires_mic():
    with pytest.raises(ValueError, match="mic"):
        SpeechConfig({"model_path": "/m", "transcript_target": "t"})
    assert SpeechConfig({"model_path": "/m", "mic": "m", "transcript_target": "t"}).dependencies() == ["m", "t"]
    with pytest.raises(ValueError, match="task"):
        SpeechConfig({"model_path": "/m", "task": "summarize"})


def test_llm_history_bounds():
    with pytest.raises(ValueError, match="max_history_messages"):
        LLMConfig({"model_path": "/m", "max_history_messages": 1})
