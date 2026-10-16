"""Attribute parsing shared by the three models.

Validation happens in ``validate_config`` so a bad config is rejected when it is saved, with an error that names
the attribute. Nothing here touches OpenVINO, so it is cheap to run and easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# Generative models compile slowly and unevenly on the NPU (a 3B VLM did not finish compiling in 20 minutes on a
# Core Ultra), and OpenVINO's AUTO never picks the NPU anyway. Default to the GPU and fall back to the CPU.
DEFAULT_DEVICE = "GPU"
DEFAULT_FALLBACK_DEVICE = "CPU"
_DEVICE_PREFIXES = ("CPU", "GPU", "NPU", "AUTO", "HETERO", "MULTI")


def _opt_str(attrs: Mapping[str, Any], key: str) -> Optional[str]:
    v = attrs.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{key} must be a non-empty string when set")
    return v.strip()


def _opt_num(attrs: Mapping[str, Any], key: str, lo: float, hi: float, integer: bool = False) -> Optional[float]:
    v = attrs.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number")
    if integer and float(v) != int(v):
        raise ValueError(f"{key} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {v}")
    return int(v) if integer else float(v)


def _device(attrs: Mapping[str, Any], key: str, default: Optional[str]) -> Optional[str]:
    v = attrs.get(key, default)
    if v is None or v == "":
        return None
    if not isinstance(v, str) or not v.strip().upper().startswith(_DEVICE_PREFIXES):
        raise ValueError(f"{key} must be an OpenVINO device such as CPU, GPU, GPU.0 or NPU, got {v!r}")
    return v.strip().upper()


@dataclass
class GenerationOptions:
    """Generation knobs passed straight to openvino_genai.GenerationConfig. Unset fields keep the model's defaults."""

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, Any], defaults: Optional[Mapping[str, Any]] = None) -> "GenerationOptions":
        merged: Dict[str, Any] = dict(defaults or {})
        merged.update({k: v for k, v in attrs.items() if v is not None})
        do_sample = merged.get("do_sample")
        if do_sample is not None and not isinstance(do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        return cls(
            max_new_tokens=_opt_num(merged, "max_new_tokens", 1, 32768, integer=True),  # type: ignore[arg-type]
            temperature=_opt_num(merged, "temperature", 0.0, 10.0),
            top_p=_opt_num(merged, "top_p", 0.0, 1.0),
            top_k=_opt_num(merged, "top_k", 0, 100000, integer=True),  # type: ignore[arg-type]
            do_sample=do_sample,
            repetition_penalty=_opt_num(merged, "repetition_penalty", 0.1, 10.0),
        )

    def overridden(self, request: Mapping[str, Any]) -> "GenerationOptions":
        """Per-request overrides from a DoCommand, validated the same way as config attributes."""
        current = {k: v for k, v in self.__dict__.items() if v is not None}
        current.update({k: request[k] for k in self.__dict__ if k in request})
        return GenerationOptions.from_attributes(current)

    def apply(self, gen_config: Any) -> Any:
        for k, v in self.__dict__.items():
            if v is not None:
                setattr(gen_config, k, v)
        return gen_config


@dataclass
class PipelineConfig:
    """Attributes every model has: where the model is and which device runs it."""

    model_path: str
    device: str = DEFAULT_DEVICE
    fallback_device: Optional[str] = DEFAULT_FALLBACK_DEVICE
    cache_dir: Optional[str] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, Any]) -> "PipelineConfig":
        model_path = _opt_str(attrs, "model_path")
        if not model_path:
            raise ValueError("model_path is required (a folder exported with optimum-cli or downloaded from the OpenVINO Hugging Face org)")
        extra = attrs.get("extra_config") or {}
        if not isinstance(extra, Mapping):
            raise ValueError("extra_config must be an object")
        cache_dir = attrs.get("cache_dir")
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise ValueError("cache_dir must be a string ('' disables the compiled-model cache)")
        device = _device(attrs, "device", DEFAULT_DEVICE)
        fallback = _device(attrs, "fallback_device", DEFAULT_FALLBACK_DEVICE)
        return cls(
            model_path=model_path,
            device=device or DEFAULT_DEVICE,
            fallback_device=None if fallback == device else fallback,
            cache_dir=cache_dir,
            extra_config=dict(extra),
        )


def camera_names(attrs: Mapping[str, Any]) -> List[str]:
    """``camera_name`` (one) and ``camera_names`` (several) both become optional dependencies."""
    names: List[str] = []
    single = _opt_str(attrs, "camera_name")
    if single:
        names.append(single)
    many = attrs.get("camera_names")
    if many is not None:
        if not isinstance(many, list) or not all(isinstance(n, str) and n.strip() for n in many):
            raise ValueError("camera_names must be a list of non-empty strings")
        names.extend(n.strip() for n in many if n.strip() not in names)
    return names
