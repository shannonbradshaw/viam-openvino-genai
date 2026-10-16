"""Loads an OpenVINO GenAI pipeline off the event loop and tracks its state.

Compiling a generative model takes seconds on the GPU and much longer elsewhere, which can exceed viam-server's
resource construction timeout. The resource is therefore built immediately and the pipeline is compiled in a
background thread; requests made before it is ready fail with a clear "still loading" error, and
DoCommand {"command": "status"} reports progress.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from viam.logging import getLogger

from .config import PipelineConfig

LOGGER = getLogger(__name__)

LOADING, READY, ERROR, CLOSED = "loading", "ready", "error", "closed"


def default_cache_dir() -> Optional[str]:
    data = os.environ.get("VIAM_MODULE_DATA")
    return os.path.join(data, "ov_cache") if data else None


def available_devices() -> list:
    import openvino as ov

    return list(ov.Core().available_devices)


class PipelineLoader:
    """Owns one pipeline. ``factory(model_path, device, **properties)`` builds it, e.g. openvino_genai.LLMPipeline."""

    def __init__(self, name: str, cfg: PipelineConfig, factory: Callable[..., Any]) -> None:
        self.name = name
        self.cfg = cfg
        self._factory = factory
        self._lock = threading.Lock()  # GenAI pipelines are not safe for concurrent generate calls
        self._pipeline: Any = None
        self.state = LOADING
        self.error: Optional[str] = None
        self.device: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self._thread = threading.Thread(target=self._load, name=f"load-{name}", daemon=True)

    def start(self) -> "PipelineLoader":
        self._thread.start()
        return self

    def _properties(self) -> Dict[str, Any]:
        props = dict(self.cfg.extra_config)
        cache = self.cfg.cache_dir if self.cfg.cache_dir is not None else default_cache_dir()
        if cache:
            os.makedirs(cache, exist_ok=True)
            props.setdefault("CACHE_DIR", cache)
        return props

    def _try(self, device: str) -> Any:
        t0 = time.perf_counter()
        pipe = self._factory(self.cfg.model_path, device, **self._properties())
        self.load_seconds = round(time.perf_counter() - t0, 2)
        return pipe

    def _load(self) -> None:
        if not os.path.exists(self.cfg.model_path):
            self._fail(f"model_path {self.cfg.model_path!r} does not exist")
            return
        devices = [self.cfg.device] + ([self.cfg.fallback_device] if self.cfg.fallback_device else [])
        last_err = ""
        for i, dev in enumerate(devices):
            try:
                LOGGER.info(f"{self.name}: compiling {self.cfg.model_path} on {dev} (available: {available_devices()})")
                pipe = self._try(dev)
            except Exception as e:  # noqa: BLE001 - surface any OpenVINO failure in status and logs
                last_err = f"{dev}: {e}"
                if i + 1 < len(devices):
                    LOGGER.warning(f"{self.name}: could not load on {dev}, falling back to {devices[i + 1]}: {e}")
                continue
            with self._lock:
                if self.state == CLOSED:
                    return
                self._pipeline, self.device, self.state = pipe, dev, READY
            LOGGER.info(f"{self.name}: ready on {dev} after {self.load_seconds}s")
            return
        self._fail(last_err)

    def _fail(self, msg: str) -> None:
        self.state, self.error = ERROR, msg
        LOGGER.error(f"{self.name}: failed to load: {msg}")

    def run(self, fn: Callable[[Any], Any]) -> Any:
        """Call ``fn(pipeline)`` while holding the pipeline lock. Raises if the pipeline is not ready."""
        if self.state == LOADING:
            raise RuntimeError(f"{self.name}: model is still loading on {self.cfg.device}; check DoCommand status")
        if self.state != READY:
            raise RuntimeError(f"{self.name}: model is not available ({self.state}): {self.error}")
        with self._lock:
            return fn(self._pipeline)

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "model_path": self.cfg.model_path,
            "requested_device": self.cfg.device,
            "device": self.device or "",
            "load_seconds": self.load_seconds if self.load_seconds is not None else -1,
            "error": self.error or "",
        }

    def close(self) -> None:
        with self._lock:
            self.state, self._pipeline = CLOSED, None
