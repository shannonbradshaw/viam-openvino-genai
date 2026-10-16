"""``viam:openvino-genai:vlm`` - a vision-language model (Qwen2.5-VL, Phi-3.5-vision, ...) as a vision service.

* DoCommand {"command": "ask", "question": "...", "camera_name": "..."} answers a free-form question about the
  camera's current frame (or an ``image_b64`` sent in the request).
* GetClassifications / GetClassificationsFromCamera return one classification whose class_name is the model's
  answer to the configured ``prompt`` (by default a one-sentence caption), so the service plugs into anything
  that consumes classifications, including data capture and the CONTROL tab.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

from .config import GenerationOptions, PipelineConfig, _opt_num, _opt_str, camera_names
from .loader import PipelineLoader

LOGGER = getLogger(__name__)

DEFAULT_PROMPT = "Describe this image in one sentence."
# Qwen2.5-VL fell into loops ("office chair, office desk, office chair, ...") on list-style prompts without this.
DEFAULT_GENERATION = {"max_new_tokens": 128, "repetition_penalty": 1.1}


class VLMConfig:
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        self.pipeline = PipelineConfig.from_attributes(attrs)
        self.generation = GenerationOptions.from_attributes(attrs, DEFAULT_GENERATION)
        self.prompt = _opt_str(attrs, "prompt") or DEFAULT_PROMPT
        self.cameras = camera_names(attrs)
        self.max_image_size = _opt_num(attrs, "max_image_size", 64, 8192, integer=True)


def _pil_to_tensor(img: Image.Image, max_size: Optional[int]):
    import openvino as ov

    img = img.convert("RGB")
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    return ov.Tensor(np.ascontiguousarray(np.array(img, dtype=np.uint8)[None]))


def _is_color(mime: str, source: str) -> bool:
    s = source.lower()
    if "depth" in s or "dep" in mime:
        return False
    return mime.startswith("image/") or "color" in s or "rgb" in s


class OpenVINOVLM(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model.from_string("viam:openvino-genai:vlm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[VLMConfig] = None
        self.loader: Optional[PipelineLoader] = None
        self.cameras: Dict[str, Camera] = {}

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        cfg = VLMConfig(struct_to_dict(config.attributes))
        # Optional so the model still loads (and DoCommand ask with image_b64 still works) if a camera is missing.
        return [], cfg.cameras

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = VLMConfig(struct_to_dict(config.attributes))
        old = self.loader
        if old is None or old.cfg != cfg.pipeline:
            import openvino_genai as ov_genai

            self.loader = PipelineLoader(self.name, cfg.pipeline, ov_genai.VLMPipeline).start()
            if old is not None:
                old.close()
        self.cfg = cfg
        by_name = {rn.name: res for rn, res in dependencies.items()}
        self.cameras = {n: by_name[n] for n in cfg.cameras if n in by_name}  # type: ignore[misc]

    # -- core ----------------------------------------------------------------------------------------------------

    def _ask_sync(self, img: Image.Image, question: str, gen: GenerationOptions) -> Dict[str, Any]:
        assert self.cfg is not None and self.loader is not None
        tensor = _pil_to_tensor(img, self.cfg.max_image_size)

        def go(pipe):
            gc = gen.apply(pipe.get_generation_config())
            t0 = time.perf_counter()
            res = pipe.generate(question, images=[tensor], generation_config=gc)
            return res, time.perf_counter() - t0

        res, secs = self.loader.run(go)
        m = res.perf_metrics
        return {
            "answer": res.texts[0].strip() if res.texts else "",
            "seconds": round(secs, 3),
            "ttft_ms": round(m.get_ttft().mean, 1),
            "tokens_per_second": round(m.get_throughput().mean, 1),
            "device": self.loader.device or "",
        }

    async def ask(self, img: Image.Image, question: str, request: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        assert self.cfg is not None
        gen = self.cfg.generation.overridden(request or {})
        return await asyncio.to_thread(self._ask_sync, img, question, gen)

    async def _camera_image(self, camera_name: Optional[str]) -> Image.Image:
        name = camera_name or (self.cfg.cameras[0] if self.cfg and self.cfg.cameras else "")
        if not name:
            raise ValueError(f"vlm {self.name!r}: no camera_name given and none configured")
        cam = self.cameras.get(name)
        if cam is None:
            raise ValueError(f"vlm {self.name!r}: camera {name!r} is not configured; add it to camera_name or camera_names")
        images, _ = await cam.get_images()
        # Pick the color frame explicitly: an RGBD camera such as a RealSense also returns a depth image.
        color = [i for i in images if _is_color(i.mime_type, getattr(i, "name", "") or "")]
        if not color:
            raise RuntimeError(f"camera {name!r} returned no color image")
        return viam_to_pil_image(ViamImage(color[0].data, color[0].mime_type))

    def _caption(self, result: Dict[str, Any]) -> List[Classification]:
        return [Classification(class_name=result["answer"], confidence=1.0)]

    # -- vision API ----------------------------------------------------------------------------------------------

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None, **kwargs) -> List[Classification]:
        assert self.cfg is not None
        prompt = str((extra or {}).get("question") or self.cfg.prompt)
        return self._caption(await self.ask(await self._camera_image(camera_name), prompt, extra))

    async def get_classifications(self, image: ViamImage, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None, **kwargs) -> List[Classification]:
        assert self.cfg is not None
        prompt = str((extra or {}).get("question") or self.cfg.prompt)
        return self._caption(await self.ask(viam_to_pil_image(image), prompt, extra))

    async def capture_all_from_camera(self, camera_name: str, return_image: bool = False, return_classifications: bool = False, return_detections: bool = False, return_object_point_clouds: bool = False, return_detections_3d: bool = False, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None, **kwargs) -> CaptureAllResult:
        assert self.cfg is not None
        img = await self._camera_image(camera_name)
        result = CaptureAllResult()
        if return_image:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            result.image = ViamImage(buf.getvalue(), "image/jpeg")  # type: ignore[arg-type]
        if return_classifications:
            prompt = str((extra or {}).get("question") or self.cfg.prompt)
            result.classifications = self._caption(await self.ask(img, prompt, extra))
        return result

    async def get_properties(self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None, **kwargs) -> GetPropertiesResponse:
        return GetPropertiesResponse(classifications_supported=True, detections_supported=False, object_point_clouds_supported=False)

    async def get_detections_from_camera(self, camera_name: str, *, extra=None, timeout=None, **kwargs) -> List[Detection]:
        raise NotImplementedError("vlm returns classifications (answers), not detections")

    async def get_detections(self, image: ViamImage, *, extra=None, timeout=None, **kwargs) -> List[Detection]:
        raise NotImplementedError("vlm returns classifications (answers), not detections")

    async def get_object_point_clouds(self, camera_name: str, *, extra=None, timeout=None, **kwargs) -> List[PointCloudObject]:
        raise NotImplementedError("vlm does not produce point clouds")

    async def get_detections_3d(self, camera_name: str, *, extra=None, timeout=None, **kwargs) -> list:
        raise NotImplementedError("vlm does not produce 3D detections")

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        assert self.loader is not None and self.cfg is not None
        cmd = command.get("command")
        if cmd == "status":
            return {**self.loader.status(), "cameras": list(self.cfg.cameras), "prompt": self.cfg.prompt}
        if cmd == "ask":
            question = command.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("ask needs a non-empty question")
            if command.get("image_b64"):
                img = Image.open(io.BytesIO(base64.b64decode(str(command["image_b64"]))))
            else:
                img = await self._camera_image(command.get("camera_name"))  # type: ignore[arg-type]
            return await self.ask(img, question.strip(), command)
        raise ValueError(f"unknown command {cmd!r}; supported: ask, status")

    async def close(self) -> None:
        if self.loader is not None:
            self.loader.close()
