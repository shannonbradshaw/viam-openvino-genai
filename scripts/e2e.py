"""End-to-end check with real OpenVINO GenAI models, driving the resource classes directly (no viam-server).

    .venv/bin/python scripts/e2e.py --whisper DIR --audio FILE --vlm DIR --images DIR --llm DIR [--device GPU]
"""

import argparse
import asyncio
import base64
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from PIL import Image  # noqa: E402
from viam.media.video import NamedImage  # noqa: E402
from viam.proto.common import ResourceName  # noqa: E402

from conftest import make_config  # noqa: E402
from models.llm import OpenVINOLLM  # noqa: E402
from models.stt import OpenVINOSpeechToText  # noqa: E402
from models.vlm import OpenVINOVLM  # noqa: E402


class FileRGBDCamera:
    """An RGBD-shaped camera backed by an image file: returns a depth frame first, then color."""

    def __init__(self, path):
        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, format="JPEG")
        self.jpeg = buf.getvalue()

    async def get_images(self, **kwargs):
        return [NamedImage("depth", b"\x00" * 16, "image/vnd.viam.dep"), NamedImage("color", self.jpeg, "image/jpeg")], None


async def wait(res, timeout=900):
    t0 = time.time()
    while True:
        st = await res.do_command({"command": "status"})
        if st["state"] != "loading" or time.time() - t0 > timeout:
            return st
        await asyncio.sleep(0.5)


def show(title, d, keys):
    print(f"  {title}: " + ", ".join(f"{k}={d[k]!r}" for k in keys if k in d))


async def main(a):
    print("== speech-to-text")
    stt = OpenVINOSpeechToText.new(make_config("stt", {"model_path": a.whisper, "device": a.device, "language": "en"}), {})
    show("status", await wait(stt), ["state", "device", "load_seconds", "error"])
    out = await stt.do_command({"command": "transcribe", "audio_path": a.audio})
    show("audio_path", out, ["text", "audio_seconds", "inference_ms", "device"])
    with open(a.audio, "rb") as f:
        out = await stt.do_command({"command": "transcribe", "audio_b64": base64.b64encode(f.read()).decode(), "return_timestamps": True})
    show("audio_b64+timestamps", out, ["inference_ms", "chunks"])
    await stt.close()

    print("== speech-to-text fallback (device GPU.7 does not exist)")
    stt2 = OpenVINOSpeechToText.new(make_config("stt2", {"model_path": a.whisper, "device": "GPU.7"}), {})
    show("status", await wait(stt2), ["state", "requested_device", "device", "load_seconds"])
    await stt2.close()

    print("== vlm")
    cams = {ResourceName(namespace="rdk", type="component", subtype="camera", name="realsense"): FileRGBDCamera(os.path.join(a.images, "office.jpg"))}
    vlm = OpenVINOVLM.new(make_config("vlm", {"model_path": a.vlm, "device": a.device, "camera_name": "realsense"}), cams)
    show("status", await wait(vlm), ["state", "device", "load_seconds", "error"])
    with open(os.path.join(a.images, "bus.jpg"), "rb") as f:
        bus = base64.b64encode(f.read()).decode()
    out = await vlm.do_command({"command": "ask", "question": "Describe this image in one sentence.", "image_b64": bus})
    show("ask bus (image_b64)", out, ["answer", "seconds", "ttft_ms", "tokens_per_second"])
    out = await vlm.do_command({"command": "ask", "question": "Is anyone sitting at a desk? Answer yes or no, then explain briefly."})
    show("ask office (camera)", out, ["answer", "seconds", "ttft_ms", "tokens_per_second"])
    out = await vlm.do_command({"command": "ask", "question": "List the objects you can see, comma separated."})
    show("list objects (repetition_penalty 1.1)", out, ["answer", "seconds"])
    cls = await vlm.get_classifications_from_camera("realsense", 1)
    print(f"  GetClassificationsFromCamera: {cls[0].class_name!r} ({cls[0].confidence})")
    await vlm.close()

    print("== llm")
    llm = OpenVINOLLM.new(make_config("llm", {"model_path": a.llm, "device": a.device, "system_prompt": "You are a helpful robot assistant. Answer in one or two sentences."}), {})
    show("status", await wait(llm), ["state", "device", "load_seconds", "error"])
    out = await llm.do_command({"command": "generate", "prompt": "My robot arm has six joints. What is a gripper?", "session": "demo"})
    show("turn 1", out, ["text", "seconds", "ttft_ms", "tokens_per_second", "history_messages"])
    out = await llm.do_command({"command": "generate", "prompt": "How many joints did I say my arm has?", "session": "demo"})
    show("turn 2 (needs history)", out, ["text", "seconds", "history_messages"])
    out = await llm.do_command({"command": "generate", "prompt": "How many joints did I say my arm has?"})
    show("no session (no history)", out, ["text"])
    await llm.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("whisper", "audio", "vlm", "images", "llm"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--device", default="GPU")
    asyncio.run(main(p.parse_args()))
