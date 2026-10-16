import asyncio

from viam.module.module import Module

# Importing the model classes registers them with the SDK's resource registry.
from models.llm import OpenVINOLLM  # noqa: F401
from models.stt import OpenVINOSpeechToText  # noqa: F401
from models.vlm import OpenVINOVLM  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
