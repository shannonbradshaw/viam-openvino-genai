# viam-openvino-genai

Generative AI on Intel hardware for Viam machines, using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai):
speech-to-text with Whisper, vision-language models that answer questions about a camera's view, and local LLMs. Everything
runs on the machine; nothing is sent to a cloud service.

> **Module ID is a placeholder.** `meta.json` uses `viam:openvino-genai`. Which namespace publishes this module has not been
> decided yet; change `module_id` before uploading.

| Model | API | What it does |
|---|---|---|
| [`viam:openvino-genai:speech-to-text`](#viamopenvino-genaispeech-to-text) | `rdk:service:generic` | Whisper transcription of audio files, or streaming from an `audio_in` mic. |
| [`viam:openvino-genai:vlm`](#viamopenvino-genaivlm) | `rdk:service:vision` | Ask a vision-language model (Qwen2.5-VL, ...) about a camera frame; captions as classifications. |
| [`viam:openvino-genai:llm`](#viamopenvino-genaillm) | `rdk:service:generic` | Text generation with optional multi-turn sessions. |

Supported platform: `linux/amd64` with an Intel CPU, and optionally an Intel GPU or NPU (with drivers installed on the host).

## Models (the AI kind)

Each service loads one model folder in OpenVINO format, given as `model_path`. Get one of two ways:

* Download a pre-converted model from the [OpenVINO organization on Hugging Face](https://huggingface.co/OpenVINO), e.g.
  `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov`, `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`.
* Export one with Optimum Intel:
  `optimum-cli export openvino --model openai/whisper-base --weight-format fp16 whisper-base-fp16`.
  For Qwen2.5-VL, `optimum-intel` currently needs `transformers<5`.

To ship a model with a machine's config, upload the folder as a Viam ML model package and set
`"model_path": "${packages.ml_model.<name>}"`.

## Common attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | required | Folder containing the OpenVINO model (`openvino_*.xml`/`.bin`, tokenizer files). |
| `device` | string | `"GPU"` | `CPU`, `GPU`, `GPU.0`, `NPU`, ... |
| `fallback_device` | string | `"CPU"` | Used if `device` fails to load; logged as a warning. `""` disables fallback. |
| `cache_dir` | string | `$VIAM_MODULE_DATA/ov_cache` | Compiled-model cache; makes later starts faster. `""` disables. |
| `extra_config` | object | `{}` | Extra OpenVINO properties passed to the pipeline. |

The GPU is the default because it was the fastest device for every generative model we measured, and the NPU compiles large
generative models very slowly (a 3B VLM did not finish compiling in 20 minutes on a Core Ultra X7 358H).

Models are compiled in a background thread so a slow compile does not hit viam-server's resource-construction timeout.
Until loading finishes, requests fail with "model is still loading"; `{"command": "status"}` reports `state`
(`loading`, `ready`, `error`), the device actually used, and the load time.

Generation knobs, settable as attributes (defaults) and per request (overrides): `max_new_tokens`, `temperature`, `top_p`,
`top_k`, `do_sample`, `repetition_penalty`.

## `viam:openvino-genai:speech-to-text`

```json
{ "model_path": "/path/to/whisper-base-fp16", "language": "en" }
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `language` | string | auto-detect | `en`, `es`, ... (or Whisper's `<\|en\|>` form). |
| `task` | string | `transcribe` | `transcribe` or `translate` (to English). |
| `mic` | string | | Optional `audio_in` component to listen to. |
| `transcript_target` | string | | Generic resource that receives each transcript (needs `mic`). |
| `max_segment_seconds` | number | `30` | Transcribe at least this often while audio keeps arriving without a segment marker. |

DoCommand:

```json
{"command": "transcribe", "audio_path": "/home/me/clip.flac"}
{"command": "transcribe", "audio_b64": "<base64 wav/flac/ogg/mp3>", "return_timestamps": true}
{"command": "transcribe", "pcm16_b64": "<base64 raw pcm16>", "sample_rate": 16000, "num_channels": 1}
{"command": "status"}
```

returns `{"text": ..., "audio_seconds": ..., "inference_ms": ..., "device": ...}` (plus `chunks` with timestamps).

**Streaming.** With `mic` set, the service reads pcm16 audio from the `audio_in` component, cuts it at the empty-chunk
segment markers that a wake-word filter (`viam:filtered-audio:wake-word-filter`) emits, transcribes each segment and calls
`transcript_target.DoCommand({"command": "deliverTranscript", "transcript": ..., "is_final": true, "confidence": 1.0,
"source": "openvino-whisper"})`. That is the same contract as Viam's `viam:speech-to-text` cloud models, so this service can
replace them in an existing voice pipeline.

**Why not the `viam-labs:service:speech` API?** It is a custom API whose clients need an extra package
(`speech_service_api`), and current Viam voice modules use the generic `deliverTranscript` contract instead.

## `viam:openvino-genai:vlm`

```json
{ "model_path": "/path/to/qwen2.5-vl-3b-int4", "camera_name": "realsense" }
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `camera_name` / `camera_names` | string / list | | Cameras the service may read. Any of them can be named in a request. |
| `prompt` | string | `Describe this image in one sentence.` | Question used for GetClassifications. |
| `max_image_size` | int | | Downscale frames so the longest side is at most this many pixels (faster). |
| `max_new_tokens` | int | `128` | |
| `repetition_penalty` | number | `1.1` | Stops the repetition loops Qwen2.5-VL falls into on list-style prompts. |

For RGBD cameras such as a RealSense the service picks the color frame from `GetImages`, not whichever comes first.

DoCommand:

```json
{"command": "ask", "question": "Is anyone at the desk?", "camera_name": "realsense"}
{"command": "ask", "question": "What is this?", "image_b64": "<base64 jpeg/png>", "max_new_tokens": 64}
{"command": "status"}
```

returns `{"answer": ..., "seconds": ..., "ttft_ms": ..., "tokens_per_second": ..., "device": ...}`.

`GetClassificationsFromCamera` / `GetClassifications` return one classification whose `class_name` is the model's answer to
`prompt` (or to `extra.question`), with confidence 1.0. `CaptureAllFromCamera` supports image and classifications.
Detections and point clouds are not supported.

## `viam:openvino-genai:llm`

```json
{ "model_path": "/path/to/qwen2.5-1.5b-instruct-int4-ov", "system_prompt": "You are a helpful robot assistant." }
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | string | | Prepended to every conversation. |
| `max_history_messages` | int | `20` | Messages kept per session (oldest dropped). |
| `max_new_tokens` | int | `256` | |
| `repetition_penalty` | number | `1.1` | |

DoCommand:

```json
{"command": "generate", "prompt": "What is a gripper?"}
{"command": "generate", "prompt": "And how many joints did I say?", "session": "operator-1", "system": "override system prompt"}
{"command": "reset", "session": "operator-1"}
{"command": "status"}
```

returns `{"text": ..., "seconds": ..., "ttft_ms": ..., "tokens_per_second": ..., "device": ...}`. Sessions are kept in memory
and lost when the module restarts.

## Packaging

The module ships as source. `first_run.sh` (run once per version, as root, before the module starts) downloads a pinned,
checksum-verified [uv](https://github.com/astral-sh/uv), installs a uv-managed Python 3.12 and creates `.venv` from the pinned
`requirements.txt`. The host needs no `python3-venv` and no particular system Python. `run.sh` is the entrypoint and repeats
the setup if the venv is missing (local modules, hot reload).

Why not a PyInstaller binary: the OpenVINO wheels are already self-contained, and a frozen binary bundles the build machine's
`libstdc++`, which shadows the newer one Intel's GPU/NPU drivers need and makes OpenVINO silently report CPU only (see
erh/viam-openvino#1). Shipping source avoids that class of bug and keeps the package small.

Host prerequisites for GPU/NPU (drivers) are the same as for `erh:viam-openvino`; on Panther Lake and other new chips use
compute-runtime 26.35+, Level Zero 1.34+ and NPU driver 1.38+ from Intel's GitHub releases.

## Development

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest pytest-asyncio
.venv/bin/python -m pytest                     # unit tests, no models needed
.venv/bin/python scripts/e2e.py --whisper DIR --audio FILE --vlm DIR --images DIR --llm DIR   # real models
./build.sh                                     # module.tar.gz
```
