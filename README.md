# llama.cpp AI Task for Home Assistant

A HACS-installable custom integration that exposes a local `llama.cpp` (`llama-server`) instance as a Home Assistant **AI Task** entity. It is intended for `ai_task.generate_data`, including structured JSON output and local image/audio attachments.

> **Why a separate integration?** Home Assistant Core added its own `llama_cpp` integration in 2026.8 for conversation agents. This project deliberately uses the unique domain `llama_cpp_ai_task` so installing it does **not** override the Core integration.

## Features

- One `ai_task` entity per configured subentry.
- Structured output using JSON Schema sent to llama.cpp, with a second Home Assistant-side validation pass before results are returned.
- Image and audio attachments when `llama-server` supports them.
- Per-task model and sampler settings.
- Prompt-cache reuse with `cache_prompt`.
- Optional API-key authentication.
- No cloud service calls from the integration.

## Requirements

- Home Assistant 2025.7 or newer.
- HACS for HACS installation.
- A reachable recent `llama-server` build.

## Start llama-server

For a normal text model:

```bash
llama-server \
  -hf unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --jinja \
  --host 0.0.0.0 --port 8080 \
  -c 8192 \
  --api-key changeme
```

For vision tasks, start the server with a multimodal projector, for example:

```bash
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --mmproj auto --jinja --host 0.0.0.0
```

From the Home Assistant host, a useful connectivity check is:

```bash
curl -s http://YOUR_SERVER:8080/props
```

## Install with HACS

1. In HACS, open **Custom repositories**.
2. Add `https://github.com/danfulton72/ha_llama_ai_task`.
3. Select repository type **Integration**.
4. Download **llama.cpp AI Task**.
5. Restart Home Assistant.
6. Go to **Settings → Devices & services → Add integration → llama.cpp AI Task**.

Manual installation is also possible by copying `custom_components/llama_cpp_ai_task` into your Home Assistant configuration directory.

## Configure

Enter the base URL of `llama-server`, for example `http://192.168.1.10:8080`, and an API key if the server requires one. Do **not** append `/v1`; the integration adds the endpoint paths itself.

An AI Task entity is created automatically. Additional task entities can be added as subentries, each with its own options.

| Option | Default | Notes |
| --- | --- | --- |
| Model | empty | Only needed when the server exposes more than one model. |
| Extra instructions | empty | Added to the system prompt for each request. |
| Maximum tokens | 1024 | Raise for larger structured responses. |
| Temperature | 0.4 | Lower values are usually better for deterministic data tasks. |
| Top P / Top K / Repeat penalty | 0.95 / 40 / 1.1 | Passed to llama.cpp. |
| Request timeout | 120 s | Increase for slow CPU inference or large models. |
| Allow the model to think | off | Usually best left off for constrained JSON output. |
| Support attachments | auto-detected | Can be forced on for builds that do not report modalities. |

## Structured data example

```yaml
action: ai_task.generate_data
data:
  task_name: morning_briefing
  entity_id: ai_task.llama_cpp_ai_task
  instructions: >-
    Outside temperature is {{ states('sensor.outside_temperature') }} °C.
    Write a short briefing and decide whether a coat is needed.
  structure:
    briefing:
      description: One short sentence
      required: true
      selector:
        text:
    coat_needed:
      required: true
      selector:
        boolean:
response_variable: result
```

The result is available under `result.data`. The integration validates the returned JSON against Home Assistant's requested structure before returning it.

## Vision example

```yaml
action: ai_task.generate_data
data:
  task_name: driveway_check
  entity_id: ai_task.llama_cpp_ai_task
  instructions: Is a car parked in the driveway?
  attachments:
    - media_content_id: media-source://media_source/local/driveway.jpg
      media_content_type: image/jpeg
  structure:
    car_present:
      required: true
      selector:
        boolean:
response_variable: result
```

## Thinking and structured output

Hybrid reasoning models are asked not to think by default using `chat_template_kwargs: {"enable_thinking": false}`. This is intentional: reasoning output and grammar-constrained JSON have had compatibility issues in llama.cpp. Enable thinking mainly for free-text tasks.

## Known limitations

- This integration does not implement Home Assistant LLM tool calling.
- It does not provide a conversation agent; Home Assistant Core's `llama.cpp` integration already covers that use case.
- Attachments must resolve to local files and are inlined into the request.
- llama.cpp has had releases where structured-output constraints could fail open. This integration therefore validates the final data again in Home Assistant and fails the task if the structure does not match.
- Streaming and image generation are not implemented.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Failed to connect during setup | `llama-server` is not reachable from Home Assistant, is bound only to loopback, or a firewall is blocking it. |
| Config entry not ready after restart | The server/model is still starting; Home Assistant retries. |
| Structured task fails validation | The model/server ignored or could not enforce the response schema. Try a newer llama.cpp build and keep thinking disabled. |
| Task stops at token limit | Increase **Maximum tokens** or reduce the requested output. |
| Attachments fail | Start llama.cpp with the appropriate multimodal projector and confirm the attachment is a supported local image/audio file. |

Debug logging:

```yaml
logger:
  logs:
    custom_components.llama_cpp_ai_task: debug
```

Debug logs intentionally do not print prompt text or base64 attachment contents.

## Repository layout

```text
custom_components/llama_cpp_ai_task/
├── __init__.py
├── ai_task.py
├── client.py
├── config_flow.py
├── const.py
├── entity.py
├── manifest.json
├── brand/
│   └── icon.png
└── translations/
    └── en.json
```
