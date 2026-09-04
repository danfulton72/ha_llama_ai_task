# llama.cpp AI Task for Home Assistant

A HACS-installable custom integration that exposes a local `llama.cpp` (`llama-server`) instance as a Home Assistant **AI Task** entity. It is intended for `ai_task.generate_data`, including structured JSON output and local image/audio attachments.

> **Why a separate integration?** Home Assistant Core added its own `llama_cpp` integration in 2026.8 for conversation agents. This project deliberately uses the unique domain `llama_cpp_ai_task` so installing it does **not** override the Core integration.

## Features

- One standalone `ai_task` entity per configured subentry; AI Tasks are not exposed as conversation/service devices.
- New tasks use the same **model-to-title conversion** as Home Assistant Core's llama.cpp integration, while task titles you rename are preserved.
- Structured output using JSON Schema sent to llama.cpp, with a second Home Assistant-side validation pass before results are returned.
- Image and audio attachments when `llama-server` supports them.
- Per-task model and sampler settings.
- Prompt-cache reuse with `cache_prompt`.
- Optional Bearer-token/API-key authentication for llama-server, llama-swap, or an authenticating reverse proxy.
- Router-aware discovery for llama-swap and llama.cpp router mode without depending on llama-swap-specific model metadata.
- Safe routed discovery: setup prefers already-loaded models and does not cold-start a sequence of unloaded models just to inspect `/props`.
- No cloud service calls from the integration.

## Requirements

- Home Assistant 2026.8 or newer. Home Assistant 2026.9 replaced `voluptuous-openapi` with `probatio`; the integration detects which converter Core ships and uses it.
- HACS for HACS installation.
- A reachable recent `llama-server` build, llama.cpp router-mode server, or llama-swap instance routing to recent llama-server builds.

## Start llama-server

For a normal text model:

```bash
llama-server \
  -hf unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --jinja \
  --host 0.0.0.0 --port 8080 \
  -c 8192
```

For vision tasks, start the server with a multimodal projector, for example:

```bash
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --mmproj auto --jinja --host 0.0.0.0
```

From the Home Assistant host, a useful direct llama-server connectivity check is:

```bash
curl -s http://YOUR_SERVER:8080/props
```

For a router, list its models first:

```bash
curl -s http://YOUR_ROUTER:8080/v1/models
```

Some routers require a model query parameter for model-specific endpoints such as `/props`. The integration handles this automatically. During setup it probes only models reported as already loaded, adds `autoload=false` for routers that honour it, and never walks a list of unloaded models causing repeated cold starts.

## Install with HACS

1. In HACS, open **Custom repositories**.
2. Add `https://github.com/danfulton72/ha_llama_ai_task`.
3. Select repository type **Integration**.
4. Download **llama.cpp AI Task**.
5. Restart Home Assistant.
6. Go to **Settings → Devices & services → Add integration → llama.cpp AI Task**.

Manual installation is also possible by copying `custom_components/llama_cpp_ai_task` into your Home Assistant configuration directory.

## Configure

Enter the server URL, for example `http://192.168.1.10:8080`. Both the server-root form and an OpenAI-compatible URL ending in `/v1` are accepted; the integration normalizes `/v1` back to the server root before using llama.cpp-specific endpoints. Older stored `/v1` URLs are normalized automatically during the VERSION 2 config-entry migration, so the same server cannot be added twice under the two spellings.

If the endpoint requires authentication, enter its API key. The integration sends it as `Authorization: Bearer <key>`. The field is optional, and reconfigure can remove it by leaving the field blank. A 401/403 is treated as an authentication failure and starts Home Assistant's reauthentication flow instead of being misreported as an invalid server or retried forever.

For plain llama-server, setup reads `/props` directly. If that fails but `/v1/models` provides a valid routed model catalogue, discovery becomes router-aware without checking vendor-specific metadata. A model is selected automatically only when the server reports one as loaded or when exactly one model exists. A multi-model endpoint with no loaded/default signal is left unselected rather than silently choosing `models[0]`.

An AI Task entity is created automatically. Additional task entities can be added as subentries, each with its own options. AI Task entities are deliberately standalone Home Assistant entities and are **not** represented as Core-style llama.cpp conversation/service devices.

The model-to-title *string conversion* mirrors Home Assistant Core. For example, model ID `qwen3.5-4b` defaults to title **Qwen3.5 4b** and normally receives entity ID `ai_task.qwen3_5_4b`. The architecture intentionally differs from Core: this integration does not create a `DeviceEntryType.SERVICE` device. After creation, the subentry title is user-owned. If you rename a task to **Kitchen classifier**, a restart, reload, or model change will not force it back to the model name. Existing autogenerated IDs such as `ai_task.llama_cpp_ai_task_<id>` are migrated once; an entity ID you manually renamed is preserved.

| Option | Default | Notes |
| --- | --- | --- |
| API key | empty | Optional Bearer token for the server/router/reverse proxy. |
| Model | automatic when unambiguous | A reported loaded model or the sole available model may be selected automatically. Multi-model routers with no loaded/default signal are left unselected until you choose one. The model supplies the default title only when a task is first created. |
| Extra instructions | empty | Added to the system prompt for each request. Rendered as a Home Assistant template. |
| Maximum tokens | 1024 | Raise for larger structured responses. |
| Temperature | 0.4 | Lower values are usually better for deterministic data tasks. |
| Top P / Top K / Repeat penalty | 0.95 / 40 / 1.1 | Passed to llama.cpp. |
| Request timeout | 120 s | Increase for slow CPU inference or large models. |
| Allow the model to think | off | Usually best left off for constrained JSON output. |
| Always allow attachments | off | Attachments are offered automatically when the server reports vision or audio support. Force this on for builds that do not report their modalities. |

## Structured data example

```yaml
action: ai_task.generate_data
data:
  task_name: morning_briefing
  entity_id: ai_task.qwen3_5_4b
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
  entity_id: ai_task.qwen3_5_4b
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
- Server-level capability information comes from `/props` when it can be inspected without intentionally cold-starting an unloaded routed model. Capability information may therefore be unknown until a routed model is already loaded.
- A task that explicitly chooses a different model may have different multimodal capabilities from the model inspected during setup.
- Attachments must resolve to local files and are inlined into the request.
- Structured output is requested as `response_format.json_schema.schema`, which needs a llama.cpp build recent enough to read that field.
- llama.cpp has had releases where structured-output constraints could fail open. This integration therefore validates the final data again in Home Assistant and fails the task if the structure does not match.
- Streaming and image generation are not implemented.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Failed to connect during setup | `llama-server`/router is not reachable from Home Assistant, is bound only to loopback, or a firewall is blocking it. |
| Authentication failed / reauthentication requested | The endpoint returned HTTP 401/403. Check the optional API key or reverse-proxy authentication settings. |
| Router has several unloaded models and no model is selected | Choose the desired model in the AI Task options. Discovery intentionally avoids cold-starting each model merely to inspect it. |
| Config entry not ready after restart | The server itself is unreachable or otherwise returned a transient non-auth error; Home Assistant retries. |
| Structured task fails validation | The model/server ignored or could not enforce the response schema. Try a newer llama.cpp build and keep thinking disabled. |
| Task stops at token limit | Increase **Maximum tokens** or reduce the requested output. |
| Attachments fail | Start llama.cpp with the appropriate multimodal projector and confirm the attachment is a supported local image/audio file. |

Debug logging:

```yaml
logger:
  logs:
    custom_components.llama_cpp_ai_task: debug
```

Debug logs intentionally do not print API keys, prompt text, or base64 attachment contents.

## Repository layout

```text
custom_components/llama_cpp_ai_task/
├── __init__.py
├── ai_task.py
├── client.py
├── config_flow.py
├── const.py
├── entity.py
├── helpers.py
├── manifest.json
├── brand/
│   └── icon.png
└── translations/
    └── en.json
```
