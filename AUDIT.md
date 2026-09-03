# Home Assistant / HACS audit

Audit date: 2026-09-03

## Result

The original ZIP was **not safe to publish as a current HACS integration without changes**. The repository version has been repackaged and adjusted so it can coexist with current Home Assistant and follow the current custom-integration layout.

## Fixed before publishing

### 1. Core domain collision — critical

The ZIP used the domain `llama_cpp`. Home Assistant Core introduced an official `llama_cpp` integration in Home Assistant 2026.8. A custom integration with that same domain overrides the Core integration, which Home Assistant explicitly discourages.

**Fix:** the custom integration domain is now `llama_cpp_ai_task` and the integration name is `llama.cpp AI Task`.

References:
- https://www.home-assistant.io/integrations/llama_cpp
- https://developers.home-assistant.io/docs/creating_integration_file_structure/

### 2. HACS repository layout — critical

The ZIP placed `__init__.py`, `manifest.json`, platform files, and other runtime files at repository root. HACS integrations should place runtime files under `custom_components/<domain>/` unless `content_in_root` is explicitly used.

**Fix:** runtime files now live under `custom_components/llama_cpp_ai_task/`, and a root `hacs.json` has been added.

Reference:
- https://hacs.xyz/docs/publish/integration/

### 3. Custom-integration translations — high

The ZIP used `strings.json`. Current Home Assistant custom integrations do not use Core's `strings.json` build step; they need language files under `translations/`.

**Fix:** English strings are now in `custom_components/llama_cpp_ai_task/translations/en.json`.

Reference:
- https://developers.home-assistant.io/docs/internationalization/custom_integration/

### 4. Structured-output fail-open risk — high

llama.cpp has had releases where a JSON schema request could be accepted without the schema actually being enforced. The original integration parsed JSON but did not validate the parsed object against Home Assistant's requested structure.

**Fix:** after JSON parsing, the result is validated again with the original Home Assistant `vol.Schema`. A shape/type mismatch now fails the task instead of silently returning incorrect structured data.

Relevant upstream examples:
- https://github.com/ggml-org/llama.cpp/issues/24097
- https://github.com/ggml-org/llama.cpp/issues/19051

### 5. Debug logging of prompt/attachment payloads — medium

The original code logged the full chat-completion payload at debug level. For image/audio tasks that can include large base64 data and private prompt text.

**Fix:** debug logging now records only model, structured/unstructured mode, and message count.

### 6. Manifest/HACS metadata — medium

The manifest had an empty `codeowners` array and the repository had no `hacs.json`.

**Fix:** `@danfulton72` is listed as code owner and `hacs.json` declares the display name and Home Assistant 2025.7 minimum.

## Current compatibility review

- AI Task was introduced in Home Assistant 2025.7.
- Current Home Assistant AI Task still uses `AITaskEntity`, `GenDataTask`, `GenDataTaskResult`, and the `_async_generate_data(task, chat_log)` implementation hook used here.
- Current entity platform setup supports `config_subentry_id`, matching this integration's subentry entity setup.
- `voluptuous-openapi`, `aiohttp`, and `voluptuous` are Home Assistant Core dependencies in the current development branch.
- The integration uses Home Assistant's shared aiohttp client session and does not create its own long-lived session.
- No API key is written to logs by the integration.

## Remaining limitations / follow-up

### Legacy tests are incomplete

The ZIP included two tests that depend on a `tests/hastub` tree that was not included in the archive. They are retained under `tests/` for reference, but are not reliable regression tests yet.

Recommended follow-up: replace them with `pytest-homeassistant-custom-component` tests covering config flow, setup/unload, subentry creation/reconfigure, structured-output validation, attachments, auth failure, and server-unavailable behavior.

### CI validation

A GitHub Actions workflow is included for HACS validation and hassfest. The first pushed workflow run should be treated as the authoritative repository-level validation and any reported metadata/brand warnings should be resolved before submitting this project to the default HACS store.

### Feature limitations

- No Home Assistant LLM tool calling.
- No conversation entity (the official Core llama.cpp integration handles conversation agents).
- No streaming.
- No AI image generation platform.
- Attachments are limited to supported local image/audio files and are inlined into the request.

## HACS install target

Custom repository: `danfulton72/ha_llama_ai_task`

Integration directory installed by HACS: `custom_components/llama_cpp_ai_task`
