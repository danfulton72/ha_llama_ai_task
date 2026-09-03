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

**Fix:** `@danfulton72` is listed as code owner and `hacs.json` declares the display name and a Home Assistant minimum version. That minimum was corrected to 2026.8.0 in the second pass below.

## Second pass, reviewed against Home Assistant 2026.9.0

### 7. Declared minimum Home Assistant version was unreachable — critical

`helpers.py` imported `probatio`, which became a Home Assistant Core dependency in 2026.9. Home Assistant 2026.8 still ships the earlier voluptuous/OpenAPI stack. Because `manifest.json` declares no requirements, a 2026.8 installation would not install `probatio` for this custom integration. The original 2025.7 floor was independently too low for attachment support.

**Fix:** the schema converter is resolved at import time, preferring `probatio.to_openapi` and falling back to `voluptuous_openapi.convert`. The declared floor is `2026.8.0` in `hacs.json` and the README. CI explicitly tests both Home Assistant 2026.8.0 and 2026.9.0 so both converter paths are exercised.

### 8. Extra instructions were never rendered — high

The subentry option used a `TemplateSelector` and the strings file described it as a template, but `entity.py` passed the raw string to the model, so any Jinja reached the prompt verbatim.

**Fix:** the value is rendered with `homeassistant.helpers.template.Template`, and a `TemplateError` is surfaced as a `HomeAssistantError` instead of being sent to the model.

### 9. Structured output used a non-canonical `response_format` — medium

The payload carried the schema twice plus OpenAI's `strict` flag. Strict mode additionally imposes schema requirements that a structure converted from Home Assistant does not necessarily satisfy, while the duplicated top-level `schema` key is not part of the canonical OpenAI-compatible nested shape.

**Fix:** only `response_format.json_schema.schema` is sent, without `strict`.

### 10. Nullability was discarded from the schema — medium

OpenAPI 3.0 represents nullability with the `nullable` keyword. The integration's llama.cpp cleanup removed that keyword because it is not part of JSON Schema grammar, which could make nullable values non-nullable.

**Fix:** conversion requests OpenAPI 3.1, which expresses nullability with a JSON-schema-compatible null branch. A regression test now proves a real `null` type survives conversion rather than merely checking that `nullable` disappeared.

### 11. Attachment support was frozen at first setup — medium

The config flow wrote `attachments` as a concrete boolean derived from the server's modalities, which made the auto-detection fallback in `ai_task.py` dead code and meant restarting llama-server with `--mmproj` had no effect until the subentry was reconfigured. The fallback also checked vision only while the flow checked vision or audio.

**Fix:** `attachments` is now a force-on override that is not written during initial setup. Support is recomputed from the modalities reported at each setup via the shared `attachments_supported` helper.

### 12. Audio attachments were not capability-checked — low

Images sent to a server without a vision projector produced a warning; audio did not.

**Fix:** both image and audio branches now warn when the server does not report the corresponding capability.

### 13. Repository plumbing — low

- Reconfigure now rejects a URL already used by another config entry.
- HACS release archives use a constant `llama_cpp_ai_task.zip` filename and hold the integration files at the archive root, matching the `zip_release`/`filename` pair in `hacs.json`. The existing v1.0.0 release is backfilled with the same constant-name asset before this HACS setting is merged, avoiding a broken migration window.
- The HACS workflow no longer ignores `description` and `topics`; both repository metadata fields are set.
- `# noqa: BLE001` was inert because `BLE` is not in the Ruff selection. The comment was removed and `RUF100` added so stale suppressions are reported.
- GitHub Actions use current `actions/checkout@v7` and `actions/setup-python@v7` majors.

## Current compatibility review

Verified against the supported Home Assistant 2026.8/2026.9 boundary and current repository CI.

- AI Task was introduced before the declared compatibility floor; attachment support is available throughout the supported range.
- Home Assistant AI Task uses `AITaskEntity`, `GenDataTask`, `GenDataTaskResult`, and the `_async_generate_data(task, chat_log)` implementation hook used here.
- `AITaskEntityFeature` exposes the features required by this integration, including data generation and attachments.
- Entity platform setup supports `config_subentry_id`, matching this integration's subentry entity setup.
- `ChatLog`, `SystemContent`, `UserContent`, `AssistantContent`, `Attachment.path`/`.mime_type`, and `async_add_assistant_content_without_tools` remain the interfaces used here.
- Home Assistant 2026.9 uses `probatio`; the integration retains a 2026.8 fallback through `voluptuous_openapi`.
- `homeassistant.helpers.llm.selector_serializer` remains available for selector-aware schema conversion.
- `aiohttp` remains a Home Assistant Core dependency. The integration uses Home Assistant's shared client session and does not create its own long-lived session.
- No API key is written to logs by the integration.

## Remaining limitations / follow-up

### Test coverage stops at the Home Assistant boundary

`tests/` covers `client.py` and the pure helpers, including schema conversion and attachment-capability logic. CI runs those tests under both Home Assistant 2026.8.0 and 2026.9.0. Nothing yet drives `config_flow.py`, `__init__.py`, `entity.py`, or `ai_task.py` through a real `hass` instance.

Recommended follow-up: add `pytest-homeassistant-custom-component` tests covering config flow, setup/unload, subentry creation/reconfigure, structured-output validation, attachments, auth failure, template errors, and server-unavailable behavior. Pin it to the release matching the Home Assistant version under test because it pins `homeassistant` exactly.

### Brand images

`custom_components/llama_cpp_ai_task/brand/icon.png` is intentionally retained. Current Home Assistant supports brand assets bundled with custom integrations, so the local icon is meaningful and does not need to be removed merely because an external `home-assistant/brands` repository also exists.

### CI validation

GitHub Actions validate Python tests/lint, HACS, hassfest, and release consistency. The Python matrix includes the declared minimum Home Assistant version and the 2026.9 compatibility boundary.

### Feature limitations

- No Home Assistant LLM tool calling.
- No conversation entity (the official Core llama.cpp integration handles conversation agents).
- No streaming.
- No AI image generation platform.
- Attachments are limited to supported local image/audio files and are inlined into the request.

## HACS install target

Custom repository: `danfulton72/ha_llama_ai_task`

Integration directory installed by HACS: `custom_components/llama_cpp_ai_task`
