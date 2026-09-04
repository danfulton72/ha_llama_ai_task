# Home Assistant / HACS audit

Audit date: 2026-09-04

This document records the current compatibility and safety review for
`danfulton72/ha_llama_ai_task`. It is intentionally updated when repository
behaviour changes; older release decisions are recorded in the history section
rather than being left as apparently-current findings.

## Current result

The repository is structured as a HACS custom integration under
`custom_components/llama_cpp_ai_task/` and deliberately uses the unique domain
`llama_cpp_ai_task` so it can coexist with Home Assistant Core's official
`llama_cpp` conversation integration.

The v1.1.0 stabilization branch is designed for Home Assistant 2026.8 and
2026.9. CI exercises both versions, plus HACS validation and hassfest.

## Current integration contract

### Home Assistant compatibility

- Minimum supported Home Assistant version: 2026.8.
- Home Assistant 2026.8 uses the older voluptuous/OpenAPI converter path;
  2026.9 uses `probatio`. The integration resolves the available converter at
  import time and CI tests both versions.
- AI Task entities use `AITaskEntity`, `GenDataTask`, `GenDataTaskResult`,
  config subentries, `ChatLog`, local attachment paths, and Home Assistant's
  shared aiohttp session.
- AI Tasks are standalone entities. This intentionally differs from Core
  llama.cpp's service-device entity structure; only the model-to-title string
  conversion is mirrored.

### Authentication

- API-key authentication is optional.
- When configured, the key is sent only as `Authorization: Bearer <key>`.
- HTTP 401/403 is a distinct authentication failure. During setup it becomes a
  Home Assistant reauthentication request instead of an endless
  `ConfigEntryNotReady` retry; config flows show `invalid_auth`.
- Reconfigure never pre-fills the stored API key into the browser. Leaving the
  new-key field blank preserves an existing key; an explicit remove control
  deletes it; entering a replacement key takes precedence.
- Reconfigure starts from the existing `entry.data` mapping so future unrelated
  config-entry keys are not silently discarded.
- Debug logging does not print API keys, prompt text, or base64 attachment data.

### VERSION 2 migration and entity identity

Persistent migration is performed by `async_migrate_entry`, not from
`async_setup_entry`.

VERSION 2:

- normalizes stored server URLs so a legacy `.../v1` value becomes the server
  root;
- removes only blank/null legacy API-key values while retaining a real key;
- converts the known legacy default subentry title to a model-derived default
  only when a model was actually stored;
- preserves any user-owned subentry title;
- detaches entities from legacy custom `DeviceEntryType.SERVICE` devices and
  removes only devices owned by this custom integration/config entry;
- **does not programmatically rename any existing entity ID**.

The last point is deliberate. Existing entity IDs may be referenced by
Home Assistant automations, scripts, dashboards, templates, or external clients.
A direct `EntityRegistry.async_update_entity(new_entity_id=...)` migration does
not safely rewrite all of those references. Legacy IDs such as
`ai_task.llama_cpp_ai_task_<id>` therefore remain valid. New entities naturally
receive the improved model-derived entity-ID convention.

Subentry titles are user-owned after creation. A model name supplies the default
for a new task, but restart/reload and later model changes do not force a custom
name back to the model name.

### Server and router detection

Direct llama-server is identified through `/props`.

For routed servers:

- `/v1/models` is used as a model catalogue, but a generic OpenAI-compatible
  catalogue by itself is **not** accepted as proof of llama.cpp;
- routed mode must be established by a successful model-specific `/props`
  request, a recognized llama.cpp/router `/props` routing error, or native
  llama.cpp router metadata such as `role: router`;
- model routing is not gated on llama-swap-specific `owned_by` or metadata, so
  native llama.cpp router mode is supported too;
- already-loaded models are preferred for model-specific `/props` capability
  inspection;
- `autoload=false` is sent where supported;
- the integration never walks an arbitrary list of unloaded models and triggers
  a sequence of cold starts;
- when exactly one routed model exists, one `autoload=false` probe is allowed so
  capabilities can be discovered without creating a multi-model cold-start
  storm;
- native llama.cpp router-level `/props` is recognized and, when safe, enriched
  with model-specific `/props` so modalities/context are not silently lost.

`async_list_models()` is intentionally pure. The stateful operation that may set
`client.default_model` is `async_refresh_models()`, making calls that establish
an automatic request default explicit. A default is selected only from a model
reported loaded or from an unambiguous sole model; the first item in a
multi-model list is never chosen arbitrarily.

### AI Task request behaviour

- `cache_prompt` is enabled to reuse llama.cpp's KV cache for repeated tasks.
- Model sampler options are passed through to llama.cpp.
- Hybrid-reasoning templates receive
  `chat_template_kwargs: {"enable_thinking": false}` by default because long
  reasoning output and constrained JSON are a poor combination.
- Structured output uses the nested
  `response_format.json_schema.schema` shape without OpenAI's `strict` flag.
- Returned structured data is validated again against the original Home
  Assistant schema so a llama.cpp structured-output fail-open cannot silently
  return the wrong shape.
- Multiple system parts are merged into one leading system message because some
  chat templates reject multiple system turns.
- Tool content is deliberately skipped because this integration does not expose
  Home Assistant LLM tool calling.
- Image and audio attachments are inlined only from local files. Capability
  support is derived from `/props`, with an explicit force-on override for
  servers that fail to report modalities.

### Release packaging

HACS uses a constant `llama_cpp_ai_task.zip` release asset.

The release quality gate runs `compileall`, but the release ZIP is built with
`git archive HEAD:custom_components/llama_cpp_ai_task`. This means generated
`__pycache__`, `.pyc`, and `.pyo` files from the compiled worktree cannot be
swept into the HACS archive. The workflow additionally inspects the archive and
fails if Python bytecode is present.

The release workflow normally increments the patch version after successful
push-triggered CI on `main`. A one-shot `RELEASE_VERSION` marker can request a
specific larger version (the current stabilization target is v1.1.0) while the
pre-release manifest remains equal to the latest published release. The marker
is consumed in the release commit. Repository tests compare the manifest and
marker semantically and do not hard-code a particular previous patch version.

## Test coverage

The repository currently has:

- in-process HTTP client tests covering direct llama-server, routed discovery,
  generic OpenAI-compatible rejection, model default selection, timeouts, and
  optional Bearer authentication;
- helper/schema/structured-output tests;
- release/manifest workflow contract tests;
- VERSION 2 migration tests using Home Assistant's real `ConfigEntry`,
  `EntityRegistry`, and `DeviceRegistry` APIs rather than invented registry
  `SimpleNamespace` mocks.

Migration tests use the public `ConfigEntries.async_add()` path while replacing
only the setup call so the test can register an entry without trying to contact a
real llama.cpp server. Each raw `HomeAssistant` test instance is explicitly shut
down with `await hass.async_stop(force=True)`.

CI runs the Python suite against both Home Assistant 2026.8.0 and 2026.9.0.

## Release/change history relevant to the audit

### Initial publication work

The original package used domain `llama_cpp`, which collided with Home Assistant
Core's integration, and its runtime files were at repository root. Publication
work moved it to `custom_components/llama_cpp_ai_task/`, added HACS metadata and
custom-integration translations, revalidated structured output in Home
Assistant, stopped logging complete model payloads, and established the 2026.8
compatibility floor.

### v1.0.1

API-key support was removed and AI Tasks were changed from custom service-device
presentation to standalone entities. The latter prevented the custom AI Task
from looking like another Core llama.cpp conversation service. The API-key
removal was later reconsidered because it also excluded authenticated
llama-server deployments and authenticating reverse proxies.

### v1.0.3

llama-swap routing support was added. Setup learned to normalize an entered
`.../v1` base URL, discover `/v1/models`, and route `/props` through a model ID.

### v1.0.4

Model-based task naming was introduced using Home Assistant Core's llama.cpp
model-to-title conversion. Review subsequently found that setup-time title
rewrites and automatic registry ID renames were too aggressive; v1.1.0 changes
model naming to a creation default and preserves existing entity IDs.

### v1.1.0 stabilization

The stabilization work restores optional API-key authentication with explicit
auth failures, moves persistent changes into VERSION 2 migration, preserves user
names and existing entity IDs, normalizes legacy URLs, hardens router detection,
avoids multi-model cold starts, makes model-refresh side effects explicit,
removes arbitrary `models[0]` selection, uses real Home Assistant registries in
migration tests, and prevents bytecode from entering release archives.

## Remaining limitations

- No Home Assistant LLM tool calling.
- No conversation entity; Home Assistant Core's `llama_cpp` integration covers
  conversation agents.
- No streaming.
- No AI image-generation platform.
- Attachments are limited to supported local image/audio files and are inlined
  into the request.
- On a multi-model router where no model is reported loaded, per-model capability
  data may remain unknown until a model is loaded or explicitly selected. This
  is preferred to starting multiple large models merely to inspect them.
- A task explicitly selecting a different model can have capabilities that differ
  from the model inspected at config-entry setup.

## Brand assets

`custom_components/llama_cpp_ai_task/brand/icon.png` is intentionally retained.
Current Home Assistant supports local brand assets bundled with custom
integrations, so it is not dead release content.

## HACS install target

Custom repository: `danfulton72/ha_llama_ai_task`

Integration directory installed by HACS: `custom_components/llama_cpp_ai_task`
