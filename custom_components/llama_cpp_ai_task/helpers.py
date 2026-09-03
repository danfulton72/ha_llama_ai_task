"""Pure helpers shared by the llama.cpp integration modules."""

from __future__ import annotations

import re
from typing import Any, Final

import voluptuous as vol

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

try:
    # Home Assistant 2026.9 replaced voluptuous/voluptuous-openapi with probatio
    # as its validation engine. `probatio` is a Core dependency there, so it must
    # not be declared in the manifest requirements.
    from probatio import to_openapi as _convert_schema
except ImportError:  # pragma: no cover - exercised by the HA 2026.8 CI job
    from voluptuous_openapi import convert as _convert_schema

# OpenAPI 3.1 preserves nullability as a JSON-schema-compatible null branch,
# which llama.cpp's json-schema-to-grammar converter understands.
OPENAPI_VERSION: Final = "3.1.0"

# Some chat templates emit reasoning inside the content when the server runs
# with `--reasoning-format none`. Strip it before parsing structured output.
THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
UNCLOSED_THINK_BLOCK = re.compile(
    r"^.*?</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE
)


def _to_json_schema(structure: vol.Schema) -> dict[str, Any]:
    """Convert a Home Assistant selector schema into JSON schema."""
    schema = _convert_schema(
        structure,
        custom_serializer=llm.selector_serializer,
        openapi_version=OPENAPI_VERSION,
    )

    if not isinstance(schema, dict):
        raise HomeAssistantError("Unsupported structure for llama.cpp")

    # llama.cpp's converter needs an explicit type on the root object.
    if "type" not in schema and "properties" in schema:
        schema["type"] = "object"

    return _clean_schema(schema)


def _clean_schema(schema: Any) -> Any:
    """Drop keys llama.cpp's json-schema-to-grammar converter chokes on."""
    if isinstance(schema, dict):
        cleaned = {
            key: _clean_schema(value)
            for key, value in schema.items()
            # `default` has no grammar equivalent, and an empty `enum` would
            # compile to a grammar that matches nothing. `nullable` should not
            # survive the 3.1 conversion, but a custom serializer can still emit
            # it, so it is dropped defensively.
            if key not in ("default", "nullable")
            and not (key == "enum" and not value)
        }
        if cleaned.get("type") == "object" and "properties" not in cleaned:
            # An object with no properties would constrain output to `{}`.
            cleaned.pop("type", None)
        return cleaned
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def attachments_supported(*, forced: bool, vision: bool, audio: bool) -> bool:
    """Return whether attachment support should be advertised.

    Support is derived from the modalities the server reports at setup, so
    restarting llama-server with a projector and reloading the entry is enough.
    ``forced`` is an override for builds that do not report their modalities.
    """
    return bool(forced or vision or audio)


def _isolate_json(text: str) -> str:
    """Return the JSON document inside ``text``."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    if text.startswith(("{", "[")):
        return text

    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end > start:
        return text[start : end + 1]
    return text


def _extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant text out of a chat completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HomeAssistantError("llama.cpp returned no choices")

    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise HomeAssistantError("llama.cpp returned a malformed message")

    text = message.get("content") or ""
    if not isinstance(text, str):
        # Some builds return typed content parts.
        text = "".join(
            part.get("text", "")
            for part in text
            if isinstance(part, dict) and part.get("type") == "text"
        )

    text = strip_thinking(text).strip()

    if not text:
        if choice.get("finish_reason") == "length":
            raise HomeAssistantError(
                "llama.cpp stopped at the token limit before producing an "
                "answer; raise 'Maximum tokens' or disable thinking"
            )
        raise HomeAssistantError("llama.cpp returned an empty response")

    return text


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks that leaked into the content."""
    text = THINK_BLOCK.sub("", text)
    lower_text = text.lower()
    if (
        "</think" in lower_text
        or "</thinking" in lower_text
        or "</reasoning" in lower_text
    ):
        text = UNCLOSED_THINK_BLOCK.sub("", text)
    return text
