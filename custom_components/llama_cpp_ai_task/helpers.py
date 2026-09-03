"""Pure helpers shared by the llama.cpp integration modules."""

from __future__ import annotations

import re
from typing import Any

from probatio import to_openapi
import voluptuous as vol

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

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
    schema = to_openapi(structure, custom_serializer=llm.selector_serializer)

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
            # `default` and `nullable` have no grammar equivalent, and an empty
            # `enum` would compile to a grammar that matches nothing.
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
