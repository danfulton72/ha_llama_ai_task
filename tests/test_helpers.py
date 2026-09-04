"""Unit tests for pure integration helper functions."""

from __future__ import annotations

import json

import pytest
import voluptuous as vol
from homeassistant.exceptions import HomeAssistantError

from custom_components.llama_cpp_ai_task.helpers import (
    _clean_schema,
    _extract_text,
    _isolate_json,
    _to_json_schema,
    attachments_supported,
    model_name_to_title,
    strip_thinking,
)


def test_strip_thinking() -> None:
    assert strip_thinking("<think>secret</think>Answer").strip() == "Answer"
    assert strip_thinking("reasoning</thinking>Answer") == "Answer"
    assert strip_thinking("Answer") == "Answer"


def test_extract_text() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "<think>x</think>Ready"},
            }
        ]
    }
    assert _extract_text(response) == "Ready"


def test_extract_text_typed_parts() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "image_url", "image_url": {}},
                        {"type": "text", "text": "world"},
                    ]
                }
            }
        ]
    }
    assert _extract_text(response) == "Hello world"


def test_extract_text_errors() -> None:
    with pytest.raises(HomeAssistantError, match="no choices"):
        _extract_text({"choices": []})
    with pytest.raises(HomeAssistantError, match="token limit"):
        _extract_text(
            {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        )


def test_isolate_json() -> None:
    assert _isolate_json('{"ok": true}') == '{"ok": true}'
    assert _isolate_json('```json\n{"ok": true}\n```') == '{"ok": true}'
    assert _isolate_json('Result: {"ok": true} done') == '{"ok": true}'
    assert _isolate_json("nothing structured") == "nothing structured"


def test_model_name_to_title_matches_core_llama_cpp() -> None:
    """Use the same inverse-slug convention as Home Assistant Core llama.cpp."""
    assert model_name_to_title("qwen3.5-4b") == "Qwen3.5 4b"
    assert model_name_to_title("deepseek-v4-flash") == "Deepseek V4 Flash"
    assert model_name_to_title("llama-3.2-3b-instruct") == "Llama 3.2 3b Instruct"
    assert model_name_to_title("anthropic/claude-fable-5") == "Anthropic Claude Fable 5"


def test_schema_conversion_and_cleaning() -> None:
    schema = vol.Schema(
        {
            vol.Required("summary"): str,
            vol.Optional("count"): int,
            vol.Optional("ok"): bool,
        }
    )
    converted = _to_json_schema(schema)
    assert converted["type"] == "object"
    assert converted["required"] == ["summary"]
    assert set(converted["properties"]) == {"summary", "count", "ok"}

    dirty = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "default": "x", "nullable": True},
            "b": {"type": "string", "enum": []},
            "c": {"type": "object"},
        },
    }
    cleaned = _clean_schema(dirty)
    assert "default" not in cleaned["properties"]["a"]
    assert "nullable" not in cleaned["properties"]["a"]
    assert "enum" not in cleaned["properties"]["b"]
    assert "type" not in cleaned["properties"]["c"]


def test_schema_conversion_is_openapi_31_and_preserves_nullability() -> None:
    """Nullability must survive 3.1 conversion instead of being discarded."""
    converted = _to_json_schema(
        vol.Schema({vol.Optional("note"): vol.Any(str, None)})
    )
    note_schema = converted["properties"]["note"]
    assert "nullable" not in json.dumps(note_schema)

    def contains_null_type(value: object) -> bool:
        if isinstance(value, dict):
            schema_type = value.get("type")
            if schema_type == "null" or (
                isinstance(schema_type, list) and "null" in schema_type
            ):
                return True
            return any(contains_null_type(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_null_type(item) for item in value)
        return False

    assert contains_null_type(note_schema), note_schema


@pytest.mark.parametrize(
    ("forced", "vision", "audio", "expected"),
    [
        (False, False, False, False),
        (False, True, False, True),
        (False, False, True, True),
        (True, False, False, True),
        (True, True, True, True),
    ],
)
def test_attachments_supported(
    forced: bool, vision: bool, audio: bool, expected: bool
) -> None:
    assert (
        attachments_supported(forced=forced, vision=vision, audio=audio) is expected
    )
