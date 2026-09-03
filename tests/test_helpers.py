"""Unit tests for pure integration helper functions."""

from __future__ import annotations

import pytest
from homeassistant.exceptions import HomeAssistantError
import voluptuous as vol

from custom_components.llama_cpp_ai_task.ai_task import (
    _clean_schema,
    _isolate_json,
    _to_json_schema,
)
from custom_components.llama_cpp_ai_task.entity import _extract_text, strip_thinking


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
