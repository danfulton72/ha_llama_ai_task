"""AI Task platform for llama.cpp."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import LlamaCppConfigEntry
from .const import AI_TASK_SUBENTRY_TYPE, CONF_ATTACHMENTS, LOGGER
from .entity import LlamaCppBaseLLMEntity

# `llm.selector_serializer` teaches voluptuous-openapi how to render Home
# Assistant selectors. The name has been private in some releases, so look it
# up defensively rather than failing to import.
SELECTOR_SERIALIZER = getattr(llm, "selector_serializer", None) or getattr(
    llm, "_selector_serializer", None
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LlamaCppConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one AI Task entity per subentry."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != AI_TASK_SUBENTRY_TYPE:
            continue
        async_add_entities(
            [LlamaCppTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LlamaCppTaskEntity(ai_task.AITaskEntity, LlamaCppBaseLLMEntity):
    """AI Task entity backed by a llama.cpp server."""

    # `__init__` deliberately comes from LlamaCppBaseLLMEntity: AITaskEntity
    # defines no initializer, so `(entry, subentry)` resolves down the MRO.

    @property
    def supported_features(self) -> ai_task.AITaskEntityFeature:
        """Return the features, which depend on how llama-server was started."""
        features = ai_task.AITaskEntityFeature.GENERATE_DATA
        if self.subentry.data.get(
            CONF_ATTACHMENTS, self.server_info.supports_vision
        ):
            features |= ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        return features

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        json_schema = _to_json_schema(task.structure) if task.structure else None

        text = await self._async_generate(chat_log, json_schema=json_schema)

        # Keep the chat log consistent so the conversation can be continued.
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=self.entity_id, content=text)
        )

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json.loads(_isolate_json(text))
            # llama.cpp has had releases where a response schema was accepted but
            # not actually enforced. Validate again in Home Assistant so a valid
            # JSON document with the wrong shape cannot silently pass through.
            data = task.structure(data)
        except (ValueError, vol.Invalid) as err:
            LOGGER.warning("llama.cpp returned invalid structured data")
            raise HomeAssistantError(
                "llama.cpp did not return data matching the requested structure"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )


def _to_json_schema(structure: vol.Schema) -> dict[str, Any]:
    """Convert a Home Assistant selector schema into JSON schema.

    llama.cpp compiles the schema into a GBNF grammar, so the output is
    guaranteed to match rather than merely requested.
    """
    kwargs: dict[str, Any] = {}
    if SELECTOR_SERIALIZER is not None:
        kwargs["custom_serializer"] = SELECTOR_SERIALIZER

    schema = convert(structure, **kwargs)

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
    """Return the JSON document inside ``text``.

    Constrained decoding normally makes this a no-op, but a server that ignored
    the schema (or a proxy that dropped it) may wrap the answer in prose or a
    markdown fence.
    """
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
