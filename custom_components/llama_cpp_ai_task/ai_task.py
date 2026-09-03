"""AI Task platform for llama.cpp."""

from __future__ import annotations

import json

import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import LlamaCppConfigEntry
from .const import AI_TASK_SUBENTRY_TYPE, CONF_ATTACHMENTS, LOGGER
from .entity import LlamaCppBaseLLMEntity
from .helpers import _isolate_json, _to_json_schema, attachments_supported


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
        if attachments_supported(
            forced=bool(self.subentry.data.get(CONF_ATTACHMENTS)),
            vision=self.server_info.supports_vision,
            audio=self.server_info.supports_audio,
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
