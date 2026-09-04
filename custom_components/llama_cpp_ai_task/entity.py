"""Shared entity plumbing for the llama.cpp AI Task integration.

Everything that talks to the model lives here so the ``ai_task`` platform can
share one implementation of "turn a ChatLog into a llama.cpp request".
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import template
from homeassistant.helpers.entity import Entity

from .client import LlamaCppClient, LlamaCppError, LlamaCppServerInfo
from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REPEAT_PENALTY,
    CONF_TEMPERATURE,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TOP_K,
    CONF_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    LOGGER,
)
from .helpers import _extract_text

if TYPE_CHECKING:
    from . import LlamaCppConfigEntry


class LlamaCppBaseLLMEntity(Entity):
    """Base class for entities backed by a llama.cpp server."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self, entry: LlamaCppConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity.

        AI Tasks intentionally remain standalone entities rather than Core
        llama.cpp-style service devices. New subentries use Core's model-to-title
        conversion as their default title, while the subentry title itself is
        user-owned thereafter and becomes the entity's friendly name.
        """
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_name = subentry.title

    @property
    def options(self) -> dict[str, Any]:
        """Return the options of this entity's subentry."""
        return dict(self.subentry.data)

    @property
    def client(self) -> LlamaCppClient:
        """Return the shared client."""
        return self.entry.runtime_data.client

    @property
    def server_info(self) -> LlamaCppServerInfo:
        """Return the server info captured at setup."""
        return self.entry.runtime_data.info

    async def _async_generate(
        self,
        chat_log: conversation.ChatLog,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Run one chat completion for the current chat log and return the text."""
        options = self.options
        messages = await self._async_build_messages(chat_log)

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "cache_prompt": True,
            "max_tokens": int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
            "temperature": float(options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
            "top_p": float(options.get(CONF_TOP_P, DEFAULT_TOP_P)),
            "top_k": int(options.get(CONF_TOP_K, DEFAULT_TOP_K)),
            "repeat_penalty": float(
                options.get(CONF_REPEAT_PENALTY, DEFAULT_REPEAT_PENALTY)
            ),
        }

        if model := options.get(CONF_MODEL):
            payload["model"] = model

        if not options.get(CONF_THINKING, DEFAULT_THINKING):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": json_schema,
                },
            }

        LOGGER.debug(
            "Sending chat completion to llama.cpp (model=%s, structured=%s, messages=%d)",
            payload.get("model") or self.client.default_model or self.server_info.model_name,
            json_schema is not None,
            len(messages),
        )

        try:
            response = await self.client.async_chat_completion(
                payload, timeout=float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
            )
        except LlamaCppError as err:
            raise HomeAssistantError(f"Error talking to llama.cpp: {err}") from err

        return _extract_text(response)

    async def _async_build_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert the chat log into llama.cpp chat messages."""
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []

        if extra_prompt := self._render_extra_prompt():
            system_parts.append(extra_prompt)

        for content in chat_log.content:
            if isinstance(content, conversation.SystemContent):
                if content.content:
                    system_parts.append(content.content.strip())
            elif isinstance(content, conversation.UserContent):
                messages.append(await self._async_user_message(content))
            elif isinstance(content, conversation.AssistantContent):
                if content.content:
                    messages.append(
                        {"role": "assistant", "content": content.content}
                    )
            else:
                LOGGER.debug("Skipping unsupported chat content %s", type(content))

        if system_parts:
            messages.insert(
                0, {"role": "system", "content": "\n\n".join(system_parts)}
            )

        return messages

    def _render_extra_prompt(self) -> str | None:
        """Render the configured extra instructions as a template."""
        raw_prompt = self.options.get(CONF_PROMPT)
        if not raw_prompt:
            return None
        try:
            rendered = template.Template(str(raw_prompt), self.hass).async_render(
                {"ha_name": self.hass.config.location_name},
                parse_result=False,
            )
        except TemplateError as err:
            raise HomeAssistantError(
                f"Error rendering the extra instructions template: {err}"
            ) from err
        return rendered.strip() or None

    async def _async_user_message(
        self, content: conversation.UserContent
    ) -> dict[str, Any]:
        """Build a user message, inlining attachments as data URLs."""
        attachments = getattr(content, "attachments", None)
        if not attachments:
            return {"role": "user", "content": content.content}

        parts: list[dict[str, Any]] = []
        if content.content:
            parts.append({"type": "text", "text": content.content})

        for attachment in attachments:
            parts.append(await self._async_attachment_part(attachment))

        return {"role": "user", "content": parts}

    async def _async_attachment_part(self, attachment: Any) -> dict[str, Any]:
        """Convert one attachment into a content part."""
        mime_type: str = getattr(attachment, "mime_type", "") or ""
        path: Path | None = getattr(attachment, "path", None)

        if path is None:
            raise HomeAssistantError("Attachment has no local path")

        def _read() -> bytes:
            if not path.exists():
                raise HomeAssistantError(f"`{path}` does not exist")
            return path.read_bytes()

        data = await self.hass.async_add_executor_job(_read)
        encoded = base64.b64encode(data).decode("utf-8")

        if mime_type.startswith("image/"):
            if not self.server_info.supports_vision:
                LOGGER.warning(
                    "Sending an image to a llama.cpp server that does not report "
                    "vision support; start llama-server with --mmproj"
                )
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }

        if mime_type.startswith("audio/"):
            if not self.server_info.supports_audio:
                LOGGER.warning(
                    "Sending audio to a llama.cpp server that does not report "
                    "audio support; start llama-server with --mmproj"
                )
            audio_format = mime_type.rsplit("/", 1)[-1].lower()
            if audio_format in ("mpeg", "mpga", "mp3"):
                audio_format = "mp3"
            elif audio_format in ("wav", "x-wav", "wave"):
                audio_format = "wav"
            else:
                raise HomeAssistantError(
                    f"Unsupported audio attachment type `{mime_type}`; "
                    "llama.cpp accepts wav and mp3"
                )
            return {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": audio_format},
            }

        raise HomeAssistantError(
            f"Unsupported attachment type `{mime_type}`; llama.cpp accepts "
            "images and audio"
        )
