"""Shared entity plumbing for the llama.cpp integration.

Everything that talks to the model lives here so that the ``ai_task`` platform
(and a future ``conversation`` platform) can share one implementation of
"turn a ChatLog into a llama.cpp request".
"""

from __future__ import annotations

import base64
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import Entity

from .client import LlamaCppAuthError, LlamaCppClient, LlamaCppError, LlamaCppServerInfo
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
    DOMAIN,
    LOGGER,
    MANUFACTURER,
)

if TYPE_CHECKING:
    from . import LlamaCppConfigEntry

# Some chat templates emit reasoning inside the content when the server runs
# with `--reasoning-format none`. Strip it before parsing structured output.
THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
UNCLOSED_THINK_BLOCK = re.compile(
    r"^.*?</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE
)


class LlamaCppBaseLLMEntity(Entity):
    """Base class for entities backed by a llama.cpp server."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_should_poll = False

    def __init__(
        self, entry: LlamaCppConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer=MANUFACTURER,
            model=subentry.data.get(CONF_MODEL) or self.server_info.model_name,
            sw_version=self.server_info.build_info,
            entry_type=DeviceEntryType.SERVICE,
            configuration_url=self.client.base_url,
        )

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
            # Reuse the KV cache between calls; big win for repeated tasks that
            # share a system prompt.
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
            # Honoured by hybrid-reasoning templates (Qwen3, GLM, ...) and
            # ignored by templates that do not use the variable. Constrained
            # decoding and long thinking blocks do not mix well, so thinking is
            # off unless the user asks for it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        if json_schema is not None:
            # Sent in three shapes on purpose: llama.cpp releases have read the
            # schema from `response_format.schema` and from
            # `response_format.json_schema.schema`, and other OpenAI-compatible
            # servers behind a proxy expect the nested form. Extra keys are
            # ignored, so this works across builds.
            payload["response_format"] = {
                "type": "json_schema",
                "schema": json_schema,
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        LOGGER.debug(
            "Sending chat completion to llama.cpp (model=%s, structured=%s, messages=%d)",
            payload.get("model") or self.server_info.model_name,
            json_schema is not None,
            len(messages),
        )

        try:
            response = await self.client.async_chat_completion(
                payload, timeout=float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
            )
        except LlamaCppAuthError as err:
            raise HomeAssistantError(
                "The llama.cpp server rejected the API key"
            ) from err
        except LlamaCppError as err:
            raise HomeAssistantError(f"Error talking to llama.cpp: {err}") from err

        return _extract_text(response)

    async def _async_build_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert the chat log into llama.cpp chat messages.

        All system content is merged into a single leading message: several chat
        templates reject more than one system turn.
        """
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []

        if extra_prompt := self.options.get(CONF_PROMPT):
            system_parts.append(str(extra_prompt).strip())

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
                # Tool calls are not exposed by this integration.
                LOGGER.debug("Skipping unsupported chat content %s", type(content))

        if system_parts:
            messages.insert(
                0, {"role": "system", "content": "\n\n".join(system_parts)}
            )

        return messages

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
            # llama.cpp accepts wav and mp3 for audio input.
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
    if "</think" in lower_text or "</thinking" in lower_text or "</reasoning" in lower_text:
        text = UNCLOSED_THINK_BLOCK.sub("", text)
    return text
