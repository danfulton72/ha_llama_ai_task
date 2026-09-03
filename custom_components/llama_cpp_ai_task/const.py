"""Constants for the llama.cpp AI Task integration."""

from __future__ import annotations

import logging
from typing import Final

DOMAIN: Final = "llama_cpp_ai_task"
LOGGER: Final = logging.getLogger(__package__)

# Subentry types. Each subentry becomes one standalone AI Task entity.
AI_TASK_SUBENTRY_TYPE: Final = "ai_task_data"

DEFAULT_URL: Final = "http://localhost:8080"
DEFAULT_AI_TASK_NAME: Final = "llama.cpp AI Task"

# Subentry option keys.
CONF_MODEL: Final = "model"
CONF_PROMPT: Final = "prompt"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TEMPERATURE: Final = "temperature"
CONF_TOP_P: Final = "top_p"
CONF_TOP_K: Final = "top_k"
CONF_REPEAT_PENALTY: Final = "repeat_penalty"
CONF_TIMEOUT: Final = "timeout"
CONF_ATTACHMENTS: Final = "attachments"
CONF_THINKING: Final = "thinking"

DEFAULT_MAX_TOKENS: Final = 1024
DEFAULT_TEMPERATURE: Final = 0.4
DEFAULT_TOP_P: Final = 0.95
DEFAULT_TOP_K: Final = 40
DEFAULT_REPEAT_PENALTY: Final = 1.1
DEFAULT_TIMEOUT: Final = 120
DEFAULT_THINKING: Final = False

# Connection check timeout, kept short so config flow / setup fails fast.
PROPS_TIMEOUT: Final = 15
