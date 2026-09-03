"""Config flow for the llama.cpp integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryData,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME, CONF_URL
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .client import (
    LlamaCppAuthError,
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppError,
    LlamaCppServerInfo,
)
from .const import (
    AI_TASK_SUBENTRY_TYPE,
    CONF_ATTACHMENTS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REPEAT_PENALTY,
    CONF_TEMPERATURE,
    CONF_THINKING,
    CONF_TIMEOUT,
    CONF_TOP_K,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_URL,
    DOMAIN,
    LOGGER,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL, default=DEFAULT_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


class LlamaCppConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the config flow for llama.cpp."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            self._async_abort_entries_match({CONF_URL: url})

            info, error = await self._async_try_connect(
                url, user_input.get(CONF_API_KEY)
            )
            if error:
                errors["base"] = error
            else:
                assert info is not None
                title = info.model_name or "llama.cpp"
                return self.async_create_entry(
                    title=title,
                    data={CONF_URL: url, CONF_API_KEY: user_input.get(CONF_API_KEY)},
                    subentries=[
                        ConfigSubentryData(
                            subentry_type=AI_TASK_SUBENTRY_TYPE,
                            title=DEFAULT_AI_TASK_NAME,
                            data={
                                CONF_ATTACHMENTS: info.supports_vision
                                or info.supports_audio
                            },
                            unique_id=None,
                        )
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the URL or API key of an existing entry."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            _, error = await self._async_try_connect(
                url, user_input.get(CONF_API_KEY)
            )
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry,
                    data_updates={
                        CONF_URL: url,
                        CONF_API_KEY: user_input.get(CONF_API_KEY),
                    },
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input or dict(entry.data)
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: dict[str, Any]
    ) -> ConfigFlowResult:
        """Handle re-authentication."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a new API key."""
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            _, error = await self._async_try_connect(
                entry.data[CONF_URL], user_input.get(CONF_API_KEY)
            )
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_KEY: user_input.get(CONF_API_KEY)}
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    )
                }
            ),
            errors=errors,
            description_placeholders={CONF_URL: entry.data[CONF_URL]},
        )

    async def _async_try_connect(
        self, url: str, api_key: str | None
    ) -> tuple[Any, str | None]:
        """Try to reach the server, returning (server_info, error_key)."""
        client = LlamaCppClient(async_get_clientsession(self.hass), url, api_key)
        try:
            return await client.async_get_server_info(), None
        except LlamaCppAuthError:
            return None, "invalid_auth"
        except LlamaCppConnectionError:
            return None, "cannot_connect"
        except LlamaCppError as err:
            LOGGER.debug("Unexpected response from %s: %s", url, err)
            return None, "invalid_server"
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error connecting to %s", url)
            return None, "unknown"

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry types this integration supports."""
        return {AI_TASK_SUBENTRY_TYPE: LlamaCppSubentryFlowHandler}


class LlamaCppSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure an AI Task entity."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a new AI Task entity."""
        return await self.async_step_set_options()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing AI Task entity."""
        return await self.async_step_set_options()

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Collect the model options."""
        entry = self._get_entry()
        is_new = self.source != SOURCE_RECONFIGURE

        if user_input is not None:
            name = user_input.pop(CONF_NAME, DEFAULT_AI_TASK_NAME)
            if is_new:
                return self.async_create_entry(title=name, data=user_input)
            return self.async_update_and_abort(
                entry, self._get_reconfigure_subentry(), data=user_input
            )

        # The entry may not be loaded (server offline), so fall back to blanks.
        runtime = getattr(entry, "runtime_data", None)
        info = runtime.info if runtime else LlamaCppServerInfo()
        models = await runtime.client.async_list_models() if runtime else []

        if is_new:
            defaults: dict[str, Any] = {
                CONF_NAME: DEFAULT_AI_TASK_NAME,
                CONF_ATTACHMENTS: info.supports_vision or info.supports_audio,
            }
        else:
            subentry = self._get_reconfigure_subentry()
            defaults = {CONF_NAME: subentry.title, **subentry.data}

        return self.async_show_form(
            step_id="set_options",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(models, include_name=is_new), defaults
            ),
            description_placeholders={
                "model": info.model_name or "unknown",
                "n_ctx": str(info.n_ctx or "unknown"),
            },
        )


def _options_schema(models: list[str], *, include_name: bool) -> vol.Schema:
    """Build the AI Task option schema."""
    schema: dict[Any, Any] = {}

    if include_name:
        schema[vol.Required(CONF_NAME, default=DEFAULT_AI_TASK_NAME)] = TextSelector()

    if models:
        model_selector: Any = SelectSelector(
            SelectSelectorConfig(
                options=models,
                mode=SelectSelectorMode.DROPDOWN,
                custom_value=True,
            )
        )
    else:
        model_selector = TextSelector()

    schema.update(
        {
            vol.Optional(CONF_MODEL): model_selector,
            vol.Optional(CONF_PROMPT): TemplateSelector(),
            vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): NumberSelector(
                NumberSelectorConfig(min=1, max=32768, mode=NumberSelectorMode.BOX)
            ),
            vol.Required(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): NumberSelector(
                NumberSelectorConfig(min=0, max=2, step=0.05)
            ),
            vol.Required(CONF_TOP_P, default=DEFAULT_TOP_P): NumberSelector(
                NumberSelectorConfig(min=0, max=1, step=0.01)
            ),
            vol.Required(CONF_TOP_K, default=DEFAULT_TOP_K): NumberSelector(
                NumberSelectorConfig(min=0, max=200, step=1)
            ),
            vol.Required(
                CONF_REPEAT_PENALTY, default=DEFAULT_REPEAT_PENALTY
            ): NumberSelector(NumberSelectorConfig(min=1, max=2, step=0.01)),
            vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): NumberSelector(
                NumberSelectorConfig(
                    min=10, max=900, step=5, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(CONF_THINKING, default=DEFAULT_THINKING): BooleanSelector(),
            vol.Required(CONF_ATTACHMENTS, default=False): BooleanSelector(),
        }
    )

    return vol.Schema(schema)
