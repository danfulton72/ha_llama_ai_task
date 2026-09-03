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
from homeassistant.const import CONF_URL
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
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppError,
    LlamaCppServerInfo,
    normalize_base_url,
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
from .helpers import model_name_to_title

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL, default=DEFAULT_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        )
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
            url = normalize_base_url(user_input[CONF_URL])
            self._async_abort_entries_match({CONF_URL: url})

            info, model, error = await self._async_try_connect(url)
            if error:
                errors["base"] = error
            else:
                assert info is not None
                title_model = model or info.model_name or "llama.cpp"
                subentry_data = {CONF_MODEL: model} if model else {}
                return self.async_create_entry(
                    title=info.model_name or "llama.cpp",
                    data={CONF_URL: url},
                    subentries=[
                        ConfigSubentryData(
                            subentry_type=AI_TASK_SUBENTRY_TYPE,
                            title=model_name_to_title(title_model),
                            # Attachment support is derived from the modalities
                            # the server reports on every setup, so nothing is
                            # frozen into the subentry here.
                            data=subentry_data,
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
        """Change the URL of an existing entry."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            url = normalize_base_url(user_input[CONF_URL])
            for other in self._async_current_entries():
                if other.entry_id != entry.entry_id and other.data.get(CONF_URL) == url:
                    return self.async_abort(reason="already_configured")

            _, _, error = await self._async_try_connect(url)
            if error:
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry,
                    data_updates={CONF_URL: url},
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA,
                user_input or {CONF_URL: entry.data[CONF_URL]},
            ),
            errors=errors,
        )

    async def _async_try_connect(
        self, url: str
    ) -> tuple[LlamaCppServerInfo | None, str | None, str | None]:
        """Try to reach the server and identify a request-safe model ID."""
        client = LlamaCppClient(async_get_clientsession(self.hass), url)
        try:
            info = await client.async_detect_server_info()
            models = await client.async_list_models()
        except LlamaCppConnectionError:
            return None, None, "cannot_connect"
        except LlamaCppError as err:
            LOGGER.debug("Unexpected response from %s: %s", url, err)
            return None, None, "invalid_server"
        except Exception:
            LOGGER.exception("Unexpected error connecting to %s", url)
            return None, None, "unknown"

        # Only persist a model proven by /v1/models (or llama-swap routing).
        # /props model_alias/model_path is useful for naming but is not guaranteed
        # to be a valid OpenAI request model ID.
        model = client.default_model or (models[0] if models else None)
        return info, model, None

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

        # The entry may not be loaded (server offline), so fall back to blanks.
        runtime = getattr(entry, "runtime_data", None)
        info = runtime.info if runtime else LlamaCppServerInfo()
        models = await runtime.client.async_list_models() if runtime else []
        request_model = runtime.client.default_model if runtime else None
        if not request_model and models:
            request_model = models[0]
        title_model = request_model or info.model_name or "llama.cpp"

        if user_input is not None:
            model = user_input.get(CONF_MODEL) or title_model
            title = model_name_to_title(str(model))
            if is_new:
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                entry,
                self._get_reconfigure_subentry(),
                data=user_input,
                title=title,
            )

        if is_new:
            # CONF_ATTACHMENTS is a force-on override, not the detected value.
            defaults: dict[str, Any] = {}
            if request_model:
                defaults[CONF_MODEL] = request_model
        else:
            subentry = self._get_reconfigure_subentry()
            defaults = dict(subentry.data)
            if CONF_MODEL not in defaults and request_model:
                defaults[CONF_MODEL] = request_model

        return self.async_show_form(
            step_id="set_options",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(models), defaults
            ),
            description_placeholders={
                "model": info.model_name or "unknown",
                "n_ctx": str(info.n_ctx or "unknown"),
            },
        )


def _options_schema(models: list[str]) -> vol.Schema:
    """Build the AI Task option schema."""
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

    return vol.Schema(
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
                    min=10, max=900, step=5, mode=NumberSelectorMode.BOX)
            ),
            vol.Required(CONF_THINKING, default=DEFAULT_THINKING): BooleanSelector(),
            vol.Required(CONF_ATTACHMENTS, default=False): BooleanSelector(),
        }
    )
