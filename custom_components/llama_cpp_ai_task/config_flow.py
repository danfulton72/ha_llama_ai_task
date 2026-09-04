"""Config flow for the llama.cpp AI Task integration."""

from __future__ import annotations

from typing import Any, Mapping

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
from homeassistant.const import CONF_API_KEY, CONF_URL
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
from .helpers import model_name_to_title

CONF_REMOVE_API_KEY = "remove_api_key"

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


def _reconfigure_schema(*, has_api_key: bool) -> vol.Schema:
    """Build reconfigure fields without sending a stored secret to the browser."""
    fields: dict[Any, Any] = {
        vol.Required(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
    if has_api_key:
        fields[vol.Optional(CONF_REMOVE_API_KEY, default=False)] = BooleanSelector()
    return vol.Schema(fields)


class LlamaCppConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the config flow for llama.cpp AI Task."""

    VERSION = 2

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = normalize_base_url(user_input[CONF_URL])
            if self._url_is_configured(url):
                return self.async_abort(reason="already_configured")

            api_key = user_input.get(CONF_API_KEY) or None
            info, model, error = await self._async_try_connect(url, api_key)
            if error:
                errors["base"] = error
            else:
                assert info is not None
                title_model = model or info.model_name or "llama.cpp"
                subentry_data = {CONF_MODEL: model} if model else {}
                data: dict[str, Any] = {CONF_URL: url}
                if api_key:
                    data[CONF_API_KEY] = api_key
                return self.async_create_entry(
                    title=info.model_name or model or "llama.cpp",
                    data=data,
                    subentries=[
                        ConfigSubentryData(
                            subentry_type=AI_TASK_SUBENTRY_TYPE,
                            title=model_name_to_title(title_model),
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
        """Change the URL or optional API key of an existing entry."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        stored_api_key = entry.data.get(CONF_API_KEY) or None

        if user_input is not None:
            url = normalize_base_url(user_input[CONF_URL])
            if self._url_is_configured(url, exclude_entry_id=entry.entry_id):
                return self.async_abort(reason="already_configured")

            replacement_api_key = user_input.get(CONF_API_KEY) or None
            remove_api_key = bool(user_input.get(CONF_REMOVE_API_KEY))
            if replacement_api_key:
                effective_api_key = replacement_api_key
            elif remove_api_key:
                effective_api_key = None
            else:
                effective_api_key = stored_api_key

            _, _, error = await self._async_try_connect(url, effective_api_key)
            if error:
                errors["base"] = error
            else:
                # Preserve any future config-entry data keys. Only the fields this
                # flow owns are changed, rather than replacing entry.data wholesale.
                data = dict(entry.data)
                data[CONF_URL] = url
                if replacement_api_key:
                    data[CONF_API_KEY] = replacement_api_key
                elif remove_api_key:
                    data.pop(CONF_API_KEY, None)
                return self.async_update_reload_and_abort(entry, data=data)

        # Never pre-fill a stored secret. Blank means "keep the existing key";
        # removal is an explicit checkbox when a key is currently stored.
        defaults: dict[str, Any] = {CONF_URL: entry.data[CONF_URL]}
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                _reconfigure_schema(has_api_key=bool(stored_api_key)),
                user_input or defaults,
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle an authentication failure."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for an optional replacement API key."""
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            api_key = user_input.get(CONF_API_KEY) or None
            _, _, error = await self._async_try_connect(entry.data[CONF_URL], api_key)
            if error:
                errors["base"] = error
            else:
                data = dict(entry.data)
                if api_key:
                    data[CONF_API_KEY] = api_key
                else:
                    data.pop(CONF_API_KEY, None)
                return self.async_update_reload_and_abort(entry, data=data)

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

    def _url_is_configured(
        self, url: str, *, exclude_entry_id: str | None = None
    ) -> bool:
        """Return whether a normalized server URL is already configured."""
        return any(
            entry.entry_id != exclude_entry_id
            and isinstance((stored := entry.data.get(CONF_URL)), str)
            and normalize_base_url(stored) == url
            for entry in self._async_current_entries()
        )

    async def _async_try_connect(
        self, url: str, api_key: str | None
    ) -> tuple[LlamaCppServerInfo | None, str | None, str | None]:
        """Try to reach the server and identify a request-safe model ID."""
        client = LlamaCppClient(async_get_clientsession(self.hass), url, api_key)
        try:
            info = await client.async_detect_server_info()
            # Refreshing is deliberately stateful: it chooses a request default
            # only from a loaded-model signal or an unambiguous sole model.
            await client.async_refresh_models()
        except LlamaCppAuthError:
            return None, None, "invalid_auth"
        except LlamaCppConnectionError:
            return None, None, "cannot_connect"
        except LlamaCppError as err:
            LOGGER.debug("Unexpected response from %s: %s", url, err)
            return None, None, "invalid_server"
        except Exception:
            LOGGER.exception("Unexpected error connecting to %s", url)
            return None, None, "unknown"

        return info, client.default_model, None

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

        runtime = getattr(entry, "runtime_data", None)
        info = runtime.info if runtime else LlamaCppServerInfo()
        models = await runtime.client.async_list_models() if runtime else []
        request_model = runtime.client.default_model if runtime else None

        if user_input is not None:
            selected_model = user_input.get(CONF_MODEL) or request_model
            new_auto_title = model_name_to_title(
                str(selected_model or info.model_name or "llama.cpp")
            )
            if is_new:
                return self.async_create_entry(title=new_auto_title, data=user_input)

            subentry = self._get_reconfigure_subentry()
            old_model = subentry.data.get(CONF_MODEL)
            known_auto_titles = {DEFAULT_AI_TASK_NAME}
            if old_model:
                known_auto_titles.add(model_name_to_title(str(old_model)))

            # Model names are defaults, not ownership of the user's title. Update
            # the title only while it still looks autogenerated; otherwise keep a
            # user-customized title intact when the model changes.
            title = (
                new_auto_title
                if subentry.title in known_auto_titles
                else subentry.title
            )
            return self.async_update_and_abort(
                entry,
                subentry,
                data=user_input,
                title=title,
            )

        if is_new:
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
                "model": info.model_name or request_model or "unknown",
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
                    min=10, max=900, step=5, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(CONF_THINKING, default=DEFAULT_THINKING): BooleanSelector(),
            # This is a force-on override. False leaves capability auto-detection
            # in charge; True lets users work around servers that omit modalities.
            vol.Required(CONF_ATTACHMENTS, default=False): BooleanSelector(),
        }
    )
