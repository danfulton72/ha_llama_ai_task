"""The llama.cpp AI Task integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_API_KEY, CONF_URL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import (
    LlamaCppAuthError,
    LlamaCppClient,
    LlamaCppError,
    LlamaCppServerInfo,
    normalize_base_url,
)
from .const import CONF_MODEL, DEFAULT_AI_TASK_NAME, DOMAIN, LOGGER
from .helpers import model_name_to_title

PLATFORMS: tuple[Platform, ...] = (Platform.AI_TASK,)
CONFIG_ENTRY_VERSION = 2


@dataclass(slots=True)
class LlamaCppRuntimeData:
    """Data shared by all entities of a config entry."""

    client: LlamaCppClient
    info: LlamaCppServerInfo


type LlamaCppConfigEntry = ConfigEntry[LlamaCppRuntimeData]


def _migration_title(subentry: ConfigSubentry) -> str:
    """Return the VERSION 2 title for a subentry.

    Only the known legacy default title is converted to a model-derived default.
    Any other title is treated as user-owned and must be preserved. Legacy tasks
    without a stored model keep the old default rather than inventing a new name.
    """
    if subentry.title != DEFAULT_AI_TASK_NAME:
        return subentry.title
    model = subentry.data.get(CONF_MODEL)
    if model:
        return model_name_to_title(str(model))
    return subentry.title


def _migrate_v1_registry_state(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> None:
    """Detach entities from legacy custom service devices and remove those devices.

    Existing entity IDs are deliberately preserved. Programmatic registry renames
    do not rewrite every automation/script/dashboard reference, so changing a
    legacy ``ai_task.llama_cpp_ai_task_*`` ID during migration would be a silent
    breaking change. New entities still receive model-derived IDs naturally.
    """
    entity_registry = er.async_get(hass)
    for entity in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity.platform == DOMAIN and entity.device_id is not None:
            entity_registry.async_update_entity(entity.entity_id, device_id=None)

    device_registry = dr.async_get(hass)
    for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
        if any(identifier[0] == DOMAIN for identifier in device.identifiers):
            device_registry.async_remove_device(device.id)


async def async_migrate_entry(
    hass: HomeAssistant, entry: LlamaCppConfigEntry
) -> bool:
    """Migrate persistent config and registry state to the current version."""
    LOGGER.debug(
        "Migrating config entry %s from version %s", entry.entry_id, entry.version
    )

    if entry.version > CONFIG_ENTRY_VERSION:
        LOGGER.error(
            "Cannot migrate config entry %s from future version %s",
            entry.entry_id,
            entry.version,
        )
        return False

    if entry.version == 1:
        data = dict(entry.data)
        if isinstance((url := data.get(CONF_URL)), str):
            data[CONF_URL] = normalize_base_url(url)

        # Older releases sometimes stored a null/blank key. Keep a real key now
        # that authentication is optional again, but remove meaningless values.
        if not data.get(CONF_API_KEY):
            data.pop(CONF_API_KEY, None)

        for subentry in tuple(entry.subentries.values()):
            title = _migration_title(subentry)
            if title != subentry.title:
                hass.config_entries.async_update_subentry(
                    entry, subentry, title=title
                )

        _migrate_v1_registry_state(hass, entry)
        hass.config_entries.async_update_entry(
            entry,
            data=data,
            version=CONFIG_ENTRY_VERSION,
        )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Set up llama.cpp AI Task from a config entry."""
    client = LlamaCppClient(
        async_get_clientsession(hass),
        entry.data[CONF_URL],
        entry.data.get(CONF_API_KEY),
    )

    try:
        info = await client.async_detect_server_info()
    except LlamaCppAuthError as err:
        # Authentication failures are persistent until credentials change, so
        # trigger Home Assistant's reauthentication flow instead of retrying.
        raise ConfigEntryAuthFailed(str(err)) from err
    except LlamaCppError as err:
        # Connection/model startup failures may be transient.
        raise ConfigEntryNotReady(str(err)) from err

    # Refreshing models is intentionally stateful: it may establish a request
    # default only when the server reports a loaded model or exactly one model.
    await client.async_refresh_models()
    model = client.default_model or info.model_name

    LOGGER.debug(
        "Connected to llama.cpp %s serving %s (n_ctx=%s, modalities=%s, model=%s)",
        info.build_info,
        info.model_name,
        info.n_ctx,
        info.modalities,
        model,
    )

    entry.runtime_data = LlamaCppRuntimeData(client=client, info=info)

    # Persistent title/URL/registry migration belongs in async_migrate_entry.
    # Keeping setup mutation-free avoids update-listener/reload ordering hazards.
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_entry))

    return True


async def async_update_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> None:
    """Reload the entry when options or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
