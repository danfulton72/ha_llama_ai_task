"""The llama.cpp AI Task integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import LlamaCppClient, LlamaCppError, LlamaCppServerInfo
from .const import DOMAIN, LOGGER

PLATFORMS: tuple[Platform, ...] = (Platform.AI_TASK,)


@dataclass(slots=True)
class LlamaCppRuntimeData:
    """Data shared by all entities of a config entry."""

    client: LlamaCppClient
    info: LlamaCppServerInfo


type LlamaCppConfigEntry = ConfigEntry[LlamaCppRuntimeData]


def _cleanup_legacy_service_devices(
    hass: HomeAssistant, entry: LlamaCppConfigEntry
) -> None:
    """Remove service devices created by older versions of this integration.

    AI Task entities are standalone entities. Older versions attached each task to
    a ``DeviceEntryType.SERVICE`` device, which made the task look like another
    llama.cpp conversation service in Home Assistant. Detach only entities and
    remove only devices owned by this custom integration/config entry; Core's
    ``llama_cpp`` registry entries use a different domain and are never touched.
    """
    entity_registry = er.async_get(hass)
    for entity in er.async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity.platform == DOMAIN and entity.device_id is not None:
            entity_registry.async_update_entity(entity.entity_id, device_id=None)

    device_registry = dr.async_get(hass)
    for device in dr.async_entries_for_config_entry(device_registry, entry.entry_id):
        if any(identifier[0] == DOMAIN for identifier in device.identifiers):
            device_registry.async_remove_device(device.id)


async def async_setup_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Set up llama.cpp AI Task from a config entry."""
    client = LlamaCppClient(
        async_get_clientsession(hass),
        entry.data[CONF_URL],
    )

    try:
        info = await client.async_detect_server_info()
    except LlamaCppError as err:
        # The server may simply still be loading the model.
        raise ConfigEntryNotReady(str(err)) from err

    LOGGER.debug(
        "Connected to llama.cpp %s serving %s (n_ctx=%s, modalities=%s, routed_model=%s)",
        info.build_info,
        info.model_name,
        info.n_ctx,
        info.modalities,
        client.default_model,
    )

    entry.runtime_data = LlamaCppRuntimeData(client=client, info=info)
    _cleanup_legacy_service_devices(hass, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_entry))

    return True


async def async_update_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> None:
    """Reload the entry when options or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
