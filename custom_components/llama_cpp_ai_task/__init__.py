"""The llama.cpp integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import LlamaCppClient, LlamaCppError, LlamaCppServerInfo
from .const import LOGGER

PLATFORMS: tuple[Platform, ...] = (Platform.AI_TASK,)


@dataclass(slots=True)
class LlamaCppRuntimeData:
    """Data shared by all entities of a config entry."""

    client: LlamaCppClient
    info: LlamaCppServerInfo


type LlamaCppConfigEntry = ConfigEntry[LlamaCppRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Set up llama.cpp from a config entry."""
    client = LlamaCppClient(
        async_get_clientsession(hass),
        entry.data[CONF_URL],
    )

    try:
        info = await client.async_get_server_info()
    except LlamaCppError as err:
        # The server may simply still be loading the model.
        raise ConfigEntryNotReady(str(err)) from err

    LOGGER.debug(
        "Connected to llama.cpp %s serving %s (n_ctx=%s, modalities=%s)",
        info.build_info,
        info.model_name,
        info.n_ctx,
        info.modalities,
    )

    entry.runtime_data = LlamaCppRuntimeData(client=client, info=info)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_entry))

    return True


async def async_update_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> None:
    """Reload the entry when options or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LlamaCppConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
