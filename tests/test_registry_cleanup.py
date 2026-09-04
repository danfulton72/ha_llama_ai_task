"""Integration-level tests for VERSION 2 registry/config migration."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from homeassistant.config_entries import ConfigEntries, ConfigEntry, SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er

from custom_components import llama_cpp_ai_task as integration
from custom_components.llama_cpp_ai_task.const import (
    AI_TASK_SUBENTRY_TYPE,
    CONF_MODEL,
    DEFAULT_AI_TASK_NAME,
    DOMAIN,
)


def _legacy_entry(
    *, api_key: str | None = "secret", model: str | None = "qwen3.5-4b"
) -> ConfigEntry:
    """Build an actual Home Assistant v1 config entry with two subentries."""
    data = {CONF_URL: "http://server:8080/v1/"}
    if api_key is not None:
        data[CONF_API_KEY] = api_key

    model_data = {CONF_MODEL: model} if model is not None else {}
    return ConfigEntry(
        data=data,
        discovery_keys=MappingProxyType({}),
        domain=DOMAIN,
        entry_id="legacy-entry",
        minor_version=1,
        options={},
        source=SOURCE_USER,
        subentries_data=[
            {
                "subentry_id": "legacy-subentry",
                "subentry_type": AI_TASK_SUBENTRY_TYPE,
                "title": DEFAULT_AI_TASK_NAME,
                "data": model_data,
                "unique_id": None,
            },
            {
                "subentry_id": "custom-subentry",
                "subentry_type": AI_TASK_SUBENTRY_TYPE,
                "title": "Kitchen classifier",
                "data": {CONF_MODEL: "small"},
                "unique_id": None,
            },
        ],
        title="llama.cpp",
        unique_id=None,
        version=1,
    )


async def _hass_with_registries(tmp_path) -> HomeAssistant:
    """Create Home Assistant with the real registries initialized as Core does."""
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = ConfigEntries(hass, {})
    await hass.config_entries.async_initialize()
    dr.async_setup(hass)
    await dr.async_load(hass, load_empty=True)
    await er.async_load(hass, load_empty=True)
    return hass


async def _add_without_setup(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Add a real ConfigEntry through Core's public API without network setup."""
    original_setup = hass.config_entries.async_setup
    hass.config_entries.async_setup = AsyncMock(return_value=True)
    try:
        await hass.config_entries.async_add(entry)
    finally:
        hass.config_entries.async_setup = original_setup


@pytest.mark.asyncio
async def test_version_2_migration_uses_real_home_assistant_registries(
    tmp_path,
) -> None:
    """Migrate persistent state while preserving legacy and user entity IDs."""
    hass = await _hass_with_registries(tmp_path)
    try:
        entry = _legacy_entry()
        await _add_without_setup(hass, entry)

        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)

        legacy_device = device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, "legacy-subentry")},
            name="Legacy llama.cpp service",
        )

        legacy_entity = entity_registry.async_get_or_create(
            "ai_task",
            DOMAIN,
            "legacy-subentry",
            suggested_object_id="llama_cpp_ai_task_01m1mp9x3wz0t795d25ambnac3",
            config_entry=entry,
            config_subentry_id="legacy-subentry",
            device_id=legacy_device.id,
        )
        custom_entity = entity_registry.async_get_or_create(
            "ai_task",
            DOMAIN,
            "custom-subentry",
            suggested_object_id="my_custom_task",
            config_entry=entry,
            config_subentry_id="custom-subentry",
        )

        legacy_entity_id = legacy_entity.entity_id
        assert legacy_entity_id.startswith("ai_task.llama_cpp_ai_task_")
        assert custom_entity.entity_id == "ai_task.my_custom_task"

        assert await integration.async_migrate_entry(hass, entry) is True

        assert entry.version == 2
        assert entry.data[CONF_URL] == "http://server:8080"
        assert entry.data[CONF_API_KEY] == "secret"
        assert entry.subentries["legacy-subentry"].title == "Qwen3.5 4b"
        assert entry.subentries["custom-subentry"].title == "Kitchen classifier"

        # Existing entity IDs are user-facing API and may be referenced by
        # automations/scripts/dashboards. Migration detaches the old device but
        # deliberately leaves both automatic and manually customized IDs intact.
        migrated_legacy = entity_registry.async_get(legacy_entity_id)
        assert migrated_legacy is not None
        assert migrated_legacy.device_id is None
        assert entity_registry.async_get("ai_task.qwen3_5_4b") is None
        assert entity_registry.async_get("ai_task.my_custom_task") is not None

        # Only the legacy custom service device is removed.
        assert device_registry.async_get(legacy_device.id) is None

        # VERSION 2 is a one-shot migration. Running it again is a no-op and does
        # not rewrite the user-owned title or legacy entity ID.
        assert await integration.async_migrate_entry(hass, entry) is True
        assert entry.subentries["custom-subentry"].title == "Kitchen classifier"
        assert entity_registry.async_get(legacy_entity_id) is not None
    finally:
        await hass.async_stop(force=True)


@pytest.mark.asyncio
async def test_legacy_task_without_model_keeps_title_and_entity_id(tmp_path) -> None:
    """A common v1 legacy task without CONF_MODEL is not pointlessly renamed."""
    hass = await _hass_with_registries(tmp_path)
    try:
        entry = _legacy_entry(model=None)
        await _add_without_setup(hass, entry)
        entity_registry = er.async_get(hass)
        legacy_entity = entity_registry.async_get_or_create(
            "ai_task",
            DOMAIN,
            "legacy-subentry",
            suggested_object_id="llama_cpp_ai_task_01m1mp9x3wz0t795d25ambnac3",
            config_entry=entry,
            config_subentry_id="legacy-subentry",
        )
        old_entity_id = legacy_entity.entity_id

        assert await integration.async_migrate_entry(hass, entry) is True
        assert entry.subentries["legacy-subentry"].title == DEFAULT_AI_TASK_NAME
        assert entity_registry.async_get(old_entity_id) is not None
    finally:
        await hass.async_stop(force=True)


@pytest.mark.asyncio
async def test_migration_drops_only_blank_legacy_api_key(tmp_path) -> None:
    """Preserve real optional credentials but remove meaningless blank values."""
    hass = await _hass_with_registries(tmp_path)
    try:
        entry = _legacy_entry(api_key="")
        await _add_without_setup(hass, entry)

        assert await integration.async_migrate_entry(hass, entry) is True
        assert entry.data == {CONF_URL: "http://server:8080"}
    finally:
        await hass.async_stop(force=True)
