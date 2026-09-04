"""Integration-level tests for VERSION 2 registry/config migration."""

from __future__ import annotations

from types import MappingProxyType

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


def _legacy_entry() -> ConfigEntry:
    """Build an actual Home Assistant v1 config entry with two subentries."""
    return ConfigEntry(
        data={
            CONF_URL: "http://server:8080/v1/",
            CONF_API_KEY: "secret",
        },
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
                "data": {CONF_MODEL: "qwen3.5-4b"},
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


@pytest.mark.asyncio
async def test_version_2_migration_uses_real_home_assistant_registries(
    tmp_path,
) -> None:
    """Migrate legacy IDs/devices while preserving a user-owned task title."""
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = ConfigEntries(hass, {})
    entry = _legacy_entry()

    # Register the real ConfigEntry without setting it up; migrations run before
    # normal entry setup in Home Assistant.
    hass.config_entries._entries[entry.entry_id] = entry

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
        suggested_object_id=(
            "llama_cpp_ai_task_01m1mp9x3wz0t795d25ambnac3"
        ),
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

    assert legacy_entity.entity_id.startswith("ai_task.llama_cpp_ai_task_")
    assert custom_entity.entity_id == "ai_task.my_custom_task"

    assert await integration.async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.data[CONF_URL] == "http://server:8080"
    assert entry.data[CONF_API_KEY] == "secret"
    assert entry.subentries["legacy-subentry"].title == "Qwen3.5 4b"
    assert entry.subentries["custom-subentry"].title == "Kitchen classifier"

    assert entity_registry.async_get(legacy_entity.entity_id) is None
    migrated_entity = entity_registry.async_get("ai_task.qwen3_5_4b")
    assert migrated_entity is not None
    assert migrated_entity.device_id is None

    # A manually renamed entity ID is not migration-owned.
    assert entity_registry.async_get("ai_task.my_custom_task") is not None

    # Only the legacy custom service device is removed.
    assert device_registry.async_get(legacy_device.id) is None

    # VERSION 2 is a one-shot migration. Running it again is a no-op and does not
    # rewrite the user-owned title.
    assert await integration.async_migrate_entry(hass, entry) is True
    assert entry.subentries["custom-subentry"].title == "Kitchen classifier"


@pytest.mark.asyncio
async def test_migration_drops_only_blank_legacy_api_key(tmp_path) -> None:
    """Preserve real optional credentials but remove meaningless blank values."""
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = ConfigEntries(hass, {})
    entry = _legacy_entry()
    object.__setattr__(
        entry,
        "data",
        MappingProxyType({CONF_URL: "http://server:8080/v1", CONF_API_KEY: ""}),
    )
    hass.config_entries._entries[entry.entry_id] = entry

    assert await integration.async_migrate_entry(hass, entry) is True
    assert entry.data == {CONF_URL: "http://server:8080"}
