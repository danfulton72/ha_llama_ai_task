"""Tests for registry migration performed during integration setup."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components import llama_cpp_ai_task as integration
from custom_components.llama_cpp_ai_task.const import DOMAIN


def test_migrate_legacy_registry_entries(monkeypatch) -> None:
    """Rename generated AI Task IDs while preserving user and Core entity IDs."""
    entity_updates: list[tuple[str, dict[str, object]]] = []
    subentry_updates: list[tuple[str, str]] = []
    removed_devices: list[str] = []
    generated_ids: list[tuple[str, str]] = []

    legacy_entity = SimpleNamespace(
        entity_id="ai_task.llama_cpp_ai_task_01m1mp9x3wz0t795d25ambnac3",
        platform=DOMAIN,
        device_id="legacy-custom-device",
        config_subentry_id="subentry-id",
        unique_id="subentry-id",
    )
    user_named_entity = SimpleNamespace(
        entity_id="ai_task.my_custom_task",
        platform=DOMAIN,
        device_id=None,
        config_subentry_id="custom-subentry-id",
        unique_id="custom-subentry-id",
    )
    core_entity = SimpleNamespace(
        entity_id="conversation.llama_cpp",
        platform="llama_cpp",
        device_id="core-device",
        config_subentry_id=None,
        unique_id="core",
    )

    def available_entity_id(domain: str, title: str) -> str:
        generated_ids.append((domain, title))
        return "ai_task.qwen3_5_4b"

    entity_registry = SimpleNamespace(
        async_update_entity=lambda entity_id, **changes: entity_updates.append(
            (entity_id, changes)
        ),
        async_get_available_entity_id=available_entity_id,
    )
    device_registry = SimpleNamespace(
        async_remove_device=lambda device_id: removed_devices.append(device_id)
    )

    custom_device = SimpleNamespace(
        id="legacy-custom-device",
        identifiers={(DOMAIN, "subentry-id")},
    )
    core_device = SimpleNamespace(
        id="core-device",
        identifiers={("llama_cpp", "subentry-id")},
    )

    monkeypatch.setattr(integration.er, "async_get", lambda _hass: entity_registry)
    monkeypatch.setattr(
        integration.er,
        "async_entries_for_config_entry",
        lambda _registry, _entry_id: [legacy_entity, user_named_entity, core_entity],
    )
    monkeypatch.setattr(integration.dr, "async_get", lambda _hass: device_registry)
    monkeypatch.setattr(
        integration.dr,
        "async_entries_for_config_entry",
        lambda _registry, _entry_id: [custom_device, core_device],
    )

    old_title = "llama.cpp AI Task"
    legacy_subentry = SimpleNamespace(
        subentry_id="subentry-id",
        title=old_title,
        data={},
    )
    custom_subentry = SimpleNamespace(
        subentry_id="custom-subentry-id",
        title=old_title,
        data={},
    )
    entry = SimpleNamespace(
        entry_id="custom-entry",
        runtime_data=SimpleNamespace(model="qwen3.5-4b"),
        subentries={
            "subentry-id": legacy_subentry,
            "custom-subentry-id": custom_subentry,
        },
    )
    hass = SimpleNamespace(
        config_entries=SimpleNamespace(
            async_update_subentry=lambda _entry, subentry, **changes: subentry_updates.append(
                (subentry.subentry_id, changes["title"])
            )
        )
    )

    integration._migrate_legacy_registry_entries(hass, entry)

    assert subentry_updates == [
        ("subentry-id", "Qwen3.5 4b"),
        ("custom-subentry-id", "Qwen3.5 4b"),
    ]
    assert generated_ids == [("ai_task", "Qwen3.5 4b")]
    assert entity_updates == [
        (
            "ai_task.llama_cpp_ai_task_01m1mp9x3wz0t795d25ambnac3",
            {
                "device_id": None,
                "new_entity_id": "ai_task.qwen3_5_4b",
            },
        )
    ]
    assert removed_devices == ["legacy-custom-device"]
