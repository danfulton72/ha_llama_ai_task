"""Tests for registry cleanup performed during integration setup."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components import llama_cpp_ai_task as integration
from custom_components.llama_cpp_ai_task.const import DOMAIN


def test_cleanup_legacy_service_devices(monkeypatch) -> None:
    """Detach custom AI Task entities and remove only custom service devices."""
    entity_updates: list[tuple[str, dict[str, object]]] = []
    removed_devices: list[str] = []

    custom_entity = SimpleNamespace(
        entity_id="ai_task.llama_cpp_ai_task",
        platform=DOMAIN,
        device_id="legacy-custom-device",
    )
    core_entity = SimpleNamespace(
        entity_id="conversation.llama_cpp",
        platform="llama_cpp",
        device_id="core-device",
    )

    entity_registry = SimpleNamespace(
        async_update_entity=lambda entity_id, **changes: entity_updates.append(
            (entity_id, changes)
        )
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
        lambda _registry, _entry_id: [custom_entity, core_entity],
    )
    monkeypatch.setattr(integration.dr, "async_get", lambda _hass: device_registry)
    monkeypatch.setattr(
        integration.dr,
        "async_entries_for_config_entry",
        lambda _registry, _entry_id: [custom_device, core_device],
    )

    entry = SimpleNamespace(entry_id="custom-entry")
    integration._cleanup_legacy_service_devices(SimpleNamespace(), entry)

    assert entity_updates == [
        ("ai_task.llama_cpp_ai_task", {"device_id": None})
    ]
    assert removed_devices == ["legacy-custom-device"]
