"""Tests for release/version helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "release.py"
SPEC = importlib.util.spec_from_file_location("release_helpers", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_normalize_version() -> None:
    assert release.normalize_version("1.2.3") == "1.2.3"
    assert release.normalize_version("v1.2.3") == "1.2.3"


@pytest.mark.parametrize("value", ["1.2", "1.2.3.4", "v1.2", "01.2.3", "latest", ""])
def test_normalize_version_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        release.normalize_version(value)


def test_manifest_version_is_semver() -> None:
    current = release.manifest_version()
    assert release.normalize_version(current) == current


def test_tag_matches_manifest() -> None:
    current = release.manifest_version()
    release.check_tag_matches_manifest(f"v{current}")
    mismatched = "v0.0.0" if current != "0.0.0" else "v999.999.999"
    with pytest.raises(ValueError):
        release.check_tag_matches_manifest(mismatched)
