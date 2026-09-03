"""Tests for release/version helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

RELEASE_SPEC = importlib.util.spec_from_file_location(
    "release", SCRIPTS / "release.py"
)
assert RELEASE_SPEC is not None and RELEASE_SPEC.loader is not None
release = importlib.util.module_from_spec(RELEASE_SPEC)
sys.modules["release"] = release
RELEASE_SPEC.loader.exec_module(release)

CONSISTENCY_SPEC = importlib.util.spec_from_file_location(
    "release_consistency", SCRIPTS / "check_release_consistency.py"
)
assert CONSISTENCY_SPEC is not None and CONSISTENCY_SPEC.loader is not None
release_consistency = importlib.util.module_from_spec(CONSISTENCY_SPEC)
CONSISTENCY_SPEC.loader.exec_module(release_consistency)


def test_normalize_version() -> None:
    assert release.normalize_version("1.2.3") == "1.2.3"
    assert release.normalize_version("v1.2.3") == "1.2.3"


@pytest.mark.parametrize("value", ["1.2", "1.2.3.4", "v1.2", "01.2.3", "latest", ""])
def test_normalize_version_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        release.normalize_version(value)


def test_next_patch_version() -> None:
    assert release.next_patch_version("1.0.0") == "1.0.1"
    assert release.next_patch_version("v2.7.99") == "2.7.100"


def test_manifest_version_is_semver() -> None:
    current = release.manifest_version()
    assert release.normalize_version(current) == current


def test_tag_matches_manifest() -> None:
    current = release.manifest_version()
    release.check_tag_matches_manifest(f"v{current}")
    mismatched = "v0.0.0" if current != "0.0.0" else "v999.999.999"
    with pytest.raises(ValueError):
        release.check_tag_matches_manifest(mismatched)


def test_newest_published_release_includes_prereleases() -> None:
    releases = [
        {
            "tag_name": "v1.0.0",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-09-01T10:00:00Z",
        },
        {
            "tag_name": "v1.1.0-beta.1",
            "draft": False,
            "prerelease": True,
            "published_at": "2026-09-02T10:00:00Z",
        },
        {
            "tag_name": "v9.0.0",
            "draft": True,
            "prerelease": False,
            "published_at": "2026-09-03T10:00:00Z",
        },
    ]
    assert release_consistency._newest_published_tag(releases) == "v1.1.0-beta.1"


def test_newest_published_release_handles_empty_list() -> None:
    assert release_consistency._newest_published_tag([]) is None
