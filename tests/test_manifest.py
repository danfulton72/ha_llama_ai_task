"""Repository-level manifest and release-workflow tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "llama_cpp_ai_task"
MANIFEST = INTEGRATION / "manifest.json"
HACS = ROOT / "hacs.json"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
RELEASE_VERSION = ROOT / "RELEASE_VERSION"
VERSION_RE = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Parse the repository's strict X.Y.Z version format."""
    assert VERSION_RE.fullmatch(version)
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def test_manifest_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["domain"] == INTEGRATION.name
    assert data["name"] == "llama.cpp AI Task"
    assert data["config_flow"] is True
    assert data["dependencies"] == ["ai_task"]
    assert data["integration_type"] == "service"
    _version_tuple(data["version"])
    assert data["documentation"].startswith("https://github.com/danfulton72/")
    assert data["requirements"] == []
    assert (INTEGRATION / "translations" / "en.json").is_file()
    assert HACS.is_file()


def test_hacs_metadata_and_clean_release_archive() -> None:
    hacs = json.loads(HACS.read_text(encoding="utf-8"))
    major, minor = (int(part) for part in hacs["homeassistant"].split(".")[:2])
    assert (major, minor) >= (2026, 8)
    assert hacs["zip_release"] is True
    assert hacs["filename"] == "llama_cpp_ai_task.zip"

    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "git archive" in workflow
    assert "HEAD:custom_components/llama_cpp_ai_task" in workflow
    assert f"dist/{hacs['filename']}" in workflow
    assert "__pycache__" in workflow
    assert r"\.py[co]$" in workflow


def test_release_workflow_runs_after_successful_main_ci() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_run:" in workflow
    assert 'workflows: ["CI"]' in workflow
    assert "branches: [main]" in workflow
    assert "github.event.workflow_run.event == 'push'" in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "python scripts/release.py next-patch" in workflow
    assert "python scripts/check_release_consistency.py" in workflow
    assert "git push --atomic origin HEAD:main" in workflow
    assert "gh release create" in workflow
    assert '--expected-tag "${TAG}"' in workflow


def test_one_shot_release_target_contract() -> None:
    """Allow a future release marker before/during release without stale bounds."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    manifest_version = _version_tuple(manifest["version"])

    assert "elif [[ -f RELEASE_VERSION ]]" in workflow
    assert 'cat RELEASE_VERSION' in workflow
    assert "git rm RELEASE_VERSION" in workflow

    if RELEASE_VERSION.exists():
        target = RELEASE_VERSION.read_text(encoding="utf-8").strip()
        target_version = _version_tuple(target)
        # Branch CI sees the currently published manifest version; the release
        # quality gate sees manifest == target after synchronization but before
        # the marker is consumed. Both are valid, and an intervening patch release
        # must not hard-code the branch to a particular previous version.
        assert manifest_version <= target_version
