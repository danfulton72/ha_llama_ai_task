"""Repository-level manifest tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "llama_cpp_ai_task"
MANIFEST = INTEGRATION / "manifest.json"
HACS = ROOT / "hacs.json"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def test_manifest_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["domain"] == INTEGRATION.name
    assert data["name"] == "llama.cpp AI Task"
    assert data["config_flow"] is True
    assert data["dependencies"] == ["ai_task"]
    assert data["integration_type"] == "service"
    assert re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", data["version"])
    assert data["documentation"].startswith("https://github.com/danfulton72/")
    assert data["requirements"] == []
    assert (INTEGRATION / "translations" / "en.json").is_file()
    assert HACS.is_file()


def test_hacs_metadata() -> None:
    hacs = json.loads(HACS.read_text(encoding="utf-8"))
    major, minor = (int(part) for part in hacs["homeassistant"].split(".")[:2])
    # AI Task attachments need 2025.8; the converter fallback and APIs used by
    # this integration are explicitly tested from Home Assistant 2026.8 onward.
    assert (major, minor) >= (2026, 8)
    assert hacs["zip_release"] is True
    assert hacs["filename"] == "llama_cpp_ai_task.zip"

    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    escaped = re.escape(hacs["filename"])
    assert re.search(rf"zip -qr .*dist/{escaped}", workflow)
    assert f"dist/{hacs['filename']}" in workflow


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
