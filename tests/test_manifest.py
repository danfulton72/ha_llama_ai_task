"""Repository-level manifest tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "llama_cpp_ai_task"
MANIFEST = INTEGRATION / "manifest.json"


def test_manifest_contract() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["domain"] == INTEGRATION.name
    assert data["name"] == "llama.cpp AI Task"
    assert data["config_flow"] is True
    assert data["dependencies"] == ["ai_task"]
    assert data["integration_type"] == "service"
    assert re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", data["version"])
    assert data["documentation"].startswith("https://github.com/danfulton72/")
    assert (INTEGRATION / "translations" / "en.json").is_file()
    assert (ROOT / "hacs.json").is_file()
