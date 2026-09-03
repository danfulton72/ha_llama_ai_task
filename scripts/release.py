#!/usr/bin/env python3
"""Release/version helpers for the integration.

The published GitHub Release is the canonical released version. This helper
keeps the Home Assistant manifest mechanically synchronized with the release
version requested by the release workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_components" / "llama_cpp_ai_task" / "manifest.json"
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def normalize_version(value: str) -> str:
    """Return a strict X.Y.Z version without an optional leading v."""
    version = value.strip()
    if version.startswith("v"):
        version = version[1:]
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(f"Invalid release version {value!r}; expected X.Y.Z")
    return version


def manifest_version() -> str:
    """Read the version from manifest.json."""
    with MANIFEST.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return normalize_version(str(data["version"]))


def set_manifest_version(value: str) -> str:
    """Set manifest.json version and return the normalized version."""
    version = normalize_version(value)
    with MANIFEST.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data["version"] = version
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return version


def tag_version(tag: str) -> str:
    """Extract a version from a release tag."""
    if not tag.startswith("v"):
        raise ValueError(f"Release tag {tag!r} must start with 'v'")
    return normalize_version(tag)


def check_tag_matches_manifest(tag: str) -> None:
    """Fail if a release tag and manifest.json disagree."""
    release_version = tag_version(tag)
    current = manifest_version()
    if release_version != current:
        raise ValueError(
            f"Release {tag} does not match manifest.json version {current}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("get")
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("version")
    check_parser = subparsers.add_parser("check-tag")
    check_parser.add_argument("tag")
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("version")
    return parser


def main() -> int:
    """CLI entry point."""
    args = _parser().parse_args()
    try:
        if args.command == "get":
            print(manifest_version())
        elif args.command == "set":
            print(set_manifest_version(args.version))
        elif args.command == "check-tag":
            check_tag_matches_manifest(args.tag)
        elif args.command == "normalize":
            print(normalize_version(args.version))
    except (KeyError, OSError, json.JSONDecodeError, ValueError) as err:
        print(err, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
