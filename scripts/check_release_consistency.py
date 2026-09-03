#!/usr/bin/env python3
"""Compare manifest.json with the latest published GitHub release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from release import check_tag_matches_manifest


def latest_release_tag(repository: str, token: str | None) -> str | None:
    """Return the latest published release tag, or None when there is none."""
    url = f"https://api.github.com/repos/{repository}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ha-llama-ai-task-release-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            data = json.load(response)
    except HTTPError as err:
        if err.code == 404:
            return None
        raise
    return str(data["tag_name"])


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-release", action="store_true")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()

    if not args.repository:
        print("Repository is required via --repository or GITHUB_REPOSITORY", file=sys.stderr)
        return 2

    try:
        tag = latest_release_tag(args.repository, os.environ.get("GITHUB_TOKEN"))
        if tag is None:
            if args.allow_no_release:
                print("No published GitHub release yet; bootstrap is allowed")
                return 0
            print("No published GitHub release exists", file=sys.stderr)
            return 1
        check_tag_matches_manifest(tag)
        print(f"manifest.json matches latest GitHub release {tag}")
        return 0
    except (HTTPError, URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as err:
        print(f"Release consistency check failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
