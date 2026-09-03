#!/usr/bin/env python3
"""Compare manifest.json with the newest published GitHub release."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from release import check_tag_matches_manifest


def _newest_published_tag(releases: list[dict[str, Any]]) -> str | None:
    """Return the newest published release tag, including prereleases."""
    published = [
        release
        for release in releases
        if not release.get("draft")
        and release.get("published_at")
        and release.get("tag_name")
    ]
    if not published:
        return None
    newest = max(published, key=lambda release: str(release["published_at"]))
    return str(newest["tag_name"])


def latest_release_tag(repository: str, token: str | None) -> str | None:
    """Return the newest published release tag, including prereleases."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ha-llama-ai-task-release-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repository}/releases"
            f"?per_page=100&page={page}"
        )
        request = Request(url, headers=headers)
        with urlopen(request, timeout=15) as response:
            data = json.load(response)
        if not isinstance(data, list):
            raise ValueError("GitHub releases response was not a list")
        releases.extend(item for item in data if isinstance(item, dict))
        if len(data) < 100:
            break
        page += 1

    return _newest_published_tag(releases)


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
        print(f"manifest.json matches newest published GitHub release {tag}")
        return 0
    except (HTTPError, URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as err:
        print(f"Release consistency check failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
