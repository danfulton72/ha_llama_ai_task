#!/usr/bin/env python3
"""Compare manifest.json with published GitHub releases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from release import check_tag_matches_manifest


def _headers(token: str | None) -> dict[str, str]:
    """Return headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ha-llama-ai-task-release-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


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


def _validate_expected_release(release: dict[str, Any], expected_tag: str) -> None:
    """Validate that an exact GitHub release is published for the expected tag."""
    actual_tag = release.get("tag_name")
    if actual_tag != expected_tag:
        raise ValueError(
            f"Expected GitHub release {expected_tag}, got {actual_tag!r}"
        )
    if release.get("draft"):
        raise ValueError(f"GitHub release {expected_tag} is still a draft")
    if not release.get("published_at"):
        raise ValueError(f"GitHub release {expected_tag} is not published")


def release_by_tag(repository: str, token: str | None, tag: str) -> dict[str, Any]:
    """Return the GitHub release object for an exact tag."""
    encoded_tag = quote(tag, safe="")
    url = f"https://api.github.com/repos/{repository}/releases/tags/{encoded_tag}"
    request = Request(url, headers=_headers(token))
    with urlopen(request, timeout=15) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("GitHub release-by-tag response was not an object")
    return data


def latest_release_tag(repository: str, token: str | None) -> str | None:
    """Return the newest published release tag, including prereleases."""
    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repository}/releases"
            f"?per_page=100&page={page}"
        )
        request = Request(url, headers=_headers(token))
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
    parser.add_argument("--expected-tag")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()

    if not args.repository:
        print("Repository is required via --repository or GITHUB_REPOSITORY", file=sys.stderr)
        return 2

    token = os.environ.get("GITHUB_TOKEN")
    try:
        if args.expected_tag:
            release = release_by_tag(args.repository, token, args.expected_tag)
            _validate_expected_release(release, args.expected_tag)
            check_tag_matches_manifest(args.expected_tag)
            print(
                "manifest.json matches published GitHub release "
                f"{args.expected_tag}"
            )
            return 0

        tag = latest_release_tag(args.repository, token)
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
