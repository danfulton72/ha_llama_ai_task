# Contributing

## Local checks

Use Python 3.14 and install the test dependencies:

```bash
python -m pip install -r requirements_test.txt
```

Run the same core checks used by CI:

```bash
python -m compileall -q custom_components scripts tests
ruff check .
ruff format --check .
pytest
```

GitHub Actions additionally runs Home Assistant Hassfest and the HACS validator.

## Pull requests

Keep normal feature/fix pull requests version-neutral: do not edit the `version` field in `manifest.json`. The published GitHub Release is the source of truth for the released version and the release workflow owns manifest version changes.

## Maintainer release procedure

1. Merge all intended changes to `main` after CI passes.
2. Open **Actions → Release → Run workflow**.
3. Enter the next semantic version as `X.Y.Z`.
4. The workflow updates `manifest.json`, runs tests/lint, commits `Release vX.Y.Z`, creates `vX.Y.Z`, and pushes the commit and tag atomically.
5. The workflow creates the GitHub Release from that existing tag and uploads:
   - `llama_cpp_ai_task-X.Y.Z.zip`
   - `llama_cpp_ai_task-X.Y.Z.zip.sha256`
6. Verify the Release workflow and the subsequent release consistency check are green.

Never publish a GitHub Release from a tag whose `manifest.json` version differs from the release tag. The automated workflow enforces this invariant.
