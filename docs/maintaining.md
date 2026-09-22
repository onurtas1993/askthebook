# Maintaining AskTheBook

## Build executables

The `.github/workflows/build-executables.yml` workflow builds Windows x64 and
Ubuntu 22.04 x64 packages using Python 3.13 and PyInstaller. It runs on manual
dispatch, `v*` tags, and pull requests affecting the application or build inputs.
CI builds and uploads packages without running tests.

For a manual build, open **Actions > Build executables > Run workflow** after the
workflow is on the default branch. Download the platform artifacts from the
completed run. The Windows artifact contains the executable and supporting files;
the Ubuntu artifact contains a tar archive that preserves executable permissions.

The tracked `config.json` is included in both packages. Review its settings before
committing and publishing. Models, llama.cpp, and document data are not bundled.

## Publish a release

Push a version tag pointing to the commit you want to publish:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

After both builds succeed, the release job publishes generated release notes,
`AskTheBook-windows-x64.zip`, and `AskTheBook-ubuntu-22.04-x64.tar.gz`.
Re-running a tagged build replaces the assets on the existing release. Pull
requests and manual branch builds publish workflow artifacts only.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover index/retrieval/generation logic, request validation, model overrides,
busy responses, real PDF extraction/chunking with mocked expensive steps, and failed
upload visibility. Actual model checks require llama.cpp and LM Studio.
