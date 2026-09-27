# Release process

Safe-ICE follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and
records every change in the [changelog](changelog.md), in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Until 1.0, a
minor version may change the public API.

The version is stated once, as `version` under `[project]` in `pyproject.toml`.
`safe_ice.__version__` reads it back from the installed package metadata.

## Cutting a release

1. **Update the changelog.** Move everything under `## [Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section, leave an empty `Unreleased` heading
   behind, and update the link definitions at the bottom of `CHANGELOG.md`.
2. **Bump the version.** Run the **Version bump** workflow from the Actions tab,
   which opens a pull request, or do it locally:

    ```bash
    python scripts/pyproject_editor.py --check bump-version patch   # preview
    python scripts/pyproject_editor.py bump-version patch           # apply
    ```

3. **Merge** both changes into `main` and wait for CI to pass on that commit.
4. **Tag** the release commit and push the tag:

    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    git push origin vX.Y.Z
    ```

The full checklist, including the local checks to run first, is
[`scripts/release_checklist.md`](https://github.com/DiogoRibeiro7/adaptive-importance-sampling/blob/main/scripts/release_checklist.md).

## What the release workflow does

Pushing a `v*` tag runs `.github/workflows/release.yml`, which:

1. runs Ruff, strict mypy and the full test suite, including slow tests;
2. fails if the tag does not match the packaged version exactly;
3. builds the sdist and wheel and checks their metadata with `twine check`;
4. creates a GitHub Release with both distributions attached, taking the release
   notes from the matching section of `CHANGELOG.md`.

Tags containing `rc`, `alpha` or `beta` are published as pre-releases.

## Documentation

The site is deployed from `main`, not from tags. Merging the release commit
changes `CHANGELOG.md` and `pyproject.toml`, which triggers the **Documentation**
workflow, so the published changelog shows the new version once it finishes.

## PyPI

Publishing to PyPI is not wired up yet, which is why installation is from
GitHub. When it is, the release workflow will publish through a PyPI Trusted
Publisher, with no stored API token; the release checklist describes the steps.
