# Release checklist

Releases are cut by pushing a tag. `.github/workflows/release.yml` then runs the
full quality gate, builds the distributions, checks that the tag matches the
packaged version, and publishes a GitHub Release with the artifacts attached.

Publishing to PyPI is **not** wired up. See "Enabling PyPI" at the bottom if you
decide to turn it on.

## 1. Before you start

- [ ] `main` (or `develop`) is green in CI
- [ ] Working tree is clean: `git status`
- [ ] You are up to date: `git pull`

## 2. Verify locally

```bash
ruff check .
ruff format --check .
mypy
pytest -m ""            # the whole suite, including slow tests
python -m build && python -m twine check --strict dist/*
```

- [ ] All of the above pass
- [ ] Examples still run: `python examples/basic_usage.py`
- [ ] Docs build: `make -C docs clean html`

## 3. Update the changelog

- [ ] Move everything under `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD`
      section in `CHANGELOG.md`
- [ ] Leave an empty `## [Unreleased]` heading behind
- [ ] Update the link definitions at the bottom of the file

The release workflow extracts the notes for the tagged version from this file,
so it is the single source of release notes.

## 4. Bump the version

The version lives in exactly one place: `version` under `[project]` in
`pyproject.toml`. `safe_ice.__version__` reads it back from the installed
package metadata, so nothing else needs editing.

Either run the **Version bump** workflow from the Actions tab (it opens a PR),
or do it locally:

```bash
python scripts/pyproject_editor.py --check bump-version patch   # preview
python scripts/pyproject_editor.py bump-version patch           # apply
```

- [ ] Version bumped and the change merged

## 5. Tag and push

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

The tag must match the version in `pyproject.toml` exactly, prefixed with `v`.
The release workflow fails the build if it does not.

- [ ] Tag pushed
- [ ] Release workflow succeeded
- [ ] GitHub Release looks right, with `dist/` artifacts attached

## 6. After the release

- [ ] Read the Docs has built the new version
- [ ] Close the milestone and open the next one
- [ ] Update `CITATION.cff` if the release should be cited

## Troubleshooting

**The tag/version check fails.** The tag does not match `[project].version`.
Delete the tag (`git tag -d vX.Y.Z && git push --delete origin vX.Y.Z`), fix the
version, and tag again.

**`twine check` fails.** Usually a malformed `README.md`; it is the long
description and must render as valid Markdown.

**Release notes come out empty.** The changelog has no section matching the
tagged version. The workflow falls back to a generic message rather than
failing, so fix `CHANGELOG.md` and edit the release.

## Enabling PyPI

1. Register the project on PyPI as a Trusted Publisher for this repository,
   pointing at `release.yml` and the `pypi` environment.
2. Add a job to `release.yml` that needs `build`, has `id-token: write`, and
   runs `pypa/gh-action-pypi-publish`. No API token is required.
3. Confirm the name `safe-ice` is available or already yours.
