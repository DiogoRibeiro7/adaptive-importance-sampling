"""A version bump must update every file that restates the version.

`pyproject.toml` is the source of truth -- `safe_ice.__version__` reads it back
from installed package metadata, and the release workflow refuses to publish a
tag that disagrees with it. But `CITATION.cff` and the conda recipe restate the
version as literals, and the bump command used to touch only `pyproject.toml`,
so both would still claim the previous release after a bump. Nothing would fail
until someone read the citation metadata.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EDITOR = REPO_ROOT / "scripts" / "pyproject_editor.py"

# The editor is repository tooling rather than part of the package, so its
# dependency is declared in the dev group and installed explicitly in CI.
requires_tomlkit = pytest.mark.skipif(
    importlib.util.find_spec("tomlkit") is None,
    reason="scripts/pyproject_editor.py needs tomlkit; install the dev group",
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A miniature copy of the files a bump has to keep in step."""
    (tmp_path / "conda.recipe").mkdir()

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "safe-ice"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: Safe-ICE\nversion: 1.2.3\nlicense: MIT\n",
        encoding="utf-8",
    )
    (tmp_path / "conda.recipe" / "meta.yaml").write_text(
        '{% set name = "safe-ice" %}\n{% set version = "1.2.3" %}\n\npackage:\n'
        "  version: {{ version }}\n",
        encoding="utf-8",
    )
    return tmp_path


def run_editor(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the editor against the miniature project."""
    return subprocess.run(
        [sys.executable, str(EDITOR), "--file", str(project / "pyproject.toml"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@requires_tomlkit
class TestBumpKeepsMetadataInStep:
    def test_all_three_files_move_together(self, project: Path) -> None:
        result = run_editor(project, "bump-version", "minor")
        assert result.returncode == 0, result.stderr

        assert 'version = "1.3.0"' in (project / "pyproject.toml").read_text()
        assert "version: 1.3.0" in (project / "CITATION.cff").read_text()
        assert (
            '{% set version = "1.3.0" %}'
            in (project / "conda.recipe" / "meta.yaml").read_text()
        )

    def test_the_recipe_keeps_its_jinja_syntax(self, project: Path) -> None:
        """The replacement is a template, so a stray brace would corrupt it."""
        run_editor(project, "bump-version", "patch")

        recipe = (project / "conda.recipe" / "meta.yaml").read_text()
        assert '{% set version = "1.2.4" %}' in recipe
        assert "{{ version }}" in recipe  # the reference below is untouched
        assert '{% set name = "safe-ice" %}' in recipe

    def test_only_the_version_line_of_the_citation_changes(self, project: Path) -> None:
        run_editor(project, "bump-version", "major")

        citation = (project / "CITATION.cff").read_text()
        assert "version: 2.0.0" in citation
        assert "cff-version: 1.2.0" in citation  # not a version to bump
        assert "license: MIT" in citation

    def test_check_mode_writes_nothing(self, project: Path) -> None:
        before = {
            path: path.read_text()
            for path in (
                project / "pyproject.toml",
                project / "CITATION.cff",
                project / "conda.recipe" / "meta.yaml",
            )
        }

        result = run_editor(project, "--check", "bump-version", "minor")
        assert result.returncode == 0, result.stderr

        for path, original in before.items():
            assert path.read_text() == original, f"{path.name} was modified"

    def test_missing_companions_are_not_an_error(self, project: Path) -> None:
        """Not every checkout has a conda recipe."""
        shutil.rmtree(project / "conda.recipe")

        result = run_editor(project, "bump-version", "patch")
        assert result.returncode == 0, result.stderr
        assert 'version = "1.2.4"' in (project / "pyproject.toml").read_text()


class TestTheRealRepositoryStaysConsistent:
    """Guards the released artefacts, not a temporary copy."""

    def test_declared_versions_agree(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject_version = tomllib.load(handle)["project"]["version"]

        citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        assert f"version: {pyproject_version}" in citation

        recipe_path = REPO_ROOT / "conda.recipe" / "meta.yaml"
        if recipe_path.exists():
            recipe = recipe_path.read_text(encoding="utf-8")
            assert f'{{% set version = "{pyproject_version}" %}}' in recipe
