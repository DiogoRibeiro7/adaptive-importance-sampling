"""The notebook staleness check, which is only useful if it can fail.

`scripts/check_notebooks.py` guards an arrangement that goes wrong quietly: each
notebook is committed as a jupytext `.py` and as an `.ipynb` with saved outputs,
and editing one without regenerating the other leaves a published notebook
showing older text and older numbers with nothing about it looking stale.

A check that cannot detect the thing it guards is worse than none, so most of
these tests are about making it fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_notebooks.py"

pytest.importorskip("jupytext", reason="the notebook check needs jupytext")


@pytest.fixture(scope="module")
def checker():
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location("check_notebooks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_notebooks"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def notebook_pair(tmp_path, checker, monkeypatch):
    """A miniature notebooks/ directory holding one matched pair."""
    import jupytext

    source = tmp_path / "example.py"
    source.write_text(
        "# %% [markdown]\n# A heading\n\n# %%\nresult = 2 + 2\nprint(result)\n",
        encoding="utf-8",
    )
    jupytext.write(jupytext.read(source), tmp_path / "example.ipynb")

    monkeypatch.setattr(checker, "NOTEBOOKS", tmp_path)
    return source, tmp_path / "example.ipynb"


class TestItPassesWhenItShould:
    def test_a_matched_pair_reports_no_problems(self, checker, notebook_pair) -> None:
        assert checker.check_sync(checker.paired_notebooks()) == 0

    def test_the_real_repository_is_in_step(self, checker) -> None:
        """The committed notebooks must match their sources right now."""
        assert checker.check_sync(checker.paired_notebooks()) == 0

    def test_it_finds_every_notebook(self, checker) -> None:
        pairs = checker.paired_notebooks()
        names = {source.name for source, _ in pairs}

        assert len(pairs) >= 6
        assert "01_getting_started.py" in names
        assert all(notebook.suffix == ".ipynb" for _, notebook in pairs)


class TestItFailsWhenItShould:
    """The half that matters."""

    def test_an_edited_source_is_caught(self, checker, notebook_pair, capsys) -> None:
        source, _notebook = notebook_pair
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# %%\nprint('added later')\n",
            encoding="utf-8",
        )

        assert checker.check_sync(checker.paired_notebooks()) == 1
        assert "STALE" in capsys.readouterr().out

    def test_a_changed_cell_is_caught(self, checker, notebook_pair, capsys) -> None:
        """Same number of cells, different contents: the easiest case to miss."""
        source, _notebook = notebook_pair
        source.write_text(
            source.read_text(encoding="utf-8").replace("2 + 2", "3 + 3"),
            encoding="utf-8",
        )

        assert checker.check_sync(checker.paired_notebooks()) == 1
        out = capsys.readouterr().out
        assert "STALE" in out
        assert "differ" in out

    def test_a_missing_notebook_is_caught(self, checker, notebook_pair, capsys) -> None:
        _source, notebook = notebook_pair
        notebook.unlink()

        assert checker.check_sync(checker.paired_notebooks()) == 1
        assert "MISSING" in capsys.readouterr().out

    def test_main_returns_a_failing_exit_code(self, checker, notebook_pair) -> None:
        source, _notebook = notebook_pair
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# %%\nprint('drift')\n",
            encoding="utf-8",
        )

        assert checker.main(["--sync"]) == 1

    def test_main_succeeds_on_a_matched_pair(self, checker, notebook_pair) -> None:
        assert checker.main(["--sync"]) == 0


class TestReporting:
    def test_it_only_claims_what_it_checked(
        self, checker, notebook_pair, capsys
    ) -> None:
        """--sync says nothing about whether the notebooks still run."""
        checker.main(["--sync"])

        out = capsys.readouterr().out
        assert "All notebooks in step." in out
        assert "running" not in out

    def test_sync_is_the_default(self, checker, notebook_pair, capsys) -> None:
        checker.main([])
        assert "in step" in capsys.readouterr().out
