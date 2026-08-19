#!/usr/bin/env python
"""Check the notebooks are in step with their sources, and still run.

Every notebook under ``notebooks/`` is committed twice: as a jupytext
percent-format ``.py``, which is the file to edit and the one that reviews as a
normal diff, and as an ``.ipynb`` carrying saved outputs so it renders on GitHub
without anyone running it.

That arrangement has two ways of going quietly wrong, and they need different
checks.

**Out of step.** Editing the ``.py`` and forgetting to regenerate leaves the
``.ipynb`` showing older text and older numbers. Nothing about it looks stale.
``--sync`` compares the cells of both and is fast enough to run on every push.

**Out of date.** The saved outputs are pinned to whatever the code did when they
were produced. A change to an estimator does not touch the notebooks, so their
numbers silently become claims about a version that no longer exists.
``--execute`` re-runs each one and fails if any cell raises. It takes minutes,
so it belongs on a schedule rather than on every push.

Neither writes anything. Regenerating is a decision for whoever is editing::

    jupytext --to ipynb --execute notebooks/01_getting_started.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


def paired_notebooks() -> list[tuple[Path, Path]]:
    """Every ``.py`` under ``notebooks/`` with its committed ``.ipynb``."""
    pairs = []
    for source in sorted(NOTEBOOKS.glob("*.py")):
        pairs.append((source, source.with_suffix(".ipynb")))
    return pairs


def cell_sources(path: Path) -> list[str]:
    """Cell contents of a notebook, whichever form it is stored in."""
    import jupytext

    notebook = jupytext.read(path)
    return [cell["source"].strip() for cell in notebook.cells]


def check_sync(pairs: list[tuple[Path, Path]]) -> int:
    """Report any ``.ipynb`` whose cells differ from its ``.py``."""
    problems = 0

    for source, notebook in pairs:
        if not notebook.exists():
            print(f"MISSING  {notebook.name}: no committed notebook for {source.name}")
            problems += 1
            continue

        from_source = cell_sources(source)
        committed = cell_sources(notebook)

        if from_source == committed:
            print(f"ok       {notebook.name}: {len(committed)} cells in step")
            continue

        problems += 1
        if len(from_source) != len(committed):
            print(
                f"STALE    {notebook.name}: {len(from_source)} cells in "
                f"{source.name}, {len(committed)} in the notebook"
            )
        else:
            differing = [
                index
                for index, (a, b) in enumerate(zip(from_source, committed, strict=True))
                if a != b
            ]
            print(
                f"STALE    {notebook.name}: cells {differing} differ from {source.name}"
            )

    return problems


def check_execute(pairs: list[tuple[Path, Path]]) -> int:
    """Re-run each notebook from its source and report any that fail.

    The result goes to stdout and is discarded, which keeps this from touching
    the committed notebooks. Writing to a file instead would also move the
    kernel: jupytext runs it in the output's directory, so an executed copy sent
    to a temporary folder broke the flood notebook's relative path to
    ``data/``. Sending it nowhere leaves the working directory where the
    notebooks expect it.
    """
    problems = 0

    for source, _notebook in pairs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "ipynb",
                "--execute",
                "--output",
                "-",
                source.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(NOTEBOOKS),
            check=False,
        )
        if result.returncode == 0:
            print(f"ok       {source.name}: runs")
        else:
            problems += 1
            tail = (result.stderr or "").strip().splitlines()[-6:]
            print(f"FAILED   {source.name}:")
            for line in tail:
                print(f"           {line}")

    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the requested checks and return a shell exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="check each .ipynb matches its .py (fast)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="re-run each notebook and fail if a cell raises (slow)",
    )
    args = parser.parse_args(argv)

    if not (args.sync or args.execute):
        args.sync = True

    pairs = paired_notebooks()
    if not pairs:
        print(f"No notebooks found under {NOTEBOOKS}.", file=sys.stderr)
        return 1

    problems = 0
    if args.sync:
        print(f"Checking {len(pairs)} notebooks are in step with their sources")
        problems += check_sync(pairs)
        print()

    if args.execute:
        print(f"Re-running {len(pairs)} notebooks")
        problems += check_execute(pairs)
        print()

    if problems:
        print(f"{problems} problem(s). Regenerate with:")
        print("  jupytext --to ipynb --execute notebooks/<name>.py")
        return 1

    # Only claim what was actually checked.
    checked = " and ".join(
        part for part, ran in (("in step", args.sync), ("running", args.execute)) if ran
    )
    print(f"All notebooks {checked}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
