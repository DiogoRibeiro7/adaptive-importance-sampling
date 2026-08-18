#!/usr/bin/env python
"""
pyproject_editor.py

A small, typed CLI to update `pyproject.toml` safely:
- Works with both Poetry (`[tool.poetry]`) and PEP 621 (`[project]`).
- Preserves formatting and comments via tomlkit.
- Subcommands:
    * bump-version {major,minor,patch}
    * set-dep <name> <spec> [--group <group>]
    * remove-dep <name> [--group <group>]
    * set-python <spec>
    * print (pretty-prints the effective model)

Examples:
    python scripts/pyproject_editor.py bump-version patch
    python scripts/pyproject_editor.py set-dep numpy "^1.26"
    python scripts/pyproject_editor.py set-dep ruff "^0.5" --group dev
    python scripts/pyproject_editor.py remove-dep numpy
    python scripts/pyproject_editor.py set-python ">=3.10,<3.13"
    python scripts/pyproject_editor.py print --check
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Literal

import tomlkit
from tomlkit.toml_document import TOMLDocument

# The file being edited contains mathematical notation, so stdout must be
# able to carry it on consoles that default to a legacy code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VersionBump = Literal["major", "minor", "patch"]

# Constants
DEFAULT_PYPROJECT_PATH = "pyproject.toml"
VERSION_PATTERN = re.compile(r"^(?P<maj>\d+)\.(?P<min>\d+)\.(?P<pat>\d+)(?P<rest>.*)?$")


@dataclass(frozen=True)
class Paths:
    root: Path
    pyproject: Path


def _ensure_file(p: Path) -> None:
    """
    Verify that a path exists and is a file.

    Args:
        p: Path to check

    Raises:
        FileNotFoundError: If the file doesn't exist
        IsADirectoryError: If the path is a directory
    """
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Expected a file, got directory: {p}")


def _load_doc(path: Path) -> TOMLDocument:
    """
    Load a TOML document from a file.

    Args:
        path: Path to the TOML file

    Returns:
        Parsed TOML document

    Raises:
        UnicodeDecodeError: If the file cannot be decoded as UTF-8
        tomlkit.exceptions.TOMLKitError: If the TOML syntax is invalid
    """
    text = path.read_text(encoding="utf-8")
    return tomlkit.parse(text)


def _dump_doc(doc: TOMLDocument) -> str:
    """
    Serialize a TOML document to a string.

    Args:
        doc: TOML document to serialize

    Returns:
        Formatted TOML string with comments preserved
    """
    # tomlkit returns a string preserving formatting/comments
    return tomlkit.dumps(doc)


def _detect_layout(doc: TOMLDocument) -> Literal["poetry", "pep621"]:
    """
    Return which layout the pyproject uses for metadata:
    - "pep621" if [project] carries the metadata
    - "poetry" if [tool.poetry] does

    Poetry 2.x projects commonly have both: [project] holds the metadata while
    [tool.poetry] is reduced to build settings such as `packages`. So [project]
    wins whenever it is present and actually declares a version, and we only
    fall back to [tool.poetry] for the older layout.

    Args:
        doc: TOML document to check

    Returns:
        Layout type ("poetry" or "pep621")

    Raises:
        ValueError: If neither layout is found
    """
    project = doc.get("project")
    has_project_version = isinstance(project, dict) and "version" in project
    tool = doc.get("tool")
    has_poetry = isinstance(tool, dict) and "poetry" in tool

    if has_project_version:
        return "pep621"
    if has_poetry:
        return "poetry"
    if project is not None:
        return "pep621"
    raise ValueError(
        "Unsupported pyproject: neither [tool.poetry] nor [project] found."
    )


def _get_version(doc: TOMLDocument) -> str:
    """
    Get the version from the document, respecting the layout.

    Args:
        doc: TOML document to extract version from

    Returns:
        Version string

    Raises:
        KeyError: If the version field is missing
        TypeError: If the version is not a string
    """
    layout = _detect_layout(doc)
    try:
        if layout == "poetry":
            v = doc["tool"]["poetry"].get("version")
        else:
            v = doc["project"].get("version")
    except KeyError as exc:
        raise KeyError(f"Version field not found in [{layout}] section") from exc

    if v is None:
        raise KeyError(f"Version field not found in [{layout}] section")
    if not isinstance(v, str):
        raise TypeError(f"Version must be a string, got {type(v).__name__}")
    return v


def _set_version(doc: TOMLDocument, version: str) -> None:
    """
    Set the version in the document, respecting the layout.

    Args:
        doc: TOML document to modify
        version: New version string to set

    Raises:
        KeyError: If required sections are missing
    """
    layout = _detect_layout(doc)
    try:
        if layout == "poetry":
            doc["tool"]["poetry"]["version"] = version
        else:
            doc["project"]["version"] = version
    except KeyError as e:
        raise KeyError(f"Failed to set version: {e}") from e


def _bump_semver(v: str, level: VersionBump) -> str:
    """
    Minimal semantic version bump for MAJOR.MINOR.PATCH[rest].
    'rest' (like -rc.1) is dropped on bump to keep things predictable.

    Args:
        v: Version string in semver format
        level: Which part to bump (major, minor, or patch)

    Returns:
        New version string

    Raises:
        ValueError: If the version string doesn't match MAJOR.MINOR.PATCH pattern
        ValueError: If the bump level is invalid
    """
    m = VERSION_PATTERN.match(v)
    if not m:
        raise ValueError(f"Version '{v}' is not in MAJOR.MINOR.PATCH[suffix] format.")

    try:
        maj, min_, pat = int(m["maj"]), int(m["min"]), int(m["pat"])
    except ValueError as exc:
        raise ValueError(f"Version components must be integers in '{v}'") from exc

    if level == "major":
        maj, min_, pat = maj + 1, 0, 0
    elif level == "minor":
        min_, pat = min_ + 1, 0
    elif level == "patch":
        pat = pat + 1
    else:
        raise ValueError(
            f"Unknown bump level: {level}. Must be major, minor, or patch."
        )

    return f"{maj}.{min_}.{pat}"


# Files that restate the version and must not drift from pyproject.toml.
# Each entry is (path relative to the project root, regex, replacement template).
VERSION_COMPANIONS: tuple[tuple[str, str, str], ...] = (
    ("CITATION.cff", r"(?m)^version: .+$", "version: {version}"),
    (".zenodo.json", r'(?m)^  "version": "[^"]+",$', '  "version": "{version}",'),
    (
        "conda.recipe/meta.yaml",
        r'(?m)^\{% set version = "[^"]+" %\}$',
        '{{% set version = "{version}" %}}',
    ),
)


def _sync_version_companions(pyproject: Path, version: str, check: bool) -> list[str]:
    """Rewrite the version in files that restate it, and report which changed.

    A bump that touches only pyproject.toml leaves CITATION.cff and the conda
    recipe claiming the previous release, which is the kind of drift that is
    only noticed after the tag is published.
    """
    changed: list[str] = []
    root = pyproject.parent

    for relative_path, pattern, template in VERSION_COMPANIONS:
        path = root / relative_path
        if not path.exists():
            continue

        before = path.read_text(encoding="utf-8")
        after = re.sub(pattern, template.format(version=version), before)
        if before == after:
            continue

        changed.append(relative_path)
        if not check:
            path.write_text(after, encoding="utf-8")

    return changed


def _get_poetry_dep_table(doc: TOMLDocument, group: str | None) -> tomlkit.items.Table:
    """
    Return a writable table for dependencies in Poetry layout.
    Supports:
      - main deps: [tool.poetry.dependencies]
      - dev deps (legacy): [tool.poetry.dev-dependencies]
      - named groups: [tool.poetry.group.<group>.dependencies]

    Args:
        doc: TOML document to get table from
        group: Group name or None for main dependencies

    Returns:
        Table object for the dependencies
    """
    tool = doc.setdefault("tool", tomlkit.table())
    poetry = tool.setdefault("poetry", tomlkit.table())

    if group is None or group == "main":
        return poetry.setdefault("dependencies", tomlkit.table())

    # Handle "dev" group - can be either legacy dev-dependencies or group.dev
    if group == "dev" and "dev-dependencies" in poetry:
        return poetry.setdefault("dev-dependencies", tomlkit.table())

    # Poetry 1.2+ group
    group_tbl = poetry.setdefault("group", tomlkit.table())
    g = group_tbl.setdefault(group, tomlkit.table())
    return g.setdefault("dependencies", tomlkit.table())


def _ensure_pep621_arrays(
    doc: TOMLDocument,
) -> tuple[tomlkit.items.Array, tomlkit.items.Table]:
    """
    Return (project.dependencies Array, project.optional-dependencies Table).
    Create them if missing.

    Args:
        doc: TOML document to get arrays from

    Returns:
        Tuple of (dependencies array, optional-dependencies table)

    Raises:
        TypeError: If the dependencies is not an array or optional-dependencies is not a table
    """
    project = doc.setdefault("project", tomlkit.table())
    deps = project.setdefault("dependencies", tomlkit.array())
    opt = project.setdefault("optional-dependencies", tomlkit.table())

    # Typing guards:
    if not isinstance(deps, tomlkit.items.Array):
        raise TypeError("[project.dependencies] must be an array.")
    if not isinstance(opt, tomlkit.items.Table):
        raise TypeError("[project.optional-dependencies] must be a table.")
    return deps, opt


def _set_dep(doc: TOMLDocument, name: str, spec: str, group: str | None) -> None:
    """
    Set or update a dependency across layouts.
    - Poetry: name = {version = spec} or name = spec (string)
    - PEP 621: strings like "name spec" in arrays

    Args:
        doc: TOML document to modify
        name: Package name
        spec: Version specification
        group: Group name or None for main dependencies

    Raises:
        TypeError: For PEP 621 projects if arrays/tables aren't the right type
    """
    if not name:
        raise ValueError("Dependency name cannot be empty")
    if not spec:
        raise ValueError("Version specification cannot be empty")

    layout = _detect_layout(doc)
    if layout == "poetry":
        deps = _get_poetry_dep_table(doc, group)
        # Use simple string pin if spec looks like a simple constraint; Poetry supports both forms.
        deps[name] = spec
        return

    # PEP 621
    deps, opt = _ensure_pep621_arrays(doc)
    target_array = (
        deps
        if (group is None or group == "main")
        else opt.setdefault(group, tomlkit.array())
    )
    if not isinstance(target_array, tomlkit.items.Array):
        raise TypeError(f"[project.optional-dependencies.{group}] must be an array.")

    # Replace existing line if package already present
    pk_prefix = f"{name} "
    new_line = f"{name} {spec}"
    found_idx = None
    for i, item in enumerate(
        list(target_array)
    ):  # make a copy to avoid iterator issues
        if isinstance(item, str) and (item == name or item.startswith(pk_prefix)):
            found_idx = i
            break
    if found_idx is not None:
        target_array[found_idx] = new_line
    else:
        target_array.append(new_line)


def _remove_dep(doc: TOMLDocument, name: str, group: str | None) -> bool:
    """
    Remove a dependency. Returns True if removed, False if not found.

    Args:
        doc: TOML document to modify
        name: Package name to remove
        group: Group name or None for main dependencies

    Returns:
        True if dependency was found and removed, False otherwise
    """
    if not name:
        raise ValueError("Dependency name cannot be empty")

    layout = _detect_layout(doc)
    if layout == "poetry":
        # For Poetry, handle both legacy dev-dependencies and new group structure
        if group == "dev":
            # Check both legacy and new structure
            legacy_deps = (
                doc.get("tool", {}).get("poetry", {}).get("dev-dependencies", {})
            )
            if name in legacy_deps:
                del legacy_deps[name]
                return True

            # Try new group.dev structure
            group_deps = (
                doc.get("tool", {})
                .get("poetry", {})
                .get("group", {})
                .get("dev", {})
                .get("dependencies", {})
            )
            if name in group_deps:
                del group_deps[name]
                return True
            return False

        # Regular group or main dependencies
        deps = _get_poetry_dep_table(doc, group)
        if name in deps:
            del deps[name]
            return True
        return False

    # PEP 621
    deps, opt = _ensure_pep621_arrays(doc)
    arrays = []
    if group is None or group == "main":
        arrays.append(deps)
    elif group in opt:
        arrays.append(opt.get(group))

    removed = False
    for arr in arrays:
        if not isinstance(arr, tomlkit.items.Array):
            continue
        keep = []
        for item in arr:
            if isinstance(item, str) and (item == name or item.startswith(f"{name} ")):
                removed = True
            else:
                keep.append(item)
        # Replace contents
        while len(arr) > 0:
            arr.pop()
        for k in keep:
            arr.append(k)
    return removed


def _set_python_constraint(doc: TOMLDocument, spec: str) -> None:
    """
    Set Python constraint:
    - Poetry: [tool.poetry.dependencies].python = "<spec>"
    - PEP 621: [project].requires-python = "<spec>"

    Args:
        doc: TOML document to modify
        spec: Python version specification

    Raises:
        ValueError: If spec is empty
    """
    if not spec:
        raise ValueError("Python version specification cannot be empty")

    layout = _detect_layout(doc)
    if layout == "poetry":
        tool = doc.setdefault("tool", tomlkit.table())
        poetry = tool.setdefault("poetry", tomlkit.table())
        deps = poetry.setdefault("dependencies", tomlkit.table())
        deps["python"] = spec
        return

    project = doc.setdefault("project", tomlkit.table())
    project["requires-python"] = spec


def _write_or_diff(path: Path, old_text: str, new_text: str, check: bool) -> int:
    """
    Write new content to a file or print a diff if check mode is enabled.

    Args:
        path: Path to write to
        old_text: Original file content
        new_text: New file content
        check: If True, print diff instead of writing

    Returns:
        0 for success

    Raises:
        PermissionError: If the file cannot be written due to permissions
        OSError: For other file system related errors
    """
    if check:
        diff = "".join(
            unified_diff(
                old_text.splitlines(True),
                new_text.splitlines(True),
                fromfile=str(path),
                tofile=str(path),
            )
        )
        sys.stdout.write(diff)
        return 0

    try:
        path.write_text(new_text, encoding="utf-8")
    except (PermissionError, OSError) as e:
        print(f"Error writing to {path}: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the script.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(description="Safely edit pyproject.toml")
    parser.add_argument(
        "--file", default=DEFAULT_PYPROJECT_PATH, help="Path to pyproject.toml"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; print a unified diff of proposed changes.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bump-version", help="Bump semantic version")
    b.add_argument("level", choices=["major", "minor", "patch"], type=str)

    sd = sub.add_parser("set-dep", help="Set or update a dependency")
    sd.add_argument("name", type=str)
    sd.add_argument(
        "spec", type=str, help='Version constraint, e.g. "^1.26" or ">=1.0,<2.0"'
    )
    sd.add_argument(
        "--group",
        type=str,
        default=None,
        help='Dependency group (Poetry: "dev", or PEP 621 optional group)',
    )

    rd = sub.add_parser("remove-dep", help="Remove a dependency")
    rd.add_argument("name", type=str)
    rd.add_argument("--group", type=str, default=None)

    sp = sub.add_parser("set-python", help="Set Python version constraint")
    sp.add_argument("spec", type=str, help='e.g. ">=3.10,<3.13"')

    sub.add_parser("print", help="Pretty-print normalized metadata")

    args = parser.parse_args(argv)

    try:
        pyproject = Path(args.file)
        _ensure_file(pyproject)
        before = pyproject.read_text(encoding="utf-8")
        doc = _load_doc(pyproject)

        if args.cmd == "bump-version":
            current = _get_version(doc)
            new = _bump_semver(current, args.level)  # type: ignore[arg-type]
            _set_version(doc, new)
            print(f"Bumped version: {current} → {new}")

            synced = _sync_version_companions(pyproject, new, args.check)
            for relative_path in synced:
                print(f"  also updated {relative_path}")

        elif args.cmd == "set-dep":
            _set_dep(doc, args.name.strip(), args.spec.strip(), args.group)
            print(
                f"Set dependency: {args.name} to {args.spec}"
                + (f" in group '{args.group}'" if args.group else "")
            )

        elif args.cmd == "remove-dep":
            removed = _remove_dep(doc, args.name.strip(), args.group)
            if not removed:
                print(f"Dependency '{args.name}' not found.", file=sys.stderr)
                return 1
            print(
                f"Removed dependency: {args.name}"
                + (f" from group '{args.group}'" if args.group else "")
            )

        elif args.cmd == "set-python":
            _set_python_constraint(doc, args.spec.strip())
            print(f"Set Python constraint: {args.spec}")

        elif args.cmd == "print":
            # For now, we just re-dump (preserves comments). Could add normalization.
            sys.stdout.write(_dump_doc(doc))
            return 0

        else:
            parser.error("Unknown command")  # unreachable with argparse choices

        after = _dump_doc(doc)
        if before == after:
            print("No changes were needed.")
            # Nothing changed; still show diff in --check mode for transparency.
            return _write_or_diff(pyproject, before, after, args.check)

        return _write_or_diff(pyproject, before, after, args.check)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
