#!/usr/bin/env python
"""Version bumping script for Safe-ICE."""

import argparse
import re
import sys
from pathlib import Path


def get_current_version(root_dir: Path) -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = root_dir / "pyproject.toml"
    content = pyproject_path.read_text()

    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int, str | None]:
    """Parse version string into components."""
    # Match versions like 0.1.0, 0.1.0rc1, 0.1.0-beta.1
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[-.]?(.+))?$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = map(int, match.groups()[:3])
    suffix = match.group(4)

    return major, minor, patch, suffix


def format_version(
    major: int, minor: int, patch: int, suffix: str | None = None
) -> str:
    """Format version components into string."""
    version = f"{major}.{minor}.{patch}"
    if suffix:
        version = f"{version}-{suffix}"
    return version


def bump_version(current: str, bump_type: str, suffix: str | None = None) -> str:
    """Bump version based on bump type."""
    major, minor, patch, current_suffix = parse_version(current)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    elif bump_type == "prerelease":
        if current_suffix:
            # Increment prerelease version
            if "rc" in current_suffix:
                num = int(re.search(r"rc(\d+)", current_suffix).group(1))
                suffix = f"rc{num + 1}"
            elif "beta" in current_suffix:
                num = int(re.search(r"beta(\d+)", current_suffix).group(1))
                suffix = f"beta{num + 1}"
            elif "alpha" in current_suffix:
                num = int(re.search(r"alpha(\d+)", current_suffix).group(1))
                suffix = f"alpha{num + 1}"
            else:
                suffix = current_suffix
        else:
            # First prerelease
            suffix = suffix or "rc1"
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

    return format_version(major, minor, patch, suffix)


def update_file_version(
    file_path: Path,
    old_version: str,  # noqa: ARG001 - kept so callers can log the transition
    new_version: str,
) -> bool:
    """Update version in a file."""
    if not file_path.exists():
        return False

    content = file_path.read_text()

    # Different patterns for different files
    patterns = [
        (r'version = "[^"]+"', f'version = "{new_version}"'),
        (r"version='[^']+'", f"version='{new_version}'"),
        (r'__version__ = "[^"]+"', f'__version__ = "{new_version}"'),
        (r"version: [^\n]+", f"version: {new_version}"),
        (r"v\d+\.\d+\.\d+(?:[-.]?\w+)?", f"v{new_version}"),
    ]

    updated = False
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            updated = True

    if updated:
        file_path.write_text(content)

    return updated


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bump Safe-ICE version")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump",
    )
    parser.add_argument(
        "--suffix", help="Suffix for prerelease versions (e.g., rc1, beta1, alpha1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    args = parser.parse_args()

    # Find project root
    root_dir = Path(__file__).parent.parent

    # Get current version
    try:
        current_version = get_current_version(root_dir)
        print(f"Current version: {current_version}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Calculate new version
    try:
        new_version = bump_version(current_version, args.bump_type, args.suffix)
        print(f"New version: {new_version}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run - no files will be modified")
        print("\nFiles that would be updated:")
    else:
        print("\nUpdating files...")

    # Files to update
    files_to_update = [
        root_dir / "pyproject.toml",
        root_dir / "setup.py",
        root_dir / "safe_ice" / "__init__.py",
        root_dir / "docs" / "source" / "conf.py",
        root_dir / "conda.recipe" / "meta.yaml",
        root_dir / "Dockerfile",
        root_dir / ".github" / "workflows" / "release.yml",
    ]

    # Update versions
    updated_files = []
    for file_path in files_to_update:
        if args.dry_run:
            if file_path.exists():
                content = file_path.read_text()
                if re.search(r'version[=:"\s]+["\']?[0-9.]+', content):
                    print(f"  - {file_path.relative_to(root_dir)}")
        else:
            if update_file_version(file_path, current_version, new_version):
                updated_files.append(file_path)
                print(f"  ✓ Updated {file_path.relative_to(root_dir)}")

    if not args.dry_run:
        print(f"\nSuccessfully updated {len(updated_files)} files")
        print("\nNext steps:")
        print("1. Review the changes: git diff")
        print(f"2. Commit the changes: git commit -am 'Bump version to {new_version}'")
        print(f"3. Create a tag: git tag v{new_version}")
        print("4. Push changes: git push && git push --tags")

    return 0


if __name__ == "__main__":
    sys.exit(main())
