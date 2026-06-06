#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_version(v):
    try:
        return tuple(int(x) for x in v.lstrip("v").split("."))
    except ValueError:
        print(f"[version-gate] ERROR: cannot parse version string: {v!r}")
        sys.exit(1)


def read_current_version():
    version_file = ROOT / "__version__.py"
    try:
        text = version_file.read_text()
    except FileNotFoundError:
        print(f"[version-gate] ERROR: {version_file} not found")
        sys.exit(1)

    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        print("[version-gate] ERROR: could not parse __version__ from __version__.py")
        sys.exit(1)

    return match.group(1)


def main():
    curr = parse_version(read_current_version())

    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
    )
    prev_tag = result.stdout.strip() if result.returncode == 0 else "v0.0.0"
    prev = parse_version(prev_tag)

    if curr <= prev:
        print(
            f"[version-gate] FAIL: bump __version__.py before pushing. "
            f"latest={'.'.join(map(str, prev))} current={'.'.join(map(str, curr))}"
        )
        sys.exit(1)

    print(f"[version-gate] OK: {'.'.join(map(str, prev))} -> {'.'.join(map(str, curr))}")


if __name__ == "__main__":
    main()
