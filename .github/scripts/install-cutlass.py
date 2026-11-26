#!/usr/bin/env python3.11
"""Stage pinned CUTLASS v4.4.2 headers for sm_70 Volta builds without network access in CMake."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request

VERSION = ("4", "4", "2")
TAG = "v4.4.2"
URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/{TAG}.tar.gz"
SHA256 = "ef62816841b8cbcd0ed3ec45d3ab56cf67d569a0f39329b43eeeebea86f58473"


def verify_tree(destination: Path) -> bool:
    gemm = destination / "include/cutlass/gemm/device/gemm.h"
    version_header = destination / "include/cutlass/version.h"
    if not gemm.is_file() or not version_header.is_file():
        return False
    text = version_header.read_text(encoding="utf-8")
    parts = dict(re.findall(r"#define\s+CUTLASS_(MAJOR|MINOR|PATCH)\s+(\d+)", text))
    observed = tuple(parts.get(key) for key in ("MAJOR", "MINOR", "PATCH"))
    if observed != VERSION:
        raise ValueError(f"CUTLASS version {observed} at {destination}; expected {VERSION}")
    return True


def stage(destination: Path) -> Path:
    destination = destination.resolve()
    if verify_tree(destination):
        return destination
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Choose an empty destination or a CUTLASS {TAG} tree: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cutlass-stage-", dir=destination.parent) as tmp:
        work = Path(tmp)
        archive = work / f"cutlass-{TAG}.tar.gz"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
                with archive.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                        raise ValueError("CUTLASS archive SHA-256 mismatch")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5)
        with tarfile.open(archive, "r:gz") as tar:
            members = [
                member for member in tar.getmembers()
                if member.name.startswith("cutlass-4.4.2/include/") or
                member.name in ("cutlass-4.4.2/LICENSE.txt", "cutlass-4.4.2/README.md")
            ]
            tar.extractall(work, members=members, filter="data")
        extracted = work / "cutlass-4.4.2"
        if not verify_tree(extracted):
            raise RuntimeError("Extracted CUTLASS archive is missing expected headers")
        try:
            if destination.is_dir():
                destination.rmdir()  # Accept an empty pre-created directory.
            extracted.rename(destination)
        except OSError:
            # A concurrent install may have published the complete tree first.
            if not verify_tree(destination):
                raise
    return destination


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=repo_root / "third_party/cutlass" / TAG,
        help="Destination directory for unpacked CUTLASS tree",
    )
    args = parser.parse_args()
    installed = stage(args.destination)
    print(f"CUTLASS {'.'.join(VERSION)} ready at {installed}")


if __name__ == "__main__":
    main()
