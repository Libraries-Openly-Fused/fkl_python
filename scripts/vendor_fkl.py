#!/usr/bin/env python3
"""Vendor the FusedKernelLibrary headers into fkl/_vendor/ so the wheel is
self-contained: `pip install fkl-python` works with no FKL checkout and no
FKL_INCLUDE/FKL_ROOT environment variables.

FKL is header-only and Apache-2.0, same license as this package, so
redistributing the headers inside the wheel is clean. The LICENSE and the
exact upstream commit are recorded alongside.

Usage:
    python scripts/vendor_fkl.py [--source /path/to/FusedKernelLibrary]
                                 [--ref LTS-C++17]

With --source, copies from a local checkout (records its HEAD commit).
Without it, shallow-clones the given ref from GitHub into a temp dir.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/Libraries-Openly-Fused/FusedKernelLibrary.git"
PKG_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PKG_ROOT / "fkl" / "_vendor" / "FusedKernelLibrary"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help="local FusedKernelLibrary checkout to copy from")
    ap.add_argument("--ref", default="LTS-C++17",
                    help="git ref to clone when --source is not given")
    args = ap.parse_args()

    tmp = None
    if args.source is not None:
        src = args.source.resolve()
    else:
        tmp = tempfile.mkdtemp(prefix="fkl_vendor_")
        subprocess.run(["git", "clone", "--depth", "1", "--branch", args.ref,
                        REPO_URL, tmp], check=True)
        src = Path(tmp)

    inc = src / "include"
    lic = src / "LICENSE"
    if not (inc / "fused_kernel" / "fused_kernel.h").exists():
        print(f"error: {inc} does not look like FKL's include dir", file=sys.stderr)
        return 1

    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(src), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()

    if VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
    VENDOR_DIR.mkdir(parents=True)
    shutil.copytree(inc, VENDOR_DIR / "include")
    if lic.exists():
        shutil.copy2(lic, VENDOR_DIR / "LICENSE")
    (VENDOR_DIR / "VENDOR_INFO.txt").write_text(
        f"FusedKernelLibrary headers vendored for self-contained wheels.\n"
        f"upstream: {REPO_URL}\nbranch: {branch}\ncommit: {commit}\n"
        f"license: Apache-2.0 (see LICENSE in this directory)\n")
    # package marker so setuptools treats it as data within the fkl package
    (PKG_ROOT / "fkl" / "_vendor" / "__init__.py").touch()

    n = sum(1 for _ in (VENDOR_DIR / "include").rglob("*.h"))
    print(f"vendored {n} headers from {branch}@{commit[:9]} -> {VENDOR_DIR}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
