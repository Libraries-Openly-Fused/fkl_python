"""Compiler backend: turns a generated .cu into a cached .so.

Two interchangeable backends (option 1 + option 2 from the design):
  - "clang": clang++ -x cuda  (single step, full device std lib, Apache-2.0,
             no NVIDIA EULA in the bundle). Preferred when clang is present.
  - "nvcc" : nvcc shared lib  (fallback; ships with CUDA SDK).

Both compile host+device together and expose extern "C" fkl_entry. The result
is cached on disk keyed by hash(signature + fkl version + arch + cuda version),
so each unique chain is compiled exactly once, ever.
"""
from __future__ import annotations
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---- configuration -------------------------------------------------------

def _resolve_fkl_include() -> str:
    """FKL headers resolution, in priority order:
    1. FKL_INCLUDE env var (explicit override / developer checkout)
    2. vendored headers shipped inside the wheel (fkl/_vendor/...)
    3. well-known sibling checkout (development convenience)
    """
    env = os.environ.get("FKL_INCLUDE")
    if env:
        return env
    vendored = Path(__file__).parent / "_vendor" / "FusedKernelLibrary" / "include"
    if (vendored / "fused_kernel" / "fused_kernel.h").exists():
        return str(vendored)
    dev = Path.home() / "Projects" / "oscar" / "FusedKernelLibrary" / "include"
    if (dev / "fused_kernel" / "fused_kernel.h").exists():
        return str(dev)
    raise RuntimeError(
        "FusedKernelLibrary headers not found. Either:\n"
        "  - pip install a wheel with vendored headers (run scripts/vendor_fkl.py"
        " before building), or\n"
        "  - set FKL_INCLUDE to <FusedKernelLibrary>/include")


_FKL_INCLUDE = None  # resolved lazily on first compile (import must not fail)


def fkl_include() -> str:
    global _FKL_INCLUDE
    if _FKL_INCLUDE is None:
        _FKL_INCLUDE = _resolve_fkl_include()
    return _FKL_INCLUDE


def fkl_root() -> str:
    # FKL_ROOT is only needed for in-repo test utilities (<tests/main.h>);
    # generated chains only need include/. Default: include's parent.
    return os.environ.get("FKL_ROOT", str(Path(fkl_include()).parent))


def _detect_cuda_home() -> str:
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        return env
    if Path("/usr/local/cuda").exists():
        return "/usr/local/cuda"
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).parent.parent)
    return "/usr/local/cuda"  # last resort; clear error surfaces at compile


_CUDA_HOME = _detect_cuda_home()
_CACHE_DIR = Path(os.environ.get("FKL_CACHE", str(Path.home() / ".cache" / "fkl")))
def _detect_arch() -> str:
    """GPU arch resolution for plug-and-play installs:
    1. FKL_ARCH env var (explicit override)
    2. compute capability of GPU 0 via the CUDA driver API (ctypes; no
       dependencies; works without nvcc/nvidia-smi on PATH)
    3. sm_75 floor as a last resort (compiles everywhere Turing+)
    """
    env = os.environ.get("FKL_ARCH")
    if env:
        return env
    try:
        import ctypes
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) == 0:
            major, minor, dev = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            if cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)   # COMPUTE_CAPABILITY_MAJOR
                cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)   # COMPUTE_CAPABILITY_MINOR
                if major.value > 0:
                    return f"sm_{major.value}{minor.value}"
    except Exception:
        pass
    return "sm_75"


_ARCH = _detect_arch()
_STD = os.environ.get("FKL_STD", "c++20")


def _cuda_version() -> str:
    try:
        out = subprocess.check_output(
            [str(Path(_CUDA_HOME) / "bin" / "nvcc"), "--version"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[1].split(",")[0].strip()
    except Exception:
        pass
    return "unknown"


def _fkl_version() -> str:
    # cheap content fingerprint of the public header to invalidate cache on bumps
    h = hashlib.sha1()
    fk_h = Path(fkl_include()) / "fused_kernel" / "fused_kernel.h"
    try:
        h.update(fk_h.read_bytes())
    except Exception:
        h.update(b"unknown")
    return h.hexdigest()[:12]


class CompilerBackend:
    CLANG = "clang"
    NVCC = "nvcc"
    CPU = "cpu"      # plain C++ (clang++/g++), ParArch::CPU, no CUDA at all

    def __init__(self, kind: str | None = None):
        self.kind = kind or self._auto()
        self.cache_dir = _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _auto() -> str:
        # Remember a prior clang->nvcc fallback for this toolkit so we don't
        # re-probe clang (which can cost ~400ms) on every fresh process.
        marker = _CACHE_DIR / ".backend"
        try:
            pref = marker.read_text().strip()
            if pref in (CompilerBackend.CLANG, CompilerBackend.NVCC):
                return pref
        except Exception:
            pass
        # clang preferred (single step, full std lib, no EULA in bundle)
        if shutil.which("clang++"):
            return CompilerBackend.CLANG
        if (Path(_CUDA_HOME) / "bin" / "nvcc").exists() or shutil.which("nvcc"):
            return CompilerBackend.NVCC
        raise RuntimeError("no CUDA compiler found (need clang++ or nvcc)")

    def _remember(self):
        try:
            (self.cache_dir / ".backend").write_text(self.kind)
        except Exception:
            pass

    # ---- cache key -------------------------------------------------------
    def cache_key(self, sig: str) -> str:
        raw = f"{sig};fkl={_fkl_version()};cuda={_cuda_version()};{self.kind};{_STD}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def so_path(self, sig: str) -> Path:
        return self.cache_dir / f"fkl_{self.cache_key(sig)}.so"

    # ---- compile ---------------------------------------------------------
    def compile(self, cu_source: str, sig: str, force: bool = False) -> Path:
        so = self.so_path(sig)
        if so.exists() and not force:
            return so  # cache HIT -- compiled once, ever

        src_path = so.with_suffix(".cu")
        src_path.write_text(cu_source)

        if self.kind == self.CPU:
            cmd = self._cpu_cmd(src_path, so)
        elif self.kind == self.CLANG:
            cmd = self._clang_cmd(src_path, so)
        else:
            cmd = self._nvcc_cmd(src_path, so)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Automatic fallback: clang's CUDA support can lag the installed CUDA
        # toolkit (e.g. clang-21 vs CUDA 13.3). If clang fails, retry with nvcc
        # transparently so the user never has to care.
        if (proc.returncode != 0 or not so.exists()) and self.kind == self.CLANG:
            nvcc_avail = (Path(_CUDA_HOME) / "bin" / "nvcc").exists() or shutil.which("nvcc")
            if nvcc_avail:
                self.kind = self.NVCC
                so = self.so_path(sig)
                if so.exists() and not force:
                    return so
                src_path = so.with_suffix(".cu")
                src_path.write_text(cu_source)
                proc = subprocess.run(self._nvcc_cmd(src_path, so),
                                      capture_output=True, text=True)

        if proc.returncode != 0 or not so.exists():
            errs = "\n".join(
                l for l in proc.stderr.splitlines()
                if "error" in l.lower() or "Error" in l
            ) or proc.stderr[-2000:]
            raise RuntimeError(
                f"FKL JIT compile failed ({self.kind}).\n"
                f"source kept at: {src_path}\n--- errors ---\n{errs}"
            )
        self._remember()
        return so

    def _cpu_cmd(self, src: Path, so: Path):
        # FKL's CPU path compiles as plain C++. NOTE: g++ rejects the
        # Stream_<ParArch::CPU>() constructor spelling in stream.h; clang++
        # accepts it, so CPU mode requires clang++.
        cxx = shutil.which("clang++")
        if cxx is None:
            raise RuntimeError("CPU backend requires clang++ (g++ chokes on "
                               "Stream_<ParArch::CPU>'s ctor spelling)")
        cpp = src.with_suffix(".cpp")
        if not cpp.exists():
            cpp.write_text(src.read_text())
        return [cxx, f"-std={_STD}", "-O2", "-shared", "-fPIC",
                "-I", fkl_include(), "-I", fkl_root(),
                str(cpp), "-o", str(so)]

    def _nvcc_cmd(self, src: Path, so: Path):
        nvcc = str(Path(_CUDA_HOME) / "bin" / "nvcc")
        return [
            nvcc, f"-std={_STD}", f"-arch={_ARCH}",
            "-shared", "-Xcompiler", "-fPIC",
            "-I", fkl_include(), "-I", fkl_root(),
            str(src), "-o", str(so),
        ]

    def _clang_cmd(self, src: Path, so: Path):
        shim = self._ensure_clang_shims()
        return [
            "clang++", "-x", "cuda", f"-std={_STD}",
            f"--cuda-gpu-arch={_ARCH}", f"--cuda-path={shim['cuda']}",
            "-isystem", str(shim["headers"]),
            # CUDA 13.3 + glibc>=2.42: header uses noexcept via macro that
            # clang's wrapper chain leaves undefined -> define it explicitly.
            "-D_NV_RSQRT_SPECIFIER=noexcept(true)",
            "-shared", "-fPIC",
            "-I", fkl_include(), "-I", fkl_root(),
            str(src), "-o", str(so),
            f"-L{_CUDA_HOME}/lib64", "-lcudart",
        ]

    def _ensure_clang_shims(self) -> dict:
        """clang<=21 lags CUDA 13: (a) it includes texture headers that CUDA 13
        removed; (b) it drives fatbinary with the removed --image= syntax.
        Build a shimmed cuda-path (symlinks + fatbinary CLI translator) and an
        empty-texture-header include dir. Idempotent, cached under FKL_CACHE."""
        shim_root = self.cache_dir / "clang_shim"
        cuda_shim = self.cache_dir / "clang_cuda"
        if not (cuda_shim / "bin" / "fatbinary").exists():
            shim_root.mkdir(parents=True, exist_ok=True)
            for h in ("texture_fetch_functions.h", "texture_indirect_functions.h"):
                (shim_root / h).write_text("// removed in CUDA 13; empty shim for clang\n")
            bind = cuda_shim / "bin"
            bind.mkdir(parents=True, exist_ok=True)
            cuda = Path(_CUDA_HOME)
            for d in ("include", "lib64", "nvvm", "targets", "version.json"):
                link = cuda_shim / d
                if not link.exists() and (cuda / d).exists():
                    link.symlink_to(cuda / d)
            for b in (cuda / "bin").iterdir():
                link = bind / b.name
                if not link.exists() and b.name != "fatbinary":
                    link.symlink_to(b)
            fb = bind / "fatbinary"
            fb.write_text(f"""#!/usr/bin/env bash
# auto-generated by fkl-python: translate clang's CUDA-12 fatbinary CLI to CUDA-13
REAL="{cuda}/bin/fatbinary"
args=()
for a in "$@"; do
  case "$a" in
    --image=profile=sm_*,file=*)
      spec="${{a#--image=profile=sm_}}"; sm="${{spec%%,*}}"; file="${{spec#*,file=}}"
      args+=("--image3=kind=elf,sm=${{sm}},file=${{file}}") ;;
    --image=profile=compute_*,file=*)
      spec="${{a#--image=profile=compute_}}"; sm="${{spec%%,*}}"; file="${{spec#*,file=}}"
      args+=("--image3=kind=ptx,sm=${{sm}},file=${{file}}") ;;
    *) args+=("$a") ;;
  esac
done
exec "$REAL" "${{args[@]}}"
""")
            fb.chmod(0o755)
        return {"cuda": cuda_shim, "headers": shim_root}


# ---- module-level singleton ---------------------------------------------
_backend: CompilerBackend | None = None


def get_backend() -> CompilerBackend:
    global _backend
    if _backend is None:
        _backend = CompilerBackend()
    return _backend


def set_backend(kind: str):
    global _backend
    _backend = CompilerBackend(kind)


def clear_cache():
    import glob
    for f in glob.glob(str(_CACHE_DIR / "fkl_*")):
        os.remove(f)
