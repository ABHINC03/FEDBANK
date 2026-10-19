"""
gpu_env.py — GPU allocator setup, VRAM helpers.
FEDBANK_preprocessing_spec.md Section 2.
"""
import os
import sys

if "CUDA_PATH" not in os.environ:
    prefix = sys.prefix
    candidate = os.path.join(prefix, "targets", "x86_64-linux")
    if os.path.exists(os.path.join(candidate, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = candidate
    elif os.path.exists(os.path.join(prefix, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = prefix

os.environ.setdefault("CUDF_SPILL", "on")

import gc
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator


def init_gpu():
    """Initialize RMM + shared cuPy allocator. managed_memory=False is REQUIRED on WSL2."""
    rmm.reinitialize(pool_allocator=False, managed_memory=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # In WSL2, pylibcudf parquet writer with kvikio posix_io causes
    # CUDA_ERROR_ILLEGAL_ADDRESS due to virtualized device pointers.
    # Route to_parquet safely through pyarrow.
    import cudf
    def _safe_to_parquet(self, path, *args, **kwargs):
        kwargs.pop("engine", None)
        compression = kwargs.pop("compression", "snappy")
        index = kwargs.pop("index", False)
        self.to_pandas().to_parquet(path, engine="pyarrow", compression=compression, index=index, **kwargs)

    cudf.DataFrame.to_parquet = _safe_to_parquet


def free_gpu():
    """Release cached GPU blocks."""
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def vram(tag: str = "") -> int:
    """Print and return free VRAM bytes."""
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[VRAM]{tag} used {(total - free) / 2**30:.2f} / {total / 2**30:.2f} GiB  "
          f"(free {free / 2**30:.2f} GiB)")
    return free


def check_vram(min_mb: int = 600, tag: str = ""):
    """Abort if free VRAM is below min_mb."""
    free_bytes = vram(tag)
    if free_bytes < min_mb * 2**20:
        raise MemoryError(
            f"[VRAM] Only {free_bytes / 2**20:.0f} MB free before stage{tag}; "
            f"need ≥ {min_mb} MB. Free GPU memory and retry."
        )
