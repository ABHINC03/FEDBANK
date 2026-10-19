# FEDBANK preprocessing package
import os
import sys

# Ensure CUDA headers path is configured for CuPy runtime compilation
if "CUDA_PATH" not in os.environ:
    prefix = sys.prefix
    candidate = os.path.join(prefix, "targets", "x86_64-linux")
    if os.path.exists(os.path.join(candidate, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = candidate
    elif os.path.exists(os.path.join(prefix, "include", "cuda.h")):
        os.environ["CUDA_PATH"] = prefix
