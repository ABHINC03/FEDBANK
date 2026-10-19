"""
check_env.py — Environment verification (Section 1.1).
Run: python -m scripts.preprocess.check_env
"""
import os
os.environ.setdefault("CUDF_SPILL", "on")

import sys
import subprocess

print("python", sys.version)

import cupy as cp
import cudf
import cuml
import rmm

print(
    f"cudf {cudf.__version__} | cuml {cuml.__version__} | "
    f"cupy {cp.__version__} | rmm {rmm.__version__}"
)

free, total = cp.cuda.runtime.memGetInfo()
print(f"GPU free/total: {free/2**30:.2f} / {total/2**30:.2f} GiB")

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
    capture_output=True, text=True
)
print(result.stdout)

# ----- functional tests -----
s = cudf.Series([1.0, None, 3.0])
assert int(s.isna().sum()) == 1, "cuDF isna check failed"

from cuml.neighbors import NearestNeighbors
X = cp.random.rand(100, 4).astype("float32")
nn = NearestNeighbors(n_neighbors=3, algorithm="brute").fit(X)
nn.kneighbors(X[:5])

from cuml.decomposition import PCA
PCA(n_components=2).fit(X)

print("ENV OK")
