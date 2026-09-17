import subprocess
import sys

# Install CUDA 12.6 PyTorch
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch==2.6.0+cu126",
    "torchvision==0.21.0+cu126",
    "torchaudio==2.6.0+cu126",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu126"
])

# Install remaining packages
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "joblib",
    "pyyaml",
    "torch-geometric",
    "opacus",
    "networkx",
    "fastapi",
    "uvicorn[standard]",
    "flwr"
])

print("\nAll packages installed successfully!")