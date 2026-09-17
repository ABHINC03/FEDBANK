#!/bin/bash

set -e

echo "=========================================="
echo " FedAgent-Bank WSL2 Environment Setup"
echo "=========================================="

# ------------------------------------------
# 1. Update Ubuntu
# ------------------------------------------

echo ""
echo "[1/7] Updating Ubuntu..."

sudo apt update
sudo apt upgrade -y

# ------------------------------------------
# 2. Install required system packages
# ------------------------------------------

echo ""
echo "[2/7] Installing Git and required tools..."

sudo apt install -y git wget curl build-essential

echo "Git version:"
git --version

# ------------------------------------------
# 3. Install Miniconda
# ------------------------------------------

echo ""
echo "[3/7] Checking Miniconda..."

if [ -d "$HOME/miniconda3" ]; then
    echo "Miniconda is already installed."
else
    echo "Downloading Miniconda..."

    wget -O "$HOME/miniconda.sh" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    echo "Installing Miniconda..."

    bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"

    rm "$HOME/miniconda.sh"
fi

# Initialize Conda for the current shell
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Initialize Conda for future shells
"$HOME/miniconda3/bin/conda" init bash

echo "Conda version:"
conda --version

# ------------------------------------------
# 4. Create fedbank environment
# ------------------------------------------

echo ""
echo "[4/7] Creating fedbank Conda environment..."

if conda env list | grep -qE "^fedbank[[:space:]]"; then
    echo "fedbank environment already exists."
else
    conda create -n fedbank python=3.11 -y
fi

conda activate fedbank

echo "Python version:"
python --version

# ------------------------------------------
# 5. Verify NVIDIA GPU
# ------------------------------------------

echo ""
echo "[5/7] Checking NVIDIA GPU..."

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo ""
    echo "WARNING: nvidia-smi was not found."
    echo "Please make sure NVIDIA drivers and WSL2 GPU support are installed."
    echo ""
fi

# ------------------------------------------
# 6. Install RAPIDS libraries
# ------------------------------------------

echo ""
echo "[6/7] Installing RAPIDS libraries..."

echo "Installing cuML..."

conda install -c rapidsai -c conda-forge cuml -y

echo "Installing cuDF..."

conda install -c rapidsai -c conda-forge cudf -y

echo "Installing cuGraph..."

conda install -c rapidsai -c conda-forge cugraph -y

# ------------------------------------------
# 7. Verify installation
# ------------------------------------------

echo ""
echo "[7/7] Verifying installations..."

echo ""
echo "Python:"
python --version

echo ""
echo "cuML:"
python -c "import cuml; print(cuml.__version__)"

echo ""
echo "cuDF:"
python -c "import cudf; print(cudf.__version__)"

echo ""
echo "cuGraph:"
python -c "import cugraph; print(cugraph.__version__)"

echo ""
echo "Testing cuML KNN..."

python -c "from cuml.neighbors import NearestNeighbors; print('cuML KNN: OK')"

echo ""
echo "=========================================="
echo " Setup completed successfully!"
echo "=========================================="

echo ""
echo "Activate the environment using:"
echo ""
echo "    conda activate fedbank"
echo ""

echo "Project environment:"
echo "    Python 3.11"
echo "    cuML"
echo "    cuDF"
echo "    cuGraph"
echo ""