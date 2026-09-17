# FedAgent-Bank – WSL2, Conda and RAPIDS Setup

This guide describes how to set up the FedAgent-Bank development environment using WSL2, Ubuntu, Miniconda, NVIDIA GPU support, and RAPIDS libraries.

## 1. Install WSL2

Open **PowerShell as Administrator** in Windows:

    wsl --install

Restart the computer if prompted.

After restarting, open **Ubuntu** from the Windows Start menu.

Check WSL:

    wsl --status

Check Ubuntu:

    lsb_release -a

## 2. Update Ubuntu

Inside Ubuntu/WSL:

    sudo apt update
    sudo apt upgrade -y

## 3. Install Git and Required Tools

    sudo apt install -y git wget curl build-essential

Verify Git:

    git --version

## 4. Install Miniconda

Download Miniconda:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Install Miniconda:

    bash Miniconda3-latest-Linux-x86_64.sh

During installation:

1. Press Enter to read the license.
2. Enter `yes` to accept the license.
3. Press Enter to accept the default installation location.
4. When asked whether to initialize Conda, enter `yes`.

If Conda initialization was skipped, run:

    /home/$USER/miniconda3/bin/conda init bash

Reload the shell:

    source ~/.bashrc

Verify Conda:

    conda --version

## 5. Create the Project Conda Environment

Create the `fedbank` environment using Python 3.11:

    conda create -n fedbank python=3.11

Activate the environment:

    conda activate fedbank

Verify Python:

    python --version

Expected:

    Python 3.11.x

The system Python inside Ubuntu may be Python 3.13. It does not need to be changed. The project uses Python 3.11 inside the `fedbank` Conda environment.

## 6. Clone the GitHub Repository

Go to the home directory:

    cd ~

Clone the project repository:

    git clone <YOUR_GITHUB_REPOSITORY_URL>

Enter the project directory:

    cd <PROJECT_FOLDER>

Check the project files:

    ls

## 7. Verify NVIDIA GPU

Run:

    nvidia-smi

The NVIDIA GPU should be detected.

For this project, the expected GPU is:

    NVIDIA RTX 4050

If `nvidia-smi` does not work, fix the WSL2/NVIDIA GPU configuration before continuing.

## 8. Install cuML

Make sure the `fedbank` environment is active:

    conda activate fedbank

Install RAPIDS cuML:

    conda install -c rapidsai -c conda-forge cuml

Verify the installation:

    python -c "import cuml; print('cuML:', cuml.__version__)"

Test cuML KNN:

    python -c "from cuml.neighbors import NearestNeighbors; print('cuML KNN: OK')"

## 9. Install cuDF

Install cuDF for GPU-accelerated dataframe processing:

    conda install -c rapidsai -c conda-forge cudf

Verify:

    python -c "import cudf; print('cuDF:', cudf.__version__)"

## 10. Install cuGraph

Install cuGraph for GPU-accelerated graph processing:

    conda install -c rapidsai -c conda-forge cugraph

Verify:

    python -c "import cugraph; print('cuGraph:', cugraph.__version__)"

## Setup Complete

The following environment is now configured:

- WSL2
- Ubuntu
- Git
- Miniconda
- Python 3.11
- `fedbank` Conda environment
- NVIDIA GPU access
- RAPIDS cuML
- RAPIDS cuDF
- RAPIDS cuGraph