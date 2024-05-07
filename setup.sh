#!/bin/bash

# Update Git submodules
git submodule update --init --recursive || {
    echo "Error updating Git submodules"
    exit 1
}

# Create a new virtual environment with Python 3.10
python3.10 -m venv .venv || {
    echo "Error creating the virtual environment"
    exit 1
}

# Activate the virtual environment
source .venv/bin/activate || {
    echo "Error activating the virtual environment"
    exit 1
}

# Install PyTorch with CUDA 12.x support
pip install torch || {
    echo "Error installing PyTorch packages"
    exit 1
}

# Install other required packages
pip install matplotlib==3.6.2 numpy==1.24.4 scikit_learn==1.1.3 setuptools==65.5.0 sympy==1.11.1 tqdm==4.66.2 || {
    echo "Error installing additional packages"
    exit 1
}

# Change directory to pykan
cd pykan || {
    echo "Error changing directory to pykan"
    exit 1
}

# Install pykan in editable mode
pip install -e. || {
    echo "Error installing pykan"
    exit 1
}

# Change directory to efficient-kan
cd ../efficient-kan || {
    echo "Error changing directory to efficient-kan"
    exit 1
}

# Install efficient-kan in editable mode
pip install -e. || {
    echo "Error installing efficient-kan"
    exit 1
}

# Change directory up
cd .. || {
    echo "Error changing directory up"
    exit 1
}

# Check execution
python benchmark.py --help || {
    echo "Error. Something is not well installed."
    exit 1
}

# Success message
echo "Setup completed successfully"