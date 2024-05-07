@echo off
REM Update Git submodules
git submodule update --init --recursive
IF ERRORLEVEL 1 (
    echo Error updating Git submodules
    pause
    exit /b 1
)

REM Create a new conda environment with Python 3.10
conda create --name kan python=3.10 -y
IF ERRORLEVEL 1 (
    echo Error creating the conda environment
    pause
    exit /b 1
)

REM Activate the conda environment
call activate kan
IF ERRORLEVEL 1 (
    echo Error activating the conda environment
    pause
    exit /b 1
)

REM Install PyTorch with CUDA 11.8 support
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
IF ERRORLEVEL 1 (
    echo Error installing PyTorch packages
    pause
    exit /b 1
)

REM Install other required packages
python -m pip install matplotlib==3.6.2 numpy==1.24.4 scikit_learn==1.1.3 setuptools==65.5.0 sympy==1.11.1 tqdm==4.66.2
IF ERRORLEVEL 1 (
    echo Error installing additional packages
    pause
    exit /b 1
)

REM Change directory to pykan
cd pykan
IF ERRORLEVEL 1 (
    echo Error changing directory to pykan
    pause
    exit /b 1
)

REM Install pykan in editable mode
python -m pip install -e .
IF ERRORLEVEL 1 (
    echo Error installing pykan
    pause
    exit /b 1
)

REM Change directory to efficient-kan
cd ..\efficient-kan
IF ERRORLEVEL 1 (
    echo Error changing directory to efficient-kan
    pause
    exit /b 1
)

REM Install efficient-kan in editable mode
python -m pip install -e .
IF ERRORLEVEL 1 (
    echo Error installing efficient-kan
    pause
    exit /b 1
)

REM Change directory up
cd ..
IF ERRORLEVEL 1 (
    echo Error changing directory up
    pause
    exit /b 1
)

REM Check execution
python benchmark.py --help
IF ERRORLEVEL 1 (
    echo Error running benchmark.py --help
    pause
    exit /b 1
)

REM Success message
echo Setup completed successfully
pause
