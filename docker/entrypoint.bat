@echo off
REM Windows entrypoint script for Combat-Ready SLAM System

echo Starting Combat-Ready Low-Light SLAM + Thermal Detection System...

REM Check system requirements
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics YOLO ready')"

REM Set environment variables
set PYTHONPATH=%cd%\src;%PYTHONPATH%
set CUDA_VISIBLE_DEVICES=0

REM Parse command line arguments
if "%1"=="--demo" (
    echo Running in demo mode...
    python src\main.py --demo
) else if "%1"=="--dataset" (
    echo Running dataset processing mode...
    python src\main.py --dataset %2
) else if "%1"=="--train" (
    echo Running training mode...
    python scripts\multi_dataset_training.py
) else if "%1"=="--evaluate" (
    echo Running evaluation mode...
    python scripts\evaluate_system.py
) else (
    echo Usage: entrypoint.bat [--demo^|--dataset PATH^|--train^|--evaluate]
    echo   --demo: Run system demonstration with synthetic data
    echo   --dataset PATH: Process specific dataset
    echo   --train: Train YOLO model on thermal datasets
    echo   --evaluate: Evaluate system performance
    echo.
    echo Running demo mode by default...
    python src\main.py --demo
)

echo System execution completed.
pause
