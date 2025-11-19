@echo off
echo =================================================================
echo Flight Price Forecast - Complete Pipeline Runner
echo =================================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo Python found!
echo.

:: Navigate to project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

:: Install/upgrade required packages
echo Installing/updating required packages...
pip install --quiet --upgrade jupyter pandas numpy scikit-learn matplotlib seaborn plotly flask flask-cors joblib
echo.

:: Run the Python pipeline script
echo =================================================================
echo Starting the Flight Price Forecast Pipeline...
echo =================================================================
echo.
echo This will:
echo 1. Execute all Jupyter notebooks in sequence
echo 2. Train machine learning models
echo 3. Launch the web application
echo.
echo The process may take several minutes...
echo.

python run_pipeline.py

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo =================================================================
    echo Pipeline execution completed with errors.
    echo Check the output above for details.
    echo =================================================================
    pause
) else (
    echo.
    echo =================================================================
    echo Pipeline completed successfully!
    echo =================================================================
)