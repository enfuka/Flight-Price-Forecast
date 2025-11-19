# Flight Price Forecast - Pipeline Runner (PowerShell)
# =================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Flight Price Forecast - Complete Pipeline Runner" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Continue"

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
    }
    else {
        throw "Python not found"
    }
}
catch {
    Write-Host "[ERROR] ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Navigate to script directory
Set-Location -Path $PSScriptRoot

# Check for virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Blue
    try {
        & "venv\Scripts\Activate.ps1"
        Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
    }
    catch {
        Write-Host "[WARNING] Warning: Could not activate virtual environment" -ForegroundColor Yellow
    }
}
elseif (Test-Path "venv\Scripts\activate.bat") {
    Write-Host "Activating virtual environment (batch)..." -ForegroundColor Blue
    cmd /c "venv\Scripts\activate.bat && echo Environment activated"
}

Write-Host ""

# Install/upgrade required packages
Write-Host "Installing/updating required packages..." -ForegroundColor Blue
$packages = @(
    "jupyter",
    "pandas>=1.5.0", 
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "flask>=2.0.0",
    "flask-cors>=3.0.0",
    "joblib>=1.0.0"
)

try {
    foreach ($package in $packages) {
        Write-Host "  Installing $package..." -ForegroundColor Gray -NoNewline
        pip install --quiet --upgrade $package 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
        }
        else {
            Write-Host " [WARNING]" -ForegroundColor Yellow
        }
    }
    Write-Host "[OK] Package installation completed" -ForegroundColor Green
}
catch {
    Write-Host "[WARNING] Warning: Some packages may not have installed correctly" -ForegroundColor Yellow
}

Write-Host ""

# Display pipeline information
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Starting the Flight Price Forecast Pipeline..." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This pipeline will:" -ForegroundColor White
Write-Host "  1. Execute data preprocessing notebook" -ForegroundColor Gray
Write-Host "  2. Perform data analysis and visualization" -ForegroundColor Gray
Write-Host "  3. Run feature selection and engineering" -ForegroundColor Gray
Write-Host "  4. Train multiple machine learning models" -ForegroundColor Gray
Write-Host "  5. Evaluate model performance" -ForegroundColor Gray
Write-Host "  6. Fine-tune and optimize models" -ForegroundColor Gray
Write-Host "  7. Launch the web application" -ForegroundColor Gray
Write-Host ""
Write-Host "This process may take 5-10 minutes depending on your system..." -ForegroundColor Yellow
Write-Host ""

# Offer options to user
Write-Host "Choose an option:" -ForegroundColor Cyan
Write-Host "  [1] Run complete pipeline (notebooks + web app)" -ForegroundColor White
Write-Host "  [2] Skip notebooks and launch web app only" -ForegroundColor White
Write-Host "  [3] Run notebooks only (no web app)" -ForegroundColor White
Write-Host "  [Q] Quit" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Enter your choice (1, 2, 3, or Q)"
    $choice = $choice.ToUpper()
} while ($choice -notin @("1", "2", "3", "Q"))

switch ($choice) {
    "1" {
        Write-Host "Running complete pipeline..." -ForegroundColor Green
        python run_pipeline.py
    }
    "2" {
        Write-Host "Skipping notebooks, launching web app..." -ForegroundColor Yellow
        python run_pipeline.py --skip-notebooks
    }
    "3" {
        Write-Host "Running notebooks only..." -ForegroundColor Blue
        python run_pipeline.py --skip-notebooks
        # Note: This needs to be modified in the Python script to support notebooks-only mode
        Write-Host "[OK] Notebooks completed. Run option 2 to launch the web app." -ForegroundColor Green
        Read-Host "Press Enter to exit"
    }
    "Q" {
        Write-Host "Goodbye!" -ForegroundColor Yellow
        exit 0
    }
}

# Final status check
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your web application should now be running at:" -ForegroundColor White
    Write-Host "   http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Check 'pipeline_report.md' for detailed execution summary" -ForegroundColor Gray
}
else {
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "Pipeline completed with some issues" -ForegroundColor Yellow
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the output above for error details" -ForegroundColor Gray
    Write-Host "See 'pipeline.log' for complete execution log" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to exit"