# Flight Price Forecast - Environment Setup Script
# Run this script to set up the conda environment with GPU support

param(
    [switch]$SkipGPU,
    [switch]$Update,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$EnvName = "flight-forecast"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Flight Price Forecast - Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
try {
    $condaVersion = conda --version
    Write-Host "[OK] Conda found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Conda not found! Please install Miniconda or Anaconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Clean option - remove existing environment
if ($Clean) {
    Write-Host ""
    Write-Host "Cleaning up existing environment..." -ForegroundColor Yellow
    conda deactivate 2>$null
    conda env remove -n $EnvName -y 2>$null
    Write-Host "[OK] Environment cleaned" -ForegroundColor Green
}

# Check if environment exists
$envExists = conda env list | Select-String -Pattern "^$EnvName\s"

if ($Update -and $envExists) {
    Write-Host ""
    Write-Host "Updating existing environment..." -ForegroundColor Yellow
    conda env update -f environment.yml --prune
    Write-Host "[OK] Environment updated" -ForegroundColor Green
} elseif ($envExists) {
    Write-Host ""
    Write-Host "[INFO] Environment '$EnvName' already exists." -ForegroundColor Yellow
    Write-Host "Use -Update to update, or -Clean to recreate." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Creating new conda environment..." -ForegroundColor Yellow
    Write-Host "This may take several minutes..." -ForegroundColor Gray
    
    if ($SkipGPU) {
        Write-Host "[INFO] Skipping GPU packages (CPU-only mode)" -ForegroundColor Yellow
        conda env create -f environment-cpu.yml
    } else {
        conda env create -f environment.yml
    }
    
    Write-Host "[OK] Environment created successfully!" -ForegroundColor Green
}

# Check for NVIDIA GPU
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow

try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        Write-Host "[OK] NVIDIA GPU detected:" -ForegroundColor Green
        $nvidiaSmi | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray }
    }
} catch {
    Write-Host "[WARNING] NVIDIA GPU not detected or drivers not installed." -ForegroundColor Yellow
    Write-Host "          Training will use CPU (slower but still works)." -ForegroundColor Yellow
}

# Register Jupyter kernel
Write-Host ""
Write-Host "Registering Jupyter kernel..." -ForegroundColor Yellow
conda activate $EnvName
python -m ipykernel install --user --name $EnvName --display-name "Python (Flight Forecast GPU)"
Write-Host "[OK] Jupyter kernel registered" -ForegroundColor Green

# Setup .env file if not exists
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[OK] Created .env file from template" -ForegroundColor Green
        Write-Host "[ACTION] Please edit .env and add your GEMINI_API_KEY" -ForegroundColor Yellow
    }
}

# Final instructions
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor White
Write-Host "  conda activate $EnvName" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the training pipeline:" -ForegroundColor White
Write-Host "  python run_pipeline.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start Jupyter Lab:" -ForegroundColor White
Write-Host "  jupyter lab" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start the web UI:" -ForegroundColor White
Write-Host "  python ui/app.py" -ForegroundColor Yellow
Write-Host ""
