#!/usr/bin/env python3
"""
Quick Start Script - Flight Price Forecast

This script provides a simplified way to quickly test the application
without running the full notebook pipeline.

Usage: python quick_start.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def quick_setup():
    """Quick setup for testing the application."""
    print("Flight Price Forecast - Quick Start")
    print("="*50)
    print()

    # Get project root
    project_root = Path(__file__).parent
    ui_dir = project_root / 'ui'
    models_dir = project_root / 'models'
    data_dir = project_root / 'data'

    # Create directories
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    print(" Installing required packages...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet',
            'flask', 'flask-cors', 'pandas', 'numpy', 'scikit-learn', 'joblib'
        ], check=True)
        print("Packages installed successfully")
    except subprocess.CalledProcessError:
        print("Warning: Package installation failed, continuing anyway...")

    print()

    # Check if models exist
    model_files = [
        models_dir / 'best_model.pkl',
        models_dir / 'scaler.pkl',
        models_dir / 'feature_columns.json'
    ]

    models_exist = all(f.exists() for f in model_files)

    if models_exist:
        print("Trained models found - using real models")
    else:
        print("No trained models found - will use mock predictions")
        print("   Run 'python run_pipeline.py' to train real models")

    print()
    print("Starting web application...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    print()

    # Launch Flask app
    try:
        os.chdir(ui_dir)
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nServer stopped!")
    except FileNotFoundError:
        print("Flask app not found. Please ensure ui/app.py exists.")
        return False
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

    return True


if __name__ == '__main__':
    success = quick_setup()
    sys.exit(0 if success else 1)
