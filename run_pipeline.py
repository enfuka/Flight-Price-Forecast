#!/usr/bin/env python3
"""
Flight Price Forecast - Complete Pipeline Runner (Windows Compatible)

This script automates the entire machine learning pipeline:
1. Runs all Jupyter notebooks in sequence (data preprocessing -> model training)
2. Validates model artifacts are created
3. Launches the Flask web application

Usage:
    python run_pipeline.py [--skip-notebooks] [--port 5000]

Arguments:
    --skip-notebooks    Skip notebook execution and go directly to UI launch
    --port              Port number for Flask app (default: 5000)
    --help              Show this help message
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Windows-compatible symbols
OK = "[OK]"
ERROR = "[ERROR]"
WARNING = "[WARNING]"
RUNNING = "[RUNNING]"
INFO = "[INFO]"

# Setup logging with Windows compatibility


class WindowsStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace problematic Unicode characters
            msg = msg.replace('[OK]', '[OK]').replace('[ERROR]', '[ERROR]')
            msg = msg.replace('[RUNNING]', '[RUNNING]').replace(
                '[INFO]', '[INFO]')
            msg = msg.replace('[ML]', '[ML]').replace('[DEMO]', '[DEMO]')
            msg = msg.replace('[STOPPED]', '[STOPPED]')

            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        WindowsStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FlightPricePipeline:
    def __init__(self, project_root=None):
        """Initialize the pipeline runner."""
        self.project_root = Path(
            project_root) if project_root else Path(__file__).parent
        self.notebooks_dir = self.project_root / 'notebooks'
        self.models_dir = self.project_root / 'models'
        self.data_dir = self.project_root / 'data'
        self.ui_dir = self.project_root / 'ui'

        # Notebook execution order
        self.notebooks = [
            '01_data_preprocessing.ipynb',
            '02_data_analysis.ipynb',
            '03_feature_selection.ipynb',
            '04_model_training.ipynb',
            '05_model_evaluation.ipynb',
            '06_model_fine_tuning.ipynb'
        ]

        # Expected output files
        self.expected_files = {
            '01_data_preprocessing.ipynb': [
                self.data_dir / 'cleaned_data.csv',
                self.data_dir / 'preprocessing_summary.json'
            ],
            '03_feature_selection.ipynb': [
                self.data_dir / 'features_selected.csv',
                self.data_dir / 'feature_metadata.json'
            ],
            '04_model_training.ipynb': [
                self.models_dir / 'best_model.pkl',
                self.models_dir / 'scaler.pkl',
                self.models_dir / 'feature_columns.json',
                self.models_dir / 'model_performance.json'
            ],
            '06_model_fine_tuning.ipynb': [
                self.models_dir / 'final_optimized_model.pkl',
                self.models_dir / 'final_scaler.pkl',
                self.models_dir / 'optimization_summary.json'
            ]
        }

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        logger.info("Setting up project directories...")

        directories = [self.data_dir, self.models_dir,
                       self.notebooks_dir, self.ui_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"{OK} Directory ready: {directory}")

    def check_dependencies(self):
        """Check if required dependencies are installed and install them if missing."""
        logger.info("Checking dependencies...")

        # Package mapping for proper imports
        required_packages = {
            'jupyter': 'jupyter',
            'nbconvert': 'nbconvert',  # Required for notebook execution
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'flask': 'flask',
            'flask-cors': 'flask_cors',
            'joblib': 'joblib'
        }

        missing_packages = []

        # First pass: check what's missing
        for display_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"{OK} {display_name} is installed")
            except ImportError:
                missing_packages.append(display_name)
                logger.warning(f"{WARNING} {display_name} is missing")

        # If packages are missing, try to install them automatically
        if missing_packages:
            logger.info(
                f"{INFO} Attempting to install missing packages: {missing_packages}")

            try:
                # Install missing packages
                install_cmd = [sys.executable, '-m',
                               'pip', 'install'] + missing_packages
                logger.info(f"{RUNNING} Installing packages...")

                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                logger.info(f"{OK} Successfully installed packages!")

                # Verify installation by checking imports again
                still_missing = []
                for display_name, import_name in required_packages.items():
                    if display_name in missing_packages:
                        try:
                            __import__(import_name)
                            logger.info(f"{OK} {display_name} now available")
                        except ImportError:
                            still_missing.append(display_name)

                if still_missing:
                    logger.error(f"{ERROR} Failed to install: {still_missing}")
                    logger.info(
                        f"{INFO} Please manually install: pip install " + " ".join(still_missing))
                    return False

                return True

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"{ERROR} Failed to install packages automatically")
                logger.error(f"Error: {e.stderr}")
                logger.info(
                    f"{INFO} Please manually install: pip install " + " ".join(missing_packages))
                return False
            except Exception as e:
                logger.error(
                    f"{ERROR} Unexpected error during installation: {e}")
                logger.info(
                    f"{INFO} Please manually install: pip install " + " ".join(missing_packages))
                return False

        return True

    def check_dataset(self):
        """Check if the dataset file exists."""
        dataset_path = self.project_root / \
            'US Airline Flight Routes and Fares 1993-2024.csv'

        if dataset_path.exists():
            logger.info(f"{OK} Dataset found: {dataset_path}")
            return True
        else:
            logger.warning(f"{WARNING} Dataset not found: {dataset_path}")
            logger.info(
                f"{INFO} The notebooks will create synthetic data for demonstration")
            return False

    def run_notebook(self, notebook_name):
        """Execute a single Jupyter notebook."""
        notebook_path = self.notebooks_dir / notebook_name

        if not notebook_path.exists():
            logger.error(f"{ERROR} Notebook not found: {notebook_path}")
            return False

        logger.info(f"{RUNNING} Executing {notebook_name}...")

        try:
            # Use current Python executable to run jupyter nbconvert
            # This ensures we use the same environment where packages are installed
            cmd = [
                sys.executable, '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=600',  # 10 minutes timeout
                str(notebook_path)
            ]

            logger.info(f"{INFO} Running command: {' '.join(cmd[:5])}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )

            if result.returncode == 0:
                logger.info(f"{OK} {notebook_name} completed successfully")
                return True
            else:
                logger.error(f"{ERROR} {notebook_name} failed with error:")
                if result.stderr:
                    logger.error(result.stderr)
                if result.stdout:
                    logger.error(result.stdout)
                return False

        except FileNotFoundError:
            logger.error(
                f"{ERROR} Python executable not found: {sys.executable}")
            return False
        except Exception as e:
            logger.error(f"{ERROR} Error running {notebook_name}: {e}")
            return False

    def validate_outputs(self, notebook_name):
        """Validate that expected output files were created."""
        if notebook_name not in self.expected_files:
            return True  # No validation needed

        expected = self.expected_files[notebook_name]
        missing_files = []

        for file_path in expected:
            if not file_path.exists():
                missing_files.append(file_path)

        if missing_files:
            logger.warning(
                f"{WARNING} {notebook_name} - Missing expected files:")
            for file_path in missing_files:
                logger.warning(f"  - {file_path}")
            return False
        else:
            logger.info(f"{OK} {notebook_name} - All expected files created")
            return True

    def run_all_notebooks(self):
        """Execute all notebooks in sequence."""
        logger.info("="*60)
        logger.info("STARTING NOTEBOOK EXECUTION PIPELINE")
        logger.info("="*60)

        start_time = time.time()
        successful_notebooks = 0

        for i, notebook in enumerate(self.notebooks, 1):
            logger.info(f"\n[{i}/{len(self.notebooks)}] Processing {notebook}")

            # Run notebook
            if self.run_notebook(notebook):
                successful_notebooks += 1

                # Validate outputs
                self.validate_outputs(notebook)

                # Brief pause between notebooks
                time.sleep(2)
            else:
                logger.error(f"{ERROR} Failed to execute {notebook}")

                # Ask user if they want to continue
                try:
                    response = input(
                        f"Continue with remaining notebooks? (y/n): ").lower().strip()
                    if response != 'y':
                        logger.info(
                            f"{INFO} Pipeline execution stopped by user")
                        return False
                except (EOFError, KeyboardInterrupt):
                    logger.info(f"{INFO} Pipeline execution interrupted")
                    return False

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("\n" + "="*60)
        logger.info("NOTEBOOK EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(
            f"Successfully executed: {successful_notebooks}/{len(self.notebooks)} notebooks")
        logger.info(f"Total execution time: {execution_time:.1f} seconds")

        if successful_notebooks == len(self.notebooks):
            logger.info(f"{OK} All notebooks executed successfully!")
            return True
        else:
            logger.warning(
                f"{WARNING} {len(self.notebooks) - successful_notebooks} notebooks failed")
            return successful_notebooks >= 4  # Need at least data prep + model training

    def check_models_ready(self):
        """Check if trained models are available for the UI."""
        model_files = [
            self.models_dir / 'best_model.pkl',
            self.models_dir / 'scaler.pkl',
            self.models_dir / 'feature_columns.json'
        ]

        missing_models = [f for f in model_files if not f.exists()]

        if missing_models:
            logger.warning(f"{WARNING} Some model files are missing:")
            for file_path in missing_models:
                logger.warning(f"  - {file_path}")
            logger.info(
                f"{INFO} The UI will use mock models for demonstration")
            return False
        else:
            logger.info(
                f"{OK} All model files found - UI will use trained models")
            return True

    def launch_ui(self, port=5000):
        """Launch the Flask web application."""
        logger.info("\n" + "="*60)
        logger.info("LAUNCHING FLIGHT PRICE FORECAST WEB APPLICATION")
        logger.info("="*60)

        app_path = self.ui_dir / 'app.py'

        if not app_path.exists():
            logger.error(f"{ERROR} Flask app not found: {app_path}")
            return False

        # Check if models are ready
        models_ready = self.check_models_ready()

        logger.info(f"{RUNNING} Starting Flask server on port {port}...")
        logger.info(
            f"{INFO} Web application will be available at: http://localhost:{port}")

        if models_ready:
            logger.info(f"{INFO} Using trained machine learning models")
        else:
            logger.info(f"{INFO} Using mock models for demonstration")

        logger.info("\n" + "="*50)
        logger.info("Press Ctrl+C to stop the server")
        logger.info("="*50)

        try:
            # Set environment variables
            env = os.environ.copy()
            env['FLASK_ENV'] = 'development'
            env['FLASK_DEBUG'] = 'True'

            # Launch Flask app
            subprocess.run([
                sys.executable, 'app.py'
            ], cwd=str(self.ui_dir), env=env)

        except KeyboardInterrupt:
            logger.info(f"\n{INFO} Server stopped by user")
            return True
        except Exception as e:
            logger.error(f"{ERROR} Error launching Flask app: {e}")
            return False

    def generate_report(self):
        """Generate a summary report of the pipeline execution."""
        report_path = self.project_root / 'pipeline_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Flight Price Forecast Pipeline Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Project Structure\n")
            f.write(f"- Project Root: `{self.project_root}`\n")
            f.write(f"- Notebooks: `{self.notebooks_dir}`\n")
            f.write(f"- Models: `{self.models_dir}`\n")
            f.write(f"- Data: `{self.data_dir}`\n")
            f.write(f"- UI: `{self.ui_dir}`\n\n")

            f.write("## Executed Notebooks\n")
            for notebook in self.notebooks:
                notebook_path = self.notebooks_dir / notebook
                status = "" if notebook_path.exists() else ""
                f.write(f"{status} {notebook}\n")

            f.write("\n## Generated Files\n")

            # Data files
            f.write("### Data Files\n")
            data_files = ['cleaned_data.csv', 'features_selected.csv',
                          'preprocessing_summary.json', 'feature_metadata.json']
            for file_name in data_files:
                file_path = self.data_dir / file_name
                status = "" if file_path.exists() else ""
                f.write(f"{status} `data/{file_name}`\n")

            # Model files
            f.write("\n### Model Files\n")
            model_files = ['best_model.pkl', 'scaler.pkl',
                           'feature_columns.json', 'model_performance.json']
            for file_name in model_files:
                file_path = self.models_dir / file_name
                status = "" if file_path.exists() else ""
                f.write(f"{status} `models/{file_name}`\n")

            f.write(f"\n## Web Application\n")
            f.write("To launch the web application manually:\n")
            f.write("```bash\n")
            f.write("cd ui\n")
            f.write("python app.py\n")
            f.write("```\n")
            f.write("Then open: http://localhost:5000\n")

        logger.info(f"{INFO} Pipeline report generated: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Flight Price Forecast Pipeline Runner')
    parser.add_argument('--skip-notebooks', action='store_true',
                        help='Skip notebook execution and launch UI directly')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number for Flask app (default: 5000)')
    parser.add_argument('--project-root', type=str, default=None,
                        help='Project root directory (default: current directory)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = FlightPricePipeline(args.project_root)

    print("Flight Price Forecast - Complete Pipeline Runner")
    print("="*60)

    # Setup
    pipeline.setup_directories()

    if not pipeline.check_dependencies():
        print(f"{ERROR} Dependency check failed. Please install missing packages.")
        return 1

    pipeline.check_dataset()

    # Execute notebooks (unless skipped)
    if not args.skip_notebooks:
        success = pipeline.run_all_notebooks()
        if not success:
            print(
                f"{WARNING} Notebook execution had issues, but continuing to UI launch...")
    else:
        print(f"{INFO} Skipping notebook execution as requested")

    # Generate report
    pipeline.generate_report()

    # Launch UI
    try:
        pipeline.launch_ui(args.port)
    except KeyboardInterrupt:
        print(f"\n{INFO} Pipeline execution completed!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
