# Flight Price Forecast Project

A comprehensive machine learning project for predicting airline ticket prices using historical flight data from 1993-2024. This project implements end-to-end data science methodology with interactive Jupyter notebooks and a user-friendly web interface.

## Project Overview

**Objective**: Predict airline fare trends and provide travelers with affordable airfare recommendations using 30+ years of historical flight data.

**Dataset**: US Airline Flight Routes and Fares (1993–2024) from Kaggle

**Methodology**: Data-driven approach using ensemble machine learning models with comprehensive evaluation and hyperparameter optimization.

## Project Structure

```
Flight Price Forecast/
│
├── US Airline Flight Routes and Fares 1993-2024.csv  # Dataset
├── requirements.txt                                   # Python dependencies
├── README.md                                         # Project documentation
│
├── notebooks/                    # Jupyter notebooks (main workflow)
│   ├── 01_data_preprocessing.ipynb   # Data cleaning and preparation
│   ├── 02_data_analysis.ipynb       # Exploratory Data Analysis (EDA)
│   ├── 03_feature_selection.ipynb   # Feature engineering and selection
│   ├── 04_model_training.ipynb      # Model selection and training
│   ├── 05_model_evaluation.ipynb    # Performance evaluation
│   └── 06_model_fine_tuning.ipynb   # Hyperparameter optimization
│
├── data/                         # Processed datasets
│   ├── cleaned_data.csv             # Cleaned dataset (generated)
│   ├── features.csv                 # Selected features (generated)
│   └── train_test_split/            # Split datasets (generated)
│
├── models/                       # Trained models and artifacts
│   ├── best_model.pkl               # Final optimized model
│   ├── scaler.pkl                   # Feature scaler
│   ├── feature_columns.pkl          # Feature metadata
│   └── model_performance.json       # Evaluation metrics
│
└── ui/                           # Web user interface
    ├── index.html                   # Main web interface
    ├── styles.css                   # Custom styling
    ├── script.js                    # Frontend functionality
    └── app.py                       # Flask backend server
```

## Methodology Workflow

### Phase 1: Data Foundation

- **A. Data Selection**: Utilize comprehensive US airline dataset (1993-2024)
- **B. Data Preprocessing**: Clean, validate, and prepare data for analysis

### Phase 2: Analysis & Feature Engineering

- **C. Data Analysis**: Conduct EDA to understand patterns and relationships
- **D. Feature Selection**: Engineer and select optimal predictive features

### Phase 3: Model Development

- **E. Model Selection & Training**: Implement Decision Trees, Random Forests, and ensemble methods
- **F. Model Evaluation**: Comprehensive performance assessment with multiple metrics
- **G. Model Fine-Tuning**: Hyperparameter optimization using Grid Search and cross-validation

### Phase 4: Deployment

- **H. User Interface**: Interactive web application for price predictions

## Quick Start

### Automated Pipeline (Recommended)

The fastest way to get started is using our automated pipeline scripts:

#### Option 1: Complete Pipeline (Windows)

```batch
# Double-click to run, or in PowerShell/Command Prompt:
run_pipeline.bat
```

#### Option 2: PowerShell Script (Windows)

```powershell
# Right-click → "Run with PowerShell", or:
.\run_pipeline.ps1
```

#### Option 3: Python Script (Cross-platform)

```bash
# Run complete pipeline (notebooks + web app)
python run_pipeline.py

# Skip notebooks and launch web app only
python run_pipeline.py --skip-notebooks

# Use custom port
python run_pipeline.py --port 8080
```

#### Option 4: Quick Test (No Model Training)

```bash
# Launch web app immediately with mock predictions
python quick_start.py
```

### Manual Setup (Advanced Users)

If you prefer to run components individually:

#### 1. Setup Environment

```bash
# Create virtual environment
python -m venv flight_forecast_env

# Activate environment
# Windows:
flight_forecast_env\Scripts\activate
# macOS/Linux:
source flight_forecast_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Run Notebooks in Order

Execute notebooks sequentially using Jupyter:

```bash
# Start Jupyter
jupyter notebook

# Or execute programmatically
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_model_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_model_fine_tuning.ipynb
```

#### 3. Launch Web Interface

```bash
cd ui
python app.py
```

Access the application at: `http://localhost:5000`

### Script Features

Our automation scripts provide:

- **Dependency Management**: Automatic package installation
- **Error Handling**: Graceful failure recovery
- **Progress Tracking**: Real-time execution status
- **Validation**: Verify model artifacts are created
- **Logging**: Detailed execution logs (`pipeline.log`)
- **Report Generation**: Summary report (`pipeline_report.md`)
- **Flexible Options**: Skip notebooks, custom ports, etc.

## Key Features

### Machine Learning Pipeline

- **Data Preprocessing**: Automated cleaning, duplicate removal, missing value handling
- **Feature Engineering**: Temporal features, seasonal patterns, route competition metrics
- **Model Selection**: Decision Trees, Random Forests, Gradient Boosting
- **Hyperparameter Optimization**: Grid Search, Random Search, Bayesian optimization
- **Cross-Validation**: Robust model evaluation with multiple CV strategies

### Web Application

- **Interactive Price Prediction**: User-friendly form for flight details
- **Real-time Analysis**: Instant price predictions with confidence intervals
- **Smart Recommendations**: Booking timing and route optimization suggestions
- **Market Insights**: Historical trends and seasonal patterns
- **Responsive Design**: Mobile-friendly interface

## Expected Outcomes

### Model Performance Targets

- **Accuracy**: Target >85% prediction accuracy on test set
- **Precision/Recall**: Balanced performance across price ranges
- **RMSE**: Minimize root mean squared error for price predictions
- **Cross-Validation**: Consistent performance across different data splits

### Business Value

- **Cost Savings**: Help travelers find optimal booking times
- **Market Insights**: Understand airline pricing patterns
- **Trend Analysis**: Identify seasonal and temporal pricing trends
- **Decision Support**: Data-driven flight booking recommendations

## Technical Stack

**Data Science**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
**Machine Learning**: Decision Trees, Random Forests, Gradient Boosting, XGBoost
**Web Framework**: Flask, HTML5, CSS3, JavaScript, Bootstrap
**Optimization**: GridSearchCV, RandomizedSearchCV, Scikit-optimize
**Visualization**: Interactive charts, statistical plots, performance dashboards

## Requirements

### System Requirements

- Python 3.8+
- 4GB+ RAM recommended
- 2GB+ free disk space for datasets and models

### Dependencies

See `requirements.txt` for complete list. Key libraries:

- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- flask, flask-cors
- jupyter, ipykernel

## Usage Examples

### Prediction API

```python
# Example API call for price prediction
import requests

prediction = requests.post('http://localhost:5000/api/predict', json={
    'originCity': 'New York, NY',
    'destCity': 'Los Angeles, CA',
    'travelDate': '2024-07-15',
    'bookingAdvance': 45,
    'airline': 'American Airlines',
    'tripType': 'roundtrip'
})

print(prediction.json())
```

### Model Usage

```python
# Load trained model for custom predictions
import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare features and predict
features = prepare_flight_features(...)
prediction = model.predict(scaler.transform(features))
```

## Documentation

Each notebook contains detailed documentation including:

- Methodology explanations
- Code comments and markdown cells
- Visualization interpretations
- Performance analysis
- Next steps and recommendations

## Contributing

1. Follow the established notebook structure
2. Document all analyses and decisions
3. Include visualizations for key insights
4. Test models thoroughly before deployment
5. Update documentation for any changes

## Future Enhancements

- **Real-time Data**: Integration with live flight APIs
- **Advanced Models**: Deep learning and neural networks
- **Multi-class Prediction**: Categorical price ranges
- **Mobile App**: Native mobile application
- **API Enhancement**: RESTful API with authentication

## Important Notes

- Predictions are based on historical data (1993-2024)
- Actual prices may vary due to current market conditions
- Always verify predictions with actual airline pricing
- Model performance depends on data quality and feature selection

## Support

For questions or issues:

1. Check notebook documentation
2. Review error logs in console
3. Ensure all dependencies are installed
4. Verify dataset is in correct location

---

**Created for CMSC 535 - Fall 2025**  
**Flight Price Forecasting Using Machine Learning**
