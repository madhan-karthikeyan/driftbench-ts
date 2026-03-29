# DriftBench-TS

A clean, modular research-grade Python framework for time series forecasting and drift detection benchmarking.

## Project Description

DriftBench-TS provides a standardized framework for:

- Loading and preprocessing time series datasets
- Running rolling window forecasting experiments
- Detecting concept drift in time series
- Evaluating forecasting model performance

The framework is designed to be modular, extensible, and easy to use for research purposes.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
driftbench-ts/
├── configs/              # Configuration files
│   └── electricity.yaml
├── driftbench/          # Main package
│   ├── datasets/        # Dataset loaders
│   ├── preprocessing/   # Data preprocessing utilities
│   ├── models/          # Forecasting models
│   ├── simulator/       # Rolling window simulator
│   ├── drift/           # Drift detection methods
│   ├── metrics/         # Evaluation metrics
│   └── utils/           # Utilities
├── notebooks/           # Jupyter notebooks for analysis
├── results/             # Experiment results
├── run_experiment.py    # Main experiment runner
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How to Run Experiment

1. Configure your experiment in `configs/electricity.yaml`

2. Run the experiment:
```bash
python run_experiment.py --config configs/electricity.yaml
```

3. Results will be saved to the `results/` directory

## Available Models

- **SeasonalNaiveModel**: Simple seasonal naive baseline
- **LGBMModel**: LightGBM-based forecaster
- **LSTMModel**: LSTM placeholder for future implementation

## Available Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- MAPE (Mean Absolute Percentage Error)

## Drift Detection

The framework includes KS-test based drift detection in `driftbench/drift/ks_test.py`.

## License

MIT
