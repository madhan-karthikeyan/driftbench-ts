# DriftBench-TS

A clean, modular research-grade Python framework for time series forecasting and concept drift detection benchmarking.

## Overview

DriftBench-TS provides a standardized framework for:

- Loading and preprocessing time series datasets (electricity, traffic, oil prices, weather, synthetic)
- Running rolling window forecasting experiments with configurable models
- Detecting concept drift using multiple algorithms (ADWIN, Page-Hinkley, CUSUM, KS Test, Wasserstein)
- Evaluating forecasting model performance with comprehensive metrics
- Comparing retraining strategies (no retrain, fixed schedule, adaptive)

The framework is designed to be modular, extensible, and easy to use for research purposes.

## Installation

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
├── configs/                     # Configuration files for experiments
├── dashboard-react/             # React dashboard
│   ├── api.py                   # Flask API server
│   ├── src/                     # React source code
│   │   ├── pages/               # Dashboard pages
│   │   ├── components/          # React components
│   │   └── services/             # API client
│   └── package.json             # npm dependencies
├── driftbench/                  # Main package
│   ├── datasets/                # Dataset loaders and utilities
│   ├── preprocessing/          # Data preprocessing utilities
│   ├── models/                 # Forecasting models
│   ├── simulator/              # Rolling window simulator
│   ├── drift/                  # Drift detection methods
│   ├── metrics/                # Evaluation metrics
│   └── utils/                  # Utilities
├── notebooks/                   # Jupyter notebooks for analysis
├── results/                     # Experiment results
├── run_experiment.py            # Main experiment runner
├── run_all.py                  # Run all experiment combinations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Run a Single Experiment

```bash
python run_experiment.py --config configs/electricity.yaml
```

### Run All Experiments

```bash
python run_all.py
```

This runs experiments for all combinations of:
- **Datasets**: traffic, brent, wti, electricity
- **Strategies**: no_retrain, fixed_retrain, adaptive_retrain
- **Models**: naive, seasonal_naive, rf, lgbm, lstm, tsmixer

### Start the Dashboard

```bash
# Terminal 1: Start the Flask API server
source venv/bin/activate
python dashboard-react/api.py --results results --port 5001

# Terminal 2: Start the React development server
cd dashboard-react
npm install
npm run dev
```

Open http://localhost:3000 to view the dashboard.

## Available Datasets

| Dataset | Description | Frequency |
|---------|-------------|-----------|
| `electricity` | UCI Electricity dataset (multiple entities) | Hourly |
| `traffic` | UCI Metro Traffic Volume (single entity) | Hourly |
| `brent` | Brent crude oil prices | Weekly |
| `wti` | WTI crude oil prices | Weekly |
| `weather` | Synthetic weather-like data | Hourly |
| `synthetic` | Synthetic data with known drift points | Configurable |

All datasets return a DataFrame with columns: `entity_id`, `timestamp`, `target`

## Available Models

| Model | Description | Dependencies |
|-------|-------------|--------------|
| `naive` | Naive baseline (previous value) | - |
| `seasonal_naive` | Seasonal naive baseline | - |
| `rf` | Random Forest regressor | scikit-learn |
| `lgbm` | LightGBM regressor | lightgbm |
| `lstm` | LSTM neural network | torch |
| `tsmixer` | TSMixer (MLP-based) | torch |

### Model Parameters

```yaml
model:
  name: lgbm
  params:
    n_estimators: 50
    max_depth: 5
    learning_rate: 0.1
    n_lags: 3
```

## Retraining Strategies

| Strategy | Description | Configuration |
|----------|-------------|--------------|
| `no_retrain` | Model never retrained | `policy: no_retraining` |
| `fixed_retrain` | Retrain on fixed schedule | `policy: fixed_schedule`, `retrain_every_n_steps: 10` |
| `adaptive_retrain` | Retrain when drift detected | `policy: drift_triggered`, `drift_threshold: 0.5` |

### Example Configuration

```yaml
dataset:
  name: electricity
  sample_entities: 5
  downsample_freq: h

model:
  name: lgbm
  params:
    n_estimators: 50

simulation:
  history_window_days: 30
  forecast_horizon_hours: 24
  step_size_hours: 24

drift:
  enabled: true

retraining:
  enabled: true
  policy: drift_triggered
  drift_threshold: 0.5
  min_steps_between_retrain: 3

seed: 42
```

## Drift Detection

The framework includes multiple drift detection algorithms:

| Detector | Description | Use Case |
|----------|-------------|----------|
| `adwin` | Adaptive Windowing detector | Online learning |
| `page_hinkley` | Page-Hinkley test | Mean shift detection |
| `cusum` | Cumulative Sum detector | Process monitoring |
| `ks_test` | Kolmogorov-Smirnov test | Distribution change |
| `wasserstein` | Wasserstein distance | Distribution shift |
| `residual_ks` | Residual-based KS test | Model performance drift |

All detectors use normalized effect size (Cohen's d) for continuous data.

### Detection Modes

- `feature`: Detect drift in input features
- `residual`: Detect drift in prediction residuals
- `error`: Detect drift in prediction errors

## Metrics

The framework computes comprehensive forecasting metrics:

| Metric | Description |
|--------|-------------|
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `SMAPE` | Symmetric Mean Absolute Percentage Error |
| `MAPE` | Mean Absolute Percentage Error |

Additional metrics available:
- Per-window metrics (saved to `window_metrics.csv`)
- Drift detection rate
- Retrain statistics (compute proxy, retrain rate)

## Output Structure

Results are saved to the `results/` directory:

```
results/
├── dataset/
│   └── strategy/
│       └── model/
│           ├── metrics.json           # Overall metrics
│           ├── drift_log.csv          # Drift events
│           ├── retraining_log.csv     # Retraining events
│           ├── window_metrics.csv     # Per-window metrics
│           └── config_snapshot.yaml  # Configuration used
```

## Using the Python API

### Load a Dataset

```python
from driftbench.datasets import get_dataset

df = get_dataset('electricity', sample_entities=5, downsample_freq='h')
```

### Create a Model

```python
from driftbench.models import get_model

model = get_model('lgbm', n_estimators=50, max_depth=5)
```

### Run a Rolling Simulation

```python
from driftbench.simulator.rolling import RollingSimulator

simulator = RollingSimulator(
    model=model,
    history_window=90*24,  # 90 days of hourly data
    horizon=24,            # 24-hour forecast
    step_size=24           # Roll every 24 hours
)

results = simulator.run(df)
```

### Create a Drift Detector

```python
from driftbench.drift.detectors import create_detector

detector = create_detector('adwin', effect_threshold=0.5)
```

## Dashboard

The project includes a modern React dashboard for visualizing experiment results.

### Features

| Page | Description |
|------|-------------|
| **Overview** | Summary cards, best model rankings, filterable results table |
| **Dataset Deep Dive** | Error timelines, drift alignment charts, model comparison |
| **Global Comparison** | Heatmaps, strategy rankings, efficiency frontier plots |
| **Run Explorer** | Window-level analysis, drift score tracking, retraining effects |

### Tech Stack

- **Frontend**: React 18 + Vite + React Router + Recharts
- **Styling**: Tailwind CSS (dark mode default)
- **Backend**: Flask API serving results data
- **Icons**: Lucide React

### Dashboard Project Structure

```
dashboard-react/
├── api.py                 # Flask API server
├── package.json           # npm dependencies
├── vite.config.js         # Vite configuration
├── tailwind.config.js     # Tailwind configuration
├── index.html             # HTML entry point
└── src/
    ├── App.jsx            # Main app with routing
    ├── index.css          # Global styles
    ├── main.jsx           # React entry point
    ├── components/
    │   └── Layout.jsx     # Navigation and layout
    ├── pages/
    │   ├── Overview.jsx    # Main overview page
    │   ├── DatasetDetail.jsx  # Dataset deep dive
    │   ├── Compare.jsx     # Global comparison
    │   └── RunExplorer.jsx # Individual run details
    ├── services/
    │   └── api.js          # API client functions
    └── types/
        └── constants.js    # Colors, labels, etc.
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/results` | All experiment runs with summary |
| `GET /api/datasets` | List of all datasets |
| `GET /api/dataset/:name` | Dataset detail with timeline |
| `GET /api/run/:d/:s/:m` | Single run metrics |
| `GET /api/run/:d/:s/:m/windows` | Window-level time series |
| `GET /api/compare` | Heatmap and best per dataset |
| `GET /api/compare/models` | Model rankings |
| `GET /api/compare/strategies` | Strategy rankings |
| `GET /api/robustness` | Error vs retrain cost data |

### Key Visualizations

The dashboard highlights the causal relationship between:
1. **Drift Detection**: When drift is detected (amber markers)
2. **Error Spike**: Corresponding error increase
3. **Retrain Event**: Model retraining (green markers)
4. **Recovery**: Error stabilization after retraining

### Build for Production

```bash
cd dashboard-react
npm run build
# Serve the dist folder with any web server
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 conventions.

## License

MIT