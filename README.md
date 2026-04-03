# DriftBench-TS

A deployment-oriented benchmark and open-source framework for rigorous evaluation of time series forecasting models under distribution shift (concept drift).

## Problem Statement

Time series forecasting models are embedded in high-stakes decision systems spanning energy load scheduling, road traffic management, financial trading, and retail demand planning. Despite the proliferation of sophisticated forecasting architectures, prevailing evaluation practices rely on a single static train-test split that implicitly assumes stationarity and ignores the realities of long-running deployment.

**Core Problem**: Models trained on historical data degrade as data distributions evolve over time due to:
- Seasonal cycles and periodic patterns
- Economic shocks and policy interventions
- Sensor degradation and evolving user behavior
- External market forces and environmental changes

**Research Questions**:
1. How do different model families perform under distribution shift?
2. What retraining strategies are most effective across different domains?
3. How do practitioners balance forecast accuracy against computational cost?

## Study Overview

DriftBench-TS provides a comprehensive experimental study across:

| Dimension | Configuration |
|-----------|---------------|
| **Datasets** | 4 real-world datasets (electricity, traffic, Brent oil, WTI oil) |
| **Domains** | Energy, Mobility, Finance |
| **Models** | 6 model families (Naive, Seasonal Naive, RF, LightGBM, LSTM, TSMixer) |
| **Strategies** | 3 maintenance policies (No Retrain, Fixed Interval, Adaptive Drift-Triggered) |
| **Total Runs** | 72 configuration combinations × 5 seeds |

## Methods

### 1. Deployment Simulation Engine

Rolling window evaluation that emulates streaming deployment:

```
┌─────────────────────────────────────────────────────────────┐
│  Initial Training → Rolling Evaluation Loop                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │  Train    │───▶│ Forecast │───▶│  Update   │            │
│  │  Window   │    │  F steps │    │  Drift    │            │
│  └──────────┘    └──────────┘    │  Metrics  │            │
│       ▲              │          └────▲─────┘            │
│       │              │               │                   │
│       └──────────────┴───────────────┘                   │
│              Retrain if Policy Requires                   │
└─────────────────────────────────────────────────────────────┘
```

### 2. Maintenance Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| **No Retrain** | Single training, never updated | Baseline, zero-cost |
| **Fixed Interval** | Retrain every K steps | Predictable compute budget |
| **Adaptive Drift-Triggered** | Retrain when KS test detects drift | Resource-efficient on gradual drift |

### 3. Drift Detection

Kolmogorov-Smirnov (KS) test monitors divergence between:
- Reference window (training data)
- Current observation window

```
Drift Score = KSStatistic(reference_window, current_window)
            = max|F_ref(x) - F_curr(x)|

Trigger: drift_score > threshold (0.5) + consecutive detections
```

### 4. Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error  
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **Temporal Robustness**: Std of per-window MAE
- **Maintenance Cost**: Cumulative retraining events

## Key Findings

### Finding 1: Model Selection is Dataset-Dependent

| Dataset | Best Model | Strategy | MAE Improvement |
|---------|-----------|----------|-----------------|
| Electricity | LightGBM | Fixed K=10 | 1495 → 119 (**92%**) |
| Traffic | LightGBM | Fixed K=10 | 213 → 195 (9%) |
| Brent Oil | TSMixer | Adaptive | 1.81 → 1.39 (23%) |
| WTI Oil | TSMixer | Adaptive | 1.87 → 1.37 (27%) |

### Finding 2: Maintenance Policy Selection is Consequential

**Fixed retraining** provides consistent improvements:
- Electricity/LightGBM: 92% MAE reduction with fixed K=10
- All models benefit from periodic retraining

**Adaptive retraining** excels on low-drift financial data:
- Brent: 45% improvement over fixed (4.00 vs 7.27 MAE)
- WTI: 38% improvement over fixed (3.61 vs 5.85 MAE)

### Finding 3: Accuracy-Compute Trade-off

| Configuration | MAE | Retrains | Compute Cost |
|--------------|-----|----------|--------------|
| LightGBM + Fixed K=10 | 119.5 | 533 | Moderate |
| TSMixer + No Retrain | 1.81 | 1 | Zero |
| TSMixer + Adaptive | 1.39 | 11 | Low |

**Key Insight**: K=10 provides optimal balance; K=1 yields only 17% marginal improvement but 10× more retrains.

### Finding 4: Model-Robustness Characteristics

| Model | Complexity | Volatile Data | Periodic Data |
|-------|------------|---------------|---------------|
| LightGBM | O(T log T) | Excellent | Good |
| TSMixer | O(T² d) | Moderate | Excellent |
| LSTM | O(T H²) | Poor | Moderate |
| Naive | O(1) | Baseline | Baseline |

## Novelty & Contributions

1. **Deployment Simulation Framework**: First open-source benchmark to make deployment simulation, configurable maintenance policies, and drift-correlated error analysis co-equal components of evaluation.

2. **Unified Model Zoo**: Common interface (`fit/predict/update`) across 6 diverse model families enabling controlled comparison.

3. **Comprehensive Retraining Analysis**: Systematic comparison of maintenance policies across multiple domains and drift characteristics.

4. **Domain-Specific Insights**: Actionable guidance for practitioners on model selection based on data characteristics.

## Project Structure

```
driftbench-ts/
├── configs/                     # 72 YAML experiment configurations
├── dashboard-react/             # React visualization dashboard
│   ├── api.py                    # Flask API server
│   └── src/                      # React components
├── datasets/                     # Raw datasets (CSV)
├── driftbench/                   # Core framework
│   ├── models/                   # Model implementations
│   ├── datasets/                 # Dataset loaders
│   ├── simulator/                # Rolling window simulator
│   ├── drift/                    # Drift detection algorithms
│   ├── metrics/                  # Evaluation metrics
│   └── utils/                    # Utilities
├── notebooks/                    # Analysis notebooks
├── results/                      # Experiment results
├── paper.pdf                     # Published paper
├── paper.tex                     # LaTeX source
└── run_*.py                      # Experiment runners
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Experiments

```bash
# Single experiment
python run_experiment.py --config configs/electricity_fixed_retrain_lgbm.yaml

# All 72 configurations
python run_all.py
```

### Dashboard

```bash
# Terminal 1: API server
python dashboard-react/api.py --results results --port 5001

# Terminal 2: React app
cd dashboard-react
npm install && npm run dev
```

## Citation

```bibtex
@article{driftbench-ts,
  title={DriftBench-TS: A Deployment-Oriented Benchmark for Time Series Forecasting Under Distribution Shift},
  author={Sachin and Madhan Karthikeyan and Madhu},
  year={2025},
  institution={Vellore Institute of Technology}
}
```

## License

MIT License
