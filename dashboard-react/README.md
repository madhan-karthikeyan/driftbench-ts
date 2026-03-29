# DriftBench-TS Dashboard

Modern React dashboard for time-series drift detection benchmarking results.

## Features

- **Overview Page**: Summary cards, best model rankings, filterable results table
- **Dataset Deep Dive**: Error timelines, drift alignment charts, model comparison
- **Global Comparison**: Heatmaps, strategy rankings, efficiency frontier plots
- **Run Explorer**: Window-level analysis, drift score tracking, retraining effects

## Tech Stack

- **Frontend**: React 18 + Vite + React Router + Recharts
- **Styling**: Tailwind CSS (dark mode default)
- **Backend**: Flask API serving existing results data
- **Icons**: Lucide React

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard-react
npm install
```

### 2. Start the Flask API Server

```bash
# From project root
source venv/bin/activate
python dashboard-react/api.py --results results --port 5001
```

### 3. Start the React Dev Server

```bash
# In another terminal
cd dashboard-react
npm run dev
```

### 4. Open Browser

Navigate to: http://localhost:3000

## Project Structure

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

## API Endpoints

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

## Color Scheme

### Models
- Naive: `#64748b` (gray)
- Seasonal Naive: `#f59e0b` (amber)
- Random Forest: `#8b5cf6` (purple)
- LightGBM: `#06b6d4` (cyan)
- LSTM: `#ec4899` (pink)
- TSMixer: `#84cc16` (lime)

### Strategies
- No Retrain: `#ef4444` (red)
- Fixed Retrain: `#3b82f6` (blue)
- Adaptive: `#22c55e` (green)

## Key Visualizations

### Drift → Error → Retrain Flow
The dashboard highlights the causal relationship between:
1. **Drift Detection**: When drift is detected (amber markers)
2. **Error Spike**: Corresponding error increase
3. **Retrain Event**: Model retraining (green markers)
4. **Recovery**: Error stabilization after retraining

### Critical Insights
- **False Positives**: Drift detected without error change
- **Missed Drift**: Error spike without detection
- **Model Robustness**: Which models handle drift better
- **Dataset Dependency**: Performance variation across datasets
