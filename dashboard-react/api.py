"""
DriftBench-TS API Server
Serves experiment results as JSON API for the React dashboard.
"""
import json
import os
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder='../dashboard-react/dist', static_url_path='')

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BASE_DIR / 'results')).resolve()


class ResultLoader:
    """Loads experiment results from the results directory."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
    
    def normalize_metrics(self, metrics, drift_count=0, retrain_count=0):
        """Normalize metric field names to standard format."""
        n_windows = metrics.get('n_windows', 1)
        return {
            'mae': metrics.get('mae', 0),
            'rmse': metrics.get('rmse', 0),
            'smape': metrics.get('smape', 0),
            'retrain_total': retrain_count or metrics.get('retrain_total_retrains', 0),
            'retrain_rate': metrics.get('retrain_retrain_rate', 0) * 100,
            'drift_detection_rate': (drift_count / max(n_windows, 1)) * 100,
            'compute_proxy': metrics.get('compute_proxy', 0),
            'n_windows': n_windows,
        }
    
    def load_drift_count(self, dataset, strategy, model):
        """Count drift events from drift_log.csv."""
        path = self.results_dir / dataset / strategy / model / 'drift_log.csv'
        if path.exists():
            df = pd.read_csv(path)
            return len(df[df.get('drift_detected', False) == True]) if 'drift_detected' in df.columns else 0
        return 0
    
    def load_retrain_count(self, dataset, strategy, model):
        """Count retrain events from retraining_log.csv."""
        path = self.results_dir / dataset / strategy / model / 'retraining_log.csv'
        if path.exists():
            df = pd.read_csv(path)
            # Count rows where retrain_reason is not 'initial_training'
            return len(df[df.get('reason', '') != 'initial_training']) if 'reason' in df.columns else 0
        return 0
    
    def load_metrics(self, dataset, strategy, model):
        """Load metrics for a single run."""
        metrics_path = self.results_dir / dataset / strategy / model / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                drift_count = self.load_drift_count(dataset, strategy, model)
                retrain_count = self.load_retrain_count(dataset, strategy, model)
                return self.normalize_metrics(metrics, drift_count, retrain_count)
        return None
    
    def load_windows(self, dataset, strategy, model):
        """Load window metrics as time series."""
        path = self.results_dir / dataset / strategy / model / 'window_metrics.csv'
        if path.exists():
            df = pd.read_csv(path)
            return df.to_dict('records')
        return []
    
    def get_all_runs(self):
        """Get all experiment runs."""
        runs = []
        if not self.results_dir.exists():
            return runs
        for dataset_dir in self.results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for strategy_dir in dataset_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue
                for model_dir in strategy_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    metrics = self.load_metrics(
                        dataset_dir.name, 
                        strategy_dir.name, 
                        model_dir.name
                    )
                    if metrics:
                        runs.append({
                            'dataset': dataset_dir.name,
                            'strategy': strategy_dir.name,
                            'model': model_dir.name,
                            'metrics': metrics,
                        })
        return runs
    
    def get_dataset_runs(self, dataset):
        """Get all runs for a specific dataset."""
        runs = []
        dataset_dir = self.results_dir / dataset
        if not dataset_dir.exists():
            return runs
        for strategy_dir in dataset_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            for model_dir in strategy_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                metrics = self.load_metrics(dataset, strategy_dir.name, model_dir.name)
                if metrics:
                    runs.append({
                        'dataset': dataset,
                        'strategy': strategy_dir.name,
                        'model': model_dir.name,
                        'metrics': metrics,
                    })
        return runs


loader = ResultLoader(RESULTS_DIR)


@app.route('/api/results')
def get_results():
    """Get overview of all results."""
    runs = loader.get_all_runs()
    
    if not runs:
        return jsonify({'error': 'No results found'}), 404
    
    # Compute overview stats
    datasets = list(set(r['dataset'] for r in runs))
    models = list(set(r['model'] for r in runs))
    
    # Best model overall (by avg MAE)
    model_maes = {}
    for model in models:
        model_runs = [r for r in runs if r['model'] == model]
        avg_mae = sum(r['metrics'].get('mae', 0) for r in model_runs) / len(model_runs)
        model_maes[model] = avg_mae
    best_model = min(model_maes, key=model_maes.get)
    
    # Best per dataset
    best_per_dataset = []
    for dataset in datasets:
        dataset_runs = [r for r in runs if r['dataset'] == dataset]
        if dataset_runs:
            best = min(dataset_runs, key=lambda r: r['metrics'].get('mae', float('inf')))
            worst = max(dataset_runs, key=lambda r: r['metrics'].get('mae', 0))
            best_per_dataset.append({
                'dataset': dataset,
                'best_strategy': best['strategy'],
                'best_model': best['model'],
                'best_mae': best['metrics'].get('mae', 0),
                'improvement': ((worst['metrics'].get('mae', 0) - best['metrics'].get('mae', 0)) 
                              / worst['metrics'].get('mae', 1) * 100) if worst['metrics'].get('mae', 0) > 0 else 0
            })
    
    return jsonify({
        'runs': runs,
        'datasets': sorted(datasets),
        'models': sorted(models),
        'strategies': ['no_retrain', 'fixed_retrain', 'adaptive_retrain'],
        'best_per_dataset': best_per_dataset,
        'summary': {
            'total_runs': len(runs),
            'total_datasets': len(datasets),
            'total_models': len(models),
            'best_model': best_model,
            'best_model_mae': round(model_maes[best_model], 2),
        }
    })


@app.route('/api/datasets')
def get_datasets():
    """Get list of all datasets."""
    runs = loader.get_all_runs()
    datasets = sorted(set(r['dataset'] for r in runs))
    return jsonify({'datasets': datasets})


@app.route('/api/dataset/<name>')
def get_dataset(name):
    """Get all runs for a specific dataset with insights."""
    runs = loader.get_dataset_runs(name)
    
    if not runs:
        return jsonify({'error': f'Dataset {name} not found'}), 404
    
    # Compute insights
    strategies = sorted(set(r['strategy'] for r in runs))
    models = sorted(set(r['model'] for r in runs))
    
    # Best combo
    best_run = min(runs, key=lambda r: r['metrics'].get('mae', float('inf')))
    
    # Error timeline data
    timeline = {}
    for run in runs:
        key = f"{run['strategy']}_{run['model']}"
        windows = loader.load_windows(name, run['strategy'], run['model'])
        timeline[key] = {
            'strategy': run['strategy'],
            'model': run['model'],
            'windows': windows,
        }
    
    return jsonify({
        'dataset': name,
        'runs': runs,
        'strategies': strategies,
        'models': models,
        'timeline': timeline,
        'insights': {
            'best_run': best_run,
            'total_runs': len(runs),
            'avg_mae': sum(r['metrics'].get('mae', 0) for r in runs) / len(runs),
        }
    })


@app.route('/api/dataset/<name>/runs')
def get_dataset_runs(name):
    """Get all runs for a dataset."""
    runs = loader.get_dataset_runs(name)
    return jsonify({'runs': runs})


@app.route('/api/run/<dataset>/<strategy>/<model>')
def get_run(dataset, strategy, model):
    """Get a single run."""
    metrics = loader.load_metrics(dataset, strategy, model)
    
    if metrics is None:
        return jsonify({'error': 'Run not found'}), 404
    
    return jsonify({
        'dataset': dataset,
        'strategy': strategy,
        'model': model,
        'metrics': metrics,
    })


@app.route('/api/run/<dataset>/<strategy>/<model>/windows')
def get_run_windows(dataset, strategy, model):
    """Get window metrics for a run."""
    windows = loader.load_windows(dataset, strategy, model)
    return jsonify({'windows': windows})


@app.route('/api/compare')
def get_comparison():
    """Get comparison data for all runs."""
    runs = loader.get_all_runs()
    
    # Heatmap data: dataset x (strategy_model)
    datasets = sorted(set(r['dataset'] for r in runs))
    models = sorted(set(r['model'] for r in runs))
    strategies = ['no_retrain', 'fixed_retrain', 'adaptive_retrain']
    
    heatmap_data = []
    for dataset in datasets:
        for strategy in strategies:
            for model in models:
                run = next((r for r in runs if r['dataset'] == dataset 
                           and r['strategy'] == strategy and r['model'] == model), None)
                if run:
                    heatmap_data.append({
                        'dataset': dataset,
                        'strategy': strategy,
                        'model': model,
                        'mae': run['metrics'].get('mae', 0),
                        'rmse': run['metrics'].get('rmse', 0),
                        'smape': run['metrics'].get('smape', 0),
                        'retrains': run['metrics'].get('retrain_total', 0),
                        'drift_rate': run['metrics'].get('drift_detection_rate', 0),
                    })
    
    # Best strategy per dataset
    best_per_dataset = []
    for dataset in datasets:
        dataset_runs = [r for r in runs if r['dataset'] == dataset]
        best = min(dataset_runs, key=lambda r: r['metrics'].get('mae', float('inf')))
        worst = max(dataset_runs, key=lambda r: r['metrics'].get('mae', 0))
        best_per_dataset.append({
            'dataset': dataset,
            'best_strategy': best['strategy'],
            'best_model': best['model'],
            'best_mae': best['metrics'].get('mae', 0),
            'improvement': ((worst['metrics'].get('mae', 0) - best['metrics'].get('mae', 0)) 
                          / worst['metrics'].get('mae', 1) * 100) if worst['metrics'].get('mae', 0) > 0 else 0
        })
    
    return jsonify({
        'heatmap': heatmap_data,
        'best_per_dataset': best_per_dataset,
        'datasets': datasets,
        'models': models,
        'strategies': strategies,
    })


@app.route('/api/compare/models')
def get_model_comparison():
    """Compare models across all datasets and strategies."""
    runs = loader.get_all_runs()
    models = sorted(set(r['model'] for r in runs))
    
    comparison = []
    for model in models:
        model_runs = [r for r in runs if r['model'] == model]
        avg_mae = sum(r['metrics'].get('mae', 0) for r in model_runs) / len(model_runs)
        avg_rmse = sum(r['metrics'].get('rmse', 0) for r in model_runs) / len(model_runs)
        total_retrains = sum(r['metrics'].get('retrain_total', 0) for r in model_runs)
        comparison.append({
            'model': model,
            'avg_mae': round(avg_mae, 2),
            'avg_rmse': round(avg_rmse, 2),
            'total_retrains': total_retrains,
            'num_runs': len(model_runs),
        })
    
    comparison.sort(key=lambda x: x['avg_mae'])
    return jsonify({'comparison': comparison})


@app.route('/api/compare/strategies')
def get_strategy_comparison():
    """Compare strategies across all datasets and models."""
    runs = loader.get_all_runs()
    strategies = ['no_retrain', 'fixed_retrain', 'adaptive_retrain']
    
    comparison = []
    for strategy in strategies:
        strat_runs = [r for r in runs if r['strategy'] == strategy]
        avg_mae = sum(r['metrics'].get('mae', 0) for r in strat_runs) / len(strat_runs) if strat_runs else 0
        avg_rmse = sum(r['metrics'].get('rmse', 0) for r in strat_runs) / len(strat_runs) if strat_runs else 0
        total_retrains = sum(r['metrics'].get('retrain_total', 0) for r in strat_runs)
        comparison.append({
            'strategy': strategy,
            'avg_mae': round(avg_mae, 2),
            'avg_rmse': round(avg_rmse, 2),
            'total_retrains': total_retrains,
            'num_runs': len(strat_runs),
        })
    
    comparison.sort(key=lambda x: x['avg_mae'])
    return jsonify({'comparison': comparison})


@app.route('/api/heatmap')
def get_heatmap():
    """Get heatmap data."""
    comparison = get_comparison()
    return comparison


@app.route('/api/robustness')
def get_robustness():
    """Get robustness plot data (retrain cost vs error)."""
    runs = loader.get_all_runs()
    
    data = []
    for run in runs:
        mae = run['metrics'].get('mae', 0)
        retrain_cost = run['metrics'].get('compute_proxy', 0)
        data.append({
            'dataset': run['dataset'],
            'strategy': run['strategy'],
            'model': run['model'],
            'mae': mae,
            'retrain_cost': retrain_cost,
        })
    
    return jsonify({'data': data})


@app.route('/')
def serve_index():
    """Serve the React app."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DriftBench-TS API Server')
    parser.add_argument('--results', '-r', default=str(BASE_DIR / 'results'), help='Results directory')
    parser.add_argument('--port', '-p', type=int, default=5001, help='Port')
    parser.add_argument('--host', default='127.0.0.1', help='Host')
    args = parser.parse_args()
    
    results_path = Path(args.results).resolve()
    app.config['RESULTS_DIR'] = str(results_path)
    loader = ResultLoader(results_path)
    
    print(f"\n{'='*60}")
    print("DriftBench-TS API Server")
    print(f"{'='*60}")
    print(f"Results: {results_path}")
    print(f"API: http://{args.host}:{args.port}/api")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=True)
