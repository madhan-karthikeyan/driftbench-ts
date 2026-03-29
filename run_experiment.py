"""
Main experiment runner for drift detection benchmarking.

This enhanced version supports:
- Synthetic drift injection
- Advanced drift detection (ADWIN, Page-Hinkley, etc.)
- Adaptive retraining policies with proper cooldown
- Comprehensive per-window metrics logging
- Reproducibility via seeds

FIXED:
- Per-window metrics storage
- Seed handling for reproducibility
- Proper result saving with window_metrics.csv
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driftbench.datasets import get_dataset
from driftbench.models import get_model, get_available_models, is_model_available, MODEL_REGISTRY
from driftbench.simulator.rolling import RollingSimulator
from driftbench.simulator.advanced_rolling import AdvancedRollingSimulator, create_advanced_simulator
from driftbench.metrics.forecasting_metrics import compute_all_metrics
from driftbench.metrics.extended_metrics import compute_extended_metrics
from driftbench.utils.logging import setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(
    config: dict, 
    logger, 
    use_advanced: bool = True,
    seed: Optional[int] = None
) -> dict:
    """
    Run the forecasting experiment with optional drift detection and retraining.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    logger : logging.Logger
        Logger instance.
    use_advanced : bool
        Whether to use advanced simulator with drift detection.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results dictionary.
    """
    logger.info("Starting experiment...")
    
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    dataset_config = config['dataset']
    logger.info(f"Loading dataset: {dataset_config['name']}")

    df = get_dataset(
        dataset_config['name'],
        sample_entities=dataset_config.get('sample_entities', None),
        downsample_freq=dataset_config.get('downsample_freq', 'h')
    )

    logger.info(f"Dataset loaded: {len(df)} rows, {df['entity_id'].nunique()} entities")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    model_config = config['model']
    logger.info(f"Initializing model: {model_config['name']}")

    model_params = model_config.get('params', {})
    if seed is not None:
        model_params['random_state'] = seed
    model = get_model(model_config['name'], **model_params)

    sim_config = config.get('simulation', {})
    drift_config = config.get('drift', {})
    retraining_config = config.get('retraining', {})

    use_advanced = (
        use_advanced and (
            drift_config.get('enabled', False) or
            retraining_config.get('enabled', False) or
            'retraining' in config
        )
    )

    if use_advanced:
        logger.info("Using advanced simulator with drift detection and retraining")
        simulator = create_advanced_simulator(model, config, seed=seed)
    else:
        logger.info("Using basic rolling simulator")
        history_window = sim_config.get('history_window_days', 90) * 24
        horizon = sim_config.get('forecast_horizon_hours', 24)
        step_size = sim_config.get('step_size_hours', 24)

        simulator = RollingSimulator(
            model=model,
            history_window=history_window,
            horizon=horizon,
            step_size=step_size
        )

    logger.info("Running rolling simulation...")
    results = simulator.run(df, verbose=True)

    logger.info("Computing overall metrics...")
    overall_metrics = results['overall_metrics']

    extended_metrics = {}
    if use_advanced and results.get('predictions'):
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        extended_metrics = compute_extended_metrics(actuals, predictions)

    logger.info("Experiment completed!")
    logger.info(f"Overall MAE: {overall_metrics.get('mae', 'N/A'):.4f}")
    logger.info(f"Overall RMSE: {overall_metrics.get('rmse', 'N/A'):.4f}")
    logger.info(f"Overall SMAPE: {overall_metrics.get('smape', 'N/A'):.4f}")

    if use_advanced:
        if 'drift_log' in results and not results['drift_log'].empty:
            logger.info(f"Drift events detected: {len(results['drift_log'])}")
            logger.info(f"Drift detection rate: {len(results['drift_log']) / max(1, results.get('total_windows', 1)):.2%}")
        
        if 'retraining_log' in results and results['retraining_log'] is not None and not results['retraining_log'].empty:
            logger.info(f"Retraining events: {len(results['retraining_log'])}")
            logger.info(f"Compute proxy (retrains): {results.get('compute_proxy', 0)}")
        
        if 'retrain_statistics' in results:
            stats = results['retrain_statistics']
            if 'retrain_rate' in stats:
                logger.info(f"Retrain rate: {stats['retrain_rate']:.2%}")

    return {
        'config': config,
        'metrics': overall_metrics,
        'extended_metrics': extended_metrics,
        'n_predictions': len(results.get('predictions', [])),
        'n_windows': results.get('total_windows', 0),
        'drift_log': results.get('drift_log', pd.DataFrame()).to_dict('records') if use_advanced else [],
        'retraining_log': results.get('retraining_log', pd.DataFrame()).to_dict('records') if use_advanced and results.get('retraining_log') is not None else [],
        'compute_proxy': results.get('compute_proxy', 0),
        'retrain_statistics': results.get('retrain_statistics', {}),
        'window_results': results.get('window_results', []),
        'detailed_results': results
    }


def save_results(
    results: dict, 
    output_dir: str, 
    prefix: str = "", 
    config: dict = None,
    save_window_metrics: bool = True
):
    """Save experiment results to files with enhanced structure."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = results.get('metrics', {})
    metrics.update(results.get('extended_metrics', {}))
    metrics['compute_proxy'] = results.get('compute_proxy', 0)
    metrics['n_windows'] = results.get('n_windows', 0)
    metrics['n_predictions'] = results.get('n_predictions', 0)
    
    if 'retrain_statistics' in results:
        metrics.update({
            f'retrain_{k}': v for k, v in results['retrain_statistics'].items()
        })

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    drift_log = results.get('drift_log', [])
    if drift_log:
        drift_df = pd.DataFrame(drift_log)
        drift_path = os.path.join(output_dir, "drift_log.csv")
        drift_df.to_csv(drift_path, index=False)

    retraining_log = results.get('retraining_log', [])
    if retraining_log:
        retrain_df = pd.DataFrame(retraining_log)
        retrain_path = os.path.join(output_dir, "retraining_log.csv")
        retrain_df.to_csv(retrain_path, index=False)

    if config:
        config_path = os.path.join(output_dir, "config_snapshot.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    if save_window_metrics and 'window_results' in results and results['window_results']:
        window_df = _create_window_metrics_df(results['window_results'])
        window_path = os.path.join(output_dir, "window_metrics.csv")
        window_df.to_csv(window_path, index=False)
        print(f"Saved per-window metrics: {len(window_df)} rows")

    print(f"Results saved to {output_dir}")


def _create_window_metrics_df(window_results: list) -> pd.DataFrame:
    """Convert window results to DataFrame."""
    records = []
    for wr in window_results:
        record = {
            'step': wr.step if hasattr(wr, 'step') else wr.get('step'),
            'entity_id': wr.entity_id if hasattr(wr, 'entity_id') else wr.get('entity_id'),
            'timestamp': wr.timestamp if hasattr(wr, 'timestamp') else wr.get('timestamp'),
            'mean_error': float(np.mean(wr.errors)) if hasattr(wr, 'errors') else wr.get('mean_error'),
            'max_error': float(np.max(wr.errors)) if hasattr(wr, 'errors') else wr.get('max_error'),
            'std_error': float(np.std(wr.errors)) if hasattr(wr, 'errors') else wr.get('std_error'),
            'drift_detected': wr.drift_detected if hasattr(wr, 'drift_detected') else wr.get('drift_detected'),
            'drift_score': wr.drift_score if hasattr(wr, 'drift_score') else wr.get('drift_score'),
            'retrained': wr.retrained if hasattr(wr, 'retrained') else wr.get('retrained'),
            'retrain_reason': wr.retrain_reason if hasattr(wr, 'retrain_reason') else wr.get('retrain_reason')
        }
        if hasattr(wr, 'window_metrics'):
            record.update(wr.window_metrics)
        records.append(record)
    
    return pd.DataFrame(records)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run drift detection experiment")
    parser.add_argument('--config', type=str, default='configs/electricity.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--basic', action='store_true',
                       help='Use basic simulator without drift detection')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-window-metrics', action='store_true',
                       help='Skip saving per-window metrics')
    args = parser.parse_args()

    logger = setup_logger("driftbench")

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    seed = args.seed if args.seed is not None else config.get('seed')
    
    results = run_experiment(config, logger, use_advanced=not args.basic, seed=seed)

    save_results(
        results, 
        args.output, 
        config=config,
        save_window_metrics=not args.no_window_metrics
    )


if __name__ == "__main__":
    main()
