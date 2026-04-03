"""
Multi-seed experiment runner for robust statistical evaluation.

This module runs experiments across multiple random seeds and computes
statistical summaries (mean, std, confidence intervals).
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driftbench.datasets import get_dataset
from driftbench.models import get_model, get_available_models
from driftbench.simulator.advanced_rolling import create_advanced_simulator
from driftbench.metrics.forecasting_metrics import compute_all_metrics
from driftbench.utils.logging import setup_logger


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run_single_seed(
    config: dict,
    seed: int,
    logger
) -> Dict[str, Any]:
    """
    Run experiment for a single seed.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    seed : int
        Random seed.
    logger : logging.Logger
        Logger instance.
    
    Returns
    -------
    dict
        Results for this seed.
    """
    logger.info(f"Running experiment with seed {seed}")
    
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    dataset_config = config['dataset']
    df = get_dataset(
        dataset_config['name'],
        sample_entities=dataset_config.get('sample_entities', None),
        downsample_freq=dataset_config.get('downsample_freq', 'h')
    )
    
    model_config = config['model']
    model_params = model_config.get('params', {})
    model_params['random_state'] = seed
    model = get_model(model_config['name'], **model_params)
    
    simulator = create_advanced_simulator(model, config, seed=seed)
    
    results = simulator.run(df, verbose=False)
    
    metrics = results.get('overall_metrics', {})
    
    return {
        'seed': seed,
        'metrics': metrics,
        'n_windows': results.get('total_windows', 0),
        'n_predictions': len(results.get('predictions', [])),
        'retrain_stats': results.get('retrain_statistics', {}),
        'drift_log': results.get('drift_log', pd.DataFrame()),
        'retraining_log': results.get('retraining_log', pd.DataFrame())
    }


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistical summaries across seeds.
    
    Parameters
    ----------
    results : List[Dict]
        List of results from each seed.
    
    Returns
    -------
    dict
        Statistical summaries.
    """
    if not results:
        return {}
    
    metric_keys = results[0]['metrics'].keys()
    
    stats = {}
    for metric in metric_keys:
        values = [r['metrics'].get(metric, np.nan) for r in results]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats[metric] = {
                'mean': mean,
                'std': std,
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    total_retrains = [r['retrain_stats'].get('total_retrains', 0) for r in results]
    if total_retrains:
        stats['total_retrains'] = {
            'mean': np.mean(total_retrains),
            'std': np.std(total_retrains, ddof=1) if len(total_retrains) > 1 else 0.0,
            'min': np.min(total_retrains),
            'max': np.max(total_retrains),
            'values': total_retrains
        }
    
    return stats


def run_multi_seed_experiment(
    config_path: str,
    seeds: List[int] = None,
    output_dir: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run experiment across multiple seeds.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    seeds : List[int]
        List of random seeds (default: [42, 43, 44, 45, 46]).
    output_dir : str
        Output directory for results.
    verbose : bool
        Whether to log progress.
    
    Returns
    -------
    dict
        Aggregated results with statistics.
    """
    seeds = seeds or DEFAULT_SEEDS
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger("driftbench_multi_seed")
    
    model_name = config.get('model', {}).get('name', 'unknown')
    dataset_name = config.get('dataset', {}).get('name', 'unknown')
    policy = config.get('retraining', {}).get('policy', 'no_retraining')
    
    logger.info(f"Running multi-seed experiment: {dataset_name}/{model_name}/{policy}")
    logger.info(f"Seeds: {seeds}")
    
    results = []
    for seed in seeds:
        try:
            result = run_single_seed(config, seed, logger)
            results.append(result)
            logger.info(f"Seed {seed} completed: MAE={result['metrics'].get('mae', 'N/A'):.2f}")
        except Exception as e:
            logger.warning(f"Seed {seed} failed: {e}")
            continue
    
    if not results:
        logger.warning("All seeds failed - returning empty statistics")
        return {
            'config': config,
            'seeds': seeds,
            'n_successful': 0,
            'results': [],
            'statistics': {},
            'timestamp': datetime.now().isoformat()
        }
    
    stats = compute_statistics(results)
    
    output = {
        'config': config,
        'seeds': seeds,
        'n_successful': len(results),
        'results': results,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'multi_seed_results.json'), 'w') as f:
            save_dict = {
                'config': output['config'],
                'seeds': output['seeds'],
                'n_successful': output['n_successful'],
                'statistics': output['statistics'],
                'timestamp': output['timestamp']
            }
            json.dump(save_dict, f, indent=2, default=str)
        
        summary_path = os.path.join(output_dir, 'summary.csv')
        with open(summary_path, 'w') as f:
            f.write("metric,mean,std,min,max\n")
            for metric, stat in stats.items():
                if isinstance(stat, dict) and 'mean' in stat:
                    min_val = stat.get('min', stat['mean'])
                    max_val = stat.get('max', stat['mean'])
                    f.write(f"{metric},{stat['mean']:.4f},{stat['std']:.4f},{min_val:.4f},{max_val:.4f}\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    return output


def run_all_experiments(
    configs_dir: str = "configs",
    seeds: List[int] = None,
    output_base: str = "results_multi_seed"
) -> pd.DataFrame:
    """
    Run all configurations with multiple seeds.
    
    Parameters
    ----------
    configs_dir : str
        Directory containing config files.
    seeds : List[int]
        Random seeds.
    output_base : str
        Base output directory.
    
    Returns
    -------
    pd.DataFrame
        Summary of all experiments.
    """
    seeds = seeds or DEFAULT_SEEDS
    
    logger = setup_logger("run_all")
    logger.info("Starting multi-seed experiment suite")
    
    config_files = sorted(Path(configs_dir).glob("*.yaml"))
    
    all_results = []
    
    for config_path in config_files:
        logger.info(f"Processing {config_path.name}")
        
        try:
            output_dir = os.path.join(
                output_base,
                config_path.stem
            )
            
            result = run_multi_seed_experiment(
                str(config_path),
                seeds=seeds,
                output_dir=output_dir,
                verbose=False
            )
            
            stats = result['statistics']
            
            row = {
                'config': config_path.stem,
                'n_seeds': result['n_successful'],
                'mae_mean': stats.get('mae', {}).get('mean', np.nan),
                'mae_std': stats.get('mae', {}).get('std', np.nan),
                'rmse_mean': stats.get('rmse', {}).get('mean', np.nan),
                'rmse_std': stats.get('rmse', {}).get('std', np.nan),
                'smape_mean': stats.get('smape', {}).get('mean', np.nan),
                'smape_std': stats.get('smape', {}).get('std', np.nan),
                'retrains_mean': stats.get('total_retrains', {}).get('mean', np.nan),
                'retrains_std': stats.get('total_retrains', {}).get('std', np.nan)
            }
            
            all_results.append(row)
            mae_val = row['mae_mean']
            mae_std = row['mae_std']
            if not np.isnan(mae_val) and not np.isnan(mae_std):
                logger.info(f"  Completed: MAE={mae_val:.2f}±{mae_std:.2f}")
            elif not np.isnan(mae_val):
                logger.info(f"  Completed: MAE={mae_val:.2f}")
            else:
                logger.info(f"  Completed: MAE=N/A (all seeds failed or no metrics)")
            
            if result.get('n_successful', 0) == 0:
                logger.warning(f"  Warning: No successful seeds for {config_path.stem}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            all_results.append({
                'config': config_path.stem,
                'n_seeds': 0,
                'error': str(e)
            })
    
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_base, 'all_experiments_summary.csv'), index=False)
    
    logger.info(f"All experiments complete. Summary saved to {output_base}/all_experiments_summary.csv")
    
    return summary_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to single config file')
    parser.add_argument('--configs-dir', type=str, default='configs',
                       help='Directory with configs to run')
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46',
                       help='Comma-separated seeds')
    parser.add_argument('--output', type=str, default='results_multi_seed',
                       help='Output directory')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.config:
        run_multi_seed_experiment(
            args.config,
            seeds=seeds,
            output_dir=args.output
        )
    else:
        run_all_experiments(
            configs_dir=args.configs_dir,
            seeds=seeds,
            output_base=args.output
        )


if __name__ == "__main__":
    main()
