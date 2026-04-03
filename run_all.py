"""
Run all experiment combinations for DriftBench-TS.

This script runs experiments for all combinations of:
- Datasets: traffic, brent, wti, electricity
- Strategies: no_retrain, fixed_retrain, adaptive_retrain
- Models: naive, seasonal_naive, rf, lgbm, lstm, tsmixer

Results are saved to results/{dataset}/{strategy}/{model}/
"""
import os
import sys
import yaml
import subprocess
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driftbench.models import get_available_models, is_model_available

DATASETS = ['traffic', 'brent', 'wti', 'electricity']
STRATEGIES = ['no_retrain', 'fixed_retrain', 'adaptive_retrain']

MODELS = ['naive', 'seasonal_naive', 'rf', 'lgbm', 'lstm', 'tsmixer']

RESULTS_BASE = Path('results')
CONFIGS_DIR = Path('configs')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_config(dataset: str) -> dict:
    """Get dataset-specific configuration."""
    configs = {
        'traffic': {
            'sample_entities': 1,
            'downsample_freq': 'h',
            'history_window_days': 30,
            'forecast_horizon_hours': 24,
            'step_size_hours': 24,
        },
        'brent': {
            'sample_entities': 1,
            'downsample_freq': 'w',
            'history_window_days': 60,
            'forecast_horizon_hours': 168,
            'step_size_hours': 168,
        },
        'wti': {
            'sample_entities': 1,
            'downsample_freq': 'w',
            'history_window_days': 60,
            'forecast_horizon_hours': 168,
            'step_size_hours': 168,
        },
        'electricity': {
            'sample_entities': 5,
            'downsample_freq': 'h',
            'history_window_days': 30,
            'forecast_horizon_hours': 24,
            'step_size_hours': 24,
        },
    }
    return configs.get(dataset, configs['traffic'])


def get_strategy_config(strategy: str) -> dict:
    """Get strategy-specific configuration."""
    configs = {
        'no_retrain': {
            'enabled': True,
            'policy': 'no_retraining',
            'min_steps_between_retrain': 999999,
        },
        'fixed_retrain': {
            'enabled': True,
            'policy': 'fixed_schedule',
            'retrain_every_n_steps': 10,
            'min_steps_between_retrain': 1,
        },
        'adaptive_retrain': {
            'enabled': True,
            'policy': 'drift_triggered',
            'min_steps_between_retrain': 3,
            'require_consecutive': 2,
            'drift_threshold': 0.5,
            'decay_consecutive': True,
            'consecutive_decay_rate': 0.5,
        },
    }
    return configs.get(strategy, configs['no_retrain'])


def get_drift_config(strategy: str) -> dict:
    """Get drift configuration based on strategy."""
    if strategy == 'adaptive_retrain':
        return {
            'enabled': True,
            'detector': 'ks_test',
            'on': 'residual',
            'detector_params': {}
        }
    else:
        return {
            'enabled': False
        }


def get_model_params(model: str) -> dict:
    """Get model-specific parameters."""
    params = {
        'naive': {},
        'seasonal_naive': {'season_length': 24},
        'rf': {'n_estimators': 50, 'max_depth': 8, 'random_state': 42, 'n_lags': 3},
        'lgbm': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'n_lags': 3},
        'lstm': {'hidden_size': 32, 'num_layers': 2, 'sequence_length': 24, 'epochs': 20},
        'tsmixer': {'input_len': 24, 'hidden_size': 32, 'n_layers': 2, 'epochs': 20},
    }
    return params.get(model, {})


def create_config(dataset: str, strategy: str, model: str) -> dict:
    """Create a full experiment configuration."""
    ds_config = get_dataset_config(dataset)
    st_config = get_strategy_config(strategy)
    md_params = get_model_params(model)
    
    return {
        'dataset': {
            'name': dataset,
            'sample_entities': ds_config['sample_entities'],
            'downsample_freq': ds_config['downsample_freq'],
        },
        'model': {
            'name': model,
            'params': md_params
        },
        'simulation': {
            'history_window_days': ds_config['history_window_days'],
            'forecast_horizon_hours': ds_config['forecast_horizon_hours'],
            'step_size_hours': ds_config['step_size_hours'],
        },
        'drift': get_drift_config(strategy),
        'retraining': st_config,
        'metrics': ['mae', 'rmse', 'smape'],
        'output': {
            'results_dir': str(RESULTS_BASE / dataset / strategy / model),
            'save_predictions': False,
            'save_metrics': True,
            'save_window_metrics': True,
        },
        'seed': 42,
    }


def run_experiment(dataset: str, strategy: str, model: str) -> bool:
    """Run a single experiment."""
    result_dir = RESULTS_BASE / dataset / strategy / model
    config_path = CONFIGS_DIR / f"{dataset}_{strategy}_{model}.yaml"
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    
    config = create_config(dataset, strategy, model)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Running {dataset}/{strategy}/{model}...")
    
    cmd = [
        sys.executable, 'run_experiment.py',
        '--config', str(config_path),
        '--output', str(result_dir),
        '--no-window-metrics'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info(f"  ✓ Completed")
            return True
        else:
            logger.error(f"  ✗ Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Timeout")
        return False
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False


def main():
    """Run all experiment combinations."""
    available_models = [m for m in MODELS if is_model_available(m)]
    unavailable_models = [m for m in MODELS if not is_model_available(m)]
    
    if unavailable_models:
        logger.warning(f"Models not available (will skip): {unavailable_models}")
    
    if not available_models:
        logger.error("No models available! Check model dependencies.")
        return
    
    print("=" * 60)
    print("DriftBench-TS - Running All Experiments")
    print("=" * 60)
    print(f"Datasets: {DATASETS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Models: {available_models}")
    print(f"Total experiments: {len(DATASETS) * len(STRATEGIES) * len(available_models)}")
    print("=" * 60)
    
    total = 0
    successful = 0
    
    for dataset in DATASETS:
        print(f"\n📊 Dataset: {dataset}")
        for strategy in STRATEGIES:
            print(f"  📈 Strategy: {strategy}")
            for model in available_models:
                total += 1
                if run_experiment(dataset, strategy, model):
                    successful += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {successful}/{total} experiments completed successfully")
    print("=" * 60)


if __name__ == '__main__':
    main()
