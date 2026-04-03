"""
Run TSMixer experiments only - to verify the fix.
"""
import os
import sys
import yaml
import subprocess
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ['traffic', 'brent', 'wti', 'electricity']
STRATEGIES = ['no_retrain', 'fixed_retrain', 'adaptive_retrain']
MODEL = 'tsmixer'

RESULTS_BASE = Path('results')
CONFIGS_DIR = Path('configs')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dataset_config(dataset: str) -> dict:
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
    if strategy == 'adaptive_retrain':
        return {
            'enabled': True,
            'detector': 'ks_test',
            'on': 'residual',
            'detector_params': {}
        }
    else:
        return {'enabled': False}


def get_model_params(model: str) -> dict:
    params = {
        'tsmixer': {'input_len': 24, 'hidden_size': 32, 'n_layers': 2, 'epochs': 20},
    }
    return params.get(model, {})


def create_config(dataset: str, strategy: str, model: str) -> dict:
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
            timeout=600
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
    print("=" * 60)
    print("DriftBench-TS - Running TSMixer Experiments Only")
    print("=" * 60)
    print(f"Dataset: {DATASETS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Model: {MODEL}")
    print(f"Total experiments: {len(DATASETS) * len(STRATEGIES)}")
    print("=" * 60)
    
    total = 0
    successful = 0
    
    for dataset in DATASETS:
        print(f"\n📊 Dataset: {dataset}")
        for strategy in STRATEGIES:
            total += 1
            if run_experiment(dataset, strategy, MODEL):
                successful += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {successful}/{total} experiments completed successfully")
    print("=" * 60)


if __name__ == '__main__':
    main()
