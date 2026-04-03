#!/usr/bin/env python3
"""
Generate all adaptive retrain results for all models in each dataset.
No timeout - runs sequentially until complete.
Skips existing complete results and models with missing dependencies.
"""

import os
import sys
import subprocess

DATASETS = ['brent', 'electricity', 'traffic']
MODELS = ['lgbm', 'lstm', 'naive', 'rf', 'seasonal_naive', 'tsmixer']

REQUIRED_FILES = ['metrics.json', 'drift_log.csv', 'retraining_log.csv']

venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'bin', 'python')

def has_complete_results(output_dir):
    """Check if results folder has all required files."""
    if not os.path.exists(output_dir):
        return False
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(output_dir, f)):
            return False
    return True

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    failed = []
    skipped = []
    skipped_dep = []
    total = 0
    
    for dataset in DATASETS:
        for model in MODELS:
            config_file = f"configs/{dataset}_adaptive_retrain_{model}.yaml"
            if not os.path.exists(config_file):
                print(f"Config not found: {config_file}, skipping...")
                continue
            
            output_dir = f"results/{dataset}/adaptive_retrain/{model}"
            total += 1
            
            if has_complete_results(output_dir):
                print(f"SKIP: {dataset}/{model} - results already exist")
                skipped.append(f"{dataset}/{model}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running: {dataset} / {model}")
            print(f"Config: {config_file}")
            print(f"Output: {output_dir}")
            print(f"{'='*60}")
            
            cmd = [
                venv_python, "run_experiment.py",
                "--config", config_file,
                "--output", output_dir
            ]
            
            result = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                output = result.stdout + result.stderr
                if "LightGBM is not installed" in output or "PyTorch not available" in output:
                    print(f"SKIP: {dataset}/{model} - missing dependency")
                    skipped_dep.append(f"{dataset}/{model}")
                else:
                    print(f"FAILED: {dataset}/{model}")
                    print(f"Error: {output[-500:]}")
                    failed.append(f"{dataset}/{model}")
            else:
                print(f"SUCCESS: {dataset}/{model}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total experiments: {total}")
    print(f"Skipped (already complete): {len(skipped)}")
    print(f"Skipped (missing dependency): {len(skipped_dep)}")
    print(f"Failed: {len(failed)}")
    
    if skipped_dep:
        print("\nSkipped due to missing dependencies:")
        for d in skipped_dep:
            print(f"  - {d}")
    
    if failed:
        print("\nFailed experiments:")
        for f in failed:
            print(f"  - {f}")
    
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
