"""
Quick verification that TSMixer produces different results than Naive.
Runs a single small experiment without the full framework overhead.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driftbench.models import get_model
from driftbench.simulator.rolling import RollingSimulator


def quick_test():
    print("=" * 60)
    print("Quick TSMixer vs Naive Verification")
    print("=" * 60)
    
    # Create small test dataset (simulate electricity with 5 entities)
    np.random.seed(42)
    n_timesteps = 500  # Small for quick test
    n_entities = 3
    
    records = []
    for entity in range(n_entities):
        base = np.random.uniform(50, 100)
        for t in range(n_timesteps):
            val = base + np.sin(t / 24 * np.pi) * 20 + np.random.randn() * 5
            records.append({
                'entity_id': f'entity_{entity}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=t),
                'target': max(0, val)
            })
    
    df = pd.DataFrame(records)
    print(f"Test dataset: {len(df)} rows, {df['entity_id'].nunique()} entities")
    
    # Test both models
    models = ['naive', 'seasonal_naive', 'tsmixer']
    results = {}
    
    for model_name in models:
        print(f"\n🔄 Testing {model_name}...")
        
        model = get_model(model_name)
        simulator = RollingSimulator(
            model=model,
            history_window=24 * 7,  # 1 week
            horizon=24,              # 1 day
            step_size=24             # roll by 1 day
        )
        
        sim_results = simulator.run(df, verbose=False)
        mae = sim_results['overall_metrics'].get('mae', float('inf'))
        results[model_name] = mae
        print(f"   {model_name} MAE: {mae:.4f}")
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for model, mae in results.items():
        print(f"  {model:15} MAE: {mae:.4f}")
    
    # Check if TSMixer is different from baselines
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    ts_mae = results['tsmixer']
    naive_mae = results['naive']
    sn_mae = results['seasonal_naive']
    
    if abs(ts_mae - naive_mae) < 0.01:
        print("❌ TSMixer produces SAME results as Naive!")
        print("   The Naive fallback is still being used.")
    else:
        print("✅ TSMixer produces DIFFERENT results from Naive!")
        print(f"   TSMixer MAE: {ts_mae:.4f}")
        print(f"   Naive MAE: {naive_mae:.4f}")
    
    if ts_mae < sn_mae:
        print(f"✅ TSMixer outperforms Seasonal Naive ({sn_mae:.4f})")
    else:
        print(f"⚠️ TSMixer MAE ({ts_mae:.4f}) >= Seasonal Naive ({sn_mae:.4f})")


if __name__ == '__main__':
    quick_test()
