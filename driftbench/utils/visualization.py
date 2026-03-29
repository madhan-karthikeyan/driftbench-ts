"""
Visualization utilities for drift detection benchmarking.

Creates temporal plots for analyzing:
- Error trajectory over time
- Drift score over time
- Retraining event overlay
- Comparison between experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse


def plot_drift_analysis(
    window_metrics_path: str,
    output_dir: Optional[str] = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive drift analysis visualizations.
    
    Parameters
    ----------
    window_metrics_path : str
        Path to window_metrics.csv file.
    output_dir : str, optional
        Directory to save plots. If None, uses current directory.
    show : bool
        Whether to display plots (default: True).
    
    Returns
    -------
    dict
        Summary statistics from the analysis.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return {}
    
    df = pd.read_csv(window_metrics_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    output_dir = Path(output_dir) if output_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    axes[0].plot(df['timestamp'], df['mae'], 'b-', alpha=0.7, label='MAE')
    if 'mae_mean' in df.columns:
        rolling_mae = df['mae'].rolling(10, min_periods=1).mean()
        axes[0].plot(df['timestamp'], rolling_mae, 'b-', linewidth=2, label='MAE (10-step MA)')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Error Trajectory Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'drift_score' in df.columns:
        axes[1].plot(df['timestamp'], df['drift_score'], 'r-', alpha=0.7, label='Drift Score')
        if 'effect_threshold' in df.columns:
            axes[1].axhline(y=df['effect_threshold'].iloc[0], color='k', linestyle='--', 
                          label='Threshold')
        axes[1].set_ylabel('Drift Score')
        axes[1].set_title('Drift Detection Score Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    drift_mask = df['drift_detected'] == True if 'drift_detected' in df.columns else pd.Series(False, index=df.index)
    axes[2].bar(df.loc[drift_mask, 'timestamp'], df.loc[drift_mask, 'drift_score'], 
                color='red', alpha=0.6, label='Drift Detected', width=0.8)
    axes[2].set_ylabel('Drift Score')
    axes[2].set_title('Drift Events')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    if 'retrained' in df.columns:
        retrain_mask = df['retrained'] == True
        for ts in df.loc[retrain_mask, 'timestamp']:
            axes[2].axvline(x=ts, color='green', alpha=0.3, linewidth=2)
    
    axes[3].scatter(df['timestamp'], df['drift_score'], c=df['mae'], 
                   cmap='viridis', alpha=0.6, s=20)
    axes[3].set_ylabel('Drift Score')
    axes[3].set_xlabel('Timestamp')
    axes[3].set_title('Drift Score vs Error (color = MAE)')
    cbar = plt.colorbar(axes[3].collections[0], ax=axes[3])
    cbar.set_label('MAE')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'drift_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved drift analysis to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return compute_summary_stats(df)


def plot_experiment_comparison(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    Compare metrics across multiple experiments.
    
    Parameters
    ----------
    results_dirs : list
        List of experiment result directories.
    labels : list, optional
        Labels for each experiment.
    output_dir : str, optional
        Directory to save plots.
    show : bool
        Whether to display plots.
    
    Returns
    -------
    dict
        Comparison statistics.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return {}
    
    labels = labels or [f"Exp {i}" for i in range(len(results_dirs))]
    output_dir = Path(output_dir) if output_dir else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    for results_dir in results_dirs:
        results_path = Path(results_dir)
        
        metrics_file = list(results_path.glob('metrics.json'))
        if metrics_file:
            import json
            with open(metrics_file[0]) as f:
                metrics = json.load(f)
            metrics_data.append(metrics)
    
    if not metrics_data:
        print("No metrics found in results directories")
        return {}
    
    metrics_names = ['mae', 'rmse', 'smape', 'compute_proxy']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_names):
        values = [m.get(metric, 0) for m in metrics_data]
        bars = axes[i].bar(labels, values, alpha=0.7)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Value')
        
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'experiment_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    comparison_df = pd.DataFrame(metrics_data, index=labels)
    comparison_df.index.name = 'Experiment'
    
    return {'comparison': comparison_df.to_dict()}


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics from window metrics."""
    stats = {
        'total_windows': len(df),
        'drift_events': int(df['drift_detected'].sum()) if 'drift_detected' in df.columns else 0,
        'drift_rate': float(df['drift_detected'].mean()) if 'drift_detected' in df.columns else 0,
        'retrain_events': int(df['retrained'].sum()) if 'retrained' in df.columns else 0,
        'retrain_rate': float(df['retrained'].mean()) if 'retrained' in df.columns else 0,
        'mean_mae': float(df['mae'].mean()) if 'mae' in df.columns else 0,
        'std_mae': float(df['mae'].std()) if 'mae' in df.columns else 0,
        'mean_drift_score': float(df['drift_score'].mean()) if 'drift_score' in df.columns else 0,
        'max_drift_score': float(df['drift_score'].max()) if 'drift_score' in df.columns else 0,
    }
    
    if 'drift_detected' in df.columns and 'retrained' in df.columns:
        drift_mae = df.loc[df['drift_detected'], 'mae'].mean() if df['drift_detected'].any() else 0
        no_drift_mae = df.loc[~df['drift_detected'], 'mae'].mean() if (~df['drift_detected']).any() else 0
        stats['mae_during_drift'] = drift_mae
        stats['mae_without_drift'] = no_drift_mae
        stats['mae_increase_during_drift'] = drift_mae - no_drift_mae
    
    return stats


def generate_report(window_metrics_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate a text report summarizing the experiment results.
    
    Parameters
    ----------
    window_metrics_path : str
        Path to window_metrics.csv.
    output_path : str, optional
        Path to save report.
    
    Returns
    -------
    str
        Report text.
    """
    df = pd.read_csv(window_metrics_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    stats = compute_summary_stats(df)
    
    report = []
    report.append("=" * 60)
    report.append("DRIFTBENCH-TS EXPERIMENT SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"Total Windows Evaluated: {stats['total_windows']}")
    report.append(f"Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    report.append("")
    
    report.append("DRIFT DETECTION:")
    report.append(f"  - Drift Events: {stats['drift_events']}")
    report.append(f"  - Drift Rate: {stats['drift_rate']:.2%}")
    report.append(f"  - Mean Drift Score: {stats['mean_drift_score']:.4f}")
    report.append(f"  - Max Drift Score: {stats['max_drift_score']:.4f}")
    report.append("")
    
    report.append("RETRAINING:")
    report.append(f"  - Retrain Events: {stats['retrain_events']}")
    report.append(f"  - Retrain Rate: {stats['retrain_rate']:.2%}")
    report.append("")
    
    report.append("ERROR METRICS:")
    report.append(f"  - Mean MAE: {stats['mean_mae']:.4f}")
    report.append(f"  - Std MAE: {stats['std_mae']:.4f}")
    
    if 'mae_during_drift' in stats:
        report.append(f"  - MAE During Drift: {stats['mae_during_drift']:.4f}")
        report.append(f"  - MAE Without Drift: {stats['mae_without_drift']:.4f}")
        report.append(f"  - MAE Increase: {stats['mae_increase_during_drift']:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_path}")
    
    return report_text


def main():
    """CLI for visualization utilities."""
    parser = argparse.ArgumentParser(description="Visualize drift detection results")
    parser.add_argument('--window-metrics', type=str, required=True,
                       help='Path to window_metrics.csv')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate text report')
    args = parser.parse_args()
    
    plot_drift_analysis(
        args.window_metrics,
        output_dir=args.output_dir,
        show=not args.no_show
    )
    
    if args.report:
        report_path = Path(args.output_dir) / 'experiment_report.txt'
        generate_report(args.window_metrics, str(report_path))
        print(generate_report(args.window_metrics))


if __name__ == "__main__":
    main()
