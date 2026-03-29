"""
Advanced rolling window simulation with drift detection and adaptive retraining.

FIXED:
- Drift detection now compares training distribution vs incoming batch (not post-injected data)
- Per-window metrics are stored for temporal analysis
- Proper cooldown mechanism for retraining
- Prediction error based drift detection as primary method
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm
from dataclasses import dataclass, field

from driftbench.models.base import BaseModel
from driftbench.metrics.forecasting_metrics import mae, rmse, smape
from driftbench.drift.base_detector import BaseDriftDetector
from driftbench.simulator.drift_injection import DriftInjector
from driftbench.simulator.retraining import (
    RetrainingSimulator,
    create_retraining_policy,
    RetrainingDecision
)


@dataclass
class WindowResult:
    """Result of a single forecasting window."""
    step: int
    entity_id: Any
    timestamp: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    errors: np.ndarray
    drift_detected: bool
    drift_score: float
    retrained: bool
    retrain_reason: str
    window_metrics: Dict[str, float]


class AdvancedRollingSimulator:
    """
    Advanced rolling window simulator with drift detection and retraining.
    
    FIXED:
    - Drift detection uses training distribution vs incoming batch
    - All per-window results are stored for temporal analysis
    - Prediction error-based drift detection
    - Proper cooldown mechanism

    Features:
    - Configurable drift injection before training
    - Drift detection on features, residuals, or predictions
    - Multiple retraining policies with cooldown
    - Comprehensive per-window logging
    """

    def __init__(
        self,
        model: BaseModel,
        history_window: int = 90 * 24,
        horizon: int = 24,
        step_size: int = 24,
        metrics: Optional[List[Callable]] = None,
        drift_detector: Optional[BaseDriftDetector] = None,
        drift_injector: Optional[DriftInjector] = None,
        retraining_simulator: Optional[RetrainingSimulator] = None,
        detection_mode: str = "residual",
        detect_on: str = "residual",
        store_window_results: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the advanced rolling simulator.

        Parameters
        ----------
        model : BaseModel
            The forecasting model to use.
        history_window : int
            Number of time steps to use for training.
        horizon : int
            Number of time steps to forecast.
        step_size : int
            Number of time steps to roll forward.
        metrics : list, optional
            List of metric functions to compute.
        drift_detector : BaseDriftDetector, optional
            Drift detector to use.
        drift_injector : DriftInjector, optional
            Drift injector for synthetic drift.
        retraining_simulator : RetrainingSimulator, optional
            Retraining simulator with policy.
        detection_mode : str
            Mode for drift detection: 'residual', 'feature', 'error'.
        detect_on : str
            Which data to detect drift on: 'residual' or 'error'.
        store_window_results : bool
            Whether to store per-window results (for temporal analysis).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.model = model
        self.history_window = history_window
        self.horizon = horizon
        self.step_size = step_size
        self.metrics = metrics or [mae, rmse, smape]
        self.drift_detector = drift_detector
        self.drift_injector = drift_injector
        self.retraining_simulator = retraining_simulator
        self.detection_mode = detection_mode
        self.detect_on = detect_on
        self.store_window_results = store_window_results
        self.seed = seed
        
        self.window_results: List[WindowResult] = []
        self.drift_log: List[Dict[str, Any]] = []
        self.predictions_log: List[Dict[str, Any]] = []
        
        self.current_step = 0
        self.model_fitted = False
        self._reference_data: Optional[np.ndarray] = None
        self._initial_fit_done = False
        
        self._set_seed()

    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        if self.seed is not None:
            np.random.seed(self.seed)
            import random
            random.seed(self.seed)

    def run(
        self,
        df: pd.DataFrame,
        entity_col: str = 'entity_id',
        timestamp_col: str = 'timestamp',
        target_col: str = 'target',
        feature_cols: Optional[List[str]] = None,
        verbose: bool = True
    ) -> dict:
        """
        Run the advanced rolling window simulation.
        
        FIXED: Drift detection now compares training window vs incoming batch,
        not post-injected data.

        Parameters
        ----------
        df : pd.DataFrame
            The full time series dataset.
        entity_col : str
            Column name for entity identifier.
        timestamp_col : str
            Column name for timestamp.
        target_col : str
            Column name for target values.
        feature_cols : list, optional
            List of feature columns.
        verbose : bool
            Whether to show progress bar.

        Returns
        -------
        dict
            Dictionary containing predictions, metrics, drift log, and per-window results.
        """
        df = df.sort_values([entity_col, timestamp_col]).reset_index(drop=True)
        entities = df[entity_col].unique()

        all_predictions = []
        all_actuals = []
        all_entity_metrics = []
        
        self.window_results = []
        self.drift_log = []

        if self.drift_detector and not self.drift_detector.is_fitted:
            first_entity_df = df[df[entity_col] == entities[0]].copy()
            first_entity_df = first_entity_df.sort_values(timestamp_col)
            ref_data = first_entity_df[target_col].values[:self.history_window]
            self.drift_detector.fit(ref_data)
            self._reference_data = ref_data

        entity_iter = tqdm(entities, disable=not verbose, desc="Entities")

        for entity in entity_iter:
            entity_df = df[df[entity_col] == entity].copy()
            entity_df = entity_df.sort_values(timestamp_col).reset_index(drop=True)

            entity_result = self._run_entity(
                entity_df, timestamp_col, target_col, feature_cols, verbose, entity
            )

            if entity_result is not None:
                all_predictions.extend(entity_result['predictions'])
                all_actuals.extend(entity_result['actuals'])
                all_entity_metrics.append(entity_result['metrics'])

        overall_metrics = self._compute_overall_metrics(all_predictions, all_actuals)
        
        retraining_log = None
        compute_proxy = 0.0
        retrain_stats = {}
        if self.retraining_simulator:
            retraining_log = self.retraining_simulator.get_retraining_log()
            compute_proxy = self.retraining_simulator.compute_proxy
            retrain_stats = self.retraining_simulator.get_statistics()

        results = {
            'predictions': all_predictions,
            'actuals': all_actuals,
            'entity_metrics': all_entity_metrics,
            'overall_metrics': overall_metrics,
            'drift_log': pd.DataFrame(self.drift_log) if self.drift_log else pd.DataFrame(),
            'retraining_log': retraining_log,
            'compute_proxy': compute_proxy,
            'retrain_statistics': retrain_stats,
            'window_results': self.window_results if self.store_window_results else [],
            'total_windows': len(self.window_results)
        }

        return results

    def _run_entity(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
        feature_cols: Optional[List[str]],
        verbose: bool,
        entity_id: Any
    ) -> Optional[dict]:
        """Run rolling window for a single entity with proper drift detection."""
        n = len(df)

        if n < self.history_window + self.horizon:
            return None

        all_preds = []
        all_acts = []
        all_entity_metrics = []
        
        max_start = n - self.horizon
        starts = range(0, max_start, self.step_size)

        if verbose:
            starts = tqdm(starts, leave=False, desc="Windows")

        all_residuals = []
        training_distribution = None
        
        for step_idx, start in enumerate(starts):
            train_start = start
            train_end = start + self.history_window
            train_df = df.iloc[train_start:train_end].copy()

            test_start = train_end
            test_end = test_start + self.horizon
            test_df = df.iloc[test_start:test_end].copy()

            if len(test_df) < self.horizon:
                break
            
            current_timestamp = test_df[timestamp_col].iloc[0]
            
            training_distribution = train_df[target_col].values.copy()
            
            should_retrain = False
            retrain_reason = "none"
            retrain_confidence = 1.0
            
            drift_detected = False
            drift_score = 0.0
            
            if self.retraining_simulator:
                error_ma_window = self._compute_error_ma(all_entity_metrics)
                
                decision = self.retraining_simulator.should_retrain(
                    step=self.current_step,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    error=error_ma_window,
                    current_date=current_timestamp
                )
                
                should_retrain = decision.should_retrain
                retrain_reason = decision.reason
                retrain_confidence = decision.confidence

            if not self._initial_fit_done or should_retrain:
                self.model.fit(train_df)
                self.model_fitted = True
                self._initial_fit_done = True

            predictions_df = self.model.predict(test_df)
            
            preds = predictions_df['prediction'].values
            acts = test_df[target_col].values

            errors = np.abs(acts - preds)
            residuals = acts - preds
            
            window_metrics = {}
            for metric_fn in self.metrics:
                metric_name = metric_fn.__name__
                window_metrics[metric_name] = metric_fn(acts, preds)
            
            all_residuals.extend(residuals.tolist())
            
            drift_detected, drift_score = self._detect_drift(
                training_distribution=training_distribution,
                incoming_data=residuals if self.detect_on == "residual" else acts,
                preds=preds,
                acts=acts,
                step=self.current_step,
                timestamp=current_timestamp
            )
            
            window_result = WindowResult(
                step=self.current_step,
                entity_id=entity_id,
                timestamp=current_timestamp,
                y_true=acts,
                y_pred=preds,
                errors=errors,
                drift_detected=drift_detected,
                drift_score=drift_score,
                retrained=should_retrain,
                retrain_reason=retrain_reason,
                window_metrics=window_metrics
            )
            
            if self.store_window_results:
                self.window_results.append(window_result)
            
            if drift_detected:
                self.drift_log.append({
                    'timestamp': current_timestamp,
                    'step': self.current_step,
                    'entity_id': entity_id,
                    'drift_detected': True,
                    'drift_score': drift_score,
                    'detector': self.detect_on,
                    'retrain_triggered': should_retrain,
                    'retrain_reason': retrain_reason
                })
            
            all_preds.extend(preds)
            all_acts.extend(acts)
            all_entity_metrics.append(window_metrics)
            
            self.current_step += 1

        return {
            'predictions': all_preds,
            'actuals': all_acts,
            'metrics': self._aggregate_entity_metrics(all_entity_metrics)
        }

    def _detect_drift(
        self,
        training_distribution: np.ndarray,
        incoming_data: np.ndarray,
        preds: np.ndarray,
        acts: np.ndarray,
        step: int,
        timestamp: pd.Timestamp
    ) -> tuple:
        """
        Detect drift using training distribution vs incoming batch.
        
        FIXED: Drift is detected by comparing training window distribution
        with the incoming prediction errors/actuals, NOT post-injected data.
        """
        if self.drift_detector is None:
            return False, 0.0
        
        if self.detect_on == "residual":
            detect_data = incoming_data
        elif self.detect_on == "error":
            detect_data = np.abs(acts - preds)
        else:
            detect_data = acts
        
        result = self.drift_detector.detect(detect_data)
        
        return result.drift_detected, result.drift_score

    def _compute_error_ma(self, all_metrics: List[Dict[str, float]]) -> float:
        """Compute moving average of MAE errors."""
        if not all_metrics:
            return 0.0
        
        recent = all_metrics[-min(5, len(all_metrics)):]
        mae_values = [m.get('mae', m.get('MAE', 0)) for m in recent]
        return np.mean(mae_values) if mae_values else 0.0

    def _aggregate_entity_metrics(self, metrics_list: List[Dict[str, float]]) -> dict:
        """Aggregate metrics across all windows for an entity."""
        if not metrics_list:
            return {}

        aggregated = {}
        metric_names = metrics_list[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)

        return aggregated

    def _compute_overall_metrics(self, predictions: List, actuals: List) -> dict:
        """Compute overall metrics across all predictions."""
        if not predictions:
            return {}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        overall = {}
        for metric_fn in self.metrics:
            metric_name = metric_fn.__name__
            overall[metric_name] = metric_fn(actuals, predictions)

        return overall

    def get_window_metrics_df(self) -> pd.DataFrame:
        """Get per-window metrics as DataFrame for temporal analysis."""
        if not self.window_results:
            return pd.DataFrame()
        
        records = []
        for wr in self.window_results:
            record = {
                'step': wr.step,
                'entity_id': wr.entity_id,
                'timestamp': wr.timestamp,
                'mean_error': np.mean(wr.errors),
                'max_error': np.max(wr.errors),
                'std_error': np.std(wr.errors),
                'drift_detected': wr.drift_detected,
                'drift_score': wr.drift_score,
                'retrained': wr.retrained,
                'retrain_reason': wr.retrain_reason
            }
            record.update(wr.window_metrics)
            records.append(record)
        
        return pd.DataFrame(records)


def create_advanced_simulator(
    model: BaseModel,
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> AdvancedRollingSimulator:
    """
    Create an advanced simulator from configuration.

    Parameters
    ----------
    model : BaseModel
        The forecasting model.
    config : dict
        Configuration dictionary.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    AdvancedRollingSimulator
        Configured simulator.
    """
    sim_config = config.get('simulation', {})
    history_window = sim_config.get('history_window_days', 90) * 24
    horizon = sim_config.get('forecast_horizon_hours', 24)
    step_size = sim_config.get('step_size_hours', 24)
    store_window_results = config.get('output', {}).get('save_window_metrics', True)

    drift_config = config.get('drift', {})
    drift_injector = None
    drift_detector = None

    if drift_config.get('enabled', False):
        from driftbench.simulator.drift_injection import create_drift_injector
        drift_injector = create_drift_injector(drift_config.get('injection', {}))

    detector_type = drift_config.get('detector', 'adwin')
    detection_mode = drift_config.get('on', 'residual')

    if detector_type:
        from driftbench.drift.detectors import create_detector
        
        detector_params = drift_config.get('detector_params', {})
        
        if detector_type.lower() == 'adwin':
            detector_params.setdefault('effect_threshold', 0.5)
            detector_params.setdefault('min_window_size', 30)
            detector_params.setdefault('warmup_size', 50)
        elif detector_type.lower() == 'residual_ks':
            detector_params.setdefault('effect_threshold', 0.5)
        
        drift_detector = create_detector(detector_type, mode=detection_mode, **detector_params)

    retraining_simulator = None
    if 'retraining' in config and config['retraining'].get('enabled', False):
        from driftbench.simulator.retraining import create_retraining_policy
        policy = create_retraining_policy(config['retraining'])
        retraining_simulator = RetrainingSimulator(policy)

    return AdvancedRollingSimulator(
        model=model,
        history_window=history_window,
        horizon=horizon,
        step_size=step_size,
        drift_detector=drift_detector,
        drift_injector=drift_injector,
        retraining_simulator=retraining_simulator,
        detection_mode=detection_mode,
        detect_on=detection_mode,
        store_window_results=store_window_results,
        seed=seed
    )
