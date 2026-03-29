"""
Rolling window simulation for time series forecasting experiments.
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable
from tqdm import tqdm

from driftbench.models.base import BaseModel
from driftbench.metrics.forecasting_metrics import mae, rmse, smape


class RollingSimulator:
    """
    Rolling window simulator for time series forecasting.

    This simulator performs rolling window forecasting where:
    1. It uses a fixed history window to train the model
    2. Makes predictions for a horizon period
    3. Rolls forward by a step size and repeats
    """

    def __init__(
        self,
        model: BaseModel,
        history_window: int = 90 * 24,  # hours
        horizon: int = 24,
        step_size: int = 24,
        metrics: Optional[list] = None
    ):
        """
        Initialize the rolling simulator.

        Parameters
        ----------
        model : BaseModel
            The forecasting model to use.
        history_window : int
            Number of time steps to use for training (default: 90 days * 24 hours).
        horizon : int
            Number of time steps to forecast (default: 24 hours).
        step_size : int
            Number of time steps to roll forward (default: 24 hours).
        metrics : list, optional
            List of metric functions to compute. Defaults to [mae, rmse, smape].
        """
        self.model = model
        self.history_window = history_window
        self.horizon = horizon
        self.step_size = step_size
        self.metrics = metrics or [mae, rmse, smape]

    def run(
        self,
        df: pd.DataFrame,
        entity_col: str = 'entity_id',
        timestamp_col: str = 'timestamp',
        target_col: str = 'target',
        verbose: bool = True
    ) -> dict:
        """
        Run the rolling window simulation.

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
        verbose : bool
            Whether to show progress bar.

        Returns
        -------
        dict
            Dictionary containing predictions and metrics.
        """
        # Sort by entity and timestamp
        df = df.sort_values([entity_col, timestamp_col]).reset_index(drop=True)

        # Get unique entities
        entities = df[entity_col].unique()

        all_predictions = []
        all_actuals = []
        all_metrics = []

        entity_iter = tqdm(entities, disable=not verbose, desc="Entities")

        for entity in entity_iter:
            entity_df = df[df[entity_col] == entity].copy()
            entity_df = entity_df.sort_values(timestamp_col).reset_index(drop=True)

            # Run rolling window for this entity
            entity_result = self._run_entity(
                entity_df, timestamp_col, target_col, verbose
            )

            if entity_result is not None:
                all_predictions.extend(entity_result['predictions'])
                all_actuals.extend(entity_result['actuals'])
                all_metrics.append(entity_result['metrics'])

        # Aggregate metrics across all entities
        results = {
            'predictions': all_predictions,
            'actuals': all_actuals,
            'entity_metrics': all_metrics,
            'overall_metrics': self._compute_overall_metrics(all_predictions, all_actuals)
        }

        return results

    def _run_entity(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
        verbose: bool
    ) -> Optional[dict]:
        """Run rolling window for a single entity."""
        n = len(df)

        if n < self.history_window + self.horizon:
            return None

        all_preds = []
        all_acts = []
        all_entity_metrics = []

        # Calculate number of rolling windows
        max_start = n - self.horizon
        starts = range(0, max_start, self.step_size)

        if verbose:
            starts = tqdm(starts, leave=False, desc="Windows")

        for start in starts:
            # Training data
            train_start = start
            train_end = start + self.history_window
            train_df = df.iloc[train_start:train_end].copy()

            # Test data (what we're predicting)
            test_start = train_end
            test_end = test_start + self.horizon
            test_df = df.iloc[test_start:test_end].copy()

            if len(test_df) < self.horizon:
                break

            # Fit model
            self.model.fit(train_df)

            # Predict
            predictions_df = self.model.predict(test_df)

            # Store predictions and actuals
            preds = predictions_df['prediction'].values
            acts = test_df[target_col].values

            all_preds.extend(preds)
            all_acts.extend(acts)

            # Compute metrics for this window
            window_metrics = {}
            for metric_fn in self.metrics:
                metric_name = metric_fn.__name__
                window_metrics[metric_name] = metric_fn(acts, preds)

            all_entity_metrics.append(window_metrics)

        return {
            'predictions': all_preds,
            'actuals': all_acts,
            'metrics': self._aggregate_entity_metrics(all_entity_metrics)
        }

    def _aggregate_entity_metrics(self, metrics_list: list) -> dict:
        """Aggregate metrics across all windows for an entity."""
        if not metrics_list:
            return {}

        aggregated = {}
        metric_names = metrics_list[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)

        return aggregated

    def _compute_overall_metrics(self, predictions: list, actuals: list) -> dict:
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
