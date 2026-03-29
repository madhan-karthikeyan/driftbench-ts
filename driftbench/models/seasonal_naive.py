"""
Seasonal naive forecasting model.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from .base import BaseModel
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model('seasonal_naive')
class SeasonalNaiveModel(BaseModel):
    """
    Seasonal naive forecasting model.

    This model predicts future values by copying the values from the same
    season in the past (e.g., same hour yesterday, same hour last week).
    """

    def __init__(self, season_length: int = 24, **kwargs):
        """
        Initialize the seasonal naive model.

        Parameters
        ----------
        season_length : int
            Length of the seasonal period (default: 24 for hourly data).
        """
        super().__init__(name="SeasonalNaive")
        self.season_length = season_length
        self.history = None

    def fit(self, df: pd.DataFrame, **kwargs) -> 'SeasonalNaiveModel':
        """
        Fit the model by storing historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with at least 'target' column.

        Returns
        -------
        self
            Fitted model instance.
        """
        self.history = df['target'].values.copy()
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using seasonal naive approach.

        Parameters
        ----------
        df : pd.DataFrame
            Data to make predictions on. Must have same length as forecast horizon.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions.
        """
        n_forecast = len(df)
        predictions = np.zeros(n_forecast)

        history_len = len(self.history)

        for i in range(n_forecast):
            # Use value from season_length ago
            idx = history_len - self.season_length + (i % self.season_length)
            if idx < history_len and idx >= 0:
                predictions[i] = self.history[idx]
            else:
                # Fallback to mean if not enough history
                predictions[i] = np.mean(self.history)

        result = df.copy()
        result['prediction'] = predictions
        return result
