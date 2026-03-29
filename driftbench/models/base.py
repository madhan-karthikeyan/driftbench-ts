"""
Base model interface for time series forecasting.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Base class for all forecasting models."""

    def __init__(self, name: str = "base_model"):
        self.name = name
        self.is_fitted = False
        self.model = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> 'BaseModel':
        """
        Fit the model to the training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and target.

        Returns
        -------
        self
            Fitted model instance.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.

        Parameters
        ----------
        df : pd.DataFrame
            Data to make predictions on.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}

    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
