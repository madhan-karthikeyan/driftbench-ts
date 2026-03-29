"""
RandomForest forecasting model.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model('rf')
class RandomForestModel(BaseModel):
    """
    RandomForest forecasting model wrapper.
    
    Uses scikit-learn RandomForestRegressor for time series forecasting
    with automatic feature engineering from time series data.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
        n_lags: int = 3,
        **kwargs
    ):
        super().__init__(name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_lags = n_lags
        self.params = kwargs
        self.feature_cols = None
        self.lag_cols = None
        
    @classmethod
    def is_available(cls) -> bool:
        return True
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from time series data."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        if 'target' in df.columns:
            for lag in range(1, self.n_lags + 1):
                df[f'lag_{lag}'] = df['target'].shift(lag)
            
            df['rolling_mean_3'] = df['target'].shift(1).rolling(3, min_periods=1).mean()
            df['rolling_std_3'] = df['target'].shift(1).rolling(3, min_periods=1).std().fillna(0)
            
            if len(df) > 24:
                df['seasonal_diff'] = df['target'].shift(1) - df['target'].shift(24)
            else:
                df['seasonal_diff'] = 0
        
        df = df.fillna(0)
        
        feature_cols = [
            c for c in df.columns
            if c not in ['target', 'prediction', 'timestamp', 'entity_id']
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']
        ]
        
        return df, feature_cols
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'RandomForestModel':
        """Fit the RandomForest model."""
        logger.info("Fitting RandomForest model...")
        
        df, feature_cols = self._prepare_features(df)
        
        X = df[feature_cols]
        y = df['target']
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            **self.params
        )
        
        self.model.fit(X, y)
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        logger.info(f"RandomForest fitted. Features: {len(feature_cols)}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        df, _ = self._prepare_features(df)
        
        X = df[self.feature_cols]
        
        predictions = self.model.predict(X)
        
        result = df.copy()
        result['prediction'] = predictions
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'n_lags': self.n_lags
        }


SklearnRFModel = RandomForestModel
