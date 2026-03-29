"""
LightGBM forecasting model.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from .base import BaseModel
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Model will use fallback.")


@register_model('lgbm')
class LGBMModel(BaseModel):
    """
    LightGBM forecasting model wrapper.
    
    Uses LightGBM for time series forecasting with automatic feature engineering
    including lag features, rolling statistics, and time-based features.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        num_leaves: int = 31,
        n_lags: int = 3,
        **kwargs
    ):
        super().__init__(name="LightGBM")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.n_lags = n_lags
        self.params = kwargs
        self.feature_cols = None
        
    @classmethod
    def is_available(cls) -> bool:
        return LGBM_AVAILABLE
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
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
            
            df['rolling_mean_3'] = df['target'].shift(1).rolling(3).mean()
            df['rolling_std_3'] = df['target'].shift(1).rolling(3).std()
            df['rolling_mean_7'] = df['target'].shift(1).rolling(7).mean()
            
            if len(df) > 24:
                df['seasonal_diff'] = df['target'].shift(1) - df['target'].shift(24)
        
        df = df.dropna()
        
        feature_cols = [
            c for c in df.columns
            if c not in ['target', 'prediction', 'timestamp', 'entity_id']
            and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']
        ]
        
        return df, feature_cols
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'LGBMModel':
        """Fit the LightGBM model."""
        if not LGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        
        logger.info("Fitting LightGBM model...")
        
        df, feature_cols = self._prepare_features(df)
        
        X = df[feature_cols]
        y = df['target']
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            n_jobs=-1,
            verbosity=-1,
            **self.params
        )
        
        self.model.fit(X, y)
        self.feature_cols = feature_cols
        self.is_fitted = True
        
        logger.info(f"LightGBM fitted. Features: {len(feature_cols)}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        original_df = df.copy()
        original_indices = original_df.index.tolist()
        
        df_features = df.copy()
        df_features, _ = self._prepare_features(df_features)
        
        for col in self.feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        
        X = df_features[self.feature_cols]
        predictions = self.model.predict(X)
        
        result = original_df.copy()
        result['prediction'] = np.nan
        
        result.loc[df_features.index, 'prediction'] = predictions
        
        result['prediction'] = result['prediction'].fillna(result['target'])
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'n_lags': self.n_lags
        }
