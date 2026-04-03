"""
TSMixer-style forecasting model using sklearn Ridge Regression.

This implementation uses Ridge Regression with rich temporal features
to capture patterns similar to TSMixer architecture.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from .base import BaseModel
from .registry import register_model

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@register_model('ridge_features')
@register_model('tsmixer_legacy')
class RidgeFeatureModel(BaseModel):
    """
    Feature-based forecasting model using Ridge Regression.
    
    This model uses Ridge Regression with rich temporal features including:
    - Time-based features (hour, day, month, cyclical encodings)
    - Lag features (multiple lookback windows)
    - Rolling statistics (mean, std)
    - Seasonal features
    
    Note: This is NOT the TSMixer architecture (Chen et al. 2023).
    TSMixer is an MLP-based deep learning architecture. This model
    is a feature-engineered linear model for CPU-efficient forecasting.
    """
    
    def __init__(
        self,
        input_len: int = 24,
        hidden_size: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        alpha: float = 1.0,
        epochs: int = 30,
        batch_size: int = 64,
        n_lags: int = 24,
        **kwargs
    ):
        super().__init__(name="RidgeFeatures")
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_lags = n_lags
        self.params = kwargs
        
        self.model = None
        self.scaler = None
        self.history = None
        self.feature_names = None
        self._fitted = False
    
    @classmethod
    def is_available(cls) -> bool:
        return True
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        for i in range(1, min(self.n_lags + 1, 48)):
            df[f'lag_{i}'] = df['target'].shift(i)
        
        for w in [3, 6, 12, 24]:
            df[f'rolling_mean_{w}'] = df['target'].shift(1).rolling(w, min_periods=1).mean()
            df[f'rolling_std_{w}'] = df['target'].shift(1).rolling(w, min_periods=1).std().fillna(0)
        
        if len(df) > 24:
            df['seasonal_diff'] = df['target'].shift(1) - df['target'].shift(24)
            df['seasonal_ratio'] = df['target'].shift(1) / (df['target'].shift(24) + 1e-6)
        
        df = df.dropna()
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        exclude = ['target', 'prediction', 'timestamp', 'entity_id']
        return [c for c in df.columns if c not in exclude 
                and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'RidgeFeatureModel':
        """Fit the Ridge Feature model."""
        logger.info("Fitting RidgeFeature model...")
        
        df_features = self._create_features(df)
        
        if len(df_features) < 100:
            logger.warning("Insufficient data, using simple fallback")
            self.history = df['target'].values[-self.n_lags:].copy()
            self._fitted = True
            return self
        
        feature_cols = self._get_feature_columns(df_features)
        self.feature_names = feature_cols
        
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        self.scaler = FeatureScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)
        
        self.history = df_features['target'].values[-self.input_len:].copy()
        self._fitted = True
        
        logger.info(f"RidgeFeature model fitted. Features: {len(feature_cols)}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions."""
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "RidgeFeatureModel has not been fitted. Call fit() before predict()."
            )
        
        result = df.copy()
        n = len(df)
        
        if 'target' not in df.columns:
            raise ValueError("Input DataFrame must contain 'target' column for feature generation")
        
        # Create extended target series: history + test targets
        extended_target = np.concatenate([self.history, df['target'].values])
        
        pred_df = df.copy()
        
        # Add time features
        if 'timestamp' in pred_df.columns:
            pred_df['hour'] = pred_df['timestamp'].dt.hour
            pred_df['day_of_week'] = pred_df['timestamp'].dt.dayofweek
            pred_df['day_of_month'] = pred_df['timestamp'].dt.day
            pred_df['month'] = pred_df['timestamp'].dt.month
            pred_df['is_weekend'] = (pred_df['day_of_week'] >= 5).astype(int)
            pred_df['hour_sin'] = np.sin(2 * np.pi * pred_df['hour'] / 24)
            pred_df['hour_cos'] = np.cos(2 * np.pi * pred_df['hour'] / 24)
            pred_df['dow_sin'] = np.sin(2 * np.pi * pred_df['day_of_week'] / 7)
            pred_df['dow_cos'] = np.cos(2 * np.pi * pred_df['day_of_week'] / 7)
        
        # Add lag features using extended target
        for j in range(1, min(self.n_lags + 1, 48)):
            pred_df[f'lag_{j}'] = extended_target[-(n + j):-j] if j <= n else extended_target[:j]
            if j <= n:
                pred_df.loc[pred_df.index, f'lag_{j}'] = extended_target[len(self.history) - j : len(self.history) - j + n]
            else:
                pred_df[f'lag_{j}'] = np.nan
        
        # Use a simpler approach: create full extended dataframe for rolling
        full_df = pd.DataFrame({'target': extended_target})
        for w in [3, 6, 12, 24]:
            pred_df[f'rolling_mean_{w}'] = full_df['target'].shift(1).rolling(w, min_periods=1).mean().values[-n:]
            pred_df[f'rolling_std_{w}'] = full_df['target'].shift(1).rolling(w, min_periods=1).std().fillna(0).values[-n:]
        
        if len(pred_df) > 24:
            pred_df['seasonal_diff'] = extended_target[len(self.history) - 1 : len(self.history) - 1 + n] - extended_target[len(self.history) - 24 : len(self.history) - 24 + n]
        
        # Get feature values
        for c in self.feature_names:
            if c not in pred_df.columns:
                pred_df[c] = 0
        
        valid_cols = [c for c in self.feature_names if c in pred_df.columns]
        
        X = pred_df[valid_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        predictions = preds
        result['prediction'] = predictions
        return result
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'input_len': self.input_len,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'alpha': self.alpha,
            'n_lags': self.n_lags
        }


class FeatureScaler:
    """Feature-wise scaler."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray) -> 'FeatureScaler':
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std < 1e-10] = 1.0
        return self
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean
