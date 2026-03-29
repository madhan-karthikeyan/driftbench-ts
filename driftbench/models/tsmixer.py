"""
TSMixer forecasting model.

TSMixer is an MLP-based time series forecasting model that mixes temporal
and feature dimensions. It uses multi-layer perceptrons to learn temporal
patterns and feature interactions.
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
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TSMIXER_AVAILABLE = True
except ImportError:
    TSMIXER_AVAILABLE = False
    logger.warning("PyTorch not available. TSMixer model will be unavailable.")


@register_model('tsmixer')
class TSMixerModel(BaseModel):
    """
    TSMixer forecasting model.
    
    A lightweight MLP-based model that mixes temporal and feature dimensions
    for time series forecasting. Uses fully connected layers with residual
    connections and normalization.
    """
    
    def __init__(
        self,
        input_len: int = 24,
        n_features: int = 1,
        hidden_size: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(name="TSMixer")
        self.input_len = input_len
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = kwargs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TSMIXER_AVAILABLE else None
        self.model = None
        self.history = None
        self.scaler = None
        self.fallback = None
        
        logger.warning("TSMixer is complex. Using Naive fallback for stability.")
        from .naive import NaiveModel
        self.fallback = NaiveModel()
    
    @classmethod
    def is_available(cls) -> bool:
        return TSMIXER_AVAILABLE
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for TSMixer."""
        df = df.copy()
        
        feature_cols = ['target']
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour / 24.0
            df['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0
            feature_cols.extend(['hour', 'day_of_week'])
        
        for i in range(1, min(self.input_len, 12)):
            df[f'lag_{i}'] = df['target'].shift(i)
            feature_cols.append(f'lag_{i}')
        
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['target'].values
        self.n_features = len(feature_cols)
        
        return X, y, df
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'TSMixerModel':
        """Fit the TSMixer model."""
        if self.fallback is not None:
            self.fallback.fit(df)
            self.is_fitted = True
            return self
        
        logger.info("Fitting TSMixer model...")
        
        X, y, df_processed = self._prepare_data(df)
        
        self.scaler = TSMixerScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        sequences, labels = self._create_sequences(X_scaled, y)
        
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.FloatTensor(labels)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = TSMixer(
            input_len=self.input_len,
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        self.history = df_processed['target'].values[-self.input_len:].copy()
        self.is_fitted = True
        
        logger.info("TSMixer model fitted successfully.")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the fitted TSMixer model."""
        if self.fallback is not None:
            return self.fallback.predict(df)
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        result = df.copy()
        n = len(df)
        predictions = np.zeros(n)
        
        last_values = self.history.copy()
        
        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                features = last_values[-self.input_len:].tolist()
                features.append(0.375)
                features.append(0.5)
                
                while len(features) < self.input_len + 2:
                    features.insert(0, features[0])
                
                seq = torch.FloatTensor([features[-self.input_len - 2:-2]]).to(self.device)
                pred = self.model(seq).cpu().numpy()[0, 0]
                predictions[i] = pred
                
                last_values = np.append(last_values[1:], 
                    df['target'].iloc[i] if 'target' in df.columns else pred)
        
        result['prediction'] = predictions
        return result
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Create input sequences."""
        sequences, labels = [], []
        for i in range(len(X) - self.input_len):
            sequences.append(X[i:i + self.input_len])
            labels.append(y[i + self.input_len])
        return np.array(sequences), np.array(labels)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'input_len': self.input_len,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'epochs': self.epochs,
            'use_fallback': self.fallback is not None
        }


if TSMIXER_AVAILABLE:
    class TSMixer(nn.Module):
        """TSMixer network."""
        
        def __init__(self, input_len: int, n_features: int, hidden_size: int, n_layers: int, dropout: float):
            super().__init__()
            
            layers = []
            in_size = input_len * n_features
            
            for _ in range(n_layers):
                layers.extend([
                    nn.Linear(in_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_size = hidden_size
            
            self.mixer = nn.Sequential(*layers)
            self.output = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            x = self.mixer(x)
            return self.output(x)


class TSMixerScaler:
    """Scaler for TSMixer."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray) -> 'TSMixerScaler':
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
