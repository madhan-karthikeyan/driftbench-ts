"""
LSTM forecasting model.
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
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM model will be unavailable.")


@register_model('lstm')
class LSTMModel(BaseModel):
    """
    LSTM forecasting model.
    
    Uses PyTorch LSTM for time series forecasting when available.
    Falls back to Naive model when PyTorch is not installed.
    """
    
    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 24,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(name="LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = kwargs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if LSTM_AVAILABLE else None
        self.model = None
        self.history = None
        self.scaler = None
        self.fallback = None
        
        if not LSTM_AVAILABLE:
            logger.warning("PyTorch not available. Using Naive fallback for LSTM.")
            from .naive import NaiveModel
            self.fallback = NaiveModel()
    
    @classmethod
    def is_available(cls) -> bool:
        return LSTM_AVAILABLE
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'LSTMModel':
        """Fit the LSTM model."""
        if self.fallback is not None:
            self.fallback.fit(df)
            self.is_fitted = True
            return self
        
        logger.info("Fitting LSTM model...")
        
        df = df.copy()
        target = df['target'].values
        
        self.scaler = StandardScaler()
        scaled_target = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        sequences, labels = self._create_sequences(scaled_target)
        
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(labels)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = LSTMRegressor(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=1
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
        
        self.history = target[-self.sequence_length:].copy()
        self.is_fitted = True
        
        logger.info("LSTM model fitted successfully.")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the fitted LSTM model."""
        if self.fallback is not None:
            return self.fallback.predict(df)
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
        
        result = df.copy()
        n = len(df)
        predictions = np.zeros(n)
        
        last_sequence = self.history[-self.sequence_length:].copy()
        
        self.model.eval()
        with torch.no_grad():
            for i in range(n):
                seq = torch.FloatTensor(last_sequence.reshape(1, -1, 1)).to(self.device)
                pred = self.model(seq).cpu().numpy()[0, 0]
                predictions[i] = pred
                
                last_sequence = np.append(last_sequence[1:], 
                    df['target'].iloc[i] if 'target' in df.columns else pred)
        
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        self.history = np.append(self.history, predictions)
        
        result['prediction'] = predictions
        return result
    
    def _create_sequences(self, data: np.ndarray) -> tuple:
        """Create input sequences for LSTM."""
        sequences, labels = [], []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            labels.append(data[i + self.sequence_length])
        sequences = np.array(sequences)
        sequences = sequences.reshape(-1, self.sequence_length, 1)
        return sequences, np.array(labels)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'use_fallback': self.fallback is not None
        }


if LSTM_AVAILABLE:
    class LSTMRegressor(nn.Module):
        """LSTM regressor network."""
        
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                     dropout: float, output_size: int):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])


class StandardScaler:
    """Simple standard scaler."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray) -> 'StandardScaler':
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std < 1e-10:
            self.std = 1.0
        return self
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean
