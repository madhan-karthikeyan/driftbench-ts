"""
Naive baseline model - predicts the previous value.
"""

import pandas as pd
from typing import Optional
import logging

from .base import BaseModel
from .registry import register_model

logger = logging.getLogger(__name__)


@register_model('naive')
class NaiveModel(BaseModel):
    """
    Naive baseline model that predicts the previous value.
    
    This is a simple but important baseline - if your model doesn't beat
    this, it has no predictive power.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Naive")
        self.last_value = None
        
    @classmethod
    def is_available(cls) -> bool:
        return True
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'NaiveModel':
        """Fit by storing the last target value."""
        logger.info("Fitting Naive model...")
        
        if 'target' in df.columns:
            self.last_value = df['target'].iloc[-1]
        
        self.is_fitted = True
        logger.info(f"Naive model fitted. Last value: {self.last_value}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict by repeating the last known value."""
        logger.debug(f"Predicting with Naive model. Last value: {self.last_value}")
        
        result = df.copy()
        result['prediction'] = self.last_value if self.last_value is not None else 0
        
        return result
    
    def get_params(self) -> dict:
        return {'last_value': self.last_value}
