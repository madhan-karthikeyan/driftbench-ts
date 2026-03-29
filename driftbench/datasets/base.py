"""
Base dataset class.
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for all datasets."""

    def __init__(self, name: str):
        self.name = name
        self.data = None

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Load the dataset."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset."""
        pass
