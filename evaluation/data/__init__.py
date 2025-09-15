"""Data loading and preprocessing utilities for evaluation."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from dataclasses import dataclass
import json

@dataclass
class DatasetConfig:
    """Configuration for loading datasets."""
    name: str
    path: Union[str, Path]
    text_col: str = "text"
    label_col: str = "label"
    split_col: Optional[str] = None
    preprocess_fn: Optional[callable] = None

class DataLoader:
    """Generic data loader for evaluation datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize with dataset configuration."""
        self.config = config
        self.data = None
        
    def load(self) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        path = Path(self.config.path)
        
        if path.suffix == '.csv':
            self.data = pd.read_csv(path)
        elif path.suffix == '.jsonl':
            self.data = pd.read_json(path, lines=True)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Apply preprocessing if specified
        if self.config.preprocess_fn is not None:
            self.data = self.config.preprocess_fn(self.data)
            
        return self.data
    
    def get_splits(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        if self.data is None:
            self.load()
            
        if self.config.split_col and self.config.split_col in self.data.columns:
            train = self.data[self.data[self.config.split_col] == 'train']
            test = self.data[self.data[self.config.split_col] == 'test']
        else:
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(
                self.data, 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.data[self.config.label_col] if self.config.label_col in self.data.columns else None
            )
        return train, test
