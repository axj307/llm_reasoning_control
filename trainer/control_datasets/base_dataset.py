"""
Base class for control system datasets.
"""

import pickle
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class BaseControlDataset(ABC):
    """Base class for control system datasets."""
    
    def __init__(self, system_name: str, num_samples: int, dt: float, steps: int, seed: int = 42):
        """
        Initialize base dataset.
        
        Args:
            system_name: Name of the control system
            num_samples: Number of samples to generate
            dt: Time step
            steps: Number of control steps
            seed: Random seed for reproducibility
        """
        self.system_name = system_name
        self.num_samples = num_samples
        self.dt = dt
        self.steps = steps
        self.seed = seed
        self.data = []
        
        # Set random seed
        np.random.seed(seed)
        
    @abstractmethod
    def generate(self) -> None:
        """Generate dataset samples. Must be implemented by subclasses."""
        pass
        
    def save(self, filepath: str, format: str = 'pickle') -> None:
        """
        Save dataset to file.
        
        Args:
            filepath: Path to save file
            format: 'pickle' or 'json'
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.data, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'")
            
    @classmethod
    def load(cls, filepath: str, format: str = 'pickle') -> List[Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to load file
            format: 'pickle' or 'json'
            
        Returns:
            List of dataset samples
        """
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'")
            
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'system_name': self.system_name,
            'num_samples': self.num_samples,
            'dt': self.dt,
            'steps': self.steps,
            'seed': self.seed,
            'actual_samples': len(self.data)
        }
        
    def validate(self) -> bool:
        """Validate dataset integrity."""
        if len(self.data) != self.num_samples:
            print(f"⚠️  Expected {self.num_samples} samples, got {len(self.data)}")
            return False
            
        # Check each sample has required fields
        required_fields = ['prompt', 'answer']
        for i, sample in enumerate(self.data):
            for field in required_fields:
                if field not in sample:
                    print(f"⚠️  Sample {i} missing required field: {field}")
                    return False
                    
        return True
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        return self.data[idx]