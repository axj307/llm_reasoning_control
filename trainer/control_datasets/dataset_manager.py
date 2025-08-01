"""
Dataset manager for loading and saving control system datasets.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .base_dataset import BaseControlDataset
from logger import logger


class DatasetManager:
    """Manage dataset generation, saving, and loading."""
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of dataset classes
        self._dataset_classes = {}
        
    def register_dataset_class(self, system_name: str, dataset_class: type) -> None:
        """
        Register a dataset class for a system.
        
        Args:
            system_name: Name of the system
            dataset_class: Dataset class (must inherit from BaseControlDataset)
        """
        if not issubclass(dataset_class, BaseControlDataset):
            raise ValueError(f"{dataset_class} must inherit from BaseControlDataset")
        self._dataset_classes[system_name] = dataset_class
        logger.info(f"Registered dataset class for {system_name}")
        
    def _get_filename(self, system: str, num_samples: int, dt: float, steps: int) -> str:
        """Generate consistent filename for dataset."""
        return f"{system}_{num_samples}samples_dt{dt}_steps{steps}.pkl"
        
    def _get_info_filename(self, system: str, num_samples: int, dt: float, steps: int) -> str:
        """Generate filename for dataset info."""
        base = self._get_filename(system, num_samples, dt, steps)
        return base.replace('.pkl', '_info.json')
        
    def get_or_create(self, system: str, num_samples: int, dt: float, steps: int, 
                     force_regenerate: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """
        Get dataset from cache or create new one.
        
        Args:
            system: System name
            num_samples: Number of samples
            dt: Time step
            steps: Number of control steps
            force_regenerate: Force regeneration even if cached
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            List of dataset samples
        """
        filename = self._get_filename(system, num_samples, dt, steps)
        filepath = self.data_dir / filename
        info_filepath = self.data_dir / self._get_info_filename(system, num_samples, dt, steps)
        
        if filepath.exists() and not force_regenerate:
            logger.info(f"📦 Loading cached dataset from {filepath}")
            data = BaseControlDataset.load(filepath)
            
            # Load and display info if available
            if info_filepath.exists():
                with open(info_filepath, 'r') as f:
                    info = json.load(f)
                logger.info(f"   Generated on: {info.get('timestamp', 'Unknown')}")
                logger.info(f"   Samples: {info.get('num_samples', len(data))}")
            
            return data
        else:
            logger.info(f"🏭 Generating new dataset for {system}...")
            
            # Get dataset class
            if system not in self._dataset_classes:
                # Try to import from systems module
                try:
                    module_name = f"datasets.systems.{system}_dataset"
                    module = __import__(module_name, fromlist=[f"{system.title()}Dataset"])
                    dataset_class = getattr(module, f"{system.title().replace('_', '')}Dataset")
                    self.register_dataset_class(system, dataset_class)
                except ImportError:
                    raise ValueError(f"No dataset class registered for system: {system}")
            
            dataset_class = self._dataset_classes[system]
            dataset = dataset_class(
                system_name=system,
                num_samples=num_samples,
                dt=dt,
                steps=steps,
                **kwargs
            )
            
            # Generate data
            dataset.generate()
            
            # Validate
            if not dataset.validate():
                logger.warning("Dataset validation failed!")
            
            # Save dataset
            dataset.save(filepath)
            logger.info(f"💾 Saved dataset to {filepath}")
            
            # Save metadata
            info = {
                'timestamp': datetime.now().isoformat(),
                'system': system,
                'num_samples': num_samples,
                'dt': dt,
                'steps': steps,
                'seed': dataset.seed,
                'metadata': dataset.get_metadata(),
                'additional_args': kwargs
            }
            with open(info_filepath, 'w') as f:
                json.dump(info, f, indent=2)
            
            return dataset.data
            
    def list_cached_datasets(self) -> List[Dict[str, Any]]:
        """List all cached datasets."""
        datasets = []
        for filepath in self.data_dir.glob("*.pkl"):
            info_file = filepath.with_suffix('').name + "_info.json"
            info_path = self.data_dir / info_file
            
            dataset_info = {
                'filename': filepath.name,
                'size_mb': filepath.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
            
            if info_path.exists():
                with open(info_path, 'r') as f:
                    dataset_info.update(json.load(f))
                    
            datasets.append(dataset_info)
            
        return datasets
        
    def clear_cache(self, system: Optional[str] = None) -> None:
        """
        Clear cached datasets.
        
        Args:
            system: If specified, only clear datasets for this system
        """
        if system:
            pattern = f"{system}_*.pkl"
        else:
            pattern = "*.pkl"
            
        for filepath in self.data_dir.glob(pattern):
            filepath.unlink()
            # Also remove info file
            info_path = filepath.with_suffix('').name + "_info.json"
            info_file = self.data_dir / info_path
            if info_file.exists():
                info_file.unlink()
                
        logger.info(f"🗑️  Cleared cache for: {system or 'all systems'}")