#!/usr/bin/env python3
"""Utilities for loading and managing pre-generated datasets."""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def load_dataset(file_path: str, format_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load a dataset from file.
    
    Args:
        file_path: Path to dataset file
        format_type: Format type ('pickle' or 'json'), auto-detected if None
        
    Returns:
        List of dataset samples
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Auto-detect format if not specified
    if format_type is None:
        if file_path.suffix == '.pkl':
            format_type = 'pickle'
        elif file_path.suffix == '.json':
            format_type = 'json'
        else:
            raise ValueError(f"Cannot auto-detect format for file: {file_path}")
    
    print(f"📂 Loading dataset from {file_path} (format: {format_type})")
    
    if format_type == 'pickle':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif format_type == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"   ✅ Loaded {len(data)} samples")
    return data

def load_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """
    Load dataset information file.
    
    Args:
        dataset_path: Path to dataset file (will look for corresponding info file)
        
    Returns:
        Dataset information dictionary
    """
    dataset_path = Path(dataset_path)
    
    # Try to find info file
    info_files = [
        dataset_path.parent / f"{dataset_path.stem}_info.pkl",
        dataset_path.parent / f"{dataset_path.stem}_info.json",
        dataset_path.with_suffix('.info.json'),
        dataset_path.with_suffix('.info.pkl')
    ]
    
    for info_file in info_files:
        if info_file.exists():
            try:
                if info_file.suffix == '.pkl':
                    with open(info_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(info_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load info from {info_file}: {e}")
                continue
    
    print(f"⚠️  No info file found for dataset: {dataset_path}")
    return {}

def filter_dataset_by_system(data: List[Dict[str, Any]], system_name: str) -> List[Dict[str, Any]]:
    """
    Filter dataset to include only samples from a specific system.
    
    Args:
        data: Full dataset
        system_name: System name to filter by
        
    Returns:
        Filtered dataset
    """
    filtered_data = [sample for sample in data if sample.get('system_type') == system_name]
    
    if not filtered_data:
        available_systems = set(sample.get('system_type', 'unknown') for sample in data)
        raise ValueError(f"No data found for system '{system_name}'. Available systems: {available_systems}")
    
    print(f"   🔍 Filtered to {len(filtered_data)} samples for system '{system_name}'")
    return filtered_data

def load_train_eval_datasets(dataset_name: str, 
                           dataset_dir: str = "datasets",
                           system_name: Optional[str] = None) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Load both training and evaluation datasets.
    
    Args:
        dataset_name: Name of the dataset (without _train/_eval suffix)
        dataset_dir: Directory containing datasets
        system_name: Filter by system name if specified
        
    Returns:
        Tuple of (train_data, eval_data, dataset_info)
    """
    dataset_dir = Path(dataset_dir)
    
    # Try to find train and eval files
    train_files = [
        dataset_dir / f"{dataset_name}_train.pkl",
        dataset_dir / f"{dataset_name}_train.json"
    ]
    
    eval_files = [
        dataset_dir / f"{dataset_name}_eval.pkl", 
        dataset_dir / f"{dataset_name}_eval.json"
    ]
    
    # Load training data
    train_data = None
    for train_file in train_files:
        if train_file.exists():
            train_data = load_dataset(train_file)
            break
    
    if train_data is None:
        raise FileNotFoundError(f"No training dataset found for: {dataset_name}")
    
    # Load evaluation data
    eval_data = None
    for eval_file in eval_files:
        if eval_file.exists():
            eval_data = load_dataset(eval_file)
            break
    
    if eval_data is None:
        print(f"⚠️  No evaluation dataset found for: {dataset_name}")
        eval_data = []
    
    # Load dataset info
    info_file = train_files[0] if train_files[0].exists() else train_files[1]
    dataset_info = load_dataset_info(info_file)
    
    # Filter by system if requested
    if system_name:
        train_data = filter_dataset_by_system(train_data, system_name)
        if eval_data:
            eval_data = filter_dataset_by_system(eval_data, system_name)
    
    print(f"📊 Dataset loaded: {len(train_data)} train, {len(eval_data)} eval samples")
    
    return train_data, eval_data, dataset_info

def list_available_datasets(dataset_dir: str = "datasets") -> List[str]:
    """
    List all available datasets in the directory.
    
    Args:
        dataset_dir: Directory to search for datasets
        
    Returns:
        List of dataset names (without _train/_eval suffixes)
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        return []
    
    # Find all train files and extract dataset names
    train_files = list(dataset_dir.glob("*_train.*"))
    dataset_names = []
    
    for train_file in train_files:
        # Remove _train.ext to get dataset name
        dataset_name = train_file.stem.replace("_train", "")
        dataset_names.append(dataset_name)
    
    return sorted(set(dataset_names))

def get_dataset_stats(dataset_path: str) -> Dict[str, Any]:
    """
    Get statistics about a dataset.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        Dictionary with dataset statistics
    """
    data = load_dataset(dataset_path)
    
    # Count samples by system
    system_counts = {}
    for sample in data:
        system = sample.get('system_type', 'unknown')
        system_counts[system] = system_counts.get(system, 0) + 1
    
    # Get sample lengths
    input_lengths = [len(sample.get('input', '')) for sample in data]
    output_lengths = [len(sample.get('output', '')) for sample in data]
    
    stats = {
        'total_samples': len(data),
        'systems': system_counts,
        'input_length': {
            'min': min(input_lengths) if input_lengths else 0,
            'max': max(input_lengths) if input_lengths else 0,
            'avg': sum(input_lengths) / len(input_lengths) if input_lengths else 0
        },
        'output_length': {
            'min': min(output_lengths) if output_lengths else 0,
            'max': max(output_lengths) if output_lengths else 0,
            'avg': sum(output_lengths) / len(output_lengths) if output_lengths else 0
        }
    }
    
    return stats

if __name__ == "__main__":
    # Test the data utilities
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Testing data utilities with: {dataset_path}")
        
        try:
            stats = get_dataset_stats(dataset_path)
            print(f"Dataset stats: {stats}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python data_utils.py <dataset_path>")
        
        # List available datasets
        datasets = list_available_datasets()
        if datasets:
            print(f"Available datasets: {datasets}")
        else:
            print("No datasets found in 'datasets/' directory")