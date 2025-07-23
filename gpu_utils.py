#!/usr/bin/env python3
"""GPU utility functions for automatic GPU selection."""

import os
import random
import subprocess
import torch
from typing import Optional, List

def get_gpu_memory_info() -> List[dict]:
    """Get memory information for all GPUs."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_info.append({
                    'index': int(parts[0]),
                    'memory_used': int(parts[1]),
                    'memory_total': int(parts[2]),
                    'memory_free': int(parts[2]) - int(parts[1]),
                    'utilization': int(parts[1]) / int(parts[2])
                })
        return gpu_info
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to torch if nvidia-smi is not available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                gpu_info.append({
                    'index': i,
                    'memory_used': allocated // (1024**2),  # Convert to MB
                    'memory_total': total // (1024**2),     # Convert to MB  
                    'memory_free': (total - allocated) // (1024**2),
                    'utilization': allocated / total
                })
            return gpu_info
        return []

def find_best_gpu(min_memory_gb: float = 4.0, prefer_empty: bool = True) -> Optional[int]:
    """
    Find the best available GPU.
    
    Args:
        min_memory_gb: Minimum free memory required in GB
        prefer_empty: Prefer GPUs with lowest utilization
        
    Returns:
        GPU index or None if no suitable GPU found
    """
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        print("⚠️  No GPU information available")
        return None
    
    min_memory_mb = min_memory_gb * 1024
    
    # Filter GPUs with sufficient memory
    suitable_gpus = [gpu for gpu in gpu_info if gpu['memory_free'] >= min_memory_mb]
    
    if not suitable_gpus:
        print(f"⚠️  No GPU with {min_memory_gb}GB+ free memory found")
        print("Available GPUs:")
        for gpu in gpu_info:
            print(f"   GPU {gpu['index']}: {gpu['memory_free']:.0f}MB free ({gpu['utilization']:.1%} used)")
        return None
    
    if prefer_empty:
        # Sort by utilization (lowest first)
        suitable_gpus.sort(key=lambda x: x['utilization'])
    else:
        # Random selection from suitable GPUs
        random.shuffle(suitable_gpus)
    
    best_gpu = suitable_gpus[0]
    print(f"🎯 Selected GPU {best_gpu['index']}: {best_gpu['memory_free']:.0f}MB free ({best_gpu['utilization']:.1%} used)")
    
    return best_gpu['index']

def set_gpu_device(gpu_id: Optional[int] = None, auto_select: bool = True) -> int:
    """
    Set the GPU device for PyTorch and environment variables.
    
    Args:
        gpu_id: Specific GPU ID to use, or None for auto-selection
        auto_select: Whether to auto-select if gpu_id is None
        
    Returns:
        The GPU ID that was set
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return -1
    
    if gpu_id is None and auto_select:
        # Auto-select best GPU
        gpu_id = find_best_gpu(min_memory_gb=4.0)
        if gpu_id is None:
            # Fallback to random selection
            gpu_id = random.randint(0, torch.cuda.device_count() - 1)
            print(f"🎲 Randomly selected GPU {gpu_id} as fallback")
    elif gpu_id is None:
        # Default to GPU 0 if no auto-selection
        gpu_id = 0
        print(f"📌 Using default GPU {gpu_id}")
    
    # Set environment variable for other libraries
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Set PyTorch device
    try:
        torch.cuda.set_device(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"🚀 Using GPU {gpu_id}: {gpu_name}")
    except Exception as e:
        print(f"🚀 Using GPU {gpu_id} (name lookup failed: {e})")
    
    return gpu_id

def print_gpu_status():
    """Print status of all GPUs."""
    print("🖥️  GPU Status:")
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        print("   No GPU information available")
        return
    
    for gpu in gpu_info:
        status = "🔴 BUSY" if gpu['utilization'] > 0.1 else "🟢 FREE"
        print(f"   GPU {gpu['index']}: {gpu['memory_free']:>6.0f}MB free / {gpu['memory_total']:>6.0f}MB total {status}")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared GPU memory cache")

def auto_gpu_config() -> dict:
    """
    Automatically configure GPU settings for training.
    
    Returns:
        Dictionary with GPU configuration
    """
    print_gpu_status()
    
    # Clear any existing cache
    clear_gpu_memory()
    
    # Auto-select best GPU
    gpu_id = set_gpu_device(auto_select=True)
    
    # Get GPU name safely
    gpu_name = 'CPU'
    if gpu_id >= 0:
        try:
            gpu_name = torch.cuda.get_device_name(gpu_id)
        except:
            gpu_name = f'GPU_{gpu_id}'
    
    return {
        'gpu_id': gpu_id,
        'device': f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': gpu_name
    }

if __name__ == "__main__":
    # Test the GPU utilities
    print("🔧 Testing GPU utilities...")
    config = auto_gpu_config()
    print(f"Final config: {config}")