#!/usr/bin/env python3
"""Main script to train a single system model."""

import os
import sys
import argparse
import torch
import gc

# Add parent directory to path
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model_manager import UniversalModelManager
from core.data_pipeline import UniversalDataGenerator
from training.sft_training import train_sft_model
from training.grpo_training import train_grpo_model
from config import get_config

def main():
    parser = argparse.ArgumentParser(description="Train a single system model.")
    parser.add_argument("--system", type=str, default="double_integrator", help="System to train.")
    parser.add_argument("--sft", action="store_true", help="Run SFT training.")
    parser.add_argument("--grpo", action="store_true", help="Run GRPO training.")
    parser.add_argument("--plot", action="store_true", help="Plot a solution.")
    args = parser.parse_args()

    config = get_config()

    # --- Data Generation ---
    print("--- Generating Data ---")
    data_generator = UniversalDataGenerator(
        systems=[args.system],
        dt=config["system"]["dt"],
        steps=config["system"]["steps"],
        reasoning_start=config["system"]["reasoning_start"],
        reasoning_end=config["system"]["reasoning_end"],
        solution_start=config["system"]["solution_start"],
        solution_end=config["system"]["solution_end"],
    )
    train_data = data_generator.generate_single_system_dataset(args.system, num_samples=500)

    # --- Model Setup (using exact notebook parameters) ---
    print("--- Setting up Model ---")
    model_manager = UniversalModelManager(base_model_name="unsloth/Qwen3-4B-Base")
    model_manager.setup_model(
        max_seq_length=2048,  # Exact notebook parameter
        lora_rank=32,         # Exact notebook parameter
        load_in_4bit=True,
        gpu_memory_utilization=0.7,  # Exact notebook parameter
        fast_inference=True,  # Enable vLLM like notebook
        working_notebook_mode=True  # Use notebook-compatible setup
    )
    model_manager.setup_chat_template(
        reasoning_start=config["system"]["reasoning_start"],
        reasoning_end=config["system"]["reasoning_end"],
        solution_start=config["system"]["solution_start"],
        solution_end=config["system"]["solution_end"],
        system_prompt=data_generator._generate_di_reasoning(train_data[0]["initial_state"], train_data[0]["controls"])
    )

    if args.sft:
        print("--- SFT Training ---")
        train_sft_model(model_manager, train_data, training_config=config["sft"])

        # --- Clear Memory ---
        gc.collect()
        torch.cuda.empty_cache()

    if args.grpo:
        print("--- GRPO Training ---")
        train_grpo_model(model_manager, train_data, training_config=config["grpo"])

        # --- Clear Memory ---
        gc.collect()
        torch.cuda.empty_cache()

    if args.plot:
        print("--- Plotting Solution ---")
        print("Plotting functionality not yet implemented")

if __name__ == "__main__":
    main()
