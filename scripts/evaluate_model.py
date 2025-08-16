#!/usr/bin/env python3
"""Evaluation script for trained models."""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ALL_CONFIG, AVAILABLE_SYSTEMS
from core.model_manager import UniversalModelManager
from evaluation.inference import run_batch_inference, run_mpc_inference
from evaluation.metrics import compute_batch_metrics
from evaluation.visualization import (
    plot_comparison, plot_metrics_comparison, plot_publication_comparison,
    plot_model_only_trajectories, generate_publication_plots, plot_control_comparison
)
# Import simplified plotting function from evaluation module
from evaluation.visualization import plot_all_trajectories_comparison
from environments import get_system
from data_utils import load_dataset, filter_dataset_by_system, load_train_eval_datasets
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained control model")
    
    # Model selection
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--model-type", choices=["universal", "single_system"], required=True,
                       help="Type of model")
    
    # System selection
    parser.add_argument("--systems", type=str,
                       help="Systems to evaluate on (comma-separated). If not specified, uses model's trained systems")
    
    # Test configuration
    parser.add_argument("--num-test-cases", type=int, default=ALL_CONFIG["eval"]["num_test_cases"],
                       help="Number of test cases per system")
    parser.add_argument("--test-type", choices=["standard", "mpc", "both"], default="standard",
                       help="Type of evaluation")
    parser.add_argument("--mpc-horizon", type=int, default=ALL_CONFIG["eval"]["mpc_horizon"],
                       help="MPC prediction horizon")
    
    # Evaluation data configuration
    parser.add_argument("--eval-dataset", type=str,
                       help="Name of pre-generated evaluation dataset")
    parser.add_argument("--eval-data-file", type=str,
                       help="Direct path to evaluation dataset file")
    parser.add_argument("--dataset-dir", type=str, default="datasets",
                       help="Directory containing datasets")
    
    # Sampling configuration
    parser.add_argument("--temperature", type=float, default=ALL_CONFIG["eval"]["sampling_params"]["temperature"],
                       help="Sampling temperature")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for test case generation")
    
    # Output configuration
    parser.add_argument("--save-plots", action="store_true", default=ALL_CONFIG["eval"]["plot_config"]["save_plots"],
                       help="Save generated plots")
    parser.add_argument("--plot-dir", type=str, default=ALL_CONFIG["eval"]["plot_config"]["plot_directory"],
                       help="Directory to save plots")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots (requires display)")
    parser.add_argument("--enhanced-plots", action="store_true",
                       help="Generate enhanced 4-subplot comparison plots")
    
    # Hardware configuration
    parser.add_argument("--gpu-id", type=str, default="0",
                       help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Using GPU: {args.gpu_id}")
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create model manager and load model
    print("Loading model...")
    manager = UniversalModelManager()
    
    if args.model_type == "universal":
        model, tokenizer, lora_request, metadata = manager.load_universal_model()
        trained_systems = metadata.get("trained_systems", AVAILABLE_SYSTEMS)
    else:
        # For single system, parse system name and training type from path
        # Assuming path format: models/single_system/{system}/{training_type}/latest
        path_parts = Path(args.model_path).parts
        if len(path_parts) >= 3 and path_parts[-3] in AVAILABLE_SYSTEMS:
            system_name = path_parts[-3]
            training_type = path_parts[-2]  # sft or grpo
        else:
            raise ValueError(f"Could not determine system from path: {args.model_path}")
        
        model, tokenizer, lora_request, metadata = manager.load_single_system_model(system_name, training_type=training_type)
        trained_systems = [system_name]
    
    print(f"Loaded model trained on: {', '.join(trained_systems)}")
    
    # Setup chat template for inference
    manager.setup_chat_template()
    print("✅ Chat template configured")
    
    # Determine systems to evaluate
    if args.systems:
        eval_systems = [s.strip() for s in args.systems.split(",")]
        # Check if all requested systems are available
        for system in eval_systems:
            if system not in AVAILABLE_SYSTEMS:
                raise ValueError(f"Unknown system: {system}. Available: {AVAILABLE_SYSTEMS}")
    else:
        eval_systems = trained_systems
    
    print(f"Evaluating on systems: {', '.join(eval_systems)}")
    
    # Run evaluation for each system
    all_results = {}
    all_metrics = {}
    
    for system_name in eval_systems:
        print(f"\n{'='*70}")
        print(f"EVALUATING ON {system_name.upper()}")
        print('='*70)
        
        # Generate test cases for this system
        system = get_system(system_name)()
        test_cases = []
        
        for _ in range(args.num_test_cases):
            initial_state = system.generate_random_initial_state()
            test_cases.append(tuple(initial_state))
        
        print(f"Generated {len(test_cases)} test cases")
        
        # Standard evaluation
        if args.test_type in ["standard", "both"]:
            print("Running standard inference...")
            
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_k=ALL_CONFIG["eval"]["sampling_params"]["top_k"],
                max_tokens=ALL_CONFIG["eval"]["sampling_params"]["max_tokens"]
            )
            
            standard_results = run_batch_inference(
                model, tokenizer, system_name, test_cases,
                lora_request=lora_request,
                sampling_params=sampling_params
            )
            
            # Compute metrics
            standard_metrics = compute_batch_metrics(standard_results)
            
            all_results[f"{system_name}_standard"] = standard_results
            all_metrics[f"{system_name}_standard"] = standard_metrics
            
            print(f"Standard evaluation results:")
            print(f"  Success rate: {standard_metrics.get('success_rate', 0.0):.2%}")
            print(f"  Mean performance: {standard_metrics.get('mean_performance_score', 0.0):.4f}")
            print(f"  Mean final error: {standard_metrics.get('mean_final_error', 'N/A')}")
        
        # MPC evaluation
        if args.test_type in ["mpc", "both"]:
            print(f"Running MPC inference (horizon={args.mpc_horizon})...")
            
            mpc_results = []
            for i, test_case in enumerate(test_cases):
                print(f"  MPC test {i+1}/{len(test_cases)}")
                
                mpc_result = run_mpc_inference(
                    model, tokenizer, system_name, test_case,
                    dt=ALL_CONFIG["system"]["dt"],
                    total_steps=ALL_CONFIG["system"]["steps"],
                    horizon=args.mpc_horizon,
                    lora_request=lora_request
                )
                mpc_results.append(mpc_result)
            
            all_results[f"{system_name}_mpc"] = mpc_results
            
            # Calculate MPC metrics
            mpc_final_errors = [r["mpc_trajectory"]["final_error"] for r in mpc_results]
            mpc_metrics = {
                "mean_final_error": np.mean(mpc_final_errors),
                "std_final_error": np.std(mpc_final_errors),
                "median_final_error": np.median(mpc_final_errors),
            }
            all_metrics[f"{system_name}_mpc"] = mpc_metrics
            
            print(f"MPC evaluation results:")
            print(f"  Mean final error: {mpc_metrics.get('mean_final_error', 0.0):.6f}")
            print(f"  Std final error: {mpc_metrics.get('std_final_error', 0.0):.6f}")
    
    # Generate plots
    if args.save_plots or args.show_plots or args.enhanced_plots:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print('='*70)
        
        os.makedirs(args.plot_dir, exist_ok=True)
        
        for key, results in all_results.items():
            if "standard" in key:
                system_name = key.replace('_standard', '')
                
                # Generate enhanced plots if requested (new main plotting option)
                if args.enhanced_plots:
                    print(f"Generating enhanced plots for {system_name}...")
                    
                    # Process results to create trajectory comparison data
                    for i, result in enumerate(results[:3]):  # Limit to first 3 for clarity
                        if result.get('valid_format', False):
                            trajectories = {}
                            
                            # Add model trajectory
                            if result.get('model_trajectory'):
                                trajectories['Model'] = result['model_trajectory']
                            
                            # Add optimal trajectory for comparison if available
                            if result.get('optimal_trajectory'):
                                trajectories['Optimal'] = result['optimal_trajectory']
                            
                            initial_state = result.get('initial_state', (0, 0))
                            
                            if trajectories:
                                # Generate enhanced 4-subplot comparison
                                save_path = os.path.join(args.plot_dir, f"{system_name}_enhanced_case_{i+1}")
                                
                                # Determine if this is a comparison or model-only plot
                                show_comparison = len(trajectories) > 1
                                
                                fig_enhanced = plot_control_comparison(
                                    trajectories, system_name, initial_state,
                                    save_path=save_path, show_comparison=show_comparison
                                )
                                
                                if args.show_plots:
                                    plt.show()
                                else:
                                    plt.close(fig_enhanced)
                                
                                print(f"Saved enhanced comparison for case {i+1}")
                
                # Skip all other plotting modes
                
                # Generate single comparison figure with all trajectories
                if args.save_plots or args.show_plots:
                    # Use simplified plotting function
                    save_path = f"{args.plot_dir}/{key}_all_trajectories.png" if args.save_plots else None
                    fig = plot_all_trajectories_comparison(results, system_name, save_path)
                    
                    if fig:
                        if args.show_plots:
                            plt.show()
                        else:
                            plt.close(fig)
                    
                    # Skip metrics comparison plot
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print('='*70)
    
    for key, metrics in all_metrics.items():
        print(f"\n{key.upper()}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.6f}")
            else:
                print(f"  {metric_name}: {value}")
    
    print(f"\nEvaluation completed for model: {args.model_path}")
    if args.save_plots:
        print(f"Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()