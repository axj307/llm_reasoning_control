#!/usr/bin/env python3
"""
Model comparison evaluation script.
Compares SFT vs GRPO vs Optimal control on the same test cases.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, AVAILABLE_SYSTEMS
from core.model_manager import UniversalModelManager
from evaluation.inference import run_batch_inference
from evaluation.metrics import compute_batch_metrics
from environments import get_system
from core.solvers import get_solver


class ModelComparator:
    """Compare different models on the same test cases."""
    
    def __init__(self, gpu_id: str = "0"):
        """Initialize model comparator."""
        self.gpu_id = gpu_id
        self.config = get_config()
        
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        self.models = {}
        self.model_manager = UniversalModelManager()
    
    def load_models(self, system_name: str, model_types: List[str] = None):
        """
        Load multiple models for comparison.
        
        Args:
            system_name: Control system name
            model_types: List of model types to load ['sft', 'grpo']
        """
        if model_types is None:
            model_types = ['sft', 'grpo']
        
        print(f"Loading models for {system_name}...")
        
        for model_type in model_types:
            try:
                model, tokenizer, lora_request, metadata = self.model_manager.load_single_system_model(
                    system_name, training_type=model_type
                )
                
                self.models[model_type] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'lora_request': lora_request,
                    'metadata': metadata
                }
                print(f"✅ Loaded {model_type} model")
                
            except Exception as e:
                print(f"❌ Failed to load {model_type} model: {e}")
        
        # Setup chat template (same for all models)
        if self.models:
            self.model_manager.setup_chat_template()
            print("✅ Chat template configured")
    
    def evaluate_all_models(self, system_name: str, test_cases: List[Tuple[float, float]],
                           temperature: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate all loaded models on the same test cases.
        
        Args:
            system_name: Control system name
            test_cases: List of initial conditions
            temperature: Sampling temperature
        
        Returns:
            Dictionary with results for each model type
        """
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=1024
        )
        
        all_results = {}
        
        # Evaluate each model
        for model_type, model_data in self.models.items():
            print(f"Evaluating {model_type} model...")
            
            results = run_batch_inference(
                model_data['model'], model_data['tokenizer'], 
                system_name, test_cases,
                lora_request=model_data['lora_request'],
                sampling_params=sampling_params
            )
            
            metrics = compute_batch_metrics(results)
            
            all_results[model_type] = {
                'results': results,
                'metrics': metrics
            }
            
            print(f"  {model_type} success rate: {metrics.get('success_rate', 0):.1%}")
        
        # Add optimal control baseline
        all_results['optimal'] = self._compute_optimal_baseline(system_name, test_cases)
        
        return all_results
    
    def _compute_optimal_baseline(self, system_name: str, test_cases: List[Tuple]) -> Dict[str, Any]:
        """Compute optimal control baseline for comparison."""
        solver = get_solver(system_name)
        system = get_system(system_name)()
        
        optimal_results = []
        
        for test_case in test_cases:
            # Solve optimal control
            optimal_controls = solver(test_case, self.config['system']['dt'], 
                                    self.config['system']['steps'])
            
            # Simulate optimal trajectory
            states = [test_case]
            current_state = np.array(test_case)
            
            for control in optimal_controls:
                next_state = system.simulate_step(current_state, control, 
                                                self.config['system']['dt'])
                states.append(next_state)
                current_state = next_state
            
            # Create result structure
            result = {
                'initial_state': test_case,
                'valid_format': True,
                'model_trajectory': {
                    'states': states,
                    'controls': optimal_controls
                }
            }
            optimal_results.append(result)
        
        optimal_metrics = compute_batch_metrics(optimal_results)
        
        return {
            'results': optimal_results,
            'metrics': optimal_metrics
        }
    
    def create_comparison_plot(self, system_name: str, comparison_data: Dict,
                             test_case_idx: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed comparison plot for a specific test case.
        
        Args:
            system_name: Control system name
            comparison_data: Results from evaluate_all_models
            test_case_idx: Which test case to visualize
            save_path: Optional save path
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Colors for different models
        colors = {'sft': 'blue', 'grpo': 'green', 'optimal': 'red'}
        linestyles = {'sft': '-', 'grpo': '-', 'optimal': '--'}
        
        # Get test case
        test_case = None
        for model_type, data in comparison_data.items():
            if test_case_idx < len(data['results']):
                test_case = data['results'][test_case_idx]['initial_state']
                break
        
        if test_case is None:
            raise ValueError(f"Test case index {test_case_idx} not found")
        
        # Plot 1: Phase space trajectories
        ax1 = axes[0, 0]
        
        for model_type, data in comparison_data.items():
            if test_case_idx < len(data['results']):
                result = data['results'][test_case_idx]
                if result.get('valid_format') and result.get('model_trajectory'):
                    states = np.array(result['model_trajectory']['states'])
                    ax1.plot(states[:, 0], states[:, 1], 
                           color=colors.get(model_type, 'black'),
                           linestyle=linestyles.get(model_type, '-'),
                           linewidth=2, marker='o', markersize=3,
                           label=model_type.upper(), alpha=0.8)
        
        # Mark important points
        ax1.scatter(test_case[0], test_case[1], c='orange', s=100, 
                   marker='o', label='Start', zorder=5)
        ax1.scatter(0, 0, c='red', s=100, marker='*', label='Target', zorder=5)
        
        # Target region
        circle = plt.Circle((0, 0), 0.1, fill=False, color='red', 
                           linewidth=2, linestyle=':')
        ax1.add_patch(circle)
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Velocity')
        ax1.set_title('Phase Space Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot 2: Position over time
        ax2 = axes[0, 1]
        
        for model_type, data in comparison_data.items():
            if test_case_idx < len(data['results']):
                result = data['results'][test_case_idx]
                if result.get('valid_format') and result.get('model_trajectory'):
                    states = np.array(result['model_trajectory']['states'])
                    times = np.arange(len(states)) * self.config['system']['dt']
                    ax2.plot(times, states[:, 0], 
                           color=colors.get(model_type, 'black'),
                           linestyle=linestyles.get(model_type, '-'),
                           linewidth=2, label=model_type.upper())
        
        ax2.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Target')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position')
        ax2.set_title('Position vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Velocity over time
        ax3 = axes[1, 0]
        
        for model_type, data in comparison_data.items():
            if test_case_idx < len(data['results']):
                result = data['results'][test_case_idx]
                if result.get('valid_format') and result.get('model_trajectory'):
                    states = np.array(result['model_trajectory']['states'])
                    times = np.arange(len(states)) * self.config['system']['dt']
                    ax3.plot(times, states[:, 1], 
                           color=colors.get(model_type, 'black'),
                           linestyle=linestyles.get(model_type, '-'),
                           linewidth=2, label=model_type.upper())
        
        ax3.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Target')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Control inputs
        ax4 = axes[1, 1]
        
        for model_type, data in comparison_data.items():
            if test_case_idx < len(data['results']):
                result = data['results'][test_case_idx]
                if result.get('valid_format') and result.get('model_trajectory'):
                    controls = result['model_trajectory']['controls']
                    times = np.arange(len(controls)) * self.config['system']['dt']
                    ax4.step(times, controls, where='post',
                           color=colors.get(model_type, 'black'),
                           linestyle=linestyles.get(model_type, '-'),
                           linewidth=2, label=model_type.upper())
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Control Input')
        ax4.set_title('Control vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{system_name.title()} Model Comparison\n'
                    f'Initial State: ({test_case[0]:.2f}, {test_case[1]:.2f})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plot saved to: {save_path}")
        
        return fig
    
    def create_performance_comparison(self, system_name: str, comparison_data: Dict,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create performance comparison across all models.
        
        Args:
            system_name: Control system name
            comparison_data: Results from evaluate_all_models
            save_path: Optional save path
        
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        model_types = list(comparison_data.keys())
        colors = {'sft': 'skyblue', 'grpo': 'lightgreen', 'optimal': 'lightcoral'}
        
        # 1. Success Rate Comparison
        success_rates = []
        for model_type in model_types:
            rate = comparison_data[model_type]['metrics'].get('success_rate', 0)
            success_rates.append(rate)
        
        bars1 = ax1.bar(model_types, success_rates, 
                       color=[colors.get(mt, 'gray') for mt in model_types],
                       alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Final Error Comparison
        final_errors_by_model = {}
        for model_type, data in comparison_data.items():
            final_errors = []
            for result in data['results']:
                if result.get('valid_format') and result.get('model_trajectory'):
                    final_state = result['model_trajectory']['states'][-1]
                    final_error = np.linalg.norm(final_state)
                    final_errors.append(final_error)
            final_errors_by_model[model_type] = final_errors
        
        # Box plot of final errors
        box_data = [final_errors_by_model.get(mt, []) for mt in model_types]
        box_colors = [colors.get(mt, 'gray') for mt in model_types]
        
        bp = ax2.boxplot(box_data, labels=model_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.set_ylabel('Final Error')
        ax2.set_title('Final Error Distribution')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Mean Performance Metrics
        metrics_to_compare = ['mean_performance_score', 'mean_final_error']
        metric_data = {metric: [] for metric in metrics_to_compare}
        
        for model_type in model_types:
            metrics = comparison_data[model_type]['metrics']
            for metric in metrics_to_compare:
                metric_data[metric].append(metrics.get(metric, 0))
        
        x = np.arange(len(model_types))
        width = 0.35
        
        bars3_1 = ax3.bar(x - width/2, metric_data['mean_performance_score'], width,
                         label='Performance Score', alpha=0.8, color='lightblue')
        
        ax3_twin = ax3.twinx()
        bars3_2 = ax3_twin.bar(x + width/2, metric_data['mean_final_error'], width,
                              label='Final Error', alpha=0.8, color='lightyellow')
        
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel('Performance Score', color='blue')
        ax3_twin.set_ylabel('Final Error', color='orange')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_types)
        
        # 4. Control Effort Comparison
        control_efforts_by_model = {}
        for model_type, data in comparison_data.items():
            control_efforts = []
            for result in data['results']:
                if result.get('valid_format') and result.get('model_trajectory'):
                    controls = result['model_trajectory']['controls']
                    effort = np.sum(np.abs(controls))
                    control_efforts.append(effort)
            control_efforts_by_model[model_type] = control_efforts
        
        mean_efforts = [np.mean(control_efforts_by_model.get(mt, [0])) for mt in model_types]
        
        bars4 = ax4.bar(model_types, mean_efforts,
                       color=[colors.get(mt, 'gray') for mt in model_types],
                       alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Mean Control Effort')
        ax4.set_title('Control Effort Comparison')
        
        # Add value labels
        for bar, effort in zip(bars4, mean_efforts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{effort:.1f}', ha='center', va='bottom')
        
        plt.suptitle(f'{system_name.title()} Model Performance Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Performance comparison saved to: {save_path}")
        
        return fig


def main():
    parser = argparse.ArgumentParser(description="Compare SFT vs GRPO vs Optimal control models")
    
    # Model configuration
    parser.add_argument("--system", type=str, required=True, choices=AVAILABLE_SYSTEMS,
                       help="Control system to evaluate")
    parser.add_argument("--model-types", type=str, default="sft,grpo",
                       help="Model types to compare (comma-separated)")
    
    # Test configuration
    parser.add_argument("--num-test-cases", type=int, default=20,
                       help="Number of test cases")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed")
    
    # Visualization
    parser.add_argument("--comparison-cases", type=int, default=3,
                       help="Number of detailed comparison cases to show")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="model_comparison_results",
                       help="Output directory")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results as JSON")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots")
    
    # Hardware
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse model types
    model_types = [mt.strip() for mt in args.model_types.split(",")]
    
    # Initialize comparator
    comparator = ModelComparator(args.gpu_id)
    
    # Load models
    comparator.load_models(args.system, model_types)
    
    if not comparator.models:
        print("❌ No models loaded successfully")
        return
    
    # Generate test cases
    system = get_system(args.system)()
    test_cases = []
    for _ in range(args.num_test_cases):
        initial_state = system.generate_random_initial_state()
        test_cases.append(tuple(initial_state))
    
    print(f"Generated {len(test_cases)} test cases for {args.system}")
    
    # Run comparison evaluation
    print(f"\n{'='*70}")
    print(f"COMPARING MODELS ON {args.system.upper()}")
    print('='*70)
    
    comparison_data = comparator.evaluate_all_models(args.system, test_cases, args.temperature)
    
    # Print summary
    print(f"\nCOMPARISON SUMMARY:")
    for model_type, data in comparison_data.items():
        success_rate = data['metrics'].get('success_rate', 0)
        mean_error = data['metrics'].get('mean_final_error', 0)
        print(f"  {model_type.upper()}: {success_rate:.1%} success, {mean_error:.4f} mean error")
    
    # Generate visualizations
    print(f"\nGenerating comparison visualizations...")
    
    # 1. Overall performance comparison
    perf_comparison_path = os.path.join(args.output_dir, f"{args.system}_performance_comparison.png")
    fig1 = comparator.create_performance_comparison(args.system, comparison_data, perf_comparison_path)
    if args.show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # 2. Detailed trajectory comparisons for selected cases
    for i in range(min(args.comparison_cases, len(test_cases))):
        case_comparison_path = os.path.join(args.output_dir, f"{args.system}_comparison_case_{i+1}.png")
        fig2 = comparator.create_comparison_plot(args.system, comparison_data, i, case_comparison_path)
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig2)
    
    # Save results if requested
    if args.save_results:
        results_path = os.path.join(args.output_dir, f"{args.system}_comparison_results.json")
        
        # Prepare serializable data
        serializable_data = {
            'system': args.system,
            'model_types': list(comparison_data.keys()),
            'test_cases': test_cases,
            'metrics_summary': {
                model_type: data['metrics'] 
                for model_type, data in comparison_data.items()
            },
            'evaluation_config': {
                'num_test_cases': args.num_test_cases,
                'temperature': args.temperature,
                'random_seed': args.random_seed
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"✅ Comparison results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON COMPLETE")
    print('='*70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()