#!/usr/bin/env python3
"""
Comprehensive model evaluation script with enhanced visualization.
Tests models on diverse initial conditions and analyzes success rates.
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


class ComprehensiveEvaluator:
    """Comprehensive evaluator for trained control models."""
    
    def __init__(self, model_path: str, model_type: str, gpu_id: str = "0"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            model_type: 'single_system' or 'universal'
            gpu_id: GPU ID to use
        """
        self.model_path = model_path
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.config = get_config()
        
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        print(f"Loading {self.model_type} model from {self.model_path}...")
        
        self.manager = UniversalModelManager()
        
        if self.model_type == "universal":
            self.model, self.tokenizer, self.lora_request, self.metadata = self.manager.load_universal_model()
            self.trained_systems = self.metadata.get("trained_systems", AVAILABLE_SYSTEMS)
        else:
            # Parse system name from path
            path_parts = Path(self.model_path).parts
            if len(path_parts) >= 3 and path_parts[-3] in AVAILABLE_SYSTEMS:
                system_name = path_parts[-3]
                training_type = path_parts[-2]
            else:
                raise ValueError(f"Could not determine system from path: {self.model_path}")
            
            self.model, self.tokenizer, self.lora_request, self.metadata = self.manager.load_single_system_model(
                system_name, training_type=training_type
            )
            self.trained_systems = [system_name]
        
        # Setup chat template
        self.manager.setup_chat_template()
        print(f"✅ Model loaded, trained on: {', '.join(self.trained_systems)}")
    
    def generate_test_grid(self, system_name: str, grid_size: int = 5) -> List[Tuple[float, float]]:
        """
        Generate a systematic grid of initial conditions for testing.
        
        Args:
            system_name: Name of the control system
            grid_size: Size of the grid (grid_size x grid_size points)
        
        Returns:
            List of (position, velocity) initial conditions
        """
        system = get_system(system_name)()
        bounds = system.get_bounds()
        
        # Get state bounds
        pos_min, pos_max = bounds['state_bounds'][0]
        vel_min, vel_max = bounds['state_bounds'][1]
        
        # Create systematic grid
        positions = np.linspace(pos_min * 0.8, pos_max * 0.8, grid_size)
        velocities = np.linspace(vel_min * 0.8, vel_max * 0.8, grid_size)
        
        test_cases = []
        for pos in positions:
            for vel in velocities:
                test_cases.append((float(pos), float(vel)))
        
        return test_cases
    
    def generate_random_test_cases(self, system_name: str, num_cases: int = 50) -> List[Tuple[float, float]]:
        """
        Generate random initial conditions for testing.
        
        Args:
            system_name: Name of the control system
            num_cases: Number of random test cases
        
        Returns:
            List of (position, velocity) initial conditions
        """
        system = get_system(system_name)()
        test_cases = []
        
        for _ in range(num_cases):
            initial_state = system.generate_random_initial_state()
            test_cases.append(tuple(initial_state))
        
        return test_cases
    
    def evaluate_system(self, system_name: str, test_cases: List[Tuple[float, float]],
                       temperature: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate model on a specific system with given test cases.
        
        Args:
            system_name: Name of the control system
            test_cases: List of initial conditions to test
            temperature: Sampling temperature
        
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating on {system_name} with {len(test_cases)} test cases...")
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=1024
        )
        
        # Run inference
        results = run_batch_inference(
            self.model, self.tokenizer, system_name, test_cases,
            lora_request=self.lora_request,
            sampling_params=sampling_params
        )
        
        # Compute metrics
        metrics = compute_batch_metrics(results)
        
        # Analyze success patterns
        success_analysis = self._analyze_success_patterns(results, test_cases, system_name)
        
        return {
            'results': results,
            'metrics': metrics,
            'success_analysis': success_analysis,
            'test_cases': test_cases
        }
    
    def _analyze_success_patterns(self, results: List[Dict], test_cases: List[Tuple], 
                                 system_name: str) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed trajectories."""
        successful_cases = []
        failed_cases = []
        
        for i, (result, test_case) in enumerate(zip(results, test_cases)):
            if result.get('valid_format', False) and result.get('model_trajectory'):
                # Check if trajectory reaches target region
                final_state = result['model_trajectory']['states'][-1]
                final_error = np.linalg.norm(final_state)
                
                if final_error < 0.1:  # Success threshold
                    successful_cases.append({
                        'initial_state': test_case,
                        'final_error': final_error,
                        'result': result
                    })
                else:
                    failed_cases.append({
                        'initial_state': test_case,
                        'final_error': final_error,
                        'result': result
                    })
        
        return {
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'success_rate': len(successful_cases) / len(results),
            'total_cases': len(results),
            'successful_count': len(successful_cases),
            'failed_count': len(failed_cases)
        }
    
    def create_success_heatmap(self, system_name: str, evaluation_data: Dict, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing success rates across initial conditions.
        
        Args:
            system_name: Name of the control system
            evaluation_data: Results from evaluate_system
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        test_cases = evaluation_data['test_cases']
        results = evaluation_data['results']
        
        # Extract positions and velocities
        positions = [tc[0] for tc in test_cases]
        velocities = [tc[1] for tc in test_cases]
        
        # Determine success for each case
        successes = []
        for result in results:
            if result.get('valid_format', False) and result.get('model_trajectory'):
                final_state = result['model_trajectory']['states'][-1]
                final_error = np.linalg.norm(final_state)
                successes.append(1 if final_error < 0.1 else 0)
            else:
                successes.append(0)
        
        # Create grid for heatmap
        if len(set(positions)) > 1 and len(set(velocities)) > 1:
            # Grid data - reshape for heatmap
            unique_pos = sorted(set(positions))
            unique_vel = sorted(set(velocities))
            
            success_grid = np.zeros((len(unique_vel), len(unique_pos)))
            
            for i, (pos, vel, success) in enumerate(zip(positions, velocities, successes)):
                pos_idx = unique_pos.index(pos)
                vel_idx = unique_vel.index(vel)
                success_grid[vel_idx, pos_idx] = success
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(success_grid, cmap='RdYlGn', aspect='auto', origin='lower')
            
            # Set labels
            ax.set_xticks(range(len(unique_pos)))
            ax.set_yticks(range(len(unique_vel)))
            ax.set_xticklabels([f'{p:.2f}' for p in unique_pos])
            ax.set_yticklabels([f'{v:.2f}' for v in unique_vel])
            
            ax.set_xlabel('Initial Position')
            ax.set_ylabel('Initial Velocity')
            ax.set_title(f'{system_name.title()} Model Success Rate Heatmap\n'
                        f'Success Rate: {evaluation_data["success_analysis"]["success_rate"]:.1%}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Success (1) / Failure (0)')
            
        else:
            # Scatter plot for non-grid data
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = ['red' if s == 0 else 'green' for s in successes]
            ax.scatter(positions, velocities, c=colors, alpha=0.7, s=50)
            
            ax.set_xlabel('Initial Position')
            ax.set_ylabel('Initial Velocity')
            ax.set_title(f'{system_name.title()} Model Success/Failure Map\n'
                        f'Success Rate: {evaluation_data["success_analysis"]["success_rate"]:.1%}')
            
            # Add legend
            ax.scatter([], [], c='green', label='Success', s=50)
            ax.scatter([], [], c='red', label='Failure', s=50)
            ax.legend()
        
        # Add target region circle
        circle = plt.Circle((0, 0), 0.1, fill=False, color='blue', linewidth=2, 
                           linestyle='--', label='Target Region')
        ax.add_patch(circle)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Success heatmap saved to: {save_path}")
        
        return fig
    
    def create_trajectory_comparison_grid(self, system_name: str, evaluation_data: Dict,
                                        max_trajectories: int = 9, 
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a grid of trajectory comparisons (model vs optimal).
        
        Args:
            system_name: Name of the control system
            evaluation_data: Results from evaluate_system
            max_trajectories: Maximum number of trajectories to show
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        results = evaluation_data['results']
        
        # Filter valid results
        valid_results = [r for r in results if r.get('valid_format', False) and r.get('model_trajectory')]
        valid_results = valid_results[:max_trajectories]
        
        # Calculate grid dimensions
        n_plots = len(valid_results)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Get solver for optimal trajectories
        solver = get_solver(system_name)
        
        for i, result in enumerate(valid_results):
            ax = axes[i]
            
            # Plot model trajectory
            model_traj = result['model_trajectory']
            model_states = np.array(model_traj['states'])
            
            ax.plot(model_states[:, 0], model_states[:, 1], 'b-', linewidth=2, 
                   marker='o', markersize=3, label='Model', alpha=0.8)
            
            # Plot optimal trajectory for comparison
            initial_state = result['initial_state']
            optimal_controls = solver(initial_state, self.config['system']['dt'], 
                                    self.config['system']['steps'])
            
            # Simulate optimal trajectory
            system = get_system(system_name)()
            optimal_states = [initial_state]
            current_state = np.array(initial_state)
            
            for control in optimal_controls:
                next_state = system.simulate_step(current_state, control, self.config['system']['dt'])
                optimal_states.append(next_state)
                current_state = next_state
            
            optimal_states = np.array(optimal_states)
            ax.plot(optimal_states[:, 0], optimal_states[:, 1], 'r--', linewidth=2,
                   marker='s', markersize=3, label='Optimal', alpha=0.8)
            
            # Mark start and end points
            ax.scatter(initial_state[0], initial_state[1], c='green', s=100, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(0, 0, c='orange', s=100, marker='*', label='Target', zorder=5)
            
            # Add target region
            circle = plt.Circle((0, 0), 0.1, fill=False, color='orange', 
                               linewidth=2, linestyle=':')
            ax.add_patch(circle)
            
            # Calculate final error
            final_error = np.linalg.norm(model_states[-1])
            success = "✅" if final_error < 0.1 else "❌"
            
            ax.set_title(f'Case {i+1} {success}\nInit: ({initial_state[0]:.2f}, {initial_state[1]:.2f})\n'
                        f'Final Error: {final_error:.3f}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Velocity')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{system_name.title()} Model vs Optimal Trajectories\n'
                    f'Success Rate: {evaluation_data["success_analysis"]["success_rate"]:.1%}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Trajectory comparison grid saved to: {save_path}")
        
        return fig
    
    def create_performance_summary(self, system_name: str, evaluation_data: Dict,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive performance summary plot.
        
        Args:
            system_name: Name of the control system
            evaluation_data: Results from evaluate_system
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        results = evaluation_data['results']
        metrics = evaluation_data['metrics']
        success_analysis = evaluation_data['success_analysis']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Final Error Distribution
        final_errors = []
        for result in results:
            if result.get('valid_format', False) and result.get('model_trajectory'):
                final_state = result['model_trajectory']['states'][-1]
                final_error = np.linalg.norm(final_state)
                final_errors.append(final_error)
        
        ax1.hist(final_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Success Threshold')
        ax1.set_xlabel('Final Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Final Errors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Success Rate by Initial Distance
        initial_distances = []
        case_successes = []
        
        for result in results:
            if result.get('valid_format', False):
                initial_state = result['initial_state']
                initial_dist = np.linalg.norm(initial_state)
                initial_distances.append(initial_dist)
                
                if result.get('model_trajectory'):
                    final_state = result['model_trajectory']['states'][-1]
                    final_error = np.linalg.norm(final_state)
                    case_successes.append(1 if final_error < 0.1 else 0)
                else:
                    case_successes.append(0)
        
        # Bin by initial distance and calculate success rate
        if initial_distances:
            bins = np.linspace(0, max(initial_distances), 10)
            bin_indices = np.digitize(initial_distances, bins)
            bin_success_rates = []
            bin_centers = []
            
            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    success_rate = np.mean([case_successes[j] for j in range(len(case_successes)) if mask[j]])
                    bin_success_rates.append(success_rate)
                    bin_centers.append((bins[i-1] + bins[i]) / 2)
            
            ax2.bar(bin_centers, bin_success_rates, width=np.diff(bins)[0]*0.8, 
                   alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Initial Distance from Target')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate vs Initial Distance')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        
        # 3. Control Effort Analysis
        control_efforts = []
        for result in results:
            if result.get('valid_format', False) and result.get('model_trajectory'):
                controls = result['model_trajectory']['controls']
                control_effort = np.sum(np.abs(controls))
                control_efforts.append(control_effort)
        
        if control_efforts:
            ax3.hist(control_efforts, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_xlabel('Total Control Effort')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Control Effort')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary Statistics
        ax4.axis('off')
        
        # Calculate additional metrics
        mean_final_error = np.mean(final_errors) if final_errors else 0
        std_final_error = np.std(final_errors) if final_errors else 0
        mean_control_effort = np.mean(control_efforts) if control_efforts else 0
        
        summary_text = f"""
PERFORMANCE SUMMARY
{system_name.title()} System

Total Test Cases: {len(results)}
Valid Predictions: {len(final_errors)}
Success Rate: {success_analysis['success_rate']:.1%}

FINAL ERROR STATISTICS:
Mean: {mean_final_error:.4f}
Std: {std_final_error:.4f}
Median: {np.median(final_errors) if final_errors else 0:.4f}

CONTROL EFFORT:
Mean: {mean_control_effort:.2f}
Max: {max(control_efforts) if control_efforts else 0:.2f}

SUCCESS THRESHOLD: 0.1
"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{system_name.title()} Model Comprehensive Performance Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Performance summary saved to: {save_path}")
        
        return fig


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation with enhanced visualizations")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model-type", choices=["single_system", "universal"], required=True, help="Type of model")
    parser.add_argument("--systems", type=str, help="Systems to evaluate (comma-separated)")
    
    # Test configuration
    parser.add_argument("--test-type", choices=["grid", "random", "both"], default="both",
                       help="Type of test cases to generate")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size for systematic testing")
    parser.add_argument("--num-random", type=int, default=50, help="Number of random test cases")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--save-results", action="store_true", help="Save results as JSON")
    parser.add_argument("--show-plots", action="store_true", help="Display plots")
    
    # Hardware
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(args.model_path, args.model_type, args.gpu_id)
    
    # Determine systems to evaluate
    if args.systems:
        eval_systems = [s.strip() for s in args.systems.split(",")]
    else:
        eval_systems = evaluator.trained_systems
    
    print(f"Will evaluate on systems: {', '.join(eval_systems)}")
    
    # Run comprehensive evaluation
    all_evaluation_data = {}
    
    for system_name in eval_systems:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE EVALUATION: {system_name.upper()}")
        print('='*70)
        
        # Generate test cases
        test_cases = []
        
        if args.test_type in ["grid", "both"]:
            grid_cases = evaluator.generate_test_grid(system_name, args.grid_size)
            test_cases.extend(grid_cases)
            print(f"Generated {len(grid_cases)} grid test cases")
        
        if args.test_type in ["random", "both"]:
            random_cases = evaluator.generate_random_test_cases(system_name, args.num_random)
            test_cases.extend(random_cases)
            print(f"Generated {len(random_cases)} random test cases")
        
        print(f"Total test cases: {len(test_cases)}")
        
        # Run evaluation
        evaluation_data = evaluator.evaluate_system(system_name, test_cases, args.temperature)
        all_evaluation_data[system_name] = evaluation_data
        
        # Print summary
        success_analysis = evaluation_data['success_analysis']
        print(f"\nSUCCESS ANALYSIS:")
        print(f"  Total cases: {success_analysis['total_cases']}")
        print(f"  Successful: {success_analysis['successful_count']}")
        print(f"  Failed: {success_analysis['failed_count']}")
        print(f"  Success rate: {success_analysis['success_rate']:.1%}")
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        
        # 1. Success heatmap/scatter plot
        heatmap_path = os.path.join(args.output_dir, f"{system_name}_success_heatmap.png")
        fig1 = evaluator.create_success_heatmap(system_name, evaluation_data, heatmap_path)
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig1)
        
        # 2. Trajectory comparison grid
        trajectories_path = os.path.join(args.output_dir, f"{system_name}_trajectory_comparison.png")
        fig2 = evaluator.create_trajectory_comparison_grid(system_name, evaluation_data, 
                                                         max_trajectories=9, save_path=trajectories_path)
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig2)
        
        # 3. Performance summary
        summary_path = os.path.join(args.output_dir, f"{system_name}_performance_summary.png")
        fig3 = evaluator.create_performance_summary(system_name, evaluation_data, summary_path)
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig3)
        
        # Save results if requested
        if args.save_results:
            results_path = os.path.join(args.output_dir, f"{system_name}_evaluation_results.json")
            
            # Prepare data for JSON serialization
            serializable_data = {
                'metrics': evaluation_data['metrics'],
                'success_analysis': success_analysis,
                'test_cases': evaluation_data['test_cases'],
                'model_path': args.model_path,
                'model_type': args.model_type,
                'evaluation_config': {
                    'test_type': args.test_type,
                    'grid_size': args.grid_size,
                    'num_random': args.num_random,
                    'temperature': args.temperature
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"✅ Results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print('='*70)
    print(f"Results saved to: {args.output_dir}")
    
    # Print final summary
    for system_name, data in all_evaluation_data.items():
        success_rate = data['success_analysis']['success_rate']
        total_cases = data['success_analysis']['total_cases']
        print(f"{system_name}: {success_rate:.1%} success rate ({total_cases} cases)")


if __name__ == "__main__":
    main()