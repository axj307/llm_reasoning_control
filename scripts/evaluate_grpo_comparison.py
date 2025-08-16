#!/usr/bin/env python3
"""
Compare SFT vs GRPO model performance on double integrator control.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔬 SFT vs GRPO MODEL COMPARISON")
    print("=" * 50)
    
    # Test cases for comparison
    test_cases = [
        (0.5, -0.3, "Positive position, negative velocity"),
        (-0.4, 0.6, "Negative position, positive velocity"), 
        (0.8, 0.2, "High positive position"),
        (-0.6, -0.4, "Both negative"),
        (0.3, 0.7, "High positive velocity"),
    ]
    
    models_to_test = [
        {
            "name": "SFT Model (Working)",
            "path": "models/working_notebook/sft_model",
            "type": "single_system"
        },
        {
            "name": "GRPO Model (New)", 
            "path": "models/working_notebook/grpo_working_params_model",
            "type": "single_system"
        }
    ]
    
    results = {}
    
    # Test each model
    for model_info in models_to_test:
        print(f"\n📊 Testing: {model_info['name']}")
        print("-" * 40)
        
        try:
            # Run evaluation
            from subprocess import run, PIPE
            
            # Create temporary test dataset
            import pickle
            temp_test_data = []
            for i, (x0, v0, desc) in enumerate(test_cases):
                temp_test_data.append({
                    "initial_state": [x0, v0], 
                    "system_type": "double_integrator",
                    "description": desc
                })
            
            # Save temporary test file
            test_file = f"datasets/temp_comparison_test.pkl"
            with open(test_file, "wb") as f:
                pickle.dump(temp_test_data, f)
            
            # Run evaluation script
            cmd = [
                "python", "scripts/evaluate_model.py",
                "--model-path", model_info["path"],
                "--model-type", model_info["type"],
                "--systems", "double_integrator",
                "--eval-data-file", test_file,
                "--num-test-cases", "5",
                "--save-plots"
            ]
            
            result = run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Parse results from output
                output = result.stdout
                
                # Extract success rate
                import re
                success_match = re.search(r"Success rate: ([\d\.]+)%", output)
                performance_match = re.search(r"Mean performance: ([\d\.]+)", output)
                error_match = re.search(r"Mean final error: ([\d\.]+)", output)
                
                results[model_info["name"]] = {
                    "success_rate": float(success_match.group(1)) if success_match else 0.0,
                    "mean_performance": float(performance_match.group(1)) if performance_match else 0.0,
                    "mean_final_error": float(error_match.group(1)) if error_match else float('inf'),
                    "status": "success"
                }
                
                print(f"✅ Success rate: {results[model_info['name']]['success_rate']:.1f}%")
                print(f"✅ Mean performance: {results[model_info['name']]['mean_performance']:.3f}")
                print(f"✅ Mean final error: {results[model_info['name']]['mean_final_error']:.3f}")
                
            else:
                print(f"❌ Evaluation failed: {result.stderr}")
                results[model_info["name"]] = {
                    "status": "failed",
                    "error": result.stderr
                }
        
        except Exception as e:
            print(f"❌ Error testing {model_info['name']}: {e}")
            results[model_info["name"]] = {
                "status": "error", 
                "error": str(e)
            }
    
    # Print comparison
    print("\n🏆 COMPARISON RESULTS")
    print("=" * 50)
    
    successful_models = {name: data for name, data in results.items() if data.get("status") == "success"}
    
    if len(successful_models) >= 2:
        print("📊 Performance Comparison:")
        print(f"{'Model':<25} {'Success Rate':<12} {'Performance':<12} {'Final Error':<12}")
        print("-" * 65)
        
        for name, data in successful_models.items():
            print(f"{name:<25} {data['success_rate']:>8.1f}%    {data['mean_performance']:>8.3f}    {data['mean_final_error']:>8.3f}")
        
        # Determine winner
        best_model = max(successful_models.items(), key=lambda x: x[1]['success_rate'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Success Rate: {best_model[1]['success_rate']:.1f}%")
        
        # Analysis
        print(f"\n📈 Analysis:")
        sft_data = successful_models.get("SFT Model (Working)")
        grpo_data = successful_models.get("GRPO Model (New)")
        
        if sft_data and grpo_data:
            success_diff = grpo_data['success_rate'] - sft_data['success_rate'] 
            error_diff = sft_data['mean_final_error'] - grpo_data['mean_final_error']
            
            print(f"   GRPO vs SFT Success Rate: {success_diff:+.1f}%")
            print(f"   GRPO vs SFT Error Reduction: {error_diff:+.3f}")
            
            if success_diff > 10:
                print("   🎉 GRPO shows significant improvement!")
            elif success_diff > 0:
                print("   📈 GRPO shows modest improvement")
            elif success_diff < -10:
                print("   📉 GRPO underperforms SFT significantly")
            else:
                print("   ⚖️  Models perform similarly")
    
    else:
        print("❌ Insufficient successful evaluations for comparison")
        for name, data in results.items():
            if data.get("status") != "success":
                print(f"   {name}: {data.get('error', 'Unknown error')}")
    
    # Cleanup
    try:
        import os
        if os.path.exists("datasets/temp_comparison_test.pkl"):
            os.remove("datasets/temp_comparison_test.pkl")
    except:
        pass
    
    print(f"\n✅ Comparison completed!")
    return results


if __name__ == "__main__":
    main()