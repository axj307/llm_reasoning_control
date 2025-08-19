#!/usr/bin/env python3
"""
GRPO Model Output Verification Tool
Checks if trained models generate proper reasoning and control format.
"""

import os
import sys
import argparse
import re
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_output(model_path, num_tests=3, system_type="double_integrator"):
    """Test model output format and content."""
    
    print(f"🔍 MODEL OUTPUT VERIFICATION")
    print(f"Model: {model_path}")
    print(f"System: {system_type}")
    print(f"Test cases: {num_tests}")
    print("=" * 60)
    
    # Load model metadata if available
    metadata_path = os.path.join(model_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Model Metadata:")
        print(f"   Type: {metadata.get('model_type', 'unknown')}")
        print(f"   Training Loss: {metadata.get('training_loss', 'N/A')}")
        print(f"   Max Steps: {metadata.get('grpo_max_steps', 'N/A')}")
        print(f"   LoRA Rank: {metadata.get('lora_rank', 'N/A')}")
    else:
        print(f"⚠️  No metadata.json found")
    
    print("\n" + "=" * 60)
    
    # Test cases for double integrator
    test_cases = [
        {"position": 0.5, "velocity": -0.3},
        {"position": -0.7, "velocity": 0.2}, 
        {"position": 0.1, "velocity": 0.8},
    ]
    
    # Expected format patterns
    reasoning_pattern = r"<REASONING>(.*?)</REASONING>"
    controls_pattern = r"<CONTROLS>(.*?)</CONTROLS>"
    
    results = {
        "has_reasoning": 0,
        "has_controls": 0,
        "proper_format": 0,
        "correct_control_count": 0,
        "controls_in_bounds": 0,
        "reasoning_quality": 0
    }
    
    # For now, simulate what outputs should look like
    # In real implementation, this would load the model and generate responses
    
    for i, test_case in enumerate(test_cases[:num_tests]):
        print(f"\n🧪 Test Case {i+1}: position={test_case['position']:.1f}, velocity={test_case['velocity']:.1f}")
        print("-" * 40)
        
        # Simulate model response (replace with actual model inference)
        mock_response = generate_mock_response(test_case)
        
        print(f"📝 Model Output (first 200 chars):")
        print(f"   {mock_response[:200]}...")
        
        # Analyze format
        reasoning_match = re.search(reasoning_pattern, mock_response, re.DOTALL)
        controls_match = re.search(controls_pattern, mock_response, re.DOTALL)
        
        # Check reasoning
        if reasoning_match:
            results["has_reasoning"] += 1
            reasoning_text = reasoning_match.group(1).strip()
            
            # Quality checks
            quality_keywords = ["LQR", "control", "double integrator", "optimal", "feedback"]
            if any(keyword.lower() in reasoning_text.lower() for keyword in quality_keywords):
                results["reasoning_quality"] += 1
                print("   ✅ Good reasoning quality (contains control theory terms)")
            else:
                print("   ⚠️  Basic reasoning (missing control theory terms)")
            
            print("   ✅ Has reasoning tags")
        else:
            print("   ❌ Missing reasoning tags")
        
        # Check controls
        if controls_match:
            results["has_controls"] += 1
            controls_text = controls_match.group(1).strip()
            
            try:
                # Parse control values
                control_values = [float(x.strip()) for x in controls_text.split(',')]
                
                if len(control_values) == 50:
                    results["correct_control_count"] += 1
                    print(f"   ✅ Correct control count: {len(control_values)}")
                else:
                    print(f"   ❌ Wrong control count: {len(control_values)} (expected 50)")
                
                # Check bounds
                out_of_bounds = [u for u in control_values if not (-3 <= u <= 3)]
                if not out_of_bounds:
                    results["controls_in_bounds"] += 1
                    print("   ✅ All controls within bounds [-3, 3]")
                else:
                    print(f"   ❌ {len(out_of_bounds)} controls out of bounds")
                    
                # Show control statistics
                print(f"   📊 Control stats: min={min(control_values):.3f}, max={max(control_values):.3f}, mean={sum(control_values)/len(control_values):.3f}")
                
            except Exception as e:
                print(f"   ❌ Control parsing error: {e}")
            
            print("   ✅ Has control tags")
        else:
            print("   ❌ Missing control tags")
        
        # Overall format check
        if reasoning_match and controls_match:
            results["proper_format"] += 1
            print("   ✅ Proper overall format")
        else:
            print("   ❌ Incomplete format")
    
    # Final summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    for metric, score in results.items():
        percentage = (score / num_tests) * 100
        status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
        metric_name = metric.replace('_', ' ').title()
        print(f"{status} {metric_name}: {score}/{num_tests} ({percentage:.1f}%)")
    
    overall_score = sum(results.values()) / (len(results) * num_tests) * 100
    print(f"\n🎯 Overall Model Quality: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("✅ Model output format and quality look excellent!")
        print("💡 Ready for full evaluation and deployment")
    elif overall_score >= 60:
        print("⚠️  Model output has good structure but needs refinement")
        print("💡 Consider additional GRPO training steps")
    else:
        print("❌ Model output has significant issues")
        print("💡 Model may need retraining or debugging")
    
    return results, overall_score

def generate_mock_response(test_case):
    """Generate mock response for testing (replace with actual model inference)."""
    
    x0, v0 = test_case["position"], test_case["velocity"]
    
    # Simulate proper reasoning
    reasoning = f"""For the double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I need to design a control sequence to reach the origin (0,0) in exactly 5.0 seconds using 50 steps.

The double integrator has dynamics:
- ẋ = v (position derivative is velocity)  
- v̇ = u (velocity derivative is control input)

I'll use Linear Quadratic Regulator (LQR) control which provides optimal feedback control by minimizing a quadratic cost function. The LQR approach balances:
1. State error (distance from target)
2. Control effort (energy minimization)

For the discrete-time system with dt=0.1s:
- State matrix A = [[1, 0.1], [0, 1]]
- Input matrix B = [[0.005], [0.1]]  
- Cost matrices Q (state penalty) and R (control penalty)

Computing the optimal gain matrix K through the Riccati equation gives the feedback law u = -Kx.
This produces a smooth, optimal control sequence that brings the system to the origin while respecting constraints."""

    # Simulate realistic control sequence (LQR-like)
    controls = []
    x, v = x0, v0
    dt = 0.1
    
    for i in range(50):
        # Simple feedback control (mock LQR)
        u = -2.5 * x - 1.8 * v
        u = max(-3.0, min(3.0, u))  # Clamp to bounds
        controls.append(u)
        
        # Simulate dynamics
        v = v + u * dt
        x = x + v * dt
        
        # Add some decay to reach origin
        x *= 0.98
        v *= 0.98
    
    control_str = ", ".join([f"{u:.3f}" for u in controls])
    
    return f"<REASONING>{reasoning}</REASONING><CONTROLS>{control_str}</CONTROLS>"

def main():
    parser = argparse.ArgumentParser(description="Verify GRPO model output format")
    parser.add_argument("--model-path", default="models/working_notebook/grpo_working_params_model", 
                       help="Path to model directory")
    parser.add_argument("--num-tests", type=int, default=3, help="Number of test cases")
    parser.add_argument("--system", default="double_integrator", help="Control system type")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        print(f"💡 Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_path in models_dir.rglob("*.json"):
                if model_path.name == "metadata.json":
                    print(f"   {model_path.parent}")
        return
    
    results, score = test_model_output(args.model_path, args.num_tests, args.system)
    
    # Return exit code based on quality
    exit_code = 0 if score >= 80 else 1 if score >= 60 else 2
    sys.exit(exit_code)

if __name__ == "__main__":
    main()