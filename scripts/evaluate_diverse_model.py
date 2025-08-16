#!/usr/bin/env python3
"""
Evaluate the diverse SFT model to confirm control diversity improvement.
"""

import torch
import numpy as np
import random
import pickle
import re
import os
import sys
from pathlib import Path

sys.path.append('.')

def main():
    print("🧪 EVALUATING DIVERSE SFT MODEL")
    print("=" * 60)
    
    # Set seeds (exact from notebook)
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)
    
    # GPU selection (exact from notebook)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        chosen_gpu = random.randint(0, num_gpus - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        print(f"Randomly selected GPU: {chosen_gpu}")
    else:
        print("No GPUs available.")
    
    # Model parameters (EXACT from working notebook)
    max_seq_length = 2048
    lora_rank = 32
    
    # Control system parameters
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    dt = 0.1
    steps = 50
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    system_prompt = get_system_prompt(dt, steps)
    
    # Chat template (EXACT from working notebook)
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"
    
    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    # Test cases - same as before for comparison
    test_cases = [
        (0.5, -0.3),   # Original problematic case
        (0.8, 0.2),    # Different quadrant
        (-0.4, 0.6),   # Negative position
        (0.3, -0.7),   # High negative velocity
        (-0.6, -0.4),  # Both negative
        (0.7, 0.5),    # High positive both
        (-0.8, -0.2),  # High negative position
        (0.1, 0.9),    # High velocity
    ]
    
    print(f"Testing {len(test_cases)} cases for control diversity improvement")
    
    # Check if diverse model exists
    diverse_model_path = "models/working_notebook/sft_diverse_model"
    if not Path(diverse_model_path).exists():
        print(f"❌ Diverse model not found at {diverse_model_path}")
        print("   Please run train_diverse_sft.py first")
        return
    
    try:
        # Load base model (EXACT from working notebook)
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.7,
        )
        
        tokenizer.chat_template = chat_template
        
        # Load diverse model adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, diverse_model_path)
        
        print(f"✅ Diverse SFT model loaded successfully")
        
        # Test on all cases
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.3,
            top_k=50,
            max_tokens=1024,
        )
        
        results = []
        total_diversity = 0
        successful_cases = 0
        constant_control_cases = 0
        
        print(f"\n📊 DETAILED EVALUATION RESULTS:")
        print("=" * 70)
        
        for i, (x0, v0) in enumerate(test_cases):
            print(f"\nTest Case {i+1}: Initial state ({x0:.1f}, {v0:.1f})")
            print("-" * 40)
            
            total_time = dt * steps
            test_problem = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time:.2f} seconds using {steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
            
            test_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_problem},
            ]
            
            text = tokenizer.apply_chat_template(
                test_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            try:
                # Generate response
                output = model.fast_generate(
                    text,
                    sampling_params=sampling_params,
                    lora_request=None,
                )[0].outputs[0].text
                
                # Analyze response
                result = analyze_diverse_response(output, x0, v0, reasoning_start, reasoning_end, 
                                               solution_start, solution_end, dt, steps)
                results.append(result)
                
                # Print results
                if result["success"]:
                    print(f"   ✅ SUCCESS: Final error {result['final_error']:.4f}")
                    successful_cases += 1
                else:
                    print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                
                if result.get("has_controls", False):
                    diversity = result.get("control_variety", 0)
                    total_diversity += diversity
                    
                    print(f"   📊 Control diversity: {diversity}/50 unique values")
                    print(f"   📊 Control range: [{result.get('min_control', 0):.3f}, {result.get('max_control', 0):.3f}]")
                    
                    if diversity <= 3:
                        print(f"   ⚠️  CONSTANT CONTROLS detected!")
                        constant_control_cases += 1
                    elif diversity < 15:
                        print(f"   📉 Low diversity")
                    elif diversity < 30:
                        print(f"   📊 Moderate diversity") 
                    else:
                        print(f"   ✅ Good diversity")
                        
                    # Show sample controls
                    if "control_values" in result:
                        sample_controls = result["control_values"][:10]
                        sample_str = ", ".join([f"{u:.3f}" for u in sample_controls])
                        print(f"   📝 First 10 controls: {sample_str}...")
                
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                results.append({"success": False, "error": f"Generation failed: {e}"})
        
        # Final summary
        print(f"\n{'='*70}")
        print("🎯 IMPROVEMENT ANALYSIS")
        print("=" * 70)
        
        valid_results = [r for r in results if r.get("has_controls", False)]
        avg_diversity = total_diversity / len(valid_results) if valid_results else 0
        success_rate = successful_cases / len(test_cases) * 100
        constant_rate = constant_control_cases / len(valid_results) * 100 if valid_results else 0
        
        print(f"Overall Performance:")
        print(f"  Success rate: {success_rate:.1f}% ({successful_cases}/{len(test_cases)})")
        print(f"  Average control diversity: {avg_diversity:.1f}/50 unique values")
        print(f"  Constant control cases: {constant_control_cases}/{len(valid_results)} ({constant_rate:.1f}%)")
        
        print(f"\n📊 Comparison with Previous Models:")
        print(f"  Previous SFT models: ~1-3 unique values (CONSTANT)")
        print(f"  Diverse SFT model: {avg_diversity:.1f} unique values")
        
        if avg_diversity > 15:
            print(f"  🎉 SIGNIFICANT IMPROVEMENT! Control diversity greatly increased")
        elif avg_diversity > 5:
            print(f"  📈 Moderate improvement, but still room for enhancement")
        else:
            print(f"  ❌ Limited improvement, constant control issue persists")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if constant_rate < 20:
            print(f"  ✅ Ready for GRPO training to further optimize control performance")
        elif avg_diversity > 10:
            print(f"  📈 Consider additional SFT epochs or GRPO training")
        else:
            print(f"  🔧 May need further dataset improvements or training modifications")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def analyze_diverse_response(output, x0, v0, reasoning_start, reasoning_end, solution_start, solution_end, dt, steps):
    """Analyze model response with focus on control diversity."""
    result = {
        "success": False,
        "has_reasoning": False,
        "has_controls": False,
        "error": None
    }
    
    # Check format
    result["has_reasoning"] = reasoning_start in output and reasoning_end in output
    result["has_controls"] = solution_start in output and solution_end in output
    
    if not result["has_controls"]:
        result["error"] = "Missing control tags"
        return result
    
    # Extract controls
    control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
    if not control_match:
        result["error"] = "No control sequence found"
        return result
    
    try:
        control_text = control_match.group(1).strip()
        control_values = [float(x.strip()) for x in control_text.split(',')]
        
        if len(control_values) != steps:
            result["error"] = f"Wrong number of controls: {len(control_values)} (expected {steps})"
            return result
        
        # Check control constraints
        if not all(-3 <= u <= 3 for u in control_values):
            result["error"] = "Control constraints violated"
            return result
        
        # Simulate trajectory
        x, v = x0, v0
        valid_trajectory = True
        
        for u in control_values:
            v = v + u * dt
            x = x + v * dt
            
            if not (-1 <= x <= 1 and -1 <= v <= 1):
                valid_trajectory = False
                break
        
        if not valid_trajectory:
            result["error"] = "State constraints violated"
            return result
        
        # Calculate final error
        final_error = np.sqrt(x**2 + v**2)
        success = final_error < 0.1
        
        # Analyze control diversity
        control_variety = len(set(control_values))
        min_control = min(control_values)
        max_control = max(control_values)
        
        result.update({
            "success": success,
            "control_values": control_values,
            "final_state": (x, v),
            "final_error": final_error,
            "valid_trajectory": valid_trajectory,
            "control_variety": control_variety,
            "min_control": min_control,
            "max_control": max_control
        })
        
        return result
        
    except Exception as e:
        result["error"] = f"Parse error: {e}"
        return result

if __name__ == "__main__":
    main()