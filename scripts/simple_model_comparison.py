#!/usr/bin/env python3
"""
Simple model comparison following exact notebook approach.
Compares all available models on the same test cases.
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
    print("🧪 SIMPLE MODEL COMPARISON - EXACT NOTEBOOK APPROACH")
    print("=" * 70)
    
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
    
    # Test cases
    test_cases = [
        (0.5, -0.3),   # Original test case
        (0.8, 0.2),    # Different quadrant
        (-0.4, 0.6),   # Negative position
        (0.3, -0.7),   # High negative velocity
        (-0.6, -0.4),  # Both negative
        (0.7, 0.5),    # High positive both
        (-0.8, -0.2),  # High negative position
        (0.1, 0.9),    # High velocity
    ]
    
    print(f"Using {len(test_cases)} test cases for evaluation")
    
    # Available models to evaluate
    models_to_evaluate = [
        {
            "name": "Original SFT",
            "path": "models/working_notebook/sft_model",
            "description": "Basic SFT training (2 epochs, loss: 0.0636)"
        },
        {
            "name": "Extended SFT", 
            "path": "models/working_notebook/sft_extended_exact_model",
            "description": "Extended SFT training (5 epochs, loss: 0.0544)"
        },
        {
            "name": "GRPO Fixed",
            "path": "models/working_notebook/grpo_fixed_model", 
            "description": "GRPO training with simplified rewards"
        },
        {
            "name": "GRPO Format",
            "path": "models/working_notebook/grpo_format_fixed_model",
            "description": "GRPO training with format-focused rewards"
        }
    ]
    
    # Filter to only available models
    available_models = [m for m in models_to_evaluate if Path(m["path"]).exists()]
    
    print(f"\n🔍 Available models ({len(available_models)}):")
    for i, model_info in enumerate(available_models, 1):
        print(f"  {i}. {model_info['name']}")
        print(f"     Description: {model_info['description']}")
    
    all_results = {}
    
    for model_info in available_models:
        print(f"\n{'='*70}")
        print(f"🧪 EVALUATING: {model_info['name']}")
        print(f"{'='*70}")
        
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
            
            # Load adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_info["path"])
            
            print(f"✅ Model loaded: {model_info['name']}")
            
            # Test on all test cases
            model_results = []
            
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.3,  # Lower temperature for consistent evaluation
                top_k=50,
                max_tokens=1024,
            )
            
            for i, (x0, v0) in enumerate(test_cases):
                print(f"   Test case {i+1}/{len(test_cases)} - Init: ({x0:.1f}, {v0:.1f})...", end=" ")
                
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
                    result = analyze_response(output, x0, v0, reasoning_start, reasoning_end, 
                                           solution_start, solution_end, dt, steps)
                    model_results.append(result)
                    
                    status = "✅" if result["success"] else "❌"
                    if result["success"]:
                        print(f"{status} Error: {result['final_error']:.4f}")
                    else:
                        print(f"{status} {result.get('error', 'Failed')}")
                    
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    model_results.append({
                        "success": False,
                        "error": f"Generation failed: {e}"
                    })
            
            all_results[model_info['name']] = {
                "info": model_info,
                "results": model_results
            }
            
            # Model summary
            successful_cases = [r for r in model_results if r.get("success", False)]
            success_rate = len(successful_cases) / len(model_results) * 100
            
            print(f"\n📊 {model_info['name']} Summary:")
            print(f"   Success rate: {success_rate:.1f}% ({len(successful_cases)}/{len(model_results)})")
            
            if successful_cases:
                avg_error = np.mean([r["final_error"] for r in successful_cases])
                print(f"   Average final error: {avg_error:.4f}")
                
                # Format compliance
                format_compliant = sum(1 for r in successful_cases if r.get("has_reasoning", False) and r.get("has_controls", False))
                print(f"   Format compliance: {format_compliant}/{len(successful_cases)} ({format_compliant/len(successful_cases)*100:.1f}%)")
                
                # Control diversity
                control_varieties = [r.get("control_variety", 0) for r in successful_cases]
                avg_variety = np.mean(control_varieties) if control_varieties else 0
                print(f"   Avg control diversity: {avg_variety:.1f} unique values")
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_info['name']}: {e}")
            all_results[model_info['name']] = {
                "info": model_info,
                "error": str(e)
            }
    
    # Final comparison table
    print(f"\n{'='*70}")
    print("📊 FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    
    print(f"{'Model':<15} {'Success %':<10} {'Avg Error':<12} {'Format %':<10} {'Diversity':<10}")
    print("-" * 70)
    
    model_rankings = []
    
    for model_name, data in all_results.items():
        if "error" in data:
            print(f"{model_name:<15} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            continue
            
        results = data["results"]
        successful_cases = [r for r in results if r.get("success", False)]
        success_rate = len(successful_cases) / len(results) * 100
        
        if successful_cases:
            avg_error = np.mean([r["final_error"] for r in successful_cases])
            format_compliant = sum(1 for r in successful_cases if r.get("has_reasoning", False) and r.get("has_controls", False))
            format_rate = format_compliant / len(successful_cases) * 100
            control_varieties = [r.get("control_variety", 0) for r in successful_cases]
            avg_variety = np.mean(control_varieties) if control_varieties else 0
            
            print(f"{model_name:<15} {success_rate:>6.1f}%    {avg_error:>8.4f}    {format_rate:>6.1f}%    {avg_variety:>6.1f}")
            
            # Calculate overall score for ranking
            score = success_rate + (1.0 - avg_error) * 10 + format_rate * 0.5
            model_rankings.append((model_name, score, success_rate))
        else:
            print(f"{model_name:<15} {success_rate:>6.1f}%    {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            model_rankings.append((model_name, 0, success_rate))
    
    # Rankings
    print(f"\n🏆 MODEL RANKINGS:")
    model_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score, success_rate) in enumerate(model_rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"   {medal} {model_name} - {success_rate:.1f}% success (score: {score:.1f})")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if model_rankings:
        best_model, best_score, best_success = model_rankings[0]
        print(f"   🏆 Best model: {best_model} with {best_success:.1f}% success rate")
        
        if best_success >= 80:
            print("   ✅ Excellent performance - ready for production use")
        elif best_success >= 60:
            print("   📈 Good performance - consider additional GRPO training")
        elif best_success >= 40:
            print("   ⚠️  Moderate performance - needs more training")
        else:
            print("   ❌ Poor performance - major improvements needed")
    
    # Analysis insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # Compare SFT models
    if "Original SFT" in all_results and "Extended SFT" in all_results:
        orig_success = len([r for r in all_results["Original SFT"]["results"] if r.get("success", False)]) / len(test_cases) * 100
        ext_success = len([r for r in all_results["Extended SFT"]["results"] if r.get("success", False)]) / len(test_cases) * 100
        improvement = ext_success - orig_success
        
        if improvement > 5:
            print(f"   📈 Extended SFT shows significant improvement (+{improvement:.1f}%)")
        elif improvement > 0:
            print(f"   📊 Extended SFT shows modest improvement (+{improvement:.1f}%)")
        else:
            print(f"   📉 Extended SFT shows no improvement ({improvement:+.1f}%)")
    
    # Check for constant control issue
    constant_control_models = []
    for model_name, data in all_results.items():
        if "error" not in data:
            results = data["results"]
            successful_cases = [r for r in results if r.get("success", False)]
            if successful_cases:
                avg_variety = np.mean([r.get("control_variety", 0) for r in successful_cases])
                if avg_variety < 5:  # Very low diversity suggests constant controls
                    constant_control_models.append(model_name)
    
    if constant_control_models:
        print(f"   ⚠️  Models with low control diversity: {', '.join(constant_control_models)}")
        print("       This suggests the constant control issue persists")
    
    print(f"\n✅ Comprehensive evaluation completed!")

def analyze_response(output, x0, v0, reasoning_start, reasoning_end, solution_start, solution_end, dt, steps):
    """Analyze model response and return evaluation metrics."""
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
        
        # Success threshold
        success = final_error < 0.1
        
        result.update({
            "success": success,
            "control_values": control_values,
            "final_state": (x, v),
            "final_error": final_error,
            "valid_trajectory": valid_trajectory,
            "control_variety": len(set(control_values))
        })
        
        return result
        
    except Exception as e:
        result["error"] = f"Parse error: {e}"
        return result

if __name__ == "__main__":
    main()