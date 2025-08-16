#!/usr/bin/env python3
"""
Evaluate the SFT model trained with working notebook approach.
"""

import os
import sys
import pickle
import numpy as np
import torch
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🧪 Evaluating SFT Model from Working Notebook Approach")
    print("=" * 60)
    
    # Load the trained SFT model
    print("🚀 Loading trained SFT model...")
    
    from unsloth import FastLanguageModel
    
    # Load base model first
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.7,
    )
    
    # Load the SFT adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "models/working_notebook/sft_model")
    
    print("✅ SFT model loaded successfully")
    
    # Setup chat template
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
    
    chat_template = ("{% if messages[0]['role'] == 'system' %}"
                    "{{ messages[0]['content'] + eos_token }}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% else %}"
                    "{{ '" + system_prompt + "' + eos_token }}"
                    "{% set loop_messages = messages %}"
                    "{% endif %}"
                    "{% for message in loop_messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ message['content'] }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
                    "{% endif %}")
    
    tokenizer.chat_template = chat_template
    
    # Test on multiple initial conditions
    test_cases = [
        (0.5, -0.3),   # Original test case
        (0.8, 0.2),    # Different quadrant
        (-0.4, 0.6),   # Negative position
        (0.3, -0.7),   # High negative velocity
        (-0.6, -0.4),  # Both negative
    ]
    
    print("\n🧪 Testing SFT model on multiple initial conditions...")
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.3,  # Lower temperature for more consistent results
        top_k=50,
        max_tokens=1024,
    )
    
    results = []
    
    for i, (x0, v0) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: x0={x0:.1f}, v0={v0:.1f} ---")
        
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
            output = model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=None,
            )[0].outputs[0].text
            
            # Analyze response
            has_reasoning = reasoning_start in output and reasoning_end in output
            has_controls = solution_start in output and solution_end in output
            
            print(f"✅ Generated response")
            print(f"   Has reasoning: {has_reasoning}")
            print(f"   Has controls: {has_controls}")
            
            if has_controls:
                control_match = re.search(rf"{solution_start}(.*?){solution_end}", output, re.DOTALL)
                if control_match:
                    try:
                        control_text = control_match.group(1).strip()
                        control_values = [float(x.strip()) for x in control_text.split(',')]
                        
                        if len(control_values) == steps:
                            print(f"   ✅ Correct number of controls: {len(control_values)}")
                            print(f"   Control range: [{min(control_values):.3f}, {max(control_values):.3f}]")
                            
                            # Simulate trajectory
                            x, v = x0, v0
                            trajectory = [(x, v)]
                            valid_trajectory = True
                            
                            for u in control_values:
                                if not (-3 <= u <= 3):
                                    valid_trajectory = False
                                    break
                                    
                                v = v + u * dt
                                x = x + v * dt
                                trajectory.append((x, v))
                                
                                if not (-1 <= x <= 1 and -1 <= v <= 1):
                                    valid_trajectory = False
                                    break
                            
                            final_error = np.sqrt(x**2 + v**2)
                            
                            print(f"   Final position: ({x:.4f}, {v:.4f})")
                            print(f"   Final error: {final_error:.4f}")
                            print(f"   Valid trajectory: {valid_trajectory}")
                            print(f"   Control constraints satisfied: {all(-3 <= u <= 3 for u in control_values)}")
                            
                            # Success criteria
                            success = final_error < 0.1 and valid_trajectory
                            print(f"   🎯 SUCCESS: {success}")
                            
                            results.append({
                                'initial_state': (x0, v0),
                                'final_state': (x, v),
                                'final_error': final_error,
                                'valid_trajectory': valid_trajectory,
                                'success': success,
                                'controls': control_values,
                                'has_reasoning': has_reasoning,
                                'response_length': len(output)
                            })
                        else:
                            print(f"   ❌ Wrong number of controls: {len(control_values)} (expected {steps})")
                            results.append({
                                'initial_state': (x0, v0),
                                'success': False,
                                'error': 'Wrong number of controls'
                            })
                    except Exception as e:
                        print(f"   ❌ Failed to parse controls: {e}")
                        results.append({
                            'initial_state': (x0, v0),
                            'success': False,
                            'error': f'Parse error: {e}'
                        })
                else:
                    print(f"   ❌ No control sequence found")
                    results.append({
                        'initial_state': (x0, v0),
                        'success': False,
                        'error': 'No control sequence found'
                    })
            else:
                print(f"   ❌ Missing control tags")
                results.append({
                    'initial_state': (x0, v0),
                    'success': False,
                    'error': 'Missing control tags'
                })
                
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            results.append({
                'initial_state': (x0, v0),
                'success': False,
                'error': f'Generation failed: {e}'
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 SFT MODEL EVALUATION SUMMARY")
    print("="*60)
    
    successful_cases = [r for r in results if r.get('success', False)]
    success_rate = len(successful_cases) / len(results) * 100
    
    print(f"Overall Success Rate: {success_rate:.1f}% ({len(successful_cases)}/{len(results)})")
    
    if successful_cases:
        avg_error = np.mean([r['final_error'] for r in successful_cases])
        print(f"Average Final Error (successful cases): {avg_error:.4f}")
    
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        initial = result['initial_state']
        if result.get('success', False):
            final = result['final_state']
            error = result['final_error']
            print(f"  {i}. Initial: ({initial[0]:.1f}, {initial[1]:.1f}) → Final: ({final[0]:.4f}, {final[1]:.4f}) | Error: {error:.4f} ✅")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  {i}. Initial: ({initial[0]:.1f}, {initial[1]:.1f}) → {error_msg} ❌")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if success_rate < 50:
        print("   ⚠️  Low success rate - consider:")
        print("   1. More SFT training epochs")
        print("   2. Better data quality")
        print("   3. GRPO training for improvement")
    elif success_rate < 80:
        print("   📈 Moderate success rate - consider:")
        print("   1. GRPO training for fine-tuning")
        print("   2. Additional training data")
    else:
        print("   🎉 Good success rate - ready for:")
        print("   1. GRPO training for optimization")
        print("   2. Production deployment")

if __name__ == "__main__":
    main()