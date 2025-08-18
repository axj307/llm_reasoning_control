#!/usr/bin/env python3
"""
Debug GRPO training outputs to see actual LLM inputs and responses.
Shows question-answer pairs during training to verify format consistency.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from unsloth import FastLanguageModel
from vllm import SamplingParams
import re

def setup_model_for_debugging():
    """Set up model for debugging GRPO outputs."""
    print("🔍 GRPO OUTPUT DEBUG TOOL")
    print("=" * 50)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
    )
    
    # Add LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer

def create_sample_questions():
    """Create sample questions in dataset format."""
    system_prompt = """You are a control systems expert. Given a control problem, provide step-by-step reasoning and optimal control inputs.

Format your response exactly as:
<REASONING>
Your detailed reasoning here
</REASONING>

<CONTROLS>
comma-separated list of control values
</CONTROLS>"""

    questions = [
        {
            "system": system_prompt,
            "problem": "Control a double integrator system to move from initial position x=1.5, velocity=0.8 to the origin in 10 time steps with dt=0.1.",
            "expected_format": "Should contain <REASONING>...</REASONING> and <CONTROLS>u1,u2,u3,...</CONTROLS>"
        },
        {
            "system": system_prompt, 
            "problem": "For a double integrator at state (x=2.0, v=1.2), compute 5 optimal control inputs to reach the origin with minimal energy.",
            "expected_format": "Should contain detailed reasoning and 5 control values"
        },
        {
            "system": system_prompt,
            "problem": "Control van der Pol oscillator from initial state (x=0.5, v=-1.0) to origin in 8 steps with dt=0.1.",
            "expected_format": "Should handle van der Pol dynamics with nonlinear terms"
        }
    ]
    
    return questions

def test_model_outputs(model, tokenizer, questions, num_generations=2):
    """Test model outputs and show input/output pairs."""
    print("\n📊 TESTING MODEL OUTPUTS")
    print("=" * 50)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</CONTROLS>", tokenizer.eos_token]
    )
    
    for i, q in enumerate(questions):
        print(f"\n🔸 TEST CASE {i+1}")
        print("-" * 30)
        
        # Format chat message
        messages = [
            {"role": "system", "content": q["system"]},
            {"role": "user", "content": q["problem"]},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        print("📝 INPUT (Chat Template Applied):")
        print(text[:500] + "..." if len(text) > 500 else text)
        print()
        
        # Generate multiple responses
        for gen in range(num_generations):
            print(f"🤖 GENERATION {gen+1}:")
            print("-" * 20)
            
            # Generate response
            outputs = model.llm_engine.generate(
                text, 
                sampling_params=sampling_params
            )
            response = outputs[0].outputs[0].text
            
            print("💬 RAW OUTPUT:")
            print(response)
            print()
            
            # Check format compliance
            has_reasoning = "<REASONING>" in response and "</REASONING>" in response
            has_controls = "<CONTROLS>" in response and "</CONTROLS>" in response
            
            print("✅ FORMAT CHECK:")
            print(f"   Reasoning tags: {'✓' if has_reasoning else '✗'}")
            print(f"   Controls tags:  {'✓' if has_controls else '✗'}")
            
            # Extract and validate controls
            if has_controls:
                control_match = re.search(r"<CONTROLS>(.*?)</CONTROLS>", response, re.DOTALL)
                if control_match:
                    control_text = control_match.group(1).strip()
                    try:
                        controls = [float(x.strip()) for x in control_text.split(',')]
                        print(f"   Controls extracted: {len(controls)} values")
                        print(f"   Sample controls: {controls[:3]}...")
                    except Exception as e:
                        print(f"   ❌ Control parsing failed: {e}")
                else:
                    print("   ❌ Controls tag found but content not extracted")
            
            print(f"   Expected: {q['expected_format']}")
            print()
    
    print("🎯 DEBUGGING COMPLETE!")
    print("Check the outputs above to verify:")
    print("1. Chat template format is correct")
    print("2. Model generates proper <REASONING> and <CONTROLS> tags")
    print("3. Control values are parseable as floats")
    print("4. Format matches your dataset expectations")

def debug_grpo_dataset_format():
    """Debug the actual dataset format used in GRPO training."""
    print("\n📊 DATASET FORMAT DEBUG")
    print("=" * 50)
    
    try:
        import pickle
        # Check if we have dataset files
        dataset_files = [
            "datasets/di_train.pkl",
            "datasets/di_quick_train.pkl", 
            "datasets/universal_train.pkl"
        ]
        
        for dataset_file in dataset_files:
            if Path(dataset_file).exists():
                print(f"\n🔍 Checking {dataset_file}:")
                with open(dataset_file, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"   Samples: {len(data)}")
                if len(data) > 0:
                    sample = data[0]
                    print(f"   Keys: {list(sample.keys())}")
                    
                    if 'Messages' in sample:
                        messages = sample['Messages']
                        print(f"   Message format: {len(messages)} messages")
                        for j, msg in enumerate(messages):
                            role = msg.get('role', 'unknown')
                            content_preview = msg.get('content', '')[:100] + "..."
                            print(f"     {j+1}. {role}: {content_preview}")
                    
                    if len(data) > 1:
                        print(f"\n   📝 Sample assistant response:")
                        if 'Messages' in sample and len(sample['Messages']) >= 3:
                            assistant_msg = sample['Messages'][2]['content']
                            print(f"   {assistant_msg[:300]}...")
                
                print()
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")

def main():
    """Main debugging function."""
    
    # Debug dataset format first
    debug_grpo_dataset_format()
    
    # Set up model for testing
    print("⚙️ Setting up model...")
    model, tokenizer = setup_model_for_debugging()
    
    # Create test questions
    questions = create_sample_questions()
    
    # Test model outputs
    test_model_outputs(model, tokenizer, questions, num_generations=2)

if __name__ == "__main__":
    main()