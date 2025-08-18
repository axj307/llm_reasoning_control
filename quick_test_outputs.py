#!/usr/bin/env python3
"""
Quick test to see LLM inputs and outputs without SLURM.
Run this locally to debug question-answer format.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def check_dataset_format():
    """Quick check of dataset format."""
    print("🔍 DATASET FORMAT CHECK")
    print("=" * 40)
    
    import pickle
    
    dataset_files = [
        "datasets/di_train.pkl",
        "datasets/di_quick_train.pkl", 
        "datasets/universal_train.pkl"
    ]
    
    for dataset_file in dataset_files:
        if Path(dataset_file).exists():
            print(f"\n📂 {dataset_file}:")
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"   📊 Samples: {len(data)}")
            if len(data) > 0:
                sample = data[0]
                print(f"   🔑 Keys: {list(sample.keys())}")
                
                if 'Messages' in sample:
                    messages = sample['Messages']
                    print(f"   💬 Messages: {len(messages)}")
                    
                    print(f"\n   📝 SAMPLE MESSAGES:")
                    for j, msg in enumerate(messages):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        print(f"      {j+1}. {role}: {len(content)} chars")
                        if role == 'user':
                            print(f"         Question: {content}")
                        elif role == 'assistant':
                            print(f"         Answer preview: {content[:200]}...")
                            
                            # Check format
                            has_reasoning = "<REASONING>" in content and "</REASONING>" in content
                            has_controls = "<CONTROLS>" in content and "</CONTROLS>" in content
                            print(f"         Format check: Reasoning ✓" if has_reasoning else "         Format check: Reasoning ✗")
                            print(f"                       Controls ✓" if has_controls else "                       Controls ✗")
        else:
            print(f"❌ {dataset_file} not found")

def test_inference_format():
    """Test inference format if models are available."""
    print("\n🤖 INFERENCE FORMAT TEST")
    print("=" * 40)
    
    try:
        from core.model_manager import ModelManager
        from evaluation.inference import run_inference
        
        manager = ModelManager()
        
        # Check for available models
        model_paths = [
            "models/single_system/double_integrator/sft/latest",
            "models/universal/sft/latest"
        ]
        
        for model_path in model_paths:
            if Path(model_path).exists():
                print(f"\n✅ Testing model: {model_path}")
                try:
                    model, tokenizer, lora_request, metadata = manager.load_checkpoint(model_path)
                    
                    # Test with sample question
                    result = run_inference(
                        model, tokenizer, 'double_integrator',
                        initial_state=(1.5, 0.8),
                        dt=0.1,
                        steps=5,
                        lora_request=lora_request
                    )
                    
                    print(f"📝 SAMPLE INFERENCE:")
                    print(f"   Input: double_integrator, state=(1.5, 0.8), 5 steps")
                    print(f"   Reasoning: {result.get('reasoning', 'None')[:150]}...")
                    print(f"   Controls: {result.get('controls', 'None')}")
                    print(f"   Success: {result.get('success', False)}")
                    break
                    
                except Exception as e:
                    print(f"❌ Error testing {model_path}: {e}")
        else:
            print("❌ No trained models found for testing")
            
    except Exception as e:
        print(f"❌ Error in inference test: {e}")

def main():
    """Main test function."""
    print("🚀 QUICK LLM OUTPUT TEST")
    print("=" * 50)
    print("This will show you:")
    print("1. Dataset format and sample questions/answers")
    print("2. Model inference format (if models available)")
    print("3. Format consistency checks")
    print()
    
    # Check dataset format
    check_dataset_format()
    
    # Test inference if possible
    test_inference_format()
    
    print("\n✅ QUICK TEST COMPLETED!")
    print("\n💡 WHAT TO LOOK FOR:")
    print("1. Questions should be clear control problems")
    print("2. Answers should have <REASONING> and <CONTROLS> tags")
    print("3. Controls should be comma-separated numbers")
    print("4. Format should be consistent across samples")

if __name__ == "__main__":
    main()