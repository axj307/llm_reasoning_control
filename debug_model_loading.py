#!/usr/bin/env python3
"""Debug script to test model loading issues."""

import os
import torch
import traceback

def test_basic_model_loading():
    """Test basic model loading without our framework."""
    print("🔧 Testing basic model loading...")
    
    try:
        from unsloth import FastLanguageModel
        
        print("   ✅ Unsloth imported successfully")
        
        # Try loading the model with minimal parameters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=512,  # Small sequence length for testing
            dtype=None,
            load_in_4bit=True,
        )
        
        print("   ✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer type: {type(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_with_lora():
    """Test model loading with LoRA configuration."""
    print("🔧 Testing model loading with LoRA...")
    
    try:
        from unsloth import FastLanguageModel
        
        # Try loading with LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print("   ✅ LoRA model configured successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ LoRA configuration failed: {e}")
        traceback.print_exc()
        return False

def test_environment_info():
    """Print environment information."""
    print("🌍 Environment Information:")
    print(f"   Python version: {torch.__version__}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")
        print(f"   GPU name: {torch.cuda.get_device_name()}")
    
    try:
        import unsloth
        print(f"   Unsloth version: {unsloth.__version__}")
    except:
        print("   Unsloth version: Not available")
    
    try:
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
    except:
        print("   Transformers version: Not available")

def test_alternative_model():
    """Test with a different model to isolate the issue."""
    print("🔧 Testing alternative model...")
    
    try:
        from unsloth import FastLanguageModel
        
        # Try a different model that might work better
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-2-7b-bnb-4bit",  # Alternative model
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        
        print("   ✅ Alternative model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Alternative model failed: {e}")
        print("   This suggests a general issue with the setup")
        return False

def main():
    """Run all debug tests."""
    print("=" * 60)
    print("🚀 MODEL LOADING DEBUG")
    print("=" * 60)
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache")
    
    tests = [
        test_environment_info,
        test_basic_model_loading,
        test_with_lora,
        test_alternative_model,
    ]
    
    passed = 0
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == 0:
        print("❌ All model loading tests failed!")
        print("Suggestions:")
        print("1. Check if you have the right versions of unsloth/transformers")
        print("2. Try restarting your Python session")  
        print("3. Check GPU memory usage")
        print("4. Try a different model")
    elif passed < len(tests):
        print("⚠️  Some tests failed - check specific errors above")
    else:
        print("✅ All tests passed - model loading should work!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()