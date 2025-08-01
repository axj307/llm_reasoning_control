#!/usr/bin/env python3
"""
Test the trainer's reward function wrapping logic.
"""

import sys
import os

# Add trainer to path
sys.path.insert(0, 'trainer')

from rewards import get_all_reward_functions

def test_trainer_reward_wrapping():
    """Test the reward function wrapping logic from the trainer."""
    
    # Mock tokenizer
    class MockTokenizer:
        eos_token = "</s>"
    
    # Simulate the trainer's reward function setup
    def create_format_wrapper(func, tokenizer):
        def wrapper(completions, **kwargs):
            return func(completions, tokenizer=tokenizer, **kwargs)
        return wrapper
    
    # Get reward functions with proper wrapping (like in trainer)
    reward_funcs = []
    mock_tokenizer = MockTokenizer()
    
    for func in get_all_reward_functions():
        if func.__name__ == "match_format_exactly":
            # Add tokenizer for format matching function
            reward_funcs.append(create_format_wrapper(func, mock_tokenizer))
        else:
            # Other functions don't need tokenizer injection
            reward_funcs.append(func)
    
    # Test data
    mock_prompts = [
        [{"role": "user", "content": "Control a double integrator system with initial state [position=0.5, velocity=-0.3] to reach the origin (0,0) in 5.00 seconds using 50 steps."}]
    ]
    
    mock_completions = [
        [{"content": "Some reasoning</REASONING><CONTROLS>1.0, 2.0, 3.0, -1.0, -2.0</CONTROLS>"}]
    ]
    
    mock_answers = ["1.0, 2.0, 3.0, -1.0, -2.0"]
    
    print("Testing wrapped reward functions...")
    
    for i, func in enumerate(reward_funcs):
        func_name = get_all_reward_functions()[i].__name__
        print(f"Testing wrapped {func_name}...")
        
        try:
            if func_name == "match_format_exactly":
                # Wrapped function should work with just completions
                scores = func(mock_completions)
                print(f"  ✓ wrapped {func_name}: score = {scores[0]}")
            elif func_name == "match_format_approximately":
                scores = func(mock_completions)
                print(f"  ✓ wrapped {func_name}: score = {scores[0]}")
            elif func_name == "evaluate_control_sequence":
                scores = func(mock_prompts, mock_completions, mock_answers)
                print(f"  ✓ wrapped {func_name}: score = {scores[0]}")
            else:
                print(f"  ? Unknown function: {func_name}")
                
        except Exception as e:
            print(f"  ✗ wrapped {func_name}: ERROR - {e}")
    
    print("\n✅ Trainer reward function wrapping test completed!")

if __name__ == "__main__":
    test_trainer_reward_wrapping()