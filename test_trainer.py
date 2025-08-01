#!/usr/bin/env python3
"""
Test script to verify the modular trainer package works correctly.
"""

import sys
import os

# Ensure we're in the right conda environment
os.system("conda activate unsloth_env")

# Test imports
print("Testing imports...")
try:
    from trainer import ControlTrainer, create_dataset
    from trainer.control import solve_double_integrator
    from trainer.utils import parse_control_output
    from trainer.config import DT, STEPS
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test data generation
print("\nTesting data generation...")
try:
    dataset = create_dataset(num_samples=5)
    print(f"✓ Generated dataset with {len(dataset)} samples")
except Exception as e:
    print(f"✗ Data generation error: {e}")
    sys.exit(1)

# Test control solver
print("\nTesting control solver...")
try:
    controls = solve_double_integrator(0.5, -0.3, DT, STEPS)
    print(f"✓ Generated {len(controls)} control values")
    print(f"  First 5 controls: {controls[:5]}")
except Exception as e:
    print(f"✗ Control solver error: {e}")
    sys.exit(1)

# Test reward functions
print("\nTesting reward functions...")
try:
    from trainer.rewards import match_format_exactly
    test_completion = [[{"content": "Some reasoning</REASONING><CONTROLS>1.0, 2.0, 3.0</CONTROLS>"}]]
    scores = match_format_exactly(test_completion)
    print(f"✓ Reward function test passed. Score: {scores[0]}")
except Exception as e:
    print(f"✗ Reward function error: {e}")
    sys.exit(1)

print("\n✅ All tests passed! The modular structure is working correctly.")
print("\nYou can now run the training with:")
print("  python trainer/main.py --mode train --num-samples 100 --sft-epochs 1 --grpo-steps 10")
print("\nOr for a quick test:")
print("  python trainer/main.py --mode eval")