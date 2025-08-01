"""
GRPO reward functions for evaluating generated control sequences.
"""

import re
import numpy as np
from config import *


def match_format_exactly(completions=None, tokenizer=None, **kwargs):
    """Reward for exact format matching."""
    # Handle GRPO calling pattern where completions might be in kwargs
    if completions is None:
        completions = kwargs.get('completions', [])
    scores = []
    eos_token = tokenizer.eos_token if tokenizer else "</s>"
    
    solution_end_regex = r"</CONTROLS>[\s]{0,}" + \
        "(?:" + re.escape(eos_token) + ")?"
    
    match_format = re.compile(
        rf"{REASONING_END}.*?"
        rf"{SOLUTION_START}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )
    
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions=None, **kwargs):
    """Reward for approximate format matching."""
    # Handle GRPO calling pattern where completions might be in kwargs
    if completions is None:
        completions = kwargs.get('completions', [])
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(REASONING_END) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
        scores.append(score)
    return scores


def evaluate_control_sequence(prompts=None, completions=None, answer=None, **kwargs):
    """Evaluate generated control sequences."""
    # Handle GRPO calling pattern where args might be in kwargs
    if prompts is None:
        prompts = kwargs.get('prompts', [])
    if completions is None:
        completions = kwargs.get('completions', [])
    if answer is None:
        answer = kwargs.get('answer', [])
    scores = []
    
    for completion, true_answer in zip(completions, answer):
        score = 0
        response = completion[0]["content"]
        
        # Extract control sequence
        control_match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", response, re.DOTALL)
        if control_match is None:
            scores.append(-2.0)
            continue
            
        try:
            # Parse control values
            control_text = control_match.group(1).strip()
            control_values = [float(x.strip()) for x in control_text.split(',')]
            
            # Check length
            if len(control_values) == STEPS:
                score += 1.0
            else:
                score -= 1.0
                
            # Check bounds
            if all(-3 <= u <= 3 for u in control_values):
                score += 1.0
            else:
                score -= 2.0
            
            # Check smoothness (LQR characteristic)
            if len(control_values) > 1:
                diffs = [abs(control_values[i] - control_values[i-1]) for i in range(1, len(control_values))]
                if max(diffs) < 1.5:
                    score += 1.5
                    
            # Extract initial conditions
            problem_text = prompts[0][-1]["content"]
            initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", problem_text)
            if initial_match:
                x0 = float(initial_match.group(1))
                v0 = float(initial_match.group(2))
                
                # Simulate system
                x, v = x0, v0
                valid_trajectory = True
                
                for u in control_values:
                    v = v + u * DT
                    x = x + v * DT
                    
                    if not (-1 <= x <= 1 and -1 <= v <= 1):
                        valid_trajectory = False
                        break
                
                # Reward valid trajectory
                if valid_trajectory:
                    score += 1.0
                else:
                    score -= 1.0
                
                # Reward based on final error
                final_error = np.sqrt(x**2 + v**2)
                if final_error < 0.1:
                    score += 3.0
                elif final_error < 0.2:
                    score += 2.0
                elif final_error < 0.5:
                    score += 1.0
                else:
                    score -= 1.0
            
            scores.append(score)
            
        except Exception as e:
            scores.append(-2.0)
            
    return scores


def get_all_reward_functions():
    """Return list of all reward functions."""
    return [
        match_format_exactly,
        match_format_approximately,
        evaluate_control_sequence,
    ]