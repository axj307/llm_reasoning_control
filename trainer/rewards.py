"""
GRPO reward functions for evaluating generated control sequences.
"""

import re
import numpy as np
import importlib.util
import sys
from pathlib import Path

# Import constants from the old config.py file
config_path = Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("old_config", config_path)
old_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_config)

# Use constants from old config
REASONING_END = old_config.REASONING_END
SOLUTION_START = old_config.SOLUTION_START
SOLUTION_END = old_config.SOLUTION_END
STEPS = old_config.STEPS
DT = old_config.DT


def match_format_exactly(completions=None, tokenizer=None, **kwargs):
    """Reward for exact format matching."""
    try:
        # Handle GRPO calling pattern where completions might be in kwargs
        if completions is None:
            completions = kwargs.get('completions', [])
        
        # Ensure completions is a list
        if not isinstance(completions, list):
            completions = [completions] if completions is not None else []
        
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
            score = 0.0
            try:
                # Handle different completion formats
                if isinstance(completion, str):
                    response = completion
                elif isinstance(completion, list) and len(completion) > 0:
                    if isinstance(completion[0], dict) and "content" in completion[0]:
                        response = completion[0]["content"]
                    else:
                        response = str(completion[0])
                else:
                    response = str(completion)
                    
                if match_format.search(response) is not None:
                    score = 3.0
            except Exception:
                score = 0.0
            
            scores.append(score)
        
        return scores
    except Exception as e:
        # Return zero scores if there's any error
        return [0.0] * (len(completions) if completions and hasattr(completions, '__len__') else 4)


def match_format_approximately(completions=None, **kwargs):
    """Reward for approximate format matching."""
    try:
        # Handle GRPO calling pattern where completions might be in kwargs
        if completions is None:
            completions = kwargs.get('completions', [])
        
        # Ensure completions is a list
        if not isinstance(completions, list):
            completions = [completions] if completions is not None else []
            
        scores = []
        for completion in completions:
            score = 0.0
            try:
                # Handle different completion formats
                if isinstance(completion, str):
                    response = completion
                elif isinstance(completion, list) and len(completion) > 0:
                    if isinstance(completion[0], dict) and "content" in completion[0]:
                        response = completion[0]["content"]
                    else:
                        response = str(completion[0])
                else:
                    response = str(completion)
                    
                score += 0.5 if response.count(REASONING_END) == 1 else -1.0
                score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
                score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
            except Exception:
                score = 0.0
            scores.append(score)
        return scores
    except Exception:
        return [0.0] * (len(completions) if completions and hasattr(completions, '__len__') else 4)


def evaluate_control_sequence(prompts=None, completions=None, answer=None, **kwargs):
    """Evaluate generated control sequences."""
    try:
        # Handle GRPO calling pattern where args might be in kwargs
        if prompts is None:
            prompts = kwargs.get('prompts', [])
        if completions is None:
            completions = kwargs.get('completions', [])
        if answer is None:
            answer = kwargs.get('answer', [])
        
        # Ensure all are lists
        if not isinstance(completions, list):
            completions = [completions] if completions is not None else []
        if not isinstance(prompts, list):
            prompts = [prompts] if prompts is not None else []
        if not isinstance(answer, list):
            answer = [answer] if answer is not None else []
            
        scores = []
        
        for i, completion in enumerate(completions):
            score = 0.0
            try:
                # Handle different completion formats
                if isinstance(completion, str):
                    response = completion
                elif isinstance(completion, list) and len(completion) > 0:
                    if isinstance(completion[0], dict) and "content" in completion[0]:
                        response = completion[0]["content"]
                    else:
                        response = str(completion[0])
                else:
                    response = str(completion)
                
                # Extract control sequence
                control_match = re.search(rf"{SOLUTION_START}(.*?){SOLUTION_END}", response, re.DOTALL)
                if control_match is None:
                    score = -2.0
                else:
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
                    if prompts and len(prompts) > 0:
                        if isinstance(prompts[0], str):
                            problem_text = prompts[0]
                        elif isinstance(prompts[0], list) and len(prompts[0]) > 0:
                            if isinstance(prompts[0][-1], dict) and "content" in prompts[0][-1]:
                                problem_text = prompts[0][-1]["content"]
                            else:
                                problem_text = str(prompts[0][-1])
                        else:
                            problem_text = str(prompts[0])
                    else:
                        problem_text = ""
                        
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
                
            except Exception:
                scores.append(-2.0)
                
        return scores
    except Exception:
        return [0.0] * (len(completions) if completions and hasattr(completions, '__len__') else 4)


def get_all_reward_functions():
    """Return list of all reward functions."""
    return [
        match_format_exactly,
        match_format_approximately,
        evaluate_control_sequence,
    ]