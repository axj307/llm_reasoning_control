"""
Exact reward functions from working Qwen3 GRPO notebook
"""
import numpy as np
import re

# Global variables
reasoning_start = "<REASONING>"
reasoning_end = "</REASONING>"
solution_start = "<CONTROLS>"
solution_end = "</CONTROLS>"
dt = 0.1
steps = 50

# Exact regex pattern from working notebook
solution_end_regex = r"</CONTROLS>[\s]{0,}"

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None: 
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores

def evaluate_control_sequence(prompts, completions, answer, **kwargs):
    """Enhanced evaluation of control sequences with LQR characteristics."""
    scores = []
    
    for completion, true_answer in zip(completions, answer):
        score = 0
        response = completion[0]["content"]
        
        # Extract control sequence
        control_match = re.search(rf"{solution_start}(.*?){solution_end}", response, re.DOTALL)
        if control_match is None:
            scores.append(-2.0)
            continue
            
        try:
            # Parse control values
            control_text = control_match.group(1).strip()
            control_values = [float(x.strip()) for x in control_text.split(',')]
            
            # Check constraints
            if len(control_values) == steps:
                score += 1.0
            else:
                score -= 1.0
                
            if all(-3 <= u <= 3 for u in control_values):
                score += 1.0
            else:
                score -= 2.0
            
            # Check for smoothness
            if len(control_values) > 1:
                diffs = [abs(control_values[i] - control_values[i-1]) for i in range(1, len(control_values))]
                if max(diffs) < 1.5:
                    score += 1.5
                    
            # Simulate system
            problem_text = prompts[0][-1]["content"]
            initial_match = re.search(r"position=([-\d\.]+), velocity=([-\d\.]+)", problem_text)
            if initial_match:
                x0 = float(initial_match.group(1))
                v0 = float(initial_match.group(2))
                
                # Simulate system with generated controls
                x, v = x0, v0
                valid_trajectory = True
                
                for u in control_values:
                    v = v + u * dt
                    x = x + v * dt
                    
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

def setup_working_chat_template(tokenizer):
    """Set up exact chat template from working notebook"""
    
    dt = 0.1
    steps = 50
    total_time = dt * steps
    
    system_prompt = f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
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
    
    tokenizer.chat_template = chat_template
    
    return system_prompt

def create_working_grpo_config(tokenizer):
    """Create exact GRPO config from working notebook"""
    
    from trl import GRPOConfig
    
    # Don't use vLLM sampling params directly to avoid serialization issues
    # Instead use the individual parameters that GRPOConfig can handle
    training_args = GRPOConfig(
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 4,  # Must be multiple of num_generations
        gradient_accumulation_steps = 1,
        num_generations = 4,
        max_completion_length = 1024,  # Reduced to avoid sequence length issues
        max_steps = 100,
        save_steps = 500,
        report_to = "none",
        output_dir = "outputs",
    # Note: Sampling params like top_p/top_k are not accepted by GRPOConfig in this TRL version.
    # Use tokenizer/chat template for stop tokens and defaults within the trainer.
    )
    
    return training_args
