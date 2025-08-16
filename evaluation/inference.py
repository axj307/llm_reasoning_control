"""Inference utilities for control models."""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from vllm import SamplingParams

from environments import get_system
from core.solvers import get_solver


def extract_controls_from_response(response_text: str,
                                 solution_start: str = "<CONTROLS>",
                                 solution_end: str = "</CONTROLS>") -> Optional[List[float]]:
    """Extract control values from model response using regex."""
    control_match = re.search(
        rf"{re.escape(solution_start)}(.*?){re.escape(solution_end)}", 
        response_text, re.DOTALL
    )
    
    if control_match is None:
        return None
    
    try:
        control_text = control_match.group(1).strip()
        control_values = [float(x.strip()) for x in control_text.split(',')]
        return control_values
    except Exception as e:
        return None


def extract_reasoning_from_response(response_text: str,
                                  reasoning_start: str = "<REASONING>",
                                  reasoning_end: str = "</REASONING>") -> Optional[str]:
    """Extract reasoning text from model response."""
    reasoning_match = re.search(
        rf"{re.escape(reasoning_start)}(.*?){re.escape(reasoning_end)}", 
        response_text, re.DOTALL
    )
    
    if reasoning_match is None:
        return None
    
    return reasoning_match.group(1).strip()


def run_inference(model, tokenizer, system_name: str,
                 initial_state: Tuple[float, float],
                 lora_request=None,
                 dt: float = 0.1,
                 steps: int = 50,
                 sampling_params: Optional[SamplingParams] = None,
                 reasoning_start: str = "<REASONING>",
                 reasoning_end: str = "</REASONING>",
                 solution_start: str = "<CONTROLS>",
                 solution_end: str = "</CONTROLS>") -> Dict[str, Any]:
    """
    Run inference with the model on a control problem.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer
        system_name: Name of the system ('double_integrator', 'van_der_pol', etc.)
        initial_state: (x0, v0) initial state
        lora_request: Optional LoRA request for inference
        dt: Time step
        steps: Number of control steps
        sampling_params: Optional sampling parameters
        
    Returns:
        Dictionary with inference results
    """
    # Create system instance
    system = get_system(system_name)(dt=dt, steps=steps)
    
    # Get system prompt
    system_prompt = system.get_system_prompt(
        reasoning_start, reasoning_end, solution_start, solution_end
    )
    
    # Get problem description
    initial_state_array = np.array(initial_state)
    problem = system.get_problem_description(initial_state_array)
    
    # Format chat message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    # Default sampling parameters
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_k=50,
            max_tokens=1024,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
    
    # Run inference - detect model type and use appropriate method
    if hasattr(model, 'fast_generate') and not hasattr(model, 'llm_engine'):
        # Unsloth model - use standard transformers generation
        generation_kwargs = {
            "max_new_tokens": getattr(sampling_params, 'max_tokens', 1024) if sampling_params else 1024,
            "temperature": getattr(sampling_params, 'temperature', 0.7) if sampling_params else 0.7,
            "top_k": getattr(sampling_params, 'top_k', 50) if sampling_params else 50,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate using standard transformers
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode only the new tokens
        output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
    else:
        # vLLM model - use fast_generate with sampling_params
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
    
    # Extract reasoning and controls
    reasoning = extract_reasoning_from_response(output, reasoning_start, reasoning_end)
    controls = extract_controls_from_response(output, solution_start, solution_end)
    
    # Get optimal solution for comparison
    solver = get_solver(system_name)
    if system_name == "double_integrator":
        optimal_controls = solver(initial_state[0], initial_state[1], dt, steps)
    elif system_name == "van_der_pol":
        optimal_controls = solver(initial_state[0], initial_state[1], 1.0, dt, steps)
    else:
        optimal_controls = solver(initial_state_array, dt, steps)
    
    # Simulate trajectories
    model_trajectory = None
    optimal_trajectory = None
    
    if controls and len(controls) == steps:
        model_trajectory = system.simulate_trajectory(initial_state_array, controls)
    
    optimal_trajectory = system.simulate_trajectory(initial_state_array, optimal_controls)
    
    return {
        "system_name": system_name,
        "initial_state": initial_state,
        "model_output": output,
        "reasoning": reasoning,
        "model_controls": controls,
        "optimal_controls": optimal_controls,
        "model_trajectory": model_trajectory,
        "optimal_trajectory": optimal_trajectory,
        "valid_extraction": controls is not None,
        "valid_format": controls is not None and len(controls) == steps,
        "messages": messages
    }


def run_batch_inference(model, tokenizer, system_name: str,
                       initial_states: List[Tuple[float, float]],
                       lora_request=None, **kwargs) -> List[Dict[str, Any]]:
    """Run inference on a batch of initial states."""
    results = []
    
    for i, initial_state in enumerate(initial_states):
        # Clean display of initial state values
        x0_clean = f"{float(initial_state[0]):.4f}"
        x1_clean = f"{float(initial_state[1]):.4f}"
        print(f"Processing {i+1}/{len(initial_states)}: ({x0_clean}, {x1_clean})")
        result = run_inference(
            model, tokenizer, system_name, initial_state, 
            lora_request=lora_request, **kwargs
        )
        results.append(result)
    
    return results


def run_mpc_inference(model, tokenizer, system_name: str,
                     initial_state: Tuple[float, float],
                     dt: float = 0.1,
                     total_steps: int = 50,
                     horizon: int = 10,
                     lora_request=None,
                     sampling_params: Optional[SamplingParams] = None) -> Dict[str, Any]:
    """
    Run Model Predictive Control (MPC) style inference.
    
    At each time step, predict a sequence of 'horizon' controls,
    but only apply the first one. Repeat for total_steps.
    """
    current_state = np.array(initial_state)
    applied_controls = []
    states_history = [current_state.copy()]
    times_history = [0.0]
    
    # Create system instance
    system = get_system(system_name)(dt=dt, steps=horizon)
    
    for step in range(total_steps):
        # Run inference for current state with prediction horizon
        current_horizon = min(horizon, total_steps - step)
        
        result = run_inference(
            model, tokenizer, system_name, 
            tuple(current_state),
            lora_request=lora_request,
            dt=dt,
            steps=current_horizon,
            sampling_params=sampling_params
        )
        
        # Extract the first control to apply
        if result["valid_format"] and len(result["model_controls"]) > 0:
            u_to_apply = result["model_controls"][0]
        else:
            # Fallback to zero control if extraction fails
            u_to_apply = 0.0
        
        # Clamp control within system bounds
        control_bounds = system.get_control_bounds()
        u_to_apply = max(control_bounds[0], min(control_bounds[1], u_to_apply))
        
        applied_controls.append(u_to_apply)
        
        # Simulate one step forward
        current_state = system.simulate_step(current_state, u_to_apply)
        
        # Clamp states within bounds
        state_bounds = system.get_state_bounds()
        for i, (low, high) in enumerate(state_bounds):
            current_state[i] = max(low, min(high, current_state[i]))
        
        states_history.append(current_state.copy())
        times_history.append((step + 1) * dt)
    
    # Create trajectory dictionary
    trajectory = {
        'states': np.array(states_history),
        'controls': applied_controls,
        'times': times_history,
        'valid_trajectory': True,  # Assume valid since we clamped
        'final_error': np.linalg.norm(current_state),
        'initial_state': np.array(initial_state),
        'final_state': current_state
    }
    
    return {
        "system_name": system_name,
        "initial_state": initial_state,
        "mpc_trajectory": trajectory,
        "applied_controls": applied_controls,
        "prediction_horizon": horizon,
        "total_steps": total_steps
    }


def create_interactive_tester(model, tokenizer, lora_request=None):
    """Create an interactive widget for testing the model (requires Jupyter)."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        # System selection
        system_dropdown = widgets.Dropdown(
            options=[('Double Integrator', 'double_integrator'), 
                    ('Van der Pol', 'van_der_pol')],
            value='double_integrator',
            description='System:',
        )
        
        # Initial state sliders
        x0_slider = widgets.FloatSlider(
            value=0.5, min=-1.5, max=1.5, step=0.1,
            description='Position:', continuous_update=False,
        )
        
        v0_slider = widgets.FloatSlider(
            value=-0.3, min=-1.5, max=1.5, step=0.1,
            description='Velocity:', continuous_update=False,
        )
        
        # Run button
        run_button = widgets.Button(
            description='Run Test', button_style='success'
        )
        
        # Output area
        output = widgets.Output()
        
        def on_button_click(b):
            with output:
                output.clear_output()
                result = run_inference(
                    model, tokenizer, 
                    system_dropdown.value,
                    (x0_slider.value, v0_slider.value),
                    lora_request=lora_request
                )
                
                # Display results
                if result["reasoning"]:
                    print("=== Model's Reasoning ===")
                    print(result["reasoning"])
                    print()
                
                if result["valid_format"]:
                    print(f"Model controls: {result['model_controls'][:5]}...")
                    if result["model_trajectory"]:
                        final_state = result["model_trajectory"]["final_state"]
                        print(f"Final state: [{final_state[0]:.4f}, {final_state[1]:.4f}]")
                        print(f"Final error: {result['model_trajectory']['final_error']:.6f}")
                else:
                    print("Failed to extract valid controls from model output")
                    print(f"Raw output: {result['model_output'][:200]}...")
        
        run_button.on_click(on_button_click)
        
        # Layout
        ui = widgets.VBox([
            system_dropdown,
            x0_slider,
            v0_slider,
            run_button,
            output
        ])
        
        return ui
    
    except ImportError:
        print("Interactive tester requires ipywidgets and Jupyter")
        return None