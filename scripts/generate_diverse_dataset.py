#!/usr/bin/env python3
"""
Generate diverse dataset with emphasis on control diversity.
Addresses the constant control generation issue.
"""

import numpy as np
import re
import pickle
import sys
from pathlib import Path

sys.path.append('.')

def solve_double_integrator_diverse(x0, v0, dt, steps, strategy="mixed"):
    """
    Generate diverse control solutions using multiple strategies.
    """
    import scipy.linalg as la
    
    # System dynamics
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5*dt**2], [dt]])
    
    total_time = dt * steps
    
    if strategy == "lqr":
        # Standard LQR solution
        Q = np.diag([10.0, 10.0])
        R = np.array([[0.1]])
        P = la.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        
        x = np.array([[x0], [v0]])
        controls = []
        
        for i in range(steps):
            u = -K @ x
            u_clamped = max(-3.0, min(3.0, float(u[0])))
            x = A @ x + B * u_clamped
            x[0,0] = max(-1.0, min(1.0, x[0,0]))
            x[1,0] = max(-1.0, min(1.0, x[1,0]))
            controls.append(u_clamped)
            
        return controls
        
    elif strategy == "bang_bang":
        # Bang-bang control strategy
        controls = []
        x, v = x0, v0
        
        for i in range(steps):
            # Simple bang-bang logic
            if abs(x) > abs(v) * dt:
                # Focus on position
                u = -3.0 if x > 0 else 3.0
            else:
                # Focus on velocity
                u = -3.0 if v > 0 else 3.0
            
            # Apply control
            v = v + u * dt
            x = x + v * dt
            controls.append(u)
            
        return controls
        
    elif strategy == "smooth_approach":
        # Smooth approach strategy
        controls = []
        x, v = x0, v0
        
        for i in range(steps):
            # Proportional control with smooth transitions
            remaining_steps = steps - i
            if remaining_steps > 0:
                # Desired final velocity to reach origin
                desired_v = -x / (remaining_steps * dt)
                v_error = desired_v - v
                u = v_error / dt
                u = max(-3.0, min(3.0, u))
            else:
                u = 0
                
            v = v + u * dt
            x = x + v * dt
            controls.append(u)
            
        return controls
        
    elif strategy == "mixed":
        # Randomly choose strategy to create diversity
        strategies = ["lqr", "bang_bang", "smooth_approach"]
        chosen_strategy = np.random.choice(strategies)
        return solve_double_integrator_diverse(x0, v0, dt, steps, chosen_strategy)
        
    elif strategy == "noisy_lqr":
        # LQR with small perturbations for diversity
        controls = solve_double_integrator_diverse(x0, v0, dt, steps, "lqr")
        noise_level = 0.2
        for i in range(len(controls)):
            noise = np.random.normal(0, noise_level)
            controls[i] = max(-3.0, min(3.0, controls[i] + noise))
        return controls
    
    else:
        # Default to LQR
        return solve_double_integrator_diverse(x0, v0, dt, steps, "lqr")

def generate_reasoning_text(x0, v0, dt, steps, controls, strategy="mixed"):
    """Generate diverse reasoning explanations based on strategy used."""
    total_time = dt * steps
    
    # Analyze the control sequence to understand what was actually generated
    control_changes = sum(1 for i in range(1, len(controls)) if abs(controls[i] - controls[i-1]) > 0.1)
    avg_control = np.mean(controls)
    max_control = max(controls)
    min_control = min(controls)
    
    if strategy == "bang_bang" or (max_control > 2.5 and min_control < -2.5):
        reasoning = f"""For this double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll use a bang-bang control strategy to reach the origin in {total_time:.2f} seconds.

The bang-bang approach uses maximum control effort (±3) to achieve rapid state changes. This strategy is optimal for time-critical scenarios where we want to reach the target as quickly as possible.

I'll analyze the current state at each step:
- If position error dominates, apply maximum control to counteract position
- If velocity error dominates, apply maximum control to counteract velocity
- Switch control sign based on the dominant error term

This produces {len(controls)} control inputs that aggressively drive the system toward the origin while respecting the constraint boundaries."""

    elif strategy == "smooth_approach" or control_changes > len(controls) * 0.7:
        reasoning = f"""For this double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll use a smooth proportional control approach to reach the origin in {total_time:.2f} seconds.

The smooth approach calculates the desired velocity at each step to gradually approach the target. This method provides:
- Continuous control adjustments based on current error
- Smooth trajectory without abrupt control changes
- Better constraint satisfaction for position and velocity bounds

At each time step, I compute the required velocity to reach the origin in the remaining time, then calculate the control input needed to achieve that velocity. This produces {len(controls)} smoothly varying control inputs."""

    elif abs(avg_control) < 0.5 and control_changes < len(controls) * 0.3:
        reasoning = f"""For this double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply a stabilization control strategy optimized for {total_time:.2f} seconds.

Given the initial conditions, the system requires moderate control effort to reach the origin. I'll use:
- Conservative control magnitudes to ensure constraint satisfaction
- Gradual adjustments to avoid overshoot
- Consistent control pattern to maintain stability

The control sequence maintains position and velocity within [-1,1] bounds while providing {len(controls)} control inputs that systematically drive the system to equilibrium."""

    else:
        # Default LQR-style reasoning
        reasoning = f"""For this double integrator system starting at position {x0:.2f} and velocity {v0:.2f}, I'll apply Linear Quadratic Regulator (LQR) control to optimally reach the origin in {total_time:.2f} seconds.

The LQR approach balances state error and control effort by minimizing a quadratic cost function. This provides:
- Optimal feedback control law u = -Kx
- Smooth convergence to the target state
- Efficient use of control energy

For the discrete-time double integrator with dynamics:
- x(k+1) = x(k) + v(k)·Δt + 0.5·u(k)·Δt²
- v(k+1) = v(k) + u(k)·Δt

The resulting {len(controls)} control inputs optimally balance convergence speed with control effort while maintaining all constraints."""

    return reasoning

def generate_diverse_control_dataset(num_samples=3000, target_dt=0.1, target_steps=50):
    """Generate diverse double integrator control problems with varied solutions."""
    print(f"🎯 Generating diverse dataset with {num_samples} samples...")
    
    data = []
    total_time_sec = target_dt * target_steps
    
    # Control system parameters
    reasoning_start = "<REASONING>"
    reasoning_end = "</REASONING>"
    solution_start = "<CONTROLS>"
    solution_end = "</CONTROLS>"
    
    def get_system_prompt(current_dt, current_steps):
        total_time = current_dt * current_steps
        return f"""You are a control systems expert.
Given a double integrator system (ẍ = u) with initial position and velocity,
generate a sequence of {current_steps} control inputs to reach the origin (0,0) in exactly {total_time:.2f} seconds.
Position and velocity must stay within [-1, 1], and control inputs must be within [-3, 3].
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {current_steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    sample_system_prompt = get_system_prompt(target_dt, target_steps)
    
    # Strategy distribution for diversity
    strategies = ["lqr", "bang_bang", "smooth_approach", "noisy_lqr", "mixed"]
    strategy_weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Favor LQR but include diversity
    
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"   Generated {i}/{num_samples} samples...")
        
        # Generate diverse initial states
        if i < num_samples * 0.7:
            # 70% from uniform distribution
            x0 = np.random.uniform(-0.8, 0.8)
            v0 = np.random.uniform(-0.8, 0.8)
        else:
            # 30% from challenging edge cases
            if np.random.random() < 0.5:
                # High initial distance
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.6, 0.9)
                x0 = radius * np.cos(angle)
                v0 = radius * np.sin(angle)
            else:
                # Near-boundary cases
                x0 = np.random.choice([-0.8, 0.8]) + np.random.normal(0, 0.1)
                v0 = np.random.choice([-0.8, 0.8]) + np.random.normal(0, 0.1)
                x0 = max(-0.9, min(0.9, x0))
                v0 = max(-0.9, min(0.9, v0))
        
        # Problem statement
        problem = f"Control a double integrator system with initial state [position={x0:.2f}, velocity={v0:.2f}] to reach the origin (0,0) in {total_time_sec:.2f} seconds using {target_steps} steps. Ensure all states remain within [-1,1] and controls within [-3,3]."
        
        # Choose strategy for diversity
        strategy = np.random.choice(strategies, p=strategy_weights)
        
        # Solve with chosen strategy
        control_inputs = solve_double_integrator_diverse(x0, v0, target_dt, target_steps, strategy)
        
        # Generate reasoning based on actual controls
        reasoning = generate_reasoning_text(x0, v0, target_dt, target_steps, control_inputs, strategy)
        
        # Format control values
        control_str = ", ".join([f"{u:.3f}" for u in control_inputs])
        
        # Create complete output
        complete_output = f"{reasoning_start}{reasoning}{reasoning_end}{solution_start}{control_str}{solution_end}"
        
        # Add to dataset
        data.append({
            "system_type": "double_integrator",
            "initial_state": [x0, v0],
            "controls": control_inputs,
            "strategy": strategy,
            "system_prompt": sample_system_prompt,
            "problem": problem,
            "reasoning": reasoning,
            "complete_output": complete_output,
            "messages": [
                {"role": "system", "content": sample_system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": complete_output}
            ]
        })
    
    # Analyze diversity
    all_controls = [item["controls"] for item in data]
    control_varieties = [len(set(controls)) for controls in all_controls]
    avg_variety = np.mean(control_varieties)
    
    strategies_used = [item["strategy"] for item in data]
    strategy_counts = {s: strategies_used.count(s) for s in strategies}
    
    print(f"\n📊 Dataset Diversity Analysis:")
    print(f"   Total samples: {len(data)}")
    print(f"   Average control diversity: {avg_variety:.1f} unique values per sequence")
    print(f"   Strategy distribution:")
    for strategy, count in strategy_counts.items():
        print(f"     {strategy}: {count} ({count/len(data)*100:.1f}%)")
    
    return data

def main():
    print("🚀 GENERATING DIVERSE CONTROL DATASET")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(3407)
    
    # Generate diverse dataset
    diverse_data = generate_diverse_control_dataset(num_samples=3000)
    
    # Split into train/eval
    split_ratio = 0.9
    split_index = int(len(diverse_data) * split_ratio)
    
    train_data = diverse_data[:split_index]
    eval_data = diverse_data[split_index:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Evaluation: {len(eval_data)} samples")
    
    # Save datasets
    train_path = "datasets/di_diverse_train.pkl"
    eval_path = "datasets/di_diverse_eval.pkl"
    
    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    
    with open(eval_path, "wb") as f:
        pickle.dump(eval_data, f)
    
    print(f"\n💾 Datasets saved:")
    print(f"   Training: {train_path}")
    print(f"   Evaluation: {eval_path}")
    
    # Quality check - analyze a few samples
    print(f"\n🔍 Quality Check:")
    for i in range(3):
        sample = train_data[i]
        controls = sample["controls"]
        variety = len(set(controls))
        strategy = sample["strategy"]
        control_range = f"[{min(controls):.2f}, {max(controls):.2f}]"
        
        print(f"   Sample {i+1}: {variety} unique controls, strategy: {strategy}, range: {control_range}")
    
    print(f"\n✅ Diverse dataset generation completed!")
    print(f"   🎯 Focus: Control diversity and varied strategies")
    print(f"   📈 Expected improvement: Reduced constant control generation")

if __name__ == "__main__":
    main()