#!/usr/bin/env python3
"""Submit Slurm jobs for control LLM training with parameter sweeps."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def submit_sft_job(environment, samples, lora_rank, learning_rate, epochs, batch_size, job_name=None):
    """Submit an SFT training job."""
    
    if job_name is None:
        job_name = f"sft_{environment}_{samples}s_{lora_rank}r"
    
    env_vars = {
        'ENVIRONMENT': environment,
        'SAMPLES': str(samples),
        'LORA_RANK': str(lora_rank), 
        'LEARNING_RATE': str(learning_rate),
        'NUM_EPOCHS': str(epochs),
        'BATCH_SIZE': str(batch_size),
        'RUN_NAME': job_name
    }
    
    cmd = ['sbatch']
    
    # Add environment variables
    for key, value in env_vars.items():
        cmd.extend(['--export', f'{key}={value}'])
    
    # Add the script
    cmd.append('slurm/train_evaluate_sft.sbatch')
    
    print(f"Submitting SFT job: {job_name}")
    print(f"Environment: {environment}, Samples: {samples}, LoRA: {lora_rank}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Job submitted successfully! Job ID: {job_id}")
        return job_id
    else:
        print(f"❌ Job submission failed: {result.stderr}")
        return None

def submit_grpo_job(environment, samples, lora_rank, sft_lr, grpo_lr, sft_epochs, grpo_steps, job_name=None):
    """Submit a GRPO training job."""
    
    if job_name is None:
        job_name = f"grpo_{environment}_{samples}s_{lora_rank}r"
    
    env_vars = {
        'ENVIRONMENT': environment,
        'SAMPLES': str(samples),
        'LORA_RANK': str(lora_rank),
        'SFT_LEARNING_RATE': str(sft_lr),
        'GRPO_LEARNING_RATE': str(grpo_lr),
        'SFT_EPOCHS': str(sft_epochs),
        'GRPO_STEPS': str(grpo_steps),
        'RUN_NAME': job_name
    }
    
    cmd = ['sbatch']
    
    # Add environment variables
    for key, value in env_vars.items():
        cmd.extend(['--export', f'{key}={value}'])
    
    # Add the script
    cmd.append('slurm/train_evaluate_grpo.sbatch')
    
    print(f"Submitting GRPO job: {job_name}")
    print(f"Environment: {environment}, Samples: {samples}, LoRA: {lora_rank}")
    print(f"SFT LR: {sft_lr}, GRPO LR: {grpo_lr}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Job submitted successfully! Job ID: {job_id}")
        return job_id
    else:
        print(f"❌ Job submission failed: {result.stderr}")
        return None

def submit_universal_job(systems, samples_per_system, lora_rank, training_type, job_name=None):
    """Submit a universal training job."""
    
    systems_str = ",".join(systems)
    
    if job_name is None:
        job_name = f"universal_{len(systems)}sys_{samples_per_system}s_{lora_rank}r"
    
    env_vars = {
        'SYSTEMS': systems_str,
        'SAMPLES_PER_SYSTEM': str(samples_per_system),
        'LORA_RANK': str(lora_rank),
        'TRAINING_TYPE': training_type,
        'RUN_NAME': job_name
    }
    
    cmd = ['sbatch']
    
    # Add environment variables
    for key, value in env_vars.items():
        cmd.extend(['--export', f'{key}={value}'])
    
    # Add the script
    cmd.append('slurm/train_evaluate_universal.sbatch')
    
    print(f"Submitting Universal job: {job_name}")
    print(f"Systems: {systems_str}, Samples per system: {samples_per_system}")
    print(f"LoRA: {lora_rank}, Training: {training_type}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Job submitted successfully! Job ID: {job_id}")
        return job_id
    else:
        print(f"❌ Job submission failed: {result.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Submit control LLM training jobs to Slurm")
    
    subparsers = parser.add_subparsers(dest='job_type', help='Type of job to submit')
    
    # SFT job arguments
    sft_parser = subparsers.add_parser('sft', help='Submit SFT training job')
    sft_parser.add_argument('--environment', default='double_integrator',
                           choices=['double_integrator', 'van_der_pol'],
                           help='Environment to train on')
    sft_parser.add_argument('--samples', type=int, default=300,
                           help='Number of training samples')
    sft_parser.add_argument('--lora-rank', type=int, default=16,
                           help='LoRA rank')
    sft_parser.add_argument('--learning-rate', type=float, default=2e-4,
                           help='Learning rate')
    sft_parser.add_argument('--epochs', type=int, default=4,
                           help='Number of epochs')
    sft_parser.add_argument('--batch-size', type=int, default=4,
                           help='Batch size')
    sft_parser.add_argument('--job-name', type=str,
                           help='Custom job name')
    
    # GRPO job arguments  
    grpo_parser = subparsers.add_parser('grpo', help='Submit GRPO training job')
    grpo_parser.add_argument('--environment', default='double_integrator',
                            choices=['double_integrator', 'van_der_pol'],
                            help='Environment to train on')
    grpo_parser.add_argument('--samples', type=int, default=300,
                            help='Number of training samples')
    grpo_parser.add_argument('--lora-rank', type=int, default=16,
                            help='LoRA rank')
    grpo_parser.add_argument('--sft-lr', type=float, default=2e-4,
                            help='SFT learning rate')
    grpo_parser.add_argument('--grpo-lr', type=float, default=5e-6,
                            help='GRPO learning rate')
    grpo_parser.add_argument('--sft-epochs', type=int, default=4,
                            help='SFT epochs')
    grpo_parser.add_argument('--grpo-steps', type=int, default=100,
                            help='GRPO steps')
    grpo_parser.add_argument('--job-name', type=str,
                            help='Custom job name')
    
    # Universal job arguments
    universal_parser = subparsers.add_parser('universal', help='Submit universal training job')
    universal_parser.add_argument('--systems', nargs='+',
                                 default=['double_integrator', 'van_der_pol'],
                                 choices=['double_integrator', 'van_der_pol'],
                                 help='Systems to train on')
    universal_parser.add_argument('--samples-per-system', type=int, default=200,
                                 help='Samples per system')
    universal_parser.add_argument('--lora-rank', type=int, default=16,
                                 help='LoRA rank')
    universal_parser.add_argument('--training-type', choices=['sft', 'grpo', 'both'],
                                 default='both', help='Type of training')
    universal_parser.add_argument('--job-name', type=str,
                                 help='Custom job name')
    
    # Sweep arguments
    sweep_parser = subparsers.add_parser('sweep', help='Submit parameter sweep')
    sweep_parser.add_argument('--type', choices=['sft', 'grpo'], required=True,
                             help='Type of sweep')
    sweep_parser.add_argument('--environment', default='double_integrator',
                             choices=['double_integrator', 'van_der_pol'],
                             help='Environment for sweep')
    
    args = parser.parse_args()
    
    if args.job_type == 'sft':
        submit_sft_job(
            args.environment, args.samples, args.lora_rank,
            args.learning_rate, args.epochs, args.batch_size,
            args.job_name
        )
        
    elif args.job_type == 'grpo':
        submit_grpo_job(
            args.environment, args.samples, args.lora_rank,
            args.sft_lr, args.grpo_lr, args.sft_epochs, args.grpo_steps,
            args.job_name
        )
        
    elif args.job_type == 'universal':
        submit_universal_job(
            args.systems, args.samples_per_system, args.lora_rank,
            args.training_type, args.job_name
        )
        
    elif args.job_type == 'sweep':
        print(f"Running {args.type} parameter sweep for {args.environment}")
        
        if args.type == 'sft':
            # SFT parameter sweep
            configs = [
                # (samples, lora_rank, learning_rate, epochs)
                (200, 16, 1e-4, 3),
                (200, 16, 2e-4, 4),
                (200, 16, 5e-4, 5),
                (300, 32, 2e-4, 4),
                (400, 16, 2e-4, 4),
            ]
            
            job_ids = []
            for i, (samples, lora_rank, lr, epochs) in enumerate(configs):
                job_name = f"sft_sweep_{args.environment}_{i+1}"
                job_id = submit_sft_job(args.environment, samples, lora_rank, lr, epochs, 4, job_name)
                if job_id:
                    job_ids.append(job_id)
                print("-" * 50)
            
            print(f"\\n✅ Submitted {len(job_ids)} SFT sweep jobs: {', '.join(job_ids)}")
            
        elif args.type == 'grpo':
            # GRPO parameter sweep
            configs = [
                # (samples, lora_rank, sft_lr, grpo_lr, sft_epochs, grpo_steps)
                (200, 16, 2e-4, 5e-6, 4, 100),
                (200, 16, 2e-4, 1e-5, 4, 100),
                (200, 16, 2e-4, 5e-6, 4, 150),
                (300, 32, 2e-4, 5e-6, 4, 100),
                (200, 16, 1e-4, 5e-6, 3, 100),
            ]
            
            job_ids = []
            for i, (samples, lora_rank, sft_lr, grpo_lr, sft_epochs, grpo_steps) in enumerate(configs):
                job_name = f"grpo_sweep_{args.environment}_{i+1}"
                job_id = submit_grpo_job(args.environment, samples, lora_rank, sft_lr, grpo_lr, sft_epochs, grpo_steps, job_name)
                if job_id:
                    job_ids.append(job_id)
                print("-" * 50)
            
            print(f"\\n✅ Submitted {len(job_ids)} GRPO sweep jobs: {', '.join(job_ids)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()