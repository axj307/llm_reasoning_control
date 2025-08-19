"""SFT training implementation."""

import torch
from typing import Dict, List, Any, Optional
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

from core.model_manager import UniversalModelManager


def train_sft_model(model_manager: UniversalModelManager,
                   train_data: List[Dict[str, Any]],
                   eval_data: Optional[List[Dict[str, Any]]] = None,
                   training_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Train SFT model on control data.
    
    Args:
        model_manager: Universal model manager with loaded model
        train_data: Training data from data pipeline
        eval_data: Optional evaluation data
        training_config: Training configuration
    
    Returns:
        Training results and metrics
    """
    # Default training config
    default_config = {
        "output_dir": "./temp_sft_output",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 10,
        "num_train_epochs": 4,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "eval_steps": 100,
        "save_steps": 100,  # Match eval_steps
        "eval_strategy": "steps",
        "save_strategy": "steps",  # Match eval_strategy
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "report_to": "none",  # Disable wandb for simplicity
        "save_total_limit": 3,
        "load_best_model_at_end": False,  # Disable for simplicity
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    
    if training_config:
        default_config.update(training_config)
    
    # Store training config in model manager
    model_manager.training_config = default_config
    
    # Check if chat template is set up, if not, set it up
    if not hasattr(model_manager.tokenizer, 'chat_template') or model_manager.tokenizer.chat_template is None:
        print("⚠️  Chat template not set, setting up default template...")
        # Set up a basic chat template
        from environments import get_system
        
        # Get system type from first example
        system_type = train_data[0].get("system_type", "double_integrator")
        
        try:
            system = get_system(system_type)()
            system_prompt = system.get_system_prompt(
                "<REASONING>", "</REASONING>", "<CONTROLS>", "</CONTROLS>"
            )
        except:
            # Fallback system prompt
            system_prompt = "You are a control systems expert."
        
        model_manager.setup_chat_template(
            reasoning_start="<REASONING>",
            reasoning_end="</REASONING>",
            solution_start="<CONTROLS>",
            solution_end="</CONTROLS>",
            system_prompt=system_prompt
        )
        print("✅ Chat template set up successfully")
    
    # Prepare datasets for SFT training - EXACT notebook approach
    def format_for_sft(example):
        """Format a single example for SFT training - EXACT working notebook approach."""
        # EXACT copy from working notebook:
        full_text = model_manager.tokenizer.apply_chat_template(
            example["Messages"],
            tokenize=False
        )
        return {"text": full_text}
    
    # Convert to HuggingFace format
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_for_sft)
    
    eval_dataset = None
    if eval_data:
        eval_dataset = Dataset.from_list(eval_data)
        eval_dataset = eval_dataset.map(format_for_sft)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(**default_config)
    
    # Pre fine-tune the model to learn the format
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,  # Increased from 1
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",
        ),
    )

    # Run pre-training
    result = trainer.train()
    
    # Save the trained SFT model
    print("\n💾 Saving SFT model...")
    save_path = "models/single_system/double_integrator/sft/latest"
    
    # Create directory
    import os
    import json
    import numpy as np
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model using proper PEFT method
    model_manager.model.save_pretrained(save_path)
    model_manager.tokenizer.save_pretrained(save_path)
    
    # Create metadata.json file for evaluation script
    metadata = {
        "model_type": "sft",
        "system_name": "double_integrator",
        "training_type": "sft",
        "base_model": "unsloth/Qwen3-4B-Base",
        "max_seq_length": 2048,
        "lora_rank": 32,
        "training_loss": float(result.training_loss),
        "timestamp": str(np.datetime64('now')),
        "training_config": default_config
    }
    
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ SFT model saved to: {save_path}")
    print(f"   📋 Metadata saved with training loss: {result.training_loss:.4f}")
    
    return result


def setup_universal_chat_template(model_manager: UniversalModelManager,
                                systems: List[str],
                                reasoning_start: str = "<REASONING>",
                                reasoning_end: str = "</REASONING>",
                                solution_start: str = "<CONTROLS>",
                                solution_end: str = "</CONTROLS>"):
    """Set up chat template for universal model."""
    
    # Universal system prompt that works for any system
    if len(systems) == 1:
        # Single system prompt
        from environments import get_system
        system = get_system(systems[0])()
        system_prompt = system.get_system_prompt(reasoning_start, reasoning_end, 
                                                solution_start, solution_end)
    else:
        # Universal prompt for multi-system model
        dt = 0.1  # Default from config
        steps = 50  # Default from config
        total_time = dt * steps
        
        systems_str = ", ".join(systems)
        system_prompt = f"""You are a universal control systems expert.
Given any control system ({systems_str}) with its description, initial state, and constraints,
generate a sequence of {steps} control inputs to reach the target state in {total_time:.2f} seconds.
Analyze the system dynamics, identify the appropriate control approach, and ensure all constraints are satisfied.
Explain your approach between {reasoning_start} and {reasoning_end}.
Then provide exactly {steps} control values as a comma-separated list between {solution_start} and {solution_end}."""
    
    model_manager.setup_chat_template(
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        solution_start=solution_start,
        solution_end=solution_end,
        system_prompt=system_prompt
    )
    
    return system_prompt


def evaluate_sft_model(trainer: SFTTrainer, eval_dataset: Dataset) -> Dict[str, float]:
    """Evaluate the SFT model."""
    print("Running SFT evaluation...")
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    return eval_result


def save_sft_model(model_manager: UniversalModelManager,
                  systems: List[str],
                  metrics: Dict[str, Any],
                  run_name: Optional[str] = None) -> str:
    """Save the trained SFT model."""
    if len(systems) == 1:
        # Single system model
        save_path = model_manager.save_single_system_checkpoint(
            system_name=systems[0],
            training_type="sft",
            run_name=run_name,
            metrics=metrics
        )
    else:
        # Universal model
        save_path = model_manager.save_universal_checkpoint(
            systems_list=systems,
            training_type="sft",
            run_name=run_name,
            metrics=metrics
        )
    
    return save_path