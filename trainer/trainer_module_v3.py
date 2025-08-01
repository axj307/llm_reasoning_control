"""
Control trainer v3 with integrated evaluation pipeline.
"""

import torch
import gc
import os
from unsloth import FastLanguageModel
from vllm import SamplingParams
from typing import Optional, Dict, Any, List, Union
from datasets import Dataset
import numpy as np

from config import *
from data import get_system_prompt, create_dataset
from trainers import TrainerFactory
from logger import logger
from utils import parse_control_output
from evaluation import EvaluationManager


class ControlTrainerV3:
    """Enhanced trainer with integrated evaluation capabilities."""
    
    def __init__(self, model_name: str = MODEL_NAME, use_modular: bool = True):
        """
        Initialize trainer.
        
        Args:
            model_name: Base model name or path to fine-tuned model
            use_modular: Whether to use new modular trainers
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_modular = use_modular
        
        # Modular trainers
        self.sft_trainer = None
        self.grpo_trainer = None
        
        # Evaluation manager
        self.eval_manager = EvaluationManager()
        
        # Check if loading a fine-tuned model
        self.is_finetuned = os.path.exists(model_name) and os.path.isdir(model_name)
        
    def setup_model(self):
        """Load model and configure LoRA."""
        logger.info(f"Loading model: {self.model_name}")
        
        if self.is_finetuned:
            # Load fine-tuned model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
                fast_inference=True,
            )
        else:
            # Load base model and configure LoRA
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
                fast_inference=True,
                max_lora_rank=LORA_RANK,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            )
            
            # Configure LoRA
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=LORA_RANK,
                target_modules=TARGET_MODULES,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=SEED,
            )
        
        # Set chat template if needed
        if "qwen" in self.model_name.lower():
            self.tokenizer.chat_template = """{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        
        logger.info("Model loaded successfully")
        
    def train_sft(self, dataset: Dataset, output_dir: str = SFT_OUTPUT_DIR):
        """Train with supervised fine-tuning."""
        if self.model is None:
            self.setup_model()
            
        if self.use_modular:
            # Use modular trainer
            if self.sft_trainer is None:
                self.sft_trainer = TrainerFactory.create(
                    "sft", 
                    self.model, 
                    self.tokenizer,
                    config={
                        'batch_size': SFT_BATCH_SIZE,
                        'learning_rate': SFT_LEARNING_RATE,
                        'epochs': SFT_EPOCHS,
                        'output_dir': output_dir,
                        'max_seq_length': MAX_SEQ_LENGTH,
                    }
                )
            
            metrics = self.sft_trainer.train(dataset)
            logger.info(f"SFT completed with metrics: {metrics}")
        else:
            # Use original implementation
            from training import train_sft_original
            train_sft_original(self.model, self.tokenizer, dataset, output_dir)
            
    def train_grpo(self, dataset: Dataset, output_dir: str = GRPO_OUTPUT_DIR):
        """Train with Group Relative Policy Optimization."""
        if self.model is None:
            self.setup_model()
            
        if self.use_modular:
            # Use modular trainer
            if self.grpo_trainer is None:
                self.grpo_trainer = TrainerFactory.create(
                    "grpo",
                    self.model,
                    self.tokenizer,
                    config={
                        'batch_size': GRPO_BATCH_SIZE,
                        'learning_rate': GRPO_LEARNING_RATE,
                        'max_steps': GRPO_MAX_STEPS,
                        'output_dir': output_dir,
                        'temperature': GRPO_TEMPERATURE,
                        'num_generations': GRPO_NUM_GENERATIONS,
                        'max_seq_length': MAX_SEQ_LENGTH,
                    }
                )
            
            metrics = self.grpo_trainer.train(dataset)
            logger.info(f"GRPO completed with metrics: {metrics}")
        else:
            # Use original implementation
            from training import train_grpo_original
            train_grpo_original(self.model, self.tokenizer, dataset, output_dir)
            
    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Generate response for a prompt."""
        if self.model is None:
            self.setup_model()
            
        # Prepare input
        if isinstance(prompt, str):
            # Simple string prompt
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        else:
            # Already formatted messages
            messages = prompt
            
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 512),
                temperature=kwargs.get('temperature', 0.1),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
            
        return response
        
    def evaluate(
        self,
        test_dataset: Optional[Dataset] = None,
        num_test_cases: int = 10,
        system: str = "double_integrator",
        visualize: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model using the integrated evaluation pipeline.
        
        Args:
            test_dataset: Test dataset (if None, creates new one)
            num_test_cases: Number of test cases to evaluate
            system: Control system to evaluate on
            visualize: Whether to create visualizations
            save_results: Whether to save results to disk
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            self.setup_model()
            
        # Create test dataset if not provided
        if test_dataset is None:
            test_dataset = create_dataset(num_samples=num_test_cases)
            
        # Prepare test cases and predictions
        test_cases = []
        predictions = []
        
        logger.info(f"Evaluating on {len(test_dataset)} test cases...")
        
        for i in range(min(num_test_cases, len(test_dataset))):
            sample = test_dataset[i]
            
            # Extract test case info
            metadata = sample.get('metadata', {})
            test_case = {
                'initial_state': metadata.get('initial_state', [0.0, 0.0]),
                'optimal_controls': metadata.get('control_sequence', []),
                'metadata': metadata
            }
            test_cases.append(test_case)
            
            # Generate prediction
            output = self.generate(sample['prompt'])
            
            # Parse controls
            controls = parse_control_output(output)
            if controls is None:
                logger.warning(f"Failed to parse controls for test case {i}")
                controls = [0.0] * metadata.get('steps', 50)
                
            predictions.append(controls)
            
            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Evaluated {i + 1}/{num_test_cases} test cases")
                
        # Run evaluation
        model_name = os.path.basename(self.model_name)
        results = self.eval_manager.evaluate_model(
            model_name=model_name,
            system=system,
            test_cases=test_cases,
            predictions=predictions,
            visualize=visualize,
            save_results=save_results,
            dt=metadata.get('dt', 0.1),
            steps=metadata.get('steps', 50)
        )
        
        # Generate report
        report = self.eval_manager.generate_report(model_name, system)
        logger.info("\n" + report)
        
        return results
        
    def evaluate_on_benchmark(
        self,
        benchmark_name: str = "standard",
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on a predefined benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            visualize: Whether to create visualizations
            
        Returns:
            Evaluation results
        """
        # Define benchmarks
        benchmarks = {
            "standard": {
                "num_samples": 20,
                "position_range": (-0.8, 0.8),
                "velocity_range": (-0.8, 0.8),
            },
            "easy": {
                "num_samples": 10,
                "position_range": (-0.5, 0.5),
                "velocity_range": (-0.5, 0.5),
            },
            "hard": {
                "num_samples": 30,
                "position_range": (-1.0, 1.0),
                "velocity_range": (-1.0, 1.0),
            }
        }
        
        if benchmark_name not in benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
        benchmark = benchmarks[benchmark_name]
        
        # Create benchmark dataset
        from control_datasets.systems.double_integrator_dataset import DoubleIntegratorDataset
        
        di_dataset = DoubleIntegratorDataset(
            num_samples=benchmark["num_samples"],
            dt=0.1,
            steps=50,
            position_range=benchmark["position_range"],
            velocity_range=benchmark["velocity_range"]
        )
        di_dataset.generate()
        
        # Convert to HF dataset
        from datasets import Dataset as HFDataset
        test_dataset = HFDataset.from_list(di_dataset.data)
        
        # Run evaluation
        results = self.evaluate(
            test_dataset=test_dataset,
            num_test_cases=benchmark["num_samples"],
            visualize=visualize
        )
        
        # Add benchmark info
        results['benchmark'] = benchmark_name
        results['benchmark_config'] = benchmark
        
        return results
        
    def save_model(self, output_dir: str):
        """Save the current model."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleaned up resources")


# Convenience function for backward compatibility
def create_trainer(model_name: str = MODEL_NAME, **kwargs) -> ControlTrainerV3:
    """Create a trainer instance."""
    return ControlTrainerV3(model_name, **kwargs)