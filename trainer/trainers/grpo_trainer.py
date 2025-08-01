"""
Group Relative Policy Optimization (GRPO) trainer implementation.
"""

from typing import Any, Dict, List, Callable, Optional
import datasets as hf_datasets
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
from vllm import SamplingParams
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_trainer_impl import BaseTrainerImpl
from logger import logger
from rewards import get_all_reward_functions


class GRPOTrainerModule(BaseTrainerImpl):
    """GRPO trainer for reinforcement learning from feedback."""
    
    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any]):
        """Initialize GRPO trainer."""
        super().__init__(model, tokenizer, config)
        self.trainer = None
        self.grpo_config = None
        self.reward_functions = []
        
    def setup(self) -> None:
        """Setup GRPO-specific configurations."""
        super().setup()
        
        # VLLM sampling parameters
        vllm_sampling_params = SamplingParams(
            min_p=self.config.get('min_p', 0.1),
            top_p=self.config.get('top_p', 1.0),
            top_k=self.config.get('top_k', -1),
            seed=self.config.get('seed', 3407),
            stop=[self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        
        # GRPO config
        self.grpo_config = GRPOConfig(
            vllm_sampling_params=vllm_sampling_params,
            temperature=self.config.get('temperature', 0.7),
            learning_rate=self.config.get('learning_rate', 5e-7),
            weight_decay=self.config.get('weight_decay', 0.01),
            warmup_ratio=self.config.get('warmup_ratio', 0.1),
            lr_scheduler_type=self.config.get('lr_scheduler_type', 'cosine'),
            optim=self.config.get('optimizer', 'adamw_bnb_8bit'),
            logging_steps=self.config.get('logging_steps', 1),
            per_device_train_batch_size=self.config.get('batch_size', 2),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            num_generations=self.config.get('num_generations', 4),
            max_completion_length=self.config.get('max_seq_length', 2048),
            max_steps=self.config.get('max_steps', 100),
            save_steps=self.config.get('save_steps', 500),
            report_to=self.config.get('report_to', 'wandb'),
            output_dir=self.config.get('output_dir', 'outputs/grpo'),
        )
        
        # Setup reward functions
        self._setup_reward_functions()
        
    def _setup_reward_functions(self) -> None:
        """Setup reward functions with proper wrapping."""
        # Create wrapper for functions that need tokenizer
        def create_format_wrapper(func, tokenizer):
            def wrapper(completions, **kwargs):
                return func(completions, tokenizer=tokenizer, **kwargs)
            return wrapper
        
        # Get all reward functions
        self.reward_functions = []
        for func in get_all_reward_functions():
            if func.__name__ == "match_format_exactly":
                # Add tokenizer for format matching function
                self.reward_functions.append(create_format_wrapper(func, self.tokenizer))
            else:
                # Other functions don't need tokenizer injection
                self.reward_functions.append(func)
                
        logger.info(f"Configured {len(self.reward_functions)} reward functions for GRPO")
        
    def set_reward_functions(self, reward_functions: List[Callable]) -> None:
        """Set custom reward functions."""
        self.reward_functions = reward_functions
        logger.info(f"Set {len(reward_functions)} custom reward functions")
        
    def train(self, dataset: hf_datasets.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Train the model using GRPO.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional training arguments
            
        Returns:
            Training metrics
        """
        if not self._is_initialized:
            self.setup()
            
        logger.info("Starting GRPO training...")
        
        if not self.reward_functions:
            raise ValueError("No reward functions configured for GRPO training")
        
        # Create trainer
        self.trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_functions,
            args=self.grpo_config,
            train_dataset=dataset,
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Extract metrics
        metrics = {
            'loss': train_result.metrics.get('train_loss', 0),
            'rewards': train_result.metrics.get('train_rewards', 0),
            'learning_rate': train_result.metrics.get('train_learning_rate', 0),
            'total_steps': train_result.global_step,
        }
        
        logger.info(f"GRPO training completed. Final reward: {metrics['rewards']:.4f}")
        
        return metrics
        
    def get_training_args(self) -> Dict[str, Any]:
        """Get current training arguments."""
        if self.grpo_config:
            return self.grpo_config.to_dict()
        return {}
        
    def get_reward_info(self) -> List[str]:
        """Get information about configured reward functions."""
        return [func.__name__ for func in self.reward_functions]