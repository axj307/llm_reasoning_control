"""
Supervised Fine-Tuning (SFT) trainer implementation.
"""

from typing import Any, Dict, Optional
from datasets import Dataset
from trl import SFTTrainer as TRLSFTTrainer, SFTConfig
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_trainer_impl import BaseTrainerImpl
from logger import logger
from data import format_dataset_for_sft


class SFTTrainerModule(BaseTrainerImpl):
    """SFT trainer for supervised fine-tuning."""
    
    def __init__(self, model: Any, tokenizer: Any, config: Dict[str, Any]):
        """Initialize SFT trainer."""
        super().__init__(model, tokenizer, config)
        self.trainer = None
        self.sft_config = None
        
    def setup(self) -> None:
        """Setup SFT-specific configurations."""
        super().setup()
        
        # Extract SFT-specific config
        self.sft_config = SFTConfig(
            per_device_train_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            learning_rate=self.config.get('learning_rate', 2e-4),
            lr_scheduler_type=self.config.get('lr_scheduler_type', 'cosine'),
            num_train_epochs=self.config.get('epochs', 1),
            max_seq_length=self.config.get('max_seq_length', 2048),
            warmup_ratio=self.config.get('warmup_ratio', 0.1),
            logging_steps=self.config.get('logging_steps', 5),
            optim=self.config.get('optimizer', 'adamw_bnb_8bit'),
            weight_decay=self.config.get('weight_decay', 0.01),
            seed=self.config.get('seed', 3407),
            report_to=self.config.get('report_to', 'wandb'),
            output_dir=self.config.get('output_dir', 'outputs/sft'),
        )
        
    def train(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Train the model using SFT.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional training arguments
            
        Returns:
            Training metrics
        """
        if not self._is_initialized:
            self.setup()
            
        logger.info("Starting SFT training...")
        
        # Format dataset for SFT
        formatted_dataset = format_dataset_for_sft(dataset, self.tokenizer)
        
        # Create trainer
        self.trainer = TRLSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            args=self.sft_config,
            dataset_text_field="text",
            packing=False,
            max_seq_length=self.config.get('max_seq_length', 2048),
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Extract metrics
        metrics = {
            'loss': train_result.metrics.get('train_loss', 0),
            'learning_rate': train_result.metrics.get('train_learning_rate', 0),
            'epoch': train_result.metrics.get('epoch', 0),
            'total_steps': train_result.global_step,
        }
        
        logger.info(f"SFT training completed. Final loss: {metrics['loss']:.4f}")
        
        return metrics
        
    def validate(self, dataset: Dataset) -> Dict[str, Any]:
        """Validate the model."""
        if self.trainer is None:
            logger.warning("No trainer available for validation")
            return {}
            
        # Use trainer's evaluate method if available
        if hasattr(self.trainer, 'evaluate'):
            eval_result = self.trainer.evaluate(eval_dataset=dataset)
            return {
                'eval_loss': eval_result.get('eval_loss', 0),
                'perplexity': eval_result.get('perplexity', 0),
            }
        
        return {}
        
    def get_training_args(self) -> Dict[str, Any]:
        """Get current training arguments."""
        if self.sft_config:
            return self.sft_config.to_dict()
        return {}