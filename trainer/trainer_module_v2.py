"""
Control trainer v2 using modular training strategies.
"""

import torch
import gc
from unsloth import FastLanguageModel
from vllm import SamplingParams
from typing import Optional, Dict, Any
from datasets import Dataset

from config import *
from data import get_system_prompt
from trainers import TrainerFactory
from logger import logger


class ControlTrainerV2:
    """Enhanced trainer using modular training strategies."""
    
    def __init__(self, model_name: str = MODEL_NAME, use_modular: bool = True):
        """
        Initialize trainer.
        
        Args:
            model_name: Base model name
            use_modular: Whether to use new modular trainers
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.use_modular = use_modular
        
        # Modular trainers
        self.sft_trainer = None
        self.grpo_trainer = None
        
    def setup_model(self):
        """Load model and configure LoRA."""
        logger.info("Loading model and tokenizer...")
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
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=LORA_RANK*2,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )
        
        # Setup chat template
        system_prompt = get_system_prompt(DT, STEPS)
        # Set the chat template for Qwen3 models
        if "qwen" in self.model_name.lower():
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
        
    def _get_sft_config(self) -> Dict[str, Any]:
        """Get SFT configuration."""
        return {
            'batch_size': SFT_BATCH_SIZE,
            'gradient_accumulation_steps': 1,  # Default value
            'learning_rate': SFT_LEARNING_RATE,
            'lr_scheduler_type': LR_SCHEDULER,
            'epochs': SFT_EPOCHS,
            'max_seq_length': MAX_SEQ_LENGTH,
            'warmup_ratio': WARMUP_RATIO,
            'logging_steps': 5,
            'optimizer': OPTIMIZER,
            'weight_decay': WEIGHT_DECAY,
            'seed': SEED,
            'report_to': 'wandb',
            'output_dir': f"{OUTPUT_DIR}/sft",
        }
        
    def _get_grpo_config(self) -> Dict[str, Any]:
        """Get GRPO configuration."""
        return {
            'min_p': 0.1,
            'top_p': 1.0,
            'top_k': -1,
            'seed': SEED,
            'temperature': GRPO_TEMPERATURE,
            'learning_rate': GRPO_LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'warmup_ratio': 0.1,
            'lr_scheduler_type': LR_SCHEDULER,
            'optimizer': OPTIMIZER,
            'logging_steps': 1,
            'batch_size': GRPO_BATCH_SIZE,
            'gradient_accumulation_steps': 1,
            'num_generations': GRPO_NUM_GENERATIONS,
            'max_seq_length': MAX_SEQ_LENGTH,
            'max_steps': GRPO_MAX_STEPS,
            'save_steps': 500,
            'report_to': 'wandb',
            'output_dir': f"{OUTPUT_DIR}/grpo",
        }
        
    def train_sft(self, dataset: Dataset):
        """Run SFT training using modular trainer."""
        if self.use_modular:
            logger.info("Using modular SFT trainer...")
            
            # Create SFT trainer
            self.sft_trainer = TrainerFactory.create(
                'sft',
                model=self.model,
                tokenizer=self.tokenizer,
                config=self._get_sft_config()
            )
            
            # Train
            metrics = self.sft_trainer.train(dataset)
            logger.info(f"SFT metrics: {metrics}")
            
            # Cleanup
            self.sft_trainer.cleanup()
        else:
            # Fallback to original implementation
            from trainer_module import ControlTrainer
            original_trainer = ControlTrainer(self.model_name)
            original_trainer.model = self.model
            original_trainer.tokenizer = self.tokenizer
            original_trainer.train_sft(dataset)
            
    def train_grpo(self, dataset: Dataset):
        """Run GRPO training using modular trainer."""
        if self.use_modular:
            logger.info("Using modular GRPO trainer...")
            
            # Create GRPO trainer
            self.grpo_trainer = TrainerFactory.create(
                'grpo',
                model=self.model,
                tokenizer=self.tokenizer,
                config=self._get_grpo_config()
            )
            
            # Train
            metrics = self.grpo_trainer.train(dataset)
            logger.info(f"GRPO metrics: {metrics}")
            
            # Cleanup
            self.grpo_trainer.cleanup()
        else:
            # Fallback to original implementation
            from trainer_module import ControlTrainer
            original_trainer = ControlTrainer(self.model_name)
            original_trainer.model = self.model
            original_trainer.tokenizer = self.tokenizer
            original_trainer.train_grpo(dataset)
            
    def save_model(self, save_name: str = MODEL_SAVE_NAME):
        """Save the trained LoRA model."""
        logger.info(f"Saving model to {save_name}...")
        self.model.save_lora(save_name)
        
    def train(self, dataset: Dataset, do_sft: bool = True, do_grpo: bool = True):
        """Run complete training pipeline."""
        if self.model is None:
            self.setup_model()
            
        if do_sft:
            self.train_sft(dataset)
            
        if do_grpo:
            self.train_grpo(dataset)
            
        self.save_model()
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response for a given prompt."""
        messages = [
            {"role": "system", "content": get_system_prompt(DT, STEPS)},
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load LoRA if saved
        try:
            self.model.load_lora(MODEL_SAVE_NAME)
        except:
            logger.warning(f"LoRA model '{MODEL_SAVE_NAME}' not found, using base model")
            
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                use_cache=True,
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
            
        return response