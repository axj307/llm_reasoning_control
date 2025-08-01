"""
Simple combined trainer for SFT and GRPO training.
"""

import torch
import gc
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from config import *
from data import get_system_prompt, format_dataset_for_sft
from rewards import get_all_reward_functions
from logger import logger


class ControlTrainer:
    """Trainer combining SFT and GRPO for control tasks."""
    
    def __init__(self, model_name=MODEL_NAME):
        """Initialize trainer with model."""
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
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
        self._setup_chat_template()
        
    def _setup_chat_template(self):
        """Configure chat template for the tokenizer."""
        system_prompt = get_system_prompt()
        
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
            .replace("'{reasoning_start}'", f"'{REASONING_START}'")
        
        self.tokenizer.chat_template = chat_template
    
    def train_sft(self, dataset):
        """Run SFT training."""
        logger.info("Starting SFT training...")
        
        # Format dataset
        formatted_dataset = format_dataset_for_sft(dataset, self.tokenizer)
        
        # Create trainer
        sft_trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=SFT_BATCH_SIZE,
                gradient_accumulation_steps=1,
                warmup_steps=SFT_WARMUP_STEPS,
                num_train_epochs=SFT_EPOCHS,
                learning_rate=SFT_LEARNING_RATE,
                logging_steps=5,
                optim=OPTIMIZER,
                weight_decay=WEIGHT_DECAY,
                lr_scheduler_type=LR_SCHEDULER,
                seed=SEED,
                report_to="wandb",
                output_dir=f"{OUTPUT_DIR}/sft",
            ),
        )
        
        # Train
        sft_trainer.train()
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
    def train_grpo(self, dataset):
        """Run GRPO training."""
        logger.info("Starting GRPO training...")
        
        # VLLM sampling parameters
        vllm_sampling_params = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=SEED,
            stop=[self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        
        # GRPO config
        grpo_args = GRPOConfig(
            vllm_sampling_params=vllm_sampling_params,
            temperature=GRPO_TEMPERATURE,
            learning_rate=GRPO_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=0.1,
            lr_scheduler_type=LR_SCHEDULER,
            optim=OPTIMIZER,
            logging_steps=1,
            per_device_train_batch_size=GRPO_BATCH_SIZE,
            gradient_accumulation_steps=1,
            num_generations=GRPO_NUM_GENERATIONS,
            max_completion_length=MAX_SEQ_LENGTH,
            max_steps=GRPO_MAX_STEPS,
            save_steps=500,
            report_to="wandb",
            output_dir=f"{OUTPUT_DIR}/grpo",
        )
        
        # Create wrapper functions to handle tokenizer injection
        def create_format_wrapper(func, tokenizer):
            def wrapper(completions, **kwargs):
                return func(completions, tokenizer=tokenizer, **kwargs)
            return wrapper
        
        # Get reward functions with proper wrapping
        reward_funcs = []
        for func in get_all_reward_functions():
            if func.__name__ == "match_format_exactly":
                # Add tokenizer for format matching function
                reward_funcs.append(create_format_wrapper(func, self.tokenizer))
            else:
                # Other functions don't need tokenizer injection
                reward_funcs.append(func)
        
        # Create trainer
        grpo_trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=dataset,
        )
        
        # Train
        grpo_trainer.train()
        
    def save_model(self, save_name=MODEL_SAVE_NAME):
        """Save the trained LoRA model."""
        logger.info(f"Saving model to {save_name}...")
        self.model.save_lora(save_name)
        
    def train(self, dataset, do_sft=True, do_grpo=True):
        """Run complete training pipeline."""
        if self.model is None:
            self.setup_model()
            
        if do_sft:
            self.train_sft(dataset)
            
        if do_grpo:
            self.train_grpo(dataset)
            
        self.save_model()
        
    def generate(self, prompt, temperature=0.7, max_tokens=1024):
        """Generate a response for a given prompt."""
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=max_tokens,
        )
        
        # Try to load LoRA if it exists, otherwise use base model
        try:
            lora_request = self.model.load_lora(MODEL_SAVE_NAME)
        except:
            logger.warning(f"LoRA model '{MODEL_SAVE_NAME}' not found, using base model")
            lora_request = None
            
        output = self.model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
        
        return output