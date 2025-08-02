"""
Enhanced trainer module with YAML configuration support.
"""

import torch
import gc
from typing import Optional, List, Dict, Any
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from vllm import SamplingParams
from data import get_system_prompt, format_dataset_for_sft
from rewards import get_all_reward_functions
from logger import logger

# Import old config for fallback
import config as old_config


class ControlTrainer:
    """Trainer combining SFT and GRPO for control tasks with configuration support."""
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with model.
        
        Args:
            model_name: Model name (overrides config if provided)
            config: Configuration dictionary
        """
        self.model = None
        self.tokenizer = None
        self.config = config or {}
        
        # Use provided model name or get from config
        self.model_name = model_name or self.config.get('model', {}).get('name', old_config.MODEL_NAME)
        
    def setup_model(
        self,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        max_seq_length: Optional[int] = None,
        **kwargs
    ):
        """Load model and configure LoRA with configuration support."""
        logger.info("Loading model and tokenizer...")
        
        # Get values from arguments, config, or fallback to old config
        lora_config = self.config.get('lora', {})
        model_config = self.config.get('model', {})
        
        lora_rank = lora_rank or lora_config.get('rank', old_config.LORA_RANK)
        lora_alpha = lora_alpha or lora_config.get('lora_alpha', lora_rank * 2)
        lora_dropout = lora_dropout or lora_config.get('lora_dropout', 0)
        target_modules = target_modules or lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        max_seq_length = max_seq_length or model_config.get('max_seq_length', old_config.MAX_SEQ_LENGTH)
        
        # GPU settings
        gpu_config = self.config.get('gpu', {})
        gpu_memory_utilization = gpu_config.get('memory_utilization', old_config.GPU_MEMORY_UTILIZATION)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=model_config.get('load_in_4bit', True),
            fast_inference=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        # Configure LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', "unsloth"),
            random_state=model_config.get('random_state', old_config.SEED),
            use_rslora=lora_config.get('use_rslora', False),
            use_dora=lora_config.get('use_dora', False),
        )
        
        # Setup chat template
        self._setup_chat_template()
        
    def _setup_chat_template(self):
        """Configure chat template for the tokenizer."""
        # Get formatting config
        formatting_config = self.config.get('formatting', {})
        
        # Get system prompt
        system_prompt = get_system_prompt(
            dt=self.config.get('control', {}).get('dt', old_config.DT),
            steps=self.config.get('control', {}).get('steps', old_config.STEPS)
        )
        
        # Get tokens
        reasoning_start = formatting_config.get('reasoning_start', old_config.REASONING_START)
        
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
        
        self.tokenizer.chat_template = chat_template
    
    def train_sft(
        self, 
        dataset,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Run SFT training with configuration support."""
        logger.info("Starting SFT training...")
        
        # Get training config
        training_config = self.config.get('training', {})
        sft_config = self.config.get('sft_training', training_config)
        
        # Format dataset
        formatted_dataset = format_dataset_for_sft(dataset, self.tokenizer)
        
        # Create trainer
        sft_trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size or sft_config.get('per_device_train_batch_size', old_config.SFT_BATCH_SIZE),
                gradient_accumulation_steps=sft_config.get('gradient_accumulation_steps', 1),
                warmup_steps=sft_config.get('warmup_steps', old_config.SFT_WARMUP_STEPS),
                num_train_epochs=epochs or sft_config.get('num_train_epochs', old_config.SFT_EPOCHS),
                learning_rate=learning_rate or sft_config.get('learning_rate', old_config.SFT_LEARNING_RATE),
                logging_steps=sft_config.get('logging_steps', 5),
                bf16=sft_config.get('bf16', False),
                fp16=sft_config.get('fp16', True),
                weight_decay=sft_config.get('weight_decay', old_config.WEIGHT_DECAY),
                lr_scheduler_type=sft_config.get('lr_scheduler_type', old_config.LR_SCHEDULER),
                seed=sft_config.get('seed', old_config.SEED),
                optim=sft_config.get('optim', old_config.OPTIMIZER),
                output_dir=output_dir or sft_config.get('output_dir', f"{old_config.OUTPUT_DIR}/sft"),
                save_strategy=sft_config.get('save_strategy', "no"),
                save_steps=sft_config.get('save_steps', 100),
            ),
        )
        
        # Train
        sft_trainer.train()
        
        # Clean up
        del sft_trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SFT training complete!")
        
    def train_grpo(
        self,
        dataset,
        max_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_generations: Optional[int] = None,
        temperature: Optional[float] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Run GRPO training with configuration support."""
        logger.info("Starting GRPO training...")
        
        # Get training config
        training_config = self.config.get('training', {})
        grpo_config = self.config.get('grpo_training', self.config.get('grpo', training_config))
        
        # Create prompts
        prompts = [sample["text"] for sample in dataset]
        
        # Create reward functions
        reward_fns = get_all_reward_functions()
        
        # Define formatting wrapper
        formatting_config = self.config.get('formatting', {})
        reasoning_start = formatting_config.get('reasoning_start', old_config.REASONING_START)
        reasoning_end = formatting_config.get('reasoning_end', old_config.REASONING_END)
        solution_start = formatting_config.get('solution_start', old_config.SOLUTION_START)
        solution_end = formatting_config.get('solution_end', old_config.SOLUTION_END)
        
        def create_format_wrapper(func, tokenizer):
            def wrapper(completions, **kwargs):
                formatted = []
                for completion in completions:
                    if reasoning_start in completion and reasoning_end in completion:
                        start = completion.index(reasoning_start) + len(reasoning_start)
                        end = completion.index(reasoning_end)
                        reasoning = completion[start:end].strip()
                    else:
                        reasoning = "No reasoning found."
                    
                    if solution_start in completion and solution_end in completion:
                        start = completion.index(solution_start) + len(solution_start)
                        end = completion.index(solution_end)
                        solution = completion[start:end].strip()
                    else:
                        solution = "No solution found."
                    
                    final_output = f"{reasoning_start}\n{reasoning}\n{reasoning_end}\n{solution_start}\n{solution}\n{solution_end}"
                    formatted.append(final_output)
                    
                return func(formatted, **kwargs)
            return wrapper
            
        # Wrap reward functions
        wrapped_reward_fns = [create_format_wrapper(fn, self.tokenizer) for fn in reward_fns]
        
        # Create GRPO trainer
        grpo_trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=GRPOConfig(
                num_generations=num_generations or grpo_config.get('num_generations', old_config.GRPO_NUM_GENERATIONS),
                logging_steps=grpo_config.get('logging_steps', 1),
                max_steps=max_steps or grpo_config.get('max_steps', old_config.GRPO_MAX_STEPS),
                learning_rate=learning_rate or grpo_config.get('learning_rate', old_config.GRPO_LEARNING_RATE),
                per_device_train_batch_size=batch_size or grpo_config.get('per_device_train_batch_size', old_config.GRPO_BATCH_SIZE),
                temperature=temperature or grpo_config.get('temperature', old_config.GRPO_TEMPERATURE),
                save_strategy=grpo_config.get('save_strategy', "no"),
                save_steps=grpo_config.get('save_steps', 10),
                output_dir=output_dir or grpo_config.get('output_dir', f"{old_config.OUTPUT_DIR}/grpo"),
                seed=grpo_config.get('seed', old_config.SEED),
                bf16=grpo_config.get('bf16', False),
                fp16=grpo_config.get('fp16', True),
            ),
            reward_funcs=wrapped_reward_fns,
            prompt_dataset=prompts,
        )
        
        # Train
        grpo_trainer.train()
        
        # Clean up
        del grpo_trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GRPO training complete!")
        
    def save_model(self, save_path: Optional[str] = None):
        """Save model to disk."""
        save_path = save_path or f"{self.config.get('training', {}).get('output_dir', old_config.OUTPUT_DIR)}/{old_config.MODEL_SAVE_NAME}"
        logger.info(f"Saving model to {save_path}...")
        self.model.save_pretrained_merged(save_path, self.tokenizer, save_method="lora")
        logger.info("Model saved!")
        
    def train(
        self,
        dataset,
        do_sft: bool = True,
        do_grpo: bool = True,
        sft_epochs: Optional[int] = None,
        sft_batch_size: Optional[int] = None,
        sft_learning_rate: Optional[float] = None,
        grpo_max_steps: Optional[int] = None,
        grpo_batch_size: Optional[int] = None,
        grpo_learning_rate: Optional[float] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """Run full training pipeline."""
        if do_sft:
            sft_output = f"{output_dir}/sft" if output_dir else None
            self.train_sft(
                dataset,
                epochs=sft_epochs,
                batch_size=sft_batch_size,
                learning_rate=sft_learning_rate,
                output_dir=sft_output
            )
            
        if do_grpo:
            grpo_output = f"{output_dir}/grpo" if output_dir else None
            self.train_grpo(
                dataset,
                max_steps=grpo_max_steps,
                batch_size=grpo_batch_size,
                learning_rate=grpo_learning_rate,
                output_dir=grpo_output
            )
            
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate a response to a prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        # Use tokenizer to format prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate with VLLM
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=[self.tokenizer.eos_token]
        )
        
        output = self.model.generate(formatted_prompt, params)[0]
        
        # Get the generated text
        response = output.outputs[0].text
        
        return response