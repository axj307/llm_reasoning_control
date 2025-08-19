"""Universal model manager for saving and loading models."""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from unsloth import FastLanguageModel

# Import GPU utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gpu_utils import auto_gpu_config, set_gpu_device


class ModelManager:
    """Base model manager for handling model checkpoints."""
    
    def __init__(self, base_model_name: str = "unsloth/Qwen3-4B-Base"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.max_seq_length = None
        self.lora_rank = None
        self.training_config = {}
    
    def setup_model(self, max_seq_length: int = 1024, lora_rank: int = 8,
                   load_in_4bit: bool = True, gpu_memory_utilization: float = 0.4,
                   gpu_id: Optional[int] = None, auto_select_gpu: bool = True,
                   fast_inference: bool = False, working_notebook_mode: bool = False):
        """Set up the base model with LoRA and automatic GPU selection."""
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        
        # Working notebook mode uses different parameters
        if working_notebook_mode:
            max_seq_length = 2048  # Higher seq length
            lora_rank = 32  # Higher LoRA rank
            gpu_memory_utilization = 0.7  # Higher GPU utilization
            fast_inference = True  # Enable vLLM
            print("🚀 Using working notebook mode (vLLM enabled, higher parameters)")
        
        # Auto-select GPU if not specified
        if auto_select_gpu and gpu_id is None:
            print("🎯 Auto-selecting best available GPU...")
            gpu_config = auto_gpu_config()
            gpu_id = gpu_config['gpu_id']
        elif gpu_id is not None:
            print(f"📌 Using specified GPU {gpu_id}")
            set_gpu_device(gpu_id, auto_select=False)
        else:
            print("⚠️  No GPU selection, using default")
        
        print(f"🚀 Loading model: {self.base_model_name}")
        print(f"   Max sequence length: {max_seq_length}")
        print(f"   LoRA rank: {lora_rank}")
        print(f"   GPU memory utilization: {gpu_memory_utilization}")
        
        # Load base model with configuration matching working notebook
        model_kwargs = {
            "model_name": self.base_model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "fast_inference": fast_inference,  # Use parameter from working notebook
            "max_lora_rank": lora_rank,
        }
        
        # Add gpu_memory_utilization only if fast_inference is True (vLLM mode)
        if fast_inference:
            model_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            print(f"   Fast inference: {fast_inference} (vLLM enabled)")
        else:
            print(f"   Fast inference: {fast_inference} (standard mode)")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        
        print("🔧 Applying LoRA configuration...")
        
        # Apply LoRA with configuration matching working notebook
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_rank * 2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print("✅ Model setup completed successfully!")
        
        return self.model, self.tokenizer
    
    def setup_chat_template(self, reasoning_start: str = "<REASONING>",
                          reasoning_end: str = "</REASONING>",
                          solution_start: str = "<CONTROLS>",
                          solution_end: str = "</CONTROLS>",
                          system_prompt: str = ""):
        """Set up chat template using EXACT notebook approach."""
        # EXACT notebook chat template
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
        
        # Replace with specific template (EXACT notebook approach)
        chat_template = chat_template\
            .replace("'{system_prompt}'",   f"'{system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")
        
        self.tokenizer.chat_template = chat_template
        print(f"✅ Chat template set up using notebook approach")
    
    def _setup_default_chat_template(self):
        """Set up default chat template for loaded models."""
        reasoning_start = "<REASONING>"
        simple_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] + eos_token }}"
            "{% elif message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
            "{% endif %}"
        )
        
        self.tokenizer.chat_template = simple_template
        print(f"✅ Default chat template configured for loaded model")
    
    def save_checkpoint(self, save_dir: str, metadata: Optional[Dict] = None):
        """Save model checkpoint using notebook approach with save_lora()."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights using Unsloth's save_lora method (EXACT notebook approach)
        self.model.save_lora(str(save_path))
        
        # Prepare metadata
        default_metadata = {
            "base_model": self.base_model_name,
            "timestamp": datetime.now().isoformat(),
            "lora_rank": self.lora_rank,
            "max_seq_length": self.max_seq_length,
            "training_config": self.training_config,
            "saved_with": "save_lora"  # Mark how this model was saved
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(default_metadata, f, indent=2)
        
        print(f"Model saved to {save_dir} using notebook approach (save_lora)")
        return save_dir
    
    def load_checkpoint(self, load_dir: str):
        """Load a saved model checkpoint using notebook approach with load_lora()."""
        load_path = Path(load_dir)
        
        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Setup base model if not already done (using exact notebook parameters)
        if self.model is None:
            self.setup_model(
                max_seq_length=metadata.get('max_seq_length', 2048),
                lora_rank=metadata.get('lora_rank', 32),  # Default to 32 like notebook
                fast_inference=True,  # Enable vLLM for inference (notebook approach)
                working_notebook_mode=True  # Use notebook-compatible parameters
            )
        
        # Load using notebook approach: model.load_lora() returns lora_request
        if (load_path / "adapter_config.json").exists():
            print(f"Loading model using notebook approach (load_lora) from {load_dir}")
            lora_request = self.model.load_lora(str(load_path))
            
            # Set up chat template (CRITICAL for proper inference)
            self._setup_default_chat_template()
            
            return self.model, self.tokenizer, lora_request, metadata
        else:
            raise FileNotFoundError(f"No adapter_config.json found in {load_path}")


class UniversalModelManager(ModelManager):
    """Model manager for universal multi-system models."""
    
    def __init__(self, base_model_name: str = "unsloth/Qwen3-4B-Base"):
        super().__init__(base_model_name)
        self.trained_systems = []
    
    def save_universal_checkpoint(self, systems_list: List[str], 
                                training_type: str = "sft",
                                run_name: Optional[str] = None,
                                metrics: Optional[Dict] = None):
        """
        Save universal model checkpoint with system information.
        
        Args:
            systems_list: List of system names this model was trained on
            training_type: 'sft' or 'grpo'
            run_name: Optional run name, auto-generates if None
            metrics: Optional training metrics to save
        """
        # Auto-generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            systems_str = '_'.join(sorted(systems_list))
            run_name = f"v{timestamp}_{systems_str}"
        
        # Construct save directory
        save_dir = f"models/universal/{training_type}/{run_name}"
        
        # Import environments to get their info
        from environments import get_system
        
        # Prepare metadata
        metadata = {
            "model_type": "universal",
            "training_type": training_type,
            "trained_systems": systems_list,
            "num_systems": len(systems_list),
            "run_name": run_name,
            "systems_info": {
                system: get_system(system)().get_info() 
                for system in systems_list
            }
        }
        
        if metrics:
            metadata["metrics"] = metrics
        
        # Save using parent method
        save_path = self.save_checkpoint(save_dir, metadata)
        
        # Update 'latest' symlink
        self._update_latest_symlink(save_dir, training_type)
        
        self.trained_systems = systems_list
        return save_path
    
    def save_single_system_checkpoint(self, system_name: str,
                                    training_type: str = "sft",
                                    run_name: Optional[str] = None,
                                    metrics: Optional[Dict] = None):
        """Save a single-system specialist model."""
        # Auto-generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"run_{timestamp}"
        
        # Construct save directory
        save_dir = f"models/single_system/{system_name}/{training_type}/{run_name}"
        
        # Import environments to get info
        from environments import get_system
        
        # Prepare metadata
        metadata = {
            "model_type": "single_system",
            "training_type": training_type,
            "system": system_name,
            "run_name": run_name,
            "system_info": get_system(system_name)().get_info()
        }
        
        if metrics:
            metadata["metrics"] = metrics
        
        # Save using parent method
        save_path = self.save_checkpoint(save_dir, metadata)
        
        # Update 'latest' symlink
        latest_dir = f"models/single_system/{system_name}/{training_type}"
        self._update_latest_symlink(save_dir, None, latest_dir)
        
        self.trained_systems = [system_name]
        return save_path
    
    def load_universal_model(self, run_name: str = "latest", 
                           training_type: str = "grpo"):
        """Load a universal model and check its capabilities."""
        if run_name == "latest":
            load_dir = f"models/universal/{training_type}/latest"
        else:
            load_dir = f"models/universal/{training_type}/{run_name}"
        
        model, tokenizer, lora_request, metadata = self.load_checkpoint(load_dir)
        
        # Extract system information
        self.trained_systems = metadata.get('trained_systems', [])
        
        print(f"Loaded universal model trained on: {', '.join(self.trained_systems)}")
        return model, tokenizer, lora_request, metadata
    
    def load_single_system_model(self, system_name: str,
                               run_name: str = "latest",
                               training_type: str = "grpo"):
        """Load a single-system specialist model."""
        if run_name == "latest":
            load_dir = f"models/single_system/{system_name}/{training_type}/latest"
        else:
            load_dir = f"models/single_system/{system_name}/{training_type}/{run_name}"
        
        model, tokenizer, lora_request, metadata = self.load_checkpoint(load_dir)
        
        self.trained_systems = [system_name]
        
        print(f"Loaded {system_name} specialist model")
        return model, tokenizer, lora_request, metadata
    
    def list_saved_models(self) -> Dict[str, Dict]:
        """List all saved models organized by type."""
        saved_models = {
            "universal": {"sft": [], "grpo": []},
            "single_system": {}
        }
        
        # List universal models
        for training_type in ["sft", "grpo"]:
            path = Path(f"models/universal/{training_type}")
            if path.exists():
                runs = [d.name for d in path.iterdir() 
                       if d.is_dir() and d.name != "latest"]
                saved_models["universal"][training_type] = sorted(runs)
        
        # List single-system models
        single_system_path = Path("models/single_system")
        if single_system_path.exists():
            for system_dir in single_system_path.iterdir():
                if system_dir.is_dir():
                    system_name = system_dir.name
                    saved_models["single_system"][system_name] = {
                        "sft": [], "grpo": []
                    }
                    
                    for training_type in ["sft", "grpo"]:
                        path = system_dir / training_type
                        if path.exists():
                            runs = [d.name for d in path.iterdir() 
                                   if d.is_dir() and d.name != "latest"]
                            saved_models["single_system"][system_name][training_type] = sorted(runs)
        
        return saved_models
    
    def _update_latest_symlink(self, target_dir: str, training_type: Optional[str] = None,
                             base_dir: Optional[str] = None):
        """Update the 'latest' symlink to point to the most recent run."""
        target_path = Path(target_dir)
        
        if base_dir:
            latest_link = Path(base_dir) / "latest"
        else:
            latest_link = target_path.parent / "latest"
        
        # Remove existing symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        # Create new symlink
        latest_link.symlink_to(target_path.name)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get information about a saved model without loading it."""
        metadata_path = Path(model_path) / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {model_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata


def load_model_from_path(model_path: str, base_model_name: str = "unsloth/Qwen3-4B-Base",
                        max_seq_length: int = 2048, lora_rank: int = 32):
    """
    Load a model directly from a path (for working notebook models).
    
    Args:
        model_path: Path to the model directory
        base_model_name: Base model to load
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank
        
    Returns:
        Tuple of (model, tokenizer, lora_request)
    """
    from unsloth import FastLanguageModel
    from peft import PeftModel
    
    # Load base model
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # Standard mode for evaluation
        max_lora_rank=lora_rank,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, tokenizer, None  # No lora_request for PeftModel