"""
Pydantic schemas for configuration validation.
Provides type safety and validation for all configuration options.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str = Field(description="Model name or path")
    max_seq_length: int = Field(ge=512, le=32768, description="Maximum sequence length")
    dtype: Optional[str] = Field(None, description="Data type (float16, bfloat16, float32)")
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization")
    use_gradient_checkpointing: Union[bool, str] = Field("unsloth", description="Gradient checkpointing mode")
    random_state: int = Field(3407, description="Random seed for model initialization")
    use_rslora: bool = Field(False, description="Use Rank-Stabilized LoRA")
    loftq_config: Optional[Dict[str, Any]] = Field(None, description="LoftQ configuration")
    
    @validator('max_seq_length')
    def validate_seq_length(cls, v):
        if v < 2048:
            raise ValueError("max_seq_length must be at least 2048")
        return v
    
    @validator('dtype')
    def validate_dtype(cls, v):
        if v is not None and v not in ['float16', 'bfloat16', 'float32', 'auto', None]:
            raise ValueError(f"Invalid dtype: {v}")
        return v


class LoRAConfig(BaseModel):
    """LoRA configuration schema."""
    rank: int = Field(32, ge=1, le=512, description="LoRA rank")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Target modules for LoRA"
    )
    lora_alpha: int = Field(16, ge=1, description="LoRA alpha parameter")
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0, description="LoRA dropout rate")
    bias: str = Field("none", description="Bias configuration")
    use_gradient_checkpointing: bool = Field(False, description="Use gradient checkpointing in LoRA")
    use_rslora: bool = Field(False, description="Use Rank-Stabilized LoRA")
    use_dora: bool = Field(False, description="Use DoRA (Weight-Decomposed Low-Rank Adaptation)")
    loftq_config: Optional[Dict[str, Any]] = Field(None, description="LoftQ configuration")
    
    @validator('bias')
    def validate_bias(cls, v):
        if v not in ['none', 'all', 'lora_only']:
            raise ValueError(f"Invalid bias configuration: {v}")
        return v


class TrainingConfig(BaseModel):
    """Training configuration schema."""
    seed: int = Field(3407, description="Random seed")
    output_dir: str = Field("outputs", description="Output directory")
    num_train_epochs: int = Field(1, ge=1, description="Number of training epochs")
    max_steps: int = Field(-1, description="Maximum training steps (-1 for no limit)")
    per_device_train_batch_size: int = Field(4, ge=1, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(4, ge=1, description="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Gradient accumulation steps")
    learning_rate: float = Field(2e-4, gt=0, le=1, description="Learning rate")
    warmup_steps: int = Field(5, ge=0, description="Warmup steps")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")
    lr_scheduler_type: str = Field("linear", description="Learning rate scheduler type")
    optim: str = Field("adamw_8bit", description="Optimizer")
    logging_steps: int = Field(1, ge=1, description="Logging frequency")
    save_strategy: str = Field("steps", description="Save strategy")
    save_steps: int = Field(100, ge=1, description="Save frequency")
    evaluation_strategy: str = Field("no", description="Evaluation strategy")
    eval_steps: Optional[int] = Field(None, description="Evaluation frequency")
    save_total_limit: Optional[int] = Field(None, description="Maximum number of checkpoints")
    load_best_model_at_end: bool = Field(False, description="Load best model at end")
    metric_for_best_model: Optional[str] = Field(None, description="Metric for best model selection")
    greater_is_better: Optional[bool] = Field(None, description="Whether higher metric is better")
    fp16: bool = Field(True, description="Use FP16 training")
    bf16: bool = Field(False, description="Use BF16 training")
    tf32: Optional[bool] = Field(None, description="Use TF32 on Ampere GPUs")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")
    group_by_length: bool = Field(True, description="Group sequences by length")
    dataloader_num_workers: int = Field(0, ge=0, description="Number of dataloader workers")
    remove_unused_columns: bool = Field(False, description="Remove unused columns")
    hub_model_id: Optional[str] = Field(None, description="Hugging Face Hub model ID")
    hub_strategy: str = Field("every_save", description="Hub push strategy")
    hub_token: Optional[str] = Field(None, description="Hugging Face Hub token")
    hub_private_repo: bool = Field(False, description="Make Hub repo private")
    push_to_hub: bool = Field(False, description="Push to Hugging Face Hub")
    resume_from_checkpoint: Optional[Union[bool, str]] = Field(None, description="Resume from checkpoint")
    
    @validator('lr_scheduler_type')
    def validate_scheduler(cls, v):
        valid_schedulers = ['linear', 'cosine', 'constant', 'constant_with_warmup', 'polynomial']
        if v not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {v}. Must be one of {valid_schedulers}")
        return v
    
    @validator('save_strategy')
    def validate_save_strategy(cls, v):
        if v not in ['no', 'epoch', 'steps']:
            raise ValueError(f"Invalid save strategy: {v}")
        return v
    
    @validator('evaluation_strategy')
    def validate_eval_strategy(cls, v):
        if v not in ['no', 'epoch', 'steps']:
            raise ValueError(f"Invalid evaluation strategy: {v}")
        return v


class ControlConfig(BaseModel):
    """Control system configuration schema."""
    dt: float = Field(0.1, gt=0, description="Time step")
    steps: int = Field(50, ge=1, description="Number of control steps")
    control_bounds: List[float] = Field([-3.0, 3.0], min_items=2, max_items=2, description="Control bounds [min, max]")
    state_bounds: Union[List[float], Dict[str, List[float]]] = Field(
        default_factory=lambda: {"position": [-1.0, 1.0], "velocity": [-1.0, 1.0]},
        description="State bounds"
    )
    target_state: List[float] = Field([0.0, 0.0], description="Target state")
    
    @validator('control_bounds')
    def validate_bounds(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Control bounds: min must be less than max")
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""
    num_samples: int = Field(500, ge=1, description="Number of training samples")
    train_split: float = Field(0.9, gt=0, lt=1, description="Training data split ratio")
    val_split: float = Field(0.1, gt=0, lt=1, description="Validation data split ratio")
    cache_dir: Optional[str] = Field("./cache", description="Cache directory")
    num_proc: int = Field(4, ge=1, description="Number of processes for data loading")
    initial_state_sampling: str = Field("uniform", description="Initial state sampling method")
    initial_state_ranges: Optional[Dict[str, List[float]]] = Field(None, description="Initial state ranges")
    noise_level: float = Field(0.0, ge=0, description="Noise level for data generation")
    
    @validator('train_split', 'val_split')
    def validate_splits(cls, v, values):
        if 'train_split' in values:
            total = values['train_split'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError("Train and validation splits must sum to 1.0")
        return v


class FormattingConfig(BaseModel):
    """Text formatting configuration schema."""
    reasoning_start: str = Field("<REASONING>", description="Start token for reasoning")
    reasoning_end: str = Field("</REASONING>", description="End token for reasoning")
    solution_start: str = Field("<CONTROLS>", description="Start token for solution")
    solution_end: str = Field("</CONTROLS>", description="End token for solution")
    system_prompt_template: str = Field(
        default="You are an expert in control theory.",
        description="System prompt template"
    )


class GPUConfig(BaseModel):
    """GPU configuration schema."""
    memory_utilization: float = Field(0.7, gt=0, le=1, description="GPU memory utilization")
    device_map: Union[str, Dict[str, Any]] = Field("auto", description="Device mapping strategy")
    compile_model: bool = Field(False, description="Use torch.compile")
    use_flash_attention: bool = Field(False, description="Use Flash Attention")
    use_liger_kernel: bool = Field(False, description="Use Liger kernel optimizations")


class LoggingConfig(BaseModel):
    """Logging configuration schema."""
    level: str = Field("INFO", description="Logging level")
    wandb_project: Optional[str] = Field("llm-control", description="Weights & Biases project")
    wandb_enabled: bool = Field(False, description="Enable W&B logging")
    log_to_file: bool = Field(True, description="Log to file")
    log_file: str = Field("training.log", description="Log file path")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}")
        return v.upper()


class EvaluationConfig(BaseModel):
    """Evaluation configuration schema."""
    num_test_cases: int = Field(10, ge=1, description="Number of test cases")
    test_initial_conditions: List[List[float]] = Field(
        default_factory=lambda: [[0.5, -0.3], [0.7, 0.2], [-0.6, -0.4]],
        description="Test initial conditions"
    )
    metrics: List[str] = Field(
        default_factory=lambda: ["final_state_error", "trajectory_smoothness", "control_effort"],
        description="Evaluation metrics"
    )
    visualize: bool = Field(True, description="Generate visualizations")
    save_plots: bool = Field(True, description="Save plot files")
    plot_dir: str = Field("plots", description="Plot directory")
    plot_formats: List[str] = Field(["png"], description="Plot file formats")
    
    @validator('plot_formats')
    def validate_formats(cls, v):
        valid_formats = ['png', 'pdf', 'svg', 'eps']
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid plot format: {fmt}")
        return v


class SystemConfig(BaseModel):
    """System-specific configuration schema."""
    name: str = Field(description="System name")
    description: str = Field("", description="System description")
    state_dim: int = Field(ge=1, description="State dimension")
    action_dim: int = Field(ge=1, description="Action dimension")


class GRPOConfig(TrainingConfig):
    """GRPO-specific training configuration."""
    num_generations: int = Field(4, ge=1, description="Number of generations per sample")
    temperature: float = Field(1.0, gt=0, description="Sampling temperature")
    do_sample: bool = Field(True, description="Use sampling")
    top_p: float = Field(0.9, gt=0, le=1, description="Top-p sampling")
    top_k: int = Field(50, ge=1, description="Top-k sampling")
    max_new_tokens: int = Field(512, ge=1, description="Maximum new tokens")
    pad_token_id: int = Field(0, description="Padding token ID")


class Config(BaseModel):
    """Complete configuration schema."""
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    control: ControlConfig
    dataset: DatasetConfig
    formatting: FormattingConfig
    gpu: GPUConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    system: Optional[SystemConfig] = None
    
    # Training strategy specific configs
    sft_training: Optional[TrainingConfig] = None
    grpo_training: Optional[GRPOConfig] = None
    grpo: Optional[GRPOConfig] = None  # Alias for grpo_training
    
    # Additional configs
    defaults: Optional[List[Union[str, Dict[str, str]]]] = None
    checkpoint: Optional[Dict[str, Any]] = None
    advanced: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields for flexibility