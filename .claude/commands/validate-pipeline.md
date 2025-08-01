# Pipeline Validator

You are tasked with validating that the training and testing pipeline for the Universal Control LLM Framework is functioning correctly.

## Instructions

1. **Use the pipeline-validator agent** to systematically check the pipeline
2. **Verify dataset availability**:
   - Check that pre-generated datasets exist
   - Confirm at least one dataset (di or vdp) is available for testing
   - Note: Data generation is handled separately and not part of this validation

3. **Verify training functionality**:
   - SFT training scripts run without errors
   - GRPO training can load SFT models and continue training
   - Models are saved properly in the models/ directory
   - Configuration files are properly parsed
   - LoRA adapters work correctly with Qwen3-4B-Base

4. **Verify evaluation functionality**:
   - Evaluation scripts can load trained models
   - Metrics computation works correctly
   - Plots are generated and saved properly
   - Both specialist and universal models can be evaluated

5. **Check the complete workflow**:
   - End-to-end training → evaluation pipeline
   - Dependencies and environment setup
   - Model versioning and management
   - GPU utilization and memory management

6. **Report issues and suggest fixes** for any problems found

## Usage Examples

- `/validate-pipeline` - Runs comprehensive pipeline validation
- Can be used after code changes, environment updates, or before experiments
- Provides detailed report of pipeline health and any issues

## Agent Integration

This command will automatically invoke the `pipeline-validator` agent with specialized knowledge of:
- Unsloth framework with SFT/GRPO training
- LLM-based control system pipelines
- Model versioning and management
- GPU optimization and debugging
- Pre-existing dataset validation

The agent will run minimal tests (5-10 training steps) to quickly verify functionality without consuming excessive resources.

**Note**: This validator assumes datasets have already been generated. If no datasets exist, you'll need to generate them separately using the data generation scripts before running the pipeline validation.