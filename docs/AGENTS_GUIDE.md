# Universal Control LLM Framework - Agents Guide

This guide provides comprehensive information about the specialized agents available in your Universal Control LLM Framework. These agents streamline complex workflows and accelerate your research by automating repetitive tasks and providing expert-level analysis.

## 🎯 Quick Reference

| Agent | Command | Purpose | Key Features |
|-------|---------|---------|--------------|
| **experiment-runner** | `/run-experiment` | Parameter sweeps & batch training | Automated experiments, statistical analysis |
| **model-analyzer** | `/analyze-models` | Model performance analysis | Statistical comparison, failure analysis |
| **system-creator** | `/add-system` | Add new control systems | Complete integration, validation |
| **config-optimizer** | `/optimize-config` | Hyperparameter optimization | Hardware-aware, Bayesian optimization |
| **data-inspector** | `/inspect-data` | Dataset quality analysis | Coverage analysis, optimality validation |
| **slurm-manager** | `/manage-slurm` | HPC cluster management | Job optimization, batch submission |
| **results-visualizer** | `/create-plots` | Publication-ready plots | Scientific visualization, statistical plots |

---

## 🚀 Core Workflow Agents

### 1. **Experiment Runner** (`/run-experiment`)

**Purpose**: Automate comprehensive training experiments with parameter sweeps and batch analysis.

**Key Capabilities**:
- **Parameter Sweep Design**: Systematic exploration of hyperparameter spaces
- **Batch Training Management**: Sequential/parallel execution with resource optimization
- **Statistical Analysis**: Automated result collection and comparison
- **Failure Recovery**: Intelligent error handling and job resubmission
- **SLURM Integration**: HPC cluster job submission and management

**Use Cases**:
```bash
# Compare LoRA ranks across systems
/run-experiment
"Run experiments with LoRA ranks 4, 8, 16, 32 for double_integrator and van_der_pol"

# Training strategy comparison
/run-experiment
"Compare SFT-only vs SFT+GRPO training for all systems with statistical analysis"

# Cross-system performance study
/run-experiment
"Train specialist and universal models with different batch sizes and compare performance"
```

**Outputs**:
- Comprehensive experiment reports with statistical analysis
- Performance ranking tables with confidence intervals
- Resource usage summaries and optimization recommendations
- Best configuration recommendations with rationale

---

### 2. **Model Analyzer** (`/analyze-models`)

**Purpose**: Deep analysis and comparison of trained models with statistical rigor.

**Key Capabilities**:
- **Performance Benchmarking**: Multi-metric evaluation across test cases
- **Statistical Analysis**: ANOVA, t-tests, effect size calculations
- **Failure Case Analysis**: Systematic identification and categorization of failures
- **Training Dynamics**: Convergence analysis and learning pattern identification
- **Hyperparameter Impact**: Correlation analysis between parameters and performance

**Use Cases**:
```bash
# Model comparison study
/analyze-models
"Compare all my double_integrator models and identify the best performer"

# Specialist vs Universal analysis
/analyze-models
"Analyze performance gap between specialist and universal models with statistical significance"

# Training strategy evaluation
/analyze-models
"Compare SFT vs GRPO models across all systems with detailed analysis"
```

**Outputs**:
- Statistical comparison reports with significance testing
- Performance ranking with confidence intervals
- Failure case analysis with visualizations
- Training dynamics analysis and recommendations
- Model selection guidance with evidence-based rationale

---

### 3. **System Creator** (`/add-system`)

**Purpose**: Add new control systems to the framework with complete integration.

**Key Capabilities**:
- **Environment Implementation**: Complete BaseEnvironment class creation
- **Solver Integration**: LQR for linear systems, numerical optimization for nonlinear
- **Configuration Management**: YAML config files and framework registration
- **Validation Testing**: Physics validation and solver convergence testing
- **Dataset Generation**: Initial dataset creation for new systems

**Use Cases**:
```bash
# Add linear system
/add-system
"Add a pendulum control system with LQR solver and proper dynamics"

# Add nonlinear system
/add-system
"Implement a cart-pole system with nonlinear dynamics and numerical optimization"

# Add complex system
/add-system
"Create a quadrotor system with 6-DOF dynamics and MPC-style solver"
```

**Outputs**:
- Complete system implementation with all required files
- Configuration files and framework integration
- Validation reports confirming correct implementation
- Initial datasets ready for training
- Documentation and usage examples

---

### 4. **Config Optimizer** (`/optimize-config`)

**Purpose**: Intelligent configuration optimization for maximum performance within constraints.

**Key Capabilities**:
- **Hardware-Aware Optimization**: GPU memory and compute optimization
- **System-Specific Tuning**: Linear vs nonlinear system parameter optimization
- **Bayesian Optimization**: Intelligent hyperparameter search
- **Multi-Objective Optimization**: Performance vs efficiency vs memory trade-offs
- **Template Generation**: Pre-optimized configurations for common scenarios

**Use Cases**:
```bash
# Memory-constrained optimization
/optimize-config
"Optimize training config for van_der_pol with 8GB GPU memory limit"

# Performance optimization
/optimize-config
"Find optimal LoRA rank and learning rate for double_integrator specialist model"

# Universal model optimization
/optimize-config
"Optimize configuration for universal model training across all systems"
```

**Outputs**:
- Optimized YAML configuration files
- Performance vs resource trade-off analysis
- Hyperparameter sensitivity analysis
- Resource utilization predictions
- Configuration recommendations with expected improvements

---

## 🔧 Development Support Agents

### 5. **Data Inspector** (`/inspect-data`)

**Purpose**: Comprehensive dataset quality analysis and validation.

**Key Capabilities**:
- **State Space Coverage**: Analysis of state space exploration completeness
- **Trajectory Optimality**: Validation against optimal control solutions
- **Statistical Analysis**: Distribution analysis and bias detection
- **Format Validation**: Consistency and completeness checking
- **Training Impact Assessment**: Prediction of dataset effects on training

**Use Cases**:
```bash
# Dataset quality check
/inspect-data
"Check if my di dataset has good state space coverage and optimal trajectories"

# Training troubleshooting
/inspect-data
"My model isn't converging well, check for dataset issues that could affect training"

# Multi-system dataset validation
/inspect-data
"Validate universal training dataset for consistency across systems"
```

**Outputs**:
- Comprehensive dataset quality reports
- State space coverage visualizations
- Optimality validation results
- Data quality issue identification
- Improvement recommendations with specific actions

---

### 6. **SLURM Manager** (`/manage-slurm`)

**Purpose**: Streamline HPC cluster usage with intelligent job management.

**Key Capabilities**:
- **Resource Optimization**: Intelligent resource allocation based on job characteristics
- **Batch Job Management**: Complex dependency chains and parameter sweeps
- **Job Monitoring**: Real-time status tracking and failure recovery
- **Queue Optimization**: Strategic job distribution across partitions
- **Cost Management**: Resource usage optimization and cost analysis

**Use Cases**:
```bash
# Parameter sweep submission
/manage-slurm
"Submit batch experiments with different LoRA ranks across multiple systems"

# Job monitoring and recovery
/manage-slurm
"Monitor my training jobs and automatically resubmit any failures"

# Resource optimization
/manage-slurm
"Optimize my SLURM job scripts for better resource utilization"
```

**Outputs**:
- Optimized SLURM job scripts
- Job monitoring dashboards
- Resource utilization reports
- Failure analysis and recovery logs
- Cost optimization recommendations

---

### 7. **Results Visualizer** (`/create-plots`)

**Purpose**: Create publication-ready scientific visualizations and figures.

**Key Capabilities**:
- **Scientific Plotting**: Publication-quality figures with proper formatting
- **Statistical Visualization**: Significance testing with error bars and p-values
- **Control System Plots**: Trajectories, phase portraits, and vector fields
- **Comparative Analysis**: Multi-model performance comparisons
- **Interactive Dashboards**: Web-based exploration and animation

**Use Cases**:
```bash
# Publication figures
/create-plots
"Create publication plots comparing specialist vs universal models with statistical significance"

# Comprehensive visualization
/create-plots
"Generate trajectory plots, phase portraits, and performance comparisons for my experiments"

# Training analysis plots
/create-plots
"Visualize training dynamics and convergence patterns across different configurations"
```

**Outputs**:
- High-resolution figures in multiple formats (PDF, PNG, SVG)
- LaTeX-ready figures with automatic caption generation
- Interactive web dashboards
- Statistical analysis plots with significance testing
- Animation files for presentations

---

## 📋 Usage Patterns & Workflows

### **Typical Research Workflow**

1. **Dataset Preparation**:
   ```bash
   /inspect-data  # Validate dataset quality
   ```

2. **Configuration Optimization**:
   ```bash
   /optimize-config  # Get optimal hyperparameters
   ```

3. **Systematic Experimentation**:
   ```bash
   /run-experiment  # Execute parameter sweeps
   ```

4. **Model Analysis**:
   ```bash
   /analyze-models  # Compare and analyze results
   ```

5. **Visualization**:
   ```bash
   /create-plots  # Generate publication figures
   ```

### **Adding New Systems Workflow**

1. **System Implementation**:
   ```bash
   /add-system  # Implement new control system
   ```

2. **Configuration Optimization**:
   ```bash
   /optimize-config  # Optimize for new system
   ```

3. **Dataset Validation**:
   ```bash
   /inspect-data  # Validate generated dataset
   ```

4. **Training and Analysis**:
   ```bash
   /run-experiment  # Train models for new system
   /analyze-models  # Compare with existing systems
   ```

### **HPC Cluster Workflow**

1. **Job Preparation**:
   ```bash
   /optimize-config  # Optimize for cluster resources
   /manage-slurm    # Generate optimized job scripts
   ```

2. **Batch Submission**:
   ```bash
   /manage-slurm    # Submit parameter sweep jobs
   ```

3. **Monitoring and Recovery**:
   ```bash
   /manage-slurm    # Monitor jobs and handle failures
   ```

4. **Result Collection**:
   ```bash
   /analyze-models  # Analyze collected results
   /create-plots    # Visualize findings
   ```

---

## 🎓 Advanced Features

### **Statistical Rigor**
- All agents use proper statistical methods (ANOVA, t-tests, multiple comparisons correction)
- Confidence intervals and effect sizes reported
- Statistical significance testing with appropriate corrections

### **Resource Optimization**
- Hardware-aware configuration recommendations
- GPU memory profiling and optimization
- Cost-effective cluster resource utilization

### **Quality Assurance**
- Automated validation at every step
- Physics-based verification for control systems
- Professional standards for all outputs

### **Reproducibility**
- Complete parameter tracking and versioning
- Reproducible experiment configurations
- Detailed logging and audit trails

---

## 🚨 Important Notes

### **Environment Setup**
All agents automatically handle conda environment activation:
```bash
conda activate unsloth_env
```

### **File Organization**
Agents maintain clean file organization:
- Results in timestamped directories
- Automatic cleanup of temporary files
- Consistent naming conventions

### **Resource Management**
- Intelligent resource allocation based on workload
- Automatic batch size adjustment for memory constraints
- GPU utilization optimization

### **Error Handling**
- Robust error recovery and retry logic
- Detailed error reporting with suggested fixes
- Graceful handling of partial failures

---

## 📞 Getting Help

### **Agent-Specific Help**
Each agent provides detailed help when called without specific instructions:
```bash
/run-experiment
"What experiments can you help me run?"

/analyze-models  
"What types of analysis can you perform?"
```

### **Troubleshooting**
- Check logs in `logs/` directory for detailed error information
- Use `/inspect-data` to validate datasets before training
- Use `/optimize-config` if experiencing memory or performance issues

### **Best Practices**
- Start with `/inspect-data` for new datasets
- Use `/optimize-config` before large experiments
- Regular `/analyze-models` to track progress
- Generate `/create-plots` for all major results

---

**Happy researching! 🎉** These agents will significantly accelerate your workflow and ensure professional-quality results at every step.