# System Creator

You are tasked with adding new control systems to the Universal Control LLM Framework with complete integration.

## Instructions

1. **Use the system-creator agent** to implement new control systems
2. **Complete System Implementation**:
   - Create environment class extending BaseEnvironment
   - Implement system dynamics and simulation
   - Add optimal control solver (LQR or numerical)
   - Define system bounds and constraints

3. **Framework Integration**:
   - Update configuration files and imports
   - Add to available systems registry
   - Create system-specific YAML configurations
   - Integrate with data generation pipeline

4. **Validation and Testing**:
   - Validate system dynamics and physics
   - Test solver convergence and optimality
   - Generate initial datasets for training
   - Verify compatibility with existing training scripts

5. **Documentation and Examples**:
   - Generate system documentation
   - Create example trajectories and visualizations
   - Provide usage examples and best practices

## Usage Examples

- `/add-system` - Add a new control system to the framework
- Supports linear systems (LQR-based) and nonlinear systems (optimization-based)
- Handles complete integration including configuration and testing
- Provides validation and initial dataset generation

## Agent Integration

This command automatically invokes the `system-creator` agent with specialized knowledge of:
- Control theory and dynamical systems
- Optimal control solver implementation
- Framework architecture and integration patterns
- Configuration management and validation
- Quality assurance and testing procedures

The agent makes adding new control systems seamless while maintaining code quality and framework consistency.