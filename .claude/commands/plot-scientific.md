# Scientific Plot Generator

You are tasked with creating professional, publication-ready scientific plots for reinforcement learning and control systems research. 

## Instructions

1. **Use the scientific-plot-styler agent** to handle all plotting tasks
2. **Analyze the context** to understand what type of plot is needed:
   - Training metrics (rewards, losses, entropy)
   - Control trajectories (states, controls, phase transitions)
   - Comparison plots (RL vs optimal control)
   - Multi-phase control visualizations

3. **Apply consistent styling**:
   - Use scientific color schemes appropriate for publications
   - Ensure proper axis labels, legends, and titles
   - Apply consistent font sizes and line weights
   - Include waypoint markers for multi-phase problems
   - Use phase-coordinated colors when applicable

4. **Generate both PDF and PNG formats** for flexibility in publication workflows

5. **Follow the project's plotting conventions** as established in the existing figures directory structure

## Usage Examples

- `/plot-scientific` - Creates publication-ready plots from training data or model results
- Can be used with specific model files, training logs, or trajectory data
- Automatically determines appropriate plot types based on data content

## Agent Integration

This command will automatically invoke the `scientific-plot-styler` agent with specialized knowledge of:
- Matplotlib styling for scientific publications
- Multi-phase control problem visualization
- RL training metrics presentation
- Comparative analysis plots