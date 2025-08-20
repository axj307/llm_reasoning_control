# Data Inspector

You are tasked with analyzing, validating, and improving control system datasets for the Universal Control LLM Framework.

## Instructions

1. **Use the data-inspector agent** for comprehensive dataset analysis
2. **Dataset Quality Assessment**:
   - Analyze state space coverage and distribution
   - Validate trajectory optimality and feasibility
   - Check constraint satisfaction and physical consistency
   - Identify potential data quality issues

3. **Statistical Analysis**:
   - Distribution analysis and bias detection
   - Outlier identification and impact assessment
   - Coverage gap analysis with recommendations
   - Data consistency and format validation

4. **Training Readiness Evaluation**:
   - Format consistency across samples
   - Reasoning explanation quality assessment
   - Input-output alignment verification
   - Training impact prediction

5. **Improvement Recommendations**:
   - Suggest additional sampling regions
   - Recommend data cleaning actions
   - Propose augmentation strategies
   - Identify recomputation needs

## Usage Examples

- `/inspect-data` - Comprehensive dataset quality analysis
- Can focus on specific aspects (coverage, optimality, format)
- Provides actionable recommendations for dataset improvement
- Generates detailed quality reports with visualizations

## Agent Integration

This command automatically invokes the `data-inspector` agent with specialized knowledge of:
- Control theory and optimal trajectory analysis
- Statistical data analysis and validation methods
- ML dataset quality assessment techniques
- Data visualization and reporting
- Dataset improvement and augmentation strategies

The agent ensures high-quality datasets that lead to effective model training and robust performance, identifying and addressing potential issues before they affect training outcomes.