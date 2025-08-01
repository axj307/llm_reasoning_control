# SLURM Manager

You are tasked with streamlining HPC cluster usage and SLURM workload management for the Universal Control LLM Framework.

## Instructions

1. **Use the slurm-manager agent** for intelligent cluster resource management
2. **Job Script Optimization**:
   - Generate optimized SLURM scripts based on workload characteristics
   - Automatically select optimal partitions and resources
   - Configure appropriate time limits and memory allocation
   - Implement robust error handling and monitoring

3. **Batch Job Management**:
   - Handle complex batch submissions with dependencies
   - Manage parameter sweep experiments across cluster nodes
   - Coordinate multi-system training with resource optimization
   - Implement intelligent queue strategies

4. **Monitoring and Recovery**:
   - Real-time job status monitoring and progress tracking
   - Automatic failure detection and recovery strategies
   - Dynamic resource adjustment based on performance
   - Resource utilization optimization and reporting

5. **Multi-Cluster Support**:
   - Support for different cluster configurations
   - Cost optimization across available resources
   - Load balancing and efficient job distribution
   - Best practices for cluster citizenship

## Usage Examples

- `/manage-slurm` - Comprehensive SLURM job management
- Can submit parameter sweeps, monitor jobs, and handle failures
- Provides resource optimization and cost-effective scheduling
- Generates cluster utilization reports and recommendations

## Agent Integration

This command automatically invokes the `slurm-manager` agent with specialized knowledge of:
- SLURM workload manager configuration and optimization
- HPC cluster resource management and scheduling
- Job dependency management and batch processing
- Resource monitoring and failure recovery strategies
- Multi-cluster support and cost optimization

The agent maximizes cluster utilization efficiency while minimizing wait times and computational costs, making HPC usage seamless and effective.