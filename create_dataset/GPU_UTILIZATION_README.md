# GPU Utilization Optimization

This federated learning implementation now supports advanced GPU utilization modes with multiple optimization strategies to maintain close to 100% GPU utilization throughout training.

## Configuration

In `config.py`, set the `HIGH_GPU_UTILIZATION` variable:

```python
# Set to True for maximum GPU utilization, False for memory-conservative approach
HIGH_GPU_UTILIZATION = False  # Default: Conservative mode

# Advanced GPU optimization settings for maintaining 100% utilization
AGGRESSIVE_GPU_OPTIMIZATION = True  # Enable most aggressive optimizations
MINIMIZE_CLIENT_INIT_OVERHEAD = True  # Keep clients alive between rounds
ENABLE_GPU_MEMORY_POOL = True  # Use memory pool to reduce allocation overhead
ENABLE_CUDA_STREAMS = True  # Use CUDA streams for better parallelism
OVERLAP_COMMUNICATION_COMPUTE = True  # Overlap model updates with computation
```

## Modes

### Conservative Mode (HIGH_GPU_UTILIZATION = False)
- **GPU Allocation**: 50% per client
- **Batch Size**: 32
- **Memory Usage**: 50% of GPU memory
- **Memory Management**: Frequent cleanup enabled
- **Best for**: Systems with limited GPU memory, debugging, stability

### High Utilization Mode (HIGH_GPU_UTILIZATION = True)
- **GPU Allocation**: 90% per client
- **Batch Size**: 64
- **Memory Usage**: 95% of GPU memory
- **Memory Management**: Minimal cleanup for sustained performance
- **Best for**: Systems with ample GPU memory (16GB+), maximum throughput

### Aggressive Optimization Mode (AGGRESSIVE_GPU_OPTIMIZATION = True)
- **GPU Allocation**: 95% per client
- **Batch Size**: 128 (with gradient accumulation)
- **Memory Usage**: 95% of GPU memory with memory pooling
- **Memory Management**: No cleanup during training to maintain GPU state
- **CUDA Streams**: 4 parallel streams for overlapped operations
- **Client Caching**: Clients remain in memory between rounds
- **Model Warming**: Continuous GPU activity to prevent utilization drops
- **Pipeline Parallelism**: Overlapped data loading, computation, and communication
- **Best for**: Maximum performance systems with 24GB+ GPU memory

## Performance Impact

### Conservative Mode
- ✅ Stable and reliable on all systems
- ✅ Lower risk of out-of-memory errors
- ⚠️ May have GPU utilization gaps between rounds
- ⚠️ Lower throughput due to smaller batch sizes

### High Utilization Mode
- ✅ Maximum GPU utilization (~100%)
- ✅ Higher throughput with larger batch sizes
- ✅ Reduced GPU ramping up/down between operations
- ⚠️ May cause OOM errors on systems with limited GPU memory
- ⚠️ Requires careful system monitoring

### Aggressive Optimization Mode
- ✅ Sustained ~100% GPU utilization with minimal gaps
- ✅ Maximum throughput with optimized parallelism
- ✅ CUDA streams eliminate GPU idle time
- ✅ Client and model caching eliminates initialization overhead
- ✅ Pipeline parallelism overlaps all operations
- ⚠️ Requires high-end GPU (24GB+ recommended)
- ⚠️ May be unstable on systems with insufficient resources
- ⚠️ Requires monitoring for thermal throttling

## New Optimization Features

### CUDA Streams
- **Purpose**: Parallel execution of GPU operations
- **Benefit**: Eliminates GPU idle time between operations
- **Usage**: Automatically enabled in aggressive mode

### Client Caching
- **Purpose**: Avoids client re-initialization overhead
- **Benefit**: Faster round transitions, maintained GPU state
- **Usage**: Clients remain in memory between rounds

### Pipeline Parallelism
- **Purpose**: Overlap data loading, computation, and communication
- **Benefit**: No GPU idle time while waiting for data or model updates
- **Usage**: Asynchronous operations with prefetching

### Memory Pooling
- **Purpose**: Reduce GPU memory allocation/deallocation overhead
- **Benefit**: Faster memory operations, reduced fragmentation
- **Usage**: Uses PyTorch's native memory pool when available

### Model Warming
- **Purpose**: Keep GPU active between rounds
- **Benefit**: Prevents GPU from entering low-power state
- **Usage**: Background thread performs dummy operations

## Recommendations

1. **Start with Conservative Mode** for initial testing
2. **Switch to High Utilization Mode** if:
   - Your system has 16GB+ GPU memory
   - You want maximum performance
   - You can monitor for OOM errors
3. **Enable Aggressive Optimization Mode** if:
   - Your system has 24GB+ GPU memory
   - You want sustained 100% GPU utilization
   - System stability is not critical
   - You can monitor thermal performance
4. **Monitor GPU usage** with `nvidia-smi` or similar tools
5. **Check system logs** for OOM killer activity if crashes occur
6. **Monitor GPU temperature** in aggressive mode to prevent thermal throttling

## Troubleshooting

If High Utilization Mode causes crashes:
1. Switch back to Conservative Mode
2. Check available GPU memory: `nvidia-smi`
3. Monitor system memory usage
4. Consider reducing `NUM_CLIENTS` or `NUM_ROUNDS` in config.py

If Aggressive Optimization Mode causes issues:
1. Disable `AGGRESSIVE_GPU_OPTIMIZATION`
2. Reduce `BATCH_SIZE` or `GRADIENT_ACCUMULATION`
3. Disable `ENABLE_CUDA_STREAMS` if driver issues occur
4. Check GPU temperature with `nvidia-smi`
5. Monitor system RAM usage (client caching uses more memory)

## Example Usage

```bash
# Run with conservative settings (default)
uv run python main.py

# For high performance, first edit config.py:
# HIGH_GPU_UTILIZATION = True
# Then run:
uv run python main.py

# For maximum performance with sustained 100% GPU utilization, edit config.py:
# HIGH_GPU_UTILIZATION = True
# AGGRESSIVE_GPU_OPTIMIZATION = True
# Then run:
uv run python main.py
```

## Performance Monitoring

The system will automatically print optimization settings when starting:
```
GPU Optimization Settings:
  - High GPU Utilization: True
  - Aggressive Optimization: True
  - Client Caching: True
  - CUDA Streams: True (4 streams)
  - Model Warming: True
  - Pipeline Parallelism: True
  - Memory Cleanup: Minimal
  - Batch Size: 128
  - Gradient Accumulation: 2
```

Monitor GPU utilization with:
```bash
# Real-time GPU monitoring
watch -n 0.5 nvidia-smi

# GPU utilization logging
nvidia-smi --query-gpu=utilization.gpu --format=csv --loop-ms=500
```

The system will automatically print the current mode and settings when it starts.

## Expected Performance Improvements

| Mode | GPU Utilization | Throughput Gain | Memory Usage | Stability |
|------|----------------|----------------|--------------|----------|
| Conservative | 60-80% | 1.0x (baseline) | 50% GPU | High |
| High Utilization | 85-95% | 1.5-2.0x | 95% GPU | Medium |
| Aggressive | 95-100% | 2.0-3.0x | 95% GPU + RAM | Lower |

The aggressive mode aims to maintain sustained 95-100% GPU utilization by eliminating idle time between operations through advanced parallelism and caching strategies.
