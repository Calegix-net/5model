# GPU Utilization Optimization

This federated learning implementation now supports two GPU utilization modes to optimize performance based on your system capabilities.

## Configuration

In `config.py`, set the `HIGH_GPU_UTILIZATION` variable:

```python
# Set to True for maximum GPU utilization, False for memory-conservative approach
HIGH_GPU_UTILIZATION = False  # Default: Conservative mode
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

## Recommendations

1. **Start with Conservative Mode** for initial testing
2. **Switch to High Utilization Mode** if:
   - Your system has 16GB+ GPU memory
   - You want maximum performance
   - You can monitor for OOM errors
3. **Monitor GPU usage** with `nvidia-smi` or similar tools
4. **Check system logs** for OOM killer activity if crashes occur

## Troubleshooting

If High Utilization Mode causes crashes:
1. Switch back to Conservative Mode
2. Check available GPU memory: `nvidia-smi`
3. Monitor system memory usage
4. Consider reducing `NUM_CLIENTS` or `NUM_ROUNDS` in config.py

## Example Usage

```bash
# Run with conservative settings (default)
uv run python main.py

# For maximum performance, first edit config.py:
# HIGH_GPU_UTILIZATION = True
# Then run:
uv run python main.py
```

The system will automatically print the current mode and settings when it starts.
