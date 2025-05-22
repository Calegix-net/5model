# Dataset Creation Tool

This tool is used to create datasets for federated learning experiments, with support for both normal and malicious node simulation.

## Configuration

Key configuration settings are in `config.py`:

- `ENABLE_MALICIOUS_NODES`: Set to `True` to enable malicious node behavior, `False` for normal mode
- `ATTACK_TYPE`: Set the attack type (e.g., "random") when malicious nodes are enabled
- `MALICIOUS_NODE_RATIO`: The ratio of malicious nodes (e.g., 0.1 for 10%)
- `MALICIOUS_DATA_RATIO`: The ratio of malicious data (e.g., 0.1 for 10%)
- `FORCE_CPU`: Set to `True` to force CPU-only mode (useful for environments with limited GPU memory)
- `NUM_CLIENTS` and `NUM_ROUNDS`: Control the scale of the simulation

## Usage

### 1. Creating a dataset

To create a dataset using the configured settings:

```bash
python run_and_collect.py
```

This will:
- Run the simulation based on the configuration in `config.py`
- Generate a dataset file (e.g., `dataset_normal.csv` or `dataset_random_10pct_20230523_120000.csv`)

### 2. Combining datasets

To combine multiple dataset files:

```bash
python combined.py
```

This will combine all dataset files in the current directory and create a combined dataset file.

### 3. Examples

#### Normal mode

```python
# In config.py
ENABLE_MALICIOUS_NODES = False
```

#### Malicious mode with random attack

```python
# In config.py
ENABLE_MALICIOUS_NODES = True
ATTACK_TYPE = "random"
MALICIOUS_NODE_RATIO = 0.1  # 10% malicious nodes
MALICIOUS_DATA_RATIO = 0.1  # 10% malicious data
```

### Memory Management

If you encounter GPU memory errors ("CUDA out of memory"), you have several options:

1. **Use CPU-only mode**:
   ```python
   # In config.py
   FORCE_CPU = True  # Force CPU-only mode regardless of CUDA availability
   ```

2. **Reduce batch size**:
   The default batch size is 8. For extremely memory-constrained environments, you can modify the batch size in `main.py` in the `load_data` function.

3. **Reduce client concurrency**:
   In `main.py`, the `client_resources` parameter in `start_simulation` controls how many resources each client can use.
   ```python
   client_resources={"num_cpus": 2, "num_gpus": 0.1}
   ```
   Lower values for `num_gpus` will reduce concurrent GPU usage.

4. **Reduce model size**:
   If you're still having memory issues, you could modify `MODEL_NAME` in `config.py` to use a smaller model.
