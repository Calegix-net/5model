import os

# Improve CUDA allocation to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Timestamp functionality for all print statements
import datetime
import builtins

import torch
import time

# Timestamp configuration - moved from timestamper.py
# Store the original print function
original_print = builtins.print

# Define a new print function that adds timestamps
def timestamped_print(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]')
    original_print(timestamp, *args, **kwargs)

# Replace the built-in print function with our timestamped version
builtins.print = timestamped_print

# Configuration for federated learning
NUM_CLIENTS = 10  # Number of clients
NUM_ROUNDS = 10  # Number of rounds

# Attack mode configuration settings
ATTACK_MODES = {
    "none": {
        "enable": False,
        "type": "normal",
        "node_ratio": 0.0,
        "data_ratio": 0.0
    },
    "random_10pct": {
        "enable": True,
        "type": "random",
        "node_ratio": 0.1,
        "data_ratio": 0.1
    },
    "random_15pct": {
        "enable": True,
        "type": "random",
        "node_ratio": 0.15,
        "data_ratio": 0.15
    },
    "custom": {
        "enable": True,
        "type": "random",
        "node_ratio": 0.25,
        "data_ratio": 0.1
    },
    "random_20pct": {
        "enable": True,
        "type": "random",
        "node_ratio": 0.2,
        "data_ratio": 0.2
    },
    "random_30pct": {
        "enable": True,
        "type": "random",
        "node_ratio": 0.3,
        "data_ratio": 0.3
    }
}

# Current attack mode - set this to change all malicious node settings at once
# Options: "none", "random_10pct", "random_15pct", "custom", "random_20pct", "random_30pct"

# Check if attack mode was set via environment variable (from run_and_collect.py)
if "ATTACK_MODE" in os.environ and os.environ["ATTACK_MODE"] in ATTACK_MODES:
    CURRENT_ATTACK_MODE = os.environ["ATTACK_MODE"]
    print(f"Using attack mode from environment: {CURRENT_ATTACK_MODE}")
else:
    # Default mode if not specified
    CURRENT_ATTACK_MODE = "none"

# Malicious node configuration
# These values are set based on CURRENT_ATTACK_MODE
ENABLE_MALICIOUS_NODES = ATTACK_MODES[CURRENT_ATTACK_MODE]["enable"]  # Flag to enable/disable malicious nodes
ATTACK_TYPE = ATTACK_MODES[CURRENT_ATTACK_MODE]["type"]  # Options: "normal", "random"
MALICIOUS_NODE_RATIO = ATTACK_MODES[CURRENT_ATTACK_MODE]["node_ratio"]  # Ratio of malicious nodes
MALICIOUS_DATA_RATIO = ATTACK_MODES[CURRENT_ATTACK_MODE]["data_ratio"]  # Ratio of malicious data

# ---------------------------------
# GPU and Memory Usage Configuration
# ---------------------------------

# Set device with memory management
FORCE_CPU = False       # Set to True to force CPU-only mode regardless of CUDA availability
USE_CUDA = torch.cuda.is_available()

# -------------------------------------------------------------------------
# Ray resource allocation and memory management settings
# -------------------------------------------------------------------------
# These settings are based on Ray's OOM prevention guidelines.
# Refer to: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
#
# To adjust the threshold, environment variables are used:
# - RAY_memory_usage_threshold=0.95 (kill threshold at 95% memory usage)
# - RAY_memory_monitor_refresh_ms=200 (check memory usage every 200ms)
# - To disable worker killing, set RAY_memory_monitor_refresh_ms=0
# -----------------------------------------------------------------------------
# GPU and CPU resource allocation per client (used in fl.simulation.start_simulation)

# GPU Utilization Mode - Toggle between conservative and high utilization settings
# Set to True for maximum GPU utilization, False for memory-conservative approach
#
# HIGH UTILIZATION MODE (HIGH_GPU_UTILIZATION = True):
# - Uses 90% GPU allocation per client for maximum throughput
# - Larger batch sizes (64) and evaluation batches for better GPU core utilization
# - Disables frequent memory cleanup to reduce GPU ramping up/down
# - Uses 95% of available GPU memory
#
# CONSERVATIVE MODE (HIGH_GPU_UTILIZATION = False):
# - Uses 50% GPU allocation per client for safer memory usage
# - Smaller batch sizes (32) to prevent out-of-memory errors
# - Enables frequent memory cleanup and logging for debugging
# - Uses 50% of available GPU memory for stability
HIGH_GPU_UTILIZATION = True  # Set to True for maximum GPU utilization (may cause OOM on some systems)

# Advanced GPU optimization settings for maintaining 100% utilization
AGGRESSIVE_GPU_OPTIMIZATION = True  # Enable most aggressive optimizations
MINIMIZE_CLIENT_INIT_OVERHEAD = True  # Keep clients alive between rounds
ENABLE_GPU_MEMORY_POOL = True  # Use memory pool to reduce allocation overhead
ENABLE_CUDA_STREAMS = True  # Use CUDA streams for better parallelism
OVERLAP_COMMUNICATION_COMPUTE = True  # Overlap model updates with computation

if HIGH_GPU_UTILIZATION:
    CLIENT_GPU_ALLOCATION = 0.9       # High GPU allocation per client (90% of one GPU)
    CLIENT_CPU_ALLOCATION = 2         # Reduced CPU allocation to allow more parallelism
    # New optimization settings for minimizing GPU utilization drops
    ENABLE_ASYNC_OPERATIONS = True    # Enable async data loading and model operations
    PRELOAD_NEXT_BATCH = True        # Preload next batch while processing current one
    KEEP_MODEL_WARM = True           # Keep model on GPU between rounds to avoid reloading
    ENABLE_PIPELINE_PARALLELISM = True # Enable overlapping operations
    PERSISTENT_WORKERS = True        # Keep dataloader workers alive between epochs
    PREFETCH_FACTOR = 4              # Number of batches to prefetch
    NUM_DATALOADER_WORKERS = 4       # Number of parallel data loading workers
    GPU_MEMORY_FRACTION = 0.99        # Start with 99% GPU memory
    BATCH_SIZE = 64                   # Larger batch size for better GPU utilization
    ENABLE_MEMORY_CLEANUP = False     # Disable frequent memory cleanup to maintain GPU state
    ENABLE_MEMORY_LOGGING = False     # Disable frequent memory logging
    # Additional optimizations for maintaining 100% GPU utilization
    if AGGRESSIVE_GPU_OPTIMIZATION:
        CLIENT_GPU_ALLOCATION = 0.95   # Even higher GPU allocation for maximum utilization
        BATCH_SIZE = 128               # Larger batch size for better tensor core utilization
        PREFETCH_FACTOR = 8            # More aggressive prefetching
        NUM_DATALOADER_WORKERS = 6     # More workers for data loading parallelism
        ENABLE_MEMORY_CLEANUP = False  # Never cleanup memory during training
    GRADIENT_ACCUMULATION = 2      # Use gradient accumulation for larger effective batch sizes
else:
    CLIENT_GPU_ALLOCATION = 0.5       # Conservative GPU allocation (50% of one GPU)
    CLIENT_CPU_ALLOCATION = 3         # Higher CPU allocation for stability
    # Conservative async settings
    ENABLE_ASYNC_OPERATIONS = False   # Disable async operations for stability
    PRELOAD_NEXT_BATCH = False       # Disable preloading for simpler debugging
    KEEP_MODEL_WARM = False          # Allow model to be moved off GPU for memory
    ENABLE_PIPELINE_PARALLELISM = False # Disable overlapping for simpler execution
    PERSISTENT_WORKERS = False       # Don't keep workers alive to save memory
    PREFETCH_FACTOR = 2              # Smaller prefetch for memory conservation
    NUM_DATALOADER_WORKERS = 2       # Fewer workers to reduce memory usage
    GPU_MEMORY_FRACTION = 0.5         # Conservative memory usage (50% of GPU memory)
    BATCH_SIZE = 32                   # Smaller batch size for memory safety
    ENABLE_MEMORY_CLEANUP = True      # Enable frequent memory cleanup
    ENABLE_MEMORY_LOGGING = True      # Enable frequent memory logging

# Track current memory fraction so it can be adjusted dynamically
CURRENT_GPU_MEMORY_FRACTION = GPU_MEMORY_FRACTION

# Initialize gradient accumulation if not set above
if 'GRADIENT_ACCUMULATION' not in locals():
    GRADIENT_ACCUMULATION = 1     # Number of batches to accumulate gradients over

# CUDA Streams configuration for better parallelism
if ENABLE_CUDA_STREAMS and USE_CUDA:
    NUM_CUDA_STREAMS = 4  # Number of CUDA streams for parallel operations
else:
    NUM_CUDA_STREAMS = 1

# Configure device (GPU or CPU)
if USE_CUDA and not FORCE_CPU:
    # Try to use CUDA with memory efficiency settings
    try:
        # Limit memory usage to specified fraction per process
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        CURRENT_GPU_MEMORY_FRACTION = GPU_MEMORY_FRACTION
        # Enable memory pool for faster allocation/deallocation if supported
        if ENABLE_GPU_MEMORY_POOL and hasattr(torch.cuda, 'memory_pool'):
            try:
                torch.cuda.memory._set_memory_pool_options(backend='native')
                print(f"Enabled CUDA memory pool for faster allocation")
            except:
                pass  # Memory pool not available in this PyTorch version
        print(f"Using GPU with {GPU_MEMORY_FRACTION*100:.0f}% memory allocation per process ({'High Utilization' if HIGH_GPU_UTILIZATION else 'Conservative'} mode)")
    except Exception as e:
        print(f"Warning: Could not set GPU memory fraction: {e}")
        print("Continuing with default GPU memory management.")
    DEVICE = torch.device("cuda")
else:
    # Fallback to CPU
    DEVICE = torch.device("cpu")
    print("Using CPU for computation (CUDA not available or disabled)")

# ------------------------------------------------------------------
# Dynamic GPU memory adjustment helpers
# ------------------------------------------------------------------
def reduce_gpu_memory_fraction(step: float = 0.05) -> None:
    """Reduce allowed GPU memory fraction to avoid OOM."""
    global CURRENT_GPU_MEMORY_FRACTION
    if DEVICE.type == "cuda":
        new_fraction = max(0.5, CURRENT_GPU_MEMORY_FRACTION - step)
        if new_fraction < CURRENT_GPU_MEMORY_FRACTION:
            try:
                torch.cuda.set_per_process_memory_fraction(new_fraction)
                CURRENT_GPU_MEMORY_FRACTION = new_fraction
                print(f"Adjusted GPU memory fraction to {new_fraction:.2f}")
            except Exception as exc:
                print(f"Could not adjust GPU memory fraction: {exc}")

MODEL_NAME = "distilbert-base-uncased"

# Directory setup based on configuration
def get_directory_names():
    attack_str = "normal" if not ENABLE_MALICIOUS_NODES else ATTACK_TYPE

    weight_dir = f"weight_pth_file_{attack_str}({NUM_CLIENTS}c{NUM_ROUNDS}r)"
    result_dir = f"result({NUM_CLIENTS}c{NUM_ROUNDS}r)"
    layer_specific_dir = os.path.join(result_dir, "layer_specific_results")
    summary_dir = f"summary_results_{attack_str}({NUM_CLIENTS}c{NUM_ROUNDS}r)"

    return weight_dir, result_dir, layer_specific_dir, summary_dir

# Get directory names
DIRECTORY1, RESULT_DIRECTORY, LAYER_SPECIFIC_DIRECTORY, SUMMARY_DIRECTORY = get_directory_names()

# Filenames used in data collection
SUMMARY_FILE = "all_layers_summary.csv"

# Dataset filename with timestamp for uniqueness
def get_dataset_filename():
    if ENABLE_MALICIOUS_NODES:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f'dataset_{ATTACK_TYPE}_{int(MALICIOUS_NODE_RATIO*100)}pct_{timestamp}.csv'
    else:
        return "dataset_normal.csv"

FINAL_DATASET_FILE = get_dataset_filename()
