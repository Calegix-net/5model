import os
# Import timestamper module to add timestamps to all print statements
import timestamper
import sys

import torch
import time

# Configuration for federated learning
NUM_CLIENTS = 2  # Number of clients
NUM_ROUNDS = 1  # Number of rounds

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
CLIENT_GPU_ALLOCATION = 0.5       # Fraction of GPU allocated per client (0.1 = 10% of one GPU)
CLIENT_CPU_ALLOCATION = 3     # Number of CPUs allocated per client

# Memory optimization settings
GPU_MEMORY_FRACTION = 0.5     # Limit memory usage per process (0.1 = 10% of GPU memory)
BATCH_SIZE = 8                # Batch size for training and evaluation
GRADIENT_ACCUMULATION = 2     # Number of batches to accumulate gradients over

# Configure device (GPU or CPU)
if USE_CUDA and not FORCE_CPU:
    # Try to use CUDA with memory efficiency settings
    try:
        # Limit memory usage to specified fraction per process
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)  
        print(f"Using GPU with {GPU_MEMORY_FRACTION*100}% memory allocation per process")
    except Exception as e:
        print(f"Warning: Could not set GPU memory fraction: {e}")
        print("Continuing with default GPU memory management.")
    DEVICE = torch.device("cuda")
else:
    # Fallback to CPU
    DEVICE = torch.device("cpu")
    print("Using CPU for computation (CUDA not available or disabled)")

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
