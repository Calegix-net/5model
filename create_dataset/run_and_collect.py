import os
# Import timestamper module to add timestamps to all print statements
import timestamper

import shutil
import concurrent.futures
import subprocess
import pandas as pd
import time
import os
from config import (
    RESULT_DIRECTORY,
    DIRECTORY1,
    SUMMARY_DIRECTORY,
    SUMMARY_FILE,
    FINAL_DATASET_FILE,
    ENABLE_MALICIOUS_NODES,
    ATTACK_TYPE,
    # Memory configuration parameters
    FORCE_CPU,
    CLIENT_GPU_ALLOCATION,
    BATCH_SIZE,
    MALICIOUS_NODE_RATIO,
)

# Number of times to run main.py
NUM_RUNS = 100  # specify number of runs

# Filename to save the dataset (from config)
# Using FINAL_DATASET_FILE directly

# Print configuration info
def print_config_info():
    print("\nRunning dataset collection with the following configuration:")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Enable malicious nodes: {ENABLE_MALICIOUS_NODES}")
    
    if ENABLE_MALICIOUS_NODES:
        print(f"Attack type: {ATTACK_TYPE}")
        print(f"Malicious node ratio: {MALICIOUS_NODE_RATIO}")
    
    # Print memory configuration for reference
    print(f"Using {'CPU only' if FORCE_CPU else 'GPU with ' + str(CLIENT_GPU_ALLOCATION*100) + '% allocation per client'}")
    print(f"Batch size: {BATCH_SIZE}")
    
    print(f"Output dataset file: {FINAL_DATASET_FILE}")
    print("\n")

# Delete directories and files
def cleanup():
    # Remove result, weight, and summary directories
    for d in (RESULT_DIRECTORY, DIRECTORY1, SUMMARY_DIRECTORY):
        if os.path.exists(d):
            shutil.rmtree(d)


# Run main.py and collect all_layers_summary.csv
def run_main_and_collect(run_id):
    print(f"--- 実行 {run_id + 1} ---")
    if ENABLE_MALICIOUS_NODES:
        print(f"(Using {ATTACK_TYPE} attack with {MALICIOUS_NODE_RATIO*100}% malicious nodes)")
    

    # Execute main.py
    subprocess.run(["python", "main.py"], check=True)

    # After successful execution, load the summary file
    summary_path = os.path.join(SUMMARY_DIRECTORY, SUMMARY_FILE)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df["Run ID"] = run_id  # add run identifier
        return df
    else:
        print(f"Error: {summary_path} does not exist")
        return None


# Function to append and save dataset
def append_to_final_dataset(df, header):
    # Write to CSV in append mode (header only for the first run)
    try:
        df.to_csv(FINAL_DATASET_FILE, mode="a", index=False, header=header)
    except Exception as e:
        print(f"Error writing to dataset file: {e}")


# Function to create required directories
def ensure_directories():
    """Make sure all needed directories exist before starting."""
    directories = [SUMMARY_DIRECTORY, DIRECTORY1]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# Function to create the dataset
def create_final_dataset():
    # Include header for the first run
    print_config_info()
    
    # Make sure directories exist
    ensure_directories()
    
    header = True

    # Run main.py the specified number of times and collect data
    for run_id in range(NUM_RUNS):
        df = run_main_and_collect(run_id)
        if df is not None:
            # Append collected data to file
            append_to_final_dataset(df, header=header)
            header = False  # include header only for the first run

        # Clean up directories and files
        cleanup()

    print(f"\nDataset creation complete!")
    print(f"Output file: {FINAL_DATASET_FILE}")
    print(f"{'Malicious' if ENABLE_MALICIOUS_NODES else 'Normal'} data collection complete.")

# Execute
if __name__ == "__main__":
    create_final_dataset()
