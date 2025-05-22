import os
import argparse
import shutil
import subprocess
import sys
import pandas as pd
import warnings

# Parse attack mode argument BEFORE importing from config
# This allows us to modify the attack mode before config sets up the related variables
def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Run and collect data with specific attack mode and run count")
    parser.add_argument("-mode", type=str,
                      choices=["none", "random_10pct", "random_15pct", "random_20pct", "random_30pct", "custom"],
                      help="Set the attack mode (none, random_10pct, random_15pct, random_20pct, random_30pct, custom)")
    parser.add_argument("-runs", "--num_runs", type=int,
                      help="Number of times to run main.py and collect data")

    # Store the parsed arguments
    args_dict = {}
    args_provided = False

    # Parse args without failing on unknown args (allows other scripts to add their own args)
    args, _ = parser.parse_known_args()

    if args.mode:
        # Set this as environment variable so config.py can use it
        print(f"\nSetting attack mode: {args.mode}")
        os.environ["ATTACK_MODE"] = args.mode
        args_dict["mode"] = args.mode
        args_provided = True

    # Store number of runs if provided
    if args.num_runs:
        args_dict["num_runs"] = args.num_runs
        args_provided = True

    args_dict["args_provided"] = args_provided
    return args_dict

# Parse attack mode before importing config
cli_args = parse_command_line_args()

# Now import from config (which will use our environment variable)
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

# Suppress specific numpy warnings that don't affect the correctness of results
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="overflow encountered in reduce")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar multiply")

print("""

   .-.  <3
  (. .)__,')
  / V      )
  \  (   \/       lets go
   `._`._ \       ======
 8===<<==`'==D
    
    """)

# Number of times to run main.py
if "num_runs" in cli_args:
    NUM_RUNS = cli_args["num_runs"]  # Use the value provided via command line
    print(f"Using number of runs from command line: {NUM_RUNS}")
else:
    NUM_RUNS = 5  # Default number of runs if not specified

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
        df["Run_ID"] = run_id  # add run identifier
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

    print("\nDataset creation complete!")
    print(f"Output file: {FINAL_DATASET_FILE}")
    print(f"{'Malicious' if ENABLE_MALICIOUS_NODES else 'Normal'} data collection complete.")

# Execute
if __name__ == "__main__":
    # If no arguments were provided, print help message and exit
    if not cli_args.get("args_provided", False):
        parser = argparse.ArgumentParser(description="Run and collect data with specific attack mode and run count")
        parser.add_argument("-mode", type=str, choices=["none", "random_10pct", "random_15pct", "random_20pct", "random_30pct", "custom"],
                          help="Set the attack mode (none, random_10pct, random_15pct, random_20pct, random_30pct, custom)")
        parser.add_argument("-runs", "--num_runs", type=int, help="Number of times to run main.py and collect data")
        parser.parse_args(["-h"])
        sys.exit(0)
    create_final_dataset()
