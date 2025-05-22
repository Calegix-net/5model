import os
import shutil
import subprocess
import pandas as pd
from config import (
    RESULT_DIRECTORY,
    DIRECTORY1,
    SUMMARY_DIRECTORY,
    SUMMARY_FILE,
    FINAL_DATASET_FILE,
)

# Number of times to run main.py
NUM_RUNS = 100  # specify number of runs

# Filename to save the dataset (from config)
# Using FINAL_DATASET_FILE directly


# Delete directories and files
def cleanup():
    # Remove result, weight, and summary directories
    for d in (RESULT_DIRECTORY, DIRECTORY1, SUMMARY_DIRECTORY):
        if os.path.exists(d):
            shutil.rmtree(d)


# Run main.py and collect all_layers_summary.csv
def run_main_and_collect(run_id):
    print(f"--- 実行 {run_id + 1} ---")

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
    df.to_csv(FINAL_DATASET_FILE, mode="a", index=False, header=header)


# Function to create the dataset
def create_final_dataset():
    # Include header for the first run
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


# Execute
if __name__ == "__main__":
    create_final_dataset()
