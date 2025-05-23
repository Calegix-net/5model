import pandas as pd
import glob
import os
import time
from config import ENABLE_MALICIOUS_NODES, ATTACK_TYPE, MALICIOUS_NODE_RATIO

# Combined output file name based on configuration
def get_output_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if ENABLE_MALICIOUS_NODES:
        return f'combined_dataset_{ATTACK_TYPE}_{int(MALICIOUS_NODE_RATIO*100)}pct_{timestamp}.csv'
    else:
        return 'combined_dataset_normal.csv'

def combine_datasets():
    output_file = get_output_filename()
    print(f"Combining datasets into: {output_file}")

    # List to store dataframes
    dataframes = []

    # Get existing max Run_ID (initial value is -1)
    if os.path.exists(output_file):
        combined_df = pd.read_csv(output_file)
        max_run_id = combined_df['Run_ID'].max()
        dataframes.append(combined_df)
    else:
        max_run_id = -1

    # Pattern to match dataset CSV files
    csv_files = glob.glob('dataset_*.csv')
    
    print(f"Found {len(csv_files)} dataset files to combine.")
    
    # Process each CSV file
    for file in csv_files:
        print(f"Processing: {file}")
        df = pd.read_csv(file)
        
        # Adjust Run_ID - ensure both are integers for addition
        df['Run_ID'] = df['Run_ID'].astype(int) + max_run_id + 1
        
        # Add file source info
        df['Source File'] = file
        
        # Add malicious flag info
        if 'random' in file or 'malicious' in file:
            df['Is Malicious Data'] = True
        else:
            df['Is Malicious Data'] = False
        
        # Add to list of dataframes
        dataframes.append(df)
        
        # Update max Run_ID
        max_run_id = df['Run_ID'].max()
    
    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved to: {output_file}")
        print(f"Total samples in combined dataset: {len(combined_df)}")
    else:
        print("No dataset files found to combine.")

if __name__ == "__main__":
    combine_datasets()
