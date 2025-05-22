import os
import torch

# Configuration for federated learning
NUM_CLIENTS = 10  # Number of clients
NUM_ROUNDS = 10  # Number of rounds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
ATTACK_TYPE = "normal"
MALICIOUS_NODE_RATIO = 0
MALICIOUS_DATA_RATIO = 0

# Directory setup for federated learning outputs
DIRECTORY1 = "weight_pth_file_normal(10c10r)"
RESULT_DIRECTORY = "result(10c10r)"
LAYER_SPECIFIC_DIRECTORY = os.path.join(RESULT_DIRECTORY, "layer_specific_results")
# Parent directory for saving summary at same level as DIRECTORY1
PARENT_DIRECTORY = os.path.dirname(DIRECTORY1)
SUMMARY_DIRECTORY = os.path.join(PARENT_DIRECTORY, "summary_results_normal(10c10r)")
# Filenames used in data collection
SUMMARY_FILE = "all_layers_summary.csv"
FINAL_DATASET_FILE = "final_dataset.csv"
