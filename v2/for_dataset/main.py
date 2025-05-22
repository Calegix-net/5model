import os
import torch
import numpy as np
import glob
import pandas as pd
from scipy import stats
import flwr as fl
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from collections import OrderedDict
from config import *

# -----------------------------------------------------------------------------
# Initialization and Setup
# -----------------------------------------------------------------------------


def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    for directory in [
        RESULT_DIRECTORY,
        LAYER_SPECIFIC_DIRECTORY,
        SUMMARY_DIRECTORY,
        DIRECTORY1,
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Create required directories
ensure_directories_exist()

# Initialize global accuracy list
global_accuracy = []

# -----------------------------------------------------------------------------
# Data Loading and Preparation
# -----------------------------------------------------------------------------


def load_data(partition_id):
    """Load and prepare data for a specific client partition.

    Args:
        partition_id: The ID of the client partition to load.

    Returns:
        A tuple of (trainloader, testloader) for the client.
    """
    from flwr_datasets import FederatedDataset  # Make sure this import is correct

    fds = FederatedDataset(dataset="imdb", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns(["text"])
    partition_train_test = partition_train_test.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=16,
        collate_fn=data_collator,
        shuffle=True,
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=16, collate_fn=data_collator
    )
    return trainloader, testloader


# -----------------------------------------------------------------------------
# Weight Analysis Functions
# -----------------------------------------------------------------------------


def load_weights(directory):
    """Load and parse saved model weights from .pth files.

    Returns:
        A dictionary mapping layer names to rounds to weights.
    """
    weights_dict = {}
    for filepath in glob.glob(os.path.join(directory, "*.pth")):
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        round_number = int(parts[0][1:])
        layer_name = "_".join(parts[1:-1])
        client_id = parts[-1][6:-4]

        if layer_name not in weights_dict:
            weights_dict[layer_name] = {}
        if round_number not in weights_dict[layer_name]:
            weights_dict[layer_name][round_number] = []

        weight = torch.load(filepath).cpu().numpy()
        weights_dict[layer_name][round_number].append(weight)

    return weights_dict


def analyze_weights(weights_dict):
    """Analyze per-layer variance and detect outliers in weights.

    Args:
        weights_dict: Dictionary containing weights organized by layer and round.

    Returns:
        Dictionary containing analysis results per layer and round.
    """
    analysis_results = {}
    for layer_name, rounds in weights_dict.items():
        analysis_results[layer_name] = {}

        for round_number, weights in rounds.items():
            weights_array = np.array(weights)

            # Compute basic statistics
            variance = np.var(weights_array, axis=0)
            mean_variance = np.mean(variance)

            # Detect outliers
            z_scores = np.abs(stats.zscore(weights_array, axis=0))
            outliers = (z_scores > 3).sum()

            # Compute weight statistics
            weight_stats = {
                "weight_min": np.min(weights_array),
                "weight_max": np.max(weights_array),
                "weight_mean": np.mean(weights_array),
                "weight_median": np.median(weights_array),
                "weight_std": np.std(weights_array),
                "weight_q25": np.percentile(weights_array, 25),
                "weight_q75": np.percentile(weights_array, 75),
                "first_weight": float(weights_array.flatten()[0]),
            }

            analysis_results[layer_name][round_number] = {
                "mean_variance": mean_variance,
                "outliers": outliers,
                **weight_stats,
            }

    return analysis_results


# -----------------------------------------------------------------------------
# Federated Learning Client Implementation
# -----------------------------------------------------------------------------


class IMDBClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config=None):
        """Extract model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.model.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """Train the model on the local dataset.

        Args:
            parameters: List of model parameters.
            config: Configuration for training.

        Returns:
            Updated parameters, number of examples, and training metrics.
        """
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        total_loss, total_examples = 0, 0
        for batch in self.trainloader:
            optimizer.zero_grad()
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_examples += batch["input_ids"].size(0)

        # Save layer weights
        state_dict = self.model.state_dict()
        for layer_name, weight in state_dict.items():
            if ".weight" in layer_name:
                existing_files = len(
                    [
                        f
                        for f in os.listdir(DIRECTORY1)
                        if f.endswith(f"_{layer_name}_client{self.cid}.pth")
                    ]
                )
                filename = f"r{existing_files + 1}_{layer_name}_client{self.cid}.pth"
                torch.save(weight, os.path.join(DIRECTORY1, filename))

        return (
            self.get_parameters(),
            total_examples,
            {"loss": total_loss / total_examples},
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset.

        Args:
            parameters: List of model parameters.
            config: Configuration for evaluation.

        Returns: Loss, number of examples, and evaluation metrics.
        """
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_correct, total_examples = 0, 0, 0
        with torch.no_grad():
            for batch in self.testloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss.item()
                total_loss += loss * batch["input_ids"].size(0)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_correct += (predictions == batch["labels"]).sum().item()
                total_examples += batch["input_ids"].size(0)

        accuracy = total_correct / total_examples
        global_accuracy.append(accuracy)
        return (
            float(total_loss / total_examples),
            total_examples,
            {"accuracy": accuracy},
        )


# -----------------------------------------------------------------------------
# Federated Learning Setup
# -----------------------------------------------------------------------------


def client_fn(cid: str) -> fl.client.Client:
    """Instantiate a federated client given its client ID."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    trainloader, testloader = load_data(int(cid))
    return IMDBClient(cid, model, trainloader, testloader).to_client()


def aggregate_metrics(results):
    """Aggregate metrics from client results.

    Args:
        results: List of client metrics to aggregate.

    Returns:
        Dictionary containing aggregated metrics.
    """
    aggregated_metrics = {}
    for result in results:
        if isinstance(result, tuple) and isinstance(result[1], dict):
            result = result[1]
        if not isinstance(result, dict):
            continue

        for key, value in result.items():
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]
    return {k: sum(v) / len(v) for k, v in aggregated_metrics.items()}


# -----------------------------------------------------------------------------
# Federated Learning Execution
# -----------------------------------------------------------------------------

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=fl.server.strategy.FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        fit_metrics_aggregation_fn=aggregate_metrics,
    ),
    client_resources={"num_cpus": 3, "num_gpus": 0.5},
)


# -----------------------------------------------------------------------------
# Results Analysis and Saving
# -----------------------------------------------------------------------------

# Load and analyze weights
weights_dict = load_weights(DIRECTORY1)
analysis_results = analyze_weights(weights_dict)

# Save results
all_layers_summary = []
for layer_name, rounds in analysis_results.items():
    for round_number, results in rounds.items():
        all_layers_summary.append(
            {
                "Round": round_number,
                "Mean_Variance": results["mean_variance"],
                "Number_of_outliers": results["outliers"],
                "Layer": layer_name,
                "Attack_Type": ATTACK_TYPE,
                "MALICIOUS_NODE_RATIO": MALICIOUS_NODE_RATIO,
                "MALICIOUS_DATA_RATIO": MALICIOUS_DATA_RATIO,
                "Weight_Min": results["weight_min"],
                "Weight_Max": results["weight_max"],
                "Weight_Mean": results["weight_mean"],
                "Weight_Median": results["weight_median"],
                "Weight_Std": results["weight_std"],
                "Weight_Q25": results["weight_q25"],
                "Weight_Q75": results["weight_q75"],
                "First_Weight": results["first_weight"],
            }
        )

summary_df = pd.DataFrame(all_layers_summary)
summary_file = os.path.join(SUMMARY_DIRECTORY, "all_layers_summary.csv")
summary_df.to_csv(summary_file, index=False)

# Check missing values in each column
print("\nMissing values in each column:")
print(summary_df.isnull().sum())

# Verify data types of each column
print("\nData types of each column:")
print(summary_df.dtypes)

# -----------------------------------------------------------------------------
# Layer-Specific Analysis
# -----------------------------------------------------------------------------

# Define layers of interest for detailed analysis
layers_of_interest = [
    "distilbert.embeddings.LayerNorm.weight",
    "distilbert.transformer.layer.3.sa_layer_norm.weight",
    "distilbert.transformer.layer.5.sa_layer_norm.weight",
    "pre_classifier.weight",
]

# Generate and save summary files for specific layers of interest
for layer in layers_of_interest:
    layer_variances = []
    layer_outliers = []
    rounds = []

    if layer in analysis_results:
        for round_number, results in analysis_results[layer].items():
            rounds.append(round_number)
            layer_variances.append(results["mean_variance"])
            layer_outliers.append(results["outliers"])

        # Convert per-layer data to DataFrame
        layer_summary_df = pd.DataFrame(
            {
                "Round": rounds,
                "Mean_Variance": layer_variances,
                "Number_of_outliers": layer_outliers,
                "First_Weight": [
                    results["first_weight"]
                    for results in analysis_results[layer].values()
                ],  # Include the first weight value
                "Weight_Min": [
                    results["weight_min"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Max": [
                    results["weight_max"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Mean": [
                    results["weight_mean"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Median": [
                    results["weight_median"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Std": [
                    results["weight_std"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Q25": [
                    results["weight_q25"]
                    for results in analysis_results[layer].values()
                ],
                "Weight_Q75": [
                    results["weight_q75"]
                    for results in analysis_results[layer].values()
                ],
                "Attack_Type": [ATTACK_TYPE] * len(rounds),
                "MALICIOUS_NODE_RATIO": [MALICIOUS_NODE_RATIO] * len(rounds),
                "MALICIOUS_DATA_RATIO": [MALICIOUS_DATA_RATIO] * len(rounds),
            }
        )

        # Save per-layer results
        layer_summary_file = os.path.join(
            LAYER_SPECIFIC_DIRECTORY, f"{layer}_summary.csv"
        )
        layer_summary_df.to_csv(layer_summary_file, index=False)
    else:
        print(f"No data found for layer: {layer}")
