import os
import torch
import numpy as np
import glob
import pandas as pd
from scipy import stats
import flwr as fl
# Import Context for Flower client_fn
# Import timestamp logging utilities
from flwr.common import Context
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from collections import OrderedDict
from config import (
    ENABLE_MALICIOUS_NODES, ATTACK_TYPE, MALICIOUS_NODE_RATIO, 
    MALICIOUS_DATA_RATIO, DEVICE, MODEL_NAME, NUM_CLIENTS, NUM_ROUNDS, 
    # Import memory optimization parameters
    BATCH_SIZE, GRADIENT_ACCUMULATION,
    # Import client resource allocation parameters
    CLIENT_GPU_ALLOCATION, CLIENT_CPU_ALLOCATION,
    DIRECTORY1, RESULT_DIRECTORY, LAYER_SPECIFIC_DIRECTORY, SUMMARY_DIRECTORY
)

# -----------------------------------------------------------------------------
# Initialization and Setup
# -----------------------------------------------------------------------------

# Add memory profiling function for debugging
def log_memory_usage(tag=""):
    if DEVICE.type == "cuda":
        print(f"Memory usage {tag}: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB (current/peak)")

# Update malicious node settings based on configuration
def configure_malicious_settings():
    global ATTACK_TYPE, MALICIOUS_NODE_RATIO, MALICIOUS_DATA_RATIO
    
    if ENABLE_MALICIOUS_NODES:
        if ATTACK_TYPE == "random":
            # Default settings for random attack if enabled
            if MALICIOUS_NODE_RATIO <= 0:
                MALICIOUS_NODE_RATIO = 0.1
            if MALICIOUS_DATA_RATIO <= 0:
                MALICIOUS_DATA_RATIO = 0.1
        else:
            # Other attack types can be configured here
            pass
    else:
        # Reset to normal when disabled
        ATTACK_TYPE = "normal"
        MALICIOUS_NODE_RATIO = 0.0
        MALICIOUS_DATA_RATIO = 0.0

# Enable timestamped logging

# Apply malicious settings configuration
configure_malicious_settings()

# Create directories if they don't exist
os.makedirs(DIRECTORY1, exist_ok=True)
os.makedirs(RESULT_DIRECTORY, exist_ok=True)
os.makedirs(LAYER_SPECIFIC_DIRECTORY, exist_ok=True)
os.makedirs(SUMMARY_DIRECTORY, exist_ok=True)

# Print configuration for verification
print("Running with configuration:")
print(f"- Enable Malicious Nodes: {ENABLE_MALICIOUS_NODES}")
print(f"- Attack Type: {ATTACK_TYPE}")
print(f"- Malicious Node Ratio: {MALICIOUS_NODE_RATIO}")
print(f"- Malicious Data Ratio: {MALICIOUS_DATA_RATIO}")

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
    import warnings

    # Note: The IMDB dataset may show a warning as it's not in the list of officially tested datasets
    # This warning is informational and doesn't affect functionality
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The currently tested dataset are")
        fds = FederatedDataset(dataset="imdb", partitioners={"train": NUM_CLIENTS})
        partition = fds.load_partition(partition_id)  # Load the specific partition for this client
        # Free memory as soon as possible by deleting unused references
        del fds
        import gc
        gc.collect()
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # Process dataset - perform tokenization in batches for efficiency
    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns(["text"])  # Remove text to save memory
    partition_train_test = partition_train_test.rename_column("label", "labels")  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create data loaders with configured batch size to optimize memory usage
    # Smaller batch sizes reduce GPU memory requirements but may increase training time
    # Use num_workers for parallel data loading when appropriate
    # Default to 2 workers for CPU, 0 for GPU (avoids CUDA contention)
    num_workers = 2 if DEVICE.type == "cpu" else 0
    # Use pin_memory for faster data transfer to GPU when using CUDA
    pin_memory = DEVICE.type == "cuda"
    # Add prefetching for more efficient data loading
    prefetch_factor = 2 if num_workers > 0 else None
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=BATCH_SIZE,  # Use batch size from config
        collate_fn=data_collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )
    
    # Test loader with potentially larger batch size for evaluation
    # We can usually use larger batches for evaluation since we don't need gradients
    eval_batch_size = BATCH_SIZE * 2  # Double batch size for evaluation is often possible
    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=data_collator
    )
    
    # Free memory after creating data loaders
    # The dataset contents are now referenced by the dataloaders, so we can delete original references
    del partition, partition_train_test
    import gc
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    
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
        Dictionary containing analysis results for each layer and round.
    """

# Add efficient weight saving helper function
def save_weight_efficiently(weight, layer_name, client_id):
    """Save weight tensor to file with optimized I/O operations."""
    try:
        # Use pattern matching to count existing files efficiently
        pattern = f"*_{layer_name}_client{client_id}.pth"
        existing_files = len(glob.glob(os.path.join(DIRECTORY1, pattern)))
        
        # Create filename and save with CPU tensor to avoid GPU->CPU transfer during save
        filename = f"r{existing_files + 1}_{layer_name}_client{client_id}.pth"
        filepath = os.path.join(DIRECTORY1, filename)
        torch.save(weight.cpu(), filepath, _use_new_zipfile_serialization=True)
    except Exception as e:
        print(f"Error saving weight file for {layer_name}: {e}")

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
        # Track device for memory management
        self.device = next(model.parameters()).device 
        # Set up amp for mixed precision training if on CUDA
        self.use_amp = self.device.type == "cuda" and hasattr(torch.cuda, 'amp')
        # Initialize scaler properly without specifying a device parameter - it's inferred automatically
        # The device parameter causes problems with unscaling FP16 gradients
        # in newer PyTorch versions (2.7.0+)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        # Clean up memory at initialization time
        if self.device.type == "cuda":
            # Free memory cache to reduce fragmentation
            torch.cuda.empty_cache()
            # Print memory usage for monitoring
            log_memory_usage(f"Client {cid} at initialization")
            # Print memory stats for debugging
            print(f"Client {cid} GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")

    def get_parameters(self, config=None):
        """Extract model parameters as a list of NumPy arrays."""
        # Memory optimization: Free CUDA cache before parameter extraction
        if self.device.type == "cuda":
            # Clear any cached GPU tensors to free memory
            torch.cuda.empty_cache()
        
        # Optimize memory usage during parameter extraction
        # Explicitly move to CPU before conversion to NumPy to prevent CUDA OOM
        params = []
        
        # Process parameters in batches to reduce peak memory usage
        state_dict = self.model.state_dict()
        for k, val in state_dict.items():
            # Convert to CPU in batches to reduce memory pressure
            val_cpu = val.cpu().detach()
            # Convert to numpy and append to params list
            params.append(val_cpu.numpy())
            # Delete the CPU tensor to free memory immediately
            del val_cpu
        
        # Free memory after parameter extraction
        del state_dict
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return params

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        # Get model keys for parameter mapping
        keys = self.model.state_dict().keys()
        # Build state dict by transferring parameters directly to target device
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """Train the model on the local dataset.

        Args:
            parameters: List of model parameters.
            config: Configuration for training.

        Returns:
            Updated parameters, number of examples, and training metrics.
        """
        # Memory optimization: Free CUDA cache before training starts
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            log_memory_usage(f"Client {self.cid} before fit")
        
        # Set the model parameters and prepare for training
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        total_loss, total_examples = 0, 0
        
        # Memory optimization: Gradient accumulation
        accumulation_steps = GRADIENT_ACCUMULATION  # Get value from config
        
        batch_count = 0
        try:
            # Use mixed precision training when available (GPU only)
            for batch in self.trainloader:
                # Only zero gradients at the beginning of each accumulation cycle
                if batch_count % accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Move batch to appropriate device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Use mixed precision for forward pass when available
                if self.use_amp:
                    # Automatic mixed precision training for better performance on GPU
                    # Note: with autocast allows computation in lower precision but keeps gradients in FP32
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch)
                        # Scale loss for accumulation
                        loss = outputs.loss / accumulation_steps
                    
                    # Scale loss gradients and perform backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Only update weights after accumulation cycle
                    if (batch_count + 1) % accumulation_steps == 0:
                        # Unscale gradients for optimizer step
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    # Standard precision training path
                    outputs = self.model(**batch)
                    loss = outputs.loss / accumulation_steps
                    loss.backward()
                    
                    if (batch_count + 1) % accumulation_steps == 0:
                        optimizer.step()
                
                # Track total loss and examples processed
                total_loss += loss.item() * accumulation_steps  # Adjust for scaling
                total_examples += batch["input_ids"].size(0)
                batch_count += 1
                
                # Periodically free memory
                if batch_count % 10 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during training on client {self.cid}: {e}")
            # Try to recover and continue
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Save layer weights
        # Optimize file saving to reduce I/O overhead
        with torch.no_grad():  # Ensure no gradients are tracked during file operations
            state_dict = self.model.state_dict()
            # Use a batch approach to save layer weights more efficiently
            save_count = 0
            for layer_name, weight in state_dict.items():
                if ".weight" in layer_name:
                    # Use optimized weight saving function instead of direct file I/O
                    save_weight_efficiently(weight, layer_name, self.cid)
                    save_count += 1
            
            print(f"Saved {save_count} layer weights for client {self.cid}") 

        # Calculate metrics
        metrics = {"loss": total_loss / total_examples if total_examples > 0 else float('inf')}
        
        # Clean up and free memory
        if self.device.type == "cuda":
            del optimizer, batch_count, state_dict
            import gc; gc.collect()
            torch.cuda.empty_cache()
            log_memory_usage(f"Client {self.cid} after fit")
        
        return (
            self.get_parameters(),
            total_examples,
            metrics
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset.

        Args:
            parameters: List of model parameters.
            config: Configuration for evaluation.

        Returns: Loss, number of examples, and evaluation metrics.
        """
        # Memory optimization: Free CUDA cache before evaluation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            log_memory_usage(f"Client {self.cid} before evaluate")
        
        # Set parameters and prepare for evaluation
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_correct, total_examples = 0, 0, 0
        
        # Perform evaluation without computing gradients to save memory
        with torch.no_grad():
            try:
                for batch in self.testloader:
                    # Move batch to device efficiently
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Use mixed precision for faster evaluation if available
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(**batch)
                    else:
                        outputs = self.model(**batch)
                    
                    # Extract results and accumulate metrics
                    loss = outputs.loss.item()
                    total_loss += loss * batch["input_ids"].size(0)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    total_correct += (predictions == batch["labels"]).sum().item()
                    total_examples += batch["input_ids"].size(0)
                    
                    # Free memory after each batch
                    del batch, outputs, predictions
            except Exception as e:
                print(f"Error during evaluation on client {self.cid}: {e}")
        
        # Calculate final accuracy
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        global_accuracy.append(accuracy)
        
        # Memory optimization: Free memory after evaluation completion
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc; gc.collect()
            log_memory_usage(f"Client {self.cid} after evaluate")
            
        return (
            float(total_loss / total_examples),
            total_examples,
            {"accuracy": accuracy},
        )

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Federated Learning Setup
# -----------------------------------------------------------------------------


def client_fn(context: Context) -> fl.client.Client:
    """Create a federated client instance.
    
    Uses memory-optimized loading techniques to prevent OOM issues.
    """
    # Handle different Flower versions with a simplified approach
    # This ensures compatibility across Flower updates
    cid = str(context.cid) if hasattr(context, 'cid') else str(context) if isinstance(context, int) else "0"
    print(f"Initializing client {cid}")

    # Log memory usage before model loading
    log_memory_usage(f"Client {cid} before model load")
    
    # Memory optimization: Configure PyTorch for efficient model loading
    try:
        # Try loading the model with memory efficiency settings
        if torch.cuda.is_available():
            # Clear cache to help with memory fragmentation
            torch.cuda.empty_cache()
            # Set PyTorch to use more efficient memory allocation algorithms
            torch.backends.cuda.matmul.allow_tf32 = True  
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                # Enable mixed precision globally for better memory efficiency
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
    except Exception as e:
        print(f"Warning: Error setting up memory optimizations: {e}")
    
    # Model loading with best practices for memory utilization
    # First load model to CPU, then transfer to target device
    model = None
    try:
        # First load data - this prevents memory usage spikes when loading model and data together
        trainloader, testloader = load_data(int(cid))
        
        # Load model with memory optimizations
        print(f"Loading model for client {cid}...")
        model_kwargs = {
            'num_labels': 2, 
            'low_cpu_mem_usage': True  # Use less CPU memory during loading
        }
        
        # Don't force dtype to float16 (this causes issues with gradient scaling)
        if False and DEVICE.type == "cuda":  # Disabled - loading FP16 model causes gradient unscaling issues
            model_kwargs['torch_dtype'] = torch.float16
        
        # Load model to CPU first
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            **model_kwargs
        )
        
        # Only after data is loaded and model initialized, move model to target device
        print(f"Moving model to {DEVICE} for client {cid}...")
        model = model.to(DEVICE)
        
        # Log memory after model is moved to device
        log_memory_usage(f"Client {cid} after model loaded to {DEVICE}")
    except Exception as e:
        print(f"Error initializing model for client {cid}: {e}")
        # Fall back to CPU if there's an issue with GPU
        print("Falling back to CPU due to GPU error")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        trainloader, testloader = load_data(int(cid)) 
        model = model.to(torch.device("cpu"))
    
    # Create and return client instance with appropriate device handling
    client = IMDBClient(cid, model, trainloader, testloader)
    return client.to_client()


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
# Start the federated learning simulation with optimized resource allocation
# -----------------------------------------------------------------------------
# Federated Learning Execution
# -----------------------------------------------------------------------------

# Configure Ray with minimal settings, primarily using environment variables for OOM prevention
# as recommended in: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
ray_init_config = {
    # Runtime environment settings
    # Note: Removed runtime_env to avoid virtualenv dependency
    # All packages should already be available in your Poetry environment
    # Set custom configuration for Ray initialization
    "ignore_reinit_error": True,  # Allow reinitialization if needed
    "_system_config": {
        # Enable object spilling to disk to prevent OOM
        "object_spilling_config": "{\"type\": \"filesystem\", \"params\": {\"directory_path\": \"/tmp\"}}",
        # Removed unsupported parameter: max_task_lease_timeout_seconds
        "worker_register_timeout_seconds": 30  # Faster worker registration and recovery
    }
}

# Print OOM prevention configuration for clarity
print("\nUsing Ray OOM prevention with:")
print("- Using default Ray memory management")
# Memory monitoring settings are now using Ray defaults
print(f"- CPU allocation: {CLIENT_CPU_ALLOCATION} cores per client (reduces parallelism)")
print(f"- GPU allocation: {CLIENT_GPU_ALLOCATION} per client\n")

fl.simulation.start_simulation(
    # Apply Ray OOM prevention configuration
    ray_init_args=ray_init_config,
    client_fn=client_fn,
    # Simulation config
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=900.0),  # Extended timeout for slower clients
    strategy=fl.server.strategy.FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        # Aggregates metrics from client training
        fit_metrics_aggregation_fn=aggregate_metrics,
    ),
    # Resource allocation per client - configurable through config.py
    client_resources={
        "num_cpus": CLIENT_CPU_ALLOCATION,  # CPU cores per client
        "num_gpus": CLIENT_GPU_ALLOCATION,  # GPU fraction per client
        # Note: Requesting more CPUs per task reduces task parallelism
        # which helps prevent OOM issues
    }
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
