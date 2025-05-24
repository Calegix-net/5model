import os
import torch
import numpy as np
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import glob
import pandas as pd
from scipy import stats
import flwr as fl
# Import Context for Flower client_fn
# Import timestamp logging utilities
from flwr.common import Context
from flwr.common.typing import Metrics
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
    # Import new GPU optimization parameters
    ENABLE_ASYNC_OPERATIONS, PRELOAD_NEXT_BATCH, KEEP_MODEL_WARM,
    ENABLE_PIPELINE_PARALLELISM, PERSISTENT_WORKERS, PREFETCH_FACTOR, NUM_DATALOADER_WORKERS,
    # Import memory optimization parameters
    BATCH_SIZE, GRADIENT_ACCUMULATION,
    # Import client resource allocation parameters
    CLIENT_GPU_ALLOCATION, CLIENT_CPU_ALLOCATION,
    DIRECTORY1, RESULT_DIRECTORY, LAYER_SPECIFIC_DIRECTORY, SUMMARY_DIRECTORY,
    # Import GPU optimization toggles
    HIGH_GPU_UTILIZATION, ENABLE_MEMORY_CLEANUP, ENABLE_MEMORY_LOGGING,
    # Import advanced optimization settings
    AGGRESSIVE_GPU_OPTIMIZATION, MINIMIZE_CLIENT_INIT_OVERHEAD, 
    ENABLE_CUDA_STREAMS, NUM_CUDA_STREAMS, OVERLAP_COMMUNICATION_COMPUTE
)

# -----------------------------------------------------------------------------
# Initialization and Setup
# -----------------------------------------------------------------------------

# Add memory profiling function for debugging
def log_memory_usage(tag=""):
    if DEVICE.type == "cuda":
        if ENABLE_MEMORY_LOGGING:
            print(f"Memory usage {tag}: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.max_memory_allocated()/1024**2:.2f}MB (current/peak)")

# Create optimized empty_cache function based on settings
torch.cuda.empty_cache = (lambda: None) if not ENABLE_MEMORY_CLEANUP else torch.cuda.empty_cache

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
print(f"- GPU Utilization Mode: {'High Utilization' if HIGH_GPU_UTILIZATION else 'Conservative'}")
print(f"- GPU Allocation per Client: {CLIENT_GPU_ALLOCATION*100:.0f}%")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Memory Cleanup: {'Disabled' if not ENABLE_MEMORY_CLEANUP else 'Enabled'}")
print(f"- Memory Logging: {'Disabled' if not ENABLE_MEMORY_LOGGING else 'Enabled'}")
print(f"- Async Operations: {'Enabled' if ENABLE_ASYNC_OPERATIONS else 'Disabled'}")
print(f"- Model Warm-up: {'Enabled' if KEEP_MODEL_WARM else 'Disabled'}")
print(f"- Pipeline Parallelism: {'Enabled' if ENABLE_PIPELINE_PARALLELISM else 'Disabled'}")
print(f"- Dataloader Workers: {NUM_DATALOADER_WORKERS}")
print(f"- Prefetch Factor: {PREFETCH_FACTOR}")
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

# Global model cache to keep models warm between rounds
_model_cache = {}
_dataloader_cache = {}

# Global client cache to minimize initialization overhead
_client_cache = {}

# CUDA streams for better parallelism
# CUDA streams are created lazily to avoid serialization issues with Ray
_cuda_streams = None

def _init_cuda_streams():
    """Initialize CUDA streams on demand.

    Streams are created inside the worker process rather than at module import
    time so that no non-serialisable objects are present when Ray serialises
    this module.  This function is idempotent and can safely be called multiple
    times.
    """
    global _cuda_streams
    if _cuda_streams is None and ENABLE_CUDA_STREAMS and DEVICE.type == "cuda":
        _cuda_streams = [torch.cuda.Stream() for _ in range(NUM_CUDA_STREAMS)]
        print(f"Initialized {NUM_CUDA_STREAMS} CUDA streams for parallel operations")

# Global GPU warmup state
_gpu_warmup_active = False

# GPU warm-up operations to prevent utilization drops
def ensure_gpu_is_warm():
    """Perform dummy GPU operations to keep GPU active between rounds."""
    global _gpu_warmup_active
    if DEVICE.type == "cuda" and KEEP_MODEL_WARM and not _gpu_warmup_active:
        _gpu_warmup_active = True
        try:
            # More intensive warm-up operations to maintain GPU state
            if AGGRESSIVE_GPU_OPTIMIZATION:
                # Larger operations that better utilize GPU cores
                dummy_tensor = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
                # Multiple operations to keep different GPU functional units active
                for _ in range(3):
                    _ = torch.mm(dummy_tensor, dummy_tensor.T)
                    _ = torch.nn.functional.relu(dummy_tensor)
                # Use different streams if available
                _init_cuda_streams()
                if _cuda_streams:
                    for i, stream in enumerate(_cuda_streams[:2]):
                        with torch.cuda.stream(stream):
                            temp = torch.randn(256, 256, device=DEVICE)
                            _ = torch.mm(temp, temp.T)
                del dummy_tensor
            else:
                # Standard warm-up operations
                dummy_tensor = torch.randn(100, 100, device=DEVICE)
                _ = torch.mm(dummy_tensor, dummy_tensor.T)
                del dummy_tensor
        except Exception as e:
            print(f"GPU warm-up failed: {e}")
        finally:
            _gpu_warmup_active = False

# Background thread for async operations
def async_gpu_warmup_worker():
    """Background worker to keep GPU warm during idle periods."""
    while True:
        if KEEP_MODEL_WARM and DEVICE.type == "cuda":
            ensure_gpu_is_warm()
        # Adjust warmup frequency based on optimization level
        sleep_time = 0.05 if AGGRESSIVE_GPU_OPTIMIZATION else 0.1
        time.sleep(sleep_time)


def load_data(partition_id):
    """Load and prepare data for a specific client partition.

    Args:
        partition_id: The ID of the client partition to load.

    Returns:
        A tuple of (trainloader, testloader) for the client.
    """
    # Check cache first to avoid reloading
    cache_key = f"partition_{partition_id}"
    if cache_key in _dataloader_cache and (KEEP_MODEL_WARM or MINIMIZE_CLIENT_INIT_OVERHEAD):
        print(f"Using cached dataloaders for partition {partition_id}")
        return _dataloader_cache[cache_key]
        
    print(f"Loading data for client partition {partition_id}...")
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
    num_workers = NUM_DATALOADER_WORKERS if ENABLE_ASYNC_OPERATIONS else (2 if DEVICE.type == "cpu" else 0)
    # Use pin_memory for faster data transfer to GPU when using CUDA
    pin_memory = DEVICE.type == "cuda"
    # Add prefetching for more efficient data loading - increased for high utilization mode
    prefetch_factor = PREFETCH_FACTOR if num_workers > 0 and ENABLE_ASYNC_OPERATIONS else (2 if num_workers > 0 else None)
    # Enable persistent workers to avoid respawning overhead
    persistent_workers = PERSISTENT_WORKERS and num_workers > 0
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=BATCH_SIZE,  # Use batch size from config
        collate_fn=data_collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    # Test loader with potentially larger batch size for evaluation
    # We can usually use larger batches for evaluation since we don't need gradients
    # In high utilization mode, use even larger evaluation batches
    eval_batch_multiplier = 3 if HIGH_GPU_UTILIZATION else 2
    eval_batch_size = BATCH_SIZE * eval_batch_multiplier
    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=data_collator
    )
    
    # Cache dataloaders if keeping model warm to avoid reloading
    if KEEP_MODEL_WARM:
        _dataloader_cache[cache_key] = (trainloader, testloader)
        print(f"Cached dataloaders for partition {partition_id}")
    
    # Free memory after creating data loaders
    # The dataset contents are now referenced by the dataloaders, so we can delete original references
    del partition, partition_train_test
    import gc
    gc.collect()
    # Only clear cache if memory cleanup is enabled
    if DEVICE.type == "cuda" and ENABLE_MEMORY_CLEANUP:
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
        
        # Advanced GPU optimization features
        self.cuda_stream = None
        _init_cuda_streams()
        if ENABLE_CUDA_STREAMS and DEVICE.type == "cuda" and _cuda_streams:
            # Assign a CUDA stream to this client for parallel operations 
            self.cuda_stream = _cuda_streams[int(cid) % len(_cuda_streams)]
            print(f"Client {cid} assigned CUDA stream {int(cid) % len(_cuda_streams)}")
        
        # Initialize batch queue for async operations
        self.batch_queue = queue.Queue(maxsize=2) if ENABLE_ASYNC_OPERATIONS else None
        self.batch_loader_thread = None
        self.preloaded_batch = None
        
        # Thread pool for parallel operations - more workers for aggressive optimization
        max_workers = 4 if AGGRESSIVE_GPU_OPTIMIZATION else 2
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers) if ENABLE_ASYNC_OPERATIONS else None
        
        # Track device for memory management
        self.device = next(model.parameters()).device 
        
        # Set up amp for mixed precision training if on CUDA
        self.use_amp = self.device.type == "cuda" and hasattr(torch.cuda, 'amp')
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        # Pre-allocate tensors for better memory management if aggressive optimization is enabled
        if AGGRESSIVE_GPU_OPTIMIZATION and self.device.type == "cuda":
            self._preallocate_tensors()
        
        # Clean up memory at initialization time
        if self.device.type == "cuda":
            # Only cleanup if not using aggressive optimization (to maintain GPU state)
            if ENABLE_MEMORY_CLEANUP and not AGGRESSIVE_GPU_OPTIMIZATION:
                torch.cuda.empty_cache()
            # Print memory usage for monitoring
            log_memory_usage(f"Client {cid} at initialization")
            # Print memory stats for debugging (only if memory logging is enabled)
            if ENABLE_MEMORY_LOGGING:
                print(f"Client {cid} GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
    
    def _preallocate_tensors(self):
        """Preallocate commonly used tensors to reduce allocation overhead."""
        try:
            # Preallocate some tensors that will be reused during training
            self._temp_tensor_pool = []
            for _ in range(4):  # Create a small pool of reusable tensors
                tensor = torch.empty(BATCH_SIZE, 512, dtype=torch.float16, device=self.device)
                self._temp_tensor_pool.append(tensor)
        except Exception as e:
            print(f"Warning: Could not preallocate tensors for client {self.cid}: {e}")
            self._temp_tensor_pool = []

    def _batch_loader_worker(self, dataloader):
        """Background worker to preload batches asynchronously."""
        try:
            for batch in dataloader:
                if self.batch_queue:
                    self.batch_queue.put(batch)
                else:
                    break
        except Exception as e:
            print(f"Batch loader worker error: {e}")

    def _start_batch_preloading(self, dataloader):
        """Start async batch preloading."""
        if ENABLE_ASYNC_OPERATIONS and self.batch_queue:
            self.batch_loader_thread = threading.Thread(
                target=self._batch_loader_worker,
                args=(dataloader,),
                daemon=True
            )
            self.batch_loader_thread.start()

    def _get_next_batch(self):
        """Get next batch from queue or return None if not using async operations."""
        if ENABLE_ASYNC_OPERATIONS and self.batch_queue:
            try:
                return self.batch_queue.get(timeout=1.0)
            except queue.Empty:
                return None
        return None

    def _warmup_model(self):
        """Perform model warmup to maintain GPU state."""
        ensure_gpu_is_warm()

    def get_parameters(self, config=None):
        """Extract model parameters as a list of NumPy arrays."""
        # Only cleanup memory if not using aggressive optimization
        if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP and not AGGRESSIVE_GPU_OPTIMIZATION:
            torch.cuda.empty_cache()
        
        if OVERLAP_COMMUNICATION_COMPUTE and self.cuda_stream:
            # Use stream for async parameter extraction
            with torch.cuda.stream(self.cuda_stream):
                params = []
                state_dict = self.model.state_dict()
                for k, val in state_dict.items():
                    # Use non-blocking transfer to CPU
                    val_cpu = val.cpu()
                    params.append(val_cpu.detach().numpy())
                del state_dict
                return params
        else:
            # Standard parameter extraction with memory optimization
            params = []
            state_dict = self.model.state_dict()
            for k, val in state_dict.items():
                val_cpu = val.cpu().detach()
                params.append(val_cpu.numpy())
                del val_cpu
            
            del state_dict
            if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP and not AGGRESSIVE_GPU_OPTIMIZATION:
                torch.cuda.empty_cache()
            
            return params

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        if OVERLAP_COMMUNICATION_COMPUTE and self.cuda_stream:
            # Overlap parameter loading with GPU operations using streams
            with torch.cuda.stream(self.cuda_stream):
                keys = self.model.state_dict().keys()
                # Use non-blocking transfers for better performance
                state_dict = OrderedDict({
                    k: torch.tensor(v, device=self.device, dtype=self.model.state_dict()[k].dtype)
                    for k, v in zip(keys, parameters)
                })
                self.model.load_state_dict(state_dict)
                # Synchronize to ensure parameters are loaded before training
                if self.cuda_stream:
                    self.cuda_stream.synchronize()
        else:
            # Standard parameter loading
            keys = self.model.state_dict().keys()
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
        # Warm up the model and GPU before starting training
        if KEEP_MODEL_WARM:
            self._warmup_model()
            
        # Start async batch preloading if enabled
        self._start_batch_preloading(self.trainloader) if ENABLE_ASYNC_OPERATIONS else None
        
        # Memory optimization: Free CUDA cache before training starts
        if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP:
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
            if ENABLE_ASYNC_OPERATIONS and ENABLE_PIPELINE_PARALLELISM:
                # Advanced pipeline with async batch loading and parallel processing
                batch_iter = iter(self.trainloader)
                current_batch = None
                next_batch = None
                
                # Preload first batch
                try:
                    current_batch = next(batch_iter)
                    next_batch = next(batch_iter)
                except StopIteration:
                    pass
                
                while current_batch is not None:
                    # Process current batch while loading next batch in parallel
                    if self.thread_pool:
                        # Async load next batch
                        next_batch_future = self.thread_pool.submit(lambda: next(batch_iter, None))
                    
                    # Process current batch
                    result = self._process_training_batch(current_batch, optimizer, accumulation_steps, batch_count)
                    if isinstance(result, dict):
                        total_loss += result.get('loss', 0)
                        total_examples += result.get('examples', 0)
                    batch_count += 1
                    
                    # Get next batch from async loading
                    if self.thread_pool and 'next_batch_future' in locals():
                        try:
                            current_batch = next_batch_future.result(timeout=0.5)
                        except:
                            current_batch = None
                    else:
                        current_batch = next_batch
                        try:
                            next_batch = next(batch_iter)
                        except StopIteration:
                            next_batch = None
            else:
                # Standard synchronous training loop
                for batch in self.trainloader:
                    result = self._process_training_batch(batch, optimizer, accumulation_steps, batch_count)
                    if isinstance(result, dict):
                        total_loss += result.get('loss', 0)
                        total_examples += result.get('examples', 0)
                    batch_count += 1

        except Exception as e:
            print(f"Error during training on client {self.cid}: {e}")
            # Try to recover and continue (only if memory cleanup is enabled)
            if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP:
                torch.cuda.empty_cache()

    def _process_training_batch(self, batch, optimizer, accumulation_steps, batch_count):
        """Process a single training batch with optimizations."""
        try:
            # Use CUDA stream for this batch if available
            stream_context = torch.cuda.stream(self.cuda_stream) if self.cuda_stream else None
            
            if stream_context:
                with stream_context:
                    return self._process_batch_with_stream(batch, optimizer, accumulation_steps, batch_count)
            else:
                return self._process_batch_standard(batch, optimizer, accumulation_steps, batch_count)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return {'loss': 0, 'examples': 0}
    
    def _process_batch_with_stream(self, batch, optimizer, accumulation_steps, batch_count):
        """Process batch using CUDA streams for better parallelism."""
        # Only zero gradients at the beginning of each accumulation cycle
        if batch_count % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Move batch to device efficiently with non-blocking transfer
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Synchronize stream before computation
        if self.cuda_stream:
            self.cuda_stream.synchronize()
        
        # Use mixed precision for forward pass when available
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs.loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                optimizer.step()
        
        return {
            'loss': loss.item() * accumulation_steps,
            'examples': batch["input_ids"].size(0)
        }
    
    def _process_batch_standard(self, batch, optimizer, accumulation_steps, batch_count):
        """Standard batch processing without streams."""
        # Only zero gradients at the beginning of each accumulation cycle
        if batch_count % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Move batch to appropriate device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Use mixed precision for forward pass when available
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs.loss / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                optimizer.step()
        
        # Only cleanup memory if not using aggressive optimization
        if ENABLE_MEMORY_CLEANUP and not AGGRESSIVE_GPU_OPTIMIZATION and batch_count % 10 == 0 and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            'loss': loss.item() * accumulation_steps,
            'examples': batch["input_ids"].size(0)
        }

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
        if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP:
            del optimizer, batch_count, state_dict
            import gc; gc.collect()
            torch.cuda.empty_cache()
            log_memory_usage(f"Client {self.cid} after fit")
        elif self.device.type == "cuda":
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
        # Warm up the model before evaluation
        if KEEP_MODEL_WARM:
            self._warmup_model()
            
        # Memory optimization: Free CUDA cache before evaluation
        if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP:
            torch.cuda.empty_cache()
            log_memory_usage(f"Client {self.cid} before evaluate")
        
        # Set parameters and prepare for evaluation
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_correct, total_examples = 0, 0, 0
        
        # Perform evaluation without computing gradients to save memory
        with torch.no_grad():
            try:
                if ENABLE_ASYNC_OPERATIONS and ENABLE_PIPELINE_PARALLELISM:
                    # Process evaluation batches with pipeline parallelism
                    batch_iter = iter(self.testloader)
                    current_batch = next(batch_iter, None)
                    
                    while current_batch is not None:
                        # Process current batch while loading next batch
                        if self.thread_pool:
                            next_batch_future = self.thread_pool.submit(lambda: next(batch_iter, None))
                        
                        result = self._process_evaluation_batch(current_batch)
                        total_loss += result['loss']
                        total_correct += result['correct']
                        total_examples += result['examples']
                        
                        # Get next batch
                        if self.thread_pool and 'next_batch_future' in locals():
                            try:
                                current_batch = next_batch_future.result(timeout=0.5)
                            except:
                                current_batch = None
                        else:
                            current_batch = next(batch_iter, None)
                else:
                    # Standard evaluation loop
                    for batch in self.testloader:
                        result = self._process_evaluation_batch(batch)
                        total_loss += result['loss']
                        total_correct += result['correct']
                        total_examples += result['examples']
            except Exception as e:
                print(f"Error during evaluation on client {self.cid}: {e}")
        
    def _process_evaluation_batch(self, batch):
        """Process a single evaluation batch."""
        try:
            # Move batch to device efficiently
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Use mixed precision for faster evaluation if available
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
            
            # Extract results and calculate metrics
            loss = outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct = (predictions == batch["labels"]).sum().item()
            examples = batch["input_ids"].size(0)
            
            # Free memory after each batch
            del batch, outputs, predictions
            
            return {
                'loss': loss * examples,
                'correct': correct,
                'examples': examples
            }
        except Exception as e:
            print(f"Error processing evaluation batch: {e}")
            return {'loss': 0, 'correct': 0, 'examples': 0}
        
        # Calculate final accuracy
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        global_accuracy.append(accuracy)
        
        # Memory optimization: Free memory after evaluation completion
        if self.device.type == "cuda" and ENABLE_MEMORY_CLEANUP:
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc; gc.collect()
            log_memory_usage(f"Client {self.cid} after evaluate")
        elif self.device.type == "cuda":
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
    
    # Check if we have a cached client to avoid reinitialization
    if MINIMIZE_CLIENT_INIT_OVERHEAD:
        client_cache_key = f"client_{cid}"
        if client_cache_key in _client_cache:
            print(f"Reusing cached client {cid} for faster initialization")
            client = _client_cache[client_cache_key]
            # Lazily initialise CUDA streams in case this worker was restarted
            _init_cuda_streams()
            if (
                ENABLE_CUDA_STREAMS
                and DEVICE.type == "cuda"
                and getattr(client.numpy_client, "cuda_stream", None) is None
                and _cuda_streams
            ):
                client.numpy_client.cuda_stream = _cuda_streams[int(cid) % len(_cuda_streams)]
            return client
    
    print(f"Initializing client {cid}")

    # Check if we have a cached model to avoid reloading
    cache_key = f"model_{MODEL_NAME}"
    if cache_key in _model_cache and (KEEP_MODEL_WARM or MINIMIZE_CLIENT_INIT_OVERHEAD):
        print(f"Using cached model for client {cid}")
        model = _model_cache[cache_key]
        trainloader, testloader = load_data(int(cid))
        client = IMDBClient(cid, model, trainloader, testloader).to_client()
        # Cache client if minimizing overhead is enabled
        if MINIMIZE_CLIENT_INIT_OVERHEAD:
            _client_cache[f"client_{cid}"] = client
        return client
    
    # Log memory usage before model loading
    log_memory_usage(f"Client {cid} before model load")
    
    # Memory optimization: Configure PyTorch for efficient model loading
    try:
        # Try loading the model with memory efficiency settings
        if torch.cuda.is_available():
            # Clear cache to help with memory fragmentation (only if memory cleanup is enabled)
            if ENABLE_MEMORY_CLEANUP:
                torch.cuda.empty_cache()
            # Set PyTorch to use more efficient memory allocation algorithms
            torch.backends.cuda.matmul.allow_tf32 = HIGH_GPU_UTILIZATION  # Enable TF32 for high utilization mode
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
        # Only enable FP16 in high utilization mode and if supported
        if HIGH_GPU_UTILIZATION and DEVICE.type == "cuda" and hasattr(torch, 'float16'):
            pass  # Keep disabled for stability in gradient unscaling
        
        # Load model to CPU first
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            **model_kwargs
        )
        
        # Only after data is loaded and model initialized, move model to target device
        print(f"Moving model to {DEVICE} for client {cid}...")
        model = model.to(DEVICE)
        
        # Cache the model if keeping warm or minimizing overhead
        if KEEP_MODEL_WARM or MINIMIZE_CLIENT_INIT_OVERHEAD:
            _model_cache[cache_key] = model
            print(f"Cached model for future clients")
        
        # Log memory after model is moved to device
        log_memory_usage(f"Client {cid} after model loaded to {DEVICE}")
    except Exception as e:
        print(f"Error initializing model for client {cid}: {e}")
        # Try to recover with less aggressive settings
        if DEVICE.type == "cuda" and ENABLE_MEMORY_CLEANUP:
            print("Attempting recovery by clearing GPU memory...")
            torch.cuda.empty_cache()
            import gc; gc.collect()
            try:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, low_cpu_mem_usage=True)
                model = model.to(DEVICE)
                trainloader, testloader = load_data(int(cid))
            except:
                print("Recovery failed, falling back to CPU")
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
                model = model.to(torch.device("cpu"))
                trainloader, testloader = load_data(int(cid))
        else:
            print("Falling back to CPU due to GPU error")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            trainloader, testloader = load_data(int(cid)) 
            model = model.to(torch.device("cpu"))
    
    # Create and return client instance with appropriate device handling
    client = IMDBClient(cid, model, trainloader, testloader)
    
    # Cache client if minimizing overhead is enabled
    if MINIMIZE_CLIENT_INIT_OVERHEAD:
        _client_cache[f"client_{cid}"] = client.to_client()
        print(f"Cached client {cid} for future reuse")
    
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


# Skip GPU warmup on the driver to avoid creating CUDA streams before Ray
# serializes this module. Streams are initialized lazily inside each worker.

# Print optimization settings summary
print("\nGPU Optimization Settings:")
print(f"  - High GPU Utilization: {HIGH_GPU_UTILIZATION}")
print(f"  - Aggressive Optimization: {AGGRESSIVE_GPU_OPTIMIZATION}")
print(f"  - Client Caching: {MINIMIZE_CLIENT_INIT_OVERHEAD}")
print(f"  - CUDA Streams: {ENABLE_CUDA_STREAMS} ({NUM_CUDA_STREAMS} streams)" if ENABLE_CUDA_STREAMS else "  - CUDA Streams: Disabled")
print(f"  - Model Warming: {KEEP_MODEL_WARM}")
print(f"  - Pipeline Parallelism: {ENABLE_PIPELINE_PARALLELISM}")
print(f"  - Memory Cleanup: {'Minimal' if not ENABLE_MEMORY_CLEANUP else 'Enabled'}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print()

fl.simulation.start_simulation(
    # Apply Ray OOM prevention configuration
    ray_init_args=ray_init_config,
    client_fn=client_fn,
    # Simulation config
    num_clients=NUM_CLIENTS,
    # Optimized server config for better performance
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS, 
        round_timeout=1200.0 if AGGRESSIVE_GPU_OPTIMIZATION else 900.0,  # Allow more time for aggressive optimization
    ),
    strategy=fl.server.strategy.FedAvg(
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        # Aggregates metrics from client training
        fit_metrics_aggregation_fn=aggregate_metrics,
        # Optimize strategy for better performance
        fraction_fit=1.0,  # Use all clients for training
        fraction_evaluate=1.0,  # Use all clients for evaluation
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
