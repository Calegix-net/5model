import os
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10
MODEL_NAME = "distilbert-base-uncased"
NUM_ROUNDS = 10
MALICIOUS_NODE_RATIO = 0.20
MALICIOUS_DATA_RATIO = 0.20

directory1 = 'adversarial_weight_pth_file'
if not os.path.exists(directory1):
    os.makedirs(directory1)

# Initialize global accuracy storage
global_accuracy = []

def load_data(partition_id, malicious: bool = False):
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

    if malicious:
        partition_train_test = load_adversarial_data(partition_train_test)
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(partition_train_test["train"], batch_size=16, collate_fn=data_collator, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=16, collate_fn=data_collator)
    return trainloader, testloader

def load_adversarial_data(dataset):
    adv_data_path = '/home/akai/federated_learning/TextFooler/data/adv_results/imdb_bert'
    orig_texts, adv_texts, labels = [], [], []
    
    with open(adv_data_path, 'r') as f:
        for line in f:
            if line.startswith("orig sent"):
                _, orig_label, orig_text = line.strip().split('\t')
                orig_texts.append(orig_text)
                labels.append(int(orig_label.split('(')[-1].rstrip('):')))
            elif line.startswith("adv sent"):
                _, adv_label, adv_text = line.strip().split('\t')
                adv_texts.append(adv_text)

    adv_train_size = int(len(adv_texts) * 0.8)
    adv_train_texts = adv_texts[:adv_train_size]
    adv_test_texts = adv_texts[adv_train_size:]
    orig_train_texts = orig_texts[:adv_train_size]
    orig_test_texts = orig_texts[adv_train_size:]
    train_labels = labels[:adv_train_size]
    test_labels = labels[adv_train_size:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    
    train_encodings = tokenizer(adv_train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(adv_test_texts, truncation=True, padding=True)
    
    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = IMDbDataset(train_encodings, train_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)
    
    dataset["train"] = train_dataset
    dataset["test"] = test_dataset
    return dataset

class IMDBClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config=None):
        print(f"Retrieving parameters with config: {config}")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.model.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
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
        
        state_dict = self.model.state_dict()
        client_id = self.cid
        for layer_name, weight in state_dict.items():
            if ".weight" in layer_name:
                existing_files = len([file for file in os.listdir(directory1) if file.endswith(f'_{layer_name}_client{client_id}.pth')])
                filename = f"r{existing_files + 1}_{layer_name}_client{client_id}.pth"
                
                for _ in range(2):
                    torch.save(weight, os.path.join(directory1, filename))
                    if os.path.exists(os.path.join(directory1, filename)):
                        break

        return self.get_parameters(), total_examples, {"loss": total_loss / total_examples}

    def evaluate(self, parameters, config):
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
        return float(total_loss / total_examples), total_examples, {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    malicious = (int(cid)+1) / NUM_CLIENTS <= MALICIOUS_NODE_RATIO
    trainloader, testloader = load_data(int(cid), malicious=malicious)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    print("Model architecture:")
    print(model)

    return IMDBClient(cid, model, trainloader, testloader)

def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated_metrics = {}
    for result in results:
        if not isinstance(result, dict):
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                result = result[1]
            else:
                raise ValueError("Expected 'result' to be a dictionary or a tuple containing a dictionary, but got type: {}".format(type(result).__name__))
        for key, value in result.items():
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]
    final_metrics = {k: sum(v) / len(v) for k, v in aggregated_metrics.items()}
    return final_metrics

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
    client_resources={"num_cpus": 3, "num_gpus": 0.5}
)

rounds = np.arange(1, NUM_ROUNDS + 1)
if global_accuracy:
    fig, axis = plt.subplots()
    axis.plot(rounds, global_accuracy, label="FedAvg")
    plt.ylim([0, 1])
    plt.title("Validation - IMDB")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    axis.set_aspect(abs((axis.get_xlim()[1] - axis.get_xlim()[0]) / (axis.get_ylim()[1] - axis.get_ylim()[0])) * 1.0)
    plt.savefig("accuracy_graph.png")
    plt.show()
else:
    print("No accuracy data to plot.")