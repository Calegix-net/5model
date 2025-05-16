import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from collections import OrderedDict
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 2
MODEL_NAME = "distilbert-base-uncased"
NUM_ROUNDS = 2

directory1 = 'weight_pth_file_normal(2c2r)'
if not os.path.exists(directory1):
    os.makedirs(directory1)

# Initialize global accuracy storage
global_accuracy = []

def load_data(partition_id):
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
    trainloader = DataLoader(partition_train_test["train"], batch_size=16, collate_fn=data_collator, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=16, collate_fn=data_collator)
    return trainloader, testloader

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
        
        # .weight パラメータのみ保存する部分
        state_dict = self.model.state_dict()
        client_id = self.cid
        for layer_name, weight in state_dict.items():
            if ".weight" in layer_name:
                existing_files = len([file for file in os.listdir(directory1) if file.endswith(f'_{layer_name}_client{client_id}.pth')])
                filename = f"r{existing_files + 1}_{layer_name}_client{client_id}.pth"
                
                # ファイルを保存し、成功したか確認するループ
                for _ in range(2):
                    torch.save(weight, os.path.join(directory1, filename))
                    if os.path.exists(os.path.join(directory1, filename)):
                        break

        # メトリクスや集計なしに結果を返す
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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    trainloader, testloader = load_data(int(cid))

    # モデル構造を表示
    print("Model architecture:")
    print(model)

    return IMDBClient(cid, model, trainloader, testloader).to_client()

def aggregate_metrics(results):
    aggregated_metrics = {}
    for result in results:
        # 結果がタプルの場合、その中の辞書を取り出す
        if isinstance(result, tuple):
            if len(result) == 2 and isinstance(result[1], dict):
                result = result[1]  # 正しい形式の辞書を取得
            else:
                print(f"Skipping unexpected tuple format: {type(result).__name__}")
                continue
        
        # 結果が辞書でない場合はスキップ
        if not isinstance(result, dict):
            print(f"Skipping unexpected result format: {type(result).__name__}")
            continue
        
        # 辞書内の値を集計
        for key, value in result.items():
            if key in aggregated_metrics:
                aggregated_metrics[key].append(value)
            else:
                aggregated_metrics[key] = [value]
    
    # 最終的なメトリクスを計算して返す
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