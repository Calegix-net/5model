import os
import torch
import numpy as np
import glob
import pandas as pd
from scipy import stats
import flwr as fl
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from collections import OrderedDict

# 定数の定義
NUM_CLIENTS = 10  # クライアント数を指定
NUM_ROUNDS = 10   # ラウンド数を指定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
attack_type = "adversarial"  # 悪意のあるノードの設定
MALICIOUS_NODE_RATIO = 0.1  # 悪意のあるノードの割合
MALICIOUS_DATA_RATIO = 0.1  # 悪意のあるデータの割合

# フェデレーテッドラーニング用ディレクトリ設定
directory1 = 'weight_pth_file_adversarial(10c10r)'
result_directory = 'result(10c10r)'
layer_specific_directory = os.path.join(result_directory, 'layer_specific_results')

# all_layers_summary.csvを保存するディレクトリを作成
parent_directory = os.path.dirname(directory1)
summary_directory = os.path.join(parent_directory, 'summary_results_adversarial(10c10r)')

# 結果を保存するディレクトリが存在しない場合は作成
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
if not os.path.exists(layer_specific_directory):
    os.makedirs(layer_specific_directory)
if not os.path.exists(summary_directory):
    os.makedirs(summary_directory)

# 重みを保存するディレクトリが存在しない場合は作成
if not os.path.exists(directory1):
    os.makedirs(directory1)

# グローバル精度の初期化
global_accuracy = []

# データをロードする関数
def load_data(partition_id, malicious=False):
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

    # 悪意のあるデータ変更
    if malicious:
        partition_train_test = load_adversarial_data(partition_train_test)
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(partition_train_test["train"], batch_size=16, collate_fn=data_collator, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=16, collate_fn=data_collator)
    return trainloader, testloader

# 悪意のあるデータをロードする関数
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

# クライアントクラス
class IMDBClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).to(self.model.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
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
        
        # 重みを保存
        state_dict = self.model.state_dict()
        for layer_name, weight in state_dict.items():
            if ".weight" in layer_name:
                existing_files = len([file for file in os.listdir(directory1) if file.endswith(f'_{layer_name}_client{self.cid}.pth')])
                filename = f"r{existing_files + 1}_{layer_name}_client{self.cid}.pth"
                torch.save(weight, os.path.join(directory1, filename))

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

# クライアントの生成
def client_fn(cid: str) -> fl.client.Client:
    malicious = (int(cid)+1) / NUM_CLIENTS <= MALICIOUS_NODE_RATIO  # 悪意のあるクライアントかどうかを判定
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    trainloader, testloader = load_data(int(cid), malicious=malicious)
    return IMDBClient(cid, model, trainloader, testloader).to_client()

# メトリクスの集約
def aggregate_metrics(results):
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

# フェデレーテッドラーニングの実行
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

# 重みのロードと解析を行う関数
def load_weights(directory):
    weights_dict = {}
    for filepath in glob.glob(os.path.join(directory, "*.pth")):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        round_number = int(parts[0][1:])
        layer_name = '_'.join(parts[1:-1])
        client_id = parts[-1][6:-4]
        
        if layer_name not in weights_dict:
            weights_dict[layer_name] = {}
        if round_number not in weights_dict[layer_name]:
            weights_dict[layer_name][round_number] = []
        
        weight = torch.load(filepath).cpu().numpy()
        weights_dict[layer_name][round_number].append(weight)
    
    return weights_dict

# 各レイヤーごとの分散と外れ値の確認を行う関数
def analyze_weights(weights_dict):
    analysis_results = {}
    for layer_name, rounds in weights_dict.items():
        analysis_results[layer_name] = {}
        for round_number, weights in rounds.items():
            weights_array = np.array(weights)
            variance = np.var(weights_array, axis=0)
            mean_variance = np.mean(variance)
            z_scores = np.abs(stats.zscore(weights_array, axis=0))
            outliers = (z_scores > 3).sum()
            
            analysis_results[layer_name][round_number] = {
                'mean_variance': mean_variance,
                'outliers': outliers
            }
    return analysis_results

# 重みのロードと解析
weights_dict = load_weights(directory1)
analysis_results = analyze_weights(weights_dict)

# 結果の保存
all_layers_summary = []
for layer_name, rounds in analysis_results.items():
    for round_number, results in rounds.items():
        all_layers_summary.append({
            'Round': round_number,
            'Mean Variance': results['mean_variance'],
            'Number of outliers': results['outliers'],
            'Layer': layer_name,
            'Attack Type': attack_type,
            'MALICIOUS_NODE_RATIO': MALICIOUS_NODE_RATIO,
            'MALICIOUS_DATA_RATIO': MALICIOUS_DATA_RATIO
        })

summary_df = pd.DataFrame(all_layers_summary)
summary_file = os.path.join(summary_directory, "all_layers_summary.csv")
summary_df.to_csv(summary_file, index=False)

# 特定のレイヤーごとのサマリーファイルも保存
layers_of_interest = [
    "distilbert.embeddings.LayerNorm.weight",
    "distilbert.transformer.layer.3.sa_layer_norm.weight",
    "distilbert.transformer.layer.5.sa_layer_norm.weight",
    "pre_classifier.weight"
]

for layer in layers_of_interest:
    layer_variances = []
    layer_outliers = []
    rounds = []

    if layer in analysis_results:
        for round_number, results in analysis_results[layer].items():
            rounds.append(round_number)
            layer_variances.append(results['mean_variance'])
            layer_outliers.append(results['outliers'])

        # レイヤーごとのデータをDataFrameに変換
        layer_summary_df = pd.DataFrame({
            "Round": rounds,
            "Mean Variance": layer_variances,
            "Number of outliers": layer_outliers,
            "Attack Type": [attack_type] * len(rounds),
            "MALICIOUS_NODE_RATIO": [MALICIOUS_NODE_RATIO] * len(rounds),
            "MALICIOUS_DATA_RATIO": [MALICIOUS_DATA_RATIO] * len(rounds)
        })

        # レイヤーごとの結果を保存
        layer_summary_file = os.path.join(layer_specific_directory, f"{layer}_summary.csv")
        layer_summary_df.to_csv(layer_summary_file, index=False)
    else:
        print(f"No data found for layer: {layer}")

# 最後に、全てのラウンドのグローバル精度をプロット
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