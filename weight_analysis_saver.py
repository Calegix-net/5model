import os
import torch
import numpy as np
import glob
import pandas as pd
from scipy import stats

# 定数の定義
attack_type = "normal"
MALICIOUS_NODE_RATIO = 0
MALICIOUS_DATA_RATIO = 0

directory1 = 'weight_pth_file_normal(10c10r)'
result_directory = 'result_3(10c10r)'
layer_specific_directory = os.path.join(result_directory, 'layer_specific_results')

# all_layers_summary.csvを保存するディレクトリをdirectory1と同じ階層に作成
parent_directory = os.path.dirname(directory1)
summary_directory = os.path.join(parent_directory, 'summary_results_normal(10c10r)')

# 結果を保存するディレクトリが存在しない場合は作成
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# 新しいディレクトリが存在しない場合は作成
if not os.path.exists(layer_specific_directory):
    os.makedirs(layer_specific_directory)

# summary_results ディレクトリが存在しない場合は作成
if not os.path.exists(summary_directory):
    os.makedirs(summary_directory)

# .pth ファイルを読み込んで辞書に整理
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

weights_dict = load_weights(directory1)

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

analysis_results = analyze_weights(weights_dict)

# 全てのレイヤーのサマリーをまとめるためのリスト
all_layers_summary = []

# 結果を表示し、保存
for layer_name, rounds in analysis_results.items():
    for round_number, results in rounds.items():
        # 各レイヤーのデータをall_layers_summaryに保存
        all_layers_summary.append({
            'Round': round_number,
            'Mean Variance': results['mean_variance'],
            'Number of outliers': results['outliers'],
            'Layer': layer_name,
            'Attack Type': attack_type,
            'MALICIOUS_NODE_RATIO': MALICIOUS_NODE_RATIO,
            'MALICIOUS_DATA_RATIO': MALICIOUS_DATA_RATIO
        })

# 全てのレイヤーのサマリーをDataFrameに変換
summary_df = pd.DataFrame(all_layers_summary)

# all_layers_summary.csvをsummary_directoryに保存
summary_file = os.path.join(summary_directory, "all_layers_summary.csv")
summary_df.to_csv(summary_file, index=False)

# レイヤーごとのサマリーファイルも保存
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

        layer_summary_df = pd.DataFrame({
            "Round": rounds,
            "Mean Variance": layer_variances,
            "Number of outliers": layer_outliers,
            "Attack Type": [attack_type] * len(rounds),
            "MALICIOUS_NODE_RATIO": [MALICIOUS_NODE_RATIO] * len(rounds),
            "MALICIOUS_DATA_RATIO": [MALICIOUS_DATA_RATIO] * len(rounds)
        })

        # 新しいディレクトリに保存
        layer_summary_file = os.path.join(layer_specific_directory, f"{layer}_summary.csv")
        layer_summary_df.to_csv(layer_summary_file, index=False)
    else:
        print(f"No data found for layer: {layer}")