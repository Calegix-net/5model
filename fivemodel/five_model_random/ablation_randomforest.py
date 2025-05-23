import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

def perform_rf_ablation_analysis(X, y, groups, output_dir):
    """
    RandomForest によるアブレーション実験を行い、各群（例："layer.5_Weight_Std"）を除去した場合の
    クロスバリデーションによる Accuracy, ROC_AUC およびベースラインとの差分を算出して保存する。
    """
    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # imblearn の Pipeline を使用（SMOTE → 標準化 → RandomForest）
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 【ベースライン】全特徴量を用いた場合の評価
    baseline_acc = np.mean(cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1))
    baseline_roc = np.mean(cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1))
    
    results = []
    
    # 各アブレーション群ごとに評価（例："layer.5_Weight_Std"）
    for group in groups:
        # 特徴量名は "layer.X_統計量" となっており、実際は "layer.X_統計量_集計関数"（例："layer.5_Weight_Std_mean"）となっているため、
        # group + "_" で始まる全列を除去する
        cols_to_drop = [col for col in X.columns if col.startswith(group + "_")]
        if len(cols_to_drop) == 0:
            print(f"{group} に該当する特徴量はありません。")
            continue

        X_ablation = X.drop(columns=cols_to_drop)
        
        # クロスバリデーションで Accuracy, ROC_AUC を計算
        acc_scores = cross_val_score(pipeline, X_ablation, y, cv=skf, scoring='accuracy', n_jobs=-1)
        roc_scores = cross_val_score(pipeline, X_ablation, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        
        mean_acc = np.mean(acc_scores)
        mean_roc = np.mean(roc_scores)
        
        # ベースラインとの差分（低下量）
        acc_drop = baseline_acc - mean_acc
        roc_drop = baseline_roc - mean_roc
        
        results.append({
            '特徴量／群名': group,
            'Accuracy': mean_acc,
            'ROC_AUC': mean_roc,
            'Accuracy低下': acc_drop,
            'ROC_AUC低下': roc_drop
        })
    
    # 結果を DataFrame 化
    results_df = pd.DataFrame(results)
    
    # ベースライン結果を先頭行に追加
    baseline_row = pd.DataFrame([{
        '特徴量／群名': 'Baseline (全特徴量)',
        'Accuracy': baseline_acc,
        'ROC_AUC': baseline_roc,
        'Accuracy低下': 0.0,
        'ROC_AUC低下': 0.0
    }])
    results_df = pd.concat([baseline_row, results_df], ignore_index=True)
    
    # CSV に保存
    csv_path = os.path.join(output_dir, 'rf_ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"結果を {csv_path} に保存しました。")
    
    # 結果を棒グラフでプロット（Accuracy と ROC_AUC）
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='特徴量／群名', y='Accuracy', data=results_df, palette='Blues_d')
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='特徴量／群名', y='ROC_AUC', data=results_df, palette='Reds_d')
    plt.xticks(rotation=45, ha='right')
    plt.title('ROC_AUC')
    plt.ylabel('ROC_AUC')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rf_ablation_performance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"プロットを {plot_path} に保存しました。")
    
    return results_df

def main():
    # 出力ディレクトリの設定
    output_dir = 'rf_ablation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # ★ ここでは、既に各レイヤーの特徴量を結合した CSV dataset.csv）を読み込む想定です。
    # dataset.csv には、calculate_layer_features の結果をマージしたデータが保存されており、
    # 各列名は "layer.X_統計量_集計関数"（例："layer.5_Weight_Std_mean"）の形式となっています。
    features_df = pd.read_csv('dataset.csv', low_memory=False)
    
    # 'Run_ID' 列が含まれている場合は削除（学習に不要なため）
    if 'Run_ID' in features_df.columns:
        features_df = features_df.drop(columns=['Run_ID'])
    
    # ターゲット変数 Attack_Type が含まれていれば分離
    if 'Attack_Type' in features_df.columns:
        y = features_df['Attack_Type']
        X = features_df.drop(columns=['Attack_Type'])
    else:
        # 例として別ファイルから読み込む場合
        y = pd.read_csv('attack_type.csv')['Attack_Type']
        X = features_df
    
    # アブレーション対象のグループリスト作成
    layers = ['layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5']
    stats = ['Number_of_outliers', 'Weight_Min', 'Weight_Max', 'Weight_Mean',
             'Weight_Median', 'Weight_Std', 'Weight_Q25', 'Weight_Q75']
    groups = []
    for layer in layers:
        for stat in stats:
            groups.append(f"{layer}_{stat}")
    
    # アブレーション実験の実行
    results_df = perform_rf_ablation_analysis(X, y, groups, output_dir)
    
    print("アブレーション実験の結果:")
    print(results_df)

if __name__ == "__main__":
    main()