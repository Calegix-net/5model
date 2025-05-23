import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_all_results():
    """全モデルの結果を読み込む"""
    models = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'xgboost']
    all_results = []
    
    for model in models:
        model_dir = f'model_results/{model}'
        if os.path.exists(model_dir):
            # First_Weight結果の読み込み
            fw_results = pd.read_csv(
                os.path.join(model_dir, f'{model}_first_weight_only_results.csv')
            )
            # Other Features結果の読み込み
            other_results = pd.read_csv(
                os.path.join(model_dir, f'{model}_other_features_results.csv')
            )
            all_results.extend([fw_results, other_results])
    
    return pd.concat(all_results, ignore_index=True)

def create_comparison_plots(results_df, output_dir):
    """モデル比較の可視化"""
    # 精度比較プロット
    plt.figure(figsize=(12, 6))
    metrics = ['Best_Score', 'Mean_CV_Score', 'Test_Accuracy']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        sns.barplot(
            data=results_df,
            x='Model',
            y=metric,
            hue='Feature_Set'
        )
        plt.xticks(rotation=45)
        plt.title(f'{metric} Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'))
    plt.close()

def main():
    # 結果保存用ディレクトリの作成
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 全モデルの結果を読み込み
    all_results = load_all_results()
    
    # 結果の比較と可視化
    create_comparison_plots(all_results, output_dir)
    
    # 詳細な比較結果をCSVファイルとして保存
    all_results.to_csv(
        os.path.join(output_dir, 'all_models_comparison.csv'),
        index=False
    )
    
    # 最良モデルの特定
    best_model = all_results.loc[all_results['Test_Accuracy'].idxmax()]
    
    # 結果のサマリーをテキストファイルとして保存
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("=== Model Comparison Summary ===\n\n")
        f.write(f"Best performing model: {best_model['Model']}\n")
        f.write(f"Feature set: {best_model['Feature_Set']}\n")
        f.write(f"Test accuracy: {best_model['Test_Accuracy']:.3f}\n")
        f.write(f"Cross-validation score: {best_model['Mean_CV_Score']:.3f}\n")
        
        f.write("\nAll models performance summary:\n")
        f.write(all_results.groupby(['Model', 'Feature_Set'])['Test_Accuracy'].mean().to_string())
    
    print("\n=== Comparison completed. Results saved in 'comparison_results' directory ===")

if __name__ == '__main__':
    main()