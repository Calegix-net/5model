import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, make_scorer
)
from sklearn.preprocessing import StandardScaler

import joblib

# imbalanced-learn の Pipeline と SMOTE をインポート
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# XGBoostのインポート
from xgboost import XGBClassifier

# データを読み込む
df = pd.read_csv('dataset.csv', low_memory=False)

# 'Attack_Type'を数値にマッピング
df['Attack_Type'] = df['Attack_Type'].map({'normal': 0, 'adversarial': 1})

# マッピング後の欠損値を確認
missing_attack_type = df['Attack_Type'].isnull().sum()
print(f"マッピング後の欠損しているAttack_Typeの数: {missing_attack_type}")

# 欠損値を削除
df = df.dropna(subset=['Attack_Type'])

# 再度欠損値を確認
missing_attack_type_after = df['Attack_Type'].isnull().sum()
print(f"欠損値処理後のAttack_Typeの数: {missing_attack_type_after}")

# さらに、他の潜在的な欠損値も確認
total_missing = df.isnull().sum()
print("各列の欠損値の数:")
print(total_missing)

# 数値列の欠損値を0で埋める
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

# 出力ディレクトリ
output_dir = 'output_files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ユニークなLayerを表示
print("Unique Layer values:")
print(df['Layer'].unique())

# 中間層と最終層のキーワードを指定
middle_layer_keywords = ['layer.3']
final_layer_keywords = ['layer.5']

# データをフィルタリング
middle_layers_df = df[df['Layer'].str.contains('|'.join(middle_layer_keywords), na=False)]
final_layers_df = df[df['Layer'].str.contains('|'.join(final_layer_keywords), na=False)]

# データフレームが空でないか確認
print("Middle layers dataframe shape:", middle_layers_df.shape)
print("Final layers dataframe shape:", final_layers_df.shape)

if middle_layers_df.empty:
    print("Error: middle_layers_df is empty. Please check the Layer keywords.")
    import sys
    sys.exit()

if final_layers_df.empty:
    print("Error: final_layers_df is empty. Please check the Layer keywords.")
    import sys
    sys.exit()

# 変化率を計算する関数
def calculate_change_rate(group):
    group = group.sort_values('Round')
    group['Mean_Variance_Change_Rate'] = group['Mean_Variance'].pct_change(fill_method=None)
    return group

# StratifiedKFoldを使用して層化を維持
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# DataFrameGroupByの警告を無視するか、別の方法で処理
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 関数を適用（警告を避けるために必要な列のみを選択）
middle_layers_df = middle_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate)[['Run_ID', 'Round', 'Mean_Variance', 'Mean_Variance_Change_Rate']].reset_index(drop=True)
final_layers_df = final_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate)[['Run_ID', 'Round', 'Mean_Variance', 'Mean_Variance_Change_Rate']].reset_index(drop=True)

# 特徴量を計算
grouped = df.groupby('Run_ID')
features = grouped.agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Number_of_outliers': ['mean', 'std', 'min', 'max', 'median'],
    'Layer': 'nunique'
}).reset_index()

features.columns = ['Run_ID'] + ['_'.join(col).strip('_') for col in features.columns.values[1:]]
features = features.rename(columns={'Layer_nunique': 'Num_Layers'})

# 中間層の特徴量
middle_features = middle_layers_df.groupby('Run_ID').agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Mean_Variance_Change_Rate': ['mean', 'std', 'min', 'max', 'median']
}).reset_index()

middle_features.columns = ['Run_ID'] + ['Middle_' + '_'.join(col).strip('_') for col in middle_features.columns.values[1:]]

# 最終層の特徴量
final_features = final_layers_df.groupby('Run_ID').agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Mean_Variance_Change_Rate': ['mean', 'std', 'min', 'max', 'median']
}).reset_index()

final_features.columns = ['Run_ID'] + ['Final_' + '_'.join(col).strip('_') for col in final_features.columns.values[1:]]

# 特徴量をマージ
features = pd.merge(features, middle_features, on='Run_ID', how='left')
features = pd.merge(features, final_features, on='Run_ID', how='left')

# 各'Run_ID'の'Attack_Type'を取得
attack_type = grouped['Attack_Type'].first().reset_index()

# 特徴量とラベルをマージ
data = pd.merge(features, attack_type, on='Run_ID')

# マージ後の欠損値を確認
missing_after_merge = data['Attack_Type'].isnull().sum()
print(f"マージ後の欠損しているAttack_Typeの数: {missing_after_merge}")

if missing_after_merge > 0:
    print("マージ後に欠損値が発生しています。Run_IDの整合性を確認してください。")
    # 欠損しているRun_IDを表示
    missing_runs = data[data['Attack_Type'].isnull()]['Run_ID'].unique()
    print(f"欠損しているRun_ID: {missing_runs}")
    # 欠損Run_IDを削除
    data = data.dropna(subset=['Attack_Type'])
    print(f"欠損Run_IDを削除後のデータ形状: {data.shape}")

# 特徴量行列Xとラベルベクトルyを準備
feature_cols = [col for col in data.columns if col not in ['Run_ID', 'Attack_Type']]
X = data[feature_cols]
y = data['Attack_Type']

# yに欠損値がないことを確認
if y.isnull().sum() > 0:
    print("Error: yにまだ欠損値が含まれています。データ前処理を見直してください。")
    import sys
    sys.exit()

# クラス分布を確認
print("クラス分布:")
print(y.value_counts())

# yの統計情報を表示
print("yの統計情報:")
print(y.describe())

# yのユニークな値を表示
print("yのユニークな値:")
print(y.unique())

# yにNaNが含まれているか確認
print(f"yにNaNが含まれているか: {y.isnull().any()}")

# データのバランスを調整（グローバルなSMOTEの適用を削除）
# SMOTEをパイプライン内でのみ適用するため、ここでは適用しない

# 相関分析と保存
X_with_target = X.copy()
X_with_target['Attack_Type'] = y
corr_matrix = X_with_target.corr()

# 相関行列を保存
corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

# ヒートマップをプロットして保存
plt.figure(figsize=(20, 18))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# 特徴量の分布をプロット
for col in feature_cols:
    plt.figure()
    sns.histplot(X[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(output_dir, f'feature_distribution_{col}.png'))
    plt.close()

# クロスバリデーションの設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # StratifiedKFoldを使用

# データを分割（層化を維持）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# y_train に欠損値がないことを確認
print(f"y_trainの数: {y_train.isnull().sum()}")

if y_train.isnull().sum() > 0:
    print("Error: y_trainに欠損値が含まれています。")
    import sys
    sys.exit()

# グローバルなスケーリングを削除
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# モデルの定義
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# 各モデルの結果を保存するディクショナリ
model_results = {}

# クロスバリデーションの実行部分を修正
def perform_cross_validation(pipeline, param_grid, X, y, cv):
    # スコアラーの修正
    def custom_scorer(y_true, y_pred):
        if len(np.unique(y_true)) < 2:
            return 0.0
        return accuracy_score(y_true, y_pred)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',  # カスタムスコアラーの代わりに標準のaccuracyを使用
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X, y)
        return grid_search
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}")
        return None

# プロット関数の修正
def plot_cv_comparison(cv_results_dict, output_dir):
    try:
        # データの準備
        results_data = []
        
        for model_name, model in cv_results_dict.items():
            if model is not None:
                # クロスバリデーションスコアの取得
                cv_scores = model.cv_results_['mean_test_score']
                valid_scores = cv_scores[~np.isnan(cv_scores)]
                
                results_data.append({
                    'Model': model_name,
                    'Best Score': model.best_score_,
                    'Mean CV Score': np.mean(valid_scores),
                    'Std CV Score': np.std(valid_scores),
                    'Test Accuracy': model.best_estimator_.score(X_test, y_test)  # テストデータでの精度を追加
                })
        
        # DataFrameの作成
        results_df = pd.DataFrame(results_data)
        
        # 結果をCSVに保存
        results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
        
        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # CV Scoresのプロット
        x_pos = np.arange(len(results_df))
        ax1.bar(x_pos, results_df['Mean CV Score'], 
                yerr=results_df['Std CV Score'], 
                align='center', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(results_df['Model'], rotation=45)
        ax1.set_ylabel('Cross-validation Score')
        ax1.set_title('Model Comparison: Cross-validation Scores')
        
        # Test Accuracyのプロット
        ax2.bar(x_pos, results_df['Test Accuracy'], 
                align='center', alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(results_df['Model'], rotation=45)
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Model Comparison: Test Accuracy')
        
        # レイアウトの調整
        plt.tight_layout()
        
        # プロットの保存
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()
        
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))
        
        return results_df  # 結果のDataFrameを返す
        
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        traceback.print_exc()
        return None

# モデルのトレーニングと評価部分の修正
def train_and_evaluate_model(model_name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv):
    print(f"Training and evaluating model: {model_name}")
    
    # クラス数をチェック
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Warning: Training data contains only one class ({unique_classes[0]})")
        return None
    
    try:
        # グリッドサーチの実行
        grid_search = perform_cross_validation(pipeline, param_grid, X_train, y_train, cv)
        
        if grid_search is None:
            print(f"Training and evaluation failed for {model_name}")
            return None
            
        # 最適なパラメータと精度を表示
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        
        # テストデータでの評価
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_name}: {accuracy:.3f}")
        
        return grid_search
        
    except Exception as e:
        print(f"Error in training {model_name}: {str(e)}")
        print(f"Training and evaluation failed for {model_name}")
        return None

# パイプラインの作成を修正
for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}")

    # パイプラインの作成（SMOTEを除外）
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # クロスバリデーションの設定を修正
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # SMOTEを個別に適用
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # ハイパーパラメータの定義（モデルごとに異なる）
    if model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_split': [2, 5, 10]
        }
    elif model_name == 'LogisticRegression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'classifier__penalty': ['l2']
        }
    elif model_name == 'SVM':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    elif model_name == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    elif model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    else:
        param_grid = {}

    # モデルのトレーニングと評価
    model_result = train_and_evaluate_model(
        model_name, 
        pipeline, 
        param_grid, 
        X_train_resampled,  # リサンプリング済みのデータを使用
        X_test, 
        y_train_resampled,  # リサンプリング済みのデータを使用
        y_test, 
        cv
    )

    if model_result is not None:
        model_results[model_name] = model_result

# 全モデルのクロスバリデーションスコアを比較
try:
    plot_cv_comparison(model_results, output_dir)
except Exception as e:
    print(f"Error plotting cross-validation scores comparison: {e}")

# 全モデルの精度をファイルに保存
try:
    accuracy_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [results['accuracy'] for results in model_results.values()],
        'Mean CV Accuracy': [results['cross_val_scores'].mean() for results in model_results.values()],
        'Std CV Accuracy': [results['cross_val_scores'].std() for results in model_results.values()]
    })

    accuracy_df.to_csv(os.path.join(output_dir, 'models_accuracy_comparison.csv'), index=False)
except Exception as e:
    print(f"Error saving models accuracy comparison: {e}")

# モデルの評価結果を保存
if model_results:
    results_df = plot_cv_comparison(model_results, output_dir)
    
    if results_df is not None:
        # 最良のモデルの選択（テスト精度に基づく）
        best_model_info = results_df.loc[results_df['Test Accuracy'].idxmax()]
        best_model_name = best_model_info['Model']
        best_model = model_results[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best parameters: {best_model.best_params_}")
        print(f"Best cross-validation score: {best_model.best_score_:.3f}")
        print(f"Test accuracy: {best_model_info['Test Accuracy']:.3f}")
        
        # 最良のモデルを保存
        joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
        
        # 最良のモデルの詳細な結果を保存
        with open(os.path.join(output_dir, 'best_model_results.txt'), 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Parameters: {best_model.best_params_}\n")
            f.write(f"Best Cross-validation Score: {best_model.best_score_:.3f}\n")
            f.write(f"Test Accuracy: {best_model_info['Test Accuracy']:.3f}\n")
            f.write("\nDetailed CV Results:\n")
            f.write(f"Mean CV Score: {best_model_info['Mean CV Score']:.3f}\n")
            f.write(f"Std CV Score: {best_model_info['Std CV Score']:.3f}\n")
else:
    print("No models were successfully trained")

print("全てのモデルのトレーニングと評価が完了しました。結果は 'output_files' ディレクトリに保存されています。")

def plot_feature_importance(model, feature_names, output_dir):
    try:
        # 特徴量の重要度を取得
        importances = model.best_estimator_.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 上位20個の特徴量のみを表示
        n_features = min(20, len(importances))
        
        plt.figure(figsize=(12, 6))
        plt.title("Top 20 Feature Importances")
        plt.bar(range(n_features), importances[indices][:n_features])
        plt.xticks(range(n_features), [feature_names[i] for i in indices][:n_features], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # プロットを保存
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # 特徴量の重要度をCSVファイルとして保存
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # 上位10個の特徴量を表示
        print("\nTop 10 Most Important Features:")
        for idx in indices[:10]:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            
    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")
        traceback.print_exc()

# モデルの評価結果を保存する部分の後に追加
if model_results and best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
    # 特徴量名のリストを作成
    feature_names = X_train.columns.tolist()
    
    # 特徴量の重要度をプロット
    plot_feature_importance(model_results[best_model_name], feature_names, output_dir)
    
    # 特徴量の重要度を保存
    try:
        importances = model_results[best_model_name].best_estimator_.named_steps['classifier'].feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        
        with open(os.path.join(output_dir, 'feature_importance_details.txt'), 'w') as f:
            f.write("Feature Importance Details:\n\n")
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{feature}: {importance:.6f}\n")
    except Exception as e:
        print(f"Error saving feature importance details: {str(e)}")

def plot_and_save_metrics(model, X_test, y_test, model_name, output_dir):
    try:
        # 予測と確率を取得
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name}.png'))
        plt.close()
        
        # 詳細な分類レポートを保存
        report = classification_report(y_test, y_pred)
        with open(os.path.join(output_dir, f'classification_report_{model_name}.txt'), 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("="*50 + "\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
            f.write(f"\n\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'classification_report': report
        }
        
    except Exception as e:
        print(f"Error plotting metrics for {model_name}: {str(e)}")
        traceback.print_exc()
        return None

# モデルの評価結果を保存する部分を修正（444行目付近）
if model_results:
    results_df = plot_cv_comparison(model_results, output_dir)
    
    if results_df is not None:
        # 最良のモデルの選択（テスト精度に基づく）
        best_model_info = results_df.loc[results_df['Test Accuracy'].idxmax()]
        best_model_name = best_model_info['Model']
        best_model = model_results[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best parameters: {best_model.best_params_}")
        print(f"Best cross-validation score: {best_model.best_score_:.3f}")
        print(f"Test accuracy: {best_model_info['Test Accuracy']:.3f}")
        
        # 最良のモデルを保存
        joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
        
        # 最良のモデルの評価指標を保存
        metrics_results = plot_and_save_metrics(
            best_model.best_estimator_,
            X_test,
            y_test,
            best_model_name,
            output_dir
        )
        
        # 最良のモデルの詳細な結果を保存
        with open(os.path.join(output_dir, 'best_model_results.txt'), 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Parameters: {best_model.best_params_}\n")
            f.write(f"Best Cross-validation Score: {best_model.best_score_:.3f}\n")
            f.write(f"Test Accuracy: {best_model_info['Test Accuracy']:.3f}\n")
            f.write("\nDetailed CV Results:\n")
            f.write(f"Mean CV Score: {best_model_info['Mean CV Score']:.3f}\n")
            f.write(f"Std CV Score: {best_model_info['Std CV Score']:.3f}\n")
            
            if metrics_results:
                f.write("\nDetailed Metrics:\n")
                f.write(f"ROC AUC Score: {metrics_results['roc_auc']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(metrics_results['classification_report'])

def plot_all_models_roc(model_results, X_test, y_test, output_dir):
    plt.figure(figsize=(10, 8))
    
    for model_name, model in model_results.items():
        if model is not None:
            try:
                # ROC曲線の計算
                y_prob = model.best_estimator_.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # ROC曲線のプロット
                plt.plot(fpr, tpr, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Error plotting ROC curve for {model_name}: {str(e)}")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    # 全モデルのROC曲線を保存
    plt.savefig(os.path.join(output_dir, 'all_models_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_models_metrics(model_results, X_test, y_test, output_dir):
    for model_name, model in model_results.items():
        if model is not None:
            try:
                # 予測
                y_pred = model.best_estimator_.predict(X_test)
                y_prob = model.best_estimator_.predict_proba(X_test)[:, 1]
                
                # Confusion Matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 個別のROC曲線
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 分類レポートの保存
                report = classification_report(y_test, y_pred)
                with open(os.path.join(output_dir, f'classification_report_{model_name}.txt'), 'w') as f:
                    f.write(f"Classification Report for {model_name}\n")
                    f.write("="*50 + "\n")
                    f.write(report)
                    f.write("\n\nConfusion Matrix:\n")
                    f.write(str(cm))
                    f.write(f"\n\nROC AUC Score: {roc_auc:.4f}")
                
            except Exception as e:
                print(f"Error plotting metrics for {model_name}: {str(e)}")
                traceback.print_exc()

# モデルの評価結果を保存する部分を修正
if model_results:
    results_df = plot_cv_comparison(model_results, output_dir)
    
    if results_df is not None:
        # 全モデルのメトリクスを保存
        plot_all_models_metrics(model_results, X_test, y_test, output_dir)
        
        # 全モデルのROC曲線を1つのグラフに保存
        plot_all_models_roc(model_results, X_test, y_test, output_dir)
        
        # ... 残りの処理は変更なし ...