import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RepeatedStratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, make_scorer, roc_auc_score
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
df['Attack_Type'] = df['Attack_Type'].map({'normal': 0, 'random': 1})

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

# レイヤーのキーワードを個別に指定
layer_keywords = ['layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5']

# データをフィルタリング
layer_dfs = {}
for layer in layer_keywords:
    layer_df = df[df['Layer'].str.contains(layer, na=False)]
    if layer_df.empty:
        print(f"Error: {layer} dataframe is empty. Please check the Layer keywords.")
        import sys
        sys.exit()
    layer_dfs[layer] = layer_df

# 各層ごとの特徴量を計算する関数
def calculate_layer_features(df, layer_name):
    """各層ごとの特徴量を計算する関数"""
    try:
        features = df.groupby('Run_ID').agg({
            # 'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
            'Number_of_outliers': ['mean', 'std', 'min', 'max', 'median'],
            'First_Weight': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Min': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Max': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Mean': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Median': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Std': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Q25': ['mean', 'std', 'min', 'max', 'median'],
            'Weight_Q75': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()
        
        # 集計後の欠損値を0で埋める
        features = features.fillna(0)
        
        # 欠損値が残っていないか確認
        if features.isnull().sum().sum() > 0:
            print(f"Warning: {layer_name} still contains NaN values after aggregation")
            print(features.isnull().sum())
        
        return features
        
    except Exception as e:
        print(f"Error in calculating features for {layer_name}: {str(e)}")
        print("Columns in df:", df.columns)
        return None

def format_column_names(df, layer_name):
    """カラム名にレイヤー名の接頭辞を追加"""
    df.columns = ['Run_ID'] + [f'{layer_name}_{col[0]}_{col[1]}' for col in df.columns[1:]]
    return df

# 各レイヤーの特徴量を計算
layer_features = {}
for layer in layer_keywords:
    # 特徴量の計算（変化率の計算を削除）
    features = calculate_layer_features(layer_dfs[layer], layer)
    
    if features is not None:
        features = format_column_names(features, layer)
        layer_features[layer] = features
    else:
        print(f"Skipping layer {layer} due to calculation error")
        continue

# 全ての特徴量をマージする前に欠損値チェック
features = layer_features[layer_keywords[0]]
for layer in layer_keywords[1:]:
    if layer in layer_features:
        # マージ前の欠損値チェック
        print(f"\nChecking NaN values before merging {layer}:")
        print("Current features NaN count:", features.isnull().sum().sum())
        print(f"{layer} features NaN count:", layer_features[layer].isnull().sum().sum())
        
        features = pd.merge(features, layer_features[layer], on='Run_ID', how='left')
        
        # マージ後の欠損値チェック
        print(f"NaN values after merging {layer}:", features.isnull().sum().sum())

# 最終的な特徴量の欠損値を確認
print("\nFinal feature set NaN check:")
print(features.isnull().sum())

# 残っている欠損値があれば0で埋める
features = features.fillna(0)

# Attack_Typeを数値に変換（既に変換済みの場合はスキップ）
if df['Attack_Type'].dtype == object:
    df['Attack_Type'] = df['Attack_Type'].map({'normal': 0, 'random': 1})

# Run_IDごとのAttack_Typeを取得
attack_type = df.groupby('Run_ID')['Attack_Type'].first().reset_index()

# 特徴量とラベルをマージ
data = pd.merge(features, attack_type, on='Run_ID', how='left')

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
    """クロスバリデーションを実行する関数"""
    try:
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search
        
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}")
        traceback.print_exc()
        return None

# プロット関数の修正
def plot_cv_comparison(cv_results_dict, output_dir, X_test, y_test):  # X_test と y_test を引数として追加
    try:
        # データの準備
        results_data = []
        
        for model_name, model in cv_results_dict.items():
            if model is not None:
                try:
                    # クロスバリデーションスコアの取得
                    cv_scores = model.cv_results_['mean_test_score']
                    valid_scores = cv_scores[~np.isnan(cv_scores)]
                    
                    # テストデータでの精度を計算
                    y_pred = model.predict(X_test)  # 渡されたX_testを使用
                    test_accuracy = accuracy_score(y_test, y_pred)  # 渡されたy_testを使用
                    
                    results_data.append({
                        'Model': model_name,
                        'Best_Score': model.best_score_,
                        'Mean_CV_Score': np.mean(valid_scores),
                        'Std_CV_Score': np.std(valid_scores),
                        'Test_Accuracy': test_accuracy
                    })
                except Exception as e:
                    print(f"Error processing model {model_name}: {str(e)}")
                    traceback.print_exc()
                    continue
        
        if not results_data:
            print("No valid results to plot")
            return None
            
        # DataFrameの作成
        results_df = pd.DataFrame(results_data)
        
        # 結果をCSVに保存
        results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
        
        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # CV Scoresのプロット
        x_pos = np.arange(len(results_df))
        ax1.bar(x_pos, results_df['Mean_CV_Score'],
                yerr=results_df['Std_CV_Score'],
                align='center', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(results_df['Model'], rotation=45)
        ax1.set_ylabel('Cross-validation Score')
        ax1.set_title('Model Comparison: Cross-validation Scores')
        
        # Test Accuracyのプロット
        ax2.bar(x_pos, results_df['Test_Accuracy'],
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
        
        return results_df
        
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        traceback.print_exc()
        return None

# モデルのトレーニングと評価部分の修正
def train_and_evaluate_model(model_name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv):
    print(f"Training and evaluating model: {model_name}")
    
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
        traceback.print_exc()
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
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 10, 20, 30, 40],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False]
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
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__gamma': [0, 0.1, 0.2]
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
    plot_cv_comparison(model_results, output_dir, X_test, y_test)
except Exception as e:
    print(f"Error plotting cross-validation scores comparison: {e}")

# 全モデルの精度をファイルに保存
try:
    accuracy_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Best Score': [model.best_score_ for model in model_results.values()],
        'Mean CV Score': [np.mean(model.cv_results_['mean_test_score']) for model in model_results.values()],
        'Std CV Score': [np.std(model.cv_results_['mean_test_score']) for model in model_results.values()]
    })

    accuracy_df.to_csv(os.path.join(output_dir, 'models_accuracy_comparison.csv'), index=False)
except Exception as e:
    print(f"Error saving models accuracy comparison: {e}")

# モデルの評価結果を保存
if model_results:
    results_df = plot_cv_comparison(model_results, output_dir, X_test, y_test)
    
    if results_df is not None:
        # 最良のモデルの選択（テスト精度に基づく）
        best_model_info = results_df.loc[results_df['Test_Accuracy'].idxmax()]
        best_model_name = best_model_info['Model']
        best_model = model_results[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best parameters: {best_model.best_params_}")
        print(f"Best cross-validation score: {best_model.best_score_:.3f}")
        print(f"Test accuracy: {best_model_info['Test_Accuracy']:.3f}")
        
        # 最良のモデルを保存
        joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
        
        # 最良のモデルの詳細な結果を保存
        with open(os.path.join(output_dir, 'best_model_results.txt'), 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Parameters: {best_model.best_params_}\n")
            f.write(f"Best Cross-validation Score: {best_model.best_score_:.3f}\n")
            f.write(f"Test Accuracy: {best_model_info['Test_Accuracy']:.3f}\n")

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
    """個別モデルの評価指標をプロットして保存する関数"""
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
        
        # 分類レポートを保存
        report = classification_report(y_test, y_pred)
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'classification_report': report
        }
        
    except Exception as e:
        print(f"Error plotting metrics for {model_name}: {str(e)}")
        traceback.print_exc()
        return None

from random_forest_model import train_random_forest
from gradient_boosting_model import train_gradient_boosting
from logistic_regression_model import train_logistic_regression
from svc_model import train_svc
from xgboost_model import train_xgboost

# ... データの前処理部分は変更なし ...

# モデルのトレーニング
model_results = {
    'RandomForest': train_random_forest(X_train, X_test, y_train, y_test, output_dir),
    'LogisticRegression': train_logistic_regression(X_train, X_test, y_train, y_test, output_dir),
    'SVM': train_svc(X_train, X_test, y_train, y_test, output_dir),
    'GradientBoosting': train_gradient_boosting(X_train, X_test, y_train, y_test, output_dir),
    'XGBoost': train_xgboost(X_train, X_test, y_train, y_test, output_dir)
}

# ... 結果の評価と保存部分は変更なし ...

# 特徴量セットの準備
def prepare_feature_sets(data):
    """First_Weightのみの特徴量セットとそれ以外の特徴量セットを準備"""
    # First_Weight関連の特徴量のみを抽出
    first_weight_cols = [col for col in data.columns if 'First_Weight' in col]
    first_weight_features = data[first_weight_cols]
    
    # First_Weight以外の特徴量を抽出
    other_feature_cols = [col for col in data.columns 
                         if col not in first_weight_cols + ['Run_ID', 'Attack_Type']]
    other_features = data[other_feature_cols]
    
    return first_weight_features, other_features

# 各特徴量セットでの学習と評価
def train_and_evaluate_both_sets(X_first_weight, X_other, y, output_dir):
    """両方の特徴量セットで学習と評価を行う"""
    # データ分割
    X_first_train, X_first_test, y_first_train, y_first_test = train_test_split(
        X_first_weight, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_other_train, X_other_test, y_other_train, y_other_test = train_test_split(
        X_other, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # First_Weightのみの特徴量での学習
    first_weight_results = {
        'RandomForest': train_random_forest(X_first_train, X_first_test, y_first_train, y_first_test, 
                                          os.path.join(output_dir, 'first_weight')),
        'LogisticRegression': train_logistic_regression(X_first_train, X_first_test, y_first_train, y_first_test, 
                                                      os.path.join(output_dir, 'first_weight')),
        'SVM': train_svc(X_first_train, X_first_test, y_first_train, y_first_test, 
                        os.path.join(output_dir, 'first_weight')),
        'GradientBoosting': train_gradient_boosting(X_first_train, X_first_test, y_first_train, y_first_test, 
                                                  os.path.join(output_dir, 'first_weight')),
        'XGBoost': train_xgboost(X_first_train, X_first_test, y_first_train, y_first_test, 
                                os.path.join(output_dir, 'first_weight'))
    }
    
    # その他の特徴量での学習
    other_features_results = {
        'RandomForest': train_random_forest(X_other_train, X_other_test, y_other_train, y_other_test, 
                                          os.path.join(output_dir, 'other_features')),
        'LogisticRegression': train_logistic_regression(X_other_train, X_other_test, y_other_train, y_other_test, 
                                                      os.path.join(output_dir, 'other_features')),
        'SVM': train_svc(X_other_train, X_other_test, y_other_train, y_other_test, 
                        os.path.join(output_dir, 'other_features')),
        'GradientBoosting': train_gradient_boosting(X_other_train, X_other_test, y_other_train, y_other_test, 
                                                  os.path.join(output_dir, 'other_features')),
        'XGBoost': train_xgboost(X_other_train, X_other_test, y_other_train, y_other_test, 
                                os.path.join(output_dir, 'other_features'))
    }
    
    return (first_weight_results, other_features_results, 
            X_first_test, X_other_test, y_first_test, y_other_test)

# 結果の比較と可視化
def compare_and_visualize_results(first_weight_results, other_features_results, output_dir):
    """両方の特徴量セットの結果を比較して可視化"""
    comparison_data = []
    
    for model_name in first_weight_results.keys():
        first_weight_score = first_weight_results[model_name].best_score_
        other_features_score = other_features_results[model_name].best_score_
        
        comparison_data.append({
            'Model': model_name,
            'First_Weight_Score': first_weight_score,
            'Other_Features_Score': other_features_score,
            'Score_Difference': other_features_score - first_weight_score
        })
    
    # 結果をDataFrameに変換
    comparison_df = pd.DataFrame(comparison_data)
    
    # 結果をCSVに保存
    comparison_df.to_csv(os.path.join(output_dir, 'comparison', 'feature_sets_comparison.csv'), 
                        index=False)
    
    # 結果の可視化
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['First_Weight_Score'], width, label='First Weight Only')
    plt.bar(x + width/2, comparison_df['Other_Features_Score'], width, label='Other Features')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy Score')
    plt.title('Performance Comparison: First Weight vs Other Features')
    plt.xticks(x, comparison_df['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'plots', 'feature_sets_comparison.png'))
    plt.close()
    
    # プロットデータの保存
    plot_data = {
        'x_positions': x.tolist(),
        'models': comparison_df['Model'].tolist(),
        'first_weight_scores': comparison_df['First_Weight_Score'].tolist(),
        'other_features_scores': comparison_df['Other_Features_Score'].tolist(),
        'width': width
    }
    save_plot_data(plot_data, 'comparison_bar', output_dir)
    
    return comparison_df

def calculate_model_metrics(model, X_test, y_test):
    """モデルの詳細な評価指標を計算"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'fpr': None,
        'tpr': None,
        'roc_auc': None
    }
    
    # ROC曲線の計算
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_auc'] = auc(fpr, tpr)
    
    return metrics

def save_plot_data(plot_data, plot_type, output_dir):
    """プロット用データをCSVとして保存"""
    plot_data_dir = os.path.join(output_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)
    
    if plot_type == 'comparison_bar':
        # 棒グラフ用データの保存
        pd.DataFrame(plot_data).to_csv(
            os.path.join(plot_data_dir, 'feature_sets_comparison_data.csv'),
            index=False
        )
        
        # プロット設定の保存
        plot_settings = {
            'title': 'Performance Comparison: First Weight vs Other Features',
            'xlabel': 'Models',
            'ylabel': 'Accuracy Score',
            'legend_labels': ['First Weight Only', 'Other Features']
        }
        pd.DataFrame([plot_settings]).to_csv(
            os.path.join(plot_data_dir, 'feature_sets_comparison_settings.csv'),
            index=False
        )
    
    elif plot_type == 'roc':
        # ROC曲線用データの保存
        for model_name, metrics in plot_data.items():
            roc_data = pd.DataFrame({
                'FPR': metrics['fpr'],
                'TPR': metrics['tpr'],
                'AUC': metrics['roc_auc'],
                'Model': model_name
            })
            roc_data.to_csv(
                os.path.join(plot_data_dir, f'roc_curve_data_{model_name}.csv'),
                index=False
            )
        
        # プロット設定の保存
        plot_settings = {
            'title': 'ROC Curves Comparison',
            'xlabel': 'False Positive Rate',
            'ylabel': 'True Positive Rate'
        }
        pd.DataFrame([plot_settings]).to_csv(
            os.path.join(plot_data_dir, 'roc_curves_settings.csv'),
            index=False
        )

def plot_roc_curves(models_metrics, title, output_path):
    """複数モデルのROC曲線を1つのグラフにプロット"""
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in models_metrics.items():
        plt.plot(metrics['fpr'], metrics['tpr'], 
                label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # プロットデータの保存
    output_dir = os.path.dirname(output_path)
    save_plot_data(models_metrics, 'roc', output_dir)

def perform_statistical_analysis(first_weight_metrics, other_features_metrics, output_dir):
    """両特徴量セット間の統計的分析"""
    analysis_results = {}
    
    # 各モデルの性能差の分析
    for model_name in first_weight_metrics.keys():
        first_weight_perf = first_weight_metrics[model_name]
        other_features_perf = other_features_metrics[model_name]
        
        # 性能指標の差分
        performance_diff = {
            'accuracy_diff': other_features_perf['accuracy'] - first_weight_perf['accuracy'],
            'roc_auc_diff': other_features_perf['roc_auc'] - first_weight_perf['roc_auc'],
        }
        
        # 混同行列の要素ごとの差分
        cm_diff = other_features_perf['confusion_matrix'] - first_weight_perf['confusion_matrix']
        
        analysis_results[model_name] = {
            'performance_differences': performance_diff,
            'confusion_matrix_difference': cm_diff,
        }
    
    # 結果をファイルに保存
    with open(os.path.join(output_dir, 'statistical_analysis.txt'), 'w') as f:
        f.write("Statistical Analysis Results\n")
        f.write("===========================\n\n")
        
        for model_name, results in analysis_results.items():
            f.write(f"\n{model_name}\n{'-' * len(model_name)}\n")
            f.write(f"Accuracy improvement: {results['performance_differences']['accuracy_diff']:.4f}\n")
            f.write(f"ROC AUC improvement: {results['performance_differences']['roc_auc_diff']:.4f}\n")
            f.write("\nConfusion Matrix Difference:\n")
            f.write(str(results['confusion_matrix_difference']))
            f.write("\n")
    
    return analysis_results

def setup_output_directories(base_dir):
    """出力ディレクトリの構造をセットアップ"""
    directories = {
        'first_weight': os.path.join(base_dir, 'first_weight'),
        'other_features': os.path.join(base_dir, 'other_features'),
        'comparison': os.path.join(base_dir, 'comparison'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'plots': os.path.join(base_dir, 'plots'),
        'models': os.path.join(base_dir, 'models'),
        'statistical_analysis': os.path.join(base_dir, 'statistical_analysis')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def save_metrics_to_csv(metrics, model_name, feature_set, output_dir):
    """評価指標をCSVファイルとして保存"""
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC_AUC', 'Precision', 'Recall', 'F1_Score'],
        'Value': [
            metrics['accuracy'],
            metrics['roc_auc'],
            metrics['classification_report']['weighted avg']['precision'],
            metrics['classification_report']['weighted avg']['recall'],
            metrics['classification_report']['weighted avg']['f1-score']
        ]
    })
    
    # 混同行列をDataFrameに変換
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        columns=['Predicted_Negative', 'Predicted_Positive'],
        index=['Actual_Negative', 'Actual_Positive']
    )
    
    # メトリクスの保存
    metrics_df.to_csv(
        os.path.join(output_dir, 'metrics', f'{feature_set}_{model_name}_metrics.csv'),
        index=False
    )
    
    # 混同行列の保存
    cm_df.to_csv(
        os.path.join(output_dir, 'metrics', f'{feature_set}_{model_name}_confusion_matrix.csv')
    )
    
    # ROC曲線のデータを保存
    roc_df = pd.DataFrame({
        'FPR': metrics['fpr'],
        'TPR': metrics['tpr']
    })
    roc_df.to_csv(
        os.path.join(output_dir, 'metrics', f'{feature_set}_{model_name}_roc_curve.csv'),
        index=False
    )

def evaluate_and_compare_models(first_weight_results, other_features_results, 
                              X_first_test, X_other_test, y_first_test, y_other_test, output_dir):
    """モデルの評価と比較を行う"""
    first_weight_metrics = {}
    other_features_metrics = {}
    
    # 各モデルの評価指標を計算と保存
    for model_name, model in first_weight_results.items():
        metrics = calculate_model_metrics(model, X_first_test, y_first_test)
        first_weight_metrics[model_name] = metrics
        save_metrics_to_csv(metrics, model_name, 'first_weight', output_dir)
    
    for model_name, model in other_features_results.items():
        metrics = calculate_model_metrics(model, X_other_test, y_other_test)
        other_features_metrics[model_name] = metrics
        save_metrics_to_csv(metrics, model_name, 'other_features', output_dir)
    
    # ROC曲線のプロット
    plot_roc_curves(
        first_weight_metrics,
        'ROC Curves (First Weight Features)',
        os.path.join(output_dir, 'plots', 'roc_curves_first_weight.png')
    )
    
    plot_roc_curves(
        other_features_metrics,
        'ROC Curves (Other Features)',
        os.path.join(output_dir, 'plots', 'roc_curves_other_features.png')
    )
    
    # 詳細な比較結果をDataFrameとして保存
    comparison_data = []
    for model_name in first_weight_metrics.keys():
        first_metrics = first_weight_metrics[model_name]
        other_metrics = other_features_metrics[model_name]
        
        comparison_data.append({
            'Model': model_name,
            'Feature_Set': 'First_Weight',
            'Accuracy': first_metrics['accuracy'],
            'ROC_AUC': first_metrics['roc_auc'],
            'Precision': first_metrics['classification_report']['weighted avg']['precision'],
            'Recall': first_metrics['classification_report']['weighted avg']['recall'],
            'F1_Score': first_metrics['classification_report']['weighted avg']['f1-score']
        })
        
        comparison_data.append({
            'Model': model_name,
            'Feature_Set': 'Other_Features',
            'Accuracy': other_metrics['accuracy'],
            'ROC_AUC': other_metrics['roc_auc'],
            'Precision': other_metrics['classification_report']['weighted avg']['precision'],
            'Recall': other_metrics['classification_report']['weighted avg']['recall'],
            'F1_Score': other_metrics['classification_report']['weighted avg']['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(
        os.path.join(output_dir, 'comparison', 'detailed_metrics_comparison.csv'),
        index=False
    )
    
    # 統計的分析の実行と保存
    analysis_results = perform_statistical_analysis(
        first_weight_metrics,
        other_features_metrics,
        os.path.join(output_dir, 'statistical_analysis')
    )
    
    # 分析結果をCSVとして保存
    analysis_df = pd.DataFrame([
        {
            'Model': model_name,
            'Accuracy_Improvement': results['performance_differences']['accuracy_diff'],
            'ROC_AUC_Improvement': results['performance_differences']['roc_auc_diff']
        }
        for model_name, results in analysis_results.items()
    ])
    
    analysis_df.to_csv(
        os.path.join(output_dir, 'statistical_analysis', 'performance_improvements.csv'),
        index=False
    )
    
    return first_weight_metrics, other_features_metrics, analysis_results

# メイン処理
if __name__ == "__main__":
    # 出力ディレクトリの構造をセットアップ
    directories = setup_output_directories(output_dir)
    
    # 必要なimportを追加
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
    
    # 特徴量セットの準備
    X_first_weight, X_other = prepare_feature_sets(data)
    
    # 両方の特徴量セットで学習と評価
    (first_weight_results, other_features_results,
     X_first_test, X_other_test, y_first_test, y_other_test) = train_and_evaluate_both_sets(
        X_first_weight, X_other, y, output_dir
    )
    
    # 詳細な評価と比較
    first_weight_metrics, other_features_metrics, analysis_results = evaluate_and_compare_models(
        first_weight_results, 
        other_features_results,
        X_first_test, 
        X_other_test, 
        y_first_test, 
        y_other_test,
        output_dir  # output_dirを追加
    )
    
    # 結果の比較と可視化
    comparison_results = compare_and_visualize_results(
        first_weight_results, 
        other_features_results, 
        output_dir
    )
    
    print("\n特徴量セット間の性能比較結果:")
    print(comparison_results.to_string(index=False))