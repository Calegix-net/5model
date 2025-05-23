import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid tkinter errors when saving figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve, RepeatedStratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, make_scorer, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

import joblib
import shap

# Import Pipeline and SMOTE from imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Import XGBoost
from xgboost import XGBClassifier

# Load data
df = pd.read_csv('dataset.csv', low_memory=False)

# Map 'Attack_Type' to numeric
df['Attack_Type'] = df['Attack_Type'].map({'normal': 0, 'random': 1})

# Check missing values after mapping
missing_attack_type = df['Attack_Type'].isnull().sum()
print(f"Number of missing Attack_Type after mapping: {missing_attack_type}")

# Remove missing values
df = df.dropna(subset=['Attack_Type'])

# Check missing values again
missing_attack_type_after = df['Attack_Type'].isnull().sum()
print(f"Missing Attack_Type count after dropping NaNs: {missing_attack_type_after}")

# Also check for other potential missing values
total_missing = df.isnull().sum()
print("Number of missing values per column:")
print(total_missing)

# Fill missing numeric columns with 0
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

# Output directory
output_dir = 'output_files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Display unique Layer values
print("Unique Layer values:")
print(df['Layer'].unique())

# Specify keywords for middle and final layers
middle_layer_keywords = ['layer.3']
final_layer_keywords = ['layer.5']

# Filter data
middle_layers_df = df[df['Layer'].str.contains('|'.join(middle_layer_keywords), na=False)]
final_layers_df = df[df['Layer'].str.contains('|'.join(final_layer_keywords), na=False)]

# Check that DataFrames are not empty
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

# Function to calculate change rate
def calculate_change_rate(group):
    group = group.sort_values('Round')
    group['Mean_Variance_Change_Rate'] = group['Mean_Variance'].pct_change(fill_method=None)
    return group

# Use StratifiedKFold to maintain stratification
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ignore DataFrameGroupBy deprecation warnings
import warnings
# Suppress deprecation and future warnings from pandas and scikit-learn internals
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Apply function (select only necessary columns to avoid warnings)
middle_layers_df = middle_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate)[['Run_ID', 'Round', 'Mean_Variance', 'Mean_Variance_Change_Rate']].reset_index(drop=True)
final_layers_df = final_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate)[['Run_ID', 'Round', 'Mean_Variance', 'Mean_Variance_Change_Rate']].reset_index(drop=True)

# Compute features
grouped = df.groupby('Run_ID')
features = grouped.agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Number_of_outliers': ['mean', 'std', 'min', 'max', 'median'],
    'Layer': 'nunique'
}).reset_index()

features.columns = ['Run_ID'] + ['_'.join(col).strip('_') for col in features.columns.values[1:]]
features = features.rename(columns={'Layer_nunique': 'Num_Layers'})

# Middle layer features
middle_features = middle_layers_df.groupby('Run_ID').agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Mean_Variance_Change_Rate': ['mean', 'std', 'min', 'max', 'median']
}).reset_index()

middle_features.columns = ['Run_ID'] + ['Middle_' + '_'.join(col).strip('_') for col in middle_features.columns.values[1:]]

# Final layer features
final_features = final_layers_df.groupby('Run_ID').agg({
    'Mean_Variance': ['mean', 'std', 'min', 'max', 'median'],
    'Mean_Variance_Change_Rate': ['mean', 'std', 'min', 'max', 'median']
}).reset_index()

final_features.columns = ['Run_ID'] + ['Final_' + '_'.join(col).strip('_') for col in final_features.columns.values[1:]]

# Merge features
features = pd.merge(features, middle_features, on='Run_ID', how='left')
features = pd.merge(features, final_features, on='Run_ID', how='left')

# Get 'Attack_Type' for each 'Run_ID'
attack_type = grouped['Attack_Type'].first().reset_index()

# Merge features and labels
data = pd.merge(features, attack_type, on='Run_ID')

# Check missing values after merge
missing_after_merge = data['Attack_Type'].isnull().sum()
print(f"Number of missing Attack_Type after merge: {missing_after_merge}")

if missing_after_merge > 0:
    print("Missing values occurred after merge. Please check Run_ID consistency.")
    # Display missing Run_IDs
    missing_runs = data[data['Attack_Type'].isnull()]['Run_ID'].unique()
    print(f"Run_IDs with missing Attack_Type: {missing_runs}")
    # Remove missing Run_IDs
    data = data.dropna(subset=['Attack_Type'])
    print(f"Data shape after dropping missing Run_IDs: {data.shape}")

# Prepare feature matrix X and label vector y
feature_cols = [col for col in data.columns if col not in ['Run_ID', 'Attack_Type', 'Weight_Std']]
X = data[feature_cols]
y = data['Attack_Type']

# Confirm that y has no missing values
if y.isnull().sum() > 0:
    print("Error: y still contains missing values. Please review data preprocessing.")
    import sys
    sys.exit()

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Display statistics of y
print("y descriptive statistics:")
print(y.describe())

# Display unique values in y
print("Unique values of y:")
print(y.unique())

# Check if y contains NaN
print(f"Does y contain NaN: {y.isnull().any()}")

# Adjust data balance (remove global SMOTE application)
# Do not apply SMOTE here; use it only within pipelines

# Correlation analysis and saving
X_with_target = X.copy()
X_with_target['Attack_Type'] = y
corr_matrix = X_with_target.corr()

# Save correlation matrix
corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))

# Plot and save heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# Plot feature distributions
for col in feature_cols:
    plt.figure()
    sns.histplot(X[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(output_dir, f'feature_distribution_{col}.png'))
    plt.close()

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Split data (maintain stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# y_train に欠損値がないことを確認
print(f"Count of missing in y_train: {y_train.isnull().sum()}")

if y_train.isnull().sum() > 0:
    print("Error: y_train contains missing values.")
    import sys
    sys.exit()

# Remove global scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Define models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Dictionary to store results for each model
model_results = {}

def perform_cross_validation(pipeline, param_grid, X, y, cv):
    """Function to perform cross-validation"""
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

def plot_cv_comparison(cv_results_dict, output_dir, X_test, y_test):
    # Modified plotting function: accepts X_test and y_test as arguments
    try:
        # Prepare data
        results_data = []
        
        for model_name, model in cv_results_dict.items():
            if model is not None:
                try:
                    # Get cross-validation scores
                    cv_scores = model.cv_results_['mean_test_score']
                    valid_scores = cv_scores[~np.isnan(cv_scores)]
                    
                    # Calculate accuracy on test data
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

def train_and_evaluate_model(model_name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv):
    # Modified model training and evaluation section
    print(f"Training and evaluating model: {model_name}")
    
    try:
        # Execute grid search
        grid_search = perform_cross_validation(pipeline, param_grid, X_train, y_train, cv)
        
        if grid_search is None:
            print(f"Training and evaluation failed for {model_name}")
            return None
            
        # Display best parameters and accuracy
        print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        
        # Evaluate on test data
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

    # Create pipeline (exclude SMOTE)
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Modified cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Apply SMOTE individually
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define hyperparameters (model-specific)
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

    # Train and evaluate model
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

try:
    # Compare cross-validation scores across all models
    plot_cv_comparison(model_results, output_dir, X_test, y_test)
except Exception as e:
    print(f"Error plotting cross-validation scores comparison: {e}")

try:
    # Save accuracy of all models to file
    accuracy_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Best Score': [model.best_score_ for model in model_results.values()],
        'Mean CV Score': [np.mean(model.cv_results_['mean_test_score']) for model in model_results.values()],
        'Std CV Score': [np.std(model.cv_results_['mean_test_score']) for model in model_results.values()]
    })

    accuracy_df.to_csv(os.path.join(output_dir, 'models_accuracy_comparison.csv'), index=False)
except Exception as e:
    print(f"Error saving models accuracy comparison: {e}")

if model_results:
    # Save model evaluation results
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

print("All model training and evaluation completed. Results saved in 'output_files' directory.")

def plot_feature_importance(model, feature_names, output_dir):
    try:
        # Get feature importances
        importances = model.best_estimator_.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Display only top 20 features
        n_features = min(20, len(importances))
        
        plt.figure(figsize=(12, 6))
        plt.title("Top 20 Feature Importances")
        plt.bar(range(n_features), importances[indices][:n_features])
        plt.xticks(range(n_features), [feature_names[i] for i in indices][:n_features], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save feature importances to CSV
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # Display top 10 most important features
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
    """Function to plot and save individual model evaluation metrics"""
    try:
        # Obtain predictions and probabilities
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
        
        # Save classification report
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

if model_results:
    # Modified section to save model evaluation results (around line 444)
    results_df = plot_cv_comparison(model_results, output_dir, X_test, y_test)
    
    if results_df is not None:
        # Select best model (based on test accuracy)
        best_model_info = results_df.loc[results_df['Test_Accuracy'].idxmax()]
        best_model_name = best_model_info['Model']
        best_model = model_results[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best parameters: {best_model.best_params_}")
        print(f"Best cross-validation score: {best_model.best_score_:.3f}")
        print(f"Test accuracy: {best_model_info['Test_Accuracy']:.3f}")
        
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
            f.write(f"Test Accuracy: {best_model_info['Test_Accuracy']:.3f}\n")
            f.write("\nDetailed CV Results:\n")
            f.write(f"Mean CV Score: {best_model_info['Mean_CV_Score']:.3f}\n")
            f.write(f"Std CV Score: {best_model_info['Std_CV_Score']:.3f}\n")
            
            if metrics_results:
                f.write("\nDetailed Metrics:\n")
                f.write(f"ROC AUC Score: {metrics_results['roc_auc']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(metrics_results['classification_report'])

def plot_all_models_metrics(model_results, X_test, y_test, output_dir):
    """全モデルの評価指標をプロットして保存する関数"""
    try:
        for model_name, model in model_results.items():
            if model is not None:
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
                    
                    # 分類レポートを保存
                    report = classification_report(y_test, y_pred)
                    with open(os.path.join(output_dir, f'classification_report_{model_name}.txt'), 'w') as f:
                        f.write(f"Classification Report for {model_name}\n")
                        f.write("="*50 + "\n")
                        f.write(report)
                        f.write("\n\nConfusion Matrix:\n")
                        f.write(str(cm))
                        
                        # ROC AUCスコアを計算して保存
                        roc_auc = roc_auc_score(y_test, y_prob)
                        f.write(f"\n\nROC AUC Score: {roc_auc:.4f}")
                except Exception as e:
                    print(f"Error processing model {model_name}: {str(e)}")
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"Error in plotting metrics: {str(e)}")
        traceback.print_exc()

def plot_all_models_roc(model_results, X_test, y_test, output_dir):
    """全モデルのROC曲線を1つのグラフにプロットする関数"""
    try:
        plt.figure(figsize=(10, 8))
        
        for model_name, model in model_results.items():
            if model is not None:
                # 予測確率を取得
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # ROC曲線を計算
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # ROC曲線をプロット
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # グラフの設定
        plt.plot([0, 1], [0, 1], 'k--')  # 対角線
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        
        # グラフを保存
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_models_roc_curves.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error in plotting ROC curves: {str(e)}")
        traceback.print_exc()

print("\n=== Starting retraining with only important features ===\n")

# 重要度が0.030以上の特徴量を選択
important_features = []
feature_importance_file = os.path.join(output_dir, 'feature_importance_details.txt')
if os.path.exists(feature_importance_file):
    with open(feature_importance_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # ヘッダー行をスキップ
            if line.strip():
                feature, importance = line.split(': ')
                if float(importance) >= 0.030:
                    important_features.append(feature)
else:
    print("Feature importance file not found. Using all features for analysis.")
    # Use all features if feature importance file doesn't exist
    important_features = X.columns.tolist()

print(f"Selected important features ({len(important_features)}):")
for feature in important_features:
    print(f"- {feature}")

# 新しい出力ディレクトリを作成
output_dir_important = os.path.join(output_dir, 'important_features_results')
if not os.path.exists(output_dir_important):
    os.makedirs(output_dir_important)

# 重要な特徴量のみを使用してデータを準備
X_important = X[important_features]

# データを分割（層化を維持）
X_train_important, X_test_important, y_train_important, y_test_important = train_test_split(
    X_important, y, test_size=0.2, random_state=42, stratify=y
)

# モデルの結果を保存する新しいディクショナリ
model_results_important = {}

# 各モデルの再学習
for model_name, model in models.items():
    print(f"\nTraining {model_name} with important features...")

    # パイプラインの作成
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # SMOTEを適用
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_important, y_train_important)

    # ハイパーパラメータグリッドの設定（既存のものを使用）
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

    # モデルのトレーニングと評価
    model_result = train_and_evaluate_model(
        model_name, 
        pipeline, 
        param_grid, 
        X_train_resampled, 
        X_test_important, 
        y_train_resampled, 
        y_test_important, 
        cv
    )

    if model_result is not None:
        model_results_important[model_name] = model_result

# 結果の評価と保存
if model_results_important:
    results_df = plot_cv_comparison(model_results_important, output_dir_important, X_test_important, y_test_important)
    
    if results_df is not None:
        # 全モデルのメトリクスを保存
        plot_all_models_metrics(model_results_important, X_test_important, y_test_important, output_dir_important)
        
        # 全モデルのROC曲線を1つのグラフに保存
        plot_all_models_roc(model_results_important, X_test_important, y_test_important, output_dir_important)
        
        # 最良のモデルの選択（テスト精度に基づく）
        best_model_info = results_df.loc[results_df['Test_Accuracy'].idxmax()]
        best_model_name = best_model_info['Model']
        best_model = model_results_important[best_model_name]
        
        print(f"\nBest performing model with important features: {best_model_name}")
        print(f"Best parameters: {best_model.best_params_}")
        print(f"Best cross-validation score: {best_model.best_score_:.3f}")
        print(f"Test accuracy: {best_model_info['Test_Accuracy']:.3f}")
        
        # 最良のモデルを保存
        joblib.dump(best_model, os.path.join(output_dir_important, 'best_model.joblib'))
        
        # 最良のモデルの評価指標を保存
        metrics_results = plot_and_save_metrics(
            best_model.best_estimator_,
            X_test_important,
            y_test_important,
            best_model_name,
            output_dir_important
        )
        
        # 最良のモデルの詳細な結果を保存
        with open(os.path.join(output_dir_important, 'best_model_results.txt'), 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Parameters: {best_model.best_params_}\n")
            f.write(f"Best Cross-validation Score: {best_model.best_score_:.3f}\n")
            f.write(f"Test Accuracy: {best_model_info['Test_Accuracy']:.3f}\n")
            f.write("\nSelected Important Features:\n")
            for feature in important_features:
                f.write(f"- {feature}\n")
            f.write("\nDetailed CV Results:\n")
            f.write(f"Mean CV Score: {best_model_info['Mean_CV_Score']:.3f}\n")
            f.write(f"Std CV Score: {best_model_info['Std_CV_Score']:.3f}\n")
            
            if metrics_results:
                f.write("\nDetailed Metrics:\n")
                f.write(f"ROC AUC Score: {metrics_results['roc_auc']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(metrics_results['classification_report'])

print("\n=== Retraining with only important features completed ===\n")

def perform_shap_analysis(model, X_train, X_test, y_test, feature_names, model_name, output_dir):
    """SHAP解析を実行し、各種プロットを生成する関数"""
    try:
        print(f"\nPerforming SHAP analysis for {model_name}...")
        
        # SHAPディレクトリを作成
        shap_dir = os.path.join(output_dir, f'shap_analysis_{model_name}')
        if not os.path.exists(shap_dir):
            os.makedirs(shap_dir)
        
        # モデルタイプに応じてExplainerを選択
        if model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            # Tree-based modelsにはTreeExplainerを使用
            explainer = shap.TreeExplainer(model.best_estimator_.named_steps['classifier'])
            
            # 標準化されたデータを取得
            scaler = model.best_estimator_.named_steps['scaler']
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # SHAP値を計算
            shap_values_train = explainer.shap_values(X_train_scaled)
            shap_values_test = explainer.shap_values(X_test_scaled)
            
            # 二値分類の場合、適切なクラスのSHAP値を選択
            if isinstance(shap_values_test, list):
                shap_values_train = shap_values_train[1]  # Positive class
                shap_values_test = shap_values_test[1]    # Positive class
        
        else:
            # 他のモデルにはKernelExplainerを使用
            def model_predict(X):
                return model.predict_proba(X)[:, 1]
            
            # 背景データとしてトレーニングデータのサンプルを使用
            background_sample = shap.sample(X_train, 100)
            explainer = shap.KernelExplainer(model_predict, background_sample)
            
            # テストデータからサンプルを選択してSHAP値を計算
            test_sample = shap.sample(X_test, 50)
            shap_values_test = explainer.shap_values(test_sample)
            
            X_test_scaled = test_sample
        
        # 1. Waterfall plot (最初のテストサンプル)
        plt.figure(figsize=(12, 8))
        if model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            expected_value = explainer.expected_value
        else:
            expected_value = explainer.expected_value
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_test[0],
                base_values=expected_value,
                data=X_test_scaled[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - {model_name} (First Test Sample)')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'waterfall_plot_{model_name}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values_test,
                base_values=expected_value if isinstance(expected_value, (int, float)) else expected_value,
                data=X_test_scaled,
                feature_names=feature_names
            ),
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Beeswarm Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'beeswarm_plot_{model_name}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Scatter plots for top 10 features
        mean_abs_shap = np.mean(np.abs(shap_values_test), axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-10:]  # Top 10 features
        
        for i, feature_idx in enumerate(top_features_idx):
            plt.figure(figsize=(10, 6))
            shap.plots.scatter(
                shap.Explanation(
                    values=shap_values_test[:, feature_idx],
                    base_values=expected_value if isinstance(expected_value, (int, float)) else expected_value,
                    data=X_test_scaled[:, feature_idx],
                    feature_names=[feature_names[feature_idx]]
                ),
                show=False
            )
            plt.title(f'SHAP Scatter Plot - {model_name} - {feature_names[feature_idx]}')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f'scatter_plot_{model_name}_{feature_names[feature_idx].replace("/", "_")}.png'), 
                        bbox_inches='tight', dpi=300)
            plt.close()
        
        # 4. Summary plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_test, 
            X_test_scaled, 
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Summary Plot (Bar) - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'summary_plot_bar_{model_name}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 5. Summary plot (dot)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_test, 
            X_test_scaled, 
            feature_names=feature_names,
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Summary Plot (Dot) - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'summary_plot_dot_{model_name}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # SHAP値の統計情報を保存
        shap_stats = {
            'mean_abs_shap': mean_abs_shap,
            'feature_names': feature_names,
            'top_10_features': [feature_names[i] for i in top_features_idx]
        }
        
        # SHAP値の詳細を保存
        with open(os.path.join(shap_dir, f'shap_analysis_results_{model_name}.txt'), 'w') as f:
            f.write(f"SHAP Analysis Results for {model_name}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Top 10 Most Important Features (by mean absolute SHAP value):\n")
            sorted_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                f.write(f"{i+1:2d}. {feature}: {importance:.6f}\n")
            
            f.write(f"\nExpected Value: {expected_value}\n")
            f.write(f"Number of test samples analyzed: {len(shap_values_test)}\n")
            f.write(f"Number of features: {len(feature_names)}\n")
        
        # SHAP値をCSVとして保存
        shap_df = pd.DataFrame(shap_values_test, columns=feature_names)
        shap_df.to_csv(os.path.join(shap_dir, f'shap_values_{model_name}.csv'), index=False)
        
        print(f"SHAP analysis completed for {model_name}. Results saved in {shap_dir}")
        
        return shap_stats
        
    except Exception as e:
        print(f"Error in SHAP analysis for {model_name}: {str(e)}")
        traceback.print_exc()
        return None

# SHAP解析の実行
if model_results_important and len(model_results_important) > 0:
    print("\n=== Starting SHAP analysis ===\n")
    
    # Tree-based modelsでSHAP解析を実行
    shap_results = {}
    for model_name, model in model_results_important.items():
        if model_name in ['RandomForest', 'GradientBoosting', 'XGBoost'] and model is not None:
            shap_stats = perform_shap_analysis(
                model,
                X_train_important,
                X_test_important,
                y_test_important,
                important_features,
                model_name,
                output_dir_important
            )
            
            if shap_stats is not None:
                shap_results[model_name] = shap_stats
    
    # その他のモデル(SVM, LogisticRegression)でもSHAP解析を実行
    for model_name, model in model_results_important.items():
        if model_name in ['SVM', 'LogisticRegression'] and model is not None:
            print(f"\nPerforming SHAP analysis for {model_name} (may take longer due to KernelExplainer)...")
            shap_stats = perform_shap_analysis(
                model,
                X_train_important,
                X_test_important,
                y_test_important,
                important_features,
                model_name,
                output_dir_important
            )
            
            if shap_stats is not None:
                shap_results[model_name] = shap_stats
    
    # SHAP解析結果の比較
    if shap_results:
        print("\n=== SHAP Analysis Results Summary ===\n")
        
        # 各モデルの主要特徴量を比較
        comparison_results = {}
        for model_name, stats in shap_results.items():
            top_10_features = stats['top_10_features'][::-1]  # Top 10 in descending order
            comparison_results[model_name] = top_10_features
            
            print(f"Top 10 features for {model_name}:")
            for i, feature in enumerate(top_10_features):
                print(f"  {i+1:2d}. {feature}")
            print()
        
        # 比較結果をファイルに保存
        with open(os.path.join(output_dir_important, 'shap_comparison_summary.txt'), 'w') as f:
            f.write("SHAP Analysis Comparison Summary\n")
            f.write("="*40 + "\n\n")
            
            for model_name, top_features in comparison_results.items():
                f.write(f"{model_name} - Top 10 Features:\n")
                for i, feature in enumerate(top_features):
                    f.write(f"  {i+1:2d}. {feature}\n")
                f.write("\n")
            
            # 共通の重要特徴量を特定
            all_features = []
            for features in comparison_results.values():
                all_features.extend(features)
            
            from collections import Counter
            feature_counts = Counter(all_features)
            common_features = [feature for feature, count in feature_counts.items() if count > 1]
            
            f.write("Common Important Features Across Models:\n")
            for feature in sorted(common_features, key=lambda x: feature_counts[x], reverse=True):
                f.write(f"  - {feature} (appears in {feature_counts[feature]} models)\n")
    
    print("\n=== SHAP analysis completed ===\n")
else:
    print("\nNo models available for SHAP analysis.")

print("All analyses completed successfully!")