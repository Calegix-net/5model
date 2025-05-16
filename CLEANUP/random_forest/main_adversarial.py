import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, cross_validate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler

import joblib

# データを読み込む
df = pd.read_csv('dataset.csv')

# 'Attack_Type'を数値にマッピング
df['Attack_Type'] = df['Attack_Type'].map({'normal': 0, 'adversarial': 1})

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
    group['Mean_Variance_Change_Rate'] = group['Mean_Variance'].pct_change()
    return group

# 関数を適用
middle_layers_df = middle_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate).reset_index(drop=True)
final_layers_df = final_layers_df.groupby(['Run_ID'], group_keys=False).apply(calculate_change_rate).reset_index(drop=True)

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

# 特徴量行列Xとラベルベクトルyを準備
feature_cols = [col for col in data.columns if col not in ['Run_ID', 'Attack_Type']]
X = data[feature_cols]
y = data['Attack_Type']

# 欠損値を処理
X = X.fillna(0)

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
cv = 5  # 分割数

# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初期モデルのトレーニング
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 予測
y_pred = clf.predict(X_test)

# モデルを評価
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy with new features:', accuracy)

# 特徴量の重要度を取得
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# 特徴量の重要度を保存
feature_importances_df = pd.DataFrame({
    'Feature': feature_names[indices],
    'Importance': importances[indices]
})
feature_importances_df.to_csv(os.path.join(output_dir, 'feature_importances_with_new_features.csv'), index=False)

# 特徴量の重要度をプロットして保存
plt.figure(figsize=(12, 8))
plt.title("Feature Importances with New Features")
plt.bar(range(len(feature_names)), importances[indices], align="center")
plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importances_with_new_features.png'))
plt.close()

# 初期モデルの学習曲線をプロット（交差検証前）
max_train_samples_initial = int((cv - 1) / cv * X_train.shape[0])
train_sizes_initial = np.linspace(10, max_train_samples_initial, 10, dtype=int)

train_sizes_initial, train_scores_initial, val_scores_initial = learning_curve(
    clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1,
    train_sizes=train_sizes_initial
)

train_scores_mean_initial = np.mean(train_scores_initial, axis=1)
val_scores_mean_initial = np.mean(val_scores_initial, axis=1)

plt.figure()
plt.plot(train_sizes_initial, train_scores_mean_initial, 'o-', color='r', label='Training score')
plt.plot(train_sizes_initial, val_scores_mean_initial, 'o-', color='g', label='Validation score')
plt.title('Learning Curve before Cross-Validation')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig(os.path.join(output_dir, 'learning_curve_before_cv.png'))
plt.close()

# ハイパーパラメータの調整
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1,
                           scoring='accuracy',
                           error_score='raise')

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# 最適なパラメータでモデルを再学習
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Best Model Accuracy with new features:', accuracy)

# トレーニング精度と検証精度を計算
y_train_pred = best_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_test, y_pred)
print('Training Accuracy:', train_accuracy)
print('Validation Accuracy:', val_accuracy)

# 精度をファイルに保存
with open(os.path.join(output_dir, 'model_accuracy.txt'), 'w') as f:
    f.write(f'Training Accuracy: {train_accuracy}\n')
    f.write(f'Validation Accuracy: {val_accuracy}\n')

# 交差検証による評価
scores = cross_val_score(best_clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())

# 学習曲線をプロット（交差検証後のモデル）
max_train_samples = int((cv - 1) / cv * X.shape[0])
train_sizes = np.linspace(10, max_train_samples, 10, dtype=int)

train_sizes, train_scores, val_scores = learning_curve(
    best_clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1,
    train_sizes=train_sizes
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation score')
plt.title('Learning Curve after Cross-Validation')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig(os.path.join(output_dir, 'learning_curve_after_cv.png'))
plt.close()

# 交差検証のトレーニング精度と検証精度を取得
cv_results = cross_validate(
    best_clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1,
    return_train_score=True
)

print('Cross-validation training scores:', cv_results['train_score'])
print('Cross-validation validation scores:', cv_results['test_score'])
print('Mean training score:', np.mean(cv_results['train_score']))
print('Mean validation score:', np.mean(cv_results['test_score']))

# 結果をファイルに保存
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)

# 初期モデルの混同行列と分類レポート
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix - Initial Model')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_initial.png'))
plt.close()

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, 'classification_report_initial.csv'))

# 初期モデルのROC曲線
y_prob = best_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Initial Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve_initial.png'))
plt.close()

# 相関が低い・重要度が低い特徴量を特定
low_corr_features = corr_matrix['Attack_Type'][abs(corr_matrix['Attack_Type']) < 0.1].index.tolist()
if 'Attack_Type' in low_corr_features:
    low_corr_features.remove('Attack_Type')  # ターゲット変数を除外

low_importance_features = feature_importances_df[feature_importances_df['Importance'] < 0.01]['Feature'].tolist()

features_to_drop = list(set(low_corr_features + low_importance_features))

print(f"Features to drop: {features_to_drop}")

# 特徴量を削除
X_reduced = X.drop(columns=features_to_drop)

# データセットのサンプル数を取得
n_samples = X_reduced.shape[0]
print("Total number of samples:", n_samples)
max_train_samples = int((cv - 1) / cv * n_samples)
print("Maximum number of training samples per fold:", max_train_samples)

# データを再分割
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# 標準化
X_train_red = scaler.fit_transform(X_train_red)
X_test_red = scaler.transform(X_test_red)

# モデルを再トレーニング
best_clf.fit(X_train_red, y_train_red)
y_pred_red = best_clf.predict(X_test_red)
accuracy_red = accuracy_score(y_test_red, y_pred_red)
print('Accuracy after feature reduction:', accuracy_red)

# トレーニング精度と検証精度を計算（特徴量削減後）
y_train_pred_red = best_clf.predict(X_train_red)
train_accuracy_red = accuracy_score(y_train_red, y_train_pred_red)
val_accuracy_red = accuracy_score(y_test_red, y_pred_red)
print('Training Accuracy after feature reduction:', train_accuracy_red)
print('Validation Accuracy after feature reduction:', val_accuracy_red)

# 精度をファイルに保存
with open(os.path.join(output_dir, 'model_accuracy_after_reduction.txt'), 'w') as f:
    f.write(f'Training Accuracy after feature reduction: {train_accuracy_red}\n')
    f.write(f'Validation Accuracy after feature reduction: {val_accuracy_red}\n')

# 交差検証
scores_red = cross_val_score(best_clf, X_reduced, y, cv=cv, scoring='accuracy', n_jobs=-1)
print('Cross-validation scores after feature reduction:', scores_red)
print('Mean cross-validation score after feature reduction:', scores_red.mean())

# 交差検証のトレーニング精度と検証精度を取得（特徴量削減後）
cv_results_red = cross_validate(
    best_clf, X_reduced, y, cv=cv, scoring='accuracy', n_jobs=-1,
    return_train_score=True
)

print('Cross-validation training scores after reduction:', cv_results_red['train_score'])
print('Cross-validation validation scores after reduction:', cv_results_red['test_score'])
print('Mean training score after reduction:', np.mean(cv_results_red['train_score']))
print('Mean validation score after reduction:', np.mean(cv_results_red['test_score']))

# 結果をファイルに保存
cv_results_red_df = pd.DataFrame(cv_results_red)
cv_results_red_df.to_csv(os.path.join(output_dir, 'cross_validation_results_after_reduction.csv'), index=False)

# 学習曲線をプロット（特徴量削減後）
train_sizes_red = np.linspace(10, max_train_samples, 10, dtype=int)

train_sizes_red, train_scores_red, val_scores_red = learning_curve(
    best_clf, X_reduced, y, cv=cv, scoring='accuracy', n_jobs=-1,
    train_sizes=train_sizes_red
)

train_scores_mean_red = np.mean(train_scores_red, axis=1)
val_scores_mean_red = np.mean(val_scores_red, axis=1)

plt.figure()
plt.plot(train_sizes_red, train_scores_mean_red, 'o-', color='r', label='Training score')
plt.plot(train_sizes_red, val_scores_mean_red, 'o-', color='g', label='Validation score')
plt.title('Learning Curve after Feature Reduction')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.ylim(0.7, 1.0)  # Y軸の範囲を調整
plt.savefig(os.path.join(output_dir, 'learning_curve_after_reduction.png'))
plt.close()

# 最終的な特徴量の重要度を保存
importances_red = best_clf.feature_importances_
feature_names_red = X_reduced.columns
feature_importances_red_df = pd.DataFrame({
    'Feature': feature_names_red,
    'Importance': importances_red
}).sort_values(by='Importance', ascending=False)
feature_importances_red_df.to_csv(os.path.join(output_dir, 'feature_importances_after_reduction.csv'), index=False)

# モデルの保存
joblib.dump(best_clf, os.path.join(output_dir, 'random_forest_model_final.pkl'))

# 特徴量削減後の混同行列と分類レポート
cm_red = confusion_matrix(y_test_red, y_pred_red)
disp_red = ConfusionMatrixDisplay(confusion_matrix=cm_red)
disp_red.plot()
plt.title('Confusion Matrix - Reduced Model')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_reduced.png'))
plt.close()

report_red = classification_report(y_test_red, y_pred_red, output_dict=True)
report_red_df = pd.DataFrame(report_red).transpose()
report_red_df.to_csv(os.path.join(output_dir, 'classification_report_reduced.csv'))

# 特徴量削減後のROC曲線
y_prob_red = best_clf.predict_proba(X_test_red)[:, 1]
fpr_red, tpr_red, thresholds_red = roc_curve(y_test_red, y_prob_red)
roc_auc_red = auc(fpr_red, tpr_red)

plt.figure()
plt.plot(fpr_red, tpr_red, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_red:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Reduced Model')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve_reduced.png'))
plt.close()

# 交差検証スコアの比較
plt.figure()
plt.boxplot([scores, scores_red], labels=['Initial Model', 'Reduced Model'])
plt.title('Cross-validation Scores Comparison')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(output_dir, 'cross_validation_scores_comparison.png'))
plt.close()