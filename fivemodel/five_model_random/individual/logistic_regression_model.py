import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_logistic_regression(X_train, X_test, y_train, y_test, output_dir):
    # パイプラインの作成
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    # ハイパーパラメータグリッド
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__penalty': ['l2']
    }

    # SMOTEの適用
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # グリッドサーチの実行
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # モデルの学習
    grid_search.fit(X_train_resampled, y_train_resampled)

    # モデルの保存
    model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
    joblib.dump(grid_search, model_path)

    return grid_search
