import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_random_forest(X_train, X_test, y_train, y_test, output_dir):
    # パイプラインの作成
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ハイパーパラメータグリッド
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 10, 20, 30, 40],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
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
    model_path = os.path.join(output_dir, 'random_forest_model.joblib')
    joblib.dump(grid_search, model_path)

    return grid_search
