import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_xgboost(X_train, X_test, y_train, y_test, output_dir):
    # パイプラインの作成
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    # ハイパーパラメータグリッド
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__min_child_weight': [1, 3, 5],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
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
    model_path = os.path.join(output_dir, 'xgboost_model.joblib')
    joblib.dump(grid_search, model_path)

    return grid_search
