
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
import pickle
import argparse

# --- 主執行函數 ---
def train_fold(fold_id, gpu_id):
    print(f"--- 開始訓練 Fold {fold_id} on GPU {gpu_id} ---")
    
    # 設定此程序使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # --- 載入資料 ---
    print(f"[Fold {fold_id}] 載入預處理的特徵...")
    artifacts_dir = 'parallel_artifacts'
    with open(os.path.join(artifacts_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(artifacts_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)['label']
    with open(os.path.join(artifacts_dir, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)

    # --- 準備 Fold 資料 ---
    NFOLDS = 5
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    
    # 根據 fold_id 找到對應的 train/validation index
    train_idx, valid_idx = list(folds.split(X_train, y_train))[fold_id]
    
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    print(f"[Fold {fold_id}] 資料準備完成。訓練集大小: {len(X_train_fold)}, 驗證集大小: {len(X_valid_fold)}")

    # --- 模型訓練 ---
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        early_stopping_rounds=100,
        tree_method='gpu_hist',
        device=f'cuda:{gpu_id}' # 明確指定 device
    )

    print(f"[Fold {fold_id}] 開始 XGBoost 訓練...")
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        verbose=500
    )

    # --- 產生預測 ---
    print(f"[Fold {fold_id}] 產生預測...")
    oof_preds = model.predict_proba(X_valid_fold)[:, 1]
    sub_preds = model.predict_proba(X_test)[:, 1]

    # --- 儲存結果 ---
    print(f"[Fold {fold_id}] 儲存預測結果...")
    np.save(os.path.join(artifacts_dir, f'oof_fold_{fold_id}.npy'), oof_preds)
    np.save(os.path.join(artifacts_dir, f'sub_fold_{fold_id}.npy'), sub_preds)
    # 也儲存 valid_idx 以便後續合併
    np.save(os.path.join(artifacts_dir, f'valid_idx_fold_{fold_id}.npy'), valid_idx)

    print(f"--- Fold {fold_id} 訓練完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single fold of an XGBoost model.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to train (0-4)")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use for training")
    args = parser.parse_args()
    
    train_fold(args.fold, args.gpu)
