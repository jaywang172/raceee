"""
Multi-Model Ensemble for Fraud Detection
使用 XGBoost + LightGBM + CatBoost 的組合
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
N_FOLDS = 5
RANDOM_SEED = 42
DEVICE = 'cuda'

np.random.seed(RANDOM_SEED)

print("="*80)
print("ENSEMBLE MODEL: XGBoost + LightGBM + CatBoost")
print("="*80)

# ============================================================================
# LOAD PREPROCESSED FEATURES
# ============================================================================
print("\n[1/4] Loading preprocessed features...")

# 假設你已經用 improved_model.py 生成了特徵
# 這裡直接讀取
# 如果沒有，需要先跑 improved_model.py 的特徵工程部分

# 為了示範，這裡展示如何組合多個模型
# 實際使用時，需要先生成 train_features 和 test_features

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    """訓練 XGBoost"""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        tree_method='gpu_hist',
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    return model

def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight):
    """訓練 LightGBM"""
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        device='gpu',
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model

def train_catboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    """訓練 CatBoost"""
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        task_type='GPU',
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    return model

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ensemble(X_train, y_train, X_test):
    """
    訓練 ensemble 模型

    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        X_test: 測試特徵

    Returns:
        oof_preds: OOF predictions (用於threshold優化)
        test_preds: 測試集預測
    """

    print(f"\n[2/4] Training ensemble with {N_FOLDS} folds...")

    # 初始化預測數組
    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))

    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    # 交叉驗證
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 訓練 XGBoost
        print("  Training XGBoost...")
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, scale_pos_weight)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # 訓練 LightGBM
        print("  Training LightGBM...")
        lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, scale_pos_weight)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # 訓練 CatBoost
        print("  Training CatBoost...")
        cat_model = train_catboost(X_tr, y_tr, X_val, y_val, scale_pos_weight)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # Fold validation metrics
        oof_ensemble = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
        val_pred = (oof_ensemble > 0.5).astype(int)
        val_f1 = f1_score(y_val, val_pred)

        print(f"  Ensemble F1: {val_f1:.4f}")

    # 組合預測（簡單平均）
    oof_preds = (oof_xgb + oof_lgb + oof_cat) / 3
    test_preds = (test_xgb + test_lgb + test_cat) / 3

    print("\n[3/4] Individual model OOF performance:")
    for name, preds in [('XGBoost', oof_xgb), ('LightGBM', oof_lgb), ('CatBoost', oof_cat)]:
        pred_binary = (preds > 0.5).astype(int)
        f1 = f1_score(y_train, pred_binary)
        print(f"  {name}: F1 = {f1:.4f}")

    return oof_preds, test_preds

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold(y_true, y_pred):
    """優化分類閾值"""
    print("\n[4/4] Optimizing threshold...")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in tqdm(np.arange(0.05, 0.95, 0.01)):
        pred_binary = (y_pred > threshold).astype(int)
        f1 = f1_score(y_true, pred_binary)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest F1: {best_f1:.5f}")
    print(f"Best threshold: {best_threshold:.4f}")

    return best_threshold, best_f1

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：

    1. 先用 improved_model.py 生成特徵：
       - train_features (46個統計特徵)
       - test_features
       - y_train

    2. 然後用這個腳本訓練 ensemble:

       oof_preds, test_preds = train_ensemble(X_train, y_train, X_test)
       best_threshold, best_f1 = optimize_threshold(y_train, oof_preds)

       final_test_pred = (test_preds > best_threshold).astype(int)
    """

    print("\n" + "="*80)
    print("USAGE:")
    print("="*80)
    print("1. First generate features using improved_model.py")
    print("2. Then load features and use this script:")
    print()
    print("   from ensemble_models import train_ensemble, optimize_threshold")
    print("   oof_preds, test_preds = train_ensemble(X_train, y_train, X_test)")
    print("   best_threshold, _ = optimize_threshold(y_train, oof_preds)")
    print("   final_pred = (test_preds > best_threshold).astype(int)")
    print("="*80)
