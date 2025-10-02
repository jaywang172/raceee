#!/bin/bash

# 設置 Bash 在遇到錯誤時立即退出
set -e

# 獲取腳本所在的目錄
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 進入腳本所在目錄
cd "$DIR"

echo "======================================="
echo "=== GNN+XGBoost 訓練流程開始 ==="
echo "======================================="

# 步驟 1: 執行特徵工程
echo "
--- 步驟 1: 執行特徵工程 (gnn_xgb_step1_features.py) ---"
python gnn_xgb_step1_features.py

# 檢查特徵檔案是否成功生成
if [ ! -f "train_features_gnn.csv" ] || [ ! -f "test_features_gnn.csv" ]; then
    echo "
錯誤：特徵檔案生成失敗，流程中止。"
    exit 1
fi

echo "
特徵檔案生成成功！"

# 步驟 2: 執行模型訓練與預測
echo "
--- 步驟 2: 執行模型訓練與預測 (gnn_xgb_step2_train.py) ---"
python gnn_xgb_step2_train.py

# 檢查提交檔案是否成功生成
if [ ! -f "submission.csv" ]; then
    echo "
錯誤：提交檔案生成失敗，流程中止。"
    exit 1
fi

echo "
======================================="
echo "=== 流程成功結束！ ==="
echo "提交檔案 'submission.csv' 已在目前目錄生成。"
echo "======================================="
