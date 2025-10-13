#!/bin/bash

# 設置 Bash 在遇到錯誤時立即退出
set -e

# 啟動虛擬環境
echo "啟動 Python 虛擬環境..."
source ../venv/bin/activate

# 步驟 1: 執行特徵工程
echo "
======================================="
echo "=== 步驟 1: 執行特徵工程         ==="
echo "======================================="
python3 parallel_step1_features.py

# 檢查特徵檔案是否成功生成
if [ ! -f "parallel_artifacts/X_train.pkl" ]; then
    echo "
錯誤：特徵檔案生成失敗，流程中止。"
    exit 1
fi

# 步驟 2: 平行化訓練 5 個 Folds
echo "
======================================="
echo "=== 步驟 2: 平行化訓練 5 個 Folds  ==="
echo "======================================="

# 使用 nohup 在背景執行，並將日誌分別儲存
# 我們假設您有 5 張 GPU，編號為 0, 1, 2, 3, 4
nohup python3 parallel_step2_train_fold.py --fold 0 --gpu 0 > fold0.log 2>&1 &
pid0=$!
nohup python3 parallel_step2_train_fold.py --fold 1 --gpu 1 > fold1.log 2>&1 &
pid1=$!
nohup python3 parallel_step2_train_fold.py --fold 2 --gpu 2 > fold2.log 2>&1 &
pid2=$!
nohup python3 parallel_step2_train_fold.py --fold 3 --gpu 3 > fold3.log 2>&1 &
pid3=$!
nohup python3 parallel_step2_train_fold.py --fold 4 --gpu 4 > fold4.log 2>&1 &
pid4=$!

echo "5 個 Fold 的訓練任務已在背景啟動。"
echo "日誌將分別寫入 fold0.log, fold1.log, ..."
echo "PID 分別為: $pid0, $pid1, $pid2, $pid3, $pid4"
echo "等待所有訓練任務完成..."

# 等待所有背景任務結束
wait $pid0 $pid1 $pid2 $pid3 $pid4

echo "
所有 Fold 訓練完成！"

# 步驟 3: 合併結果
echo "
======================================="
echo "=== 步驟 3: 合併結果並生成提交檔案 ==="
echo "======================================="
python3 parallel_step3_aggregate.py

echo "
所有步驟已成功完成！"
