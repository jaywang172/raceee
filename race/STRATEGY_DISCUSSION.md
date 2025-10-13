# 🎯 5張3090並行策略深度討論

## ⚠️ 關鍵發現

### 數據時間結構
```
訓練集警示帳戶: 1004個，警示日期範圍: 1-121天
測試集待預測: 4780個
交易數據範圍: 1-121天
```

**重要**: 訓練截止日期 = 數據最大日期 = 121天
- 這意味著訓練集和測試集在**相同時間段內**
- 這是一個**空間分割**而非時間分割的問題
- 測試集帳戶在訓練期間也可能有交易記錄

### 這改變了什麼？

1. **不是時間序列預測問題**，而是帳戶分類問題
2. **測試集帳戶的特徵可以使用全部121天的數據**
3. **數據洩漏的定義不同**：不能用測試集帳戶的標籤，但可以用其交易數據

---

## 硬體環境
- **GPU**: 5張NVIDIA 3090（無NVLink，24GB VRAM each）
- **並行方式**: 獨立進程，通過CUDA_VISIBLE_DEVICES控制

---

## 三個可行策略（詳細分析）

### 策略A：多模型並行訓練 + 集成（⭐推薦）

#### 核心思路
同時訓練5個不同的模型，每個使用一張GPU，最後集成。

#### 具體分配

**GPU 0**: `run_gnn_xgb_pipeline.py`
- GNN特徵提取 + XGBoost 5-Fold CV
- 預計時間：30-60分鐘
- 輸出：submission_gnn_xgb.csv + OOF預測

**GPU 1**: `three.py` (LightGBM)
- 穩健的時間窗口特徵
- 預計時間：5-10分鐘
- 輸出：submission_lgb.csv

**GPU 2**: 修復的`ultimate_gnn_model.py`
- 純GNN模型（修正數據洩漏）
- 預計時間：20-40分鐘
- 輸出：submission_gnn.csv

**GPU 3**: `breakthrough_model.py`
- 需要先檢查代碼確認無洩漏
- 預計時間：待定
- 輸出：submission_breakthrough.csv

**GPU 4**: XGBoost變體（不同參數）
- 使用不同的特徵組合或超參數
- 預計時間：20-30分鐘
- 輸出：submission_xgb_variant.csv

#### 最後步驟
```bash
# 所有模型完成後，在主機上執行
python3 ensemble_final.py
```
- 讀取5個模型的預測
- 加權平均（權重基於驗證集F1）
- 最終提交文件

#### 優點
✅ 最大化利用5張GPU
✅ 模型多樣性高，集成效果好
✅ 總時間 = 最慢模型的時間（約60分鐘）

#### 缺點
⚠️ 需要仔細檢查每個模型
⚠️ 協調5個進程較複雜
⚠️ 如果有一個模型出錯，需要調試

---

### 策略B：並行5-Fold XGBoost（需要改寫）

#### 核心思路
將`run_gnn_xgb_pipeline.py`拆解為3步：
1. 特徵工程（1張GPU，20分鐘）
2. 並行5個Fold訓練（5張GPU同時，5分鐘）
3. 聚合結果（CPU，1分鐘）

#### 具體步驟

**步驟1**: 在GPU 0上執行特徵工程
```bash
CUDA_VISIBLE_DEVICES=0 python3 parallel_step1_features.py
```
- 生成GNN embeddings
- 計算所有靜態特徵
- 保存到`parallel_artifacts/`

**步驟2**: 並行訓練5個Fold
```bash
# 同時啟動5個進程
CUDA_VISIBLE_DEVICES=0 python3 parallel_step2_train_fold.py --fold 0 --gpu 0 &
CUDA_VISIBLE_DEVICES=1 python3 parallel_step2_train_fold.py --fold 1 --gpu 1 &
CUDA_VISIBLE_DEVICES=2 python3 parallel_step2_train_fold.py --fold 2 --gpu 2 &
CUDA_VISIBLE_DEVICES=3 python3 parallel_step2_train_fold.py --fold 3 --gpu 3 &
CUDA_VISIBLE_DEVICES=4 python3 parallel_step2_train_fold.py --fold 4 --gpu 4 &
wait  # 等待所有進程完成
```

**步驟3**: 聚合結果
```bash
python3 parallel_step3_aggregate.py
```

#### 優點
✅ 專注於優化單一強模型
✅ 5-Fold並行，總時間大幅縮短
✅ 技術上最優雅

#### 缺點
⚠️ 已有parallel_step系列腳本，但需要檢查
⚠️ 如果特徵工程階段出錯，全部重來
⚠️ 缺乏模型多樣性

---

### 策略C：保守順序執行 + 人工監控（最安全）

#### 核心思路
不冒險，按順序執行已驗證的模型。

#### 執行順序

**第1輪**（GPU 0）:
```bash
CUDA_VISIBLE_DEVICES=0 python3 three.py
```
- 最快，5-10分鐘
- 先拿到一個baseline

**第2輪**（GPU 0）:
```bash
CUDA_VISIBLE_DEVICES=0 python3 run_gnn_xgb_pipeline.py
```
- 最穩健的模型，30-60分鐘
- 檢查結果，如果F1 > 0，繼續

**第3輪**（如果時間允許）:
- 嘗試其他模型或集成

#### 優點
✅ 最安全，出錯機率低
✅ 每步都可以檢查結果
✅ 不需要複雜的並行協調

#### 缺點
⚠️ 沒有充分利用5張GPU
⚠️ 總時間較長（60-90分鐘）
⚠️ 可能錯過更好的集成機會

---

## 我的推薦

### 🎯 最佳方案：**策略A的簡化版**

基於你明天需要在機房執行，我建議：

#### Phase 1：快速驗證（10分鐘）
```bash
# 先用最快的模型驗證環境
CUDA_VISIBLE_DEVICES=0 python3 three.py
```
檢查：
- 環境是否正常
- 輸出文件是否合理
- 預測分布是否正常

#### Phase 2：並行主力模型（60分鐘）
```bash
# 同時啟動3個最穩健的模型
CUDA_VISIBLE_DEVICES=0 python3 run_gnn_xgb_pipeline.py > log_gnn_xgb.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 three.py --output submission_lgb2.csv > log_lgb.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 check_and_run_breakthrough.py > log_breakthrough.txt 2>&1 &

# 監控進度
tail -f log_gnn_xgb.txt
```

#### Phase 3：集成（5分鐘）
```bash
python3 simple_ensemble.py
```

### 為什麼這樣？

1. **風險控制**: 先快速驗證，再並行執行
2. **時間效率**: 3個模型並行，總時間60分鐘
3. **結果保證**: 至少有3個獨立模型，可以互相驗證
4. **靈活性**: 如果某個模型出錯，還有其他backup

---

## 需要準備的腳本

### 1. 檢查並修復數據洩漏（必須）
- [ ] 檢查`ultimate_gnn_model.py`
- [ ] 檢查`breakthrough_model.py`
- [ ] 確保所有模型的時間邏輯正確

### 2. 簡單的集成腳本（必須）
- [ ] `simple_ensemble.py`：讀取多個submission，加權平均

### 3. 監控腳本（可選但推薦）
- [ ] `monitor_gpus.sh`：實時顯示GPU使用情況
- [ ] `check_progress.py`：檢查各模型進度

---

## 關鍵注意事項

### ⚠️ 數據洩漏檢查
雖然這是空間分割問題，仍需注意：
- ❌ 不能用測試集帳戶的標籤信息
- ✅ 可以用測試集帳戶的交易記錄做特徵
- ⚠️ 小心不要在特徵工程時引入標籤信息

### 💡 調試技巧
```bash
# 先用小數據測試
head -10000 acct_transaction.csv > acct_transaction_small.csv

# 快速測試腳本
python3 model.py --test-mode --max-iter 10
```

### 📊 結果驗證
每個模型完成後檢查：
```python
import pandas as pd
df = pd.read_csv('submission.csv')
print(f"預測為1的數量: {df['label'].sum()}")
print(f"預測為1的比例: {df['label'].mean():.2%}")
# 合理範圍：5-15%左右
```

---

## 你的決定？

請告訴我：
1. 你傾向哪個策略？（A/B/C 或混合）
2. 你對風險的容忍度？（激進/穩健/保守）
3. 你在機房有多少時間？（2小時/4小時/一整天）
4. 你想要單一最佳模型，還是集成？

我會根據你的選擇，準備相應的腳本和執行計劃。
