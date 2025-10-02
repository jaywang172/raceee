# 🏦 警示帳戶預測競賽 - 提交歷史分析總結

## 📌 快速總結

你有**10個提交文件**，對應到**至少6個不同的Python模型**。

### ⚠️ 關鍵發現
**submission_ultimate.csv (F1-score = 0)** 的原因已找到：
- **數據洩漏問題**: `ultimate_gnn_model.py` 在計算特徵時使用了完整數據集的最大日期(121天)
- 模型在訓練時"看到"了測試期間的信息，導致過度擬合
- 提交到平台後，真實測試集與訓練假設不符，導致F1=0

## 📁 提交文件與模型對應關係

### ✅ 確認的對應關係

1. **submission_ultimate.csv** ← `ultimate_gnn_model.py`
   - GNN模型，**有數據洩漏**，F1=0

2. **submission.csv** ← `run_gnn_xgb_pipeline.py`
   - GNN + XGBoost混合模型
   - 使用5-Fold交叉驗證
   - **推薦使用** - 有明確的時間分割

3. **submission_exp3_robust_anomaly.csv** ← `three.py`
   - LightGBM模型
   - 穩定的時間窗口特徵工程
   - **可用**

4. **submission_quick_test.csv** ← `quick_test.py`
   - 快速測試模型

5. **submission_breakthrough.csv** ← `breakthrough_model.py`
   - 複雜模型，待驗證

6. **submission_a100_optimized.csv** ← `a100_optimized_model.py`
   - GPU優化模型，待驗證

### ❓ 未明確對應的文件

7. **submission_final.csv** - 可能是最終整合版本
8. **submission_v2.csv** - 版本2
9. **submission_A.csv** - A/B測試版本A
10. **submission_B.csv** - A/B測試版本B

## 🎯 評分方式

```
F1-score = 2 × (Precision × Recall) / (Precision + Recall)

其中:
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)
- 四捨五入至小數點後7位
```

## 📊 數據規模

- **交易數據**: 4,435,890 筆
- **警示帳戶**: 1,004 個
- **待預測帳戶**: 4,780 個
- **時間跨度**: 121天 (day 1 - day 121)

## 🔍 ultimate_gnn_model.py 的數據洩漏詳解

### 問題代碼位置

**第51-76行** - `precompute_node_features()` 函數:
```python
def precompute_node_features(df_trans, all_accts):
    # ❌ 錯誤：使用了全部數據的最大日期
    latest_date = df_trans['txn_date'].max()  # = 121
    
    # ❌ 錯誤：為所有帳戶（包括測試集）計算了基於最新數據的特徵
    window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
```

**第273行** - 測試集預測:
```python
# ❌ 錯誤：使用最大日期作為測試集的參考日期
max_date = df_trans['txn_date'].max()  # = 121
test_samples = [(acct, max_date, 0) for acct in test_accts]
```

### 為什麼會導致F1=0？

1. **訓練階段**：模型使用全部121天的數據計算特徵，學習了測試期間的模式
2. **線上評估**：競賽平台的測試集可能:
   - 只提供到某個截止日期的數據（如90天）
   - 或者測試集的時間分佈與訓練時的假設完全不同
3. **結果**：模型的預測完全失效，導致F1-score = 0

## ✅ 正確的做法

### 修改建議

```python
# ✅ 正確：明確設定訓練截止日期
TRAIN_CUTOFF_DATE = 90  # 只使用前90天訓練
TEST_REFERENCE_DATE = 121  # 測試集的參考日期

def precompute_node_features(df_trans, all_accts, cutoff_date):
    '''為節點計算特徵，只使用cutoff_date之前的數據'''
    # ✅ 正確：只使用截止日期前的數據
    train_trans = df_trans[df_trans['txn_date'] <= cutoff_date]
    window_trans = train_trans[
        train_trans['txn_date'] >= cutoff_date - STATIC_FEATURE_WINDOW
    ]
    # ... 特徵計算 ...
    return features_df

# 使用時
# 訓練階段：只使用90天的數據
train_features = precompute_node_features(df_trans, all_accts, TRAIN_CUTOFF_DATE)

# 測試階段：也只能使用90天的數據
test_features = precompute_node_features(df_trans, test_accts, TRAIN_CUTOFF_DATE)
```

## 🚀 立即可執行的推薦模型

### 選項1：GNN + XGBoost 管線（推薦）

```bash
cd /Users/jaywang/Desktop/race/初賽資料
python3 run_gnn_xgb_pipeline.py
```

**優點**：
- 有明確的時間分割邏輯
- 使用5-Fold交叉驗證
- OOF預測優化閾值
- 避免數據洩漏

### 選項2：穩健的LightGBM模型

```bash
cd /Users/jaywang/Desktop/race/初賽資料
python3 three.py
```

**優點**：
- 專注於穩定的時間窗口（1, 3, 7天）
- 異常特徵工程
- 較快的訓練速度

## 📋 數據洩漏檢查清單

在執行任何模型前，檢查這些關鍵點：

### ✅ 時間截止
- [ ] 是否有明確的 `TRAIN_CUTOFF_DATE`？
- [ ] 特徵計算是否只用截止日期前的數據？
- [ ] 測試集特徵是否避免使用未來信息？

### ✅ 測試集隔離
- [ ] 特徵工程時沒用測試集統計？
- [ ] 標準化沒用測試集的mean/std？
- [ ] 訓練集和測試集完全分離？

### ✅ 交叉驗證
- [ ] Fold分割按時間順序？
- [ ] 每個fold的特徵獨立計算？
- [ ] 沒有驗證集到訓練集的信息洩漏？

### ✅ 圖結構（針對GNN）
- [ ] 圖的邊只用訓練期間交易？
- [ ] 節點特徵基於正確時間窗口？
- [ ] 測試集節點沒用未來連接？

## 📈 下一步建議

1. **立即執行穩健模型**
   ```bash
   cd 初賽資料/
   python3 run_gnn_xgb_pipeline.py
   ```

2. **檢查其他模型**
   - `breakthrough_model.py` - 檢查時間分割
   - `a100_optimized_model.py` - 檢查特徵計算
   - `quick_test.py` - 檢查數據洩漏

3. **修復 ultimate_gnn_model.py**（如果需要GNN）
   - 添加明確的訓練截止日期
   - 修改特徵計算函數
   - 分離訓練和測試流程

4. **模型集成**
   - 使用多個穩健模型的預測
   - 加權平均或投票
   - 進一步提升F1-score

## 📚 生成的文檔

1. **SUBMISSION_ANALYSIS.md** - 詳細的技術分析
2. **MODEL_SUBMISSION_MAP.md** - 模型對應關係和執行指南
3. **README_ANALYSIS.md** - 本文檔（快速總結）

---

**總結**: 你的 `ultimate_gnn_model.py` 因為數據洩漏導致F1=0。推薦使用 `run_gnn_xgb_pipeline.py` 或 `three.py`，這兩個模型有正確的時間分割邏輯。
