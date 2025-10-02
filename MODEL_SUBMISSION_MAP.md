# 模型與提交文件對應表

## 📊 快速參考表

| 提交文件 | 對應Python文件 | 模型類型 | 主要特點 | 狀態 |
|---------|--------------|---------|---------|------|
| submission_ultimate.csv | ultimate_gnn_model.py | GNN (PyTorch Geometric) | SAGEConv, 128維隱藏層 | ⚠️ F1=0 (數據洩漏) |
| submission.csv | run_gnn_xgb_pipeline.py | GNN + XGBoost | 5-Fold CV, OOF閾值優化 | ✅ 推薦使用 |
| submission_exp3_robust_anomaly.csv | three.py | LightGBM | 穩定時間窗口, 異常特徵 | ✅ 可用 |
| submission_quick_test.csv | quick_test.py | 快速測試 | 輕量級驗證 | 🔄 測試用 |
| submission_breakthrough.csv | breakthrough_model.py | 複雜模型 | 24KB代碼 | 🔄 待驗證 |
| submission_a100_optimized.csv | a100_optimized_model.py | GPU優化 | A100專用, 29KB代碼 | 🔄 待驗證 |
| submission_final.csv | ? | 未知 | 可能是整合版本 | ❓ 待確認 |
| submission_v2.csv | ? | 未知 | 版本2 | ❓ 待確認 |
| submission_A.csv | ? | 未知 | A/B測試版本A | ❓ 待確認 |
| submission_B.csv | ? | 未知 | A/B測試版本B | ❓ 待確認 |

## 🔍 數據洩漏分析

### ultimate_gnn_model.py 的問題診斷

#### 🚨 主要問題：使用了完整數據集的最大日期

```python
# ❌ 錯誤做法 (第51-76行)
def precompute_node_features(df_trans, all_accts):
    latest_date = df_trans['txn_date'].max()  # 121天，包含測試期間！
    window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
    # 為所有帳戶計算特徵，包括測試集帳戶
```

**結果**: 模型在訓練時已經"看到"了測試集的未來信息，導致:
- 訓練時過度擬合測試集的分佈
- 提交到競賽平台後，真實測試集與訓練時的假設不符
- F1-score = 0

#### ✅ 正確做法

```python
# ✅ 正確做法
TRAIN_CUTOFF_DATE = 90  # 明確的訓練截止日期

def precompute_node_features(df_trans, all_accts, cutoff_date):
    # 只使用cutoff_date之前的數據
    train_trans = df_trans[df_trans['txn_date'] <= cutoff_date]
    window_trans = train_trans[train_trans['txn_date'] >= cutoff_date - STATIC_FEATURE_WINDOW]
    # 計算特徵...
```

## 📈 推薦的模型執行順序

### 1. 驗證已有的穩健模型
```bash
cd 初賽資料/

# 首選：GNN + XGBoost 管線
python3 run_gnn_xgb_pipeline.py

# 備選：穩健的LightGBM模型
python3 three.py
```

### 2. 檢查其他模型的數據洩漏問題
需要仔細檢查以下文件:
- breakthrough_model.py
- a100_optimized_model.py
- quick_test.py

### 3. 使用並行處理加速 (如果需要)
```bash
# 步驟1: 特徵提取
python3 parallel_step1_features.py

# 步驟2: 訓練模型 (多個fold並行)
python3 parallel_step2_train_fold.py

# 步驟3: 聚合結果
python3 parallel_step3_aggregate.py
```

## 🛠️ 避免數據洩漏的檢查清單

### ✅ 必須檢查的點

1. **時間截止日期**
   - [ ] 是否明確設定了訓練截止日期？
   - [ ] 特徵工程是否只使用截止日期前的數據？
   - [ ] 測試集的特徵計算是否避免使用測試期間的信息？

2. **測試集隔離**
   - [ ] 是否在訓練前完全分離了測試集？
   - [ ] 特徵工程時是否使用了測試集的統計信息？
   - [ ] 數據標準化/歸一化時是否用了測試集的mean/std？

3. **交叉驗證**
   - [ ] Fold分割是否按時間順序？
   - [ ] 每個fold的特徵是否獨立計算？
   - [ ] 是否有信息從驗證集洩漏到訓練集？

4. **圖結構構建** (針對GNN)
   - [ ] 圖的邊是否只使用訓練期間的交易？
   - [ ] 測試集節點的連接是否使用了未來信息？
   - [ ] 節點特徵是否基於正確的時間窗口？

## 🎯 下一步行動建議

1. **立即執行**:
   ```bash
   cd /Users/jaywang/Desktop/race/初賽資料
   python3 run_gnn_xgb_pipeline.py
   ```
   這個模型有明確的時間分割邏輯，應該不會有數據洩漏問題。

2. **修復 ultimate_gnn_model.py**:
   如果需要使用GNN模型，必須修正數據洩漏問題。

3. **評估所有提交**:
   如果能拿到實際的F1分數，可以判斷哪些模型最有效。

4. **集成多個模型**:
   將表現穩定的模型進行加權平均或投票。

## 📚 相關文檔

- `資料欄位說明.csv` - 完整的數據欄位說明
- `INSTRUCTIONS.md` - 環境設置和執行指南
- `SUBMISSION_ANALYSIS.md` - 詳細的分析報告
