# 提交歷史分析報告

## 評分方式總結
- **評分指標**: F1-score (Precision和Recall的調和平均)
- **精確度計算**: 四捨五入至小數點後7位
- **目標**: 預測警示帳戶 (label=1) vs 正常帳戶 (label=0)

### F1-score 公式
```
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
```

## 提交歷史與對應模型

### 1. submission_ultimate.csv → ultimate_gnn_model.py
**模型類型**: GNN (Graph Neural Network) with PyTorch Geometric
**結果**: **F1-score = 0** (數據洩漏問題，模型過擬合)
**特點**:
- 使用SAGEConv進行圖神經網絡學習
- HIDDEN_CHANNELS = 128, GNN_OUT_CHANNELS = 64
- DYNAMIC_GRAPH_WINDOW = 7天
- 問題: 可能使用了未來信息或測試集信息導致過擬合

### 2. submission.csv → run_gnn_xgb_pipeline.py 或 parallel_step3_aggregate.py
**模型類型**: GNN + XGBoost 混合模型
**特點**:
- 階段1: GNN特徵提取
- 階段2: XGBoost分類
- 使用5-Fold交叉驗證
- OOF (Out-of-Fold) 預測尋找最佳閾值

### 3. submission_exp3_robust_anomaly.csv → three.py
**模型類型**: LightGBM with Robust Feature Engineering
**特點**:
- 專注於穩定時間窗口 (1, 3, 7天)
- 創建異常特徵: ratio和difference features
- 使用異常檢測方法

### 4. submission_quick_test.csv → quick_test.py
**模型類型**: 快速測試模型
**特點**: 輕量級模型用於快速驗證

### 5. submission_breakthrough.csv → breakthrough_model.py
**模型類型**: 突破性模型 (較大規模)
**特點**: 24KB代碼，可能包含更複雜的特徵工程

### 6. submission_a100_optimized.csv → a100_optimized_model.py
**模型類型**: A100 GPU優化模型
**特點**: 
- 針對A100 GPU優化
- 29KB代碼，較複雜的實現

### 7. submission_final.csv → (未明確對應)
可能是最終整合版本

### 8. submission_v2.csv → (未明確對應)
版本2的提交

### 9. submission_A.csv, submission_B.csv → (未明確對應)
A/B測試版本

## 關鍵發現

### 數據洩漏問題 (Data Leakage)
**ultimate_gnn_model.py 得分0的原因**:
1. 可能在特徵工程時使用了未來信息
2. 可能在訓練時不當使用了測試集信息
3. 模型過度擬合訓練集特定模式

### 避免數據洩漏的關鍵點:
1. **時間截止**: 特徵計算必須嚴格使用 cutoff_date 之前的數據
2. **測試集隔離**: 絕對不能使用 acct_predict.csv 中的任何統計信息
3. **交叉驗證**: 確保fold分割正確，避免信息洩漏

## 建議的下一步

1. **檢查時間洩漏**: 仔細審查每個模型的cutoff_date使用
2. **特徵驗證**: 確認所有特徵只使用歷史信息
3. **模型選擇**: 優先使用表現穩定的模型 (如 run_gnn_xgb_pipeline.py)
4. **閾值調優**: 使用OOF預測找到最佳F1閾值
5. **集成方法**: 考慮多個穩健模型的集成

## 檔案結構
```
初賽資料/
├── acct_transaction.csv    # 交易數據 (703MB)
├── acct_alert.csv           # 警示帳戶標籤 (67KB)
├── acct_predict.csv         # 待預測帳戶 (313KB)
└── [各種Python模型文件]
```

## 數據詳細分析

### 數據規模
- **交易數據**: 4,435,890 筆交易
- **警示帳戶**: 1,004 個
- **待預測帳戶**: 4,780 個
- **時間範圍**: 第1天到第121天

### 數據欄位說明
**acct_transaction.csv** (交易數據):
- from_acct: 匯款帳戶
- from_acct_type: 匯款帳戶類型 (01:玉山, 02:他行)
- to_acct: 收款帳戶
- to_acct_type: 收款帳戶類型
- is_self_txn: 是否為自己轉帳 (Y/N/UNK)
- txn_amt: 交易金額
- txn_date: 交易日期 (1-121)
- txn_time: 交易時間
- currency_type: 幣別 (TWD/USD等)
- channel_type: 交易通路 (1:ATM, 2:臨櫃, 3:行銀, 4:網銀等)

**acct_alert.csv** (警示帳戶):
- acct: 帳戶ID
- event_date: 警示日期 (1-121)

**acct_predict.csv** (待預測):
- acct: 帳戶ID
- label: 預測結果 (初始全為0，需要填入預測值)

## 數據洩漏檢查清單

### ultimate_gnn_model.py 可能的數據洩漏點:

#### 問題1: 測試集特徵計算
```python
# 第273行
max_date = df_trans['txn_date'].max()  # max_date = 121
test_samples = [(acct, max_date, 0) for acct in test_accts]
```
**風險**: 使用全部交易數據的最大日期(121)來計算測試集特徵

#### 問題2: 節點特徵預計算
```python
# 第51-76行: precompute_node_features函數
latest_date = df_trans['txn_date'].max()  # 使用所有數據的最大日期
window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
```
**風險**: 為所有帳戶(包括測試集帳戶)計算特徵時使用了完整數據集

#### 問題3: 圖結構構建
```python
# 第95-97行: FraudGraphDataset.get方法
sub_trans = self.trans_df[(self.trans_df['txn_date'] < ref_date) & 
                          (self.trans_df['txn_date'] >= ref_date - DYNAMIC_GRAPH_WINDOW)]
```
這部分看起來是正確的，使用了時間截止

### 正確的做法 (參考run_gnn_xgb_pipeline.py):

1. **設定訓練截止日期**: 
   - 應該設定一個訓練集的截止日期 (例如: 90天)
   - 只使用截止日期前的數據進行特徵工程和模型訓練

2. **測試集特徵計算**:
   - 測試集的特徵也只能使用訓練截止日期前的信息
   - 不能使用測試集本身的任何統計信息

3. **時間窗口一致性**:
   - 訓練和測試時使用相同的時間窗口邏輯
   - 確保沒有未來信息洩漏

## 修正建議

### 對於 ultimate_gnn_model.py 的修正:

1. 添加明確的訓練截止日期:
```python
TRAIN_CUTOFF_DATE = 90  # 只使用前90天的數據訓練
TEST_CUTOFF_DATE = 121  # 測試集的參考日期
```

2. 修改節點特徵計算:
```python
def precompute_node_features(df_trans, all_accts, cutoff_date):
    # 只使用cutoff_date之前的數據
    train_trans = df_trans[df_trans['txn_date'] <= cutoff_date]
    # ... 其他計算
```

3. 分離訓練和測試的特徵計算流程

## 提交文件狀態

**注意**: 所有submission文件目前都是Git LFS指針，實際內容未能讀取。
需要確認:
1. 各個submission的實際F1分數
2. 哪些模型的表現最好
3. 是否有其他模型也存在數據洩漏問題

