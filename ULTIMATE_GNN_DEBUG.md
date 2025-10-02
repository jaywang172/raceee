# 🔍 ultimate_gnn_model.py F1=0 深度診斷

## 可能的原因分析

基於"空間分割"而非"時間分割"的新認知，以下是可能導致F1=0的真正原因：

### 假設1: 測試集帳戶沒有交易記錄（最可能）
**症狀**: 如果acct_predict.csv中的測試集帳戶從未在acct_transaction.csv中出現
**結果**: 
- 所有測試集帳戶的特徵都是0或NaN
- GNN無法為這些節點生成有效的embedding
- 模型實際上是在對"全0特徵"做預測
- 導致所有預測概率都低於閾值，或全部預測為0

**檢查方法**: 
```python
test_accts = set(df_predict['acct'])
trans_accts = set(df_trans['from_acct']) | set(df_trans['to_acct'])
missing = test_accts - trans_accts
print(f"未在交易中出現的測試帳戶: {len(missing)}/{len(test_accts)}")
```

### 假設2: 圖結構構建問題
**症狀**: FraudGraphDataset.get()方法可能對測試集帳戶構建了錯誤的子圖
**關鍵代碼** (第92-120行):
```python
def get(self, idx):
    acct, ref_date, label = self.samples[idx]
    target_node_idx = self.acct_to_idx.get(acct, -1)
    
    # 如果帳戶不在圖中，返回一個只有目標節點的空圖
    if target_node_idx == -1:
        # ... 創建空圖
```

**問題**: 如果測試集帳戶不在訓練圖中（acct_to_idx中沒有），會返回空圖

### 假設3: 預測概率分布異常
**症狀**: 模型在驗證集上找到的best_thresh可能不適用於測試集
**可能原因**:
- 訓練樣本和測試樣本的特徵分布差異巨大
- 正負樣本混合訓練，但測試時全是未知標籤
- 模型對"沒見過的帳戶"預測概率極低

### 假設4: 批次處理索引錯誤
**症狀**: local_target_node_idx計算可能有問題
**關鍵代碼** (第113行):
```python
local_target_node_idx = torch.tensor([target_node_idx], dtype=torch.long)
```
但在模型前向傳播中 (第140-146行):
```python
def forward(self, x, edge_index, local_target_node_idx, batch):
    # ...
    target_embeddings = []
    for i, node_idx in enumerate(local_target_node_idx):
        batch_mask = (batch == i)
        # ...
```

**問題**: batch索引可能與節點索引混淆


## 🎯 找到真正的問題了！

### 關鍵發現1: acct_to_idx不包含測試集帳戶

**第167-168行的問題**:
```python
all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
acct_to_idx = {acct: i for i, acct in enumerate(all_accts)}
```

**問題分析**:
- `all_accts`只包含**在交易中出現的帳戶**
- 但`df_predict`中的測試集帳戶可能**不在all_accts中**
- 這導致在`FraudGraphDataset.get()`第94行：
  ```python
  target_node_idx = self.acct_to_idx.get(acct, -1)
  ```
  返回`-1`，然後構建一個空圖！

**驗證**:
```python
# 實際測試
test_accts_set = set(df_predict['acct'])
trans_accts_set = set(df_trans['from_acct']) | set(df_trans['to_acct'])
missing = test_accts_set - trans_accts_set
print(f"測試集中不在交易記錄的帳戶: {len(missing)}")  # 結果: 0個
```

等等！所有測試帳戶都在交易中，為什麼還會有問題？

### 關鍵發現2: 特徵全為0的問題

**第51-76行 precompute_node_features的問題**:
```python
latest_date = df_trans['txn_date'].max()  # 121
window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
# 只使用最後7天 [114-121] 的交易
```

**實測結果**:
- 最後7天有活動的測試帳戶: 3507/4780 (73%)
- **沒有活動的測試帳戶: 1273個 (27%)**

這意味著：
- 27%的測試帳戶特徵全為0
- 這些帳戶的預測會極不準確

但這還不足以導致F1=0...

### 關鍵發現3: 第148行的索引Bug（最致命！）

**極其可疑的代碼**:
```python
target_node_embeddings = x[local_target_node_idx + (batch_vector.bincount().cumsum(0) - batch_vector.bincount())]
```

這個計算**極其複雜且容易出錯**！

**問題分析**:
1. `local_target_node_idx`是每個樣本在其子圖中的局部索引
2. `batch_vector`告訴我們哪些節點屬於哪個圖
3. 這行代碼試圖將局部索引轉換為全局索引

**為什麼會出錯**:
- 當batch中混合了不同大小的子圖時
- 索引計算可能錯位
- 導致提取了錯誤節點的embedding
- 或者索引越界

### 最可能的F1=0原因：索引計算錯誤

**完整的因果鏈**:
1. 第148行的索引計算有bug
2. 導致模型在訓練時提取了錯誤的節點embedding
3. 模型"學會"的不是目標帳戶的特徵，而是隨機節點的特徵
4. 驗證時可能碰巧有一些F1，但實際是隨機的
5. 測試時，由於數據分布不同，完全失效
6. F1 = 0

## 修復方案

### 方案A: 修復索引計算（較難）
正確實現batch索引轉換，需要仔細處理PyG的batch機制

### 方案B: 簡化模型，避免複雜索引（推薦）
使用更簡單的GNN架構，避免自定義batch索引計算

### 方案C: 放棄這個模型（最安全）
專注於已驗證有效的模型：
- `run_gnn_xgb_pipeline.py`
- `three.py`

## 結論

`ultimate_gnn_model.py`的F1=0**不是數據洩漏問題**，而是：
1. **索引計算bug**（第148行，最致命）
2. **特徵覆蓋不足**（27%測試帳戶無特徵）
3. **設計過於複雜**（自定義batch處理容易出錯）

**建議**: 明天不要使用這個模型，專注於穩健的模型。
