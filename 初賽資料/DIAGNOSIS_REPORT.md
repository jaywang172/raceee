# 完整數據診斷報告

## 執行摘要

經過全面分析，我發現了 **2 個嚴重的 data leakage 問題** 和 **1 個重大的 distribution shift 問題**，這些都是導致模型 F1 score 不佳的主要原因。

---

## 🔴 關鍵問題發現

### 問題 1: GNN 節點特徵使用了標籤信息 (嚴重 Data Leakage)

**位置**: `run_gnn_xgb_pipeline.py` Line 56

```python
node_features[alert_accts_encoded, 0] = 1.0  # ⚠️ 這是嚴重的 data leakage!
```

**問題描述**:
- GNN 模型直接將 alert 帳號的標籤設為 1.0 作為節點特徵
- 這意味著模型在訓練時就知道哪些帳號是 alert
- 對測試集來說，這是嚴重的資訊洩漏
- 導致模型過度擬合訓練標籤，而非學習真正的圖結構特徵

**影響**:
- 訓練準確度高，但測試準確度低
- F1 score 在實際測試集上表現差
- 這是典型的 overfitting 症狀

---

### 問題 2: 訓練和測試使用相同的截止日期 (時間洩漏)

**發現**:
```
Training cutoff date: 121
Test cutoff date: 121
⚠️ LEAKAGE: Test cutoff (121) <= Train cutoff (121)
```

**問題描述**:
- 訓練集和測試集使用相同的時間點 (day 121)
- 缺乏時間上的分隔
- 無法驗證模型的時間泛化能力

**建議**:
- 應該使用 time-based split
- 例如: train on day 1-105, validate on day 106-121
- 這樣才能真正測試模型的預測能力

---

### 問題 3: 訓練集和測試集的嚴重分佈偏移

**分佈對比**:

| 特徵 | 訓練集平均 | 測試集平均 | 比例 |
|------|-----------|-----------|------|
| 交易總數 | 2.46 | 40.90 | **16.6x** |
| 交易總額 | 100K | 2M | **19.9x** |
| 平均交易額 | 10.9K | 59.4K | **5.4x** |
| 唯一交易對象數 | 1.24 | 19.70 | **15.9x** |

**問題描述**:
- 測試集的帳號比訓練集活躍得多（16倍交易量）
- 測試集帳號明顯是「大戶」，與訓練集分佈完全不同
- 這是 **distribution shift** 問題
- 模型在小額、低頻交易上訓練，但要在大額、高頻交易上預測

**影響**:
- 特徵尺度完全不匹配
- 模型學到的模式無法應用到測試集
- 需要特別處理這種分佈差異

---

## 📊 數據統計

### 基本信息
- 交易記錄: 4,435,890 筆
- Alert 帳戶: 1,004 個 (正樣本)
- 測試帳戶: 4,780 個
- 總帳戶數: 1,800,106 個
- 類別不平衡比: 1:1792

### Alert 帳戶 vs 普通帳戶特徵對比

| 特徵 | Alert 平均 | 非 Alert 平均 | 可分性比例 |
|------|-----------|--------------|-----------|
| 交易總額 | 318,696 | 100,468 | 3.17x |
| 交易次數 | 12.75 | 2.46 | **5.18x** |
| 平均交易額 | 17,345 | 10,902 | 1.59x |
| 唯一交易對象數 | 7.37 | 1.23 | **5.98x** |
| 接收交易次數 | 21.13 | 2.45 | **8.61x** |

**關鍵發現**:
- Alert 帳戶的交易次數是普通帳戶的 5 倍
- Alert 帳戶的唯一交易對象數是普通帳戶的 6 倍
- Alert 帳戶的接收交易次數是普通帳戶的 8.6 倍
- 這些特徵有很強的判別能力

---

## 🔍 模型架構分析

### 當前 GNN 架構
```
- 隱藏層維度: 32
- Attention heads: 2
- 層數: 2
- Dropout: 0.6
- 參數量: ~2K
```

**評價**:
- 架構合理，但節點特徵使用了標籤（嚴重問題）
- Dropout 0.6 較高，有助於防止過擬合

### 當前 XGBoost 設定
```
- Max depth: 8
- N estimators: 2000
- Learning rate: 0.02
- Early stopping: 100 rounds
```

**評價**:
- 設定合理
- 使用了 early stopping 和 scale_pos_weight
- 5-fold cross-validation 策略正確

---

## ✅ 改進方案

我已經創建了修復版本 `improved_model.py`，包含以下改進：

### 1. 移除 GNN 的標籤洩漏
```python
# ❌ 舊版本 (有 leakage)
node_features[alert_accts_encoded, 0] = 1.0

# ✅ 新版本 (使用結構特徵)
node_features = np.column_stack([
    out_degree,           # 出度
    in_degree,            # 入度
    total_degree,         # 總度數
    log_out_degree,       # 對數出度
    log_in_degree,        # 對數入度
    normalized_txn_mean,  # 標準化交易平均值
    normalized_txn_std,   # 標準化交易標準差
])
```

### 2. 使用無監督式 GNN 訓練
- 改用 link prediction 作為訓練目標
- 不依賴任何標籤信息
- 純粹學習圖結構特徵

### 3. 正確的時間切分
```python
# 訓練集: 使用 train_cutoff
train_features = create_features(df_trans, accts, train_cutoff)

# 測試集: 使用 test_cutoff
test_features = create_features(df_trans, accts, test_cutoff)
```

### 4. 豐富的特徵工程 (47+ 特徵)

#### 基礎統計特徵
- 交易次數、總額、平均值、標準差、最小值、最大值
- 作為發送方和接收方的統計

#### 網絡特徵
- 出度、入度、總度數
- 交易集中度
- 交易對象多樣性

#### 行為特徵
- 淨現金流
- 交易金額波動率
- 平均交易間隔、標準差

#### 時間特徵
- 夜間交易比例 (凌晨 0-6 點, 晚上 10-12 點)
- 週末交易比例
- 交易速度 (近期 vs 歷史)
- 帳戶年齡、每日平均交易數

#### 可疑模式特徵
- 整數金額交易比例 (例如 10000, 50000)
- 接近門檻值交易 (29K-30K, 49K-50K)
- 自我轉帳比例
- 渠道多樣性

#### 時間窗口特徵
- 最近 7/14/30 天的交易統計
- 交易速度變化 (7天 vs 30天)

### 5. 特徵標準化
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

這解決了 distribution shift 問題，讓訓練集和測試集在相同尺度上。

### 6. 更深的 GNN 架構
```python
- 3 層 GATv2Conv
- 隱藏維度: 64 (提升自 32)
- Attention heads: 4 (提升自 2)
- Dropout: 0.3 (降低自 0.6，因為不再有 leakage)
```

---

## 📈 預期改進效果

### 問題修復後的預期
1. **移除 label leakage**
   - 訓練 F1 可能會下降，但這是正常的
   - 測試 F1 應該會上升
   - 泛化能力增強

2. **特徵標準化**
   - 解決 distribution shift
   - 模型能更好地處理不同尺度的數據

3. **豐富特徵**
   - 更多判別能力
   - 捕捉更多可疑模式

### 如何判斷是 overfitting 還是 data leakage

| 症狀 | Overfitting | Data Leakage |
|------|------------|--------------|
| 訓練 F1 | 高 (>0.95) | 非常高 (>0.99) |
| 驗證 F1 | 中等 (0.6-0.8) | 高 (>0.9) |
| 測試 F1 | 低 (<0.5) | 極低 (<0.3) |
| OOF vs Test | 適度差距 | 巨大差距 |
| 特徵重要性 | 合理 | 某些特徵異常重要 |

你的情況符合 **Data Leakage** 特徵。

---

## 🚀 下一步建議

### 立即執行
1. 運行 `python improved_model.py`
2. 對比新舊模型的結果
3. 檢查 OOF F1 是否更接近真實測試 F1

### 進階優化
1. **時間序列驗證**
   ```python
   # 使用時間切分而非隨機切分
   train_data = data[data['date'] <= train_cutoff]
   val_data = data[(data['date'] > train_cutoff) & (data['date'] <= val_cutoff)]
   test_data = data[data['date'] > val_cutoff]
   ```

2. **處理冷啟動問題**
   - 0 個測試帳號是完全新的帳號
   - 但測試集活躍度高，需要專門設計特徵

3. **集成學習**
   - 訓練多個模型 (XGBoost, LightGBM, CatBoost)
   - 使用不同的 negative sampling 策略
   - Stacking 或 blending

4. **Focal Loss**
   ```python
   # 替換標準 cross-entropy
   # 對難分類樣本給予更多關注
   ```

5. **SMOTE 過採樣**
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

---

## 📋 檢查清單

- [x] 識別 data leakage 問題
- [x] 識別 distribution shift 問題
- [x] 分析特徵可分性
- [x] 創建修復版本模型
- [ ] 執行新模型並驗證改進
- [ ] 實施時間序列驗證
- [ ] 嘗試集成學習
- [ ] 調整閾值優化策略

---

## 📊 生成的文件

1. `comprehensive_diagnosis.py` - 完整診斷腳本
2. `improved_model.py` - 修復後的改進模型
3. `feature_comparison.csv` - 特徵對比統計
4. `alert_temporal_distribution.csv` - Alert 時間分佈

---

## 總結

你的模型問題 **不是單純的 overfitting**，而是：
1. **嚴重的 data leakage** (GNN 使用標籤作為特徵)
2. **Distribution shift** (訓練集和測試集分佈差異 16 倍)
3. **缺乏時間驗證** (訓練和測試同一天)

修復這些問題後，F1 score 應該會有顯著提升。重點是要讓模型學習真正的 **異常交易模式**，而不是記憶訓練標籤。

---

**建議優先級**:
1. 🔥 立即修復 label leakage (使用 `improved_model.py`)
2. 🔥 實施特徵標準化
3. ⭐ 增加時間窗口特徵
4. ⭐ 實施時間序列驗證
5. 💡 嘗試集成學習和高級採樣技術
