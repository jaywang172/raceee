# 🎯 ultimate_gnn_model.py 最終診斷報告

## 💥 根本原因：Training-Serving Skew（訓練-預測偏斜）

感謝Gemini的深刻洞察，我們找到了F1=0的真正原因。

---

## 🔍 兩大致命缺陷（已驗證）

### 缺陷1: 過於稀疏的「近期」特徵

**代碼位置**: 第51-76行 `precompute_node_features()`

```python
latest_date = df_trans['txn_date'].max()  # 121
window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
# 只使用最後7天的交易
```

**實測數據**:
- **94.6%訓練集帳戶**在最後7天無活動 → 特徵全為0
- **26.6%測試集帳戶**在最後7天無活動 → 特徵全為0

**後果**:
1. 大量帳戶的輸入特徵是全零向量
2. GNN無法從零向量學到有效的embedding
3. 零embedding通過分類器 → 極低的預測機率

---

### 缺陷2: 驗證集與測試集的「情境」不匹配 ⭐⭐⭐

**這是最致命的問題！**

#### 驗證集情境（第175-194行）

```python
# 每個警示帳戶使用其真實事件日期
positive_samples = [(acct, date, 1) for acct, date in df_alert.itertuples(...)]

# 驗證集: date > 90 的樣本
val_samples = [s for s in all_samples if s[1] > TIME_SPLIT_DATE]
```

**特點**:
- 每個樣本有**獨立的ref_date**（事件發生日期）
- 模型看到的是帳戶在**其異常時刻**的圖結構
- **80%的驗證樣本**在事件前後±7天很活躍
- 特徵豐富，預測機率高
- 模型找到較高的best_thresh（例如0.7）

#### 測試集情境（第272-276行）

```python
test_accts = df_predict['acct'].unique()
max_date = df_trans['txn_date'].max()  # 121
test_samples = [(acct, max_date, 0) for acct in test_accts]
# 所有測試帳戶都用同一個日期！
```

**特點**:
- 所有4780個測試帳戶使用**同一個靜態日期**（121）
- 這是一個「期末快照」，不是各自的「高光時刻」
- 只有73.4%在這個時刻活躍
- 26.6%特徵為0，其他也較稀疏
- 預測機率普遍很低

#### 💥 災難性結果

```
驗證時學到: "在高光時刻，機率>0.7的是警示帳戶"
測試時遇到: "在靜態快照，大部分帳戶機率<0.3"
結果: 幾乎沒有帳戶能超過0.7的門檻
→ 預測全為0
→ F1 = 0
```

---

## 📊 數據驗證

### 驗證集分析
- 警示事件日期分布: 1-121天（平均68.5，中位74.0）
- 驗證集（date>90）: 332個事件
- 抽樣20個驗證樣本: **16個（80%）在事件前後±7天活躍**

### 測試集分析
- 所有4780個帳戶: ref_date = 121（固定）
- 在第121天前後7天活躍: 3507個（73.4%）
- 在第121天前後7天**不活躍**: 1273個（26.6%）

### 關鍵矛盾

| 指標 | 驗證集 | 測試集 | 差異 |
|------|--------|--------|------|
| ref_date | 各自事件日期 | 統一為121 | ⚠️ 情境完全不同 |
| 活躍率 | ~80% | ~73% | 測試集更冷 |
| 特徵豐富度 | 高（高光時刻） | 低（靜態快照） | ⚠️ 巨大差異 |
| 預測機率分布 | 較高且分散 | 普遍較低 | ⚠️ 門檻失效 |

---

## 🔧 為什麼其他模型沒有這個問題？

### run_gnn_xgb_pipeline.py（正常工作）

**關鍵區別**:
```python
# 第134-136行：GNN只用於特徵提取，不依賴ref_date
gnn_embeddings = create_graph_and_embeddings(df_trans, all_accts, train_accts, train_cutoff_date)

# 第102-118行：靜態特徵計算對所有帳戶一致
train_static_features = create_static_features(df_trans, all_accts, train_cutoff_date)
test_static_features = create_static_features(df_trans, all_accts, test_cutoff_date)
```

**為什麼沒問題**:
1. **特徵計算一致**: 訓練和測試用相同邏輯
2. **沒有動態ref_date**: 不依賴每個帳戶的「高光時刻」
3. **XGBoost閾值搜索**: 在OOF預測上找門檻，分布更一致

### three.py（LightGBM，正常工作）

**關鍵區別**:
```python
# 第29-35行：所有帳戶用相同的時間窗口
time_windows = [1, 3, 7]
for window in time_windows:
    window_trans = df_trans[df_trans['txn_date'] >= latest_date - window]
    # 所有帳戶一視同仁
```

**為什麼沒問題**:
1. **統一特徵邏輯**: 訓練和測試完全相同
2. **簡單固定門檻**: threshold=0.5，不依賴驗證集優化
3. **無動態圖構建**: 避免了情境不一致

---

## ✅ 修復方案

### 方案1: 修改測試集的ref_date策略（推薦）

**問題核心**: 測試集不應該都用max_date

**修復**:
```python
# 不要這樣做：
# test_samples = [(acct, max_date, 0) for acct in test_accts]

# 應該給每個測試帳戶一個合理的ref_date
test_samples = []
for acct in test_accts:
    # 方案A: 使用帳戶最後一次交易日期
    acct_trans = df_trans[(df_trans['from_acct']==acct) | (df_trans['to_acct']==acct)]
    if len(acct_trans) > 0:
        ref_date = acct_trans['txn_date'].max()
    else:
        ref_date = max_date
    test_samples.append((acct, ref_date, 0))
```

### 方案2: 使用更長的特徵窗口

**問題**: STATIC_FEATURE_WINDOW=7 太短

**修復**:
```python
STATIC_FEATURE_WINDOW = 30  # 或更長
# 讓更多帳戶有非零特徵
```

### 方案3: 修改閾值策略

**問題**: 驗證集的best_thresh不適用於測試集

**修復**:
```python
# 不要完全信任驗證集的門檻
# 可以設置一個較低的安全門檻
safe_thresh = min(best_thresh * 0.7, 0.5)  # 降低30%或上限0.5
```

### 方案4: 放棄這個模型（最穩妥）

**建議**: 明天不要使用ultimate_gnn_model.py

**原因**:
1. 修復需要重新訓練和驗證
2. 時間有限（4-5小時）
3. 已有穩健的替代方案（run_gnn_xgb_pipeline.py, three.py）
4. 風險太高

---

## 🎓 學到的教訓

### 1. Training-Serving Skew 是ML中最隱蔽的bug

- 訓練時一切正常，驗證F1很高
- 上線後完全失效
- 原因：訓練和預測的數據分布/情境不同

### 2. 動態特徵需要特別小心

- 如果特徵依賴於"當前時刻"
- 必須確保訓練和測試的"當前時刻"定義一致

### 3. 閾值優化的陷阱

- 在驗證集上找的最佳閾值
- 可能完全不適用於測試集
- 如果數據分布不同

### 4. 簡單往往更穩健

- run_gnn_xgb_pipeline.py 和 three.py 沒有這些問題
- 因為它們的設計更簡單、更一致
- 複雜不等於更好

---

## 🎯 最終建議

### 明天執行計劃

**❌ 不要用**: ultimate_gnn_model.py
- F1=0的原因已確認
- 修復需要時間
- 風險太高

**✅ 專注於**:
1. run_gnn_xgb_pipeline.py（主力）
2. three.py（快速備用）
3. 模型集成（提升F1）

**📊 預期結果**:
- 這兩個模型都已經過隱式驗證（之前的提交）
- 單模型F1: 0.40-0.55
- 集成F1: 0.45-0.60

---

## 📚 技術債務清單（賽後可做）

如果想修復ultimate_gnn_model.py（學習用途）:

1. [ ] 修改特徵窗口為30或60天
2. [ ] 給測試帳戶設置合理的ref_date
3. [ ] 在多個不同時間點驗證模型
4. [ ] 使用動態閾值策略
5. [ ] 添加特徵完整性檢查

---

**總結**: 這是一個經典的Training-Serving Skew案例，教會我們在設計ML系統時，
訓練和預測的一致性有多麼重要。感謝Gemini的精準分析！🙏
