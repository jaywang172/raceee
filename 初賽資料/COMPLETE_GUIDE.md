# 🚀 Ultimate Fraud Detection - 完整方案總結

## 📊 競賽分析

### 當前狀況
- **最佳成績**: 0.1068 (submission_exp3_robust_anomaly.csv)
- **數據規模**: 
  - 交易記錄: 4,435,891筆
  - 警示帳戶: 1,005個
  - 待預測帳戶: 4,780個
- **評分標準**: F1-score（Precision和Recall的調和平均）

### 競賽難點
1. **極度不平衡**: 正樣本（詐騙）< 1%
2. **時序性**: 需要考慮交易發生的時間順序
3. **圖結構**: 帳戶間的轉帳構成複雜網絡
4. **特徵噪音**: 正常交易和詐騙交易模式有重疊

---

## 🎯 解決方案架構

### 核心創新點

#### 1. **超級特徵工程** (200+ features)
```
時間維度 (12個窗口):
├─ 統計特徵: 均值、標準差、最大/最小值、分位數
├─ 加速度特徵: 短期vs長期變化率
├─ 趨勢特徵: 線性擬合斜率
└─ 異常特徵: Z-score、熵

圖結構維度:
├─ 度數特徵: 入度、出度、總度數
├─ 鄰居特徵: 一階、二階鄰居數量
├─ 集中度: Herfindahl指數
└─ PageRank風格分數

行為模式維度:
├─ 淨流量: 轉入-轉出
├─ 時間模式: 夜間交易、工作時間交易
├─ 通路偏好: ATM、網銀、臨櫃比例
└─ 自我交易比例
```

#### 2. **多模型Ensemble**
```
模型1: LightGBM (GPU加速)
  ├─ 3000棵樹
  ├─ 127片葉子
  └─ AUC權重: ~35%

模型2: XGBoost (GPU加速)  
  ├─ 3000棵樹
  ├─ 深度10
  └─ AUC權重: ~35%

模型3: Deep Neural Network
  ├─ 4層 [1024→512→256→128]
  ├─ Batch Normalization
  ├─ Dropout 0.2
  ├─ 混合精度訓練 (A100優化)
  └─ AUC權重: ~30%

可選模型4: Graph Attention Network
  ├─ 4層GATv2
  ├─ 8個注意力頭
  └─ 全局池化 (mean + max)
```

#### 3. **智能閾值優化**
- 在驗證集上搜索0.05-0.95，步長0.01
- 最大化F1-score而非AUC
- 考慮Precision-Recall權衡

---

## 📁 文件說明

### 主要訓練腳本

| 文件名 | 說明 | 推薦場景 | 預計時間 |
|--------|------|----------|----------|
| `a100_optimized_model.py` | A100優化模型，深度學習+樹模型 | **推薦首選** | 60-90分鐘 |
| `ultimate_ensemble_model.py` | 完整ensemble，含GNN | 追求極致性能 | 120-180分鐘 |
| `quick_run.py` | 自動化運行腳本 | 快速開始 | - |

### 輔助工具

| 文件名 | 用途 |
|--------|------|
| `model_analyzer.py` | 分析特徵重要性、模型性能 |
| `config.ini` | 配置文件，調整超參數 |
| `install_dependencies.sh` | 自動安裝依賴 |
| `README_ULTIMATE.md` | 詳細文檔 |

---

## 🚀 快速開始（3步驟）

### Step 1: 安裝依賴
```bash
cd 初賽資料
chmod +x install_dependencies.sh
./install_dependencies.sh
```

或手動安裝：
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install lightgbm xgboost catboost
pip install pandas numpy scikit-learn tqdm
```

### Step 2: 訓練模型
```bash
# 方法A: 使用快速運行腳本（推薦）
python quick_run.py

# 方法B: 直接運行
python a100_optimized_model.py
```

### Step 3: 提交結果
```bash
# 檢查生成的文件
ls -lh submission_*.csv

# 上傳到T-Brain平台
# submission_a100_optimized.csv
```

---

## 🎨 特徵工程詳解

### 為什麼這些特徵有效？

#### 時間窗口特徵
```python
# 詐騙帳戶特徵:
✓ 短期內大量交易 (from_count_1d, from_count_3d高)
✓ 金額突然暴增 (from_amt_accel_1_7高)
✓ 快速清空帳戶 (net_flow_7d負且絕對值大)
✓ 交易對手高度分散 (from_unique_accts_7d高)

# 正常帳戶特徵:
✓ 交易頻率穩定 (accel特徵接近1)
✓ 固定交易對手 (unique_accts低)
✓ 交易金額穩定 (std低)
```

#### 圖結構特徵
```python
# 詐騙帳戶在圖中的特徵:
✓ 出度遠大於入度 (快速轉出)
✓ 二階鄰居數量多 (中繼節點)
✓ 對手集中度低 (廣撒網)

# PageRank思想:
✓ 與多個高風險帳戶交易 → 自己也高風險
```

#### 異常檢測特徵
```python
# Z-score異常:
✓ 相對全局: amt_zscore_global
✓ 相對歷史: recent_amt_zscore

# 時間異常:
✓ 夜間交易比例高 (0-6點)
✓ 交易間隔突然變化
```

---

## 🔧 調優指南

### 階段1: 基礎優化（提升到0.11-0.12）

```python
# 在a100_optimized_model.py中調整:

# 1. 增加負樣本比例
NEG_POS_RATIO = 6  # 從5增加到6

# 2. 更多時間窗口
TIME_WINDOWS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120]

# 3. 調整閾值
# 模型會自動搜索，但可以手動微調
best_threshold = 0.42  # 根據驗證集調整
```

### 階段2: 深度優化（提升到0.12-0.13）

```python
# 深度學習模型
HIDDEN_DIM = 1024  # 增加容量
EPOCHS = 40        # 更多訓練
DROPOUT = 0.3      # 防止過擬合

# LightGBM
lgb_params = {
    'num_leaves': 255,      # 增加
    'learning_rate': 0.01,  # 降低並增加n_estimators
    'n_estimators': 5000,
}

# XGBoost  
xgb_params = {
    'max_depth': 12,        # 增加
    'learning_rate': 0.01,
    'n_estimators': 5000,
}
```

### 階段3: 高級技巧（提升到0.13+）

#### Technique 1: Test Time Augmentation (TTA)
```python
# 對測試集使用多個參考時間點
max_date = df_trans['txn_date'].max()
ref_dates = [max_date, max_date-2, max_date-5, max_date-7]

predictions = []
for ref_date in ref_dates:
    pred = model.predict(...)
    predictions.append(pred)

final_pred = np.mean(predictions, axis=0)
```

#### Technique 2: Pseudo-Labeling
```python
# 第一輪訓練後
high_conf_mask = (test_pred > 0.95) | (test_pred < 0.05)
pseudo_labels = (test_pred > 0.5).astype(int)

# 將高置信度樣本加入訓練集
X_train_extended = np.vstack([X_train, X_test[high_conf_mask]])
y_train_extended = np.hstack([y_train, pseudo_labels[high_conf_mask]])

# 重新訓練
```

#### Technique 3: Stacking
```python
# Level 1: 基礎模型
lgb_pred = lgb_model.predict_proba(X)[:, 1]
xgb_pred = xgb_model.predict_proba(X)[:, 1]
deep_pred = deep_model.predict(X)

# Level 2: Meta模型
meta_features = np.column_stack([lgb_pred, xgb_pred, deep_pred])
meta_model = LogisticRegression()
meta_model.fit(meta_features, y)
```

#### Technique 4: Feature Selection
```python
# 基於重要性選擇Top-K特徵
importance = lgb_model.feature_importances_
top_k_indices = np.argsort(importance)[-150:]  # Top 150
X_selected = X[:, top_k_indices]

# 或使用遞歸特徵消除
from sklearn.feature_selection import RFE
selector = RFE(lgb_model, n_features_to_select=150)
X_selected = selector.fit_transform(X, y)
```

---

## 📊 預期性能

| 方法 | 預期F1 | 訓練時間 | GPU記憶體 |
|------|--------|----------|-----------|
| 當前最佳 | 0.1068 | - | - |
| A100 Basic | 0.110-0.120 | 60分鐘 | 8GB |
| A100 Optimized | 0.120-0.130 | 90分鐘 | 12GB |
| Ultimate Ensemble | 0.125-0.135 | 150分鐘 | 16GB |
| + TTA | 0.130-0.140 | +30分鐘 | - |
| + Pseudo-labeling | 0.135-0.145 | +60分鐘 | - |

---

## 🐛 常見問題

### Q1: GPU記憶體不足
```python
# 解決方案1: 減小批量
BATCH_SIZE = 128  # 或64

# 解決方案2: 減小模型
HIDDEN_DIM = 256

# 解決方案3: 禁用AMP
USE_AMP = False
```

### Q2: 訓練太慢
```python
# 解決方案1: 減少樣本
NEG_POS_RATIO = 3

# 解決方案2: 減少epochs
EPOCHS = 15

# 解決方案3: 使用更少特徵
TIME_WINDOWS = [1, 3, 7, 14, 30]
```

### Q3: 過擬合
```python
# 解決方案1: 增加正則化
DROPOUT = 0.4
reg_alpha = 0.5
reg_lambda = 2.0

# 解決方案2: Early stopping
early_stopping_rounds = 100

# 解決方案3: 減小模型容量
num_leaves = 31
max_depth = 6
```

### Q4: F1分數不理想
```python
# 檢查清單:
✓ 驗證集正負樣本比例是否合理
✓ 特徵是否有NaN或Inf
✓ 閾值是否最優
✓ 模型是否收斂
✓ 特徵工程是否充分

# 使用分析工具:
python model_analyzer.py
```

---

## 📈 性能監控

### 訓練過程中關注的指標

```python
# 好的訓練曲線:
Epoch 1  | Loss: 0.35 | Val F1: 0.08 | Val AUC: 0.75
Epoch 5  | Loss: 0.25 | Val F1: 0.10 | Val AUC: 0.82
Epoch 10 | Loss: 0.20 | Val F1: 0.11 | Val AUC: 0.85
Epoch 15 | Loss: 0.18 | Val F1: 0.12 | Val AUC: 0.86  ← 最佳
Epoch 20 | Loss: 0.16 | Val F1: 0.11 | Val AUC: 0.87  ← 開始過擬合

# 警告信號:
⚠ Loss持續下降但F1不提升 → 過擬合
⚠ Train F1 >> Val F1 → 過擬合
⚠ F1和AUC都很低 → 欠擬合或特徵問題
```

---

## 🎯 提交策略

### 每日3次提交機會的使用策略

#### 提交1: 穩健模型
```bash
# 使用保守的參數
NEG_POS_RATIO = 4
threshold = 0.45  # 稍高，提高precision
python a100_optimized_model.py
```

#### 提交2: 激進模型  
```bash
# 追求recall
NEG_POS_RATIO = 6
threshold = 0.35  # 稍低，提高recall
python a100_optimized_model.py
```

#### 提交3: Ensemble
```bash
# 綜合兩者
predictions = 0.6 * pred_conservative + 0.4 * pred_aggressive
```

### Public vs Private LB

```python
# Public LB優化可能導致過擬合
# 策略:
1. 以驗證集F1為主要指標
2. Public LB作為參考
3. 保持模型穩健性（不要過度調參）
4. 最後選擇驗證集表現最穩定的模型
```

---

## 🏆 競賽建議

### 時間規劃（假設11/12截止）

**Week 1-2: 建立Baseline**
- [ ] 運行a100_optimized_model.py
- [ ] 獲得初始分數
- [ ] 分析特徵重要性
- [ ] 理解數據模式

**Week 3-4: 特徵工程**
- [ ] 添加新的時間窗口
- [ ] 實驗圖特徵
- [ ] 嘗試異常檢測特徵
- [ ] A/B測試不同特徵組合

**Week 5-6: 模型優化**
- [ ] 超參數調優
- [ ] 嘗試不同的ensemble權重
- [ ] 實現TTA
- [ ] 閾值精細調整

**最後3天: 穩定化**
- [ ] 選擇最穩定的模型
- [ ] 多次驗證
- [ ] 準備文檔
- [ ] 最終提交

### 複賽準備

如果進入複賽，需要準備：
1. **代碼文檔**: 清晰的README和註釋
2. **運行說明**: 如何復現結果
3. **環境文件**: requirements.txt
4. **簡報**: 解釋方法和創新點
5. **消融實驗**: 各個模塊的貢獻

---

## 💡 關鍵洞察

### 詐騙帳戶的典型模式

1. **時間特徵**: 開戶後短期內大量交易
2. **金額特徵**: 突然的大額轉出
3. **網絡特徵**: 充當中繼節點
4. **行為特徵**: 夜間交易比例高
5. **通路特徵**: 偏好ATM和網銀

### 容易誤判的情況

**False Positive (誤報)**:
- 新開戶的正常用戶
- 臨時性的大額交易（如買房）
- 企業帳戶（交易量大且多樣）

**False Negative (漏報)**:
- 小額多次的詐騙
- 休眠後突然活躍的帳戶
- 模擬正常交易模式的詐騙

### 針對性優化

```python
# 減少誤報
- 添加帳戶年齡特徵
- 考慮交易對手的可信度
- 使用更保守的閾值

# 減少漏報
- 增加異常檢測特徵
- 關注交易間隔的突變
- 考慮多跳鄰居的風險傳播
```

---

## 📚 參考資料

### 論文
1. "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
2. "Graph Attention Networks"
3. "XGBoost: A Scalable Tree Boosting System"
4. "TabNet: Attentive Interpretable Tabular Learning"

### 相關競賽
1. IEEE-CIS Fraud Detection (Kaggle)
2. Santander Customer Transaction Prediction
3. PAKDD 2009: Credit Card Fraud Detection

---

## ✅ 最終檢查清單

出發前確認：
- [ ] 所有依賴已安裝
- [ ] GPU可正常使用
- [ ] 數據文件在正確位置
- [ ] 有足夠的磁碟空間（至少10GB）
- [ ] 設置了合適的隨機種子
- [ ] 驗證集劃分合理
- [ ] 閾值搜索範圍合適

訓練後確認：
- [ ] 模型收斂（loss下降）
- [ ] 驗證集F1 > 當前最佳
- [ ] 沒有過度過擬合
- [ ] 提交文件格式正確
- [ ] 預測分布合理（5-15%陽性）
- [ ] 保存了模型和日誌

---

## 🎉 結語

這套方案整合了：
- ✅ 深度特徵工程（200+特徵）
- ✅ 先進的模型（樹模型+深度學習+GNN）
- ✅ 智能ensemble（基於AUC的權重）
- ✅ GPU優化（A100混合精度訓練）
- ✅ 完整的工具鏈（訓練、分析、調優）

**預期提升**: 從0.1068 → 0.12-0.14

祝競賽順利！加油！🚀

有任何問題，檢查README_ULTIMATE.md或代碼中的註釋。
