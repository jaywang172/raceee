"""
快速測試版 - 驗證核心突破思路
只用最關鍵的特徵快速驗證是否有效
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict

print("="*80)
print("快速測試版 - 驗證核心思路")
print("="*80)

# 加載數據
print("\n加載數據...")
df_trans = pd.read_csv('acct_transaction.csv')
df_alert = pd.read_csv('acct_alert.csv')
df_predict = pd.read_csv('acct_predict.csv')

# 構建交易圖
print("構建交易網絡...")
graph = defaultdict(lambda: {'out': set(), 'in': set(), 'out_amt': [], 'in_amt': []})

for _, row in tqdm(df_trans.iterrows(), total=len(df_trans), desc="Building graph"):
    from_acct = row['from_acct']
    to_acct = row['to_acct']
    amt = row['txn_amt']
    
    graph[from_acct]['out'].add(to_acct)
    graph[from_acct]['out_amt'].append(amt)
    graph[to_acct]['in'].add(from_acct)
    graph[to_acct]['in_amt'].append(amt)

# 計算風險傳播分數
print("\n計算風險傳播...")
alert_accts = set(df_alert['acct'])
risk_score = defaultdict(float)

# 初始化：alert帳戶=1
for acct in alert_accts:
    risk_score[acct] = 1.0

# 3次迭代傳播
for iteration in range(3):
    new_risk = defaultdict(float)
    for acct in tqdm(graph.keys(), desc=f"Propagation iter {iteration+1}"):
        # 從鄰居獲得風險
        for in_acct in graph[acct]['in']:
            new_risk[acct] += risk_score[in_acct] * 0.7
        for out_acct in graph[acct]['out']:
            new_risk[acct] += risk_score[out_acct] * 0.3
        
        # 保留原有風險
        new_risk[acct] = max(new_risk[acct], risk_score[acct] * 0.9)
    
    risk_score = new_risk

print(f"風險分數範圍: {min(risk_score.values()):.4f} - {max(risk_score.values()):.4f}")

# 特徵工程
print("\n特徵工程...")

def extract_features(acct, ref_date):
    """提取核心特徵"""
    features = {}
    
    # 獲取交易
    acct_txns = df_trans[
        ((df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)) &
        (df_trans['txn_date'] < ref_date)
    ]
    
    # 短期窗口統計（從exp3學到）
    for window in [1, 3, 7]:
        window_txns = acct_txns[acct_txns['txn_date'] >= ref_date - window]
        
        from_txns = window_txns[window_txns['from_acct'] == acct]
        features[f'from_count_{window}d'] = len(from_txns)
        features[f'from_amt_sum_{window}d'] = from_txns['txn_amt'].sum()
        features[f'from_amt_mean_{window}d'] = from_txns['txn_amt'].mean() if len(from_txns) > 0 else 0
        features[f'from_unique_{window}d'] = from_txns['to_acct'].nunique()
        
        to_txns = window_txns[window_txns['to_acct'] == acct]
        features[f'to_count_{window}d'] = len(to_txns)
        features[f'to_amt_sum_{window}d'] = to_txns['txn_amt'].sum()
        features[f'to_amt_mean_{window}d'] = to_txns['txn_amt'].mean() if len(to_txns) > 0 else 0
    
    # 簡單比率（從exp3學到）
    epsilon = 1e-8
    features['ratio_from_amt_3d_7d'] = features['from_amt_sum_3d'] / (features['from_amt_sum_7d'] + epsilon)
    features['ratio_from_cnt_1d_7d'] = features['from_count_1d'] / (features['from_count_7d'] + epsilon)
    features['ratio_to_amt_3d_7d'] = features['to_amt_sum_3d'] / (features['to_amt_sum_7d'] + epsilon)
    
    # 網絡特徵（新增）
    features['out_degree'] = len(graph[acct]['out'])
    features['in_degree'] = len(graph[acct]['in'])
    features['degree_ratio'] = features['out_degree'] / (features['in_degree'] + epsilon)
    features['risk_score'] = risk_score.get(acct, 0)
    
    # 領域特徵（新增）
    if len(acct_txns) > 0:
        features['atm_ratio'] = (acct_txns['channel_type'] == '1').mean()
        features['night_ratio'] = 0  # 簡化版暫不計算
        
        # 快速轉移
        recent_7d = acct_txns[acct_txns['txn_date'] >= ref_date - 7]
        features['recent_activity'] = len(recent_7d) / (len(acct_txns) + epsilon)
    else:
        features['atm_ratio'] = 0
        features['night_ratio'] = 0
        features['recent_activity'] = 0
    
    return features

# 創建訓練樣本
print("\n創建訓練樣本...")
positive_samples = [(row['acct'], row['event_date'], 1) for _, row in df_alert.iterrows()]

all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
non_alert_accts = np.setdiff1d(all_accts, list(alert_accts))

np.random.seed(42)
selected_neg = np.random.choice(non_alert_accts, len(positive_samples) * 8, replace=False)

negative_samples = []
for acct in tqdm(selected_neg, desc="Sampling negatives"):
    dates = df_trans[(df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)]['txn_date'].unique()
    if len(dates) > 2:
        valid_dates = dates[(dates > dates.min()) & (dates < dates.max())]
        if len(valid_dates) > 0:
            pseudo_date = np.random.choice(valid_dates)
            negative_samples.append((acct, pseudo_date, 0))

all_samples = positive_samples + negative_samples
print(f"總樣本: {len(all_samples)}")

# 提取特徵
feature_list = []
for acct, ref_date, label in tqdm(all_samples, desc="Extracting features"):
    features = extract_features(acct, ref_date)
    features['acct'] = acct
    features['label'] = label
    features['ref_date'] = ref_date
    feature_list.append(features)

df_features = pd.DataFrame(feature_list).fillna(0)

# 時間分割
TIME_SPLIT = 90
train_mask = df_features['ref_date'] <= TIME_SPLIT
val_mask = df_features['ref_date'] > TIME_SPLIT

feature_cols = [col for col in df_features.columns if col not in ['acct', 'label', 'ref_date']]

X_train = df_features.loc[train_mask, feature_cols].values
y_train = df_features.loc[train_mask, 'label'].values
X_val = df_features.loc[val_mask, feature_cols].values
y_val = df_features.loc[val_mask, 'label'].values

print(f"\n訓練集: {len(X_train)} (正樣本率: {y_train.mean():.4f})")
print(f"驗證集: {len(X_val)} (正樣本率: {y_val.mean():.4f})")

# 訓練模型
print("\n訓練LightGBM...")
scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1e-8)

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# 驗證
val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)

# 找最佳閾值
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_val, val_pred > thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n{'='*80}")
print(f"驗證結果:")
print(f"{'='*80}")
print(f"AUC: {val_auc:.4f}")
print(f"最佳F1: {best_f1:.6f}")
print(f"最佳閾值: {best_thresh:.4f}")

# 特徵重要性
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 重要特徵:")
print(importance.head(15).to_string(index=False))

# 測試集預測
print(f"\n{'='*80}")
print("測試集預測...")
print(f"{'='*80}")

test_accts = df_predict['acct'].tolist()
max_date = df_trans['txn_date'].max()

test_features = []
for acct in tqdm(test_accts, desc="Test features"):
    features = extract_features(acct, max_date)
    test_features.append(features)

df_test = pd.DataFrame(test_features).fillna(0)
X_test = df_test[feature_cols].values

test_pred = model.predict_proba(X_test)[:, 1]
final_labels = (test_pred > best_thresh).astype(int)

# 保存
submission = pd.DataFrame({'acct': test_accts, 'label': final_labels})
submission.to_csv('submission_quick_test.csv', index=False)

print(f"\n{'='*80}")
print(f"完成!")
print(f"{'='*80}")
print(f"提交文件: submission_quick_test.csv")
print(f"驗證F1: {best_f1:.6f}")
print(f"陽性預測: {final_labels.sum()} / {len(final_labels)} ({final_labels.mean()*100:.2f}%)")

# 與exp3比較
print(f"\n與exp3 (0.1067762) 比較:")
if best_f1 > 0.1067:
    print(f"✅ 提升: {(best_f1 - 0.1067) / 0.1067 * 100:.2f}%")
else:
    print(f"❌ 下降: {(0.1067 - best_f1) / 0.1067 * 100:.2f}%")

print(f"\n關鍵創新:")
print(f"  1. 風險傳播分數 (重要性: {importance[importance['feature']=='risk_score']['importance'].values[0] if 'risk_score' in importance['feature'].values else 0:.1f})")
print(f"  2. 網絡度數特徵")
print(f"  3. 領域知識特徵")
print(f"={'='*80}")
