"""
ULTIMATE FRAUD DETECTION MODEL
終極版本 - 包含所有優化策略

改進點：
1. 增強特徵工程（加入圖統計特徵）
2. Ensemble: XGBoost + LightGBM + CatBoost
3. 智能閾值優化
4. 特徵選擇和重要性分析
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm
import warnings
import networkx as nx
from collections import Counter
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE FRAUD DETECTION MODEL")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_FOLDS = 5
RANDOM_SEED = 42
USE_ENSEMBLE = True
ADD_GRAPH_FEATURES = True

np.random.seed(RANDOM_SEED)

print(f"Ensemble: {USE_ENSEMBLE}")
print(f"Graph Features: {ADD_GRAPH_FEATURES}")

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[1/6] Loading data...")
df_trans = pd.read_csv('acct_transaction.csv')
df_alert = pd.read_csv('acct_alert.csv')
df_predict = pd.read_csv('acct_predict.csv')

print(f"Transactions: {len(df_trans):,}")
print(f"Alert accounts: {len(df_alert):,}")
print(f"Test accounts: {len(df_predict):,}")

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================
def create_enhanced_features(df_trans, accts, cutoff_date):
    """
    增強版特徵工程 - 包含更多特徵
    """
    print(f"Creating enhanced features for {len(accts):,} accounts (cutoff: {cutoff_date})...")

    df_subset = df_trans[df_trans['txn_date'] <= cutoff_date].copy()
    features = pd.DataFrame(index=accts)
    features.index.name = 'acct'

    # Parse time
    dt_series = pd.to_datetime(df_subset['txn_time'], format='%H:%M:%S', errors='coerce')
    df_subset['hour'] = dt_series.dt.hour
    df_subset['minute'] = dt_series.dt.minute
    df_subset['is_night'] = (df_subset['hour'] < 6) | (df_subset['hour'] >= 22)
    df_subset['is_weekend'] = (df_subset['txn_date'] % 7 >= 5)
    df_subset['is_business_hours'] = (df_subset['hour'] >= 9) & (df_subset['hour'] <= 17)

    # ========================================
    # 1. Basic Transaction Statistics (Enhanced)
    # ========================================
    from_stats = df_subset.groupby('from_acct')['txn_amt'].agg([
        'count', 'sum', 'mean', 'std', 'min', 'max', 'median',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])
    from_stats.columns = ['from_count', 'from_sum', 'from_mean', 'from_std',
                          'from_min', 'from_max', 'from_median', 'from_q25', 'from_q75']
    features = features.join(from_stats, how='left')

    to_stats = df_subset.groupby('to_acct')['txn_amt'].agg([
        'count', 'sum', 'mean', 'std', 'median'
    ])
    to_stats.columns = ['to_count', 'to_sum', 'to_mean', 'to_std', 'to_median']
    features = features.join(to_stats, how='left')

    # ========================================
    # 2. Network Features
    # ========================================
    features['out_degree'] = df_subset.groupby('from_acct')['to_acct'].nunique().reindex(accts).fillna(0)
    features['in_degree'] = df_subset.groupby('to_acct')['from_acct'].nunique().reindex(accts).fillna(0)
    features['total_degree'] = features['out_degree'] + features['in_degree']
    features['degree_ratio'] = features['out_degree'] / (features['in_degree'] + 1e-6)

    # ========================================
    # 3. Advanced Behavioral Features
    # ========================================
    epsilon = 1e-6
    features['avg_out_amt'] = features['from_sum'] / (features['from_count'] + epsilon)
    features['avg_in_amt'] = features['to_sum'] / (features['to_count'] + epsilon)
    features['net_flow'] = features['to_sum'] - features['from_sum']
    features['amt_volatility'] = features['from_std'] / (features['from_mean'] + epsilon)
    features['amt_range'] = features['from_max'] - features['from_min']
    features['amt_iqr'] = features['from_q75'] - features['from_q25']

    # Transaction concentration
    features['out_concentration'] = features['from_count'] / (features['out_degree'] + epsilon)
    features['in_concentration'] = features['to_count'] / (features['in_degree'] + epsilon)

    # Balance ratio
    features['balance_ratio'] = features['to_sum'] / (features['from_sum'] + epsilon)

    # ========================================
    # 4. Enhanced Temporal Features
    # ========================================
    # Night transactions
    night_from = df_subset[df_subset['is_night']].groupby('from_acct').size()
    night_to = df_subset[df_subset['is_night']].groupby('to_acct').size()
    total_txn = features['from_count'] + features['to_count']
    features['night_ratio'] = (night_from.reindex(accts).fillna(0) +
                               night_to.reindex(accts).fillna(0)) / (total_txn + epsilon)

    # Weekend transactions
    weekend_from = df_subset[df_subset['is_weekend']].groupby('from_acct').size()
    features['weekend_ratio'] = weekend_from.reindex(accts).fillna(0) / (features['from_count'] + epsilon)

    # Business hours
    business_from = df_subset[df_subset['is_business_hours']].groupby('from_acct').size()
    features['business_hours_ratio'] = business_from.reindex(accts).fillna(0) / (features['from_count'] + epsilon)

    # Transaction timing patterns
    df_subset['timestamp'] = df_subset['txn_date'] * 86400 + df_subset['hour'] * 3600 + df_subset['minute'] * 60
    df_sorted = df_subset.sort_values(['from_acct', 'timestamp'])
    df_sorted['interval'] = df_sorted.groupby('from_acct')['timestamp'].diff()

    interval_stats = df_sorted.groupby('from_acct')['interval'].agg(['mean', 'std', 'min', 'max'])
    interval_stats.columns = ['interval_mean', 'interval_std', 'interval_min', 'interval_max']
    features = features.join(interval_stats, how='left')

    # ========================================
    # 5. Suspicious Pattern Features (Enhanced)
    # ========================================
    # Round numbers
    round_from = df_subset[df_subset['txn_amt'] % 1000 == 0].groupby('from_acct').size()
    features['round_ratio'] = round_from.reindex(accts).fillna(0) / (features['from_count'] + epsilon)

    # Just-under-threshold
    under_30k = df_subset[(df_subset['txn_amt'] >= 29000) & (df_subset['txn_amt'] < 30000)].groupby('from_acct').size()
    under_50k = df_subset[(df_subset['txn_amt'] >= 49000) & (df_subset['txn_amt'] < 50000)].groupby('from_acct').size()
    features['under_threshold_ratio'] = (under_30k.reindex(accts).fillna(0) +
                                         under_50k.reindex(accts).fillna(0)) / (features['from_count'] + epsilon)

    # Large transactions (> 90th percentile)
    amt_90th = df_subset['txn_amt'].quantile(0.9)
    large_txn = df_subset[df_subset['txn_amt'] > amt_90th].groupby('from_acct').size()
    features['large_txn_ratio'] = large_txn.reindex(accts).fillna(0) / (features['from_count'] + epsilon)

    # Small transactions (< 10th percentile)
    amt_10th = df_subset['txn_amt'].quantile(0.1)
    small_txn = df_subset[df_subset['txn_amt'] < amt_10th].groupby('from_acct').size()
    features['small_txn_ratio'] = small_txn.reindex(accts).fillna(0) / (features['from_count'] + epsilon)

    # ========================================
    # 6. Time Window Features (Multiple windows)
    # ========================================
    for window in [3, 7, 14, 30]:
        window_start = cutoff_date - window
        df_window = df_subset[df_subset['txn_date'] >= window_start]

        window_from = df_window.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
        window_from.columns = [f'from_count_{window}d', f'from_sum_{window}d', f'from_mean_{window}d']
        features = features.join(window_from, how='left')

    # Velocity features
    features['velocity_3d_vs_30d'] = features['from_count_3d'] / (features['from_count_30d'] + epsilon)
    features['velocity_7d_vs_30d'] = features['from_count_7d'] / (features['from_count_30d'] + epsilon)
    features['velocity_14d_vs_30d'] = features['from_count_14d'] / (features['from_count_30d'] + epsilon)

    # Amount velocity
    features['amt_velocity_7d_vs_30d'] = features['from_sum_7d'] / (features['from_sum_30d'] + epsilon)

    # ========================================
    # 7. Account Age and Activity
    # ========================================
    first_from = df_subset.groupby('from_acct')['txn_date'].min()
    first_to = df_subset.groupby('to_acct')['txn_date'].min()
    last_from = df_subset.groupby('from_acct')['txn_date'].max()

    features['first_txn_date'] = pd.concat([first_from, first_to], axis=1).min(axis=1).reindex(accts)
    features['last_txn_date'] = last_from.reindex(accts)
    features['account_age'] = cutoff_date - features['first_txn_date']
    features['days_since_last_txn'] = cutoff_date - features['last_txn_date']
    features['txn_per_day'] = features['from_count'] / (features['account_age'] + epsilon)
    features['active_days_ratio'] = (features['last_txn_date'] - features['first_txn_date']) / (features['account_age'] + epsilon)

    # Fill NaN and inf
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    print(f"Generated {len(features.columns)} enhanced features")

    return features

# ============================================================================
# GRAPH-BASED FEATURES (NetworkX)
# ============================================================================
def add_graph_features(df_trans, accts, cutoff_date, features):
    """
    使用NetworkX計算圖特徵
    """
    if not ADD_GRAPH_FEATURES:
        return features

    print("Computing graph-based features...")

    df_subset = df_trans[df_trans['txn_date'] <= cutoff_date].copy()

    # Build directed graph
    G = nx.DiGraph()

    # Add edges with weights
    edge_list = df_subset[['from_acct', 'to_acct', 'txn_amt']].values
    for from_acct, to_acct, amt in edge_list:
        if G.has_edge(from_acct, to_acct):
            G[from_acct][to_acct]['weight'] += amt
            G[from_acct][to_acct]['count'] += 1
        else:
            G.add_edge(from_acct, to_acct, weight=amt, count=1)

    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Compute PageRank
    print("  Computing PageRank...")
    try:
        pagerank = nx.pagerank(G, max_iter=50)
        features['pagerank'] = pd.Series(pagerank).reindex(accts).fillna(0)
    except:
        features['pagerank'] = 0

    # Compute Betweenness Centrality (sampled for speed)
    print("  Computing Betweenness...")
    try:
        sample_size = min(10000, G.number_of_nodes())
        betweenness = nx.betweenness_centrality(G, k=sample_size)
        features['betweenness'] = pd.Series(betweenness).reindex(accts).fillna(0)
    except:
        features['betweenness'] = 0

    # Compute Clustering Coefficient
    print("  Computing Clustering...")
    try:
        clustering = nx.clustering(G.to_undirected())
        features['clustering'] = pd.Series(clustering).reindex(accts).fillna(0)
    except:
        features['clustering'] = 0

    # In/Out strength (weighted degree)
    features['in_strength'] = 0
    features['out_strength'] = 0

    for acct in accts:
        if G.has_node(acct):
            features.loc[acct, 'in_strength'] = sum([G[u][acct]['weight'] for u in G.predecessors(acct)])
            features.loc[acct, 'out_strength'] = sum([G[acct][v]['weight'] for v in G.successors(acct)])

    print(f"Total features after graph: {len(features.columns)}")

    return features

# ============================================================================
# ENSEMBLE MODEL TRAINING
# ============================================================================
def train_ensemble_models(X_train, y_train, X_test):
    """
    訓練 XGBoost + LightGBM + CatBoost ensemble
    """
    print("\n[4/6] Training ensemble models...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))

    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            tree_method='gpu_hist',
            verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=7,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            device='gpu',
            verbosity=-1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        test_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.02,
            depth=7,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            task_type='GPU',
            verbose=False
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS

        # Fold performance
        oof_ensemble = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cat[val_idx]) / 3
        val_f1 = f1_score(y_val, (oof_ensemble > 0.5).astype(int))
        print(f"Ensemble F1: {val_f1:.4f}")

    # Weighted average (可調整權重)
    oof_preds = (oof_xgb * 0.4 + oof_lgb * 0.3 + oof_cat * 0.3)
    test_preds = (test_xgb * 0.4 + test_lgb * 0.3 + test_cat * 0.3)

    return oof_preds, test_preds

# ============================================================================
# SMART THRESHOLD OPTIMIZATION
# ============================================================================
def optimize_threshold_smart(y_true, y_pred):
    """
    智能閾值優化 - 考慮多個指標
    """
    print("\n[5/6] Smart threshold optimization...")

    best_f1 = 0
    best_threshold = 0.5
    threshold_scores = []

    for threshold in np.arange(0.01, 0.99, 0.01):
        pred_binary = (y_pred > threshold).astype(int)

        if pred_binary.sum() == 0:
            continue

        f1 = f1_score(y_true, pred_binary)
        precision = precision_score(y_true, pred_binary)
        recall = recall_score(y_true, pred_binary)

        threshold_scores.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'predicted_positive': pred_binary.sum()
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Print top thresholds
    df_scores = pd.DataFrame(threshold_scores).sort_values('f1', ascending=False)
    print("\nTop 5 thresholds by F1:")
    print(df_scores.head())

    print(f"\nBest F1: {best_f1:.5f} at threshold: {best_threshold:.4f}")

    return best_threshold, best_f1, df_scores

# ============================================================================
# MAIN PIPELINE
# ============================================================================
print("\n[2/6] Preparing data...")

train_cutoff = df_alert['event_date'].max()
test_cutoff = df_trans['txn_date'].max()

all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
alert_accts = df_alert['acct'].unique()
test_accts = df_predict['acct'].unique()
train_pool_accts = np.setdiff1d(all_accts, test_accts)

print(f"Train accounts: {len(train_pool_accts):,}")
print(f"Test accounts: {len(test_accts):,}")
print(f"Alert accounts: {len(alert_accts):,}")

# Feature generation
print("\n[3/6] Feature generation...")
train_features = create_enhanced_features(df_trans, train_pool_accts, train_cutoff)
test_features = create_enhanced_features(df_trans, test_accts, test_cutoff)

# Add graph features
train_features = add_graph_features(df_trans, train_pool_accts, train_cutoff, train_features)
test_features = add_graph_features(df_trans, test_accts, test_cutoff, test_features)

print(f"\nFinal feature count: {len(train_features.columns)}")

# Labels
y_train = pd.Series(0, index=train_pool_accts)
y_train.loc[y_train.index.isin(alert_accts)] = 1

# Feature alignment
common_features = train_features.columns.tolist()
X_train = train_features[common_features].replace([np.inf, -np.inf], 0).fillna(0)
X_test = test_features[common_features].replace([np.inf, -np.inf], 0).fillna(0)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=common_features, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=common_features, index=X_test.index)

# Train models
if USE_ENSEMBLE:
    oof_preds, test_preds = train_ensemble_models(X_train, y_train, X_test)
else:
    # Single XGBoost
    oof_preds, test_preds = train_single_xgboost(X_train, y_train, X_test)

# Optimize threshold
best_threshold, best_f1, df_scores = optimize_threshold_smart(y_train, oof_preds)

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print("\n[6/6] Generating submission...")

final_oof_pred = (oof_preds > best_threshold).astype(int)
print("\nOOF Performance:")
print(classification_report(y_train, final_oof_pred))

# Try multiple thresholds for submission
print("\nGenerating predictions with different strategies...")

# Strategy 1: Best F1 threshold
pred_best_f1 = (test_preds > best_threshold).astype(int)
print(f"Strategy 1 (Best F1={best_f1:.4f}, threshold={best_threshold:.4f}): {pred_best_f1.sum()} positives")

# Strategy 2: Top percentile
percentile_threshold = np.percentile(test_preds, 99)  # Top 1%
pred_percentile = (test_preds > percentile_threshold).astype(int)
print(f"Strategy 2 (Top 1%): {pred_percentile.sum()} positives")

# Strategy 3: Expected positive rate
expected_rate = len(alert_accts) / len(train_pool_accts)
n_expected = int(len(test_accts) * expected_rate * 5)  # 5x buffer
top_n_indices = np.argsort(test_preds)[-n_expected:]
pred_top_n = np.zeros(len(test_accts), dtype=int)
pred_top_n[top_n_indices] = 1
print(f"Strategy 3 (Expected rate x5): {pred_top_n.sum()} positives")

# Use the one with most predictions (more recall)
final_test_pred = pred_top_n if pred_top_n.sum() > pred_best_f1.sum() else pred_best_f1

submission = pd.DataFrame({'acct': test_accts, 'label': final_test_pred})
submission.to_csv('submission_ultimate.csv', index=False)

print(f"\n✅ Submission saved: submission_ultimate.csv")
print(f"Predicted {final_test_pred.sum():,} alert accounts out of {len(test_accts):,}")
print(f"Positive rate: {final_test_pred.sum() / len(test_accts) * 100:.2f}%")

# Save all predictions for analysis
pd.DataFrame({
    'acct': test_accts,
    'score': test_preds,
    'pred_best_f1': pred_best_f1,
    'pred_percentile': pred_percentile,
    'pred_top_n': pred_top_n,
    'final_pred': final_test_pred
}).to_csv('prediction_analysis.csv', index=False)

print("="*80)
print("DONE!")
print("="*80)
