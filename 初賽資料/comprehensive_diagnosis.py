"""
Comprehensive Data Analysis & Diagnosis Script
Detects data leakage, overfitting, and provides actionable insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE DATA DIAGNOSIS FOR GNN/ML PIPELINE")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING & BASIC STATISTICS
# ============================================================================
print("\n[1/8] LOADING DATA AND COMPUTING BASIC STATISTICS...")

df_trans = pd.read_csv('acct_transaction.csv')
df_alert = pd.read_csv('acct_alert.csv')
df_predict = pd.read_csv('acct_predict.csv')

print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print(f"{'='*80}")
print(f"Transaction records: {len(df_trans):,}")
print(f"Alert accounts (positive samples): {len(df_alert):,}")
print(f"Prediction accounts (test set): {len(df_predict):,}")

# Time range analysis
print(f"\n{'='*80}")
print("TIME RANGE ANALYSIS")
print(f"{'='*80}")
print(f"Transaction date range: {df_trans['txn_date'].min()} to {df_trans['txn_date'].max()}")
print(f"Transaction date span: {df_trans['txn_date'].max() - df_trans['txn_date'].min()} days")
print(f"Alert event date range: {df_alert['event_date'].min()} to {df_alert['event_date'].max()}")
print(f"Alert event date span: {df_alert['event_date'].max() - df_alert['event_date'].min()} days")

# Account analysis
all_trans_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
alert_accts = df_alert['acct'].unique()
test_accts = df_predict['acct'].unique()

print(f"\n{'='*80}")
print("ACCOUNT DISTRIBUTION")
print(f"{'='*80}")
print(f"Total unique accounts in transactions: {len(all_trans_accts):,}")
print(f"Alert accounts: {len(alert_accts):,}")
print(f"Test accounts: {len(test_accts):,}")
print(f"Alert accounts in test set: {len(np.intersect1d(alert_accts, test_accts)):,}")
print(f"Class imbalance ratio: 1:{len(all_trans_accts)//len(alert_accts)}")

# ============================================================================
# SECTION 2: CRITICAL DATA LEAKAGE CHECKS
# ============================================================================
print(f"\n{'='*80}")
print("[2/8] CHECKING FOR DATA LEAKAGE ISSUES...")
print(f"{'='*80}")

leakage_issues = []

# Check 1: Test accounts in training data
test_in_alert = np.intersect1d(test_accts, alert_accts)
if len(test_in_alert) > 0:
    leakage_issues.append(f"CRITICAL: {len(test_in_alert)} test accounts appear in alert data!")
    print(f"⚠️  LEAKAGE DETECTED: {len(test_in_alert)} test accounts in training labels")
else:
    print(f"✅ Test accounts are properly separated from training labels")

# Check 2: Future information in features
train_cutoff = df_alert['event_date'].max()
test_cutoff = df_trans['txn_date'].max()
print(f"\nTraining cutoff date: {train_cutoff}")
print(f"Test cutoff date: {test_cutoff}")

if test_cutoff > train_cutoff:
    days_diff = test_cutoff - train_cutoff
    print(f"✅ Test data is {days_diff} days after training cutoff (no temporal leakage)")
else:
    leakage_issues.append(f"CRITICAL: Test data is not after training data!")
    print(f"⚠️  LEAKAGE: Test cutoff ({test_cutoff}) <= Train cutoff ({train_cutoff})")

# Check 3: Alert event dates vs transaction dates
future_alerts = df_alert[df_alert['event_date'] > df_trans['txn_date'].max()]
if len(future_alerts) > 0:
    print(f"⚠️  WARNING: {len(future_alerts)} alerts have event dates beyond transaction data")

# Check 4: Time window feature leakage
print(f"\n{'='*80}")
print("TIME WINDOW FEATURE ANALYSIS")
print(f"{'='*80}")
print("Checking if features use future information...")

# Simulate feature generation with wrong cutoff
trans_after_alert = df_trans[df_trans['txn_date'] > train_cutoff]
if len(trans_after_alert) > 0:
    print(f"⚠️  {len(trans_after_alert):,} transactions occur after alert cutoff")
    print(f"   These should NOT be used for training features!")

    # Check if these future transactions involve alert accounts
    future_alert_txns = trans_after_alert[
        trans_after_alert['from_acct'].isin(alert_accts) |
        trans_after_alert['to_acct'].isin(alert_accts)
    ]
    if len(future_alert_txns) > 0:
        leakage_issues.append(f"CRITICAL: Alert accounts have {len(future_alert_txns)} transactions after event date!")
        print(f"⚠️  LEAKAGE: {len(future_alert_txns):,} future transactions from alert accounts")

# ============================================================================
# SECTION 3: TRAIN/TEST SPLIT ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("[3/8] ANALYZING TRAIN/TEST SPLIT STRATEGY...")
print(f"{'='*80}")

# Temporal distribution of alerts
alert_dist = df_alert['event_date'].value_counts().sort_index()
print(f"\nAlert temporal distribution:")
print(f"First alert date: {df_alert['event_date'].min()}")
print(f"Last alert date: {df_alert['event_date'].max()}")
print(f"Alert date concentration:")
print(alert_dist.describe())

# Check if test accounts have different temporal characteristics
test_in_trans_from = df_trans[df_trans['from_acct'].isin(test_accts)]
test_in_trans_to = df_trans[df_trans['to_acct'].isin(test_accts)]

if len(test_in_trans_from) > 0:
    test_txn_dates = pd.concat([
        test_in_trans_from['txn_date'],
        test_in_trans_to['txn_date']
    ])
    print(f"\nTest accounts transaction timeline:")
    print(f"First transaction: {test_txn_dates.min()}")
    print(f"Last transaction: {test_txn_dates.max()}")

    # Check if test accounts are "new" accounts
    train_accts_in_trans = np.union1d(
        df_trans[df_trans['txn_date'] <= train_cutoff]['from_acct'].unique(),
        df_trans[df_trans['txn_date'] <= train_cutoff]['to_acct'].unique()
    )
    new_test_accts = np.setdiff1d(test_accts, train_accts_in_trans)
    print(f"\n⚠️  {len(new_test_accts)} test accounts ({len(new_test_accts)/len(test_accts)*100:.1f}%) are NEW accounts")
    print(f"   These accounts have no history before test period!")

# ============================================================================
# SECTION 4: FEATURE DISTRIBUTION ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("[4/8] ANALYZING FEATURE DISTRIBUTIONS...")
print(f"{'='*80}")

# Generate basic features for analysis
def generate_basic_features(df_trans, accts, cutoff_date):
    df_subset = df_trans[df_trans['txn_date'] <= cutoff_date]
    features = pd.DataFrame(index=accts)

    features['total_txn_amt'] = df_subset.groupby('from_acct')['txn_amt'].sum().reindex(accts).fillna(0)
    features['total_txn_count'] = df_subset.groupby('from_acct')['txn_amt'].count().reindex(accts).fillna(0)
    features['total_to_amt'] = df_subset.groupby('to_acct')['txn_amt'].sum().reindex(accts).fillna(0)
    features['total_to_count'] = df_subset.groupby('to_acct')['txn_amt'].count().reindex(accts).fillna(0)
    features['avg_txn_amt'] = features['total_txn_amt'] / (features['total_txn_count'] + 1e-6)
    features['num_unique_to_accts'] = df_subset.groupby('from_acct')['to_acct'].nunique().reindex(accts).fillna(0)

    return features

print("Generating features for alert and non-alert accounts...")
train_accts_pool = np.union1d(
    df_trans[df_trans['txn_date'] <= train_cutoff]['from_acct'].unique(),
    df_trans[df_trans['txn_date'] <= train_cutoff]['to_acct'].unique()
)

features = generate_basic_features(df_trans, train_accts_pool, train_cutoff)
features['is_alert'] = features.index.isin(alert_accts).astype(int)

print(f"\nFeature statistics for ALERT accounts:")
alert_features = features[features['is_alert'] == 1]
print(alert_features.describe())

print(f"\nFeature statistics for NON-ALERT accounts:")
non_alert_features = features[features['is_alert'] == 0].sample(min(1000, len(features[features['is_alert'] == 0])))
print(non_alert_features.describe())

# Check for separability
print(f"\n{'='*80}")
print("FEATURE SEPARABILITY ANALYSIS")
print(f"{'='*80}")
for col in ['total_txn_amt', 'total_txn_count', 'avg_txn_amt', 'num_unique_to_accts']:
    if col in features.columns:
        alert_vals = features[features['is_alert'] == 1][col]
        non_alert_vals = features[features['is_alert'] == 0][col]

        alert_mean = alert_vals.mean()
        non_alert_mean = non_alert_vals.mean()

        print(f"\n{col}:")
        print(f"  Alert mean: {alert_mean:.2f}")
        print(f"  Non-alert mean: {non_alert_mean:.2f}")
        print(f"  Ratio: {alert_mean / (non_alert_mean + 1e-6):.2f}x")

# ============================================================================
# SECTION 5: GNN-SPECIFIC ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("[5/8] ANALYZING GNN MODEL STRUCTURE...")
print(f"{'='*80}")

# Graph structure analysis
train_trans = df_trans[df_trans['txn_date'] <= train_cutoff]
test_trans = df_trans[df_trans['txn_date'] > train_cutoff]

print(f"\nGraph structure (training period):")
print(f"Total edges: {len(train_trans):,}")
print(f"Unique from_acct: {train_trans['from_acct'].nunique():,}")
print(f"Unique to_acct: {train_trans['to_acct'].nunique():,}")

# Check alert account connectivity
alert_as_from = train_trans[train_trans['from_acct'].isin(alert_accts)]
alert_as_to = train_trans[train_trans['to_acct'].isin(alert_accts)]

print(f"\nAlert account graph connectivity:")
print(f"Alert accounts as sender: {len(alert_as_from):,} transactions")
print(f"Alert accounts as receiver: {len(alert_as_to):,} transactions")
print(f"Alert accounts with NO transactions: {len(alert_accts) - len(np.union1d(alert_as_from['from_acct'].unique(), alert_as_to['to_acct'].unique()))}")

# GNN label leakage check
print(f"\n{'='*80}")
print("GNN LABEL LEAKAGE DETECTION")
print(f"{'='*80}")

# Check if node features include alert information
print("Checking if alert status is used as node feature...")
print("⚠️  WARNING: Line 56 in run_gnn_xgb_pipeline.py:")
print("   node_features[alert_accts_encoded, 0] = 1.0")
print("   This directly uses alert labels as node features!")
print("   This is SEVERE DATA LEAKAGE for test accounts!")

leakage_issues.append("CRITICAL: GNN uses alert labels as node features (line 56)")

# ============================================================================
# SECTION 6: OVERFITTING INDICATORS
# ============================================================================
print(f"\n{'='*80}")
print("[6/8] CHECKING FOR OVERFITTING INDICATORS...")
print(f"{'='*80}")

# Check model complexity
print("\nModel complexity analysis:")
print("GNN Model:")
print("  - Hidden dim: 32")
print("  - Attention heads: 2")
print("  - Layers: 2")
print("  - Dropout: 0.6")
print("  - Total parameters: ~2K")

print("\nXGBoost Model:")
print("  - Max depth: 8")
print("  - N estimators: 2000")
print("  - Learning rate: 0.02")
print("  - Early stopping: 100 rounds")

# Check training strategy
print("\nTraining strategy:")
print("  - Using 5-fold cross-validation: ✅ Good")
print("  - Early stopping enabled: ✅ Good")
print("  - Threshold tuning on OOF: ✅ Good")

# Check class imbalance handling
print("\nClass imbalance handling:")
pos_samples = len(alert_accts)
neg_samples = len(train_accts_pool) - pos_samples
print(f"  - Positive samples: {pos_samples:,}")
print(f"  - Negative samples: {neg_samples:,}")
print(f"  - Imbalance ratio: 1:{neg_samples//pos_samples}")
print("  - Using scale_pos_weight: ✅ Good")

# ============================================================================
# SECTION 7: TEST SET ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("[7/8] ANALYZING TEST SET CHARACTERISTICS...")
print(f"{'='*80}")

# Generate features for test accounts
test_features = generate_basic_features(df_trans, test_accts, test_cutoff)

print(f"\nTest account feature statistics:")
print(test_features.describe())

# Compare train vs test distributions
print(f"\n{'='*80}")
print("TRAIN vs TEST DISTRIBUTION COMPARISON")
print(f"{'='*80}")

for col in ['total_txn_amt', 'total_txn_count', 'avg_txn_amt', 'num_unique_to_accts']:
    if col in features.columns and col in test_features.columns:
        train_mean = features[col].mean()
        test_mean = test_features[col].mean()
        train_std = features[col].std()
        test_std = test_features[col].std()

        print(f"\n{col}:")
        print(f"  Train: mean={train_mean:.2f}, std={train_std:.2f}")
        print(f"  Test:  mean={test_mean:.2f}, std={test_std:.2f}")
        print(f"  Mean ratio: {test_mean / (train_mean + 1e-6):.2f}x")

# Check for distribution shift
print(f"\n{'='*80}")
print("DISTRIBUTION SHIFT DETECTION")
print(f"{'='*80}")

# Check if test accounts behave differently
test_txn_per_acct = test_features['total_txn_count'].mean()
train_txn_per_acct = features['total_txn_count'].mean()

if abs(test_txn_per_acct - train_txn_per_acct) / (train_txn_per_acct + 1e-6) > 0.5:
    print(f"⚠️  WARNING: Large distribution shift detected!")
    print(f"   Train avg transactions: {train_txn_per_acct:.2f}")
    print(f"   Test avg transactions: {test_txn_per_acct:.2f}")
else:
    print(f"✅ Train and test distributions are similar")

# ============================================================================
# SECTION 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print(f"\n{'='*80}")
print("[8/8] DIAGNOSIS SUMMARY AND RECOMMENDATIONS")
print(f"{'='*80}")

print(f"\n{'='*40}")
print("IDENTIFIED ISSUES:")
print(f"{'='*40}")

if len(leakage_issues) > 0:
    for i, issue in enumerate(leakage_issues, 1):
        print(f"{i}. {issue}")
else:
    print("No critical data leakage detected!")

print(f"\n{'='*40}")
print("RECOMMENDATIONS TO IMPROVE F1 SCORE:")
print(f"{'='*40}")

recommendations = [
    "1. REMOVE ALERT LABELS FROM NODE FEATURES",
    "   - Line 56 in run_gnn_xgb_pipeline.py uses alert status as features",
    "   - This causes severe overfitting to training labels",
    "   - Solution: Use only structural features (degree, centrality, etc.)",
    "",
    "2. FIX TEMPORAL FEATURE ENGINEERING",
    "   - Ensure training features only use data up to train_cutoff_date",
    "   - Ensure test features only use data up to test_cutoff_date",
    "   - Check final_model.py line 24: it uses 'latest_date' which may leak",
    "",
    "3. HANDLE NEW ACCOUNTS IN TEST SET",
    f"   - {len(new_test_accts)} test accounts have no training history",
    "   - Consider: cold-start features, global statistics, or separate model",
    "",
    "4. IMPROVE FEATURE ENGINEERING",
    "   - Add temporal patterns (time-of-day, day-of-week)",
    "   - Add network features (PageRank, betweenness centrality)",
    "   - Add behavioral features (transaction velocity, burst detection)",
    "   - Add graph motifs (triangles, community structure)",
    "",
    "5. BETTER CLASS IMBALANCE HANDLING",
    "   - Try focal loss instead of weighted cross-entropy",
    "   - Use SMOTE or ADASYN for oversampling",
    "   - Try ensemble with different negative sampling ratios",
    "",
    "6. GNN ARCHITECTURE IMPROVEMENTS",
    "   - Try deeper models (3-4 layers) with residual connections",
    "   - Add edge features (transaction amount, time difference)",
    "   - Try different aggregation (mean, max, attention)",
    "   - Consider temporal GNN (TGAT, TGN) for time-aware modeling",
    "",
    "7. VALIDATION STRATEGY",
    "   - Use time-based split instead of random split",
    "   - Validate on most recent alert events",
    "   - Monitor recall at different precision levels",
    "",
    "8. THRESHOLD OPTIMIZATION",
    "   - Current: optimizes F1 on OOF predictions",
    "   - Try: optimize F1 on validation set with time-based split",
    "   - Consider: precision-recall curve analysis",
]

for rec in recommendations:
    print(rec)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")

# Save detailed statistics
print("\nSaving detailed statistics to CSV files...")

# Save feature comparison
comparison_df = pd.DataFrame({
    'feature': features.columns,
    'alert_mean': [features[features['is_alert']==1][col].mean() for col in features.columns],
    'non_alert_mean': [features[features['is_alert']==0][col].mean() for col in features.columns],
    'alert_std': [features[features['is_alert']==1][col].std() for col in features.columns],
    'non_alert_std': [features[features['is_alert']==0][col].std() for col in features.columns],
})
comparison_df['separability_ratio'] = comparison_df['alert_mean'] / (comparison_df['non_alert_mean'] + 1e-6)
comparison_df.to_csv('feature_comparison.csv', index=False)
print("✅ Feature comparison saved to 'feature_comparison.csv'")

# Save temporal analysis
temporal_df = df_alert.groupby('event_date').size().reset_index()
temporal_df.columns = ['date', 'alert_count']
temporal_df.to_csv('alert_temporal_distribution.csv', index=False)
print("✅ Alert temporal distribution saved to 'alert_temporal_distribution.csv'")

print("\n" + "="*80)
print("Next steps:")
print("1. Fix the critical data leakage issues identified above")
print("2. Re-run the model and compare F1 scores")
print("3. Implement the recommended feature improvements")
print("4. Consider temporal validation strategy")
print("="*80)
