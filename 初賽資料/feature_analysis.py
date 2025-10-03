import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Part 1: Setup & Data Loading ---
print("--- Comprehensive Feature Importance Analysis (v4 - LEAKAGE FIXED) ---")
print("Loading data...")
try:
    df_trans = pd.read_csv('acct_transaction.csv')
    df_alert = pd.read_csv('acct_alert.csv')
except FileNotFoundError:
    print("Error: Required CSV files not found.")
    exit()

# --- Part 2: Initial Feature Engineering ---
print("\nGenerating initial feature sets...")

# 2.1 Time-Window Numeric Features
latest_date = df_trans['txn_date'].max()
all_accts_in_trans = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
features_df = pd.DataFrame(index=all_accts_in_trans)
features_df.index.name = 'acct'

time_windows = [1, 3, 7, 14, 30]
for window in tqdm(time_windows, desc="Processing numeric time windows"):
    window_trans = df_trans[df_trans['txn_date'] >= latest_date - window]
    from_feats = window_trans.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean', 'std'])
    from_feats.columns = [f'from_count_{window}d', f'from_amt_sum_{window}d', f'from_amt_mean_{window}d', f'from_amt_std_{window}d']
    to_feats = window_trans.groupby('to_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
    to_feats.columns = [f'to_count_{window}d', f'to_amt_sum_{window}d', f'to_amt_mean_{window}d']
    features_df = features_df.join(from_feats, how='left').join(to_feats, how='left')

# 2.2 Basic Behavioral & Network Features
dt_series = pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce')
df_trans['txn_datetime_num'] = df_trans['txn_date'] * 86400 + (dt_series - dt_series.dt.normalize()).dt.total_seconds().fillna(0)

grouped_from = df_trans.groupby('from_acct')
grouped_to = df_trans.groupby('to_acct')
total_txns_from = grouped_from.size().reindex(features_df.index).fillna(0)
total_txns_to = grouped_to.size().reindex(features_df.index).fillna(0)
total_txns = total_txns_from + total_txns_to
epsilon = 1e-6

features_df['night_txn_ratio'] = (df_trans[pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour < 6].groupby('from_acct').size() + df_trans[pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour < 6].groupby('to_acct').size()).reindex(features_df.index).fillna(0) / (total_txns + epsilon)
features_df['unique_from_acct_count'] = grouped_from['to_acct'].nunique().reindex(features_df.index).fillna(0)
features_df['unique_to_acct_count'] = grouped_to['from_acct'].nunique().reindex(features_df.index).fillna(0)

features_df.fillna(0, inplace=True)

# --- Part 3: Advanced & Computationally Intensive Features ---
print("\nGenerating advanced (slower) features...")

# 3.1 Transaction Interval Features
print("Calculating transaction intervals (this may take a while)...")
df_trans_sorted = df_trans.sort_values(by=['from_acct', 'txn_datetime_num'])
df_trans_sorted['interval'] = df_trans_sorted.groupby('from_acct')['txn_datetime_num'].diff()
interval_feats = df_trans_sorted.groupby('from_acct')['interval'].agg(['mean', 'std']).reindex(features_df.index).fillna(0)
interval_feats.columns = ['txn_interval_mean', 'txn_interval_std']
features_df = features_df.join(interval_feats)

# 3.2 Weekend Transaction Ratio
print("Calculating weekend transaction ratios...")
df_trans['is_weekend'] = (df_trans['txn_date'] % 7 >= 5) # Assuming day 5 and 6 are weekend
weekend_txns = df_trans[df_trans['is_weekend']].groupby('from_acct').size().reindex(features_df.index).fillna(0) + \
               df_trans[df_trans['is_weekend']].groupby('to_acct').size().reindex(features_df.index).fillna(0)
features_df['weekend_txn_ratio'] = weekend_txns / (total_txns + epsilon)

# 3.3 Under Threshold Transaction Ratio
print("Calculating under-threshold transaction ratios...")
under_30k = df_trans[(df_trans['txn_amt'] > 29000) & (df_trans['txn_amt'] < 30000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
under_50k = df_trans[(df_trans['txn_amt'] > 49000) & (df_trans['txn_amt'] < 50000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
features_df['under_threshold_ratio'] = (under_30k + under_50k) / (total_txns_from + epsilon)

# 3.4 One-Time Partner Ratio
print("Calculating one-time partner ratios...")
pair_counts_out = df_trans.groupby(['from_acct', 'to_acct']).size()
one_time_pairs_out = pair_counts_out[pair_counts_out == 1].groupby('from_acct').size().reindex(features_df.index).fillna(0)
features_df['one_time_partner_out_ratio'] = one_time_pairs_out / (features_df['unique_from_acct_count'] + epsilon)

features_df.fillna(0, inplace=True)

# --- Part 4: Model Training for Feature Analysis ---
print("\n--- Part 4: Training Model for Feature Importance Analysis ---")
final_feature_columns = list(features_df.columns)
print(f"Final feature set contains {len(final_feature_columns)} features.")

alert_accts = df_alert['acct'].unique()
np.random.seed(42)
non_alert_accts = np.setdiff1d(features_df.index, alert_accts)

num_neg_samples = len(alert_accts) * 10
if len(non_alert_accts) < num_neg_samples:
    num_neg_samples = len(non_alert_accts)
    
selected_neg_accts = np.random.choice(non_alert_accts, size=num_neg_samples, replace=False)
train_accts = np.union1d(alert_accts, selected_neg_accts)

train_df = features_df.loc[train_accts].copy()
train_df['label'] = 0
train_df.loc[train_df.index.isin(alert_accts), 'label'] = 1

X_train = train_df[final_feature_columns]
y_train = train_df['label']

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Training with {len(train_df)} samples. Scale_pos_weight: {scale_pos_weight:.2f}")

lgb_params = {
    'objective': 'binary', 'boosting_type': 'gbdt', 'n_estimators': 1000,
    'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight
}
model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Part 5: Feature Importance Analysis ---
print("\n--- Feature Importance Analysis ---")
importance_df = pd.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

pd.set_option('display.max_rows', 100)
print("Feature importances:")
print(importance_df)
