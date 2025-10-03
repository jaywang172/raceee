import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Part 1: Setup & Data Loading ---
print("--- Final Model Submission Script (v3 Features) ---")
print("Loading data...")
try:
    df_trans = pd.read_csv('acct_transaction.csv')
    df_alert = pd.read_csv('acct_alert.csv')
    df_predict = pd.read_csv('acct_predict.csv')
except FileNotFoundError:
    print("Error: Required CSV files not found.")
    exit()

# --- Part 2: Comprehensive Feature Engineering ---
print("\nGenerating comprehensive feature set...")

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
features_df.fillna(0, inplace=True)

epsilon = 1e-6
features_df['ratio_from_amt_3d_vs_30d'] = features_df['from_amt_sum_3d'] / (features_df['from_amt_sum_30d'] + epsilon)
features_df['ratio_from_cnt_1d_vs_7d'] = features_df['from_count_1d'] / (features_df['from_count_7d'] + epsilon)
features_df['diff_from_mean_amt_3d_vs_30d'] = features_df['from_amt_mean_3d'] - features_df['from_amt_mean_30d']

# 2.2 Behavioral & Network Features
dt_series = pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce')
df_trans['txn_time_num'] = (dt_series - dt_series.dt.normalize()).dt.total_seconds().fillna(0)

grouped_from = df_trans.groupby('from_acct')
grouped_to = df_trans.groupby('to_acct')

total_txns_from = grouped_from.size().reindex(features_df.index).fillna(0)
total_txns_to = grouped_to.size().reindex(features_df.index).fillna(0)
total_txns = total_txns_from + total_txns_to

features_df['night_txn_ratio'] = (df_trans[df_trans['txn_time_num'] < 6 * 3600].groupby('from_acct').size() + df_trans[df_trans['txn_time_num'] < 6 * 3600].groupby('to_acct').size()).reindex(features_df.index).fillna(0) / (total_txns + epsilon)
features_df['self_txn_ratio'] = (df_trans[df_trans['is_self_txn'] == 'Y'].groupby('from_acct').size() + df_trans[df_trans['is_self_txn'] == 'Y'].groupby('to_acct').size()).reindex(features_df.index).fillna(0) / (total_txns + epsilon)

channel_dummies = pd.get_dummies(df_trans['channel_type'], prefix='channel')
channel_total = pd.concat([df_trans[['from_acct', 'to_acct']], channel_dummies], axis=1).groupby('from_acct')[channel_dummies.columns].sum() + pd.concat([df_trans[['from_acct', 'to_acct']], channel_dummies], axis=1).groupby('to_acct')[channel_dummies.columns].sum()
for col in channel_dummies.columns:
    features_df[f'{col}_ratio'] = channel_total[col].reindex(features_df.index).fillna(0) / (total_txns + epsilon)

features_df['unique_from_acct_count'] = grouped_from['to_acct'].nunique().reindex(features_df.index).fillna(0)
features_df['unique_to_acct_count'] = grouped_to['from_acct'].nunique().reindex(features_df.index).fillna(0)

from_max_mean = grouped_from['txn_amt'].agg(['max', 'mean'])
from_max_mean.columns = ['from_max_amt', 'from_mean_amt']
features_df = features_df.join(from_max_mean, how='left')
features_df['from_max_vs_mean_ratio'] = features_df['from_max_amt'] / (features_df['from_mean_amt'] + epsilon)

first_txn_from = grouped_from['txn_date'].min().reindex(features_df.index)
first_txn_to = grouped_to['txn_date'].min().reindex(features_df.index)
features_df['first_txn_date'] = pd.concat([first_txn_from, first_txn_to], axis=1).min(axis=1)
features_df['days_since_first_txn'] = latest_date - features_df['first_txn_date']

round_num_from = df_trans[df_trans['txn_amt'] % 1000 == 0].groupby('from_acct').size().reindex(features_df.index).fillna(0)
round_num_to = df_trans[df_trans['txn_amt'] % 1000 == 0].groupby('to_acct').size().reindex(features_df.index).fillna(0)
features_df['round_number_txn_ratio'] = (round_num_from + round_num_to) / (total_txns + epsilon)

features_df.fillna(0, inplace=True)

# --- Part 3: Model Training ---
print("\n--- Part 3: Training Final Model ---")
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
print("Final model training complete.")

# --- Part 4: Prediction and Submission ---
print("\n--- Part 4: Predicting and Generating Submission File ---")
test_accts = df_predict['acct'].unique()
# Ensure all test accounts have a feature row, even if all zeros
test_features_df = features_df.reindex(np.union1d(features_df.index, test_accts)).loc[test_accts][final_feature_columns].copy()

test_predictions_proba = model.predict_proba(test_features_df)[:, 1]

# Dynamic thresholding: select top N percent riskiest accounts
num_predictions_target = int(len(test_accts) * 0.02) # Target top 2%
if num_predictions_target > 0:
    submission_threshold = np.sort(test_predictions_proba)[-num_predictions_target]
    print(f"Using a dynamic threshold of {submission_threshold:.4f} to select the top {num_predictions_target} accounts.")
else:
    submission_threshold = 0.5 # Fallback
    print(f"Using a fixed fallback threshold of {submission_threshold}.")

test_predictions_final = (test_predictions_proba >= submission_threshold).astype(int)

submission_df = pd.DataFrame({'acct': test_accts, 'label': test_predictions_final})
submission_df.to_csv('submission_final.csv', index=False)

print("\n--- Submission File 'submission_final.csv' Generated ---")
print(f"Prediction distribution:\n{submission_df['label'].value_counts(dropna=False)}")
print("Preview of submission file:")
print(submission_df.head())
