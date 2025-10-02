import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import f1_score

# --- Part 1: Setup & Data Loading ---
print("--- Experiment 3: Robust Anomaly Feature Engineering ---")
print("Loading data...")
try:
    df_trans = pd.read_csv('acct_transaction.csv')
    df_alert = pd.read_csv('acct_alert.csv')
    df_predict = pd.read_csv('acct_predict.csv')
except FileNotFoundError:
    print("Error: Required CSV files not found.")
    exit()

# --- Part 2: Advanced ROBUST Feature Engineering ---
print("\nGenerating a new set of robust features for ALL accounts...")

# 2.1 Generate base features ONLY from stable time windows
latest_date = df_trans['txn_date'].max()
all_accts_in_trans = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
features_df = pd.DataFrame(index=all_accts_in_trans)
features_df.index.name = 'acct'

# Only use the windows we have validated as stable
time_windows = [1, 3, 7]
for window in tqdm(time_windows, desc="Processing base time windows"):
    window_trans = df_trans[df_trans['txn_date'] >= latest_date - window]
    from_feats = window_trans.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
    from_feats.columns = [f'from_count_{window}d', f'from_amt_sum_{window}d', f'from_amt_mean_{window}d']
    to_feats = window_trans.groupby('to_acct')['txn_amt'].agg(['count', 'sum'])
    to_feats.columns = [f'to_count_{window}d', f'to_amt_sum_{window}d']
    features_df = features_df.join(from_feats, how='left').join(to_feats, how='left')
features_df.fillna(0, inplace=True)

# 2.2 Create NEW "short-term internal" anomaly features
print("Creating new robust anomaly features (ratios and diffs)...")
epsilon = 1e-6

# Ratio Features (Activity Concentration)
features_df['ratio_from_amt_3d_vs_7d'] = features_df['from_amt_sum_3d'] / (features_df['from_amt_sum_7d'] + epsilon)
features_df['ratio_from_cnt_1d_vs_7d'] = features_df['from_count_1d'] / (features_df['from_count_7d'] + epsilon)
features_df['ratio_to_amt_3d_vs_7d'] = features_df['to_amt_sum_3d'] / (features_df['to_amt_sum_7d'] + epsilon)

# Difference Features (Behavioral Pattern Change)
features_df['diff_from_mean_amt_3d_vs_7d'] = features_df['from_amt_mean_3d'] - features_df['from_amt_mean_7d']

# 2.3 Define the FINAL feature set
# It includes the original stable features PLUS our new robust anomaly features
final_feature_columns = list(features_df.columns)
print(f"\nFinal feature set contains {len(final_feature_columns)} features.")

# --- Part 3: Model Training (Using the same robust setup) ---
print("\n--- Part 3: Training Model on ENHANCED Robust Feature Set ---")
alert_accts = df_alert['acct'].unique()
np.random.seed(42)
non_alert_accts = np.setdiff1d(all_accts_in_trans, alert_accts)
selected_neg_accts = np.random.choice(non_alert_accts, size=len(alert_accts) * 10, replace=False)
train_accts = np.union1d(alert_accts, selected_neg_accts)

train_df = features_df.loc[train_accts].copy()
train_df['label'] = 0
train_df.loc[train_df.index.isin(alert_accts), 'label'] = 1

X_train = train_df[final_feature_columns]
y_train = train_df['label']

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Training with {len(train_df)} samples. Scale_pos_weight: {scale_pos_weight:.2f}")

lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'scale_pos_weight': scale_pos_weight}
model = lgb.LGBMClassifier(**lgb_params)
model.fit(X_train, y_train)
print("Final model training complete.")

# --- Part 4: Prediction and Submission ---
print("\n--- Part 4: Predicting and Generating Submission File ---")
test_accts = df_predict['acct'].unique()
test_features_df = features_df.loc[test_accts][final_feature_columns].copy()

test_predictions_proba = model.predict_proba(test_features_df)[:, 1]

submission_threshold = 0.5
print(f"Using a fixed threshold of {submission_threshold} for this submission.")
test_predictions_final = (test_predictions_proba > submission_threshold).astype(int)

submission_df = pd.DataFrame({'acct': test_accts, 'label': test_predictions_final})
submission_df.to_csv('submission_exp3_robust_anomaly.csv', index=False)

print("\n--- Submission File 'submission_exp3_robust_anomaly.csv' Generated ---")
print(f"Prediction distribution:\n{submission_df['label'].value_counts(dropna=False)}")
print("Preview of submission file:")
print(submission_df.head())