
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
FILE_PATH_TRANS = 'acct_transaction.csv'
FILE_PATH_ALERT = 'acct_alert.csv'
FILE_PATH_PREDICT = 'acct_predict.csv'
SUBMISSION_FILENAME = 'final_robust_submission.csv'

# Feature Engineering Constants
STABLE_TIME_WINDOWS = [1, 3, 7]
EPSILON = 1e-6

# Modeling Constants
NEG_POS_RATIO = 3
TIME_SPLIT_DATE = 90
RANDOM_SEED = 42
LGB_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1,
}

# --- 2. Feature Engineering Functions ---

def generate_features_for_samples(df_trans, accounts_with_ref_date):
    """
    Generates time-aware features for a given list of accounts, each with a reference date.
    """
    feature_list = []
    for acct, ref_date, label in tqdm(accounts_with_ref_date, desc="Generating features for train/val set"):
        acct_txns = df_trans[((df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)) & (df_trans['txn_date'] < ref_date)].copy()
        if acct_txns.empty: continue

        features = {'acct': acct, 'label': label, 'ref_date': ref_date}
        for window in STABLE_TIME_WINDOWS:
            window_txns = acct_txns[acct_txns['txn_date'] >= ref_date - window]
            from_txns = window_txns[window_txns['from_acct'] == acct]
            features[f'from_count_{window}d'] = len(from_txns)
            features[f'from_amt_sum_{window}d'] = from_txns['txn_amt'].sum()
            features[f'from_amt_mean_{window}d'] = from_txns['txn_amt'].mean()
            to_txns = window_txns[window_txns['to_acct'] == acct]
            features[f'to_count_{window}d'] = len(to_txns)
            features[f'to_amt_sum_{window}d'] = to_txns['txn_amt'].sum()

        features['ratio_from_amt_3d_vs_7d'] = features['from_amt_sum_3d'] / (features['from_amt_sum_7d'] + EPSILON)
        features['ratio_from_cnt_1d_vs_7d'] = features['from_count_1d'] / (features['from_count_7d'] + EPSILON)
        features['ratio_to_amt_3d_vs_7d'] = features['to_amt_sum_3d'] / (features['to_amt_sum_7d'] + EPSILON)
        features['diff_from_mean_amt_3d_vs_7d'] = features['from_amt_mean_3d'] - features['from_amt_mean_7d']
        feature_list.append(features)

    return pd.DataFrame(feature_list).fillna(0)

def generate_features_for_test_tta(df_trans, test_accts, ref_date):
    """
    Generates features for the test set based on a given reference date (for TTA).
    """
    features_df = pd.DataFrame(index=test_accts)
    features_df.index.name = 'acct'

    for window in STABLE_TIME_WINDOWS:
        window_trans = df_trans[df_trans['txn_date'] >= ref_date - window]
        from_feats = window_trans.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
        from_feats.columns = [f'from_count_{window}d', f'from_amt_sum_{window}d', f'from_amt_mean_{window}d']
        to_feats = window_trans.groupby('to_acct')['txn_amt'].agg(['count', 'sum'])
        to_feats.columns = [f'to_count_{window}d', f'to_amt_sum_{window}d']
        features_df = features_df.join(from_feats, how='left').join(to_feats, how='left')
    
    features_df.fillna(0, inplace=True)
    features_df['ratio_from_amt_3d_vs_7d'] = features_df['from_amt_sum_3d'] / (features_df['from_amt_sum_7d'] + EPSILON)
    features_df['ratio_from_cnt_1d_vs_7d'] = features_df['from_count_1d'] / (features_df['from_count_7d'] + EPSILON)
    features_df['ratio_to_amt_3d_vs_7d'] = features_df['to_amt_sum_3d'] / (features_df['to_amt_sum_7d'] + EPSILON)
    features_df['diff_from_mean_amt_3d_vs_7d'] = features_df['from_amt_mean_3d'] - features_df['from_amt_mean_7d']
    return features_df.fillna(0)

# --- 3. Main Execution Logic ---
if __name__ == "__main__":
    print("--- Loading Data ---")
    df_trans = pd.read_csv(FILE_PATH_TRANS)
    df_alert = pd.read_csv(FILE_PATH_ALERT)
    df_predict = pd.read_csv(FILE_PATH_PREDICT)
    print("Data loaded.")

    print("\n--- Creating Training & Validation Set ---")
    positive_samples = [(acct, date, 1) for acct, date in df_alert.itertuples(index=False, name=None)]
    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    alert_accts = df_alert['acct'].unique()
    non_alert_accts = np.setdiff1d(all_accts, alert_accts)
    num_negative_samples = len(positive_samples) * NEG_POS_RATIO
    np.random.seed(RANDOM_SEED)
    selected_neg_accts = np.random.choice(non_alert_accts, size=num_negative_samples, replace=False)

    negative_samples = []
    for acct in tqdm(selected_neg_accts, desc="Sampling pseudo dates"):
        acct_txn_dates = df_trans[(df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)]['txn_date'].unique()
        if len(acct_txn_dates) > 1:
            pseudo_event_date = np.random.choice(acct_txn_dates[acct_txn_dates > np.min(acct_txn_dates)])
            negative_samples.append((acct, pseudo_event_date, 0))

    full_feature_set = generate_features_for_samples(df_trans, positive_samples + negative_samples)

    print("\n--- Performing Time-Consistent Validation to Find Best Threshold ---")
    train_df = full_feature_set[full_feature_set['ref_date'] <= TIME_SPLIT_DATE]
    val_df = full_feature_set[full_feature_set['ref_date'] > TIME_SPLIT_DATE]

    if len(train_df) == 0 or len(val_df) == 0:
        print("Error: Data split resulted in an empty set.")
        exit()

    features_to_drop = ['acct', 'label', 'ref_date']
    X_train_val, y_train_val = train_df.drop(columns=features_to_drop), train_df['label']
    X_val, y_val = val_df.drop(columns=features_to_drop), val_df['label']
    X_val = X_val[X_train_val.columns]

    scale_pos_weight_val = y_train_val.value_counts().get(0, 0) / (y_train_val.value_counts().get(1, 1) + EPSILON)
    val_model = lgb.LGBMClassifier(**{**LGB_PARAMS, 'scale_pos_weight': scale_pos_weight_val})
    val_model.fit(X_train_val, y_train_val, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
    
    val_preds_proba = val_model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (val_preds_proba > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"Best F1 on Time-Split Validation: {best_f1:.4f} at threshold {best_thresh:.2f}")

    print("\n--- Training Final Model on All Data ---")
    X_full = full_feature_set.drop(columns=features_to_drop)
    y_full = full_feature_set['label']
    scale_pos_weight_full = y_full.value_counts().get(0, 0) / (y_full.value_counts().get(1, 1) + EPSILON)
    final_model = lgb.LGBMClassifier(**{**LGB_PARAMS, 'scale_pos_weight': scale_pos_weight_full, 'n_estimators': 500})
    final_model.fit(X_full, y_full)
    print("Final model training complete.")

    print("\n--- Predicting with Test Time Augmentation (TTA) ---")
    test_accts = df_predict['acct'].unique()
    tta_predictions = []
    # Predict based on features from the last day, 7 days ago, and 14 days ago
    for time_shift in [0, 7, 14]:
        ref_date = df_trans['txn_date'].max() - time_shift
        print(f"Generating TTA features for reference date: {ref_date}")
        test_features_df = generate_features_for_test_tta(df_trans, test_accts, ref_date)
        test_features_df = test_features_df[X_full.columns] # Ensure column order
        preds = final_model.predict_proba(test_features_df)[:, 1]
        tta_predictions.append(preds)

    # Average the predictions from different time points
    final_predictions_proba = np.mean(tta_predictions, axis=0)
    print("Averaged predictions from TTA.")

    # Generate submission file using the robust threshold
    final_predictions = (final_predictions_proba > best_thresh).astype(int)
    submission_df = pd.DataFrame({'acct': test_accts, 'label': final_predictions})
    submission_df.to_csv(SUBMISSION_FILENAME, index=False)

    print(f"\n--- Submission file '{SUBMISSION_FILENAME}' created. ---")
    print(f"Prediction distribution:\n{submission_df['label'].value_counts(dropna=False)}")
