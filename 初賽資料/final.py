import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 1. Configuration (No Changes) ---
FILE_PATH_TRANS = 'acct_transaction.csv'
FILE_PATH_ALERT = 'acct_alert.csv'
FILE_PATH_PREDICT = 'acct_predict.csv'
FILE_PATH_GRAPH_FEATURES = 'graph_features.csv'
SUBMISSION_FILENAME = 'submission_with_graph_optimized_v4.csv'  # <-- New version

STABLE_TIME_WINDOWS = [1, 3, 7]
EPSILON = 1e-6
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


# --- 2. Optimized Feature Engineering Functions (Revised) ---

def precompute_cumulative_features(df_trans):
    """
    Precomputes daily transaction aggregates and their cumulative sums for each account.
    This version uses transform() to robustly keep the 'acct' key.
    """
    print("Pre-computing daily and cumulative features...")
    # Melt the dataframe
    from_trans = df_trans[['from_acct', 'txn_date', 'txn_amt']].rename(columns={'from_acct': 'acct'})
    from_trans['type'] = 'from'
    to_trans = df_trans[['to_acct', 'txn_date', 'txn_amt']].rename(columns={'to_acct': 'acct'})
    to_trans['type'] = 'to'

    all_acct_trans = pd.concat([from_trans, to_trans])

    # Get daily stats
    daily_summary = all_acct_trans.groupby(['acct', 'txn_date', 'type'])['txn_amt'].agg(
        amt_count='count', amt_sum='sum'
    ).unstack('type', fill_value=0)
    daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
    daily_summary = daily_summary.reset_index()
    daily_summary = daily_summary.sort_values(['acct', 'txn_date'])

    # --- FIX START ---
    # Use groupby().transform('cumsum') to calculate cumulative features.
    # This preserves the original DataFrame's structure, including the 'acct' column.
    cols_to_sum = ['amt_count_from', 'amt_sum_from', 'amt_count_to', 'amt_sum_to']

    # Create a new DataFrame for cumulative features to keep things clean.
    # It starts with the keys we need: 'acct' and 'txn_date'.
    cumulative_features = daily_summary[['acct', 'txn_date']].copy()

    # Calculate cumulative sums and assign them to the new DataFrame.
    # transform() returns a result with the same index as daily_summary, so direct assignment works.
    cumulative_features[cols_to_sum] = daily_summary.groupby('acct')[cols_to_sum].transform('cumsum')
    # --- FIX END ---

    # The resulting cumulative_features DataFrame now correctly contains 'acct', 'txn_date',
    # and the cumulative sums, ready for merge_asof.
    return cumulative_features


def generate_features_with_merge_asof(cumulative_features, samples_df, df_graph):
    """
    Generates features for a given set of samples using the efficient merge_asof method.
    (No changes needed in this function)
    """
    print("Generating features using merge_asof...")
    # Ensure data is sorted for merge_asof
    samples_df = samples_df.sort_values('ref_date').reset_index(drop=True)
    cumulative_features = cumulative_features.rename(columns={'txn_date': 'date'}).sort_values('date')

    # Create a base lookup table from our samples
    samples_lookup = samples_df[['acct', 'ref_date']].copy()
    samples_lookup['date_to_join'] = samples_df['ref_date'] - 1

    # Get cumulative features at T-1
    merged_features = pd.merge_asof(
        samples_lookup.sort_values('date_to_join'),
        cumulative_features,
        left_on='date_to_join',
        right_on='date',
        by='acct',  # This will now work correctly
        direction='backward'
    )

    for window in tqdm(STABLE_TIME_WINDOWS, desc="Calculating window features"):
        samples_lookup[f'date_to_join_{window}d_ago'] = samples_df['ref_date'] - 1 - window

        merged_features_ago = pd.merge_asof(
            samples_lookup.sort_values(f'date_to_join_{window}d_ago'),
            cumulative_features,
            left_on=f'date_to_join_{window}d_ago',
            right_on='date',
            by='acct',  # This will now work correctly
            direction='backward'
        )

        feature_cols_map = {
            'amt_count_from': 'from_count', 'amt_sum_from': 'from_amt_sum',
            'amt_count_to': 'to_count', 'amt_sum_to': 'to_amt_sum'
        }

        for col, new_base_name in feature_cols_map.items():
            val_now = merged_features[col].fillna(0)
            val_ago = merged_features_ago[col].fillna(0)
            samples_df[f'{new_base_name}_{window}d'] = val_now - val_ago

        samples_df[f'from_amt_mean_{window}d'] = samples_df[f'from_amt_sum_{window}d'] / (
                    samples_df[f'from_count_{window}d'] + EPSILON)

    samples_df.fillna(0, inplace=True)

    # Calculate ratio and diff features
    samples_df['ratio_from_amt_3d_vs_7d'] = samples_df['from_amt_sum_3d'] / (samples_df['from_amt_sum_7d'] + EPSILON)
    samples_df['ratio_from_cnt_1d_vs_7d'] = samples_df['from_count_1d'] / (samples_df['from_count_7d'] + EPSILON)
    samples_df['ratio_to_amt_3d_vs_7d'] = samples_df['to_amt_sum_3d'] / (samples_df['to_amt_sum_7d'] + EPSILON)
    samples_df['diff_from_mean_amt_3d_vs_7d'] = samples_df['from_amt_mean_3d'] - samples_df['from_amt_mean_7d']

    # Merge graph features
    samples_df = samples_df.merge(df_graph, on='acct', how='left')
    return samples_df.fillna(0)


# --- 3. Main Execution Logic (No Changes) ---

if __name__ == "__main__":
    # --- Load Data ---
    print("--- Loading All Data ---")
    try:
        df_trans = pd.read_csv(FILE_PATH_TRANS)
        df_alert = pd.read_csv(FILE_PATH_ALERT)
        df_predict = pd.read_csv(FILE_PATH_PREDICT)
        df_graph = pd.read_csv(FILE_PATH_GRAPH_FEATURES, index_col=0).reset_index().rename(columns={'index': 'acct'})
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all required CSV files are present.")
        exit()
    print("All data loaded successfully.")

    # --- Pre-computation Step ---
    cumulative_features = precompute_cumulative_features(df_trans)

    # --- Create Training/Validation Set ---
    print("\n--- Creating Training & Validation Set ---")
    positive_samples_df = df_alert.copy()
    positive_samples_df['label'] = 1
    positive_samples_df = positive_samples_df.rename(columns={'date': 'ref_date'})

    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    alert_accts = df_alert['acct'].unique()
    non_alert_accts = np.setdiff1d(all_accts, alert_accts)
    num_negative_samples = len(positive_samples_df) * NEG_POS_RATIO
    np.random.seed(RANDOM_SEED)
    selected_neg_accts = np.random.choice(non_alert_accts, size=num_negative_samples, replace=True)

    acct_to_map = pd.DataFrame({'acct': selected_neg_accts})
    neg_trans = df_trans.melt(
        id_vars=['txn_date'], value_vars=['from_acct', 'to_acct'], value_name='acct'
    ).drop(columns='variable')
    neg_trans_dates = acct_to_map.merge(neg_trans, on='acct', how='left').dropna()

    negative_samples_df = neg_trans_dates.groupby('acct').sample(n=1, random_state=RANDOM_SEED, replace=True)
    negative_samples_df = negative_samples_df[['acct', 'txn_date']].rename(columns={'txn_date': 'ref_date'})
    negative_samples_df['label'] = 0

    samples_to_process = pd.concat([positive_samples_df, negative_samples_df], ignore_index=True)

    full_feature_set = generate_features_with_merge_asof(cumulative_features, samples_to_process, df_graph)

    # --- Time-Split Validation ---
    print("\n--- Performing Time-Consistent Validation ---")
    train_df = full_feature_set[full_feature_set['ref_date'] <= TIME_SPLIT_DATE]
    val_df = full_feature_set[full_feature_set['ref_date'] > TIME_SPLIT_DATE]

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    if len(train_df) == 0 or len(val_df) == 0:
        print("Error: Data split resulted in an empty set.")
        exit()

    features_to_drop = ['acct', 'label', 'ref_date']
    X_train = train_df.drop(columns=features_to_drop)
    y_train = train_df['label']
    X_val = val_df.drop(columns=features_to_drop)
    y_val = val_df['label']

    train_cols = X_train.columns
    X_val = X_val[train_cols]

    scale_pos_weight_val = y_train.value_counts().get(0, 0) / (y_train.value_counts().get(1, 1) + EPSILON)
    val_model = lgb.LGBMClassifier(**{**LGB_PARAMS, 'scale_pos_weight': scale_pos_weight_val})
    val_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

    val_preds_proba = val_model.predict_proba(X_val)[:, 1]

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (val_preds_proba > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Validation AUC: {roc_auc_score(y_val, val_preds_proba):.4f}")
    print(f"Best F1-Score on Validation Set: {best_f1:.4f} at threshold {best_thresh:.2f}")

    # --- Final Model Training ---
    print("\n--- Training Final Model on All Data ---")
    X_full = full_feature_set.drop(columns=features_to_drop)[train_cols]
    y_full = full_feature_set['label']

    scale_pos_weight_full = y_full.value_counts().get(0, 0) / (y_full.value_counts().get(1, 1) + EPSILON)
    final_model = lgb.LGBMClassifier(**{**LGB_PARAMS, 'scale_pos_weight': scale_pos_weight_full, 'n_estimators': 500})
    final_model.fit(X_full, y_full)
    print("Final model training complete.")

    # --- Prediction and Submission ---
    print("\n--- Generating Submission File ---")
    test_accts_df = df_predict.copy()
    test_accts_df['ref_date'] = df_trans['txn_date'].max() + 1

    test_features_df = generate_features_with_merge_asof(cumulative_features, test_accts_df, df_graph)

    submission_accts = test_features_df['acct']
    test_features_df_aligned = test_features_df[train_cols]

    test_predictions_proba = final_model.predict_proba(test_features_df_aligned)[:, 1]
    test_predictions_final = (test_predictions_proba > best_thresh).astype(int)

    submission_df = pd.DataFrame({'acct': submission_accts, 'label': test_predictions_final})
    submission_df.to_csv(SUBMISSION_FILENAME, index=False)

    print(f"\nSubmission file '{SUBMISSION_FILENAME}' created successfully.")
    print(f"Prediction distribution:\n{submission_df['label'].value_counts(dropna=False)}")