
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)


def generate_features(df_trans, all_accts, cutoff_date):
    """Generates all features for a given transaction dataframe and cutoff date."""
    
    features_df = pd.DataFrame(index=all_accts)
    features_df.index.name = 'acct'

    # --- Time-Window Numeric Features ---
    time_windows = [1, 3, 7, 14, 30]
    for window in time_windows:
        # Use data up to the cutoff date for feature calculation
        window_trans = df_trans[df_trans['txn_date'] <= cutoff_date]
        # Further filter for the specific time window relative to the cutoff date
        window_trans = window_trans[window_trans['txn_date'] >= cutoff_date - window]
        
        from_feats = window_trans.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean', 'std'])
        from_feats.columns = [f'from_count_{window}d', f'from_amt_sum_{window}d', f'from_amt_mean_{window}d', f'from_amt_std_{window}d']
        to_feats = window_trans.groupby('to_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
        to_feats.columns = [f'to_count_{window}d', f'to_amt_sum_{window}d', f'to_amt_mean_{window}d']
        features_df = features_df.join(from_feats, how='left').join(to_feats, how='left')

    features_df.fillna(0, inplace=True)

    # --- Advanced Behavioral, Network, and Granular Features ---
    df_trans_cutoff = df_trans[df_trans['txn_date'] <= cutoff_date]
    dt_series = pd.to_datetime(df_trans_cutoff['txn_time'], format='%H:%M:%S', errors='coerce')
    df_trans_cutoff['txn_datetime_num'] = df_trans_cutoff['txn_date'] * 86400 + (dt_series - dt_series.dt.normalize()).dt.total_seconds().fillna(0)

    grouped_from = df_trans_cutoff.groupby('from_acct')
    grouped_to = df_trans_cutoff.groupby('to_acct')
    total_txns_from = grouped_from.size().reindex(features_df.index).fillna(0)
    total_txns_to = grouped_to.size().reindex(features_df.index).fillna(0)
    total_txns = total_txns_from + total_txns_to
    epsilon = 1e-6

    features_df['night_txn_ratio'] = (df_trans_cutoff[dt_series.dt.hour < 6].groupby('from_acct').size() + df_trans_cutoff[dt_series.dt.hour < 6].groupby('to_acct').size()).reindex(features_df.index).fillna(0) / (total_txns + epsilon)
    features_df['unique_from_acct_count'] = grouped_from['to_acct'].nunique().reindex(features_df.index).fillna(0)
    features_df['unique_to_acct_count'] = grouped_to['from_acct'].nunique().reindex(features_df.index).fillna(0)

    df_trans_sorted = df_trans_cutoff.sort_values(by=['from_acct', 'txn_datetime_num'])
    df_trans_sorted['interval'] = df_trans_sorted.groupby('from_acct')['txn_datetime_num'].diff()
    interval_feats = df_trans_sorted.groupby('from_acct')['interval'].agg(['mean', 'std']).reindex(features_df.index).fillna(0)
    interval_feats.columns = ['txn_interval_mean', 'txn_interval_std']
    features_df = features_df.join(interval_feats)

    df_trans_cutoff['is_weekend'] = (df_trans_cutoff['txn_date'] % 7 >= 5)
    weekend_txns = df_trans_cutoff[df_trans_cutoff['is_weekend']].groupby('from_acct').size().reindex(features_df.index).fillna(0) + \
                   df_trans_cutoff[df_trans_cutoff['is_weekend']].groupby('to_acct').size().reindex(features_df.index).fillna(0)
    features_df['weekend_txn_ratio'] = weekend_txns / (total_txns + epsilon)

    under_30k = df_trans_cutoff[(df_trans_cutoff['txn_amt'] > 29000) & (df_trans_cutoff['txn_amt'] < 30000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    under_50k = df_trans_cutoff[(df_trans_cutoff['txn_amt'] > 49000) & (df_trans_cutoff['txn_amt'] < 50000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    features_df['under_threshold_ratio'] = (under_30k + under_50k) / (total_txns_from + epsilon)

    pair_counts_out = df_trans_cutoff.groupby(['from_acct', 'to_acct']).size()
    one_time_pairs_out = pair_counts_out[pair_counts_out == 1].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    features_df['one_time_partner_out_ratio'] = one_time_pairs_out / (features_df['unique_from_acct_count'] + epsilon)

    return features_df.fillna(0)


if __name__ == "__main__":
    print("--- Final Model v2: Backtesting & Submission ---")
    print("Loading data...")
    try:
        df_trans = pd.read_csv('acct_transaction.csv')
        df_alert = pd.read_csv('acct_alert.csv')
        df_predict = pd.read_csv('acct_predict.csv')
    except FileNotFoundError:
        print("Error: Required CSV files not found.")
        exit()

    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    
    top_20_features = [
        'weekend_txn_ratio', 'txn_interval_mean', 'unique_to_acct_count', 'txn_interval_std',
        'to_amt_sum_30d', 'unique_from_acct_count', 'one_time_partner_out_ratio', 'to_amt_mean_30d',
        'to_count_30d', 'to_amt_sum_14d', 'from_amt_sum_30d', 'from_amt_mean_30d',
        'to_amt_mean_14d', 'from_count_30d', 'night_txn_ratio', 'to_count_14d',
        'from_amt_std_30d', 'to_amt_mean_7d', 'to_amt_sum_7d', 'to_count_7d'
    ]

    # --- Part 1: Backtesting for Robust Threshold Finding ---
    print("\n--- Part 1: Performing Time-Series Backtesting to find best threshold ---")
    backtest_splits = [
        {'train_end': 90, 'val_start': 91, 'val_end': 105},
        {'train_end': 105, 'val_start': 106, 'val_end': 121}
    ]
    
    oof_preds = []
    oof_labels = []

    for split in backtest_splits:
        train_end_date, val_start_date, val_end_date = split['train_end'], split['val_start'], split['val_end']
        print(f"\nBacktesting split: Training <= {train_end_date}, Validating on {val_start_date}-{val_end_date}")

        print("Generating features for this time split...")
        features = generate_features(df_trans, all_accts, train_end_date)

        # Define train set for this split
        train_alert_accts = df_alert[df_alert['event_date'] <= train_end_date]['acct'].unique()
        train_non_alert_accts = np.setdiff1d(features.index, train_alert_accts)
        num_neg_samples = len(train_alert_accts) * 10
        if len(train_non_alert_accts) < num_neg_samples: num_neg_samples = len(train_non_alert_accts)
        selected_neg_accts = np.random.choice(train_non_alert_accts, size=num_neg_samples, replace=False)
        train_accts_split = np.union1d(train_alert_accts, selected_neg_accts)
        
        train_df = features.loc[train_accts_split].copy()
        train_df['label'] = 0
        train_df.loc[train_df.index.isin(train_alert_accts), 'label'] = 1
        X_train = train_df[top_20_features]
        y_train = train_df['label']

        # Define validation set for this split
        val_alert_accts = df_alert[(df_alert['event_date'] >= val_start_date) & (df_alert['event_date'] <= val_end_date)]['acct'].unique()
        X_val = features.loc[val_alert_accts][top_20_features]
        y_val = pd.Series(np.ones(len(val_alert_accts)), index=val_alert_accts)

        # Train model
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        lgb_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 7, 'seed': 42, 'n_jobs': -1, 'verbose': -1, 'scale_pos_weight': scale_pos_weight}
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)

        # Predict and store
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds.append(preds)
        oof_labels.append(y_val.values)

    oof_preds = np.concatenate(oof_preds)
    oof_labels = np.concatenate(oof_labels)

    # Find best threshold on out-of-time predictions
    best_f1 = 0
    best_threshold = 0.5
    for threshold in tqdm(np.arange(0.01, 0.99, 0.01), desc="Finding Best Threshold"):
        f1 = f1_score(oof_labels, (oof_preds > threshold).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"\nBest F1 score on backtest: {best_f1:.5f} at threshold {best_threshold:.3f}")

    # --- Part 2: Final Model Training & Submission ---
    print("\n--- Part 2: Training final model on all data and generating submission ---")
    print("Generating features on all data...")
    final_features = generate_features(df_trans, all_accts, df_trans['txn_date'].max())

    final_alert_accts = df_alert['acct'].unique()
    final_non_alert_accts = np.setdiff1d(final_features.index, final_alert_accts)
    num_neg_samples = len(final_alert_accts) * 10
    if len(final_non_alert_accts) < num_neg_samples: num_neg_samples = len(final_non_alert_accts)
    selected_neg_accts = np.random.choice(final_non_alert_accts, size=num_neg_samples, replace=False)
    final_train_accts = np.union1d(final_alert_accts, selected_neg_accts)

    train_df_final = final_features.loc[final_train_accts].copy()
    train_df_final['label'] = 0
    train_df_final.loc[train_df_final.index.isin(final_alert_accts), 'label'] = 1
    X_train_final = train_df_final[top_20_features]
    y_train_final = train_df_final['label']

    scale_pos_weight_final = y_train_final.value_counts()[0] / y_train_final.value_counts()[1]
    lgb_params['scale_pos_weight'] = scale_pos_weight_final
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X_train_final, y_train_final)
    print("Final model training complete.")

    test_accts = df_predict['acct'].unique()
    test_features_df = final_features.reindex(np.union1d(final_features.index, test_accts)).loc[test_accts][top_20_features].copy()
    test_predictions_proba = final_model.predict_proba(test_features_df)[:, 1]
    test_predictions_final = (test_predictions_proba >= best_threshold).astype(int)

    submission_df = pd.DataFrame({'acct': test_accts, 'label': test_predictions_final})
    submission_df.to_csv('submission_v2.csv', index=False)

    print("\n--- Submission File 'submission_v2.csv' Generated ---")
    print(f"Prediction distribution:\n{submission_df['label'].value_counts(dropna=False)}")
    print("Preview of submission file:")
    print(submission_df.head())
