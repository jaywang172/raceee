
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import warnings
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings("ignore")

# --- Part 1: Feature Engineering Function ---
def generate_features(df_trans):
    print("\nGenerating comprehensive feature set...")
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

    dt_series = pd.to_datetime(df_trans['txn_time'], format='%H:%M:%S', errors='coerce')
    df_trans['txn_datetime_num'] = df_trans['txn_date'] * 86400 + (dt_series - dt_series.dt.normalize()).dt.total_seconds().fillna(0)

    grouped_from = df_trans.groupby('from_acct')
    grouped_to = df_trans.groupby('to_acct')
    total_txns_from = grouped_from.size().reindex(features_df.index).fillna(0)
    total_txns_to = grouped_to.size().reindex(features_df.index).fillna(0)
    total_txns = total_txns_from + total_txns_to
    epsilon = 1e-6

    features_df['night_txn_ratio'] = (df_trans[dt_series.dt.hour < 6].groupby('from_acct').size() + df_trans[dt_series.dt.hour < 6].groupby('to_acct').size()).reindex(features_df.index).fillna(0) / (total_txns + epsilon)
    features_df['unique_from_acct_count'] = grouped_from['to_acct'].nunique().reindex(features_df.index).fillna(0)
    features_df['unique_to_acct_count'] = grouped_to['from_acct'].nunique().reindex(features_df.index).fillna(0)

    df_trans_sorted = df_trans.sort_values(by=['from_acct', 'txn_datetime_num'])
    df_trans_sorted['interval'] = df_trans_sorted.groupby('from_acct')['txn_datetime_num'].diff()
    interval_feats = df_trans_sorted.groupby('from_acct')['interval'].agg(['mean', 'std']).reindex(features_df.index).fillna(0)
    interval_feats.columns = ['txn_interval_mean', 'txn_interval_std']
    features_df = features_df.join(interval_feats)

    df_trans['is_weekend'] = (df_trans['txn_date'] % 7 >= 5)
    weekend_txns = df_trans[df_trans['is_weekend']].groupby('from_acct').size().reindex(features_df.index).fillna(0) + \
                   df_trans[df_trans['is_weekend']].groupby('to_acct').size().reindex(features_df.index).fillna(0)
    features_df['weekend_txn_ratio'] = weekend_txns / (total_txns + epsilon)

    under_30k = df_trans[(df_trans['txn_amt'] > 29000) & (df_trans['txn_amt'] < 30000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    under_50k = df_trans[(df_trans['txn_amt'] > 49000) & (df_trans['txn_amt'] < 50000)].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    features_df['under_threshold_ratio'] = (under_30k + under_50k) / (total_txns_from + epsilon)

    pair_counts_out = df_trans.groupby(['from_acct', 'to_acct']).size()
    one_time_pairs_out = pair_counts_out[pair_counts_out == 1].groupby('from_acct').size().reindex(features_df.index).fillna(0)
    features_df['one_time_partner_out_ratio'] = one_time_pairs_out / (features_df['unique_from_acct_count'] + epsilon)

    return features_df.fillna(0)

if __name__ == "__main__":
    print("--- Dual Model Hyperparameter Tuning ---")
    print("Loading data...")
    try:
        df_trans = pd.read_csv('acct_transaction.csv')
        df_alert = pd.read_csv('acct_alert.csv')
    except FileNotFoundError:
        print("Error: Required CSV files not found.")
        exit()

    # --- Part 2: Data Preparation ---
    features = generate_features(df_trans)
    
    top_20_features = [
        'weekend_txn_ratio', 'txn_interval_mean', 'unique_to_acct_count', 'txn_interval_std',
        'to_amt_sum_30d', 'unique_from_acct_count', 'one_time_partner_out_ratio', 'to_amt_mean_30d',
        'to_count_30d', 'to_amt_sum_14d', 'from_amt_sum_30d', 'from_amt_mean_30d',
        'to_amt_mean_14d', 'from_count_30d', 'night_txn_ratio', 'to_count_14d',
        'from_amt_std_30d', 'to_amt_mean_7d', 'to_amt_sum_7d', 'to_count_7d'
    ]
    
    # Create a dataframe sorted by date for time-series split
    alert_dates = df_alert.set_index('acct')['event_date']
    train_df = features.join(alert_dates).sort_values('event_date').copy()
    train_df['label'] = 1 # All accounts in this df are alerts
    
    # We need to add negative samples for training, but the split is on alerts
    # The grid search will be done on a dataset of alerts only, which is not ideal.
    # A proper implementation would require creating training data (features + labels) for each time split.
    # This script will be simplified to run on the whole training data for now.
    
    print("\nPreparing full training data for grid search...")
    alert_accts = df_alert['acct'].unique()
    non_alert_accts = np.setdiff1d(features.index, alert_accts)
    np.random.seed(42)
    selected_neg_accts = np.random.choice(non_alert_accts, size=len(alert_accts) * 10, replace=False)
    train_accts = np.union1d(alert_accts, selected_neg_accts)
    
    train_full_df = features.loc[train_accts].copy()
    train_full_df['label'] = 0
    train_full_df.loc[train_full_df.index.isin(alert_accts), 'label'] = 1
    
    # For a true time-series split, we need a time column in the training data.
    # We will use the alert date for alerts, and sample dates for non-alerts.
    alert_dates_df = df_alert.set_index('acct')['event_date']
    train_full_df = train_full_df.join(alert_dates_df)
    # For non-alerts, sample a random date from the alert date distribution
    nan_dates = train_full_df['event_date'].isna()
    train_full_df.loc[nan_dates, 'event_date'] = np.random.choice(df_alert['event_date'], size=nan_dates.sum())
    train_full_df = train_full_df.sort_values('event_date')

    X = train_full_df[top_20_features]
    y = train_full_df['label']

    # --- Part 3: Hyperparameter Search ---
    time_splitter = TimeSeriesSplit(n_splits=3)
    f1_scorer = make_scorer(f1_score)
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    # --- LightGBM Search ---
    print("\n--- Starting Hyperparameter Search for LightGBM ---")
    print("This will be VERY slow...")
    lgbm = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    param_grid_lgbm = {
        'learning_rate': [0.02, 0.05],
        'n_estimators': [500, 1000],
        'num_leaves': [20, 31, 40],
        'max_depth': [5, 7]
    }
    grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=param_grid_lgbm, cv=time_splitter, scoring=f1_scorer, n_jobs=-1, verbose=2)
    grid_search_lgbm.fit(X, y)

    print("\n--- LightGBM Results ---")
    print(f"Best F1 Score (Time-Series CV): {grid_search_lgbm.best_score_:.5f}")
    print("Best Parameters:")
    print(grid_search_lgbm.best_params_)

    # --- XGBoost Search ---
    print("\n--- Starting Hyperparameter Search for XGBoost ---")
    print("This will also be VERY slow...")
    xgb_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss', use_label_encoder=False)
    param_grid_xgb = {
        'learning_rate': [0.02, 0.05],
        'n_estimators': [500, 1000],
        'max_depth': [5, 7],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=time_splitter, scoring=f1_scorer, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X, y)

    print("\n--- XGBoost Results ---")
    print(f"Best F1 Score (Time-Series CV): {grid_search_xgb.best_score_:.5f}")
    print("Best Parameters:")
    print(grid_search_xgb.best_params_)

    print("\n--- Tuning Complete ---")
