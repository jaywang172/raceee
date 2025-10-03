
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import xgboost as xgb
import gc
import os

# --- 全局設定 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GNN_EPOCHS = 5
GNN_HIDDEN_DIM = 32
GNN_HEADS = 2
BATCH_SIZE = 1024
XGB_EARLY_STOPPING_ROUNDS = 100

# --- 階段一：特徵工程函數 ---

def load_data(base_path='.'):
    print("正在載入資料...")
    path_trans = os.path.join(base_path, 'acct_transaction.csv')
    path_alert = os.path.join(base_path, 'acct_alert.csv')
    path_predict = os.path.join(base_path, 'acct_predict.csv')
    try:
        df_trans = pd.read_csv(path_trans)
        df_alert = pd.read_csv(path_alert)
        df_predict = pd.read_csv(path_predict)
    except FileNotFoundError as e:
        print(f"錯誤：找不到必要的 CSV 檔案: {e.filename}。請確認資料檔案都在腳本的同一目錄下。")
        return None, None, None
    return df_trans, df_alert, df_predict

def create_graph_and_embeddings(df_trans, all_accts, alert_accts, cutoff_date):
    print(f"建立圖結構與節點嵌入 (截止日期: {cutoff_date})...")
    acct_encoder = LabelEncoder()
    acct_encoder.fit(all_accts)
    num_nodes = len(acct_encoder.classes_)
    train_trans = df_trans[df_trans['txn_date'] <= cutoff_date].copy()
    known_accts_mask = train_trans['from_acct'].isin(acct_encoder.classes_) & train_trans['to_acct'].isin(acct_encoder.classes_)
    train_trans = train_trans[known_accts_mask]
    from_accts_encoded = acct_encoder.transform(train_trans['from_acct'])
    to_accts_encoded = acct_encoder.transform(train_trans['to_acct'])
    edge_index_np = np.vstack([from_accts_encoded, to_accts_encoded])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    node_features = torch.zeros(num_nodes, 1)
    known_alert_accts = [acct for acct in alert_accts if acct in acct_encoder.classes_]
    alert_accts_encoded = acct_encoder.transform(known_alert_accts)
    node_features[alert_accts_encoded, 0] = 1.0
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[alert_accts_encoded] = 1
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[alert_accts_encoded] = True
    data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask).to(DEVICE)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
    )
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads):
            super().__init__()
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
            self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x
    model = GNN(data.num_node_features, GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, heads=GNN_HEADS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    print("使用鄰居採樣訓練 GNN...")
    for epoch in range(GNN_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{GNN_EPOCHS}"):
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    print("生成節點嵌入...")
    model.eval()
    with torch.no_grad():
        try:
            embeddings = model(data.x, data.edge_index).cpu().numpy()
        except torch.OutOfMemoryError:
            print("警告：全圖推理仍然記憶體不足。GNN嵌入將是零向量。")
            embeddings = np.zeros((num_nodes, GNN_HIDDEN_DIM))
    embedding_df = pd.DataFrame(embeddings, index=acct_encoder.classes_, columns=[f'gnn_{i}' for i in range(GNN_HIDDEN_DIM)])
    return embedding_df

def create_static_features(df_trans, all_accts, cutoff_date):
    print(f"為截止日期 {cutoff_date} 計算靜態特徵...")
    df_subset = df_trans[df_trans['txn_date'] <= cutoff_date].copy()
    features = pd.DataFrame(index=all_accts)
    features['total_txn_amt'] = df_subset.groupby('from_acct')['txn_amt'].sum()
    features['total_txn_count'] = df_subset.groupby('from_acct')['txn_amt'].count()
    features['total_to_amt'] = df_subset.groupby('to_acct')['txn_amt'].sum()
    features['total_to_count'] = df_subset.groupby('to_acct')['txn_amt'].count()
    features['net_flow'] = features['total_to_amt'].fillna(0) - features['total_txn_amt'].fillna(0)
    features['avg_txn_amt'] = features['total_txn_amt'] / features['total_txn_count']
    features['num_unique_to_accts'] = df_subset.groupby('from_acct')['to_acct'].nunique()
    features['num_unique_from_accts'] = df_subset.groupby('to_acct')['to_acct'].nunique()
    # last_from_txn = df_subset.groupby('from_acct')['txn_date'].max()
    # last_to_txn = df_subset.groupby('to_acct')['txn_date'].max()
    # last_txn = pd.concat([last_from_txn, last_to_txn], axis=1).max(axis=1)
    # features['days_since_last_txn'] = cutoff_date - last_txn
    return features.fillna(0)

if __name__ == "__main__":
    print(f"使用設備: {DEVICE}")
    print("\n--- 階段一：特徵工程 ---")
    df_trans, df_alert, df_predict = load_data(base_path='.')
    if df_trans is None: exit()
    train_cutoff_date = df_alert['event_date'].max()
    test_cutoff_date = df_trans['txn_date'].max()
    print(f"訓練集截止日期 (train_cutoff_date): {train_cutoff_date}")
    print(f"測試集截止日期 (test_cutoff_date): {test_cutoff_date}")
    train_accts = df_alert['acct'].unique()
    test_accts = df_predict['acct'].unique()
    all_accts = np.unique(np.concatenate([train_accts, test_accts, df_trans['from_acct'].unique(), df_trans['to_acct'].unique()]))
    labels = pd.DataFrame(index=all_accts)
    labels['label'] = np.isin(labels.index, train_accts).astype(int)
    gnn_embeddings = create_graph_and_embeddings(df_trans, all_accts, train_accts, train_cutoff_date)
    train_static_features = create_static_features(df_trans, all_accts, train_cutoff_date)
    test_static_features = create_static_features(df_trans, all_accts, test_cutoff_date)
    print("合併特徵...")
    train_full_features = train_static_features.join(gnn_embeddings, how='left').fillna(0)
    test_full_features = test_static_features.join(gnn_embeddings, how='left').fillna(0)
    train_pool_accts = train_full_features.index.difference(test_accts)
    X_train = train_full_features.loc[train_pool_accts]
    y_train = labels.loc[train_pool_accts]['label']
    X_test = test_full_features.loc[test_accts]
    del df_trans, gnn_embeddings, train_static_features, test_static_features, train_full_features, test_full_features
    gc.collect()
    print("特徵工程完成！")

    print("\n--- 階段二：XGBoost 模型訓練與預測 ---")
    X_test = X_test[X_train.columns]

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        tree_method='gpu_hist' if DEVICE.type == 'cuda' else 'hist'
    )

    print("進行 5-Fold 交叉驗證訓練...")
    NFOLDS = 5
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(X_train.shape[0])
    sub_preds = np.zeros(X_test.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_valid_fold, y_valid_fold = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        print(f"--- Fold {n_fold+1} ---")
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=500
        )
        oof_preds[valid_idx] = model.predict_proba(X_valid_fold)[:, 1]
    sub_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits

    # Save raw prediction scores for analysis
    score_df = pd.DataFrame({'acct': X_test.index, 'score': sub_preds})
    score_df.to_csv('prediction_scores.csv', index=False)
    print("\n原始預測分數已儲存至 'prediction_scores.csv'")


    print("\n在 OOF 預測上尋找最佳 F1 門檻值...")
    best_f1 = 0
    best_threshold = 0
    for threshold in tqdm(np.arange(0.1, 0.9, 0.01), desc="Threshold Search"):
        f1 = f1_score(y_train, (oof_preds > threshold).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"最佳 F1 分數 (OOF): {best_f1:.5f}")
    print(f"最佳門檻值: {best_threshold:.3f}")
    p = precision_score(y_train, (oof_preds > best_threshold).astype(int))
    r = recall_score(y_train, (oof_preds > best_threshold).astype(int))
    print(f"OOF Precision: {p:.5f}, Recall: {r:.5f}")

    print("\n生成提交檔案...")
    predictions = (sub_preds > best_threshold).astype(int)
    submission = pd.DataFrame({'acct': X_test.index, 'label': predictions})
    if submission['label'].sum() == 0:
        print("警告：模型在最佳門檻值下沒有預測出任何正樣本。將使用一個較低的門檻值(0.1)重試。")
        predictions = (sub_preds > 0.1).astype(int)
        submission['label'] = predictions

    submission.to_csv('submission.csv', index=False)

    print("\n=======================================")
    print(f"流程成功結束！")
    print(f"提交檔案 'submission.csv' 已生成於目前目錄下。")
    print(f"預測出 {submission['label'].sum()} 個警示帳戶。")
    print("=======================================")
