# ultimate_gnn_model.py
# ==============================================================================
#
# # 運行前請務必確認環境，可參考以下指令 (根據您的CUDA版本調整):
# # pip install pandas numpy scikit-learn tqdm
# # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# # pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
# # pip install torch_geometric
#
# ==============================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
# File Paths
FILE_PATH_TRANS = 'acct_transaction.csv'
FILE_PATH_ALERT = 'acct_alert.csv'
FILE_PATH_PREDICT = 'acct_predict.csv'
PROB_OUTPUT_FILENAME = 'gnn_model_probs.csv' # Changed for ensembling

# Feature Engineering Constants
STATIC_FEATURE_WINDOW = 7
DYNAMIC_GRAPH_WINDOW = 7
EPSILON = 1e-8

# Modeling Constants
NEG_POS_RATIO = 3
TIME_SPLIT_DATE = 90
RANDOM_SEED = 42

# GNN & Training Hyperparameters
HIDDEN_CHANNELS = 128
GNN_OUT_CHANNELS = 64
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 2. Node Feature Pre-computation ---
def precompute_node_features(df_trans, all_accts):
    print("\n--- Pre-computing Base Node Features ---")
    latest_date = df_trans['txn_date'].max()
    
    features_df = pd.DataFrame(index=all_accts)
    features_df.index.name = 'acct'

    window_trans = df_trans[df_trans['txn_date'] >= latest_date - STATIC_FEATURE_WINDOW]
    
    from_feats = window_trans.groupby('from_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
    from_feats.columns = [f'from_count_{STATIC_FEATURE_WINDOW}d', f'from_amt_sum_{STATIC_FEATURE_WINDOW}d', f'from_amt_mean_{STATIC_FEATURE_WINDOW}d']
    
    to_feats = window_trans.groupby('to_acct')['txn_amt'].agg(['count', 'sum', 'mean'])
    to_feats.columns = [f'to_count_{STATIC_FEATURE_WINDOW}d', f'to_amt_sum_{STATIC_FEATURE_WINDOW}d', f'to_amt_mean_{STATIC_FEATURE_WINDOW}d']
    
    features_df = features_df.join(from_feats, how='left').join(to_feats, how='left').fillna(0)
    
    for col in features_df.columns:
        features_df[col] = (features_df[col] - features_df[col].mean()) / (features_df[col].std() + EPSILON)
        
    print(f"Pre-computed {features_df.shape[1]} features for {len(features_df)} nodes.")
    return features_df.fillna(0)

# --- 3. PyTorch Geometric Dataset Class ---
class FraudGraphDataset(Dataset):
    def __init__(self, samples, trans_df, node_features_df, acct_to_idx):
        super().__init__()
        self.samples = samples
        self.trans_df = trans_df
        self.node_features_df = node_features_df
        self.acct_to_idx = acct_to_idx
        self.idx_to_acct = {i: a for a, i in acct_to_idx.items()}

    def len(self):
        return len(self.samples)

    def get(self, idx):
        acct, ref_date, label = self.samples[idx]
        target_node_idx = self.acct_to_idx.get(acct, -1)

        sub_trans = self.trans_df[(self.trans_df['txn_date'] < ref_date) & 
                                  (self.trans_df['txn_date'] >= ref_date - DYNAMIC_GRAPH_WINDOW)]

        if sub_trans.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            subgraph_nodes = [target_node_idx]
            subgraph_map = {target_node_idx: 0}
        else:
            subgraph_accts = np.union1d(sub_trans['from_acct'].unique(), sub_trans['to_acct'].unique())
            subgraph_nodes = [self.acct_to_idx[a] for a in subgraph_accts if a in self.acct_to_idx]
            subgraph_map = {node_id: i for i, node_id in enumerate(subgraph_nodes)}
            
            from_local = [subgraph_map[self.acct_to_idx[a]] for a in sub_trans['from_acct'] if a in self.acct_to_idx and self.acct_to_idx[a] in subgraph_map]
            to_local = [subgraph_map[self.acct_to_idx[a]] for a in sub_trans['to_acct'] if a in self.acct_to_idx and self.acct_to_idx[a] in subgraph_map]
            
            edge_index = torch.tensor([from_local, to_local], dtype=torch.long)

        x = torch.tensor(self.node_features_df.iloc[subgraph_nodes].values, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.float)
        local_target_node_idx = subgraph_map.get(target_node_idx, -1)

        return Data(x=x, edge_index=edge_index, y=y, local_target_node_idx=local_target_node_idx)

# --- 4. GNN Model Definition ---
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, gnn_out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, gnn_out_channels)
        
        self.classifier = Sequential(
            Linear(gnn_out_channels, gnn_out_channels // 2),
            ReLU(),
            Linear(gnn_out_channels // 2, 1)
        )

    def forward(self, x, edge_index, local_target_node_idx, batch_vector):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        target_node_embeddings = x[local_target_node_idx + (batch_vector.bincount().cumsum(0) - batch_vector.bincount())]
        
        return self.classifier(target_node_embeddings)

# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"--- Using device: {device} ---")

    print("--- Loading Data ---")
    df_trans = pd.read_csv(FILE_PATH_TRANS)
    df_alert = pd.read_csv(FILE_PATH_ALERT)
    df_predict = pd.read_csv(FILE_PATH_PREDICT)

    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    acct_to_idx = {acct: i for i, acct in enumerate(all_accts)}
    
    node_features_df = precompute_node_features(df_trans, all_accts)

    print("\n--- Creating Training & Validation Samples ---")
    positive_samples = [(acct, date, 1) for acct, date in df_alert.itertuples(index=False, name=None)]
    
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

    all_samples = positive_samples + negative_samples
    
    train_samples = [s for s in all_samples if s[1] <= TIME_SPLIT_DATE]
    val_samples = [s for s in all_samples if s[1] > TIME_SPLIT_DATE]

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    train_dataset = FraudGraphDataset(train_samples, df_trans, node_features_df, acct_to_idx)
    val_dataset = FraudGraphDataset(val_samples, df_trans, node_features_df, acct_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n--- Starting Model Training ---")
    model = GNNModel(
        in_channels=node_features_df.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        gnn_out_channels=GNN_OUT_CHANNELS
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.local_target_node_idx, batch.batch)
            loss = loss_fn(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.local_target_node_idx, batch.batch)
                all_preds.append(out.squeeze().sigmoid().cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
        
        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)
        
        current_best_f1, _ = 0, 0
        for thresh in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(val_labels, (val_preds > thresh).astype(int))
            if f1 > current_best_f1:
                current_best_f1 = f1

        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Val F1: {current_best_f1:.4f}")

        if current_best_f1 > best_f1:
            best_f1 = current_best_f1
            best_model_state = model.state_dict()
            print(f"*** New best validation F1: {best_f1:.4f} ***")

    print("\n--- Generating Final Predictions ---")
    if best_model_state is None:
        print("Warning: Model did not improve. Using last model state for prediction.")
        best_model_state = model.state_dict()
        
    model.load_state_dict(best_model_state)
    model.eval()

    test_accts = df_predict['acct'].unique()
    max_date = df_trans['txn_date'].max()
    test_samples = [(acct, max_date, 0) for acct in test_accts]

    test_dataset = FraudGraphDataset(test_samples, df_trans, node_features_df, acct_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting on Test Set"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.local_target_node_idx, batch.batch)
            test_preds.append(out.squeeze().sigmoid().cpu().numpy())
            
    final_predictions_proba = np.concatenate(test_preds)

    prob_df = pd.DataFrame({'acct': test_accts, 'probability': final_predictions_proba})
    prob_df.to_csv(PROB_OUTPUT_FILENAME, index=False)

    print(f"--- Probability file '{PROB_OUTPUT_FILENAME}' created. ---")
