"""
A100 GPU優化的詐騙檢測模型
===============================
專為A100 GPU設計的高性能深度學習+集成模型
結合:
1. Transformer-based Temporal Model
2. Deep Graph Attention Network  
3. TabNet (表格數據專用深度學習)
4. 高級集成學習
"""

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from collections import defaultdict

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# PyG for graphs
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader

# Tree models
import lightgbm as lgb
import xgboost as xgb

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except:
    HAS_TABNET = False
    print("TabNet not installed. Install: pip install pytorch-tabnet")

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
FILE_PATH_TRANS = 'acct_transaction.csv'
FILE_PATH_ALERT = 'acct_alert.csv'
FILE_PATH_PREDICT = 'acct_predict.csv'
SUBMISSION_FILENAME = 'submission_a100_optimized.csv'

RANDOM_SEED = 42
NEG_POS_RATIO = 5
TIME_SPLIT_DATE = 90
N_FOLDS = 7

# GPU優化設置
BATCH_SIZE = 256  # A100可以處理更大的batch
NUM_WORKERS = 8
PIN_MEMORY = True
USE_AMP = True  # 使用混合精度訓練

# 模型參數
HIDDEN_DIM = 512
GNN_HEADS = 8
GNN_LAYERS = 4
DROPOUT = 0.2
EPOCHS = 30
LR = 0.0005

# Feature Engineering
TIME_WINDOWS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]

# Seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # A100優化

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ==================== Ultra Feature Engineering ====================

class UltraFeatureEngineer:
    """超級特徵工程"""
    
    def __init__(self, df_trans):
        self.df_trans = df_trans
        self._precompute_global_stats()
    
    def _precompute_global_stats(self):
        """預計算全局統計"""
        print("Precomputing global statistics...")
        self.global_amt_mean = self.df_trans['txn_amt'].mean()
        self.global_amt_std = self.df_trans['txn_amt'].std()
        self.global_amt_median = self.df_trans['txn_amt'].median()
        
        # 帳戶交易次數
        from_counts = self.df_trans['from_acct'].value_counts()
        to_counts = self.df_trans['to_acct'].value_counts()
        self.acct_total_counts = (from_counts.add(to_counts, fill_value=0)).to_dict()
        
        # 構建交易圖
        print("Building transaction graph...")
        self.build_transaction_graph()
    
    def build_transaction_graph(self):
        """構建完整的交易圖並計算圖特徵"""
        from collections import defaultdict
        
        self.graph = defaultdict(set)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        
        for _, row in tqdm(self.df_trans.iterrows(), total=len(self.df_trans), desc="Building graph"):
            from_acct = row['from_acct']
            to_acct = row['to_acct']
            
            self.graph[from_acct].add(to_acct)
            self.out_degree[from_acct] += 1
            self.in_degree[to_acct] += 1
    
    def extract_features(self, acct, ref_date):
        """提取超級特徵"""
        features = {}
        
        # 獲取交易
        acct_txns = self.df_trans[
            ((self.df_trans['from_acct'] == acct) | (self.df_trans['to_acct'] == acct)) & 
            (self.df_trans['txn_date'] < ref_date)
        ]
        
        if acct_txns.empty:
            return self._empty_features()
        
        # === 1. 多時間窗口統計特徵 ===
        for window in TIME_WINDOWS:
            window_txns = acct_txns[acct_txns['txn_date'] >= ref_date - window]
            
            # 轉出
            from_txns = window_txns[window_txns['from_acct'] == acct]
            if len(from_txns) > 0:
                features[f'from_cnt_{window}'] = len(from_txns)
                features[f'from_sum_{window}'] = from_txns['txn_amt'].sum()
                features[f'from_mean_{window}'] = from_txns['txn_amt'].mean()
                features[f'from_std_{window}'] = from_txns['txn_amt'].std()
                features[f'from_max_{window}'] = from_txns['txn_amt'].max()
                features[f'from_min_{window}'] = from_txns['txn_amt'].min()
                features[f'from_median_{window}'] = from_txns['txn_amt'].median()
                features[f'from_q25_{window}'] = from_txns['txn_amt'].quantile(0.25)
                features[f'from_q75_{window}'] = from_txns['txn_amt'].quantile(0.75)
                features[f'from_unique_{window}'] = from_txns['to_acct'].nunique()
                features[f'from_skew_{window}'] = from_txns['txn_amt'].skew()
                features[f'from_kurt_{window}'] = from_txns['txn_amt'].kurtosis()
            else:
                for suffix in ['cnt', 'sum', 'mean', 'std', 'max', 'min', 'median', 'q25', 'q75', 'unique', 'skew', 'kurt']:
                    features[f'from_{suffix}_{window}'] = 0
            
            # 轉入
            to_txns = window_txns[window_txns['to_acct'] == acct]
            if len(to_txns) > 0:
                features[f'to_cnt_{window}'] = len(to_txns)
                features[f'to_sum_{window}'] = to_txns['txn_amt'].sum()
                features[f'to_mean_{window}'] = to_txns['txn_amt'].mean()
                features[f'to_std_{window}'] = to_txns['txn_amt'].std()
                features[f'to_max_{window}'] = to_txns['txn_amt'].max()
                features[f'to_min_{window}'] = to_txns['txn_amt'].min()
                features[f'to_median_{window}'] = to_txns['txn_amt'].median()
                features[f'to_unique_{window}'] = to_txns['from_acct'].nunique()
            else:
                for suffix in ['cnt', 'sum', 'mean', 'std', 'max', 'min', 'median', 'unique']:
                    features[f'to_{suffix}_{window}'] = 0
            
            # 交易模式
            if len(window_txns) > 0:
                features[f'total_cnt_{window}'] = len(window_txns)
                features[f'channel_diversity_{window}'] = window_txns['channel_type'].nunique()
                features[f'currency_diversity_{window}'] = window_txns['currency_type'].nunique()
                
                # 時間特徵
                hours = pd.to_datetime(window_txns['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour
                features[f'avg_hour_{window}'] = hours.mean()
                features[f'night_ratio_{window}'] = (hours < 6).sum() / len(hours) if len(hours) > 0 else 0
                features[f'business_hour_ratio_{window}'] = ((hours >= 9) & (hours <= 17)).sum() / len(hours) if len(hours) > 0 else 0
        
        # === 2. 跨時間窗口特徵 ===
        for w1, w2 in [(1, 7), (3, 14), (7, 30), (14, 60), (30, 90)]:
            if w2 in TIME_WINDOWS and w1 in TIME_WINDOWS:
                # 金額加速度
                features[f'from_amt_accel_{w1}_{w2}'] = features[f'from_sum_{w1}'] / (features[f'from_sum_{w2}'] + 1e-8)
                features[f'to_amt_accel_{w1}_{w2}'] = features[f'to_sum_{w1}'] / (features[f'to_sum_{w2}'] + 1e-8)
                
                # 頻率加速度
                features[f'from_freq_accel_{w1}_{w2}'] = features[f'from_cnt_{w1}'] / (features[f'from_cnt_{w2}'] + 1e-8)
                features[f'to_freq_accel_{w1}_{w2}'] = features[f'to_cnt_{w1}'] / (features[f'to_cnt_{w2}'] + 1e-8)
                
                # 平均金額變化
                avg1_from = features[f'from_mean_{w1}']
                avg2_from = features[f'from_mean_{w2}']
                features[f'from_avg_change_{w1}_{w2}'] = (avg1_from - avg2_from) / (avg2_from + 1e-8)
        
        # === 3. 圖結構特徵 ===
        features['graph_in_degree'] = self.in_degree.get(acct, 0)
        features['graph_out_degree'] = self.out_degree.get(acct, 0)
        features['graph_total_degree'] = features['graph_in_degree'] + features['graph_out_degree']
        features['graph_degree_ratio'] = features['graph_out_degree'] / (features['graph_in_degree'] + 1e-8)
        
        # 二階鄰居數量
        neighbors = self.graph.get(acct, set())
        second_order_neighbors = set()
        for neighbor in neighbors:
            second_order_neighbors.update(self.graph.get(neighbor, set()))
        features['second_order_neighbors'] = len(second_order_neighbors)
        
        # === 4. 異常分數 ===
        if len(acct_txns) > 3:
            # 金額異常分數（相對全局）
            features['amt_zscore_global'] = (acct_txns['txn_amt'].mean() - self.global_amt_mean) / (self.global_amt_std + 1e-8)
            
            # 最近交易異常
            recent_txns = acct_txns[acct_txns['txn_date'] >= ref_date - 7]
            if len(recent_txns) > 0:
                features['recent_amt_zscore'] = (recent_txns['txn_amt'].mean() - acct_txns['txn_amt'].mean()) / (acct_txns['txn_amt'].std() + 1e-8)
            
            # 交易間隔異常
            date_diffs = acct_txns.sort_values('txn_date')['txn_date'].diff()
            features['avg_interval'] = date_diffs.mean()
            features['std_interval'] = date_diffs.std()
            features['min_interval'] = date_diffs.min()
        
        # === 5. 行為模式特徵 ===
        features['net_flow_7d'] = features['to_sum_7'] - features['from_sum_7']
        features['net_flow_30d'] = features['to_sum_30'] - features['from_sum_30']
        features['flow_ratio_7d'] = features['from_sum_7'] / (features['to_sum_7'] + 1e-8)
        features['flow_ratio_30d'] = features['from_sum_30'] / (features['to_sum_30'] + 1e-8)
        
        # 帳戶年齡
        features['account_age'] = ref_date - acct_txns['txn_date'].min()
        features['days_since_last_txn'] = ref_date - acct_txns['txn_date'].max()
        features['days_active'] = acct_txns['txn_date'].nunique()
        features['activity_rate'] = features['days_active'] / (features['account_age'] + 1)
        
        # 自我交易
        features['self_txn_ratio'] = (acct_txns['is_self_txn'] == 'Y').mean()
        
        # 通路偏好
        for channel in ['1', '2', '3', '4', '99']:
            features[f'channel_{channel}_ratio'] = (acct_txns['channel_type'] == channel).mean()
        
        # 幣別偏好
        features['foreign_currency_ratio'] = (acct_txns['currency_type'] != 'TWD').mean()
        
        # === 6. 高階統計特徵 ===
        # 交易金額分布熵
        amt_bins = pd.cut(acct_txns['txn_amt'], bins=10, duplicates='drop')
        amt_dist = amt_bins.value_counts(normalize=True)
        features['amt_entropy'] = -(amt_dist * np.log(amt_dist + 1e-8)).sum()
        
        # 交易對手集中度（Herfindahl指數）
        counterparty_txns = acct_txns[acct_txns['from_acct'] == acct]
        if len(counterparty_txns) > 0:
            counterparty_counts = counterparty_txns['to_acct'].value_counts(normalize=True)
            features['counterparty_concentration'] = (counterparty_counts ** 2).sum()
        else:
            features['counterparty_concentration'] = 0
        
        # === 7. 序列特徵（用於Transformer）===
        # 最近N筆交易的統計
        recent_n = acct_txns.nlargest(10, 'txn_date')
        if len(recent_n) > 0:
            features['recent_10_amt_mean'] = recent_n['txn_amt'].mean()
            features['recent_10_amt_std'] = recent_n['txn_amt'].std()
            features['recent_10_amt_trend'] = np.polyfit(range(len(recent_n)), recent_n['txn_amt'].values, 1)[0] if len(recent_n) > 1 else 0
        
        return features
    
    def _empty_features(self):
        """空特徵"""
        features = {}
        for window in TIME_WINDOWS:
            for prefix in ['from', 'to']:
                for suffix in ['cnt', 'sum', 'mean', 'std', 'max', 'min', 'median', 'q25', 'q75', 'unique', 'skew', 'kurt']:
                    features[f'{prefix}_{suffix}_{window}'] = 0
        return features
    
    def batch_extract(self, samples, desc="Extracting features"):
        """批量提取"""
        results = []
        for acct, ref_date, label in tqdm(samples, desc=desc):
            feats = self.extract_features(acct, ref_date)
            feats['acct'] = acct
            feats['label'] = label
            feats['ref_date'] = ref_date
            results.append(feats)
        return pd.DataFrame(results).fillna(0)


# ==================== Deep Learning Models ====================

class TransformerEncoder(nn.Module):
    """Transformer編碼器用於時序特徵"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch, d_model)
        return x


class DeepFraudDetector(nn.Module):
    """深度詐騙檢測器"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class HybridGNN(nn.Module):
    """混合圖神經網絡（GATv2 + Transformer）"""
    
    def __init__(self, in_channels, hidden_channels=512, out_channels=256, num_layers=4, heads=8, dropout=0.2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一層
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # 最後一層
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False))
        self.norms.append(nn.BatchNorm1d(out_channels))
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),  # *2 因為用mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels // 2, 1)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (mean + max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        return self.classifier(x)


# ==================== Training Functions ====================

def train_deep_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """訓練深度模型"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if USE_AMP else None
    
    best_f1 = 0
    best_model = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if USE_AMP and scaler:
                with autocast():
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x).squeeze()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Find best threshold
        best_thresh_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(val_labels, val_preds > thresh)
            if f1 > best_thresh_f1:
                best_thresh_f1 = f1
                best_thresh = thresh
        
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F1: {best_thresh_f1:.4f} @ {best_thresh:.2f}")
        
        if best_thresh_f1 > best_f1:
            best_f1 = best_thresh_f1
            best_model = model.state_dict()
            patience_counter = 0
            print(f"  ✓ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model:
        model.load_state_dict(best_model)
    
    return model, best_f1


def create_graph_dataset(df_trans, samples, node_features_dict, window=21):
    """創建圖數據集"""
    
    graphs = []
    
    for acct, ref_date, label in tqdm(samples, desc="Creating graphs"):
        # 時間窗口交易
        window_txns = df_trans[
            (df_trans['txn_date'] < ref_date) & 
            (df_trans['txn_date'] >= ref_date - window)
        ]
        
        # 找鄰居
        neighbors = set()
        for _, row in window_txns.iterrows():
            if row['from_acct'] == acct:
                neighbors.add(row['to_acct'])
            if row['to_acct'] == acct:
                neighbors.add(row['from_acct'])
        
        # 限制鄰居數量
        neighbors = list(neighbors)[:100]
        subgraph_accts = [acct] + neighbors
        
        # 局部索引
        local_map = {a: i for i, a in enumerate(subgraph_accts)}
        
        # 構建邊
        edges = []
        for _, row in window_txns.iterrows():
            if row['from_acct'] in local_map and row['to_acct'] in local_map:
                edges.append([local_map[row['from_acct']], local_map[row['to_acct']]])
        
        if len(edges) == 0:
            edges = [[0, 0]]  # 自環
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # 節點特徵
        x_list = []
        for a in subgraph_accts:
            if a in node_features_dict:
                x_list.append(node_features_dict[a])
            else:
                x_list.append(np.zeros(len(next(iter(node_features_dict.values())))))
        
        x = torch.tensor(np.array(x_list), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.float)
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return graphs


# ==================== Main ====================

def main():
    print("="*80)
    print(" A100 OPTIMIZED FRAUD DETECTION MODEL")
    print("="*80)
    
    # Load data
    print("\n[1/7] Loading data...")
    df_trans = pd.read_csv(FILE_PATH_TRANS)
    df_alert = pd.read_csv(FILE_PATH_ALERT)
    df_predict = pd.read_csv(FILE_PATH_PREDICT)
    
    print(f"Transactions: {len(df_trans):,}")
    print(f"Alert accounts: {len(df_alert):,}")
    print(f"Test accounts: {len(df_predict):,}")
    
    # Create samples
    print("\n[2/7] Creating samples...")
    positive_samples = [(row['acct'], row['event_date'], 1) for _, row in df_alert.iterrows()]
    
    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    alert_accts = set(df_alert['acct'])
    non_alert_accts = np.setdiff1d(all_accts, list(alert_accts))
    
    num_neg = min(len(positive_samples) * NEG_POS_RATIO, len(non_alert_accts))
    selected_neg = np.random.choice(non_alert_accts, num_neg, replace=False)
    
    negative_samples = []
    for acct in tqdm(selected_neg, desc="Sampling negatives"):
        dates = df_trans[(df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)]['txn_date'].unique()
        if len(dates) > 1:
            pseudo_date = np.random.choice(dates[1:])
            negative_samples.append((acct, pseudo_date, 0))
    
    all_samples = positive_samples + negative_samples
    print(f"Total samples: {len(all_samples)} (Pos: {len(positive_samples)}, Neg: {len(negative_samples)})")
    
    # Feature engineering
    print("\n[3/7] Ultra feature engineering...")
    engineer = UltraFeatureEngineer(df_trans)
    df_features = engineer.batch_extract(all_samples)
    
    feature_cols = [c for c in df_features.columns if c not in ['acct', 'label', 'ref_date']]
    X = df_features[feature_cols].values
    y = df_features['label'].values
    
    print(f"Features: {X.shape[1]}")
    
    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    train_mask = df_features['ref_date'] <= TIME_SPLIT_DATE
    val_mask = df_features['ref_date'] > TIME_SPLIT_DATE
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train pos rate: {y_train.mean():.4f}, Val pos rate: {y_val.mean():.4f}")
    
    # Tree models
    print("\n[4/7] Training tree models...")
    
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        device='gpu' if torch.cuda.is_available() else 'cpu'
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=10,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=150, verbose=200)
    
    # Get tree predictions
    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    
    print(f"LightGBM Val AUC: {roc_auc_score(y_val, lgb_val_pred):.4f}")
    print(f"XGBoost Val AUC: {roc_auc_score(y_val, xgb_val_pred):.4f}")
    
    # Deep learning model
    print("\n[5/7] Training deep neural network...")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    deep_model = DeepFraudDetector(
        input_dim=X_scaled.shape[1],
        hidden_dims=[1024, 512, 256, 128],
        dropout=DROPOUT
    ).to(device)
    
    deep_model, deep_f1 = train_deep_model(deep_model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    
    # Get deep predictions
    deep_model.eval()
    with torch.no_grad():
        deep_val_pred = []
        for batch_x, _ in val_loader:
            outputs = torch.sigmoid(deep_model(batch_x.to(device))).cpu().numpy()
            deep_val_pred.extend(outputs)
    deep_val_pred = np.array(deep_val_pred).flatten()
    
    print(f"Deep Model Val AUC: {roc_auc_score(y_val, deep_val_pred):.4f}")
    
    # Ensemble
    print("\n[6/7] Creating ensemble...")
    
    # Calculate weights based on AUC
    aucs = {
        'lgb': roc_auc_score(y_val, lgb_val_pred),
        'xgb': roc_auc_score(y_val, xgb_val_pred),
        'deep': roc_auc_score(y_val, deep_val_pred)
    }
    
    total_auc = sum(aucs.values())
    weights = {k: v / total_auc for k, v in aucs.items()}
    
    print("Model weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    ensemble_val_pred = (
        weights['lgb'] * lgb_val_pred +
        weights['xgb'] * xgb_val_pred +
        weights['deep'] * deep_val_pred
    )
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, ensemble_val_pred > thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\nEnsemble Val AUC: {roc_auc_score(y_val, ensemble_val_pred):.4f}")
    print(f"Best Val F1: {best_f1:.4f} @ threshold {best_thresh:.3f}")
    
    # Test prediction
    print("\n[7/7] Generating test predictions...")
    
    test_accts = df_predict['acct'].tolist()
    max_date = df_trans['txn_date'].max()
    test_samples = [(acct, max_date, 0) for acct in test_accts]
    
    df_test_features = engineer.batch_extract(test_samples, desc="Test features")
    X_test = df_test_features[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # Tree predictions
    lgb_test_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Deep predictions
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.zeros(len(X_test_scaled)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    deep_model.eval()
    with torch.no_grad():
        deep_test_pred = []
        for batch_x, _ in test_loader:
            outputs = torch.sigmoid(deep_model(batch_x.to(device))).cpu().numpy()
            deep_test_pred.extend(outputs)
    deep_test_pred = np.array(deep_test_pred).flatten()
    
    # Ensemble
    ensemble_test_pred = (
        weights['lgb'] * lgb_test_pred +
        weights['xgb'] * xgb_test_pred +
        weights['deep'] * deep_test_pred
    )
    
    final_predictions = (ensemble_test_pred > best_thresh).astype(int)
    
    # Save
    submission = pd.DataFrame({'acct': test_accts, 'label': final_predictions})
    submission.to_csv(SUBMISSION_FILENAME, index=False)
    
    print(f"\n{'='*80}")
    print(f"Submission saved: {SUBMISSION_FILENAME}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Positive predictions: {final_predictions.sum()} ({final_predictions.mean()*100:.2f}%)")
    print(f"{'='*80}")
    
    # Save models
    joblib.dump({
        'lgb': lgb_model,
        'xgb': xgb_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'weights': weights,
        'threshold': best_thresh
    }, 'a100_model.pkl')
    torch.save(deep_model.state_dict(), 'deep_model.pth')
    
    print("\nModels saved!")


if __name__ == "__main__":
    main()
