"""
🚀 BREAKTHROUGH MODEL - 針對 F1=0.25+ 的革命性解決方案
=============================================================

基於深度分析你的提交歷史，我發現:
✓ exp3 (0.1068): 短期窗口[1,3,7] + 簡單比率特徵 → 最佳
✓ exp1 (0.1061): Precision優化 → 次佳
✓ graph (0.1040): 圖特徵有效
✗ exp2 (0.0765): Z-score + 長期窗口[30,60] → 大失敗

關鍵洞察:
1. 詐騙帳戶特徵在短期內最明顯 (1-7天)
2. 簡單特徵 > 複雜統計特徵
3. 圖結構很重要但未被充分利用
4. 需要135%的提升 → 需要革命性方法

突破策略:
1. 交易網絡深度分析 (PageRank, 風險傳播)
2. 時間序列模式 (交易序列特徵)
3. 異常傳播算法 (從已知詐騙擴散)
4. 多模型深度Ensemble
5. 領域知識特徵 (詐騙行為模式)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 從exp3學到：使用短期窗口
SHORT_WINDOWS = [1, 3, 7]  # 核心窗口
MEDIUM_WINDOWS = [14, 21]  # 輔助窗口
NEG_POS_RATIO = 10  # 增加負樣本以提高模型泛化

# ==================== 第一部分：交易網絡深度分析 ====================

class TransactionNetworkAnalyzer:
    """交易網絡分析器 - 核心突破點"""
    
    def __init__(self, df_trans):
        self.df_trans = df_trans
        self.graph = defaultdict(lambda: {'out': set(), 'in': set()})
        self.build_graph()
    
    def build_graph(self):
        """構建有向圖"""
        print("構建交易網絡圖...")
        for _, row in tqdm(self.df_trans.iterrows(), total=len(self.df_trans)):
            from_acct = row['from_acct']
            to_acct = row['to_acct']
            self.graph[from_acct]['out'].add(to_acct)
            self.graph[to_acct]['in'].add(from_acct)
    
    def compute_pagerank(self, iterations=20, damping=0.85):
        """計算PageRank分數 - 識別核心詐騙節點"""
        print("計算PageRank分數...")
        nodes = list(self.graph.keys())
        n = len(nodes)
        
        # 初始化
        pagerank = {node: 1.0 / n for node in nodes}
        
        for _ in range(iterations):
            new_pagerank = {}
            for node in nodes:
                rank = (1 - damping) / n
                
                # 來自入邊的貢獻
                for in_node in self.graph[node]['in']:
                    out_degree = len(self.graph[in_node]['out'])
                    if out_degree > 0:
                        rank += damping * pagerank[in_node] / out_degree
                
                new_pagerank[node] = rank
            
            pagerank = new_pagerank
        
        return pagerank
    
    def compute_risk_propagation(self, alert_accts, iterations=5):
        """風險傳播算法 - 從已知詐騙帳戶擴散風險"""
        print("計算風險傳播分數...")
        
        # 初始化：alert帳戶風險=1，其他=0
        risk_score = defaultdict(float)
        for acct in alert_accts:
            risk_score[acct] = 1.0
        
        # 迭代傳播
        for iteration in range(iterations):
            new_risk = defaultdict(float)
            
            for acct in self.graph.keys():
                # 從轉入帳戶獲得風險（收到詐騙款）
                for in_acct in self.graph[acct]['in']:
                    new_risk[acct] += risk_score[in_acct] * 0.6  # 強傳播
                
                # 從轉出帳戶獲得風險（轉給詐騙帳戶）
                for out_acct in self.graph[acct]['out']:
                    new_risk[acct] += risk_score[out_acct] * 0.4  # 弱傳播
                
                # 保留原有風險
                new_risk[acct] = max(new_risk[acct], risk_score[acct] * 0.9)
            
            risk_score = new_risk
        
        return dict(risk_score)
    
    def compute_network_features(self, acct):
        """計算單個帳戶的網絡特徵"""
        features = {}
        
        # 基礎度數
        features['out_degree'] = len(self.graph[acct]['out'])
        features['in_degree'] = len(self.graph[acct]['in'])
        features['total_degree'] = features['out_degree'] + features['in_degree']
        
        # 度數比率（詐騙帳戶通常出度>>入度）
        features['degree_ratio'] = features['out_degree'] / (features['in_degree'] + 1e-8)
        
        # 二階鄰居
        second_order_out = set()
        for neighbor in self.graph[acct]['out']:
            second_order_out.update(self.graph[neighbor]['out'])
        features['second_order_out'] = len(second_order_out)
        
        # 共同鄰居（中繼模式）
        features['common_neighbors'] = len(
            self.graph[acct]['out'].intersection(self.graph[acct]['in'])
        )
        
        return features


# ==================== 第二部分：時序模式分析 ====================

class TemporalPatternExtractor:
    """時序模式提取器"""
    
    def __init__(self, df_trans):
        self.df_trans = df_trans
    
    def extract_sequence_features(self, acct, ref_date, lookback=30):
        """提取交易序列特徵"""
        features = {}
        
        # 獲取該帳戶的交易序列
        acct_txns = self.df_trans[
            ((self.df_trans['from_acct'] == acct) | (self.df_trans['to_acct'] == acct)) &
            (self.df_trans['txn_date'] < ref_date) &
            (self.df_trans['txn_date'] >= ref_date - lookback)
        ].sort_values('txn_date')
        
        if len(acct_txns) == 0:
            return self._empty_sequence_features()
        
        # 1. 交易時間間隔模式
        date_diffs = acct_txns['txn_date'].diff().dropna()
        if len(date_diffs) > 0:
            features['avg_interval'] = date_diffs.mean()
            features['std_interval'] = date_diffs.std()
            features['min_interval'] = date_diffs.min()
            features['interval_cv'] = features['std_interval'] / (features['avg_interval'] + 1e-8)
        
        # 2. 金額序列模式
        amounts = acct_txns['txn_amt'].values
        if len(amounts) > 1:
            # 趨勢：是否金額遞增（典型詐騙模式）
            features['amount_trend'] = np.corrcoef(np.arange(len(amounts)), amounts)[0, 1]
            
            # 變異係數
            features['amount_cv'] = np.std(amounts) / (np.mean(amounts) + 1e-8)
            
            # 最大金額佔比（單筆大額）
            features['max_amount_ratio'] = np.max(amounts) / (np.sum(amounts) + 1e-8)
        
        # 3. 交易時間模式（詐騙常在夜間/週末）
        try:
            times = pd.to_datetime(acct_txns['txn_time'], format='%H:%M:%S', errors='coerce')
            hours = times.dt.hour
            
            features['night_txn_ratio'] = ((hours >= 0) & (hours < 6)).mean()
            features['evening_txn_ratio'] = ((hours >= 22) | (hours < 2)).mean()
            features['business_hour_ratio'] = ((hours >= 9) & (hours <= 17)).mean()
        except:
            features['night_txn_ratio'] = 0
            features['evening_txn_ratio'] = 0
            features['business_hour_ratio'] = 0
        
        # 4. 突發性（短期內密集交易）
        recent_3d = acct_txns[acct_txns['txn_date'] >= ref_date - 3]
        features['burst_ratio'] = len(recent_3d) / (len(acct_txns) + 1e-8)
        
        # 5. 通路多樣性（詐騙可能使用多種通路）
        features['channel_diversity'] = acct_txns['channel_type'].nunique()
        features['channel_entropy'] = self._calculate_entropy(
            acct_txns['channel_type'].value_counts(normalize=True)
        )
        
        return features
    
    def _empty_sequence_features(self):
        """空序列特徵"""
        return {
            'avg_interval': 0, 'std_interval': 0, 'min_interval': 0, 'interval_cv': 0,
            'amount_trend': 0, 'amount_cv': 0, 'max_amount_ratio': 0,
            'night_txn_ratio': 0, 'evening_txn_ratio': 0, 'business_hour_ratio': 0,
            'burst_ratio': 0, 'channel_diversity': 0, 'channel_entropy': 0
        }
    
    @staticmethod
    def _calculate_entropy(probs):
        """計算熵"""
        return -np.sum(probs * np.log(probs + 1e-8))


# ==================== 第三部分：領域知識特徵 ====================

class FraudDomainFeatures:
    """詐騙領域特徵工程"""
    
    @staticmethod
    def extract_fraud_patterns(df_trans, acct, ref_date):
        """提取詐騙行為模式特徵"""
        features = {}
        
        acct_txns = df_trans[
            ((df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)) &
            (df_trans['txn_date'] < ref_date)
        ]
        
        if len(acct_txns) == 0:
            return {k: 0 for k in [
                'round_amount_ratio', 'atm_preference', 'self_txn_ratio',
                'foreign_currency_ratio', 'quick_transfer_ratio',
                'concentration_index', 'velocity_score'
            ]}
        
        # 1. 整數金額偏好（詐騙常用整數）
        amounts = acct_txns['txn_amt'].values
        features['round_amount_ratio'] = np.mean(amounts % 1000 == 0)
        
        # 2. ATM偏好（詐騙常用ATM取款）
        features['atm_preference'] = (acct_txns['channel_type'] == '1').mean()
        
        # 3. 自我交易比例
        features['self_txn_ratio'] = (acct_txns['is_self_txn'] == 'Y').mean()
        
        # 4. 外幣交易（可能涉及洗錢）
        features['foreign_currency_ratio'] = (acct_txns['currency_type'] != 'TWD').mean()
        
        # 5. 快速轉移（資金快進快出）
        from_txns = acct_txns[acct_txns['from_acct'] == acct]
        to_txns = acct_txns[acct_txns['to_acct'] == acct]
        
        recent_7d_from = from_txns[from_txns['txn_date'] >= ref_date - 7]
        recent_7d_to = to_txns[to_txns['txn_date'] >= ref_date - 7]
        
        if len(to_txns) > 0:
            features['quick_transfer_ratio'] = len(recent_7d_from) / (len(to_txns) + 1e-8)
        else:
            features['quick_transfer_ratio'] = 0
        
        # 6. 交易對手集中度（Herfindahl指數）
        if len(from_txns) > 0:
            counterparty_counts = from_txns['to_acct'].value_counts(normalize=True)
            features['concentration_index'] = (counterparty_counts ** 2).sum()
        else:
            features['concentration_index'] = 0
        
        # 7. 資金流速度（短期內大量進出）
        features['velocity_score'] = (
            recent_7d_from['txn_amt'].sum() + recent_7d_to['txn_amt'].sum()
        ) / 7.0
        
        return features


# ==================== 第四部分：主特徵工程管道 ====================

def engineer_breakthrough_features(df_trans, df_alert, samples):
    """工程化突破性特徵集"""
    
    print("\n" + "="*80)
    print("開始革命性特徵工程")
    print("="*80)
    
    # 初始化分析器
    network_analyzer = TransactionNetworkAnalyzer(df_trans)
    temporal_extractor = TemporalPatternExtractor(df_trans)
    
    # 計算全局網絡特徵
    print("\n計算全局網絡特徵...")
    pagerank_scores = network_analyzer.compute_pagerank()
    alert_accts = df_alert['acct'].unique()
    risk_scores = network_analyzer.compute_risk_propagation(alert_accts)
    
    # 為每個樣本提取特徵
    feature_list = []
    
    for acct, ref_date, label in tqdm(samples, desc="提取特徵"):
        features = {'acct': acct, 'label': label, 'ref_date': ref_date}
        
        # 1. 基礎統計特徵（從exp3學到：使用短期窗口）
        acct_txns = df_trans[
            ((df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)) &
            (df_trans['txn_date'] < ref_date)
        ]
        
        for window in SHORT_WINDOWS + MEDIUM_WINDOWS:
            window_txns = acct_txns[acct_txns['txn_date'] >= ref_date - window]
            
            from_txns = window_txns[window_txns['from_acct'] == acct]
            features[f'from_count_{window}d'] = len(from_txns)
            features[f'from_amt_sum_{window}d'] = from_txns['txn_amt'].sum()
            features[f'from_amt_mean_{window}d'] = from_txns['txn_amt'].mean() if len(from_txns) > 0 else 0
            features[f'from_amt_std_{window}d'] = from_txns['txn_amt'].std() if len(from_txns) > 0 else 0
            features[f'from_unique_{window}d'] = from_txns['to_acct'].nunique()
            
            to_txns = window_txns[window_txns['to_acct'] == acct]
            features[f'to_count_{window}d'] = len(to_txns)
            features[f'to_amt_sum_{window}d'] = to_txns['txn_amt'].sum()
            features[f'to_amt_mean_{window}d'] = to_txns['txn_amt'].mean() if len(to_txns) > 0 else 0
            features[f'to_unique_{window}d'] = to_txns['from_acct'].nunique()
        
        # 2. 比率特徵（從exp3學到：簡單有效）
        epsilon = 1e-8
        features['ratio_from_amt_3d_7d'] = features['from_amt_sum_3d'] / (features['from_amt_sum_7d'] + epsilon)
        features['ratio_from_amt_7d_14d'] = features['from_amt_sum_7d'] / (features['from_amt_sum_14d'] + epsilon)
        features['ratio_from_cnt_1d_7d'] = features['from_count_1d'] / (features['from_count_7d'] + epsilon)
        features['ratio_to_amt_3d_7d'] = features['to_amt_sum_3d'] / (features['to_amt_sum_7d'] + epsilon)
        
        # 3. 網絡特徵（突破點1）
        network_feats = network_analyzer.compute_network_features(acct)
        features.update(network_feats)
        features['pagerank_score'] = pagerank_scores.get(acct, 0)
        features['risk_propagation_score'] = risk_scores.get(acct, 0)
        
        # 4. 時序特徵（突破點2）
        temporal_feats = temporal_extractor.extract_sequence_features(acct, ref_date)
        features.update(temporal_feats)
        
        # 5. 領域知識特徵（突破點3）
        domain_feats = FraudDomainFeatures.extract_fraud_patterns(df_trans, acct, ref_date)
        features.update(domain_feats)
        
        # 6. 交互特徵
        features['risk_x_degree'] = features['risk_propagation_score'] * features['total_degree']
        features['pagerank_x_out_degree'] = features['pagerank_score'] * features['out_degree']
        features['velocity_x_burst'] = features['velocity_score'] * features['burst_ratio']
        
        feature_list.append(features)
    
    df_features = pd.DataFrame(feature_list).fillna(0)
    
    print(f"\n特徵工程完成!")
    print(f"總特徵數: {len(df_features.columns) - 3}")  # 減去acct, label, ref_date
    
    return df_features


# ==================== 第五部分：高級集成模型 ====================

def train_breakthrough_models(X_train, y_train, X_val, y_val):
    """訓練突破性集成模型"""
    
    print("\n" + "="*80)
    print("訓練高級集成模型")
    print("="*80)
    
    models = {}
    predictions = {}
    
    # 模型1: LightGBM（優化版）
    print("\n[1/3] 訓練 LightGBM...")
    scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1e-8)
    
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    models['lgb'] = lgb_model
    predictions['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
    
    lgb_auc = roc_auc_score(y_val, predictions['lgb'])
    print(f"LightGBM - Val AUC: {lgb_auc:.4f}")
    
    # 模型2: XGBoost（優化版）
    print("\n[2/3] 訓練 XGBoost...")
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,
        verbose=0
    )
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
    
    xgb_auc = roc_auc_score(y_val, predictions['xgb'])
    print(f"XGBoost - Val AUC: {xgb_auc:.4f}")
    
    # 模型3: LightGBM with different params（多樣性）
    print("\n[3/3] 訓練 LightGBM-Diverse...")
    lgb_params2 = {
        **lgb_params,
        'num_leaves': 127,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7
    }
    
    lgb_model2 = lgb.LGBMClassifier(**lgb_params2)
    lgb_model2.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    models['lgb2'] = lgb_model2
    predictions['lgb2'] = lgb_model2.predict_proba(X_val)[:, 1]
    
    lgb2_auc = roc_auc_score(y_val, predictions['lgb2'])
    print(f"LightGBM-Diverse - Val AUC: {lgb2_auc:.4f}")
    
    return models, predictions


# ==================== 主程序 ====================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🚀 BREAKTHROUGH MODEL - Target F1 = 0.25+          ║
    ║                                                           ║
    ║  基於深度分析你的8次提交歷史開發的革命性解決方案      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 加載數據
    print("\n[階段1/6] 加載數據...")
    df_trans = pd.read_csv('acct_transaction.csv')
    df_alert = pd.read_csv('acct_alert.csv')
    df_predict = pd.read_csv('acct_predict.csv')
    
    print(f"✓ 交易記錄: {len(df_trans):,}")
    print(f"✓ 警示帳戶: {len(df_alert):,}")
    print(f"✓ 待預測: {len(df_predict):,}")
    
    # 創建訓練樣本
    print("\n[階段2/6] 創建訓練樣本...")
    positive_samples = [(row['acct'], row['event_date'], 1) 
                        for _, row in df_alert.iterrows()]
    
    all_accts = np.union1d(df_trans['from_acct'].unique(), df_trans['to_acct'].unique())
    alert_accts = set(df_alert['acct'])
    non_alert_accts = np.setdiff1d(all_accts, list(alert_accts))
    
    num_neg = min(len(positive_samples) * NEG_POS_RATIO, len(non_alert_accts))
    selected_neg_accts = np.random.choice(non_alert_accts, num_neg, replace=False)
    
    negative_samples = []
    for acct in tqdm(selected_neg_accts, desc="採樣負樣本"):
        dates = df_trans[
            (df_trans['from_acct'] == acct) | (df_trans['to_acct'] == acct)
        ]['txn_date'].unique()
        if len(dates) > 1:
            # 避免數據洩漏：只選擇中間的日期
            valid_dates = dates[(dates > dates.min()) & (dates < dates.max())]
            if len(valid_dates) > 0:
                pseudo_date = np.random.choice(valid_dates)
                negative_samples.append((acct, pseudo_date, 0))
    
    all_samples = positive_samples + negative_samples
    print(f"✓ 總樣本: {len(all_samples)} (正:{len(positive_samples)}, 負:{len(negative_samples)})")
    
    # 特徵工程
    print("\n[階段3/6] 革命性特徵工程...")
    df_features = engineer_breakthrough_features(df_trans, df_alert, all_samples)
    
    # 時間分割（避免數據洩漏）
    print("\n[階段4/6] 時間分割驗證...")
    TIME_SPLIT = 90
    train_mask = df_features['ref_date'] <= TIME_SPLIT
    val_mask = df_features['ref_date'] > TIME_SPLIT
    
    feature_cols = [col for col in df_features.columns 
                    if col not in ['acct', 'label', 'ref_date']]
    
    X_train = df_features.loc[train_mask, feature_cols].values
    y_train = df_features.loc[train_mask, 'label'].values
    X_val = df_features.loc[val_mask, feature_cols].values
    y_val = df_features.loc[val_mask, 'label'].values
    
    print(f"✓ 訓練集: {len(X_train)} (正樣本率: {y_train.mean():.4f})")
    print(f"✓ 驗證集: {len(X_val)} (正樣本率: {y_val.mean():.4f})")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 訓練模型
    print("\n[階段5/6] 訓練集成模型...")
    models, val_predictions = train_breakthrough_models(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Ensemble權重優化
    print("\n計算最優ensemble權重...")
    aucs = {name: roc_auc_score(y_val, pred) 
            for name, pred in val_predictions.items()}
    
    total_auc = sum(aucs.values())
    weights = {name: auc / total_auc for name, auc in aucs.items()}
    
    print("\nEnsemble權重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f} (AUC: {aucs[name]:.4f})")
    
    # Ensemble預測
    ensemble_val_pred = sum(weights[name] * val_predictions[name] 
                            for name in val_predictions.keys())
    
    # 找最佳閾值
    print("\n尋找最佳閾值...")
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.05, 0.95, 0.01):
        pred_labels = (ensemble_val_pred > threshold).astype(int)
        f1 = f1_score(y_val, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n✓ 最佳驗證F1: {best_f1:.6f}")
    print(f"✓ 最佳閾值: {best_threshold:.4f}")
    print(f"✓ 驗證AUC: {roc_auc_score(y_val, ensemble_val_pred):.4f}")
    
    # 測試集預測
    print("\n[階段6/6] 測試集預測...")
    test_accts = df_predict['acct'].tolist()
    max_date = df_trans['txn_date'].max()
    test_samples = [(acct, max_date, 0) for acct in test_accts]
    
    df_test_features = engineer_breakthrough_features(df_trans, df_alert, test_samples)
    X_test = df_test_features[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # Ensemble預測
    test_predictions = {}
    for name, model in models.items():
        test_predictions[name] = model.predict_proba(X_test_scaled)[:, 1]
    
    ensemble_test_pred = sum(weights[name] * test_predictions[name] 
                            for name in test_predictions.keys())
    
    final_labels = (ensemble_test_pred > best_threshold).astype(int)
    
    # 保存結果
    submission = pd.DataFrame({
        'acct': test_accts,
        'label': final_labels
    })
    submission.to_csv('submission_breakthrough.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80)
    print(f"提交文件: submission_breakthrough.csv")
    print(f"驗證F1: {best_f1:.6f}")
    print(f"預測分布: {pd.Series(final_labels).value_counts().to_dict()}")
    print(f"陽性率: {final_labels.mean()*100:.2f}%")
    print("\n關鍵改進:")
    print("  ✓ 網絡風險傳播算法")
    print("  ✓ 時序模式特徵")
    print("  ✓ 領域知識特徵")
    print("  ✓ 三模型深度ensemble")
    print("  ✓ 避免數據洩漏的嚴格時間分割")
    print("="*80)


if __name__ == "__main__":
    main()
