# 🚀 明天機房執行計劃（4-5小時）

基於深入分析，這是經過仔細設計的執行方案。

## 📋 Phase 1: 快速驗證（10-15分鐘）

### 目標：驗證環境，拿到baseline

```bash
cd 初賽資料/

# 檢查GPU
nvidia-smi

# 快速baseline（LightGBM）
CUDA_VISIBLE_DEVICES=0 python3 three.py 2>&1 | tee log_phase1.txt
```

**檢查結果**:
```python
import pandas as pd
df = pd.read_csv('submission_exp3_robust_anomaly.csv')
print(f"預測為1: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
```

合理範圍：5-20%為警示帳戶

---

## 🚀 Phase 2: 並行主力模型（60-90分鐘）

### 同時訓練3個模型

```bash
# GPU 0: GNN + XGBoost（最穩健）
CUDA_VISIBLE_DEVICES=0 python3 run_gnn_xgb_pipeline.py > log_gnn_xgb.txt 2>&1 &

# GPU 1: LightGBM v2
CUDA_VISIBLE_DEVICES=1 python3 three.py > log_lgb2.txt 2>&1 &

# GPU 2: 備用（如果需要）
# CUDA_VISIBLE_DEVICES=2 python3 other_model.py > log_other.txt 2>&1 &

# 監控進度
tail -f log_gnn_xgb.txt
```

**等待完成**:
```bash
wait
echo "所有模型完成！"
ls -lh submission*.csv
```

---

## 🎨 Phase 3: 模型集成（5-10分鐘）

### 準備集成

編輯`simple_ensemble.py`：
```python
SUBMISSION_FILES = [
    'submission.csv',                      # GNN+XGB
    'submission_exp3_robust_anomaly.csv',  # LightGBM
]
```

### 執行集成
```bash
python3 simple_ensemble.py
```

### 驗證結果
```python
import pandas as pd
final = pd.read_csv('submission_ensemble_final.csv')
print(f"總數: {len(final)}")
print(f"警示: {final['label'].sum()} ({final['label'].mean()*100:.2f}%)")
assert final['label'].isin([0, 1]).all()
print("✅ 格式正確！")
```

---

## 📦 Phase 4: 打包（5分鐘）

```bash
mkdir -p final_submissions
cp submission_ensemble_final.csv final_submissions/
cp submission.csv final_submissions/
tar -czf submissions_$(date +%Y%m%d).tar.gz final_submissions/
```

---

## ⚠️ 常見問題

### CUDA Out of Memory
```bash
pkill -9 python
nvidia-smi  # 檢查GPU狀態
```

### 預測全0或全1
- 檢查訓練日誌：`grep "F1" log_*.txt`
- 不使用異常的模型

### 運行時間過長
- 檢查進度：`tail -100 log_*.txt`
- 必要時中止：`kill -9 PID`

---

## 📊 預期結果

### 時間分配
- Phase 1: 15分鐘
- Phase 2: 90分鐘
- Phase 3: 10分鐘
- Phase 4: 5分鐘
- **總計: 120分鐘**

### 最終產出
1. **submission_ensemble_final.csv** - 主提交
2. **submission.csv** - 備用（GNN+XGB）
3. **submission_exp3_robust_anomaly.csv** - 備用（LightGBM）

### 預期F1
- 單模型: 0.40-0.55
- 集成: 0.45-0.60

---

## 🔑 關鍵點

1. ❌ **不要用ultimate_gnn_model.py** - 有bug，F1=0
2. ✅ **優先用run_gnn_xgb_pipeline.py** - 最穩健
3. ✅ **LightGBM很快** - 適合快速驗證
4. ✅ **集成通常更好** - 但每個模型必須有效
5. 📊 **監控GPU** - 用nvidia-smi確認運行

---

## ✅ 檢查清單

**出發前**:
- [ ] 數據文件齊全（3個CSV）
- [ ] 腳本準備好（.py文件）
- [ ] 打印此執行計劃

**執行後**:
- [ ] 檢查submission格式
- [ ] 備份所有結果
- [ ] 記錄每個模型的表現

**祝順利！** 🍀

---

## 📝 快速命令備忘

```bash
# 檢查環境
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# 快速驗證
CUDA_VISIBLE_DEVICES=0 python3 three.py

# 並行訓練
CUDA_VISIBLE_DEVICES=0 python3 run_gnn_xgb_pipeline.py > log1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 three.py > log2.txt 2>&1 &

# 監控
tail -f log1.txt
watch -n 5 nvidia-smi

# 集成
python3 simple_ensemble.py

# 檢查結果
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('submission_ensemble_final.csv')
print(f"警示帳戶: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
EOF
```
