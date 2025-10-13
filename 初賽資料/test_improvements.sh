#!/bin/bash

# 測試改進效果的腳本
# 比較修復前後的差異

echo "=============================================================================="
echo "測試 Data Leakage 修復效果"
echo "=============================================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}步驟 1: 運行診斷分析${NC}"
echo "執行: python comprehensive_diagnosis.py"
echo ""
python comprehensive_diagnosis.py
echo ""

echo "=============================================================================="
echo -e "${YELLOW}步驟 2: 運行改進模型${NC}"
echo "執行: python improved_model.py"
echo ""
echo "⚠️  這會需要 15-30 分鐘，取決於你的硬體"
echo "⚠️  如果遇到 OOM，請編輯 improved_model.py 設定 USE_GNN = False"
echo ""
read -p "按 Enter 繼續，或 Ctrl+C 取消..."

python improved_model.py

echo ""
echo "=============================================================================="
echo -e "${YELLOW}步驟 3: 對比結果${NC}"
echo "=============================================================================="
echo ""

# 檢查是否生成了新文件
if [ -f "submission_improved.csv" ]; then
    echo -e "${GREEN}✅ 改進模型提交文件生成成功${NC}"
    echo ""

    # 統計預測數量
    OLD_COUNT=$(tail -n +2 submission.csv | awk -F',' '{sum+=$2} END {print sum}')
    NEW_COUNT=$(tail -n +2 submission_improved.csv | awk -F',' '{sum+=$2} END {print sum}')

    echo "預測對比:"
    echo "  舊模型 (submission.csv): 預測 $OLD_COUNT 個 alert 帳戶"
    echo "  新模型 (submission_improved.csv): 預測 $NEW_COUNT 個 alert 帳戶"
    echo ""

    # 顯示前幾筆預測
    echo "舊模型預測樣本:"
    head -6 submission.csv
    echo ""

    echo "新模型預測樣本:"
    head -6 submission_improved.csv
    echo ""

    # 檢查預測分數
    if [ -f "prediction_scores_improved.csv" ]; then
        echo "預測分數統計 (新模型):"
        python -c "
import pandas as pd
df = pd.read_csv('prediction_scores_improved.csv')
print(df['score'].describe())
print(f'\n預測為 1 的帳戶: {df[\"prediction\"].sum()}')
print(f'預測為 0 的帳戶: {(df[\"prediction\"] == 0).sum()}')
"
    fi
else
    echo -e "${RED}❌ 改進模型執行失敗，請檢查錯誤信息${NC}"
fi

echo ""
echo "=============================================================================="
echo -e "${YELLOW}步驟 4: 檢查生成的分析文件${NC}"
echo "=============================================================================="
echo ""

FILES=(
    "feature_comparison.csv"
    "alert_temporal_distribution.csv"
    "prediction_scores_improved.csv"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file (未生成)${NC}"
    fi
done

echo ""
echo "=============================================================================="
echo -e "${GREEN}完成！${NC}"
echo "=============================================================================="
echo ""
echo "📊 查看完整報告:"
echo "   cat DIAGNOSIS_REPORT.md"
echo "   cat SUMMARY.md"
echo ""
echo "🚀 提交文件:"
echo "   submission_improved.csv"
echo ""
echo "📈 下一步:"
echo "   1. 將 submission_improved.csv 提交到競賽平台"
echo "   2. 查看實際 F1 score"
echo "   3. 如果 F1 有提升，說明修復成功！"
echo "   4. 可以進一步調整特徵和超參數"
echo ""
