#!/usr/bin/env python3
"""
一鍵運行腳本 - 自動選擇最佳方案
"""

import os
import sys
import time
import subprocess

def print_header(text):
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")

def check_files():
    """檢查必要文件"""
    required = ['acct_transaction.csv', 'acct_alert.csv', 'acct_predict.csv']
    for file in required:
        if not os.path.exists(file):
            print(f"❌ 缺少文件: {file}")
            return False
    return True

def check_gpu():
    """檢查GPU"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def estimate_time(model_type):
    """估計訓練時間"""
    times = {
        'quick': '15-20分鐘',
        'breakthrough': '40-60分鐘',
        'a100': '90-120分鐘'
    }
    return times.get(model_type, '未知')

def run_model(script_name):
    """運行模型"""
    print(f"開始運行: {script_name}")
    print("這可能需要一些時間，請耐心等待...\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=7200  # 2小時超時
        )
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ 成功! 耗時: {minutes}分{seconds}秒")
            return True, result.stdout
        else:
            print(f"\n❌ 失敗!")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("\n⏰ 超時! 訓練時間過長")
        return False, "超時"
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷")
        return False, "中斷"
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        return False, str(e)

def parse_f1_score(output):
    """從輸出中解析F1分數"""
    for line in output.split('\n'):
        if '最佳F1' in line or 'Best F1' in line or '驗證F1' in line:
            try:
                # 嘗試提取數字
                import re
                numbers = re.findall(r'0\.\d+', line)
                if numbers:
                    return float(numbers[0])
            except:
                pass
    return None

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              🚀 一鍵運行 - 自動選擇最佳方案             ║
    ║                                                           ║
    ║  基於你的8次提交歷史深度分析開發的智能運行工具         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 檢查環境
    print_header("環境檢查")
    
    if not check_files():
        print("請確保所有數據文件在當前目錄")
        return
    
    print("✅ 數據文件檢查通過")
    
    has_gpu = check_gpu()
    if has_gpu:
        print("✅ 檢測到GPU，可以使用所有模型")
    else:
        print("⚠️  未檢測到GPU，建議使用quick_test或breakthrough模型")
    
    # 選擇方案
    print_header("選擇運行方案")
    
    print("可用方案:")
    print()
    print("1. 快速驗證版 (quick_test.py)")
    print("   - 預期F1: 0.12-0.15")
    print("   - 訓練時間: 15-20分鐘")
    print("   - 推薦: ⭐⭐⭐⭐⭐ (新手首選)")
    print()
    print("2. 突破模型 (breakthrough_model.py)")
    print("   - 預期F1: 0.15-0.20")
    print("   - 訓練時間: 40-60分鐘")
    print("   - 推薦: ⭐⭐⭐⭐⭐ (追求性能)")
    print()
    print("3. A100優化版 (a100_optimized_model.py)")
    print("   - 預期F1: 0.18-0.22")
    print("   - 訓練時間: 90-120分鐘")
    if has_gpu:
        print("   - 推薦: ⭐⭐⭐⭐ (有GPU推薦)")
    else:
        print("   - 推薦: ⭐⭐ (需要GPU)")
    print()
    print("4. 智能選擇 (推薦)")
    print("   - 自動選擇最適合的方案")
    print()
    
    choice = input("請選擇 (1/2/3/4) [默認: 4]: ").strip()
    
    if not choice:
        choice = '4'
    
    # 確定運行方案
    results = []
    
    if choice == '1':
        print_header("運行: 快速驗證版")
        success, output = run_model('quick_test.py')
        if success:
            f1 = parse_f1_score(output)
            results.append(('quick_test', f1, 'submission_quick_test.csv'))
            
    elif choice == '2':
        print_header("運行: 突破模型")
        success, output = run_model('breakthrough_model.py')
        if success:
            f1 = parse_f1_score(output)
            results.append(('breakthrough', f1, 'submission_breakthrough.csv'))
            
    elif choice == '3':
        print_header("運行: A100優化版")
        if not has_gpu:
            print("⚠️  警告: 未檢測到GPU，訓練會非常慢")
            confirm = input("確定繼續? (y/n): ").lower()
            if confirm != 'y':
                print("已取消")
                return
        success, output = run_model('a100_optimized_model.py')
        if success:
            f1 = parse_f1_score(output)
            results.append(('a100_optimized', f1, 'submission_a100_optimized.csv'))
            
    elif choice == '4':
        print_header("智能選擇模式")
        
        # 策略: 先快速驗證，如果好再運行完整版
        print("步驟1: 運行快速驗證版...")
        success1, output1 = run_model('quick_test.py')
        
        if success1:
            f1_quick = parse_f1_score(output1)
            results.append(('quick_test', f1_quick, 'submission_quick_test.csv'))
            
            if f1_quick and f1_quick > 0.12:
                print(f"\n✅ 快速驗證成功! F1={f1_quick:.6f}")
                print("這個結果已經不錯，可以直接提交")
                print()
                
                run_full = input("是否運行完整的突破模型? (y/n) [推薦: y]: ").lower()
                
                if run_full != 'n':
                    print("\n步驟2: 運行突破模型...")
                    success2, output2 = run_model('breakthrough_model.py')
                    
                    if success2:
                        f1_breakthrough = parse_f1_score(output2)
                        results.append(('breakthrough', f1_breakthrough, 'submission_breakthrough.csv'))
            else:
                print(f"\n⚠️  快速驗證F1={f1_quick:.6f} 不太理想")
                print("建議檢查數據和參數")
        else:
            print("\n❌ 快速驗證失敗，請檢查錯誤信息")
    
    else:
        print(f"無效選擇: {choice}")
        return
    
    # 總結結果
    print_header("運行總結")
    
    if not results:
        print("❌ 沒有成功的模型")
        return
    
    print("成功運行的模型:\n")
    
    best_f1 = 0
    best_model = None
    
    for model_name, f1, submission_file in results:
        if f1:
            print(f"📊 {model_name}")
            print(f"   F1 Score: {f1:.6f}")
            print(f"   提交文件: {submission_file}")
            
            # 與exp3比較
            improvement = (f1 - 0.1068) / 0.1068 * 100
            if improvement > 0:
                print(f"   vs exp3: +{improvement:.2f}% ✅")
            else:
                print(f"   vs exp3: {improvement:.2f}% ❌")
            
            # 與第一名比較
            gap = (0.2514 - f1) / 0.2514 * 100
            print(f"   距第一名: {gap:.2f}%")
            print()
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = (model_name, submission_file)
        else:
            print(f"⚠️  {model_name}: 無法解析F1分數")
            print()
    
    if best_model:
        print("="*80)
        print(f"🏆 推薦提交: {best_model[1]}")
        print(f"   預期F1: {best_f1:.6f}")
        print("="*80)
    
    # 下一步建議
    print("\n下一步建議:\n")
    
    if best_f1 > 0.15:
        print("✅ 優秀! 這個結果非常好")
        print("   1. 立即提交到平台")
        print("   2. 記錄Public LB分數")
        print("   3. 可以嘗試超參數微調")
    elif best_f1 > 0.12:
        print("✅ 不錯! 已經超越了exp3")
        print("   1. 可以提交這個結果")
        print("   2. 考慮運行更完整的模型")
        print("   3. 分析特徵重要性")
    elif best_f1 > 0.10:
        print("⚠️  一般，與exp3接近")
        print("   1. 檢查特徵工程")
        print("   2. 調整超參數")
        print("   3. 考慮其他方案")
    else:
        print("❌ 結果不理想")
        print("   1. 檢查數據是否正確")
        print("   2. 查看錯誤日誌")
        print("   3. 聯繫技術支持")
    
    print("\n" + "="*80)
    print("運行完成! 祝你在競賽中取得好成績! 🚀")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷程序")
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
