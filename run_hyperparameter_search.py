#!/usr/bin/env python3
"""
快速超參數搜尋腳本
使用推薦的參數範圍進行搜尋
"""

import subprocess
import sys
import os

def main():
    """運行超參數搜尋"""
    
    print("EW-PSSK 超參數搜尋")
    print("γ（熵權重平滑指數）與 C（SVM 懲罰）聯合搜尋")
    print("="*60)
    
    # 檢查數據集
    # dataset_path = "src/dataset/acp740.txt"
    dataset_path = "src/dataset/acp20.txt"
    if not os.path.exists(dataset_path):
        print(f"錯誤: 找不到數據集文件 {dataset_path}")
        return False
    
    print("推薦配置:")
    print("- γ 範圍: [0.3, 3.0] (8 點, log-uniform)")
    print("- C 範圍: [0.001, 100] (8 點, log-uniform)")
    print("- 使用 HalvingGridSearchCV 加速搜尋")
    print("- 5 折交叉驗證")
    print()
    
    # 詢問用戶確認
    confirm = input("是否使用推薦配置開始搜尋？ (y/N): ")
    if confirm.lower() not in ['y', 'yes']:
        print("搜尋已取消")
        return False
    
    # 運行超參數搜尋
    cmd = [
        sys.executable, "hyperparameter_search.py",
        "--dataset", dataset_path,
        # "--search_method", "halving",
        "--search_method", "manual",  # 使用手動搜尋以便於測試
        "--kernel_method", "linear",
        # "--kernel_method", "rbf",  # 使用 RBF 核方法
        "--cv_folds", "5",
        "--gamma_min", "2.5",
        "--gamma_max", "3.5",
        "--C_min", "3.0",
        "--C_max", "4.0",
        "--n_gamma", "10",
        "--n_C", "10"
    ]
    
    print("執行命令:", " ".join(cmd))
    print("開始超參數搜尋...")
    print("預計時間: 2-5 分鐘（取決於 CPU 性能）")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\\n超參數搜尋成功完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n超參數搜尋執行失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n搜尋被用戶中斷")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
