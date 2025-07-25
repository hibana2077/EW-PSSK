#!/usr/bin/env python3
"""
快速運行腳本 - 使用 ACP740 數據集進行實驗
"""

import subprocess
import sys
import os

def main():
    """運行 EW-PSSK 實驗"""
    
    print("EW-PSSK 抗癌胜肽預測實驗")
    print("="*50)
    
    # 檢查數據集
    dataset_path = "src/dataset/acp740.txt"
    if not os.path.exists(dataset_path):
        print(f"錯誤: 找不到數據集文件 {dataset_path}")
        return False
    
    # 運行主實驗
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset_path,
        "--gamma", "1.4591",
        "--C", "0.5273",
        "--cv_folds", "10",
        "--kernel_method", "linear",
        # "--kernel_method", "precomputed",
        "--max_length", "50",
        "--random_state", "42"
    ]
    
    print("執行命令:", " ".join(cmd))
    print("開始實驗...")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n實驗成功完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n實驗執行失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n實驗被用戶中斷")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
