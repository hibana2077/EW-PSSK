#!/usr/bin/env python3
"""
EW-PSSK 超參數搜尋腳本
gamma（熵權重平滑指數）與 C（SVM 懲罰）聯合搜尋
"""

import sys
import os
import argparse
from datetime import datetime

# 添加 src 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.hyperparameter_search import run_hyperparameter_search


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='EW-PSSK 超參數搜尋')
    parser.add_argument('--dataset', type=str, default='src/dataset/acp740.txt',
                       help='數據集文件路徑')
    parser.add_argument('--search_method', type=str, default='halving',
                       choices=['halving', 'manual'],
                       help='搜尋方法 (halving: HalvingGridSearchCV, manual: 手動網格搜尋)')
    parser.add_argument('--kernel_method', type=str, default='linear',
                       choices=['linear', 'precomputed'],
                       help='核方法類型')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='交叉驗證折數')
    parser.add_argument('--gamma_min', type=float, default=0.3,
                       help='gamma 最小值')
    parser.add_argument('--gamma_max', type=float, default=3.0,
                       help='gamma 最大值')
    parser.add_argument('--C_min', type=float, default=1e-3,
                       help='C 最小值')
    parser.add_argument('--C_max', type=float, default=1e2,
                       help='C 最大值')
    parser.add_argument('--n_gamma', type=int, default=8,
                       help='gamma 採樣點數')
    parser.add_argument('--n_C', type=int, default=8,
                       help='C 採樣點數')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    print("="*70)
    print("EW-PSSK 超參數搜尋")
    print("gamma（熵權重平滑指數）與 C（SVM 懲罰）聯合搜尋")
    print("="*70)
    print(f"搜尋開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"數據集: {args.dataset}")
    print(f"搜尋方法: {args.search_method}")
    print(f"核方法: {args.kernel_method}")
    print(f"gamma 範圍: [{args.gamma_min:.3f}, {args.gamma_max:.3f}] ({args.n_gamma} 點)")
    print(f"C 範圍: [{args.C_min:.3f}, {args.C_max:.3f}] ({args.n_C} 點)")
    print(f"交叉驗證: {args.cv_folds} 折")
    print("="*70)
    
    # 檢查數據集文件
    if not os.path.exists(args.dataset):
        print(f"錯誤: 找不到數據集文件 {args.dataset}")
        return False
    
    # 運行超參數搜尋
    try:
        results = run_hyperparameter_search(
            dataset_path=args.dataset,
            search_method=args.search_method,
            kernel_method=args.kernel_method,
            cv_folds=args.cv_folds,
            gamma_range=(args.gamma_min, args.gamma_max),
            C_range=(args.C_min, args.C_max),
            n_gamma=args.n_gamma,
            n_C=args.n_C,
            output_dir=args.output_dir
        )
        
        if results:
            print("\n超參數搜尋成功完成！")
            
            # 提供後續建議
            best_params = results['search_results']['best_params']
            print("\n建議後續步驟:")
            print(f"1. 使用最佳參數進行完整評估:")
            print(f"   python main.py --gamma {best_params['gamma']:.4f} --C {best_params['C']:.4f} --cv_folds 10")
            print(f"2. 檢查結果文件和熱力圖以了解參數敏感性")
            print(f"3. 如需要，可在最佳參數附近進行更細粒度搜尋")
            
            return True
        else:
            print("\n超參數搜尋失敗！")
            return False
            
    except KeyboardInterrupt:
        print("\n搜尋被用戶中斷")
        return False
    except Exception as e:
        print(f"\n搜尋執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
