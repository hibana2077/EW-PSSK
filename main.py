"""
EW-PSSK 主執行腳本
完整的抗癌胜肽預測實驗
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加 src 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_loader import DataLoader
from src.models.ewpssk import EWPSSK
from src.models.classifier import EWPSSKClassifier, cross_validation_evaluation
from src.utils.performance_monitor import PerformanceMonitor, format_performance_report
from src.utils.result_saver import ResultSaver

import numpy as np
import argparse
from datetime import datetime


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='EW-PSSK 抗癌胜肽預測實驗')
    parser.add_argument('--dataset', type=str, default='src/dataset/acp740.txt',
                       help='數據集文件路徑')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='EW-PSSK gamma 參數')
    parser.add_argument('--C', type=float, default=1.0,
                       help='分類器正則化參數')
    parser.add_argument('--cv_folds', type=int, default=10,
                       help='交叉驗證折數')
    parser.add_argument('--kernel_method', type=str, default='linear',
                       choices=['linear', 'precomputed', 'rbf'],
                       help='核方法類型')
    parser.add_argument('--max_length', type=int, default=50,
                       help='序列最大長度')
    parser.add_argument('--random_state', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='結果輸出目錄')
    parser.add_argument('--hyperparameter_search', action='store_true',
                       help='執行超參數搜尋而非單次實驗')
    parser.add_argument('--search_method', type=str, default='halving',
                       choices=['halving', 'manual'],
                       help='超參數搜尋方法')
    parser.add_argument('--gamma_min', type=float, default=0.3,
                       help='γ 最小值（超參數搜尋）')
    parser.add_argument('--gamma_max', type=float, default=3.0,
                       help='γ 最大值（超參數搜尋）')
    parser.add_argument('--C_min', type=float, default=1e-3,
                       help='C 最小值（超參數搜尋）')
    parser.add_argument('--C_max', type=float, default=1e2,
                       help='C 最大值（超參數搜尋）')
    parser.add_argument('--n_gamma', type=int, default=8,
                       help='γ 採樣點數（超參數搜尋）')
    parser.add_argument('--n_C', type=int, default=8,
                       help='C 採樣點數（超參數搜尋）')
    
    args = parser.parse_args()
    
    # 檢查是否執行超參數搜尋
    if args.hyperparameter_search:
        return run_hyperparameter_search_main(args)
    else:
        return run_single_experiment(args)


def run_hyperparameter_search_main(args):
    """運行超參數搜尋"""
    from src.utils.hyperparameter_search import run_hyperparameter_search
    
    print("="*60)
    print("EW-PSSK 超參數搜尋模式")
    print("="*60)
    
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
    
    return results


def run_single_experiment(args):
    """運行單次實驗"""
    print("="*60)
    print("EW-PSSK (Entropy-Weighted Position-Specific Scoring Kernel)")
    print("抗癌胜肽預測實驗")
    print("="*60)
    print(f"實驗開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"數據集: {args.dataset}")
    print(f"參數設置: gamma={args.gamma}, C={args.C}, cv_folds={args.cv_folds}")
    print("="*60)
    
    # 初始化效能監控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # 1. 載入數據
        print("\n1. 載入和預處理數據...")
        data_loader = DataLoader()
        sequences, labels = data_loader.load_fasta(args.dataset)
        
        print(f"原始數據: {len(sequences)} 個序列")
        
        # 數據清理
        sequences = data_loader.validate_sequences(sequences)
        # 確保標籤數量匹配
        if len(sequences) != len(labels):
            labels = labels[:len(sequences)]
        
        # 序列對齊（填充）
        sequences, actual_max_length = data_loader.pad_sequences(
            sequences, max_length=args.max_length
        )
        
        # 獲取數據集信息
        dataset_info = data_loader.get_dataset_info(sequences, labels)
        print(f"處理後數據: {dataset_info['total_samples']} 個序列")
        print(f"正樣本: {dataset_info['positive_samples']}, "
              f"負樣本: {dataset_info['negative_samples']}")
        print(f"序列長度: {actual_max_length}")
        
        # 2. 初始化分類器
        print("\n2. 初始化 EW-PSSK 分類器...")
        classifier = EWPSSKClassifier(
            kernel_method=args.kernel_method,
            gamma=args.gamma,
            C=args.C,
            random_state=args.random_state
        )
        
        # 3. 執行交叉驗證
        print("\n3. 執行交叉驗證評估...")
        cv_results = cross_validation_evaluation(
            classifier=classifier,
            sequences=sequences,
            labels=labels,
            cv_folds=args.cv_folds,
            random_state=args.random_state
        )
        
        # 4. 訓練完整模型以獲取權重信息
        print("\n4. 訓練完整模型...")
        full_classifier = EWPSSKClassifier(
            kernel_method=args.kernel_method,
            gamma=args.gamma,
            C=args.C,
            random_state=args.random_state
        )
        full_classifier.fit(sequences, labels)
        weights_info = full_classifier.ewpssk.get_weights_info()
        
        # 5. 停止效能監控
        performance_stats = monitor.stop_monitoring()
        
        # 6. 整理實驗結果
        print("\n5. 整理實驗結果...")
        experiment_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': args.dataset,
                'experiment_type': 'cross_validation'
            },
            'model_params': {
                'gamma': args.gamma,
                'C': args.C,
                'kernel_method': args.kernel_method,
                'cv_folds': args.cv_folds,
                'max_length': args.max_length,
                'random_state': args.random_state
            },
            'dataset_info': dataset_info,
            'cv_results': cv_results,
            'weights_info': weights_info,
            'performance': performance_stats,
            'system_info': monitor.get_system_info()
        }
        
        # 7. 保存結果
        print("\n6. 保存實驗結果...")
        result_saver = ResultSaver(output_dir=args.output_dir)
        output_path = result_saver.save_experiment_results(
            experiment_results, 
            experiment_name="ewpssk_acp_prediction"
        )
        
        # 創建可視化圖表
        try:
            result_saver.create_visualization(
                experiment_results,
                experiment_name="ewpssk_acp_prediction"
            )
            print("可視化圖表已生成")
        except Exception as e:
            print(f"可視化生成失敗: {e}")
        
        # 8. 打印結果摘要
        print("\n" + "="*60)
        print("實驗結果摘要")
        print("="*60)
        
        if 'average_results' in cv_results:
            avg_results = cv_results['average_results']
            std_results = cv_results.get('std_results', {})
            
            print(f"交叉驗證結果 ({args.cv_folds} 折):")
            print("-" * 40)
            metrics = ['Accuracy', 'Sn', 'Sp', 'F1', 'MCC', 'AUC']
            for metric in metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in avg_results:
                    mean_val = avg_results[mean_key]
                    std_val = std_results.get(std_key, 0)
                    print(f"{metric:>8}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n效能統計:")
        print("-" * 40)
        print(f"總執行時間: {performance_stats.get('wall_time', 0):.4f} 秒")
        print(f"CPU 時間: {performance_stats.get('cpu_time', 0):.4f} 秒")
        print(f"記憶體峰值: {performance_stats.get('peak_memory_mb', 0):.2f} MB")
        print(f"記憶體增加: {performance_stats.get('memory_increase_mb', 0):.2f} MB")

        print(f"\n結果已保存到: {output_path}")
        print("="*60)
        
        return experiment_results
        
    except Exception as e:
        print(f"\n實驗執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop_monitoring()
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print("\n實驗成功完成！")
    else:
        print("\n實驗執行失敗！")
        sys.exit(1)
