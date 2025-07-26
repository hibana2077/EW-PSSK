"""
純 SVC 分類器測試腳本
不使用 EW-PSSK 特徵提取，僅使用基本的氨基酸編碼
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加 src 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.data_loader import DataLoader
from src.utils.performance_monitor import PerformanceMonitor, format_performance_report
from src.utils.result_saver import ResultSaver

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, make_scorer
)
from datetime import datetime
import argparse


class SequenceEncoder:
    """序列編碼器 - 不使用 EW-PSSK"""
    
    def __init__(self, max_length=50):
        """
        初始化編碼器
        
        Args:
            max_length: 序列最大長度
        """
        self.max_length = max_length
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
    
    def encode_sequences(self, sequences):
        """
        將序列編碼為數值特徵
        使用簡單的 one-hot 編碼
        
        Args:
            sequences: 序列列表
            
        Returns:
            編碼後的特徵矩陣
        """
        n_sequences = len(sequences)
        n_features = self.max_length * len(self.amino_acids)
        
        X = np.zeros((n_sequences, n_features))
        
        for i, seq in enumerate(sequences):
            # 截斷或填充序列到固定長度
            seq = seq[:self.max_length]
            seq = seq.ljust(self.max_length, 'A')  # 用 'A' 填充
            
            # One-hot 編碼
            for j, aa in enumerate(seq):
                if aa in self.aa_to_idx:
                    aa_idx = self.aa_to_idx[aa]
                    feature_idx = j * len(self.amino_acids) + aa_idx
                    X[i, feature_idx] = 1
        
        return X


def calculate_sn_sp(y_true, y_pred):
    """計算敏感度和特異度"""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sn, sp


def cross_validation_evaluation(X, y, classifier, cv_folds=5, random_state=42):
    """
    交叉驗證評估
    
    Args:
        X: 特徵矩陣
        y: 標籤
        classifier: 分類器
        cv_folds: 交叉驗證折數
        random_state: 隨機種子
        
    Returns:
        評估結果字典
    """
    # 定義評估指標
    def sn_scorer(y_true, y_pred):
        sn, _ = calculate_sn_sp(y_true, y_pred)
        return sn
    
    def sp_scorer(y_true, y_pred):
        _, sp = calculate_sn_sp(y_true, y_pred)
        return sp
    
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'auc': 'roc_auc',
        'mcc': make_scorer(matthews_corrcoef),
        'sn': make_scorer(sn_scorer),
        'sp': make_scorer(sp_scorer)
    }
    
    # 設置交叉驗證
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # 執行交叉驗證
    cv_results = cross_validate(
        classifier, X, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )
    
    # 計算統計結果
    results = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    return results


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='純 SVC 抗癌胜肽預測實驗')
    parser.add_argument('--dataset', type=str, default='src/dataset/acp740.txt',
                       help='數據集文件路徑')
    parser.add_argument('--C', type=float, default=1.0,
                       help='SVC 正則化參數')
    parser.add_argument('--gamma', type=str, default='scale',
                       help='SVC gamma 參數')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'rbf', 'poly', 'sigmoid'],
                       help='SVC 核函數')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='交叉驗證折數')
    parser.add_argument('--max_length', type=int, default=50,
                       help='序列最大長度')
    parser.add_argument('--random_state', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    print("純 SVC 抗癌胜肽預測實驗")
    print("=" * 50)
    print(f"數據集: {args.dataset}")
    print(f"SVC 參數: C={args.C}, gamma={args.gamma}, kernel={args.kernel}")
    print(f"交叉驗證折數: {args.cv_folds}")
    print(f"序列最大長度: {args.max_length}")
    print()
    
    # 初始化性能監控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # 載入數據
        print("載入數據...")
        data_loader = DataLoader()
        sequences, labels = data_loader.load_fasta(args.dataset)
        
        print(f"成功載入 {len(sequences)} 個序列")
        print(f"正樣本數: {sum(labels)}")
        print(f"負樣本數: {len(labels) - sum(labels)}")
        print(f"正樣本比例: {sum(labels) / len(labels):.4f}")
        print()
        
        # 序列統計
        seq_lengths = [len(seq) for seq in sequences]
        print(f"序列長度範圍: {min(seq_lengths)}-{max(seq_lengths)}")
        print(f"平均序列長度: {np.mean(seq_lengths):.2f} ± {np.std(seq_lengths):.2f}")
        print()
        
        # 編碼序列
        print("編碼序列...")
        encoder = SequenceEncoder(max_length=args.max_length)
        X = encoder.encode_sequences(sequences)
        y = np.array(labels)
        
        print(f"特徵矩陣形狀: {X.shape}")
        print()
        
        # 標準化特徵
        print("標準化特徵...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print()
        
        # 初始化分類器
        print(f"初始化 SVC 分類器...")
        if args.gamma == 'scale':
            gamma = 'scale'
        elif args.gamma == 'auto':
            gamma = 'auto'
        else:
            gamma = float(args.gamma)
        
        classifier = SVC(
            C=args.C,
            gamma=gamma,
            kernel=args.kernel,
            probability=True,
            random_state=args.random_state
        )
        print(f"分類器參數: {classifier.get_params()}")
        print()
        
        # 執行交叉驗證
        print(f"執行 {args.cv_folds} 折交叉驗證...")
        cv_results = cross_validation_evaluation(
            X_scaled, y, classifier, 
            cv_folds=args.cv_folds, 
            random_state=args.random_state
        )
        
        # 停止性能監控並獲取結果
        performance_info = monitor.stop_monitoring()
        
        # 準備結果
        experiment_name = f"svc_only_acp_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 模型參數信息
        model_params = {
            'C': args.C,
            'gamma': args.gamma,
            'kernel': args.kernel,
            'cv_folds': args.cv_folds,
            'max_length': args.max_length,
            'random_state': args.random_state,
            'encoding_method': 'one_hot',
            'feature_scaling': 'standard_scaler'
        }
        
        # 數據集信息
        dataset_info = {
            'total_samples': len(sequences),
            'positive_samples': sum(labels),
            'negative_samples': len(labels) - sum(labels),
            'positive_ratio': sum(labels) / len(labels),
            'sequence_length_range': f"{min(seq_lengths)}-{max(seq_lengths)}",
            'average_sequence_length': np.mean(seq_lengths),
            'sequence_length_std': np.std(seq_lengths)
        }
        
        # 交叉驗證詳細結果
        cv_details = {}
        for fold in range(args.cv_folds):
            cv_details[f'fold_{fold+1}'] = {
                'accuracy': cv_results['accuracy']['scores'][fold],
                'sn': cv_results['sn']['scores'][fold],
                'sp': cv_results['sp']['scores'][fold],
                'f1': cv_results['f1']['scores'][fold],
                'mcc': cv_results['mcc']['scores'][fold],
                'auc': cv_results['auc']['scores'][fold]
            }
        
        # 保存結果
        result_saver = ResultSaver(args.output_dir)
        
        # 保存完整結果
        full_results = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': datetime.now().isoformat(),
                'method': 'SVC_only',
                'dataset': args.dataset
            },
            'model_params': model_params,
            'dataset_info': dataset_info,
            'cv_results': {
                'accuracy': cv_results['accuracy'],
                'sn': cv_results['sn'],
                'sp': cv_results['sp'],
                'f1': cv_results['f1'],
                'mcc': cv_results['mcc'],
                'auc': cv_results['auc'],
                'average_results': {
                    'Accuracy_mean': cv_results['accuracy']['mean'],
                    'Sn_mean': cv_results['sn']['mean'],
                    'Sp_mean': cv_results['sp']['mean'],
                    'F1_mean': cv_results['f1']['mean'],
                    'MCC_mean': cv_results['mcc']['mean'],
                    'AUC_mean': cv_results['auc']['mean']
                },
                'std_results': {
                    'Accuracy_std': cv_results['accuracy']['std'],
                    'Sn_std': cv_results['sn']['std'],
                    'Sp_std': cv_results['sp']['std'],
                    'F1_std': cv_results['f1']['std'],
                    'MCC_std': cv_results['mcc']['std'],
                    'AUC_std': cv_results['auc']['std']
                }
            },
            'cv_details': cv_details,
            'performance': performance_info
        }
        
        # 保存實驗結果
        result_dir = result_saver.save_experiment_results(full_results, "svc_only_acp_prediction")
        
        # 顯示結果
        print("\n交叉驗證結果:")
        print("-" * 20)
        print(f"Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"Sn:       {cv_results['sn']['mean']:.4f} ± {cv_results['sn']['std']:.4f}")
        print(f"Sp:       {cv_results['sp']['mean']:.4f} ± {cv_results['sp']['std']:.4f}")
        print(f"F1:       {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        print(f"MCC:      {cv_results['mcc']['mean']:.4f} ± {cv_results['mcc']['std']:.4f}")
        print(f"AUC:      {cv_results['auc']['mean']:.4f} ± {cv_results['auc']['std']:.4f}")
        print()
        print(f"結果已保存到: {result_dir}")
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"實驗執行失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()
