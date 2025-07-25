"""
模型訓練和評估模塊
包含邏輯回歸分類器和評估指標
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from typing import List, Dict, Tuple, Optional
import time


class EWPSSKClassifier:
    """
    基於 EW-PSSK 的抗癌胜肽分類器
    """
    
    def __init__(self, kernel_method: str = 'linear', gamma: float = 1.0,
                 C: float = 1.0, random_state: int = 42):
        """
        初始化分類器
        
        Args:
            kernel_method: 'linear' 使用線性核，'precomputed' 使用預計算核
            gamma: EW-PSSK 的 gamma 參數
            C: 正則化參數
            random_state: 隨機種子
        """
        self.kernel_method = kernel_method
        self.gamma = gamma
        self.C = C
        self.random_state = random_state
        
        # 根據核方法選擇分類器
        if kernel_method == 'linear':
            self.classifier = LogisticRegression(
                C=C, random_state=random_state, max_iter=1000
            )
        elif kernel_method == 'precomputed':
            self.classifier = SVC(
                kernel='precomputed', C=C, probability=True, random_state=random_state
            )
        else:
            raise ValueError(f"不支持的核方法: {kernel_method}")
        
        self.ewpssk = None
        self.is_fitted = False
        self.train_sequences = None  # 保存訓練序列用於預計算核
    
    def fit(self, sequences: List[str], labels: List[int]) -> 'EWPSSKClassifier':
        """
        訓練分類器
        
        Args:
            sequences: 訓練序列
            labels: 訓練標籤
            
        Returns:
            self
        """
        print(f"使用 {self.kernel_method} 核方法訓練分類器...")
        start_time = time.time()
        
        # 導入 EW-PSSK
        from ..models.ewpssk import EWPSSK
        
        # 初始化並訓練 EW-PSSK
        self.ewpssk = EWPSSK(gamma=self.gamma)
        self.ewpssk.fit(sequences)
        
        # 根據核方法準備特徵
        if self.kernel_method == 'linear':
            X = self.ewpssk.transform(sequences)
        else:  # precomputed
            X = self.ewpssk.compute_kernel_matrix(sequences)
        
        # 訓練分類器
        y = np.array(labels)
        self.classifier.fit(X, y)
        
        # 保存訓練序列用於預計算核
        if self.kernel_method == 'precomputed':
            self.train_sequences = sequences.copy()
        
        self.is_fitted = True
        fit_time = time.time() - start_time
        print(f"分類器訓練完成，總耗時: {fit_time:.4f} 秒")
        
        return self
    
    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        預測序列標籤
        
        Args:
            sequences: 待預測序列
            
        Returns:
            預測標籤
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        if self.kernel_method == 'linear':
            X = self.ewpssk.transform(sequences)
        else:  # precomputed
            # 計算測試序列與訓練序列之間的核矩陣
            if self.train_sequences is None:
                raise ValueError("預計算核方法需要訓練序列信息")
            X = self._compute_test_kernel_matrix(sequences, self.train_sequences)
        
        return self.classifier.predict(X)
    
    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """
        預測序列的機率
        
        Args:
            sequences: 待預測序列
            
        Returns:
            預測機率矩陣
        """
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        if self.kernel_method == 'linear':
            X = self.ewpssk.transform(sequences)
        else:  # precomputed
            # 計算測試序列與訓練序列之間的核矩陣
            if self.train_sequences is None:
                raise ValueError("預計算核方法需要訓練序列信息")
            X = self._compute_test_kernel_matrix(sequences, self.train_sequences)
        
        return self.classifier.predict_proba(X)
    
    def _compute_test_kernel_matrix(self, test_sequences: List[str], 
                                   train_sequences: List[str]) -> np.ndarray:
        """
        計算測試序列與訓練序列之間的核矩陣
        
        Args:
            test_sequences: 測試序列列表
            train_sequences: 訓練序列列表
            
        Returns:
            核矩陣 (n_test x n_train)
        """
        n_test = len(test_sequences)
        n_train = len(train_sequences)
        kernel_matrix = np.zeros((n_test, n_train))
        
        for i, test_seq in enumerate(test_sequences):
            for j, train_seq in enumerate(train_sequences):
                kernel_matrix[i, j] = self.ewpssk.kernel_function(test_seq, train_seq)
        
        return kernel_matrix


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    計算分類評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        y_proba: 預測機率 (可選)
        
    Returns:
        評估指標字典
    """
    # 基本指標
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity (Sn)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 混淆矩陣
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity (Sp)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,  # Sensitivity (Sn)
        'Sn': recall,  # 敏感度
        'Sp': specificity,  # 特異度
        'F1': f1,
        'MCC': mcc,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }
    
    # AUC (如果提供了機率)
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                # 取正類的機率
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba)
            metrics['AUC'] = auc
        except ValueError:
            metrics['AUC'] = 0.0
    
    return metrics


def cross_validation_evaluation(classifier: EWPSSKClassifier, 
                               sequences: List[str], 
                               labels: List[int],
                               cv_folds: int = 10,
                               random_state: int = 42) -> Dict[str, any]:
    """
    交叉驗證評估
    
    Args:
        classifier: 分類器
        sequences: 序列列表
        labels: 標籤列表
        cv_folds: 交叉驗證折數
        random_state: 隨機種子
        
    Returns:
        交叉驗證結果
    """
    print(f"開始 {cv_folds} 折交叉驗證...")
    start_time = time.time()
    
    # 導入 EW-PSSK
    from ..models.ewpssk import EWPSSK
    
    # 分層交叉驗證
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_times = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, labels)):
        print(f"處理第 {fold + 1}/{cv_folds} 折...")
        fold_start = time.time()
        
        # 準備訓練和測試數據
        train_sequences = [sequences[i] for i in train_idx]
        test_sequences = [sequences[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        
        # 訓練模型
        fold_classifier = EWPSSKClassifier(
            kernel_method=classifier.kernel_method,
            gamma=classifier.gamma,
            C=classifier.C,
            random_state=random_state
        )
        fold_classifier.fit(train_sequences, train_labels)
        
        # 預測
        y_pred = fold_classifier.predict(test_sequences)
        y_proba = fold_classifier.predict_proba(test_sequences)
        
        # 計算指標
        fold_metrics = calculate_metrics(
            np.array(test_labels), y_pred, y_proba
        )
        
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        fold_results.append(fold_metrics)
        
        print(f"第 {fold + 1} 折完成 - Acc: {fold_metrics['Accuracy']:.4f}, "
              f"AUC: {fold_metrics.get('AUC', 0):.4f}, 耗時: {fold_time:.2f}秒")
    
    # 計算平均結果
    avg_results = {}
    std_results = {}
    
    metric_names = ['Accuracy', 'Sn', 'Sp', 'F1', 'MCC', 'AUC']
    for metric in metric_names:
        values = [result.get(metric, 0) for result in fold_results]
        avg_results[f'{metric}_mean'] = np.mean(values)
        std_results[f'{metric}_std'] = np.std(values)
    
    total_time = time.time() - start_time
    
    return {
        'fold_results': fold_results,
        'average_results': avg_results,
        'std_results': std_results,
        'total_time': total_time,
        'avg_fold_time': np.mean(fold_times),
        'cv_folds': cv_folds
    }
