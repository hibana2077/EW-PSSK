"""
EW-PSSK (Entropy-Weighted Position-Specific Scoring Kernel) 核心實現
基於理論文檔中的數學推導
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
import time


class EWPSSK:
    """
    熵權重化位置專一計分核心 (Entropy-Weighted Position-Specific Scoring Kernel)
    
    基於論文理論：
    K_EW(s,t) = Σ_{i=1}^{L} w_i * [[s_i = t_i]]
    其中 w_i = (H_max - H_i)^γ
    """
    
    def __init__(self, gamma: float = 1.0, max_entropy: Optional[float] = None):
        """
        初始化 EW-PSSK
        
        Args:
            gamma: 平滑指數，控制熵權重的非線性程度
            max_entropy: 最大熵值，如果為 None 則使用 log(20)
        """
        self.gamma = gamma
        self.max_entropy = max_entropy if max_entropy is not None else np.log(20)
        self.weights_ = None
        self.position_entropies_ = None
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
    
    def _calculate_position_entropies(self, sequences: List[str]) -> np.ndarray:
        """
        計算每個位置的 Shannon 熵
        
        Args:
            sequences: 蛋白質序列列表 (已對齊)
            
        Returns:
            每個位置的熵值陣列
        """
        if not sequences:
            raise ValueError("序列列表不能為空")
        
        L = len(sequences[0])  # 序列長度
        entropies = np.zeros(L)
        
        for pos in range(L):
            # 統計該位置各胺基酸的頻率
            aa_counts = {}
            valid_count = 0
            
            for seq in sequences:
                if pos < len(seq):
                    aa = seq[pos]
                    if aa in self.amino_acids:  # 忽略填充符號 'X'
                        aa_counts[aa] = aa_counts.get(aa, 0) + 1
                        valid_count += 1
            
            if valid_count == 0:
                # 如果該位置全是填充符號，設熵為最大值
                entropies[pos] = self.max_entropy
                continue
            
            # 計算機率分布
            probabilities = np.array([aa_counts.get(aa, 0) / valid_count 
                                    for aa in self.amino_acids])
            
            # 計算 Shannon 熵: H = -Σ p_i * log(p_i)
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log(p)
            
            entropies[pos] = entropy
        
        return entropies
    
    def _calculate_weights(self, entropies: np.ndarray) -> np.ndarray:
        """
        根據熵計算位置權重
        
        Args:
            entropies: 位置熵陣列
            
        Returns:
            位置權重陣列
        """
        # w_i = (H_max - H_i)^γ
        information_content = self.max_entropy - entropies
        # 確保權重非負
        information_content = np.maximum(information_content, 0)
        weights = np.power(information_content, self.gamma)
        
        return weights
    
    def fit(self, sequences: List[str]) -> 'EWPSSK':
        """
        訓練 EW-PSSK 核，計算權重
        
        Args:
            sequences: 訓練序列列表
            
        Returns:
            self
        """
        print("計算位置熵...")
        start_time = time.time()
        
        self.position_entropies_ = self._calculate_position_entropies(sequences)
        self.weights_ = self._calculate_weights(self.position_entropies_)
        
        fit_time = time.time() - start_time
        print(f"權重計算完成，耗時: {fit_time:.4f} 秒")
        
        return self
    
    def kernel_function(self, seq1: str, seq2: str) -> float:
        """
        計算兩個序列之間的核值
        
        Args:
            seq1, seq2: 兩個序列
            
        Returns:
            核值
        """
        if self.weights_ is None:
            raise ValueError("模型尚未訓練，請先呼叫 fit() 方法")
        
        min_len = min(len(seq1), len(seq2), len(self.weights_))
        kernel_value = 0.0
        
        for i in range(min_len):
            if seq1[i] == seq2[i] and seq1[i] in self.amino_acids:
                kernel_value += self.weights_[i]
        
        return kernel_value
    
    def compute_kernel_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        計算序列集的 Gram 矩陣
        
        Args:
            sequences: 序列列表
            
        Returns:
            Gram 矩陣 (N x N)
        """
        if self.weights_ is None:
            raise ValueError("模型尚未訓練，請先呼叫 fit() 方法")
        
        print("計算 Gram 矩陣...")
        start_time = time.time()
        
        N = len(sequences)
        gram_matrix = np.zeros((N, N))
        
        # 計算上三角矩陣
        for i in range(N):
            for j in range(i, N):
                kernel_value = self.kernel_function(sequences[i], sequences[j])
                gram_matrix[i, j] = kernel_value
                gram_matrix[j, i] = kernel_value  # 對稱性
            
            if (i + 1) % 50 == 0:
                print(f"已處理 {i + 1}/{N} 個序列")
        
        compute_time = time.time() - start_time
        print(f"Gram 矩陣計算完成，耗時: {compute_time:.4f} 秒")
        
        return gram_matrix
    
    def get_feature_vector(self, sequence: str) -> np.ndarray:
        """
        將序列轉換為特徵向量 (用於線性核 SVM)
        
        Args:
            sequence: 蛋白質序列
            
        Returns:
            特徵向量
        """
        if self.weights_ is None:
            raise ValueError("模型尚未訓練，請先呼叫 fit() 方法")
        
        L = len(self.weights_)
        feature_dim = L * 20  # L 個位置 × 20 個胺基酸
        features = np.zeros(feature_dim)
        
        for pos in range(min(len(sequence), L)):
            aa = sequence[pos]
            if aa in self.aa_to_idx:
                aa_idx = self.aa_to_idx[aa]
                feat_idx = pos * 20 + aa_idx
                features[feat_idx] = np.sqrt(self.weights_[pos])  # 使用平方根權重
        
        return features
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        將序列列表轉換為特徵矩陣
        
        Args:
            sequences: 序列列表
            
        Returns:
            特徵矩陣 (N x feature_dim)
        """
        print("轉換序列為特徵向量...")
        start_time = time.time()
        
        N = len(sequences)
        feature_matrix = np.array([self.get_feature_vector(seq) for seq in sequences])
        
        transform_time = time.time() - start_time
        print(f"特徵轉換完成，耗時: {transform_time:.4f} 秒")
        
        return feature_matrix
    
    def get_weights_info(self) -> dict:
        """
        獲取權重統計信息
        
        Returns:
            權重信息字典
        """
        if self.weights_ is None:
            return {}
        
        return {
            'min_weight': float(np.min(self.weights_)),
            'max_weight': float(np.max(self.weights_)),
            'mean_weight': float(np.mean(self.weights_)),
            'std_weight': float(np.std(self.weights_)),
            'min_entropy': float(np.min(self.position_entropies_)),
            'max_entropy': float(np.max(self.position_entropies_)),
            'mean_entropy': float(np.mean(self.position_entropies_)),
            'gamma': self.gamma,
            'sequence_length': len(self.weights_)
        }
