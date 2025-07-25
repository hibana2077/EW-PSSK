"""
數據載入和預處理模塊
負責載入 FASTA 格式的數據並進行預處理
"""

import re
from typing import List, Tuple, Dict
import numpy as np


class DataLoader:
    """數據載入器類"""
    
    def __init__(self):
        """初始化數據載入器"""
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
    
    def load_fasta(self, filepath: str) -> Tuple[List[str], List[int]]:
        """
        載入 FASTA 格式的數據
        
        Args:
            filepath: FASTA 文件路徑
            
        Returns:
            tuple: (序列列表, 標籤列表)
        """
        sequences = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_sequence = ""
        current_label = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # 保存前一個序列
                if current_sequence and current_label is not None:
                    sequences.append(current_sequence)
                    labels.append(current_label)
                
                # 解析標籤
                # 格式: >ACP_1|1 或 >non-ACP_1|0
                parts = line.split('|')
                if len(parts) >= 2:
                    current_label = int(parts[1])
                else:
                    current_label = 1 if 'ACP' in line and 'non-ACP' not in line else 0
                
                current_sequence = ""
            else:
                # 序列行
                current_sequence += line.upper()
        
        # 保存最後一個序列
        if current_sequence and current_label is not None:
            sequences.append(current_sequence)
            labels.append(current_label)
        
        return sequences, labels
    
    def validate_sequences(self, sequences: List[str]) -> List[str]:
        """
        驗證和清理序列
        
        Args:
            sequences: 蛋白質序列列表
            
        Returns:
            清理後的序列列表
        """
        cleaned_sequences = []
        
        for seq in sequences:
            # 移除非胺基酸字符
            cleaned_seq = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', '', seq)
            
            # 檢查序列長度
            if len(cleaned_seq) >= 5:  # 最小長度限制
                cleaned_sequences.append(cleaned_seq)
            else:
                print(f"警告: 序列太短，已跳過: {seq}")
        
        return cleaned_sequences
    
    def pad_sequences(self, sequences: List[str], max_length: int = None) -> Tuple[List[str], int]:
        """
        填充序列到統一長度
        
        Args:
            sequences: 序列列表
            max_length: 最大長度，如果為 None 則使用最長序列的長度
            
        Returns:
            tuple: (填充後的序列列表, 實際使用的最大長度)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                # 截斷過長的序列
                padded_seq = seq[:max_length]
            else:
                # 填充短序列 (用 X 表示空位)
                padded_seq = seq + 'X' * (max_length - len(seq))
            
            padded_sequences.append(padded_seq)
        
        return padded_sequences, max_length
    
    def sequences_to_onehot(self, sequences: List[str]) -> np.ndarray:
        """
        將序列轉換為 one-hot 編碼
        
        Args:
            sequences: 序列列表
            
        Returns:
            one-hot 編碼矩陣 (N, L, 20)
        """
        N = len(sequences)
        L = len(sequences[0])  # 假設所有序列長度相同
        
        # 使用 21 維 (20個胺基酸 + 1個填充符號 X)
        onehot = np.zeros((N, L, 21), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in self.aa_to_idx:
                    onehot[i, j, self.aa_to_idx[aa]] = 1.0
                elif aa == 'X':  # 填充符號
                    onehot[i, j, 20] = 1.0
                else:
                    # 未知胺基酸，設為均勻分布
                    onehot[i, j, :20] = 1.0 / 20
        
        return onehot
    
    def get_dataset_info(self, sequences: List[str], labels: List[int]) -> Dict:
        """
        獲取數據集基本信息
        
        Args:
            sequences: 序列列表
            labels: 標籤列表
            
        Returns:
            數據集信息字典
        """
        lengths = [len(seq) for seq in sequences]
        labels_array = np.array(labels)
        
        info = {
            'total_samples': len(sequences),
            'positive_samples': int(np.sum(labels_array == 1)),
            'negative_samples': int(np.sum(labels_array == 0)),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }
        
        info['positive_ratio'] = info['positive_samples'] / info['total_samples']
        
        return info
