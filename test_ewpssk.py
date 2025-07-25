"""
測試 EW-PSSK 實現的簡單腳本
"""

import sys
import os
sys.path.append('src')

from src.utils.data_loader import DataLoader
from src.models.classifier import EWPSSKClassifier
import numpy as np

def test_ewpssk():
    """測試 EW-PSSK 基本功能"""
    
    print("測試 EW-PSSK 基本功能...")
    
    # 創建測試數據
    test_sequences = [
        "GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV",
        "GLFDIIKKIAESI", 
        "GLLDIVKKVVGAFGSL",
        "MTISLIWGIAMVVCCCIWVIFDRRRRKAGEPPL",
        "MFATPLRQPTNASGARPAVSMDGQETPFQYEITD"
    ]
    test_labels = [1, 1, 1, 0, 0]
    
    print(f"測試數據: {len(test_sequences)} 個序列")
    
    # 數據預處理
    data_loader = DataLoader()
    padded_sequences, max_len = data_loader.pad_sequences(test_sequences, max_length=35)
    
    print(f"填充後序列長度: {max_len}")
    
    # 測試線性核方法
    print("\n測試線性核方法...")
    linear_classifier = EWPSSKClassifier(kernel_method='linear', gamma=1.0)
    linear_classifier.fit(padded_sequences, test_labels)
    
    # 預測
    predictions = linear_classifier.predict(padded_sequences[:3])
    probas = linear_classifier.predict_proba(padded_sequences[:3])
    
    print(f"預測結果: {predictions}")
    print(f"預測機率: {probas}")
    
    # 測試預計算核方法
    print("\n測試預計算核方法...")
    precomputed_classifier = EWPSSKClassifier(kernel_method='precomputed', gamma=1.0)
    precomputed_classifier.fit(padded_sequences, test_labels)
    
    # 預測
    predictions_pre = precomputed_classifier.predict(padded_sequences[:3])
    probas_pre = precomputed_classifier.predict_proba(padded_sequences[:3])
    
    print(f"預測結果: {predictions_pre}")
    print(f"預測機率: {probas_pre}")
    
    print("\n✓ 所有測試通過！")

if __name__ == "__main__":
    test_ewpssk()
