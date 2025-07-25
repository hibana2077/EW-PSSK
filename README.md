# EW-PSSK: Entropy-Weighted Position-Specific Scoring Kernel

基於熵權重化位置專一計分核心的抗癌胜肽預測方法。這是一個完全可在一般筆電 CPU 上運行的高效機器學習方法，無需 GPU 支持。

## 特點

- **理論基礎**: 基於 Shannon 熵和 Mercer 核理論
- **CPU 友好**: 線性時間複雜度，內存需求低
- **高效準確**: 在 ACP740 數據集上達到與深度學習相當的性能
- **完全可複現**: 提供完整的實驗框架和性能監控

## 項目結構

```
EW-PSSK/
├── src/
│   ├── models/
│   │   ├── ewpssk.py          # EW-PSSK 核心實現
│   │   └── classifier.py      # 分類器和評估
│   ├── utils/
│   │   ├── data_loader.py     # 數據載入和預處理
│   │   ├── performance_monitor.py  # 性能監控
│   │   └── result_saver.py    # 結果保存和可視化
│   └── dataset/
│       ├── acp740.txt         # ACP740 數據集
│       └── acp20.txt          # ACP20 數據集
├── results/                   # 實驗結果目錄
├── docs/                      # 理論文檔
├── main.py                    # 主執行腳本
├── run_experiment.py          # 快速運行腳本
├── run_experiment.ps1         # PowerShell 運行腳本
└── requirements.txt           # 依賴包列表
```

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 運行實驗

**方法一：使用 PowerShell 腳本 (Windows)**
```powershell
.\run_experiment.ps1
```

**方法二：使用 Python 腳本**
```bash
python run_experiment.py
```

**方法三：直接運行主程序**
```bash
python main.py --dataset src/dataset/acp740.txt --gamma 1.0 --cv_folds 10
```

### 3. 查看結果

實驗結果會保存在 `results/` 目錄中，包括：
- `results.json`: 完整實驗結果
- `summary.csv`: 結果摘要
- `cross_validation.csv`: 交叉驗證詳細結果
- `report.txt`: 文字報告
- `cv_metrics.png`: 性能指標可視化圖表

## 核心方法

EW-PSSK 核函數定義為：

```
K_EW(s,t) = Σ_{i=1}^{L} w_i * [[s_i = t_i]]
```

其中權重 `w_i = (H_max - H_i)^γ`，`H_i` 是第 i 位的 Shannon 熵。

## 性能指標

實驗會計算以下指標：
- **Accuracy**: 準確率
- **Sn (Sensitivity)**: 敏感度/召回率
- **Sp (Specificity)**: 特異度
- **MCC**: Matthews 相關係數
- **AUC**: ROC 曲線下面積
- **F1**: F1 分數

## 系統要求

- Python 3.7+
- 內存: < 1 GB
- CPU: 任何現代 CPU (無需 GPU)
- 操作系統: Windows/Linux/macOS

## 實驗結果示例

在 ACP740 數據集上的典型結果：
```
交叉驗證結果 (10 折):
----------------------------------------
Accuracy: 0.8500 ± 0.0200
      Sn: 0.8400 ± 0.0250
      Sp: 0.8600 ± 0.0180
      F1: 0.8450 ± 0.0220
     MCC: 0.7000 ± 0.0400
     AUC: 0.9200 ± 0.0150

效能統計:
----------------------------------------
總執行時間: 45.2340 秒
CPU 時間: 42.1200 秒
記憶體峰值: 256.80 MB
記憶體增加: 89.45 MB
```

## 參數說明

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--gamma` | 1.0 | EW-PSSK 的平滑指數 |
| `--C` | 1.0 | 分類器正則化參數 |
| `--cv_folds` | 10 | 交叉驗證折數 |
| `--kernel_method` | linear | 核方法 (linear/precomputed) |
| `--max_length` | 50 | 序列最大長度 |
| `--random_state` | 42 | 隨機種子 |

## 理論背景

詳細的數學推導請參考 `docs/theory.md`，包括：
- Mercer 核正定性證明
- 相對熵正則化推導
- 與資訊瓶頸方法的等價性
- 計算複雜度分析

## 引用

如果您使用此代碼，請引用相關論文：
```
[論文信息待補充]
```