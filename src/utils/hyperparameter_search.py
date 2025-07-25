"""
EW-PSSK 超參數搜尋模塊
支援 γ（熵權重平滑指數）與 C（SVM 懲罰）的聯合搜尋
"""

import numpy as np
import pandas as pd
try:
    from sklearn.experimental import enable_halving_search_cv  # 啟用實驗性功能
    from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold, GridSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    HALVING_AVAILABLE = False
    print("警告: HalvingGridSearchCV 不可用，將使用 GridSearchCV")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict, List, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

from ..models.classifier import EWPSSKClassifier, calculate_metrics
from ..utils.performance_monitor import PerformanceMonitor
from ..utils.result_saver import ResultSaver


class EWPSSKWrapper(BaseEstimator, ClassifierMixin):
    """
    EW-PSSK 包裝器，符合 scikit-learn 接口
    用於超參數搜尋
    """
    
    def __init__(self, gamma=1.0, C=1.0, kernel_method='linear', random_state=42):
        self.gamma = gamma
        self.C = C
        self.kernel_method = kernel_method
        self.random_state = random_state
        self.classifier_ = None
    
    def fit(self, X, y):
        """訓練模型，X 是序列列表"""
        self.classifier_ = EWPSSKClassifier(
            gamma=self.gamma,
            C=self.C,
            kernel_method=self.kernel_method,
            random_state=self.random_state
        )
        self.classifier_.fit(X, y)
        return self
    
    def predict(self, X):
        """預測"""
        if self.classifier_ is None:
            raise ValueError("模型尚未訓練")
        return self.classifier_.predict(X)
    
    def predict_proba(self, X):
        """預測機率"""
        if self.classifier_ is None:
            raise ValueError("模型尚未訓練")
        return self.classifier_.predict_proba(X)
    
    def score(self, X, y):
        """計算 AUC 分數"""
        try:
            y_proba = self.predict_proba(X)
            return roc_auc_score(y, y_proba[:, 1])
        except:
            return 0.0


class HyperparameterSearch:
    """超參數搜尋類"""
    
    def __init__(self, kernel_method='linear', cv_folds=5, random_state=42,
                 n_jobs=1, verbose=1):
        """
        初始化超參數搜尋
        
        Args:
            kernel_method: 核方法 ('linear' 或 'precomputed')
            cv_folds: 交叉驗證折數
            random_state: 隨機種子
            n_jobs: 並行作業數
            verbose: 詳細程度
        """
        self.kernel_method = kernel_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.search_results_ = None
    
    def create_param_grid(self, gamma_range=(0.3, 3.0), C_range=(1e-3, 1e2),
                         n_gamma=10, n_C=10) -> Dict[str, List]:
        """
        創建參數網格
        
        Args:
            gamma_range: γ 範圍 (min, max)
            C_range: C 範圍 (min, max)
            n_gamma: γ 的採樣點數
            n_C: C 的採樣點數
            
        Returns:
            參數網格字典
        """
        # Log-uniform 採樣
        gamma_values = np.logspace(
            np.log10(gamma_range[0]), 
            np.log10(gamma_range[1]), 
            n_gamma
        )
        C_values = np.logspace(
            np.log10(C_range[0]), 
            np.log10(C_range[1]), 
            n_C
        )
        
        param_grid = {
            'gamma': gamma_values.tolist(),
            'C': C_values.tolist()
        }
        
        total_combinations = len(param_grid['gamma']) * len(param_grid['C'])
        print(f"創建參數網格: {len(param_grid['gamma'])} × {len(param_grid['C'])} = {total_combinations} 種組合")
        print(f"γ 範圍: {gamma_range[0]:.3f} - {gamma_range[1]:.3f}")
        print(f"C 範圍: {C_range[0]:.3f} - {C_range[1]:.3f}")
        
        return param_grid
    
    def halving_grid_search(self, sequences: List[str], labels: List[int],
                           param_grid: Dict[str, List],
                           factor: int = 2,
                           min_resources: int = 20,
                           max_resources: str = 'auto') -> Dict[str, Any]:
        """
        使用 HalvingGridSearchCV 或 GridSearchCV 進行超參數搜尋
        
        Args:
            sequences: 序列列表
            labels: 標籤列表
            param_grid: 參數網格
            factor: 每次迭代的減少因子（僅用於 HalvingGridSearchCV）
            min_resources: 最小資源數（僅用於 HalvingGridSearchCV）
            max_resources: 最大資源數（僅用於 HalvingGridSearchCV）
            
        Returns:
            搜尋結果
        """
        if HALVING_AVAILABLE:
            print(f"開始 Halving Grid Search...")
            print(f"參數: factor={factor}, min_resources={min_resources}")
        else:
            print(f"開始 Grid Search...")
            
        start_time = time.time()
        
        # 創建估計器
        estimator = EWPSSKWrapper(
            kernel_method=self.kernel_method,
            random_state=self.random_state
        )
        
        # 創建評分函數
        scoring = make_scorer(roc_auc_score, needs_proba=True)
        
        # 創建交叉驗證
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 創建搜尋器
        if HALVING_AVAILABLE:
            search = HalvingGridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                factor=factor,
                min_resources=min_resources,
                max_resources=max_resources,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state
            )
        else:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        
        # 執行搜尋
        search.fit(sequences, labels)
        
        # 保存結果
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.cv_results_ = search.cv_results_
        
        search_time = time.time() - start_time
        
        print(f"\\n搜尋完成！耗時: {search_time:.2f} 秒")
        print(f"最佳參數: {self.best_params_}")
        print(f"最佳 AUC: {self.best_score_:.4f}")
        
        # 整理結果
        if HALVING_AVAILABLE:
            self.search_results_ = {
                'best_params': self.best_params_,
                'best_score': self.best_score_,
                'search_time': search_time,
                'n_splits_': getattr(search, 'n_splits_', self.cv_folds),
                'n_candidates_': getattr(search, 'n_candidates_', len(param_grid['gamma']) * len(param_grid['C'])),
                'n_resources_': getattr(search, 'n_resources_', 'N/A'),
                'cv_results': search.cv_results_
            }
        else:
            self.search_results_ = {
                'best_params': self.best_params_,
                'best_score': self.best_score_,
                'search_time': search_time,
                'cv_results': search.cv_results_
            }
        
        return self.search_results_
    
    def manual_grid_search(self, sequences: List[str], labels: List[int],
                          param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        手動網格搜尋（用於預計算核或自定義需求）
        
        Args:
            sequences: 序列列表
            labels: 標籤列表
            param_grid: 參數網格
            
        Returns:
            搜尋結果
        """
        print(f"開始手動網格搜尋...")
        start_time = time.time()
        
        best_score = 0
        best_params = None
        results = []
        
        total_combinations = len(param_grid['gamma']) * len(param_grid['C'])
        current = 0
        
        for gamma in param_grid['gamma']:
            for C in param_grid['C']:
                current += 1
                print(f"\\n進度 {current}/{total_combinations}: γ={gamma:.3f}, C={C:.3f}")
                
                # 執行交叉驗證
                scores = self._evaluate_params(sequences, labels, gamma, C)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                result = {
                    'gamma': gamma,
                    'C': C,
                    'mean_auc': mean_score,
                    'std_auc': std_score,
                    'scores': scores
                }
                results.append(result)
                
                print(f"  AUC: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'C': C}
                    print(f"  ★ 新的最佳結果！")
        
        search_time = time.time() - start_time
        
        print(f"\\n手動搜尋完成！耗時: {search_time:.2f} 秒")
        print(f"最佳參數: {best_params}")
        print(f"最佳 AUC: {best_score:.4f}")
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        self.search_results_ = {
            'best_params': best_params,
            'best_score': best_score,
            'search_time': search_time,
            'results': results
        }
        
        return self.search_results_
    
    def _evaluate_params(self, sequences: List[str], labels: List[int],
                        gamma: float, C: float) -> List[float]:
        """
        評估單組參數
        
        Args:
            sequences: 序列列表
            labels: 標籤列表
            gamma: γ 參數
            C: C 參數
            
        Returns:
            交叉驗證 AUC 分數列表
        """
        from sklearn.model_selection import StratifiedKFold
        
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(sequences, labels)):
            # 準備數據
            train_sequences = [sequences[i] for i in train_idx]
            test_sequences = [sequences[i] for i in test_idx]
            train_labels = [labels[i] for i in train_idx]
            test_labels = [labels[i] for i in test_idx]
            
            # 訓練模型
            classifier = EWPSSKClassifier(
                kernel_method=self.kernel_method,
                gamma=gamma,
                C=C,
                random_state=self.random_state
            )
            classifier.fit(train_sequences, train_labels)
            
            # 預測和評估
            y_proba = classifier.predict_proba(test_sequences)
            auc = roc_auc_score(test_labels, y_proba[:, 1])
            scores.append(auc)
        
        return scores
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """
        創建結果 DataFrame
        
        Returns:
            結果 DataFrame
        """
        if self.search_results_ is None:
            return pd.DataFrame()
        
        if 'cv_results' in self.search_results_:
            # HalvingGridSearchCV 結果
            df = pd.DataFrame(self.search_results_['cv_results'])
            
            # 提取參數
            params_df = pd.json_normalize(df['params'])
            df = pd.concat([params_df, df.drop('params', axis=1)], axis=1)
            
            # 選擇重要列
            important_cols = ['gamma', 'C', 'mean_test_score', 'std_test_score', 
                            'rank_test_score', 'mean_fit_time']
            df = df[important_cols]
            
        else:
            # 手動搜尋結果
            df = pd.DataFrame(self.search_results_['results'])
        
        return df.sort_values('mean_test_score' if 'mean_test_score' in df.columns 
                            else 'mean_auc', ascending=False)
    
    def plot_search_results(self, save_path: str = None):
        """
        繪製搜尋結果熱力圖
        
        Args:
            save_path: 保存路徑
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = self.create_results_dataframe()
            if df.empty:
                print("無搜尋結果可繪製")
                return
            
            # 創建透視表
            score_col = 'mean_test_score' if 'mean_test_score' in df.columns else 'mean_auc'
            pivot_table = df.pivot(index='gamma', columns='C', values=score_col)
            
            # 繪製熱力圖
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis',
                       cbar_kws={'label': 'AUC Score'})
            
            # plt.title('EW-PSSK 超參數搜尋結果 (γ vs C)', fontsize=14, fontweight='bold')
            # plt.xlabel('C (SVM 懲罰參數)', fontsize=12)
            # plt.ylabel('γ (熵權重平滑指數)', fontsize=12)
            plt.title('EW-PSSK Hyperparameter Search Results (γ vs C)', fontsize=14, fontweight='bold')
            plt.xlabel('C (SVM Penalty Parameter)', fontsize=12)
            plt.ylabel('γ (Entropy Weight Smoothing Index)', fontsize=12)
            
            # 標記最佳點
            if self.best_params_:
                best_gamma = self.best_params_['gamma']
                best_C = self.best_params_['C']
                
                # 找到最佳點在圖中的位置
                gamma_idx = pivot_table.index.get_loc(best_gamma)
                C_idx = pivot_table.columns.get_loc(best_C)
                
                plt.scatter(C_idx + 0.5, gamma_idx + 0.5, 
                           marker='*', s=500, color='red', 
                           label=f'Best: γ={best_gamma:.3f}, C={best_C:.3f}')
                plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"結果圖表已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("需要 matplotlib 和 seaborn 來繪製圖表")
        except Exception as e:
            print(f"繪圖出錯: {e}")


def run_hyperparameter_search(dataset_path: str,
                             search_method: str = 'halving',
                             kernel_method: str = 'linear',
                             cv_folds: int = 5,
                             gamma_range: Tuple[float, float] = (0.3, 3.0),
                             C_range: Tuple[float, float] = (1e-3, 1e2),
                             n_gamma: int = 8,
                             n_C: int = 8,
                             output_dir: str = 'results') -> Dict[str, Any]:
    """
    運行超參數搜尋的主函數
    
    Args:
        dataset_path: 數據集路徑
        search_method: 搜尋方法 ('halving' 或 'manual')
        kernel_method: 核方法
        cv_folds: 交叉驗證折數
        gamma_range: γ 搜尋範圍
        C_range: C 搜尋範圍
        n_gamma: γ 採樣點數
        n_C: C 採樣點數
        output_dir: 輸出目錄
        
    Returns:
        搜尋結果字典
    """
    print("="*60)
    print("EW-PSSK 超參數搜尋")
    print("="*60)
    print(f"數據集: {dataset_path}")
    print(f"搜尋方法: {search_method}")
    print(f"核方法: {kernel_method}")
    print(f"交叉驗證: {cv_folds} 折")
    print("="*60)
    
    # 初始化效能監控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # 載入數據
        from ..utils.data_loader import DataLoader
        
        print("\\n1. 載入和預處理數據...")
        data_loader = DataLoader()
        sequences, labels = data_loader.load_fasta(dataset_path)
        sequences = data_loader.validate_sequences(sequences)
        sequences, max_length = data_loader.pad_sequences(sequences, max_length=50)
        
        if len(sequences) != len(labels):
            labels = labels[:len(sequences)]
        
        dataset_info = data_loader.get_dataset_info(sequences, labels)
        print(f"數據集信息: {dataset_info['total_samples']} 個序列 "
              f"({dataset_info['positive_samples']} 正, {dataset_info['negative_samples']} 負)")
        
        # 初始化搜尋器
        print(f"\\n2. 初始化超參數搜尋器...")
        searcher = HyperparameterSearch(
            kernel_method=kernel_method,
            cv_folds=cv_folds,
            random_state=42,
            verbose=1
        )
        
        # 創建參數網格
        print(f"\\n3. 創建參數網格...")
        param_grid = searcher.create_param_grid(
            gamma_range=gamma_range,
            C_range=C_range,
            n_gamma=n_gamma,
            n_C=n_C
        )
        
        # 執行搜尋
        print(f"\\n4. 執行超參數搜尋...")
        if search_method == 'halving':
            search_results = searcher.halving_grid_search(sequences, labels, param_grid)
        else:
            search_results = searcher.manual_grid_search(sequences, labels, param_grid)
        
        # 停止效能監控
        performance_stats = monitor.stop_monitoring()
        
        # 整理最終結果
        final_results = {
            'search_info': {
                'dataset_path': dataset_path,
                'search_method': search_method,
                'kernel_method': kernel_method,
                'cv_folds': cv_folds,
                'gamma_range': gamma_range,
                'C_range': C_range,
                'n_gamma': n_gamma,
                'n_C': n_C
            },
            'dataset_info': dataset_info,
            'search_results': search_results,
            'performance': performance_stats
        }
        
        # 保存結果
        print(f"\\n5. 保存搜尋結果...")
        result_saver = ResultSaver(output_dir=output_dir)
        output_path = result_saver.save_experiment_results(
            final_results,
            experiment_name=f"ewpssk_hyperparameter_search_{search_method}"
        )
        
        # 保存詳細結果表格
        df = searcher.create_results_dataframe()
        csv_path = f"{output_path}/hyperparameter_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"詳細結果已保存到: {csv_path}")
        
        # 繪製結果圖表
        plot_path = f"{output_path}/hyperparameter_heatmap.png"
        searcher.plot_search_results(save_path=plot_path)
        
        # 打印摘要
        print("\\n" + "="*60)
        print("超參數搜尋結果摘要")
        print("="*60)
        print(f"最佳參數: γ={search_results['best_params']['gamma']:.4f}, "
              f"C={search_results['best_params']['C']:.4f}")
        print(f"最佳 AUC: {search_results['best_score']:.4f}")
        print(f"搜尋耗時: {search_results['search_time']:.2f} 秒")
        print(f"總記憶體峰值: {performance_stats['peak_memory_mb']:.2f} MB")
        print(f"結果保存到: {output_path}")
        print("="*60)
        
        return final_results
        
    except Exception as e:
        print(f"\\n超參數搜尋執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop_monitoring()
        return None
