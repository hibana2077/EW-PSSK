"""
結果保存和可視化模塊
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResultSaver:
    """結果保存器"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化結果保存器
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_experiment_results(self, results: Dict[str, Any], 
                              experiment_name: str = "ewpssk_experiment") -> str:
        """
        保存實驗結果
        
        Args:
            results: 實驗結果字典
            experiment_name: 實驗名稱
            
        Returns:
            保存的文件路徑
        """
        # 創建實驗目錄
        exp_dir = os.path.join(self.output_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存完整結果 (JSON)
        json_path = os.path.join(exp_dir, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存摘要結果 (CSV)
        csv_path = os.path.join(exp_dir, "summary.csv")
        self._save_summary_csv(results, csv_path)
        
        # 保存交叉驗證詳細結果
        if 'cv_results' in results:
            cv_path = os.path.join(exp_dir, "cross_validation.csv")
            self._save_cv_results_csv(results['cv_results'], cv_path)
        
        # 生成報告
        report_path = os.path.join(exp_dir, "report.txt")
        self._generate_text_report(results, report_path)
        
        print(f"實驗結果已保存到: {exp_dir}")
        return exp_dir
    
    def _save_summary_csv(self, results: Dict[str, Any], filepath: str):
        """保存摘要結果到 CSV"""
        summary_data = []
        
        # 基本信息
        if 'dataset_info' in results:
            info = results['dataset_info']
            summary_data.append(['Dataset', 'Total Samples', info.get('total_samples', 0)])
            summary_data.append(['Dataset', 'Positive Samples', info.get('positive_samples', 0)])
            summary_data.append(['Dataset', 'Negative Samples', info.get('negative_samples', 0)])
            summary_data.append(['Dataset', 'Positive Ratio', f"{info.get('positive_ratio', 0):.4f}"])
        
        # 交叉驗證結果
        if 'cv_results' in results and 'average_results' in results['cv_results']:
            avg_results = results['cv_results']['average_results']
            std_results = results['cv_results'].get('std_results', {})
            
            for metric in ['Accuracy', 'Sn', 'Sp', 'F1', 'MCC', 'AUC']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in avg_results:
                    mean_val = avg_results[mean_key]
                    std_val = std_results.get(std_key, 0)
                    summary_data.append(['CV Results', metric, f"{mean_val:.4f} ± {std_val:.4f}"])
        
        # 效能信息
        if 'performance' in results:
            perf = results['performance']
            summary_data.append(['Performance', 'Total Time (s)', f"{perf.get('total_time', 0):.4f}"])
            summary_data.append(['Performance', 'Peak Memory (MB)', f"{perf.get('peak_memory_mb', 0):.2f}"])
            summary_data.append(['Performance', 'CPU Time (s)', f"{perf.get('cpu_time', 0):.4f}"])
        
        # 保存 CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Value'])
            writer.writerows(summary_data)
    
    def _save_cv_results_csv(self, cv_results: Dict[str, Any], filepath: str):
        """保存交叉驗證詳細結果到 CSV"""
        if 'fold_results' not in cv_results:
            return
        
        fold_results = cv_results['fold_results']
        
        # 準備數據
        rows = []
        for i, fold_result in enumerate(fold_results):
            row = {'Fold': i + 1}
            for metric, value in fold_result.items():
                if isinstance(value, (int, float)):
                    row[metric] = value
            rows.append(row)
        
        # 添加平均值行
        if 'average_results' in cv_results:
            avg_row = {'Fold': 'Mean'}
            for key, value in cv_results['average_results'].items():
                if key.endswith('_mean'):
                    metric = key.replace('_mean', '')
                    avg_row[metric] = value
            rows.append(avg_row)
        
        # 添加標準差行
        if 'std_results' in cv_results:
            std_row = {'Fold': 'Std'}
            for key, value in cv_results['std_results'].items():
                if key.endswith('_std'):
                    metric = key.replace('_std', '')
                    std_row[metric] = value
            rows.append(std_row)
        
        # 保存到 CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
    
    def _generate_text_report(self, results: Dict[str, Any], filepath: str):
        """生成文字報告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("EW-PSSK 實驗報告\n")
            f.write("="*50 + "\n")
            f.write(f"實驗時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 數據集信息
            if 'dataset_info' in results:
                f.write("數據集信息:\n")
                f.write("-"*20 + "\n")
                info = results['dataset_info']
                f.write(f"總樣本數: {info.get('total_samples', 0)}\n")
                f.write(f"正樣本數: {info.get('positive_samples', 0)}\n")
                f.write(f"負樣本數: {info.get('negative_samples', 0)}\n")
                f.write(f"正樣本比例: {info.get('positive_ratio', 0):.4f}\n")
                f.write(f"序列長度範圍: {info.get('min_length', 0)}-{info.get('max_length', 0)}\n")
                f.write(f"平均序列長度: {info.get('avg_length', 0):.2f} ± {info.get('std_length', 0):.2f}\n\n")
            
            # 模型參數
            if 'model_params' in results:
                f.write("模型參數:\n")
                f.write("-"*20 + "\n")
                params = results['model_params']
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 交叉驗證結果
            if 'cv_results' in results:
                f.write("交叉驗證結果:\n")
                f.write("-"*20 + "\n")
                cv = results['cv_results']
                f.write(f"折數: {cv.get('cv_folds', 10)}\n")
                
                if 'average_results' in cv:
                    avg = cv['average_results']
                    std = cv.get('std_results', {})
                    
                    metrics = ['Accuracy', 'Sn', 'Sp', 'F1', 'MCC', 'AUC']
                    for metric in metrics:
                        mean_key = f'{metric}_mean'
                        std_key = f'{metric}_std'
                        if mean_key in avg:
                            mean_val = avg[mean_key]
                            std_val = std.get(std_key, 0)
                            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
                f.write("\n")
            
            # 效能信息
            if 'performance' in results:
                f.write("效能信息:\n")
                f.write("-"*20 + "\n")
                perf = results['performance']
                f.write(f"總執行時間: {perf.get('total_time', 0):.4f} 秒\n")
                f.write(f"CPU 時間: {perf.get('cpu_time', 0):.4f} 秒\n")
                f.write(f"CPU 使用率: {perf.get('cpu_usage_percent', 0):.2f}%\n")
                f.write(f"記憶體峰值: {perf.get('peak_memory_mb', 0):.2f} MB\n")
                f.write(f"記憶體增加: {perf.get('memory_increase_mb', 0):.2f} MB\n")
                f.write("\n")
            
            # EW-PSSK 權重信息
            if 'weights_info' in results:
                f.write("EW-PSSK 權重信息:\n")
                f.write("-"*20 + "\n")
                weights = results['weights_info']
                f.write(f"序列長度: {weights.get('sequence_length', 0)}\n")
                f.write(f"Gamma 參數: {weights.get('gamma', 0)}\n")
                f.write(f"權重範圍: {weights.get('min_weight', 0):.4f} - {weights.get('max_weight', 0):.4f}\n")
                f.write(f"平均權重: {weights.get('mean_weight', 0):.4f} ± {weights.get('std_weight', 0):.4f}\n")
                f.write(f"熵範圍: {weights.get('min_entropy', 0):.4f} - {weights.get('max_entropy', 0):.4f}\n")
                f.write(f"平均熵: {weights.get('mean_entropy', 0):.4f}\n")
    
    def create_visualization(self, results: Dict[str, Any], 
                           experiment_name: str = "ewpssk_experiment"):
        """
        創建結果可視化圖表
        
        Args:
            results: 實驗結果
            experiment_name: 實驗名稱
        """
        exp_dir = os.path.join(self.output_dir, f"{experiment_name}_{self.timestamp}")
        
        # 1. 交叉驗證結果柱狀圖
        if 'cv_results' in results and 'average_results' in results['cv_results']:
            self._plot_cv_metrics(results['cv_results'], 
                                os.path.join(exp_dir, "cv_metrics.png"))
        
        # 2. 權重分布圖
        if 'weights_info' in results:
            # 這個需要原始權重數據，暫時跳過
            pass
        
        # 3. 效能對比圖
        if 'performance' in results:
            self._plot_performance(results['performance'], 
                                 os.path.join(exp_dir, "performance.png"))
    
    def _plot_cv_metrics(self, cv_results: Dict[str, Any], filepath: str):
        """繪製交叉驗證指標圖"""
        if 'average_results' not in cv_results:
            return
        
        avg_results = cv_results['average_results']
        std_results = cv_results.get('std_results', {})
        
        metrics = ['Accuracy', 'Sn', 'Sp', 'F1', 'MCC', 'AUC']
        values = []
        errors = []
        labels = []
        
        for metric in metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in avg_results:
                values.append(avg_results[mean_key])
                errors.append(std_results.get(std_key, 0))
                labels.append(metric)
        
        if not values:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, yerr=errors, capsize=5, 
                      color='steelblue', alpha=0.7, edgecolor='black')
        
        # plt.title('EW-PSSK 交叉驗證結果', fontsize=14, fontweight='bold')
        # plt.ylabel('分數', fontsize=12)
        # plt.xlabel('評估指標', fontsize=12)
        plt.title('EW-PSSK Cross-Validation Results', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylim(0, 1.1)
        
        # 添加數值標籤
        for bar, value, error in zip(bars, values, errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance(self, performance: Dict[str, float], filepath: str):
        """繪製效能圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 時間圖
        time_metrics = ['wall_time', 'cpu_time']
        time_values = [performance.get(metric, 0) for metric in time_metrics]
        time_labels = ['Total Time', 'CPU Time']
        
        ax1.bar(time_labels, time_values, color=['lightblue', 'lightcoral'])
        ax1.set_title('Execution Time', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')

        for i, v in enumerate(time_values):
            ax1.text(i, v + max(time_values) * 0.01, f'{v:.3f}s', 
                    ha='center', va='bottom')
        
        # 記憶體圖
        memory_metrics = ['start_memory_mb', 'peak_memory_mb', 'end_memory_mb']
        memory_values = [performance.get(metric, 0) for metric in memory_metrics]
        memory_labels = ['Start', 'Peak', 'End']
        
        ax2.bar(memory_labels, memory_values, color=['lightgreen', 'orange', 'lightblue'])
        ax2.set_title('Memory Usage', fontweight='bold')
        ax2.set_ylabel('Memory (MB)')

        for i, v in enumerate(memory_values):
            ax2.text(i, v + max(memory_values) * 0.01, f'{v:.1f}MB', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
