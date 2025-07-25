"""
系統效能監控模塊
監控 CPU 時間和記憶體使用情況
"""

import time
import psutil
import os
from typing import Dict, Optional
import threading
from contextlib import contextmanager


class PerformanceMonitor:
    """系統效能監控器"""
    
    def __init__(self):
        """初始化監控器"""
        self.process = psutil.Process(os.getpid())
        self.reset()
    
    def reset(self):
        """重置監控狀態"""
        self.start_time = None
        self.end_time = None
        self.start_cpu_time = None
        self.end_cpu_time = None
        self.peak_memory = 0
        self.start_memory = 0
        self.monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = False
    
    def start_monitoring(self):
        """開始監控"""
        self.start_time = time.time()
        self.start_cpu_time = self.process.cpu_times()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.monitoring = True
        self._stop_monitoring = False
        
        # 啟動記憶體監控線程
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        print(f"開始效能監控 - 初始記憶體: {self.start_memory:.2f} MB")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        停止監控並返回結果
        
        Returns:
            效能統計字典
        """
        if not self.monitoring:
            return {}
        
        self.end_time = time.time()
        self.end_cpu_time = self.process.cpu_times()
        self.monitoring = False
        self._stop_monitoring = True
        
        # 等待監控線程結束
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        # 計算統計結果
        wall_time = self.end_time - self.start_time
        cpu_time = ((self.end_cpu_time.user - self.start_cpu_time.user) + 
                   (self.end_cpu_time.system - self.start_cpu_time.system))
        
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.start_memory
        
        stats = {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_usage_percent': (cpu_time / wall_time) * 100 if wall_time > 0 else 0,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': memory_increase,
            'peak_memory_increase_mb': self.peak_memory - self.start_memory
        }
        
        print(f"效能監控結束:")
        print(f"  總時間: {wall_time:.4f} 秒")
        print(f"  CPU 時間: {cpu_time:.4f} 秒")
        print(f"  CPU 使用率: {stats['cpu_usage_percent']:.2f}%")
        print(f"  記憶體峰值: {self.peak_memory:.2f} MB")
        print(f"  記憶體增加: {memory_increase:.2f} MB")
        
        return stats
    
    def _monitor_memory(self):
        """記憶體監控線程"""
        while not self._stop_monitoring:
            try:
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                time.sleep(0.1)  # 每 100ms 檢查一次
            except:
                break
    
    def get_current_memory(self) -> float:
        """
        獲取當前記憶體使用量
        
        Returns:
            記憶體使用量 (MB)
        """
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_system_info(self) -> Dict[str, any]:
        """
        獲取系統信息
        
        Returns:
            系統信息字典
        """
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'cpu_percent': psutil.cpu_percent(interval=1)
        }
        
        memory_info = psutil.virtual_memory()._asdict()
        memory_info = {k: v / 1024 / 1024 for k, v in memory_info.items() 
                      if isinstance(v, (int, float))}  # 轉換為 MB
        
        return {
            'cpu_info': cpu_info,
            'memory_info': memory_info,
            'python_process_id': os.getpid()
        }


@contextmanager
def monitor_performance():
    """
    效能監控上下文管理器
    
    使用方式:
    with monitor_performance() as monitor:
        # 執行需要監控的代碼
        pass
    stats = monitor.get_stats()
    """
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        stats = monitor.stop_monitoring()
        monitor._stats = stats


class BenchmarkTimer:
    """基準測試計時器"""
    
    def __init__(self, name: str = ""):
        """
        初始化計時器
        
        Args:
            name: 計時器名稱
        """
        self.name = name
        self.times = []
        self.current_start = None
    
    def start(self):
        """開始計時"""
        self.current_start = time.time()
    
    def stop(self) -> float:
        """
        停止計時並記錄
        
        Returns:
            本次計時時間
        """
        if self.current_start is None:
            raise ValueError("計時器尚未啟動")
        
        elapsed = time.time() - self.current_start
        self.times.append(elapsed)
        self.current_start = None
        
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """
        獲取計時統計
        
        Returns:
            統計字典
        """
        if not self.times:
            return {}
        
        import numpy as np
        
        return {
            'total_time': sum(self.times),
            'mean_time': np.mean(self.times),
            'std_time': np.std(self.times),
            'min_time': min(self.times),
            'max_time': max(self.times),
            'count': len(self.times)
        }
    
    @contextmanager
    def time_it(self):
        """
        計時上下文管理器
        
        使用方式:
        timer = BenchmarkTimer("測試")
        with timer.time_it():
            # 執行需要計時的代碼
            pass
        """
        self.start()
        try:
            yield
        finally:
            self.stop()


def format_performance_report(stats: Dict[str, float], 
                            title: str = "效能報告") -> str:
    """
    格式化效能報告
    
    Args:
        stats: 效能統計字典
        title: 報告標題
        
    Returns:
        格式化的報告字串
    """
    report = f"\n{'='*50}\n{title:^50}\n{'='*50}\n"
    
    if 'wall_time' in stats:
        report += f"執行時間:\n"
        report += f"  總時間: {stats['wall_time']:.4f} 秒\n"
        report += f"  CPU 時間: {stats.get('cpu_time', 0):.4f} 秒\n"
        report += f"  CPU 使用率: {stats.get('cpu_usage_percent', 0):.2f}%\n\n"
    
    if 'peak_memory_mb' in stats:
        report += f"記憶體使用:\n"
        report += f"  起始記憶體: {stats.get('start_memory_mb', 0):.2f} MB\n"
        report += f"  結束記憶體: {stats.get('end_memory_mb', 0):.2f} MB\n"
        report += f"  記憶體峰值: {stats['peak_memory_mb']:.2f} MB\n"
        report += f"  記憶體增加: {stats.get('memory_increase_mb', 0):.2f} MB\n"
    
    report += "="*50
    
    return report
