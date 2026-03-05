"""
性能监控模块

提供函数性能监控、资源使用测量和性能分析功能。
"""

import time
import functools
import psutil
import os
from typing import Dict, Any, Callable, Optional, Tuple
import threading
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    execution_time: float  # 执行时间（秒）
    memory_usage_mb: float  # 内存使用（MB）
    cpu_percent: float  # CPU使用率（百分比）
    peak_memory_mb: float  # 峰值内存（MB）
    call_count: int  # 调用次数
    success: bool  # 是否成功
    timestamp: str  # 时间戳
    error_message: Optional[str] = None  # 错误信息（如果有）


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, name: str = "unknown"):
        """初始化性能监控器
        
        Args:
            name: 监控器名称
        """
        self.name = name
        self.metrics_history: list[PerformanceMetrics] = []
        self._process = psutil.Process(os.getpid())
        self._lock = threading.Lock()
    
    def measure(self, func: Callable) -> Callable:
        """测量函数性能的装饰器
        
        Args:
            func: 要测量的函数
            
        Returns:
            包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 测量前的资源使用
            memory_before = self._get_memory_usage_mb()
            cpu_before = self._get_cpu_percent()
            
            # 执行函数并测量时间
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                error_message = None
            except Exception as e:
                result = None
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                
                # 测量后的资源使用
                memory_after = self._get_memory_usage_mb()
                cpu_after = self._get_cpu_percent()
                
                # 计算指标
                execution_time = end_time - start_time
                memory_usage = memory_after - memory_before
                cpu_usage = max(0, cpu_after - cpu_before)  # 防止负值
                peak_memory = self._get_peak_memory_mb()
                
                # 记录指标
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_percent=cpu_usage,
                    peak_memory_mb=peak_memory,
                    call_count=1,
                    success=success,
                    timestamp=datetime.now().isoformat(),
                    error_message=error_message
                )
                
                with self._lock:
                    self.metrics_history.append(metrics)
            
            return result
        
        return wrapper
    
    def measure_context(self, context_name: str):
        """上下文管理器测量性能
        
        Args:
            context_name: 上下文名称
            
        Returns:
            上下文管理器
        """
        class PerformanceContext:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.memory_before = 0
                self.cpu_before = 0
                self.start_time = 0
            
            def __enter__(self):
                self.memory_before = self.monitor._get_memory_usage_mb()
                self.cpu_before = self.monitor._get_cpu_percent()
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                memory_after = self.monitor._get_memory_usage_mb()
                cpu_after = self.monitor._get_cpu_percent()
                peak_memory = self.monitor._get_peak_memory_mb()
                
                execution_time = end_time - self.start_time
                memory_usage = memory_after - self.memory_before
                cpu_usage = max(0, cpu_after - self.cpu_before)
                success = exc_type is None
                error_message = str(exc_val) if exc_val else None
                
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    memory_usage_mb=memory_usage,
                    cpu_percent=cpu_usage,
                    peak_memory_mb=peak_memory,
                    call_count=1,
                    success=success,
                    timestamp=datetime.now().isoformat(),
                    error_message=error_message
                )
                
                with self.monitor._lock:
                    self.monitor.metrics_history.append(metrics)
                
                # 不处理异常，让异常正常传播
                return False
        
        return PerformanceContext(self, context_name)
    
    def _get_memory_usage_mb(self) -> float:
        """获取当前内存使用（MB）"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # 转换为MB
        except:
            return 0.0
    
    def _get_cpu_percent(self) -> float:
        """获取当前CPU使用率"""
        try:
            return self._process.cpu_percent(interval=0.0)
        except:
            return 0.0
    
    def _get_peak_memory_mb(self) -> float:
        """获取峰值内存使用（MB）"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.vms / (1024 * 1024)  # 转换为MB
        except:
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            性能摘要字典
        """
        if not self.metrics_history:
            return {}
        
        successful_metrics = [m for m in self.metrics_history if m.success]
        if not successful_metrics:
            return {}
        
        # 计算统计信息
        execution_times = [m.execution_time for m in successful_metrics]
        memory_usages = [m.memory_usage_mb for m in successful_metrics]
        cpu_usages = [m.cpu_percent for m in successful_metrics]
        peak_memories = [m.peak_memory_mb for m in successful_metrics]
        
        import statistics
        
        return {
            "name": self.name,
            "total_calls": len(self.metrics_history),
            "successful_calls": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics_history) if self.metrics_history else 0,
            "execution_time": {
                "avg": statistics.mean(execution_times) if execution_times else 0,
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "std": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            },
            "memory_usage_mb": {
                "avg": statistics.mean(memory_usages) if memory_usages else 0,
                "min": min(memory_usages) if memory_usages else 0,
                "max": max(memory_usages) if memory_usages else 0,
            },
            "cpu_percent": {
                "avg": statistics.mean(cpu_usages) if cpu_usages else 0,
                "min": min(cpu_usages) if cpu_usages else 0,
                "max": max(cpu_usages) if cpu_usages else 0,
            },
            "peak_memory_mb": {
                "avg": statistics.mean(peak_memories) if peak_memories else 0,
                "min": min(peak_memories) if peak_memories else 0,
                "max": max(peak_memories) if peak_memories else 0,
            },
        }
    
    def reset(self) -> None:
        """重置监控器"""
        with self._lock:
            self.metrics_history.clear()
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """将性能数据转换为JSON
        
        Args:
            filepath: 可选的文件路径，如果提供则保存到文件
            
        Returns:
            JSON字符串
        """
        summary = self.get_summary()
        history = []
        for metrics in self.metrics_history:
            history.append({
                "execution_time": metrics.execution_time,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_percent": metrics.cpu_percent,
                "peak_memory_mb": metrics.peak_memory_mb,
                "success": metrics.success,
                "timestamp": metrics.timestamp,
                "error_message": metrics.error_message,
            })
        
        data = {
            "name": self.name,
            "summary": summary,
            "history": history,
            "export_timestamp": datetime.now().isoformat(),
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str


# 全局性能监控器
_global_monitors: Dict[str, PerformanceMonitor] = {}


def get_monitor(name: str = "default") -> PerformanceMonitor:
    """获取或创建性能监控器
    
    Args:
        name: 监控器名称
        
    Returns:
        性能监控器实例
    """
    if name not in _global_monitors:
        _global_monitors[name] = PerformanceMonitor(name)
    return _global_monitors[name]


def measure_performance(name: str = "default"):
    """性能测量装饰器工厂
    
    Args:
        name: 监控器名称
        
    Returns:
        装饰器函数
    """
    monitor = get_monitor(name)
    
    def decorator(func):
        return monitor.measure(func)
    
    return decorator


def print_performance_summary(name: str = "default") -> None:
    """打印性能摘要
    
    Args:
        name: 监控器名称
    """
    monitor = get_monitor(name)
    summary = monitor.get_summary()
    
    if not summary:
        print(f"监控器 '{name}' 没有性能数据")
        return
    
    print(f"\n性能摘要 - {summary['name']}:")
    print(f"  调用次数: {summary['total_calls']} (成功率: {summary['success_rate']*100:.1f}%)")
    print(f"  执行时间: {summary['execution_time']['avg']*1000:.2f}ms "
          f"(min: {summary['execution_time']['min']*1000:.2f}ms, "
          f"max: {summary['execution_time']['max']*1000:.2f}ms)")
    print(f"  内存使用: {summary['memory_usage_mb']['avg']:.2f}MB "
          f"(min: {summary['memory_usage_mb']['min']:.2f}MB, "
          f"max: {summary['memory_usage_mb']['max']:.2f}MB)")
    print(f"  CPU使用: {summary['cpu_percent']['avg']:.1f}% "
          f"(min: {summary['cpu_percent']['min']:.1f}%, "
          f"max: {summary['cpu_percent']['max']:.1f}%)")
    print(f"  峰值内存: {summary['peak_memory_mb']['avg']:.2f}MB "
          f"(min: {summary['peak_memory_mb']['min']:.2f}MB, "
          f"max: {summary['peak_memory_mb']['max']:.2f}MB)")


# 测试函数
def test_performance_monitor():
    """测试性能监控器"""
    print("测试性能监控器...")
    
    monitor = PerformanceMonitor("test_monitor")
    
    # 测试装饰器
    @monitor.measure
    def test_function(n: int):
        """测试函数"""
        result = 0
        for i in range(n):
            result += i * i
        time.sleep(0.01)  # 模拟工作
        return result
    
    # 测试多次调用
    for i in range(5):
        test_function(1000)
    
    # 测试上下文管理器
    with monitor.measure_context("test_context"):
        time.sleep(0.02)
        result = sum(i * i for i in range(1000))
    
    # 获取摘要
    summary = monitor.get_summary()
    print(f"性能摘要: {summary}")
    
    # 打印摘要
    print_performance_summary("test_monitor")
    
    # 测试JSON导出
    json_str = monitor.to_json()
    print(f"JSON数据长度: {len(json_str)} 字符")
    
    print("✅ 性能监控器测试完成!")


if __name__ == "__main__":
    test_performance_monitor()