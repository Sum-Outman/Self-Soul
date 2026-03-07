"""
监控系统

提供系统监控、性能指标收集和资源使用跟踪功能。
"""

import time
import threading
import psutil
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = {}
        self.lock = threading.Lock()
        self.enabled = True
        
        # 默认指标
        self.register_metric("cycle_time")
        self.register_metric("memory_usage")
        self.register_metric("cpu_usage")
        self.register_metric("tensor_transfers")
        self.register_metric("cache_hit_rate")
        
        logger.info("性能监控器已初始化")
    
    def register_metric(self, name: str):
        """注册新的监控指标"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """记录指标值"""
        if not self.enabled:
            return
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(metric)
            
            # 保持最近1000个记录
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            recent_values = values[-100:] if len(values) > 100 else values
            
            return {
                'count': len(values),
                'mean': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'recent_mean': sum(recent_values) / len(recent_values) if recent_values else 0,
                'last': values[-1] if values else 0
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的统计信息"""
        result = {}
        with self.lock:
            for name in self.metrics:
                result[name] = self.get_metric_stats(name)
        return result
    
    def reset_metrics(self):
        """重置所有指标"""
        with self.lock:
            self.metrics.clear()


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, update_interval: float = 5.0):
        self.update_interval = update_interval
        self.enabled = True
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # 资源数据
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = 0.0  # 如果可用
        self.disk_usage = 0.0
        self.network_io = {"sent": 0, "received": 0}
        
        # 进程信息
        self.process = psutil.Process()
        
        logger.info("资源监控器已初始化")
    
    def start_monitoring(self):
        """开始资源监控"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            self.monitoring_thread = None
        logger.info("资源监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_event.is_set():
            try:
                self._update_resource_usage()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(self.update_interval)
    
    def _update_resource_usage(self):
        """更新资源使用情况"""
        # CPU使用率
        self.cpu_usage = psutil.cpu_percent(interval=None)
        
        # 内存使用率
        memory_info = psutil.virtual_memory()
        self.memory_usage = memory_info.percent
        
        # 磁盘使用率
        disk_info = psutil.disk_usage('/')
        self.disk_usage = disk_info.percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        self.network_io = {
            "sent": net_io.bytes_sent,
            "received": net_io.bytes_recv
        }
        
        # GPU使用率（如果可用）
        try:
            if torch.cuda.is_available():
                self.gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        except Exception:
            self.gpu_usage = 0.0
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        return {
            'cpu_usage_percent': self.cpu_usage,
            'memory_usage_percent': self.memory_usage,
            'gpu_usage_percent': self.gpu_usage,
            'disk_usage_percent': self.disk_usage,
            'network_io_bytes': self.network_io,
            'process_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'process_cpu_percent': self.process.cpu_percent(interval=None),
            'process_threads': self.process.num_threads(),
            'process_open_files': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
        }
    
    def get_resource_history(self, duration_seconds: float = 300) -> Dict[str, List[float]]:
        """获取资源使用历史（简化版）"""
        # 在实际实现中，这里会存储历史数据
        # 现在返回空历史
        return {}


class AGIMonitor:
    """AGI特定监控器"""
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 resource_monitor: ResourceMonitor):
        self.performance_monitor = performance_monitor
        self.resource_monitor = resource_monitor
        
        # AGI特定指标
        self.cognitive_cycles = 0
        self.successful_cycles = 0
        self.failed_cycles = 0
        
        # 组件性能
        self.component_performance: Dict[str, List[float]] = {}
        
        logger.info("AGI监控器已初始化")
    
    def record_cognitive_cycle(self, success: bool, cycle_time: float, 
                              component_times: Dict[str, float]):
        """记录认知循环"""
        self.cognitive_cycles += 1
        if success:
            self.successful_cycles += 1
        else:
            self.failed_cycles += 1
        
        # 记录性能指标
        self.performance_monitor.record_metric("cycle_time", cycle_time)
        
        # 记录组件性能
        for component, time_taken in component_times.items():
            if component not in self.component_performance:
                self.component_performance[component] = []
            self.component_performance[component].append(time_taken)
            
            # 保持最近100个记录
            if len(self.component_performance[component]) > 100:
                self.component_performance[component] = self.component_performance[component][-100:]
    
    def get_agi_metrics(self) -> Dict[str, Any]:
        """获取AGI特定指标"""
        success_rate = 0
        if self.cognitive_cycles > 0:
            success_rate = self.successful_cycles / self.cognitive_cycles
        
        # 组件平均性能
        component_avg = {}
        for component, times in self.component_performance.items():
            if times:
                component_avg[component] = sum(times) / len(times)
        
        return {
            'total_cycles': self.cognitive_cycles,
            'successful_cycles': self.successful_cycles,
            'failed_cycles': self.failed_cycles,
            'success_rate': success_rate,
            'component_performance': component_avg,
            'resource_usage': self.resource_monitor.get_resource_usage(),
            'performance_metrics': self.performance_monitor.get_all_metrics()
        }
    
    def reset_agi_metrics(self):
        """重置AGI指标"""
        self.cognitive_cycles = 0
        self.successful_cycles = 0
        self.failed_cycles = 0
        self.component_performance.clear()
        self.performance_monitor.reset_metrics()