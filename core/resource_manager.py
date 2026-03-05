#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源管理器 - Resource Manager

提供全面的系统资源监控和管理功能：
1. 实时资源监控：CPU、内存、GPU、磁盘、网络
2. 资源使用预测和趋势分析
3. 动态资源分配和调整
4. 资源限制和配额管理
5. 资源使用报告和优化建议

设计特点：
- 实时性：低延迟资源监控
- 全面性：支持多种资源类型
- 预测性：基于历史数据的资源使用预测
- 可配置：灵活的资源配置和限制
- 集成性：与演化系统紧密集成
"""

import logging
import time
import threading
import queue
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import warnings

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """资源指标数据类"""
    timestamp: float
    cpu_percent: float  # CPU使用率百分比
    memory_percent: float  # 内存使用率百分比
    memory_mb: float  # 内存使用量（MB）
    disk_usage_percent: Optional[float] = None  # 磁盘使用率百分比
    disk_io_read_mb: Optional[float] = None  # 磁盘读取（MB/s）
    disk_io_write_mb: Optional[float] = None  # 磁盘写入（MB/s）
    network_sent_mb: Optional[float] = None  # 网络发送（MB/s）
    network_recv_mb: Optional[float] = None  # 网络接收（MB/s）
    gpu_usage_percent: Optional[float] = None  # GPU使用率百分比
    gpu_memory_mb: Optional[float] = None  # GPU内存使用量（MB）
    gpu_temperature: Optional[float] = None  # GPU温度（摄氏度）
    process_count: Optional[int] = None  # 进程数量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_resource_pressure(self) -> Dict[str, float]:
        """获取资源压力指标（0-1范围）"""
        pressures = {
            "cpu": min(1.0, self.cpu_percent / 100.0),
            "memory": min(1.0, self.memory_percent / 100.0),
        }
        
        if self.disk_usage_percent is not None:
            pressures["disk"] = min(1.0, self.disk_usage_percent / 100.0)
        
        if self.gpu_usage_percent is not None:
            pressures["gpu"] = min(1.0, self.gpu_usage_percent / 100.0)
        
        return pressures
    
    def is_critical(self, thresholds: Dict[str, float]) -> bool:
        """检查是否超过临界阈值"""
        pressures = self.get_resource_pressure()
        
        for resource, threshold in thresholds.items():
            if resource in pressures and pressures[resource] > threshold:
                return True
        
        return False


@dataclass
class ResourceQuota:
    """资源配额"""
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 2000.0
    max_memory_percent: float = 85.0
    max_disk_usage_percent: float = 90.0
    max_gpu_memory_mb: Optional[float] = None
    max_gpu_usage_percent: Optional[float] = 80.0
    max_concurrent_tasks: int = 5
    max_total_tasks: int = 100
    soft_threshold: float = 0.7  # 软限制阈值（0-1）
    hard_threshold: float = 0.9  # 硬限制阈值（0-1）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ResourceAllocation:
    """资源分配"""
    task_id: str
    allocated_cpu_percent: float = 0.0
    allocated_memory_mb: float = 0.0
    allocated_gpu_memory_mb: float = 0.0
    allocated_bandwidth_mbps: float = 0.0
    priority: int = 1  # 1-10，10为最高优先级
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def get_utilization_ratio(self, current_metrics: ResourceMetrics) -> Dict[str, float]:
        """获取资源利用率比率"""
        ratios = {}
        
        if self.allocated_cpu_percent > 0:
            ratios["cpu"] = min(1.0, current_metrics.cpu_percent / self.allocated_cpu_percent)
        
        if self.allocated_memory_mb > 0:
            ratios["memory"] = min(1.0, current_metrics.memory_mb / self.allocated_memory_mb)
        
        if (self.allocated_gpu_memory_mb > 0 and 
            current_metrics.gpu_memory_mb is not None):
            ratios["gpu_memory"] = min(1.0, current_metrics.gpu_memory_mb / self.allocated_gpu_memory_mb)
        
        return ratios


class ResourcePredictor:
    """资源使用预测器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.trend_cache: Dict[str, Dict[str, float]] = {}
        self.cache_expiry = 5.0  # 缓存过期时间（秒）
        self.last_update_time: Dict[str, float] = {}
    
    def add_metric(self, metric_type: str, value: float):
        """添加指标数据"""
        self.history[metric_type].append((time.time(), value))
        self.trend_cache.pop(metric_type, None)  # 清除缓存
        self.last_update_time[metric_type] = time.time()
    
    def predict_trend(self, metric_type: str, window_minutes: int = 30) -> Dict[str, float]:
        """预测趋势"""
        cache_key = f"{metric_type}_{window_minutes}"
        
        # 检查缓存
        if (cache_key in self.trend_cache and 
            time.time() - self.last_update_time.get(metric_type, 0) < self.cache_expiry):
            return self.trend_cache[cache_key]
        
        # 获取历史数据
        history = self.history.get(metric_type, deque())
        if len(history) < 2:
            return {"current": 0.0, "trend": 0.0, "predicted": 0.0, "confidence": 0.0}
        
        # 提取最近的数据点
        window_seconds = window_minutes * 60
        recent_data = [(t, v) for t, v in history if time.time() - t <= window_seconds]
        
        if len(recent_data) < 2:
            # 使用所有数据
            recent_data = list(history)
        
        if len(recent_data) < 2:
            return {"current": 0.0, "trend": 0.0, "predicted": 0.0, "confidence": 0.0}
        
        # 计算趋势
        timestamps = [t for t, _ in recent_data]
        values = [v for _, v in recent_data]
        
        # 线性回归计算趋势
        n = len(recent_data)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(timestamps[i] * values[i] for i in range(n))
        sum_x2 = sum(t * t for t in timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 计算当前值和预测值
        current_value = values[-1]
        current_time = timestamps[-1]
        future_time = current_time + 300  # 预测5分钟后
        
        # 计算置信度（基于数据点数量和分布）
        value_std = statistics.stdev(values) if len(values) > 1 else 0.0
        avg_value = statistics.mean(values)
        confidence = max(0.0, min(1.0, 1.0 - (value_std / (avg_value + 0.001)) if avg_value > 0 else 0.0))
        
        predicted_value = current_value + slope * (future_time - current_time)
        
        # 确保预测值非负
        predicted_value = max(0.0, predicted_value)
        
        result = {
            "current": current_value,
            "trend": slope * 60,  # 每分钟的变化率
            "predicted": predicted_value,
            "confidence": confidence,
            "data_points": len(recent_data),
            "window_minutes": window_minutes
        }
        
        # 缓存结果
        self.trend_cache[cache_key] = result
        
        return result
    
    def predict_peak_usage(self, metric_type: str, lookahead_minutes: int = 60) -> Dict[str, Any]:
        """预测峰值使用"""
        trend = self.predict_trend(metric_type, window_minutes=30)
        
        if trend["data_points"] < 10:
            return {
                "peak_value": trend["current"],
                "peak_time": time.time(),
                "confidence": 0.0,
                "warning_level": "low"
            }
        
        # 基于趋势预测峰值
        current_time = time.time()
        lookahead_seconds = lookahead_minutes * 60
        
        # 简单线性预测
        predicted_peak = trend["current"] + trend["trend"] * (lookahead_minutes / 60)
        
        # 考虑历史峰值
        history_values = [v for _, v in self.history.get(metric_type, deque())]
        if history_values:
            historical_peak = max(history_values)
            # 综合历史峰值和趋势预测
            predicted_peak = max(predicted_peak, historical_peak)
        
        # 确定警告级别
        if predicted_peak > trend["current"] * 1.5:
            warning_level = "high"
        elif predicted_peak > trend["current"] * 1.2:
            warning_level = "medium"
        else:
            warning_level = "low"
        
        return {
            "peak_value": predicted_peak,
            "peak_time": current_time + lookahead_seconds,
            "confidence": trend["confidence"],
            "warning_level": warning_level,
            "trend": trend["trend"]
        }


class ResourceManager:
    """
    资源管理器主类
    提供全面的资源监控和管理功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化资源管理器
        
        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()
        
        # 资源配额
        self.quota = ResourceQuota(**self.config.get("quota", {}))
        
        # 资源分配
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # 资源指标历史
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = self.config.get("max_history_size", 10000)
        
        # 预测器
        self.predictor = ResourcePredictor(
            history_size=self.config.get("prediction_history_size", 1000)
        )
        
        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get("monitoring_interval", 2.0)
        
        # 告警阈值
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "cpu": 0.9,  # 90%使用率
            "memory": 0.9,  # 90%使用率
            "disk": 0.95,  # 95%使用率
            "gpu": 0.85,  # 85%使用率
        })
        
        # 回调函数
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.metric_callbacks: List[Callable[[ResourceMetrics], None]] = []
        
        # 统计信息
        self.statistics = {
            "total_metrics_collected": 0,
            "resource_violations": 0,
            "allocation_count": 0,
            "deallocation_count": 0,
            "peak_cpu_usage": 0.0,
            "peak_memory_usage": 0.0,
            "last_alert_time": None
        }
        
        self.logger.info("资源管理器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring_interval": 2.0,
            "max_history_size": 10000,
            "prediction_history_size": 1000,
            "quota": {
                "max_cpu_percent": 80.0,
                "max_memory_mb": 2000.0,
                "max_memory_percent": 85.0,
                "max_disk_usage_percent": 90.0,
                "max_gpu_usage_percent": 80.0,
                "max_concurrent_tasks": 5,
                "soft_threshold": 0.7,
                "hard_threshold": 0.9
            },
            "alert_thresholds": {
                "cpu": 0.9,
                "memory": 0.9,
                "disk": 0.95,
                "gpu": 0.85
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        try:
            # 获取当前资源使用情况
            usage = self.get_resource_usage()
            current_metrics = usage.get("current_metrics", {})
            
            # 获取资源压力
            pressure = usage.get("resource_pressure", {})
            
            # 构建系统状态信息
            status = {
                "status": "active" if self.monitoring_active else "inactive",
                "resource_usage": {
                    "cpu_percent": current_metrics.get("cpu_percent", 0.0),
                    "memory_mb": current_metrics.get("memory_mb", 0.0),
                    "memory_percent": current_metrics.get("memory_percent", 0.0),
                    "disk_usage_percent": current_metrics.get("disk_usage_percent", 0.0),
                    "gpu_usage_percent": current_metrics.get("gpu_usage_percent", 0.0)
                },
                "resource_pressure": {
                    "cpu": pressure.get("cpu", 0.0),
                    "memory": pressure.get("memory", 0.0),
                    "disk": pressure.get("disk", 0.0),
                    "gpu": pressure.get("gpu", 0.0)
                },
                "allocations": {
                    "active_allocations": len(self.allocations),
                    "total_allocations": self.statistics["allocation_count"],
                    "total_deallocations": self.statistics["deallocation_count"]
                },
                "statistics": self.statistics.copy(),
                "config": {
                    "monitoring_active": self.monitoring_active,
                    "monitoring_interval": self.monitoring_interval,
                    "quota": {
                        "max_cpu_percent": self.quota.max_cpu_percent,
                        "max_memory_mb": self.quota.max_memory_mb,
                        "max_concurrent_tasks": self.quota.max_concurrent_tasks
                    }
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "resource_usage": {},
                "resource_pressure": {},
                "allocations": {},
                "statistics": {},
                "config": {}
            }
    
    def start_monitoring(self):
        """开始资源监控"""
        if self.monitoring_active:
            self.logger.warning("资源监控已在运行中")
            return
        
        self.monitoring_active = True
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_thread_func,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"资源监控已启动，监控间隔: {self.monitoring_interval}秒")
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("资源监控已停止")
    
    def allocate_resources(self, task_id: str, 
                          cpu_percent: float = 0.0,
                          memory_mb: float = 0.0,
                          gpu_memory_mb: float = 0.0,
                          priority: int = 1,
                          duration_seconds: Optional[float] = None) -> bool:
        """分配资源"""
        try:
            # 检查资源是否可用
            if not self._check_resource_availability(cpu_percent, memory_mb, gpu_memory_mb):
                self.logger.warning(f"资源不足，无法分配给任务: {task_id}")
                return False
            
            # 创建资源分配
            expires_at = None
            if duration_seconds:
                expires_at = time.time() + duration_seconds
            
            allocation = ResourceAllocation(
                task_id=task_id,
                allocated_cpu_percent=cpu_percent,
                allocated_memory_mb=memory_mb,
                allocated_gpu_memory_mb=gpu_memory_mb,
                priority=priority,
                expires_at=expires_at
            )
            
            # 保存分配
            self.allocations[task_id] = allocation
            
            # 更新统计
            self.statistics["allocation_count"] += 1
            
            self.logger.info(f"资源已分配给任务: {task_id} (CPU: {cpu_percent}%, 内存: {memory_mb}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"分配资源失败: {str(e)}")
            return False
    
    def deallocate_resources(self, task_id: str) -> bool:
        """释放资源"""
        try:
            if task_id not in self.allocations:
                self.logger.warning(f"任务 {task_id} 没有分配的资源")
                return False
            
            # 移除分配
            del self.allocations[task_id]
            
            # 更新统计
            self.statistics["deallocation_count"] += 1
            
            self.logger.info(f"资源已从任务释放: {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"释放资源失败: {str(e)}")
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        try:
            # 收集系统资源指标
            metrics = self._collect_system_metrics()
            
            # 计算分配的资源使用情况
            allocation_usage = self._calculate_allocation_usage(metrics)
            
            # 计算资源压力
            resource_pressure = metrics.get_resource_pressure()
            
            # 检查配额限制
            quota_violations = self._check_quota_violations(metrics)
            
            # 获取预测
            predictions = self._get_resource_predictions()
            
            # 构建使用报告
            usage_report = {
                "timestamp": time.time(),
                "current_metrics": metrics.to_dict(),
                "resource_pressure": resource_pressure,
                "allocation_usage": allocation_usage,
                "active_allocations": len(self.allocations),
                "quota_violations": quota_violations,
                "predictions": predictions,
                "statistics": self.statistics.copy()
            }
            
            return usage_report
            
        except Exception as e:
            self.logger.error(f"获取资源使用情况失败: {str(e)}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "current_metrics": {},
                "resource_pressure": {}
            }
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """获取资源优化建议"""
        try:
            usage = self.get_resource_usage()
            metrics = usage.get("current_metrics", {})
            pressure = usage.get("resource_pressure", {})
            predictions = usage.get("predictions", {})
            
            recommendations = []
            warnings = []
            optimizations = []
            
            # CPU建议
            cpu_pressure = pressure.get("cpu", 0.0)
            if cpu_pressure > 0.8:
                warnings.append("CPU使用率过高，考虑减少并发任务或优化计算密集型操作")
            elif cpu_pressure < 0.3:
                optimizations.append("CPU使用率较低，可考虑增加并发任务以提高利用率")
            
            # 内存建议
            memory_pressure = pressure.get("memory", 0.0)
            if memory_pressure > 0.85:
                warnings.append("内存使用率过高，可能影响系统稳定性")
                recommendations.append("考虑减少内存分配或优化内存使用")
            
            # 磁盘建议
            if "disk" in pressure and pressure["disk"] > 0.9:
                warnings.append("磁盘使用率过高，可能影响I/O性能")
            
            # GPU建议
            if "gpu" in pressure and pressure["gpu"] > 0.8:
                warnings.append("GPU使用率过高，可能影响图形计算性能")
            
            # 预测建议
            cpu_prediction = predictions.get("cpu", {}).get("predicted", 0.0)
            if cpu_prediction > 0.9:
                warnings.append(f"预测CPU使用率将达到{cpu_prediction:.1%}，建议提前扩容")
            
            return {
                "timestamp": time.time(),
                "recommendations": recommendations,
                "warnings": warnings,
                "optimizations": optimizations,
                "summary": {
                    "total_recommendations": len(recommendations),
                    "total_warnings": len(warnings),
                    "total_optimizations": len(optimizations),
                    "overall_health": "healthy" if len(warnings) == 0 else "warning"
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取资源建议失败: {str(e)}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [],
                "warnings": []
            }
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable[[ResourceMetrics], None]):
        """添加指标回调函数"""
        self.metric_callbacks.append(callback)
    
    def _monitoring_thread_func(self):
        """监控线程函数"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # 收集系统指标
                metrics = self._collect_system_metrics()
                
                # 保存历史
                self._save_metrics_history(metrics)
                
                # 更新预测器
                self._update_predictor(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                # 清理过期分配
                self._cleanup_expired_allocations()
                
                # 调用指标回调
                for callback in self.metric_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        self.logger.error(f"指标回调执行失败: {str(e)}")
                
                # 计算执行时间
                execution_time = time.time() - start_time
                
                # 休眠直到下一个监控周期
                sleep_time = max(0.1, self.monitoring_interval - execution_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"监控线程异常: {str(e)}", exc_info=True)
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> ResourceMetrics:
        """收集系统资源指标"""
        try:
            import psutil
            
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 获取内存信息
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_mb = memory_info.used / (1024 * 1024)
            
            # 获取磁盘信息
            try:
                disk_info = psutil.disk_usage('/')
                disk_usage_percent = disk_info.percent
            except:
                disk_usage_percent = None
            
            # 获取磁盘IO（如果可用）
            try:
                disk_io = psutil.disk_io_counters()
                disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if hasattr(disk_io, 'read_bytes') else None
                disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if hasattr(disk_io, 'write_bytes') else None
            except:
                disk_io_read_mb = None
                disk_io_write_mb = None
            
            # 获取网络IO（如果可用）
            try:
                net_io = psutil.net_io_counters()
                network_sent_mb = net_io.bytes_sent / (1024 * 1024) if hasattr(net_io, 'bytes_sent') else None
                network_recv_mb = net_io.bytes_recv / (1024 * 1024) if hasattr(net_io, 'bytes_recv') else None
            except:
                network_sent_mb = None
                network_recv_mb = None
            
            # 获取进程数量
            try:
                process_count = len(psutil.pids())
            except:
                process_count = None
            
            # GPU信息需要额外库支持（如pynvml）
            gpu_usage_percent = None
            gpu_memory_mb = None
            gpu_temperature = None
            
            # 尝试获取GPU信息
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    # 使用第一个GPU
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU使用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage_percent = utilization.gpu
                    
                    # GPU内存
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_mb = memory_info.used / (1024 * 1024)
                    
                    # GPU温度
                    try:
                        gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        gpu_temperature = None
                
                pynvml.nvmlShutdown()
            except ImportError:
                # pynvml不可用
                pass
            except Exception as e:
                self.logger.debug(f"获取GPU信息失败: {str(e)}")
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_usage_percent=disk_usage_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                gpu_usage_percent=gpu_usage_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_temperature=gpu_temperature,
                process_count=process_count
            )
            
        except ImportError:
            # psutil不可用，返回基本指标
            self.logger.warning("psutil不可用，使用基本资源指标")
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0
            )
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {str(e)}")
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0
            )
    
    def _save_metrics_history(self, metrics: ResourceMetrics):
        """保存指标历史"""
        self.metrics_history.append(metrics)
        self.statistics["total_metrics_collected"] += 1
        
        # 更新峰值统计
        if metrics.cpu_percent > self.statistics["peak_cpu_usage"]:
            self.statistics["peak_cpu_usage"] = metrics.cpu_percent
        
        if metrics.memory_mb > self.statistics["peak_memory_usage"]:
            self.statistics["peak_memory_usage"] = metrics.memory_mb
        
        # 限制历史记录大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def _update_predictor(self, metrics: ResourceMetrics):
        """更新预测器"""
        try:
            # 更新CPU预测
            self.predictor.add_metric("cpu_percent", metrics.cpu_percent)
            
            # 更新内存预测
            self.predictor.add_metric("memory_mb", metrics.memory_mb)
            self.predictor.add_metric("memory_percent", metrics.memory_percent)
            
            # 更新GPU预测（如果可用）
            if metrics.gpu_usage_percent is not None:
                self.predictor.add_metric("gpu_usage_percent", metrics.gpu_usage_percent)
            
            if metrics.gpu_memory_mb is not None:
                self.predictor.add_metric("gpu_memory_mb", metrics.gpu_memory_mb)
                
        except Exception as e:
            self.logger.error(f"更新预测器失败: {str(e)}")
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """检查告警"""
        try:
            # 检查资源压力
            pressures = metrics.get_resource_pressure()
            
            for resource, pressure in pressures.items():
                if resource in self.alert_thresholds and pressure > self.alert_thresholds[resource]:
                    # 触发告警
                    alert = {
                        "alert_id": f"resource_{resource}_{int(time.time())}",
                        "resource": resource,
                        "pressure": pressure,
                        "threshold": self.alert_thresholds[resource],
                        "timestamp": time.time(),
                        "severity": "warning" if pressure > 0.95 else "info",
                        "message": f"资源 {resource} 使用率过高: {pressure:.1%} (阈值: {self.alert_thresholds[resource]:.1%})"
                    }
                    
                    # 调用告警回调
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.error(f"告警回调执行失败: {str(e)}")
                    
                    self.statistics["resource_violations"] += 1
                    self.statistics["last_alert_time"] = time.time()
                    
                    self.logger.warning(alert["message"])
            
        except Exception as e:
            self.logger.error(f"检查告警失败: {str(e)}")
    
    def _cleanup_expired_allocations(self):
        """清理过期分配"""
        expired_tasks = []
        
        for task_id, allocation in list(self.allocations.items()):
            if allocation.is_expired():
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            self.deallocate_resources(task_id)
            self.logger.info(f"任务 {task_id} 的资源分配已过期并释放")
    
    def _check_resource_availability(self, cpu_percent: float, memory_mb: float, 
                                   gpu_memory_mb: float) -> bool:
        """检查资源是否可用"""
        try:
            # 获取当前资源使用情况
            current_metrics = self._collect_system_metrics()
            
            # 检查CPU
            if cpu_percent > 0:
                available_cpu = 100.0 - current_metrics.cpu_percent
                if cpu_percent > available_cpu * 0.9:  # 保留10%缓冲
                    self.logger.debug(f"CPU不足: 需要{cpu_percent}%，可用{available_cpu}%")
                    return False
            
            # 检查内存
            if memory_mb > 0:
                available_memory = self.quota.max_memory_mb - current_metrics.memory_mb
                if memory_mb > available_memory * 0.9:  # 保留10%缓冲
                    self.logger.debug(f"内存不足: 需要{memory_mb}MB，可用{available_memory}MB")
                    return False
            
            # 检查GPU内存（如果支持）
            if gpu_memory_mb > 0 and current_metrics.gpu_memory_mb is not None:
                if self.quota.max_gpu_memory_mb:
                    available_gpu_memory = self.quota.max_gpu_memory_mb - current_metrics.gpu_memory_mb
                    if gpu_memory_mb > available_gpu_memory * 0.9:
                        self.logger.debug(f"GPU内存不足: 需要{gpu_memory_mb}MB，可用{available_gpu_memory}MB")
                        return False
            
            # 检查并发任务限制
            if len(self.allocations) >= self.quota.max_concurrent_tasks:
                self.logger.debug(f"并发任务数已达上限: {len(self.allocations)}/{self.quota.max_concurrent_tasks}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查资源可用性失败: {str(e)}")
            return False
    
    def _calculate_allocation_usage(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """计算分配的资源使用情况"""
        total_cpu_allocated = 0.0
        total_memory_allocated = 0.0
        total_gpu_memory_allocated = 0.0
        
        active_allocations = []
        
        for task_id, allocation in self.allocations.items():
            if allocation.is_active and not allocation.is_expired():
                total_cpu_allocated += allocation.allocated_cpu_percent
                total_memory_allocated += allocation.allocated_memory_mb
                total_gpu_memory_allocated += allocation.allocated_gpu_memory_mb
                
                active_allocations.append({
                    "task_id": task_id,
                    "allocation": allocation.to_dict(),
                    "utilization": allocation.get_utilization_ratio(metrics)
                })
        
        return {
            "total_cpu_allocated_percent": total_cpu_allocated,
            "total_memory_allocated_mb": total_memory_allocated,
            "total_gpu_memory_allocated_mb": total_gpu_memory_allocated,
            "active_allocations_count": len(active_allocations),
            "active_allocations": active_allocations,
            "allocation_efficiency": {
                "cpu": total_cpu_allocated / metrics.cpu_percent if metrics.cpu_percent > 0 else 0.0,
                "memory": total_memory_allocated / metrics.memory_mb if metrics.memory_mb > 0 else 0.0
            }
        }
    
    def _check_quota_violations(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """检查配额违规"""
        violations = []
        
        # 检查CPU配额
        if metrics.cpu_percent > self.quota.max_cpu_percent:
            violations.append({
                "resource": "cpu",
                "current": metrics.cpu_percent,
                "limit": self.quota.max_cpu_percent,
                "excess": metrics.cpu_percent - self.quota.max_cpu_percent
            })
        
        # 检查内存配额
        if metrics.memory_percent > self.quota.max_memory_percent:
            violations.append({
                "resource": "memory_percent",
                "current": metrics.memory_percent,
                "limit": self.quota.max_memory_percent,
                "excess": metrics.memory_percent - self.quota.max_memory_percent
            })
        
        if metrics.memory_mb > self.quota.max_memory_mb:
            violations.append({
                "resource": "memory_mb",
                "current": metrics.memory_mb,
                "limit": self.quota.max_memory_mb,
                "excess": metrics.memory_mb - self.quota.max_memory_mb
            })
        
        # 检查磁盘配额
        if (metrics.disk_usage_percent is not None and 
            metrics.disk_usage_percent > self.quota.max_disk_usage_percent):
            violations.append({
                "resource": "disk",
                "current": metrics.disk_usage_percent,
                "limit": self.quota.max_disk_usage_percent,
                "excess": metrics.disk_usage_percent - self.quota.max_disk_usage_percent
            })
        
        # 检查GPU配额
        if (metrics.gpu_usage_percent is not None and 
            self.quota.max_gpu_usage_percent is not None and
            metrics.gpu_usage_percent > self.quota.max_gpu_usage_percent):
            violations.append({
                "resource": "gpu_usage",
                "current": metrics.gpu_usage_percent,
                "limit": self.quota.max_gpu_usage_percent,
                "excess": metrics.gpu_usage_percent - self.quota.max_gpu_usage_percent
            })
        
        return violations
    
    def _get_resource_predictions(self) -> Dict[str, Dict[str, Any]]:
        """获取资源预测"""
        predictions = {}
        
        try:
            # CPU预测
            cpu_prediction = self.predictor.predict_trend("cpu_percent", window_minutes=30)
            cpu_peak = self.predictor.predict_peak_usage("cpu_percent", lookahead_minutes=60)
            predictions["cpu"] = {**cpu_prediction, "peak": cpu_peak}
            
            # 内存预测
            memory_prediction = self.predictor.predict_trend("memory_mb", window_minutes=30)
            memory_peak = self.predictor.predict_peak_usage("memory_mb", lookahead_minutes=60)
            predictions["memory"] = {**memory_prediction, "peak": memory_peak}
            
            # GPU预测（如果可用）
            gpu_prediction = self.predictor.predict_trend("gpu_usage_percent", window_minutes=30)
            if gpu_prediction["data_points"] > 0:
                gpu_peak = self.predictor.predict_peak_usage("gpu_usage_percent", lookahead_minutes=60)
                predictions["gpu"] = {**gpu_prediction, "peak": gpu_peak}
            
        except Exception as e:
            self.logger.error(f"获取资源预测失败: {str(e)}")
        
        return predictions


# 全局资源管理器实例
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(config: Optional[Dict[str, Any]] = None) -> ResourceManager:
    """
    获取全局资源管理器实例
    
    Args:
        config: 配置字典
        
    Returns:
        资源管理器实例
    """
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(config)
    
    return _global_resource_manager


def start_resource_monitoring(config: Optional[Dict[str, Any]] = None):
    """启动全局资源监控"""
    manager = get_resource_manager(config)
    manager.start_monitoring()
    return manager


def stop_resource_monitoring():
    """停止全局资源监控"""
    global _global_resource_manager
    if _global_resource_manager:
        _global_resource_manager.stop_monitoring()


# 测试代码
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("资源管理器测试")
    print("=" * 80)
    
    try:
        # 创建资源管理器
        manager = ResourceManager()
        
        print("\n1. 测试资源管理器初始化:")
        print(f"   配置: 监控间隔={manager.config['monitoring_interval']}s, 最大历史记录={manager.config['max_history_size']}")
        print(f"   配额: CPU={manager.quota.max_cpu_percent}%, 内存={manager.quota.max_memory_mb}MB")
        
        print("\n2. 测试资源分配:")
        # 分配资源给测试任务
        allocation_success = manager.allocate_resources(
            task_id="test_task_1",
            cpu_percent=20.0,
            memory_mb=500.0,
            priority=5
        )
        print(f"   资源分配结果: {'成功' if allocation_success else '失败'}")
        
        print("\n3. 测试资源使用情况获取:")
        usage = manager.get_resource_usage()
        print(f"   当前CPU使用率: {usage.get('current_metrics', {}).get('cpu_percent', 0.0):.1f}%")
        print(f"   当前内存使用: {usage.get('current_metrics', {}).get('memory_mb', 0.0):.1f}MB")
        print(f"   资源压力: CPU={usage.get('resource_pressure', {}).get('cpu', 0.0):.2f}")
        
        print("\n4. 测试资源建议:")
        recommendations = manager.get_resource_recommendations()
        print(f"   建议数量: {len(recommendations.get('recommendations', []))}")
        print(f"   警告数量: {len(recommendations.get('warnings', []))}")
        
        print("\n5. 测试资源释放:")
        deallocation_success = manager.deallocate_resources("test_task_1")
        print(f"   资源释放结果: {'成功' if deallocation_success else '失败'}")
        
        print("\n✓ 资源管理器测试完成")
        print("\n说明:")
        print("  1. 资源管理器提供全面的系统资源监控和管理")
        print("  2. 支持CPU、内存、GPU、磁盘、网络等多种资源")
        print("  3. 提供资源分配、使用预测和优化建议")
        print("  4. 可集成到演化系统中进行智能资源调度")
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)