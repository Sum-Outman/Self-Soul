#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性能监控仪表板 - 提供实时性能监控、可视化、预警和优化建议

核心功能：
1. 实时性能监控 - CPU、内存、GPU、磁盘IO、网络等
2. 多维度可视化 - 时间序列、热力图、分布图等
3. 智能预警系统 - 基于阈值和趋势的预警
4. 优化建议生成 - 基于性能数据的智能建议
5. 资源使用预测 - 基于历史数据的资源需求预测

设计目标：
- 提供全面的系统性能监控
- 实现智能预警和主动优化
- 支持多种可视化格式
- 易于集成到现有系统
"""

import sys
import os
import time
import threading
import json
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd

# 尝试导入性能监控库
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil不可用，某些监控功能将受限")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """警报级别"""
    INFO = "info"        # 信息级别
    WARNING = "warning"  # 警告级别
    CRITICAL = "critical"  # 严重级别


class MetricType(Enum):
    """指标类型"""
    CPU = "cpu"                # CPU使用率
    MEMORY = "memory"          # 内存使用
    GPU = "gpu"                # GPU使用率
    DISK = "disk"              # 磁盘使用
    NETWORK = "network"        # 网络IO
    PROCESS = "process"        # 进程信息
    TASK = "task"              # 任务执行
    INFERENCE = "inference"    # 推理性能
    CACHE = "cache"            # 缓存效率


@dataclass
class PerformanceAlert:
    """性能警报"""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    timestamp: datetime
    value: float
    threshold: Optional[float] = None
    trend: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SystemMetric:
    """系统指标"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitoringDashboard:
    """系统性能监控仪表板"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化性能监控仪表板"""
        self.config = config or self._get_default_config()
        
        # 监控数据存储
        self.metrics_history: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=self.config.get("history_size", 1000))
            for metric_type in MetricType
        }
        
        # 推理延迟跟踪
        self.inference_latencies: List[float] = []
        self.inference_history_size = self.config.get("inference_tracking", {}).get("history_size", 100)
        self.inference_start_times: Dict[str, Dict[str, Any]] = {}  # 改为存储更多信息
        
        # 按模型类型的推理统计
        self.inference_by_model: Dict[str, Dict[str, Any]] = {}
        self.inference_success_count = 0
        self.inference_failure_count = 0
        
        # 任务吞吐量跟踪
        self.task_completion_times: List[float] = []
        self.task_history_size = self.config.get("task_tracking", {}).get("history_size", 100)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        
        self.alerts: List[PerformanceAlert] = []
        self.optimization_suggestions: List[Dict[str, Any]] = []
        
        # 监控配置
        self.monitor_interval = self.config.get("monitor_interval", 5.0)
        self.monitor_thread = None
        self.monitor_running = False
        
        # 警报阈值
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "cpu": {"warning": 80.0, "critical": 95.0},
            "memory": {"warning": 85.0, "critical": 95.0},
            "gpu": {"warning": 85.0, "critical": 95.0},
            "disk": {"warning": 90.0, "critical": 98.0}
        })
        
        # 趋势分析窗口
        self.trend_window = self.config.get("trend_window", 10)
        
        # 回调函数
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.metric_callbacks: List[Callable[[SystemMetric], None]] = []
        
        # 输出目录
        self.output_dir = self.config.get("output_dir", "performance_dashboard")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"系统性能监控仪表板初始化完成，监控间隔: {self.monitor_interval}秒")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitor_interval": 5.0,
            "history_size": 1000,
            "trend_window": 10,
            "output_dir": "performance_dashboard",
            "enable_cpu_monitoring": True,
            "enable_memory_monitoring": True,
            "enable_gpu_monitoring": GPU_AVAILABLE,
            "enable_disk_monitoring": True,
            "enable_network_monitoring": True,
            "enable_process_monitoring": True,
            "enable_inference_monitoring": True,
            "enable_task_monitoring": True,
            "alert_thresholds": {
                "cpu": {"warning": 80.0, "critical": 95.0},
                "memory": {"warning": 85.0, "critical": 95.0},
                "gpu": {"warning": 85.0, "critical": 95.0},
                "disk": {"warning": 90.0, "critical": 98.0},
                "inference_latency": {"warning": 5000.0, "critical": 10000.0},  # 毫秒
                "task_throughput": {"warning": 10.0, "critical": 5.0}  # 任务/秒，越低越差
            },
            "inference_tracking": {
                "enable_auto_tracking": True,
                "history_size": 100,
                "calculate_percentiles": [50, 90, 95, 99]
            },
            "task_tracking": {
                "enable_auto_tracking": True,
                "history_size": 100,
                "track_completed_tasks": True,
                "track_failed_tasks": True
            },
            "visualization": {
                "use_plotly": PLOTLY_AVAILABLE,
                "use_matplotlib": MATPLOTLIB_AVAILABLE,
                "auto_generate_reports": True,
                "report_interval": 300  # 5分钟生成一次报告
            }
        }
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitor_running:
            logger.warning("性能监控已经在运行")
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitor_thread.start()
        
        logger.info("系统性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        
        logger.info("系统性能监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        last_report_time = time.time()
        report_interval = self.config.get("visualization", {}).get("report_interval", 300)
        
        while self.monitor_running:
            try:
                # 收集各项指标
                self._collect_system_metrics()
                
                # 检查警报条件
                self._check_alert_conditions()
                
                # 生成优化建议
                self._generate_optimization_suggestions()
                
                # 定期生成报告
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    self._generate_performance_report()
                    last_report_time = current_time
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            # 等待下一次监控
            time.sleep(self.monitor_interval)
        
        # 监控停止时生成最终报告
        self._generate_performance_report()
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = datetime.now()
        
        # CPU使用率
        if self.config.get("enable_cpu_monitoring", True) and PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._add_metric(
                SystemMetric(
                    timestamp=timestamp,
                    metric_type=MetricType.CPU,
                    value=cpu_percent,
                    unit="percent",
                    metadata={"cores": psutil.cpu_count()}
                )
            )
        
        # 内存使用
        if self.config.get("enable_memory_monitoring", True) and PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            self._add_metric(
                SystemMetric(
                    timestamp=timestamp,
                    metric_type=MetricType.MEMORY,
                    value=memory_info.percent,
                    unit="percent",
                    metadata={
                        "total_gb": memory_info.total / (1024**3),
                        "available_gb": memory_info.available / (1024**3),
                        "used_gb": memory_info.used / (1024**3)
                    }
                )
            )
        
        # GPU使用率
        if self.config.get("enable_gpu_monitoring", True) and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self._add_metric(
                        SystemMetric(
                            timestamp=timestamp,
                            metric_type=MetricType.GPU,
                            value=gpu.load * 100,
                            unit="percent",
                            metadata={
                                "gpu_index": i,
                                "name": gpu.name,
                                "memory_total": gpu.memoryTotal,
                                "memory_used": gpu.memoryUsed,
                                "temperature": gpu.temperature
                            }
                        )
                    )
            except Exception as e:
                logger.warning(f"GPU监控失败: {e}")
        
        # 磁盘使用
        if self.config.get("enable_disk_monitoring", True) and PSUTIL_AVAILABLE:
            try:
                disk_usage = psutil.disk_usage('/')
                self._add_metric(
                    SystemMetric(
                        timestamp=timestamp,
                        metric_type=MetricType.DISK,
                        value=disk_usage.percent,
                        unit="percent",
                        metadata={
                            "total_gb": disk_usage.total / (1024**3),
                            "used_gb": disk_usage.used / (1024**3),
                            "free_gb": disk_usage.free / (1024**3)
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"磁盘监控失败: {e}")
        
        # 网络IO
        if self.config.get("enable_network_monitoring", True) and PSUTIL_AVAILABLE:
            try:
                net_io = psutil.net_io_counters()
                self._add_metric(
                    SystemMetric(
                        timestamp=timestamp,
                        metric_type=MetricType.NETWORK,
                        value=net_io.bytes_sent + net_io.bytes_recv,
                        unit="bytes",
                        metadata={
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "packets_sent": net_io.packets_sent,
                            "packets_recv": net_io.packets_recv
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"网络监控失败: {e}")
        
        # 收集推理延迟指标
        if self.config.get("enable_inference_monitoring", True):
            self._collect_inference_metrics(timestamp)
        
        # 收集任务吞吐量指标
        if self.config.get("enable_task_monitoring", True):
            self._collect_task_metrics(timestamp)
    
    def _collect_inference_metrics(self, timestamp: datetime):
        """收集推理延迟指标"""
        if not self.inference_latencies:
            return
        
        # 计算平均推理延迟
        avg_latency = np.mean(self.inference_latencies) if self.inference_latencies else 0
        
        # 计算百分位数
        percentiles = {}
        if self.inference_latencies:
            for p in self.config.get("inference_tracking", {}).get("calculate_percentiles", [50, 90, 95, 99]):
                percentiles[f"p{p}"] = np.percentile(self.inference_latencies, p)
        
        # 计算成功率
        total_inferences = self.inference_success_count + self.inference_failure_count
        success_rate = 0
        if total_inferences > 0:
            success_rate = self.inference_success_count / total_inferences * 100
        
        # 按模型类型的统计
        model_statistics = {}
        for model_type, stats in self.inference_by_model.items():
            if stats["total_count"] > 0:
                model_success_rate = stats["success_count"] / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
                model_avg_latency = stats["total_latency"] / stats["total_count"] if stats["total_count"] > 0 else 0
                
                model_statistics[model_type] = {
                    "total_count": stats["total_count"],
                    "success_rate": model_success_rate,
                    "avg_latency": model_avg_latency,
                    "min_latency": stats["min_latency"] if stats["min_latency"] != float('inf') else 0,
                    "max_latency": stats["max_latency"]
                }
        
        self._add_metric(
            SystemMetric(
                timestamp=timestamp,
                metric_type=MetricType.INFERENCE,
                value=avg_latency,
                unit="milliseconds",
                metadata={
                    "sample_count": len(self.inference_latencies),
                    "min_latency": min(self.inference_latencies) if self.inference_latencies else 0,
                    "max_latency": max(self.inference_latencies) if self.inference_latencies else 0,
                    "percentiles": percentiles,
                    "success_rate": success_rate,
                    "total_inferences": total_inferences,
                    "success_count": self.inference_success_count,
                    "failure_count": self.inference_failure_count,
                    "model_statistics": model_statistics
                }
            )
        )
    
    def _collect_task_metrics(self, timestamp: datetime):
        """收集任务吞吐量指标"""
        # 计算任务吞吐量（任务/秒）
        throughput = 0
        if self.task_completion_times:
            # 基于最近完成的任务计算吞吐量
            recent_tasks = self.task_completion_times[-min(10, len(self.task_completion_times)):]
            if len(recent_tasks) >= 2:
                time_window = recent_tasks[-1] - recent_tasks[0]
                if time_window > 0:
                    throughput = len(recent_tasks) / time_window
        
        # 计算任务成功率
        success_rate = 0
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            success_rate = len(self.completed_tasks) / total_tasks * 100
        
        self._add_metric(
            SystemMetric(
                timestamp=timestamp,
                metric_type=MetricType.TASK,
                value=throughput,
                unit="tasks_per_second",
                metadata={
                    "active_tasks": len(self.active_tasks),
                    "completed_tasks": len(self.completed_tasks),
                    "failed_tasks": len(self.failed_tasks),
                    "success_rate": success_rate,
                    "total_tasks": total_tasks
                }
            )
        )
    
    # 推理跟踪方法
    def start_inference_tracking(self, inference_id: str, model_type: str = "unknown", metadata: Optional[Dict[str, Any]] = None):
        """开始跟踪推理延迟
        
        Args:
            inference_id: 推理ID
            model_type: 模型类型 (如: "resnet50", "bert", "gpt2")
            metadata: 附加元数据
            
        Returns:
            推理ID或None
        """
        if self.config.get("enable_inference_monitoring", True):
            self.inference_start_times[inference_id] = {
                "start_time": time.time(),
                "model_type": model_type,
                "metadata": metadata or {}
            }
            return inference_id
        return None
    
    def stop_inference_tracking(self, inference_id: str, success: bool = True, result: Optional[Any] = None):
        """停止跟踪推理延迟
        
        Args:
            inference_id: 推理ID
            success: 推理是否成功
            result: 推理结果
            
        Returns:
            推理延迟(毫秒)或None
        """
        if inference_id in self.inference_start_times:
            start_info = self.inference_start_times.pop(inference_id)
            start_time = start_info["start_time"]
            model_type = start_info.get("model_type", "unknown")
            metadata = start_info.get("metadata", {})
            
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 添加到延迟列表
            self.inference_latencies.append(latency)
            
            # 保持列表大小
            if len(self.inference_latencies) > self.inference_history_size:
                self.inference_latencies = self.inference_latencies[-self.inference_history_size:]
            
            # 更新成功率统计
            if success:
                self.inference_success_count += 1
            else:
                self.inference_failure_count += 1
            
            # 按模型类型统计
            if model_type not in self.inference_by_model:
                self.inference_by_model[model_type] = {
                    "total_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "latencies": [],
                    "total_latency": 0.0,
                    "min_latency": float('inf'),
                    "max_latency": 0.0
                }
            
            model_stats = self.inference_by_model[model_type]
            model_stats["total_count"] += 1
            if success:
                model_stats["success_count"] += 1
            else:
                model_stats["failure_count"] += 1
            
            model_stats["latencies"].append(latency)
            model_stats["total_latency"] += latency
            model_stats["min_latency"] = min(model_stats["min_latency"], latency)
            model_stats["max_latency"] = max(model_stats["max_latency"], latency)
            
            # 保持模型统计的延迟列表大小
            if len(model_stats["latencies"]) > self.inference_history_size:
                model_stats["latencies"] = model_stats["latencies"][-self.inference_history_size:]
            
            # 记录推理结果
            inference_record = {
                "inference_id": inference_id,
                "model_type": model_type,
                "timestamp": time.time(),
                "latency": latency,
                "success": success,
                "result": result,
                "metadata": metadata
            }
            
            return latency
        return None
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息
        
        Returns:
            包含推理统计信息的字典
        """
        total_inferences = self.inference_success_count + self.inference_failure_count
        success_rate = 0
        if total_inferences > 0:
            success_rate = self.inference_success_count / total_inferences * 100
        
        # 按模型类型的详细统计
        model_details = {}
        for model_type, stats in self.inference_by_model.items():
            if stats["total_count"] > 0:
                model_success_rate = stats["success_count"] / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
                model_avg_latency = stats["total_latency"] / stats["total_count"] if stats["total_count"] > 0 else 0
                
                # 计算模型延迟百分位数
                model_percentiles = {}
                if stats["latencies"]:
                    for p in self.config.get("inference_tracking", {}).get("calculate_percentiles", [50, 90, 95, 99]):
                        model_percentiles[f"p{p}"] = np.percentile(stats["latencies"], p)
                
                model_details[model_type] = {
                    "total_count": stats["total_count"],
                    "success_count": stats["success_count"],
                    "failure_count": stats["failure_count"],
                    "success_rate": model_success_rate,
                    "avg_latency": model_avg_latency,
                    "min_latency": stats["min_latency"] if stats["min_latency"] != float('inf') else 0,
                    "max_latency": stats["max_latency"],
                    "percentiles": model_percentiles,
                    "recent_latencies": stats["latencies"][-10:] if len(stats["latencies"]) > 10 else stats["latencies"]
                }
        
        # 总体统计
        overall_percentiles = {}
        if self.inference_latencies:
            for p in self.config.get("inference_tracking", {}).get("calculate_percentiles", [50, 90, 95, 99]):
                overall_percentiles[f"p{p}"] = np.percentile(self.inference_latencies, p)
        
        return {
            "overall": {
                "total_inferences": total_inferences,
                "success_count": self.inference_success_count,
                "failure_count": self.inference_failure_count,
                "success_rate": success_rate,
                "avg_latency": np.mean(self.inference_latencies) if self.inference_latencies else 0,
                "min_latency": min(self.inference_latencies) if self.inference_latencies else 0,
                "max_latency": max(self.inference_latencies) if self.inference_latencies else 0,
                "percentiles": overall_percentiles,
                "sample_count": len(self.inference_latencies),
                "active_trackings": len(self.inference_start_times)
            },
            "by_model": model_details,
            "recent_latencies": self.inference_latencies[-20:] if len(self.inference_latencies) > 20 else self.inference_latencies
        }
    
    def get_model_inference_statistics(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取特定模型类型的推理统计信息
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型统计信息或None
        """
        if model_type not in self.inference_by_model:
            return None
        
        stats = self.inference_by_model[model_type]
        if stats["total_count"] == 0:
            return None
        
        model_success_rate = stats["success_count"] / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
        model_avg_latency = stats["total_latency"] / stats["total_count"] if stats["total_count"] > 0 else 0
        
        # 计算百分位数
        percentiles = {}
        if stats["latencies"]:
            for p in self.config.get("inference_tracking", {}).get("calculate_percentiles", [50, 90, 95, 99]):
                percentiles[f"p{p}"] = np.percentile(stats["latencies"], p)
        
        return {
            "model_type": model_type,
            "total_count": stats["total_count"],
            "success_count": stats["success_count"],
            "failure_count": stats["failure_count"],
            "success_rate": model_success_rate,
            "avg_latency": model_avg_latency,
            "min_latency": stats["min_latency"] if stats["min_latency"] != float('inf') else 0,
            "max_latency": stats["max_latency"],
            "percentiles": percentiles,
            "recent_latencies": stats["latencies"][-10:] if len(stats["latencies"]) > 10 else stats["latencies"],
            "total_latency": stats["total_latency"]
        }
    
    def reset_inference_statistics(self, model_type: Optional[str] = None):
        """重置推理统计信息
        
        Args:
            model_type: 如果提供，只重置指定模型类型的统计信息；否则重置所有统计
        """
        if model_type is None:
            # 重置所有统计
            self.inference_latencies.clear()
            self.inference_start_times.clear()
            self.inference_by_model.clear()
            self.inference_success_count = 0
            self.inference_failure_count = 0
        elif model_type in self.inference_by_model:
            # 重置特定模型类型的统计
            model_stats = self.inference_by_model[model_type]
            
            # 从总延迟列表中移除该模型的延迟
            model_latencies = set(model_stats["latencies"])
            self.inference_latencies = [lat for lat in self.inference_latencies if lat not in model_latencies]
            
            # 更新总成功/失败计数
            self.inference_success_count -= model_stats["success_count"]
            self.inference_failure_count -= model_stats["failure_count"]
            
            # 移除模型统计
            del self.inference_by_model[model_type]
    
    # 任务跟踪方法
    def start_task_tracking(self, task_id: str, task_type: str = "generic", metadata: Optional[Dict[str, Any]] = None):
        """开始跟踪任务"""
        if self.config.get("enable_task_monitoring", True):
            self.active_tasks[task_id] = {
                "task_id": task_id,
                "task_type": task_type,
                "start_time": time.time(),
                "metadata": metadata or {}
            }
            return task_id
        return None
    
    def stop_task_tracking(self, task_id: str, success: bool = True, result: Optional[Any] = None):
        """停止跟踪任务"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks.pop(task_id)
            completion_time = time.time() - task_info["start_time"]
            
            # 记录完成时间
            self.task_completion_times.append(completion_time)
            
            # 保持列表大小
            if len(self.task_completion_times) > self.task_history_size:
                self.task_completion_times = self.task_completion_times[-self.task_history_size:]
            
            # 记录任务结果
            task_record = {
                "task_id": task_id,
                "task_type": task_info["task_type"],
                "start_time": task_info["start_time"],
                "completion_time": completion_time,
                "success": success,
                "result": result,
                "metadata": task_info["metadata"]
            }
            
            if success:
                self.completed_tasks.append(task_record)
                # 保持列表大小
                if len(self.completed_tasks) > self.task_history_size:
                    self.completed_tasks = self.completed_tasks[-self.task_history_size:]
            else:
                self.failed_tasks.append(task_record)
                # 保持列表大小
                if len(self.failed_tasks) > self.task_history_size:
                    self.failed_tasks = self.failed_tasks[-self.task_history_size:]
            
            return completion_time
        return None
    
    def _add_metric(self, metric: SystemMetric):
        """添加指标到历史记录"""
        self.metrics_history[metric.metric_type].append(metric)
        
        # 触发指标回调
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"指标回调错误: {e}")
    
    def _check_alert_conditions(self):
        """检查警报条件"""
        current_time = datetime.now()
        
        # 检查CPU警报
        if self.metrics_history[MetricType.CPU]:
            latest_cpu = self.metrics_history[MetricType.CPU][-1]
            cpu_thresholds = self.alert_thresholds.get("cpu", {})
            
            if latest_cpu.value >= cpu_thresholds.get("critical", 95.0):
                self._create_alert(
                    metric_type=MetricType.CPU,
                    value=latest_cpu.value,
                    threshold=cpu_thresholds.get("critical", 95.0),
                    level=AlertLevel.CRITICAL,
                    message=f"CPU使用率过高: {latest_cpu.value:.1f}%"
                )
            elif latest_cpu.value >= cpu_thresholds.get("warning", 80.0):
                self._create_alert(
                    metric_type=MetricType.CPU,
                    value=latest_cpu.value,
                    threshold=cpu_thresholds.get("warning", 80.0),
                    level=AlertLevel.WARNING,
                    message=f"CPU使用率偏高: {latest_cpu.value:.1f}%"
                )
        
        # 检查内存警报
        if self.metrics_history[MetricType.MEMORY]:
            latest_memory = self.metrics_history[MetricType.MEMORY][-1]
            memory_thresholds = self.alert_thresholds.get("memory", {})
            
            if latest_memory.value >= memory_thresholds.get("critical", 95.0):
                self._create_alert(
                    metric_type=MetricType.MEMORY,
                    value=latest_memory.value,
                    threshold=memory_thresholds.get("critical", 95.0),
                    level=AlertLevel.CRITICAL,
                    message=f"内存使用率过高: {latest_memory.value:.1f}%"
                )
            elif latest_memory.value >= memory_thresholds.get("warning", 85.0):
                self._create_alert(
                    metric_type=MetricType.MEMORY,
                    value=latest_memory.value,
                    threshold=memory_thresholds.get("warning", 85.0),
                    level=AlertLevel.WARNING,
                    message=f"内存使用率偏高: {latest_memory.value:.1f}%"
                )
        
        # 检查推理延迟警报
        if self.metrics_history[MetricType.INFERENCE]:
            latest_inference = self.metrics_history[MetricType.INFERENCE][-1]
            inference_thresholds = self.alert_thresholds.get("inference_latency", {})
            
            if latest_inference.value >= inference_thresholds.get("critical", 10000.0):
                self._create_alert(
                    metric_type=MetricType.INFERENCE,
                    value=latest_inference.value,
                    threshold=inference_thresholds.get("critical", 10000.0),
                    level=AlertLevel.CRITICAL,
                    message=f"推理延迟过高: {latest_inference.value:.1f}ms"
                )
            elif latest_inference.value >= inference_thresholds.get("warning", 5000.0):
                self._create_alert(
                    metric_type=MetricType.INFERENCE,
                    value=latest_inference.value,
                    threshold=inference_thresholds.get("warning", 5000.0),
                    level=AlertLevel.WARNING,
                    message=f"推理延迟偏高: {latest_inference.value:.1f}ms"
                )
        
        # 检查任务吞吐量警报（吞吐量越低越差）
        if self.metrics_history[MetricType.TASK]:
            latest_task = self.metrics_history[MetricType.TASK][-1]
            task_thresholds = self.alert_thresholds.get("task_throughput", {})
            
            if latest_task.value <= task_thresholds.get("critical", 5.0):
                self._create_alert(
                    metric_type=MetricType.TASK,
                    value=latest_task.value,
                    threshold=task_thresholds.get("critical", 5.0),
                    level=AlertLevel.CRITICAL,
                    message=f"任务吞吐量过低: {latest_task.value:.1f}任务/秒"
                )
            elif latest_task.value <= task_thresholds.get("warning", 10.0):
                self._create_alert(
                    metric_type=MetricType.TASK,
                    value=latest_task.value,
                    threshold=task_thresholds.get("warning", 10.0),
                    level=AlertLevel.WARNING,
                    message=f"任务吞吐量偏低: {latest_task.value:.1f}任务/秒"
                )
    
    def _create_alert(self, metric_type: MetricType, value: float, 
                     threshold: float, level: AlertLevel, message: str):
        """创建警报"""
        alert_id = f"{metric_type.value}_{int(time.time())}"
        
        # 计算趋势
        trend = None
        if len(self.metrics_history[metric_type]) >= self.trend_window:
            recent_values = [
                m.value for m in list(self.metrics_history[metric_type])[-self.trend_window:]
            ]
            if len(recent_values) >= 2:
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now(),
            value=value,
            threshold=threshold,
            trend=trend,
            suggestions=self._generate_alert_suggestions(metric_type, value, level)
        )
        
        self.alerts.append(alert)
        
        # 触发警报回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警报回调错误: {e}")
        
        logger.warning(f"性能警报: {message}")
    
    def _generate_alert_suggestions(self, metric_type: MetricType, 
                                   value: float, level: AlertLevel) -> List[str]:
        """生成警报建议"""
        suggestions = []
        
        if metric_type == MetricType.CPU:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即检查CPU密集型任务",
                    "考虑优化算法复杂度",
                    "增加CPU核心数或升级硬件",
                    "启用任务调度优化"
                ]
            elif level == AlertLevel.WARNING:
                suggestions = [
                    "监控CPU使用趋势",
                    "优化代码执行效率",
                    "考虑任务分批处理",
                    "检查是否有异常进程"
                ]
        
        elif metric_type == MetricType.MEMORY:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即检查内存泄漏",
                    "增加系统内存",
                    "优化内存使用模式",
                    "启用内存回收机制"
                ]
            elif level == AlertLevel.WARNING:
                suggestions = [
                    "监控内存增长趋势",
                    "优化数据结构",
                    "及时释放不再使用的对象",
                    "检查缓存策略"
                ]
        
        elif metric_type == MetricType.INFERENCE:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即检查推理模型性能",
                    "优化模型推理流程",
                    "考虑模型量化或剪枝",
                    "增加推理硬件资源"
                ]
            elif level == AlertLevel.WARNING:
                suggestions = [
                    "监控推理延迟趋势",
                    "优化输入数据处理",
                    "考虑批处理推理",
                    "检查模型缓存策略"
                ]
        
        elif metric_type == MetricType.TASK:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即检查任务调度系统",
                    "优化任务并行度",
                    "增加计算资源",
                    "检查任务依赖关系"
                ]
            elif level == AlertLevel.WARNING:
                suggestions = [
                    "监控任务吞吐量趋势",
                    "优化任务执行流程",
                    "考虑任务优先级调整",
                    "检查系统资源使用情况"
                ]
        
        return suggestions
    
    def _generate_optimization_suggestions(self):
        """生成优化建议"""
        current_time = datetime.now()
        
        # 清理旧的建议
        self.optimization_suggestions = [
            s for s in self.optimization_suggestions
            if (current_time - s.get("timestamp", current_time)).total_seconds() < 3600
        ]
        
        # 基于历史数据生成建议
        suggestions = []
        
        # CPU使用率趋势分析
        if self.metrics_history[MetricType.CPU]:
            cpu_values = [m.value for m in self.metrics_history[MetricType.CPU]]
            if len(cpu_values) >= 20:
                avg_cpu = np.mean(cpu_values[-20:])
                if avg_cpu > 70.0:
                    suggestions.append({
                        "type": "high_cpu_usage",
                        "priority": "high" if avg_cpu > 85.0 else "medium",
                        "description": f"CPU平均使用率偏高: {avg_cpu:.1f}%",
                        "suggestion": "考虑优化计算密集型任务或增加计算资源",
                        "timestamp": current_time
                    })
        
        # 内存使用趋势分析
        if self.metrics_history[MetricType.MEMORY]:
            memory_values = [m.value for m in self.metrics_history[MetricType.MEMORY]]
            if len(memory_values) >= 20:
                avg_memory = np.mean(memory_values[-20:])
                if avg_memory > 75.0:
                    suggestions.append({
                        "type": "high_memory_usage",
                        "priority": "high" if avg_memory > 90.0 else "medium",
                        "description": f"内存平均使用率偏高: {avg_memory:.1f}%",
                        "suggestion": "优化内存使用，检查内存泄漏，考虑增加内存",
                        "timestamp": current_time
                    })
        
        # 推理延迟趋势分析
        if self.metrics_history[MetricType.INFERENCE]:
            inference_values = [m.value for m in self.metrics_history[MetricType.INFERENCE]]
            if len(inference_values) >= 10:
                avg_inference = np.mean(inference_values[-10:])
                if avg_inference > 3000.0:  # 超过3秒
                    suggestions.append({
                        "type": "high_inference_latency",
                        "priority": "high" if avg_inference > 5000.0 else "medium",
                        "description": f"推理延迟偏高: {avg_inference:.1f}ms",
                        "suggestion": "优化模型推理流程，考虑模型压缩或硬件加速",
                        "timestamp": current_time
                    })
        
        # 任务吞吐量趋势分析
        if self.metrics_history[MetricType.TASK]:
            task_values = [m.value for m in self.metrics_history[MetricType.TASK]]
            if len(task_values) >= 10:
                avg_task = np.mean(task_values[-10:])
                if avg_task < 15.0:  # 低于15任务/秒
                    suggestions.append({
                        "type": "low_task_throughput",
                        "priority": "high" if avg_task < 5.0 else "medium",
                        "description": f"任务吞吐量偏低: {avg_task:.1f}任务/秒",
                        "suggestion": "优化任务调度和执行流程，增加并行度",
                        "timestamp": current_time
                    })
        
        # 添加新的建议
        self.optimization_suggestions.extend(suggestions)
    
    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.output_dir, f"performance_report_{timestamp}.json")
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_summary(),
                "alerts": [
                    {
                        "id": alert.alert_id,
                        "level": alert.level.value,
                        "metric_type": alert.metric_type.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "trend": alert.trend,
                        "suggestions": alert.suggestions
                    }
                    for alert in self.alerts[-50:]  # 最近50条警报
                ],
                "optimization_suggestions": self.optimization_suggestions,
                "metrics_summary": self._get_metrics_summary(),
                "visualizations": self._generate_visualizations()
            }
            
            # 转换为JSON可序列化的格式
            serializable_report = self._convert_to_serializable(report)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"性能报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        summary = {
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            "warning_alerts": len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
            "optimization_suggestions_count": len(self.optimization_suggestions),
            "monitoring_duration_hours": self._get_monitoring_duration_hours()
        }
        
        # 添加指标摘要
        for metric_type in MetricType:
            if self.metrics_history[metric_type]:
                values = [m.value for m in self.metrics_history[metric_type]]
                summary[f"{metric_type.value}_avg"] = np.mean(values) if values else 0.0
                summary[f"{metric_type.value}_max"] = np.max(values) if values else 0.0
                summary[f"{metric_type.value}_min"] = np.min(values) if values else 0.0
        
        return summary
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        
        for metric_type in MetricType:
            if self.metrics_history[metric_type]:
                metrics_list = list(self.metrics_history[metric_type])
                summary[metric_type.value] = {
                    "count": len(metrics_list),
                    "latest_value": metrics_list[-1].value if metrics_list else None,
                    "latest_timestamp": metrics_list[-1].timestamp.isoformat() if metrics_list else None,
                    "unit": metrics_list[-1].unit if metrics_list else None
                }
        
        return summary
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """生成可视化图表"""
        visualizations = {}
        
        if not PLOTLY_AVAILABLE:
            return visualizations
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 生成CPU使用率图表
            if self.metrics_history[MetricType.CPU]:
                cpu_file = self._plot_metric_timeseries(
                    MetricType.CPU, 
                    f"{self.output_dir}/cpu_timeseries_{timestamp}.html"
                )
                if cpu_file:
                    visualizations["cpu_timeseries"] = cpu_file
            
            # 生成内存使用率图表
            if self.metrics_history[MetricType.MEMORY]:
                memory_file = self._plot_metric_timeseries(
                    MetricType.MEMORY,
                    f"{self.output_dir}/memory_timeseries_{timestamp}.html"
                )
                if memory_file:
                    visualizations["memory_timeseries"] = memory_file
            
            # 生成警报分布图表
            if self.alerts:
                alerts_file = self._plot_alerts_distribution(
                    f"{self.output_dir}/alerts_distribution_{timestamp}.html"
                )
                if alerts_file:
                    visualizations["alerts_distribution"] = alerts_file
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
        
        return visualizations
    
    def _plot_metric_timeseries(self, metric_type: MetricType, 
                               output_file: str) -> Optional[str]:
        """绘制指标时间序列图"""
        try:
            if not self.metrics_history[metric_type] or not PLOTLY_AVAILABLE:
                return None
            
            metrics = list(self.metrics_history[metric_type])
            timestamps = [m.timestamp for m in metrics]
            values = [m.value for m in metrics]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=metric_type.value,
                line=dict(color='blue', width=2)
            ))
            
            # 添加阈值线
            thresholds = self.alert_thresholds.get(metric_type.value, {})
            warning_threshold = thresholds.get("warning")
            critical_threshold = thresholds.get("critical")
            
            if warning_threshold:
                fig.add_hline(
                    y=warning_threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="警告阈值",
                    annotation_position="bottom right"
                )
            
            if critical_threshold:
                fig.add_hline(
                    y=critical_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="严重阈值",
                    annotation_position="top right"
                )
            
            fig.update_layout(
                title=f"{metric_type.value.upper()} 使用率时间序列",
                xaxis_title="时间",
                yaxis_title=f"使用率 ({metrics[0].unit})",
                template="plotly_white",
                height=500
            )
            
            fig.write_html(output_file)
            return output_file
            
        except Exception as e:
            logger.error(f"绘制时间序列图失败: {e}")
            return None
    
    def _plot_alerts_distribution(self, output_file: str) -> Optional[str]:
        """绘制警报分布图"""
        try:
            if not self.alerts or not PLOTLY_AVAILABLE:
                return None
            
            # 按级别分组
            levels = ["critical", "warning", "info"]
            level_counts = {level: 0 for level in levels}
            
            for alert in self.alerts:
                level_counts[alert.level.value] += 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(level_counts.keys()),
                    values=list(level_counts.values()),
                    hole=0.3,
                    marker_colors=["red", "orange", "blue"]
                )
            ])
            
            fig.update_layout(
                title="警报级别分布",
                template="plotly_white",
                height=500
            )
            
            fig.write_html(output_file)
            return output_file
            
        except Exception as e:
            logger.error(f"绘制警报分布图失败: {e}")
            return None
    
    def _get_monitoring_duration_hours(self) -> float:
        """获取监控持续时间（小时）"""
        if not self.metrics_history[MetricType.CPU]:
            return 0.0
        
        first_metric = self.metrics_history[MetricType.CPU][0]
        last_metric = self.metrics_history[MetricType.CPU][-1]
        
        duration_seconds = (last_metric.timestamp - first_metric.timestamp).total_seconds()
        return duration_seconds / 3600.0
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """注册警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def register_metric_callback(self, callback: Callable[[SystemMetric], None]):
        """注册指标回调函数"""
        self.metric_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        status = {
            "monitoring": self.monitor_running,
            "total_metrics": sum(len(history) for history in self.metrics_history.values()),
            "total_alerts": len(self.alerts),
            "active_optimization_suggestions": len(self.optimization_suggestions),
            "monitoring_duration_hours": self._get_monitoring_duration_hours()
        }
        
        # 添加最新指标值
        for metric_type in MetricType:
            if self.metrics_history[metric_type]:
                latest = self.metrics_history[metric_type][-1]
                status[f"latest_{metric_type.value}"] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return status
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取指定时间范围内的性能摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            "time_range_hours": hours,
            "metrics_summary": {},
            "alerts_summary": {},
            "optimization_summary": {}
        }
        
        # 筛选指定时间范围内的指标
        for metric_type in MetricType:
            recent_metrics = [
                m for m in self.metrics_history[metric_type]
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["metrics_summary"][metric_type.value] = {
                    "count": len(values),
                    "avg": np.mean(values) if values else 0.0,
                    "max": np.max(values) if values else 0.0,
                    "min": np.min(values) if values else 0.0
                }
        
        # 筛选指定时间范围内的警报
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        summary["alerts_summary"] = {
            "total": len(recent_alerts),
            "by_level": {
                level.value: len([a for a in recent_alerts if a.level == level])
                for level in AlertLevel
            }
        }
        
        # 筛选指定时间范围内的优化建议
        recent_suggestions = [
            s for s in self.optimization_suggestions
            if s.get("timestamp", cutoff_time) >= cutoff_time
        ]
        summary["optimization_summary"] = {
            "total": len(recent_suggestions),
            "by_priority": {
                "high": len([s for s in recent_suggestions if s.get("priority") == "high"]),
                "medium": len([s for s in recent_suggestions if s.get("priority") == "medium"]),
                "low": len([s for s in recent_suggestions if s.get("priority") == "low"])
            }
        }
        
        return summary
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化的格式
        
        处理numpy数组、标量和其他非标准类型
        """
        if obj is None:
            return None
        
        # 处理numpy标量
        if hasattr(obj, 'item'):
            try:
                return obj.item()  # 将numpy标量转换为Python标量
            except:
                pass
        
        # 处理numpy数组
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()  # 将numpy数组转换为Python列表
            except:
                pass
        
        # 处理字典
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        
        # 处理列表、元组等可迭代对象
        if isinstance(obj, (list, tuple, set)):
            return [self._convert_to_serializable(item) for item in obj]
        
        # 处理基本类型
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # 处理datetime对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # 尝试转换为字符串
        try:
            return str(obj)
        except:
            # 如果无法转换，返回None或空字符串
            return None


def create_performance_monitoring_dashboard(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitoringDashboard:
    """创建性能监控仪表板实例"""
    return PerformanceMonitoringDashboard(config)