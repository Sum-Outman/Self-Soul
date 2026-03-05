#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源调度管理器 - 集成动态资源调度和内存优化功能

核心功能：
1. 动态资源调度 - 集成现有的DynamicResourceScheduler进行智能资源分配
2. 内存优化管理 - 监控内存使用、实施垃圾回收、管理缓存
3. 性能监控 - 实时监控系统性能指标
4. 优化建议 - 基于资源使用情况生成优化建议

设计目标：
- 解决工业故障诊断系统中的资源管理不足问题
- 实现内存使用优化，防止内存泄漏
- 提供实时性能监控和调优建议
"""

import sys
import os
import time
import threading
import json
import logging
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch

# 导入动态资源调度器
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.dynamic_resource_scheduler import (
        DynamicResourceScheduler, TaskRequirement, TaskPriority,
        ResourceType, SystemResource, create_dynamic_resource_scheduler
    )
    DYNAMIC_SCHEDULER_AVAILABLE = True
except ImportError:
    DYNAMIC_SCHEDULER_AVAILABLE = False
    print("警告: 动态资源调度器不可用，将使用简化实现")

logger = logging.getLogger(__name__)


class MemoryOptimizationStrategy(Enum):
    """内存优化策略"""
    AGGRESSIVE = "aggressive"      # 激进优化，频繁回收内存
    MODERATE = "moderate"          # 中等优化，平衡性能和内存
    CONSERVATIVE = "conservative"  # 保守优化，优先保证性能
    ADAPTIVE = "adaptive"          # 自适应优化，根据系统负载调整


@dataclass
class MemoryUsageSnapshot:
    """内存使用快照"""
    timestamp: float
    process_memory_mb: float
    system_memory_percent: float
    system_memory_available_mb: float
    gpu_memory_mb: Optional[float] = None
    tensor_count: Optional[int] = None
    tensor_memory_mb: Optional[float] = None
    cache_size_mb: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    error_rate: Optional[float] = None


class ResourceSchedulerManager:
    """资源调度管理器 - 集成资源调度和内存优化"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化资源调度管理器"""
        self.config = config or self._get_default_config()
        
        # 初始化动态资源调度器
        self.dynamic_scheduler = None
        if DYNAMIC_SCHEDULER_AVAILABLE and self.config.get("enable_dynamic_scheduling", True):
            scheduler_config = self.config.get("scheduler_config", {})
            self.dynamic_scheduler = create_dynamic_resource_scheduler(scheduler_config)
        
        # 内存优化配置
        self.memory_strategy = MemoryOptimizationStrategy(
            self.config.get("memory_strategy", "adaptive")
        )
        
        # 检测系统总内存并设置合理的阈值
        self.total_system_memory_mb = self._detect_system_memory()
        
        # 动态计算内存阈值（默认为系统内存的50%或配置值）
        default_threshold = min(self.total_system_memory_mb * 0.5, 4096)  # 最多4GB
        self.memory_threshold_mb = self.config.get("memory_threshold_mb", default_threshold)
        
        # 动态计算缓存限制（默认为系统内存的25%或配置值）
        default_cache_limit = min(self.total_system_memory_mb * 0.25, 2048)  # 最多2GB
        self.cache_size_limit_mb = self.config.get("cache_size_limit_mb", default_cache_limit)
        
        # 内存使用趋势跟踪
        self.memory_trend: List[float] = []
        self.memory_trend_window = 10  # 跟踪最近10个样本
        self.memory_leak_detected = False
        
        # 监控数据
        self.memory_snapshots: List[MemoryUsageSnapshot] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # 缓存管理
        self.data_cache: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 内存优化状态
        self.last_gc_time = time.time()
        self.gc_interval = self.config.get("gc_interval", 60)  # 默认60秒一次GC
        
        # 性能监控线程
        self.monitor_thread = None
        self.monitor_running = False
        self.monitor_interval = self.config.get("monitor_interval", 5)  # 5秒监控间隔
        
        # 锁
        self.cache_lock = threading.RLock()
        self.monitor_lock = threading.RLock()
        
        logger.info(f"资源调度管理器初始化完成，内存策略: {self.memory_strategy.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enable_dynamic_scheduling": True,
            "memory_strategy": "adaptive",
            "memory_threshold_mb": 1024,
            "cache_size_limit_mb": 512,
            "gc_interval": 60,  # 秒
            "monitor_interval": 5,  # 秒
            "performance_history_size": 100,
            "scheduler_config": {
                "scheduling_interval": 2.0,
                "max_concurrent_tasks": 8,
                "resource_utilization_target": 75.0,
                "enable_preemption": True
            },
            "enable_gpu_monitoring": True,
            "enable_tensor_tracking": True,
            "auto_optimization": True
        }
    
    def _detect_system_memory(self) -> float:
        """检测系统总内存（MB）"""
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            total_mb = system_memory.total / (1024 * 1024)
            logger.info(f"检测到系统总内存: {total_mb:.1f}MB")
            return float(total_mb)
        except Exception as e:
            logger.warning(f"无法检测系统内存: {e}")
            # 默认返回8GB
            return 8192.0  # 8GB in MB
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitor_running:
            logger.warning("性能监控已经在运行")
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动动态调度器
        if self.dynamic_scheduler:
            self.dynamic_scheduler.start_scheduler()
        
        logger.info("资源监控和调度已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # 停止动态调度器
        if self.dynamic_scheduler:
            self.dynamic_scheduler.stop_scheduler()
        
        logger.info("资源监控和调度已停止")
    
    def _update_memory_trend(self, current_memory_mb: float):
        """更新内存使用趋势"""
        self.memory_trend.append(current_memory_mb)
        # 保持趋势窗口大小
        if len(self.memory_trend) > self.memory_trend_window:
            self.memory_trend.pop(0)
    
    def _check_memory_leak(self):
        """检查内存泄漏"""
        if len(self.memory_trend) < 2:
            return
        
        # 计算内存增长趋势
        # 使用线性回归检测趋势
        x = list(range(len(self.memory_trend)))
        y = self.memory_trend
        
        # 简单趋势检测：检查是否持续增长
        if len(y) >= 5:
            # 计算最近5个点的斜率
            recent_x = x[-5:]
            recent_y = y[-5:]
            
            # 简单线性回归
            n = len(recent_x)
            sum_x = sum(recent_x)
            sum_y = sum(recent_y)
            sum_xy = sum(x_i * y_i for x_i, y_i in zip(recent_x, recent_y))
            sum_x2 = sum(x_i * x_i for x_i in recent_x)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # 如果斜率大于阈值，可能内存泄漏
                if slope > 10.0:  # 每秒增长超过10MB
                    if not self.memory_leak_detected:
                        self.memory_leak_detected = True
                        logger.warning(f"检测到可能的内存泄漏，斜率: {slope:.2f} MB/样本")
                elif slope < -5.0:  # 显著下降
                    self.memory_leak_detected = False
    
    def _adjust_gc_interval_based_on_memory_pressure(self, snapshot: MemoryUsageSnapshot):
        """根据内存压力调整GC间隔"""
        memory_pressure = snapshot.process_memory_mb / self.memory_threshold_mb
        
        if memory_pressure > 0.9:
            # 高内存压力：更频繁的GC
            self.gc_interval = max(10, int(self.config.get("gc_interval", 60) * 0.3))
        elif memory_pressure > 0.7:
            # 中等内存压力：适度频繁的GC
            self.gc_interval = max(20, int(self.config.get("gc_interval", 60) * 0.6))
        elif memory_pressure < 0.3:
            # 低内存压力：减少GC频率
            self.gc_interval = min(120, int(self.config.get("gc_interval", 60) * 1.5))
        else:
            # 正常内存压力：使用默认间隔
            self.gc_interval = self.config.get("gc_interval", 60)
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.monitor_running:
            try:
                # 收集性能指标
                self._collect_performance_metrics()
                
                # 收集内存快照
                self._collect_memory_snapshot()
                
                # 执行内存优化
                self._optimize_memory_usage()
                
                # 管理缓存
                self._manage_cache()
                
                # 清理历史数据
                self._cleanup_history_data()
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            # 等待下一次监控
            time.sleep(self.monitor_interval)
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        process = psutil.Process()
        
        # 收集CPU和内存指标
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
        
        # 收集GPU指标（如果可用）
        gpu_percent = None
        gpu_memory_mb = None
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_percent = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None
                gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        except:
            pass
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb
        )
        
        # 保存指标
        with self.monitor_lock:
            self.performance_metrics.append(metrics)
            
            # 限制历史数据大小
            max_history = self.config.get("performance_history_size", 100)
            if len(self.performance_metrics) > max_history:
                self.performance_metrics = self.performance_metrics[-max_history:]
    
    def _collect_memory_snapshot(self):
        """收集内存快照"""
        process = psutil.Process()
        system_memory = psutil.virtual_memory()
        
        # 收集进程内存
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # 收集系统内存
        system_memory_percent = system_memory.percent
        system_memory_available_mb = system_memory.available / (1024 * 1024)
        
        # 收集GPU内存（如果可用）
        gpu_memory_mb = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        except:
            pass
        
        # 收集张量信息（如果启用）
        tensor_count = None
        tensor_memory_mb = None
        if self.config.get("enable_tensor_tracking", True):
            tensor_count, tensor_memory_mb = self._get_tensor_memory_info()
        
        # 计算缓存大小
        cache_size_mb = self._calculate_cache_size()
        
        # 创建内存快照
        snapshot = MemoryUsageSnapshot(
            timestamp=time.time(),
            process_memory_mb=process_memory_mb,
            system_memory_percent=system_memory_percent,
            system_memory_available_mb=system_memory_available_mb,
            gpu_memory_mb=gpu_memory_mb,
            tensor_count=tensor_count,
            tensor_memory_mb=tensor_memory_mb,
            cache_size_mb=cache_size_mb
        )
        
        # 保存快照
        with self.monitor_lock:
            self.memory_snapshots.append(snapshot)
            
            # 限制历史数据大小
            max_history = self.config.get("performance_history_size", 100)
            if len(self.memory_snapshots) > max_history:
                self.memory_snapshots = self.memory_snapshots[-max_history:]
    
    def _get_tensor_memory_info(self) -> Tuple[Optional[int], Optional[float]]:
        """获取张量内存信息"""
        try:
            import torch
            tensor_count = 0
            total_memory = 0
            
            # 收集所有张量的信息
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor):
                    tensor_count += 1
                    if obj.is_cuda:
                        total_memory += obj.element_size() * obj.nelement()
            
            tensor_memory_mb = total_memory / (1024 * 1024) if total_memory > 0 else 0
            return tensor_count, tensor_memory_mb
        except:
            return None, None
    
    def _calculate_cache_size(self) -> float:
        """计算缓存大小（MB）"""
        with self.cache_lock:
            # 简化实现：估算缓存大小
            total_size = 0
            
            # 估算数据缓存大小
            for key, value in self.data_cache.items():
                # 粗略估算：字符串长度 + 数据大小
                if isinstance(value, (str, bytes)):
                    total_size += len(value) / 1024  # KB
                elif isinstance(value, (list, dict)):
                    total_size += len(str(value)) / 1024  # KB
            
            # 估算模型缓存大小（如果有）
            for key, value in self.model_cache.items():
                if hasattr(value, '__sizeof__'):
                    total_size += value.__sizeof__() / 1024  # KB
            
            return total_size / 1024  # 转换为MB
    
    def _optimize_memory_usage(self):
        """优化内存使用"""
        current_time = time.time()
        
        # 检查内存趋势
        if len(self.memory_snapshots) > 0:
            latest_snapshot = self.memory_snapshots[-1]
            
            # 更新内存趋势
            self._update_memory_trend(latest_snapshot.process_memory_mb)
            
            # 检查内存泄漏
            if len(self.memory_trend) >= self.memory_trend_window:
                self._check_memory_leak()
            
            # 自适应调整GC间隔
            self._adjust_gc_interval_based_on_memory_pressure(latest_snapshot)
            
            # 检查内存阈值
            threshold_multiplier = 1.0
            if self.memory_leak_detected:
                threshold_multiplier = 0.8  # 内存泄漏时使用更低的阈值
                logger.warning("检测到可能的内存泄漏，使用更保守的内存阈值")
            
            # 如果进程内存超过阈值，执行更激进的内存优化
            if latest_snapshot.process_memory_mb > self.memory_threshold_mb * threshold_multiplier:
                self._execute_aggressive_memory_optimization(latest_snapshot)
        
        # 检查是否需要执行垃圾回收
        if current_time - self.last_gc_time > self.gc_interval:
            self._execute_garbage_collection()
            self.last_gc_time = current_time
    
    def _execute_garbage_collection(self):
        """执行垃圾回收"""
        logger.debug("执行垃圾回收")
        
        # 收集垃圾回收前的内存使用情况
        before_count = len(gc.get_objects())
        
        # 执行垃圾回收
        gc.collect()
        
        # 收集垃圾回收后的内存使用情况
        after_count = len(gc.get_objects())
        
        logger.info(f"垃圾回收完成，对象数: {before_count} -> {after_count}，释放了 {before_count - after_count} 个对象")
    
    def _execute_aggressive_memory_optimization(self, snapshot: MemoryUsageSnapshot):
        """执行激进的内存优化"""
        logger.warning(f"进程内存使用过高 ({snapshot.process_memory_mb:.1f}MB > {self.memory_threshold_mb}MB)，执行激进内存优化")
        
        # 1. 清理缓存
        self._clear_cache(aggressive=True)
        
        # 2. 强制垃圾回收
        gc.collect()
        
        # 3. 清空PyTorch缓存（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清空PyTorch GPU缓存")
        except:
            pass
        
        # 4. 根据内存策略调整优化强度
        if self.memory_strategy == MemoryOptimizationStrategy.AGGRESSIVE:
            # 激进策略：减少GC间隔
            self.gc_interval = max(10, self.gc_interval // 2)
            logger.info(f"调整GC间隔为 {self.gc_interval} 秒")
    
    def _manage_cache(self):
        """管理缓存"""
        with self.cache_lock:
            # 计算当前缓存大小
            cache_size_mb = self._calculate_cache_size()
            
            # 如果缓存超过限制，清理最旧的缓存项
            if cache_size_mb > self.cache_size_limit_mb:
                logger.info(f"缓存大小超过限制 ({cache_size_mb:.1f}MB > {self.cache_size_limit_mb}MB)，开始清理")
                
                # 清理数据缓存（最旧的50%）
                if self.data_cache:
                    items_to_remove = len(self.data_cache) // 2
                    keys_to_remove = list(self.data_cache.keys())[:items_to_remove]
                    for key in keys_to_remove:
                        del self.data_cache[key]
                    
                    logger.info(f"清理了 {len(keys_to_remove)} 个数据缓存项")
                
                # 清理模型缓存（如果有）
                if self.model_cache:
                    items_to_remove = len(self.model_cache) // 2
                    keys_to_remove = list(self.model_cache.keys())[:items_to_remove]
                    for key in keys_to_remove:
                        del self.model_cache[key]
                    
                    logger.info(f"清理了 {len(keys_to_remove)} 个模型缓存项")
    
    def _clear_cache(self, aggressive: bool = False):
        """清理缓存"""
        with self.cache_lock:
            items_removed = 0
            
            if aggressive:
                # 激进清理：清空所有缓存
                items_removed = len(self.data_cache) + len(self.model_cache)
                self.data_cache.clear()
                self.model_cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0
            else:
                # 常规清理：只清理最旧的缓存项
                if self.data_cache:
                    items_to_remove = max(1, len(self.data_cache) // 4)  # 清理25%
                    keys_to_remove = list(self.data_cache.keys())[:items_to_remove]
                    for key in keys_to_remove:
                        del self.data_cache[key]
                    items_removed += len(keys_to_remove)
            
            if items_removed > 0:
                logger.info(f"清理了 {items_removed} 个缓存项")
    
    def _cleanup_history_data(self):
        """清理历史数据"""
        with self.monitor_lock:
            max_history = self.config.get("performance_history_size", 100)
            
            # 清理性能指标历史
            if len(self.performance_metrics) > max_history * 2:
                self.performance_metrics = self.performance_metrics[-max_history:]
            
            # 清理内存快照历史
            if len(self.memory_snapshots) > max_history * 2:
                self.memory_snapshots = self.memory_snapshots[-max_history:]
    
    def get_cache(self, key: str, cache_type: str = "data") -> Any:
        """从缓存中获取数据"""
        with self.cache_lock:
            if cache_type == "data":
                if key in self.data_cache:
                    self.cache_hits += 1
                    return self.data_cache[key]
            elif cache_type == "model":
                if key in self.model_cache:
                    self.cache_hits += 1
                    return self.model_cache[key]
            
            self.cache_misses += 1
            return None
    
    def set_cache(self, key: str, value: Any, cache_type: str = "data"):
        """设置缓存数据"""
        with self.cache_lock:
            if cache_type == "data":
                self.data_cache[key] = value
            elif cache_type == "model":
                self.model_cache[key] = value
    
    def submit_task(self, task_requirement: Any) -> Optional[str]:
        """提交任务到动态调度器"""
        if self.dynamic_scheduler and DYNAMIC_SCHEDULER_AVAILABLE:
            try:
                task_id = self.dynamic_scheduler.submit_task(task_requirement)
                return task_id
            except Exception as e:
                logger.error(f"提交任务到动态调度器失败: {e}")
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.monitor_lock:
            if not self.performance_metrics:
                # 没有数据时返回基本报告
                try:
                    import psutil
                    process = psutil.Process()
                    
                    # 获取CPU使用率 - 使用更可靠的方法
                    # 第一次调用初始化，第二次获取实际值
                    process.cpu_percent()  # 初始化
                    time.sleep(0.05)  # 短暂延迟
                    cpu_percent = process.cpu_percent()
                    
                    # 限制CPU使用率在合理范围内（0-100%或0-核心数*100%）
                    # 对于多核系统，cpu_percent可能超过100%，但702%显然异常
                    # 我们将限制在0-500%之间（假设最多5个核心完全利用）
                    cpu_percent = min(max(cpu_percent, 0.0), 500.0)
                    
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "current": {
                            "cpu_percent": float(cpu_percent),  # 确保转换为Python float
                            "memory_mb": float(memory_mb),      # 确保转换为Python float
                            "gpu_percent": None,
                            "gpu_memory_mb": None
                        },
                        "statistics": {
                            "cpu_avg": float(cpu_percent),
                            "cpu_max": float(cpu_percent),
                            "cpu_min": float(cpu_percent),
                            "memory_avg_mb": float(memory_mb),
                            "memory_max_mb": float(memory_mb),
                            "memory_min_mb": float(memory_mb),
                        },
                        "monitoring_duration_seconds": 0.0,
                        "samples_collected": 1
                    }
                except Exception as e:
                    logger.warning(f"获取性能数据失败: {e}")
                    return {"error": "无法获取性能数据"}
            
            latest = self.performance_metrics[-1]
            earliest = self.performance_metrics[0] if len(self.performance_metrics) > 1 else latest
            
            # 计算统计信息，确保数据类型正确
            cpu_values = []
            memory_values = []
            
            for m in self.performance_metrics:
                # 确保CPU值在合理范围内
                cpu_val = getattr(m, "cpu_percent", 0.0)
                if isinstance(cpu_val, (int, float, np.number)):
                    cpu_val = float(min(max(cpu_val, 0.0), 500.0))  # 限制在0-500%
                    cpu_values.append(cpu_val)
                
                # 确保内存值正确
                mem_val = getattr(m, "memory_mb", 0.0)
                if isinstance(mem_val, (int, float, np.number)):
                    memory_values.append(float(mem_val))
            
            # 计算统计信息，确保返回Python原生类型
            cpu_avg = float(np.mean(cpu_values)) if cpu_values else 0.0
            cpu_max = float(np.max(cpu_values)) if cpu_values else 0.0
            cpu_min = float(np.min(cpu_values)) if cpu_values else 0.0
            
            memory_avg = float(np.mean(memory_values)) if memory_values else 0.0
            memory_max = float(np.max(memory_values)) if memory_values else 0.0
            memory_min = float(np.min(memory_values)) if memory_values else 0.0
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "current": {
                    "cpu_percent": float(getattr(latest, "cpu_percent", 0.0)),
                    "memory_mb": float(getattr(latest, "memory_mb", 0.0)),
                    "gpu_percent": float(getattr(latest, "gpu_percent", 0.0)) if getattr(latest, "gpu_percent") is not None else None,
                    "gpu_memory_mb": float(getattr(latest, "gpu_memory_mb", 0.0)) if getattr(latest, "gpu_memory_mb") is not None else None
                },
                "statistics": {
                    "cpu_avg": cpu_avg,
                    "cpu_max": cpu_max,
                    "cpu_min": cpu_min,
                    "memory_avg_mb": memory_avg,
                    "memory_max_mb": memory_max,
                    "memory_min_mb": memory_min,
                },
                "monitoring_duration_seconds": float(latest.timestamp - earliest.timestamp),
                "samples_collected": int(len(self.performance_metrics))
            }
            
            return report
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存报告"""
        with self.monitor_lock:
            if not self.memory_snapshots:
                # 没有数据时返回基本报告
                try:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    process_memory_mb = memory_info.rss / (1024 * 1024)
                    
                    # 获取系统内存
                    system_memory = psutil.virtual_memory()
                    
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "current": {
                            "process_memory_mb": process_memory_mb,
                            "system_memory_percent": system_memory.percent,
                            "system_memory_available_mb": system_memory.available / (1024 * 1024),
                            "cache_size_mb": 0.0,
                            "gpu_memory_mb": None,
                            "tensor_count": 0,
                            "tensor_memory_mb": 0.0
                        },
                        "cache_statistics": {
                            "cache_hit_rate": 0.0,
                            "cache_hits": self.cache_hits,
                            "cache_misses": self.cache_misses,
                            "data_cache_items": len(self.data_cache),
                            "model_cache_items": len(self.model_cache)
                        },
                        "optimization": {
                            "memory_strategy": self.memory_strategy.value,
                            "memory_threshold_mb": self.memory_threshold_mb,
                            "cache_size_limit_mb": self.cache_size_limit_mb,
                            "gc_interval": self.gc_interval
                        }
                    }
                except Exception:
                    return {"error": "无法获取内存数据"}
            
            latest = self.memory_snapshots[-1]
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "current": {
                    "process_memory_mb": float(getattr(latest, "process_memory_mb", 0.0)),
                    "system_memory_percent": float(getattr(latest, "system_memory_percent", 0.0)),
                    "system_memory_available_mb": float(getattr(latest, "system_memory_available_mb", 0.0)),
                    "gpu_memory_mb": float(getattr(latest, "gpu_memory_mb", 0.0)) if getattr(latest, "gpu_memory_mb") is not None else None,
                    "tensor_count": int(getattr(latest, "tensor_count", 0)),
                    "tensor_memory_mb": float(getattr(latest, "tensor_memory_mb", 0.0)),
                    "cache_size_mb": float(getattr(latest, "cache_size_mb", 0.0))
                },
                "cache_statistics": {
                    "cache_hits": int(self.cache_hits),
                    "cache_misses": int(self.cache_misses),
                    "cache_hit_rate": float(self.cache_hits / (self.cache_hits + self.cache_misses) 
                                 if (self.cache_hits + self.cache_misses) > 0 else 0),
                    "data_cache_items": int(len(self.data_cache)),
                    "model_cache_items": int(len(self.model_cache))
                },
                "optimization": {
                    "memory_strategy": str(self.memory_strategy.value),
                    "memory_threshold_mb": int(self.memory_threshold_mb),
                    "cache_size_limit_mb": int(self.cache_size_limit_mb),
                    "gc_interval": int(self.gc_interval)
                }
            }
            
            return report
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        suggestions = []
        
        # 获取当前性能报告
        perf_report = self.get_performance_report()
        mem_report = self.get_memory_report()
        
        if "error" in perf_report or "error" in mem_report:
            return suggestions
        
        # 确保从报告中提取的值是Python原生类型
        try:
            process_memory_mb = float(mem_report["current"]["process_memory_mb"])
            cache_hit_rate = float(mem_report["cache_statistics"]["cache_hit_rate"])
            data_cache_items = int(mem_report["cache_statistics"]["data_cache_items"])
            cpu_avg = float(perf_report["statistics"]["cpu_avg"])
            system_memory_percent = float(mem_report["current"]["system_memory_percent"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"提取优化建议数据失败: {e}")
            return suggestions
        
        # 1. 内存使用建议
        if process_memory_mb > self.memory_threshold_mb * 0.8:
            suggestions.append({
                "type": "high_memory_usage",
                "priority": "high",
                "current_mb": process_memory_mb,
                "threshold_mb": int(self.memory_threshold_mb),
                "suggestion": "进程内存使用接近阈值，建议优化内存使用或增加内存阈值"
            })
        
        # 2. 缓存效率建议
        if cache_hit_rate < 0.3 and data_cache_items > 10:
            suggestions.append({
                "type": "low_cache_efficiency",
                "priority": "medium",
                "cache_hit_rate": cache_hit_rate,
                "suggestion": "缓存命中率较低，建议优化缓存策略或清理不常用的缓存项"
            })
        
        # 3. CPU使用建议
        if cpu_avg > 80:
            suggestions.append({
                "type": "high_cpu_usage",
                "priority": "medium",
                "cpu_avg": cpu_avg,
                "suggestion": "CPU平均使用率较高，建议优化计算密集型任务或考虑任务调度"
            })
        
        # 4. 系统内存建议
        if system_memory_percent > 85:
            suggestions.append({
                "type": "high_system_memory_usage",
                "priority": "high",
                "system_memory_percent": system_memory_percent,
                "suggestion": "系统内存使用率过高，可能影响整体性能，建议增加系统内存或优化内存使用"
            })
        
        return suggestions
    
    def clear_all_cache(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.data_cache.clear()
            self.model_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("已清空所有缓存")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        status = {
            "monitoring_running": self.monitor_running,
            "dynamic_scheduler_available": DYNAMIC_SCHEDULER_AVAILABLE and self.dynamic_scheduler is not None,
            "memory_strategy": self.memory_strategy.value,
            "optimization_enabled": self.config.get("auto_optimization", True)
        }
        
        # 添加性能报告
        try:
            status["performance"] = self.get_performance_report()
        except:
            status["performance"] = {"error": "无法获取性能报告"}
        
        # 添加内存报告
        try:
            status["memory"] = self.get_memory_report()
        except:
            status["memory"] = {"error": "无法获取内存报告"}
        
        # 添加优化建议
        try:
            status["suggestions"] = self.get_optimization_suggestions()
        except:
            status["suggestions"] = []
        
        return status


def create_resource_scheduler_manager(config: Optional[Dict[str, Any]] = None) -> ResourceSchedulerManager:
    """创建资源调度管理器实例"""
    return ResourceSchedulerManager(config)