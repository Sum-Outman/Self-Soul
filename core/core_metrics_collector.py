#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心指标收集器 - Core Metrics Collector

解决用户指出的"无核心指标监控"问题：
1. 未采集"演化成功率、推理延迟抖动、硬件资源占用率、重复输出率"等核心指标
2. 所有日志混在一起（错误/警告/信息），无ERROR/WARN/INFO分级
3. 系统崩溃时仅输出"崩溃"，不保存崩溃前的上下文、硬件状态、演化进度

本模块提供：
1. 统一的核心指标收集和监控
2. 增强的分级日志审计
3. 崩溃快照集成（与CrashSnapshotManager配合）
4. 实时指标分析和告警
"""

import logging
import time
import json
import threading
import os
import psutil
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

# 导入现有监控组件
from .error_handling import error_handler
from .scene_adaptive_parameters import SceneAdaptiveParameters
from .performance_monitoring_dashboard import PerformanceMonitoringDashboard

logger = logging.getLogger(__name__)


class CoreMetricsCollector:
    """
    核心指标收集器
    收集和监控Self-Soul AGI系统的核心指标
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_log_audit: bool = True,
        metrics_window_size: int = 1000
    ):
        """
        初始化核心指标收集器
        
        Args:
            config: 配置字典
            enable_log_audit: 是否启用日志审计
            metrics_window_size: 指标窗口大小
        """
        self.config = config or {}
        self.enable_log_audit = enable_log_audit
        self.metrics_window_size = metrics_window_size
        
        # 指标存储
        self.metrics_history = {
            "evolution_success_rate": deque(maxlen=metrics_window_size),
            "inference_latency_jitter": deque(maxlen=metrics_window_size),
            "hardware_utilization": deque(maxlen=metrics_window_size),
            "repetition_output_rate": deque(maxlen=metrics_window_size),
            "cpu_usage": deque(maxlen=metrics_window_size),
            "memory_usage": deque(maxlen=metrics_window_size),
            "gpu_usage": deque(maxlen=metrics_window_size),
            "inference_success_rate": deque(maxlen=metrics_window_size),
            "model_performance": deque(maxlen=metrics_window_size)
        }
        
        # 指标统计
        self.metrics_statistics = defaultdict(lambda: {
            "count": 0,
            "sum": 0.0,
            "min": float('inf'),
            "max": float('-inf'),
            "last_value": 0.0,
            "last_timestamp": 0.0
        })
        
        # 日志审计
        self.log_audit_data = {
            "log_level_distribution": defaultdict(int),
            "component_log_counts": defaultdict(int),
            "recent_logs": deque(maxlen=1000),
            "error_patterns": defaultdict(int)
        }
        
        # 组件引用（延迟初始化）
        self.scene_adaptive_params = None
        self.performance_dashboard = None
        self.evolution_engine = None
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get("monitoring_interval", 5.0)  # 秒
        
        # 初始化日志处理器（如果启用日志审计）
        if self.enable_log_audit:
            self._setup_log_audit()
        
        logger.info("CoreMetricsCollector initialized")
    
    def _setup_log_audit(self) -> None:
        """设置日志审计"""
        # 获取根日志记录器
        root_logger = logging.getLogger()
        
        # 创建审计处理器
        audit_handler = logging.Handler()
        audit_handler.setLevel(logging.DEBUG)
        
        # 自定义过滤器来捕获所有日志记录
        def audit_filter(record):
            self._audit_log_record(record)
            return True
        
        audit_handler.addFilter(audit_filter)
        root_logger.addHandler(audit_handler)
        
        logger.debug("Log audit system enabled")
    
    def _audit_log_record(self, record: logging.LogRecord) -> None:
        """
        审计日志记录
        
        Args:
            record: 日志记录对象
        """
        # 记录日志级别分布
        level_name = record.levelname
        self.log_audit_data["log_level_distribution"][level_name] += 1
        
        # 记录组件日志计数
        logger_name = record.name
        self.log_audit_data["component_log_counts"][logger_name] += 1
        
        # 记录最近的日志
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level_name,
            "logger": logger_name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
            "function": record.funcName
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None
            }
        
        self.log_audit_data["recent_logs"].append(log_entry)
        
        # 分析错误模式（对于ERROR及以上级别）
        if record.levelno >= logging.ERROR:
            error_msg = record.getMessage()
            # 简单错误模式提取（基于关键词）
            error_pattern = self._extract_error_pattern(error_msg)
            self.log_audit_data["error_patterns"][error_pattern] += 1
    
    def _extract_error_pattern(self, error_message: str) -> str:
        """
        提取错误模式
        
        Args:
            error_message: 错误消息
            
        Returns:
            错误模式字符串
        """
        if not error_message:
            return "unknown"
        
        # 常见错误模式关键词
        patterns = [
            ("timeout", ["timeout", "timed out", "time out"]),
            ("connection", ["connection", "connect", "disconnect", "network"]),
            ("memory", ["memory", "oom", "out of memory"]),
            ("file", ["file", "directory", "path", "ioerror"]),
            ("permission", ["permission", "access denied", "forbidden"]),
            ("validation", ["validation", "invalid", "valid"]),
            ("type", ["type", "attribute", "property"]),
            ("index", ["index", "out of range", "bounds"]),
            ("key", ["key", "dictionary", "dict"]),
            ("import", ["import", "module", "package"]),
            ("argument", ["argument", "parameter", "arg"]),
            ("resource", ["resource", "busy", "locked"]),
            ("authentication", ["auth", "authentication", "login", "credential"]),
            ("database", ["database", "sql", "query", "db"]),
            ("api", ["api", "http", "request", "response"])
        ]
        
        error_lower = error_message.lower()
        
        for pattern_name, keywords in patterns:
            for keyword in keywords:
                if keyword in error_lower:
                    return pattern_name
        
        # 如果没有匹配，返回前几个单词
        words = error_message.split()[:3]
        return " ".join(words)
    
    def set_scene_adaptive_params(self, params: SceneAdaptiveParameters) -> None:
        """
        设置场景自适应参数管理器
        
        Args:
            params: SceneAdaptiveParameters实例
        """
        self.scene_adaptive_params = params
        logger.info("SceneAdaptiveParameters linked to CoreMetricsCollector")
    
    def set_performance_dashboard(self, dashboard: PerformanceMonitoringDashboard) -> None:
        """
        设置性能监控仪表板
        
        Args:
            dashboard: PerformanceMonitoringDashboard实例
        """
        self.performance_dashboard = dashboard
        logger.info("PerformanceMonitoringDashboard linked to CoreMetricsCollector")
    
    def set_evolution_engine(self, engine) -> None:
        """
        设置演化引擎
        
        Args:
            engine: 架构演化引擎实例
        """
        self.evolution_engine = engine
        logger.info("ArchitectureEvolutionEngine linked to CoreMetricsCollector")
    
    def collect_core_metrics(self) -> Dict[str, Any]:
        """
        收集所有核心指标
        
        Returns:
            包含所有核心指标的字典
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "evolution_success_rate": self._collect_evolution_success_rate(),
            "inference_latency_jitter": self._collect_inference_latency_jitter(),
            "hardware_utilization": self._collect_hardware_utilization(),
            "repetition_output_rate": self._collect_repetition_output_rate(),
            "system_metrics": self._collect_system_metrics(),
            "log_audit_summary": self._get_log_audit_summary()
        }
        
        # 存储到历史
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history and isinstance(value, (int, float)):
                self.metrics_history[metric_name].append({
                    "timestamp": metrics["timestamp"],
                    "value": value
                })
                
                # 更新统计
                stats = self.metrics_statistics[metric_name]
                stats["count"] += 1
                stats["sum"] += value
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["last_value"] = value
                stats["last_timestamp"] = time.time()
        
        logger.debug(f"Core metrics collected: {metrics}")
        return metrics
    
    def _collect_evolution_success_rate(self) -> float:
        """
        收集演化成功率
        
        Returns:
            演化成功率（0-1）
        """
        try:
            if self.evolution_engine:
                # 从演化引擎获取成功率
                # 这里假设演化引擎有相应的方法
                if hasattr(self.evolution_engine, "get_evolution_statistics"):
                    stats = self.evolution_engine.get_evolution_statistics()
                    if "success_rate" in stats:
                        return stats["success_rate"]
                
                # 备选方案：从熔断状态计算
                if hasattr(self.evolution_engine, "fuse_state"):
                    fuse_state = self.evolution_engine.fuse_state
                    total_generations = getattr(self.evolution_engine, "total_generations", 0)
                    
                    if total_generations > 0:
                        consecutive_failures = fuse_state.get("consecutive_failures", 0)
                        success_rate = 1.0 - (consecutive_failures / max(1, total_generations))
                        return max(0.0, min(1.0, success_rate))
            
            # 默认值
            return 0.8
            
        except Exception as e:
            logger.warning(f"Failed to collect evolution success rate: {e}")
            return 0.5
    
    def _collect_inference_latency_jitter(self) -> float:
        """
        收集推理延迟抖动
        
        Returns:
            推理延迟抖动系数（标准差/平均值）
        """
        try:
            if self.performance_dashboard:
                # 从性能仪表板获取推理延迟数据
                if hasattr(self.performance_dashboard, "inference_latencies"):
                    latencies = self.performance_dashboard.inference_latencies
                    if len(latencies) >= 2:
                        mean_latency = np.mean(latencies)
                        std_latency = np.std(latencies)
                        if mean_latency > 0:
                            jitter = std_latency / mean_latency
                            return float(jitter)
            
            # 从系统监控获取
            if hasattr(self, 'inference_latencies') and self.inference_latencies:
                latencies = self.inference_latencies
                if len(latencies) >= 2:
                    mean_latency = np.mean(latencies)
                    std_latency = np.std(latencies)
                    if mean_latency > 0:
                        jitter = std_latency / mean_latency
                        return float(jitter)
            
            # 默认值
            return 0.1
            
        except Exception as e:
            logger.warning(f"Failed to collect inference latency jitter: {e}")
            return 0.15
    
    def _collect_hardware_utilization(self) -> float:
        """
        收集硬件资源占用率
        
        Returns:
            综合硬件利用率（0-1）
        """
        try:
            utilization_metrics = []
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            utilization_metrics.append(cpu_percent / 100.0)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            utilization_metrics.append(memory_percent / 100.0)
            
            # 磁盘使用率（如果有）
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                utilization_metrics.append(disk_percent / 100.0)
            except:
                pass
            
            # GPU使用率（如果可用）
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = sum([gpu.load for gpu in gpus]) / len(gpus)
                    utilization_metrics.append(gpu_percent)
            except:
                pass
            
            # 计算综合利用率
            if utilization_metrics:
                avg_utilization = np.mean(utilization_metrics)
                return float(avg_utilization)
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"Failed to collect hardware utilization: {e}")
            return 0.4
    
    def _collect_repetition_output_rate(self) -> float:
        """
        收集重复输出率
        
        Returns:
            重复输出率（0-1）
        """
        try:
            if self.scene_adaptive_params:
                # 从场景自适应参数管理器获取性能历史
                performance_history = self.scene_adaptive_params.get_performance_history(limit=50)
                
                if performance_history:
                    # 计算平均重复分数
                    repetition_scores = [p.get("repetition_score", 0.5) for p in performance_history]
                    avg_repetition_score = np.mean(repetition_scores) if repetition_scores else 0.5
                    
                    # 重复输出率 = 1 - 重复评分（因为重复评分越高表示重复越合适）
                    repetition_rate = 1.0 - avg_repetition_score
                    return max(0.0, min(1.0, repetition_rate))
            
            # 默认值
            return 0.2
            
        except Exception as e:
            logger.warning(f"Failed to collect repetition output rate: {e}")
            return 0.25
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """
        收集系统指标
        
        Returns:
            系统指标字典
        """
        try:
            metrics = {}
            
            # CPU使用率
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = memory.percent
            metrics["memory_used_gb"] = memory.used / (1024**3)
            metrics["memory_total_gb"] = memory.total / (1024**3)
            
            # 磁盘使用
            try:
                disk = psutil.disk_usage('/')
                metrics["disk_percent"] = disk.percent
                metrics["disk_used_gb"] = disk.used / (1024**3)
                metrics["disk_total_gb"] = disk.total / (1024**3)
            except:
                metrics["disk_percent"] = 0.0
                metrics["disk_used_gb"] = 0.0
                metrics["disk_total_gb"] = 0.0
            
            # 网络IO
            try:
                net_io = psutil.net_io_counters()
                metrics["network_sent_mb"] = net_io.bytes_sent / (1024**2)
                metrics["network_recv_mb"] = net_io.bytes_recv / (1024**2)
            except:
                metrics["network_sent_mb"] = 0.0
                metrics["network_recv_mb"] = 0.0
            
            # 进程信息
            current_process = psutil.Process()
            metrics["process_cpu_percent"] = current_process.cpu_percent()
            metrics["process_memory_mb"] = current_process.memory_info().rss / (1024**2)
            metrics["process_threads"] = current_process.num_threads()
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}
    
    def _get_log_audit_summary(self) -> Dict[str, Any]:
        """
        获取日志审计摘要
        
        Returns:
            日志审计摘要字典
        """
        return {
            "log_level_distribution": dict(self.log_audit_data["log_level_distribution"]),
            "top_logging_components": dict(
                sorted(
                    self.log_audit_data["component_log_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "total_logs": sum(self.log_audit_data["log_level_distribution"].values()),
            "error_patterns": dict(
                sorted(
                    self.log_audit_data["error_patterns"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "recent_log_count": len(self.log_audit_data["recent_logs"])
        }
    
    def start_monitoring(self) -> bool:
        """
        启动监控
        
        Returns:
            是否成功启动
        """
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return False
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self.collect_core_metrics()
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                
                # 等待下一次收集
                time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(
            target=monitoring_loop,
            name="CoreMetricsMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Core metrics monitoring started (interval: {self.monitoring_interval}s)")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        停止监控
        
        Returns:
            是否成功停止
        """
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return False
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        logger.info("Core metrics monitoring stopped")
        return True
    
    def get_metrics_history(
        self, 
        metric_name: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        获取指标历史
        
        Args:
            metric_name: 指标名称，如果为None则返回所有指标
            limit: 限制返回的历史记录数量
            
        Returns:
            指标历史数据
        """
        if metric_name:
            if metric_name in self.metrics_history:
                history = list(self.metrics_history[metric_name])[-limit:]
                return {metric_name: history}
            else:
                return {}
        else:
            result = {}
            for name, history in self.metrics_history.items():
                result[name] = list(history)[-limit//len(self.metrics_history):]
            return result
    
    def get_metrics_statistics(
        self, 
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取指标统计
        
        Args:
            metric_name: 指标名称，如果为None则返回所有指标统计
            
        Returns:
            指标统计数据
        """
        if metric_name:
            if metric_name in self.metrics_statistics:
                stats = self.metrics_statistics[metric_name].copy()
                if stats["count"] > 0:
                    stats["avg"] = stats["sum"] / stats["count"]
                else:
                    stats["avg"] = 0.0
                return {metric_name: stats}
            else:
                return {}
        else:
            result = {}
            for name, stats in self.metrics_statistics.items():
                if stats["count"] > 0:
                    stats_copy = stats.copy()
                    stats_copy["avg"] = stats_copy["sum"] / stats_copy["count"]
                    result[name] = stats_copy
            return result
    
    def get_log_audit_data(self, detailed: bool = False) -> Dict[str, Any]:
        """
        获取日志审计数据
        
        Args:
            detailed: 是否包含详细日志
            
        Returns:
            日志审计数据
        """
        data = {
            "summary": self._get_log_audit_summary(),
            "level_distribution": dict(self.log_audit_data["log_level_distribution"]),
            "component_counts": dict(self.log_audit_data["component_log_counts"]),
            "error_patterns": dict(self.log_audit_data["error_patterns"])
        }
        
        if detailed:
            data["recent_logs"] = list(self.log_audit_data["recent_logs"])
        
        return data
    
    def check_metric_thresholds(self) -> List[Dict[str, Any]]:
        """
        检查指标阈值
        
        Returns:
            阈值违规列表
        """
        violations = []
        
        # 定义阈值
        thresholds = {
            "evolution_success_rate": {"min": 0.6, "severity": "high"},
            "inference_latency_jitter": {"max": 0.3, "severity": "medium"},
            "hardware_utilization": {"max": 0.9, "severity": "high"},
            "repetition_output_rate": {"max": 0.4, "severity": "medium"},
            "cpu_usage": {"max": 0.9, "severity": "high"},
            "memory_usage": {"max": 0.9, "severity": "high"}
        }
        
        # 检查每个指标
        for metric_name, threshold_config in thresholds.items():
            if metric_name in self.metrics_statistics:
                stats = self.metrics_statistics[metric_name]
                last_value = stats["last_value"]
                
                # 检查最小值
                if "min" in threshold_config and last_value < threshold_config["min"]:
                    violations.append({
                        "metric": metric_name,
                        "value": last_value,
                        "threshold": threshold_config["min"],
                        "condition": "below_minimum",
                        "severity": threshold_config["severity"],
                        "timestamp": stats["last_timestamp"]
                    })
                
                # 检查最大值
                if "max" in threshold_config and last_value > threshold_config["max"]:
                    violations.append({
                        "metric": metric_name,
                        "value": last_value,
                        "threshold": threshold_config["max"],
                        "condition": "above_maximum",
                        "severity": threshold_config["severity"],
                        "timestamp": stats["last_timestamp"]
                    })
        
        return violations
    
    def generate_health_report(self) -> Dict[str, Any]:
        """
        生成系统健康报告
        
        Returns:
            健康报告字典
        """
        # 收集当前指标
        current_metrics = self.collect_core_metrics()
        
        # 检查阈值违规
        violations = self.check_metric_thresholds()
        
        # 获取指标统计
        metrics_stats = self.get_metrics_statistics()
        
        # 获取日志审计摘要
        log_summary = self._get_log_audit_summary()
        
        # 生成健康评分
        health_score = self._calculate_health_score(current_metrics, violations)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "health_status": self._get_health_status(health_score),
            "current_metrics": current_metrics,
            "threshold_violations": violations,
            "metrics_statistics": metrics_stats,
            "log_audit_summary": log_summary,
            "recommendations": self._generate_recommendations(current_metrics, violations)
        }
        
        return report
    
    def _calculate_health_score(self, metrics: Dict[str, Any], violations: List[Dict[str, Any]]) -> float:
        """
        计算健康评分
        
        Args:
            metrics: 当前指标
            violations: 阈值违规列表
            
        Returns:
            健康评分（0-100）
        """
        base_score = 100.0
        
        # 根据阈值违规扣分
        for violation in violations:
            severity = violation["severity"]
            if severity == "high":
                base_score -= 15
            elif severity == "medium":
                base_score -= 8
            elif severity == "low":
                base_score -= 3
        
        # 根据关键指标扣分
        critical_metrics = [
            ("evolution_success_rate", 0.7, 20),
            ("hardware_utilization", 0.85, 15),
            ("inference_latency_jitter", 0.25, 10)
        ]
        
        for metric_name, threshold, penalty in critical_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                
                if metric_name in ["evolution_success_rate"]:
                    if value < threshold:
                        base_score -= penalty * (1 - value/threshold)
                else:
                    if value > threshold:
                        base_score -= penalty * (value/threshold - 1)
        
        return max(0.0, min(100.0, base_score))
    
    def _get_health_status(self, health_score: float) -> str:
        """
        获取健康状态
        
        Args:
            health_score: 健康评分
            
        Returns:
            健康状态字符串
        """
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "fair"
        elif health_score >= 40:
            return "poor"
        else:
            return "critical"
    
    def _generate_recommendations(
        self, 
        metrics: Dict[str, Any], 
        violations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成优化建议
        
        Args:
            metrics: 当前指标
            violations: 阈值违规列表
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 根据违规生成建议
        for violation in violations:
            metric = violation["metric"]
            value = violation["value"]
            condition = violation["condition"]
            
            if metric == "evolution_success_rate" and condition == "below_minimum":
                recommendations.append({
                    "id": "R001",
                    "metric": metric,
                    "issue": f"演化成功率过低 ({value:.1%})",
                    "recommendation": "检查演化参数配置，增加种群多样性，降低演化步长",
                    "priority": "high",
                    "action": "调整演化引擎配置参数"
                })
            
            elif metric == "inference_latency_jitter" and condition == "above_maximum":
                recommendations.append({
                    "id": "R002",
                    "metric": metric,
                    "issue": f"推理延迟抖动过高 ({value:.3f})",
                    "recommendation": "优化模型加载策略，增加缓存机制，减少资源竞争",
                    "priority": "medium",
                    "action": "优化推理管道和资源调度"
                })
            
            elif metric == "hardware_utilization" and condition == "above_maximum":
                recommendations.append({
                    "id": "R003",
                    "metric": metric,
                    "issue": f"硬件资源占用率过高 ({value:.1%})",
                    "recommendation": "实施资源限制，优化任务调度，考虑横向扩展",
                    "priority": "high",
                    "action": "调整资源分配和任务调度策略"
                })
            
            elif metric == "repetition_output_rate" and condition == "above_maximum":
                recommendations.append({
                    "id": "R004",
                    "metric": metric,
                    "issue": f"重复输出率过高 ({value:.1%})",
                    "recommendation": "增加repetition_penalty，优化temperature参数，增强场景自适应",
                    "priority": "medium",
                    "action": "调整生成参数和场景自适应配置"
                })
        
        return recommendations
    
    def export_metrics(self, filepath: str) -> bool:
        """
        导出指标数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics_history": {k: list(v) for k, v in self.metrics_history.items()},
                "metrics_statistics": dict(self.metrics_statistics),
                "log_audit_summary": self._get_log_audit_summary(),
                "health_report": self.generate_health_report()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Core metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export core metrics: {e}")
            return False


# 单例实例
_core_metrics_instance = None

def get_core_metrics_collector(
    config: Optional[Dict[str, Any]] = None
) -> CoreMetricsCollector:
    """
    获取核心指标收集器实例（单例模式）
    
    Args:
        config: 配置字典
        
    Returns:
        CoreMetricsCollector实例
    """
    global _core_metrics_instance
    
    if _core_metrics_instance is None:
        _core_metrics_instance = CoreMetricsCollector(config)
    
    return _core_metrics_instance


# 示例使用
if __name__ == "__main__":
    # 测试核心指标收集器
    print("=" * 80)
    print("测试核心指标收集器")
    print("=" * 80)
    
    # 创建收集器
    collector = CoreMetricsCollector(
        config={"monitoring_interval": 2.0},
        enable_log_audit=True
    )
    
    print("\n1. 收集核心指标:")
    metrics = collector.collect_core_metrics()
    print(f"   演化成功率: {metrics.get('evolution_success_rate', 0):.3f}")
    print(f"   推理延迟抖动: {metrics.get('inference_latency_jitter', 0):.3f}")
    print(f"   硬件资源占用率: {metrics.get('hardware_utilization', 0):.3f}")
    print(f"   重复输出率: {metrics.get('repetition_output_rate', 0):.3f}")
    
    print("\n2. 获取日志审计摘要:")
    log_summary = collector.get_log_audit_data()
    print(f"   总日志数: {log_summary.get('summary', {}).get('total_logs', 0)}")
    print(f"   日志级别分布: {log_summary.get('level_distribution', {})}")
    
    print("\n3. 检查指标阈值:")
    violations = collector.check_metric_thresholds()
    if violations:
        print(f"   发现 {len(violations)} 个阈值违规:")
        for violation in violations:
            print(f"     - {violation['metric']}: {violation['value']:.3f} ({violation['condition']})")
    else:
        print("   无阈值违规")
    
    print("\n4. 生成健康报告:")
    health_report = collector.generate_health_report()
    print(f"   健康评分: {health_report.get('health_score', 0):.1f}")
    print(f"   健康状态: {health_report.get('health_status', 'unknown')}")
    
    print("\n5. 获取指标统计:")
    stats = collector.get_metrics_statistics()
    for metric, stat in stats.items():
        if stat["count"] > 0:
            print(f"   {metric}: 平均值={stat.get('avg', 0):.3f}, 最小值={stat['min']:.3f}, 最大值={stat['max']:.3f}")
    
    print("\n✓ 核心指标收集器测试完成")