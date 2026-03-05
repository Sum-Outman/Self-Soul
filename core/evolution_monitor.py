#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演化监控系统 - Evolution Monitoring System

提供演化过程的实时监控、告警和报告功能：
1. 演化任务状态监控
2. 性能指标收集和分析
3. 实时告警和通知
4. 演化报告生成
5. 可视化数据导出

设计特点：
- 实时性：低延迟监控和告警
- 可扩展：支持多种监控指标
- 集成性：与演化管理器紧密集成
- 可视化：提供数据导出和可视化支持
"""

import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

from core.error_handling import error_handler
from core.evolution_manager import EvolutionManager, get_evolution_manager
from core.evolution_module import EvolutionModule, get_evolution_module
from core.resource_manager import ResourceManager, get_resource_manager

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetric:
    """监控指标数据类"""

    metric_id: str
    metric_type: str  # task_status, performance, resource, error, custom
    value: Any
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class AlertRule:
    """告警规则数据类"""

    rule_id: str
    metric_type: str
    condition: str  # >, <, >=, <=, ==, !=, contains, regex
    threshold: Any
    severity: str  # info, warning, error, critical
    message_template: str
    cooldown_seconds: float = 300  # 5分钟冷却时间
    last_triggered: Optional[float] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class Alert:
    """告警数据类"""

    alert_id: str
    rule_id: str
    severity: str
    message: str
    triggered_at: float
    metric_value: Any
    metric_timestamp: float
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def is_resolved(self) -> bool:
        """检查告警是否已解决"""
        return self.resolved_at is not None


class EvolutionMonitor:
    """
    演化监控系统主类
    提供演化过程的实时监控和告警功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化演化监控系统

        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()

        # 演化管理器和模块
        self.evolution_manager = get_evolution_manager()
        self.evolution_module = get_evolution_module()
        
        # 资源管理器
        self.resource_manager = get_resource_manager(self.config.get("resource_manager_config"))

        # 指标存储
        self.metrics: Dict[str, List[MonitoringMetric]] = defaultdict(list)
        self.metrics_lock = threading.Lock()
        self.max_metrics_per_type = self.config.get("max_metrics_per_type", 1000)

        # 告警规则
        self.alert_rules: Dict[str, AlertRule] = {}
        self._initialize_default_alert_rules()

        # 告警存储
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_alert_history = self.config.get("max_alert_history", 1000)
        self.alerts_lock = threading.Lock()

        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get("monitoring_interval", 5.0)

        # 性能统计
        self.performance_stats = {
            "total_metrics_collected": 0,
            "total_alerts_triggered": 0,
            "metrics_collection_rate": 0.0,
            "last_collection_time": None,
        }

        # 回调函数
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.metric_callbacks: List[Callable[[MonitoringMetric], None]] = []

        self.logger.info("演化监控系统初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring_interval": 5.0,  # 监控间隔（秒）
            "max_metrics_per_type": 1000,  # 每种指标类型最大存储数量
            "max_alert_history": 1000,  # 最大告警历史记录
            "metrics_retention_days": 7,  # 指标保留天数
            "alert_cooldown_default": 300,  # 默认告警冷却时间（秒）
            "performance_window_hours": 24,  # 性能分析窗口（小时）
            "resource_manager_config": {
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
            },
            "health_scoring": {
                "weights": {
                    "task_success_rate": 0.3,
                    "resource_health": 0.3,
                    "performance_trend": 0.2,
                    "system_stability": 0.2
                },
                "thresholds": {
                    "healthy": 0.8,
                    "warning": 0.6,
                    "critical": 0.4
                }
            },
            "reporting": {
                "generate_hourly": True,
                "generate_daily": True,
                "generate_weekly": True,
                "retain_reports_days": 30
            }
        }

    def _initialize_default_alert_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="task_failure_rate_high",
                metric_type="task_status",
                condition=">",
                threshold=0.3,  # 30%失败率
                severity="error",
                message_template="演化任务失败率过高: {value:.1%} (阈值: {threshold:.1%})",
            ),
            AlertRule(
                rule_id="evolution_timeout",
                metric_type="task_duration",
                condition=">",
                threshold=1800,  # 30分钟
                severity="warning",
                message_template="演化任务超时: {value:.0f}秒 > {threshold:.0f}秒",
            ),
            AlertRule(
                rule_id="resource_usage_high",
                metric_type="resource_usage",
                condition=">",
                threshold=0.8,  # 80%资源使用率
                severity="warning",
                message_template="资源使用率过高: {value:.1%} (阈值: {threshold:.1%})",
            ),
            AlertRule(
                rule_id="performance_degradation",
                metric_type="performance_improvement",
                condition="<",
                threshold=0.0,  # 性能下降
                severity="error",
                message_template="演化性能下降: {value:.3f} (阈值: > {threshold:.3f})",
            ),
            AlertRule(
                rule_id="consecutive_failures",
                metric_type="consecutive_failures",
                condition=">=",
                threshold=3,  # 连续3次失败
                severity="critical",
                message_template="连续演化失败: {value}次 (阈值: {threshold}次)",
            ),
            AlertRule(
                rule_id="fuse_triggered",
                metric_type="fuse_status",
                condition="==",
                threshold=True,
                severity="critical",
                message_template="演化熔断已触发: {value}",
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

        self.logger.info(f"已初始化 {len(default_rules)} 个默认告警规则")

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring_active:
            self.logger.warning("监控已在运行中")
            return

        self.monitoring_active = True

        # 启动监控线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_thread_func, name="EvolutionMonitor", daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info(f"演化监控已启动，监控间隔: {self.monitoring_interval}秒")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("演化监控已停止")

    def add_alert_rule(self, rule: AlertRule) -> bool:
        """添加告警规则"""
        try:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"告警规则已添加: {rule.rule_id}")
            return True
        except Exception as e:
            self.logger.error(f"添加告警规则失败: {str(e)}")
            return False

    def remove_alert_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"告警规则已移除: {rule_id}")
                return True
            else:
                self.logger.warning(f"告警规则不存在: {rule_id}")
                return False
        except Exception as e:
            self.logger.error(f"移除告警规则失败: {str(e)}")
            return False

    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """获取所有告警规则"""
        return [rule.to_dict() for rule in self.alert_rules.values()]

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self.alerts_lock:
            return [alert.to_dict() for alert in self.active_alerts.values()]

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警历史"""
        with self.alerts_lock:
            recent_alerts = self.alert_history[-limit:] if self.alert_history else []
            return [alert.to_dict() for alert in recent_alerts]

    def resolve_alert(
        self, alert_id: str, resolved_by: str = "system", resolution_note: str = "自动解决"
    ) -> bool:
        """解决告警"""
        try:
            with self.alerts_lock:
                if alert_id not in self.active_alerts:
                    return False

                alert = self.active_alerts[alert_id]
                alert.resolved_at = time.time()
                alert.resolved_by = resolved_by
                alert.resolution_note = resolution_note

                # 移动到历史记录
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]

                # 限制历史记录大小
                if len(self.alert_history) > self.max_alert_history:
                    self.alert_history = self.alert_history[-self.max_alert_history :]

            self.logger.info(f"告警已解决: {alert_id} (解决者: {resolved_by})")
            return True

        except Exception as e:
            self.logger.error(f"解决告警失败: {str(e)}")
            return False

    def add_metric(self, metric: MonitoringMetric):
        """添加监控指标"""
        try:
            metric_type = metric.metric_type

            with self.metrics_lock:
                # 添加到指标列表
                self.metrics[metric_type].append(metric)

                # 限制指标数量
                if len(self.metrics[metric_type]) > self.max_metrics_per_type:
                    self.metrics[metric_type] = self.metrics[metric_type][
                        -self.max_metrics_per_type :
                    ]

            # 更新统计
            self.performance_stats["total_metrics_collected"] += 1
            self.performance_stats["last_collection_time"] = time.time()

            # 检查告警规则
            self._check_alert_rules(metric)

            # 调用指标回调
            for callback in self.metric_callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    self.logger.error(f"指标回调执行失败: {str(e)}")

        except Exception as e:
            self.logger.error(f"添加监控指标失败: {str(e)}")

    def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取监控指标"""
        try:
            with self.metrics_lock:
                if metric_type not in self.metrics:
                    return []

                metrics = self.metrics[metric_type]

                # 时间过滤
                if start_time is not None:
                    metrics = [m for m in metrics if m.timestamp >= start_time]
                if end_time is not None:
                    metrics = [m for m in metrics if m.timestamp <= end_time]

                # 限制数量
                metrics = metrics[-limit:] if metrics else []

                return [m.to_dict() for m in metrics]

        except Exception as e:
            self.logger.error(f"获取监控指标失败: {str(e)}")
            return []

    def get_metric_statistics(
        self,
        metric_type: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """获取指标统计信息"""
        try:
            metrics = self.get_metrics(metric_type, start_time, end_time, limit=10000)

            if not metrics:
                return {
                    "count": 0,
                    "metric_type": metric_type,
                    "time_range": {"start": start_time, "end": end_time},
                }

            # 提取数值型指标
            numeric_values = []
            for metric in metrics:
                value = metric.get("value")
                if isinstance(value, (int, float)):
                    numeric_values.append(value)

            if numeric_values:
                stats = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": statistics.mean(numeric_values),
                    "median": statistics.median(numeric_values),
                    "stddev": statistics.stdev(numeric_values)
                    if len(numeric_values) > 1
                    else 0.0,
                }
            else:
                stats = {"count": len(metrics)}

            # 添加时间信息
            timestamps = [m.get("timestamp") for m in metrics if m.get("timestamp")]
            if timestamps:
                stats["time_range"] = {
                    "start": min(timestamps),
                    "end": max(timestamps),
                    "duration": max(timestamps) - min(timestamps),
                }

            stats["metric_type"] = metric_type
            return stats

        except Exception as e:
            self.logger.error(f"获取指标统计失败: {str(e)}")
            return {"error": str(e), "metric_type": metric_type}

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        try:
            # 获取演化管理器统计
            manager_stats = self.evolution_manager.get_statistics()

            # 获取演化模块状态
            module_status = self.evolution_module.get_evolution_status()

            # 计算关键指标
            current_time = time.time()
            one_hour_ago = current_time - 3600

            # 任务成功率
            task_metrics = self.get_metrics("task_status", start_time=one_hour_ago)
            total_tasks = len(task_metrics)
            successful_tasks = len(
                [
                    m
                    for m in task_metrics
                    if m.get("value", {}).get("status") == "completed"
                ]
            )
            failure_rate = (
                1.0 - (successful_tasks / total_tasks) if total_tasks > 0 else 0.0
            )

            # 性能改进
            performance_metrics = self.get_metrics(
                "performance_improvement", start_time=one_hour_ago
            )
            performance_values = [
                m.get("value")
                for m in performance_metrics
                if isinstance(m.get("value"), (int, float))
            ]
            avg_performance_improvement = (
                statistics.mean(performance_values) if performance_values else 0.0
            )

            # 活跃告警
            active_alerts = self.get_active_alerts()
            critical_alerts = [
                a for a in active_alerts if a.get("severity") == "critical"
            ]

            # 资源使用率
            resource_metrics = self.get_metrics(
                "resource_usage", start_time=current_time - 300
            )  # 最近5分钟
            resource_values = [
                m.get("value")
                for m in resource_metrics
                if isinstance(m.get("value"), (int, float))
            ]
            avg_resource_usage = (
                statistics.mean(resource_values) if resource_values else 0.0
            )

            dashboard = {
                "timestamp": current_time,
                "evolution_manager": {
                    "active_tasks": manager_stats.get("active_tasks", 0),
                    "pending_tasks": manager_stats.get("pending_tasks", 0),
                    "total_tasks": manager_stats.get("total_tasks", 0),
                    "fuse_triggered": manager_stats.get("fuse_triggered", False),
                },
                "evolution_module": {
                    "evolution_active": module_status.get("module_status", {}).get(
                        "evolution_active", False
                    ),
                    "history_size": module_status.get("module_status", {}).get(
                        "history_size", 0
                    ),
                    "best_fitness": module_status.get("engine_status", {}).get(
                        "best_fitness", 0.0
                    ),
                },
                "key_metrics": {
                    "task_failure_rate": failure_rate,
                    "average_performance_improvement": avg_performance_improvement,
                    "average_resource_usage": avg_resource_usage,
                    "metrics_collection_rate": self.performance_stats.get(
                        "metrics_collection_rate", 0.0
                    ),
                },
                "alerts": {
                    "total_active": len(active_alerts),
                    "critical": len(critical_alerts),
                    "recent_alerts": self.get_alert_history(limit=10),
                },
                "monitoring_stats": self.performance_stats,
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"获取监控仪表板失败: {str(e)}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "monitoring_stats": self.performance_stats,
            }

    def generate_evolution_report(
        self, start_time: float, end_time: float
    ) -> Dict[str, Any]:
        """生成演化报告"""
        try:
            report_id = f"report_{int(time.time())}"

            # 收集报告数据
            tasks = self.evolution_manager.get_all_tasks()
            tasks_in_period = [
                t for t in tasks if start_time <= t.get("created_at", 0) <= end_time
            ]

            # 计算任务统计
            total_tasks = len(tasks_in_period)
            completed_tasks = len(
                [t for t in tasks_in_period if t.get("status") == "completed"]
            )
            failed_tasks = len(
                [t for t in tasks_in_period if t.get("status") == "failed"]
            )
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0

            # 收集性能指标
            performance_metrics = self.get_metrics(
                "performance_improvement", start_time, end_time
            )
            performance_values = [
                m.get("value")
                for m in performance_metrics
                if isinstance(m.get("value"), (int, float))
            ]

            # 收集告警
            alerts_in_period = []
            with self.alerts_lock:
                for alert in self.alert_history:
                    if start_time <= alert.triggered_at <= end_time:
                        alerts_in_period.append(alert)

            # 生成报告
            report = {
                "report_id": report_id,
                "generated_at": time.time(),
                "time_period": {
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                },
                "task_statistics": {
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "success_rate": success_rate,
                    "average_duration": self._calculate_average_duration(
                        tasks_in_period
                    ),
                },
                "performance_statistics": {
                    "total_evolutions": len(performance_values),
                    "average_improvement": statistics.mean(performance_values)
                    if performance_values
                    else 0.0,
                    "max_improvement": max(performance_values)
                    if performance_values
                    else 0.0,
                    "min_improvement": min(performance_values)
                    if performance_values
                    else 0.0,
                },
                "alert_statistics": {
                    "total_alerts": len(alerts_in_period),
                    "critical_alerts": len(
                        [a for a in alerts_in_period if a.severity == "critical"]
                    ),
                    "error_alerts": len(
                        [a for a in alerts_in_period if a.severity == "error"]
                    ),
                    "warning_alerts": len(
                        [a for a in alerts_in_period if a.severity == "warning"]
                    ),
                },
                "resource_usage": self.get_metric_statistics(
                    "resource_usage", start_time, end_time
                ),
                "key_insights": self._generate_insights(
                    tasks_in_period, performance_values, alerts_in_period
                ),
                "recommendations": self._generate_recommendations(
                    tasks_in_period, performance_values
                ),
            }

            self.logger.info(f"演化报告已生成: {report_id} (时间段: {start_time} - {end_time})")
            return report

        except Exception as e:
            self.logger.error(f"生成演化报告失败: {str(e)}")
            return {
                "report_id": f"error_report_{int(time.time())}",
                "error": str(e),
                "timestamp": time.time(),
            }

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)

    def add_metric_callback(self, callback: Callable[[MonitoringMetric], None]):
        """添加指标回调函数"""
        self.metric_callbacks.append(callback)

    def _monitoring_thread_func(self):
        """监控线程函数"""
        thread_name = threading.current_thread().name

        while self.monitoring_active:
            try:
                start_time = time.time()

                # 收集演化管理器状态
                self._collect_manager_metrics()

                # 收集演化模块状态
                self._collect_module_metrics()

                # 收集系统资源
                self._collect_resource_metrics()

                # 清理旧指标
                self._cleanup_old_metrics()

                # 计算收集速率
                collection_time = time.time() - start_time
                self.performance_stats["metrics_collection_rate"] = (
                    self.performance_stats["total_metrics_collected"]
                    / (
                        time.time()
                        - self.performance_stats.get("initial_time", start_time)
                    )
                    if self.performance_stats.get("initial_time")
                    else 0.0
                )

                if not hasattr(self.performance_stats, "initial_time"):
                    self.performance_stats["initial_time"] = start_time

                # 休眠直到下一个监控周期
                sleep_time = max(0.1, self.monitoring_interval - collection_time)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"{thread_name} 监控线程异常: {str(e)}", exc_info=True)
                time.sleep(self.monitoring_interval)

    def _collect_manager_metrics(self):
        """收集演化管理器指标"""
        try:
            stats = self.evolution_manager.get_statistics()

            # 任务状态指标
            task_metric = MonitoringMetric(
                metric_id=f"manager_stats_{int(time.time())}",
                metric_type="task_status",
                value=stats,
                timestamp=time.time(),
                tags={"source": "evolution_manager"},
            )
            self.add_metric(task_metric)

            # 失败率指标
            total_tasks = stats.get("total_tasks", 0)
            failed_tasks = stats.get("failed_tasks", 0)
            failure_rate = failed_tasks / total_tasks if total_tasks > 0 else 0.0

            failure_metric = MonitoringMetric(
                metric_id=f"failure_rate_{int(time.time())}",
                metric_type="failure_rate",
                value=failure_rate,
                timestamp=time.time(),
                tags={"source": "evolution_manager"},
            )
            self.add_metric(failure_metric)

        except Exception as e:
            self.logger.error(f"收集演化管理器指标失败: {str(e)}")

    def _collect_module_metrics(self):
        """收集演化模块指标（支持增强模块）"""
        try:
            status = self.evolution_module.get_evolution_status()
            statistics = self.evolution_module.get_evolution_statistics()

            # 演化状态指标
            status_metric = MonitoringMetric(
                metric_id=f"module_status_{int(time.time())}",
                metric_type="module_status",
                value=status,
                timestamp=time.time(),
                tags={"source": "evolution_module"},
            )
            self.add_metric(status_metric)

            # 性能改进指标 - 收集所有数值型指标
            for stat_name, stat_value in statistics.items():
                if isinstance(stat_value, (int, float)):
                    # 创建性能指标
                    performance_metric = MonitoringMetric(
                        metric_id=f"performance_{stat_name}_{int(time.time())}",
                        metric_type="performance_improvement",
                        value=float(stat_value),
                        timestamp=time.time(),
                        tags={
                            "source": "evolution_module",
                            "stat_name": stat_name,
                            "module_type": type(self.evolution_module).__name__
                        },
                    )
                    self.add_metric(performance_metric)
                    
                    # 特别处理增强模块特有指标
                    if stat_name in ["best_multi_objective_score", "learning_progress"]:
                        enhanced_metric = MonitoringMetric(
                            metric_id=f"enhanced_{stat_name}_{int(time.time())}",
                            metric_type="enhanced_performance",
                            value=float(stat_value),
                            timestamp=time.time(),
                            tags={
                                "source": "enhanced_evolution_module",
                                "stat_name": stat_name,
                                "module_type": type(self.evolution_module).__name__
                            },
                        )
                        self.add_metric(enhanced_metric)
            
            # 收集算法使用统计（如果存在）
            if "algorithm_usage" in statistics and isinstance(statistics["algorithm_usage"], dict):
                algorithm_usage = statistics["algorithm_usage"]
                for algorithm_name, usage_count in algorithm_usage.items():
                    if isinstance(usage_count, (int, float)):
                        algorithm_metric = MonitoringMetric(
                            metric_id=f"algorithm_usage_{algorithm_name}_{int(time.time())}",
                            metric_type="algorithm_usage",
                            value=float(usage_count),
                            timestamp=time.time(),
                            tags={
                                "source": "evolution_module",
                                "algorithm": algorithm_name,
                                "module_type": type(self.evolution_module).__name__
                            },
                        )
                        self.add_metric(algorithm_metric)
            
            # 收集增强状态信息（如果存在）
            if "enhanced_features" in status:
                enhanced_status = status.get("enhanced_features", {})
                enhanced_status_metric = MonitoringMetric(
                    metric_id=f"enhanced_status_{int(time.time())}",
                    metric_type="enhanced_status",
                    value=enhanced_status,
                    timestamp=time.time(),
                    tags={
                        "source": "enhanced_evolution_module",
                        "module_type": type(self.evolution_module).__name__
                    },
                )
                self.add_metric(enhanced_status_metric)

        except Exception as e:
            self.logger.error(f"收集演化模块指标失败: {str(e)}")

    def _collect_resource_metrics(self):
        """收集系统资源指标"""
        try:
            import psutil

            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)

            cpu_metric = MonitoringMetric(
                metric_id=f"cpu_usage_{int(time.time())}",
                metric_type="resource_usage",
                value=cpu_percent / 100.0,  # 转换为0-1范围
                timestamp=time.time(),
                tags={"resource_type": "cpu"},
            )
            self.add_metric(cpu_metric)

            # 内存使用率
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            memory_metric = MonitoringMetric(
                metric_id=f"memory_usage_{int(time.time())}",
                metric_type="resource_usage",
                value=memory_percent / 100.0,  # 转换为0-1范围
                timestamp=time.time(),
                tags={"resource_type": "memory"},
            )
            self.add_metric(memory_metric)

        except ImportError:
            # psutil不可用，跳过资源收集
            pass
        except Exception as e:
            self.logger.error(f"收集系统资源指标失败: {str(e)}")

    def _check_alert_rules(self, metric: MonitoringMetric):
        """检查告警规则"""
        try:
            current_time = time.time()

            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue

                # 检查指标类型匹配
                if rule.metric_type != metric.metric_type:
                    continue

                # 检查冷却时间
                if (
                    rule.last_triggered
                    and (current_time - rule.last_triggered) < rule.cooldown_seconds
                ):
                    continue

                # 检查条件
                should_trigger = self._evaluate_condition(
                    metric.value, rule.condition, rule.threshold
                )

                if should_trigger:
                    # 触发告警
                    self._trigger_alert(rule, metric)

                    # 更新规则最后触发时间
                    rule.last_triggered = current_time

        except Exception as e:
            self.logger.error(f"检查告警规则失败: {str(e)}")

    def _trigger_alert(self, rule: AlertRule, metric: MonitoringMetric):
        """触发告警"""
        try:
            alert_id = f"alert_{int(time.time())}_{len(self.active_alerts)}"

            # 生成告警消息
            message = rule.message_template.format(
                value=metric.value, threshold=rule.threshold, timestamp=metric.timestamp
            )

            # 创建告警
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                severity=rule.severity,
                message=message,
                triggered_at=time.time(),
                metric_value=metric.value,
                metric_timestamp=metric.timestamp,
            )

            # 保存告警
            with self.alerts_lock:
                self.active_alerts[alert_id] = alert

            # 更新统计
            self.performance_stats["total_alerts_triggered"] += 1

            # 调用告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"告警回调执行失败: {str(e)}")

            self.logger.warning(f"告警触发: {rule.severity.upper()} - {message}")

        except Exception as e:
            self.logger.error(f"触发告警失败: {str(e)}")

    def _evaluate_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """评估条件"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            elif condition == "contains":
                return threshold in str(value)
            elif condition == "regex":
                import re

                return bool(re.search(threshold, str(value)))
            else:
                self.logger.warning(f"未知条件操作符: {condition}")
                return False
        except Exception as e:
            self.logger.error(f"评估条件失败: {str(e)}")
            return False

    def _cleanup_old_metrics(self):
        """清理旧指标"""
        try:
            retention_seconds = self.config.get("metrics_retention_days", 7) * 24 * 3600
            cutoff_time = time.time() - retention_seconds

            with self.metrics_lock:
                for metric_type in list(self.metrics.keys()):
                    # 过滤掉旧指标
                    self.metrics[metric_type] = [
                        m
                        for m in self.metrics[metric_type]
                        if m.timestamp >= cutoff_time
                    ]

        except Exception as e:
            self.logger.error(f"清理旧指标失败: {str(e)}")

    def _calculate_average_duration(self, tasks: List[Dict[str, Any]]) -> float:
        """计算平均任务持续时间"""
        durations = []
        for task in tasks:
            if task.get("started_at") and task.get("completed_at"):
                duration = task["completed_at"] - task["started_at"]
                if duration > 0:
                    durations.append(duration)

        return statistics.mean(durations) if durations else 0.0

    def _generate_insights(
        self,
        tasks: List[Dict[str, Any]],
        performance_values: List[float],
        alerts: List[Alert],
    ) -> List[str]:
        """生成洞察"""
        insights = []

        # 任务成功率洞察
        total_tasks = len(tasks)
        if total_tasks > 0:
            completed_tasks = len([t for t in tasks if t.get("status") == "completed"])
            success_rate = completed_tasks / total_tasks

            if success_rate > 0.8:
                insights.append(f"任务成功率良好: {success_rate:.1%}")
            elif success_rate < 0.5:
                insights.append(f"任务成功率较低: {success_rate:.1%}，建议检查演化配置")

        # 性能改进洞察
        if performance_values:
            avg_improvement = statistics.mean(performance_values)
            if avg_improvement > 0.1:
                insights.append(f"平均性能改进显著: +{avg_improvement:.3f}")
            elif avg_improvement < 0:
                insights.append(f"平均性能下降: {avg_improvement:.3f}，需要优化演化策略")

        # 告警洞察
        if alerts:
            critical_alerts = [a for a in alerts if a.severity == "critical"]
            if critical_alerts:
                insights.append(f"存在{len(critical_alerts)}个严重告警，需要立即处理")

        return insights

    def _generate_recommendations(
        self, tasks: List[Dict[str, Any]], performance_values: List[float]
    ) -> List[str]:
        """生成推荐"""
        recommendations = []

        # 基于失败率的推荐
        total_tasks = len(tasks)
        if total_tasks > 10:  # 有足够的数据
            failed_tasks = len([t for t in tasks if t.get("status") == "failed"])
            failure_rate = failed_tasks / total_tasks

            if failure_rate > 0.4:
                recommendations.append("任务失败率过高，建议降低演化复杂度或增加资源")
            elif failure_rate < 0.1:
                recommendations.append("任务失败率较低，可以适当增加演化难度")

        # 基于性能改进的推荐
        if performance_values and len(performance_values) > 5:
            improvements = [p for p in performance_values if p > 0]
            if len(improvements) / len(performance_values) < 0.3:
                recommendations.append("性能改进比例较低，建议调整演化目标或约束条件")

        return recommendations

    def export_metrics_to_file(self, filepath: str = "evolution_metrics.json", max_metrics: int = 1000) -> bool:
        """导出监控指标到JSON文件
        
        Args:
            filepath: 导出文件路径
            max_metrics: 每个指标类型最大导出数量
            
        Returns:
            导出是否成功
        """
        try:
            import json
            from datetime import datetime
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "monitor_id": self.monitor_id,
                "metrics": {}
            }
            
            with self.metrics_lock:
                for metric_type, metric_list in self.metrics.items():
                    # 限制每个类型的指标数量
                    metrics_to_export = metric_list[-max_metrics:] if len(metric_list) > max_metrics else metric_list
                    
                    # 转换为可序列化的字典
                    export_data["metrics"][metric_type] = [
                        {
                            "metric_id": metric.metric_id,
                            "metric_type": metric.metric_type,
                            "value": metric.value,
                            "timestamp": metric.timestamp,
                            "tags": metric.tags,
                            "metadata": metric.metadata
                        }
                        for metric in metrics_to_export
                    ]
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"成功导出 {sum(len(v) for v in export_data['metrics'].values())} 个指标到 {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出指标到文件失败: {str(e)}")
            return False


# 全局演化监控实例
_global_evolution_monitor: Optional[EvolutionMonitor] = None


def get_evolution_monitor(config: Optional[Dict[str, Any]] = None) -> EvolutionMonitor:
    """
    获取全局演化监控实例

    Args:
        config: 配置字典

    Returns:
        演化监控实例
    """
    global _global_evolution_monitor

    if _global_evolution_monitor is None:
        _global_evolution_monitor = EvolutionMonitor(config)

    return _global_evolution_monitor


def start_evolution_monitoring(config: Optional[Dict[str, Any]] = None):
    """启动全局演化监控"""
    monitor = get_evolution_monitor(config)
    monitor.start_monitoring()
    return monitor


def stop_evolution_monitoring():
    """停止全局演化监控"""
    global _global_evolution_monitor
    if _global_evolution_monitor:
        _global_evolution_monitor.stop_monitoring()


def export_evolution_metrics(filepath: str = "evolution_metrics.json", max_metrics: int = 1000) -> bool:
    """导出全局演化监控指标到文件
    
    Args:
        filepath: 导出文件路径
        max_metrics: 每个指标类型最大导出数量
        
    Returns:
        导出是否成功
    """
    global _global_evolution_monitor
    if _global_evolution_monitor is None:
        print("警告: 没有活动的演化监控实例，无法导出指标")
        return False
    
    return _global_evolution_monitor.export_metrics_to_file(filepath, max_metrics)


# 测试代码
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("演化监控系统测试")
    print("=" * 80)

    try:
        # 创建演化监控实例
        monitor = EvolutionMonitor()

        print("\n1. 测试演化监控初始化:")
        print(
            f"   配置: 监控间隔={monitor.config['monitoring_interval']}s, 最大指标数={monitor.config['max_metrics_per_type']}"
        )
        print(f"   告警规则: {len(monitor.alert_rules)}个规则已初始化")

        print("\n2. 测试指标收集:")
        # 添加测试指标
        test_metric = MonitoringMetric(
            metric_id="test_metric_1",
            metric_type="test",
            value=0.75,
            timestamp=time.time(),
            tags={"test": "true"},
        )
        monitor.add_metric(test_metric)
        print(f"   测试指标已添加: {test_metric.metric_id}")

        print("\n3. 测试告警规则:")
        rules = monitor.get_alert_rules()
        print(f"   告警规则数量: {len(rules)}")
        for rule in rules[:3]:  # 显示前3个规则
            print(
                f"   - {rule['rule_id']}: {rule['condition']} {rule['threshold']} ({rule['severity']})"
            )

        print("\n4. 测试仪表板数据:")
        dashboard = monitor.get_monitoring_dashboard()
        print(f"   仪表板数据已生成，包含{len(dashboard)}个部分")
        print(
            f"   关键指标: 任务数={dashboard.get('evolution_manager', {}).get('total_tasks', 0)}"
        )

        print("\n5. 测试报告生成:")
        end_time = time.time()
        start_time = end_time - 3600  # 过去1小时
        report = monitor.generate_evolution_report(start_time, end_time)
        print(f"   演化报告已生成: {report.get('report_id')}")
        print(f"   报告统计: {report.get('task_statistics', {}).get('total_tasks', 0)}个任务")

        print("\n✓ 演化监控系统测试完成")
        print("\n说明:")
        print("  1. 演化监控系统提供实时监控、告警和报告功能")
        print("  2. 支持自定义告警规则和指标回调")
        print("  3. 提供仪表板数据和详细报告")
        print("  4. 需要与演化管理器和模块集成进行完整监控")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
