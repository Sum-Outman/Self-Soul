#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演化管理器服务 - Evolution Manager Service

中心化的演化协调服务，提供以下功能：
1. 模型演化注册和调度
2. 演化资源管理
3. 演化进度监控
4. 演化结果处理和分析
5. 演化策略优化

设计原则：
- 集中管理：统一协调所有模型的演化过程
- 资源感知：根据系统资源动态调整演化策略
- 安全优先：包含熔断机制和安全约束
- 可扩展：支持多种演化算法和策略
"""

import logging
import time
import json
import threading
import queue
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import concurrent.futures
import numpy as np

from core.error_handling import error_handler
from core.evolution_module import EvolutionModule, get_evolution_module
from core.models.base_model import BaseModel
from core.model_registry import get_model_registry
from core.resource_manager import ResourceManager, get_resource_manager

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class EvolutionTask:
    """演化任务数据类"""

    task_id: str
    model_id: str
    model_instance: Optional[BaseModel] = None
    performance_targets: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10，10为最高优先级
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, scheduled, running, completed, failed, cancelled
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def get_duration(self) -> Optional[float]:
        """获取任务持续时间（秒）"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    def is_expired(self, timeout_seconds: float = 3600) -> bool:
        """检查任务是否已过期"""
        if self.status in ["completed", "failed", "cancelled"]:
            return False

        elapsed = time.time() - self.created_at
        return elapsed > timeout_seconds


@dataclass
class EvolutionSchedule:
    """演化调度计划"""

    schedule_id: str
    model_ids: List[str]
    schedule_type: str  # immediate, periodic, batch
    interval_seconds: Optional[float] = None  # 对于定期调度
    next_run: Optional[float] = None
    last_run: Optional[float] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class EvolutionManager:
    """
    演化管理器主类
    负责协调所有模型的演化过程
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化演化管理器

        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()

        # 演化模块
        self.evolution_module = get_evolution_module(
            self.config.get("evolution_module_config")
        )

        # 模型注册表
        self.model_registry = get_model_registry()

        # 任务管理
        self.tasks: Dict[str, EvolutionTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.task_lock = threading.Lock()

        # 调度管理
        self.schedules: Dict[str, EvolutionSchedule] = {}
        self.schedule_lock = threading.Lock()

        # 资源管理器
        self.resource_manager = get_resource_manager(self.config.get("resource_manager_config"))
        
        # 向后兼容的资源管理字段
        self.resource_limits = self.config.get("resource_limits", {})
        self.current_resources = {
            "memory_mb": 0.0,
            "cpu_percent": 0.0,
            "gpu_memory_mb": 0.0,
            "active_tasks": 0,
        }

        # 演化统计
        self.statistics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
            "total_evolution_time": 0.0,
            "average_evolution_time": 0.0,
            "total_models_evolved": 0,
            "best_performance_improvement": 0.0,
            "last_evolution_time": None,
        }

        # 线程和异步处理
        self.worker_threads = []
        self.max_workers = self.config.get("max_workers", 3)
        self.running = False
        self.shutdown_requested = False

        # 熔断机制
        self.fuse_state = {
            "consecutive_failures": 0,
            "fuse_triggered": False,
            "last_failure_time": None,
            "fuse_reason": None,
            "auto_reset_after_seconds": 300,  # 5分钟后自动重置
        }

        # 模型演化历史
        self.model_evolution_history: Dict[str, List[Dict[str, Any]]] = {}

        self.logger.info("演化管理器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "max_workers": 3,
            "max_concurrent_tasks": 2,
            "task_timeout_seconds": 3600,  # 1小时
            "resource_limits": {
                "max_memory_mb": 2000,
                "max_cpu_percent": 80,
                "max_gpu_memory_mb": 4000,
                "max_active_tasks": 2,
            },
            "evolution_module_config": {
                "max_history_size": 100,
                "max_rollback_points": 10,
            },
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
            "scheduling": {
                "immediate_priority": 10,
                "batch_priority": 5,
                "periodic_priority": 3,
            },
            "fuse_mechanism": {
                "enabled": True,
                "failure_threshold": 5,
                "auto_reset": True,
                "auto_reset_after_seconds": 300,
            },
        }

    def start(self):
        """启动演化管理器"""
        if self.running:
            self.logger.warning("演化管理器已在运行")
            return

        self.running = True
        self.shutdown_requested = False

        # 启动工作线程
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_thread_func,
                name=f"EvolutionWorker-{i}",
                daemon=True,
            )
            thread.start()
            self.worker_threads.append(thread)

        # 启动调度器线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_thread_func, name="EvolutionScheduler", daemon=True
        )
        self.scheduler_thread.start()

        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_thread_func, name="EvolutionMonitor", daemon=True
        )
        self.monitor_thread.start()

        # 启动资源管理器监控
        try:
            self.resource_manager.start_monitoring()
            self.logger.info("资源管理器监控已启动")
        except Exception as e:
            self.logger.error(f"启动资源管理器监控失败: {str(e)}")

        self.logger.info(f"演化管理器已启动，{self.max_workers}个工作线程")

    def stop(self):
        """停止演化管理器"""
        self.shutdown_requested = True
        self.running = False

        # 等待工作线程完成
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)

        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        # 停止资源管理器监控
        try:
            self.resource_manager.stop_monitoring()
            self.logger.info("资源管理器监控已停止")
        except Exception as e:
            self.logger.error(f"停止资源管理器监控失败: {str(e)}")

        self.logger.info("演化管理器已停止")

    def submit_evolution_task(
        self,
        model_id: str,
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        model_instance: Optional[BaseModel] = None,
    ) -> str:
        """
        提交演化任务

        Args:
            model_id: 模型ID
            performance_targets: 性能目标
            constraints: 约束条件
            priority: 任务优先级（1-10）
            model_instance: 模型实例（可选）

        Returns:
            任务ID
        """
        try:
            # 检查熔断机制
            if self.fuse_state["fuse_triggered"]:
                raise Exception(
                    f"演化熔断已触发: {self.fuse_state.get('fuse_reason', '未知原因')}"
                )

            # 检查模型是否存在
            if model_instance is None:
                model_instance = self.model_registry.get_model(model_id)
                if model_instance is None:
                    raise Exception(f"模型不存在: {model_id}")

            # 生成任务ID
            task_id = f"evo_task_{int(time.time())}_{len(self.tasks)}"

            # 创建演化任务
            task = EvolutionTask(
                task_id=task_id,
                model_id=model_id,
                model_instance=model_instance,
                performance_targets=performance_targets,
                constraints=constraints or {},
                priority=min(max(priority, 1), 10),  # 限制在1-10范围内
                status="pending",
            )

            # 保存任务
            with self.task_lock:
                self.tasks[task_id] = task

            # 将任务加入队列（使用负优先级，因为PriorityQueue是最小堆）
            self.task_queue.put((-task.priority, task_id))

            # 更新统计
            self.statistics["total_tasks"] += 1

            self.logger.info(
                f"演化任务已提交: {task_id} (模型: {model_id}, 优先级: {task.priority})"
            )
            return task_id

        except Exception as e:
            self.logger.error(f"提交演化任务失败: {str(e)}")
            raise

    def schedule_periodic_evolution(
        self,
        model_ids: List[str],
        interval_seconds: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调度定期演化

        Args:
            model_ids: 模型ID列表
            interval_seconds: 间隔时间（秒）
            config: 配置

        Returns:
            调度ID
        """
        try:
            schedule_id = f"schedule_{int(time.time())}_{len(self.schedules)}"

            schedule = EvolutionSchedule(
                schedule_id=schedule_id,
                model_ids=model_ids,
                schedule_type="periodic",
                interval_seconds=interval_seconds,
                next_run=time.time() + interval_seconds,
                enabled=True,
                config=config or {},
            )

            with self.schedule_lock:
                self.schedules[schedule_id] = schedule

            self.logger.info(
                f"定期演化已调度: {schedule_id} (模型: {model_ids}, 间隔: {interval_seconds}s)"
            )
            return schedule_id

        except Exception as e:
            self.logger.error(f"调度定期演化失败: {str(e)}")
            raise

    def schedule_batch_evolution(
        self, model_ids: List[str], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        调度批量演化

        Args:
            model_ids: 模型ID列表
            config: 配置

        Returns:
            调度ID
        """
        try:
            schedule_id = f"batch_schedule_{int(time.time())}_{len(self.schedules)}"

            schedule = EvolutionSchedule(
                schedule_id=schedule_id,
                model_ids=model_ids,
                schedule_type="batch",
                enabled=True,
                config=config or {},
            )

            with self.schedule_lock:
                self.schedules[schedule_id] = schedule

            # 立即提交批量任务
            for model_id in model_ids:
                try:
                    self.submit_evolution_task(
                        model_id=model_id,
                        performance_targets=config.get("performance_targets", {}),
                        constraints=config.get("constraints", {}),
                        priority=config.get("priority", 5),
                    )
                except Exception as e:
                    self.logger.warning(f"批量演化任务提交失败 (模型: {model_id}): {str(e)}")

            self.logger.info(f"批量演化已调度: {schedule_id} (模型: {model_ids})")
            return schedule_id

        except Exception as e:
            self.logger.error(f"调度批量演化失败: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self.task_lock:
            task = self.tasks.get(task_id)

        if task:
            return task.to_dict()
        return None

    def get_all_tasks(
        self, status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取所有任务"""
        with self.task_lock:
            tasks = list(self.tasks.values())

        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]

        return [t.to_dict() for t in tasks]

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status in ["completed", "failed", "cancelled"]:
                return False

            task.status = "cancelled"
            task.completed_at = time.time()
            self.statistics["cancelled_tasks"] += 1

        self.logger.info(f"任务已取消: {task_id}")
        return True

    def retry_task(self, task_id: str) -> bool:
        """重试失败的任务"""
        with self.task_lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status != "failed":
                return False

            if task.retry_count >= task.max_retries:
                return False

            # 重置任务状态
            task.status = "pending"
            task.started_at = None
            task.completed_at = None
            task.result = None
            task.error_message = None
            task.retry_count += 1

            # 重新加入队列
            self.task_queue.put((-task.priority, task_id))

        self.logger.info(f"任务重试: {task_id} (重试次数: {task.retry_count})")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.task_lock:
            stats = self.statistics.copy()

        # 添加当前状态
        stats.update(
            {
                "active_tasks": len(
                    [t for t in self.tasks.values() if t.status == "running"]
                ),
                "pending_tasks": len(
                    [t for t in self.tasks.values() if t.status == "pending"]
                ),
                "total_schedules": len(self.schedules),
                "running": self.running,
                "fuse_triggered": self.fuse_state["fuse_triggered"],
                "timestamp": time.time(),
            }
        )

        return stats

    def get_model_evolution_history(self, model_id: str) -> List[Dict[str, Any]]:
        """获取模型演化历史"""
        return self.model_evolution_history.get(model_id, [])

    def _worker_thread_func(self):
        """工作线程函数"""
        thread_name = threading.current_thread().name

        while not self.shutdown_requested:
            try:
                # 从队列获取任务
                try:
                    priority, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 获取任务
                with self.task_lock:
                    task = self.tasks.get(task_id)
                    if not task or task.status != "pending":
                        self.task_queue.task_done()
                        continue

                    # 检查资源限制
                    if not self._check_resource_limits():
                        # 资源不足，将任务放回队列
                        self.task_queue.put((priority, task_id))
                        self.task_queue.task_done()
                        time.sleep(5.0)  # 等待资源
                        continue

                    # 更新任务状态
                    task.status = "running"
                    task.started_at = time.time()
                    self.current_resources["active_tasks"] += 1

                # 分配任务资源
                resource_allocated = False
                try:
                    resource_allocated = self._allocate_task_resources(task)
                    if not resource_allocated:
                        # 资源分配失败，将任务放回队列
                        with self.task_lock:
                            task.status = "pending"
                            self.current_resources["active_tasks"] = max(
                                0, self.current_resources["active_tasks"] - 1
                            )
                        self.task_queue.put((priority, task_id))
                        self.task_queue.task_done()
                        self.logger.warning(
                            f"{thread_name} 任务资源分配失败，将任务放回队列: {task_id}"
                        )
                        time.sleep(5.0)  # 等待资源
                        continue
                except Exception as e:
                    self.logger.error(f"资源分配异常: {str(e)}")
                    # 资源分配异常，将任务放回队列
                    with self.task_lock:
                        task.status = "pending"
                        self.current_resources["active_tasks"] = max(
                            0, self.current_resources["active_tasks"] - 1
                        )
                    self.task_queue.put((priority, task_id))
                    self.task_queue.task_done()
                    time.sleep(5.0)  # 等待资源
                    continue

                self.logger.info(
                    f"{thread_name} 开始处理任务: {task_id} (模型: {task.model_id})"
                )

                # 执行演化任务
                try:
                    result = self._execute_evolution_task(task)
                    task.result = result
                    task.status = "completed"
                    task.completed_at = time.time()

                    # 更新统计
                    self.statistics["completed_tasks"] += 1
                    duration = task.get_duration()
                    if duration:
                        self.statistics["total_evolution_time"] += duration
                        completed_count = self.statistics["completed_tasks"]
                        self.statistics["average_evolution_time"] = (
                            self.statistics["total_evolution_time"] / completed_count
                        )

                    # 记录演化历史
                    self._record_evolution_history(task, result)

                    # 重置熔断失败计数
                    if self.fuse_state["consecutive_failures"] > 0:
                        self.fuse_state["consecutive_failures"] = 0

                    self.logger.info(
                        f"{thread_name} 任务完成: {task_id} (耗时: {duration:.2f}s)"
                    )

                except Exception as e:
                    task.status = "failed"
                    task.completed_at = time.time()
                    task.error_message = str(e)
                    task.retry_count += 1

                    # 更新统计
                    self.statistics["failed_tasks"] += 1

                    # 更新熔断状态
                    self._update_fuse_state(e)

                    self.logger.error(f"{thread_name} 任务失败: {task_id} - {str(e)}")

                    # 检查是否需要重试
                    if task.retry_count < task.max_retries:
                        # 将任务放回队列
                        self.task_queue.put((priority, task_id))
                        self.logger.info(
                            f"{thread_name} 任务将重试: {task_id} (重试次数: {task.retry_count})"
                        )

                finally:
                    # 释放资源
                    if resource_allocated:
                        try:
                            self._deallocate_task_resources(task)
                        except Exception as e:
                            self.logger.error(f"释放任务资源时发生异常: {str(e)}")
                    
                    with self.task_lock:
                        self.current_resources["active_tasks"] = max(
                            0, self.current_resources["active_tasks"] - 1
                        )

                    self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"{thread_name} 工作线程异常: {str(e)}", exc_info=True)
                time.sleep(1.0)

    def _scheduler_thread_func(self):
        """调度器线程函数"""
        while not self.shutdown_requested:
            try:
                current_time = time.time()

                with self.schedule_lock:
                    schedules_to_run = []

                    for schedule in self.schedules.values():
                        if not schedule.enabled:
                            continue

                        if schedule.schedule_type == "periodic":
                            if schedule.next_run and current_time >= schedule.next_run:
                                schedules_to_run.append(schedule)

                # 执行到期的调度
                for schedule in schedules_to_run:
                    self._execute_schedule(schedule)

                time.sleep(1.0)  # 每秒检查一次

            except Exception as e:
                self.logger.error(f"调度器线程异常: {str(e)}", exc_info=True)
                time.sleep(5.0)

    def _monitor_thread_func(self):
        """监控线程函数"""
        while not self.shutdown_requested:
            try:
                # 检查任务过期
                current_time = time.time()
                timeout_seconds = self.config.get("task_timeout_seconds", 3600)

                with self.task_lock:
                    for task in self.tasks.values():
                        if task.status == "running" and task.started_at:
                            elapsed = current_time - task.started_at
                            if elapsed > timeout_seconds:
                                task.status = "failed"
                                task.completed_at = current_time
                                task.error_message = (
                                    f"任务超时 ({elapsed:.0f}s > {timeout_seconds}s)"
                                )
                                self.statistics["failed_tasks"] += 1
                                self.logger.warning(f"任务超时: {task.task_id}")

                # 检查熔断自动重置
                if (
                    self.fuse_state["fuse_triggered"]
                    and self.fuse_state.get("last_failure_time")
                    and self.config["fuse_mechanism"]["auto_reset"]
                ):

                    elapsed = current_time - self.fuse_state["last_failure_time"]
                    reset_after = self.fuse_state["auto_reset_after_seconds"]

                    if elapsed > reset_after:
                        self.fuse_state["fuse_triggered"] = False
                        self.fuse_state["consecutive_failures"] = 0
                        self.fuse_state["fuse_reason"] = None
                        self.logger.info(f"熔断机制自动重置 (触发后 {elapsed:.0f}s)")

                time.sleep(10.0)  # 每10秒检查一次

            except Exception as e:
                self.logger.error(f"监控线程异常: {str(e)}", exc_info=True)
                time.sleep(30.0)

    def _execute_evolution_task(self, task: EvolutionTask) -> Dict[str, Any]:
        """执行演化任务"""
        try:
            # 检查模型实例
            if not task.model_instance:
                raise Exception("模型实例不可用")

            # 执行架构演化
            evolution_result = task.model_instance.evolve_architecture(
                performance_targets=task.performance_targets,
                constraints=task.constraints,
            )

            if not evolution_result.get("success", False):
                raise Exception(f"架构演化失败: {evolution_result.get('error', '未知错误')}")

            # 更新模型演化次数
            self.statistics["total_models_evolved"] += 1

            # 记录最佳性能改进
            improvement = evolution_result.get("performance_improvement", 0.0)
            if improvement > self.statistics["best_performance_improvement"]:
                self.statistics["best_performance_improvement"] = improvement

            self.statistics["last_evolution_time"] = time.time()

            return evolution_result

        except Exception as e:
            self.logger.error(f"执行演化任务失败: {str(e)}", exc_info=True)
            raise

    def _execute_schedule(self, schedule: EvolutionSchedule):
        """执行调度"""
        try:
            self.logger.info(
                f"执行调度: {schedule.schedule_id} (类型: {schedule.schedule_type})"
            )

            if schedule.schedule_type == "periodic":
                # 为每个模型提交演化任务
                for model_id in schedule.model_ids:
                    try:
                        self.submit_evolution_task(
                            model_id=model_id,
                            performance_targets=schedule.config.get(
                                "performance_targets", {}
                            ),
                            constraints=schedule.config.get("constraints", {}),
                            priority=schedule.config.get("priority", 3),
                        )
                    except Exception as e:
                        self.logger.warning(f"调度任务提交失败 (模型: {model_id}): {str(e)}")

                # 更新下一次运行时间
                schedule.last_run = time.time()
                schedule.next_run = schedule.last_run + schedule.interval_seconds

                self.logger.info(
                    f"调度执行完成: {schedule.schedule_id} (下次运行: {schedule.next_run})"
                )

        except Exception as e:
            self.logger.error(f"执行调度失败: {str(e)}", exc_info=True)

    def _check_resource_limits(self) -> bool:
        """检查资源限制（增强版，使用资源管理器）"""
        try:
            # 使用资源管理器检查资源
            usage = self.resource_manager.get_resource_usage()
            
            # 检查资源压力
            resource_pressure = usage.get("resource_pressure", {})
            
            # 获取配额违规
            quota_violations = usage.get("quota_violations", [])
            
            # 检查活动任务限制
            max_active_tasks = self.resource_limits.get("max_active_tasks", 2)
            active_tasks_ratio = (
                self.current_resources["active_tasks"] / max_active_tasks
            )
            
            # 检查资源压力是否过高
            high_pressure = False
            for resource, pressure in resource_pressure.items():
                if pressure > 0.9:  # 90%使用率阈值
                    self.logger.debug(
                        f"资源压力过高: {resource}={pressure:.2f}"
                    )
                    high_pressure = True
            
            # 检查是否有配额违规
            has_quota_violations = len(quota_violations) > 0
            
            # 检查活动任务限制
            tasks_exceeded = active_tasks_ratio > 1.0
            
            # 如果任何检查失败，返回False
            if high_pressure or has_quota_violations or tasks_exceeded:
                self.logger.debug(
                    f"资源限制检查失败: 压力过高={high_pressure}, "
                    f"配额违规={has_quota_violations}, 任务超限={tasks_exceeded}"
                )
                return False
            
            return True
            
        except Exception as e:
            # 如果资源管理器失败，回退到原始方法
            self.logger.warning(f"资源管理器检查失败，使用原始检查方法: {str(e)}")
            
            try:
                # 原始检查方法
                import psutil

                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # 检查内存限制
                max_memory_mb = self.resource_limits.get("max_memory_mb", 2000)
                current_memory_mb = memory_info.used / (1024 * 1024)
                memory_ratio = current_memory_mb / max_memory_mb

                # 检查CPU限制
                max_cpu_percent = self.resource_limits.get("max_cpu_percent", 80)
                cpu_ratio = cpu_percent / max_cpu_percent

                # 检查活动任务限制
                max_active_tasks = self.resource_limits.get("max_active_tasks", 2)
                active_tasks_ratio = (
                    self.current_resources["active_tasks"] / max_active_tasks
                )

                # 检查所有限制
                if memory_ratio > 0.9 or cpu_ratio > 0.9 or active_tasks_ratio > 1.0:
                    self.logger.debug(
                        f"资源限制检查失败: 内存={memory_ratio:.2f}, CPU={cpu_ratio:.2f}, 任务={active_tasks_ratio:.2f}"
                    )
                    return False

                return True

            except ImportError:
                # psutil不可用，使用简单检查
                max_active_tasks = self.resource_limits.get("max_active_tasks", 2)
                return self.current_resources["active_tasks"] < max_active_tasks

    def _update_fuse_state(self, error: Exception):
        """更新熔断状态"""
        if not self.config["fuse_mechanism"]["enabled"]:
            return

        self.fuse_state["consecutive_failures"] += 1
        self.fuse_state["last_failure_time"] = time.time()

        failure_threshold = self.config["fuse_mechanism"]["failure_threshold"]

        if self.fuse_state["consecutive_failures"] >= failure_threshold:
            self.fuse_state["fuse_triggered"] = True
            self.fuse_state["fuse_reason"] = f"连续{failure_threshold}次演化失败"
            self.logger.error(f"演化熔断触发: {self.fuse_state['fuse_reason']}")

    def _estimate_task_resource_requirements(self, task: EvolutionTask) -> Dict[str, float]:
        """估算演化任务资源需求
        
        Args:
            task: 演化任务
            
        Returns:
            资源需求字典，包含cpu_percent, memory_mb, gpu_memory_mb等
        """
        try:
            # 基于任务复杂性估算资源需求
            # 性能目标数量越多，资源需求越高
            complexity = len(task.performance_targets)
            
            # 约束条件也会影响资源需求
            constraints = task.constraints
            
            # 估算CPU需求（百分比）
            # 基础CPU需求 + 复杂性加成
            base_cpu = 10.0  # 基础CPU需求
            complexity_cpu = complexity * 2.0  # 每个性能目标增加2% CPU
            cpu_percent = min(base_cpu + complexity_cpu, 50.0)  # 不超过50%
            
            # 估算内存需求（MB）
            # 基础内存需求 + 模型大小估计
            base_memory = 200.0  # 基础内存需求
            model_memory = 100.0  # 模型内存估计
            memory_mb = base_memory + model_memory
            
            # 估算GPU内存需求（如果约束中指定了GPU）
            gpu_memory_mb = 0.0
            if constraints.get("use_gpu", False):
                gpu_memory_mb = 500.0  # 默认GPU内存需求
            
            # 任务优先级影响资源需求（高优先级任务可能需要更多资源）
            priority_factor = task.priority / 10.0  # 1.0为最高优先级
            
            # 应用优先级因子
            cpu_percent = cpu_percent * (0.8 + 0.4 * priority_factor)  # 0.8-1.2倍
            memory_mb = memory_mb * (0.8 + 0.4 * priority_factor)  # 0.8-1.2倍
            gpu_memory_mb = gpu_memory_mb * (0.8 + 0.4 * priority_factor)  # 0.8-1.2倍
            
            resource_requirements = {
                "cpu_percent": max(5.0, min(cpu_percent, 80.0)),  # 限制在5-80%之间
                "memory_mb": max(100.0, min(memory_mb, 2000.0)),  # 限制在100-2000MB之间
                "gpu_memory_mb": max(0.0, min(gpu_memory_mb, 4000.0)),  # 限制在0-4000MB之间
                "estimated_duration_seconds": 300.0,  # 默认5分钟
                "priority": task.priority
            }
            
            self.logger.debug(
                f"任务 {task.task_id} 资源需求估算: "
                f"CPU={resource_requirements['cpu_percent']:.1f}%, "
                f"内存={resource_requirements['memory_mb']:.1f}MB, "
                f"GPU内存={resource_requirements['gpu_memory_mb']:.1f}MB"
            )
            
            return resource_requirements
            
        except Exception as e:
            self.logger.error(f"估算任务资源需求失败: {str(e)}")
            # 返回默认资源需求
            return {
                "cpu_percent": 20.0,
                "memory_mb": 500.0,
                "gpu_memory_mb": 0.0,
                "estimated_duration_seconds": 300.0,
                "priority": task.priority
            }

    def _allocate_task_resources(self, task: EvolutionTask) -> bool:
        """为任务分配资源
        
        Args:
            task: 演化任务
            
        Returns:
            分配是否成功
        """
        try:
            # 估算资源需求
            resource_req = self._estimate_task_resource_requirements(task)
            
            # 分配资源
            allocation_success = self.resource_manager.allocate_resources(
                task_id=task.task_id,
                cpu_percent=resource_req["cpu_percent"],
                memory_mb=resource_req["memory_mb"],
                gpu_memory_mb=resource_req["gpu_memory_mb"],
                priority=resource_req["priority"],
                duration_seconds=resource_req.get("estimated_duration_seconds", 300.0)
            )
            
            if allocation_success:
                self.logger.info(f"任务 {task.task_id} 资源分配成功")
                return True
            else:
                self.logger.warning(f"任务 {task.task_id} 资源分配失败，资源不足")
                return False
                
        except Exception as e:
            self.logger.error(f"分配任务资源失败: {str(e)}")
            return False

    def _deallocate_task_resources(self, task: EvolutionTask) -> bool:
        """释放任务资源
        
        Args:
            task: 演化任务
            
        Returns:
            释放是否成功
        """
        try:
            deallocation_success = self.resource_manager.deallocate_resources(task.task_id)
            
            if deallocation_success:
                self.logger.info(f"任务 {task.task_id} 资源释放成功")
                return True
            else:
                self.logger.warning(f"任务 {task.task_id} 资源释放失败")
                return False
                
        except Exception as e:
            self.logger.error(f"释放任务资源失败: {str(e)}")
            return False

    def _record_evolution_history(self, task: EvolutionTask, result: Dict[str, Any]):
        """记录演化历史"""
        try:
            model_id = task.model_id

            if model_id not in self.model_evolution_history:
                self.model_evolution_history[model_id] = []

            history_entry = {
                "task_id": task.task_id,
                "timestamp": time.time(),
                "performance_targets": task.performance_targets,
                "constraints": task.constraints,
                "result": result,
                "duration": task.get_duration(),
                "success": result.get("success", False),
            }

            self.model_evolution_history[model_id].append(history_entry)

            # 限制历史记录大小
            max_history = 100
            if len(self.model_evolution_history[model_id]) > max_history:
                self.model_evolution_history[model_id] = self.model_evolution_history[
                    model_id
                ][-max_history:]

        except Exception as e:
            self.logger.error(f"记录演化历史失败: {str(e)}")


# 全局演化管理器实例
_global_evolution_manager: Optional[EvolutionManager] = None


def get_evolution_manager(config: Optional[Dict[str, Any]] = None) -> EvolutionManager:
    """
    获取全局演化管理器实例

    Args:
        config: 配置字典

    Returns:
        演化管理器实例
    """
    global _global_evolution_manager

    if _global_evolution_manager is None:
        _global_evolution_manager = EvolutionManager(config)

    return _global_evolution_manager


def start_evolution_manager(config: Optional[Dict[str, Any]] = None):
    """启动全局演化管理器"""
    manager = get_evolution_manager(config)
    manager.start()
    return manager


def stop_evolution_manager():
    """停止全局演化管理器"""
    global _global_evolution_manager
    if _global_evolution_manager:
        _global_evolution_manager.stop()
        _global_evolution_manager = None


# 测试代码
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("演化管理器测试")
    print("=" * 80)

    try:
        # 创建演化管理器
        manager = EvolutionManager()

        print("\n1. 测试演化管理器初始化:")
        print(
            f"   配置: 最大工作线程={manager.config['max_workers']}, 最大并发任务={manager.config['max_concurrent_tasks']}"
        )

        print("\n2. 测试统计信息获取:")
        stats = manager.get_statistics()
        print(f"   初始统计: 总任务数={stats['total_tasks']}, 运行中={stats['running']}")

        print("\n3. 测试任务提交:")
        # 注意：这里需要实际的模型实例，我们模拟一个任务
        print("   任务提交功能已实现，需要实际模型实例进行测试")

        print("\n4. 测试调度功能:")
        # 测试批量调度
        model_ids = ["test_model_1", "test_model_2"]
        schedule_id = manager.schedule_batch_evolution(
            model_ids=model_ids,
            config={
                "performance_targets": {"accuracy": 0.9},
                "constraints": {"memory_mb": 500},
                "priority": 5,
            },
        )
        print(f"   批量调度已创建: {schedule_id} (模型: {model_ids})")

        print("\n5. 测试资源检查:")
        resource_ok = manager._check_resource_limits()
        print(f"   资源检查: {'通过' if resource_ok else '不通过'}")

        print("\n✓ 演化管理器测试完成")
        print("\n说明:")
        print("  1. 演化管理器提供了中心化的演化任务调度和协调")
        print("  2. 支持优先级队列、资源感知调度和熔断机制")
        print("  3. 可以调度定期、批量演化任务")
        print("  4. 需要集成实际模型实例进行完整测试")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
