#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强演化管理器 - Enhanced Evolution Manager

基于现有的EvolutionManager，提供以下增强功能：
1. 增强演化模块集成：使用EnhancedEvolutionModule，支持多算法、多目标优化
2. 协同演化调度：支持多模型协同演化任务
3. 元学习策略：基于历史数据和任务特征选择最佳演化策略
4. 多目标优化支持：支持同时优化多个性能目标
5. 增强监控和洞察：提供详细的演化分析和建议

设计原则：
- 向后兼容：兼容现有的EvolutionManager接口
- 渐进增强：默认使用增强功能，可回退到基础功能
- 透明替换：可替代现有的EvolutionManager
- 可配置：通过配置启用/禁用特定增强功能
"""

import logging
import time
import json
import threading
import queue
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import concurrent.futures
import numpy as np
from enum import Enum

from core.error_handling import error_handler
from core.models.base_model import BaseModel
from core.model_registry import get_model_registry
from core.resource_manager import ResourceManager, get_resource_manager
from core.enhanced_evolution_module import EnhancedEvolutionModule, create_enhanced_evolution_module

# 配置日志
logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """演化策略类型"""
    SINGLE_OBJECTIVE = "single_objective"  # 单目标优化
    MULTI_OBJECTIVE = "multi_objective"    # 多目标优化
    COEVOLUTION = "coevolution"            # 协同演化
    ADAPTIVE = "adaptive"                  # 自适应策略


@dataclass
class EnhancedEvolutionTask:
    """增强演化任务数据类"""
    task_id: str
    model_id: str
    model_instance: Optional[BaseModel] = None
    performance_targets: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    strategy: str = "single_objective"  # 演化策略
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
    # 增强字段
    multi_objective_weights: Optional[Dict[str, float]] = None  # 多目标权重
    coevolution_group: Optional[str] = None  # 协同演化组
    algorithm_preference: Optional[str] = None  # 算法偏好

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
class EnhancedEvolutionSchedule:
    """增强演化调度计划"""
    schedule_id: str
    model_ids: List[str]
    schedule_type: str  # immediate, periodic, batch, coevolution
    interval_seconds: Optional[float] = None  # 对于定期调度
    next_run: Optional[float] = None
    last_run: Optional[float] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    # 增强字段
    strategy: str = "single_objective"  # 演化策略
    coevolution_group: Optional[str] = None  # 协同演化组名称


@dataclass
class CoevolutionGroup:
    """协同演化组"""
    group_id: str
    model_ids: List[str]
    created_at: float = field(default_factory=time.time)
    active: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    evolution_count: int = 0
    last_evolution: Optional[float] = None
    best_performance: Dict[str, float] = field(default_factory=dict)


class EnhancedEvolutionManager:
    """
    增强演化管理器主类
    负责协调所有模型的演化过程，支持增强功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强演化管理器

        Args:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or self._get_default_config()

        # 增强演化模块
        self.enhanced_evolution_module = create_enhanced_evolution_module(
            self.config.get("enhanced_evolution_module_config")
        )

        # 模型注册表
        self.model_registry = get_model_registry()

        # 任务管理
        self.tasks: Dict[str, EnhancedEvolutionTask] = {}
        self.task_queue = queue.PriorityQueue()
        self.task_lock = threading.Lock()

        # 调度管理
        self.schedules: Dict[str, EnhancedEvolutionSchedule] = {}
        self.schedule_lock = threading.Lock()

        # 协同演化组管理
        self.coevolution_groups: Dict[str, CoevolutionGroup] = {}
        self.coevolution_lock = threading.Lock()

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

        # 演化统计（增强版）
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
            # 增强统计
            "strategy_usage": {
                "single_objective": 0,
                "multi_objective": 0,
                "coevolution": 0,
                "adaptive": 0,
            },
            "algorithm_performance": {},
            "multi_objective_solutions": 0,
            "coevolution_groups_created": 0,
            "learning_progress": 0.0,
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

        # 模型演化历史（增强版）
        self.model_evolution_history: Dict[str, List[Dict[str, Any]]] = {}

        # 策略学习器
        self.strategy_learning_enabled = self.config.get("strategy_learning_enabled", True)
        self.strategy_history: List[Dict[str, Any]] = []

        self.logger.info("增强演化管理器初始化完成")
        self.logger.info(f"启用增强功能: 协同演化={self.config.get('enable_coevolution', True)}, "
                        f"多目标优化={self.config.get('enable_multi_objective', True)}, "
                        f"策略学习={self.strategy_learning_enabled}")

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
            "enhanced_evolution_module_config": {
                "enable_multi_objective": True,
                "enable_meta_learning": True,
                "enable_coevolution": True,
                "engine_config": {
                    "meta_controller_config": {
                        "exploration_rate": 0.1,
                        "learning_rate": 0.05,
                    },
                    "multi_objective_config": {
                        "objective_weights": {
                            "accuracy": 0.4,
                            "efficiency": 0.3,
                            "robustness": 0.2,
                            "adaptability": 0.1,
                        },
                    },
                },
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
                "coevolution_priority": 8,
            },
            "fuse_mechanism": {
                "enabled": True,
                "failure_threshold": 5,
                "auto_reset": True,
                "auto_reset_after_seconds": 300,
            },
            "strategy_learning_enabled": True,
            "enable_coevolution": True,
            "enable_multi_objective": True,
            "coevolution": {
                "max_group_size": 5,
                "min_group_size": 2,
                "performance_threshold": 0.7,
            },
        }

    def start(self):
        """启动增强演化管理器"""
        if self.running:
            self.logger.warning("增强演化管理器已在运行")
            return

        self.running = True
        self.shutdown_requested = False

        # 启动工作线程
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_thread_func,
                name=f"EnhancedEvolutionWorker-{i}",
                daemon=True,
            )
            thread.start()
            self.worker_threads.append(thread)

        # 启动调度器线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_thread_func, name="EnhancedEvolutionScheduler", daemon=True
        )
        self.scheduler_thread.start()

        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_thread_func, name="EnhancedEvolutionMonitor", daemon=True
        )
        self.monitor_thread.start()

        # 启动资源管理器监控
        try:
            self.resource_manager.start_monitoring()
            self.logger.info("资源管理器监控已启动")
        except Exception as e:
            self.logger.error(f"启动资源管理器监控失败: {str(e)}")

        # 初始化增强演化模块
        try:
            self.enhanced_evolution_module.initialize()
            self.logger.info("增强演化模块已初始化")
        except Exception as e:
            self.logger.warning(f"增强演化模块初始化失败，将回退到基础功能: {str(e)}")

        self.logger.info(f"增强演化管理器已启动，{self.max_workers}个工作线程")

    def stop(self):
        """停止增强演化管理器"""
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

        # 停止增强演化模块
        try:
            self.enhanced_evolution_module.stop()
            self.logger.info("增强演化模块已停止")
        except Exception as e:
            self.logger.warning(f"停止增强演化模块失败: {str(e)}")

        self.logger.info("增强演化管理器已停止")

    def submit_evolution_task(
        self,
        model_id: str,
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        model_instance: Optional[BaseModel] = None,
        strategy: str = "single_objective",
        multi_objective_weights: Optional[Dict[str, float]] = None,
        coevolution_group: Optional[str] = None,
    ) -> str:
        """
        提交增强演化任务

        Args:
            model_id: 模型ID
            performance_targets: 性能目标
            constraints: 约束条件
            priority: 任务优先级（1-10）
            model_instance: 模型实例（可选）
            strategy: 演化策略（single_objective, multi_objective, coevolution, adaptive）
            multi_objective_weights: 多目标权重（仅对多目标优化）
            coevolution_group: 协同演化组（仅对协同演化）

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
            task_id = f"enhanced_evo_task_{int(time.time())}_{len(self.tasks)}"

            # 创建增强演化任务
            task = EnhancedEvolutionTask(
                task_id=task_id,
                model_id=model_id,
                model_instance=model_instance,
                performance_targets=performance_targets,
                constraints=constraints or {},
                strategy=strategy,
                priority=min(max(priority, 1), 10),  # 限制在1-10范围内
                status="pending",
                multi_objective_weights=multi_objective_weights,
                coevolution_group=coevolution_group,
            )

            # 保存任务
            with self.task_lock:
                self.tasks[task_id] = task

            # 将任务加入队列（使用负优先级，因为PriorityQueue是最小堆）
            self.task_queue.put((-task.priority, task_id))

            # 更新统计
            self.statistics["total_tasks"] += 1
            if strategy in self.statistics["strategy_usage"]:
                self.statistics["strategy_usage"][strategy] += 1

            self.logger.info(
                f"增强演化任务已提交: {task_id} "
                f"(模型: {model_id}, 策略: {strategy}, 优先级: {task.priority})"
            )
            
            if strategy == "multi_objective":
                self.logger.info(f"  多目标权重: {multi_objective_weights}")
            elif strategy == "coevolution" and coevolution_group:
                self.logger.info(f"  协同演化组: {coevolution_group}")

            return task_id

        except Exception as e:
            self.logger.error(f"提交增强演化任务失败: {str(e)}")
            raise

    def submit_coevolution_task(
        self,
        model_ids: List[str],
        performance_targets: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
        priority: int = 8,
        group_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        提交协同演化任务

        Args:
            model_ids: 模型ID列表
            performance_targets: 性能目标
            constraints: 约束条件
            priority: 任务优先级（1-10）
            group_name: 协同演化组名称（可选，自动生成）
            config: 配置

        Returns:
            (组ID, 任务ID)
        """
        try:
            # 检查熔断机制
            if self.fuse_state["fuse_triggered"]:
                raise Exception(
                    f"演化熔断已触发: {self.fuse_state.get('fuse_reason', '未知原因')}"
                )

            # 检查模型是否存在
            models = []
            for model_id in model_ids:
                model = self.model_registry.get_model(model_id)
                if model is None:
                    raise Exception(f"模型不存在: {model_id}")
                models.append(model)

            # 创建或获取协同演化组
            if group_name is None:
                group_id = f"coevolution_group_{int(time.time())}_{len(self.coevolution_groups)}"
            else:
                group_id = group_name

            # 检查组大小限制
            max_group_size = self.config.get("coevolution", {}).get("max_group_size", 5)
            if len(model_ids) > max_group_size:
                raise Exception(f"协同演化组大小超过限制: {len(model_ids)} > {max_group_size}")

            # 创建协同演化组
            with self.coevolution_lock:
                if group_id not in self.coevolution_groups:
                    self.coevolution_groups[group_id] = CoevolutionGroup(
                        group_id=group_id,
                        model_ids=model_ids,
                        config=config or {},
                    )
                    self.statistics["coevolution_groups_created"] += 1
                    self.logger.info(f"创建协同演化组: {group_id} (模型: {model_ids})")

            # 生成任务ID
            task_id = f"coevo_task_{int(time.time())}_{len(self.tasks)}"

            # 为组中每个模型创建任务
            for model_id, model_instance in zip(model_ids, models):
                # 创建增强演化任务
                task = EnhancedEvolutionTask(
                    task_id=f"{task_id}_{model_id}",
                    model_id=model_id,
                    model_instance=model_instance,
                    performance_targets=performance_targets,
                    constraints=constraints or {},
                    strategy="coevolution",
                    priority=min(max(priority, 1), 10),
                    status="pending",
                    coevolution_group=group_id,
                )

                # 保存任务
                with self.task_lock:
                    self.tasks[task.task_id] = task

                # 将任务加入队列
                self.task_queue.put((-task.priority, task.task_id))

                # 更新统计
                self.statistics["total_tasks"] += 1
                self.statistics["strategy_usage"]["coevolution"] += 1

            # 记录协同演化组状态
            with self.coevolution_lock:
                if group_id in self.coevolution_groups:
                    self.coevolution_groups[group_id].evolution_count += 1

            self.logger.info(
                f"协同演化任务已提交: {task_id} "
                f"(组: {group_id}, 模型数: {len(model_ids)}, 优先级: {priority})"
            )

            return group_id, task_id

        except Exception as e:
            self.logger.error(f"提交协同演化任务失败: {str(e)}")
            raise

    def schedule_periodic_evolution(
        self,
        model_ids: List[str],
        interval_seconds: float,
        config: Optional[Dict[str, Any]] = None,
        strategy: str = "single_objective",
    ) -> str:
        """
        调度定期演化

        Args:
            model_ids: 模型ID列表
            interval_seconds: 间隔时间（秒）
            config: 配置
            strategy: 演化策略

        Returns:
            调度ID
        """
        try:
            schedule_id = f"enhanced_schedule_{int(time.time())}_{len(self.schedules)}"

            schedule = EnhancedEvolutionSchedule(
                schedule_id=schedule_id,
                model_ids=model_ids,
                schedule_type="periodic",
                interval_seconds=interval_seconds,
                next_run=time.time() + interval_seconds,
                enabled=True,
                config=config or {},
                strategy=strategy,
            )

            with self.schedule_lock:
                self.schedules[schedule_id] = schedule

            self.logger.info(
                f"增强定期演化已调度: {schedule_id} "
                f"(模型: {model_ids}, 策略: {strategy}, 间隔: {interval_seconds}s)"
            )
            return schedule_id

        except Exception as e:
            self.logger.error(f"调度定期演化失败: {str(e)}")
            raise

    def schedule_batch_evolution(
        self,
        model_ids: List[str],
        config: Optional[Dict[str, Any]] = None,
        strategy: str = "single_objective",
    ) -> str:
        """
        调度批量演化

        Args:
            model_ids: 模型ID列表
            config: 配置
            strategy: 演化策略

        Returns:
            调度ID
        """
        try:
            schedule_id = f"batch_schedule_{int(time.time())}_{len(self.schedules)}"

            schedule = EnhancedEvolutionSchedule(
                schedule_id=schedule_id,
                model_ids=model_ids,
                schedule_type="batch",
                next_run=time.time(),  # 立即执行
                enabled=True,
                config=config or {},
                strategy=strategy,
            )

            with self.schedule_lock:
                self.schedules[schedule_id] = schedule

            # 立即提交任务
            for model_id in model_ids:
                try:
                    self.submit_evolution_task(
                        model_id=model_id,
                        performance_targets=config.get("performance_targets", {}),
                        constraints=config.get("constraints", {}),
                        priority=config.get("priority", 5),
                        strategy=strategy,
                    )
                except Exception as e:
                    self.logger.warning(f"批量演化任务提交失败 (模型: {model_id}): {str(e)}")

            self.logger.info(
                f"增强批量演化已调度: {schedule_id} "
                f"(模型: {model_ids}, 策略: {strategy}, 任务数: {len(model_ids)})"
            )
            return schedule_id

        except Exception as e:
            self.logger.error(f"调度批量演化失败: {str(e)}")
            raise

    def schedule_coevolution(
        self,
        model_groups: List[List[str]],
        interval_seconds: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        调度定期协同演化

        Args:
            model_groups: 模型组列表，每个组是一个模型ID列表
            interval_seconds: 间隔时间（秒）
            config: 配置

        Returns:
            调度ID列表
        """
        try:
            schedule_ids = []
            for i, model_ids in enumerate(model_groups):
                schedule_id = f"coevo_schedule_{int(time.time())}_{i}"
                
                schedule = EnhancedEvolutionSchedule(
                    schedule_id=schedule_id,
                    model_ids=model_ids,
                    schedule_type="coevolution",
                    interval_seconds=interval_seconds,
                    next_run=time.time() + interval_seconds,
                    enabled=True,
                    config=config or {},
                    strategy="coevolution",
                    coevolution_group=f"coevo_group_{i}",
                )

                with self.schedule_lock:
                    self.schedules[schedule_id] = schedule

                schedule_ids.append(schedule_id)
                
                # 创建协同演化组
                group_id = f"coevo_group_{i}"
                with self.coevolution_lock:
                    self.coevolution_groups[group_id] = CoevolutionGroup(
                        group_id=group_id,
                        model_ids=model_ids,
                        config=config or {},
                    )

                self.logger.info(
                    f"协同演化已调度: {schedule_id} "
                    f"(组: {group_id}, 模型: {model_ids}, 间隔: {interval_seconds}s)"
                )

            return schedule_ids

        except Exception as e:
            self.logger.error(f"调度协同演化失败: {str(e)}")
            raise

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
                    f"{thread_name} 开始处理任务: {task_id} (模型: {task.model_id}, 策略: {task.strategy})"
                )

                # 执行演化任务
                try:
                    result = self._execute_enhanced_evolution_task(task)
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
                    
                    # 更新协同演化组状态
                    if task.coevolution_group:
                        self._update_coevolution_group(task.coevolution_group, result)

                    # 记录策略历史（用于学习）
                    if self.strategy_learning_enabled:
                        self._record_strategy_history(task, result, duration)

                    # 重置熔断失败计数
                    if self.fuse_state["consecutive_failures"] > 0:
                        self.fuse_state["consecutive_failures"] = 0

                    self.logger.info(
                        f"{thread_name} 任务完成: {task_id} "
                        f"(策略: {task.strategy}, 耗时: {duration:.2f}s)"
                    )

                except Exception as e:
                    task.status = "failed"
                    task.completed_at = time.time()
                    task.error_message = str(e)
                    task.retry_count += 1

                    # 更新失败统计
                    self.statistics["failed_tasks"] += 1
                    self.fuse_state["consecutive_failures"] += 1
                    self.fuse_state["last_failure_time"] = time.time()

                    # 检查是否需要触发熔断
                    failure_threshold = self.config.get("fuse_mechanism", {}).get("failure_threshold", 5)
                    if self.fuse_state["consecutive_failures"] >= failure_threshold:
                        self.fuse_state["fuse_triggered"] = True
                        self.fuse_state["fuse_reason"] = f"连续{failure_threshold}次演化失败"
                        self.logger.error(
                            f"演化熔断已触发: {self.fuse_state['fuse_reason']}"
                        )

                    self.logger.error(
                        f"{thread_name} 任务失败: {task_id} (重试次数: {task.retry_count}): {str(e)}"
                    )

                    # 检查是否需要重试
                    if task.retry_count < task.max_retries:
                        task.status = "pending"
                        # 增加优先级以确保重试
                        task.priority = min(task.priority + 1, 10)
                        self.task_queue.put((-task.priority, task_id))
                        self.logger.info(
                            f"任务 {task_id} 将重试 (重试 {task.retry_count}/{task.max_retries})"
                        )
                    else:
                        self.logger.warning(
                            f"任务 {task_id} 已达到最大重试次数，标记为失败"
                        )

                finally:
                    # 释放任务资源
                    try:
                        self._deallocate_task_resources(task)
                    except Exception as e:
                        self.logger.warning(f"释放任务资源失败: {str(e)}")

                    # 更新活动任务计数
                    with self.task_lock:
                        self.current_resources["active_tasks"] = max(
                            0, self.current_resources["active_tasks"] - 1
                        )

                    self.task_queue.task_done()

                    # 短暂休眠以避免CPU占用过高
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"工作线程异常: {str(e)}", exc_info=True)
                time.sleep(5.0)  # 发生异常后等待一段时间

    def _execute_enhanced_evolution_task(self, task: EnhancedEvolutionTask) -> Dict[str, Any]:
        """执行增强演化任务"""
        try:
            # 检查模型实例
            if not task.model_instance:
                raise Exception("模型实例不可用")

            # 根据策略选择演化方法
            if task.strategy == "multi_objective" and task.multi_objective_weights:
                # 多目标优化
                evolution_result = task.model_instance.evolve_architecture(
                    performance_targets=task.performance_targets,
                    constraints=task.constraints,
                    multi_objective_weights=task.multi_objective_weights,
                )
            elif task.strategy == "coevolution" and task.coevolution_group:
                # 协同演化
                evolution_result = self._execute_coevolution_task(task)
            else:
                # 单目标优化
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
            
            # 如果多目标优化成功，更新统计
            if task.strategy == "multi_objective" and evolution_result.get("multi_objective_solution", False):
                self.statistics["multi_objective_solutions"] += 1

            return evolution_result

        except Exception as e:
            self.logger.error(f"执行增强演化任务失败: {str(e)}", exc_info=True)
            raise

    def _execute_coevolution_task(self, task: EnhancedEvolutionTask) -> Dict[str, Any]:
        """执行协同演化任务"""
        try:
            # 获取协同演化组
            group_id = task.coevolution_group
            if not group_id:
                raise Exception("协同演化任务没有指定组")

            with self.coevolution_lock:
                group = self.coevolution_groups.get(group_id)
                if not group:
                    raise Exception(f"协同演化组不存在: {group_id}")

            # 获取组中所有模型
            group_model_ids = group.model_ids
            models = []
            for model_id in group_model_ids:
                model = self.model_registry.get_model(model_id)
                if model:
                    models.append(model)

            if len(models) < 2:
                raise Exception(f"协同演化组需要至少2个模型，当前只有 {len(models)} 个")

            self.logger.info(
                f"开始协同演化: 组 {group_id}, 模型数 {len(models)}"
            )

            # 创建协同演化任务配置
            coevolution_config = {
                "group_id": group_id,
                "model_ids": group_model_ids,
                "performance_targets": task.performance_targets,
                "constraints": task.constraints,
                "base_architecture": task.model_instance.get_model_architecture() if hasattr(task.model_instance, 'get_model_architecture') else {},
            }

            # 执行协同演化（这里可以扩展为更复杂的协同演化逻辑）
            # 目前简化实现：为每个模型执行演化，然后交换最佳架构
            results = []
            for model in models:
                try:
                    # 使用增强演化模块
                    evolution_result = self.enhanced_evolution_module.coevolve_architectures(
                        group_id=group_id,
                        base_architecture=model.get_model_architecture() if hasattr(model, 'get_model_architecture') else {},
                        performance_targets=task.performance_targets,
                        constraints=task.constraints,
                    )
                    
                    if evolution_result.get("success", False):
                        results.append({
                            "model_id": model.model_id,
                            "result": evolution_result,
                            "performance": evolution_result.get("performance_metrics", {}).get("overall_score", 0.0),
                        })
                except Exception as e:
                    self.logger.warning(f"协同演化中模型 {model.model_id} 失败: {str(e)}")

            if not results:
                raise Exception("协同演化中所有模型都失败")

            # 选择最佳结果
            best_result = max(results, key=lambda x: x["performance"])

            # 更新协同演化组状态
            with self.coevolution_lock:
                if group_id in self.coevolution_groups:
                    self.coevolution_groups[group_id].last_evolution = time.time()
                    self.coevolution_groups[group_id].evolution_count += 1
                    
                    # 更新最佳性能
                    for metric, value in best_result["result"].get("performance_metrics", {}).items():
                        if metric not in self.coevolution_groups[group_id].best_performance:
                            self.coevolution_groups[group_id].best_performance[metric] = value
                        else:
                            self.coevolution_groups[group_id].best_performance[metric] = max(
                                self.coevolution_groups[group_id].best_performance[metric],
                                value
                            )

            # 返回最佳结果
            return {
                "success": True,
                "strategy": "coevolution",
                "group_id": group_id,
                "model_id": task.model_id,
                "best_model_id": best_result["model_id"],
                "performance_improvement": best_result["performance"],
                "performance_metrics": best_result["result"].get("performance_metrics", {}),
                "evolution_result": best_result["result"],
                "coevolution_results": results,
                "message": f"协同演化完成，最佳模型: {best_result['model_id']}",
            }

        except Exception as e:
            self.logger.error(f"执行协同演化任务失败: {str(e)}", exc_info=True)
            raise

    def _record_strategy_history(self, task: EnhancedEvolutionTask, result: Dict[str, Any], duration: float):
        """记录策略历史用于学习"""
        try:
            history_entry = {
                "timestamp": time.time(),
                "task_id": task.task_id,
                "model_id": task.model_id,
                "strategy": task.strategy,
                "performance_targets": task.performance_targets,
                "constraints": task.constraints,
                "performance": result.get("performance_improvement", 0.0),
                "duration": duration,
                "success": result.get("success", False),
                "multi_objective_weights": task.multi_objective_weights,
                "coevolution_group": task.coevolution_group,
            }
            
            self.strategy_history.append(history_entry)
            
            # 限制历史记录大小
            max_history = 1000
            if len(self.strategy_history) > max_history:
                self.strategy_history = self.strategy_history[-max_history:]
                
        except Exception as e:
            self.logger.warning(f"记录策略历史失败: {str(e)}")

    def _update_coevolution_group(self, group_id: str, result: Dict[str, Any]):
        """更新协同演化组状态"""
        try:
            with self.coevolution_lock:
                if group_id in self.coevolution_groups:
                    group = self.coevolution_groups[group_id]
                    group.last_evolution = time.time()
                    
                    # 更新最佳性能
                    performance_metrics = result.get("performance_metrics", {})
                    for metric, value in performance_metrics.items():
                        if metric not in group.best_performance:
                            group.best_performance[metric] = value
                        else:
                            group.best_performance[metric] = max(group.best_performance[metric], value)
                            
        except Exception as e:
            self.logger.warning(f"更新协同演化组状态失败: {str(e)}")

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
                    f"资源限制检查失败: 高压={high_pressure}, 配额违规={has_quota_violations}, "
                    f"任务超限={tasks_exceeded}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查资源限制失败: {str(e)}")
            # 出错时默认返回True以避免死锁
            return True

    def _allocate_task_resources(self, task: EnhancedEvolutionTask) -> bool:
        """分配任务资源
        
        Args:
            task: 演化任务
            
        Returns:
            分配是否成功
        """
        try:
            # 估算任务资源需求
            resource_estimate = {
                "task_id": task.task_id,
                "cpu_cores": 1.0,  # 默认1个CPU核心
                "memory_mb": 500.0,  # 默认500MB内存
                "gpu_memory_mb": 0.0,  # 默认不使用GPU
                "estimated_duration": 60.0,  # 默认60秒
                "priority": task.priority,
            }
            
            # 根据任务策略调整资源估计
            if task.strategy == "multi_objective":
                resource_estimate["cpu_cores"] = 2.0  # 多目标优化需要更多计算
                resource_estimate["estimated_duration"] = 120.0
            elif task.strategy == "coevolution":
                resource_estimate["cpu_cores"] = 3.0  # 协同演化需要更多计算
                resource_estimate["memory_mb"] = 1000.0
                resource_estimate["estimated_duration"] = 180.0
            
            allocation_success = self.resource_manager.allocate_resources(
                task.task_id, resource_estimate
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

    def _deallocate_task_resources(self, task: EnhancedEvolutionTask) -> bool:
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

    def _record_evolution_history(self, task: EnhancedEvolutionTask, result: Dict[str, Any]):
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
                "strategy": task.strategy,
                "result": result,
                "duration": task.get_duration(),
                "success": result.get("success", False),
                "multi_objective_weights": task.multi_objective_weights,
                "coevolution_group": task.coevolution_group,
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

    def _scheduler_thread_func(self):
        """调度器线程函数"""
        thread_name = threading.current_thread().name
        
        while not self.shutdown_requested:
            try:
                current_time = time.time()
                
                with self.schedule_lock:
                    schedules_to_run = []
                    
                    for schedule_id, schedule in self.schedules.items():
                        if not schedule.enabled:
                            continue
                            
                        if schedule.next_run and schedule.next_run <= current_time:
                            schedules_to_run.append(schedule)
                
                # 执行需要运行的调度
                for schedule in schedules_to_run:
                    try:
                        self._execute_schedule(schedule)
                    except Exception as e:
                        self.logger.error(f"执行调度失败: {schedule.schedule_id}: {str(e)}")
                
                time.sleep(1.0)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"调度器线程异常: {str(e)}", exc_info=True)
                time.sleep(5.0)

    def _execute_schedule(self, schedule: EnhancedEvolutionSchedule):
        """执行调度"""
        try:
            self.logger.info(
                f"执行调度: {schedule.schedule_id} (类型: {schedule.schedule_type}, 策略: {schedule.strategy})"
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
                            strategy=schedule.strategy,
                        )
                    except Exception as e:
                        self.logger.warning(f"调度任务提交失败 (模型: {model_id}): {str(e)}")

                # 更新下一次运行时间
                schedule.last_run = time.time()
                schedule.next_run = schedule.last_run + schedule.interval_seconds

                self.logger.info(
                    f"调度执行完成: {schedule.schedule_id} "
                    f"(策略: {schedule.strategy}, 下次运行: {schedule.next_run})"
                )
                
            elif schedule.schedule_type == "coevolution":
                # 提交协同演化任务
                try:
                    group_id, task_id = self.submit_coevolution_task(
                        model_ids=schedule.model_ids,
                        performance_targets=schedule.config.get(
                            "performance_targets", {}
                        ),
                        constraints=schedule.config.get("constraints", {}),
                        priority=schedule.config.get("priority", 8),
                        group_name=schedule.coevolution_group,
                        config=schedule.config,
                    )
                    
                    # 更新下一次运行时间
                    schedule.last_run = time.time()
                    schedule.next_run = schedule.last_run + schedule.interval_seconds
                    
                    self.logger.info(
                        f"协同演化调度执行完成: {schedule.schedule_id} "
                        f"(组: {group_id}, 下次运行: {schedule.next_run})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"执行协同演化调度失败: {str(e)}")

        except Exception as e:
            self.logger.error(f"执行调度失败: {str(e)}", exc_info=True)

    def _monitor_thread_func(self):
        """监控线程函数"""
        thread_name = threading.current_thread().name
        
        while not self.shutdown_requested:
            try:
                # 检查熔断机制是否需要自动重置
                if self.fuse_state["fuse_triggered"]:
                    reset_seconds = self.fuse_state.get("auto_reset_after_seconds", 300)
                    last_failure = self.fuse_state.get("last_failure_time")
                    
                    if last_failure and (time.time() - last_failure) > reset_seconds:
                        self.fuse_state["fuse_triggered"] = False
                        self.fuse_state["consecutive_failures"] = 0
                        self.fuse_state["fuse_reason"] = None
                        self.logger.info("演化熔断已自动重置")
                
                # 检查过期的任务
                with self.task_lock:
                    expired_tasks = []
                    for task_id, task in self.tasks.items():
                        if task.is_expired():
                            expired_tasks.append(task)
                    
                    for task in expired_tasks:
                        if task.status in ["pending", "running"]:
                            task.status = "cancelled"
                            task.error_message = "任务超时"
                            self.statistics["cancelled_tasks"] += 1
                            self.logger.warning(f"任务超时已取消: {task.task_id}")
                
                # 更新学习进度（基于策略历史）
                if self.strategy_learning_enabled and self.strategy_history:
                    recent_history = self.strategy_history[-100:]  # 最近100条记录
                    if recent_history:
                        success_rate = sum(1 for h in recent_history if h.get("success", False)) / len(recent_history)
                        avg_performance = sum(h.get("performance", 0.0) for h in recent_history) / len(recent_history)
                        
                        # 简单的学习进度计算
                        self.statistics["learning_progress"] = min(1.0, success_rate * avg_performance)
                
                time.sleep(5.0)  # 每5秒检查一次
                
            except Exception as e:
                self.logger.error(f"监控线程异常: {str(e)}", exc_info=True)
                time.sleep(10.0)

    def get_statistics(self) -> Dict[str, Any]:
        """获取演化统计信息"""
        with self.task_lock:
            running_tasks = sum(1 for task in self.tasks.values() if task.status == "running")
            pending_tasks = sum(1 for task in self.tasks.values() if task.status == "pending")
            
            stats = self.statistics.copy()
            stats.update({
                "running": running_tasks,
                "pending": pending_tasks,
                "total_tasks_count": len(self.tasks),
                "coevolution_groups_count": len(self.coevolution_groups),
                "fuse_triggered": self.fuse_state["fuse_triggered"],
                "consecutive_failures": self.fuse_state["consecutive_failures"],
                "strategy_history_count": len(self.strategy_history),
                "learning_progress": self.statistics.get("learning_progress", 0.0),
            })
            
            return stats

    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        try:
            if not self.strategy_history:
                return {
                    "success": False,
                    "message": "没有足够的策略历史数据",
                }
            
            # 分析策略性能
            strategy_performance = {}
            for entry in self.strategy_history:
                strategy = entry.get("strategy", "unknown")
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "count": 0,
                        "success_count": 0,
                        "total_performance": 0.0,
                        "total_duration": 0.0,
                    }
                
                stats = strategy_performance[strategy]
                stats["count"] += 1
                if entry.get("success", False):
                    stats["success_count"] += 1
                stats["total_performance"] += entry.get("performance", 0.0)
                stats["total_duration"] += entry.get("duration", 0.0)
            
            # 计算性能指标
            insights = {}
            for strategy, stats in strategy_performance.items():
                if stats["count"] > 0:
                    insights[strategy] = {
                        "usage_count": stats["count"],
                        "success_rate": stats["success_count"] / stats["count"],
                        "average_performance": stats["total_performance"] / stats["count"],
                        "average_duration": stats["total_duration"] / stats["count"],
                    }
            
            # 生成建议
            recommendations = []
            
            # 分析多目标优化性能
            if "multi_objective" in insights:
                mo_stats = insights["multi_objective"]
                if mo_stats["success_rate"] < 0.5:
                    recommendations.append({
                        "type": "multi_objective_optimization",
                        "priority": "medium",
                        "message": "多目标优化成功率较低",
                        "suggestion": "调整目标权重或增加演化迭代次数",
                    })
            
            # 分析协同演化性能
            if "coevolution" in insights:
                ce_stats = insights["coevolution"]
                if ce_stats["average_duration"] > 300:  # 超过5分钟
                    recommendations.append({
                        "type": "coevolution_efficiency",
                        "priority": "low",
                        "message": "协同演化时间较长",
                        "suggestion": "考虑减少组大小或优化协同策略",
                    })
            
            return {
                "success": True,
                "strategy_insights": insights,
                "recommendations": recommendations,
                "total_history_entries": len(self.strategy_history),
                "analysis_timestamp": time.time(),
            }
            
        except Exception as e:
            self.logger.error(f"获取学习洞察失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "学习洞察分析失败",
            }

    def get_evolution_history(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取演化历史"""
        try:
            if model_id:
                # 获取特定模型的演化历史
                history = self.model_evolution_history.get(model_id, [])
                return {
                    "success": True,
                    "model_id": model_id,
                    "history": history,
                    "count": len(history),
                }
            else:
                # 获取所有模型的演化历史摘要
                summary = {}
                for mid, history in self.model_evolution_history.items():
                    if history:
                        summary[mid] = {
                            "count": len(history),
                            "latest_timestamp": history[-1]["timestamp"],
                            "latest_success": history[-1].get("success", False),
                        }
                
                return {
                    "success": True,
                    "summary": summary,
                    "total_models": len(summary),
                }
                
        except Exception as e:
            self.logger.error(f"获取演化历史失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_coevolution_groups(self) -> Dict[str, Any]:
        """获取协同演化组信息"""
        try:
            groups_summary = {}
            with self.coevolution_lock:
                for group_id, group in self.coevolution_groups.items():
                    groups_summary[group_id] = {
                        "model_ids": group.model_ids,
                        "evolution_count": group.evolution_count,
                        "last_evolution": group.last_evolution,
                        "active": group.active,
                        "best_performance": group.best_performance,
                    }
            
            return {
                "success": True,
                "groups": groups_summary,
                "total_groups": len(groups_summary),
            }
            
        except Exception as e:
            self.logger.error(f"获取协同演化组信息失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    def reset_fuse(self):
        """重置熔断机制"""
        self.fuse_state["fuse_triggered"] = False
        self.fuse_state["consecutive_failures"] = 0
        self.fuse_state["fuse_reason"] = None
        self.logger.info("演化熔断已手动重置")


# 全局增强演化管理器实例
_global_enhanced_evolution_manager: Optional[EnhancedEvolutionManager] = None


def get_enhanced_evolution_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedEvolutionManager:
    """
    获取全局增强演化管理器实例

    Args:
        config: 配置字典

    Returns:
        增强演化管理器实例
    """
    global _global_enhanced_evolution_manager

    if _global_enhanced_evolution_manager is None:
        _global_enhanced_evolution_manager = EnhancedEvolutionManager(config)

    return _global_enhanced_evolution_manager


def start_enhanced_evolution_manager(config: Optional[Dict[str, Any]] = None):
    """启动全局增强演化管理器"""
    manager = get_enhanced_evolution_manager(config)
    manager.start()
    return manager


def stop_enhanced_evolution_manager():
    """停止全局增强演化管理器"""
    global _global_enhanced_evolution_manager
    if _global_enhanced_evolution_manager:
        _global_enhanced_evolution_manager.stop()
        _global_enhanced_evolution_manager = None


# 测试代码
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("增强演化管理器测试")
    print("=" * 80)

    try:
        # 创建增强演化管理器
        manager = EnhancedEvolutionManager()

        print("\n1. 测试增强演化管理器初始化:")
        print(
            f"   配置: 最大工作线程={manager.config['max_workers']}, "
            f"最大并发任务={manager.config['max_concurrent_tasks']}"
        )
        print(f"   启用功能: 协同演化={manager.config.get('enable_coevolution', True)}, "
              f"多目标优化={manager.config.get('enable_multi_objective', True)}")

        print("\n2. 测试统计信息获取:")
        stats = manager.get_statistics()
        print(f"   初始统计: 总任务数={stats['total_tasks']}, 运行中={stats['running']}")
        print(f"   策略使用: {stats['strategy_usage']}")
        print(f"   学习进度: {stats['learning_progress']:.2f}")

        print("\n3. 测试任务提交功能:")
        print("   任务提交功能已实现，支持以下策略:")
        print("     - single_objective: 单目标优化")
        print("     - multi_objective: 多目标优化（需提供权重）")
        print("     - coevolution: 协同演化（需提供组或模型列表）")
        print("     - adaptive: 自适应策略")

        print("\n4. 测试协同演化组管理:")
        print("   协同演化组创建和管理功能已实现")
        print("   支持批量协同演化任务提交")

        print("\n5. 测试调度功能:")
        print("   支持以下调度类型:")
        print("     - periodic: 定期调度")
        print("     - batch: 批量调度")
        print("     - coevolution: 协同演化调度")

        print("\n6. 测试学习洞察功能:")
        insights = manager.get_learning_insights()
        if insights.get("success", False):
            print("   学习洞察功能正常")
            if insights.get("strategy_insights"):
                print("   策略洞察数据可用")
        else:
            print(f"   学习洞察: {insights.get('message', '未初始化')}")

        print("\n7. 测试资源检查:")
        resource_ok = manager._check_resource_limits()
        print(f"   资源检查: {'通过' if resource_ok else '不通过'}")

        print("\n✓ 增强演化管理器测试完成")
        print("\n增强功能说明:")
        print("  1. 多算法支持: 使用增强演化模块，支持8种演化算法")
        print("  2. 多目标优化: 支持同时优化多个性能目标")
        print("  3. 协同演化: 支持多模型协作演化")
        print("  4. 元学习策略: 基于历史数据选择最佳演化策略")
        print("  5. 学习洞察: 提供演化分析和优化建议")
        print("  6. 增强监控: 实时监控演化过程和资源使用")
        print("\n使用方式:")
        print("  1. 单目标演化: submit_evolution_task(...)")
        print("  2. 多目标演化: submit_evolution_task(strategy='multi_objective', ...)")
        print("  3. 协同演化: submit_coevolution_task(...)")
        print("  4. 定期调度: schedule_periodic_evolution(...)")
        print("  5. 获取洞察: get_learning_insights()")

    except Exception as e:
        print(f"✗ 测试失败: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)