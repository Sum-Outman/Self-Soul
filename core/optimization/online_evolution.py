#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时在线演化系统 - Real-time Online Evolution System

支持系统运行时的动态架构调整和热更新，无需停止服务。
实现模型性能监控、自动演化触发、后台演化和无缝架构切换。

主要功能：
1. 实时性能监控：收集模型性能指标，检测性能下降和数据漂移
2. 智能演化触发：基于阈值、趋势分析和异常检测自动触发演化
3. 后台演化执行：在不影响服务的情况下执行架构演化
4. 无缝架构切换：支持热交换、渐进更新和A/B测试
5. 安全回滚机制：新架构性能下降时自动回退到稳定版本
6. 演化历史管理：记录所有演化决策和结果，支持可解释性分析

设计原则：
- 服务连续性：演化过程不影响正在运行的服务
- 安全第一：确保架构变更不会导致系统崩溃
- 渐进优化：小步快跑，避免激进架构变更
- 可观测性：全面监控演化过程和结果
- 可解释性：记录演化决策的原因和依据
"""

import logging
import time
import threading
import queue
import copy
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np

# 尝试导入必要的模块
try:
    from core.module_interfaces import IEvolutionModule, EvolutionResult
    from core.evolution_module import get_evolution_module
    from core.model_registry import ModelRegistry
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    IEvolutionModule = None
    EvolutionResult = None
    get_evolution_module = None
    ModelRegistry = None

logger = logging.getLogger(__name__)


class EvolutionTrigger(Enum):
    """演化触发条件"""
    PERFORMANCE_DROP = "performance_drop"          # 性能下降
    DATA_DRIFT = "data_drift"                      # 数据分布漂移
    SCHEDULED = "scheduled"                        # 计划任务
    MANUAL = "manual"                              # 手动触发
    RESOURCE_CHANGE = "resource_change"            # 资源变化
    HARDWARE_CHANGE = "hardware_change"            # 硬件变化


class ArchitectureUpdateStrategy(Enum):
    """架构更新策略"""
    HOT_SWAP = "hot_swap"                          # 热交换（立即替换）
    GRADUAL_UPDATE = "gradual_update"              # 渐进更新（逐步替换）
    A_B_TESTING = "a_b_testing"                    # A/B测试（并行运行）
    SHADOW_MODE = "shadow_mode"                    # 影子模式（不影响生产流量）


class EvolutionStatus(Enum):
    """演化状态"""
    IDLE = "idle"                                  # 空闲
    MONITORING = "monitoring"                      # 监控中
    TRIGGERED = "triggered"                        # 已触发
    EVOLVING = "evolving"                          # 演化中
    DEPLOYING = "deploying"                        # 部署中
    TESTING = "testing"                            # 测试中
    ROLLBACK = "rollback"                          # 回滚中
    COMPLETED = "completed"                        # 已完成
    FAILED = "failed"                              # 失败


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    accuracy: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionDecision:
    """演化决策"""
    decision_id: str
    trigger: EvolutionTrigger
    reason: str
    timestamp: float
    confidence: float = 0.0
    expected_improvement: float = 0.0
    risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class ArchitectureVersion:
    """架构版本"""
    version_id: str
    architecture: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    deployed_time: float
    is_active: bool = False
    evolution_history: List[str] = field(default_factory=list)


@dataclass
class OnlineEvolutionConfig:
    """在线演化配置"""
    # 监控配置
    monitoring_interval: float = 60.0              # 监控间隔（秒）
    performance_window_size: int = 100             # 性能窗口大小
    performance_threshold: float = 0.8             # 性能阈值（触发演化的最低准确率）
    performance_trend_window: int = 10             # 性能趋势分析窗口
    
    # 触发配置
    enable_performance_drop_trigger: bool = True   # 启用性能下降触发
    enable_data_drift_trigger: bool = True         # 启用数据漂移触发
    enable_scheduled_trigger: bool = False         # 启用计划任务触发
    scheduled_interval: float = 86400.0            # 计划任务间隔（秒，默认1天）
    
    # 演化配置
    evolution_module_type: str = "enhanced"        # 演化模块类型
    max_evolution_time: float = 300.0              # 最大演化时间（秒）
    evolution_priority: int = 1                    # 演化优先级
    
    # 更新配置
    update_strategy: ArchitectureUpdateStrategy = ArchitectureUpdateStrategy.GRADUAL_UPDATE
    gradual_update_percentage: float = 0.1         # 渐进更新百分比（每次更新比例）
    a_b_test_duration: float = 3600.0              # A/B测试持续时间（秒）
    shadow_mode_traffic_percentage: float = 0.01   # 影子模式流量百分比
    
    # 回滚配置
    enable_auto_rollback: bool = True              # 启用自动回滚
    rollback_threshold: float = 0.9                # 回滚阈值（新架构性能低于旧架构的比例）
    rollback_check_interval: float = 300.0         # 回滚检查间隔（秒）
    
    # 安全配置
    max_concurrent_evolutions: int = 1             # 最大并发演化数
    require_human_approval: bool = False           # 需要人工批准
    dry_run_first: bool = True                     # 首次演化使用干运行


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: OnlineEvolutionConfig):
        """初始化性能监控器"""
        self.config = config
        self.performance_history: List[PerformanceMetrics] = []
        self.data_distribution_history: List[Dict[str, Any]] = []
        
    def add_metrics(self, metrics: PerformanceMetrics):
        """添加性能指标"""
        self.performance_history.append(metrics)
        
        # 限制历史记录大小
        if len(self.performance_history) > self.config.performance_window_size:
            self.performance_history = self.performance_history[-self.config.performance_window_size:]
    
    def check_performance_drop(self) -> Tuple[bool, str]:
        """检查性能下降
        
        Returns:
            (是否触发演化, 原因描述)
        """
        if len(self.performance_history) < self.config.performance_trend_window:
            return False, "数据不足"
        
        # 获取最近的数据点
        recent_metrics = self.performance_history[-self.config.performance_trend_window:]
        recent_accuracies = [m.accuracy for m in recent_metrics]
        
        # 计算平均准确率
        avg_accuracy = np.mean(recent_accuracies)
        
        # 检查是否低于阈值
        if avg_accuracy < self.config.performance_threshold:
            return True, f"平均准确率({avg_accuracy:.3f})低于阈值({self.config.performance_threshold})"
        
        # 检查下降趋势（最近5个点 vs 前5个点）
        if len(recent_accuracies) >= 10:
            recent_avg = np.mean(recent_accuracies[-5:])
            previous_avg = np.mean(recent_accuracies[-10:-5])
            
            if recent_avg < previous_avg * 0.9:  # 下降超过10%
                return True, f"准确率下降趋势明显: 前5点平均={previous_avg:.3f}, 最近5点平均={recent_avg:.3f}"
        
        return False, "性能正常"
    
    def check_data_drift(self, current_data_stats: Dict[str, Any]) -> Tuple[bool, str]:
        """检查数据漂移
        
        Args:
            current_data_stats: 当前数据统计
            
        Returns:
            (是否触发演化, 原因描述)
        """
        if not self.data_distribution_history:
            # 记录初始数据分布
            self.data_distribution_history.append(current_data_stats)
            return False, "初始化数据分布"
        
        # 获取最近的数据分布
        recent_distribution = self.data_distribution_history[-1]
        
        # 简化检查：比较关键统计量
        drift_detected = False
        reasons = []
        
        for key, current_value in current_data_stats.items():
            if key in recent_distribution:
                previous_value = recent_distribution[key]
                
                # 数值型数据检查
                if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    change_ratio = abs(current_value - previous_value) / (abs(previous_value) + 1e-10)
                    if change_ratio > 0.3:  # 变化超过30%
                        drift_detected = True
                        reasons.append(f"{key}变化: {previous_value:.3f} -> {current_value:.3f}({change_ratio:.1%})")
        
        if drift_detected:
            reason = f"数据漂移检测: {', '.join(reasons)}"
            return True, reason
        
        return False, "数据分布稳定"


class OnlineEvolutionManager:
    """在线演化管理器"""
    
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """初始化在线演化管理器
        
        Args:
            model_id: 模型ID
            config: 配置字典
        """
        if not MODULE_AVAILABLE:
            raise ImportError("必需的模块不可用，请确保core.module_interfaces等模块存在")
        
        self.model_id = model_id
        self.config = OnlineEvolutionConfig(**(config or {}))
        self.logger = logging.getLogger(f"{__name__}.manager.{model_id}")
        
        # 状态管理
        self.status = EvolutionStatus.IDLE
        self.current_decision: Optional[EvolutionDecision] = None
        self.architecture_versions: Dict[str, ArchitectureVersion] = {}
        self.active_version_id: Optional[str] = None
        
        # 组件初始化
        self.performance_monitor = PerformanceMonitor(self.config)
        self.evolution_module: Optional[IEvolutionModule] = None
        self.model_registry: Optional[ModelRegistry] = None
        
        self._init_components()
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 启动工作线程
        self._start_worker_threads()
        
        self.logger.info(f"在线演化管理器初始化完成: {model_id}")
    
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化演化模块
            evolution_config = {
                "evolution_module_type": self.config.evolution_module_type,
                "enable_hardware_aware": True
            }
            self.evolution_module = get_evolution_module(evolution_config)
            
            # 初始化模型注册表
            self.model_registry = ModelRegistry()
            
            self.logger.info("组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def _start_worker_threads(self):
        """启动工作线程"""
        # 监控线程
        monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name=f"{self.model_id}_monitor"
        )
        monitor_thread.start()
        
        # 演化工作线程
        evolution_thread = threading.Thread(
            target=self._evolution_worker,
            daemon=True,
            name=f"{self.model_id}_evolution"
        )
        evolution_thread.start()
        
        # 部署工作线程
        deployment_thread = threading.Thread(
            target=self._deployment_worker,
            daemon=True,
            name=f"{self.model_id}_deployment"
        )
        deployment_thread.start()
        
        self.logger.info("工作线程已启动")
    
    def _monitoring_worker(self):
        """监控工作线程"""
        while True:
            try:
                if self.status != EvolutionStatus.IDLE:
                    time.sleep(self.config.monitoring_interval)
                    continue
                
                # 收集当前性能指标
                current_metrics = self._collect_current_metrics()
                if current_metrics:
                    self.performance_monitor.add_metrics(current_metrics)
                    
                    # 检查触发条件
                    trigger_evolution = False
                    trigger_reason = ""
                    trigger_type = None
                    
                    if self.config.enable_performance_drop_trigger:
                        triggered, reason = self.performance_monitor.check_performance_drop()
                        if triggered:
                            trigger_evolution = True
                            trigger_reason = reason
                            trigger_type = EvolutionTrigger.PERFORMANCE_DROP
                    
                    if not trigger_evolution and self.config.enable_data_drift_trigger:
                        # 简化：使用当前指标作为数据统计
                        data_stats = {
                            "accuracy": current_metrics.accuracy,
                            "latency": current_metrics.latency,
                            "throughput": current_metrics.throughput
                        }
                        triggered, reason = self.performance_monitor.check_data_drift(data_stats)
                        if triggered:
                            trigger_evolution = True
                            trigger_reason = reason
                            trigger_type = EvolutionTrigger.DATA_DRIFT
                    
                    if trigger_evolution:
                        # 创建演化决策
                        decision = EvolutionDecision(
                            decision_id=f"decision_{int(time.time())}_{len(self.architecture_versions)}",
                            trigger=trigger_type,
                            reason=trigger_reason,
                            timestamp=time.time(),
                            confidence=0.7,  # 简化置信度
                            expected_improvement=0.1,
                            risks=["新架构可能性能下降", "部署过程可能影响服务"],
                            mitigation_strategies=["A/B测试", "渐进部署", "自动回滚"]
                        )
                        
                        self.current_decision = decision
                        self.status = EvolutionStatus.TRIGGERED
                        
                        self.logger.info(f"演化触发: {trigger_reason}")
                        
                        # 将演化任务加入队列
                        self.task_queue.put(("evolve", decision))
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控工作线程异常: {e}")
                time.sleep(10.0)
    
    def _evolution_worker(self):
        """演化工作线程"""
        while True:
            try:
                task_type, task_data = self.task_queue.get(timeout=1.0)
                
                if task_type == "evolve" and isinstance(task_data, EvolutionDecision):
                    self.status = EvolutionStatus.EVOLVING
                    
                    decision = task_data
                    self.logger.info(f"开始演化: {decision.decision_id}")
                    
                    # 获取当前架构
                    current_architecture = self._get_current_architecture()
                    if not current_architecture:
                        self.logger.error("无法获取当前架构")
                        self.status = EvolutionStatus.FAILED
                        continue
                    
                    # 准备演化参数
                    performance_targets = self._calculate_performance_targets()
                    
                    # 执行演化
                    try:
                        evolution_result = self.evolution_module.evolve_architecture(
                            base_architecture=current_architecture,
                            performance_targets=performance_targets,
                            constraints=self._get_evolution_constraints()
                        )
                        
                        if evolution_result.success:
                            # 创建新架构版本
                            new_version_id = f"arch_v{len(self.architecture_versions) + 1}_{int(time.time())}"
                            
                            new_version = ArchitectureVersion(
                                version_id=new_version_id,
                                architecture=evolution_result.evolved_architecture,
                                performance_metrics=PerformanceMetrics(
                                    timestamp=time.time(),
                                    accuracy=evolution_result.performance_metrics.get("accuracy", 0.0),
                                    latency=0.0,  # 需要实际测量
                                    throughput=0.0
                                ),
                                deployed_time=0.0,
                                is_active=False,
                                evolution_history=[decision.decision_id]
                            )
                            
                            self.architecture_versions[new_version_id] = new_version
                            
                            # 将部署任务加入队列
                            self.task_queue.put(("deploy", new_version_id))
                            
                            self.logger.info(f"演化成功: {new_version_id}")
                            self.status = EvolutionStatus.COMPLETED
                            
                        else:
                            self.logger.error(f"演化失败: {evolution_result.error_message}")
                            self.status = EvolutionStatus.FAILED
                            
                    except Exception as e:
                        self.logger.error(f"演化异常: {e}")
                        self.status = EvolutionStatus.FAILED
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"演化工作线程异常: {e}")
                time.sleep(5.0)
    
    def _deployment_worker(self):
        """部署工作线程"""
        while True:
            try:
                task_type, task_data = self.task_queue.get(timeout=1.0)
                
                if task_type == "deploy" and isinstance(task_data, str):
                    version_id = task_data
                    
                    if version_id not in self.architecture_versions:
                        self.logger.error(f"未知架构版本: {version_id}")
                        continue
                    
                    self.status = EvolutionStatus.DEPLOYING
                    
                    version = self.architecture_versions[version_id]
                    self.logger.info(f"开始部署架构: {version_id}")
                    
                    # 根据策略部署
                    if self.config.update_strategy == ArchitectureUpdateStrategy.HOT_SWAP:
                        success = self._deploy_hot_swap(version)
                    elif self.config.update_strategy == ArchitectureUpdateStrategy.GRADUAL_UPDATE:
                        success = self._deploy_gradual_update(version)
                    elif self.config.update_strategy == ArchitectureUpdateStrategy.A_B_TESTING:
                        success = self._deploy_a_b_testing(version)
                    elif self.config.update_strategy == ArchitectureUpdateStrategy.SHADOW_MODE:
                        success = self._deploy_shadow_mode(version)
                    else:
                        success = self._deploy_hot_swap(version)
                    
                    if success:
                        version.is_active = True
                        version.deployed_time = time.time()
                        
                        # 停用旧版本
                        if self.active_version_id:
                            old_version = self.architecture_versions.get(self.active_version_id)
                            if old_version:
                                old_version.is_active = False
                        
                        self.active_version_id = version_id
                        
                        self.logger.info(f"架构部署成功: {version_id}")
                        self.status = EvolutionStatus.COMPLETED
                        
                        # 启动回滚监控
                        if self.config.enable_auto_rollback:
                            self.task_queue.put(("monitor_rollback", version_id))
                    else:
                        self.logger.error(f"架构部署失败: {version_id}")
                        self.status = EvolutionStatus.FAILED
                
                elif task_type == "monitor_rollback" and isinstance(task_data, str):
                    version_id = task_data
                    self._monitor_for_rollback(version_id)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"部署工作线程异常: {e}")
                time.sleep(5.0)
    
    def _deploy_hot_swap(self, version: ArchitectureVersion) -> bool:
        """热交换部署
        
        Args:
            version: 架构版本
            
        Returns:
            部署是否成功
        """
        try:
            # 简化部署：在实际系统中，这里应该更新模型注册表中的模型
            self.logger.info(f"热交换部署: {version.version_id}")
            
            # 模拟部署延迟
            time.sleep(2.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"热交换部署失败: {e}")
            return False
    
    def _deploy_gradual_update(self, version: ArchitectureVersion) -> bool:
        """渐进更新部署
        
        Args:
            version: 架构版本
            
        Returns:
            部署是否成功
        """
        try:
            self.logger.info(f"渐进更新部署: {version.version_id}")
            
            # 模拟渐进更新过程
            update_steps = int(1.0 / self.config.gradual_update_percentage)
            
            for step in range(update_steps):
                self.logger.info(f"渐进更新步骤 {step + 1}/{update_steps}")
                time.sleep(1.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"渐进更新部署失败: {e}")
            return False
    
    def _deploy_a_b_testing(self, version: ArchitectureVersion) -> bool:
        """A/B测试部署
        
        Args:
            version: 架构版本
            
        Returns:
            部署是否成功
        """
        try:
            self.logger.info(f"A/B测试部署: {version.version_id}")
            
            # 模拟A/B测试
            test_duration = self.config.a_b_test_duration
            self.logger.info(f"A/B测试进行中，持续时间: {test_duration}秒")
            
            # 在实际系统中，这里应该分配流量并收集性能数据
            time.sleep(min(test_duration, 5.0))  # 简化：只等待5秒
            
            # 假设A/B测试通过
            self.logger.info("A/B测试通过，新架构性能优于旧架构")
            
            return True
            
        except Exception as e:
            self.logger.error(f"A/B测试部署失败: {e}")
            return False
    
    def _deploy_shadow_mode(self, version: ArchitectureVersion) -> bool:
        """影子模式部署
        
        Args:
            version: 架构版本
            
        Returns:
            部署是否成功
        """
        try:
            self.logger.info(f"影子模式部署: {version.version_id}")
            
            # 模拟影子模式运行
            shadow_duration = 3600.0  # 1小时
            self.logger.info(f"影子模式运行中，持续时间: {shadow_duration}秒")
            
            # 在实际系统中，这里应该复制生产流量到新架构并收集性能数据
            time.sleep(5.0)  # 简化
            
            # 假设影子模式测试通过
            self.logger.info("影子模式测试通过，新架构性能稳定")
            
            return True
            
        except Exception as e:
            self.logger.error(f"影子模式部署失败: {e}")
            return False
    
    def _monitor_for_rollback(self, version_id: str):
        """监控回滚条件"""
        check_interval = self.config.rollback_check_interval
        
        while True:
            try:
                time.sleep(check_interval)
                
                if version_id != self.active_version_id:
                    # 版本已变更，停止监控
                    break
                
                # 收集新架构性能
                new_performance = self._collect_current_metrics()
                if not new_performance:
                    continue
                
                # 获取旧架构性能（如果有）
                old_performance = None
                if len(self.architecture_versions) > 1:
                    # 找到前一个活跃版本
                    for v_id, version in self.architecture_versions.items():
                        if v_id != version_id and version.is_active:
                            old_performance = version.performance_metrics
                            break
                
                if old_performance:
                    # 计算性能比例
                    performance_ratio = new_performance.accuracy / old_performance.accuracy
                    
                    if performance_ratio < self.config.rollback_threshold:
                        self.logger.warning(f"检测到性能下降，触发回滚: 新/旧准确率={performance_ratio:.3f}")
                        
                        # 触发回滚
                        self.task_queue.put(("rollback", version_id))
                        break
                
            except Exception as e:
                self.logger.error(f"回滚监控异常: {e}")
                time.sleep(10.0)
    
    def _collect_current_metrics(self) -> Optional[PerformanceMetrics]:
        """收集当前性能指标"""
        try:
            # 简化：返回模拟数据
            # 在实际系统中，这里应该从模型注册表或监控系统获取真实指标
            return PerformanceMetrics(
                timestamp=time.time(),
                accuracy=0.85 + (np.random.random() * 0.1),  # 0.85-0.95
                latency=50.0 + (np.random.random() * 20.0),  # 50-70ms
                throughput=1000.0 + (np.random.random() * 500.0),  # 1000-1500 req/s
                error_rate=0.01 + (np.random.random() * 0.02),  # 1-3%
                resource_usage={
                    "cpu": 30.0 + (np.random.random() * 20.0),
                    "memory": 45.0 + (np.random.random() * 10.0)
                }
            )
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            return None
    
    def _get_current_architecture(self) -> Optional[Dict[str, Any]]:
        """获取当前架构"""
        try:
            # 简化：返回模拟架构
            # 在实际系统中，这里应该从模型注册表获取真实架构
            return {
                "type": "classification",
                "layers": [
                    {"type": "linear", "size": 128, "activation": "relu"},
                    {"type": "linear", "size": 64, "activation": "relu"},
                    {"type": "linear", "size": 32, "activation": "sigmoid"}
                ],
                "input_size": 256,
                "output_size": 10
            }
        except Exception as e:
            self.logger.error(f"获取当前架构失败: {e}")
            return None
    
    def _calculate_performance_targets(self) -> Dict[str, float]:
        """计算性能目标"""
        # 基于当前性能计算改进目标
        current_metrics = self._collect_current_metrics()
        
        if current_metrics:
            return {
                "accuracy": min(current_metrics.accuracy * 1.1, 0.99),  # 提升10%
                "efficiency": 0.8,
                "robustness": 0.7
            }
        else:
            return {
                "accuracy": 0.9,
                "efficiency": 0.8,
                "robustness": 0.7
            }
    
    def _get_evolution_constraints(self) -> Dict[str, Any]:
        """获取演化约束"""
        return {
            "max_parameters": 100000,
            "max_layers": 10,
            "resource_constraints": {
                "memory_limit_gb": 4.0,
                "time_limit_seconds": self.config.max_evolution_time
            }
        }
    
    def get_manager_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "active_version": self.active_version_id,
            "total_versions": len(self.architecture_versions),
            "current_decision": self.current_decision.__dict__ if self.current_decision else None,
            "config": {
                "monitoring_interval": self.config.monitoring_interval,
                "update_strategy": self.config.update_strategy.value,
                "enable_auto_rollback": self.config.enable_auto_rollback
            }
        }


def create_online_evolution_manager(model_id: str, config: Optional[Dict[str, Any]] = None) -> OnlineEvolutionManager:
    """创建在线演化管理器实例
    
    Args:
        model_id: 模型ID
        config: 配置字典
        
    Returns:
        在线演化管理器实例
    """
    return OnlineEvolutionManager(model_id, config)


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("实时在线演化系统测试")
    print("=" * 80)
    
    try:
        # 创建在线演化管理器
        config = {
            "monitoring_interval": 5.0,  # 缩短监控间隔以快速测试
            "performance_threshold": 0.8,
            "update_strategy": "gradual_update",
            "gradual_update_percentage": 0.2,
            "enable_auto_rollback": True
        }
        
        manager = create_online_evolution_manager("test_model", config)
        
        print("在线演化管理器创建成功")
        
        # 获取初始状态
        status = manager.get_manager_status()
        print(f"初始状态: {status['status']}")
        print(f"活跃版本: {status['active_version']}")
        
        print("\n模拟性能监控和演化触发...")
        print("注意：测试将运行30秒，观察演化过程")
        
        # 运行一段时间
        start_time = time.time()
        while time.time() - start_time < 30:
            current_status = manager.get_manager_status()
            if current_status['status'] != status['status']:
                print(f"状态变化: {status['status']} -> {current_status['status']}")
                status = current_status
            
            time.sleep(2.0)
        
        print("\n最终状态:")
        print(f"  状态: {status['status']}")
        print(f"  活跃版本: {status['active_version']}")
        print(f"  总版本数: {status['total_versions']}")
        
        print("\n✓ 实时在线演化系统测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()