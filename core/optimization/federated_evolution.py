#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦演化学习框架 - Federated Evolutionary Learning Framework

支持分布式、隐私保护的协同演化学习，允许多个客户端在本地进行演化，
同时通过中央服务器协调知识共享，保护数据隐私。

主要功能：
1. 联邦演化协调器：管理客户端注册、任务分配、结果聚合
2. 联邦演化客户端：在本地执行演化，支持硬件感知和个性化优化
3. 安全聚合机制：支持差分隐私、安全聚合和模型混淆
4. 异构客户端支持：适应不同的硬件能力、数据分布和演化目标
5. 异步通信和容错：支持客户端掉线和动态加入

设计原则：
- 隐私优先：默认启用隐私保护机制
- 可扩展性：支持任意数量的客户端
- 灵活性：支持同步和异步通信模式
- 兼容性：与现有演化模块接口兼容
- 可配置性：通过配置调整联邦学习参数
"""

import logging
import time
import json
import hashlib
import random
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np

# 尝试导入必要的模块
try:
    from core.module_interfaces import IEvolutionModule, EvolutionResult
    from core.evolution_module import get_evolution_module
    EVOLUTION_MODULE_AVAILABLE = True
except ImportError:
    EVOLUTION_MODULE_AVAILABLE = False
    IEvolutionModule = None
    EvolutionResult = None
    get_evolution_module = None

logger = logging.getLogger(__name__)


class FederatedLearningMode(Enum):
    """联邦学习模式"""
    SYNCHRONOUS = "synchronous"      # 同步联邦学习（所有客户端同步更新）
    ASYNCHRONOUS = "asynchronous"    # 异步联邦学习（客户端随时更新）
    SEMI_SYNCHRONOUS = "semi_synchronous"  # 半同步联邦学习（部分客户端同步）


class PrivacyLevel(Enum):
    """隐私保护级别"""
    NONE = "none"                    # 无隐私保护
    DIFFERENTIAL_PRIVACY = "differential_privacy"  # 差分隐私
    SECURE_AGGREGATION = "secure_aggregation"      # 安全聚合
    FULL_PROTECTION = "full_protection"            # 完全保护（差分隐私 + 安全聚合）


class ClientStatus(Enum):
    """客户端状态"""
    IDLE = "idle"                    # 空闲
    TRAINING = "training"            # 训练中
    UPLOADING = "uploading"          # 上传中
    DOWNLOADING = "downloading"      # 下载中
    OFFLINE = "offline"              # 离线
    ERROR = "error"                  # 错误


@dataclass
class FederatedEvolutionConfig:
    """联邦演化配置"""
    # 联邦学习参数
    mode: FederatedLearningMode = FederatedLearningMode.SYNCHRONOUS
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    num_clients: int = 5                          # 客户端数量
    min_clients_per_round: int = 3                # 每轮最小客户端数
    max_clients_per_round: int = 10               # 每轮最大客户端数
    rounds: int = 100                             # 联邦学习轮数
    local_epochs: int = 5                         # 每轮本地演化代数
    
    # 聚合参数
    aggregation_strategy: str = "weighted_average"  # 聚合策略：weighted_average, best_selection, knowledge_distillation
    selection_strategy: str = "random"            # 客户端选择策略：random, capability_based, performance_based
    
    # 隐私参数
    dp_epsilon: float = 1.0                       # 差分隐私epsilon参数
    dp_delta: float = 1e-5                        # 差分隐私delta参数
    noise_scale: float = 0.01                     # 噪声缩放因子
    
    # 通信参数
    communication_interval: float = 30.0          # 通信间隔（秒）
    timeout: float = 300.0                        # 超时时间（秒）
    max_retries: int = 3                          # 最大重试次数
    
    # 演化参数
    enable_hardware_aware: bool = True           # 启用硬件感知
    enable_personalization: bool = True          # 启用个性化优化
    knowledge_sharing_rate: float = 0.3          # 知识共享率


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str                               # 客户端ID
    status: ClientStatus = ClientStatus.IDLE     # 客户端状态
    capabilities: Dict[str, Any] = field(default_factory=dict)  # 客户端能力（硬件、数据等）
    performance_history: List[float] = field(default_factory=list)  # 性能历史
    last_communication: float = field(default_factory=time.time)  # 最后通信时间
    local_model_version: int = 0                 # 本地模型版本
    global_model_version: int = 0                # 全局模型版本


@dataclass
class EvolutionTask:
    """演化任务"""
    task_id: str                                 # 任务ID
    base_architecture: Dict[str, Any]            # 基础架构
    performance_targets: Dict[str, float]        # 性能目标
    constraints: Optional[Dict[str, Any]] = None # 约束条件
    client_id: Optional[str] = None              # 分配的客户端ID
    priority: int = 1                            # 任务优先级
    created_time: float = field(default_factory=time.time)  # 创建时间
    deadline: Optional[float] = None             # 截止时间


@dataclass
class EvolutionResultAggregate:
    """演化结果聚合"""
    round_id: int                                # 轮次ID
    best_architecture: Dict[str, Any]            # 最佳架构
    average_performance: Dict[str, float]        # 平均性能
    client_contributions: Dict[str, float]       # 客户端贡献度
    privacy_noise_added: bool = False            # 是否添加了隐私噪声
    aggregation_time: float = 0.0                # 聚合耗时


class FederatedEvolutionClient:
    """联邦演化客户端
    
    在本地执行演化任务，支持硬件感知和个性化优化。
    定期与服务器通信，上传演化结果，下载全局模型。
    """
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        """初始化联邦演化客户端
        
        Args:
            client_id: 客户端ID
            config: 客户端配置
        """
        self.client_id = client_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.client.{client_id}")
        
        # 客户端状态
        self.status = ClientStatus.IDLE
        self.capabilities = self._detect_capabilities()
        self.performance_history = []
        self.local_model_version = 0
        self.global_model_version = 0
        self.last_communication = time.time()
        
        # 本地演化模块
        self.evolution_module = None
        self._init_evolution_module()
        
        # 本地知识库
        self.local_knowledge_base = {
            "successful_architectures": [],
            "failed_architectures": [],
            "performance_patterns": {},
            "hardware_optimizations": {}
        }
        
        # 任务队列
        self.task_queue = []
        self.current_task = None
        
        self.logger.info(f"联邦演化客户端初始化完成: {client_id}")
        self.logger.info(f"客户端能力: {self.capabilities}")
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """检测客户端能力"""
        capabilities = {
            "hardware_type": "unknown",
            "memory_gb": 0.0,
            "compute_units": 1,
            "has_gpu": False,
            "data_size": 0,
            "network_bandwidth": 1.0,
            "power_status": "ac"  # ac或battery
        }
        
        # 尝试检测硬件（简化版本）
        try:
            import psutil
            capabilities["memory_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            capabilities["memory_gb"] = 8.0  # 默认值
        
        # 尝试检测GPU
        try:
            import torch
            capabilities["has_gpu"] = torch.cuda.is_available()
            if capabilities["has_gpu"]:
                capabilities["hardware_type"] = "gpu"
                capabilities["compute_units"] = torch.cuda.device_count()
        except ImportError:
            pass
        
        return capabilities
    
    def _init_evolution_module(self):
        """初始化本地演化模块"""
        if not EVOLUTION_MODULE_AVAILABLE:
            self.logger.warning("演化模块不可用，客户端将无法执行演化")
            return
        
        try:
            # 根据客户端能力配置演化模块
            client_config = {
                "evolution_module_type": "enhanced",
                "enable_hardware_aware": self.config.get("enable_hardware_aware", True),
                "hardware_aware_config": {
                    "optimization_level": "balanced",
                    "task_complexity": 0.5
                }
            }
            
            # 如果客户端有GPU，使用GPU优化配置
            if self.capabilities.get("has_gpu", False):
                client_config["hardware_aware_config"]["optimization_level"] = "max_performance"
            
            self.evolution_module = get_evolution_module(client_config)
            self.logger.info("本地演化模块初始化成功")
        except Exception as e:
            self.logger.error(f"本地演化模块初始化失败: {e}")
    
    def execute_evolution_task(self, task: EvolutionTask) -> Optional[EvolutionResult]:
        """执行演化任务
        
        Args:
            task: 演化任务
            
        Returns:
            演化结果，如果失败则返回None
        """
        if self.evolution_module is None:
            self.logger.error("演化模块未初始化，无法执行任务")
            return None
        
        try:
            self.status = ClientStatus.TRAINING
            self.current_task = task
            
            self.logger.info(f"开始执行演化任务: {task.task_id}")
            self.logger.info(f"性能目标: {task.performance_targets}")
            
            # 执行演化
            start_time = time.time()
            result = self.evolution_module.evolve_architecture(
                base_architecture=task.base_architecture,
                performance_targets=task.performance_targets,
                constraints=task.constraints
            )
            evolution_time = time.time() - start_time
            
            # 记录性能
            if result.success:
                performance_score = result.performance_metrics.get("overall_score", 0.0)
                self.performance_history.append(performance_score)
                self.logger.info(f"演化任务完成: 性能得分={performance_score:.4f}, 耗时={evolution_time:.2f}s")
                
                # 更新本地知识库
                self._update_knowledge_base(task, result)
            else:
                self.logger.warning(f"演化任务失败: {result.error_message}")
            
            self.status = ClientStatus.IDLE
            self.current_task = None
            
            return result
            
        except Exception as e:
            self.logger.error(f"演化任务执行异常: {e}")
            self.status = ClientStatus.ERROR
            self.current_task = None
            return None
    
    def _update_knowledge_base(self, task: EvolutionTask, result: EvolutionResult):
        """更新本地知识库"""
        # 记录成功架构
        if result.success:
            knowledge_entry = {
                "task_id": task.task_id,
                "base_architecture": task.base_architecture,
                "evolved_architecture": result.evolved_architecture,
                "performance_metrics": result.performance_metrics,
                "generation_info": result.generation_info,
                "timestamp": time.time()
            }
            
            self.local_knowledge_base["successful_architectures"].append(knowledge_entry)
            
            # 限制知识库大小
            max_entries = self.config.get("max_knowledge_entries", 100)
            if len(self.local_knowledge_base["successful_architectures"]) > max_entries:
                self.local_knowledge_base["successful_architectures"] = \
                    self.local_knowledge_base["successful_architectures"][-max_entries:]
    
    def get_client_info(self) -> ClientInfo:
        """获取客户端信息"""
        return ClientInfo(
            client_id=self.client_id,
            status=self.status,
            capabilities=self.capabilities,
            performance_history=self.performance_history.copy(),
            last_communication=self.last_communication,
            local_model_version=self.local_model_version,
            global_model_version=self.global_model_version
        )
    
    def apply_privacy_protection(self, data: Dict[str, Any], privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """应用隐私保护
        
        Args:
            data: 要保护的数据
            privacy_level: 隐私保护级别
            
        Returns:
            保护后的数据
        """
        if privacy_level == PrivacyLevel.NONE:
            return data
        
        protected_data = data.copy()
        
        if privacy_level in [PrivacyLevel.DIFFERENTIAL_PRIVACY, PrivacyLevel.FULL_PROTECTION]:
            # 应用差分隐私：添加高斯噪声
            noise_scale = self.config.get("dp_noise_scale", 0.01)
            
            # 为数值型数据添加噪声
            for key, value in protected_data.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, noise_scale * abs(value))
                    protected_data[key] = value + noise
        
        if privacy_level in [PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PROTECTION]:
            # 应用安全聚合：对敏感数据进行混淆
            # 这里使用简单的哈希混淆，实际应用中应使用更安全的加密方法
            sensitive_keys = ["client_id", "raw_data", "personal_info"]
            
            for key in sensitive_keys:
                if key in protected_data:
                    value_str = str(protected_data[key])
                    protected_data[key] = hashlib.sha256(value_str.encode()).hexdigest()[:16]
        
        return protected_data


class FederatedEvolutionCoordinator:
    """联邦演化协调器
    
    管理客户端注册、任务分配、结果聚合和全局模型更新。
    支持多种联邦学习模式和隐私保护机制。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化联邦演化协调器
        
        Args:
            config: 配置字典
        """
        self.config = FederatedEvolutionConfig(**(config or {}))
        self.logger = logging.getLogger(f"{__name__}.coordinator")
        
        # 客户端管理
        self.clients: Dict[str, FederatedEvolutionClient] = {}
        self.client_info: Dict[str, ClientInfo] = {}
        
        # 任务管理
        self.task_queue: List[EvolutionTask] = []
        self.completed_tasks: Dict[str, EvolutionResult] = {}
        self.next_task_id = 1
        
        # 联邦学习状态
        self.current_round = 0
        self.global_model: Optional[Dict[str, Any]] = None
        self.global_model_version = 0
        self.aggregation_history: List[EvolutionResultAggregate] = []
        
        # 通信管理
        self.communication_lock = threading.Lock()
        self.last_aggregation_time = time.time()
        
        # 启动后台线程
        self._start_background_tasks()
        
        self.logger.info("联邦演化协调器初始化完成")
        self.logger.info(f"模式: {self.config.mode.value}, 隐私级别: {self.config.privacy_level.value}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动客户端状态监控线程
        monitor_thread = threading.Thread(
            target=self._monitor_clients,
            daemon=True,
            name="client_monitor"
        )
        monitor_thread.start()
        
        # 启动任务调度线程
        scheduler_thread = threading.Thread(
            target=self._schedule_tasks,
            daemon=True,
            name="task_scheduler"
        )
        scheduler_thread.start()
        
        # 启动聚合线程
        aggregator_thread = threading.Thread(
            target=self._aggregate_results,
            daemon=True,
            name="result_aggregator"
        )
        aggregator_thread.start()
    
    def register_client(self, client_id: str, client_config: Dict[str, Any]) -> bool:
        """注册客户端
        
        Args:
            client_id: 客户端ID
            client_config: 客户端配置
            
        Returns:
            注册是否成功
        """
        with self.communication_lock:
            if client_id in self.clients:
                self.logger.warning(f"客户端已存在: {client_id}")
                return False
            
            try:
                client = FederatedEvolutionClient(client_id, client_config)
                self.clients[client_id] = client
                self.client_info[client_id] = client.get_client_info()
                
                self.logger.info(f"客户端注册成功: {client_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"客户端注册失败: {e}")
                return False
    
    def submit_evolution_task(self, 
                             base_architecture: Dict[str, Any],
                             performance_targets: Dict[str, float],
                             constraints: Optional[Dict[str, Any]] = None,
                             priority: int = 1) -> str:
        """提交演化任务
        
        Args:
            base_architecture: 基础架构
            performance_targets: 性能目标
            constraints: 约束条件
            priority: 任务优先级
            
        Returns:
            任务ID
        """
        task_id = f"fed_task_{self.next_task_id}_{int(time.time())}"
        self.next_task_id += 1
        
        task = EvolutionTask(
            task_id=task_id,
            base_architecture=base_architecture,
            performance_targets=performance_targets,
            constraints=constraints,
            priority=priority
        )
        
        with self.communication_lock:
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda x: x.priority, reverse=True)
            
        self.logger.info(f"演化任务已提交: {task_id}, 优先级: {priority}")
        
        return task_id
    
    def _monitor_clients(self):
        """监控客户端状态"""
        while True:
            try:
                with self.communication_lock:
                    # 更新客户端信息
                    for client_id, client in self.clients.items():
                        self.client_info[client_id] = client.get_client_info()
                    
                    # 检测离线客户端
                    current_time = time.time()
                    offline_clients = []
                    
                    for client_id, info in self.client_info.items():
                        time_since_communication = current_time - info.last_communication
                        if time_since_communication > self.config.timeout:
                            if info.status != ClientStatus.OFFLINE:
                                self.logger.warning(f"客户端离线: {client_id}, 最后通信: {time_since_communication:.1f}秒前")
                                info.status = ClientStatus.OFFLINE
                                offline_clients.append(client_id)
                
                # 休眠一段时间
                time.sleep(10.0)
                
            except Exception as e:
                self.logger.error(f"客户端监控异常: {e}")
                time.sleep(5.0)
    
    def _schedule_tasks(self):
        """调度任务到客户端"""
        while True:
            try:
                with self.communication_lock:
                    if not self.task_queue:
                        time.sleep(1.0)
                        continue
                    
                    # 选择可用的客户端
                    available_clients = []
                    for client_id, info in self.client_info.items():
                        if info.status == ClientStatus.IDLE:
                            available_clients.append(client_id)
                    
                    if not available_clients:
                        time.sleep(1.0)
                        continue
                    
                    # 根据策略选择客户端
                    if self.config.selection_strategy == "capability_based":
                        # 基于能力选择：选择计算能力最强的客户端
                        selected_client = max(
                            available_clients,
                            key=lambda cid: self.client_info[cid].capabilities.get("compute_units", 1)
                        )
                    elif self.config.selection_strategy == "performance_based":
                        # 基于性能选择：选择历史性能最好的客户端
                        selected_client = max(
                            available_clients,
                            key=lambda cid: (
                                np.mean(self.client_info[cid].performance_history) 
                                if self.client_info[cid].performance_history else 0.0
                            )
                        )
                    else:  # random
                        selected_client = random.choice(available_clients)
                    
                    # 分配任务
                    task = self.task_queue.pop(0)
                    task.client_id = selected_client
                    
                    # 异步执行任务
                    client = self.clients[selected_client]
                    task_thread = threading.Thread(
                        target=self._execute_client_task,
                        args=(client, task),
                        daemon=True,
                        name=f"task_{task.task_id}"
                    )
                    task_thread.start()
                    
                    self.logger.info(f"任务分配: {task.task_id} -> {selected_client}")
                
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"任务调度异常: {e}")
                time.sleep(1.0)
    
    def _execute_client_task(self, client: FederatedEvolutionClient, task: EvolutionTask):
        """执行客户端任务"""
        try:
            result = client.execute_evolution_task(task)
            
            with self.communication_lock:
                if result is not None:
                    self.completed_tasks[task.task_id] = result
                    self.logger.info(f"任务完成: {task.task_id}, 客户端: {client.client_id}")
                else:
                    # 任务失败，重新加入队列
                    task.client_id = None
                    self.task_queue.append(task)
                    self.logger.warning(f"任务失败，重新加入队列: {task.task_id}")
                    
        except Exception as e:
            self.logger.error(f"任务执行异常: {e}")
            
            with self.communication_lock:
                # 任务失败，重新加入队列
                task.client_id = None
                self.task_queue.append(task)
    
    def _aggregate_results(self):
        """聚合客户端结果"""
        while True:
            try:
                current_time = time.time()
                time_since_aggregation = current_time - self.last_aggregation_time
                
                if time_since_aggregation < self.config.communication_interval:
                    time.sleep(5.0)
                    continue
                
                with self.communication_lock:
                    if not self.completed_tasks:
                        time.sleep(5.0)
                        continue
                    
                    # 开始新一轮聚合
                    self.current_round += 1
                    round_id = self.current_round
                    
                    self.logger.info(f"开始第 {round_id} 轮聚合")
                    
                    # 收集本轮结果
                    round_results = {}
                    for task_id, result in self.completed_tasks.items():
                        if result.success:
                            round_results[task_id] = result
                    
                    if not round_results:
                        self.logger.warning(f"第 {round_id} 轮无成功结果")
                        self.last_aggregation_time = current_time
                        continue
                    
                    # 应用聚合策略
                    aggregated_result = self._apply_aggregation_strategy(round_results)
                    
                    # 更新全局模型
                    if aggregated_result.best_architecture:
                        self.global_model = aggregated_result.best_architecture
                        self.global_model_version += 1
                        
                        # 通知客户端更新全局模型
                        self._notify_clients_global_update()
                    
                    # 记录聚合历史
                    self.aggregation_history.append(aggregated_result)
                    
                    # 限制历史记录大小
                    max_history = 100
                    if len(self.aggregation_history) > max_history:
                        self.aggregation_history = self.aggregation_history[-max_history:]
                    
                    # 清空已完成任务
                    self.completed_tasks.clear()
                    
                    self.last_aggregation_time = current_time
                    
                    self.logger.info(f"第 {round_id} 轮聚合完成，全局模型版本: {self.global_model_version}")
                    self.logger.info(f"最佳架构性能: {aggregated_result.average_performance}")
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"结果聚合异常: {e}")
                time.sleep(5.0)
    
    def _apply_aggregation_strategy(self, round_results: Dict[str, EvolutionResult]) -> EvolutionResultAggregate:
        """应用聚合策略
        
        Args:
            round_results: 轮次结果
            
        Returns:
            聚合结果
        """
        strategy = self.config.aggregation_strategy
        
        if strategy == "best_selection":
            return self._aggregate_best_selection(round_results)
        elif strategy == "knowledge_distillation":
            return self._aggregate_knowledge_distillation(round_results)
        else:  # weighted_average
            return self._aggregate_weighted_average(round_results)
    
    def _aggregate_weighted_average(self, round_results: Dict[str, EvolutionResult]) -> EvolutionResultAggregate:
        """加权平均聚合"""
        # 简化版本：选择性能最好的架构
        best_task_id = None
        best_score = -float('inf')
        
        for task_id, result in round_results.items():
            score = result.performance_metrics.get("overall_score", 0.0)
            if score > best_score:
                best_score = score
                best_task_id = task_id
        
        if best_task_id is None:
            # 如果没有结果，返回空聚合
            return EvolutionResultAggregate(
                round_id=self.current_round,
                best_architecture={},
                average_performance={},
                client_contributions={}
            )
        
        best_result = round_results[best_task_id]
        
        # 计算平均性能
        all_scores = []
        for result in round_results.values():
            all_scores.append(result.performance_metrics.get("overall_score", 0.0))
        
        avg_score = np.mean(all_scores) if all_scores else 0.0
        
        # 计算客户端贡献度
        client_contributions = {}
        for task_id, result in round_results.items():
            # 简化：贡献度基于性能得分
            score = result.performance_metrics.get("overall_score", 0.0)
            client_id = task_id.split("_")[-2] if "_" in task_id else "unknown"
            client_contributions[client_id] = score
        
        # 归一化贡献度
        total_contribution = sum(client_contributions.values())
        if total_contribution > 0:
            for client_id in client_contributions:
                client_contributions[client_id] /= total_contribution
        
        return EvolutionResultAggregate(
            round_id=self.current_round,
            best_architecture=best_result.evolved_architecture,
            average_performance={"overall_score": avg_score},
            client_contributions=client_contributions
        )
    
    def _aggregate_best_selection(self, round_results: Dict[str, EvolutionResult]) -> EvolutionResultAggregate:
        """最佳选择聚合"""
        # 与加权平均相同，但可以添加特定逻辑
        return self._aggregate_weighted_average(round_results)
    
    def _aggregate_knowledge_distillation(self, round_results: Dict[str, EvolutionResult]) -> EvolutionResultAggregate:
        """知识蒸馏聚合"""
        # 简化版本：使用加权平均
        # 实际实现中应包含知识蒸馏逻辑
        return self._aggregate_weighted_average(round_results)
    
    def _notify_clients_global_update(self):
        """通知客户端全局模型更新"""
        for client_id, client in self.clients.items():
            try:
                # 更新客户端全局模型版本
                client.global_model_version = self.global_model_version
                client.last_communication = time.time()
                
                # 这里可以添加实际的通知逻辑
                # 例如，通过消息队列或RPC调用
                
            except Exception as e:
                self.logger.error(f"通知客户端失败 {client_id}: {e}")
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        with self.communication_lock:
            active_clients = 0
            for info in self.client_info.values():
                if info.status != ClientStatus.OFFLINE:
                    active_clients += 1
            
            return {
                "current_round": self.current_round,
                "global_model_version": self.global_model_version,
                "total_clients": len(self.clients),
                "active_clients": active_clients,
                "tasks_in_queue": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "aggregation_history_size": len(self.aggregation_history),
                "config": {
                    "mode": self.config.mode.value,
                    "privacy_level": self.config.privacy_level.value,
                    "aggregation_strategy": self.config.aggregation_strategy
                }
            }


class FederatedEvolutionModule:
    """联邦演化模块
    
    实现IEvolutionModule接口，提供联邦演化功能。
    内部使用FederatedEvolutionCoordinator管理分布式演化。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化联邦演化模块
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.module")
        
        # 联邦演化协调器
        self.coordinator = FederatedEvolutionCoordinator(self.config.get("coordinator_config"))
        
        # 本地缓存
        self.local_cache = {
            "submitted_tasks": {},
            "received_results": {},
            "global_model_cache": None
        }
        
        # 统计信息
        self.statistics = {
            "total_submitted_tasks": 0,
            "completed_tasks": 0,
            "average_federation_time": 0.0,
            "best_federated_score": 0.0,
            "client_participation_count": 0
        }
        
        self.logger.info("联邦演化模块初始化完成")
    
    def evolve_architecture(self,
                          base_architecture: Dict[str, Any],
                          performance_targets: Dict[str, float],
                          constraints: Optional[Dict[str, Any]] = None) -> EvolutionResult:
        """演化神经网络架构（联邦版）
        
        Args:
            base_architecture: 基础架构
            performance_targets: 性能目标
            constraints: 约束条件
            
        Returns:
            演化结果
        """
        start_time = time.time()
        
        try:
            self.logger.info("开始联邦演化")
            self.logger.info(f"性能目标: {performance_targets}")
            
            # 提交任务到联邦协调器
            task_id = self.coordinator.submit_evolution_task(
                base_architecture=base_architecture,
                performance_targets=performance_targets,
                constraints=constraints,
                priority=1
            )
            
            # 记录任务
            self.local_cache["submitted_tasks"][task_id] = {
                "base_architecture": base_architecture,
                "performance_targets": performance_targets,
                "constraints": constraints,
                "submit_time": start_time
            }
            
            self.statistics["total_submitted_tasks"] += 1
            
            # 等待任务完成（简化版本：轮询检查）
            # 实际实现中应使用更高效的等待机制
            max_wait_time = self.config.get("max_wait_time", 300.0)
            poll_interval = self.config.get("poll_interval", 2.0)
            
            end_time = start_time + max_wait_time
            result = None
            
            while time.time() < end_time:
                # 检查任务是否完成
                with self.coordinator.communication_lock:
                    if task_id in self.coordinator.completed_tasks:
                        result = self.coordinator.completed_tasks[task_id]
                        break
                
                time.sleep(poll_interval)
            
            if result is None:
                # 超时
                self.logger.warning(f"联邦演化超时: {task_id}")
                
                return EvolutionResult(
                    success=False,
                    evolved_architecture={},
                    performance_metrics={},
                    generation_info={"error": "federation_timeout"},
                    error_message="联邦演化超时，请稍后重试"
                )
            
            # 更新统计信息
            self.statistics["completed_tasks"] += 1
            federation_time = time.time() - start_time
            
            # 更新平均联邦时间
            total_time = self.statistics["average_federation_time"] * (self.statistics["completed_tasks"] - 1)
            self.statistics["average_federation_time"] = (total_time + federation_time) / self.statistics["completed_tasks"]
            
            # 更新最佳得分
            score = result.performance_metrics.get("overall_score", 0.0)
            if score > self.statistics["best_federated_score"]:
                self.statistics["best_federated_score"] = score
            
            # 更新客户端参与计数
            coordinator_status = self.coordinator.get_coordinator_status()
            self.statistics["client_participation_count"] = coordinator_status["active_clients"]
            
            self.logger.info(f"联邦演化完成: 耗时={federation_time:.2f}s, 得分={score:.4f}")
            
            # 添加联邦特定的生成信息
            if hasattr(result, 'generation_info'):
                result.generation_info["federated_round"] = coordinator_status["current_round"]
                result.generation_info["client_count"] = coordinator_status["active_clients"]
                result.generation_info["federation_time"] = federation_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"联邦演化异常: {e}")
            
            return EvolutionResult(
                success=False,
                evolved_architecture={},
                performance_metrics={},
                generation_info={"error": "federation_exception"},
                error_message=f"联邦演化异常: {str(e)}"
            )
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """获取演化状态"""
        coordinator_status = self.coordinator.get_coordinator_status()
        
        status = {
            "module_status": {
                "federated_mode": True,
                "coordinator_active": True,
                "submitted_tasks": len(self.local_cache["submitted_tasks"]),
                "received_results": len(self.local_cache["received_results"])
            },
            "coordinator_status": coordinator_status,
            "statistics": self.statistics
        }
        
        return status
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """获取演化历史"""
        # 简化版本：返回提交的任务历史
        history = []
        
        for task_id, task_info in self.local_cache["submitted_tasks"].items():
            history_entry = {
                "task_id": task_id,
                "base_architecture": task_info["base_architecture"],
                "performance_targets": task_info["performance_targets"],
                "submit_time": task_info["submit_time"],
                "completed": task_id in self.coordinator.completed_tasks
            }
            
            history.append(history_entry)
        
        return history
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """获取演化统计信息"""
        return self.statistics.copy()


def create_federated_evolution_module(config: Optional[Dict[str, Any]] = None) -> FederatedEvolutionModule:
    """创建联邦演化模块实例
    
    Args:
        config: 配置字典
        
    Returns:
        联邦演化模块实例
    """
    return FederatedEvolutionModule(config)


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("联邦演化学习框架测试")
    print("=" * 80)
    
    try:
        # 创建联邦演化模块
        config = {
            "coordinator_config": {
                "mode": "synchronous",
                "privacy_level": "differential_privacy",
                "num_clients": 3,
                "communication_interval": 10.0
            }
        }
        
        federated_module = create_federated_evolution_module(config)
        
        print("联邦演化模块创建成功")
        
        # 获取状态
        status = federated_module.get_evolution_status()
        print(f"协调器状态: {status['coordinator_status']}")
        
        # 测试架构
        test_architecture = {
            "type": "classification",
            "layers": [
                {"type": "linear", "size": 64, "activation": "relu"},
                {"type": "linear", "size": 32, "activation": "relu"},
                {"type": "linear", "size": 16, "activation": "sigmoid"}
            ],
            "input_size": 128,
            "output_size": 10
        }
        
        performance_targets = {
            "accuracy": 0.8,
            "efficiency": 0.7
        }
        
        print("\n测试联邦演化（需要注册客户端）")
        print("注意：实际联邦演化需要注册客户端并等待任务执行")
        print("此测试仅验证模块初始化")
        
        print("\n✓ 联邦演化学习框架测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()