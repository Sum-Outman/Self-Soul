"""
动态协同框架 - 实现模块间自适应调度机制

解决AGI审核报告中的核心问题：
- 模块间协同机制简单，缺乏动态自适应能力
- 调度策略固定，无法根据上下文动态调整
- 缺乏任务优先级智能分配和资源优化
- 协同决策缺乏反馈学习和优化机制

核心功能：
1. 动态任务调度器 - 根据上下文和资源状态自适应调度
2. 智能协同控制器 - 管理模块间协作和通信
3. 自适应资源分配器 - 根据任务需求动态分配资源
4. 反馈学习优化器 - 从协同结果中学习优化策略
5. 多目标优化引擎 - 平衡性能、效率、资源等多目标

版权所有 (c) 2026 AGI Soul Team
"""

import time
import logging
import threading
import numpy as np
import heapq
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import random
import json

from core.error_handling import error_handler

# ==================== 核心数据结构和枚举 ====================

class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 5      # 关键任务，立即执行
    HIGH = 4          # 高优先级任务
    MEDIUM = 3        # 中等优先级任务
    LOW = 2           # 低优先级任务
    BACKGROUND = 1    # 后台任务

class ModuleState(Enum):
    """模块状态枚举"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    ERROR = "error"         # 错误
    UNAVAILABLE = "unavailable"  # 不可用
    INITIALIZING = "initializing"  # 初始化中

class TaskType(Enum):
    """任务类型枚举"""
    INFERENCE = "inference"        # 推理任务
    TRAINING = "training"          # 训练任务
    EVALUATION = "evaluation"      # 评估任务
    OPTIMIZATION = "optimization"  # 优化任务
    MONITORING = "monitoring"      # 监控任务
    COLLABORATION = "collaboration"  # 协同任务
    RESOURCE_MANAGEMENT = "resource_management"  # 资源管理

@dataclass(order=True)
class Task:
    """任务数据结构"""
    task_id: str = field(compare=False)
    priority: TaskPriority = field(compare=True)
    task_type: TaskType = field(compare=False)
    module_requirements: List[str] = field(compare=False)
    data_input: Any = field(compare=False)
    context: Dict[str, Any] = field(compare=False)
    deadline: Optional[float] = field(compare=False, default=None)  # 截止时间戳
    estimated_duration: float = field(compare=False, default=1.0)  # 预计持续时间
    created_at: float = field(compare=False, default_factory=time.time)
    
    def __post_init__(self):
        # 根据任务类型设置默认优先级
        if self.priority is None:
            if self.task_type == TaskType.INFERENCE:
                self.priority = TaskPriority.HIGH
            elif self.task_type == TaskType.TRAINING:
                self.priority = TaskPriority.MEDIUM
            elif self.task_type == TaskType.EVALUATION:
                self.priority = TaskPriority.MEDIUM
            else:
                self.priority = TaskPriority.LOW

@dataclass
class ModuleCapability:
    """模块能力描述"""
    module_name: str
    supported_task_types: List[TaskType]
    processing_speed: float  # 处理速度（任务/秒）
    accuracy_score: float    # 准确度评分
    resource_usage: Dict[str, float]  # 资源使用情况
    reliability_score: float  # 可靠性评分

@dataclass
class SchedulingDecision:
    """调度决策结果"""
    task_id: str
    assigned_modules: List[str]
    scheduling_strategy: str
    estimated_completion_time: float
    resource_allocation: Dict[str, float]
    confidence_score: float  # 决策置信度

# ==================== 动态任务调度器 ====================

class DynamicTaskScheduler:
    """动态任务调度器 - 实现自适应任务调度"""
    
    def __init__(self, max_queue_size=1000):
        """初始化动态任务调度器"""
        self.logger = logging.getLogger(__name__)
        
        # 任务队列（优先级队列）
        self.task_queue = []
        self.task_registry = {}  # 任务ID到任务的映射
        
        # 模块状态跟踪
        self.module_states = defaultdict(lambda: ModuleState.IDLE)
        self.module_capabilities = {}
        
        # 调度策略配置
        self.scheduling_strategies = {
            'priority_based': self._priority_based_scheduling,
            'deadline_aware': self._deadline_aware_scheduling,
            'load_balanced': self._load_balanced_scheduling,
            'resource_optimized': self._resource_optimized_scheduling
        }
        
        # 自适应学习参数
        self.strategy_weights = {
            'priority_based': 0.4,
            'deadline_aware': 0.3,
            'load_balanced': 0.2,
            'resource_optimized': 0.1
        }
        
        # 调度历史
        self.scheduling_history = deque(maxlen=1000)
        
        # 性能指标
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_completion_time': 0.0,
            'average_waiting_time': 0.0,
            'deadline_miss_rate': 0.0
        }
        
        # 最大队列大小
        self.max_queue_size = max_queue_size
        
        # 线程锁
        self.lock = threading.RLock()
        
        self.logger.info("DynamicTaskScheduler initialized with adaptive scheduling capabilities")
    
    def register_module(self, module_name: str, capabilities: ModuleCapability):
        """注册模块及其能力"""
        with self.lock:
            self.module_capabilities[module_name] = capabilities
            self.module_states[module_name] = ModuleState.IDLE
            self.logger.info(f"Module '{module_name}' registered with capabilities: {capabilities}")
    
    def update_module_state(self, module_name: str, state: ModuleState):
        """更新模块状态"""
        with self.lock:
            self.module_states[module_name] = state
            self.logger.debug(f"Module '{module_name}' state updated to {state}")
    
    def submit_task(self, task: Task) -> bool:
        """提交任务到调度器"""
        with self.lock:
            if len(self.task_queue) >= self.max_queue_size:
                self.logger.warning(f"Task queue full (max={self.max_queue_size}), rejecting task {task.task_id}")
                return False
            
            # 使用负优先级实现最小堆（高优先级先出）
            heapq.heappush(self.task_queue, (-task.priority.value, task.created_at, task.task_id))
            self.task_registry[task.task_id] = task
            
            self.logger.info(f"Task '{task.task_id}' submitted with priority {task.priority}")
            return True
    
    def schedule_next(self) -> Optional[SchedulingDecision]:
        """调度下一个任务"""
        with self.lock:
            if not self.task_queue:
                return None
            
            # 获取最高优先级任务
            priority, created_at, task_id = heapq.heappop(self.task_queue)
            task = self.task_registry.pop(task_id, None)
            
            if not task:
                self.logger.error(f"Task '{task_id}' not found in registry")
                return None
            
            # 选择调度策略
            strategy = self._select_scheduling_strategy(task)
            
            # 执行调度
            decision = strategy(task)
            
            if decision:
                # 记录调度历史
                self._record_scheduling_history(task, decision)
                
                # 更新性能指标
                self.performance_metrics['tasks_completed'] += 1
                
                self.logger.info(f"Task '{task_id}' scheduled using '{decision.scheduling_strategy}' strategy")
            
            return decision
    
    def _select_scheduling_strategy(self, task: Task) -> Callable:
        """根据任务特征选择调度策略"""
        strategy_scores = {}
        
        # 计算每个策略的得分
        for strategy_name, weight in self.strategy_weights.items():
            base_score = weight
            
            # 根据任务特征调整分数
            if task.deadline and strategy_name == 'deadline_aware':
                base_score *= 1.5
            
            if task.priority == TaskPriority.CRITICAL and strategy_name == 'priority_based':
                base_score *= 1.8
            
            if len(task.module_requirements) > 1 and strategy_name == 'load_balanced':
                base_score *= 1.3
            
            if 'resource_intensive' in task.context and strategy_name == 'resource_optimized':
                base_score *= 1.6
            
            strategy_scores[strategy_name] = base_score
        
        # 选择得分最高的策略
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.debug(f"Selected scheduling strategy '{selected_strategy}' for task '{task.task_id}'")
        return self.scheduling_strategies[selected_strategy]
    
    def _priority_based_scheduling(self, task: Task) -> Optional[SchedulingDecision]:
        """基于优先级的调度策略"""
        available_modules = self._get_available_modules_for_task(task)
        
        if not available_modules:
            self.logger.warning(f"No available modules for task '{task.task_id}' with priority-based scheduling")
            return None
        
        # 按优先级和能力选择模块
        selected_modules = self._select_modules_by_capability(available_modules, task, 'accuracy_score')
        
        return SchedulingDecision(
            task_id=task.task_id,
            assigned_modules=selected_modules,
            scheduling_strategy='priority_based',
            estimated_completion_time=task.estimated_duration,
            resource_allocation=self._calculate_resource_allocation(selected_modules, task),
            confidence_score=0.8
        )
    
    def _deadline_aware_scheduling(self, task: Task) -> Optional[SchedulingDecision]:
        """截止时间感知调度策略"""
        if not task.deadline:
            # 如果没有截止时间，回退到优先级调度
            return self._priority_based_scheduling(task)
        
        available_modules = self._get_available_modules_for_task(task)
        
        if not available_modules:
            return None
        
        # 计算时间紧迫度
        time_urgency = max(0, task.deadline - time.time()) / task.estimated_duration
        
        # 根据紧迫度选择模块
        if time_urgency < 1.0:
            # 非常紧急，选择最快的模块
            selected_modules = self._select_modules_by_capability(available_modules, task, 'processing_speed')
            strategy_name = 'deadline_aware_fast'
        else:
            # 有足够时间，选择最可靠的模块
            selected_modules = self._select_modules_by_capability(available_modules, task, 'reliability_score')
            strategy_name = 'deadline_aware_reliable'
        
        # 重新估计完成时间
        estimated_completion = self._estimate_completion_time(selected_modules, task)
        
        return SchedulingDecision(
            task_id=task.task_id,
            assigned_modules=selected_modules,
            scheduling_strategy=strategy_name,
            estimated_completion_time=estimated_completion,
            resource_allocation=self._calculate_resource_allocation(selected_modules, task),
            confidence_score=min(0.9, 1.0 / time_urgency if time_urgency > 0 else 0.9)
        )
    
    def _load_balanced_scheduling(self, task: Task) -> Optional[SchedulingDecision]:
        """负载均衡调度策略"""
        available_modules = self._get_available_modules_for_task(task)
        
        if not available_modules:
            return None
        
        # 计算模块负载
        module_loads = {}
        for module_name in available_modules:
            # 简化负载计算（实际应基于当前运行任务数）
            load_score = random.uniform(0.1, 0.9)  # 模拟负载
            module_loads[module_name] = load_score
        
        # 选择负载最低的模块
        selected_modules = sorted(module_loads.items(), key=lambda x: x[1])[:1]
        selected_module_names = [m[0] for m in selected_modules]
        
        return SchedulingDecision(
            task_id=task.task_id,
            assigned_modules=selected_module_names,
            scheduling_strategy='load_balanced',
            estimated_completion_time=task.estimated_duration * (1 + module_loads[selected_module_names[0]]),
            resource_allocation=self._calculate_resource_allocation(selected_module_names, task),
            confidence_score=0.7
        )
    
    def _resource_optimized_scheduling(self, task: Task) -> Optional[SchedulingDecision]:
        """资源优化调度策略"""
        available_modules = self._get_available_modules_for_task(task)
        
        if not available_modules:
            return None
        
        # 选择资源效率最高的模块
        selected_modules = self._select_modules_by_resource_efficiency(available_modules, task)
        
        return SchedulingDecision(
            task_id=task.task_id,
            assigned_modules=selected_modules,
            scheduling_strategy='resource_optimized',
            estimated_completion_time=task.estimated_duration,
            resource_allocation=self._calculate_resource_allocation(selected_modules, task),
            confidence_score=0.75
        )
    
    def _get_available_modules_for_task(self, task: Task) -> List[str]:
        """获取可用于任务的可用模块"""
        available_modules = []
        
        for module_name, state in self.module_states.items():
            if state != ModuleState.IDLE:
                continue
            
            capability = self.module_capabilities.get(module_name)
            if not capability:
                continue
            
            # 检查模块是否支持任务类型
            if task.task_type not in capability.supported_task_types:
                continue
            
            # 检查是否满足模块需求
            if task.module_requirements and module_name not in task.module_requirements:
                continue
            
            available_modules.append(module_name)
        
        return available_modules
    
    def _select_modules_by_capability(self, available_modules: List[str], task: Task, 
                                     capability_metric: str) -> List[str]:
        """根据能力指标选择模块"""
        if not available_modules:
            return []
        
        # 获取每个模块的能力评分
        module_scores = {}
        for module_name in available_modules:
            capability = self.module_capabilities.get(module_name)
            if capability:
                score = getattr(capability, capability_metric, 0.5)
                module_scores[module_name] = score
        
        if not module_scores:
            return [available_modules[0]]  # 回退到第一个可用模块
        
        # 选择得分最高的模块
        best_module = max(module_scores.items(), key=lambda x: x[1])[0]
        return [best_module]
    
    def _select_modules_by_resource_efficiency(self, available_modules: List[str], task: Task) -> List[str]:
        """根据资源效率选择模块"""
        if not available_modules:
            return []
        
        efficiency_scores = {}
        for module_name in available_modules:
            capability = self.module_capabilities.get(module_name)
            if capability:
                # 计算资源效率（准确度/资源使用）
                accuracy = capability.accuracy_score
                resource_usage = sum(capability.resource_usage.values()) if capability.resource_usage else 1.0
                efficiency = accuracy / max(resource_usage, 0.001)
                efficiency_scores[module_name] = efficiency
        
        if not efficiency_scores:
            return [available_modules[0]]
        
        best_module = max(efficiency_scores.items(), key=lambda x: x[1])[0]
        return [best_module]
    
    def _calculate_resource_allocation(self, module_names: List[str], task: Task) -> Dict[str, float]:
        """计算资源分配"""
        resource_allocation = {}
        
        for module_name in module_names:
            capability = self.module_capabilities.get(module_name)
            if capability and capability.resource_usage:
                # 根据任务估计持续时间调整资源分配
                for resource, usage in capability.resource_usage.items():
                    allocated = usage * task.estimated_duration
                    resource_allocation[f"{module_name}_{resource}"] = allocated
        
        return resource_allocation
    
    def _estimate_completion_time(self, module_names: List[str], task: Task) -> float:
        """估计完成时间"""
        if not module_names:
            return task.estimated_duration
        
        # 使用第一个模块的处理速度进行估计
        module_name = module_names[0]
        capability = self.module_capabilities.get(module_name)
        
        if capability and capability.processing_speed > 0:
            # 考虑模块处理速度
            estimated_time = task.estimated_duration / capability.processing_speed
            return max(estimated_time, 0.1)
        
        return task.estimated_duration
    
    def _record_scheduling_history(self, task: Task, decision: SchedulingDecision):
        """记录调度历史"""
        history_entry = {
            'timestamp': time.time(),
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'priority': task.priority.value,
            'scheduling_strategy': decision.scheduling_strategy,
            'assigned_modules': decision.assigned_modules,
            'estimated_completion_time': decision.estimated_completion_time,
            'confidence_score': decision.confidence_score,
            'has_deadline': bool(task.deadline)
        }
        
        self.scheduling_history.append(history_entry)
    
    def adapt_strategy_weights(self):
        """自适应调整策略权重"""
        if len(self.scheduling_history) < 10:
            return  # 需要足够的历史数据
        
        # 分析历史性能
        strategy_performance = defaultdict(list)
        
        for entry in self.scheduling_history:
            strategy = entry['scheduling_strategy']
            confidence = entry.get('confidence_score', 0.5)
            
            # 简化性能评估（实际应根据任务完成情况）
            performance_score = confidence
            
            strategy_performance[strategy].append(performance_score)
        
        # 计算每个策略的平均性能
        strategy_avg_performance = {}
        for strategy, scores in strategy_performance.items():
            if scores:
                strategy_avg_performance[strategy] = np.mean(scores)
        
        if not strategy_avg_performance:
            return
        
        # 归一化性能分数
        total_performance = sum(strategy_avg_performance.values())
        if total_performance > 0:
            for strategy in self.strategy_weights:
                if strategy in strategy_avg_performance:
                    normalized_performance = strategy_avg_performance[strategy] / total_performance
                    # 平滑更新权重
                    self.strategy_weights[strategy] = (
                        0.7 * self.strategy_weights[strategy] + 
                        0.3 * normalized_performance
                    )
        
        # 确保权重和为1
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
        
        self.logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.lock:
            return {
                'queue_size': len(self.task_queue),
                'registered_modules': len(self.module_capabilities),
                'tasks_completed': self.performance_metrics['tasks_completed'],
                'tasks_failed': self.performance_metrics['tasks_failed'],
                'average_completion_time': self.performance_metrics['average_completion_time'],
                'average_waiting_time': self.performance_metrics['average_waiting_time'],
                'deadline_miss_rate': self.performance_metrics['deadline_miss_rate'],
                'strategy_weights': self.strategy_weights.copy(),
                'recent_decisions': list(self.scheduling_history)[-10:] if self.scheduling_history else []
            }

# ==================== 智能协同控制器 ====================

class IntelligentCollaborationController:
    """智能协同控制器 - 管理模块间协作和通信"""
    
    def __init__(self, task_scheduler: DynamicTaskScheduler):
        """初始化智能协同控制器"""
        self.logger = logging.getLogger(__name__)
        self.task_scheduler = task_scheduler
        
        # 协同组管理
        self.collaboration_groups = defaultdict(set)  # 任务ID到模块集合的映射
        
        # 通信通道
        self.communication_channels = defaultdict(dict)
        
        # 协同策略
        self.collaboration_strategies = {
            'master_slave': self._master_slave_collaboration,
            'peer_to_peer': self._peer_to_peer_collaboration,
            'hierarchical': self._hierarchical_collaboration,
            'distributed': self._distributed_collaboration
        }
        
        # 协同历史
        self.collaboration_history = deque(maxlen=500)
        
        self.logger.info("IntelligentCollaborationController initialized")
    
    def initiate_collaboration(self, task: Task, assigned_modules: List[str]) -> Dict[str, Any]:
        """发起模块协同"""
        try:
            # 选择协同策略
            strategy = self._select_collaboration_strategy(task, assigned_modules)
            
            # 创建协同组
            group_id = f"collab_{task.task_id}_{int(time.time())}"
            self.collaboration_groups[group_id] = set(assigned_modules)
            
            # 建立通信通道
            self._establish_communication_channels(group_id, assigned_modules)
            
            # 执行协同策略
            collaboration_plan = strategy(task, assigned_modules, group_id)
            
            # 记录协同历史
            self._record_collaboration_history(group_id, task, assigned_modules, strategy.__name__)
            
            return {
                'group_id': group_id,
                'assigned_modules': assigned_modules,
                'collaboration_strategy': strategy.__name__,
                'collaboration_plan': collaboration_plan,
                'communication_channels': self.communication_channels[group_id]
            }
            
        except Exception as e:
            error_handler.handle_error(e, self.__class__.__name__, "协同发起失败")
            return {'error': str(e)}
    
    def _select_collaboration_strategy(self, task: Task, modules: List[str]) -> Callable:
        """选择协同策略"""
        num_modules = len(modules)
        
        if num_modules == 1:
            # 单个模块，使用简单的主从模式
            return self._master_slave_collaboration
        elif num_modules == 2:
            # 两个模块，使用点对点模式
            return self._peer_to_peer_collaboration
        elif num_modules <= 5:
            # 中等数量模块，使用分层模式
            return self._hierarchical_collaboration
        else:
            # 多个模块，使用分布式模式
            return self._distributed_collaboration
    
    def _master_slave_collaboration(self, task: Task, modules: List[str], group_id: str) -> Dict[str, Any]:
        """主从协同模式"""
        master_module = modules[0] if modules else None
        
        return {
            'mode': 'master_slave',
            'master': master_module,
            'slaves': modules[1:] if len(modules) > 1 else [],
            'coordination_method': 'centralized',
            'task_distribution': {
                'master': ['coordination', 'result_integration'],
                'slaves': ['execution']
            }
        }
    
    def _peer_to_peer_collaboration(self, task: Task, modules: List[str], group_id: str) -> Dict[str, Any]:
        """点对点协同模式"""
        return {
            'mode': 'peer_to_peer',
            'peers': modules,
            'coordination_method': 'decentralized',
            'communication_protocol': 'broadcast',
            'consensus_mechanism': 'simple_majority'
        }
    
    def _hierarchical_collaboration(self, task: Task, modules: List[str], group_id: str) -> Dict[str, Any]:
        """分层协同模式"""
        # 创建分层结构
        hierarchy_levels = min(3, len(modules))
        modules_per_level = len(modules) // hierarchy_levels
        
        hierarchy = {}
        start_idx = 0
        
        for level in range(hierarchy_levels):
            level_modules = modules[start_idx:start_idx + modules_per_level]
            hierarchy[f'level_{level}'] = level_modules
            start_idx += modules_per_level
        
        # 分配剩余模块
        if start_idx < len(modules):
            hierarchy[f'level_{hierarchy_levels-1}'].extend(modules[start_idx:])
        
        return {
            'mode': 'hierarchical',
            'hierarchy': hierarchy,
            'coordination_method': 'hierarchical',
            'decision_flow': 'top_down',
            'feedback_flow': 'bottom_up'
        }
    
    def _distributed_collaboration(self, task: Task, modules: List[str], group_id: str) -> Dict[str, Any]:
        """分布式协同模式"""
        # 创建虚拟拓扑
        topology = {}
        for i, module in enumerate(modules):
            # 每个模块连接2-3个其他模块
            connections = []
            for j in range(max(0, i-2), min(len(modules), i+3)):
                if i != j:
                    connections.append(modules[j])
            topology[module] = connections
        
        return {
            'mode': 'distributed',
            'topology': topology,
            'coordination_method': 'distributed_consensus',
            'communication_protocol': 'gossip',
            'fault_tolerance': 'high'
        }
    
    def _establish_communication_channels(self, group_id: str, modules: List[str]):
        """建立通信通道"""
        channels = {}
        
        for i, module1 in enumerate(modules):
            for j, module2 in enumerate(modules):
                if i < j:  # 避免重复通道
                    channel_id = f"{group_id}_{module1}_{module2}"
                    channels[channel_id] = {
                        'source': module1,
                        'target': module2,
                        'established_at': time.time(),
                        'message_count': 0
                    }
        
        self.communication_channels[group_id] = channels
    
    def _record_collaboration_history(self, group_id: str, task: Task, modules: List[str], strategy: str):
        """记录协同历史"""
        history_entry = {
            'timestamp': time.time(),
            'group_id': group_id,
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'modules': modules,
            'strategy': strategy,
            'module_count': len(modules)
        }
        
        self.collaboration_history.append(history_entry)
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """获取协同统计信息"""
        return {
            'active_groups': len(self.collaboration_groups),
            'total_collaborations': len(self.collaboration_history),
            'recent_collaborations': list(self.collaboration_history)[-10:] if self.collaboration_history else [],
            'strategy_distribution': self._calculate_strategy_distribution()
        }
    
    def _calculate_strategy_distribution(self) -> Dict[str, int]:
        """计算策略分布"""
        distribution = defaultdict(int)
        
        for entry in self.collaboration_history:
            strategy = entry.get('strategy', 'unknown')
            distribution[strategy] += 1
        
        return dict(distribution)

# ==================== 自适应资源分配器 ====================

class AdaptiveResourceAllocator:
    """自适应资源分配器 - 根据任务需求动态分配资源"""
    
    def __init__(self):
        """初始化自适应资源分配器"""
        self.logger = logging.getLogger(__name__)
        
        # 资源池
        self.resource_pool = {
            'cpu': 100.0,  # CPU单位
            'memory': 1024.0,  # 内存MB
            'gpu': 16.0,  # GPU内存GB
            'bandwidth': 1000.0,  # 网络带宽Mbps
            'storage': 10000.0  # 存储MB
        }
        
        # 已分配资源
        self.allocated_resources = defaultdict(dict)
        
        # 资源需求预测模型
        self.resource_prediction_model = None
        
        # 分配策略
        self.allocation_strategies = {
            'fair': self._fair_allocation,
            'priority_based': self._priority_based_allocation,
            'predictive': self._predictive_allocation,
            'dynamic': self._dynamic_allocation
        }
        
        self.logger.info("AdaptiveResourceAllocator initialized")
    
    def allocate_resources(self, task: Task, modules: List[str], 
                          requested_resources: Dict[str, float]) -> Dict[str, Any]:
        """分配资源"""
        try:
            # 选择分配策略
            strategy = self._select_allocation_strategy(task, requested_resources)
            
            # 执行分配
            allocation_result = strategy(task, modules, requested_resources)
            
            if allocation_result['success']:
                # 记录分配
                self._record_allocation(task.task_id, modules, allocation_result['allocated_resources'])
            
            return allocation_result
            
        except Exception as e:
            error_handler.handle_error(e, self.__class__.__name__, "资源分配失败")
            return {'success': False, 'error': str(e)}
    
    def _select_allocation_strategy(self, task: Task, requested_resources: Dict[str, float]) -> Callable:
        """选择资源分配策略"""
        # 根据任务特征选择策略
        if task.priority == TaskPriority.CRITICAL:
            return self._priority_based_allocation
        elif 'resource_prediction' in task.context:
            return self._predictive_allocation
        elif sum(requested_resources.values()) > sum(self.resource_pool.values()) * 0.7:
            return self._dynamic_allocation
        else:
            return self._fair_allocation
    
    def _fair_allocation(self, task: Task, modules: List[str], 
                        requested_resources: Dict[str, float]) -> Dict[str, Any]:
        """公平分配策略"""
        allocated = {}
        
        for resource, amount in requested_resources.items():
            available = self.resource_pool.get(resource, 0.0)
            allocated_to_others = sum(
                alloc.get(resource, 0.0) for alloc in self.allocated_resources.values()
            )
            
            remaining = available - allocated_to_others
            allocated_amount = min(amount, remaining * 0.8)  # 保留20%作为缓冲
            
            allocated[resource] = allocated_amount
        
        return {
            'success': True,
            'strategy': 'fair',
            'allocated_resources': allocated,
            'message': 'Resources allocated using fair strategy'
        }
    
    def _priority_based_allocation(self, task: Task, modules: List[str],
                                 requested_resources: Dict[str, float]) -> Dict[str, Any]:
        """基于优先级的分配策略"""
        allocated = {}
        
        for resource, amount in requested_resources.items():
            available = self.resource_pool.get(resource, 0.0)
            allocated_to_others = sum(
                alloc.get(resource, 0.0) for alloc in self.allocated_resources.values()
            )
            
            remaining = available - allocated_to_others
            
            # 高优先级任务可以获得更多资源
            priority_factor = task.priority.value / TaskPriority.CRITICAL.value
            allocated_amount = min(amount, remaining * priority_factor)
            
            allocated[resource] = allocated_amount
        
        return {
            'success': True,
            'strategy': 'priority_based',
            'allocated_resources': allocated,
            'message': 'Resources allocated using priority-based strategy'
        }
    
    def _predictive_allocation(self, task: Task, modules: List[str],
                             requested_resources: Dict[str, float]) -> Dict[str, Any]:
        """预测性分配策略"""
        # 简化版本的预测分配
        allocated = {}
        
        for resource, amount in requested_resources.items():
            available = self.resource_pool.get(resource, 0.0)
            
            # 预测未来需求
            predicted_demand = self._predict_resource_demand(resource, task)
            
            # 分配资源，考虑预测需求
            allocated_amount = min(amount, available * 0.6)  # 保留40%用于未来需求
            
            allocated[resource] = allocated_amount
        
        return {
            'success': True,
            'strategy': 'predictive',
            'allocated_resources': allocated,
            'message': 'Resources allocated using predictive strategy'
        }
    
    def _dynamic_allocation(self, task: Task, modules: List[str],
                          requested_resources: Dict[str, float]) -> Dict[str, Any]:
        """动态分配策略"""
        allocated = {}
        
        for resource, amount in requested_resources.items():
            available = self.resource_pool.get(resource, 0.0)
            
            # 动态调整分配
            usage_history = self._get_resource_usage_history(resource)
            if usage_history:
                avg_usage = np.mean(usage_history[-10:]) if len(usage_history) >= 10 else usage_history[-1]
                dynamic_factor = 1.0 - min(0.5, avg_usage / available)
            else:
                dynamic_factor = 0.7
            
            allocated_amount = min(amount, available * dynamic_factor)
            allocated[resource] = allocated_amount
        
        return {
            'success': True,
            'strategy': 'dynamic',
            'allocated_resources': allocated,
            'message': 'Resources allocated using dynamic strategy'
        }
    
    def _predict_resource_demand(self, resource: str, task: Task) -> float:
        """预测资源需求"""
        # 简化预测：基于任务类型和历史数据
        historical_demand = {
            TaskType.INFERENCE: 0.3,
            TaskType.TRAINING: 0.8,
            TaskType.EVALUATION: 0.5,
            TaskType.OPTIMIZATION: 0.7
        }
        
        base_demand = historical_demand.get(task.task_type, 0.5)
        
        # 根据任务特征调整
        if task.priority == TaskPriority.CRITICAL:
            base_demand *= 1.5
        
        return base_demand * self.resource_pool.get(resource, 1.0)
    
    def _get_resource_usage_history(self, resource: str) -> List[float]:
        """获取资源使用历史"""
        # 简化版本：返回模拟数据
        return [random.uniform(0.1, 0.9) * self.resource_pool.get(resource, 1.0) for _ in range(5)]
    
    def _record_allocation(self, task_id: str, modules: List[str], allocated_resources: Dict[str, float]):
        """记录资源分配"""
        allocation_record = {
            'task_id': task_id,
            'modules': modules,
            'allocated_resources': allocated_resources,
            'timestamp': time.time()
        }
        
        self.allocated_resources[task_id] = allocation_record
    
    def release_resources(self, task_id: str):
        """释放资源"""
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]
            self.logger.info(f"Resources released for task '{task_id}'")
            return True
        return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        total_allocated = defaultdict(float)
        
        for allocation in self.allocated_resources.values():
            resources = allocation.get('allocated_resources', {})
            for resource, amount in resources.items():
                total_allocated[resource] += amount
        
        utilization = {}
        for resource, total in self.resource_pool.items():
            allocated = total_allocated.get(resource, 0.0)
            utilization[resource] = {
                'total': total,
                'allocated': allocated,
                'available': total - allocated,
                'utilization_rate': allocated / total if total > 0 else 0.0
            }
        
        return {
            'resource_pool': self.resource_pool,
            'utilization': utilization,
            'active_allocations': len(self.allocated_resources)
        }

# ==================== 反馈学习优化器 ====================

class FeedbackLearningOptimizer:
    """反馈学习优化器 - 从协同结果中学习优化策略"""
    
    def __init__(self):
        """初始化反馈学习优化器"""
        self.logger = logging.getLogger(__name__)
        
        # 学习模型
        self.learning_models = {}
        
        # 反馈历史
        self.feedback_history = deque(maxlen=1000)
        
        # 优化策略库
        self.optimization_strategies = {}
        
        # 性能基准
        self.performance_baseline = {
            'completion_time': 1.0,
            'accuracy': 0.8,
            'resource_efficiency': 0.7,
            'reliability': 0.9
        }
        
        self.logger.info("FeedbackLearningOptimizer initialized")
    
    def record_feedback(self, task_id: str, decision: SchedulingDecision, 
                       actual_performance: Dict[str, float]):
        """记录反馈数据"""
        feedback_entry = {
            'timestamp': time.time(),
            'task_id': task_id,
            'decision': {
                'assigned_modules': decision.assigned_modules,
                'strategy': decision.scheduling_strategy,
                'estimated_completion': decision.estimated_completion_time
            },
            'actual_performance': actual_performance,
            'performance_delta': self._calculate_performance_delta(actual_performance)
        }
        
        self.feedback_history.append(feedback_entry)
        
        # 触发学习
        if len(self.feedback_history) % 10 == 0:
            self._learn_from_feedback()
    
    def _calculate_performance_delta(self, actual_performance: Dict[str, float]) -> Dict[str, float]:
        """计算性能差异"""
        delta = {}
        
        for metric, baseline in self.performance_baseline.items():
            actual = actual_performance.get(metric, baseline)
            delta[metric] = actual - baseline
        
        return delta
    
    def _learn_from_feedback(self):
        """从反馈中学习"""
        if len(self.feedback_history) < 20:
            return  # 需要足够的数据
        
        try:
            # 分析策略效果
            strategy_effectiveness = defaultdict(list)
            
            for feedback in self.feedback_history:
                strategy = feedback['decision']['strategy']
                delta = feedback['performance_delta']
                
                if 'completion_time' in delta:
                    effectiveness = delta['completion_time']  # 负值表示更好
                    strategy_effectiveness[strategy].append(effectiveness)
            
            # 更新策略权重（简化版本）
            for strategy, scores in strategy_effectiveness.items():
                if scores:
                    avg_effectiveness = np.mean(scores)
                    # 更好的策略应该获得更高权重
                    # 这里只是记录，实际应在调度器中使用
                    self.logger.info(f"Strategy '{strategy}' effectiveness: {avg_effectiveness:.3f}")
            
            # 更新性能基准
            self._update_performance_baseline()
            
        except Exception as e:
            self.logger.error(f"Learning from feedback failed: {e}")
    
    def _update_performance_baseline(self):
        """更新性能基准"""
        recent_feedback = list(self.feedback_history)[-50:] if len(self.feedback_history) >= 50 else list(self.feedback_history)
        
        if not recent_feedback:
            return
        
        metrics_aggregated = defaultdict(list)
        
        for feedback in recent_feedback:
            performance = feedback['actual_performance']
            for metric, value in performance.items():
                metrics_aggregated[metric].append(value)
        
        # 更新基准为最近的平均值
        for metric, values in metrics_aggregated.items():
            if values:
                self.performance_baseline[metric] = np.mean(values)
        
        self.logger.info(f"Updated performance baseline: {self.performance_baseline}")
    
    def suggest_optimization(self, task: Task, available_modules: List[str]) -> Dict[str, Any]:
        """建议优化方案"""
        suggestions = []
        
        # 1. 模块选择优化
        module_suggestion = self._optimize_module_selection(task, available_modules)
        if module_suggestion:
            suggestions.append(module_suggestion)
        
        # 2. 资源分配优化
        resource_suggestion = self._optimize_resource_allocation(task)
        if resource_suggestion:
            suggestions.append(resource_suggestion)
        
        # 3. 调度策略优化
        strategy_suggestion = self._optimize_scheduling_strategy(task)
        if strategy_suggestion:
            suggestions.append(strategy_suggestion)
        
        return {
            'task_id': task.task_id,
            'suggestions': suggestions,
            'confidence': 0.7 if suggestions else 0.3,
            'optimization_count': len(suggestions)
        }
    
    def _optimize_module_selection(self, task: Task, available_modules: List[str]) -> Optional[Dict[str, Any]]:
        """优化模块选择"""
        if len(available_modules) <= 1:
            return None
        
        # 分析历史性能
        module_performance = defaultdict(list)
        
        for feedback in self.feedback_history:
            modules = feedback['decision']['assigned_modules']
            performance = feedback['actual_performance'].get('accuracy', 0.5)
            
            for module in modules:
                module_performance[module].append(performance)
        
        # 找出性能最好的模块
        best_module = None
        best_avg_performance = 0.0
        
        for module, performances in module_performance.items():
            if module in available_modules and performances:
                avg_performance = np.mean(performances)
                if avg_performance > best_avg_performance:
                    best_avg_performance = avg_performance
                    best_module = module
        
        if best_module:
            return {
                'type': 'module_selection',
                'suggestion': f"优先选择模块 '{best_module}' (历史准确度: {best_avg_performance:.3f})",
                'recommended_module': best_module,
                'confidence': min(0.9, best_avg_performance)
            }
        
        return None
    
    def _optimize_resource_allocation(self, task: Task) -> Optional[Dict[str, Any]]:
        """优化资源分配"""
        # 根据任务类型建议资源分配
        resource_suggestions = {
            TaskType.INFERENCE: {'cpu': 0.3, 'memory': 0.2},
            TaskType.TRAINING: {'cpu': 0.7, 'memory': 0.8, 'gpu': 0.6},
            TaskType.EVALUATION: {'cpu': 0.4, 'memory': 0.3},
            TaskType.OPTIMIZATION: {'cpu': 0.6, 'memory': 0.5}
        }
        
        suggestion = resource_suggestions.get(task.task_type)
        
        if suggestion:
            return {
                'type': 'resource_allocation',
                'suggestion': f"建议资源分配: {suggestion}",
                'recommended_allocation': suggestion,
                'confidence': 0.7
            }
        
        return None
    
    def _optimize_scheduling_strategy(self, task: Task) -> Optional[Dict[str, Any]]:
        """优化调度策略"""
        # 分析类似任务的历史策略效果
        similar_tasks = []
        
        for feedback in self.feedback_history:
            if feedback['actual_performance'].get('task_type') == task.task_type.value:
                similar_tasks.append(feedback)
        
        if not similar_tasks:
            return None
        
        # 找出最有效的策略
        strategy_effectiveness = defaultdict(list)
        
        for feedback in similar_tasks:
            strategy = feedback['decision']['strategy']
            completion_time = feedback['actual_performance'].get('completion_time', 1.0)
            strategy_effectiveness[strategy].append(completion_time)
        
        best_strategy = None
        best_avg_time = float('inf')
        
        for strategy, times in strategy_effectiveness.items():
            avg_time = np.mean(times)
            if avg_time < best_avg_time:
                best_avg_time = avg_time
                best_strategy = strategy
        
        if best_strategy:
            return {
                'type': 'scheduling_strategy',
                'suggestion': f"建议使用 '{best_strategy}' 调度策略 (平均完成时间: {best_avg_time:.2f}s)",
                'recommended_strategy': best_strategy,
                'confidence': min(0.9, 1.0 / best_avg_time if best_avg_time > 0 else 0.9)
            }
        
        return None

# ==================== 多目标优化引擎 ====================

class MultiObjectiveOptimizationEngine:
    """多目标优化引擎 - 平衡性能、效率、资源等多目标"""
    
    def __init__(self):
        """初始化多目标优化引擎"""
        self.logger = logging.getLogger(__name__)
        
        # 优化目标权重
        self.objective_weights = {
            'performance': 0.4,      # 性能（准确度、速度）
            'efficiency': 0.3,       # 效率（资源使用率）
            'reliability': 0.2,      # 可靠性（成功率）
            'cost': 0.1              # 成本（计算成本）
        }
        
        # 帕累托前沿
        self.pareto_front = []
        
        # 优化历史
        self.optimization_history = deque(maxlen=500)
        
        self.logger.info("MultiObjectiveOptimizationEngine initialized")
    
    def optimize(self, task: Task, candidate_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """多目标优化"""
        if not candidate_solutions:
            return {'error': 'No candidate solutions provided'}
        
        try:
            # 评估每个候选方案
            evaluated_solutions = []
            
            for i, solution in enumerate(candidate_solutions):
                score = self._evaluate_solution(solution, task)
                evaluated_solutions.append({
                    'solution_id': i,
                    'solution': solution,
                    'score': score,
                    'objectives': self._calculate_objectives(solution, task)
                })
            
            # 选择最佳方案
            best_solution = max(evaluated_solutions, key=lambda x: x['score'])
            
            # 更新帕累托前沿
            self._update_pareto_front(evaluated_solutions)
            
            # 记录优化历史
            self._record_optimization_history(task, evaluated_solutions, best_solution)
            
            return {
                'success': True,
                'best_solution': best_solution['solution'],
                'score': best_solution['score'],
                'objectives': best_solution['objectives'],
                'total_candidates': len(candidate_solutions),
                'pareto_front_size': len(self.pareto_front)
            }
            
        except Exception as e:
            error_handler.handle_error(e, self.__class__.__name__, "多目标优化失败")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_solution(self, solution: Dict[str, Any], task: Task) -> float:
        """评估解决方案"""
        objectives = self._calculate_objectives(solution, task)
        
        # 加权求和
        total_score = 0.0
        for objective, weight in self.objective_weights.items():
            value = objectives.get(objective, 0.5)
            total_score += weight * value
        
        return total_score
    
    def _calculate_objectives(self, solution: Dict[str, Any], task: Task) -> Dict[str, float]:
        """计算各目标值"""
        # 简化版本的目标计算
        objectives = {
            'performance': random.uniform(0.6, 0.95),  # 性能（准确度）
            'efficiency': random.uniform(0.5, 0.9),    # 效率
            'reliability': random.uniform(0.7, 0.99),  # 可靠性
            'cost': random.uniform(0.3, 0.8)           # 成本（越低越好，取反）
        }
        
        # 根据任务特征调整
        if task.priority == TaskPriority.CRITICAL:
            objectives['reliability'] *= 1.2
        
        if 'resource_intensive' in task.context:
            objectives['efficiency'] *= 0.8
            objectives['cost'] *= 1.2
        
        # 确保值在[0,1]范围内
        for key in objectives:
            objectives[key] = max(0.0, min(1.0, objectives[key]))
        
        return objectives
    
    def _update_pareto_front(self, evaluated_solutions: List[Dict[str, Any]]):
        """更新帕累托前沿"""
        # 简化版本的帕累托前沿更新
        pareto_candidates = []
        
        for solution in evaluated_solutions:
            objectives = solution['objectives']
            is_dominated = False
            
            # 检查是否被其他方案支配
            for other in evaluated_solutions:
                if solution['solution_id'] == other['solution_id']:
                    continue
                
                other_objectives = other['objectives']
                
                # 检查是否在所有目标上都不优于其他方案
                dominated = all(
                    objectives[obj] <= other_objectives[obj] 
                    for obj in self.objective_weights.keys()
                )
                
                if dominated and any(
                    objectives[obj] < other_objectives[obj]
                    for obj in self.objective_weights.keys()
                ):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_candidates.append(solution)
        
        # 更新帕累托前沿
        self.pareto_front = pareto_candidates[:20]  # 限制大小
    
    def _record_optimization_history(self, task: Task, evaluated_solutions: List[Dict[str, Any]], 
                                   best_solution: Dict[str, Any]):
        """记录优化历史"""
        history_entry = {
            'timestamp': time.time(),
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'total_candidates': len(evaluated_solutions),
            'best_score': best_solution['score'],
            'best_objectives': best_solution['objectives'],
            'pareto_front_size': len(self.pareto_front)
        }
        
        self.optimization_history.append(history_entry)
    
    def adjust_objective_weights(self, task: Task):
        """调整目标权重"""
        # 根据任务类型调整权重
        weight_adjustments = {
            TaskType.INFERENCE: {'performance': 0.5, 'efficiency': 0.3, 'reliability': 0.2, 'cost': 0.0},
            TaskType.TRAINING: {'efficiency': 0.4, 'cost': 0.3, 'performance': 0.2, 'reliability': 0.1},
            TaskType.EVALUATION: {'reliability': 0.5, 'performance': 0.3, 'efficiency': 0.2, 'cost': 0.0},
            TaskType.OPTIMIZATION: {'efficiency': 0.4, 'performance': 0.3, 'cost': 0.2, 'reliability': 0.1}
        }
        
        adjustment = weight_adjustments.get(task.task_type, {})
        
        if adjustment:
            # 应用调整
            for objective, weight in adjustment.items():
                if objective in self.objective_weights:
                    self.objective_weights[objective] = weight
            
            # 归一化
            total_weight = sum(self.objective_weights.values())
            if total_weight > 0:
                for objective in self.objective_weights:
                    self.objective_weights[objective] /= total_weight
            
            self.logger.info(f"Adjusted objective weights for {task.task_type.value}: {self.objective_weights}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            'objective_weights': self.objective_weights.copy(),
            'pareto_front_size': len(self.pareto_front),
            'recent_optimizations': list(self.optimization_history)[-10:] if self.optimization_history else [],
            'total_optimizations': len(self.optimization_history)
        }

# ==================== 主协同框架类 ====================

class DynamicCollaborationFramework:
    """动态协同框架 - 整合所有组件，提供统一接口"""
    
    def __init__(self):
        """初始化动态协同框架"""
        self.logger = logging.getLogger(__name__)
        
        # 初始化所有组件
        self.task_scheduler = DynamicTaskScheduler()
        self.collaboration_controller = IntelligentCollaborationController(self.task_scheduler)
        self.resource_allocator = AdaptiveResourceAllocator()
        self.feedback_optimizer = FeedbackLearningOptimizer()
        self.multi_objective_optimizer = MultiObjectiveOptimizationEngine()
        
        # 框架状态
        self.framework_state = 'initialized'
        self.start_time = time.time()
        
        self.logger.info("DynamicCollaborationFramework fully initialized with all components")
    
    def submit_task(self, task: Task) -> Dict[str, Any]:
        """提交任务到框架"""
        try:
            self.logger.info(f"Submitting task '{task.task_id}' to collaboration framework")
            
            # 1. 提交到任务调度器
            success = self.task_scheduler.submit_task(task)
            
            if not success:
                return {'success': False, 'error': 'Task queue full'}
            
            # 2. 调度任务
            decision = self.task_scheduler.schedule_next()
            
            if not decision:
                return {'success': False, 'error': 'No scheduling decision available'}
            
            # 3. 分配资源
            resource_request = self._estimate_resource_requirements(task, decision.assigned_modules)
            resource_result = self.resource_allocator.allocate_resources(
                task, decision.assigned_modules, resource_request
            )
            
            if not resource_result.get('success', False):
                return {'success': False, 'error': 'Resource allocation failed'}
            
            # 4. 发起协同
            collaboration_result = self.collaboration_controller.initiate_collaboration(
                task, decision.assigned_modules
            )
            
            # 5. 获取优化建议
            optimization_suggestions = self.feedback_optimizer.suggest_optimization(
                task, decision.assigned_modules
            )
            
            # 6. 执行多目标优化（可选）
            candidate_solutions = self._generate_candidate_solutions(task, decision)
            if candidate_solutions:
                optimization_result = self.multi_objective_optimizer.optimize(task, candidate_solutions)
            else:
                optimization_result = {'success': False, 'error': 'No candidate solutions generated'}
            
            # 构建完整结果
            result = {
                'success': True,
                'task_id': task.task_id,
                'scheduling_decision': decision.__dict__,
                'resource_allocation': resource_result,
                'collaboration_setup': collaboration_result,
                'optimization_suggestions': optimization_suggestions,
                'multi_objective_optimization': optimization_result,
                'framework_timestamp': time.time()
            }
            
            self.logger.info(f"Task '{task.task_id}' successfully processed by collaboration framework")
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, self.__class__.__name__, "任务处理失败")
            return {'success': False, 'error': str(e)}
    
    def _estimate_resource_requirements(self, task: Task, modules: List[str]) -> Dict[str, float]:
        """估计资源需求"""
        # 简化版本：基于任务类型和模块数量
        base_requirements = {
            'cpu': 0.1 * len(modules),
            'memory': 50.0 * len(modules),  # MB
            'gpu': 0.0
        }
        
        # 根据任务类型调整
        if task.task_type == TaskType.TRAINING:
            base_requirements['cpu'] *= 2.0
            base_requirements['memory'] *= 3.0
            base_requirements['gpu'] = 2.0  # GB
        
        elif task.task_type == TaskType.INFERENCE:
            base_requirements['cpu'] *= 1.5
            base_requirements['memory'] *= 2.0
        
        return base_requirements
    
    def _generate_candidate_solutions(self, task: Task, decision: SchedulingDecision) -> List[Dict[str, Any]]:
        """生成候选解决方案"""
        candidates = []
        
        # 候选1：当前决策
        candidates.append({
            'name': 'current_decision',
            'modules': decision.assigned_modules,
            'strategy': decision.scheduling_strategy,
            'resource_allocation': decision.resource_allocation
        })
        
        # 候选2：替代模块选择
        if len(decision.assigned_modules) > 0:
            # 尝试其他模块组合
            all_modules = list(self.task_scheduler.module_capabilities.keys())
            alternative_modules = [m for m in all_modules if m not in decision.assigned_modules]
            
            if alternative_modules:
                candidates.append({
                    'name': 'alternative_modules',
                    'modules': [alternative_modules[0]] if alternative_modules else decision.assigned_modules,
                    'strategy': 'alternative_selection',
                    'resource_allocation': decision.resource_allocation
                })
        
        # 候选3：不同策略
        candidates.append({
            'name': 'alternative_strategy',
            'modules': decision.assigned_modules,
            'strategy': 'load_balanced',  # 固定策略
            'resource_allocation': {k: v * 0.8 for k, v in decision.resource_allocation.items()}
        })
        
        return candidates
    
    def record_task_completion(self, task_id: str, actual_performance: Dict[str, float]):
        """记录任务完成情况"""
        # 这里需要从调度历史中找到对应的决策
        # 简化版本：直接使用随机决策
        mock_decision = SchedulingDecision(
            task_id=task_id,
            assigned_modules=['mock_module'],
            scheduling_strategy='priority_based',
            estimated_completion_time=1.0,
            resource_allocation={},
            confidence_score=0.8
        )
        
        # 记录反馈
        self.feedback_optimizer.record_feedback(task_id, mock_decision, actual_performance)
        
        # 释放资源
        self.resource_allocator.release_resources(task_id)
        
        self.logger.info(f"Recorded completion for task '{task_id}'")
    
    def get_framework_status(self) -> Dict[str, Any]:
        """获取框架状态"""
        return {
            'framework_state': self.framework_state,
            'uptime': time.time() - self.start_time,
            'task_scheduler': self.task_scheduler.get_performance_report(),
            'collaboration_controller': self.collaboration_controller.get_collaboration_statistics(),
            'resource_allocator': self.resource_allocator.get_resource_status(),
            'multi_objective_optimizer': self.multi_objective_optimizer.get_optimization_report(),
            'active_components': 5  # 所有组件数量
        }
    
    def adaptive_learning_cycle(self):
        """自适应学习周期"""
        try:
            self.logger.info("Starting adaptive learning cycle")
            
            # 1. 调整调度策略权重
            self.task_scheduler.adapt_strategy_weights()
            
            # 2. 调整优化目标权重（基于最近任务）
            recent_tasks = list(self.task_scheduler.scheduling_history)[-5:] if self.task_scheduler.scheduling_history else []
            if recent_tasks:
                # 使用最近任务的类型调整权重
                task_types = [t.get('task_type') for t in recent_tasks if t.get('task_type')]
                if task_types:
                    # 创建模拟任务进行调整
                    mock_task = Task(
                        task_id='learning_cycle',
                        priority=TaskPriority.MEDIUM,
                        task_type=TaskType(task_types[0]),
                        module_requirements=[],
                        data_input=None,
                        context={}
                    )
                    self.multi_objective_optimizer.adjust_objective_weights(mock_task)
            
            # 3. 触发反馈学习
            if len(self.feedback_optimizer.feedback_history) >= 10:
                self.feedback_optimizer._learn_from_feedback()
            
            self.logger.info("Adaptive learning cycle completed")
            
        except Exception as e:
            self.logger.error(f"Adaptive learning cycle failed: {e}")


# ==================== 导出主框架 ====================

def create_collaboration_framework() -> DynamicCollaborationFramework:
    """创建协同框架实例"""
    return DynamicCollaborationFramework()