"""
多模型协同调度器 - Multi-Model Collaborative Scheduler

整合智能任务分配器和模型协作协调器，提供完整的AGI多模型协同调度能力。
实现感知-决策-行动-反馈闭环，支持动态任务分配、负载均衡和冲突解决。

核心功能：
1. 模型注册和能力管理
2. 智能任务分配与调度
3. 协同工作流编排
4. 性能监控与优化
5. 冲突检测与解决
6. 自主闭环管理
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import uuid

logger = logging.getLogger(__name__)

from core.collaboration.intelligent_task_distributor import (
    ModelCapability, TaskComplexity, ModelCapabilityProfile,
    TaskRequirements, IntelligentTaskDistributor, get_intelligent_distributor,
    get_capability_registry
)
from core.collaboration.model_collaborator import (
    CollaborationMode, ModelCollaborationOrchestrator, collaboration_orchestrator
)

class SchedulingStrategy(Enum):
    """调度策略枚举"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # 性能优化
    LOAD_BALANCED = "load_balanced"                  # 负载均衡
    FAULT_TOLERANT = "fault_tolerant"               # 容错优先
    ENERGY_EFFICIENT = "energy_efficient"           # 能效优先
    QUALITY_FOCUSED = "quality_focused"             # 质量优先

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    ASSIGNED = "assigned"        # 已分配
    EXECUTING = "executing"      # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 取消

class FeedbackType(Enum):
    """反馈类型枚举"""
    PERFORMANCE = "performance"      # 性能反馈
    QUALITY = "quality"              # 质量反馈
    RESOURCE_USAGE = "resource_usage"  # 资源使用反馈
    COLLABORATION = "collaboration"  # 协作反馈
    ERROR = "error"                  # 错误反馈

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_name: str
    task_description: str
    required_capabilities: List[ModelCapability]
    assigned_models: List[str] = field(default_factory=list)
    assigned_decision: Optional[Dict[str, Any]] = None
    collaboration_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    priority: int = 1  # 1-10，越高越优先
    deadline: Optional[float] = None  # 截止时间戳

@dataclass
class FeedbackData:
    """反馈数据"""
    feedback_id: str
    task_id: str
    feedback_type: FeedbackType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    processed: bool = False

@dataclass
class SchedulingMetrics:
    """调度指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    avg_assignment_time: float = 0.0
    success_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    model_utilization: Dict[str, float] = field(default_factory=dict)
    last_update_time: float = field(default_factory=time.time)

class MultiModelScheduler:
    """多模型协同调度器"""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.PERFORMANCE_OPTIMIZED):
        """初始化调度器"""
        self.scheduling_strategy = strategy
        self.task_distributor = get_intelligent_distributor()
        self.collaboration_orchestrator = collaboration_orchestrator
        self.capability_registry = get_capability_registry()
        
        # 任务管理
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue = deque()
        self.completed_tasks = deque(maxlen=1000)
        
        # 反馈系统
        self.feedback_history: List[FeedbackData] = []
        self.feedback_handlers: Dict[FeedbackType, List[Callable]] = defaultdict(list)
        
        # 调度指标
        self.metrics = SchedulingMetrics()
        self.performance_history = deque(maxlen=100)
        
        # 调度锁
        self.lock = threading.RLock()
        self.scheduling_thread = None
        self.running = False
        
        # 初始化调度器
        self._initialize_scheduler()
        
        logger.info(f"多模型协同调度器初始化完成，策略: {strategy.value}")
    
    def _initialize_scheduler(self):
        """初始化调度器"""
        # 注册反馈处理器
        self._register_feedback_handlers()
        
        # 初始化默认调度策略
        self._initialize_scheduling_strategies()
        
        # 初始化性能监控
        self._initialize_performance_monitoring()
    
    def _register_feedback_handlers(self):
        """注册反馈处理器"""
        # 性能反馈处理器
        self.feedback_handlers[FeedbackType.PERFORMANCE].append(
            self._handle_performance_feedback
        )
        
        # 质量反馈处理器
        self.feedback_handlers[FeedbackType.QUALITY].append(
            self._handle_quality_feedback
        )
        
        # 资源使用反馈处理器
        self.feedback_handlers[FeedbackType.RESOURCE_USAGE].append(
            self._handle_resource_feedback
        )
        
        # 协作反馈处理器
        self.feedback_handlers[FeedbackType.COLLABORATION].append(
            self._handle_collaboration_feedback
        )
        
        # 错误反馈处理器
        self.feedback_handlers[FeedbackType.ERROR].append(
            self._handle_error_feedback
        )
    
    def _initialize_scheduling_strategies(self):
        """初始化调度策略"""
        self.strategy_handlers = {
            SchedulingStrategy.PERFORMANCE_OPTIMIZED: self._performance_optimized_scheduling,
            SchedulingStrategy.LOAD_BALANCED: self._load_balanced_scheduling,
            SchedulingStrategy.FAULT_TOLERANT: self._fault_tolerant_scheduling,
            SchedulingStrategy.ENERGY_EFFICIENT: self._energy_efficient_scheduling,
            SchedulingStrategy.QUALITY_FOCUSED: self._quality_focused_scheduling
        }
    
    def _initialize_performance_monitoring(self):
        """初始化性能监控"""
        # 启动性能监控线程
        self.running = True
        self.scheduling_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.scheduling_thread.start()
    
    def submit_task(self, task_name: str, task_description: str,
                   required_capabilities: List[ModelCapability],
                   priority: int = 1, deadline: Optional[float] = None) -> str:
        """提交任务到调度器"""
        task_id = str(uuid.uuid4())
        
        with self.lock:
            task_info = TaskInfo(
                task_id=task_id,
                task_name=task_name,
                task_description=task_description,
                required_capabilities=required_capabilities,
                priority=priority,
                deadline=deadline
            )
            
            self.tasks[task_id] = task_info
            self.task_queue.append(task_id)
            self.metrics.total_tasks += 1
            
            logger.info(f"任务提交成功: {task_id} - {task_name}, 优先级: {priority}")
        
        # 异步触发调度
        self._trigger_scheduling()
        
        return task_id
    
    async def submit_task_async(self, task_name: str, task_description: str,
                              required_capabilities: List[ModelCapability],
                              priority: int = 1, deadline: Optional[float] = None) -> str:
        """异步提交任务"""
        task_id = self.submit_task(task_name, task_description, required_capabilities, priority, deadline)
        return task_id
    
    def _trigger_scheduling(self):
        """触发调度"""
        # 在后台线程中执行调度
        threading.Thread(target=self._schedule_pending_tasks, daemon=True).start()
    
    def _schedule_pending_tasks(self):
        """调度待处理任务"""
        with self.lock:
            if not self.task_queue:
                return
            
            # 按优先级排序任务
            sorted_tasks = self._sort_tasks_by_priority()
            
            for task_id in sorted_tasks:
                if task_id not in self.tasks:
                    continue
                
                task_info = self.tasks[task_id]
                if task_info.status != TaskStatus.PENDING:
                    continue
                
                # 检查截止时间
                if task_info.deadline and time.time() > task_info.deadline:
                    task_info.status = TaskStatus.FAILED
                    task_info.error = "任务超时"
                    self._record_task_completion(task_info)
                    continue
                
                try:
                    # 执行任务分配
                    self._assign_task(task_info)
                    
                    # 执行任务
                    self._execute_task(task_info)
                    
                except Exception as e:
                    logger.error(f"任务调度失败: {task_id} - {str(e)}")
                    task_info.status = TaskStatus.FAILED
                    task_info.error = str(e)
                    self._record_task_completion(task_info)
    
    def _sort_tasks_by_priority(self) -> List[str]:
        """按优先级排序任务"""
        # 获取待处理任务
        pending_tasks = []
        for task_id in self.task_queue:
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.PENDING:
                pending_tasks.append((task_id, self.tasks[task_id]))
        
        # 按优先级降序排序
        pending_tasks.sort(key=lambda x: (x[1].priority, -x[1].created_at), reverse=True)
        
        return [task_id for task_id, _ in pending_tasks]
    
    def _assign_task(self, task_info: TaskInfo):
        """分配任务给模型"""
        assignment_start_time = time.time()
        
        # 使用智能任务分配器分配任务
        required_models = self._get_models_for_capabilities(task_info.required_capabilities)
        
        decision = self.task_distributor.allocate_task(
            task_id=task_info.task_id,
            task_name=task_info.task_name,
            task_description=task_info.task_description,
            required_models=required_models,
            strategy='balanced'
        )
        
        assignment_time = time.time() - assignment_start_time
        
        if not decision.allocated_models:
            raise ValueError(f"无法为任务 {task_info.task_id} 分配模型")
        
        task_info.assigned_decision = decision.__dict__
        task_info.assigned_models = decision.allocated_models
        task_info.status = TaskStatus.ASSIGNED
        
        logger.info(f"任务分配完成: {task_info.task_id}, 分配模型: {decision.allocated_models}")
        
        # 更新分配时间指标
        self.metrics.avg_assignment_time = (
            self.metrics.avg_assignment_time * (self.metrics.total_tasks - 1) + assignment_time
        ) / self.metrics.total_tasks
    
    def _get_models_for_capabilities(self, capabilities: List[ModelCapability]) -> List[str]:
        """获取具备特定能力的模型列表"""
        models = []
        for capability in capabilities:
            capable_models = self.capability_registry.capability_model_map.get(capability, [])
            models.extend(capable_models)
        
        return list(set(models))
    
    def _execute_task(self, task_info: TaskInfo):
        """执行任务"""
        task_info.started_at = time.time()
        task_info.status = TaskStatus.EXECUTING
        
        # 根据分配的模型数量决定协作模式
        if len(task_info.assigned_models) == 1:
            # 单个模型，直接执行
            self._execute_single_model_task(task_info)
        else:
            # 多个模型，使用协作模式
            self._execute_collaborative_task(task_info)
    
    def _execute_single_model_task(self, task_info: TaskInfo):
        """执行单模型任务"""
        model_id = task_info.assigned_models[0]
        
        # 在实际应用中，这里应该调用模型执行任务
        # 这里使用模拟实现
        def execute_task():
            try:
                # 模拟执行时间
                execution_time = 0.5 + (task_info.priority * 0.1)  # 优先级越高，执行时间略长
                time.sleep(execution_time)
                
                result = {
                    "success": True,
                    "model_id": model_id,
                    "execution_time": execution_time,
                    "result": f"任务 {task_info.task_id} 执行完成",
                    "confidence": 0.9
                }
                
                task_info.result = result
                task_info.status = TaskStatus.COMPLETED
                task_info.execution_time = execution_time
                task_info.completed_at = time.time()
                
                # 记录任务完成
                self._record_task_completion(task_info)
                
                # 生成性能反馈
                self._generate_feedback(task_info, FeedbackType.PERFORMANCE, {
                    "execution_time": execution_time,
                    "success": True,
                    "model_id": model_id,
                    "confidence": 0.9
                })
                
            except Exception as e:
                logger.error(f"单模型任务执行失败: {task_info.task_id} - {str(e)}")
                task_info.status = TaskStatus.FAILED
                task_info.error = str(e)
                self._record_task_completion(task_info)
                
                # 生成错误反馈
                self._generate_feedback(task_info, FeedbackType.ERROR, {
                    "error": str(e),
                    "model_id": model_id
                })
        
        # 在后台线程中执行任务
        threading.Thread(target=execute_task, daemon=True).start()
    
    def _execute_collaborative_task(self, task_info: TaskInfo):
        """执行协作任务"""
        # 使用模型协作协调器执行协作任务
        async def execute_collaboration():
            try:
                # 确定协作模式
                if len(task_info.assigned_models) == 2:
                    pattern_name = "vision_language"  # 示例模式
                else:
                    pattern_name = "multimodal_fusion"  # 示例模式
                
                # 准备输入数据
                input_data = {
                    "task_id": task_info.task_id,
                    "task_description": task_info.task_description,
                    "assigned_models": task_info.assigned_models
                }
                
                # 执行协作
                collaboration_result = await self.collaboration_orchestrator.initiate_collaboration(
                    pattern_name=pattern_name,
                    input_data=input_data,
                    custom_config={
                        "timeout": 30.0,
                        "max_retries": 3,
                        "models": task_info.assigned_models
                    }
                )
                
                task_info.collaboration_id = collaboration_result.get("collaboration_id")
                task_info.result = collaboration_result
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = time.time()
                task_info.execution_time = task_info.completed_at - task_info.started_at
                
                # 记录任务完成
                self._record_task_completion(task_info)
                
                # 生成协作反馈
                self._generate_feedback(task_info, FeedbackType.COLLABORATION, {
                    "collaboration_id": task_info.collaboration_id,
                    "models": task_info.assigned_models,
                    "execution_time": task_info.execution_time,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"协作任务执行失败: {task_info.task_id} - {str(e)}")
                task_info.status = TaskStatus.FAILED
                task_info.error = str(e)
                self._record_task_completion(task_info)
                
                # 生成错误反馈
                self._generate_feedback(task_info, FeedbackType.ERROR, {
                    "error": str(e),
                    "collaboration_id": task_info.collaboration_id
                })
        
        # 创建异步任务
        asyncio.create_task(execute_collaboration())
    
    def _record_task_completion(self, task_info: TaskInfo):
        """记录任务完成"""
        with self.lock:
            # 更新指标
            if task_info.status == TaskStatus.COMPLETED:
                self.metrics.completed_tasks += 1
                
                if task_info.execution_time:
                    # 更新平均执行时间
                    total_execution_time = self.metrics.avg_execution_time * (self.metrics.completed_tasks - 1)
                    self.metrics.avg_execution_time = (total_execution_time + task_info.execution_time) / self.metrics.completed_tasks
                
            elif task_info.status == TaskStatus.FAILED:
                self.metrics.failed_tasks += 1
            
            # 更新成功率
            total_processed = self.metrics.completed_tasks + self.metrics.failed_tasks
            if total_processed > 0:
                self.metrics.success_rate = self.metrics.completed_tasks / total_processed * 100
            
            self.metrics.last_update_time = time.time()
            
            # 将任务移动到完成队列
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                self.completed_tasks.append(task_info)
                if task_info.task_id in self.tasks:
                    del self.tasks[task_info.task_id]
                
                # 从任务队列中移除
                if task_info.task_id in self.task_queue:
                    self.task_queue.remove(task_info.task_id)
    
    def _generate_feedback(self, task_info: TaskInfo, feedback_type: FeedbackType, data: Dict[str, Any]):
        """生成反馈"""
        feedback_id = str(uuid.uuid4())
        feedback = FeedbackData(
            feedback_id=feedback_id,
            task_id=task_info.task_id,
            feedback_type=feedback_type,
            data=data
        )
        
        self.feedback_history.append(feedback)
        
        # 处理反馈
        self._process_feedback(feedback)
        
        logger.info(f"反馈生成: {feedback_id}, 类型: {feedback_type.value}, 任务: {task_info.task_id}")
    
    def _process_feedback(self, feedback: FeedbackData):
        """处理反馈"""
        handlers = self.feedback_handlers.get(feedback.feedback_type, [])
        for handler in handlers:
            try:
                handler(feedback)
            except Exception as e:
                logger.error(f"反馈处理失败: {feedback.feedback_id} - {str(e)}")
        
        feedback.processed = True
    
    def _handle_performance_feedback(self, feedback: FeedbackData):
        """处理性能反馈"""
        data = feedback.data
        model_id = data.get("model_id")
        execution_time = data.get("execution_time")
        
        if model_id and execution_time:
            # 更新模型性能统计
            logger.info(f"性能反馈: 模型 {model_id} 执行时间 {execution_time:.2f}s")
    
    def _handle_quality_feedback(self, feedback: FeedbackData):
        """处理质量反馈"""
        data = feedback.data
        quality_score = data.get("quality_score", 0.0)
        
        logger.info(f"质量反馈: 质量分数 {quality_score:.2f}")
    
    def _handle_resource_feedback(self, feedback: FeedbackData):
        """处理资源反馈"""
        data = feedback.data
        resource_usage = data.get("resource_usage", {})
        
        for resource, usage in resource_usage.items():
            self.metrics.resource_utilization[resource] = usage
        
        logger.info(f"资源反馈: {resource_usage}")
    
    def _handle_collaboration_feedback(self, feedback: FeedbackData):
        """处理协作反馈"""
        data = feedback.data
        collaboration_id = data.get("collaboration_id")
        
        logger.info(f"协作反馈: 协作 {collaboration_id}")
    
    def _handle_error_feedback(self, feedback: FeedbackData):
        """处理错误反馈"""
        data = feedback.data
        error = data.get("error")
        
        logger.warning(f"错误反馈: {error}")
    
    def _performance_optimized_scheduling(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """性能优化调度策略"""
        # 根据历史性能数据选择最佳模型
        return sorted(tasks, key=lambda x: x.priority, reverse=True)
    
    def _load_balanced_scheduling(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """负载均衡调度策略"""
        # 考虑模型当前负载进行调度
        return sorted(tasks, key=lambda x: (x.priority, x.created_at), reverse=True)
    
    def _fault_tolerant_scheduling(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """容错优先调度策略"""
        # 优先选择可靠性高的模型
        return sorted(tasks, key=lambda x: (x.priority, -x.created_at), reverse=True)
    
    def _energy_efficient_scheduling(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """能效优先调度策略"""
        # 优先选择能效高的模型
        return sorted(tasks, key=lambda x: (x.priority, x.created_at), reverse=True)
    
    def _quality_focused_scheduling(self, tasks: List[TaskInfo]) -> List[TaskInfo]:
        """质量优先调度策略"""
        # 优先选择输出质量高的模型
        return sorted(tasks, key=lambda x: (x.priority, -x.created_at), reverse=True)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                with self.lock:
                    # 更新模型利用率统计
                    total_models = len(self.capability_registry.capability_profiles)
                    if total_models > 0:
                        for model_id, profile in self.capability_registry.capability_profiles.items():
                            utilization = profile.current_task_count / max(1, profile.max_concurrent_tasks)
                            self.metrics.model_utilization[model_id] = utilization
                    
                    # 记录性能历史
                    self.performance_history.append({
                        "timestamp": time.time(),
                        "active_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.EXECUTING]),
                        "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                        "success_rate": self.metrics.success_rate,
                        "avg_execution_time": self.metrics.avg_execution_time
                    })
                
                # 每秒更新一次
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(5)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                return {
                    "task_id": task_info.task_id,
                    "status": task_info.status.value,
                    "assigned_models": task_info.assigned_models,
                    "started_at": task_info.started_at,
                    "execution_time": task_info.execution_time,
                    "error": task_info.error
                }
            
            # 检查已完成的任务
            for completed_task in self.completed_tasks:
                if completed_task.task_id == task_id:
                    return {
                        "task_id": completed_task.task_id,
                        "status": completed_task.status.value,
                        "assigned_models": completed_task.assigned_models,
                        "started_at": completed_task.started_at,
                        "completed_at": completed_task.completed_at,
                        "execution_time": completed_task.execution_time,
                        "error": completed_task.error,
                        "result": completed_task.result
                    }
            
            return None
    
    def get_scheduler_metrics(self) -> Dict[str, Any]:
        """获取调度器指标"""
        with self.lock:
            return {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "success_rate": self.metrics.success_rate,
                "avg_execution_time": self.metrics.avg_execution_time,
                "avg_assignment_time": self.metrics.avg_assignment_time,
                "active_tasks": len([t for t in self.tasks.values() if t.status in [TaskStatus.ASSIGNED, TaskStatus.EXECUTING]]),
                "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "resource_utilization": self.metrics.resource_utilization,
                "model_utilization": self.metrics.model_utilization,
                "last_update_time": self.metrics.last_update_time
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task_info = self.tasks[task_id]
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            task_info.status = TaskStatus.CANCELLED
            task_info.error = "用户取消"
            self._record_task_completion(task_info)
            
            logger.info(f"任务取消: {task_id}")
            return True
    
    def shutdown(self):
        """关闭调度器"""
        self.running = False
        if self.scheduling_thread:
            self.scheduling_thread.join(timeout=5)
        
        logger.info("多模型协同调度器已关闭")


# 全局调度器实例
_scheduler_instance = None

def get_scheduler() -> MultiModelScheduler:
    """获取调度器单例"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = MultiModelScheduler()
    return _scheduler_instance

def initialize_scheduler(strategy: SchedulingStrategy = SchedulingStrategy.PERFORMANCE_OPTIMIZED):
    """初始化调度器"""
    global _scheduler_instance
    if _scheduler_instance is not None:
        _scheduler_instance.shutdown()
    
    _scheduler_instance = MultiModelScheduler(strategy)
    return _scheduler_instance

def shutdown_scheduler():
    """关闭调度器"""
    global _scheduler_instance
    if _scheduler_instance is not None:
        _scheduler_instance.shutdown()
        _scheduler_instance = None
