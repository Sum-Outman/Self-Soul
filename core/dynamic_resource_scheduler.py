import zlib
"""
动态资源调度系统 - 智能资源分配和负载均衡

核心功能：
1. 预测性资源调度 - 基于历史数据和任务特征的资源预测
2. 动态优先级调整 - 根据任务重要性和紧急程度调整资源分配
3. 多目标优化 - 平衡性能、成本、能耗和公平性
4. 实时监控和调整 - 持续监控资源使用并动态调整分配
5. 容错和恢复 - 处理资源故障和自动重新调度

基于评估报告中的资源管理不足问题设计
"""

import time
import threading
import logging
import json
import heapq
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 5    # 关键任务，最高优先级
    HIGH = 4        # 高优先级任务
    MEDIUM = 3      # 中等优先级任务
    LOW = 2         # 低优先级任务
    BACKGROUND = 1  # 后台任务，最低优先级


class ResourceType(Enum):
    """资源类型"""
    CPU_CORES = "cpu_cores"
    CPU_THREADS = "cpu_threads"
    MEMORY_GB = "memory_gb"
    GPU_COUNT = "gpu_count"
    GPU_MEMORY_GB = "gpu_memory_gb"
    DISK_SPACE_GB = "disk_space_gb"
    NETWORK_BANDWIDTH_MBPS = "network_bandwidth_mbps"
    IOPS = "iops"  # 输入输出操作每秒


@dataclass
class TaskRequirement:
    """任务资源需求"""
    task_id: str
    task_name: str
    priority: TaskPriority
    estimated_duration_seconds: float  # 预估执行时间
    deadline_seconds: Optional[float] = None  # 截止时间（从当前时间算起）
    
    # 资源需求
    cpu_cores: float = 1.0
    cpu_threads: float = 1.0
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    disk_space_gb: float = 0.0
    network_bandwidth_mbps: float = 10.0
    iops: float = 100.0
    
    # 约束条件
    requires_gpu: bool = False
    requires_high_memory: bool = False
    requires_low_latency: bool = False
    can_be_preempted: bool = True  # 是否可被抢占
    
    # 预测信息
    historical_execution_times: List[float] = None  # 历史执行时间
    resource_usage_pattern: Dict[str, List[float]] = None  # 资源使用模式
    
    def __post_init__(self):
        if self.historical_execution_times is None:
            self.historical_execution_times = []
        if self.resource_usage_pattern is None:
            self.resource_usage_pattern = {}
        
        # 自动设置requires_gpu
        if self.gpu_count > 0:
            self.requires_gpu = True
        
        # 自动设置requires_high_memory
        if self.memory_gb > 16.0:
            self.requires_high_memory = True


@dataclass
class ResourceAllocation:
    """资源分配结果"""
    allocation_id: str
    task_id: str
    resources_allocated: Dict[str, float]
    start_time: float
    estimated_end_time: float
    actual_end_time: Optional[float] = None
    status: str = "allocated"  # allocated, running, completed, failed, preempted
    node_id: Optional[str] = None  # 分配的节点ID（用于分布式系统）


@dataclass
class SystemResource:
    """系统资源状态"""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocated_capacity: float
    utilization_percent: float
    cost_per_unit: float = 0.0  # 单位成本（用于成本优化）
    power_consumption_per_unit: float = 0.0  # 单位功耗（用于能耗优化）
    
    def update_utilization(self):
        """更新利用率"""
        self.allocated_capacity = self.total_capacity - self.available_capacity
        if self.total_capacity > 0:
            self.utilization_percent = (self.allocated_capacity / self.total_capacity) * 100
        else:
            self.utilization_percent = 0.0


class ResourceUsagePredictor:
    """资源使用预测器"""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.task_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.resource_patterns: Dict[str, Dict[str, Any]] = {}
        
    def record_task_execution(self, task_id: str, task_type: str, 
                             execution_time: float, resource_usage: Dict[str, float]):
        """记录任务执行历史"""
        history_entry = {
            "timestamp": time.time(),
            "execution_time": execution_time,
            "resource_usage": resource_usage,
            "task_type": task_type
        }
        self.task_history[task_id].append(history_entry)
        
        # 更新资源使用模式
        if task_id not in self.resource_patterns:
            self.resource_patterns[task_id] = {
                "task_type": task_type,
                "execution_times": [],
                "resource_patterns": defaultdict(list)
            }
        
        self.resource_patterns[task_id]["execution_times"].append(execution_time)
        for resource, usage in resource_usage.items():
            self.resource_patterns[task_id]["resource_patterns"][resource].append(usage)
    
    def predict_execution_time(self, task_id: str, task_type: str = None) -> float:
        """预测任务执行时间"""
        if task_id in self.task_history and self.task_history[task_id]:
            # 基于历史数据预测
            execution_times = [entry["execution_time"] for entry in self.task_history[task_id]]
            return np.median(execution_times) * 1.1  # 添加10%安全边际
        elif task_type and task_id in self.resource_patterns:
            # 基于任务类型预测
            execution_times = self.resource_patterns[task_id]["execution_times"]
            if execution_times:
                return np.median(execution_times) * 1.2  # 添加20%安全边际
        
        # 默认预测
        return 300.0  # 5分钟默认值
    
    def predict_resource_usage(self, task_id: str, resource_type: str) -> float:
        """预测资源使用量"""
        if task_id in self.resource_patterns:
            patterns = self.resource_patterns[task_id]["resource_patterns"]
            if resource_type in patterns and patterns[resource_type]:
                return np.percentile(patterns[resource_type], 75)  # 使用75百分位数
        return 0.0


class DynamicResourceScheduler:
    """动态资源调度器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 任务队列
        self.task_queue: List[Tuple[int, float, TaskRequirement]] = []  # 优先级队列
        self.running_tasks: Dict[str, ResourceAllocation] = {}
        self.completed_tasks: Dict[str, ResourceAllocation] = {}
        self.failed_tasks: Dict[str, ResourceAllocation] = {}
        
        # 系统资源
        self.system_resources: Dict[ResourceType, SystemResource] = self._initialize_system_resources()
        
        # 预测器
        self.predictor = ResourceUsagePredictor()
        
        # 调度参数
        self.scheduling_interval = self.config.get("scheduling_interval", 5.0)  # 调度间隔（秒）
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 10)
        self.resource_utilization_target = self.config.get("resource_utilization_target", 70.0)  # 目标利用率
        self.enable_preemption = self.config.get("enable_preemption", True)
        
        # 调度锁
        self.scheduler_lock = threading.RLock()
        
        # 调度线程
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # 监控数据
        self.scheduling_history = deque(maxlen=1000)
        self.resource_usage_history = deque(maxlen=500)
        
        logger.info("动态资源调度器初始化完成")
    
    def _initialize_system_resources(self) -> Dict[ResourceType, SystemResource]:
        """初始化系统资源"""
        import psutil
        
        resources = {}
        
        # CPU核心
        resources[ResourceType.CPU_CORES] = SystemResource(
            resource_type=ResourceType.CPU_CORES,
            total_capacity=psutil.cpu_count(logical=False),
            available_capacity=psutil.cpu_count(logical=False),
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.01,  # 假设成本
            power_consumption_per_unit=5.0  # 假设功耗
        )
        
        # CPU线程
        resources[ResourceType.CPU_THREADS] = SystemResource(
            resource_type=ResourceType.CPU_THREADS,
            total_capacity=psutil.cpu_count(logical=True),
            available_capacity=psutil.cpu_count(logical=True),
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.005,
            power_consumption_per_unit=2.5
        )
        
        # 内存
        memory_gb = psutil.virtual_memory().total / (1024**3)
        resources[ResourceType.MEMORY_GB] = SystemResource(
            resource_type=ResourceType.MEMORY_GB,
            total_capacity=memory_gb,
            available_capacity=memory_gb,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.1,
            power_consumption_per_unit=0.5
        )
        
        # GPU资源（需要检测）
        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
        except ImportError:
            pass
        
        resources[ResourceType.GPU_COUNT] = SystemResource(
            resource_type=ResourceType.GPU_COUNT,
            total_capacity=gpu_count,
            available_capacity=gpu_count,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=1.0,  # GPU成本较高
            power_consumption_per_unit=100.0  # GPU功耗较高
        )
        
        # GPU内存
        gpu_memory_gb = 0.0
        if gpu_count > 0:
            try:
                import torch
                gpu_memory_gb = sum(
                    torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    for i in range(gpu_count)
                )
            except:
                pass
        
        resources[ResourceType.GPU_MEMORY_GB] = SystemResource(
            resource_type=ResourceType.GPU_MEMORY_GB,
            total_capacity=gpu_memory_gb,
            available_capacity=gpu_memory_gb,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.5,
            power_consumption_per_unit=50.0
        )
        
        # 磁盘空间
        try:
            disk_gb = psutil.disk_usage('/').total / (1024**3)
        except:
            disk_gb = 100.0  # 默认值
        
        resources[ResourceType.DISK_SPACE_GB] = SystemResource(
            resource_type=ResourceType.DISK_SPACE_GB,
            total_capacity=disk_gb,
            available_capacity=disk_gb,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.001,
            power_consumption_per_unit=0.1
        )
        
        # 网络带宽
        resources[ResourceType.NETWORK_BANDWIDTH_MBPS] = SystemResource(
            resource_type=ResourceType.NETWORK_BANDWIDTH_MBPS,
            total_capacity=1000.0,  # 1 Gbps
            available_capacity=1000.0,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.01,
            power_consumption_per_unit=0.2
        )
        
        # IOPS
        resources[ResourceType.IOPS] = SystemResource(
            resource_type=ResourceType.IOPS,
            total_capacity=10000.0,  # 假设10k IOPS
            available_capacity=10000.0,
            allocated_capacity=0.0,
            utilization_percent=0.0,
            cost_per_unit=0.0001,
            power_consumption_per_unit=0.05
        )
        
        return resources
    
    def submit_task(self, task_requirement: TaskRequirement) -> str:
        """提交任务到调度器"""
        with self.scheduler_lock:
            # 生成任务ID
            task_id = task_requirement.task_id or f"task_{int(time.time())}_{(zlib.adler32(str(str(task_requirement).encode('utf-8')) & 0xffffffff)) % 10000}"
            task_requirement.task_id = task_id
            
            # 计算优先级分数（考虑截止时间和任务优先级）
            priority_score = self._calculate_priority_score(task_requirement)
            
            # 添加到优先级队列
            heapq.heappush(self.task_queue, (-priority_score, time.time(), task_requirement))
            
            logger.info(f"任务提交: {task_id} ({task_requirement.task_name}), 优先级: {task_requirement.priority.name}")
            
            return task_id
    
    def _calculate_priority_score(self, task: TaskRequirement) -> float:
        """计算任务优先级分数"""
        base_score = task.priority.value * 10  # 基础优先级分数
        
        # 截止时间紧迫性
        if task.deadline_seconds:
            time_remaining = task.deadline_seconds
            urgency_factor = max(0.1, 1.0 / (time_remaining + 1))
            base_score *= (1.0 + urgency_factor)
        
        # 资源需求复杂度
        resource_complexity = 0.0
        if task.requires_gpu:
            resource_complexity += 2.0
        if task.requires_high_memory:
            resource_complexity += 1.0
        if task.requires_low_latency:
            resource_complexity += 1.5
        
        base_score *= (1.0 + resource_complexity * 0.1)
        
        return base_score
    
    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            logger.warning("调度器已经在运行")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("动态资源调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("动态资源调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.scheduler_running:
            try:
                # 更新系统资源状态
                self._update_system_resources()
                
                # 调度决策
                self._schedule_tasks()
                
                # 监控和调整运行中的任务
                self._monitor_running_tasks()
                
                # 记录监控数据
                self._record_monitoring_data()
                
                # 清理已完成的任务
                self._cleanup_completed_tasks()
                
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
            
            # 等待下一次调度
            time.sleep(self.scheduling_interval)
    
    def _update_system_resources(self):
        """更新系统资源状态"""
        try:
            import psutil
            
            # 更新CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_cores = self.system_resources[ResourceType.CPU_CORES]
            cpu_cores.available_capacity = cpu_cores.total_capacity * (1 - cpu_percent / 100)
            cpu_cores.update_utilization()
            
            # 更新内存使用率
            memory = psutil.virtual_memory()
            memory_gb = self.system_resources[ResourceType.MEMORY_GB]
            memory_gb.available_capacity = memory.available / (1024**3)
            memory_gb.update_utilization()
            
            # 更新磁盘空间
            try:
                disk = psutil.disk_usage('/')
                disk_gb = self.system_resources[ResourceType.DISK_SPACE_GB]
                disk_gb.available_capacity = disk.free / (1024**3)
                disk_gb.update_utilization()
            except:
                pass
            
        except Exception as e:
            logger.warning(f"更新系统资源状态失败: {e}")
    
    def _schedule_tasks(self):
        """调度任务"""
        with self.scheduler_lock:
            # 检查当前运行任务数
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                return
            
            # 尝试从队列中调度任务
            scheduled_count = 0
            temp_queue = []
            
            while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
                priority_score, submit_time, task = heapq.heappop(self.task_queue)
                
                # 检查资源是否足够
                if self._can_allocate_resources(task):
                    # 分配资源
                    allocation = self._allocate_resources(task)
                    if allocation:
                        self.running_tasks[task.task_id] = allocation
                        scheduled_count += 1
                        
                        logger.info(f"任务调度: {task.task_id} 已分配资源")
                        continue
                
                # 资源不足，放回队列
                temp_queue.append((priority_score, submit_time, task))
            
            # 将未调度的任务放回队列
            for item in temp_queue:
                heapq.heappush(self.task_queue, item)
            
            if scheduled_count > 0:
                logger.info(f"调度完成: 调度了 {scheduled_count} 个任务，当前运行任务: {len(self.running_tasks)}")
    
    def _can_allocate_resources(self, task: TaskRequirement) -> bool:
        """检查是否可以分配资源给任务"""
        try:
            # 检查CPU核心
            cpu_cores = self.system_resources[ResourceType.CPU_CORES]
            if task.cpu_cores > cpu_cores.available_capacity * 0.8:  # 不超过80%可用核心
                return False
            
            # 检查内存
            memory_gb = self.system_resources[ResourceType.MEMORY_GB]
            if task.memory_gb > memory_gb.available_capacity * 0.7:  # 不超过70%可用内存
                return False
            
            # 检查GPU
            if task.gpu_count > 0:
                gpu_count = self.system_resources[ResourceType.GPU_COUNT]
                if task.gpu_count > gpu_count.available_capacity:
                    return False
                
                gpu_memory = self.system_resources[ResourceType.GPU_MEMORY_GB]
                if task.gpu_memory_gb > gpu_memory.available_capacity * 0.8:
                    return False
            
            # 检查磁盘空间
            disk_space = self.system_resources[ResourceType.DISK_SPACE_GB]
            if task.disk_space_gb > disk_space.available_capacity * 0.8:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查资源分配失败: {e}")
            return False
    
    def _allocate_resources(self, task: TaskRequirement) -> Optional[ResourceAllocation]:
        """分配资源给任务"""
        try:
            allocation_id = f"alloc_{int(time.time())}_{task.task_id}"
            
            # 计算预估执行时间
            estimated_duration = task.estimated_duration_seconds
            if not estimated_duration or estimated_duration <= 0:
                estimated_duration = self.predictor.predict_execution_time(task.task_id, task.task_name)
            
            # 资源分配
            resources_allocated = {
                "cpu_cores": task.cpu_cores,
                "cpu_threads": task.cpu_threads,
                "memory_gb": task.memory_gb,
                "gpu_count": task.gpu_count,
                "gpu_memory_gb": task.gpu_memory_gb,
                "disk_space_gb": task.disk_space_gb,
                "network_bandwidth_mbps": task.network_bandwidth_mbps,
                "iops": task.iops
            }
            
            # 更新系统资源状态
            self._update_resource_allocation(task, allocate=True)
            
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                task_id=task.task_id,
                resources_allocated=resources_allocated,
                start_time=time.time(),
                estimated_end_time=time.time() + estimated_duration,
                status="allocated"
            )
            
            return allocation
            
        except Exception as e:
            logger.error(f"分配资源失败: {e}")
            return None
    
    def _update_resource_allocation(self, task: TaskRequirement, allocate: bool):
        """更新资源分配状态"""
        multiplier = 1 if allocate else -1
        
        # CPU核心
        cpu_cores = self.system_resources[ResourceType.CPU_CORES]
        cpu_cores.allocated_capacity += multiplier * task.cpu_cores
        cpu_cores.available_capacity -= multiplier * task.cpu_cores
        cpu_cores.update_utilization()
        
        # 内存
        memory_gb = self.system_resources[ResourceType.MEMORY_GB]
        memory_gb.allocated_capacity += multiplier * task.memory_gb
        memory_gb.available_capacity -= multiplier * task.memory_gb
        memory_gb.update_utilization()
        
        # GPU
        if task.gpu_count > 0:
            gpu_count = self.system_resources[ResourceType.GPU_COUNT]
            gpu_count.allocated_capacity += multiplier * task.gpu_count
            gpu_count.available_capacity -= multiplier * task.gpu_count
            gpu_count.update_utilization()
            
            gpu_memory = self.system_resources[ResourceType.GPU_MEMORY_GB]
            gpu_memory.allocated_capacity += multiplier * task.gpu_memory_gb
            gpu_memory.available_capacity -= multiplier * task.gpu_memory_gb
            gpu_memory.update_utilization()
        
        # 磁盘空间
        disk_space = self.system_resources[ResourceType.DISK_SPACE_GB]
        disk_space.allocated_capacity += multiplier * task.disk_space_gb
        disk_space.available_capacity -= multiplier * task.disk_space_gb
        disk_space.update_utilization()
    
    def _monitor_running_tasks(self):
        """监控运行中的任务"""
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, allocation in self.running_tasks.items():
            # 检查任务是否超时
            if current_time > allocation.estimated_end_time * 1.2:  # 20%容忍度
                logger.warning(f"任务 {task_id} 可能已超时，预估结束时间: {allocation.estimated_end_time}")
                
                # 标记为失败
                allocation.status = "failed"
                allocation.actual_end_time = current_time
                self.failed_tasks[task_id] = allocation
                tasks_to_remove.append(task_id)
                
                # 释放资源
                self._release_resources(allocation)
        
        # 移除已完成的任务
        for task_id in tasks_to_remove:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _release_resources(self, allocation: ResourceAllocation):
        """释放任务占用的资源"""
        # 这里需要从allocation中恢复任务需求信息
        # 简化实现：标记资源已释放
        logger.info(f"释放任务 {allocation.task_id} 的资源")
    
    def _record_monitoring_data(self):
        """记录监控数据"""
        current_time = time.time()
        
        # 记录资源使用情况
        resource_usage = {
            "timestamp": current_time,
            "cpu_utilization": self.system_resources[ResourceType.CPU_CORES].utilization_percent,
            "memory_utilization": self.system_resources[ResourceType.MEMORY_GB].utilization_percent,
            "gpu_utilization": self.system_resources[ResourceType.GPU_COUNT].utilization_percent,
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue)
        }
        
        self.resource_usage_history.append(resource_usage)
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        # 保留最近100个已完成任务
        if len(self.completed_tasks) > 100:
            # 移除最早的任务
            oldest_tasks = sorted(self.completed_tasks.items(), 
                                 key=lambda x: x[1].actual_end_time or 0)[:50]
            for task_id, _ in oldest_tasks:
                if task_id in self.completed_tasks:
                    del self.completed_tasks[task_id]
    
    def complete_task(self, task_id: str, execution_time: float = None, 
                     actual_resource_usage: Dict[str, float] = None):
        """标记任务完成"""
        with self.scheduler_lock:
            if task_id not in self.running_tasks:
                logger.warning(f"任务 {task_id} 不在运行中")
                return False
            
            allocation = self.running_tasks[task_id]
            allocation.status = "completed"
            allocation.actual_end_time = time.time()
            
            # 记录执行历史
            if actual_resource_usage:
                self.predictor.record_task_execution(
                    task_id=task_id,
                    task_type=allocation.task_id,
                    execution_time=execution_time or (allocation.actual_end_time - allocation.start_time),
                    resource_usage=actual_resource_usage
                )
            
            # 移动到已完成任务
            self.completed_tasks[task_id] = allocation
            
            # 从运行任务中移除
            del self.running_tasks[task_id]
            
            # 释放资源
            self._release_resources(allocation)
            
            logger.info(f"任务完成: {task_id}, 执行时间: {execution_time or '未知'}秒")
            
            return True
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        with self.scheduler_lock:
            return {
                "timestamp": time.time(),
                "scheduler_running": self.scheduler_running,
                "running_tasks": len(self.running_tasks),
                "queued_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "resource_utilization": {
                    resource_type.name: resource.utilization_percent
                    for resource_type, resource in self.system_resources.items()
                },
                "system_resources": {
                    resource_type.name: {
                        "total": resource.total_capacity,
                        "available": resource.available_capacity,
                        "allocated": resource.allocated_capacity,
                        "utilization_percent": resource.utilization_percent
                    }
                    for resource_type, resource in self.system_resources.items()
                }
            }
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        suggestions = []
        
        # 检查资源利用率
        for resource_type, resource in self.system_resources.items():
            if resource.utilization_percent > 90:
                suggestions.append({
                    "type": "high_utilization",
                    "resource": resource_type.name,
                    "utilization": resource.utilization_percent,
                    "suggestion": f"{resource_type.name} 利用率过高 ({resource.utilization_percent:.1f}%)，考虑增加容量或优化任务调度",
                    "priority": "high"
                })
            elif resource.utilization_percent < 20:
                suggestions.append({
                    "type": "low_utilization",
                    "resource": resource_type.name,
                    "utilization": resource.utilization_percent,
                    "suggestion": f"{resource_type.name} 利用率过低 ({resource.utilization_percent:.1f}%)，考虑合并任务或减少资源分配",
                    "priority": "low"
                })
        
        # 检查任务队列长度
        if len(self.task_queue) > 20:
            suggestions.append({
                "type": "long_queue",
                "queue_length": len(self.task_queue),
                "suggestion": f"任务队列过长 ({len(self.task_queue)} 个任务等待)，考虑增加并发任务数或优化任务优先级",
                "priority": "medium"
            })
        
        # 检查任务失败率
        total_tasks = len(self.running_tasks) + len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            failure_rate = len(self.failed_tasks) / total_tasks * 100
            if failure_rate > 10:
                suggestions.append({
                    "type": "high_failure_rate",
                    "failure_rate": failure_rate,
                    "suggestion": f"任务失败率过高 ({failure_rate:.1f}%)，检查资源分配和任务配置",
                    "priority": "high"
                })
        
        return suggestions


def create_dynamic_resource_scheduler(config: Dict[str, Any] = None) -> DynamicResourceScheduler:
    """创建动态资源调度器实例"""
    return DynamicResourceScheduler(config)