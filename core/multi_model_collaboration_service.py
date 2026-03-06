"""
多模型协同服务 - Multi-Model Collaboration Service

在8016端口提供完整的多模型协同服务，包括：
1. 任务分发和调度
2. 模型间通信协议（RPC/消息队列）
3. 结果融合和冲突解决
4. 优先级调度机制
5. 协同性能监控

端口：8016（根据README要求）
"""

import asyncio
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import threading
import concurrent.futures

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import uvicorn
import httpx

from core.error_handling import error_handler
from core.model_registry import get_model_registry
from core.enhanced_model_collaboration import EnhancedModelCollaboration
from core.collaboration.model_collaborator import ModelCollaborationOrchestrator, CollaborationMode
from core.system_monitor import SystemMonitor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 服务配置
COLLABORATION_SERVICE_PORT = 8016
SERVICE_HOST = "127.0.0.1"
MAX_WORKERS = 10
TASK_TIMEOUT = 30.0

class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = "critical"    # 关键任务：立即执行，最高优先级
    HIGH = "high"            # 高优先级：尽快执行
    MEDIUM = "medium"        # 中优先级：正常调度
    LOW = "low"              # 低优先级：空闲时执行
    BACKGROUND = "background" # 后台任务：最低优先级

class CommunicationProtocol(Enum):
    """通信协议（扩展版本）"""
    HTTP_RPC = "http_rpc"           # HTTP REST调用
    GRPC = "grpc"                   # gRPC调用
    MESSAGE_QUEUE = "message_queue" # 消息队列
    SHARED_MEMORY = "shared_memory" # 共享内存
    WEBSOCKET = "websocket"         # WebSocket实时通信
    DIRECT_MODEL_CALL = "direct_model_call" # 直接模型调用

@dataclass
class CollaborativeTask:
    """协同任务定义"""
    task_id: str
    name: str
    description: str
    required_models: List[str]
    input_data: Any
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: float = TASK_TIMEOUT
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    success: bool
    output: Any
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    model_results: Dict[str, Any] = field(default_factory=dict)
    fusion_strategy: str = ""
    conflict_resolution: Dict[str, Any] = field(default_factory=dict)

class TaskScheduler:
    """任务调度器 - 实现优先级调度机制"""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.task_queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.MEDIUM: deque(),
            TaskPriority.LOW: deque(),
            TaskPriority.BACKGROUND: deque()
        }
        self.active_tasks: Dict[str, CollaborativeTask] = {}
        self.task_history: Dict[str, TaskResult] = {}
        self.lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 性能统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "queue_sizes": defaultdict(int)
        }
    
    def schedule_task(self, task: CollaborativeTask) -> str:
        """调度任务到相应优先级的队列"""
        with self.lock:
            task.scheduled_at = time.time()
            self.task_queues[task.priority].append(task)
            self.active_tasks[task.task_id] = task
            self.stats["total_tasks"] += 1
            self.stats["queue_sizes"][task.priority.value] += 1
            
            logger.info(f"任务已调度: {task.task_id}, 优先级: {task.priority.value}, 队列大小: {len(self.task_queues[task.priority])}")
            return task.task_id
    
    def get_next_task(self) -> Optional[CollaborativeTask]:
        """获取下一个要执行的任务（基于优先级）"""
        with self.lock:
            # 按优先级顺序检查队列
            for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                           TaskPriority.MEDIUM, TaskPriority.LOW, 
                           TaskPriority.BACKGROUND]:
                queue = self.task_queues[priority]
                if queue:
                    task = queue.popleft()
                    self.stats["queue_sizes"][priority.value] -= 1
                    return task
            return None
    
    def update_task_status(self, task_id: str, result: TaskResult):
        """更新任务状态"""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.completed_at = time.time()
                
                # 记录结果
                self.task_history[task_id] = result
                
                # 更新统计
                self.stats["completed_tasks"] += 1
                if not result.success:
                    self.stats["failed_tasks"] += 1
                
                # 计算平均执行时间
                if result.execution_time > 0:
                    total_time = self.stats["avg_execution_time"] * (self.stats["completed_tasks"] - 1)
                    self.stats["avg_execution_time"] = (total_time + result.execution_time) / self.stats["completed_tasks"]
                
                # 从活动任务中移除
                del self.active_tasks[task_id]
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self.lock:
            return {
                "total_tasks": self.stats["total_tasks"],
                "completed_tasks": self.stats["completed_tasks"],
                "failed_tasks": self.stats["failed_tasks"],
                "active_tasks": len(self.active_tasks),
                "avg_execution_time": self.stats["avg_execution_time"],
                "queue_sizes": dict(self.stats["queue_sizes"]),
                "executor_status": {
                    "max_workers": self.max_workers,
                    "active_threads": threading.active_count() - 1
                }
            }

class ModelRPCClient:
    """模型RPC客户端 - 实现HTTP RPC通信"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.model_ports = self._load_model_ports()
        self.cache = {}
        
    def _load_model_ports(self) -> Dict[str, int]:
        """加载模型端口配置"""
        try:
            from core.model_ports_config import MODEL_PORTS
            # 过滤掉非模型的服务端口（如multi_model_collaboration）
            model_ports = {}
            for model_id, port in MODEL_PORTS.items():
                # 排除协同服务本身
                if model_id not in ['multi_model_collaboration']:
                    model_ports[model_id] = port
            return model_ports
        except ImportError:
            logger.warning("无法加载模型端口配置，使用默认端口映射")
            # 默认端口映射（根据更新后的model_ports_config.py）
            return {
                'manager': 8001,
                'language': 8002,
                'knowledge': 8003,
                'vision': 8004,
                'audio': 8005,
                'autonomous': 8006,
                'programming': 8007,
                'planning': 8008,
                'emotion': 8009,
                'spatial': 8010,
                'computer_vision': 8011,
                'sensor': 8012,
                'motion': 8013,
                'prediction': 8014,
                'advanced_reasoning': 8015,
                'data_fusion': 8028,  # 从8016移到8028
                'creative_problem_solving': 8017,
                'meta_cognition': 8018,
                'value_alignment': 8019,
                'vision_image': 8020,
                'vision_video': 8021,
                'finance': 8022,
                'medical': 8023,
                'collaboration': 8024,
                'optimization': 8025,
                'computer': 8026,
                'mathematics': 8027
            }
    
    async def call_model(self, model_id: str, endpoint: str, 
                        data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
        """调用模型服务API"""
        try:
            if model_id not in self.model_ports:
                return {"error": f"未知模型ID: {model_id}"}
            
            port = self.model_ports[model_id]
            url = f"http://localhost:{port}/{endpoint}"
            
            # 记录请求
            logger.debug(f"调用模型 {model_id} (端口:{port}): {endpoint}")
            
            # 发送请求
            if method.upper() == "GET":
                response = await self.client.get(url, params=data)
            else:
                response = await self.client.post(url, json=data)
            
            # 处理响应
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"模型 {model_id} 调用失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except httpx.ConnectError:
            error_msg = f"无法连接到模型 {model_id} (端口:{port})"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"模型调用异常: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

class ResultFusionEngine:
    """结果融合引擎 - 合并多个模型的结果"""
    
    def __init__(self):
        self.fusion_strategies = {
            "weighted_average": self._weighted_average_fusion,
            "majority_vote": self._majority_vote_fusion,
            "expert_consensus": self._expert_consensus_fusion,
            "confidence_based": self._confidence_based_fusion,
            "hierarchical": self._hierarchical_fusion
        }
    
    def fuse_results(self, model_results: Dict[str, Any], 
                    strategy: str = "confidence_based") -> Dict[str, Any]:
        """融合多个模型的结果"""
        if not model_results:
            return {"error": "没有可融合的结果"}
        
        if strategy not in self.fusion_strategies:
            logger.warning(f"未知融合策略: {strategy}，使用默认策略")
            strategy = "confidence_based"
        
        try:
            fusion_func = self.fusion_strategies[strategy]
            fused_result = fusion_func(model_results)
            
            return {
                "success": True,
                "fused_result": fused_result,
                "strategy": strategy,
                "source_models": list(model_results.keys()),
                "fusion_timestamp": time.time()
            }
        except Exception as e:
            error_msg = f"结果融合失败: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _weighted_average_fusion(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """加权平均融合"""
        # 实现加权平均逻辑
        fused = {}
        total_weight = 0
        
        for model_id, result in model_results.items():
            weight = result.get("confidence", 0.5)
            total_weight += weight
            
            # 合并数值型结果
            if isinstance(result.get("value"), (int, float)):
                if "value" not in fused:
                    fused["value"] = 0
                fused["value"] += result["value"] * weight
        
        if total_weight > 0 and "value" in fused:
            fused["value"] /= total_weight
        
        return fused
    
    def _majority_vote_fusion(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """多数投票融合"""
        votes = defaultdict(int)
        
        for model_id, result in model_results.items():
            vote = result.get("decision")
            if vote is not None:
                votes[vote] += 1
        
        if votes:
            majority_decision = max(votes.items(), key=lambda x: x[1])[0]
            return {"decision": majority_decision, "vote_count": votes[majority_decision]}
        
        return {"decision": None, "vote_count": 0}
    
    def _expert_consensus_fusion(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """专家共识融合"""
        # 基于模型专业领域权重进行融合
        expert_weights = {
            "language": 0.9, "knowledge": 0.8, "vision": 0.85,
            "audio": 0.8, "planning": 0.75, "reasoning": 0.9
        }
        
        consensus_result = {}
        total_weight = 0
        
        for model_id, result in model_results.items():
            weight = expert_weights.get(model_id, 0.5)
            total_weight += weight
            
            # 合并结果逻辑
            for key, value in result.items():
                if key not in consensus_result:
                    consensus_result[key] = 0
                if isinstance(value, (int, float)):
                    consensus_result[key] += value * weight
        
        if total_weight > 0:
            for key in consensus_result:
                consensus_result[key] /= total_weight
        
        return consensus_result
    
    def _confidence_based_fusion(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """置信度加权融合"""
        fused_result = {}
        total_confidence = 0
        
        for model_id, result in model_results.items():
            confidence = result.get("confidence", 0.5)
            total_confidence += confidence
            
            # 合并数值结果
            for key, value in result.items():
                if key == "confidence":
                    continue
                    
                if key not in fused_result:
                    fused_result[key] = 0
                
                if isinstance(value, (int, float)):
                    fused_result[key] += value * confidence
                elif isinstance(value, dict):
                    # 递归处理嵌套字典
                    if "nested" not in fused_result:
                        fused_result["nested"] = {}
                    fused_result["nested"][key] = value
        
        # 归一化
        if total_confidence > 0:
            for key in fused_result:
                if isinstance(fused_result[key], (int, float)):
                    fused_result[key] /= total_confidence
        
        fused_result["overall_confidence"] = total_confidence / len(model_results) if model_results else 0
        
        return fused_result
    
    def _hierarchical_fusion(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """层次化融合"""
        # 基于任务类型和模型层次进行融合
        hierarchical_result = {
            "raw_results": model_results,
            "processed_results": {},
            "metadata": {
                "fusion_strategy": "hierarchical",
                "model_count": len(model_results)
            }
        }
        
        # 按模型类型分组
        model_groups = defaultdict(dict)
        for model_id, result in model_results.items():
            model_type = model_id.split("_")[0] if "_" in model_id else model_id
            model_groups[model_type][model_id] = result
        
        # 对每个组进行融合
        for group_type, group_results in model_groups.items():
            if len(group_results) > 1:
                # 组内融合
                group_fused = self._confidence_based_fusion(group_results)
                hierarchical_result["processed_results"][group_type] = group_fused
            else:
                # 单个模型，直接使用结果
                hierarchical_result["processed_results"][group_type] = list(group_results.values())[0]
        
        return hierarchical_result

class MultiModelCollaborationService:
    """多模型协同服务主类"""
    
    def __init__(self, port: int = COLLABORATION_SERVICE_PORT):
        self.port = port
        self.app = FastAPI(
            title="Multi-Model Collaboration Service",
            version="1.0.0",
            description="AGI多模型协同服务，提供任务分发、结果融合、冲突解决等功能"
        )
        
        # 核心组件
        self.scheduler = TaskScheduler()
        self.rpc_client = ModelRPCClient()
        self.fusion_engine = ResultFusionEngine()
        self.enhanced_collaboration = EnhancedModelCollaboration()
        self.system_monitor = SystemMonitor()
        
        # 任务执行器
        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # 服务状态
        self.is_running = False
        self.start_time = None
        
        # 设置API路由
        self._setup_routes()
        
        logger.info(f"多模型协同服务初始化完成，端口: {port}")
    
    def _setup_routes(self):
        """设置FastAPI路由"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Multi-Model Collaboration Service",
                "port": self.port,
                "status": "running" if self.is_running else "stopped",
                "uptime": time.time() - self.start_time if self.start_time else 0
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "service": "collaboration",
                "version": "1.0.0"
            }
        
        @self.app.post("/api/v1/tasks")
        async def create_task(task_request: Dict[str, Any]):
            """创建新的协同任务"""
            try:
                task_id = str(uuid.uuid4())
                
                # 验证请求
                if "name" not in task_request:
                    raise HTTPException(status_code=400, detail="任务名称不能为空")
                
                if "required_models" not in task_request or not task_request["required_models"]:
                    raise HTTPException(status_code=400, detail="必须指定至少一个模型")
                
                # 创建任务对象
                priority_str = task_request.get("priority", "medium")
                try:
                    priority = TaskPriority(priority_str)
                except ValueError:
                    priority = TaskPriority.MEDIUM
                
                task = CollaborativeTask(
                    task_id=task_id,
                    name=task_request["name"],
                    description=task_request.get("description", ""),
                    required_models=task_request["required_models"],
                    input_data=task_request.get("input_data", {}),
                    priority=priority,
                    timeout=task_request.get("timeout", TASK_TIMEOUT),
                    max_retries=task_request.get("max_retries", 3),
                    dependencies=task_request.get("dependencies", []),
                    metadata=task_request.get("metadata", {})
                )
                
                # 调度任务
                self.scheduler.schedule_task(task)
                
                # 在后台执行任务
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self._execute_task, task)
                
                return {
                    "task_id": task_id,
                    "status": "scheduled",
                    "message": f"任务 '{task.name}' 已调度",
                    "priority": priority.value,
                    "scheduled_at": task.scheduled_at
                }
                
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "创建任务失败")
                raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")
        
        @self.app.get("/api/v1/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """获取任务状态"""
            try:
                # 检查活动任务
                if task_id in self.scheduler.active_tasks:
                    task = self.scheduler.active_tasks[task_id]
                    return {
                        "task_id": task_id,
                        "status": "active",
                        "name": task.name,
                        "priority": task.priority.value,
                        "created_at": task.created_at,
                        "scheduled_at": task.scheduled_at,
                        "started_at": task.started_at
                    }
                
                # 检查历史任务
                if task_id in self.scheduler.task_history:
                    result = self.scheduler.task_history[task_id]
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "completed_at": result.task_id in self.scheduler.active_tasks and 
                                      self.scheduler.active_tasks[result.task_id].completed_at or None
                    }
                
                raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
                
            except HTTPException:
                raise
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "获取任务状态失败")
                raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")
        
        @self.app.delete("/api/v1/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """取消任务"""
            try:
                # 检查任务是否存在
                if task_id not in self.scheduler.active_tasks:
                    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在或已完成")
                
                # 从活动任务中移除
                if task_id in self.scheduler.active_tasks:
                    del self.scheduler.active_tasks[task_id]
                
                # 从调度器队列中移除（简化实现）
                # 注意：实际实现需要遍历所有优先级队列
                
                logger.info(f"任务 {task_id} 已取消")
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "message": "任务已成功取消"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "取消任务失败")
                raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")
        
        @self.app.get("/api/v1/tasks")
        async def list_tasks(status: str = None, limit: int = 50):
            """列出任务"""
            try:
                tasks = []
                
                # 活动任务
                for task_id, task in self.scheduler.active_tasks.items():
                    if status and status != "active":
                        continue
                    tasks.append({
                        "task_id": task_id,
                        "name": task.name,
                        "status": "active",
                        "priority": task.priority.value,
                        "created_at": task.created_at
                    })
                
                # 历史任务
                if not status or status == "completed":
                    for task_id, result in list(self.scheduler.task_history.items())[:limit]:
                        tasks.append({
                            "task_id": task_id,
                            "name": "已完成任务",
                            "status": "completed",
                            "success": result.success,
                            "execution_time": result.execution_time
                        })
                
                return {
                    "tasks": tasks[:limit],
                    "total": len(tasks),
                    "active_count": len(self.scheduler.active_tasks),
                    "completed_count": len(self.scheduler.task_history)
                }
                
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "列出任务失败")
                raise HTTPException(status_code=500, detail=f"列出任务失败: {str(e)}")
        
        @self.app.get("/api/v1/models")
        async def list_available_models():
            """列出可用模型"""
            try:
                model_registry = get_model_registry()
                models = []
                
                for model_id, port in self.rpc_client.model_ports.items():
                    models.append({
                        "model_id": model_id,
                        "port": port,
                        "available": True  # 简化：假设所有模型都可用
                    })
                
                return {
                    "models": models,
                    "total": len(models)
                }
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "列出模型失败")
                raise HTTPException(status_code=500, detail=f"列出模型失败: {str(e)}")
        
        @self.app.get("/api/v1/stats")
        async def get_service_stats():
            """获取服务统计信息"""
            try:
                scheduler_stats = self.scheduler.get_scheduler_stats()
                system_stats = self.system_monitor.get_system_stats() if hasattr(self.system_monitor, 'get_system_stats') else {}
                
                return {
                    "service": {
                        "port": self.port,
                        "is_running": self.is_running,
                        "uptime": time.time() - self.start_time if self.start_time else 0,
                        "start_time": self.start_time
                    },
                    "scheduler": scheduler_stats,
                    "system": system_stats
                }
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "获取统计信息失败")
                raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")
        
        @self.app.post("/api/v1/collaborate")
        async def direct_collaboration(collaboration_request: Dict[str, Any]):
            """直接协作请求"""
            try:
                pattern_name = collaboration_request.get("pattern", "multimodal_fusion")
                input_data = collaboration_request.get("input_data", {})
                config = collaboration_request.get("config", {})
                
                # 使用增强协作系统
                result = await self.enhanced_collaboration.initiate_collaboration(
                    pattern_name, input_data, config
                )
                
                return result
            except Exception as e:
                error_handler.handle_error(e, "MultiModelCollaborationService", "协作请求失败")
                raise HTTPException(status_code=500, detail=f"协作请求失败: {str(e)}")
    
    async def _execute_task(self, task: CollaborativeTask):
        """执行协同任务"""
        task.started_at = time.time()
        start_time = time.time()
        
        try:
            logger.info(f"开始执行任务: {task.task_id} - {task.name}")
            
            # 1. 并行调用所有所需模型
            model_results = await self._call_models_parallel(task)
            
            # 2. 检查是否有失败
            failed_models = []
            successful_results = {}
            
            for model_id, result in model_results.items():
                if "error" in result:
                    failed_models.append(model_id)
                else:
                    successful_results[model_id] = result
            
            # 3. 如果有失败且允许重试
            if failed_models and task.max_retries > 0:
                logger.warning(f"任务 {task.task_id} 有模型调用失败: {failed_models}，尝试重试")
                # 简化：这里可以实现重试逻辑
            
            # 4. 融合结果
            fusion_result = self.fusion_engine.fuse_results(successful_results)
            
            # 5. 创建任务结果
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.task_id,
                success="error" not in fusion_result and len(successful_results) > 0,
                output=fusion_result.get("fused_result", {}) if "error" not in fusion_result else fusion_result,
                errors=failed_models if failed_models else [],
                execution_time=execution_time,
                model_results=model_results,
                fusion_strategy=fusion_result.get("strategy", ""),
                conflict_resolution={}  # 可以添加冲突解决结果
            )
            
            # 6. 更新任务状态
            self.scheduler.update_task_status(task.task_id, task_result)
            
            logger.info(f"任务完成: {task.task_id}，执行时间: {execution_time:.2f}s，成功: {task_result.success}")
            
        except Exception as e:
            error_msg = f"任务执行失败: {str(e)}"
            logger.error(error_msg)
            error_handler.handle_error(e, "MultiModelCollaborationService", f"任务 {task.task_id} 执行失败")
            
            # 记录失败结果
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                output={"error": error_msg},
                errors=[str(e)],
                execution_time=execution_time
            )
            
            self.scheduler.update_task_status(task.task_id, task_result)
    
    async def _call_models_parallel(self, task: CollaborativeTask) -> Dict[str, Any]:
        """并行调用多个模型"""
        model_results = {}
        
        # 创建异步任务
        async def call_single_model(model_id: str):
            try:
                # 构建请求数据
                request_data = {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "input_data": task.input_data,
                    "priority": task.priority.value,
                    "metadata": task.metadata
                }
                
                # 调用模型
                result = await self.rpc_client.call_model(
                    model_id, "api/process", request_data
                )
                
                return model_id, result
            except Exception as e:
                return model_id, {"error": f"模型调用异常: {str(e)}"}
        
        # 并行调用所有模型
        tasks = [call_single_model(model_id) for model_id in task.required_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for result in results:
            if isinstance(result, Exception):
                model_id = "unknown"
                model_results[model_id] = {"error": f"任务执行异常: {str(result)}"}
            else:
                model_id, result_data = result
                model_results[model_id] = result_data
        
        return model_results
    
    async def start(self):
        """启动协同服务"""
        if self.is_running:
            logger.warning("协同服务已经在运行")
            return
        
        try:
            import time
            self.start_time = time.time()
            self.is_running = True
            
            # 启动系统监控
            if hasattr(self.system_monitor, 'start'):
                self.system_monitor.start()
            
            logger.info(f"多模型协同服务启动在端口 {self.port}")
            
            # 启动FastAPI服务器（在后台线程中运行以避免阻塞主事件循环）
            import threading
            
            def run_server():
                """在后台线程中运行uvicorn服务器"""
                try:
                    import asyncio
                    import uvicorn
                    
                    # 为新线程创建事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    config = uvicorn.Config(
                        app=self.app,
                        host=SERVICE_HOST,
                        port=self.port,
                        log_level="info"
                    )
                    server = uvicorn.Server(config)
                    
                    logger.info(f"在后台线程中启动uvicorn服务器，端口: {self.port}")
                    loop.run_until_complete(server.serve())
                    
                except Exception as e:
                    error_msg = f"后台线程中启动协同服务失败: {str(e)}"
                    logger.error(error_msg)
                    error_handler.handle_error(e, "MultiModelCollaborationService", "后台启动失败")
            
            # 启动后台线程
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # 等待一小段时间让服务器启动
            time.sleep(0.5)
            
            logger.info(f"多模型协同服务启动完成，端口: {self.port}")
            
        except Exception as e:
            error_msg = f"启动协同服务失败: {str(e)}"
            logger.error(error_msg)
            error_handler.handle_error(e, "MultiModelCollaborationService", "启动失败")
            raise
    
    async def stop(self):
        """停止协同服务"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # 关闭RPC客户端
            await self.rpc_client.close()
            
            # 停止任务执行器
            self.task_executor.shutdown(wait=False)
            
            # 停止系统监控
            if hasattr(self.system_monitor, 'stop'):
                self.system_monitor.stop()
            
            logger.info("多模型协同服务已停止")
            
        except Exception as e:
            error_msg = f"停止协同服务失败: {str(e)}"
            logger.error(error_msg)
            error_handler.handle_error(e, "MultiModelCollaborationService", "停止失败")

# 全局服务实例
_collaboration_service_instance = None

def get_collaboration_service(port: int = COLLABORATION_SERVICE_PORT) -> MultiModelCollaborationService:
    """获取协同服务实例（单例模式）"""
    global _collaboration_service_instance
    if _collaboration_service_instance is None:
        _collaboration_service_instance = MultiModelCollaborationService(port)
    return _collaboration_service_instance

async def start_collaboration_service(port: int = COLLABORATION_SERVICE_PORT):
    """启动协同服务"""
    service = get_collaboration_service(port)
    await service.start()

async def stop_collaboration_service():
    """停止协同服务"""
    global _collaboration_service_instance
    if _collaboration_service_instance:
        await _collaboration_service_instance.stop()
        _collaboration_service_instance = None

# 主函数：直接运行服务
if __name__ == "__main__":
    import sys
    import asyncio
    
    port = COLLABORATION_SERVICE_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"错误：无效的端口号 {sys.argv[1]}，使用默认端口 {COLLABORATION_SERVICE_PORT}")
    
    print(f"启动多模型协同服务，端口: {port}")
    print(f"API文档: http://localhost:{port}/docs")
    
    try:
        # 直接运行服务并保持主线程运行
        import asyncio
        import time
        
        # 创建服务实例
        service = get_collaboration_service(port)
        
        # 启动服务（异步）
        async def run_service():
            await service.start()
            print(f"多模型协同服务已启动在端口 {port}")
            print("服务正在运行...")
            
            # 保持主线程运行，直到收到中断信号
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n接收到中断信号，正在停止服务...")
                await service.stop()
                print("服务已停止")
        
        # 运行服务
        asyncio.run(run_service())
        
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止服务...")
        try:
            asyncio.run(stop_collaboration_service())
        except:
            pass
    except Exception as e:
        print(f"服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)