"""
应急终止管理器 - 全局一键停止和安全终止机制

主要功能：
1. 全局应急终止接口，前端/后端/硬件层三重停止按钮
2. 终止后清理执行队列，禁止残留指令下发
3. 应急终止记录原因 + 操作人
4. 响应时间≤1s；无残留指令执行

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import time
import json
import logging
import threading
import inspect
import signal
import asyncio
import queue
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError

from core.error_handling import error_handler

# 日志记录器
logger = logging.getLogger(__name__)


class EmergencyStopLevel(Enum):
    """应急终止级别"""
    SOFT_STOP = "soft_stop"        # 软停止：允许任务优雅完成
    HARD_STOP = "hard_stop"        # 硬停止：立即终止所有任务
    EMERGENCY_KILL = "emergency_kill"  # 紧急终止：强制终止进程


class StopTriggerSource(Enum):
    """停止触发来源"""
    FRONTEND = "frontend"          # 前端界面
    BACKEND_API = "backend_api"    # 后端API
    HARDWARE_BUTTON = "hardware_button"  # 硬件按钮
    AUTOMATED_MONITOR = "automated_monitor"  # 自动化监控
    ETHICAL_CONSTRAINT = "ethical_constraint"  # 伦理约束
    SECURITY_ALERT = "security_alert"  # 安全警报


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_name: str
    executor_id: str
    submit_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, cancelled, failed
    progress: float = 0.0  # 进度 0.0-1.0
    metadata: Optional[Dict[str, Any]] = None
    future: Optional[Future] = None


@dataclass
class ExecutorInfo:
    """执行器信息"""
    executor_id: str
    executor_type: str  # ThreadPoolExecutor, ProcessPoolExecutor, etc.
    max_workers: int
    created_at: float
    active_tasks: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    cancelled_tasks: int = 0
    is_stopped: bool = False
    executor_ref: Optional[Any] = None


@dataclass
class StopEvent:
    """停止事件记录"""
    event_id: str
    timestamp: float
    stop_level: EmergencyStopLevel
    trigger_source: StopTriggerSource
    initiated_by: str
    reason: str
    affected_tasks: int = 0
    cancelled_tasks: int = 0
    successful: bool = True
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class EmergencyStopManager:
    """应急终止管理器
    
    主要功能：
    1. 全局应急终止接口
    2. 执行队列清理
    3. 任务状态跟踪
    4. 审计日志记录
    """
    
    def __init__(self):
        """初始化应急终止管理器"""
        # 执行器注册表
        self.executors: Dict[str, ExecutorInfo] = {}
        
        # 任务注册表
        self.tasks: Dict[str, TaskInfo] = {}
        
        # 停止事件历史
        self.stop_events: List[StopEvent] = []
        
        # 互斥锁
        self.lock = threading.RLock()
        
        # 状态标志
        self.emergency_stop_active = False
        self.stop_level = None
        self.last_stop_time = 0.0
        
        # 回调函数
        self.pre_stop_callbacks: List[Callable] = []
        self.post_stop_callbacks: List[Callable] = []
        
        # 配置
        self.config = {
            'max_stop_events': 1000,
            'default_response_timeout_ms': 1000,  # 1秒响应超时
            'max_workers_per_executor': 10,
            'enable_task_tracking': True,
            'enable_audit_logging': True,
            'allow_new_tasks_after_stop': False,
            'auto_cleanup_delay_seconds': 5,  # 停止后自动清理延迟
        }
        
        # 硬件层接口（模拟）
        self.hardware_stop_registered = False
        
        # 信号处理
        self._setup_signal_handlers()
        
        logger.info("应急终止管理器初始化完成")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            # 注册SIGINT (Ctrl+C) 和 SIGTERM 信号处理
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("信号处理器已注册")
        except (ValueError, AttributeError) as e:
            # 在某些环境中可能不支持信号处理
            logger.warning(f"无法注册信号处理器: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.warning(f"收到终止信号 {signum}, 执行应急停止")
        self.emergency_stop(
            stop_level=EmergencyStopLevel.HARD_STOP,
            trigger_source=StopTriggerSource.AUTOMATED_MONITOR,
            initiated_by="system_signal",
            reason=f"收到系统信号 {signum}"
        )
    
    def register_executor(self, executor: ThreadPoolExecutor, 
                         executor_id: str = None,
                         max_workers: int = None) -> str:
        """注册执行器以进行监控
        
        Args:
            executor: 要注册的执行器实例
            executor_id: 执行器ID，None则自动生成
            max_workers: 最大工作线程数
            
        Returns:
            执行器ID
        """
        with self.lock:
            if executor_id is None:
                executor_id = f"executor_{uuid.uuid4().hex[:8]}"
            
            if executor_id in self.executors:
                logger.warning(f"执行器 {executor_id} 已存在，跳过注册")
                return executor_id
            
            if max_workers is None:
                # 尝试获取执行器的最大工作线程数
                try:
                    max_workers = executor._max_workers
                except AttributeError:
                    max_workers = self.config['max_workers_per_executor']
            
            executor_info = ExecutorInfo(
                executor_id=executor_id,
                executor_type=type(executor).__name__,
                max_workers=max_workers,
                created_at=time.time(),
                executor_ref=executor
            )
            
            self.executors[executor_id] = executor_info
            logger.info(f"注册执行器: {executor_id} ({executor_info.executor_type}, max_workers={max_workers})")
            
            return executor_id
    
    def unregister_executor(self, executor_id: str) -> bool:
        """注销执行器
        
        Args:
            executor_id: 执行器ID
            
        Returns:
            是否成功注销
        """
        with self.lock:
            if executor_id not in self.executors:
                logger.warning(f"执行器 {executor_id} 未找到")
                return False
            
            # 标记执行器为已停止
            executor_info = self.executors[executor_id]
            executor_info.is_stopped = True
            
            # 从注册表中移除
            del self.executors[executor_id]
            
            logger.info(f"注销执行器: {executor_id}")
            return True
    
    def register_task(self, task_id: str, task_name: str, executor_id: str,
                     future: Optional[Future] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册任务以进行跟踪
        
        Args:
            task_id: 任务ID
            task_name: 任务名称
            executor_id: 执行器ID
            future: Future对象（可选）
            metadata: 任务元数据（可选）
            
        Returns:
            是否成功注册
        """
        if not self.config['enable_task_tracking']:
            return True
        
        with self.lock:
            if self.emergency_stop_active and not self.config['allow_new_tasks_after_stop']:
                logger.warning(f"应急停止激活中，拒绝新任务: {task_id}")
                return False
            
            if task_id in self.tasks:
                logger.warning(f"任务 {task_id} 已存在，跳过注册")
                return True
            
            if executor_id not in self.executors:
                logger.warning(f"执行器 {executor_id} 未注册，任务跟踪可能不完整")
            
            task_info = TaskInfo(
                task_id=task_id,
                task_name=task_name,
                executor_id=executor_id,
                submit_time=time.time(),
                metadata=metadata or {},
                future=future
            )
            
            self.tasks[task_id] = task_info
            
            # 更新执行器统计
            if executor_id in self.executors:
                self.executors[executor_id].pending_tasks += 1
            
            logger.debug(f"注册任务: {task_id} ({task_name}), 执行器: {executor_id}")
            return True
    
    def update_task_status(self, task_id: str, status: str, 
                          progress: float = None, metadata_update: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            progress: 进度 (0.0-1.0)
            metadata_update: 元数据更新
            
        Returns:
            是否成功更新
        """
        if not self.config['enable_task_tracking']:
            return True
        
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"任务 {task_id} 未找到")
                return False
            
            task_info = self.tasks[task_id]
            old_status = task_info.status
            
            # 更新状态
            task_info.status = status
            
            if progress is not None:
                task_info.progress = max(0.0, min(1.0, progress))
            
            if metadata_update:
                task_info.metadata.update(metadata_update)
            
            # 更新执行时间
            if status == "running" and task_info.start_time is None:
                task_info.start_time = time.time()
                # 更新执行器统计
                if task_info.executor_id in self.executors:
                    executor = self.executors[task_info.executor_id]
                    executor.pending_tasks = max(0, executor.pending_tasks - 1)
                    executor.active_tasks += 1
            
            elif status in ["completed", "cancelled", "failed"]:
                task_info.end_time = time.time()
                # 更新执行器统计
                if task_info.executor_id in self.executors:
                    executor = self.executors[task_info.executor_id]
                    
                    if old_status == "running":
                        executor.active_tasks = max(0, executor.active_tasks - 1)
                    elif old_status == "pending":
                        executor.pending_tasks = max(0, executor.pending_tasks - 1)
                    
                    if status == "completed":
                        executor.completed_tasks += 1
                    elif status == "cancelled":
                        executor.cancelled_tasks += 1
            
            logger.debug(f"更新任务状态: {task_id} {old_status} -> {status}, 进度: {task_info.progress:.1%}")
            return True
    
    def register_pre_stop_callback(self, callback: Callable):
        """注册停止前回调函数
        
        Args:
            callback: 回调函数，接受(stop_level, trigger_source, reason)参数
        """
        with self.lock:
            self.pre_stop_callbacks.append(callback)
            logger.debug(f"注册停止前回调: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def register_post_stop_callback(self, callback: Callable):
        """注册停止后回调函数
        
        Args:
            callback: 回调函数，接受(stop_event)参数
        """
        with self.lock:
            self.post_stop_callbacks.append(callback)
            logger.debug(f"注册停止后回调: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def register_hardware_stop_button(self, callback: Callable):
        """注册硬件停止按钮回调
        
        Args:
            callback: 硬件按钮回调函数
        """
        self.hardware_stop_registered = True
        self.register_pre_stop_callback(callback)
        logger.info("硬件停止按钮已注册")
    
    def emergency_stop(self, stop_level: EmergencyStopLevel = EmergencyStopLevel.HARD_STOP,
                      trigger_source: StopTriggerSource = StopTriggerSource.BACKEND_API,
                      initiated_by: str = "unknown", reason: str = "紧急终止") -> Dict[str, Any]:
        """执行应急停止
        
        Args:
            stop_level: 停止级别
            trigger_source: 触发来源
            initiated_by: 发起者
            reason: 停止原因
            
        Returns:
            停止结果
        """
        start_time = time.time()
        
        with self.lock:
            # 检查是否已经在停止过程中
            if self.emergency_stop_active:
                logger.warning("应急停止已在执行中")
                return {
                    "success": False,
                    "error": "应急停止已在执行中",
                    "stop_id": None
                }
            
            # 设置停止标志
            self.emergency_stop_active = True
            self.stop_level = stop_level
            self.last_stop_time = start_time
            
            # 创建停止事件
            event_id = f"stop_{uuid.uuid4().hex[:8]}"
            stop_event = StopEvent(
                event_id=event_id,
                timestamp=start_time,
                stop_level=stop_level,
                trigger_source=trigger_source,
                initiated_by=initiated_by,
                reason=reason
            )
        
        logger.warning(f"开始应急停止: 级别={stop_level.value}, 来源={trigger_source.value}, 原因={reason}")
        
        try:
            # 执行停止前回调
            self._execute_pre_stop_callbacks(stop_level, trigger_source, reason)
            
            # 根据停止级别执行不同的停止策略
            if stop_level == EmergencyStopLevel.SOFT_STOP:
                result = self._execute_soft_stop()
            elif stop_level == EmergencyStopLevel.HARD_STOP:
                result = self._execute_hard_stop()
            elif stop_level == EmergencyStopLevel.EMERGENCY_KILL:
                result = self._execute_emergency_kill()
            else:
                result = {"success": False, "error": f"未知停止级别: {stop_level}"}
            
            # 计算响应时间
            response_time_ms = (time.time() - start_time) * 1000
            
            # 更新停止事件
            with self.lock:
                stop_event.affected_tasks = len(self.tasks)
                stop_event.cancelled_tasks = result.get("cancelled_tasks", 0)
                stop_event.successful = result.get("success", False)
                stop_event.response_time_ms = response_time_ms
                stop_event.error_message = result.get("error")
                
                # 保存停止事件
                self.stop_events.append(stop_event)
                
                # 限制事件数量
                if len(self.stop_events) > self.config['max_stop_events']:
                    self.stop_events = self.stop_events[-self.config['max_stop_events']:]
            
            # 执行停止后回调
            self._execute_post_stop_callbacks(stop_event)
            
            # 记录结果
            logger.warning(f"应急停止完成: 事件ID={event_id}, 响应时间={response_time_ms:.1f}ms, "
                          f"取消任务={stop_event.cancelled_tasks}/{stop_event.affected_tasks}")
            
            return {
                "success": True,
                "stop_id": event_id,
                "response_time_ms": response_time_ms,
                "cancelled_tasks": stop_event.cancelled_tasks,
                "total_tasks": stop_event.affected_tasks,
                "stop_level": stop_level.value,
                "trigger_source": trigger_source.value
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EmergencyStopManager", "应急停止执行失败")
            
            # 更新停止事件
            with self.lock:
                stop_event.successful = False
                stop_event.error_message = str(e)
                self.stop_events.append(stop_event)
            
            return {
                "success": False,
                "error": str(e),
                "stop_id": event_id
            }
        finally:
            # 重置停止标志（如果允许新任务）
            if self.config['allow_new_tasks_after_stop']:
                with self.lock:
                    self.emergency_stop_active = False
                    self.stop_level = None
    
    def _execute_pre_stop_callbacks(self, stop_level: EmergencyStopLevel,
                                  trigger_source: StopTriggerSource, reason: str):
        """执行停止前回调"""
        for callback in self.pre_stop_callbacks:
            try:
                callback(stop_level, trigger_source, reason)
                logger.debug(f"停止前回调执行成功: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
            except Exception as e:
                error_handler.handle_error(e, "EmergencyStopManager", "停止前回调执行失败")
    
    def _execute_post_stop_callbacks(self, stop_event: StopEvent):
        """执行停止后回调"""
        for callback in self.post_stop_callbacks:
            try:
                callback(stop_event)
                logger.debug(f"停止后回调执行成功: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
            except Exception as e:
                error_handler.handle_error(e, "EmergencyStopManager", "停止后回调执行失败")
    
    def _execute_soft_stop(self) -> Dict[str, Any]:
        """执行软停止：允许正在运行的任务完成，拒绝新任务"""
        cancelled_tasks = 0
        
        with self.lock:
            # 拒绝新任务
            for executor_id, executor_info in self.executors.items():
                executor_info.is_stopped = True
            
            # 取消所有待处理任务
            for task_id, task_info in list(self.tasks.items()):
                if task_info.status == "pending" and task_info.future is not None:
                    try:
                        if task_info.future.cancel():
                            task_info.status = "cancelled"
                            task_info.end_time = time.time()
                            cancelled_tasks += 1
                            logger.debug(f"取消待处理任务: {task_id}")
                    except Exception as e:
                        logger.warning(f"取消任务失败 {task_id}: {e}")
        
        logger.info(f"软停止完成: 取消 {cancelled_tasks} 个待处理任务")
        return {"success": True, "cancelled_tasks": cancelled_tasks}
    
    def _execute_hard_stop(self) -> Dict[str, Any]:
        """执行硬停止：立即停止所有任务"""
        cancelled_tasks = 0
        
        with self.lock:
            # 停止所有执行器
            for executor_id, executor_info in list(self.executors.items()):
                executor_info.is_stopped = True
                
                # 关闭执行器
                if executor_info.executor_ref is not None:
                    try:
                        # ThreadPoolExecutor.shutdown(wait=False) 会立即返回
                        executor_info.executor_ref.shutdown(wait=False, cancel_futures=True)
                        logger.debug(f"停止执行器: {executor_id}")
                    except Exception as e:
                        logger.warning(f"停止执行器失败 {executor_id}: {e}")
            
            # 取消所有任务
            for task_id, task_info in list(self.tasks.items()):
                if task_info.status in ["pending", "running"] and task_info.future is not None:
                    try:
                        if task_info.future.cancel():
                            task_info.status = "cancelled"
                            task_info.end_time = time.time()
                            cancelled_tasks += 1
                            logger.debug(f"取消任务: {task_id}")
                    except Exception as e:
                        logger.warning(f"取消任务失败 {task_id}: {e}")
                elif task_info.status == "pending":
                    # 即使没有future也标记为取消
                    task_info.status = "cancelled"
                    task_info.end_time = time.time()
                    cancelled_tasks += 1
        
        # 清理队列（通过重新创建执行器）
        self._cleanup_executor_queues()
        
        logger.info(f"硬停止完成: 取消 {cancelled_tasks} 个任务")
        return {"success": True, "cancelled_tasks": cancelled_tasks}
    
    def _execute_emergency_kill(self) -> Dict[str, Any]:
        """执行紧急终止：强制终止进程"""
        logger.critical("执行紧急终止：强制终止进程")
        
        # 首先执行硬停止
        result = self._execute_hard_stop()
        
        # 记录强制终止事件
        error_handler.log_critical("系统执行紧急终止", "EmergencyStopManager", {
            "stop_level": "emergency_kill",
            "cancelled_tasks": result.get("cancelled_tasks", 0),
            "timestamp": time.time()
        })
        
        # 在实际部署中，这里可能会调用os._exit()或发送SIGKILL
        # 但在开发环境中，我们只记录日志
        
        return {
            "success": True,
            "cancelled_tasks": result.get("cancelled_tasks", 0),
            "warning": "开发环境：实际部署中将强制终止进程"
        }
    
    def _cleanup_executor_queues(self):
        """清理执行器队列"""
        with self.lock:
            # 重新创建执行器来清理队列
            for executor_id, executor_info in list(self.executors.items()):
                if executor_info.executor_ref is not None:
                    try:
                        # 尝试获取执行器的工作队列
                        if hasattr(executor_info.executor_ref, '_work_queue'):
                            queue_size = executor_info.executor_ref._work_queue.qsize()
                            if queue_size > 0:
                                logger.info(f"清理执行器 {executor_id} 队列: {queue_size} 个待处理项目")
                                # 清空队列
                                try:
                                    while not executor_info.executor_ref._work_queue.empty():
                                        try:
                                            executor_info.executor_ref._work_queue.get_nowait()
                                        except queue.Empty:
                                            break
                                except Exception as e:
                                    logger.warning(f"清理队列失败 {executor_id}: {e}")
                    except Exception as e:
                        logger.debug(f"无法访问执行器队列 {executor_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取应急终止管理器状态"""
        with self.lock:
            total_tasks = len(self.tasks)
            
            # 统计任务状态
            task_status_counts = {}
            for task in self.tasks.values():
                status = task.status
                task_status_counts[status] = task_status_counts.get(status, 0) + 1
            
            # 统计执行器状态
            active_executors = len([e for e in self.executors.values() if not e.is_stopped])
            stopped_executors = len([e for e in self.executors.values() if e.is_stopped])
            
            # 计算执行器统计
            executor_stats = {}
            for executor in self.executors.values():
                executor_stats[executor.executor_id] = {
                    "type": executor.executor_type,
                    "max_workers": executor.max_workers,
                    "active_tasks": executor.active_tasks,
                    "pending_tasks": executor.pending_tasks,
                    "completed_tasks": executor.completed_tasks,
                    "cancelled_tasks": executor.cancelled_tasks,
                    "is_stopped": executor.is_stopped,
                    "created_at": executor.created_at
                }
            
            status = {
                "emergency_stop_active": self.emergency_stop_active,
                "stop_level": self.stop_level.value if self.stop_level else None,
                "last_stop_time": self.last_stop_time,
                "total_executors": len(self.executors),
                "active_executors": active_executors,
                "stopped_executors": stopped_executors,
                "total_tasks": total_tasks,
                "task_status_counts": task_status_counts,
                "executor_stats": executor_stats,
                "hardware_button_registered": self.hardware_stop_registered,
                "pre_stop_callbacks": len(self.pre_stop_callbacks),
                "post_stop_callbacks": len(self.post_stop_callbacks),
                "total_stop_events": len(self.stop_events),
                "allow_new_tasks": not self.emergency_stop_active or self.config['allow_new_tasks_after_stop'],
                "timestamp": time.time()
            }
            
            return status
    
    def get_recent_stop_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的停止事件
        
        Args:
            limit: 返回最大数量
            
        Returns:
            停止事件列表
        """
        with self.lock:
            recent_events = self.stop_events[-limit:] if self.stop_events else []
            
            # 转换为字典
            result = []
            for event in recent_events:
                event_dict = asdict(event)
                event_dict['stop_level'] = event.stop_level.value
                event_dict['trigger_source'] = event.trigger_source.value
                event_dict['timestamp_iso'] = datetime.fromtimestamp(event.timestamp).isoformat()
                result.append(event_dict)
            
            return result
    
    def reset(self, keep_events: bool = True) -> bool:
        """重置应急终止管理器（用于测试或恢复）
        
        Args:
            keep_events: 是否保留停止事件历史
            
        Returns:
            是否成功重置
        """
        with self.lock:
            # 重置状态
            self.emergency_stop_active = False
            self.stop_level = None
            
            # 清空任务列表
            self.tasks.clear()
            
            # 重置执行器状态
            for executor in self.executors.values():
                executor.is_stopped = False
                executor.active_tasks = 0
                executor.pending_tasks = 0
            
            # 可选清空停止事件
            if not keep_events:
                self.stop_events.clear()
            
            logger.info("应急终止管理器已重置")
            return True
    
    def create_frontend_stop_button(self) -> Dict[str, Any]:
        """创建前端停止按钮配置
        
        Returns:
            前端配置
        """
        return {
            "component_type": "emergency_stop_button",
            "button_text": "紧急停止",
            "button_color": "red",
            "confirmation_required": True,
            "confirmation_message": "确认执行紧急停止？所有正在运行的任务将被取消。",
            "api_endpoint": "/api/emergency/stop",
            "stop_levels": [
                {"value": "soft_stop", "label": "软停止", "description": "允许当前任务完成"},
                {"value": "hard_stop", "label": "硬停止", "description": "立即停止所有任务"},
                {"value": "emergency_kill", "label": "紧急终止", "description": "强制终止进程（仅限紧急情况）"}
            ],
            "default_stop_level": "hard_stop",
            "timestamp": time.time()
        }


# 全局实例
_emergency_stop_manager_instance = None

def get_emergency_stop_manager() -> EmergencyStopManager:
    """获取全局应急终止管理器实例"""
    global _emergency_stop_manager_instance
    if _emergency_stop_manager_instance is None:
        _emergency_stop_manager_instance = EmergencyStopManager()
    return _emergency_stop_manager_instance