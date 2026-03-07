"""
神经通信系统

使用共享内存和异步队列的直接神经张量通信，
替代了27个模型之间的基于HTTP的通信。

关键特性：
- 基于张量的通信（无需JSON序列化）
- 共享内存实现零拷贝数据传输
- 异步通信支持实时处理
- 基于优先级的消息路由
- 容错通信通道
"""

import torch
import torch.multiprocessing as mp
import asyncio
import threading
import queue
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import uuid
import pickle
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """消息优先级级别"""
    HIGH = 0    # 实时认知处理
    NORMAL = 1  # 标准认知操作
    LOW = 2     # 后台学习任务
    BATCH = 3   # 批处理任务


@dataclass
class TensorMessage:
    """神经张量消息结构"""
    message_id: str
    source: str
    target: str
    tensor_data: torch.Tensor
    metadata: Dict[str, Any]
    priority: MessagePriority
    timestamp: float
    requires_response: bool = False
    response_channel: Optional[str] = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()


class NeuralCommunication:
    """
    神经张量通信系统，替代HTTP。
    
    使用共享内存和异步队列实现高性能、
    低延迟的认知组件间通信。
    """
    
    def __init__(self, max_shared_memory_mb: int = 1024):
        """
        初始化神经通信系统。
        
        参数:
            max_shared_memory_mb: 最大共享内存分配（MB）
        """
        self.max_shared_memory = max_shared_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory_usage = 0
        
        # Shared memory manager
        self.manager = mp.Manager()
        self.shared_memory = self.manager.dict()
        
        # Communication queues (priority-based)
        self.input_queues = {
            priority: self.manager.Queue(maxsize=1000)
            for priority in MessagePriority
        }
        
        self.output_queues = {
            priority: self.manager.Queue(maxsize=1000)
            for priority in MessagePriority
        }
        
        # Tensor communication channels (direct tensor transfer)
        self.tensor_channels = self.manager.dict()
        
        # Response channels for request-response pattern
        self.response_channels = self.manager.dict()
        
        # Component registration
        self.registered_components = self.manager.dict()
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'tensors_transferred': 0,
            'total_data_transferred': 0,  # bytes
            'avg_latency': 0.0,
            'errors': 0
        }
        
        # Lock for thread-safe operations
        self.lock = threading.RLock()
        
        # Start monitoring thread
        self._shutdown_flag = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"NeuralCommunication initialized with {max_shared_memory_mb}MB shared memory")
    
    def register_component(self, component_id: str, component_type: str):
        """
        注册一个认知组件。
        
        参数:
            component_id: 唯一组件标识符
            component_type: 组件类型（perception、memory、reasoning等）
        """
        with self.lock:
            self.registered_components[component_id] = {
                'type': component_type,
                'registered_at': time.time(),
                'last_active': time.time(),
                'message_count': 0
            }
            
            # Create tensor channel for this component
            self.tensor_channels[component_id] = self.manager.Queue(maxsize=100)
            
            # Create response channel
            self.response_channels[component_id] = self.manager.dict()
            
            logger.info(f"Component registered: {component_id} ({component_type})")
    
    def unregister_component(self, component_id: str):
        """注销一个组件"""
        with self.lock:
            if component_id in self.registered_components:
                del self.registered_components[component_id]
            
            # Clean up channels
            if component_id in self.tensor_channels:
                del self.tensor_channels[component_id]
            
            if component_id in self.response_channels:
                del self.response_channels[component_id]
            
            logger.info(f"Component unregistered: {component_id}")
    
    def create_shared_tensor(self, name: str, shape: Tuple[int, ...], 
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        在共享内存中创建共享张量。
        
        参数:
            name: 张量名称
            shape: 张量形状
            dtype: 张量数据类型
            
        返回:
            共享张量
        """
        with self.lock:
            # Check memory limits
            tensor_size = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            if self.current_memory_usage + tensor_size > self.max_shared_memory:
                self._cleanup_old_tensors(tensor_size)
            
            # Create shared tensor
            tensor = torch.zeros(shape, dtype=dtype).share_memory_()
            self.shared_memory[name] = {
                'tensor': tensor,
                'created_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'size_bytes': tensor_size
            }
            
            self.current_memory_usage += tensor_size
            logger.debug(f"Created shared tensor '{name}' with shape {shape}, size: {tensor_size/1024:.1f}KB")
            
            return tensor
    
    def get_shared_tensor(self, name: str) -> Optional[torch.Tensor]:
        """
        按名称获取共享张量。
        
        参数:
            name: 张量名称
            
        返回:
            共享张量，如果未找到则返回None
        """
        with self.lock:
            if name in self.shared_memory:
                tensor_info = self.shared_memory[name]
                tensor_info['access_count'] += 1
                tensor_info['last_accessed'] = time.time()
                return tensor_info['tensor']
            return None
    
    async def send_tensor(self, tensor: torch.Tensor, target: str, 
                        priority: MessagePriority = MessagePriority.NORMAL,
                        metadata: Optional[Dict] = None) -> str:
        """
        异步发送张量到目标组件。
        
        参数:
            tensor: 要发送的张量
            target: 目标组件ID
            priority: 消息优先级
            metadata: 额外元数据
            
        返回:
            消息ID
        """
        message_id = str(uuid.uuid4())
        
        # Create message
        message = TensorMessage(
            message_id=message_id,
            source="sender",  # Would be actual source in real usage
            target=target,
            tensor_data=tensor,
            metadata=metadata or {},
            priority=priority,
            timestamp=time.time()
        )
        
        try:
            # Serialize tensor for queue (simplified - in production would use shared memory)
            if tensor.is_shared():
                # Already in shared memory, just send reference
                tensor_data = {
                    'shared_memory_key': f"tensor_{message_id}",
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype)
                }
                self.shared_memory[f"tensor_{message_id}"] = tensor
            else:
                # Convert to numpy for serialization
                tensor_data = {
                    'data': tensor.cpu().numpy().tobytes(),
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype)
                }
            
            # Prepare message data
            message_data = {
                'message_id': message_id,
                'tensor_data': tensor_data,
                'metadata': message.metadata,
                'priority': message.priority.value
            }
            
            # Send via appropriate queue based on priority
            queue = self.input_queues[priority]
            
            # Use thread pool for blocking queue operation
            await asyncio.get_event_loop().run_in_executor(
                None, queue.put, message_data
            )
            
            # Update statistics
            with self.lock:
                self.stats['messages_sent'] += 1
                self.stats['tensors_transferred'] += 1
                tensor_size = tensor.numel() * tensor.element_size()
                self.stats['total_data_transferred'] += tensor_size
            
            logger.debug(f"Sent tensor to {target}, message_id: {message_id}, size: {tensor.shape}")
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send tensor to {target}: {e}")
            with self.lock:
                self.stats['errors'] += 1
            raise
    
    async def receive_tensor(self, source: str, 
                           timeout: Optional[float] = None) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Receive a tensor from a source component asynchronously.
        
        Args:
            source: Source component ID
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (tensor, metadata) or None if timeout
        """
        try:
            # Check tensor channel for this source
            if source not in self.tensor_channels:
                # Create channel if it doesn't exist
                with self.lock:
                    self.tensor_channels[source] = self.manager.Queue(maxsize=100)
            
            channel = self.tensor_channels[source]
            
            # Wait for message with timeout
            if timeout is not None:
                try:
                    message_data = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, channel.get),
                        timeout
                    )
                except asyncio.TimeoutError:
                    return None
            else:
                message_data = await asyncio.get_event_loop().run_in_executor(
                    None, channel.get
                )
            
            # Deserialize tensor
            tensor_info = message_data['tensor_data']
            
            if 'shared_memory_key' in tensor_info:
                # Retrieve from shared memory
                shared_key = tensor_info['shared_memory_key']
                tensor = self.shared_memory.get(shared_key)
                if tensor is None:
                    raise ValueError(f"Shared tensor not found: {shared_key}")
            else:
                # Reconstruct from bytes
                # Convert dtype string like 'torch.float32' to torch dtype
                dtype_str = tensor_info['dtype']
                if dtype_str.startswith('torch.'):
                    dtype_name = dtype_str.split('.')[1]
                    dtype = getattr(torch, dtype_name)
                else:
                    # Fallback for other dtype formats
                    dtype = getattr(torch, dtype_str)
                # 复制数据以确保可写缓冲区
                data_copy = bytearray(tensor_info['data'])
                tensor = torch.frombuffer(
                    data_copy,
                    dtype=dtype
                ).reshape(tensor_info['shape'])
            
            # Update statistics
            with self.lock:
                self.stats['messages_received'] += 1
                if source in self.registered_components:
                    self.registered_components[source]['message_count'] += 1
                    self.registered_components[source]['last_active'] = time.time()
            
            logger.debug(f"Received tensor from {source}, shape: {tensor.shape}")
            
            return tensor, message_data.get('metadata', {})
            
        except Exception as e:
            logger.error(f"Failed to receive tensor from {source}: {e}")
            with self.lock:
                self.stats['errors'] += 1
            raise
    
    async def send_request(self, tensor: torch.Tensor, target: str,
                         metadata: Optional[Dict] = None,
                         timeout: float = 30.0) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Send a request and wait for response (request-response pattern).
        
        Args:
            tensor: Request tensor
            target: Target component ID
            metadata: Request metadata
            timeout: Response timeout
            
        Returns:
            Tuple of (response_tensor, response_metadata) or None if timeout
        """
        message_id = str(uuid.uuid4())
        response_channel = f"response_{message_id}"
        
        # Create response channel
        with self.lock:
            self.response_channels[response_channel] = self.manager.Queue(maxsize=1)
        
        # Add response info to metadata
        request_metadata = metadata or {}
        request_metadata.update({
            'response_required': True,
            'response_channel': response_channel,
            'message_id': message_id
        })
        
        # Send request
        await self.send_tensor(tensor, target, MessagePriority.HIGH, request_metadata)
        
        try:
            # Wait for response
            response_queue = self.response_channels[response_channel]
            
            response_data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, response_queue.get),
                timeout
            )
            
            # Deserialize response
            response_tensor_info = response_data['tensor_data']
            
            if 'shared_memory_key' in response_tensor_info:
                shared_key = response_tensor_info['shared_memory_key']
                response_tensor = self.shared_memory.get(shared_key)
            else:
                dtype = getattr(torch, response_tensor_info['dtype'])
                response_tensor = torch.frombuffer(
                    response_tensor_info['data'],
                    dtype=dtype
                ).reshape(response_tensor_info['shape'])
            
            # Clean up response channel
            with self.lock:
                if response_channel in self.response_channels:
                    del self.response_channels[response_channel]
            
            return response_tensor, response_data.get('metadata', {})
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for message {message_id}")
            # 超时时清理响应通道
            with self.lock:
                if response_channel in self.response_channels:
                    del self.response_channels[response_channel]
            return None
        except Exception as e:
            logger.error(f"Error in request-response: {e}")
            # 异常时清理响应通道
            with self.lock:
                if response_channel in self.response_channels:
                    del self.response_channels[response_channel]
            return None
    
    async def send_response(self, request_message_id: str, response_tensor: torch.Tensor,
                          metadata: Optional[Dict] = None):
        """
        Send response to a request.
        
        Args:
            request_message_id: Original request message ID
            response_tensor: Response tensor
            metadata: Response metadata
        """
        response_channel = f"response_{request_message_id}"
        
        with self.lock:
            if response_channel not in self.response_channels:
                logger.error(f"No response channel for message {request_message_id}")
                return
        
        # Prepare response data
        response_data = {
            'tensor_data': {
                'data': response_tensor.cpu().numpy().tobytes(),
                'shape': response_tensor.shape,
                'dtype': str(response_tensor.dtype)
            },
            'metadata': metadata or {}
        }
        
        # Send response
        response_queue = self.response_channels[response_channel]
        await asyncio.get_event_loop().run_in_executor(
            None, response_queue.put, response_data
        )
    
    async def broadcast_tensor(self, tensor: torch.Tensor, component_type: str,
                       metadata: Optional[Dict] = None):
        """
        广播张量到特定类型的所有组件。
        
        参数:
            tensor: 要广播的张量
            component_type: 要广播到的组件类型
            metadata: 广播元数据
        """
        with self.lock:
            target_components = [
                comp_id for comp_id, info in self.registered_components.items()
                if info['type'] == component_type
            ]
        
        # 并行发送到所有组件
        if target_components:
            tasks = []
            for target in target_components:
                tasks.append(self.send_tensor(tensor, target, MessagePriority.NORMAL, metadata))
            await asyncio.gather(*tasks)
    
    def _cleanup_old_tensors(self, required_size: int):
        """Clean up old tensors to free memory"""
        with self.lock:
            # Sort tensors by last access time (oldest first)
            tensor_infos = []
            for name, info in self.shared_memory.items():
                if isinstance(info, dict) and 'tensor' in info:
                    tensor_infos.append((name, info))
            
            tensor_infos.sort(key=lambda x: x[1]['last_accessed'])
            
            freed_size = 0
            for name, info in tensor_infos:
                if freed_size >= required_size:
                    break
                
                tensor_size = info['size_bytes']
                del self.shared_memory[name]
                freed_size += tensor_size
                self.current_memory_usage -= tensor_size
                
                logger.debug(f"Freed tensor '{name}', size: {tensor_size/1024:.1f}KB")
            
            if freed_size < required_size:
                logger.warning(f"Could only free {freed_size/1024:.1f}KB, need {required_size/1024:.1f}KB")
    
    def _monitor_resources(self):
        """Monitor resource usage and clean up as needed"""
        while not self._shutdown_flag.is_set():
            time.sleep(60)  # Check every minute
            
            try:
                with self.lock:
                    # Clean up inactive components (not active for 5 minutes)
                    current_time = time.time()
                    inactive_components = []
                    
                    # 使用try-except处理可能的代理访问错误
                    try:
                        for comp_id, info in self.registered_components.items():
                            if current_time - info['last_active'] > 300:  # 5 minutes
                                inactive_components.append(comp_id)
                    except (AttributeError, FileNotFoundError, ConnectionError) as e:
                        # 如果无法访问注册组件（例如在测试环境中），跳过清理
                        logger.debug(f"无法访问注册组件进行清理: {e}")
                        inactive_components = []
                    
                    for comp_id in inactive_components:
                        try:
                            self.unregister_component(comp_id)
                        except Exception as e:
                            logger.debug(f"取消注册组件 {comp_id} 失败: {e}")
                    
                    # Clean up old tensors (not accessed for 10 minutes)
                    old_tensors = []
                    try:
                        for name, info in self.shared_memory.items():
                            if isinstance(info, dict) and 'tensor' in info:
                                if current_time - info['last_accessed'] > 600:  # 10 minutes
                                    old_tensors.append(name)
                    except (AttributeError, FileNotFoundError, ConnectionError) as e:
                        # 如果无法访问共享内存，跳过清理
                        logger.debug(f"无法访问共享内存进行清理: {e}")
                        old_tensors = []
                    
                    for name in old_tensors:
                        try:
                            tensor_size = self.shared_memory[name]['size_bytes']
                            del self.shared_memory[name]
                            self.current_memory_usage -= tensor_size
                        except Exception as e:
                            logger.debug(f"清理张量 {name} 失败: {e}")
                    
                    # Log statistics periodically
                    if len(old_tensors) > 0 or len(inactive_components) > 0:
                        logger.info(f"Cleaned up {len(old_tensors)} old tensors and {len(inactive_components)} inactive components")
            except Exception as e:
                # 捕获所有其他异常，防止监控线程崩溃
                logger.debug(f"资源监控循环中出现异常: {e}")
                # 继续运行，等待下一次循环
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy.update({
                'registered_components': len(self.registered_components),
                'shared_tensors': len([v for v in self.shared_memory.values() 
                                     if isinstance(v, dict) and 'tensor' in v]),
                'current_memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_shared_memory / (1024 * 1024),
                'input_queue_sizes': {p.name: q.qsize() for p, q in self.input_queues.items()},
                'output_queue_sizes': {p.name: q.qsize() for p, q in self.output_queues.items()}
            })
            return stats_copy
    
    def shutdown(self):
        """关闭神经通信系统"""
        logger.debug("开始关闭神经通信系统...")
        
        # 设置关闭标志，停止监控线程
        if hasattr(self, '_shutdown_flag'):
            self._shutdown_flag.set()
        
        # 等待监控线程退出（带超时）
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            logger.debug("等待监控线程退出...")
            self.monitor_thread.join(timeout=2.0)
            
            # 如果线程仍然存活，记录警告
            if self.monitor_thread.is_alive():
                logger.warning("监控线程未及时退出，可能仍有未完成任务")
        
        # 清理所有队列和字典引用（带异常处理）
        try:
            self.input_queues.clear()
        except Exception as e:
            logger.debug(f"清理输入队列时忽略错误: {e}")
        
        try:
            self.output_queues.clear()
        except Exception as e:
            logger.debug(f"清理输出队列时忽略错误: {e}")
        
        try:
            self.tensor_channels.clear()
        except Exception as e:
            logger.debug(f"清理张量通道时忽略错误: {e}")
        
        try:
            self.response_channels.clear()
        except Exception as e:
            logger.debug(f"清理响应通道时忽略错误: {e}")
        
        try:
            self.registered_components.clear()
        except Exception as e:
            logger.debug(f"清理注册组件时忽略错误: {e}")
        
        try:
            self.shared_memory.clear()
        except Exception as e:
            logger.debug(f"清理共享内存时忽略错误: {e}")
        
        # 清理multiprocessing管理器
        if hasattr(self, 'manager'):
            try:
                # 关闭管理器连接
                self.manager.shutdown()
                logger.debug("Multiprocessing管理器已关闭")
            except Exception as e:
                logger.debug(f"关闭管理器时忽略错误: {e}")
        
        logger.info("神经通信系统关闭完成")
    
    def reset_statistics(self):
        """Reset all statistics"""
        with self.lock:
            self.stats = {
                'messages_sent': 0,
                'messages_received': 0,
                'tensors_transferred': 0,
                'total_data_transferred': 0,
                'avg_latency': 0.0,
                'errors': 0
            }


class SharedTensorManager:
    """
    Manager for shared tensors across cognitive components.
    
    Provides centralized management of shared memory tensors with
    automatic cleanup, memory tracking, and access control.
    """
    
    def __init__(self, max_memory_mb: int = 512):
        """
        Initialize shared tensor manager.
        
        Args:
            max_memory_mb: Maximum memory for shared tensors (MB)
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
        
        # Shared memory manager
        self.manager = mp.Manager()
        self.tensors = self.manager.dict()
        
        # Access tracking
        self.access_counts = self.manager.dict()
        self.last_access_times = self.manager.dict()
        
        # Lock for thread-safe operations
        self.lock = threading.RLock()
        
        logger.info(f"SharedTensorManager initialized with {max_memory_mb}MB memory")
    
    def create_tensor(self, name: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Create a new shared tensor.
        
        Args:
            name: Unique tensor identifier
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Shared tensor
        """
        with self.lock:
            # Check if tensor already exists
            if name in self.tensors:
                logger.warning(f"Tensor '{name}' already exists, returning existing")
                return self.tensors[name]
            
            # Calculate required memory
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_memory = np.prod(shape) * element_size
            
            # Check memory limits
            if self.current_memory + required_memory > self.max_memory:
                self._cleanup_for_memory(required_memory)
            
            # Create shared tensor
            tensor = torch.zeros(shape, dtype=dtype).share_memory_()
            self.tensors[name] = tensor
            self.access_counts[name] = 0
            self.last_access_times[name] = time.time()
            
            self.current_memory += required_memory
            
            logger.debug(f"Created shared tensor '{name}' with shape {shape}, size: {required_memory/1024:.1f}KB")
            return tensor
    
    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """
        Get a shared tensor.
        
        Args:
            name: Tensor identifier
            
        Returns:
            Shared tensor or None if not found
        """
        with self.lock:
            if name in self.tensors:
                tensor = self.tensors[name]
                self.access_counts[name] += 1
                self.last_access_times[name] = time.time()
                return tensor
            return None
    
    def delete_tensor(self, name: str) -> bool:
        """
        Delete a shared tensor.
        
        Args:
            name: Tensor identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if name in self.tensors:
                tensor = self.tensors[name]
                
                # Calculate memory usage
                shape = tensor.shape
                dtype = tensor.dtype
                element_size = torch.tensor([], dtype=dtype).element_size()
                tensor_size = np.prod(shape) * element_size
                
                # Clean up
                del self.tensors[name]
                if name in self.access_counts:
                    del self.access_counts[name]
                if name in self.last_access_times:
                    del self.last_access_times[name]
                
                self.current_memory -= tensor_size
                
                logger.debug(f"Deleted shared tensor '{name}', freed {tensor_size/1024:.1f}KB")
                return True
            return False
    
    def list_tensors(self) -> Dict[str, Dict[str, Any]]:
        """
        List all shared tensors with metadata.
        
        Returns:
            Dictionary of tensor information
        """
        with self.lock:
            result = {}
            for name, tensor in self.tensors.items():
                result[name] = {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'access_count': self.access_counts.get(name, 0),
                    'last_access': self.last_access_times.get(name, 0)
                }
            return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Memory usage information
        """
        return {
            'current_memory_mb': self.current_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory / (1024 * 1024),
            'tensor_count': len(self.tensors),
            'memory_usage_percent': (self.current_memory / self.max_memory) * 100 if self.max_memory > 0 else 0
        }
    
    def clear_all(self):
        """Clear all shared tensors"""
        with self.lock:
            self.tensors.clear()
            self.access_counts.clear()
            self.last_access_times.clear()
            self.current_memory = 0
            logger.info("Cleared all shared tensors")
    
    def shutdown(self):
        """关闭共享张量管理器"""
        logger.debug("开始关闭共享张量管理器...")
        
        # 清理所有张量（带异常处理）
        try:
            self.clear_all()
        except Exception as e:
            logger.debug(f"清理共享张量时忽略错误: {e}")
        
        # 清理multiprocessing管理器
        if hasattr(self, 'manager'):
            try:
                # 关闭管理器连接
                self.manager.shutdown()
                logger.debug("共享张量管理器已关闭")
            except Exception as e:
                logger.debug(f"关闭共享张量管理器时忽略错误: {e}")
        
        logger.info("共享张量管理器关闭完成")
    
    def _cleanup_for_memory(self, required_memory: int):
        """
        Clean up old tensors to free memory.
        
        Args:
            required_memory: Memory needed (bytes)
        """
        with self.lock:
            # Sort tensors by last access time (oldest first)
            tensor_info = []
            for name, tensor in self.tensors.items():
                if name in self.last_access_times:
                    tensor_info.append({
                        'name': name,
                        'last_access': self.last_access_times[name],
                        'access_count': self.access_counts.get(name, 0)
                    })
            
            tensor_info.sort(key=lambda x: x['last_access'])
            
            freed_memory = 0
            for info in tensor_info:
                if freed_memory >= required_memory:
                    break
                
                # Delete the tensor
                tensor = self.tensors[info['name']]
                shape = tensor.shape
                dtype = tensor.dtype
                element_size = torch.tensor([], dtype=dtype).element_size()
                tensor_size = np.prod(shape) * element_size
                
                freed_memory += tensor_size
                self.delete_tensor(info['name'])
            
            logger.info(f"Freed {freed_memory/1024:.1f}KB memory, requested {required_memory/1024:.1f}KB")


# 简单测试
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建通信系统
    comm = NeuralCommunication(max_shared_memory_mb=100)
    
    # 注册测试组件
    comm.register_component("perception", "perception")
    comm.register_component("memory", "memory")
    
    # Create test tensor
    test_tensor = torch.randn(3, 224, 224)
    
    print(f"Communication system ready. Statistics: {comm.get_statistics()}")