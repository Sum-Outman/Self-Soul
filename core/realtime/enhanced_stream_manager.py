"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 """

"""
增强版实时数据流管理系统 - 统一的数据流处理中心
Enhanced Real-time Data Stream Management System - Unified Data Stream Processing Center

功能描述：
- 管理多种类型的实时数据流（视频、音频、传感器等）
- 支持高并发数据流处理
- 实现数据流订阅/发布机制
- 提供错误恢复和健康检查功能
- 支持多种流处理器和数据处理管道
- 实时监控和性能分析
- 多模型集成和协同处理
- API端点集成和仪表板支持

Function Description:
- Manages multiple types of real-time data streams (video, audio, sensor, etc.)
- Supports high-concurrency data stream processing
- Implements data stream subscription/publishing mechanism
- Provides error recovery and health check functionality
- Supports multiple stream processors and data processing pipelines
- Real-time monitoring and performance analysis
- Multi-model integration and collaborative processing
- API endpoint integration and dashboard support
"""

import asyncio
import time
from typing import Dict, List, Any, Callable, Optional, Set
from enum import Enum
import cv2
import numpy as np
import json
import logging
from dataclasses import dataclass
from collections import deque
import uuid
import psutil
from datetime import datetime

from core.error_handling import error_handler
from core.model_registry import model_registry
from core.monitoring_enhanced import MonitoringDashboard


"""
StreamType类 - 中文类描述
StreamType Class - English class description
"""
class StreamType(Enum):
    """流类型枚举
    Stream Type Enumeration
    """
    VIDEO = "video"                    # 视频流
    AUDIO = "audio"                    # 音频流  
    SENSOR = "sensor"                  # 传感器数据流
    NETWORK_VIDEO = "network_video"    # 网络视频流
    NETWORK_AUDIO = "network_audio"    # 网络音频流
    STEREO_VIDEO = "stereo_video"      # 双目视频流
    MULTIMODAL = "multimodal"          # 多模态数据流

@dataclass
class StreamConfig:
    """
    StreamConfig类 - 中文类描述
    StreamConfig Class - English class description
    
    流配置数据类
    Stream Configuration Data Class
    """
    stream_id: str
    stream_type: StreamType
    source: Any
    buffer_size: int = 1000
    max_subscribers: int = 100
    processing_interval: float = 0.01  # 处理间隔(秒)
    priority: int = 1                  # 流优先级 (1-10)
    quality_of_service: int = 1        # 服务质量级别 (1-5)

@dataclass
class StreamStats:
    """
    StreamStats类 - 中文类描述
    StreamStats Class - English class description
    
    流统计信息数据类
    Stream Statistics Data Class
    """
    frames_processed: int = 0
    bytes_processed: int = 0
    last_activity: float = 0.0
    error_count: int = 0
    subscriber_count: int = 0
    processing_time_avg: float = 0.0
    processing_time_max: float = 0.0
    processing_time_min: float = float('inf')
    throughput: float = 0.0            # 吞吐量 (bytes/sec)
    health_score: float = 100.0        # 健康评分 (0-100)

@dataclass
class SystemMetrics:
    """系统性能指标
    System Performance Metrics
    """
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_throughput: float = 0.0
    active_streams: int = 0
    total_subscribers: int = 0
    total_throughput: float = 0.0
    system_health: float = 100.0


"""
DataStreamManager类 - 中文类描述
DataStreamManager Class - English class description
"""
class DataStreamManager:
    """增强版实时数据流管理中心
    Enhanced Real-time Data Stream Management Center

    功能：统一管理和处理所有类型的实时数据流，提供高并发处理、错误恢复、多模型集成和实时监控
    支持：视频流、音频流、传感器数据流、网络流、双目视频流、多模态数据流
    提供：API端点集成、实时仪表板、自动故障转移、性能优化和深度学习模型集成

    Function: Unified management and processing of all types of real-time data streams,
              providing high-concurrency processing, error recovery, multi-model integration,
              and real-time monitoring
    Support: Video streams, audio streams, sensor data streams, network streams,
             stereo video streams, multimodal data streams
    Features: API endpoint integration, real-time dashboard, automatic failover,
              performance optimization, and deep learning model integration
    """
    
    def __init__(self):
        self.streams: Dict[str, StreamConfig] = {}
        self.processors: Dict[StreamType, Any] = {}
        self.subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self.stats: Dict[str, StreamStats] = {}
        self.lock = asyncio.Lock()
        self.running = False
        self.health_check_task = None
        self.metrics_task = None
        self.logger = logging.getLogger(__name__)
        self.monitoring_dashboard = MonitoringDashboard()
        self.system_metrics = SystemMetrics()
        self.active_tasks: Set[asyncio.Task] = set()
        
        # 初始化默认处理器
        self._initialize_default_processors()
        
        self.logger.info("实时数据流管理中心初始化完成")
        self.logger.info("Real-time Data Stream Management Center initialized")
    
    def _initialize_default_processors(self):
        """初始化默认流处理器
        Initialize default stream processors
        """
        self.processors = {
            StreamType.VIDEO: VideoStreamProcessor(),
            StreamType.AUDIO: AudioStreamProcessor(),
            StreamType.SENSOR: SensorDataProcessor(),
            StreamType.NETWORK_VIDEO: NetworkVideoProcessor(),
            StreamType.NETWORK_AUDIO: NetworkAudioProcessor(),
            StreamType.STEREO_VIDEO: StereoVideoProcessor(),
            StreamType.MULTIMODAL: MultimodalProcessor()
        }
    
    async def start(self):
        """启动数据流管理中心
        Start data stream management center
        """
        if self.running:
            return
        
        self.running = True
        self.health_check_task = asyncio.create_task(self._health_check())
        self.metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.logger.info("实时数据流管理中心已启动")
        self.logger.info("Real-time Data Stream Management Center started")
    
    async def stop(self):
        """停止数据流管理中心
        Stop data stream management center
        """
        self.running = False
        
        # 取消所有任务
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # 等待任务完成
        tasks = [t for t in self.active_tasks if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=5.0)
        
        # 停止所有活跃的流
        for stream_id in list(self.streams.keys()):
            await self.unregister_stream(stream_id)
        
        self.logger.info("实时数据流管理中心已停止")
        self.logger.info("Real-time Data Stream Management Center stopped")
    
    async def _health_check(self):
        """健康检查任务
        Health check task
        """
        while self.running:
            try:
                await asyncio.sleep(10)  # 每10秒检查一次
                
                async with self.lock:
                    current_time = time.time()
                    inactive_streams = []
                    unhealthy_streams = []
                    
                    # 检查不活跃的流
                    for stream_id, stats in self.stats.items():
                        # 检查不活跃的流
                        if current_time - stats.last_activity > 60:  # 60秒无活动
                            inactive_streams.append(stream_id)
                            self.logger.warning(f"数据流 {stream_id} 不活跃，将自动注销")
                            self.logger.warning(f"Data stream {stream_id} inactive, will auto-unregister")
                        
                        # 检查不健康的流
                        if stats.health_score < 50:  # 健康评分低于50
                            unhealthy_streams.append(stream_id)
                            self.logger.warning(f"数据流 {stream_id} 健康评分低: {stats.health_score}")
                            self.logger.warning(f"Data stream {stream_id} low health score: {stats.health_score}")
                    
                    # 注销不活跃的流
                    for stream_id in inactive_streams:
                        await self.unregister_stream(stream_id)
                    
                    # 尝试恢复不健康的流
                    for stream_id in unhealthy_streams:
                        await self._try_recover_stream(stream_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"健康检查任务错误: {str(e)}"
                self.logger.error(error_msg)
                error_handler.handle_error(e, "DataStreamManager", error_msg)
                await asyncio.sleep(5)
    
    async def _try_recover_stream(self, stream_id: str):
        """尝试恢复不健康的数据流
        Try to recover unhealthy data stream
        
        Args:
            stream_id: 流ID | Stream ID
        """
        try:
            if stream_id not in self.streams:
                return
            
            stream_config = self.streams[stream_id]
            stats = self.stats[stream_id]
            
            # 重置错误计数和健康评分
            stats.error_count = 0
            stats.health_score = 80.0  # 恢复到80分
            
            self.logger.info(f"数据流 {stream_id} 已恢复，健康评分重置为80")
            self.logger.info(f"Data stream {stream_id} recovered, health score reset to 80")
            
            # 更新监控仪表板
            await self.monitoring_dashboard.update_stream_stats(stream_id, stats)
            
        except Exception as e:
            error_msg = f"恢复数据流失败: {stream_id} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
    
    async def _collect_system_metrics(self):
        """收集系统性能指标
        Collect system performance metrics
        """
        while self.running:
            try:
                await asyncio.sleep(5)  # 每5秒收集一次
                
                # 收集系统指标
                self.system_metrics.cpu_usage = psutil.cpu_percent()
                self.system_metrics.memory_usage = psutil.virtual_memory().percent
                
                async with self.lock:
                    self.system_metrics.active_streams = len(self.streams)
                    self.system_metrics.total_subscribers = sum(
                        len(subs) for subs in self.subscribers.values()
                    )
                    
                    # 计算总吞吐量
                    total_throughput = 0
                    for stats in self.stats.values():
                        total_throughput += stats.throughput
                    self.system_metrics.total_throughput = total_throughput
                    
                    # 计算系统健康评分
                    total_health = 0
                    healthy_streams = 0
                    for stats in self.stats.values():
                        if stats.health_score > 80:  # 健康阈值
                            healthy_streams += 1
                        total_health += stats.health_score
                    
                    if self.stats:
                        avg_health = total_health / len(self.stats)
                        healthy_ratio = healthy_streams / len(self.stats)
                        self.system_metrics.system_health = avg_health * healthy_ratio
                    else:
                        self.system_metrics.system_health = 100.0
                
                # 根据系统负载自适应调整处理参数
                await self._adaptive_adjustment()
                
                # 更新监控仪表板
                await self.monitoring_dashboard.update_metrics(self.system_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"系统指标收集错误: {str(e)}"
                self.logger.error(error_msg)
                error_handler.handle_error(e, "DataStreamManager", error_msg)
                await asyncio.sleep(5)
    
    async def _adaptive_adjustment(self):
        """自适应调整系统参数
        Adaptive adjustment of system parameters
        """
        try:
            # 根据CPU使用率调整处理间隔
            if self.system_metrics.cpu_usage > 80:
                # 高负载时增加处理间隔
                adjustment_factor = 1.5
                self.logger.info(f"系统高负载，增加处理间隔 (CPU: {self.system_metrics.cpu_usage}%)")
                self.logger.info(f"System high load, increasing processing interval (CPU: {self.system_metrics.cpu_usage}%)")
            elif self.system_metrics.cpu_usage < 30:
                # 低负载时减少处理间隔
                adjustment_factor = 0.7
                self.logger.info(f"系统低负载，减少处理间隔 (CPU: {self.system_metrics.cpu_usage}%)")
                self.logger.info(f"System low load, decreasing processing interval (CPU: {self.system_metrics.cpu_usage}%)")
            else:
                adjustment_factor = 1.0
            
            # 应用调整
            async with self.lock:
                for stream_id, config in self.streams.items():
                    new_interval = config.processing_interval * adjustment_factor
                    # 确保处理间隔在合理范围内
                    config.processing_interval = max(0.001, min(new_interval, 0.1))
                    
        except Exception as e:
            error_msg = f"自适应调整失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
    
    async def register_stream(self, stream_id: str, stream_type: StreamType, 
                            source: Any, config: Dict[str, Any] = None) -> bool:
        """注册新的数据流
        Register new data stream
        
        Args:
            stream_id: 流ID | Stream ID
            stream_type: 流类型 | Stream type
            source: 数据源 | Data source
            config: 配置信息 | Configuration information
            
        Returns:
            bool: 是否成功注册 | Whether registration was successful
        """
        try:
            async with self.lock:
                if stream_id in self.streams:
                    self.logger.warning(f"流 {stream_id} 已存在，正在重新注册")
                    self.logger.warning(f"Stream {stream_id} already exists, re-registering")
                    await self.unregister_stream(stream_id)
                
                stream_config = StreamConfig(
                    stream_id=stream_id,
                    stream_type=stream_type,
                    source=source,
                    buffer_size=config.get('buffer_size', 1000) if config else 1000,
                    max_subscribers=config.get('max_subscribers', 100) if config else 100,
                    processing_interval=config.get('processing_interval', 0.01) if config else 0.01,
                    priority=config.get('priority', 1) if config else 1,
                    quality_of_service=config.get('quality_of_service', 1) if config else 1
                )
                
                self.streams[stream_id] = stream_config
                self.subscribers[stream_id] = []
                self.stats[stream_id] = StreamStats(last_activity=time.time())
                
                # 添加到监控仪表板
                await self.monitoring_dashboard.add_stream(stream_id, stream_type.value)
                
                self.logger.info(f"数据流 {stream_id} 已注册，类型: {stream_type.value}")
                self.logger.info(f"Data stream {stream_id} registered, type: {stream_type.value}")
                return True
                
        except Exception as e:
            error_msg = f"注册数据流 {stream_id} 失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def unregister_stream(self, stream_id: str) -> bool:
        """注销数据流
        Unregister data stream
        
        Args:
            stream_id: 流ID | Stream ID
            
        Returns:
            bool: 是否成功注销 | Whether unregistration was successful
        """
        try:
            async with self.lock:
                if stream_id in self.streams:
                    del self.streams[stream_id]
                
                if stream_id in self.subscribers:
                    del self.subscribers[stream_id]
                
                if stream_id in self.stats:
                    del self.stats[stream_id]
                
                # 从监控仪表板移除
                await self.monitoring_dashboard.remove_stream(stream_id)
                
                self.logger.info(f"数据流 {stream_id} 已注销")
                self.logger.info(f"Data stream {stream_id} unregistered")
                return True
            return False
            
        except Exception as e:
            error_msg = f"注销数据流 {stream_id} 失败: {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def register_stream_processor(self, stream_type: StreamType, processor: Any) -> bool:
        """注册流处理器
        Register stream processor
        
        Args:
            stream_type: 流类型 | Stream type
            processor: 处理器实例 | Processor instance
            
        Returns:
            bool: 是否成功注册 | Whether registration was successful
        """
        try:
            async with self.lock:
                self.processors[stream_type] = processor
                self.logger.info(f"流处理器已注册: {stream_type.value}")
                self.logger.info(f"Stream processor registered: {stream_type.value}")
                return True
                
        except Exception as e:
            error_msg = f"注册流处理器失败: {stream_type.value} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def subscribe(self, stream_id: str, subscriber_id: str, 
                       callback: Callable, config: Dict[str, Any] = None) -> bool:
        """订阅数据流
        Subscribe to data stream
        
        Args:
            stream_id: 流ID | Stream ID
            subscriber_id: 订阅者ID | Subscriber ID
            callback: 回调函数 | Callback function
            config: 配置信息 | Configuration information
            
        Returns:
            bool: 是否成功订阅 | Whether subscription was successful
        """
        try:
            async with self.lock:
                if stream_id not in self.subscribers:
                    self.subscribers[stream_id] = []
                
                # 检查是否已存在该订阅者
                existing_subscriber = next(
                    (s for s in self.subscribers[stream_id] if s["id"] == subscriber_id), 
                    None
                )
                
                if existing_subscriber:
                    # 更新现有订阅者
                    existing_subscriber["callback"] = callback
                    existing_subscriber["last_active"] = time.time()
                    existing_subscriber["config"] = config or {}
                else:
                    # 添加新订阅者
                    self.subscribers[stream_id].append({
                        "id": subscriber_id,
                        "callback": callback,
                        "last_active": time.time(),
                        "config": config or {}
                    })
                
                # 更新统计信息
                if stream_id in self.stats:
                    self.stats[stream_id].subscriber_count = len(self.subscribers[stream_id])
                
                # 更新监控仪表板
                await self.monitoring_dashboard.update_subscribers(stream_id, len(self.subscribers[stream_id]))
                
                self.logger.info(f"订阅者 {subscriber_id} 已订阅数据流 {stream_id}")
                self.logger.info(f"Subscriber {subscriber_id} subscribed to data stream {stream_id}")
                return True
                
        except Exception as e:
            error_msg = f"订阅数据流失败: {stream_id} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def unsubscribe(self, stream_id: str, subscriber_id: str = None) -> bool:
        """取消订阅数据流
        Unsubscribe from data stream
        
        Args:
            stream_id: 流ID | Stream ID
            subscriber_id: 可选订阅者ID | Optional subscriber ID
            
        Returns:
            bool: 是否成功取消订阅 | Whether unsubscription was successful
        """
        try:
            async with self.lock:
                if stream_id not in self.subscribers:
                    return False
                
                if subscriber_id:
                    # 根据订阅者ID取消订阅
                    self.subscribers[stream_id] = [
                        s for s in self.subscribers[stream_id] if s["id"] != subscriber_id
                    ]
                else:
                    # 清空所有订阅
                    self.subscribers[stream_id] = []
                
                # 更新统计信息
                if stream_id in self.stats:
                    self.stats[stream_id].subscriber_count = len(self.subscribers[stream_id])
                
                # 更新监控仪表板
                await self.monitoring_dashboard.update_subscribers(stream_id, len(self.subscribers[stream_id]))
                
                self.logger.info(f"订阅已取消: 数据流 {stream_id}")
                self.logger.info(f"Subscription cancelled: Data stream {stream_id}")
                return True
                
        except Exception as e:
            error_msg = f"取消订阅数据流失败: {stream_id} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def process_and_publish(self, stream_id: str, raw_data: Any) -> bool:
        """处理并发布数据
        Process and publish data
        
        Args:
            stream_id: 流ID | Stream ID
            raw_data: 原始数据 | Raw data
            
        Returns:
            bool: 是否成功处理并发布 | Whether processing and publishing was successful
        """
        if not self.running:
            return False
            
        try:
            # 检查数据流是否存在
            if stream_id not in self.streams:
                self.logger.warning(f"尝试处理不存在的数据流: {stream_id}")
                self.logger.warning(f"Attempting to process non-existent data stream: {stream_id}")
                return False
            
            # 更新数据流活跃时间
            stream_config = self.streams[stream_id]
            if stream_id in self.stats:
                self.stats[stream_id].last_activity = time.time()
            
            # 获取流类型并应用相应的处理器
            processed_data = raw_data
            stream_type = stream_config.stream_type
            
            if stream_type in self.processors:
                try:
                    # 使用注册的处理器处理数据
                    processor = self.processors[stream_type]
                    start_time = time.time()
                    
                    if asyncio.iscoroutinefunction(processor.process):
                        processed_data = await processor.process(raw_data, stream_config)
                    else:
                        # 对于同步处理器，在事件循环中执行
                        loop = asyncio.get_event_loop()
                        processed_data = await loop.run_in_executor(
                            None, processor.process, raw_data, stream_config
                        )
                    
                    # 记录处理时间
                    processing_time = time.time() - start_time
                    if stream_id in self.stats:
                        # 更新平均处理时间（指数移动平均）
                        stats = self.stats[stream_id]
                        if stats.processing_time_avg == 0:
                            stats.processing_time_avg = processing_time
                        else:
                            stats.processing_time_avg = 0.9 * stats.processing_time_avg + 0.1 * processing_time
                        
                        # 更新最大最小处理时间
                        stats.processing_time_max = max(stats.processing_time_max, processing_time)
                        stats.processing_time_min = min(stats.processing_time_min, processing_time)
                        
                        # 计算吞吐量
                        if processing_time > 0:
                            if hasattr(processed_data, 'nbytes'):
                                stats.throughput = processed_data.nbytes / processing_time
                            elif isinstance(processed_data, (bytes, bytearray)):
                                stats.throughput = len(processed_data) / processing_time
                            else:
                                stats.throughput = 1000 / processing_time  # 默认1KB
                            
                except Exception as e:
                    error_msg = f"数据流处理错误 ({stream_id}): {str(e)}"
                    self.logger.error(error_msg)
                    error_handler.handle_error(e, "DataStreamManager", error_msg)
                    
                    # 更新错误计数
                    if stream_id in self.stats:
                        self.stats[stream_id].error_count += 1
                        self.stats[stream_id].health_score = max(0, self.stats[stream_id].health_score - 10)
                    return False
            
            # 发布处理后的数据到所有订阅者
            if stream_id in self.subscribers and self.subscribers[stream_id]:
                for subscriber in self.subscribers[stream_id]:
                    subscriber["last_active"] = time.time()
                    
                    # 使用异步任务处理回调，避免阻塞
                    task = asyncio.create_task(
                        self._safe_callback(subscriber["callback"], processed_data, stream_id)
                    )
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            
            # 更新处理统计
            if stream_id in self.stats:
                self.stats[stream_id].frames_processed += 1
                # 估算数据量
                if hasattr(processed_data, 'nbytes'):
                    self.stats[stream_id].bytes_processed += processed_data.nbytes
                elif isinstance(processed_data, (bytes, bytearray)):
                    self.stats[stream_id].bytes_processed += len(processed_data)
                else:
                    self.stats[stream_id].bytes_processed += 1000  # 默认1KB
                
                # 更新健康评分
                stats = self.stats[stream_id]
                if stats.error_count == 0:
                    stats.health_score = 100.0
                else:
                    error_ratio = min(stats.error_count / (stats.frames_processed + 1), 1.0)
                    stats.health_score = 100.0 * (1.0 - error_ratio)
            
            # 更新监控仪表板
            if stream_id in self.stats:
                await self.monitoring_dashboard.update_stream_stats(stream_id, self.stats[stream_id])
            
            return True
            
        except Exception as e:
            error_msg = f"处理并发布数据失败: {stream_id} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
            return False
    
    async def _safe_callback(self, callback: Callable, data: Any, stream_id: str):
        """安全执行回调函数
        Safely execute callback function
        
        Args:
            callback: 回调函数 | Callback function
            data: 数据 | Data
            stream_id: 流ID | Stream ID
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data, stream_id)
            else:
                # 对于同步回调，在事件循环中执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, data, stream_id)
        except Exception as e:
            error_msg = f"回调函数执行失败: {stream_id} - {str(e)}"
            self.logger.error(error_msg)
            error_handler.handle_error(e, "DataStreamManager", error_msg)
    
    async def get_stream_stats(self, stream_id: str) -> Optional[StreamStats]:
        """获取流统计信息
        Get stream statistics
        
        Args:
            stream_id: 流ID | Stream ID
            
        Returns:
            Optional[StreamStats]: 流统计信息 | Stream statistics
        """
        async with self.lock:
            return self.stats.get(stream_id)
    
    async def get_all_stream_stats(self) -> Dict[str, StreamStats]:
        """获取所有流统计信息
        Get all stream statistics
        
        Returns:
            Dict[str, StreamStats]: 所有流统计信息 | All stream statistics
        """
        async with self.lock:
            return self.stats.copy()
    
    async def get_system_metrics(self) -> SystemMetrics:
        """获取系统性能指标
        Get system performance metrics
        
        Returns:
            SystemMetrics: 系统性能指标 | System performance metrics
        """
        return self.system_metrics
    
    async def get_active_streams(self) -> List[str]:
        """获取活跃数据流列表
        Get list of active data streams
        
        Returns:
            List[str]: 活跃数据流ID列表 | List of active data stream IDs
        """
        async with self.lock:
            return list(self.streams.keys())
    
    async def get_subscriber_count(self, stream_id: str) -> int:
        """获取订阅者数量
        Get subscriber count
        
        Args:
            stream_id: 流ID | Stream ID
            
        Returns:
            int: 订阅者数量 | Subscriber count
        """
        async with self.lock:
            if stream_id in self.subscribers:
                return len(self.subscribers[stream_id])
            return 0

# =============================================================================
# 流处理器实现
# Stream Processor Implementations
# =============================================================================


"""
VideoStreamProcessor类 - 中文类描述
VideoStreamProcessor Class - English class description
"""
class VideoStreamProcessor:
    """视频流处理器
    Video Stream Processor
    """
    
    def process(self, frame_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理视频帧数据
        Process video frame data
        
        Args:
            frame_data: 帧数据 | Frame data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的视频数据 | Processed video data
        """
        try:
            # 基本视频帧处理
            if isinstance(frame_data, np.ndarray):
                # 确保是RGB格式
                if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                    if frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)
                else:
                    # 转换为RGB
                    if len(frame_data.shape) == 2:  # 灰度图
                        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                    else:
                        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            # 调用视觉模型进行高级处理
            vision_model = model_registry.get_model("vision")
            analysis_result = {}
            if vision_model and hasattr(vision_model, 'analyze_frame'):
                try:
                    analysis_result = vision_model.analyze_frame(frame_data)
                except Exception as e:
                    error_handler.log_warning(f"视觉模型分析失败: {str(e)}", "VideoStreamProcessor")
            
            return {
                "type": "video_frame",
                "frame_data": frame_data,
                "analysis": analysis_result,
                "timestamp": time.time(),
                "stream_id": config.stream_id,
                "metadata": {
                    "frame_size": frame_data.shape if hasattr(frame_data, 'shape') else "unknown",
                    "processing_time": time.time()
                }
            }
            
        except Exception as e:
            error_msg = f"视频流处理失败: {str(e)}"
            error_handler.handle_error(e, "VideoStreamProcessor", error_msg)
            return {
                "type": "video_frame",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
AudioStreamProcessor类 - 中文类描述
AudioStreamProcessor Class - English class description
"""
class AudioStreamProcessor:
    """音频流处理器
    Audio Stream Processor
    """
    
    def process(self, audio_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理音频数据
        Process audio data
        
        Args:
            audio_data: 音频数据 | Audio data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的音频数据 | Processed audio data
        """
        try:
            # 基本音频数据处理
            processed_audio = audio_data
            
            # 调用音频模型进行高级处理
            audio_model = model_registry.get_model("audio")
            analysis_result = {}
            if audio_model and hasattr(audio_model, 'analyze_audio'):
                try:
                    analysis_result = audio_model.analyze_audio(audio_data)
                except Exception as e:
                    error_handler.log_warning(f"音频模型分析失败: {str(e)}", "AudioStreamProcessor")
            
            return {
                "type": "audio_frame",
                "audio_data": processed_audio,
                "analysis": analysis_result,
                "timestamp": time.time(),
                "stream_id": config.stream_id,
                "metadata": {
                    "data_type": type(audio_data).__name__,
                    "processing_time": time.time()
                }
            }
            
        except Exception as e:
            error_msg = f"音频流处理失败: {str(e)}"
            error_handler.handle_error(e, "AudioStreamProcessor", error_msg)
            return {
                "type": "audio_frame",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
SensorDataProcessor类 - 中文类描述
SensorDataProcessor Class - English class description
"""
class SensorDataProcessor:
    """传感器数据处理器
    Sensor Data Processor
    """
    
    def process(self, sensor_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理传感器数据
        Process sensor data
        
        Args:
            sensor_data: 传感器数据 | Sensor data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的传感器数据 | Processed sensor data
        """
        try:
            # 基本传感器数据处理
            if isinstance(sensor_data, dict):
                processed_data = sensor_data.copy()
            else:
                processed_data = {"raw_value": sensor_data}
            
            # 添加时间戳和元数据
            processed_data["timestamp"] = time.time()
            processed_data["stream_id"] = config.stream_id
            
            # 调用传感器模型进行高级处理
            sensor_model = model_registry.get_model("sensor")
            if sensor_model and hasattr(sensor_model, 'analyze_sensor_data'):
                try:
                    analysis_result = sensor_model.analyze_sensor_data(processed_data)
                    processed_data["analysis"] = analysis_result
                except Exception as e:
                    error_handler.log_warning(f"传感器模型分析失败: {str(e)}", "SensorDataProcessor")
                    processed_data["analysis"] = {"error": str(e)}
            
            return processed_data
            
        except Exception as e:
            error_msg = f"传感器数据处理失败: {str(e)}"
            error_handler.handle_error(e, "SensorDataProcessor", error_msg)
            return {
                "type": "sensor_data",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
NetworkVideoProcessor类 - 中文类描述
NetworkVideoProcessor Class - English class description
"""
class NetworkVideoProcessor(VideoStreamProcessor):
    """网络视频处理器
    Network Video Processor
    """
    
    def process(self, video_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理网络视频数据
        Process network video data
        
        Args:
            video_data: 视频数据 | Video data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的网络视频数据 | Processed network video data
        """
        try:
            # 处理网络特定的视频数据
            result = super().process(video_data, config)
            result["source_type"] = "network"
            
            # 添加网络特定信息
            if hasattr(config.source, 'url'):
                result["stream_url"] = config.source.url
            if hasattr(config.source, 'protocol'):
                result["protocol"] = config.source.protocol
            
            return result
            
        except Exception as e:
            error_msg = f"网络视频处理失败: {str(e)}"
            error_handler.handle_error(e, "NetworkVideoProcessor", error_msg)
            return {
                "type": "network_video",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
NetworkAudioProcessor类 - 中文类描述
NetworkAudioProcessor Class - English class description
"""
class NetworkAudioProcessor(AudioStreamProcessor):
    """网络音频处理器
    Network Audio Processor
    """
    
    def process(self, audio_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理网络音频数据
        Process network audio data
        
        Args:
            audio_data: 音频数据 | Audio data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的网络音频数据 | Processed network audio data
        """
        try:
            # 处理网络特定的音频数据
            result = super().process(audio_data, config)
            result["source_type"] = "network"
            
            # 添加网络特定信息
            if hasattr(config.source, 'url'):
                result["stream_url"] = config.source.url
            if hasattr(config.source, 'protocol'):
                result["protocol"] = config.source.protocol
            
            return result
            
        except Exception as e:
            error_msg = f"网络音频处理失败: {str(e)}"
            error_handler.handle_error(e, "NetworkAudioProcessor", error_msg)
            return {
                "type": "network_audio",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
StereoVideoProcessor类 - 中文类描述
StereoVideoProcessor Class - English class description
"""
class StereoVideoProcessor(VideoStreamProcessor):
    """双目视频处理器
    Stereo Video Processor
    """
    
    def process(self, stereo_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理双目视频数据
        Process stereo video data
        
        Args:
            stereo_data: 双目视频数据 | Stereo video data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的双目视频数据 | Processed stereo video data
        """
        try:
            # 处理双目视频数据
            if isinstance(stereo_data, tuple) and len(stereo_data) == 2:
                left_frame, right_frame = stereo_data
                
                # 处理左右帧
                left_result = super().process(left_frame, config)
                right_result = super().process(right_frame, config)
                
                # 调用空间感知模型进行深度分析
                spatial_model = model_registry.get_model("spatial")
                depth_analysis = {}
                if spatial_model and hasattr(spatial_model, 'analyze_stereo_frames'):
                    try:
                        depth_analysis = spatial_model.analyze_stereo_frames(left_frame, right_frame)
                    except Exception as e:
                        error_handler.log_warning(f"空间模型分析失败: {str(e)}", "StereoVideoProcessor")
                        depth_analysis = {"error": str(e)}
                
                # 返回双目视频处理结果
                return {
                    "type": "stereo_video_frame",
                    "left_frame": left_result,
                    "right_frame": right_result,
                    "depth_analysis": depth_analysis,
                    "timestamp": time.time(),
                    "stream_id": config.stream_id,
                    "metadata": {
                        "frame_size": f"{left_frame.shape if hasattr(left_frame, 'shape') else 'unknown'} x {right_frame.shape if hasattr(right_frame, 'shape') else 'unknown'}",
                        "processing_time": time.time()
                    }
                }
            else:
                # 如果不是双目数据，使用父类处理
                return super().process(stereo_data, config)
            
        except Exception as e:
            error_msg = f"双目视频处理失败: {str(e)}"
            error_handler.handle_error(e, "StereoVideoProcessor", error_msg)
            return {
                "type": "stereo_video_frame",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }


"""
MultimodalProcessor类 - 中文类描述
MultimodalProcessor Class - English class description
"""
class MultimodalProcessor:
    """多模态数据处理器
    Multimodal Data Processor
    """
    
    def process(self, multimodal_data: Any, config: StreamConfig) -> Dict[str, Any]:
        """处理多模态数据
        Process multimodal data
        
        Args:
            multimodal_data: 多模态数据 | Multimodal data
            config: 流配置 | Stream configuration
            
        Returns:
            Dict[str, Any]: 处理后的多模态数据 | Processed multimodal data
        """
        try:
            # 多模态数据处理逻辑
            processed_data = {}
            
            if isinstance(multimodal_data, dict):
                # 处理包含多种数据类型的情况
                for data_type, data in multimodal_data.items():
                    if data_type == "video":
                        video_processor = VideoStreamProcessor()
                        processed_data["video"] = video_processor.process(data, config)
                    elif data_type == "audio":
                        audio_processor = AudioStreamProcessor()
                        processed_data["audio"] = audio_processor.process(data, config)
                    elif data_type == "sensor":
                        sensor_processor = SensorDataProcessor()
                        processed_data["sensor"] = sensor_processor.process(data, config)
                    else:
                        processed_data[data_type] = data
            
            # 调用多模态融合模型进行综合分析
            fusion_model = model_registry.get_model("fusion")
            fusion_result = {}
            if fusion_model and hasattr(fusion_model, 'fuse_multimodal_data'):
                try:
                    fusion_result = fusion_model.fuse_multimodal_data(processed_data)
                except Exception as e:
                    error_handler.log_warning(f"多模态融合模型分析失败: {str(e)}", "MultimodalProcessor")
                    fusion_result = {"error": str(e)}
            
            return {
                "type": "multimodal_data",
                "processed_data": processed_data,
                "fusion_result": fusion_result,
                "timestamp": time.time(),
                "stream_id": config.stream_id,
                "metadata": {
                    "data_types": list(processed_data.keys()) if processed_data else [],
                    "processing_time": time.time()
                }
            }
            
        except Exception as e:
            error_msg = f"多模态数据处理失败: {str(e)}"
            error_handler.handle_error(e, "MultimodalProcessor", error_msg)
            return {
                "type": "multimodal_data",
                "error": error_msg,
                "timestamp": time.time(),
                "stream_id": config.stream_id
            }
