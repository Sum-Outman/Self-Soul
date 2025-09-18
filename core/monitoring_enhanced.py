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
增强监控系统 - 扩展系统监控功能
Enhanced Monitoring System - Extended system monitoring capabilities

提供高级监控功能，包括：
- 实时性能指标收集
- 模型协作监控
- 情感状态跟踪
- 任务执行分析
- 系统健康检查

Provides advanced monitoring features including:
- Real-time performance metrics collection
- Model collaboration monitoring
- Emotion state tracking
- Task execution analysis
- System health checks
"""
"""
monitoring_enhanced.py - 中文描述
monitoring_enhanced.py - English description

版权所有 (c) 2025 AGI Brain Team
Licensed under the Apache License, Version 2.0
"""
import logging
import time
import psutil
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from collections import deque

from core.system_monitor import SystemMonitor


"""
EnhancedSystemMonitor类 - 中文类描述
EnhancedSystemMonitor Class - English class description
"""
class EnhancedSystemMonitor(SystemMonitor):
    """增强型系统监控器 / Enhanced System Monitor
    
    扩展功能：
    1. 模型性能监控
    2. 任务执行跟踪
    3. 情感状态分析
    4. 协作效率统计
    5. 实时数据流监控
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        """初始化增强监控器 / Initialize enhanced monitor"""
        super().__init__(config_path)
        self.logger = logging.getLogger("EnhancedSystemMonitor")
        
        # 扩展指标 / Extended metrics
        self.model_metrics = {}
        self.task_metrics = deque(maxlen=1000)  # 任务指标历史 / Task metrics history
        self.emotion_metrics = deque(maxlen=500)  # 情感指标历史 / Emotion metrics history
        self.collaboration_metrics = {}
        
        # 实时数据流 / Real-time data streams
        self.data_streams = {}
        self.stream_update_interval = 1.0  # 秒 / seconds
        
        # 启动监控线程 / Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._enhanced_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("增强监控系统已启动 | Enhanced monitoring system started")
    
    def _enhanced_monitoring_loop(self):
        """增强监控循环 / Enhanced monitoring loop"""
        while True:
            try:
                # 收集扩展指标 / Collect extended metrics
                self._collect_model_metrics()
                self._collect_collaboration_metrics()
                self._update_data_streams()
                
                time.sleep(self.stream_update_interval)
            except Exception as e:
                self.logger.error(f"增强监控循环错误: {str(e)} | Enhanced monitoring loop error: {str(e)}")
                time.sleep(5)
    
    def _collect_model_metrics(self):
        """收集模型性能指标 / Collect model performance metrics"""
        # 这里应该从模型注册表获取模型状态
        # This should get model status from model registry
        try:
            # 模拟模型指标收集 / Simulate model metrics collection
            model_types = ["language", "audio", "vision", "video", "spatial", 
                          "sensor", "computer", "motion", "knowledge", "programming"]
            
            for model_type in model_types:
                if model_type not in self.model_metrics:
                    self.model_metrics[model_type] = {
                        "cpu_usage": 0,
                        "memory_usage": 0,
                        "throughput": 0,
                        "error_rate": 0,
                        "last_update": datetime.now()
                    }
                else:
                    # 更新现有指标 / Update existing metrics
                    self.model_metrics[model_type].update({
                        "cpu_usage": psutil.cpu_percent() / 10,  # 模拟数据 / simulated data
                        "memory_usage": psutil.virtual_memory().percent / 2,
                        "last_update": datetime.now()
                    })
        except Exception as e:
            self.logger.error(f"模型指标收集错误: {str(e)} | Model metrics collection error: {str(e)}")
    
    def _collect_collaboration_metrics(self):
        """收集协作指标 / Collect collaboration metrics"""
        try:
            # 模拟协作数据 / Simulate collaboration data
            collaboration_types = ["model_to_model", "task_based", "data_sharing", "knowledge_transfer"]
            
            for collab_type in collaboration_types:
                if collab_type not in self.collaboration_metrics:
                    self.collaboration_metrics[collab_type] = {
                        "success_rate": 95.0,  # 成功率 / success rate
                        "avg_latency": 0.5,    # 平均延迟秒 / average latency in seconds
                        "throughput": 10.0,    # 事务/秒 / transactions per second
                        "last_updated": datetime.now()
                    }
        except Exception as e:
            self.logger.error(f"协作指标收集错误: {str(e)} | Collaboration metrics collection error: {str(e)}")
    
    def _update_data_streams(self):
        """更新数据流 / Update data streams"""
        try:
            stream_types = ["performance", "tasks", "emotions", "collaboration"]
            
            for stream_type in stream_types:
                if stream_type not in self.data_streams:
                    self.data_streams[stream_type] = {
                        "subscribers": set(),
                        "last_data": None,
                        "update_count": 0
                    }
                
                # 生成流数据 / Generate stream data
                stream_data = self._generate_stream_data(stream_type)
                self.data_streams[stream_type]["last_data"] = stream_data
                self.data_streams[stream_type]["update_count"] += 1
                
                # 通知订阅者（在实际实现中） / Notify subscribers (in actual implementation)
                # self._notify_subscribers(stream_type, stream_data)
                
        except Exception as e:
            self.logger.error(f"数据流更新错误: {str(e)} | Data stream update error: {str(e)}")
    
    def _generate_stream_data(self, stream_type: str) -> Dict[str, Any]:
        """生成流数据 / Generate stream data"""
        timestamp = datetime.now().isoformat()
        
        if stream_type == "performance":
            return {
                "type": "performance",
                "timestamp": timestamp,
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict()
            }
        elif stream_type == "tasks":
            return {
                "type": "tasks",
                "timestamp": timestamp,
                "active_tasks": len([t for t in self.task_metrics if t.get("status") == "active"]),
                "completed_tasks": len([t for t in self.task_metrics if t.get("status") == "completed"]),
                "failed_tasks": len([t for t in self.task_metrics if t.get("status") == "failed"])
            }
        elif stream_type == "emotions":
            return {
                "type": "emotions",
                "timestamp": timestamp,
                "current_emotion": "neutral",  # 应从情感分析器获取 / should get from emotion analyzer
                "intensity": 0.5,
                "emotion_history": list(self.emotion_metrics)[-10:] if self.emotion_metrics else []
            }
        elif stream_type == "collaboration":
            return {
                "type": "collaboration",
                "timestamp": timestamp,
                "model_collaborations": self.collaboration_metrics,
                "data_exchanges": 0  # 应从实际数据获取 / should get from actual data
            }
        else:
            return {"type": stream_type, "timestamp": timestamp, "data": "unknown_stream_type"}
    
    def add_task_metric(self, task_info: Dict[str, Any]):
        """添加任务指标 / Add task metric"""
        task_info["timestamp"] = datetime.now().isoformat()
        self.task_metrics.append(task_info)
    
    def add_emotion_metric(self, emotion_info: Dict[str, Any]):
        """添加情感指标 / Add emotion metric"""
        emotion_info["timestamp"] = datetime.now().isoformat()
        self.emotion_metrics.append(emotion_info)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """获取增强指标 / Get enhanced metrics"""
        base_metrics = super().collect_metrics()
        
        enhanced_metrics = {
            "base_metrics": base_metrics,
            "model_metrics": self.model_metrics,
            "task_metrics": list(self.task_metrics)[-10:],  # 最近10个任务 / last 10 tasks
            "emotion_metrics": list(self.emotion_metrics)[-10:],  # 最近10个情感 / last 10 emotions
            "collaboration_metrics": self.collaboration_metrics,
            "data_streams": {k: v["update_count"] for k, v in self.data_streams.items()}
        }
        
        return enhanced_metrics
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标（兼容接口） / Get system metrics (compatible interface)"""
        return self.get_enhanced_metrics()
    
    def subscribe_to_stream(self, stream_type: str, callback: callable):
        """订阅数据流 / Subscribe to data stream"""
        if stream_type not in self.data_streams:
            self.data_streams[stream_type] = {
                "subscribers": set(),
                "last_data": None,
                "update_count": 0
            }
        
        self.data_streams[stream_type]["subscribers"].add(callback)
        self.logger.info(f"已订阅数据流: {stream_type} | Subscribed to data stream: {stream_type}")
    
    def unsubscribe_from_stream(self, stream_type: str, callback: callable):
        """取消订阅数据流 / Unsubscribe from data stream"""
        if stream_type in self.data_streams and callback in self.data_streams[stream_type]["subscribers"]:
            self.data_streams[stream_type]["subscribers"].remove(callback)
            self.logger.info(f"已取消订阅数据流: {stream_type} | Unsubscribed from data stream: {stream_type}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取系统性能统计
        Get system performance statistics
        
        Returns:
            性能统计数据字典 | Performance statistics dictionary
        """
        try:
            # 获取基础系统指标
            base_metrics = super().collect_metrics()
            
            # 获取增强指标
            enhanced_metrics = self.get_enhanced_metrics()
            
            # 构建性能统计
            performance_stats = {
                "system": {
                    "uptime": base_metrics.get("uptime", 0),
                    "cpu_usage": base_metrics.get("cpu_usage", 0),
                    "memory_usage": base_metrics.get("memory_usage", 0),
                    "disk_usage": base_metrics.get("disk_usage", 0),
                    "network_io": base_metrics.get("network_io", {}),
                    "process_count": base_metrics.get("process_count", 0)
                },
                "models": {
                    "total_models": len(enhanced_metrics.get("model_metrics", {})),
                    "active_models": sum(1 for m in enhanced_metrics.get("model_metrics", {}).values() 
                                      if m.get("cpu_usage", 0) > 0),
                    "model_types": list(enhanced_metrics.get("model_metrics", {}).keys())
                },
                "tasks": {
                    "total_tasks": len(enhanced_metrics.get("task_metrics", [])),
                    "active_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "active"]),
                    "completed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                         if t.get("status") == "completed"]),
                    "failed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "failed"])
                },
                "collaboration": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": enhanced_metrics.get("data_streams", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            return performance_stats
            
        except Exception as e:
            self.logger.error(f"获取性能统计失败: {str(e)} | Failed to get performance stats: {str(e)}")
            # 返回基本的性能统计作为后备
            return {
                "system": {
                    "uptime": 0,
                    "cpu_usage": 0,
                    "memory_usage": 0,
                    "disk_usage": 0,
                    "network_io": {},
                    "process_count": 0
                },
                "models": {
                    "total_models": 0,
                    "active_models": 0,
                    "model_types": []
                },
                "tasks": {
                    "total_tasks": 0,
                    "active_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0
                },
                "collaboration": {},
                "data_streams": {},
                "timestamp": datetime.now().isoformat(),
                "error": f"获取性能统计失败: {str(e)}"
            }

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的日志记录
        Get recent log records
        
        Args:
            limit: 返回的日志数量限制 | Limit for number of logs to return
            
        Returns:
            最近的日志记录列表 | List of recent log records
        """
        try:
            # 从日志文件读取最近的日志
            log_file = self.config.get("log_file", "logs/system_monitor.log")
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-limit:]  # 读取最后limit行
                
                for line in lines:
                    try:
                        # 解析日志行格式 (假设格式: [时间戳] [级别] 消息)
                        if '] [' in line and line.startswith('['):
                            timestamp_end = line.find(']')
                            level_start = line.find('[', timestamp_end + 1)
                            level_end = line.find(']', level_start + 1)
                            
                            if timestamp_end != -1 and level_start != -1 and level_end != -1:
                                timestamp = line[1:timestamp_end].strip()
                                level = line[level_start+1:level_end].strip()
                                message = line[level_end+1:].strip()
                                
                                logs.append({
                                    "timestamp": timestamp,
                                    "level": level.lower(),
                                    "message": message
                                })
                        else:
                            # 简单格式处理
                            logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "level": "info",
                                "message": line.strip()
                            })
                    except:
                        # 如果解析失败，添加原始行
                        logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "level": "info", 
                            "message": line.strip()
                        })
            
            # 如果日志文件不存在或为空，返回模拟数据
            if not logs:
                levels = ["info", "warning", "error", "debug"]
                for i in range(min(limit, 10)):
                    logs.append({
                        "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                        "level": levels[i % len(levels)],
                        "message": f"示例日志消息 {i+1} - 系统运行正常"
                    })
            
            # 按时间倒序排序（最新的在前）
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            return logs[:limit]
            
        except Exception as e:
            self.logger.error(f"获取最近日志失败: {str(e)} | Failed to get recent logs: {str(e)}")
            # 返回模拟日志作为后备
            return [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "error",
                    "message": f"无法读取日志文件: {str(e)}"
                }
            ]

    def get_realtime_monitoring(self) -> Dict[str, Any]:
        """获取实时监控数据
        Get real-time monitoring data
        
        Returns:
            实时监控数据字典 | Real-time monitoring data dictionary
        """
        try:
            # 获取基础系统指标
            base_metrics = super().collect_metrics()
            
            # 获取增强指标
            enhanced_metrics = self.get_enhanced_metrics()
            
            # 获取性能统计
            performance_stats = self.get_performance_stats()
            
            # 获取最近日志
            recent_logs = self.get_recent_logs(20)
            
            # 构建实时监控数据
            realtime_data = {
                "system": {
                    "uptime": base_metrics.get("uptime", 0),
                    "cpu_usage": base_metrics.get("cpu_usage", 0),
                    "memory_usage": base_metrics.get("memory_usage", 0),
                    "disk_usage": base_metrics.get("disk_usage", 0),
                    "network_io": base_metrics.get("network_io", {}),
                    "process_count": base_metrics.get("process_count", 0),
                    "timestamp": datetime.now().isoformat()
                },
                "models": {
                    "total_models": len(enhanced_metrics.get("model_metrics", {})),
                    "active_models": sum(1 for m in enhanced_metrics.get("model_metrics", {}).values() 
                                      if m.get("cpu_usage", 0) > 0),
                    "model_types": list(enhanced_metrics.get("model_metrics", {}).keys()),
                    "model_metrics": enhanced_metrics.get("model_metrics", {})
                },
                "tasks": {
                    "total_tasks": len(enhanced_metrics.get("task_metrics", [])),
                    "active_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "active"]),
                    "completed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                         if t.get("status") == "completed"]),
                    "failed_tasks": len([t for t in enhanced_metrics.get("task_metrics", []) 
                                      if t.get("status") == "failed"]),
                    "recent_tasks": enhanced_metrics.get("task_metrics", [])[-10:]
                },
                "collaboration": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": {
                    "total_streams": len(enhanced_metrics.get("data_streams", {})),
                    "stream_types": list(enhanced_metrics.get("data_streams", {}).keys()),
                    "update_counts": enhanced_metrics.get("data_streams", {})
                },
                "emotions": {
                    "recent_emotions": enhanced_metrics.get("emotion_metrics", []),
                    "current_emotion": "neutral"  # 应从情感分析器获取实际数据
                },
                "logs": {
                    "recent_logs": recent_logs,
                    "total_logs": len(recent_logs)
                },
                "performance": performance_stats,
                "timestamp": datetime.now().isoformat(),
                "status": "healthy"
            }
            
            return realtime_data
            
        except Exception as e:
            self.logger.error(f"获取实时监控数据失败: {str(e)} | Failed to get real-time monitoring data: {str(e)}")
            # 返回错误信息作为后备
            return {
                "error": f"获取实时监控数据失败: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }

# 别名，以便兼容旧的导入代码
# Alias for compatibility with old import code
EnhancedMonitor = EnhancedSystemMonitor


"""
MonitoringDashboard类 - 中文类描述
MonitoringDashboard Class - English class description
"""
class MonitoringDashboard:
    """实时监控仪表板
    Real-time Monitoring Dashboard
    
    功能：提供实时数据流监控仪表板功能，支持流统计、系统指标、订阅者管理等
    Function: Provides real-time data stream monitoring dashboard functionality,
              supports stream statistics, system metrics, subscriber management
    """
    
    def __init__(self):
        """初始化监控仪表板
        Initialize monitoring dashboard
        """
        self.logger = logging.getLogger("MonitoringDashboard")
        self.streams = {}  # 流监控数据 / Stream monitoring data
        self.system_metrics = {}  # 系统指标 / System metrics
        self.subscriber_counts = {}  # 订阅者计数 / Subscriber counts
        
        self.logger.info("监控仪表板已初始化 | Monitoring dashboard initialized")
    
    async def update_stream_stats(self, stream_id: str, stats: Any) -> None:
        """更新流统计信息
        Update stream statistics
        
        Args:
            stream_id: 流ID | Stream ID
            stats: 统计信息对象 | Statistics object
        """
        try:
            if stream_id not in self.streams:
                self.streams[stream_id] = {}
            
            # 更新流统计信息
            self.streams[stream_id]["stats"] = {
                "frames_processed": getattr(stats, "frames_processed", 0),
                "bytes_processed": getattr(stats, "bytes_processed", 0),
                "error_count": getattr(stats, "error_count", 0),
                "subscriber_count": getattr(stats, "subscriber_count", 0),
                "processing_time_avg": getattr(stats, "processing_time_avg", 0.0),
                "throughput": getattr(stats, "throughput", 0.0),
                "health_score": getattr(stats, "health_score", 100.0),
                "last_update": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"更新流统计信息失败: {stream_id} - {str(e)}")
            self.logger.error(f"Failed to update stream stats: {stream_id} - {str(e)}")
    
    async def update_metrics(self, metrics: Any) -> None:
        """更新系统指标
        Update system metrics
        
        Args:
            metrics: 系统指标对象 | System metrics object
        """
        try:
            self.system_metrics = {
                "cpu_usage": getattr(metrics, "cpu_usage", 0.0),
                "memory_usage": getattr(metrics, "memory_usage", 0.0),
                "active_streams": getattr(metrics, "active_streams", 0),
                "total_subscribers": getattr(metrics, "total_subscribers", 0),
                "total_throughput": getattr(metrics, "total_throughput", 0.0),
                "system_health": getattr(metrics, "system_health", 100.0),
                "last_update": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"更新系统指标失败: {str(e)}")
            self.logger.error(f"Failed to update system metrics: {str(e)}")
    
    async def add_stream(self, stream_id: str, stream_type: str) -> None:
        """添加数据流到监控
        Add data stream to monitoring
        
        Args:
            stream_id: 流ID | Stream ID
            stream_type: 流类型 | Stream type
        """
        try:
            if stream_id not in self.streams:
                self.streams[stream_id] = {
                    "stream_type": stream_type,
                    "created_time": time.time(),
                    "stats": {},
                    "subscribers": 0
                }
                self.logger.info(f"已添加流到监控: {stream_id} ({stream_type})")
                self.logger.info(f"Stream added to monitoring: {stream_id} ({stream_type})")
                
        except Exception as e:
            self.logger.error(f"添加流到监控失败: {stream_id} - {str(e)}")
            self.logger.error(f"Failed to add stream to monitoring: {stream_id} - {str(e)}")
    
    async def remove_stream(self, stream_id: str) -> None:
        """从监控中移除数据流
        Remove data stream from monitoring
        
        Args:
            stream_id: 流ID | Stream ID
        """
        try:
            if stream_id in self.streams:
                del self.streams[stream_id]
                self.logger.info(f"已从监控中移除流: {stream_id}")
                self.logger.info(f"Stream removed from monitoring: {stream_id}")
                
        except Exception as e:
            self.logger.error(f"移除流失败: {stream_id} - {str(e)}")
            self.logger.error(f"Failed to remove stream: {stream_id} - {str(e)}")
    
    async def update_subscribers(self, stream_id: str, count: int) -> None:
        """更新订阅者数量
        Update subscriber count
        
        Args:
            stream_id: 流ID | Stream ID
            count: 订阅者数量 | Subscriber count
        """
        try:
            if stream_id in self.streams:
                self.streams[stream_id]["subscribers"] = count
                self.subscriber_counts[stream_id] = count
                
        except Exception as e:
            self.logger.error(f"更新订阅者数量失败: {stream_id} - {str(e)}")
            self.logger.error(f"Failed to update subscriber count: {stream_id} - {str(e)}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据
        Get dashboard data
        
        Returns:
            Dict[str, Any]: 仪表板数据 | Dashboard data
        """
        return {
            "streams": self.streams,
            "system_metrics": self.system_metrics,
            "total_streams": len(self.streams),
            "total_subscribers": sum(self.subscriber_counts.values()) if self.subscriber_counts else 0,
            "last_updated": time.time()
        }
