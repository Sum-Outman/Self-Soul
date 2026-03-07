"""
WebSocket通信模块

提供实时WebSocket通信功能，支持认知状态流式传输和实时交互。
"""

import asyncio
import json
import logging
from typing import Dict, Any, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """客户端信息"""
    websocket: WebSocket
    client_id: str
    connected_at: float
    last_active: float
    subscriptions: Set[str] = field(default_factory=set)


class WebSocketManager:
    """WebSocket管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, ClientInfo] = {}
        self.lock = asyncio.Lock()
        
        # 消息类型处理器
        self.message_handlers = {
            'cognitive_request': self._handle_cognitive_request,
            'get_diagnostics': self._handle_get_diagnostics,
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe,
            'ping': self._handle_ping
        }
        
        logger.info("WebSocket管理器已初始化")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """处理客户端连接"""
        await websocket.accept()
        
        client_info = ClientInfo(
            websocket=websocket,
            client_id=client_id,
            connected_at=time.time(),
            last_active=time.time()
        )
        
        async with self.lock:
            self.active_connections[client_id] = client_info
        
        logger.info(f"客户端 {client_id} 已连接")
        
        # 发送连接确认
        await self.send_to_client(client_id, {
            'type': 'connection_established',
            'client_id': client_id,
            'timestamp': time.time()
        })
    
    async def disconnect(self, client_id: str):
        """处理客户端断开连接"""
        async with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"客户端 {client_id} 已断开连接")
    
    async def receive_message(self, client_id: str, message: Dict[str, Any]):
        """处理接收到的消息"""
        if client_id not in self.active_connections:
            logger.warning(f"未知客户端 {client_id}")
            return
        
        # 更新最后活跃时间
        self.active_connections[client_id].last_active = time.time()
        
        # 获取消息类型
        message_type = message.get('type')
        
        if not message_type:
            await self.send_error(client_id, "消息类型未指定")
            return
        
        # 查找处理器
        handler = self.message_handlers.get(message_type)
        
        if not handler:
            await self.send_error(client_id, f"未知的消息类型: {message_type}")
            return
        
        try:
            # 处理消息
            await handler(client_id, message)
        except Exception as e:
            logger.error(f"处理消息失败 {message_type}: {e}", exc_info=True)
            await self.send_error(client_id, f"处理消息失败: {str(e)}")
    
    async def _handle_cognitive_request(self, client_id: str, message: Dict[str, Any]):
        """处理认知请求"""
        from ..cognitive.architecture import UnifiedCognitiveArchitecture
        
        # 获取全局AGI实例（简化版，实际应从依赖注入获取）
        agi_instance = None
        # 这里应该从应用程序状态获取
        
        if not agi_instance:
            await self.send_error(client_id, "AGI系统未初始化")
            return
        
        # 获取请求数据
        request_data = message.get('data', {})
        
        if not request_data:
            await self.send_error(client_id, "请求数据为空")
            return
        
        # 执行认知循环
        try:
            result = await agi_instance.cognitive_cycle(request_data)
            
            # 发送响应
            await self.send_to_client(client_id, {
                'type': 'cognitive_response',
                'request_id': message.get('request_id'),
                'data': result,
                'timestamp': time.time()
            })
        except Exception as e:
            await self.send_error(client_id, f"认知处理失败: {str(e)}")
    
    async def _handle_get_diagnostics(self, client_id: str, message: Dict[str, Any]):
        """处理获取诊断信息"""
        from ..cognitive.architecture import UnifiedCognitiveArchitecture
        
        # 获取全局AGI实例
        agi_instance = None
        # 这里应该从应用程序状态获取
        
        if not agi_instance:
            await self.send_error(client_id, "AGI系统未初始化")
            return
        
        try:
            diagnostics = agi_instance.get_diagnostics()
            
            await self.send_to_client(client_id, {
                'type': 'diagnostics',
                'request_id': message.get('request_id'),
                'data': diagnostics,
                'timestamp': time.time()
            })
        except Exception as e:
            await self.send_error(client_id, f"获取诊断信息失败: {str(e)}")
    
    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]):
        """处理订阅请求"""
        channels = message.get('channels', [])
        
        if not isinstance(channels, list):
            await self.send_error(client_id, "channels应为列表")
            return
        
        client_info = self.active_connections.get(client_id)
        if not client_info:
            return
        
        # 添加订阅
        for channel in channels:
            if channel in ['cognitive_state', 'performance_metrics', 'system_events']:
                client_info.subscriptions.add(channel)
        
        await self.send_to_client(client_id, {
            'type': 'subscription_confirmation',
            'channels': list(client_info.subscriptions),
            'timestamp': time.time()
        })
        
        logger.info(f"客户端 {client_id} 订阅了频道: {channels}")
    
    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]):
        """处理取消订阅请求"""
        channels = message.get('channels', [])
        
        client_info = self.active_connections.get(client_id)
        if not client_info:
            return
        
        # 移除订阅
        for channel in channels:
            client_info.subscriptions.discard(channel)
        
        await self.send_to_client(client_id, {
            'type': 'unsubscription_confirmation',
            'channels': list(client_info.subscriptions),
            'timestamp': time.time()
        })
        
        logger.info(f"客户端 {client_id} 取消了订阅: {channels}")
    
    async def _handle_ping(self, client_id: str, message: Dict[str, Any]):
        """处理ping请求"""
        await self.send_to_client(client_id, {
            'type': 'pong',
            'timestamp': time.time(),
            'original_timestamp': message.get('timestamp')
        })
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """发送消息到客户端"""
        if client_id not in self.active_connections:
            return
        
        try:
            await self.active_connections[client_id].websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送消息到客户端 {client_id} 失败: {e}")
            # 如果发送失败，断开连接
            await self.disconnect(client_id)
    
    async def send_error(self, client_id: str, error_message: str):
        """发送错误消息"""
        await self.send_to_client(client_id, {
            'type': 'error',
            'message': error_message,
            'timestamp': time.time()
        })
    
    async def broadcast(self, message: Dict[str, Any], channel: Optional[str] = None):
        """
        广播消息到所有客户端或特定频道的订阅者
        
        参数:
            message: 消息内容
            channel: 频道名称（如果为None，则广播到所有客户端）
        """
        async with self.lock:
            for client_id, client_info in self.active_connections.items():
                if channel is None or channel in client_info.subscriptions:
                    try:
                        await client_info.websocket.send_json(message)
                    except Exception as e:
                        logger.error(f"广播消息到客户端 {client_id} 失败: {e}")
    
    async def broadcast_cognitive_state(self, state: Dict[str, Any]):
        """广播认知状态更新"""
        await self.broadcast({
            'type': 'cognitive_state_update',
            'data': state,
            'timestamp': time.time()
        }, channel='cognitive_state')
    
    async def broadcast_performance_metrics(self, metrics: Dict[str, Any]):
        """广播性能指标"""
        await self.broadcast({
            'type': 'performance_update',
            'data': metrics,
            'timestamp': time.time()
        }, channel='performance_metrics')
    
    async def broadcast_system_event(self, event_type: str, data: Dict[str, Any]):
        """广播系统事件"""
        await self.broadcast({
            'type': 'system_event',
            'event_type': event_type,
            'data': data,
            'timestamp': time.time()
        }, channel='system_events')
    
    async def cleanup_inactive_clients(self, timeout_seconds: float = 300):
        """清理不活跃的客户端"""
        current_time = time.time()
        clients_to_remove = []
        
        async with self.lock:
            for client_id, client_info in self.active_connections.items():
                if current_time - client_info.last_active > timeout_seconds:
                    clients_to_remove.append(client_id)
        
        # 断开不活跃的客户端
        for client_id in clients_to_remove:
            logger.info(f"清理不活跃客户端: {client_id}")
            await self.disconnect(client_id)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        async with self.lock:
            total_clients = len(self.active_connections)
            
            # 计算平均连接时间
            current_time = time.time()
            total_connection_time = 0
            
            for client_info in self.active_connections.values():
                total_connection_time += current_time - client_info.connected_at
            
            avg_connection_time = total_connection_time / total_clients if total_clients > 0 else 0
            
            # 统计订阅数量
            total_subscriptions = 0
            for client_info in self.active_connections.values():
                total_subscriptions += len(client_info.subscriptions)
            
            return {
                'total_clients': total_clients,
                'avg_connection_time_seconds': avg_connection_time,
                'total_subscriptions': total_subscriptions,
                'timestamp': current_time
            }


# 全局WebSocket管理器实例
_websocket_manager = None

def get_websocket_manager() -> WebSocketManager:
    """获取全局WebSocket管理器实例"""
    global _websocket_manager
    
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    
    return _websocket_manager