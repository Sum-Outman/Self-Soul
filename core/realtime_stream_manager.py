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
realtime_stream_manager.py - 中文描述
realtime_stream_manager.py - English description

版权所有 (c) 2025 AGI Brain Team
Licensed under the Apache License, Version 2.0
"""
import asyncio
import websockets
import json
import time
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ports_config import REALTIME_STREAM_MANAGER_PORT

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTimeStreamManager")

# Create FastAPI app
app = FastAPI(title="Real-Time Stream Manager", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RealTimeStreamManager:
    """实时数据流管理器 / Real-time Data Stream Manager
    
    增强功能：
    1. 支持模型间数据路由
    2. 数据格式标准化
    3. 错误处理机制
    4. 性能监控
    """
    
    def __init__(self, host='0.0.0.0', port=REALTIME_STREAM_MANAGER_PORT):
        """初始化实时数据流管理器 / Initialize real-time data stream manager"""
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.data_streams = {}
        self.subscriptions = {}
        self.model_routing_table = {}  # 模型路由表: {model_id: set(stream_ids)}
        self.health_check_interval = 5
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.is_running = False
        self.server = None
        
        # Performance metrics
        self.performance_metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'total_data_transferred': 0,
            'error_count': 0
        }
    
    def initialize(self) -> Dict[str, Any]:
        """初始化流管理器资源 / Initialize stream manager resources"""
        try:
            # 初始化数据结构
            self.data_streams = {
                'camera': {'callback': self._mock_camera_data, 'interval': 0.1, 'last_update': time.time()},
                'sensor': {'callback': self._mock_sensor_data, 'interval': 0.5, 'last_update': time.time()},
                'model': {'callback': self._mock_model_data, 'interval': 1.0, 'last_update': time.time()},
                'system': {'callback': self._mock_system_data, 'interval': 2.0, 'last_update': time.time()}
            }
            
            # 初始化订阅
            self.subscriptions = {
                'camera': set(),
                'sensor': set(),
                'model': set(),
                'system': set()
            }
            
            # 启动清理任务
            asyncio.create_task(self._cleanup_task())
            
            logger.info("RealTimeStreamManager initialized successfully")
            return {"success": True, "message": "RealTimeStreamManager initialized"}
        except Exception as e:
            logger.error(f"RealTimeStreamManager initialization failed: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _mock_camera_data(self):
        """模拟相机数据用于测试"""
        return {
            "type": "camera",
            "timestamp": datetime.now().isoformat(),
            "camera_id": "cam_1",
            "frame_count": int(time.time() % 1000),
            "status": "active"
        }
        
    def _mock_sensor_data(self):
        """模拟传感器数据用于测试"""
        import random
        return {
            "type": "sensor",
            "timestamp": datetime.now().isoformat(),
            "temperature": round(20 + random.uniform(-2, 2), 2),
            "humidity": round(45 + random.uniform(-5, 5), 2),
            "pressure": round(1013 + random.uniform(-5, 5), 1)
        }
        
    def _mock_model_data(self):
        """模拟模型数据用于测试"""
        import random
        return {
            "type": "model",
            "timestamp": datetime.now().isoformat(),
            "model_id": "main",
            "status": "running",
            "load": round(random.uniform(0.1, 0.9), 2)
        }
        
    def _mock_system_data(self):
        """模拟系统数据用于测试"""
        import random
        return {
            "type": "system",
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": round(random.uniform(10, 50), 1),
            "memory_usage": round(random.uniform(20, 70), 1)
        }
        
    async def _cleanup_task(self):
        """定期清理断开的连接"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            for stream_id, clients in list(self.subscriptions.items()):
                for client in list(clients):
                    if client.closed:
                        clients.remove(client)
                        self.performance_metrics['active_connections'] -= 1
            logger.info(f"Cleanup completed: {self.performance_metrics['active_connections']} active connections")
    
    async def start_server(self):
        """启动WebSocket服务器 / Start WebSocket server"""
        server = await websockets.serve(self.handle_client, self.host, self.port)
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        return server
    
    async def handle_client(self, websocket, path):
        """处理客户端连接 / Handle client connection"""
        self.connected_clients.add(websocket)
        client_ip = websocket.remote_address[0]
        logger.info(f"New client connected: {client_ip}")
        
        try:
            async for message in websocket:
                await self.process_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {client_ip}, code: {e.code}")
        except Exception as e:
            logger.error(f"Error handling client {client_ip}: {str(e)}")
        finally:
            self.connected_clients.remove(websocket)
            logger.info(f"Client removed: {client_ip}")
    
    async def process_client_message(self, websocket, message):
        """处理客户端消息 / Process client message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self.handle_subscription(websocket, data)
            elif message_type == 'unsubscribe':
                await self.handle_unsubscription(websocket, data)
            elif message_type == 'model_register':
                await self.handle_model_registration(websocket, data)
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except KeyError as e:
            logger.error(f"Missing required field in message: {str(e)}")
    
    async def handle_subscription(self, websocket, data):
        """处理订阅请求 / Handle subscription request"""
        stream_id = data.get('stream_id')
        if stream_id:
            if stream_id not in self.subscriptions:
                self.subscriptions[stream_id] = set()
            self.subscriptions[stream_id].add(websocket)
            logger.info(f"Client subscribed to stream: {stream_id}")
    
    async def handle_unsubscription(self, websocket, data):
        """处理取消订阅请求 / Handle unsubscription request"""
        stream_id = data.get('stream_id')
        if stream_id and stream_id in self.subscriptions:
            self.subscriptions[stream_id].discard(websocket)
            logger.info(f"Client unsubscribed from stream: {stream_id}")
    
    async def handle_model_registration(self, websocket, data):
        """处理模型注册请求 / Handle model registration"""
        model_id = data.get('model_id')
        stream_ids = data.get('stream_ids', [])
        
        if not model_id:
            logger.error("Model registration missing model_id")
            return
            
        self.model_routing_table[model_id] = set(stream_ids)
        logger.info(f"Model {model_id} registered for streams: {stream_ids}")
    
    async def broadcast_data(self, stream_id: str, data: Any, source_model: Optional[str] = None):
        """广播标准化数据到订阅者 / Broadcast standardized data to subscribers"""
        if stream_id in self.subscriptions:
            # 标准化数据格式 / Standardize data format
            message = json.dumps({
                'stream_id': stream_id,
                'timestamp': datetime.utcnow().isoformat() + 'Z',  # UTC时间
                'source_model': source_model,
                'data_type': type(data).__name__,
                'data': data,
                'metadata': {
                    'version': '1.0',
                    'encoding': 'json'
                }
            })
            
            tasks = []
            for client in self.subscriptions[stream_id]:
                try:
                    tasks.append(client.send(message))
                except websockets.exceptions.ConnectionClosed:
                    self.subscriptions[stream_id].discard(client)
                    
            if tasks:
                await asyncio.gather(*tasks)
    
    def register_data_stream(self, stream_id: str, data_callback: Callable, model_id: Optional[str] = None):
        """注册数据流并关联模型 / Register data stream with optional model association"""
        self.data_streams[stream_id] = {
            'callback': data_callback,
            'last_update': time.time(),
            'model_id': model_id
        }
        
        if model_id:
            if model_id not in self.model_routing_table:
                self.model_routing_table[model_id] = set()
            self.model_routing_table[model_id].add(stream_id)
    
    async def route_to_model(self, model_id: str, data: Any):
        """路由数据到指定模型 / Route data to specific model"""
        if model_id in self.model_routing_table:
            for stream_id in self.model_routing_table[model_id]:
                await self.broadcast_data(stream_id, data, source_model="Router")
        else:
            logger.warning(f"No routing found for model: {model_id}")
    
    async def start_data_streaming(self, interval: float = 1.0):
        """启动数据流推送 / Start data streaming with monitoring"""
        logger.info("Starting data streaming service")
        while True:
            start_time = time.time()
            stream_count = 0
            error_count = 0
            
            # 使用异步并发处理多个数据流 / Use async concurrency for multiple streams
            tasks = []
            for stream_id, stream_info in self.data_streams.items():
                task = self._process_stream(stream_id, stream_info)
                tasks.append(task)
            
            # 等待所有流处理完成 / Wait for all stream processing to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    logger.error(f"Stream processing error: {str(result)}")
                else:
                    stream_count += result
            
            # 性能监控 / Performance monitoring
            elapsed = time.time() - start_time
            logger.info(f"Streaming cycle completed: {stream_count} streams, "
                       f"{error_count} errors, took {elapsed:.3f} seconds")
            
            await asyncio.sleep(max(0, interval - elapsed))
    
    async def _process_stream(self, stream_id: str, stream_info: dict) -> int:
        """处理单个数据流 / Process single data stream"""
        try:
            data = stream_info['callback']()
            await self.broadcast_data(stream_id, data, stream_info.get('model_id'))
            self.data_streams[stream_id]['last_update'] = time.time()
            return 1
        except Exception as e:
            logger.error(f"Error streaming data for {stream_id}: {str(e)}")
            raise e
    
    def get_performance_metrics(self) -> dict:
        """获取性能指标 / Get performance metrics"""
        metrics = {
            'active_streams': len(self.data_streams),
            'active_subscriptions': sum(len(s) for s in self.subscriptions.values()),
            'registered_models': len(self.model_routing_table),
            'last_activity': datetime.now().isoformat(),
            **self.performance_metrics
        }
        return metrics

# Create a global instance of the stream manager
stream_manager = RealTimeStreamManager()


@app.websocket("/ws/streams/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """\WebSocket endpoint for real-time data streams"""
    await websocket.accept()
    stream_manager.performance_metrics['total_connections'] += 1
    stream_manager.performance_metrics['active_connections'] += 1
    
    try:
        # Add client to subscription if stream exists
        if stream_id in stream_manager.subscriptions:
            stream_manager.subscriptions[stream_id].add(websocket)
            await websocket.send_text(f"Subscribed to stream: {stream_id}")
            
            # Keep connection alive
            while True:
                # Wait for pong or timeout
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text("ping")
        else:
            await websocket.send_text(f"Error: Stream {stream_id} not found")
    except WebSocketDisconnect:
        stream_manager.logger.info(f"Client disconnected from stream: {stream_id}")
    except Exception as e:
        stream_manager.logger.error(f"Error in WebSocket connection: {str(e)}")
        stream_manager.performance_metrics['error_count'] += 1
    finally:
        # Remove client from all subscriptions
        for stream, clients in stream_manager.subscriptions.items():
            if websocket in clients:
                clients.remove(websocket)
        stream_manager.performance_metrics['active_connections'] -= 1


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real-Time Stream Manager",
        "version": "1.0",
        "metrics": stream_manager.get_performance_metrics(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/streams")
async def list_streams():
    """List all available streams"""
    return {
        "streams": list(stream_manager.data_streams.keys()),
        "total_streams": len(stream_manager.data_streams)
    }


@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    return stream_manager.get_performance_metrics()


@app.on_event("startup")
async def startup_event():
    """Initialize stream manager on startup"""
    await stream_manager.initialize()
    # Start the streaming loop in a background task
    asyncio.create_task(stream_manager._start_streaming_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    stream_manager.logger.info("Real-Time Stream Manager is shutting down")


if __name__ == "__main__":
    """Main entry point to start the server"""
    logger.info(f"Starting Real-Time Stream Manager on ws://{stream_manager.host}:{stream_manager.port}")
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host=stream_manager.host,
        port=stream_manager.port,
        reload=False,
        log_level="info"
    )
