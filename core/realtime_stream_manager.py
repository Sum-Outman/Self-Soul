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

版权所有 (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""
import asyncio
import websockets
import json
import time
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import logging
from core.production_security import get_production_security_manager
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ports_config import REALTIME_STREAM_MANAGER_PORT
from core.error_handling import error_handler

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTimeStreamManager")

# Create FastAPI app
app = FastAPI(
    title="Real-Time Stream Manager", 
    version="1.0",
    docs_url="/docs",
    redoc_url=None,
    swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js",
    swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css"
)

# Add CORS middleware - configure based on environment
cors_origins_str = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

# Configure CORS based on environment
environment = os.environ.get("ENVIRONMENT", "development")
if environment == "production":
    # Production: strict CORS configuration
    if not cors_origins:
        cors_origins = []  # 生产环境应明确指定允许的源
    allow_credentials = False  # 生产环境通常不需要凭证
    logger.info(f"Production CORS配置: 允许的源={cors_origins}, 允许凭证={allow_credentials}")
else:
    # Development: more permissive but still safe
    if not cors_origins or cors_origins == ["*"]:
        cors_origins = ["http://localhost:3000", "http://localhost:5173"]
    allow_credentials = False  # 开发环境默认不允许凭证，需要时可配置
    logger.info(f"Development CORS配置: 允许的源={cors_origins}, 允许凭证={allow_credentials}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Add API key authentication middleware
@app.middleware("http")
async def authenticate_request_realtime(request: Request, call_next):
    # Skip authentication for health check endpoints and documentation
    if request.url.path.endswith("/health") or request.url.path.endswith("/docs") or request.url.path.endswith("/openapi.json"):
        return await call_next(request)
    
    # Get API key from environment variable
    api_key = os.environ.get("REALTIME_STREAM_API_KEY")
    # If no API key is set, skip authentication (for development only)
    if not api_key:
        return await call_next(request)
    
    # Check API key in request headers
    request_api_key = request.headers.get("X-API-Key")
    if request_api_key and request_api_key == api_key:
        return await call_next(request)
    else:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid or missing API key"}
        )


class RealTimeStreamManager:
    """实时数据流管理器 / Real-time Data Stream Manager
    
    增强功能：
    1. 支持模型间数据路由
    2. 数据格式标准化
    3. 错误处理机制
    4. 性能监控
    """
    
    def __init__(self, host='127.0.0.1', port=REALTIME_STREAM_MANAGER_PORT):
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
        
        # Camera state cache to reduce OpenCV warnings
        self.camera_cache = {}
        self.camera_cache_ttl = 5.0  # Cache TTL in seconds
        self.last_camera_check_time = 0
        
        # Callback registry for model stream processing
        self._callbacks = {}
    
    def register_callback(self, callback_id: str, callback_function: Callable, stream_ids: Optional[List[str]] = None):
        """注册回调函数用于流处理 / Register callback function for stream processing
        
        Args:
            callback_id: 回调标识符 / Callback identifier
            callback_function: 回调函数 / Callback function
            stream_ids: 关联的流ID列表 / Associated stream IDs
        """
        self._callbacks[callback_id] = {
            'function': callback_function,
            'stream_ids': stream_ids or [],
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"Callback registered: {callback_id} for streams: {stream_ids}")
    
    def unregister_callback(self, callback_id: str):
        """注销回调函数 / Unregister callback function
        
        Args:
            callback_id: 回调标识符 / Callback identifier
        """
        if callback_id in self._callbacks:
            del self._callbacks[callback_id]
            logger.info(f"Callback unregistered: {callback_id}")
    
    async def trigger_callbacks(self, stream_id: str, data: Any):
        """触发指定流的回调函数 / Trigger callbacks for specified stream
        
        Args:
            stream_id: 流标识符 / Stream identifier
            data: 要传递的数据 / Data to pass to callbacks
        """
        triggered_count = 0
        for callback_id, callback_info in self._callbacks.items():
            # 如果回调关联了特定流，只在这些流上触发
            if not callback_info['stream_ids'] or stream_id in callback_info['stream_ids']:
                try:
                    # 异步调用回调函数
                    if asyncio.iscoroutinefunction(callback_info['function']):
                        await callback_info['function'](stream_id, data)
                    else:
                        callback_info['function'](stream_id, data)
                    triggered_count += 1
                except Exception as e:
                    logger.error(f"Error in callback {callback_id}: {str(e)}")
                    self.performance_metrics['error_count'] += 1
        
        if triggered_count > 0:
            logger.debug(f"Triggered {triggered_count} callbacks for stream: {stream_id}")
    
    async def initialize(self) -> Dict[str, Any]:
        """初始化流管理器资源 / Initialize stream manager resources"""
        try:
            # 初始化数据结构 - 使用真实数据回调函数
            self.data_streams = {
                'camera': {'callback': self._real_camera_data, 'interval': 1.0, 'last_update': time.time()},
                'sensor': {'callback': self._real_sensor_data, 'interval': 0.5, 'last_update': time.time()},
                'model': {'callback': self._real_model_data, 'interval': 1.0, 'last_update': time.time()},
                'system': {'callback': self._real_system_data, 'interval': 2.0, 'last_update': time.time()}
            }
            
            # 初始化订阅
            self.subscriptions = {
                'camera': set(),
                'sensor': set(),
                'model': set(),
                'system': set()
            }
            
            # 启动清理任务 - 直接创建任务
            asyncio.create_task(self._cleanup_task())
            
            logger.info("RealTimeStreamManager initialized successfully")
            return {"success": True, "message": "RealTimeStreamManager initialized"}
        except Exception as e:
            logger.error(f"RealTimeStreamManager initialization failed: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _real_camera_data(self):
        """真实相机数据采集（优化版，减少OpenCV警告）"""
        # 检查缓存是否有效
        current_time = time.time()
        if current_time - self.last_camera_check_time < self.camera_cache_ttl and self.camera_cache:
            # 返回缓存数据
            logger.debug(f"Returning cached camera data (cache TTL: {self.camera_cache_ttl}s)")
            return self.camera_cache
        
        try:
            import cv2
            camera_data = []
            active_cameras = 0
            
            # 优化：减少检测的摄像头数量，只检查前2个摄像头
            max_cameras_to_check = 2
            
            for camera_index in range(max_cameras_to_check):
                try:
                    # 尝试不同的后端来减少MSMF警告
                    backends_to_try = [
                        cv2.CAP_DSHOW,    # Windows DirectShow (最稳定)
                        cv2.CAP_ANY,      # 自动检测
                        cv2.CAP_MSMF      # Windows Media Foundation (最后尝试)
                    ]
                    
                    cap = None
                    backend_used = None
                    
                    for backend in backends_to_try:
                        try:
                            cap = cv2.VideoCapture(camera_index, backend)
                            if cap.isOpened():
                                backend_used = backend
                                break
                            else:
                                if cap:
                                    cap.release()
                        except Exception as backend_error:
                            logger.debug(f"Backend {backend} failed for camera {camera_index}: {str(backend_error)}")
                            continue
                    
                    if not cap or not cap.isOpened():
                        camera_data.append({
                            "camera_id": f"cam_{camera_index}",
                            "status": "inactive",
                            "error": "Camera not accessible"
                        })
                        continue
                    
                    # 优化：减少缓冲区大小来减少异步回调警告
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception as buffer_error:
                        logger.debug(f"Could not set buffer size for camera {camera_index}: {str(buffer_error)}")
                    
                    ret, frame = cap.read()
                    if ret:
                        # 获取摄像头信息
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_data.append({
                            "camera_id": f"cam_{camera_index}",
                            "frame_size": f"{width}x{height}",
                            "fps": fps,
                            "status": "active",
                            "frame_count": int(time.time() % 10000),
                            "backend": str(backend_used) if backend_used else "unknown"
                        })
                        active_cameras += 1
                    else:
                        camera_data.append({
                            "camera_id": f"cam_{camera_index}",
                            "status": "inactive",
                            "error": "Frame read failed"
                        })
                    
                    # 优化：改进摄像头资源清理
                    # 先清除待处理帧缓冲区
                    for _ in range(3):
                        try:
                            cap.grab()
                        except Exception as e:
                            logger.debug(f"Failed to grab frame during cleanup: {e}")
                            break
                    
                    # 等待异步回调清理
                    if backend_used == cv2.CAP_MSMF:
                        time.sleep(0.05)  # MSMF需要更多时间清理
                    
                    # 释放摄像头
                    cap.release()
                    
                    # 释放后的延迟
                    if backend_used == cv2.CAP_MSMF:
                        time.sleep(0.02)  # MSMF需要更多时间
                    
                except Exception as e:
                    camera_data.append({
                        "camera_id": f"cam_{camera_index}",
                        "status": "error",
                        "error": str(e)
                    })
            
            result = {
                "type": "camera",
                "timestamp": datetime.now().isoformat(),
                "cameras": camera_data,
                "total_cameras": active_cameras,
                "hardware_available": active_cameras > 0,
                "cache_info": {
                    "is_cached": False,
                    "max_cameras_checked": max_cameras_to_check
                }
            }
            
            # 更新缓存
            self.camera_cache = result
            self.last_camera_check_time = current_time
            
            logger.debug(f"Camera check completed: {active_cameras} active cameras found")
            return result
            
        except ImportError:
            # OpenCV不可用
            result = {
                "type": "camera",
                "timestamp": datetime.now().isoformat(),
                "status": "opencv_not_available",
                "message": "OpenCV required for camera access"
            }
            self.camera_cache = result
            self.last_camera_check_time = current_time
            return result
        except Exception as e:
            # 其他错误
            result = {
                "type": "camera",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            self.camera_cache = result
            self.last_camera_check_time = current_time
            return result
        
    def _real_sensor_data(self):
        """真实传感器数据采集 - AGI增强版本"""
        try:
            sensor_data = {}
            
            # 尝试导入机器人硬件接口
            try:
                from core.hardware.robot_hardware_interface import RobotHardwareInterface
                from core.hardware.external_device_interface import ExternalDeviceInterface
                
                # 检查硬件接口是否已初始化
                if not hasattr(self, '_robot_hardware_interface'):
                    self._robot_hardware_interface = RobotHardwareInterface()
                    self._external_device_interface = ExternalDeviceInterface()
                    
                    # 尝试初始化硬件接口
                    try:
                        init_result = self._robot_hardware_interface.initialize_hardware()
                        if init_result.get("success", False):
                            logger.info("机器人硬件接口初始化成功")
                        else:
                            logger.warning(f"机器人硬件接口初始化失败: {init_result.get('error', 'Unknown error')}")
                    except Exception as init_error:
                        logger.warning(f"机器人硬件接口初始化异常: {str(init_error)}")
                
                # 尝试从硬件接口获取传感器数据
                try:
                    hardware_sensor_data = self._robot_hardware_interface.get_sensor_data()
                    
                    if hardware_sensor_data and isinstance(hardware_sensor_data, dict):
                        # 转换硬件数据格式
                        for sensor_id, sensor_info in hardware_sensor_data.items():
                            if isinstance(sensor_info, dict) and 'value' in sensor_info:
                                sensor_type = sensor_info.get('sensor_type', sensor_id)
                                sensor_data[sensor_type] = {
                                    "value": sensor_info['value'],
                                    "unit": sensor_info.get('unit', 'unknown'),
                                    "sensor_type": sensor_info.get('sensor_type', 'hardware'),
                                    "accuracy": sensor_info.get('accuracy', 'unknown'),
                                    "timestamp": sensor_info.get('timestamp', datetime.now().isoformat()),
                                    "source": "hardware_interface"
                                }
                    
                    # 如果有硬件数据，直接返回
                    if sensor_data:
                        return {
                            "type": "sensor",
                            "timestamp": datetime.now().isoformat(),
                            "sensors": sensor_data,
                            "hardware_available": True,
                            "source": "robot_hardware_interface"
                        }
                        
                except Exception as hardware_error:
                    logger.debug(f"从硬件接口获取传感器数据失败: {str(hardware_error)}")
                    
            except ImportError as import_error:
                logger.debug(f"硬件接口导入失败: {str(import_error)}")
            except Exception as interface_error:
                logger.debug(f"硬件接口访问失败: {str(interface_error)}")
            
            # 备用方案：尝试通过外部设备接口扫描传感器
            try:
                if hasattr(self, '_external_device_interface'):
                    # 扫描可用设备
                    available_devices = self._external_device_interface.scan_all_devices()
                    
                    sensor_devices = [dev for dev in available_devices if dev.get('type') in ['sensor', 'temperature', 'humidity', 'pressure']]
                    
                    for device in sensor_devices:
                        sensor_type = device.get('type', 'unknown')
                        sensor_id = device.get('id', f'{sensor_type}_{len(sensor_data)}')
                        
                        # 尝试连接并读取传感器数据
                        try:
                            connect_result = self._external_device_interface.connect_device(
                                device_id=sensor_id,
                                protocol=device.get('protocol', 'serial'),
                                connection_params=device.get('params', {})
                            )
                            
                            if connect_result.get('success', False):
                                # 这里可以添加真实数据读取逻辑
                                # 暂时标记为硬件存在但需要具体实现
                                sensor_data[sensor_type] = {
                                    "value": None,
                                    "unit": "unknown",
                                    "sensor_type": "hardware_detected",
                                    "accuracy": "unknown",
                                    "status": "hardware_present_implementation_required",
                                    "device_id": sensor_id,
                                    "source": "external_device_interface"
                                }
                        except Exception as connect_error:
                            logger.debug(f"传感器设备连接失败 {sensor_id}: {str(connect_error)}")
                            
            except Exception as scan_error:
                logger.debug(f"设备扫描失败: {str(scan_error)}")
            
            # 最终备用：如果没有检测到硬件，提供清晰的诊断信息
            if not sensor_data:
                sensor_data["system_status"] = {
                    "value": "no_hardware_detected",
                    "unit": "status",
                    "sensor_type": "diagnostic",
                    "accuracy": "exact",
                    "message": "未检测到连接的传感器硬件。请连接传感器设备或配置硬件接口。",
                    "recommendation": "1. 连接物理传感器设备\n2. 配置硬件驱动程序\n3. 检查设备连接状态",
                    "source": "system_diagnostic"
                }
            
            return {
                "type": "sensor",
                "timestamp": datetime.now().isoformat(),
                "sensors": sensor_data,
                "hardware_available": any(s.get('sensor_type') in ['hardware', 'hardware_detected'] for s in sensor_data.values()),
                "diagnostic_info": {
                    "hardware_detected": any(s.get('sensor_type') in ['hardware', 'hardware_detected'] for s in sensor_data.values()),
                    "simulation_mode": False,
                    "implementation_status": "real_hardware_interface_attempted"
                }
            }
        except Exception as e:
            logger.error(f"传感器数据采集失败: {str(e)}")
            return {
                "type": "sensor",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "diagnostic": "传感器数据采集系统遇到错误，请检查硬件连接和接口配置。",
                "hardware_available": False
            }
        
    def _real_model_data(self):
        """真实模型性能数据"""
        try:
            import psutil
            import os
            
            # 获取当前进程信息
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # 获取系统CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            return {
                "type": "model",
                "timestamp": datetime.now().isoformat(),
                "model_id": "main",
                "status": "running",
                "performance": {
                    "cpu_usage": round(cpu_percent, 1),
                    "memory_usage": round(memory_percent, 1),
                    "process_memory_mb": round(memory_info.rss / 1024 / 1024, 1),
                    "thread_count": process.num_threads()
                },
                "load": round(cpu_percent / 100, 2)  # 转换为0-1范围
            }
        except ImportError:
            # psutil不可用，返回诊断信息而不是模拟数据
            return {
                "type": "model",
                "timestamp": datetime.now().isoformat(),
                "model_id": "main",
                "status": "diagnostic",
                "performance": {
                    "cpu_usage": "monitoring_unavailable",
                    "memory_usage": "monitoring_unavailable",
                    "process_memory_mb": "monitoring_unavailable",
                    "thread_count": "monitoring_unavailable"
                },
                "load": "monitoring_unavailable",
                "diagnostic": {
                    "monitoring_system": "psutil_not_installed",
                    "recommendation": "Install psutil package for real performance monitoring: pip install psutil",
                    "simulation_mode": False
                }
            }
        except Exception as e:
            return {
                "type": "model",
                "timestamp": datetime.now().isoformat(),
                "model_id": "main",
                "status": "error",
                "error": str(e)
            }
        
    def _real_system_data(self):
        """真实系统监控数据"""
        try:
            import psutil
            
            # 获取系统信息
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # 获取系统启动时间
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            return {
                "type": "system",
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "cores": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown"
                },
                "memory": {
                    "usage_percent": round(memory.percent, 1),
                    "total_gb": round(memory.total / 1024 / 1024 / 1024, 1),
                    "available_gb": round(memory.available / 1024 / 1024 / 1024, 1),
                    "used_gb": round(memory.used / 1024 / 1024 / 1024, 1)
                },
                "disk": {
                    "usage_percent": round(disk.percent, 1),
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 1),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 1)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "system": {
                    "boot_time": boot_time.isoformat(),
                    "uptime_seconds": int(uptime.total_seconds()),
                    "platform": sys.platform
                }
            }
        except ImportError:
            return {
                "type": "system",
                "timestamp": datetime.now().isoformat(),
                "status": "psutil_not_available",
                "message": "psutil required for system monitoring"
            }
        except Exception as e:
            return {
                "type": "system",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
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

    def add_data(self, stream_id: str, data: Any):
        """Add data to a stream and broadcast to subscribers."""
        # 使用 asyncio.ensure_future 调度异步任务
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有事件循环，则无法广播数据，记录警告
            logger.warning(f"No event loop found, cannot broadcast data to stream: {stream_id}")
            return
        # 调度异步任务
        loop.create_task(self._async_add_data(stream_id, data))

    async def _async_add_data(self, stream_id: str, data: Any):
        """Async implementation of add_data."""
        await self.trigger_callbacks(stream_id, data)
        await self.broadcast_data(stream_id, data, source_model=None)
    
    async def route_to_model(self, model_id: str, data: Any):
        """路由数据到指定模型 / Route data to specific model"""
        if model_id in self.model_routing_table:
            for stream_id in self.model_routing_table[model_id]:
                await self.broadcast_data(stream_id, data, source_model="Router")
        else:
            error_handler.log_warning(f"No routing found for model: {model_id}", "RealTimeStreamManager")
    
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

# Create a singleton instance of the stream manager
_stream_manager_instance = None

def get_stream_manager():
    """获取流管理器单例实例 / Get stream manager singleton instance"""
    global _stream_manager_instance
    if _stream_manager_instance is None:
        _stream_manager_instance = RealTimeStreamManager()
    return _stream_manager_instance


@app.websocket("/ws/streams/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for real-time data streams with authentication"""
    # 检查认证
    api_key = os.environ.get("REALTIME_STREAM_API_KEY")
    auth_type = os.environ.get("REALTIME_STREAM_AUTH_TYPE", "api_key").lower()
    environment = os.environ.get("ENVIRONMENT", "development")
    
    # 如果生产环境未设置认证，则拒绝连接
    if environment == "production" and not api_key and auth_type != "jwt":
        await websocket.close(code=1008, reason="Authentication required in production")
        logger.warning(f"生产环境WebSocket连接缺少认证配置: stream_id={stream_id}")
        return
    
    query_params = dict(websocket.query_params)
    authenticated = False
    
    # API密钥认证
    if api_key:
        client_api_key = query_params.get("api_key")
        if client_api_key and client_api_key == api_key:
            authenticated = True
            logger.debug(f"WebSocket连接通过API密钥认证: stream_id={stream_id}")
    
    # JWT令牌认证（如果API密钥认证失败或未设置API密钥）
    if not authenticated and auth_type == "jwt":
        client_token = query_params.get("token")
        if client_token:
            try:
                security_manager = get_production_security_manager()
                payload = security_manager.verify_jwt_token(client_token)
                if payload:
                    authenticated = True
                    logger.debug(f"WebSocket连接通过JWT令牌认证: stream_id={stream_id}, user={payload.get('user_id', 'unknown')}")
            except Exception as e:
                logger.warning(f"JWT令牌验证失败: {str(e)}")
    
    # 如果认证失败，拒绝连接
    if not authenticated:
        await websocket.close(code=1008, reason="Invalid or missing authentication")
        logger.warning(f"WebSocket连接认证失败: stream_id={stream_id}")
        return
    
    stream_manager = get_stream_manager()
    await websocket.accept()
    stream_manager.performance_metrics['total_connections'] += 1
    stream_manager.performance_metrics['active_connections'] += 1
    
    # 记录连接信息（不含敏感数据）
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket连接已建立: stream_id={stream_id}, client_ip={client_ip}")
    
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
        logger.info(f"Client disconnected from stream: {stream_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        stream_manager.performance_metrics['error_count'] += 1
    finally:
        # Remove client from all subscriptions
        for stream, clients in stream_manager.subscriptions.items():
            if websocket in clients:
                clients.remove(websocket)
        stream_manager.performance_metrics['active_connections'] -= 1
        logger.info(f"WebSocket连接已关闭: stream_id={stream_id}, client_ip={client_ip}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stream_manager = get_stream_manager()
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
    stream_manager = get_stream_manager()
    return {
        "streams": list(stream_manager.data_streams.keys()),
        "total_streams": len(stream_manager.data_streams)
    }


@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    stream_manager = get_stream_manager()
    return stream_manager.get_performance_metrics()


@app.on_event("startup")
async def startup_event():
    """Initialize stream manager on startup"""
    stream_manager = get_stream_manager()
    # Initialize asynchronously since initialize() is now an async method
    await stream_manager.initialize()
    # Start the streaming loop in a background task
    asyncio.ensure_future(stream_manager.start_data_streaming())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    stream_manager = get_stream_manager()
    logger.info("Real-Time Stream Manager is shutting down")


if __name__ == "__main__":
    """Main entry point to start the server"""
    # Get singleton instance to prevent duplicate loading
    stream_manager = get_stream_manager()
    logger.info(f"Starting Real-Time Stream Manager on ws://{stream_manager.host}:{stream_manager.port}")
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host=stream_manager.host,
        port=stream_manager.port,
        reload=False,
        log_level="info"
    )
