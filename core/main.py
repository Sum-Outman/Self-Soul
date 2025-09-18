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
Self Soul 系统主入口文件
AGI Brain System Main Entry File

版权所有 (c) 2025 AGI Brain Team
Licensed under the Apache License, Version 2.0
"""
import os
import sys
import tempfile
from datetime import datetime

# Add the root directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import six compatibility fix before other imports
try:
    import six_compat
    print("Six compatibility fix imported successfully")
except ImportError as e:
    print(f"Warning: Could not import six compatibility fix: {e}")

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.error_handling import error_handler
from core.model_registry import ModelRegistry
from core.training_manager import TrainingManager
from core.emotion_awareness import EmotionAwarenessSystem
from core.autonomous_learning_manager import AutonomousLearningManager
from core.system_settings_manager import SystemSettingsManager
from core.api_model_connector import api_model_connector
from core.monitoring_enhanced import EnhancedSystemMonitor
from core.i18n_manager import set_language, get_supported_languages, _i18n_instance
from core.dataset_manager import dataset_manager

# 导入新的AGI组件
from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture
from core.enhanced_meta_cognition import EnhancedMetaCognition
from core.structured_knowledge_base import StructuredKnowledgeBase
from core.intrinsic_motivation_system import IntrinsicMotivationSystem
from core.explainable_ai import ExplainableAI
from core.value_alignment import ValueAlignment

# WebSocket连接管理器
# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# 创建FastAPI应用实例
# Create FastAPI application instance

# 全局模型注册表和训练管理器
# Global model registry and training manager
model_registry = ModelRegistry()
training_manager = TrainingManager(model_registry)

# 情感意识系统
# Emotion awareness system
emotion_system = EmotionAwarenessSystem()

# 自主学习管理器
# Autonomous learning manager
autonomous_learning_manager = AutonomousLearningManager(model_registry)

# 系统设置管理器
# System settings manager
system_settings_manager = SystemSettingsManager()

# 增强系统监控器
# Enhanced system monitor
system_monitor = EnhancedSystemMonitor()

# WebSocket连接管理器实例
# WebSocket connection manager instance
connection_manager = ConnectionManager()

# 新的AGI组件实例
# New AGI components instances
unified_cognitive_architecture = UnifiedCognitiveArchitecture()
enhanced_meta_cognition = EnhancedMetaCognition()
structured_knowledge_base = StructuredKnowledgeBase()
intrinsic_motivation_system = IntrinsicMotivationSystem()
explainable_ai = ExplainableAI()
value_alignment = ValueAlignment()

# 模型模式管理相关全局变量
# Global variables for model mode management
_model_modes = {}
# 存储外部模型连接器实例
# Store external model connector instances
_external_model_connectors = {}
# 批量处理的最大数量
# Maximum number for batch processing
BATCH_PROCESSING_LIMIT = 10

# ========== 模型模式管理辅助函数 ==========
# ========== Model Mode Management Helper Functions ==========

def load_model_modes_from_settings():
    """
    load_model_modes_from_settings函数 - 从系统设置中加载模型模式信息
    load_model_modes_from_settings Function - Load model modes from system settings
    """
    global _model_modes
    _model_modes.clear()
    
    try:
        settings = system_settings_manager.get_settings()
        models = settings.get("models", {})
        
        for model_id, model_config in models.items():
            # 根据模型来源确定模式
            # Determine mode based on model source
            if model_config.get("source") == "api":
                _model_modes[model_id] = "external"
            else:
                _model_modes[model_id] = "local"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", "加载模型模式信息失败")

def get_all_models_mode():
    """
    get_all_models_mode函数 - 获取所有模型的运行模式
    get_all_models_mode Function - Get running modes of all models
    
    Returns:
        模型ID到运行模式的映射 | Mapping of model IDs to running modes
    """
    return _model_modes.copy()

def get_model_mode(model_id: str) -> str:
    """
    get_model_mode函数 - 获取指定模型的运行模式
    get_model_mode Function - Get running mode of specified model
    
    Args:
        model_id: 模型ID | Model ID
    
    Returns:
        模型运行模式 (local/external) | Model running mode (local/external)
    """
    return _model_modes.get(model_id, "local")

def switch_model_to_external(model_id: str, api_config: dict) -> str:
    """
    switch_model_to_external函数 - 切换模型到外部API模式
    switch_model_to_external Function - Switch model to external API mode
    
    Args:
        model_id: 模型ID | Model ID
        api_config: API配置 | API configuration
    
    Returns:
        操作结果消息 | Operation result message
    """
    try:
        # 先卸载本地模型（如果已加载）
        # Unload local model first (if loaded)
        if model_registry.is_model_loaded(model_id):
            model_registry.unload_model(model_id)
        
        # 断开旧的外部连接（如果存在）
        # Disconnect old external connection (if exists)
        if model_id in _external_model_connectors:
            try:
                _external_model_connectors[model_id].disconnect()
            except Exception as e:
                error_handler.handle_error(e, "Model Mode", f"断开模型 {model_id} 的旧外部连接失败")
        
        # 创建新的外部模型连接器
        # Create new external model connector
        connector = api_model_connector
        
        # 配置连接参数
        # Configure connection parameters
        connection_params = {
            "api_url": api_config.get("api_url"),
            "api_key": api_config.get("api_key"),
            "model_name": api_config.get("model_name")
        }
        
        # 连接到外部API
        # Connect to external API
        success = connector.connect(connection_params)
        
        if not success:
            raise Exception(f"连接到外部API失败: {model_id}")
        
        # 保存连接器实例
        # Save connector instance
        _external_model_connectors[model_id] = connector
        
        # 更新模型模式
        # Update model mode
        _model_modes[model_id] = "external"
        
        # 更新系统设置
        # Update system settings
        system_settings_manager.update_model_setting(model_id, {
            "source": "api",
            "api_config": api_config
        })
        
        return f"模型 {model_id} 已成功切换到外部API模式"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", f"切换模型 {model_id} 到外部API模式失败")
        raise

def switch_model_to_local(model_id: str) -> str:
    """
    switch_model_to_local函数 - 切换模型到本地模式
    switch_model_to_local Function - Switch model to local mode
    
    Args:
        model_id: 模型ID | Model ID
    
    Returns:
        操作结果消息 | Operation result message
    """
    try:
        # 断开外部连接（如果存在）
        # Disconnect external connection (if exists)
        if model_id in _external_model_connectors:
            try:
                _external_model_connectors[model_id].disconnect()
                del _external_model_connectors[model_id]
            except Exception as e:
                error_handler.handle_error(e, "Model Mode", f"断开模型 {model_id} 的外部连接失败")
        
        # 加载本地模型
        # Load local model
        model_registry.load_model(model_id)
        
        # 更新模型模式
        # Update model mode
        _model_modes[model_id] = "local"
        
        # 更新系统设置
        # Update system settings
        system_settings_manager.update_model_setting(model_id, {
            "source": "local"
        })
        
        return f"模型 {model_id} 已成功切换到本地模式"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", f"切换模型 {model_id} 到本地模式失败")
        raise

def batch_switch_model_modes(models_data: list) -> list:
    """
    batch_switch_model_modes函数 - 批量切换模型模式
    batch_switch_model_modes Function - Batch switch model modes
    
    Args:
        models_data: 模型数据列表 | List of model data
    
    Returns:
        每个模型的切换结果列表 | List of switch results for each model
    """
    results = []
    
    # 限制批量处理数量
    # Limit batch processing quantity
    models_to_process = models_data[:BATCH_PROCESSING_LIMIT]
    
    for model_data in models_to_process:
        model_id = model_data.get("id")
        if not model_id:
            results.append({
                "id": None,
                "success": False,
                "message": "模型ID不能为空"
            })
            continue
        
        target_mode = model_data.get("mode", "local")
        
        try:
            if target_mode == "external":
                api_config = model_data.get("api_config", {})
                message = switch_model_to_external(model_id, api_config)
            else:
                message = switch_model_to_local(model_id)
            
            results.append({
                "id": model_id,
                "success": True,
                "message": message
            })
        except Exception as e:
            results.append({
                "id": model_id,
                "success": False,
                "message": str(e)
            })
    
    # 如果超出限制数量，提示用户
    # If exceeding limit quantity, prompt user
    if len(models_data) > BATCH_PROCESSING_LIMIT:
        results.append({
            "id": "batch_info",
            "success": True,
            "message": f"由于批量处理限制，仅处理了{len(models_to_process)}个模型（共{len(models_data)}个）"
        })
    
    return results

def test_external_api_connection(connection_data: dict) -> dict:
    """
    test_external_api_connection函数 - 测试外部API连接
    test_external_api_connection Function - Test external API connection
    
    Args:
        connection_data: 连接测试数据 | Connection test data
    
    Returns:
        连接测试结果 | Connection test result
    """
    try:
        # 创建临时连接器用于测试
        # Create temporary connector for testing
        connector = api_model_connector
        
        # 准备连接参数
        # Prepare connection parameters
        connection_params = {
            "api_url": connection_data.get("api_url"),
            "api_key": connection_data.get("api_key"),
            "model_name": connection_data.get("model_name")
        }
        
        # 连接到外部API
        # Connect to external API
        success = connector.connect(connection_params)
        
        if success:
            # 执行简单测试
            # Perform simple test
            test_result = connector.test_connection()
            
            # 断开连接
            # Disconnect
            connector.disconnect()
            
            if test_result["success"]:
                return {
                    "status": "success",
                    "message": "API连接测试成功",
                    "details": test_result.get("details", {})
                }
            else:
                return {
                    "status": "error",
                    "message": test_result.get("message", "API连接测试失败")
                }
        else:
            return {
                "status": "error",
                "message": "无法建立API连接"
            }
    except Exception as e:
        error_handler.handle_error(e, "API Connection", "测试外部API连接失败")
        return {
            "status": "error",
            "message": str(e)
        }

# 在启动时加载模型模式
load_model_modes_from_settings()

app = FastAPI(
    title="Self Soul 系统",
    description="Self Soul ",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# WebSocket端点
# WebSocket endpoints
@app.websocket("/ws/training/{job_id}")
async def websocket_training_endpoint(websocket: WebSocket, job_id: str):
    """
    websocket_training_endpoint函数 - 训练进度WebSocket端点
    websocket_training_endpoint Function - Training progress WebSocket endpoint
    
    Args:
        websocket: WebSocket连接 | WebSocket connection
        job_id: 训练任务ID | Training job ID
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            # 获取训练状态并发送给客户端
            status = training_manager.get_job_status(job_id)
            await websocket.send_json({
                "type": "training_status",
                "job_id": job_id,
                "status": status
            })
            
            # 如果训练完成或失败，关闭连接
            if status.get("status") in ["completed", "failed", "stopped"]:
                break
                
            # 每秒更新一次
            import asyncio
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", f"训练WebSocket错误: {job_id}")
        connection_manager.disconnect(websocket)

@app.websocket("/ws/monitoring")
async def websocket_monitoring_endpoint(websocket: WebSocket):
    """
    websocket_monitoring_endpoint函数 - 实时监控WebSocket端点
    websocket_monitoring_endpoint Function - Real-time monitoring WebSocket endpoint
    
    Args:
        websocket: WebSocket连接 | WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            # 获取实时监控数据并发送给客户端
            monitoring_data = await get_realtime_monitoring()
            await websocket.send_json({
                "type": "monitoring_data",
                "data": monitoring_data["data"]
            })
            
            # 每2秒更新一次
            import asyncio
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "监控WebSocket错误")
        connection_manager.disconnect(websocket)

@app.websocket("/ws/test-connection")
async def websocket_test_connection(websocket: WebSocket):
    """
    websocket_test_connection函数 - WebSocket连接测试端点
    websocket_test_connection Function - WebSocket connection test endpoint
    
    Args:
        websocket: WebSocket连接 | WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connection_test",
            "status": "success",
            "message": "WebSocket连接成功"
        })
        
        # 保持连接活跃
        while True:
            import asyncio
            await asyncio.sleep(10)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "WebSocket测试连接错误")
        connection_manager.disconnect(websocket)

# 启动事件处理
# Startup event handler
@app.on_event("startup")
async def startup_event():
    """系统启动事件处理
    System startup event handler
    """
    error_handler.log_info("Self Soul 系统正在启动...", "System")
    error_handler.log_info("Loading all models...", "System")
    
    # 加载所有模型
    # Load all models
    try:
        loaded_models = model_registry.load_all_models()
        error_handler.log_info(f"成功加载 {len(loaded_models)} 个模型: {', '.join(loaded_models)}", "System")
        
        # 初始化所有已加载的模型
        # Initialize all loaded models
        error_handler.log_info("正在初始化所有模型...", "System")
        initialized_count = 0
        for model_id in loaded_models:
            model = model_registry.get_model(model_id)
            if model and hasattr(model, 'initialize'):
                try:
                    result = model.initialize()
                    if result and isinstance(result, dict) and result.get("success", True):
                        error_handler.log_info(f"模型 {model_id} 初始化成功", "System")
                        initialized_count += 1
                    else:
                        error_handler.log_warning(f"模型 {model_id} 初始化可能失败: {result}", "System")
                except Exception as e:
                    error_handler.handle_error(e, "System", f"模型 {model_id} 初始化失败")
        
        error_handler.log_info(f"成功初始化 {initialized_count}/{len(loaded_models)} 个模型", "System")
        
    except Exception as e:
        error_handler.handle_error(e, "System", "加载模型失败")

# 关闭事件处理
# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """系统关闭事件处理
    System shutdown event handler
    """
    error_handler.log_info("Self Soul 系统正在关闭...", "System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义健康检查端点
# Health check endpoint
@app.get("/health")
async def health_check():
    """
    health_check函数 - 中文函数描述
    health_check Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    return {"status": "ok", "message": "Self Soul 系统运行正常"}

# 获取所有模型状态
# Get all models status
@app.get("/api/models/status")
async def get_models_status():
    """
    get_models_status函数 - 中文函数描述
    get_models_status Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    try:
        statuses = model_registry.get_all_models_status()
        return {"status": "success", "data": statuses}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取模型状态失败")
        raise HTTPException(status_code=500, detail="获取模型状态失败")

# 处理文本输入
# Process text input
@app.post("/api/process/text")
async def process_text(input_data: dict):
    """
    process_text函数 - 处理文本输入
    process_text Function - Process text input

    Args:
        input_data: 包含文本和语言信息的字典 | Dictionary containing text and language info
            - text: 要处理的文本 | Text to process
            - lang: 语言代码 (zh, en等) | Language code
            
    Returns:
        处理结果 | Processing result
    """
    try:
        manager_model = model_registry.get_model("manager")
        if not manager_model:
            raise HTTPException(status_code=500, detail="管理模型未加载")
        
        result = manager_model.process_input({"text": input_data.get("text", ""), "type": "text"})
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "处理文本输入失败")
        raise HTTPException(status_code=500, detail="处理文本输入失败")

# 处理视频输入
# Process video input
@app.post("/api/process/video")
async def process_video(
    video: UploadFile = File(...),
    lang: str = Form("zh")
):
    """
    process_video函数 - 处理视频输入
    process_video Function - Process video input
    
    Args:
        video: 上传的视频文件 | Uploaded video file
        lang: 语言代码 | Language code
        
    Returns:
        视频分析结果 | Video analysis result
    """
    try:
        # 检查视频模型是否可用
        video_model = model_registry.get_model("vision_video")
        if not video_model:
            raise HTTPException(status_code=500, detail="视频模型未加载")
        
        # 创建临时文件保存视频
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 使用视频模型处理视频
            result = video_model.process_input({
                "video_path": temp_file_path,
                "type": "video",
                "lang": lang
            })
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            return {"status": "success", "data": result}
        except Exception as processing_error:
            # 确保临时文件被清理
            try:
                os.unlink(temp_file_path)
            except:
                pass
            raise processing_error
            
    except Exception as e:
        error_handler.handle_error(e, "API", "处理视频输入失败")
        raise HTTPException(status_code=500, detail="处理视频输入失败")

# 处理图像输入
# Process image input
@app.post("/api/process/image")
async def process_image(
    image: UploadFile = File(...),
    lang: str = Form("zh")
):
    """
    process_image函数 - 处理图像输入
    process_image Function - Process image input
    
    Args:
        image: 上传的图像文件 | Uploaded image file
        lang: 语言代码 | Language code
        
    Returns:
        图像分析结果 | Image analysis result
    """
    try:
        # 检查图像模型是否可用
        image_model = model_registry.get_model("vision_image")
        if not image_model:
            raise HTTPException(status_code=500, detail="图像模型未加载")
        
        # 创建临时文件保存图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 使用图像模型处理图像
            result = image_model.process_input({
                "image_path": temp_file_path,
                "type": "image",
                "lang": lang
            })
            
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            return {"status": "success", "data": result}
        except Exception as processing_error:
            # 确保临时文件被清理
            try:
                os.unlink(temp_file_path)
            except:
                pass
            raise processing_error
            
    except Exception as e:
        error_handler.handle_error(e, "API", "处理图像输入失败")
        raise HTTPException(status_code=500, detail="处理图像输入失败")

# 实时音频流WebSocket端点
# Real-time audio stream WebSocket endpoint
@app.websocket("/ws/audio-stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    websocket_audio_stream函数 - 实时音频流处理
    websocket_audio_stream Function - Real-time audio stream processing
    
    Args:
        websocket: WebSocket连接 | WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # 获取音频模型
        audio_model = model_registry.get_model("audio")
        if not audio_model:
            await websocket.send_json({
                "type": "error",
                "message": "音频模型未加载"
            })
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": "音频流连接成功"
        })
        
        while True:
            # 接收音频数据
            data = await websocket.receive()
            
            if data.get("type") == "websocket.disconnect":
                break
                
            if data.get("type") == "websocket.receive":
                # 处理音频数据
                audio_data = data.get("bytes") or data.get("text")
                if audio_data:
                    try:
                        result = audio_model.process_input({
                            "audio_data": audio_data,
                            "type": "audio_stream",
                            "lang": "zh"
                        })
                        await websocket.send_json({
                            "type": "audio_processed",
                            "data": result
                        })
                    except Exception as e:
                        error_handler.handle_error(e, "WebSocket", "处理音频流数据失败")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"处理音频失败: {str(e)}"
                        })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "音频流WebSocket错误")
        connection_manager.disconnect(websocket)

# 实时视频流WebSocket端点
# Real-time video stream WebSocket endpoint
@app.websocket("/ws/video-stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    websocket_video_stream函数 - 实时视频流处理
    websocket_video_stream Function - Real-time video stream processing
    
    Args:
        websocket: WebSocket连接 | WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # 获取视频模型
        video_model = model_registry.get_model("vision_video")
        if not video_model:
            await websocket.send_json({
                "type": "error",
                "message": "视频模型未加载"
            })
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": "视频流连接成功"
        })
        
        while True:
            # 接收视频帧数据
            data = await websocket.receive()
            
            if data.get("type") == "websocket.disconnect":
                break
                
            if data.get("type") == "websocket.receive":
                # 处理视频帧数据
                frame_data = data.get("bytes") or data.get("text")
                if frame_data:
                    try:
                        result = video_model.process_input({
                            "video_frame": frame_data,
                            "type": "video_stream",
                            "lang": "zh"
                        })
                        await websocket.send_json({
                            "type": "video_processed",
                            "data": result
                        })
                    except Exception as e:
                        error_handler.handle_error(e, "WebSocket", "处理视频流数据失败")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"处理视频帧失败: {str(e)}"
                        })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "视频流WebSocket错误")
        connection_manager.disconnect(websocket)

# 启动模型训练 (兼容旧端点)
# Start model training (compatible with old endpoint)
@app.post("/api/train")
async def train(training_config: dict):
    """
    train函数 - 启动模型训练 (兼容前端)
    train Function - Start model training (compatible with frontend)

    Args:
        training_config: 训练配置 | Training configuration
        
    Returns:
        训练任务ID | Training job ID
    """
    try:
        mode = training_config.get("mode", "individual")
        models = training_config.get("models", [])
        dataset = training_config.get("dataset", "")
        parameters = training_config.get("parameters", {})
        
        # 转换参数格式以匹配训练管理器
        training_params = {
            "epochs": parameters.get("epochs", 10),
            "batch_size": parameters.get("batchSize", 32),
            "learning_rate": parameters.get("learningRate", 0.001),
            "validation_split": parameters.get("validationSplit", 0.2),
            "dropout_rate": parameters.get("dropoutRate", 0.1),
            "weight_decay": parameters.get("weightDecay", 0.0001),
            "momentum": parameters.get("momentum", 0.9),
            "optimizer": parameters.get("optimizer", "adam"),
            "training_mode": mode,
            "strategy": parameters.get("strategy", "standard"),
            "knowledge_assist": parameters.get("knowledge_assist")
        }
        
        job_id = training_manager.start_training(models, training_params)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "启动训练失败")
        raise HTTPException(status_code=500, detail=str(e))

# 启动模型训练 (新端点)
# Start model training (new endpoint)
@app.post("/api/training/start")
async def start_training(training_config: dict):
    """
    start_training函数 - 中文函数描述
    start_training Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    try:
        model_ids = training_config.get("model_ids", [])
        parameters = training_config.get("parameters", {})
        
        job_id = training_manager.start_training(model_ids, parameters)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "启动训练失败")
        raise HTTPException(status_code=500, detail=str(e))

# 获取训练任务状态
# Get training job status
@app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """
    get_training_status函数 - 中文函数描述
    get_training_status Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    try:
        status = training_manager.get_job_status(job_id)
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取训练状态失败")
        raise HTTPException(status_code=500, detail="获取训练状态失败")

# 停止训练任务
# Stop training job
@app.post("/api/training/stop/{job_id}")
async def stop_training(job_id: str):
    """
    stop_training函数 - 中文函数描述
    stop_training Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    try:
        success = training_manager.stop_training(job_id)
        return {"status": "success", "stopped": success}
    except Exception as e:
        error_handler.handle_error(e, "API", "停止训练失败")
        raise HTTPException(status_code=500, detail="停止训练失败")

# 获取训练历史
# Get training history
@app.get("/api/training/history")
async def get_training_history():
    """
    get_training_history函数 - 中文函数描述
    get_training_history Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    try:
        history = training_manager.get_training_history()
        return {"status": "success", "data": history}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取训练历史失败")
        raise HTTPException(status_code=500, detail="获取训练历史失败")

# 获取所有训练会话状态
# Get all training sessions status
@app.get("/api/training/sessions")
async def get_training_sessions():
    """
    get_training_sessions函数 - 获取所有训练会话状态
    get_training_sessions Function - Get all training sessions status
    
    Returns:
        所有训练任务的状态 | Status of all training jobs
    """
    try:
        sessions = training_manager.get_all_jobs_status()
        return {"status": "success", "data": sessions}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取训练会话失败")
        raise HTTPException(status_code=500, detail="获取训练会话失败")

# 验证联合训练模型组合
# Validate joint training model combination
@app.post("/api/train/validate-combination")
async def validate_training_combination(combination_data: dict):
    """
    validate_training_combination函数 - 验证联合训练模型组合
    validate_training_combination Function - Validate joint training model combination

    Args:
        combination_data: 包含模型列表和训练模式的字典 | Dictionary containing model list and training mode
        
    Returns:
        验证结果 | Validation result
    """
    try:
        models = combination_data.get("models", [])
        mode = combination_data.get("mode", "joint")
        
        # 验证模型组合
        validation_result = training_manager.validate_model_combination(models, mode)
        
        return {
            "status": "success",
            "valid": validation_result["valid"],
            "message": validation_result["message"],
            "details": validation_result.get("details", {})
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "验证模型组合失败")
        raise HTTPException(status_code=500, detail="验证模型组合失败")

# 联合训练专用端点
# Joint training specific endpoints

# 获取联合训练推荐组合
# Get joint training recommended combinations
@app.get("/api/joint-training/recommendations")
async def get_joint_training_recommendations():
    """
    get_joint_training_recommendations函数 - 获取联合训练推荐组合
    get_joint_training_recommendations Function - Get joint training recommended combinations

    Returns:
        推荐的联合训练模型组合列表 | List of recommended joint training model combinations
    """
    try:
        recommendations = training_manager.get_joint_training_recommendations()
        return {"status": "success", "data": recommendations}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取联合训练推荐组合失败")
        raise HTTPException(status_code=500, detail="获取联合训练推荐组合失败")

# 启动联合训练
# Start joint training
@app.post("/api/joint-training/start")
async def start_joint_training(joint_config: dict):
    """
    start_joint_training函数 - 启动联合训练
    start_joint_training Function - Start joint training

    Args:
        joint_config: 联合训练配置 | Joint training configuration
            - model_ids: 参与联合训练的模型ID列表 | List of model IDs for joint training
            - strategy: 训练策略 (standard, knowledge_assisted, adaptive) | Training strategy
            - parameters: 训练参数 | Training parameters
        
    Returns:
        训练任务ID | Training job ID
    """
    try:
        model_ids = joint_config.get("model_ids", [])
        strategy = joint_config.get("strategy", "standard")
        parameters = joint_config.get("parameters", {})
        
        # 设置训练模式为联合训练
        parameters["training_mode"] = "joint"
        parameters["strategy"] = strategy
        
        job_id = training_manager.start_training(model_ids, parameters)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "启动联合训练失败")
        raise HTTPException(status_code=500, detail=str(e))

# 数据集管理端点
# Dataset management endpoints

# 获取所有模型支持的数据格式
# Get all supported data formats for all models
@app.get("/api/datasets/supported-formats")
async def get_supported_formats():
    """
    get_supported_formats函数 - 获取所有模型支持的数据格式
    get_supported_formats Function - Get all supported data formats for all models
    
    Returns:
        各模型支持的数据格式 | Supported data formats for each model
    """
    try:
        formats = dataset_manager.get_all_supported_formats()
        return {"status": "success", "data": formats}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取支持格式失败")
        raise HTTPException(status_code=500, detail="获取支持格式失败")

# 获取指定模型支持的数据格式
# Get supported data formats for specific model
@app.get("/api/datasets/supported-formats/{model_id}")
async def get_model_supported_formats(model_id: str):
    """
    get_model_supported_formats函数 - 获取指定模型支持的数据格式
    get_model_supported_formats Function - Get supported data formats for specific model
    
    Args:
        model_id: 模型ID | Model ID
        
    Returns:
        支持的格式列表 | List of supported formats
    """
    try:
        formats = dataset_manager.get_model_supported_formats(model_id)
        return {"status": "success", "data": formats}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取模型支持格式失败")
        raise HTTPException(status_code=500, detail="获取模型支持格式失败")

# 验证数据集文件
# Validate dataset file
@app.post("/api/datasets/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    validate_dataset函数 - 验证数据集文件
    validate_dataset Function - Validate dataset file
    
    Args:
        file: 上传的文件 | Uploaded file
        model_id: 目标模型ID | Target model ID
        
    Returns:
        验证结果 | Validation result
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 验证数据集
        validation_result = dataset_manager.validate_dataset(temp_file_path, model_id)
        
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        return {"status": "success", "data": validation_result}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "数据集验证失败")
        raise HTTPException(status_code=500, detail="数据集验证失败")

# 上传数据集文件
# Upload dataset file
@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    dataset_name: str = Form(None)
):
    """
    upload_dataset函数 - 上传数据集文件
    upload_dataset Function - Upload dataset file
    
    Args:
        file: 上传的文件 | Uploaded file
        model_id: 目标模型ID | Target model ID
        dataset_name: 数据集名称 (可选) | Dataset name (optional)
        
    Returns:
        上传结果 | Upload result
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 保存数据集
        save_result = dataset_manager.save_dataset(temp_file_path, model_id, dataset_name)
        
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        if save_result["success"]:
            return {"status": "success", "data": save_result}
        else:
            return {"status": "error", "error": save_result.get("error", "保存失败")}
            
    except Exception as e:
        error_handler.handle_error(e, "API", "数据集上传失败")
        raise HTTPException(status_code=500, detail="数据集上传失败")

# 列出所有数据集
# List all datasets
@app.get("/api/datasets")
async def list_datasets(model_id: str = None):
    """
    list_datasets函数 - 列出所有数据集
    list_datasets Function - List all datasets
    
    Args:
        model_id: 特定模型的ID (可选) | Specific model ID (optional)
        
    Returns:
        数据集列表 | List of datasets
    """
    try:
        datasets = dataset_manager.list_datasets(model_id)
        return {"status": "success", "data": datasets}
    except Exception as e:
        error_handler.handle_error(e, "API", "列出数据集失败")
        raise HTTPException(status_code=500, detail="列出数据集失败")

# 获取数据集信息
# Get dataset information
@app.get("/api/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    get_dataset_info函数 - 获取数据集信息
    get_dataset_info Function - Get dataset information
    
    Args:
        dataset_id: 数据集ID | Dataset ID
        
    Returns:
        数据集信息 | Dataset information
    """
    try:
        dataset_info = dataset_manager.get_dataset_info(dataset_id)
        if dataset_info:
            return {"status": "success", "data": dataset_info}
        else:
            raise HTTPException(status_code=404, detail="数据集不存在")
    except Exception as e:
        error_handler.handle_error(e, "API", "获取数据集信息失败")
        raise HTTPException(status_code=500, detail="获取数据集信息失败")

# 删除数据集
# Delete dataset
@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    delete_dataset函数 - 删除数据集
    delete_dataset Function - Delete dataset
    
    Args:
        dataset_id: 数据集ID | Dataset ID
        
    Returns:
        删除结果 | Delete result
    """
    try:
        delete_result = dataset_manager.delete_dataset(dataset_id)
        if delete_result["success"]:
            return {"status": "success", "data": delete_result}
        else:
            raise HTTPException(status_code=404, detail=delete_result.get("error", "删除失败"))
    except Exception as e:
        error_handler.handle_error(e, "API", "删除数据集失败")
        raise HTTPException(status_code=500, detail="删除数据集失败")

# 获取数据集统计信息
# Get dataset statistics
@app.get("/api/datasets/stats")
async def get_dataset_stats():
    """
    get_dataset_stats函数 - 获取数据集统计信息
    get_dataset_stats Function - Get dataset statistics
    
    Returns:
        统计信息 | Statistics
    """
    try:
        stats = dataset_manager.get_dataset_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取数据集统计失败")
        raise HTTPException(status_code=500, detail="获取数据集统计失败")

# 语言设置端点
# Language settings endpoints

# 设置系统语言
# Set system language
@app.post("/api/language/set")
async def set_language_endpoint(language_data: dict):
    """
    set_language_endpoint函数 - 设置系统语言
    set_language_endpoint Function - Set system language
    
    Args:
        language_data: 语言设置数据 | Language setting data
            - language: 语言代码 (zh, en, de, ja, ru) | Language code
        
    Returns:
        设置结果 | Setting result
    """
    try:
        language = language_data.get("language", "zh")
        result = set_language(language)
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "设置语言失败")
        raise HTTPException(status_code=500, detail="设置语言失败")

# 获取支持的语言列表
# Get supported languages list
@app.get("/api/language/supported")
async def get_supported_languages_endpoint():
    """
    get_supported_languages_endpoint函数 - 获取支持的语言列表
    get_supported_languages_endpoint Function - Get supported languages list
    
    Returns:
        支持的语言列表 | Supported languages list
    """
    try:
        languages = get_supported_languages()
        return {"status": "success", "data": languages}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取支持语言失败")
        raise HTTPException(status_code=500, detail="获取支持语言失败")

# 系统监控端点
# System monitoring endpoints

# 获取实时监控数据
# Get real-time monitoring data
@app.get("/api/monitoring/realtime")
async def get_realtime_monitoring():
    """
    get_realtime_monitoring函数 - 获取实时监控数据
    get_realtime_monitoring Function - Get real-time monitoring data
    
    Returns:
        监控数据 | Monitoring data
    """
    try:
        monitoring_data = system_monitor.get_realtime_monitoring()
        return {"status": "success", "data": monitoring_data}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取监控数据失败")
        raise HTTPException(status_code=500, detail="获取监控数据失败")

# 获取系统性能统计
# Get system performance statistics
@app.get("/api/monitoring/performance")
async def get_performance_stats():
    """
    get_performance_stats函数 - 获取系统性能统计
    get_performance_stats Function - Get system performance statistics
    
    Returns:
        性能统计 | Performance statistics
    """
    try:
        performance_stats = system_monitor.get_performance_stats()
        return {"status": "success", "data": performance_stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取性能统计失败")
        raise HTTPException(status_code=500, detail="获取性能统计失败")

# 测试外部API连接
# Test external API connection
@app.post("/api/test-connection")
async def test_connection(connection_data: dict):
    """
    test_connection函数 - 测试外部API连接
    test_connection Function - Test external API connection
    
    Args:
        connection_data: 连接测试数据，包含api_endpoint、api_key、model_name、api_type等
        Connection test data including api_endpoint, api_key, model_name, api_type, etc.
        
    Returns:
        连接测试结果 | Connection test result
    """
    try:
        endpoint = connection_data.get("api_endpoint", "")
        api_key = connection_data.get("api_key", "")
        model_name = connection_data.get("model_name", "")
        api_type = connection_data.get("api_type", "generic")
        
        if not endpoint or not api_key:
            return {"success": False, "message": "API端点和密钥不能为空 | API endpoint and key cannot be empty"}
        
        # 使用API模型连接器测试连接
        test_result = api_model_connector._test_connection(endpoint, api_key)
        
        # 如果连接成功，保存配置到系统设置
        if test_result["success"]:
            model_id = connection_data.get("modelId", "")
            if model_id:
                # 保存API配置
                config = {
                    "api_url": endpoint,
                    "api_key": api_key,
                    "model_name": model_name,
                    "api_type": api_type,
                    "source": "external"
                }
                system_settings_manager.update_model_setting(model_id, config)
        
        return test_result
        
    except Exception as e:
        error_handler.handle_error(e, "API", "连接测试失败")
        return {"success": False, "message": f"连接测试异常: {str(e)} | Connection test exception: {str(e)}"}

# 仪表盘API端点
# Dashboard API endpoints

# 获取仪表盘系统指标
# Get dashboard system metrics
@app.get("/api/dashboard/system")
async def get_dashboard_system():
    """
    get_dashboard_system函数 - 获取仪表盘系统指标
    get_dashboard_system Function - Get dashboard system metrics
    
    Returns:
        系统性能指标 | System performance metrics
    """
    try:
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        base_metrics = enhanced_metrics.get("base_metrics", {})
        
        metrics = [
            {
                "id": "cpu_usage",
                "title": "CPU使用率",
                "value": base_metrics.get("cpu_usage", 0),
                "unit": "%",
                "threshold": 90,
                "warning": 70,
                "trend": "stable",
                "change": "0%"
            },
            {
                "id": "memory_usage",
                "title": "内存使用率",
                "value": base_metrics.get("memory_usage", 0),
                "unit": "%",
                "threshold": 85,
                "warning": 75,
                "trend": "stable",
                "change": "0%"
            },
            {
                "id": "disk_usage",
                "title": "磁盘使用率",
                "value": base_metrics.get("disk_usage", 0),
                "unit": "%",
                "threshold": 95,
                "warning": 85,
                "trend": "stable",
                "change": "0%"
            },
            {
                "id": "network_in",
                "title": "网络流入",
                "value": base_metrics.get("network_io", {}).get("bytes_recv", 0) / 1024 / 1024,
                "unit": "MB/s",
                "threshold": 100,
                "warning": 50,
                "trend": "stable",
                "change": "0MB/s"
            },
            {
                "id": "network_out",
                "title": "网络流出",
                "value": base_metrics.get("network_io", {}).get("bytes_sent", 0) / 1024 / 1024,
                "unit": "MB/s",
                "threshold": 100,
                "warning": 50,
                "trend": "stable",
                "change": "0MB/s"
            },
            {
                "id": "response_time",
                "title": "平均响应时间",
                "value": enhanced_metrics.get("task_metrics", [{}])[0].get("avg_response_time", 0),
                "unit": "ms",
                "threshold": 1000,
                "warning": 500,
                "trend": "stable",
                "change": "0ms"
            }
        ]
        
        return {"status": "success", "metrics": metrics}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘系统指标失败")
        raise HTTPException(status_code=500, detail="获取仪表盘系统指标失败")

# 获取仪表盘日志
# Get dashboard logs
@app.get("/api/dashboard/logs")
async def get_dashboard_logs(limit: int = 50):
    """
    get_dashboard_logs函数 - 获取仪表盘日志
    get_dashboard_logs Function - Get dashboard logs
    
    Args:
        limit: 日志条数限制 | Log entries limit
        
    Returns:
        最近的系统日志 | Recent system logs
    """
    try:
        logs = system_monitor.get_recent_logs(limit=limit)
        return {"status": "success", "logs": logs}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘日志失败")
        raise HTTPException(status_code=500, detail="获取仪表盘日志失败")

# 获取仪表盘活动
# Get dashboard activities
@app.get("/api/dashboard/activities")
async def get_dashboard_activities(limit: int = 30):
    """
    get_dashboard_activities函数 - 获取仪表盘活动
    get_dashboard_activities Function - Get dashboard activities
    
    Args:
        limit: 活动条数限制 | Activities limit
        
    Returns:
        最近的模型活动 | Recent model activities
    """
    try:
        # 从增强监控中获取活动数据
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        activities = enhanced_metrics.get("data_streams", {}).get("model_activities", [])
        
        # 限制返回数量
        activities = activities[:limit]
        
        return {"status": "success", "activities": activities}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘活动失败")
        raise HTTPException(status_code=500, detail="获取仪表盘活动失败")

# 获取仪表盘训练会话
# Get dashboard training sessions
@app.get("/api/dashboard/training-sessions")
async def get_dashboard_training_sessions():
    """
    get_dashboard_training_sessions函数 - 获取仪表盘训练会话
    get_dashboard_training_sessions Function - Get dashboard training sessions
    
    Returns:
        活跃的训练会话 | Active training sessions
    """
    try:
        all_jobs = training_manager.get_all_jobs_status()
        active_sessions = []
        
        for job_id, job_data in all_jobs.items():
            if job_data.get("status") in ["running", "pending"]:
                active_sessions.append({
                    "id": job_id,
                    "models": job_data.get("model_ids", []),
                    "status": job_data.get("status", "unknown"),
                    "progress": job_data.get("progress", 0),
                    "currentEpoch": job_data.get("current_epoch", 0),
                    "totalEpochs": job_data.get("total_epochs", 0),
                    "loss": job_data.get("current_loss", 0.0),
                    "accuracy": job_data.get("current_accuracy", 0.0) * 100,
                    "elapsedTime": job_data.get("elapsed_time", "00:00:00")
                })
        
        return {"status": "success", "sessions": active_sessions}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘训练会话失败")
        raise HTTPException(status_code=500, detail="获取仪表盘训练会话失败")

# 获取仪表盘情感数据
# Get dashboard emotion data
@app.get("/api/dashboard/emotion")
async def get_dashboard_emotion():
    """
    get_dashboard_emotion函数 - 获取仪表盘情感数据
    get_dashboard_emotion Function - Get dashboard emotion data
    
    Returns:
        情感分析数据 | Emotion analysis data
    """
    try:
        # 获取情感意识系统的当前状态
        emotion_state = emotion_system.get_current_state()
        
        return {
            "status": "success",
            "valence": emotion_state.get("valence", 0.5),
            "arousal": emotion_state.get("arousal", 0.5),
            "dominance": emotion_state.get("dominance", 0.5),
            "mood": emotion_state.get("mood", "neutral"),
            "intensity": emotion_state.get("intensity", 0.5)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘情感数据失败")
        raise HTTPException(status_code=500, detail="获取仪表盘情感数据失败")

# 获取仪表盘模型数据
# Get dashboard model data
@app.get("/api/dashboard/models")
async def get_dashboard_models():
    """
    get_dashboard_models函数 - 获取仪表盘模型数据
    get_dashboard_models Function - Get dashboard model data
    
    Returns:
        模型状态和协作数据 | Model status and collaboration data
    """
    try:
        model_statuses = model_registry.get_all_models_status()
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        collaboration_metrics = enhanced_metrics.get("collaboration_metrics", {})
        
        models = []
        for status in model_statuses:
            models.append({
                "id": status.get("id"),
                "name": status.get("name"),
                "type": status.get("type"),
                "status": status.get("status"),
                "version": status.get("version", "1.0.0"),
                "metrics": {
                    "cpu": status.get("cpu_usage", 0),
                    "memory": status.get("memory_usage", 0),
                    "responseTime": status.get("response_time", 0),
                    "successRate": status.get("success_rate", 0) * 100,
                    "throughput": status.get("throughput", 0)
                }
            })
        
        return {
            "status": "success",
            "models": models,
            "activeCollaborations": collaboration_metrics.get("active_collaborations", 0),
            "collaborationEfficiency": collaboration_metrics.get("efficiency", 0) * 100,
            "dataTransferRate": collaboration_metrics.get("data_transfer_rate", 0)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘模型数据失败")
        raise HTTPException(status_code=500, detail="获取仪表盘模型数据失败")

# 获取仪表盘详细模型指标
# Get dashboard detailed model metrics
@app.get("/api/dashboard/model-metrics")
async def get_dashboard_model_metrics():
    """
    get_dashboard_model_metrics函数 - 获取仪表盘详细模型指标
    get_dashboard_model_metrics Function - Get dashboard detailed model metrics
    
    Returns:
        详细的模型性能指标 | Detailed model performance metrics
    """
    try:
        model_statuses = model_registry.get_all_models_status()
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        collaboration_metrics = enhanced_metrics.get("collaboration_metrics", {})
        
        models = []
        for status in model_statuses:
            models.append({
                "id": status.get("id"),
                "name": status.get("name"),
                "type": status.get("type"),
                "status": status.get("status"),
                "version": status.get("version", "1.0.0"),
                "metrics": {
                    "cpu": status.get("cpu_usage", 0),
                    "memory": status.get("memory_usage", 0),
                    "responseTime": status.get("response_time", 0),
                    "successRate": status.get("success_rate", 0) * 100,
                    "throughput": status.get("throughput", 0)
                }
            })
        
        return {
            "status": "success",
            "models": models,
            "activeCollaborations": collaboration_metrics.get("active_collaborations", 0),
            "collaborationEfficiency": collaboration_metrics.get("efficiency", 0) * 100,
            "dataTransferRate": collaboration_metrics.get("data_transfer_rate", 0)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "获取仪表盘模型指标失败")
        raise HTTPException(status_code=500, detail="获取仪表盘模型指标失败")

# AGI系统状态端点
# AGI System status endpoints

# 获取AGI系统整体状态
# Get AGI system overall status
@app.get("/api/agi/status")
async def get_agi_system_status():
    """
    get_agi_system_status函数 - 获取AGI系统整体状态
    get_agi_system_status Function - Get AGI system overall status
    
    Returns:
        AGI系统状态信息 | AGI system status information
    """
    try:
        # 获取各组件状态
        cognitive_status = unified_cognitive_architecture.get_system_status()
        meta_cognition_status = enhanced_meta_cognition.get_system_status()
        knowledge_base_status = structured_knowledge_base.get_knowledge_base_status()
        motivation_status = intrinsic_motivation_system.get_motivation_report()
        explainability_status = explainable_ai.get_system_report()
        alignment_status = value_alignment.get_alignment_report()
        
        # 计算整体健康度
        health_scores = []
        if 'health_score' in cognitive_status:
            health_scores.append(cognitive_status['health_score'])
        if 'health_score' in meta_cognition_status:
            health_scores.append(meta_cognition_status['health_score'])
        if 'health_score' in knowledge_base_status:
            health_scores.append(knowledge_base_status['health_score'])
        if 'overall_motivation' in motivation_status:
            health_scores.append(motivation_status['overall_motivation'])
        if 'reliability_score' in explainability_status.get('confidence_calibration', {}):
            health_scores.append(explainability_status['confidence_calibration']['reliability_score'])
        if 'overall_alignment_score' in alignment_status.get('value_system', {}):
            health_scores.append(alignment_status['value_system']['overall_alignment_score'])
        
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0.5
        
        return {
            "status": "success",
            "data": {
                "cognitive_architecture": cognitive_status,
                "meta_cognition": meta_cognition_status,
                "knowledge_base": knowledge_base_status,
                "intrinsic_motivation": motivation_status,
                "explainable_ai": explainability_status,
                "value_alignment": alignment_status,
                "overall_health": overall_health,
                "health_status": "EXCELLENT" if overall_health > 0.8 else
                                "GOOD" if overall_health > 0.6 else
                                "FAIR" if overall_health > 0.4 else "POOR"
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "获取AGI系统状态失败")
        raise HTTPException(status_code=500, detail="获取AGI系统状态失败")

# 获取综合系统统计信息
# Get comprehensive system statistics
@app.get("/api/statistics")
async def get_system_statistics():
    """
    get_system_statistics函数 - 获取综合系统统计信息
    get_system_statistics Function - Get comprehensive system statistics
    
    Returns:
        系统统计信息，包括模型状态、训练统计、性能指标等
        System statistics including model status, training stats, performance metrics, etc.
    """
    try:
        # 获取模型状态
        model_statuses = model_registry.get_all_models_status()
        
        # 获取训练统计
        training_stats = {
            "total_jobs": len(training_manager.get_all_jobs_status()),
            "active_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                              if job.get("status") in ["running", "pending"]]),
            "completed_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                                 if job.get("status") == "completed"]),
            "failed_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                              if job.get("status") == "failed"])
        }
        
        # 获取数据集统计
        dataset_stats = dataset_manager.get_dataset_stats()
        
        # 获取系统性能指标
        performance_stats = system_monitor.get_performance_stats()
        
        # 获取增强监控指标
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        
        # 获取最近的日志
        recent_logs = system_monitor.get_recent_logs(limit=20)
        
        # 构建综合统计信息
        statistics = {
            "system": {
                "uptime": enhanced_metrics.get("base_metrics", {}).get("uptime", 0),
                "cpu_usage": enhanced_metrics.get("base_metrics", {}).get("cpu_usage", 0),
                "memory_usage": enhanced_metrics.get("base_metrics", {}).get("memory_usage", 0),
                "disk_usage": enhanced_metrics.get("base_metrics", {}).get("disk_usage", 0),
                "network_io": enhanced_metrics.get("base_metrics", {}).get("network_io", {})
            },
            "models": {
                "total_models": len(model_statuses),
                "active_models": len([m for m in model_statuses if m.get("status") == "active"]),
                "inactive_models": len([m for m in model_statuses if m.get("status") != "active"]),
                "model_types": list(set([m.get("type", "unknown") for m in model_statuses])),
                "details": model_statuses
            },
            "training": training_stats,
            "datasets": dataset_stats,
            "monitoring": {
                "task_metrics": enhanced_metrics.get("task_metrics", []),
                "emotion_metrics": enhanced_metrics.get("emotion_metrics", []),
                "collaboration_metrics": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": enhanced_metrics.get("data_streams", {})
            },
            "logs": recent_logs
        }
        
        return {"status": "success", "data": statistics}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "获取系统统计信息失败")
        raise HTTPException(status_code=500, detail="获取系统统计信息失败")

# 系统设置端点
# System settings endpoints

# 获取系统设置
# Get system settings
@app.get("/api/settings")
async def get_system_settings():
    """
    get_system_settings函数 - 获取系统设置
    get_system_settings Function - Get system settings
    
    Returns:
        系统设置 | System settings
    """
    try:
        settings = system_settings_manager.get_settings()
        return {"status": "success", "data": settings}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取系统设置失败")
        raise HTTPException(status_code=500, detail="获取系统设置失败")

# 更新系统设置
# Update system settings
@app.post("/api/settings/update")
async def update_system_settings(settings_data: dict):
    """
    update_system_settings函数 - 更新系统设置
    update_system_settings Function - Update system settings
    
    Args:
        settings_data: 新的设置数据 | New settings data
        
    Returns:
        更新结果 | Update result
    """
    try:
        result = system_settings_manager.update_settings(settings_data)
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "更新系统设置失败")
        raise HTTPException(status_code=500, detail="更新系统设置失败")

# 重置系统设置为默认值
# Reset system settings to default
@app.post("/api/settings/reset")
async def reset_system_settings():
    """
    reset_system_settings函数 - 重置系统设置为默认值
    reset_system_settings Function - Reset system settings to default
    
    Returns:
        重置结果 | Reset result
    """
    try:
        result = system_settings_manager.reset_to_default()
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "重置系统设置失败")
        raise HTTPException(status_code=500, detail="重置系统设置失败")

# 模型管理端点
# Model management endpoints

# 获取所有模型配置
# Get all models configuration
@app.get("/api/models")
async def get_all_models_config():
    """
    get_all_models_config函数 - 获取所有模型配置
    get_all_models_config Function - Get all models configuration
    
    Returns:
        模型配置列表 | List of model configurations
    """
    try:
        models_config = system_settings_manager.get_settings().get("models", {})
        models_status = model_registry.get_all_models_status()
        models_mode = get_all_models_mode()
        
        # 合并配置、状态和模式信息
        result = []
        for model_id, config in models_config.items():
            status = next((s for s in models_status if s["id"] == model_id), {})
            mode = models_mode.get(model_id, "local")
            
            result.append({
                "id": model_id,
                "name": config.get("name", model_id),
                "type": config.get("type", "unknown"),
                "source": config.get("source", "local"),
                "mode": mode,
                "active": config.get("active", True),
                "status": status.get("status", "unknown"),
                "version": status.get("version", "1.0.0"),
                "api_config": config.get("api_config", {})
            })
        
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "获取模型配置失败")
        raise HTTPException(status_code=500, detail="获取模型配置失败")

# 添加新模型
# Add new model
@app.post("/api/models")
async def add_model(model_data: dict):
    """
    add_model函数 - 添加新模型
    add_model Function - Add new model
    
    Args:
        model_data: 模型数据 | Model data
            - id: 模型ID | Model ID
            - name: 模型名称 | Model name
            - type: 模型类型 | Model type
            - source: 模型来源 (local/api) | Model source
            - active: 是否激活 | Whether active
            - api_config: API配置 (如果是API模型) | API configuration
    
    Returns:
        添加结果 | Addition result
    """
    try:
        model_id = model_data.get("id")
        if not model_id:
            raise HTTPException(status_code=400, detail="模型ID不能为空")
        
        # 准备模型配置
        model_config = {
            "name": model_data.get("name", model_id),
            "type": model_data.get("type", "unknown"),
            "source": model_data.get("source", "local"),
            "active": model_data.get("active", True)
        }
        
        # 如果是API模型，添加API配置
        if model_data.get("source") == "api":
            api_config = model_data.get("api_config", {})
            model_config["api_config"] = api_config
        
        # 保存模型配置
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # 如果是激活状态，加载模型
        if model_config["active"]:
            if model_config["source"] == "api":
                switch_model_to_external(model_id, model_config["api_config"])
            else:
                try:
                    model_registry.load_model(model_id)
                except Exception as load_error:
                    error_handler.handle_error(load_error, "API", f"加载模型 {model_id} 失败")
        
        return {"status": "success", "message": f"模型 {model_id} 添加成功"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", "添加模型失败")
        raise HTTPException(status_code=500, detail=str(e))

# 更新模型配置
# Update model configuration
@app.put("/api/models")
async def update_models_config(models_data: list):
    """
    update_models_config函数 - 更新模型配置
    update_models_config Function - Update model configuration
    
    Args:
        models_data: 模型数据列表 | List of model data
    
    Returns:
        更新结果 | Update result
    """
    try:
        updated_count = 0
        
        for model_data in models_data:
            model_id = model_data.get("id")
            if not model_id:
                continue
            
            # 获取现有配置
            current_config = system_settings_manager.get_model_setting(model_id)
            if not current_config:
                current_config = {}
            
            # 更新配置
            updated_config = {
                **current_config,
                "name": model_data.get("name", current_config.get("name", model_id)),
                "type": model_data.get("type", current_config.get("type", "unknown")),
                "source": model_data.get("source", current_config.get("source", "local")),
                "active": model_data.get("active", current_config.get("active", True))
            }
            
            # 更新API配置
            if model_data.get("source") == "api":
                updated_config["api_config"] = model_data.get("api_config", {})
            
            # 保存更新后的配置
            system_settings_manager.update_model_setting(model_id, updated_config)
            updated_count += 1
            
            # 根据新模式和激活状态处理模型
            current_mode = get_model_mode(model_id)
            new_source = updated_config["source"]
            
            if updated_config["active"]:
                if new_source == "api" and current_mode != "external":
                    # 切换到外部API模式
                    switch_model_to_external(model_id, updated_config["api_config"])
                elif new_source == "local" and current_mode != "local":
                    # 切换到本地模式
                    switch_model_to_local(model_id)
            else:
                # 停用模型
                model_registry.unload_model(model_id)
        
        return {"status": "success", "updated_count": updated_count, "message": "模型配置更新成功"}
    except Exception as e:
        error_handler.handle_error(e, "API", "更新模型配置失败")
        raise HTTPException(status_code=500, detail=str(e))

# 更新单个模型配置
# Update single model configuration
@app.patch("/api/models/{model_id}")
async def update_model_config(model_id: str, model_data: dict):
    """
    update_model_config函数 - 更新单个模型配置
    update_model_config Function - Update single model configuration
    
    Args:
        model_id: 模型ID | Model ID
        model_data: 模型数据 | Model data
    
    Returns:
        更新结果 | Update result
    """
    try:
        # 获取现有配置
        current_config = system_settings_manager.get_model_setting(model_id)
        if not current_config:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
        
        # 更新配置
        updated_config = {**current_config, **model_data}
        
        # 保存更新后的配置
        system_settings_manager.update_model_setting(model_id, updated_config)
        
        # 根据新模式和激活状态处理模型
        if "active" in model_data:
            if model_data["active"]:
                if updated_config.get("source") == "api":
                    switch_model_to_external(model_id, updated_config.get("api_config", {}))
                else:
                    model_registry.load_model(model_id)
            else:
                model_registry.unload_model(model_id)
        
        return {"status": "success", "message": f"模型 {model_id} 配置更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"更新模型 {model_id} 配置失败")
        raise HTTPException(status_code=500, detail=str(e))

# 删除模型
# Delete model
@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """
    delete_model函数 - 删除模型
    delete_model Function - Delete model
    
    Args:
        model_id: 模型ID | Model ID
    
    Returns:
        删除结果 | Delete result
    """
    try:
        # 卸载模型
        model_registry.unload_model(model_id)
        
        # 从系统设置中删除配置
        success = system_settings_manager.delete_model_setting(model_id)
        
        if success:
            return {"status": "success", "message": f"模型 {model_id} 删除成功"}
        else:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"删除模型 {model_id} 失败")
        raise HTTPException(status_code=500, detail=str(e))

# 切换模型到外部API模式
# Switch model to external API mode
@app.post("/api/models/{model_id}/switch-to-external")
async def api_switch_to_external(model_id: str, api_config: dict):
    """
    api_switch_to_external函数 - 切换模型到外部API模式
    api_switch_to_external Function - Switch model to external API mode
    
    Args:
        model_id: 模型ID | Model ID
        api_config: API配置 | API configuration
    
    Returns:
        切换结果 | Switch result
    """
    try:
        result = switch_model_to_external(model_id, api_config)
        return {"status": "success", "message": result}
    except Exception as e:
        error_handler.handle_error(e, "API", f"切换模型 {model_id} 到外部API模式失败")
        raise HTTPException(status_code=500, detail=str(e))

# 切换模型到本地模式
# Switch model to local mode
@app.post("/api/models/{model_id}/switch-to-local")
async def api_switch_to_local(model_id: str):
    """
    api_switch_to_local函数 - 切换模型到本地模式
    api_switch_to_local Function - Switch model to local mode
    
    Args:
        model_id: 模型ID | Model ID
    
    Returns:
        切换结果 | Switch result
    """
    try:
        result = switch_model_to_local(model_id)
        return {"status": "success", "message": result}
    except Exception as e:
        error_handler.handle_error(e, "API", f"切换模型 {model_id} 到本地模式失败")
        raise HTTPException(status_code=500, detail=str(e))

# 获取模型运行模式
# Get model running mode
@app.get("/api/models/{model_id}/mode")
async def api_get_model_mode(model_id: str):
    """
    api_get_model_mode函数 - 获取模型运行模式
    api_get_model_mode Function - Get model running mode
    
    Args:
        model_id: 模型ID | Model ID
    
    Returns:
        模型运行模式 | Model running mode
    """
    try:
        mode = get_model_mode(model_id)
        return {"status": "success", "mode": mode}
    except Exception as e:
        error_handler.handle_error(e, "API", f"获取模型 {model_id} 运行模式失败")
        raise HTTPException(status_code=500, detail=str(e))

# 批量切换模型模式
# Batch switch model modes
@app.post("/api/models/batch-switch-mode")
async def api_batch_switch_mode(models_data: list):
    """
    api_batch_switch_mode函数 - 批量切换模型模式
    api_batch_switch_mode Function - Batch switch model modes
    
    Args:
        models_data: 模型数据列表 | List of model data
            - id: 模型ID | Model ID
            - mode: 目标模式 (local/external) | Target mode
            - api_config: API配置 (如果切换到external模式) | API configuration
    
    Returns:
        批量切换结果 | Batch switch result
    """
    try:
        results = batch_switch_model_modes(models_data)
        return {"status": "success", "results": results}
    except Exception as e:
        error_handler.handle_error(e, "API", "批量切换模型模式失败")
        raise HTTPException(status_code=500, detail=str(e))

# 测试外部API连接
# Test external API connection
@app.post("/api/models/test-connection")
async def api_test_connection(connection_data: dict):
    """
    api_test_connection函数 - 测试外部API连接
    api_test_connection Function - Test external API connection
    
    Args:
        connection_data: 连接测试数据 | Connection test data
            - model_id: 模型ID | Model ID
            - api_url: API URL
            - api_key: API密钥 | API key
            - model_name: 模型名称 | Model name
            - api_type: API类型 | API type
    
    Returns:
        连接测试结果 | Connection test result
    """
    try:
        result = test_external_api_connection(connection_data)
        return result
    except Exception as e:
        error_handler.handle_error(e, "API", "测试外部API连接失败")
        return {"status": "error", "message": str(e)}

# 错误处理中间件
# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """
    error_handling_middleware函数 - 错误处理中间件
    error_handling_middleware Function - Error handling middleware
    
    Args:
        request: 请求对象 | Request object
        call_next: 下一个中间件函数 | Next middleware function
        
    Returns:
        响应对象 | Response object
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_handler.handle_error(e, "Middleware", f"请求处理错误: {request.url}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "内部服务器错误", "detail": str(e)}
        )

# 主函数
# Main function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
