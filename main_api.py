#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Self Soul Main API Server
Unified Cognitive Architecture Main API Gateway
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
import asyncio
import logging
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/main_api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SelfBrain_API')

# 创建FastAPI应用
app = FastAPI(
    title="Self Soul API Gateway",
    description="Unified Cognitive Architecture API Gateway",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# 数据模型
class TextRequest(BaseModel):
    text: str
    model_id: Optional[str] = "language"
    context: Optional[Dict[str, Any]] = None

class ModelStatus(BaseModel):
    id: str
    name: str
    type: str
    status: str  # active, inactive, error, connecting
    isActive: bool
    port: int
    performance: Optional[Dict[str, float]] = None
    last_updated: Optional[str] = None

class TrainingRequest(BaseModel):
    model_id: str
    parameters: Optional[Dict[str, Any]] = None

# 全局状态管理
class SystemState:
    def __init__(self):
        # 初始化模型状态
        self.models = self._initialize_models()
        self.active_connections = {}
        self.training_jobs = {}
        
    def _initialize_models(self) -> Dict[str, ModelStatus]:
        """初始化模型状态"""
        # 从配置文件加载模型配置
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'model_services_config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_ports = config.get('model_ports', {})
        except Exception as e:
            logger.error(f"Failed to load model config: {str(e)}")
            # 使用默认配置
            model_ports = {
                "manager": 8001,
                "language": 8002,
                "knowledge": 8003,
                "vision": 8004,
                "audio": 8005,
                "autonomous": 8006,
                "programming": 8007,
                "planning": 8008,
                "emotion": 8009,
                "spatial": 8010,
                "computer_vision": 8011,
                "sensor": 8012,
                "motion": 8013,
                "prediction": 8014,
                "advanced_reasoning": 8015,
                "data_fusion": 8016,
                "creative_problem_solving": 8017,
                "meta_cognition": 8018,
                "value_alignment": 8019
            }
        
        # 模型显示名称映射
        model_names = {
            "manager": "Manager Model",
            "language": "Language Model",
            "knowledge": "Knowledge Model",
            "vision": "Vision Model",
            "audio": "Audio Model",
            "autonomous": "Autonomous Model",
            "programming": "Programming Model",
            "planning": "Planning Model",
            "emotion": "Emotion Model",
            "spatial": "Spatial Model",
            "computer_vision": "Computer Vision Model",
            "sensor": "Sensor Model",
            "motion": "Motion Model",
            "prediction": "Prediction Model",
            "advanced_reasoning": "Advanced Reasoning Model",
            "data_fusion": "Data Fusion Model",
            "creative_problem_solving": "Creative Problem Solving Model",
            "meta_cognition": "Meta Cognition Model",
            "value_alignment": "Value Alignment Model"
        }
        
        # 初始化所有模型为inactive状态
        models = {}
        for model_id, port in model_ports.items():
            models[model_id] = ModelStatus(
                id=model_id,
                name=model_names.get(model_id, model_id.title().replace('_', ' ')),
                type=model_names.get(model_id, model_id.title().replace('_', ' ')),
                status="inactive",
                isActive=False,
                port=port,
                last_updated=datetime.utcnow().isoformat()
            )
        
        # 激活管理模型作为默认模型
        if "manager" in models:
            models["manager"].status = "active"
            models["manager"].isActive = True
            models["manager"].performance = {
                "success_rate": 1.0,
                "latency": 0.1,
                "accuracy": 1.0,
                "resource_usage": 10.0
            }
        
        return models
    
    def update_model_status(self, model_id: str, status: str, performance: Optional[Dict[str, float]] = None):
        """更新模型状态"""
        if model_id in self.models:
            self.models[model_id].status = status
            self.models[model_id].isActive = status == "active"
            if performance:
                self.models[model_id].performance = performance
            self.models[model_id].last_updated = datetime.utcnow().isoformat()
    
    def get_model(self, model_id: str) -> Optional[ModelStatus]:
        """获取模型状态"""
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[ModelStatus]:
        """获取所有模型状态"""
        return list(self.models.values())

# 创建系统状态实例
system_state = SystemState()

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

# 创建连接管理器实例
manager = ConnectionManager()

# 辅助函数
async def simulate_model_processing(input_data: str, model_id: str) -> Dict[str, Any]:
    """模拟模型处理过程"""
    # 简单的延迟模拟处理时间
    await asyncio.sleep(0.5 + len(input_data) * 0.01)
    
    # 基于模型类型生成不同的响应
    responses = {
        "manager": f"Manager Model has coordinated the processing of your request: {input_data}",
        "language": f"Language Model analysis: {input_data}",
        "knowledge": f"Knowledge Model retrieved information related to: {input_data}",
        "vision": f"Vision Model would analyze visual content related to: {input_data}",
        "audio": f"Audio Model would process sound content related to: {input_data}",
        "programming": f"Programming Model suggests code structure for: {input_data}",
        "planning": f"Planning Model has developed a strategy for: {input_data}",
        "emotion": f"Emotion Model analysis detects neutral sentiment in: {input_data}",
    }
    
    response_text = responses.get(model_id, f"Model {model_id} processed your request")
    
    return {
        "status": "success",
        "data": response_text,
        "model_id": model_id,
        "processing_time": round(len(input_data) * 0.01 + 0.5, 2)
    }

# API端点
@app.get("/health", tags=["System"])
async def health_check():
    """系统健康检查"""
    return JSONResponse(content={"status": "ok", "message": "Self Soul system is running normally"})

@app.get("/api/models/status", tags=["Models"])
async def get_models_status():
    """获取所有模型状态"""
    models = system_state.get_all_models()
    # 转换为字典列表以便JSON序列化
    models_data = [model.dict() for model in models]
    return {"status": "success", "models": models_data}

@app.post("/api/process/text", tags=["Processing"])
async def process_text(request: TextRequest):
    """处理文本输入"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    logger.info(f"Processing text with model {request.model_id}: {request.text[:100]}...")
    
    try:
        # 检查模型是否存在
        if not system_state.get_model(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # 模拟模型处理
        result = await simulate_model_processing(request.text, request.model_id)
        
        # 更新模型性能指标
        if result["status"] == "success":
            system_state.update_model_status(
                request.model_id,
                "active",
                {"success_rate": 0.98, "latency": result["processing_time"], "accuracy": 0.95, "resource_usage": 20.0}
            )
        
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@app.post("/api/process/image", tags=["Processing"])
async def process_image(image: UploadFile = File(...), lang: str = "en"):
    """处理图像上传及分析"""
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # 验证文件类型
    valid_image_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml']
    if image.content_type not in valid_image_types:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # 验证文件大小
    contents = await image.read()
    file_size = len(contents)
    max_size = 5 * 1024 * 1024  # 5MB
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
    
    logger.info(f"Processing image: {image.filename}, size: {file_size/1024:.2f} KB")
    
    # 模拟图像处理
    await asyncio.sleep(1.5)
    
    # 激活视觉模型
    system_state.update_model_status(
        "vision",
        "active",
        {"success_rate": 0.96, "latency": 1.5, "accuracy": 0.94, "resource_usage": 40.0}
    )
    
    return {
        "status": "success",
        "data": f"Image {image.filename} processed successfully. Vision Model detected content features.",
        "model_id": "vision",
        "file_info": {
            "name": image.filename,
            "type": image.content_type,
            "size": f"{file_size/1024:.2f} KB"
        }
    }

@app.post("/api/process/video", tags=["Processing"])
async def process_video(video: UploadFile = File(...), lang: str = "en"):
    """处理视频上传及分析"""
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # 验证文件类型
    valid_video_types = ['video/mp4', 'video/webm', 'video/avi', 'video/mov']
    if video.content_type not in valid_video_types:
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # 验证文件大小
    contents = await video.read()
    file_size = len(contents)
    max_size = 20 * 1024 * 1024  # 20MB
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds 20MB limit")
    
    logger.info(f"Processing video: {video.filename}, size: {file_size/1024:.2f} KB")
    
    # 模拟视频处理
    await asyncio.sleep(3.0)
    
    # 激活视觉和音频模型
    system_state.update_model_status(
        "vision",
        "active",
        {"success_rate": 0.95, "latency": 3.0, "accuracy": 0.93, "resource_usage": 60.0}
    )
    system_state.update_model_status(
        "audio",
        "active",
        {"success_rate": 0.97, "latency": 2.5, "accuracy": 0.96, "resource_usage": 50.0}
    )
    
    return {
        "status": "success",
        "data": f"Video {video.filename} processed successfully. Multimodal analysis completed.",
        "model_id": "vision,audio",
        "file_info": {
            "name": video.filename,
            "type": video.content_type,
            "size": f"{file_size/1024:.2f} KB"
        }
    }

@app.post("/api/training/start", tags=["Training"])
async def start_training(request: TrainingRequest):
    """启动模型训练"""
    # 检查模型是否存在
    if not system_state.get_model(request.model_id):
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    job_id = f"training_{request.model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 创建训练任务
    system_state.training_jobs[job_id] = {
        "job_id": job_id,
        "model_id": request.model_id,
        "status": "started",
        "start_time": datetime.utcnow().isoformat(),
        "parameters": request.parameters or {}
    }
    
    logger.info(f"Started training job {job_id} for model {request.model_id}")
    
    return {
        "status": "success",
        "job_id": job_id,
        "message": f"Training started successfully for model {request.model_id}",
        "model_id": request.model_id
    }

# WebSocket端点
@app.websocket("/ws/training-progress/{client_id}")
async def websocket_training_progress(websocket: WebSocket, client_id: str):
    """WebSocket端点：训练进度"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # 接收来自客户端的消息
            data = await websocket.receive_text()
            logger.debug(f"Received message from {client_id}: {data}")
            
            # 解析消息以获取job_id
            try:
                message = json.loads(data)
                job_id = message.get('job_id')
                
                if job_id and job_id in system_state.training_jobs:
                    # 模拟训练进度
                    progress = min(100, int(datetime.now().timestamp() % 100))
                    
                    await manager.send_personal_message(
                        json.dumps({
                            "job_id": job_id,
                            "progress": progress,
                            "status": "training" if progress < 100 else "completed",
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        client_id
                    )
                else:
                    await manager.send_personal_message(
                        json.dumps({"error": "Invalid or missing job_id"}),
                        client_id
                    )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    json.dumps({"error": str(e)}),
                    client_id
                )
                
            # 发送心跳包保持连接
            await asyncio.sleep(5)
            await manager.send_personal_message(
                json.dumps({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}),
                client_id
            )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
        manager.disconnect(client_id)

@app.websocket("/ws/models-monitor/{client_id}")
async def websocket_models_monitor(websocket: WebSocket, client_id: str):
    """WebSocket端点：模型监控"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # 发送模型状态更新
            models = system_state.get_all_models()
            models_data = [model.dict() for model in models]
            
            await manager.send_personal_message(
                json.dumps({
                    "type": "models_update",
                    "models": models_data,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                client_id
            )
            
            # 每10秒更新一次
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
        manager.disconnect(client_id)

@app.websocket("/ws/audio-stream/{client_id}")
async def websocket_audio_stream(websocket: WebSocket, client_id: str):
    """WebSocket端点：音频流处理"""
    await manager.connect(websocket, client_id)
    try:
        # 激活音频模型
        system_state.update_model_status(
            "audio",
            "active",
            {"success_rate": 0.98, "latency": 0.2, "accuracy": 0.97, "resource_usage": 30.0}
        )
        
        while True:
            # 接收音频数据（二进制）
            try:
                # 这里假设客户端发送的是文本格式的音频数据描述
                data = await websocket.receive_text()
                
                # 模拟音频处理
                await asyncio.sleep(0.1)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "audio_processed",
                        "status": "success",
                        "message": "Audio data processed successfully",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    client_id
                )
            except Exception as e:
                logger.error(f"Error processing audio data: {str(e)}")
                await manager.send_personal_message(
                    json.dumps({"error": str(e)}),
                    client_id
                )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
        manager.disconnect(client_id)

# Startup event
@app.on_event("startup")
async def startup_event():
    """System startup event"""
    logger.info("Self Soul Main API Server is starting up...")
    
    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Load all model configurations
    logger.info(f"Loaded {len(system_state.models)} models from configuration")
    
    # Activate manager model
    logger.info("Manager Model activated as the primary coordination model")
    
    logger.info("Self Soul Main API Server started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """System shutdown event"""
    logger.info("Self Soul Main API Server is shutting down...")
    
    # Clean up active connections
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except:
            pass
        manager.disconnect(client_id)
    
    logger.info("Self Soul Main API Server shut down successfully")

if __name__ == "__main__":
    # 从配置文件加载端口配置
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'model_services_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            port = config.get('main_api', {}).get('port', 8000)
            host = config.get('main_api', {}).get('host', '0.0.0.0')
    except Exception as e:
        logger.error(f"Failed to load port config: {str(e)}")
        port = 8000
        host = '0.0.0.0'
    
    # 启动服务器
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)