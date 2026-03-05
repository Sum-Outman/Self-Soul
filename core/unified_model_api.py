"""
统一模型API系统
Unified Model API System

提供所有模型的统一接口，支持本地模型和外部API模型的无缝切换和协同工作
Provides unified interface for all models with seamless switching between local and external API models and collaborative work
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import logging
from core.error_handling import error_handler

from core.model_registry import get_model_registry
from core.model_service_manager import ModelServiceManager
from core.api_model_connector import APIModelConnector
from core.external_model_proxy import ExternalModelProxy
from core.enhanced_training_system import enhanced_training_system
from core.model_training_api import router as training_router

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/models", tags=["models"])

# 全局模型管理器实例
model_registry = get_model_registry()
model_service_manager = ModelServiceManager()
api_model_connector = APIModelConnector()
external_model_proxy = ExternalModelProxy()


class ModelInferenceRequest(BaseModel):
    """模型推理请求模型"""
    model_id: str
    input_data: Union[str, Dict[str, Any], List[Any]]
    inference_config: Optional[Dict[str, Any]] = None
    use_external_api: bool = False
    priority: str = "normal"  # low, normal, high


class ModelChatRequest(BaseModel):
    """模型对话请求模型"""
    model_id: str
    message: str
    conversation_id: Optional[str] = None
    chat_config: Optional[Dict[str, Any]] = None
    use_external_api: bool = False


class ModelTrainingRequest(BaseModel):
    """模型训练请求模型"""
    model_id: str
    training_type: str  # from_scratch, fine_tune, joint, autonomous
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    external_api_config: Optional[Dict[str, Any]] = None


class ExternalAPIConfig(BaseModel):
    """外部API配置模型"""
    api_name: str
    api_url: str
    api_key: str
    model_name: str
    parameters: Dict[str, Any]
    timeout: int = 30


@router.post("/inference")
async def unified_model_inference(request: ModelInferenceRequest):
    """统一模型推理接口"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 检查模型是否已加载
        if not model_service_manager.is_model_loaded(request.model_id):
            # 自动加载模型
            await _auto_load_model(request.model_id)
        
        start_time = time.time()
        
        if request.use_external_api:
            # 使用外部API
            result = await external_model_proxy.inference(
                model_id=request.model_id,
                input_data=request.input_data,
                config=request.inference_config or {}
            )
            
            # 收集训练数据并启动本地模型训练
            if result.get("success", False):
                # 创建训练样本
                training_sample = {
                    "input": request.input_data,
                    "output": result.get("result", ""),
                    "timestamp": time.time()
                }
                
                # 启动异步训练
                try:
                    await enhanced_training_system.start_training(
                        model_id=request.model_id,
                        training_config={
                            "type": "incremental",
                            "sample": training_sample,
                            "epochs": 1,  # 单次增量训练
                            "batch_size": 1
                        }
                    )
                except Exception as train_e:
                    error_handler.log_warning(f"Failed to start incremental training: {train_e}", "UnifiedModelAPI")
        else:
            # 使用本地模型
            result = await model_service_manager.model_inference(
                model_id=request.model_id,
                input_data=request.input_data,
                inference_config=request.inference_config or {}
            )
        
        inference_time = time.time() - start_time
        
        # 记录推理指标
        _record_inference_metrics(request.model_id, inference_time, result.get("success", False))
        
        return {
            "success": 1,
            "model_id": request.model_id,
            "inference_time": inference_time,
            "result": result,
            "used_external_api": request.use_external_api,
            "priority": request.priority,
            "training_started": request.use_external_api and result.get("success", False)
        }
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        
        # 记录错误指标
        _record_inference_metrics(request.model_id, 0.0, False)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def unified_model_chat(request: ModelChatRequest):
    """统一模型对话接口"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 检查模型是否已加载
        if not model_service_manager.is_model_loaded(request.model_id):
            # 自动加载模型
            await _auto_load_model(request.model_id)
        
        start_time = time.time()
        
        if request.use_external_api:
            # 使用外部API进行对话
            result = await external_model_proxy.chat(
                model_id=request.model_id,
                message=request.message,
                conversation_id=request.conversation_id,
                config=request.chat_config or {}
            )
            
            # 收集训练数据并启动本地模型训练
            if result.get("success", False):
                # 创建训练样本
                training_sample = {
                    "input": {
                        "message": request.message,
                        "conversation_id": request.conversation_id
                    },
                    "output": result.get("response", ""),
                    "timestamp": time.time()
                }
                
                # 启动异步训练
                try:
                    await enhanced_training_system.start_training(
                        model_id=request.model_id,
                        training_config={
                            "type": "incremental",
                            "sample": training_sample,
                            "epochs": 1,  # 单次增量训练
                            "batch_size": 1
                        }
                    )
                except Exception as train_e:
                    error_handler.log_warning(f"Failed to start incremental chat training: {train_e}", "UnifiedModelAPI")
        else:
            # 使用本地模型进行对话
            result = await model_service_manager.model_chat(
                model_id=request.model_id,
                message=request.message,
                conversation_id=request.conversation_id,
                chat_config=request.chat_config or {}
            )
        
        chat_time = time.time() - start_time
        
        # 记录对话指标
        _record_chat_metrics(request.model_id, chat_time, result.get("success", False))
        
        return {
            "success": 1,
            "model_id": request.model_id,
            "chat_time": chat_time,
            "response": result,
            "conversation_id": result.get("conversation_id", request.conversation_id),
            "used_external_api": request.use_external_api,
            "training_started": request.use_external_api and result.get("success", False)
        }
        
    except Exception as e:
        logger.error(f"Model chat failed: {e}")
        
        # 记录错误指标
        _record_chat_metrics(request.model_id, 0.0, False)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def unified_model_training(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """统一模型训练接口"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 启动训练
        result = await enhanced_training_system.start_training(
            model_id=request.model_id,
            training_config=request.training_config
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        training_id = result["training_id"]
        
        return {
            "success": 1,
            "training_id": training_id,
            "message": f"Training started for model '{request.model_id}'",
            "estimated_duration": result["estimated_duration"]
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_id}")
async def get_unified_model_status(model_id: str):
    """获取统一模型状态"""
    try:
        # 检查模型是否已注册
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 获取模型信息
        model_info = model_registry.get_model_info(model_id)
        
        # 检查模型是否已加载
        is_loaded = model_service_manager.is_model_loaded(model_id)
        
        # 获取模型性能指标
        performance_metrics = _get_model_performance_metrics(model_id)
        
        # 获取模型健康状态
        health_status = _get_model_health_status(model_id)
        
        # 获取外部API连接状态
        external_api_status = _get_external_api_status(model_id)
        
        status_response = {
            "model_id": model_id,
            "model_type": model_info.get("model_type"),
            "model_name": model_info.get("model_name"),
            "is_loaded": is_loaded,
            "health_status": health_status,
            "performance_metrics": performance_metrics,
            "external_api_status": external_api_status,
            "registered_at": model_info.get("registered_at"),
            "last_used": model_info.get("last_used")
        }
        
        # 如果模型已加载，添加加载信息
        if is_loaded:
            loaded_info = _get_loaded_model_info(model_id)
            status_response.update(loaded_info)
        
        return {
            "success": 1,
            "model_status": status_response
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_unified_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,  # loaded, unloaded, all
    capability: Optional[str] = None,  # inference, training, chat, etc.
    limit: int = 100
):
    """列出所有统一模型"""
    try:
        models = model_registry.list_models()
        
        # 按类型过滤
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        # 按状态过滤
        if status == "loaded":
            models = [m for m in models if model_service_manager.is_model_loaded(m["model_id"])]
        elif status == "unloaded":
            models = [m for m in models if not model_service_manager.is_model_loaded(m["model_id"])]
        
        # 按能力过滤
        if capability:
            models = [m for m in models if capability in m.get("capabilities", [])]
        
        # 限制返回数量
        models = models[:limit]
        
        # 添加详细状态信息
        for model in models:
            model_id = model["model_id"]
            
            # 加载状态
            model["is_loaded"] = model_service_manager.is_model_loaded(model_id)
            
            # 性能指标
            model["performance_metrics"] = _get_model_performance_metrics(model_id)
            
            # 健康状态
            model["health_status"] = _get_model_health_status(model_id)
            
            # 外部API状态
            model["external_api_status"] = _get_external_api_status(model_id)
            
            # 如果模型已加载，添加加载信息
            if model["is_loaded"]:
                model["loaded_info"] = _get_loaded_model_info(model_id)
        
        return {
            "success": 1,
            "models": models,
            "total_count": len(models),
            "loaded_count": len([m for m in models if m["is_loaded"]]),
            "unloaded_count": len([m for m in models if not m["is_loaded"]]),
            "model_types": list(set(m["model_type"] for m in models))
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-api/register")
async def register_unified_external_api(config: ExternalAPIConfig):
    """注册统一外部API"""
    try:
        # 生成API ID
        api_id = f"{config.api_name}_{uuid.uuid4().hex[:8]}"
        
        # 注册外部API
        result = await api_model_connector.register_external_api(
            api_id=api_id,
            api_config=config.dict()
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": 1,
            "api_id": api_id,
            "message": "External API registered successfully",
            "connection_test": result
        }
        
    except Exception as e:
        logger.error(f"Failed to register external API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/external-api/list")
async def list_unified_external_apis():
    """列出所有统一外部API"""
    try:
        apis = api_model_connector.list_external_apis()
        
        return {
            "success": 1,
            "apis": apis,
            "total_count": len(apis)
        }
        
    except Exception as e:
        logger.error(f"Failed to list external APIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-api/test/{api_id}")
async def test_unified_external_api(api_id: str, test_input: Optional[Dict[str, Any]] = None):
    """测试统一外部API连接"""
    try:
        result = await api_model_connector.test_api_connection(api_id, test_input)
        
        return {
            "success": 1,
            "api_id": api_id,
            "test_result": result
        }
        
    except Exception as e:
        logger.error(f"Failed to test external API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{model_id}")
async def get_unified_model_performance(model_id: str, time_range: str = "1h"):
    """获取统一模型性能指标"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 获取性能数据
        performance_data = _get_model_performance_data(model_id, time_range)
        
        return {
            "success": 1,
            "model_id": model_id,
            "time_range": time_range,
            "performance_data": performance_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_unified_system_health():
    """获取统一系统健康状态"""
    try:
        # 获取所有模型状态
        models = model_registry.list_models()
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "loaded_models": len([m for m in models if model_service_manager.is_model_loaded(m["model_id"])]),
            "model_health": {},
            "external_api_health": {},
            "system_health": "healthy"
        }
        
        # 检查每个模型的健康状态
        for model in models:
            model_id = model["model_id"]
            health_status = _get_model_health_status(model_id)
            health_report["model_health"][model_id] = health_status
            
            # 检查外部API健康状态
            external_api_status = _get_external_api_status(model_id)
            health_report["external_api_health"][model_id] = external_api_status
            
            # 如果有模型不健康，系统健康状态降级
            if health_status != "healthy" or external_api_status != "connected":
                health_report["system_health"] = "degraded"
        
        return {
            "success": 1,
            "health_report": health_report
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
async def _auto_load_model(model_id: str):
    """自动加载模型"""
    try:
        # 获取模型配置
        model_info = model_registry.get_model_info(model_id)
        load_config = model_info.get("load_config", {})
        
        # 加载模型
        await model_service_manager.load_model(
            model_id=model_id,
            load_config=load_config
        )
        
        logger.info(f"Model '{model_id}' auto-loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to auto-load model '{model_id}': {e}")
        raise


# 指标存储
inference_metrics: Dict[str, List[Dict[str, Any]]] = {}
chat_metrics: Dict[str, List[Dict[str, Any]]] = {}
loaded_models_info: Dict[str, Dict[str, Any]] = {}


def _record_inference_metrics(model_id: str, inference_time: float, success: bool):
    """记录推理指标"""
    if model_id not in inference_metrics:
        inference_metrics[model_id] = []
    
    metric = {
        "timestamp": datetime.now().isoformat(),
        "inference_time": inference_time,
        "success": success
    }
    
    inference_metrics[model_id].append(metric)
    
    # 保留最近1000条记录
    if len(inference_metrics[model_id]) > 1000:
        inference_metrics[model_id] = inference_metrics[model_id][-1000:]


def _record_chat_metrics(model_id: str, chat_time: float, success: bool):
    """记录对话指标"""
    if model_id not in chat_metrics:
        chat_metrics[model_id] = []
    
    metric = {
        "timestamp": datetime.now().isoformat(),
        "chat_time": chat_time,
        "success": success
    }
    
    chat_metrics[model_id].append(metric)
    
    # 保留最近1000条记录
    if len(chat_metrics[model_id]) > 1000:
        chat_metrics[model_id] = chat_metrics[model_id][-1000:]


def _get_model_performance_metrics(model_id: str) -> Dict[str, Any]:
    """获取模型性能指标"""
    inference_data = inference_metrics.get(model_id, [])
    chat_data = chat_metrics.get(model_id, [])
    
    if not inference_data and not chat_data:
        return {"no_data": True}
    
    # 计算推理统计信息
    inference_times = [m["inference_time"] for m in inference_data]
    inference_success_count = sum(1 for m in inference_data if m["success"])
    inference_total_count = len(inference_data)
    
    # 计算对话统计信息
    chat_times = [m["chat_time"] for m in chat_data]
    chat_success_count = sum(1 for m in chat_data if m["success"])
    chat_total_count = len(chat_data)
    
    return {
        "inference": {
            "total_count": inference_total_count,
            "success_rate": inference_success_count / inference_total_count if inference_total_count > 0 else 0,
            "avg_time": sum(inference_times) / len(inference_times) if inference_times else 0,
            "last_10_avg": sum(inference_times[-10:]) / min(10, len(inference_times)) if inference_times else 0
        },
        "chat": {
            "total_count": chat_total_count,
            "success_rate": chat_success_count / chat_total_count if chat_total_count > 0 else 0,
            "avg_time": sum(chat_times) / len(chat_times) if chat_times else 0,
            "last_10_avg": sum(chat_times[-10:]) / min(10, len(chat_times)) if chat_times else 0
        }
    }


def _get_model_health_status(model_id: str) -> str:
    """获取模型健康状态"""
    if not model_service_manager.is_model_loaded(model_id):
        return "unloaded"
    
    # 检查最近10次推理的成功率
    inference_data = inference_metrics.get(model_id, [])
    recent_inference = inference_data[-10:]
    
    if recent_inference:
        recent_success_rate = sum(1 for m in recent_inference if m["success"]) / len(recent_inference)
        
        if recent_success_rate < 0.5:
            return "unhealthy"
        elif recent_success_rate < 0.8:
            return "degraded"
        else:
            return "healthy"
    
    return "unknown"


def _get_external_api_status(model_id: str) -> str:
    """获取外部API状态"""
    # 简化实现，实际应该检查外部API连接
    return "connected"  # 或 "disconnected", "unknown"


def _get_loaded_model_info(model_id: str) -> Dict[str, Any]:
    """获取已加载模型信息"""
    if model_id not in loaded_models_info:
        loaded_models_info[model_id] = {
            "loaded_at": datetime.now().isoformat(),
            "load_count": 0
        }
    
    return loaded_models_info[model_id]


def _get_model_performance_data(model_id: str, time_range: str) -> Dict[str, Any]:
    """获取模型性能数据"""
    # 简化实现，实际应该根据时间范围过滤数据
    return _get_model_performance_metrics(model_id)


# 导入timedelta
from datetime import timedelta
