"""
模型管理API系统
Model Management API System

提供完整的模型管理功能，包括模型加载、卸载、配置、监控和接口调用
Provides complete model management functionality including model loading, unloading, configuration, monitoring, and interface calls
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

from core.model_registry import get_model_registry
from core.model_service_manager import ModelServiceManager
from core.api_model_connector import APIModelConnector
from core.external_model_proxy import ExternalModelProxy
from core.monitoring_enhanced import EnhancedSystemMonitor

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
enhanced_monitor = EnhancedSystemMonitor()


class ModelLoadRequest(BaseModel):
    """模型加载请求模型"""
    model_id: str
    load_config: Dict[str, Any]
    port: Optional[int] = None
    external_api_config: Optional[Dict[str, Any]] = None


class ModelInferenceRequest(BaseModel):
    """模型推理请求模型"""
    model_id: str
    input_data: Union[str, Dict[str, Any], List[Any]]
    inference_config: Optional[Dict[str, Any]] = None
    use_external_api: bool = False


class ModelConfigUpdate(BaseModel):
    """模型配置更新模型"""
    model_id: str
    config_updates: Dict[str, Any]
    restart_required: bool = False


class ExternalAPIConfig(BaseModel):
    """外部API配置模型"""
    api_name: str
    api_url: str
    api_key: str
    model_name: str
    parameters: Dict[str, Any]
    timeout: int = 30


@router.post("/load")
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """加载模型"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 检查模型是否已加载
        if model_service_manager.is_model_loaded(request.model_id):
            raise HTTPException(status_code=400, detail=f"Model '{request.model_id}' is already loaded")
        
        # 在后台加载模型
        async def load_model_async():
            try:
                # 加载模型
                result = await model_service_manager.load_model(
                    model_id=request.model_id,
                    load_config=request.load_config,
                    port=request.port,
                    external_api_config=request.external_api_config
                )
                
                # 更新模型状态
                loaded_models[request.model_id]["status"] = "loaded"
                loaded_models[request.model_id]["port"] = result.get("port")
                loaded_models[request.model_id]["loaded_at"] = datetime.now().isoformat()
                loaded_models[request.model_id]["external_api"] = bool(request.external_api_config)
                
                logger.info(f"Model '{request.model_id}' loaded successfully")
                
            except Exception as e:
                loaded_models[request.model_id]["status"] = "failed"
                loaded_models[request.model_id]["error"] = str(e)
                loaded_models[request.model_id]["failed_at"] = datetime.now().isoformat()
                logger.error(f"Failed to load model '{request.model_id}': {e}")
        
        # 启动后台任务
        background_tasks.add_task(load_model_async)
        
        # 记录模型加载状态
        loaded_models[request.model_id] = {
            "model_id": request.model_id,
            "status": "loading",
            "load_config": request.load_config,
            "external_api_config": request.external_api_config,
            "started_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "model_id": request.model_id,
            "message": f"Model '{request.model_id}' is being loaded",
            "estimated_load_time": _estimate_load_time(request.load_config)
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload/{model_id}")
async def unload_model(model_id: str):
    """卸载模型"""
    try:
        # 验证模型是否已加载
        if not model_service_manager.is_model_loaded(model_id):
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not loaded")
        
        # 卸载模型
        result = await model_service_manager.unload_model(model_id)
        
        # 更新模型状态
        if model_id in loaded_models:
            loaded_models[model_id]["status"] = "unloaded"
            loaded_models[model_id]["unloaded_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "model_id": model_id,
            "message": f"Model '{model_id}' unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference")
async def model_inference(request: ModelInferenceRequest):
    """模型推理"""
    try:
        # 验证模型是否已加载
        if not model_service_manager.is_model_loaded(request.model_id):
            raise HTTPException(status_code=400, detail=f"Model '{request.model_id}' is not loaded")
        
        # 执行推理
        start_time = time.time()
        
        if request.use_external_api:
            # 使用外部API
            result = await external_model_proxy.inference(
                model_id=request.model_id,
                input_data=request.input_data,
                config=request.inference_config or {}
            )
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
            "success": True,
            "model_id": request.model_id,
            "inference_time": inference_time,
            "result": result,
            "used_external_api": request.use_external_api
        }
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        
        # 记录错误指标
        _record_inference_metrics(request.model_id, 0.0, False)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_id}")
async def get_model_status(model_id: str):
    """获取模型状态"""
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
        
        status_response = {
            "model_id": model_id,
            "model_type": model_info.get("model_type"),
            "model_name": model_info.get("model_name"),
            "is_loaded": is_loaded,
            "health_status": health_status,
            "performance_metrics": performance_metrics,
            "registered_at": model_info.get("registered_at"),
            "last_used": model_info.get("last_used")
        }
        
        # 如果模型已加载，添加加载信息
        if is_loaded:
            loaded_info = loaded_models.get(model_id, {})
            status_response.update({
                "loaded_at": loaded_info.get("loaded_at"),
                "port": loaded_info.get("port"),
                "external_api": loaded_info.get("external_api", False),
                "load_config": loaded_info.get("load_config")
            })
        
        return {
            "success": True,
            "model_status": status_response
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,  # loaded, unloaded, all
    limit: int = 100
):
    """列出所有模型"""
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
        
        # 限制返回数量
        models = models[:limit]
        
        # 添加加载状态信息
        for model in models:
            model_id = model["model_id"]
            model["is_loaded"] = model_service_manager.is_model_loaded(model_id)
            
            if model["is_loaded"] and model_id in loaded_models:
                model["loaded_info"] = loaded_models[model_id]
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models),
            "loaded_count": len([m for m in models if m["is_loaded"]]),
            "unloaded_count": len([m for m in models if not m["is_loaded"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update")
async def update_model_config(request: ModelConfigUpdate):
    """更新模型配置"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(request.model_id):
            raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
        
        # 更新配置
        result = model_registry.update_model_config(
            model_id=request.model_id,
            config_updates=request.config_updates
        )
        
        # 如果需要重启，卸载并重新加载模型
        if request.restart_required and model_service_manager.is_model_loaded(request.model_id):
            await model_service_manager.unload_model(request.model_id)
            
            # 重新加载模型
            model_info = model_registry.get_model_info(request.model_id)
            load_config = model_info.get("load_config", {})
            
            await model_service_manager.load_model(
                model_id=request.model_id,
                load_config=load_config
            )
        
        return {
            "success": True,
            "model_id": request.model_id,
            "message": "Model configuration updated successfully",
            "restart_performed": request.restart_required
        }
        
    except Exception as e:
        logger.error(f"Failed to update model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-api/register")
async def register_external_api(config: ExternalAPIConfig):
    """注册外部API"""
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
            "success": True,
            "api_id": api_id,
            "message": "External API registered successfully",
            "connection_test": result
        }
        
    except Exception as e:
        logger.error(f"Failed to register external API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/external-api/list")
async def list_external_apis():
    """列出所有外部API"""
    try:
        apis = api_model_connector.list_external_apis()
        
        return {
            "success": True,
            "apis": apis,
            "total_count": len(apis)
        }
        
    except Exception as e:
        logger.error(f"Failed to list external APIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/external-api/test/{api_id}")
async def test_external_api(api_id: str, test_input: Optional[Dict[str, Any]] = None):
    """测试外部API连接"""
    try:
        result = await api_model_connector.test_api_connection(api_id, test_input)
        
        return {
            "success": True,
            "api_id": api_id,
            "test_result": result
        }
        
    except Exception as e:
        logger.error(f"Failed to test external API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{model_id}")
async def get_model_performance(model_id: str, time_range: str = "1h"):
    """获取模型性能指标"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 获取性能指标
        performance_data = _get_model_performance_data(model_id, time_range)
        
        return {
            "success": True,
            "model_id": model_id,
            "time_range": time_range,
            "performance_data": performance_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_system_health():
    """获取系统健康状态"""
    try:
        # 获取所有模型状态
        models = model_registry.list_models()
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "loaded_models": len([m for m in models if model_service_manager.is_model_loaded(m["model_id"])]),
            "model_health": {},
            "system_health": "healthy"
        }
        
        # 检查每个模型的健康状态
        for model in models:
            model_id = model["model_id"]
            health_status = _get_model_health_status(model_id)
            health_report["model_health"][model_id] = health_status
            
            # 如果有模型不健康，系统健康状态降级
            if health_status != "healthy":
                health_report["system_health"] = "degraded"
        
        return {
            "success": True,
            "health_report": health_report
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart/{model_id}")
async def restart_model(model_id: str):
    """重启模型"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 如果模型已加载，先卸载
        if model_service_manager.is_model_loaded(model_id):
            await model_service_manager.unload_model(model_id)
        
        # 重新加载模型
        model_info = model_registry.get_model_info(model_id)
        load_config = model_info.get("load_config", {})
        
        result = await model_service_manager.load_model(
            model_id=model_id,
            load_config=load_config
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model restarted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to restart model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 已加载模型存储
loaded_models: Dict[str, Dict[str, Any]] = {}

# 推理指标存储
inference_metrics: Dict[str, List[Dict[str, Any]]] = {}


def _estimate_load_time(load_config: Dict[str, Any]) -> int:
    """估计模型加载时间（秒）"""
    model_size = load_config.get("model_size_mb", 100)
    complexity = load_config.get("complexity", "medium")
    
    # 简单的估计公式
    base_time = 10  # 基础加载时间
    size_factor = model_size / 100  # 每100MB增加1倍时间
    
    if complexity == "low":
        complexity_factor = 0.5
    elif complexity == "high":
        complexity_factor = 2.0
    else:
        complexity_factor = 1.0
    
    total_time = base_time * size_factor * complexity_factor
    return int(total_time)


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


def _get_model_performance_metrics(model_id: str) -> Dict[str, Any]:
    """获取模型性能指标"""
    if model_id not in inference_metrics:
        return {"no_data": True}
    
    metrics = inference_metrics[model_id]
    
    if not metrics:
        return {"no_data": True}
    
    # 计算基本统计信息
    inference_times = [m["inference_time"] for m in metrics]
    success_count = sum(1 for m in metrics if m["success"])
    total_count = len(metrics)
    
    return {
        "total_inferences": total_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
        "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0,
        "max_inference_time": max(inference_times) if inference_times else 0,
        "min_inference_time": min(inference_times) if inference_times else 0,
        "last_10_avg": sum(inference_times[-10:]) / min(10, len(inference_times)) if inference_times else 0
    }


def _get_model_health_status(model_id: str) -> str:
    """获取模型健康状态"""
    if not model_service_manager.is_model_loaded(model_id):
        return "unloaded"
    
    # 简单的健康检查
    if model_id not in inference_metrics:
        return "unknown"
    
    metrics = inference_metrics[model_id]
    
    if not metrics:
        return "unknown"
    
    # 检查最近10次推理的成功率
    recent_metrics = metrics[-10:]
    if recent_metrics:
        recent_success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
        
        if recent_success_rate < 0.5:
            return "unhealthy"
        elif recent_success_rate < 0.8:
            return "degraded"
        else:
            return "healthy"
    
    return "unknown"


def _get_model_performance_data(model_id: str, time_range: str) -> Dict[str, Any]:
    """获取模型性能数据"""
    if model_id not in inference_metrics:
        return {"no_data": True}
    
    metrics = inference_metrics[model_id]
    
    if not metrics:
        return {"no_data": True}
    
    # 根据时间范围过滤数据
    if time_range == "1h":
        cutoff_time = datetime.now() - timedelta(hours=1)
    elif time_range == "24h":
        cutoff_time = datetime.now() - timedelta(hours=24)
    elif time_range == "7d":
        cutoff_time = datetime.now() - timedelta(days=7)
    else:
        cutoff_time = datetime.now() - timedelta(hours=1)
    
    filtered_metrics = [
        m for m in metrics 
        if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
    ]
    
    if not filtered_metrics:
        return {"no_data_in_time_range": True}
    
    # 计算性能指标
    inference_times = [m["inference_time"] for m in filtered_metrics]
    success_count = sum(1 for m in filtered_metrics if m["success"])
    
    return {
        "time_range": time_range,
        "data_points": len(filtered_metrics),
        "success_rate": success_count / len(filtered_metrics) if filtered_metrics else 0,
        "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0,
        "p95_inference_time": _calculate_percentile(inference_times, 95) if inference_times else 0,
        "p99_inference_time": _calculate_percentile(inference_times, 99) if inference_times else 0,
        "throughput": len(filtered_metrics) / (3600 if time_range == "1h" else 86400 if time_range == "24h" else 604800)
    }


def _calculate_percentile(values: List[float], percentile: float) -> float:
    """计算百分位数"""
    if not values:
        return 0.0
    
    import math
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1
# 前端需要的额外API端点
@router.post("/")
async def add_model(model_data: Dict[str, Any]):
    """添加新模型（前端兼容）"""
    try:
        # 验证必要字段
        required_fields = ["id", "name", "type", "status"]
        for field in required_fields:
            if field not in model_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # 生成模型ID
        model_id = model_data["id"]
        
        # 检查模型是否已存在
        if model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' already exists")
        
        # 注册模型
        model_info = {
            "model_id": model_id,
            "model_name": model_data["name"],
            "model_type": model_data["type"],
            "status": model_data["status"],
            "registered_at": datetime.now().isoformat(),
            "config": model_data.get("config", {})
        }
        
        model_registry.register_model(model_id, model_info)
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型（前端兼容）"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 如果模型已加载，先卸载
        if model_service_manager.is_model_loaded(model_id):
            await model_service_manager.unload_model(model_id)
        
        # 从注册表中删除
        model_registry.unregister_model(model_id)
        
        # 从已加载模型中删除
        if model_id in loaded_models:
            del loaded_models[model_id]
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/start")
async def start_model(model_id: str):
    """启动模型（前端兼容）"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 检查模型是否已加载
        if model_service_manager.is_model_loaded(model_id):
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is already loaded")
        
        # 获取模型配置
        model_info = model_registry.get_model_info(model_id)
        load_config = model_info.get("load_config", {})
        
        # 加载模型
        result = await model_service_manager.load_model(
            model_id=model_id,
            load_config=load_config
        )
        
        # 更新已加载模型状态
        loaded_models[model_id] = {
            "model_id": model_id,
            "status": "loaded",
            "port": result.get("port"),
            "loaded_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/stop")
async def stop_model(model_id: str):
    """停止模型（前端兼容）"""
    try:
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 检查模型是否已加载
        if not model_service_manager.is_model_loaded(model_id):
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not loaded")
        
        # 卸载模型
        await model_service_manager.unload_model(model_id)
        
        # 更新已加载模型状态
        if model_id in loaded_models:
            loaded_models[model_id]["status"] = "unloaded"
            loaded_models[model_id]["unloaded_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "model_id": model_id,
            "message": "Model stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-all")
async def start_all_models():
    """启动所有模型（前端兼容）"""
    try:
        models = model_registry.list_models()
        unloaded_models = [m for m in models if not model_service_manager.is_model_loaded(m["model_id"])]
        
        results = []
        
        for model in unloaded_models:
            try:
                model_id = model["model_id"]
                load_config = model.get("load_config", {})
                
                result = await model_service_manager.load_model(
                    model_id=model_id,
                    load_config=load_config
                )
                
                # 更新已加载模型状态
                loaded_models[model_id] = {
                    "model_id": model_id,
                    "status": "loaded",
                    "port": result.get("port"),
                    "loaded_at": datetime.now().isoformat()
                }
                
                results.append({
                    "model_id": model_id,
                    "success": True,
                    "message": "Model started successfully"
                })
                
            except Exception as e:
                results.append({
                    "model_id": model["model_id"],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total_models": len(models),
            "started_models": len([r for r in results if r["success"]]),
            "failed_models": len([r for r in results if not r["success"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to start all models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-all")
async def stop_all_models():
    """停止所有模型（前端兼容）"""
    try:
        models = model_registry.list_models()
        loaded_model_ids = [m["model_id"] for m in models if model_service_manager.is_model_loaded(m["model_id"])]
        
        results = []
        
        for model_id in loaded_model_ids:
            try:
                await model_service_manager.unload_model(model_id)
                
                # 更新已加载模型状态
                if model_id in loaded_models:
                    loaded_models[model_id]["status"] = "unloaded"
                    loaded_models[model_id]["unloaded_at"] = datetime.now().isoformat()
                
                results.append({
                    "model_id": model_id,
                    "success": True,
                    "message": "Model stopped successfully"
                })
                
            except Exception as e:
                results.append({
                    "model_id": model_id,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total_models": len(models),
            "stopped_models": len([r for r in results if r["success"]]),
            "failed_models": len([r for r in results if not r["success"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to stop all models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart-all")
async def restart_all_models():
    """重启所有模型（前端兼容）"""
    try:
        # 先停止所有模型
        stop_result = await stop_all_models()
        
        # 等待一段时间
        await asyncio.sleep(2)
        
        # 再启动所有模型
        start_result = await start_all_models()
        
        return {
            "success": True,
            "stop_results": stop_result,
            "start_results": start_result,
            "message": "All models restarted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to restart all models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 模型状态管理端点（前端兼容）
@router.put("/{model_id}/activation")
async def update_model_activation(model_id: str, activation_data: Dict[str, Any]):
    """更新模型激活状态（前端兼容）"""
    try:
        is_active = activation_data.get("isActive", False)
        
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 更新模型配置
        model_registry.update_model_config(
            model_id=model_id,
            config_updates={"is_active": is_active}
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "is_active": is_active,
            "message": "Model activation updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update model activation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{model_id}/primary")
async def set_primary_model(model_id: str, primary_data: Dict[str, Any]):
    """设置主模型（前端兼容）"""
    try:
        is_primary = primary_data.get("isPrimary", False)
        
        # 验证模型是否存在
        if not model_registry.is_model_registered(model_id):
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        # 如果设置为primary，先取消其他模型的primary状态
        if is_primary:
            models = model_registry.list_models()
            for model in models:
                if model["model_id"] != model_id:
                    model_registry.update_model_config(
                        model_id=model["model_id"],
                        config_updates={"is_primary": False}
                    )
        
        # 更新当前模型的primary状态
        model_registry.update_model_config(
            model_id=model_id,
            config_updates={"is_primary": is_primary}
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "is_primary": is_primary,
            "message": "Primary model status updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to set primary model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 批量更新模型配置（前端兼容）
@router.put("/")
async def update_all_models(models_data: List[Dict[str, Any]]):
    """批量更新模型配置（前端兼容）"""
    try:
        results = []
        
        for model_data in models_data:
            try:
                model_id = model_data.get("id")
                
                # 验证模型是否存在
                if not model_registry.is_model_registered(model_id):
                    results.append({
                        "model_id": model_id,
                        "success": False,
                        "error": "Model not found"
                    })
                    continue
                
                # 更新模型配置
                config_updates = {k: v for k, v in model_data.items() if k != "id"}
                model_registry.update_model_config(
                    model_id=model_id,
                    config_updates=config_updates
                )
                
                results.append({
                    "model_id": model_id,
                    "success": True,
                    "message": "Model configuration updated successfully"
                })
                
            except Exception as e:
                results.append({
                    "model_id": model_data.get("id", "unknown"),
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total_updated": len([r for r in results if r["success"]]),
            "total_failed": len([r for r in results if not r["success"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to update all models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# 将辅助方法绑定到类
router._estimate_load_time = _estimate_load_time.__get__(router, type(router))
router._record_inference_metrics = _record_inference_metrics.__get__(router, type(router))
router._get_model_performance_metrics = _get_model_performance_metrics.__get__(router, type(router))
router._get_model_health_status = _get_model_health_status.__get__(router, type(router))
router._get_model_performance_data = _get_model_performance_data.__get__(router, type(router))
router._calculate_percentile = _calculate_percentile.__get__(router, type(router))

# 导入timedelta
from datetime import timedelta