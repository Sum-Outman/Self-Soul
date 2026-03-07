"""
遗留系统API兼容性层

为原有Self-Soul系统提供API兼容性，
允许new_agi_system直接接管原有系统。
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["legacy"])


class LegacyRequest(BaseModel):
    """遗留系统请求格式"""
    input: Dict[str, Any]
    model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class LegacyResponse(BaseModel):
    """遗留系统响应格式"""
    output: Dict[str, Any]
    model: str
    processing_time: float
    success: bool
    error: Optional[str] = None


# 模型名称映射：旧模型名称 -> 新系统组件
MODEL_MAPPING = {
    # 语言模型
    "language_model": "text_processor",
    "unified_language_model": "text_processor",
    
    # 视觉模型
    "vision_model": "image_processor",
    "unified_vision_model": "image_processor",
    
    # 音频模型
    "audio_model": "audio_processor",
    "unified_audio_model": "audio_processor",
    
    # 知识模型
    "knowledge_model": "knowledge_system",
    "unified_knowledge_model": "knowledge_system",
    
    # 推理模型
    "reasoning_model": "reasoning_engine",
    "advanced_reasoning_model": "reasoning_engine",
    
    # 规划模型
    "planning_model": "planning_system",
    "unified_planning_model": "planning_system",
    
    # 决策模型
    "decision_model": "decision_system",
    "value_alignment_model": "decision_system",
    
    # 记忆模型
    "memory_model": "memory_system",
    "unified_memory_model": "memory_system",
    
    # 协调器
    "coordinator": "cognitive_architecture",
    "manager_model": "cognitive_architecture",
    
    # 其他模型
    "training_model": "training_manager",
    "learning_model": "learning_system",
    "autonomous_model": "cognitive_architecture"
}


@router.post("/process", response_model=LegacyResponse)
async def legacy_process(request: LegacyRequest):
    """
    处理遗留系统API请求。
    
    这个端点模拟原有27模型系统的API，但实际使用统一认知架构处理。
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        # 获取请求模型
        model_name = request.model or "coordinator"
        
        # 映射到新系统组件
        component_name = MODEL_MAPPING.get(model_name, "cognitive_architecture")
        
        # 准备输入数据
        input_data = request.input
        
        # 根据模型类型处理
        result = await _process_with_component(component_name, input_data, request.parameters)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return LegacyResponse(
            output=result,
            model=model_name,
            processing_time=processing_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"遗留API处理失败: {e}")
        return LegacyResponse(
            output={},
            model=request.model or "unknown",
            processing_time=0.0,
            success=False,
            error=str(e)
        )


async def _process_with_component(component_name: str, input_data: Dict[str, Any], 
                                 parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """使用指定组件处理数据"""
    # 这里应该从全局上下文中获取组件实例
    # 现在返回模拟结果
    
    # 模拟组件处理
    await asyncio.sleep(0.1)  # 模拟处理延迟
    
    # 根据组件类型返回不同结果
    if component_name == "text_processor":
        return {
            "text": input_data.get("text", ""),
            "processed": True,
            "embeddings": [0.1, 0.2, 0.3],  # 模拟嵌入
            "tokens": input_data.get("text", "").split()[:10]
        }
    elif component_name == "image_processor":
        return {
            "image_processed": True,
            "features": [0.5, 0.6, 0.7],  # 模拟特征
            "dimensions": [224, 224, 3]
        }
    elif component_name == "audio_processor":
        return {
            "audio_processed": True,
            "features": [0.3, 0.4, 0.5],
            "duration_ms": 1000
        }
    elif component_name == "reasoning_engine":
        return {
            "reasoning_result": "逻辑推理完成",
            "confidence": 0.85,
            "steps": ["解析输入", "应用逻辑", "得出结论"]
        }
    elif component_name == "cognitive_architecture":
        # 使用统一认知架构处理
        return await _process_with_cognitive_architecture(input_data)
    else:
        # 默认响应
        return {
            "component": component_name,
            "input_received": input_data,
            "parameters": parameters,
            "status": "processed"
        }


async def _process_with_cognitive_architecture(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """使用统一认知架构处理"""
    # 这里应该调用实际的认知架构
    # 现在返回模拟结果
    
    # 模拟认知循环
    await asyncio.sleep(0.2)
    
    return {
        "cognitive_output": {
            "perception": "多模态感知完成",
            "reasoning": "通用推理完成",
            "decision": "价值基础决策完成",
            "action": "自适应行动规划完成"
        },
        "cognitive_state": {
            "attention_level": 0.8,
            "memory_usage": 0.3,
            "processing_complete": True
        },
        "performance": {
            "processing_time_ms": 200,
            "success_rate": 1.0
        }
    }


@router.post("/train")
async def legacy_train(request: Dict[str, Any]):
    """遗留训练API"""
    try:
        # 映射到新训练系统
        training_type = request.get("training_type", "generic")
        model_name = request.get("model_name", "default_model")
        
        # 模拟训练开始
        training_id = f"train_{int(asyncio.get_event_loop().time())}"
        
        return {
            "training_id": training_id,
            "status": "started",
            "model": model_name,
            "training_type": training_type,
            "estimated_completion": "5分钟"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_name}")
async def legacy_status(model_name: str):
    """遗留状态检查API"""
    # 检查模型状态
    status = {
        "model": model_name,
        "status": "active",
        "port": 9000,  # 统一端口
        "health": "healthy",
        "last_active": asyncio.get_event_loop().time(),
        "system": "unified_cognitive_architecture"
    }
    
    return status


@router.get("/models")
async def legacy_models():
    """遗留模型列表API"""
    # 返回所有支持的模型（映射到新系统）
    models = []
    
    for old_name, new_component in MODEL_MAPPING.items():
        models.append({
            "name": old_name,
            "type": "legacy_compatible",
            "component": new_component,
            "status": "available",
            "endpoint": f"/api/process?model={old_name}"
        })
    
    return {"models": models}


@router.post("/multimodal")
async def legacy_multimodal(request: Dict[str, Any]):
    """遗留多模态API"""
    try:
        # 提取多模态输入
        text = request.get("text")
        image = request.get("image")  # base64
        audio = request.get("audio")  # base64
        
        # 模拟多模态处理
        results = {}
        
        if text:
            results["text_processing"] = {
                "processed": True,
                "embeddings_size": 512
            }
        
        if image:
            results["image_processing"] = {
                "processed": True,
                "dimensions": "224x224"
            }
        
        if audio:
            results["audio_processing"] = {
                "processed": True,
                "duration_ms": 1000
            }
        
        # 模拟融合
        if len(results) > 1:
            results["multimodal_fusion"] = {
                "fused": True,
                "fusion_method": "attention_weighted",
                "confidence": 0.9
            }
        
        return {
            "multimodal_results": results,
            "processing_complete": True,
            "unified_representation": "生成统一表征"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))