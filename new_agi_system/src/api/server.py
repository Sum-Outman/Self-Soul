"""
统一认知架构API服务器

这是新统一认知架构的主要API服务器。
它用单一的神经张量基础AGI系统替换了27模型HTTP协调系统。

端口: 9000 (避免与现有27模型系统的端口8000冲突)
"""

import asyncio
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import time
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入统一认知架构
from cognitive.architecture import UnifiedCognitiveArchitecture
from cognitive.representation import UnifiedRepresentationSpace

# 导入训练API
try:
    from .training import router as training_router
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger.warning("训练API不可用，跳过训练功能")

# 导入演化API
try:
    from .evolution import router as evolution_router
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    logger.warning("演化API不可用，跳过演化功能")

# 导入自我认知API
try:
    from .self_cognition import router as self_cognition_router
    SELF_COGNITION_AVAILABLE = True
except ImportError as e:
    SELF_COGNITION_AVAILABLE = False
    logger.warning(f"自我认知API不可用，跳过自我认知功能: {e}")
    import traceback
    traceback.print_exc()

# 导入控制API
try:
    from .control import router as control_router
    CONTROL_AVAILABLE = True
except ImportError as e:
    CONTROL_AVAILABLE = False
    logger.warning(f"控制API不可用，跳过控制功能: {e}")
    import traceback
    traceback.print_exc()

# 导入自主意识API
try:
    from .autonomy import router as autonomy_router
    AUTONOMY_AVAILABLE = True
except ImportError:
    AUTONOMY_AVAILABLE = False
    logger.warning("自主意识API不可用，跳过自主意识功能")

# 导入人形机器人API
try:
    from .humanoid import router as humanoid_router
    HUMANOID_AVAILABLE = True
except ImportError:
    HUMANOID_AVAILABLE = False
    logger.warning("人形机器人API不可用，跳过人形机器人功能")

# 导入遗留兼容性API
try:
    from .legacy_compatibility import router as legacy_router
    LEGACY_COMPATIBILITY_AVAILABLE = True
except ImportError:
    LEGACY_COMPATIBILITY_AVAILABLE = False
    logger.warning("遗留兼容性API不可用，跳过遗留兼容性功能")

# FastAPI应用，支持生命周期管理
app = FastAPI(
    title="统一认知架构API",
    description="单一统一AGI系统，替换27模型HTTP协调系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中，请指定实际来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含训练API路由
if TRAINING_AVAILABLE:
    app.include_router(training_router)
    logger.info("训练API已启用")
else:
    logger.warning("训练API未启用，训练功能不可用")

# 包含演化API路由
if EVOLUTION_AVAILABLE:
    app.include_router(evolution_router)
    logger.info("演化API已启用")
else:
    logger.warning("演化API未启用，演化功能不可用")

# 包含自我认知API路由
if SELF_COGNITION_AVAILABLE:
    app.include_router(self_cognition_router)
    logger.info("自我认知API已启用")
else:
    logger.warning("自我认知API未启用，自我认知功能不可用")

# 包含控制API路由
if CONTROL_AVAILABLE:
    app.include_router(control_router)
    logger.info("控制API已启用")
else:
    logger.warning("控制API未启用，控制功能不可用")

# 包含自主意识API路由
if AUTONOMY_AVAILABLE:
    app.include_router(autonomy_router)
    logger.info("自主意识API已启用")
else:
    logger.warning("自主意识API未启用，自主意识功能不可用")

# 包含人形机器人API路由
if HUMANOID_AVAILABLE:
    app.include_router(humanoid_router)
    logger.info("人形机器人API已启用")
else:
    logger.warning("人形机器人API未启用，人形机器人功能不可用")

# 包含遗留兼容性API路由
if LEGACY_COMPATIBILITY_AVAILABLE:
    app.include_router(legacy_router)
    logger.info("遗留兼容性API已启用")
else:
    logger.warning("遗留兼容性API未启用，遗留系统兼容性不可用")

# 全局AGI实例
agi_instance = None


class CognitiveRequest(BaseModel):
    """认知处理请求"""
    text: Optional[str] = None
    image: Optional[str] = None  # Base64编码
    audio: Optional[str] = None  # Base64编码
    structured_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    goal: Optional[str] = None
    priority: str = "normal"  # low, normal, high, realtime


class CognitiveResponse(BaseModel):
    """认知处理响应"""
    output: Dict[str, Any]
    reasoning_trace: Optional[Dict[str, Any]] = None
    cognitive_state: Dict[str, Any]
    performance: Dict[str, Any]
    error: Optional[str] = None


class DiagnosticsResponse(BaseModel):
    """系统诊断响应"""
    cognitive_state: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    representation_cache: Dict[str, Any]
    communication_stats: Dict[str, Any]
    system_info: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """启动时初始化AGI系统"""
    global agi_instance
    
    logger.info("正在启动统一认知架构...")
    
    try:
        # 初始化统一认知架构
        config = {
            'embedding_dim': 1024,
            'max_shared_memory_mb': 1024,
            'port': 9000
        }
        
        agi_instance = UnifiedCognitiveArchitecture(config)
        
        # 初始化组件
        await agi_instance.initialize()
        
        logger.info("统一认知架构初始化成功")
        logger.info(f"API服务器运行在端口 {config['port']}")
        
    except Exception as e:
        logger.error(f"初始化AGI系统失败: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """关闭AGI系统"""
    global agi_instance
    
    if agi_instance:
        logger.info("正在关闭统一认知架构...")
        await agi_instance.shutdown()
        logger.info("统一认知架构关闭完成")


@app.get("/")
async def root():
    """根端点"""
    return {
        "name": "统一认知架构API",
        "version": "1.0.0",
        "description": "单一统一AGI系统，替换27模型HTTP协调系统",
        "status": "operational",
        "port": 9000
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        # 获取诊断信息
        diagnostics = agi_instance.get_diagnostics()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_info": diagnostics['system_info'],
            "performance": {
                "total_cycles": diagnostics['system_info']['total_cycles'],
                "avg_response_time": diagnostics['system_info']['avg_response_time'],
                "success_rate": diagnostics['system_info']['success_rate']
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail=f"健康检查失败: {str(e)}")


@app.post("/api/cognitive/process", response_model=CognitiveResponse)
async def process_cognitive(request: CognitiveRequest):
    """
    通过统一架构处理认知请求。
    
    这个单一端点替换了27模型HTTP协调系统，
    通过神经张量通信提供真正的认知统一性。
    """
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    start_time = time.time()
    
    try:
        # 准备输入数据
        input_data = {}
        
        if request.text:
            input_data['text'] = request.text
        
        if request.image:
            # 解码base64图像
            import base64
            try:
                image_bytes = base64.b64decode(request.image)
                # 转换为numpy数组（简化版）
                import numpy as np
                # 实际应用中，会解码为真实图像
                input_data['image'] = np.random.randn(3, 224, 224)  # 占位符
            except Exception as e:
                logger.warning(f"图像解码失败: {e}")
        
        if request.audio:
            # 解码base64音频
            import base64
            try:
                audio_bytes = base64.b64decode(request.audio)
                # 转换为numpy数组（简化版）
                import numpy as np
                input_data['audio'] = np.random.randn(16000)  # 占位符
            except Exception as e:
                logger.warning(f"音频解码失败: {e}")
        
        if request.structured_data:
            input_data['structured'] = request.structured_data
        
        # 添加上下文
        if request.context:
            input_data['context'] = request.context
        
        # 如果指定了目标，则添加
        if request.goal:
            agi_instance.cognitive_state.add_goal({
                'description': request.goal,
                'priority': request.priority,
                'type': 'user_request'
            })
        
        logger.info(f"正在处理认知请求: {request.text[:100] if request.text else '无文本'}")
        
        # 执行认知循环
        result = await agi_instance.cognitive_cycle(input_data)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 准备响应
        response = CognitiveResponse(
            output=result.get('output', {}),
            reasoning_trace=result.get('reasoning_trace'),
            cognitive_state=result.get('cognitive_state', {}),
            performance={
                **result.get('performance', {}),
                'total_processing_time': processing_time
            },
            error=result.get('error')
        )
        
        logger.info(f"认知请求处理完成，耗时 {processing_time:.3f}秒")
        
        return response
        
    except Exception as e:
        logger.error(f"认知处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"认知处理失败: {str(e)}")


@app.get("/api/cognitive/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics():
    """获取系统诊断信息"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        diagnostics = agi_instance.get_diagnostics()
        
        return DiagnosticsResponse(
            cognitive_state=diagnostics['cognitive_state'],
            performance_metrics=diagnostics['system_info'],  # 使用system_info作为performance_metrics
            representation_cache=diagnostics['representation_cache'],
            communication_stats=diagnostics['communication_stats'],
            system_info=diagnostics['system_info']
        )
    except Exception as e:
        logger.error(f"获取诊断信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取诊断信息失败: {str(e)}")


@app.post("/api/cognitive/clear_cache")
async def clear_cache():
    """清理表征缓存"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        agi_instance.representation_space.clear_cache()
        return {"status": "success", "message": "缓存已清理"}
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")


@app.post("/api/cognitive/add_goal")
async def add_goal(goal: Dict[str, Any]):
    """向认知状态添加目标"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        agi_instance.cognitive_state.add_goal(goal)
        return {"status": "success", "message": "目标已添加"}
    except Exception as e:
        logger.error(f"添加目标失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加目标失败: {str(e)}")


@app.get("/api/cognitive/current_goal")
async def get_current_goal():
    """获取当前目标"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        current_goal = agi_instance.cognitive_state.get_current_goal()
        return {"current_goal": current_goal}
    except Exception as e:
        logger.error(f"获取当前目标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取当前目标失败: {str(e)}")


@app.websocket("/ws/cognitive/stream")
async def cognitive_stream(websocket: WebSocket):
    """实时认知流式处理的WebSocket端点"""
    global agi_instance
    
    if not agi_instance:
        await websocket.close(code=1008, reason="AGI系统未初始化")
        return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process cognitive request
            if data.get('type') == 'cognitive_request':
                input_data = data.get('data', {})
                
                # Execute cognitive cycle
                result = await agi_instance.cognitive_cycle(input_data)
                
                # Send response
                await websocket.send_json({
                    'type': 'cognitive_response',
                    'data': result
                })
            
            # Get diagnostics
            elif data.get('type') == 'get_diagnostics':
                diagnostics = agi_instance.get_diagnostics()
                await websocket.send_json({
                    'type': 'diagnostics',
                    'data': diagnostics
                })
            
            # Unknown message type
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f"未知消息类型: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket连接已断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception as close_error:
            logger.debug(f"WebSocket关闭时发生异常: {close_error}")


# 特定认知能力的额外端点
@app.post("/api/cognitive/reasoning")
async def reasoning_only(request: CognitiveRequest):
    """仅执行推理组件"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        # 编码输入
        input_data = {'text': request.text} if request.text else {}
        unified_repr = agi_instance.representation_space.encode(input_data)
        
        # 获取记忆上下文
        context = agi_instance.cognitive_state.get_relevant_context(unified_repr)
        
        # 执行推理
        reasoning_result = await agi_instance._process_reasoning(unified_repr, {'context': context})
        
        return {
            "reasoning_result": reasoning_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/api/cognitive/planning")
async def planning_only(request: CognitiveRequest):
    """仅执行规划组件"""
    global agi_instance
    
    if not agi_instance:
        raise HTTPException(status_code=503, detail="AGI系统未初始化")
    
    try:
        # 为规划创建推理结果
        reasoning_result = {
            'reasoning_output': torch.randn(1, 512),  # 占位符
            'confidence': 0.8
        }
        
        # 执行规划
        plan_result = await agi_instance._process_planning(reasoning_result)
        
        return {
            "planning_result": plan_result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"规划失败: {e}")
        raise HTTPException(status_code=500, detail=f"规划失败: {str(e)}")


# 导入torch用于占位符张量
try:
    import torch
except ImportError:
    # 为测试创建模拟torch模块
    class MockTorch:
        class tensor:
            @staticmethod
            def __call__(self, data):
                return MockTensor(data)
        
        class randn:
            @staticmethod
            def __call__(self, *args):
                return MockTensor(args)
    
    class MockTensor:
        def __init__(self, shape):
            self.shape = shape if isinstance(shape, tuple) else (shape,)
        
        def mean(self, *args, **kwargs):
            return MockTensor(self.shape)
        
        def unsqueeze(self, *args, **kwargs):
            return MockTensor(self.shape)
    
    torch = MockTorch()


if __name__ == "__main__":
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一认知架构API服务器")
    parser.add_argument("--host", default="127.0.0.1", help="主机地址")
    parser.add_argument("--port", type=int, default=9000, help="端口号")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    
    args = parser.parse_args()
    
    logger.info(f"正在启动统一认知架构API服务器，地址: {args.host}:{args.port}")
    
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )