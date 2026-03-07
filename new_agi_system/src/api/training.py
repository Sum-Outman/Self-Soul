"""
训练API模块

为统一认知架构提供训练API端点。
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

from training.trainer import TrainingManager
from training.schemas import TrainingRequest, TrainingResponse, TrainingStatus, TrainingStatistics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# 全局训练管理器实例
training_manager = None


def get_training_manager():
    """获取训练管理器实例"""
    global training_manager
    if training_manager is None:
        # 需要从主应用中获取认知架构实例
        # 这里先创建一个模拟的，实际使用时需要注入
        from cognitive.architecture import UnifiedCognitiveArchitecture
        config = {'embedding_dim': 1024, 'max_shared_memory_mb': 1024}
        cognitive_arch = UnifiedCognitiveArchitecture(config)
        training_manager = TrainingManager(cognitive_arch)
    return training_manager


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """开始训练任务"""
    try:
        manager = get_training_manager()
        
        # 准备训练配置
        config = {
            'type': request.type,
            'model_name': request.model_name,
            'dataset': request.dataset,
            'hyperparameters': request.hyperparameters
        }
        
        # 开始训练
        result = await manager.start_training(config)
        return result
        
    except Exception as e:
        logger.error(f"开始训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """获取训练任务状态"""
    try:
        manager = get_training_manager()
        status = manager.get_job_status(job_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str):
    """取消训练任务"""
    try:
        manager = get_training_manager()
        result = manager.cancel_job(job_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"取消训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=TrainingStatistics)
async def get_training_statistics():
    """获取训练统计信息"""
    try:
        manager = get_training_manager()
        stats = manager.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"获取训练统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_training_jobs():
    """列出所有训练任务"""
    try:
        manager = get_training_manager()
        # 获取管理器中的任务信息
        jobs = []
        for job_id, job in manager.training_jobs.items():
            jobs.append({
                'job_id': job_id,
                'status': job.status,
                'model_name': job.config.get('model_name', 'unknown'),
                'progress': job.progress,
                'start_time': job.start_time
            })
        return {'jobs': jobs}
    except Exception as e:
        logger.error(f"列出训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """获取可训练模型列表"""
    # 返回支持的模型类型
    return {
        'models': [
            {'name': 'cognitive_model', 'description': '认知模型训练'},
            {'name': 'neural_network', 'description': '神经网络训练'},
            {'name': 'from_scratch', 'description': '从零开始训练'},
            {'name': 'fine_tuning', 'description': '微调训练'},
            {'name': 'multimodal_fusion', 'description': '多模态融合训练'},
            {'name': 'robot_control', 'description': '机器人控制训练'}
        ]
    }


@router.post("/robot/training")
async def start_robot_training(request: Dict[str, Any]):
    """开始机器人训练"""
    try:
        manager = get_training_manager()
        
        # 机器人训练配置
        config = {
            'type': 'robot_control',
            'model_name': request.get('model_name', 'robot_control_model'),
            'dataset': request.get('dataset', {}),
            'hyperparameters': request.get('hyperparameters', {}),
            'robot_config': request.get('robot_config', {})
        }
        
        result = await manager.start_training(config)
        return result
    except Exception as e:
        logger.error(f"开始机器人训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))