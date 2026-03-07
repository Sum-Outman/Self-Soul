"""
自主演化API

为统一认知架构提供自主演化功能的API接口。
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evolution", tags=["evolution"])


class EvolutionRequest(BaseModel):
    """演化请求"""
    component: str = Field(..., description="要演化的组件名称")
    performance_targets: Dict[str, float] = Field(
        ..., 
        description="性能目标字典，如 {'accuracy': 0.9, 'efficiency': 0.8}"
    )
    priority: str = Field("normal", description="优先级: low, normal, high")
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="约束条件，如 {'memory_mb': 500, 'compute_gflops': 2.0}"
    )


class EvolutionResponse(BaseModel):
    """演化响应"""
    evolution_id: str = Field(..., description="演化任务ID")
    status: str = Field(..., description="状态: pending, running, completed, failed")
    message: str = Field(..., description="状态消息")
    queue_position: Optional[int] = Field(None, description="队列位置（如果排队中）")


class EvolutionStatus(BaseModel):
    """演化状态"""
    evolution_id: str = Field(..., description="演化任务ID")
    status: str = Field(..., description="状态: pending, running, completed, failed")
    component: str = Field(..., description="演化的组件")
    created_time: float = Field(..., description="创建时间戳")
    start_time: Optional[float] = Field(None, description="开始时间戳")
    completion_time: Optional[float] = Field(None, description="完成时间戳")
    result: Optional[Dict[str, Any]] = Field(None, description="演化结果")
    queue_position: Optional[int] = Field(None, description="队列位置")


class EvolutionStatistics(BaseModel):
    """演化统计信息"""
    total_evolutions: int = Field(..., description="总演化任务数")
    successful_evolutions: int = Field(..., description="成功演化任务数")
    failed_evolutions: int = Field(..., description="失败演化任务数")
    average_improvement: float = Field(..., description="平均性能改进")
    best_performance: float = Field(..., description="最佳性能分数")
    total_computation_time: float = Field(..., description="总计算时间（秒）")
    last_evolution_time: float = Field(..., description="上次演化时间（秒）")
    gene_pool_size: int = Field(..., description="基因库大小")
    active_genes: int = Field(..., description="活跃基因数")


class GeneInfo(BaseModel):
    """基因信息"""
    component_id: str = Field(..., description="基因ID")
    component_type: str = Field(..., description="组件类型")
    performance_score: float = Field(..., description="性能分数")
    resource_cost: float = Field(..., description="资源成本")
    complexity_score: float = Field(..., description="复杂度分数")
    usage_count: int = Field(..., description="使用次数")
    last_used: float = Field(..., description="最后使用时间戳")
    mutation_count: int = Field(..., description="变异次数")
    active: bool = Field(..., description="是否活跃")


# 全局演化管理器实例
_evolution_manager = None


def get_evolution_manager():
    """获取演化管理器实例"""
    global _evolution_manager
    
    if _evolution_manager is None:
        try:
            # 导入统一认知架构
            from cognitive.architecture import UnifiedCognitiveArchitecture
            
            # 创建或获取全局认知架构实例
            # 注意：在实际部署中，这应该从应用状态中获取
            agi_instance = UnifiedCognitiveArchitecture()
            
            # 从认知架构获取演化系统
            _evolution_manager = agi_instance.evolution
            
            logger.info("演化管理器已初始化")
            
        except ImportError as e:
            logger.error(f"无法导入认知架构: {e}")
            raise HTTPException(
                status_code=500,
                detail="认知架构未正确初始化"
            )
        except Exception as e:
            logger.error(f"初始化演化管理器失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"演化管理器初始化失败: {e}"
            )
    
    return _evolution_manager


@router.post("/start", response_model=EvolutionResponse)
async def start_evolution(request: EvolutionRequest):
    """开始演化任务"""
    try:
        manager = get_evolution_manager()
        
        # 准备演化配置
        evolution_config = {
            'component': request.component,
            'performance_targets': request.performance_targets,
            'constraints': request.constraints or {}
        }
        
        # 开始演化
        result = await manager.request_evolution(
            component=request.component,
            performance_targets=request.performance_targets,
            priority=request.priority
        )
        
        return EvolutionResponse(
            evolution_id=result['evolution_id'],
            status=result['status'],
            message=result['message'],
            queue_position=result.get('queue_position')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"开始演化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{evolution_id}", response_model=EvolutionStatus)
async def get_evolution_status(evolution_id: str):
    """获取演化任务状态"""
    try:
        manager = get_evolution_manager()
        status = await manager.get_evolution_status(evolution_id)
        
        return EvolutionStatus(
            evolution_id=status['evolution_id'],
            status=status['status'],
            component=status['component'],
            created_time=status['created_time'],
            start_time=status.get('start_time'),
            completion_time=status.get('completion_time'),
            result=status.get('result'),
            queue_position=status.get('queue_position')
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取演化状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=EvolutionStatistics)
async def get_evolution_statistics():
    """获取演化统计信息"""
    try:
        manager = get_evolution_manager()
        stats = manager.get_system_status()
        
        evolution_stats = stats.get('architecture_engine', {}).get('evolution_stats', {})
        
        return EvolutionStatistics(
            total_evolutions=evolution_stats.get('total_evolutions', 0),
            successful_evolutions=evolution_stats.get('successful_evolutions', 0),
            failed_evolutions=evolution_stats.get('failed_evolutions', 0),
            average_improvement=evolution_stats.get('average_improvement', 0.0),
            best_performance=evolution_stats.get('best_performance', 0.0),
            total_computation_time=evolution_stats.get('total_computation_time', 0.0),
            last_evolution_time=evolution_stats.get('last_evolution_time', 0.0),
            gene_pool_size=stats.get('architecture_engine', {}).get('gene_pool_size', 0),
            active_genes=stats.get('architecture_engine', {}).get('active_genes', 0)
        )
    except Exception as e:
        logger.error(f"获取演化统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gene-pool", response_model=List[GeneInfo])
async def get_gene_pool():
    """获取基因库信息"""
    try:
        manager = get_evolution_manager()
        genes = manager.get_gene_pool()
        
        return [
            GeneInfo(
                component_id=gene.get('component_id', ''),
                component_type=gene.get('component_type', ''),
                performance_score=gene.get('performance_score', 0.0),
                resource_cost=gene.get('resource_cost', 0.0),
                complexity_score=gene.get('complexity_score', 0.0),
                usage_count=gene.get('usage_count', 0),
                last_used=gene.get('last_used', 0),
                mutation_count=gene.get('mutation_count', 0),
                active=gene.get('active', False)
            )
            for gene in genes
        ]
    except Exception as e:
        logger.error(f"获取基因库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue-status")
async def get_queue_status():
    """获取队列状态"""
    try:
        manager = get_evolution_manager()
        system_status = manager.get_system_status()
        queue_stats = system_status.get('queue_stats', {})
        
        return {
            "queue_status": {
                "total_tasks": queue_stats.get('total_tasks', 0),
                "pending_tasks": queue_stats.get('pending_tasks', 0),
                "running_tasks": queue_stats.get('running_tasks', 0),
                "completed_tasks": queue_stats.get('completed_tasks', 0),
                "failed_tasks": queue_stats.get('failed_tasks', 0),
                "max_concurrent": 2  # 硬编码，应与演化系统配置一致
            },
            "system_status": system_status.get('system_status', {}),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{evolution_id}")
async def stop_evolution(evolution_id: str):
    """停止演化任务"""
    try:
        # 注意：当前演化系统实现不支持停止运行中的任务
        # 这里返回成功但实际上是模拟的
        
        return {
            "success": True,
            "evolution_id": evolution_id,
            "message": "演化任务停止请求已接受（模拟实现）",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"停止演化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve-component")
async def evolve_specific_component(component: str, target: str = "performance"):
    """快速演化指定组件"""
    try:
        manager = get_evolution_manager()
        
        # 根据组件类型设置性能目标
        performance_targets = {}
        
        if component == "attention":
            performance_targets = {
                "accuracy": 0.9,
                "efficiency": 0.8,
                "adaptability": 0.85
            }
        elif component == "fusion":
            performance_targets = {
                "fusion_quality": 0.9,
                "efficiency": 0.75,
                "robustness": 0.8
            }
        elif component == "memory":
            performance_targets = {
                "recall_accuracy": 0.85,
                "storage_efficiency": 0.9,
                "retrieval_speed": 0.8
            }
        else:
            performance_targets = {
                "performance": 0.85,
                "efficiency": 0.7,
                "stability": 0.9
            }
        
        # 根据目标调整
        if target == "efficiency":
            performance_targets["efficiency"] = 0.9
            performance_targets["performance"] = 0.7
        elif target == "accuracy":
            performance_targets["accuracy"] = 0.95
            performance_targets["efficiency"] = 0.6
        
        # 开始演化
        result = await manager.request_evolution(
            component=component,
            performance_targets=performance_targets,
            priority="high"
        )
        
        return {
            "success": True,
            "evolution_id": result['evolution_id'],
            "component": component,
            "target": target,
            "status": result['status'],
            "message": result['message'],
            "performance_targets": performance_targets,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"快速演化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health():
    """获取演化系统健康状态"""
    try:
        manager = get_evolution_manager()
        system_status = manager.get_system_status()
        
        health_status = system_status.get('system_status', {})
        queue_stats = system_status.get('queue_stats', {})
        
        # 计算健康分数
        health_score = 1.0
        
        # 检查活动演化数量
        active_evolutions = health_status.get('active_evolutions', 0)
        if active_evolutions > 2:  # 超过最大并发数
            health_score -= 0.2
        
        # 检查失败任务比例
        total_tasks = queue_stats.get('total_tasks', 0)
        failed_tasks = queue_stats.get('failed_tasks', 0)
        if total_tasks > 0:
            failure_rate = failed_tasks / total_tasks
            if failure_rate > 0.3:  # 失败率超过30%
                health_score -= 0.3
        
        # 检查系统健康状态
        system_health = health_status.get('system_health', 'unknown')
        if system_health != 'healthy':
            health_score -= 0.2
        
        # 确保分数在0-1之间
        health_score = max(0.0, min(1.0, health_score))
        
        return {
            "system_health": system_health,
            "health_score": round(health_score, 2),
            "active_evolutions": active_evolutions,
            "total_successful_evolutions": health_status.get('total_successful_evolutions', 0),
            "initialized": health_status.get('initialized', False),
            "queue_status": queue_stats,
            "timestamp": time.time(),
            "recommendations": _generate_health_recommendations(
                system_health, health_score, active_evolutions, failure_rate if total_tasks > 0 else 0
            )
        }
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {e}")
        return {
            "system_health": "unknown",
            "health_score": 0.0,
            "error": str(e),
            "timestamp": time.time()
        }


def _generate_health_recommendations(system_health, health_score, active_evolutions, failure_rate):
    """生成健康建议"""
    recommendations = []
    
    if system_health != 'healthy':
        recommendations.append("系统健康状态异常，建议检查日志")
    
    if health_score < 0.7:
        recommendations.append("健康分数较低，建议优化演化配置")
    
    if active_evolutions > 2:
        recommendations.append("活动演化任务过多，建议增加最大并发数或优化任务调度")
    
    if failure_rate > 0.3:
        recommendations.append(f"演化失败率较高({failure_rate:.1%})，建议检查演化参数和约束条件")
    
    if not recommendations:
        recommendations.append("系统运行正常，无需特别建议")
    
    return recommendations