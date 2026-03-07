"""
自我认知API

为统一认知架构提供自我认知功能的API接口。
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import time
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/self_cognition", tags=["self_cognition"])


class SelfAwarenessReportRequest(BaseModel):
    """自我认知报告请求"""
    include_details: bool = Field(True, description="是否包含详细数据")
    include_history: bool = Field(True, description="是否包含历史数据")
    max_history_items: int = Field(10, description="最大历史项目数")


class PerformanceDimension(BaseModel):
    """性能维度"""
    dimension: str = Field(..., description="维度名称")
    value: float = Field(..., description="维度值 (0-1)")
    confidence: float = Field(..., description="置信度")
    trend: Optional[str] = Field(None, description="趋势: improving, stable, declining")
    insight: Optional[str] = Field(None, description="关键洞察")


class SelfAwarenessState(BaseModel):
    """自我认知状态"""
    overall_score: float = Field(..., description="总体自我认知分数")
    confidence_level: float = Field(..., description="置信度水平")
    last_update_time: float = Field(..., description="最后更新时间戳")
    system_health: str = Field(..., description="系统健康状态: healthy, warning, critical")


class PerformanceMetric(BaseModel):
    """性能指标"""
    dimension: str = Field(..., description="维度名称")
    value: float = Field(..., description="指标值")
    confidence: float = Field(..., description="置信度")
    timestamp: float = Field(..., description="时间戳")
    context: Dict[str, Any] = Field(..., description="上下文信息")
    evidence: List[str] = Field(..., description="证据列表")


class CriticalEvent(BaseModel):
    """关键事件"""
    event_id: str = Field(..., description="事件ID")
    event_type: str = Field(..., description="事件类型")
    description: str = Field(..., description="事件描述")
    timestamp: float = Field(..., description="时间戳")
    severity: float = Field(..., description="严重程度 (0-1)")
    impact_dimensions: List[str] = Field(..., description="影响维度")
    initial_assessment: Optional[str] = Field(None, description="初步评估")


class ReflectionAnalysis(BaseModel):
    """反思分析"""
    analysis_id: str = Field(..., description="分析ID")
    event_id: str = Field(..., description="事件ID")
    depth: str = Field(..., description="反思深度: surface, moderate, deep, comprehensive")
    root_causes: List[str] = Field(..., description="根本原因")
    recommendations: List[str] = Field(..., description="改进建议")
    confidence: float = Field(..., description="分析置信度")
    completion_time: float = Field(..., description="完成时间戳")
    analysis_duration: float = Field(..., description="分析持续时间")


class DeepReflectionRequest(BaseModel):
    """深度反思请求"""
    topic: str = Field(..., description="反思主题")
    focus_dimensions: Optional[List[str]] = Field(
        None,
        description="关注维度列表，如果为None则包括所有维度"
    )
    urgency: str = Field("normal", description="紧急程度: low, normal, high")


class SelfCognitionReport(BaseModel):
    """自我认知报告"""
    awareness_state: SelfAwarenessState = Field(..., description="自我认知状态")
    performance_dimensions: List[PerformanceDimension] = Field(..., description="性能维度")
    recent_metrics: List[PerformanceMetric] = Field(..., description="最近性能指标")
    active_events: List[CriticalEvent] = Field(..., description="活跃事件")
    recent_reflections: List[ReflectionAnalysis] = Field(..., description="最近反思分析")
    system_statistics: Dict[str, Any] = Field(..., description="系统统计信息")
    insights: List[str] = Field(..., description="关键洞察")
    timestamp: float = Field(..., description="报告时间戳")


class SystemHealthStatus(BaseModel):
    """系统健康状态"""
    overall_health: str = Field(..., description="总体健康状态")
    health_score: float = Field(..., description="健康分数 (0-1)")
    performance_summary: Dict[str, float] = Field(..., description="性能摘要")
    recommendations: List[str] = Field(..., description="改进建议")
    monitoring_active: bool = Field(..., description="监控是否活跃")
    last_check_time: float = Field(..., description="最后检查时间")


# 全局自我认知管理器实例
_self_cognition_manager = None


def get_self_cognition_manager():
    """获取自我认知管理器实例"""
    global _self_cognition_manager
    
    if _self_cognition_manager is None:
        try:
            # 导入统一认知架构
            from cognitive.architecture import UnifiedCognitiveArchitecture
            
            # 创建或获取全局认知架构实例
            # 注意：在实际部署中，这应该从应用状态中获取
            agi_instance = UnifiedCognitiveArchitecture()
            
            # 从认知架构获取自我认知系统
            _self_cognition_manager = agi_instance.self_cognition
            
            logger.info("自我认知管理器已初始化")
            
        except ImportError as e:
            logger.error(f"无法导入认知架构: {e}")
            raise HTTPException(
                status_code=500,
                detail="认知架构未正确初始化"
            )
        except Exception as e:
            logger.error(f"初始化自我认知管理器失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"自我认知管理器初始化失败: {e}"
            )
    
    return _self_cognition_manager


@router.get("/report", response_model=SelfCognitionReport)
async def get_self_awareness_report(request: SelfAwarenessReportRequest = None):
    """获取自我认知报告"""
    if request is None:
        request = SelfAwarenessReportRequest()
    
    try:
        manager = get_self_cognition_manager()
        
        # 获取原始报告数据
        raw_report = await manager.get_self_awareness_report()
        
        # 准备响应数据
        awareness_state = SelfAwarenessState(
            overall_score=raw_report["awareness_state"]["overall_score"],
            confidence_level=raw_report["awareness_state"]["confidence_level"],
            last_update_time=raw_report["awareness_state"]["last_update_time"],
            system_health=raw_report["system_status"]["overall_health"]
        )
        
        # 准备性能维度
        performance_dimensions = []
        for dim_name, dim_value in raw_report["awareness_state"]["dimension_scores"].items():
            trend = raw_report["awareness_state"]["trend_indicators"].get(dim_name)
            
            # 查找相关洞察
            insight = None
            for insight_text in raw_report["awareness_state"]["insights"]:
                if dim_name in insight_text:
                    insight = insight_text
                    break
            
            performance_dimensions.append(PerformanceDimension(
                dimension=dim_name,
                value=dim_value,
                confidence=raw_report["awareness_state"]["confidence_level"],
                trend=trend,
                insight=insight
            ))
        
        # 准备最近指标（如果请求包含）
        recent_metrics = []
        if request.include_details:
            recent_metrics = [
                PerformanceMetric(
                    dimension=metric["dimension"],
                    value=metric["value"],
                    confidence=metric["confidence"],
                    timestamp=metric["timestamp"],
                    context=metric["context"],
                    evidence=metric["evidence"]
                )
                for metric in raw_report.get("recent_performance", [])[:request.max_history_items]
            ]
        
        # 准备活跃事件（如果请求包含）
        active_events = []
        if request.include_details:
            active_events = [
                CriticalEvent(
                    event_id=event["event_id"],
                    event_type=event["event_type"],
                    description=event["description"],
                    timestamp=event["timestamp"],
                    severity=event["severity"],
                    impact_dimensions=event["impact_dimensions"],
                    initial_assessment=event.get("initial_assessment")
                )
                for event in raw_report.get("active_events", [])[:request.max_history_items]
            ]
        
        # 准备最近反思（如果请求包含）
        recent_reflections = []
        if request.include_details:
            recent_reflections = [
                ReflectionAnalysis(
                    analysis_id=analysis["analysis_id"],
                    event_id=analysis["event_id"],
                    depth=analysis["depth"],
                    root_causes=analysis["root_causes"],
                    recommendations=analysis["recommendations"],
                    confidence=analysis["confidence"],
                    completion_time=analysis["completion_time"],
                    analysis_duration=analysis["analysis_duration"]
                )
                for analysis in raw_report.get("recent_reflections", [])[:request.max_history_items]
            ]
        
        return SelfCognitionReport(
            awareness_state=awareness_state,
            performance_dimensions=performance_dimensions,
            recent_metrics=recent_metrics,
            active_events=active_events,
            recent_reflections=recent_reflections,
            system_statistics=raw_report.get("statistics", {}),
            insights=raw_report["awareness_state"]["insights"],
            timestamp=raw_report["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"获取自我认知报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=SystemHealthStatus)
async def get_system_health():
    """获取系统健康状态"""
    try:
        manager = get_self_cognition_manager()
        
        # 获取报告数据
        raw_report = await manager.get_self_awareness_report()
        
        # 提取健康信息
        awareness_state = raw_report["awareness_state"]
        system_status = raw_report["system_status"]
        
        # 计算健康分数
        health_score = awareness_state["overall_score"]
        
        # 确定总体健康状态
        overall_health = system_status["overall_health"]
        
        # 准备性能摘要
        performance_summary = awareness_state["dimension_scores"]
        
        # 生成建议
        recommendations = []
        
        if health_score < 0.5:
            recommendations.append("系统健康状态需立即关注，建议进行深度诊断")
        elif health_score < 0.7:
            recommendations.append("系统健康状态有待提升，建议优化关键维度")
        
        # 从洞察中提取建议
        for insight in awareness_state["insights"][:3]:  # 最多3个洞察
            if "需要" in insight or "建议" in insight or "优化" in insight:
                recommendations.append(insight)
        
        # 去重
        recommendations = list(set(recommendations))[:5]  # 最多5个建议
        
        return SystemHealthStatus(
            overall_health=overall_health,
            health_score=health_score,
            performance_summary=performance_summary,
            recommendations=recommendations,
            monitoring_active=system_status["monitoring_active"],
            last_check_time=raw_report["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"获取系统健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep-reflection", response_model=Dict[str, Any])
async def request_deep_reflection(request: DeepReflectionRequest):
    """请求深度反思"""
    try:
        manager = get_self_cognition_manager()
        
        # 请求深度反思
        result = await manager.request_deep_reflection(
            topic=request.topic,
            focus_dimensions=request.focus_dimensions
        )
        
        if result["success"]:
            return {
                "success": True,
                "analysis_id": result["analysis_id"],
                "topic": request.topic,
                "dimensions": result.get("dimensions", []),
                "message": result.get("message", "深度反思已启动"),
                "timestamp": time.time()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("message", "深度反思启动失败")
            )
            
    except Exception as e:
        logger.error(f"请求深度反思失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-history/{dimension}")
async def get_performance_history(
    dimension: str,
    limit: int = 50,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """获取性能历史数据"""
    try:
        manager = get_self_cognition_manager()
        
        # 获取性能历史
        raw_metrics = await manager.get_performance_history(
            dimension=dimension,
            limit=limit * 2  # 获取更多，用于时间过滤
        )
        
        # 时间过滤
        if start_time is not None:
            raw_metrics = [m for m in raw_metrics if m["timestamp"] >= start_time]
        
        if end_time is not None:
            raw_metrics = [m for m in raw_metrics if m["timestamp"] <= end_time]
        
        # 限制数量
        raw_metrics = raw_metrics[:limit]
        
        # 转换为响应格式
        metrics = [
            PerformanceMetric(
                dimension=m["dimension"],
                value=m["value"],
                confidence=m["confidence"],
                timestamp=m["timestamp"],
                context=m["context"],
                evidence=m["evidence"]
            )
            for m in raw_metrics
        ]
        
        return {
            "dimension": dimension,
            "metrics": metrics,
            "count": len(metrics),
            "time_range": {
                "start": metrics[0]["timestamp"] if metrics else None,
                "end": metrics[-1]["timestamp"] if metrics else None
            },
            "average_value": sum(m.value for m in metrics) / len(metrics) if metrics else 0.0,
            "timestamp": time.time()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取性能历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{analysis_id}")
async def get_reflection_analysis(analysis_id: str):
    """获取反思分析详情"""
    try:
        manager = get_self_cognition_manager()
        
        # 获取分析详情
        analysis = await manager.get_detailed_analysis(analysis_id)
        
        if analysis is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到分析ID: {analysis_id}"
            )
        
        return {
            "analysis": analysis,
            "success": True,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取反思分析详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_self_cognition_statistics():
    """获取自我认知统计信息"""
    try:
        manager = get_self_cognition_manager()
        
        # 获取统计信息
        stats = manager.get_system_statistics()
        
        return {
            "statistics": stats,
            "success": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取自我认知统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-monitoring")
async def start_self_monitoring():
    """启动自我监控"""
    try:
        manager = get_self_cognition_manager()
        
        # 启动监控
        await manager.start_monitoring()
        
        return {
            "success": True,
            "message": "自我监控已启动",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"启动自我监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-monitoring")
async def stop_self_monitoring():
    """停止自我监控"""
    try:
        manager = get_self_cognition_manager()
        
        # 停止监控
        await manager.stop_monitoring()
        
        return {
            "success": True,
            "message": "自我监控已停止",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"停止自我监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dimensions")
async def get_available_dimensions():
    """获取可用的性能维度"""
    try:
        # 这些维度来自SelfCognitionSystem中的SelfAwarenessDimension枚举
        dimensions = [
            {
                "name": "performance",
                "description": "性能表现",
                "importance": "high",
                "ideal_range": [0.7, 1.0]
            },
            {
                "name": "decision_quality",
                "description": "决策质量",
                "importance": "high",
                "ideal_range": [0.7, 1.0]
            },
            {
                "name": "efficiency",
                "description": "效率",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            },
            {
                "name": "stability",
                "description": "稳定性",
                "importance": "high",
                "ideal_range": [0.8, 1.0]
            },
            {
                "name": "adaptability",
                "description": "适应能力",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            },
            {
                "name": "learning_capacity",
                "description": "学习能力",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            },
            {
                "name": "memory_effectiveness",
                "description": "记忆效果",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            },
            {
                "name": "attention_effectiveness",
                "description": "注意力效果",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            },
            {
                "name": "reasoning_quality",
                "description": "推理质量",
                "importance": "high",
                "ideal_range": [0.7, 1.0]
            },
            {
                "name": "planning_effectiveness",
                "description": "规划效果",
                "importance": "medium",
                "ideal_range": [0.6, 1.0]
            }
        ]
        
        return {
            "dimensions": dimensions,
            "count": len(dimensions),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取可用维度失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-event")
async def trigger_critical_event(
    event_type: str,
    description: str,
    severity: float = 0.5,
    impact_dimensions: List[str] = None
):
    """手动触发关键事件（用于测试）"""
    try:
        manager = get_self_cognition_manager()
        
        # 这里需要一个方法来手动触发事件
        # 简化实现：返回模拟响应
        
        event_id = f"manual_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        return {
            "success": True,
            "event_id": event_id,
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "impact_dimensions": impact_dimensions or [],
            "message": "关键事件已触发（模拟）",
            "timestamp": time.time(),
            "note": "此功能为模拟实现，实际系统中应调用相应的方法"
        }
        
    except Exception as e:
        logger.error(f"触发关键事件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring-status")
async def get_monitoring_status():
    """获取监控状态"""
    try:
        manager = get_self_cognition_manager()
        stats = manager.get_system_statistics()
        
        return {
            "monitoring_active": stats.get("active_monitoring", False),
            "initialized": stats.get("initialized", False),
            "performance_history_size": stats.get("performance_history_size", 0),
            "critical_events_count": stats.get("critical_events_count", 0),
            "reflection_analyses_count": stats.get("reflection_analyses_count", 0),
            "last_update": time.time(),
            "recommendations": [
                "确保监控持续运行以获取准确的自我认知数据",
                "定期检查性能趋势以发现潜在问题"
            ]
        }
        
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))