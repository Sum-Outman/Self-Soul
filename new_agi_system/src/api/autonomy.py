"""
自主意识API

为统一认知架构提供自主意识功能的API接口。
包括内在动机管理、自主目标生成和好奇心驱动探索等功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/autonomy", tags=["autonomy"])


class IntrinsicMotivationType(str, Enum):
    """内在动机类型"""
    CURIOSITY = "curiosity"
    COMPETENCE = "competence"
    KNOWLEDGE_COMPLETENESS = "knowledge_completeness"
    EXPLORATION = "exploration"
    MASTERY = "mastery"
    NOVELTY = "novelty"


class GoalType(str, Enum):
    """目标类型"""
    EXPLORATORY = "exploratory"
    EXPLOITATIVE = "exploitative"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    CREATIVE = "creative"


class MotivationState(BaseModel):
    """动机状态"""
    motivation_type: IntrinsicMotivationType = Field(..., description="动机类型")
    strength: float = Field(..., description="动机强度 (0-1)")
    satisfaction: float = Field(..., description="满意度 (0-1)")
    last_activation: float = Field(..., description="最后激活时间戳")
    activation_count: int = Field(..., description="激活次数")


class AutonomousGoal(BaseModel):
    """自主目标"""
    goal_id: str = Field(..., description="目标ID")
    goal_type: GoalType = Field(..., description="目标类型")
    description: str = Field(..., description="目标描述")
    motivation_source: IntrinsicMotivationType = Field(..., description="动机来源")
    priority: float = Field(..., description="优先级 (0-1)")
    feasibility: float = Field(..., description="可行性 (0-1)")
    expected_value: float = Field(..., description="预期价值 (0-1)")
    creation_time: float = Field(..., description="创建时间戳")
    deadline: Optional[float] = Field(None, description="截止时间戳")
    progress: float = Field(..., description="进度 (0-1)")
    status: str = Field(..., description="状态")


class ExplorationState(BaseModel):
    """探索状态"""
    novelty_score: float = Field(..., description="新奇性评分 (0-1)")
    uncertainty: float = Field(..., description="不确定性 (0-1)")
    information_gain: float = Field(..., description="信息增益 (0-1)")
    coverage: float = Field(..., description="覆盖度 (0-1)")
    last_exploration: float = Field(..., description="最后探索时间戳")


class AutonomyRequest(BaseModel):
    """自主性请求"""
    action: str = Field(..., description="请求动作: generate_goal, adjust_motivation, explore")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="动作参数")


class AutonomyResponse(BaseModel):
    """自主性响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(default_factory=dict, description="响应数据")
    timestamp: float = Field(..., description="响应时间戳")


class AutonomyStatus(BaseModel):
    """自主性状态"""
    autonomy_level: float = Field(..., description="自主性水平 (0-1)")
    motivation_states: Dict[str, MotivationState] = Field(..., description="动机状态")
    active_goals: List[AutonomousGoal] = Field(..., description="活跃目标")
    exploration_state: ExplorationState = Field(..., description="探索状态")
    performance_metrics: Dict[str, Any] = Field(..., description="性能指标")
    timestamp: float = Field(..., description="状态时间戳")


class AutonomyReport(BaseModel):
    """自主性报告"""
    status: AutonomyStatus = Field(..., description="当前状态")
    knowledge_state: Dict[str, Any] = Field(..., description="知识状态")
    goal_statistics: Dict[str, Any] = Field(..., description="目标统计")
    insights: List[str] = Field(..., description="关键洞察")
    recommendations: List[str] = Field(..., description="改进建议")
    timestamp: float = Field(..., description="报告时间戳")


def get_agi_architecture():
    """获取统一认知架构实例"""
    from ..cognitive.architecture import UnifiedCognitiveArchitecture
    # 这里应该从应用程序状态获取实例
    # 简化实现：创建新实例
    return UnifiedCognitiveArchitecture()


@router.get("/status", response_model=AutonomyStatus)
async def get_autonomy_status(
    agi = Depends(get_agi_architecture)
):
    """获取自主性系统状态"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取自主性报告
        report = await agi.autonomy.get_autonomy_report()
        
        # 获取动机状态
        motivation_states = await agi.autonomy.get_motivation_states()
        
        # 获取活跃目标
        active_goals = await agi.autonomy.get_active_goals()
        
        # 构建状态响应
        return AutonomyStatus(
            autonomy_level=report['performance_metrics'].get('autonomy_level', 0.5),
            motivation_states={
                mt: MotivationState(
                    motivation_type=IntrinsicMotivationType(mt),
                    strength=state['strength'],
                    satisfaction=state['satisfaction'],
                    last_activation=state['last_activation'],
                    activation_count=state['activation_count']
                )
                for mt, state in motivation_states.items()
            },
            active_goals=[
                AutonomousGoal(
                    goal_id=goal.goal_id,
                    goal_type=GoalType(goal.goal_type.value),
                    description=goal.description,
                    motivation_source=IntrinsicMotivationType(goal.motivation_source.value),
                    priority=goal.priority,
                    feasibility=goal.feasibility,
                    expected_value=goal.expected_value,
                    creation_time=goal.creation_time,
                    deadline=goal.deadline,
                    progress=goal.progress,
                    status=goal.status
                )
                for goal in active_goals
            ],
            exploration_state=ExplorationState(
                novelty_score=report['exploration_state']['novelty_score'],
                uncertainty=report['exploration_state']['uncertainty'],
                information_gain=report['exploration_state']['information_gain'],
                coverage=report['exploration_state']['coverage'],
                last_exploration=report['exploration_state']['last_exploration']
            ),
            performance_metrics=report['performance_metrics'],
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"获取自主性状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report", response_model=AutonomyReport)
async def get_autonomy_report(
    agi = Depends(get_agi_architecture)
):
    """获取完整自主性报告"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取自主性报告
        report = await agi.autonomy.get_autonomy_report()
        
        # 构建响应
        status_response = AutonomyStatus(
            autonomy_level=report['performance_metrics'].get('autonomy_level', 0.5),
            motivation_states={},
            active_goals=[],
            exploration_state=ExplorationState(
                novelty_score=report['exploration_state']['novelty_score'],
                uncertainty=report['exploration_state']['uncertainty'],
                information_gain=report['exploration_state']['information_gain'],
                coverage=report['exploration_state']['coverage'],
                last_exploration=report['exploration_state']['last_exploration']
            ),
            performance_metrics=report['performance_metrics'],
            timestamp=time.time()
        )
        
        return AutonomyReport(
            status=status_response,
            knowledge_state=report.get('knowledge_state_summary', {}),
            goal_statistics={
                'active_goals_count': report.get('active_goals_count', 0),
                'completed_goals_count': report.get('completed_goals_count', 0),
                'total_goals_generated': report['performance_metrics'].get('total_goals_generated', 0),
                'goal_success_rate': report['performance_metrics'].get('goal_success_rate', 0.0)
            },
            insights=[
                f"自主性水平: {report['performance_metrics'].get('autonomy_level', 0.5):.2f}",
                f"目标成功率: {report['performance_metrics'].get('goal_success_rate', 0.0):.2f}",
                f"动机满意度: {report['performance_metrics'].get('motivation_satisfaction', 0.5):.2f}"
            ],
            recommendations=[
                "定期检查动机状态，确保系统保持好奇心",
                "平衡探索和利用，避免陷入局部最优",
                "根据知识缺口调整学习目标"
            ],
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"获取自主性报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_goal", response_model=AutonomyResponse)
async def generate_autonomous_goal(
    request: AutonomyRequest,
    agi = Depends(get_agi_architecture)
):
    """生成自主目标"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取参数
        motivation_type = request.parameters.get('motivation_type', 'curiosity')
        priority = request.parameters.get('priority', 0.5)
        
        # 生成目标（在自主性系统内部实现）
        # 这里只是模拟，实际应该在自主性系统内部实现
        
        return AutonomyResponse(
            success=True,
            message=f"自主目标生成请求已接收 (动机类型: {motivation_type})",
            data={
                'motivation_type': motivation_type,
                'priority': priority,
                'estimated_completion': time.time() + 300.0  # 5分钟后
            },
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"生成自主目标失败: {e}")
        return AutonomyResponse(
            success=False,
            message=str(e),
            data={},
            timestamp=time.time()
        )


@router.post("/adjust_motivation", response_model=AutonomyResponse)
async def adjust_motivation(
    request: AutonomyRequest,
    agi = Depends(get_agi_architecture)
):
    """调整动机强度"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取参数
        motivation_type = request.parameters.get('motivation_type')
        adjustment = request.parameters.get('adjustment', 0.0)
        
        if not motivation_type:
            return AutonomyResponse(
                success=False,
                message="必须指定动机类型",
                data={},
                timestamp=time.time()
            )
        
        return AutonomyResponse(
            success=True,
            message=f"动机调整请求已接收 (类型: {motivation_type}, 调整: {adjustment})",
            data={
                'motivation_type': motivation_type,
                'adjustment': adjustment,
                'timestamp': time.time()
            },
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"调整动机失败: {e}")
        return AutonomyResponse(
            success=False,
            message=str(e),
            data={},
            timestamp=time.time()
        )


@router.post("/explore", response_model=AutonomyResponse)
async def start_exploration(
    request: AutonomyRequest,
    agi = Depends(get_agi_architecture)
):
    """开始探索"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取参数
        exploration_type = request.parameters.get('type', 'balanced')
        duration = request.parameters.get('duration', 300.0)
        
        return AutonomyResponse(
            success=True,
            message=f"探索请求已接收 (类型: {exploration_type}, 持续时间: {duration}秒)",
            data={
                'exploration_type': exploration_type,
                'duration': duration,
                'start_time': time.time(),
                'estimated_end_time': time.time() + duration
            },
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"开始探索失败: {e}")
        return AutonomyResponse(
            success=False,
            message=str(e),
            data={},
            timestamp=time.time()
        )


@router.get("/motivations", response_model=Dict[str, MotivationState])
async def get_motivation_states(
    agi = Depends(get_agi_architecture)
):
    """获取所有动机状态"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取动机状态
        motivation_states = await agi.autonomy.get_motivation_states()
        
        # 转换响应格式
        return {
            mt: MotivationState(
                motivation_type=IntrinsicMotivationType(mt),
                strength=state['strength'],
                satisfaction=state['satisfaction'],
                last_activation=state['last_activation'],
                activation_count=state['activation_count']
            )
            for mt, state in motivation_states.items()
        }
        
    except Exception as e:
        logger.error(f"获取动机状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/goals/active", response_model=List[AutonomousGoal])
async def get_active_goals(
    agi = Depends(get_agi_architecture)
):
    """获取活跃目标列表"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取活跃目标
        active_goals = await agi.autonomy.get_active_goals()
        
        # 转换响应格式
        return [
            AutonomousGoal(
                goal_id=goal.goal_id,
                goal_type=GoalType(goal.goal_type.value),
                description=goal.description,
                motivation_source=IntrinsicMotivationType(goal.motivation_source.value),
                priority=goal.priority,
                feasibility=goal.feasibility,
                expected_value=goal.expected_value,
                creation_time=goal.creation_time,
                deadline=goal.deadline,
                progress=goal.progress,
                status=goal.status
            )
            for goal in active_goals
        ]
        
    except Exception as e:
        logger.error(f"获取活跃目标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exploration/state", response_model=ExplorationState)
async def get_exploration_state(
    agi = Depends(get_agi_architecture)
):
    """获取探索状态"""
    try:
        # 检查自主性系统是否可用
        if not agi.autonomy:
            raise HTTPException(
                status_code=503,
                detail="自主性系统不可用"
            )
        
        # 获取自主性报告
        report = await agi.autonomy.get_autonomy_report()
        
        # 提取探索状态
        exploration_data = report['exploration_state']
        
        return ExplorationState(
            novelty_score=exploration_data['novelty_score'],
            uncertainty=exploration_data['uncertainty'],
            information_gain=exploration_data['information_gain'],
            coverage=exploration_data['coverage'],
            last_exploration=exploration_data['last_exploration']
        )
        
    except Exception as e:
        logger.error(f"获取探索状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))