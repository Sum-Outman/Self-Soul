"""
人形机器人AGI API

为统一认知架构提供人形机器人AGI功能的API接口。
包括平衡控制、双足行走、任务执行和人机交互等功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/humanoid", tags=["humanoid"])


class HumanoidTaskType(str, Enum):
    """人形机器人任务类型"""
    WALK_TO_LOCATION = "walk_to_location"
    PICK_UP_OBJECT = "pick_up_object"
    PLACE_OBJECT = "place_object"
    INTERACT_WITH_HUMAN = "interact_with_human"
    NAVIGATE_ENVIRONMENT = "navigate_environment"
    MAINTAIN_BALANCE = "maintain_balance"
    PERFORM_GESTURE = "perform_gesture"


class WalkingDirection(str, Enum):
    """行走方向"""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class TaskRequest(BaseModel):
    """任务请求"""
    task_type: HumanoidTaskType = Field(..., description="任务类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="任务参数")


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str = Field(..., description="任务ID")
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    estimated_duration: Optional[float] = Field(None, description="预计持续时间")
    timestamp: float = Field(..., description="响应时间戳")


class SystemState(BaseModel):
    """系统状态"""
    operational_mode: str = Field(..., description="操作模式")
    autonomy_level: float = Field(..., description="自主性水平 (0-1)")
    safety_status: str = Field(..., description="安全状态")
    battery_level: float = Field(..., description="电池电量 (0-1)")
    temperature: float = Field(..., description="系统温度 (°C)")
    total_operating_time: float = Field(..., description="总运行时间")


class BalanceReport(BaseModel):
    """平衡报告"""
    balance_state: str = Field(..., description="平衡状态")
    stability_margin: float = Field(..., description="稳定裕度")
    tilt_angles: List[float] = Field(..., description="倾斜角度")
    active_strategy: str = Field(..., description="活跃策略")
    timestamp: float = Field(..., description="报告时间戳")


class WalkingReport(BaseModel):
    """行走报告"""
    is_walking: bool = Field(..., description="是否在行走中")
    gait_phase: str = Field(..., description="步态相位")
    walking_direction: str = Field(..., description="行走方向")
    current_step: int = Field(..., description="当前步数")
    total_steps: int = Field(..., description="总步数")
    walking_distance: float = Field(..., description="行走距离")
    timestamp: float = Field(..., description="报告时间戳")


class HumanoidReport(BaseModel):
    """人形机器人报告"""
    system_state: SystemState = Field(..., description="系统状态")
    balance_report: BalanceReport = Field(..., description="平衡报告")
    walking_report: WalkingReport = Field(..., description="行走报告")
    performance_metrics: Dict[str, Any] = Field(..., description="性能指标")
    active_tasks: int = Field(..., description="活跃任务数")
    task_success_rate: float = Field(..., description="任务成功率")
    timestamp: float = Field(..., description="报告时间戳")


class AutonomyAdjustment(BaseModel):
    """自主性调整"""
    autonomy_level: float = Field(..., ge=0.0, le=1.0, description="自主性水平 (0-1)")


def get_agi_architecture():
    """获取统一认知架构实例"""
    from ..cognitive.architecture import UnifiedCognitiveArchitecture
    # 这里应该从应用程序状态获取实例
    # 简化实现：创建新实例
    return UnifiedCognitiveArchitecture()


@router.get("/status", response_model=HumanoidReport)
async def get_humanoid_status(
    agi = Depends(get_agi_architecture)
):
    """获取人形机器人状态"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 获取人形机器人报告
        report = await agi.humanoid_agi.get_humanoid_report()
        
        # 提取子系统报告
        balance_report = report.get('balance_report', {})
        walking_report = report.get('walking_report', {})
        system_state = report.get('system_state', {})
        
        # 构建响应
        return HumanoidReport(
            system_state=SystemState(
                operational_mode=system_state.get('operational_mode', 'normal'),
                autonomy_level=system_state.get('autonomy_level', 0.7),
                safety_status=system_state.get('safety_status', 'safe'),
                battery_level=system_state.get('battery_level', 0.8),
                temperature=system_state.get('temperature', 25.0),
                total_operating_time=system_state.get('total_operating_time', 0.0)
            ),
            balance_report=BalanceReport(
                balance_state=balance_report.get('balance_state', 'stable'),
                stability_margin=balance_report.get('current_metrics', {}).get('stability_margin', 0.0),
                tilt_angles=balance_report.get('current_metrics', {}).get('tilt_angles', [0.0, 0.0]),
                active_strategy=balance_report.get('active_strategy', 'ankle_strategy'),
                timestamp=balance_report.get('timestamp', time.time())
            ),
            walking_report=WalkingReport(
                is_walking=walking_report.get('is_walking', False),
                gait_phase=walking_report.get('gait_phase', 'double_support'),
                walking_direction=walking_report.get('walking_direction', 'forward'),
                current_step=walking_report.get('current_step', 0),
                total_steps=walking_report.get('total_steps', 0),
                walking_distance=walking_report.get('performance_metrics', {}).get('walking_distance', 0.0),
                timestamp=walking_report.get('timestamp', time.time())
            ),
            performance_metrics=report.get('performance_metrics', {}),
            active_tasks=report.get('active_tasks', 0),
            task_success_rate=report.get('task_success_rate', 0.0),
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"获取人形机器人状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/execute", response_model=TaskResponse)
async def execute_humanoid_task(
    request: TaskRequest,
    agi = Depends(get_agi_architecture)
):
    """执行人形机器人任务"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 导入任务类型枚举
        from ..humanoid.humanoid_agi import HumanoidTaskType as InternalTaskType
        
        # 转换任务类型
        task_type_map = {
            HumanoidTaskType.WALK_TO_LOCATION: InternalTaskType.WALK_TO_LOCATION,
            HumanoidTaskType.PICK_UP_OBJECT: InternalTaskType.PICK_UP_OBJECT,
            HumanoidTaskType.PLACE_OBJECT: InternalTaskType.PLACE_OBJECT,
            HumanoidTaskType.INTERACT_WITH_HUMAN: InternalTaskType.INTERACT_WITH_HUMAN,
            HumanoidTaskType.NAVIGATE_ENVIRONMENT: InternalTaskType.NAVIGATE_ENVIRONMENT,
            HumanoidTaskType.MAINTAIN_BALANCE: InternalTaskType.MAINTAIN_BALANCE,
            HumanoidTaskType.PERFORM_GESTURE: InternalTaskType.PERFORM_GESTURE
        }
        
        internal_task_type = task_type_map.get(request.task_type)
        if not internal_task_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的任务类型: {request.task_type}"
            )
        
        # 执行任务
        result = await agi.humanoid_agi.execute_task(internal_task_type, request.parameters)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', '任务执行失败')
            )
        
        return TaskResponse(
            task_id=result.get('task_id', 'unknown'),
            success=True,
            message=result.get('result', {}).get('message', '任务已开始执行'),
            estimated_duration=result.get('result', {}).get('estimated_duration'),
            timestamp=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行人形机器人任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk/start")
async def start_walking(
    direction: WalkingDirection,
    distance: Optional[float] = None,
    agi = Depends(get_agi_architecture)
):
    """开始行走"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 导入行走方向枚举
        from ..humanoid.walking_gait import WalkingDirection as InternalWalkingDirection
        
        # 转换行走方向
        direction_map = {
            WalkingDirection.FORWARD: InternalWalkingDirection.FORWARD,
            WalkingDirection.BACKWARD: InternalWalkingDirection.BACKWARD,
            WalkingDirection.LEFT: InternalWalkingDirection.LEFT,
            WalkingDirection.RIGHT: InternalWalkingDirection.RIGHT,
            WalkingDirection.TURN_LEFT: InternalWalkingDirection.TURN_LEFT,
            WalkingDirection.TURN_RIGHT: InternalWalkingDirection.TURN_RIGHT
        }
        
        internal_direction = direction_map.get(direction)
        if not internal_direction:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的行走方向: {direction}"
            )
        
        # 通过任务执行行走
        result = await agi.humanoid_agi.execute_task(
            task_type=InternalTaskType.WALK_TO_LOCATION,  # 需要导入
            parameters={
                'direction': direction.value,
                'distance': distance
            }
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', '开始行走失败')
            )
        
        return {
            "success": True,
            "message": f"开始{direction.value}行走" + (f"，距离: {distance}米" if distance else ""),
            "task_id": result.get('task_id'),
            "estimated_duration": result.get('result', {}).get('estimated_duration'),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"开始行走失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk/stop")
async def stop_walking(
    agi = Depends(get_agi_architecture)
):
    """停止行走"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 停止行走
        result = await agi.humanoid_agi.walking_gait.stop_walking()
        
        return {
            "success": result.get('success', False),
            "message": result.get('message', '停止行走'),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"停止行走失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balance/report", response_model=BalanceReport)
async def get_balance_report(
    agi = Depends(get_agi_architecture)
):
    """获取平衡报告"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 获取平衡报告
        report = await agi.humanoid_agi.balance_control.get_balance_report()
        
        return BalanceReport(
            balance_state=report.get('balance_state', 'stable'),
            stability_margin=report.get('current_metrics', {}).get('stability_margin', 0.0),
            tilt_angles=report.get('current_metrics', {}).get('tilt_angles', [0.0, 0.0]),
            active_strategy=report.get('active_strategy', 'ankle_strategy'),
            timestamp=report.get('timestamp', time.time())
        )
        
    except Exception as e:
        logger.error(f"获取平衡报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/walking/report", response_model=WalkingReport)
async def get_walking_report(
    agi = Depends(get_agi_architecture)
):
    """获取行走报告"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 获取行走报告
        report = await agi.humanoid_agi.walking_gait.get_walking_report()
        
        return WalkingReport(
            is_walking=report.get('is_walking', False),
            gait_phase=report.get('gait_phase', 'double_support'),
            walking_direction=report.get('walking_direction', 'forward'),
            current_step=report.get('current_step', 0),
            total_steps=report.get('total_steps', 0),
            walking_distance=report.get('performance_metrics', {}).get('walking_distance', 0.0),
            timestamp=report.get('timestamp', time.time())
        )
        
    except Exception as e:
        logger.error(f"获取行走报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/autonomy/adjust")
async def adjust_autonomy_level(
    adjustment: AutonomyAdjustment,
    agi = Depends(get_agi_architecture)
):
    """调整自主性水平"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 调整自主性水平
        result = await agi.humanoid_agi.adjust_autonomy_level(adjustment.autonomy_level)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', '调整自主性水平失败')
            )
        
        return {
            "success": True,
            "autonomy_level": result.get('autonomy_level'),
            "operational_mode": result.get('operational_mode'),
            "max_walking_speed": result.get('max_walking_speed'),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"调整自主性水平失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency/stop")
async def emergency_stop(
    agi = Depends(get_agi_architecture)
):
    """紧急停止"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 执行紧急停止
        result = await agi.humanoid_agi.emergency_stop()
        
        return {
            "success": result.get('success', False),
            "message": result.get('message', '紧急停止'),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"紧急停止失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/active")
async def get_active_tasks(
    agi = Depends(get_agi_architecture)
):
    """获取活跃任务"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 这里应该从系统获取活跃任务列表
        # 简化实现：返回空列表
        return {
            "active_tasks": [],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取活跃任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/history")
async def get_task_history(
    limit: int = 10,
    agi = Depends(get_agi_architecture)
):
    """获取任务历史"""
    try:
        # 检查人形机器人AGI系统是否可用
        if not agi.humanoid_agi:
            raise HTTPException(
                status_code=503,
                detail="人形机器人AGI系统不可用"
            )
        
        # 这里应该从系统获取任务历史
        # 简化实现：返回空列表
        return {
            "task_history": [],
            "total_count": 0,
            "limit": limit,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取任务历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))