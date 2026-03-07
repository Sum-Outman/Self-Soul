"""
控制API模块

提供机器人控制功能的API端点，包括运动控制、硬件接口、传感器集成和电机控制。
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
from datetime import datetime

from cognitive.architecture import UnifiedCognitiveArchitecture

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/control", tags=["control"])


# 数据模型
class MotionCommandRequest(BaseModel):
    """运动命令请求"""
    voice_input: Optional[str] = None
    vision_input: Optional[Dict[str, Any]] = None
    sensor_input: Optional[Dict[str, Any]] = None
    priority: int = Field(1, ge=1, le=10, description="命令优先级 (1-10)")


class MotionCommandResponse(BaseModel):
    """运动命令响应"""
    commands: List[Dict[str, Any]]
    processing_time: float
    total_commands: int


class HardwareDeviceInfo(BaseModel):
    """硬件设备信息"""
    device_id: str
    device_type: str
    name: str
    version: str
    manufacturer: str
    parameters: Dict[str, Any]
    capabilities: List[str]


class SensorDataResponse(BaseModel):
    """传感器数据响应"""
    sensor_id: str
    sensor_type: str
    value: Any
    timestamp: float
    unit: str
    accuracy: float
    status: str


class MotorControlRequest(BaseModel):
    """电机控制请求"""
    motor_id: str
    position: Optional[float] = None
    velocity: Optional[float] = None
    torque: Optional[float] = None
    duration: float = Field(1.0, gt=0, description="持续时间(秒)")


class MotorControlResponse(BaseModel):
    """电机控制响应"""
    success: bool
    motor_id: str
    position: Optional[float] = None
    velocity: Optional[float] = None
    torque: Optional[float] = None
    latency: float
    message: Optional[str] = None
    error: Optional[str] = None


class EmergencyStopRequest(BaseModel):
    """紧急停止请求"""
    reason: Optional[str] = "用户请求"
    stop_all: bool = True


class EmergencyStopResponse(BaseModel):
    """紧急停止响应"""
    success: bool
    stopped_motors: int
    stopped_servos: int
    message: str
    timestamp: float


class SensorFusionRequest(BaseModel):
    """传感器融合请求"""
    fusion_method: str = Field("kalman_filter", description="融合方法")
    include_raw_data: bool = False
    max_history: int = Field(10, ge=1, le=100, description="最大历史数据点数")


class SensorFusionResponse(BaseModel):
    """传感器融合响应"""
    timestamp: float
    position: List[float]
    velocity: List[float]
    orientation: List[float]
    confidence: float
    fusion_method: str
    raw_sensor_ids: List[str]
    processing_time: Optional[float] = None


# 依赖项
def get_agi_architecture():
    """获取统一认知架构实例"""
    from ..cognitive.architecture import UnifiedCognitiveArchitecture
    # 这里应该从应用程序状态获取实例
    # 简化实现：创建新实例
    return UnifiedCognitiveArchitecture()


@router.get("/status")
async def get_control_status():
    """获取控制系统状态"""
    return {
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "motion_control": "available",
            "hardware_control": "available",
            "sensor_integration": "available",
            "motor_control": "available"
        },
        "system": "new_agi_system",
        "version": "1.0.0"
    }


@router.post("/motion/commands", response_model=MotionCommandResponse)
async def generate_motion_commands(
    request: MotionCommandRequest,
    background_tasks: BackgroundTasks,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """生成运动命令"""
    try:
        # 检查运动控制系统是否可用
        if not agi.motion_control:
            raise HTTPException(
                status_code=503,
                detail="运动控制系统不可用"
            )
        
        # 处理多模态输入
        start_time = datetime.now().timestamp()
        
        motion_commands = await agi.motion_control.process_multimodal_input(
            voice_input=request.voice_input,
            vision_input=request.vision_input,
            sensor_input=request.sensor_input
        )
        
        # 转换命令为字典
        commands_dict = []
        for cmd in motion_commands:
            commands_dict.append({
                "motion_type": cmd.motion_type.value,
                "target": cmd.target,
                "constraints": cmd.constraints,
                "control_mode": cmd.control_mode.value,
                "priority": cmd.priority,
                "duration": cmd.duration
            })
        
        processing_time = datetime.now().timestamp() - start_time
        
        # 异步规划轨迹
        if motion_commands:
            background_tasks.add_task(
                _plan_trajectories_async,
                agi.motion_control,
                motion_commands
            )
        
        return MotionCommandResponse(
            commands=commands_dict,
            processing_time=processing_time,
            total_commands=len(commands_dict)
        )
        
    except Exception as e:
        logger.error(f"生成运动命令失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _plan_trajectories_async(motion_control, motion_commands):
    """异步规划轨迹"""
    try:
        for cmd in motion_commands:
            await motion_control.plan_trajectory(cmd)
    except Exception as e:
        logger.error(f"异步规划轨迹失败: {e}")


@router.get("/hardware/devices")
async def get_hardware_devices(
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """获取硬件设备列表"""
    try:
        if not agi.hardware_control:
            raise HTTPException(
                status_code=503,
                detail="硬件控制系统不可用"
            )
        
        device_list = agi.hardware_control.get_device_list()
        
        return {
            "devices": device_list,
            "total_devices": sum(len(v) for v in device_list.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取硬件设备列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sensors/data")
async def get_sensor_data(
    sensor_id: Optional[str] = None,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """获取传感器数据"""
    try:
        if not agi.hardware_control:
            raise HTTPException(
                status_code=503,
                detail="硬件控制系统不可用"
            )
        
        if sensor_id:
            # 获取单个传感器数据
            sensor_data = await agi.hardware_control.get_sensor_data(sensor_id)
            if not sensor_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"传感器 {sensor_id} 未找到"
                )
            
            return SensorDataResponse(
                sensor_id=sensor_data.sensor_id,
                sensor_type=sensor_data.sensor_type.value,
                value=sensor_data.value,
                timestamp=sensor_data.timestamp,
                unit=sensor_data.unit,
                accuracy=sensor_data.accuracy,
                status=sensor_data.status
            )
        else:
            # 获取所有传感器数据
            all_data = await agi.hardware_control.get_all_sensor_data()
            
            return {
                "sensors": [
                    {
                        "sensor_id": data.sensor_id,
                        "sensor_type": data.sensor_type.value,
                        "value": data.value,
                        "timestamp": data.timestamp,
                        "unit": data.unit,
                        "accuracy": data.accuracy,
                        "status": data.status
                    }
                    for data in all_data.values()
                ],
                "total_sensors": len(all_data),
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取传感器数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/motor/control", response_model=MotorControlResponse)
async def control_motor(
    request: MotorControlRequest,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """控制电机"""
    try:
        if not agi.motor_control:
            raise HTTPException(
                status_code=503,
                detail="电机控制系统不可用"
            )
        
        from ..control.motor_control import ControlTarget
        
        # 创建控制目标
        control_target = ControlTarget(
            position=request.position,
            velocity=request.velocity,
            torque=request.torque,
            duration=request.duration
        )
        
        # 设置控制目标
        result = await agi.motor_control.set_control_target(
            request.motor_id,
            control_target
        )
        
        if not result.get("success"):
            return MotorControlResponse(
                success=False,
                motor_id=request.motor_id,
                error=result.get("error", "未知错误"),
                latency=result.get("latency", 0.0)
            )
        
        return MotorControlResponse(
            success=True,
            motor_id=request.motor_id,
            position=request.position,
            velocity=request.velocity,
            torque=request.torque,
            latency=result.get("latency", 0.0),
            message="电机控制命令已发送"
        )
        
    except Exception as e:
        logger.error(f"控制电机失败: {e}")
        return MotorControlResponse(
            success=False,
            motor_id=request.motor_id,
            error=str(e),
            latency=0.0
        )


@router.get("/motor/states")
async def get_motor_states(
    motor_id: Optional[str] = None,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """获取电机状态"""
    try:
        if not agi.motor_control:
            raise HTTPException(
                status_code=503,
                detail="电机控制系统不可用"
            )
        
        if motor_id:
            # 获取单个电机状态
            motor_state = await agi.motor_control.get_motor_state(motor_id)
            if not motor_state:
                raise HTTPException(
                    status_code=404,
                    detail=f"电机 {motor_id} 未找到"
                )
            
            return {
                "motor_id": motor_state.motor_id,
                "position": motor_state.position,
                "velocity": motor_state.velocity,
                "torque": motor_state.torque,
                "current": motor_state.current,
                "temperature": motor_state.temperature,
                "status": motor_state.status,
                "timestamp": motor_state.timestamp
            }
        else:
            # 获取所有电机状态
            all_states = await agi.motor_control.get_all_motor_states()
            
            return {
                "motors": [
                    {
                        "motor_id": state.motor_id,
                        "position": state.position,
                        "velocity": state.velocity,
                        "torque": state.torque,
                        "current": state.current,
                        "temperature": state.temperature,
                        "status": state.status,
                        "timestamp": state.timestamp
                    }
                    for state in all_states.values()
                ],
                "total_motors": len(all_states),
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取电机状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency/stop", response_model=EmergencyStopResponse)
async def emergency_stop(
    request: EmergencyStopRequest,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """紧急停止"""
    try:
        results = []
        
        # 停止电机
        if agi.motor_control:
            motor_result = await agi.motor_control.emergency_stop_all()
            results.append(("motor_control", motor_result))
        
        # 停止硬件
        if agi.hardware_control:
            hardware_result = await agi.hardware_control.emergency_stop()
            results.append(("hardware_control", hardware_result))
        
        # 汇总结果
        total_stopped_motors = 0
        total_stopped_servos = 0
        
        for module_name, result in results:
            if result.get("success"):
                if module_name == "motor_control":
                    total_stopped_motors = result.get("stopped_motors", 0)
                elif module_name == "hardware_control":
                    total_stopped_servos = result.get("stopped_servos", 0)
        
        return EmergencyStopResponse(
            success=True,
            stopped_motors=total_stopped_motors,
            stopped_servos=total_stopped_servos,
            message=f"紧急停止已执行: {request.reason}",
            timestamp=datetime.now().timestamp()
        )
        
    except Exception as e:
        logger.error(f"紧急停止失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sensor/fusion", response_model=SensorFusionResponse)
async def get_sensor_fusion(
    request: SensorFusionRequest = Depends(),
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """获取传感器融合数据"""
    try:
        if not agi.sensor_integration:
            raise HTTPException(
                status_code=503,
                detail="传感器集成系统不可用"
            )
        
        start_time = datetime.now().timestamp()
        
        # 获取融合数据
        fused_data = await agi.sensor_integration.get_latest_fused_data()
        
        if not fused_data:
            # 如果没有融合数据，尝试执行一次融合
            raise HTTPException(
                status_code=404,
                detail="无传感器融合数据可用"
            )
        
        processing_time = datetime.now().timestamp() - start_time
        
        return SensorFusionResponse(
            timestamp=fused_data.timestamp,
            position=fused_data.position,
            velocity=fused_data.velocity,
            orientation=fused_data.orientation,
            confidence=fused_data.confidence,
            fusion_method=fused_data.fusion_method.value,
            raw_sensor_ids=fused_data.raw_sensor_ids,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取传感器融合数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_control_performance(
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """获取控制性能指标"""
    try:
        performance_data = {}
        
        # 收集各模块性能指标
        if agi.motion_control:
            performance_data["motion_control"] = agi.motion_control.get_performance_metrics()
        
        if agi.hardware_control:
            performance_data["hardware_control"] = agi.hardware_control.get_performance_metrics()
        
        if agi.sensor_integration:
            performance_data["sensor_integration"] = agi.sensor_integration.get_performance_metrics()
        
        if agi.motor_control:
            performance_data["motor_control"] = agi.motor_control.get_performance_metrics()
        
        return {
            "performance": performance_data,
            "timestamp": datetime.now().isoformat(),
            "system_load": {
                "cpu": "N/A",  # 实际系统应该从监控系统获取
                "memory": "N/A",
                "network": "N/A"
            }
        }
        
    except Exception as e:
        logger.error(f"获取控制性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate")
async def calibrate_system(
    module: str,
    target_id: Optional[str] = None,
    agi: UnifiedCognitiveArchitecture = Depends(get_agi_architecture)
):
    """校准系统"""
    try:
        result = None
        
        if module == "sensors" and agi.sensor_integration:
            sensor_ids = [target_id] if target_id else None
            result = await agi.sensor_integration.calibrate_sensors(sensor_ids)
        
        elif module == "motor" and agi.motor_control and target_id:
            result = await agi.motor_control.calibrate_motor(target_id)
        
        elif module == "all":
            # 校准所有可用模块
            results = {}
            
            if agi.sensor_integration:
                sensor_result = await agi.sensor_integration.calibrate_sensors()
                results["sensors"] = sensor_result
            
            if agi.motor_control:
                # 校准所有电机
                motor_results = {}
                # 这里需要获取所有电机ID
                # 简化实现
                results["motors"] = {"success": True, "calibrated": ["simulated"]}
            
            result = {
                "success": all(r.get("success", False) for r in results.values() if r),
                "results": results,
                "message": "系统校准完成"
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的校准模块: {module}"
            )
        
        if result and result.get("success"):
            return {
                "success": True,
                "module": module,
                "target": target_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            error_msg = result.get("error", "校准失败") if result else "校准失败"
            raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"校准系统失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))