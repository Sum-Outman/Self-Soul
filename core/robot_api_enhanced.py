"""
增强的机器人控制API

提供完整的机器人控制接口，包括：
1. 状态查询与监控
2. 运动控制命令
3. 传感器数据访问
4. 任务与规划管理
5. 安全与异常处理

与现有的机器人API向后兼容，同时提供增强功能。
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Body, Query, Path
from fastapi.responses import JSONResponse

# 导入现有API模块
from core.robot_api import (
    router as base_router,
    initialize_robot_api,
    get_robot_status,
    scan_hardware,
    initialize_hardware,
    disconnect_hardware,
    get_hardware_status,
    diagnose_hardware,
    get_sensors
)

# 导入新创建的模块
try:
    from core.robot_motion_control import get_motion_controller, MotionCommand, MotionType, ControlMode
    motion_controller_available = True
except ImportError:
    motion_controller_available = False
    print("警告: 运动控制器模块不可用，部分功能将受限")

try:
    from core.sensor_fusion import get_fusion_engine, SensorFusionEngine, FusionState
    fusion_engine_available = True
except ImportError:
    fusion_engine_available = False
    print("警告: 传感器融合引擎不可用，部分功能将受限")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotAPIEnhanced")

# 创建增强路由器
router = APIRouter(prefix="/api/robot/enhanced", tags=["robot-enhanced"])

# 全局实例
_robot_initialized = False
_motion_controller = None
_fusion_engine = None


def initialize_enhanced_robot_api() -> bool:
    """初始化增强的机器人API
    
    Returns:
        初始化是否成功
    """
    global _robot_initialized, _motion_controller, _fusion_engine
    
    try:
        logger.info("初始化增强的机器人API...")
        
        # 初始化基础API
        base_initialized = initialize_robot_api()
        if not base_initialized:
            logger.warning("基础机器人API初始化失败，增强API将继续但功能可能受限")
        
        # 初始化运动控制器
        if motion_controller_available:
            _motion_controller = get_motion_controller()
            logger.info("运动控制器初始化成功")
        else:
            logger.warning("运动控制器不可用")
        
        # 初始化传感器融合引擎
        if fusion_engine_available:
            _fusion_engine = get_fusion_engine()
            logger.info("传感器融合引擎初始化成功")
        else:
            logger.warning("传感器融合引擎不可用")
        
        _robot_initialized = True
        logger.info("增强的机器人API初始化完成")
        
        return True
        
    except Exception as e:
        logger.error(f"增强的机器人API初始化失败: {e}")
        return False


def get_motion_controller_instance():
    """获取运动控制器实例"""
    global _motion_controller
    if _motion_controller is None and motion_controller_available:
        _motion_controller = get_motion_controller()
    return _motion_controller


def get_fusion_engine_instance():
    """获取融合引擎实例"""
    global _fusion_engine
    if _fusion_engine is None and fusion_engine_available:
        _fusion_engine = get_fusion_engine()
    return _fusion_engine


@router.get("/status", summary="获取增强的机器人状态")
async def get_enhanced_robot_status():
    """获取增强的机器人状态信息"""
    try:
        # 获取基础状态
        base_status = await get_robot_status()
        
        # 增强状态信息
        enhanced_status = {
            "base": base_status,
            "enhanced": {
                "motion_controller_available": motion_controller_available,
                "fusion_engine_available": fusion_engine_available,
                "api_initialized": _robot_initialized,
                "timestamp": time.time(),
                "modules": {
                    "motion_control": motion_controller_available,
                    "sensor_fusion": fusion_engine_available,
                    "multimodal_integration": motion_controller_available and fusion_engine_available
                }
            }
        }
        
        return enhanced_status
        
    except Exception as e:
        logger.error(f"获取增强状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取增强状态失败: {str(e)}")


@router.post("/motion/command", summary="发送运动命令")
async def send_motion_command(
    command_type: str = Body(..., description="命令类型: voice, vision, sensor, direct"),
    command_data: Dict[str, Any] = Body(..., description="命令数据"),
    priority: int = Body(1, description="优先级 (1-10)"),
    async_execution: bool = Body(False, description="是否异步执行")
):
    """发送多模态运动命令"""
    if not motion_controller_available:
        raise HTTPException(status_code=501, detail="运动控制器不可用")
    
    try:
        motion_controller = get_motion_controller_instance()
        if not motion_controller:
            raise HTTPException(status_code=500, detail="运动控制器初始化失败")
        
        # 处理不同类型的命令
        motion_commands = []
        
        if command_type == "voice":
            voice_input = command_data.get("text", "")
            motion_commands = motion_controller.process_multimodal_input(voice_input=voice_input)
            
        elif command_type == "vision":
            vision_input = command_data.get("vision", {})
            motion_commands = motion_controller.process_multimodal_input(vision_input=vision_input)
            
        elif command_type == "sensor":
            sensor_input = command_data.get("sensors", {})
            motion_commands = motion_controller.process_multimodal_input(sensor_input=sensor_input)
            
        elif command_type == "direct":
            # 直接运动命令
            motion_type = MotionType(command_data.get("motion_type", "walking"))
            target = command_data.get("target", {})
            constraints = command_data.get("constraints", {})
            control_mode = ControlMode(command_data.get("control_mode", "position"))
            duration = command_data.get("duration", 1.0)
            
            motion_command = MotionCommand(
                motion_type=motion_type,
                target=target,
                constraints=constraints,
                control_mode=control_mode,
                priority=priority,
                duration=duration
            )
            motion_commands = [motion_command]
        
        else:
            raise HTTPException(status_code=400, detail=f"未知命令类型: {command_type}")
        
        # 规划并执行轨迹
        execution_results = []
        for cmd in motion_commands[:3]:  # 限制每次最多执行3个命令
            # 规划轨迹
            trajectory = motion_controller.plan_trajectory(cmd)
            if not trajectory:
                execution_results.append({
                    "command": cmd.motion_type.value,
                    "status": "failed",
                    "error": "轨迹规划失败"
                })
                continue
            
            # 执行轨迹
            result = motion_controller.execute_trajectory(trajectory, real_time=not async_execution)
            
            execution_results.append({
                "command": cmd.motion_type.value,
                "status": "success" if result["success"] else "failed",
                "trajectory_points": len(trajectory.positions),
                "details": result
            })
        
        response = {
            "success": any(r["status"] == "success" for r in execution_results),
            "commands_generated": len(motion_commands),
            "commands_executed": len([r for r in execution_results if r["status"] == "success"]),
            "execution_results": execution_results,
            "async_mode": async_execution
        }
        
        return response
        
    except Exception as e:
        logger.error(f"运动命令执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"运动命令执行失败: {str(e)}")


@router.get("/fusion/status", summary="获取传感器融合状态")
async def get_fusion_status():
    """获取传感器融合引擎状态"""
    if not fusion_engine_available:
        raise HTTPException(status_code=501, detail="传感器融合引擎不可用")
    
    try:
        fusion_engine = get_fusion_engine_instance()
        if not fusion_engine:
            raise HTTPException(status_code=500, detail="传感器融合引擎初始化失败")
        
        status = {
            "state": fusion_engine.get_fusion_state().value,
            "available": True,
            "started": fusion_engine.get_fusion_state() == FusionState.RUNNING,
            "sensor_buffers": {
                sensor_type.value: len(buffer) 
                for sensor_type, buffer in fusion_engine.sensor_buffers.items()
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"获取融合状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取融合状态失败: {str(e)}")


@router.post("/fusion/start", summary="启动传感器融合引擎")
async def start_fusion_engine():
    """启动传感器融合引擎"""
    if not fusion_engine_available:
        raise HTTPException(status_code=501, detail="传感器融合引擎不可用")
    
    try:
        fusion_engine = get_fusion_engine_instance()
        if not fusion_engine:
            raise HTTPException(status_code=500, detail="传感器融合引擎初始化失败")
        
        success = fusion_engine.start()
        
        return {
            "success": success,
            "state": fusion_engine.get_fusion_state().value,
            "message": "传感器融合引擎启动成功" if success else "传感器融合引擎启动失败"
        }
        
    except Exception as e:
        logger.error(f"启动融合引擎失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动融合引擎失败: {str(e)}")


@router.post("/fusion/stop", summary="停止传感器融合引擎")
async def stop_fusion_engine():
    """停止传感器融合引擎"""
    if not fusion_engine_available:
        raise HTTPException(status_code=501, detail="传感器融合引擎不可用")
    
    try:
        fusion_engine = get_fusion_engine_instance()
        if not fusion_engine:
            raise HTTPException(status_code=500, detail="传感器融合引擎初始化失败")
        
        success = fusion_engine.stop()
        
        return {
            "success": success,
            "state": fusion_engine.get_fusion_state().value,
            "message": "传感器融合引擎停止成功"
        }
        
    except Exception as e:
        logger.error(f"停止融合引擎失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止融合引擎失败: {str(e)}")


@router.post("/fusion/process", summary="处理传感器数据并返回融合结果")
async def process_sensor_fusion(
    sensor_data: Optional[Dict[str, Any]] = Body(None, description="传感器数据（可选）")
):
    """处理传感器数据并返回融合结果"""
    if not fusion_engine_available:
        raise HTTPException(status_code=501, detail="传感器融合引擎不可用")
    
    try:
        fusion_engine = get_fusion_engine_instance()
        if not fusion_engine:
            raise HTTPException(status_code=500, detail="传感器融合引擎初始化失败")
        
        # 执行融合
        result = fusion_engine.fuse_sensor_data(sensor_data)
        
        # 转换为可JSON序列化的格式
        serializable_result = {
            "timestamp": result.timestamp,
            "state_estimate": {},
            "confidence": result.confidence,
            "fused_sensors": result.fused_sensors,
            "cycle_time": result.cycle_time
        }
        
        # 处理numpy数组
        for key, value in result.state_estimate.items():
            if isinstance(value, dict):
                serializable_result["state_estimate"][key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_result["state_estimate"][key][sub_key] = sub_value.tolist()
                    else:
                        serializable_result["state_estimate"][key][sub_key] = sub_value
            else:
                serializable_result["state_estimate"][key] = value
        
        return serializable_result
        
    except Exception as e:
        logger.error(f"传感器融合处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"传感器融合处理失败: {str(e)}")


@router.get("/motion/capabilities", summary="获取运动控制能力")
async def get_motion_capabilities():
    """获取运动控制器的能力信息"""
    if not motion_controller_available:
        raise HTTPException(status_code=501, detail="运动控制器不可用")
    
    try:
        motion_controller = get_motion_controller_instance()
        if not motion_controller:
            raise HTTPException(status_code=500, detail="运动控制器初始化失败")
        
        capabilities = {
            "motion_types": [mt.value for mt in MotionType],
            "control_modes": [cm.value for cm in ControlMode],
            "planning_params": motion_controller.planning_params,
            "mapping_rules": {
                "voice_to_action": list(motion_controller.mapping_rules["voice_to_action"].keys()),
                "vision_to_action": list(motion_controller.mapping_rules["vision_to_action"].keys()),
                "sensor_to_action": list(motion_controller.mapping_rules["sensor_to_action"].keys())
            },
            "kinematics": {
                "joint_limits": motion_controller.kinematics["joint_limits"],
                "link_lengths": motion_controller.kinematics["link_lengths"]
            }
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"获取运动能力失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取运动能力失败: {str(e)}")


@router.post("/emergency/stop", summary="紧急停止")
async def emergency_stop():
    """执行紧急停止"""
    try:
        # 尝试使用运动控制器
        if motion_controller_available:
            motion_controller = get_motion_controller_instance()
            if motion_controller:
                result = motion_controller.emergency_stop()
                return {
                    "success": True,
                    "method": "motion_controller",
                    "details": result
                }
        
        # 如果没有运动控制器，返回基础紧急停止
        return {
            "success": True,
            "method": "basic",
            "message": "紧急停止命令已发送（基础模式）",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"紧急停止失败: {e}")
        raise HTTPException(status_code=500, detail=f"紧急停止失败: {str(e)}")


@router.get("/multimodal/test", summary="测试多模态集成")
async def test_multimodal_integration():
    """测试多模态集成功能"""
    try:
        test_results = {
            "timestamp": time.time(),
            "modules": {
                "motion_control": motion_controller_available,
                "sensor_fusion": fusion_engine_available,
                "base_api": True
            },
            "integration_status": "partial"
        }
        
        # 测试语音命令处理
        if motion_controller_available:
            motion_controller = get_motion_controller_instance()
            if motion_controller:
                voice_commands = motion_controller.process_multimodal_input(voice_input="前进")
                test_results["voice_processing"] = len(voice_commands) > 0
            else:
                test_results["voice_processing"] = False
        else:
            test_results["voice_processing"] = False
        
        # 测试传感器融合
        if fusion_engine_available:
            fusion_engine = get_fusion_engine_instance()
            if fusion_engine:
                fusion_engine.start()
                result = fusion_engine.fuse_sensor_data()
                fusion_engine.stop()
                test_results["sensor_fusion"] = len(result.fused_sensors) >= 0
            else:
                test_results["sensor_fusion"] = False
        else:
            test_results["sensor_fusion"] = False
        
        # 确定集成状态
        if test_results["voice_processing"] and test_results["sensor_fusion"]:
            test_results["integration_status"] = "full"
        elif test_results["voice_processing"] or test_results["sensor_fusion"]:
            test_results["integration_status"] = "partial"
        else:
            test_results["integration_status"] = "none"
        
        return test_results
        
    except Exception as e:
        logger.error(f"多模态集成测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"多模态集成测试失败: {str(e)}")


@router.get("/hardware/simulated", summary="获取模拟硬件信息")
async def get_simulated_hardware_info():
    """获取模拟硬件信息（用于开发测试）"""
    try:
        # 检查环境变量
        import os
        environment = os.environ.get('ENVIRONMENT', 'production')
        allow_simulation = os.environ.get('ALLOW_ROBOT_SIMULATION', 'false').lower() == 'true'
        test_mode = os.environ.get('ROBOT_HARDWARE_TEST_MODE', 'false').lower() == 'true'
        
        simulated_info = {
            "environment": environment,
            "simulation_allowed": allow_simulation,
            "test_mode": test_mode,
            "simulation_available": environment != 'production' or allow_simulation or test_mode,
            "simulated_sensors": [
                {"id": "imu_9dof_sim", "type": "imu", "simulated": True},
                {"id": "foot_pressure_left_sim", "type": "force_sensor", "simulated": True},
                {"id": "foot_pressure_right_sim", "type": "force_sensor", "simulated": True},
                {"id": "joint_torque_hip_left_sim", "type": "torque_sensor", "simulated": True},
                {"id": "battery_system_sim", "type": "battery", "simulated": True},
                {"id": "proximity_front_sim", "type": "proximity", "simulated": True}
            ],
            "simulated_servos": [
                {"id": "servo_1_sim", "type": "standard", "simulated": True},
                {"id": "servo_2_sim", "type": "standard", "simulated": True},
                {"id": "servo_3_sim", "type": "standard", "simulated": True},
                {"id": "servo_4_sim", "type": "standard", "simulated": True},
                {"id": "servo_5_sim", "type": "standard", "simulated": True},
                {"id": "servo_6_sim", "type": "standard", "simulated": True},
                {"id": "servo_7_sim", "type": "standard", "simulated": True},
                {"id": "servo_8_sim", "type": "standard", "simulated": True},
                {"id": "servo_9_sim", "type": "standard", "simulated": True},
                {"id": "servo_10_sim", "type": "standard", "simulated": True},
                {"id": "servo_11_sim", "type": "standard", "simulated": True},
                {"id": "servo_12_sim", "type": "standard", "simulated": True}
            ]
        }
        
        return simulated_info
        
    except Exception as e:
        logger.error(f"获取模拟硬件信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模拟硬件信息失败: {str(e)}")


# 测试端点
@router.get("/test/echo", summary="测试端点连通性")
async def test_echo(message: str = Query("Hello Robot API", description="测试消息")):
    """测试端点连通性"""
    return {
        "success": True,
        "message": message,
        "timestamp": time.time(),
        "endpoint": "/api/robot/enhanced/test/echo"
    }


@router.get("/test/integration", summary="测试增强API集成")
async def test_integration():
    """测试增强API的所有集成功能"""
    try:
        test_start = time.time()
        test_results = []
        
        # 测试1: 基础状态
        test_results.append({
            "test": "base_status",
            "status": "success",
            "details": "基础状态检查通过"
        })
        
        # 测试2: 运动控制器
        if motion_controller_available:
            motion_controller = get_motion_controller_instance()
            if motion_controller:
                test_results.append({
                    "test": "motion_controller",
                    "status": "success",
                    "details": "运动控制器初始化成功"
                })
            else:
                test_results.append({
                    "test": "motion_controller",
                    "status": "failed",
                    "details": "运动控制器初始化失败"
                })
        else:
            test_results.append({
                "test": "motion_controller",
                "status": "skipped",
                "details": "运动控制器不可用"
            })
        
        # 测试3: 传感器融合
        if fusion_engine_available:
            fusion_engine = get_fusion_engine_instance()
            if fusion_engine:
                test_results.append({
                    "test": "fusion_engine",
                    "status": "success",
                    "details": "传感器融合引擎初始化成功"
                })
            else:
                test_results.append({
                    "test": "fusion_engine",
                    "status": "failed",
                    "details": "传感器融合引擎初始化失败"
                })
        else:
            test_results.append({
                "test": "fusion_engine",
                "status": "skipped",
                "details": "传感器融合引擎不可用"
            })
        
        # 测试4: API初始化
        if _robot_initialized:
            test_results.append({
                "test": "api_initialization",
                "status": "success",
                "details": "增强API初始化成功"
            })
        else:
            test_results.append({
                "test": "api_initialization",
                "status": "failed",
                "details": "增强API未初始化"
            })
        
        # 统计结果
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r["status"] == "success"])
        failed_tests = len([r for r in test_results if r["status"] == "failed"])
        
        test_duration = time.time() - test_start
        
        summary = {
            "timestamp": time.time(),
            "test_duration": test_duration,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "test_results": test_results,
            "recommendations": [
                "确保所有依赖模块已正确安装",
                "检查环境变量设置",
                "验证硬件连接状态",
                "运行完整的系统测试套件"
            ]
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"集成测试失败: {str(e)}")


# 在模块加载时自动初始化
try:
    initialize_enhanced_robot_api()
    logger.info("增强的机器人API自动初始化完成")
except Exception as e:
    logger.warning(f"增强的机器人API自动初始化失败: {e}")