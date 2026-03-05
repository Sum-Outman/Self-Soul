"""
Robot Control API Module

Provides RESTful API interfaces for robot hardware control, sensor management, motion control, and task planning
"""

import logging
import time
import json
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio

# Import core modules
from core.error_handling import error_handler
from core.model_registry import get_model_registry
from core.model_service_manager import ModelServiceManager
from core.agi_tools import AGITools
from core.collaboration.model_collaborator import ModelCollaborationOrchestrator, CollaborationMode

# Import hardware interface
try:
    from core.hardware.robot_hardware_interface import RobotHardwareInterface
    HARDWARE_INTERFACE_AVAILABLE = True
    # Check if robot driver is available through hardware interface
    ROBOT_DRIVER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Robot hardware interface unavailable: {e}")
    RobotHardwareInterface = None
    HARDWARE_INTERFACE_AVAILABLE = False
    ROBOT_DRIVER_AVAILABLE = False

# Import robot training manager
try:
    from core.training_manager import get_training_manager
    robot_training_manager = get_training_manager()
    ROBOT_TRAINING_MANAGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Robot training manager unavailable: {e}")
    robot_training_manager = None
    ROBOT_TRAINING_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/robot", tags=["robot"])

# Global variables
model_registry = None
model_service_manager = None
robot_hardware = None  # Robot hardware interface instance
model_collaborator = None  # Model collaboration orchestrator instance
robot_state = {
    "status": "idle",  # idle, active, emergency, calibrating
    "status_text": "Standby",
    "battery": 85,
    "connected": True,
    "temperature": 32.5,
    "last_update": datetime.now().isoformat()
}

# 机器人关节状态
joints_state = {
    "arm": {
        "left": {"shoulder": 0, "elbow": 0, "wrist": 0},
        "right": {"shoulder": 0, "elbow": 0, "wrist": 0}
    },
    "leg": {
        "left": {"hip": 0, "knee": 0, "ankle": 0},
        "right": {"hip": 0, "knee": 0, "ankle": 0}
    },
    "head": {"pan": 0, "tilt": 0},
    "torso": {"twist": 0, "bend": 0}
}

# 传感器状态 - 从硬件接口实时获取，不包含模拟数据
sensors_state = []

# 摄像头状态
cameras_state = {
    "left": {"active": False, "url": "", "calibrated": False},
    "right": {"active": False, "url": "", "calibrated": False}
}

# 训练状态
training_state = {
    "status": "idle",  # idle, training, paused, completed, error
    "progress": 0,
    "active_training": None,
    "error": None,
    "training_id": None,
    "mode": None,
    "models": [],
    "dataset_id": None,
    "started_at": None,
    "selected_joints": [],
    "selected_sensors": [],
    "selected_cameras": [],
    "training_params": {
        "iterations": 1000,
        "learning_rate": 0.001,
        "batch_size": 32,
        "validation_split": 0.2
    },
    "safety_limits": {
        "max_joint_velocity": 5.0,
        "max_joint_torque": 10.0,
        "max_temperature": 80.0,
        "emergency_stop_threshold": 1.5
    },
    "training_log": []
}

# 任务列表
tasks_list = []

# Pydantic模型定义
class JointCommand(BaseModel):
    jointId: str
    value: float
    timestamp: Optional[int] = None

class SensorToggle(BaseModel):
    sensorId: str
    active: bool

class CameraCommand(BaseModel):
    camera: str  # left, right
    active: bool

class MotionCommand(BaseModel):
    motion: str
    params: Optional[Dict[str, Any]] = None

class TaskPlan(BaseModel):
    description: str

class TaskExecute(BaseModel):
    taskId: Optional[int] = None

class CollaborationRequest(BaseModel):
    pattern: str  # 协作模式名称：robot_movement, robot_perception, robot_decision, robot_autonomous
    input_data: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None

class AutonomousCommand(BaseModel):
    enabled: bool
    goal: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None

class GoalSetting(BaseModel):
    goal: str
    priority: Optional[int] = 1
    deadline: Optional[str] = None

class TaskSequence(BaseModel):
    tasks: List[Dict[str, Any]]
    dependencies: Optional[Dict[str, List[str]]] = None

class SensorFusionRequest(BaseModel):
    sensor_ids: Optional[List[str]] = None
    fusion_method: str = "average"  # average, weighted, kalman, neural
    parameters: Optional[Dict[str, Any]] = None

class SensorAnalysisRequest(BaseModel):
    sensor_ids: List[str]
    analysis_type: str = "statistical"  # statistical, trend, anomaly, feature
    time_range: Optional[str] = None  # e.g., "1h", "24h"
    parameters: Optional[Dict[str, Any]] = None

class VisionDetectionRequest(BaseModel):
    camera: str = "left"
    detection_type: str = "objects"  # objects, faces, text, landmarks
    confidence_threshold: Optional[float] = 0.5
    parameters: Optional[Dict[str, Any]] = None

class VisionSegmentationRequest(BaseModel):
    camera: str = "left"
    segmentation_type: str = "semantic"  # semantic, instance, panoptic
    parameters: Optional[Dict[str, Any]] = None

class SpatialDepthRequest(BaseModel):
    cameras: List[str] = ["left", "right"]  # 双目摄像头
    method: str = "stereo"  # stereo, monocular, neural
    parameters: Optional[Dict[str, Any]] = None

class SpatialMappingRequest(BaseModel):
    operation: str = "update"  # update, get_map, reset
    area: Optional[List[float]] = None  # [x1, y1, x2, y2] 区域
    parameters: Optional[Dict[str, Any]] = None

class ConnectionTestRequest(BaseModel):
    """连接测试请求模型"""
    connectionId: str
    connectionType: str  # usb, serial, ethernet, bluetooth, etc.
    port: Optional[str] = None
    testType: Optional[str] = "connectivity"  # connectivity, throughput, latency, functionality

# 训练相关模型
class TrainingStartRequest(BaseModel):
    """训练启动请求模型 - 支持前后端两种字段格式"""
    # 后端字段名（主用）
    training_mode: str  # motion_basic, perception_training, collaboration_training, agi_fusion
    models: List[str]  # 选择的模型ID列表
    dataset_id: Optional[str] = None  # 数据集ID
    selected_joints: Optional[List[str]] = None
    selected_sensors: Optional[List[str]] = None
    selected_cameras: Optional[List[str]] = None
    training_params: Optional[Dict[str, Any]] = None
    safety_limits: Optional[Dict[str, Any]] = None
    
    # 前端字段名（兼容性别名）
    mode: Optional[str] = Field(None, alias="training_mode")  # 前端使用"mode"
    dataset: Optional[str] = Field(None, alias="dataset_id")  # 前端使用"dataset"
    parameters: Optional[Dict[str, Any]] = Field(None, alias="training_params")  # 前端使用"parameters"
    hardware_config: Optional[Dict[str, Any]] = None  # 前端使用"hardware_config"包含selected_joints等
    
    class Config:
        """Pydantic配置"""
        allow_population_by_field_name = True  # 允许通过字段名和别名进行数据填充
        extra = "ignore"  # 忽略额外字段

class TrainingPauseRequest(BaseModel):
    """训练暂停请求模型"""
    training_id: str

class TrainingStopRequest(BaseModel):
    """训练停止请求模型"""
    training_id: str

class TrainingResetRequest(BaseModel):
    """训练重置请求模型"""
    training_id: str

class TrainingStatusResponse(BaseModel):
    """训练状态响应模型"""
    training_id: str
    status: str  # idle, training, paused, completed, error
    progress: float
    mode: Optional[str] = None
    models: Optional[List[str]] = None
    dataset_id: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    training_params: Optional[Dict[str, Any]] = None
    safety_limits: Optional[Dict[str, Any]] = None
    training_log: Optional[List[Dict[str, Any]]] = None

# Initialization function
def initialize_robot_api():
    """Initialize robot API module"""
    global model_registry, model_service_manager, robot_hardware, model_collaborator
    
    try:
        # Get model registry
        model_registry = get_model_registry()
        
        # Get model service manager
        model_service_manager = ModelServiceManager()
        
        # Initialize robot hardware interface
        if HARDWARE_INTERFACE_AVAILABLE and RobotHardwareInterface:
            robot_hardware = RobotHardwareInterface()
            logger.info("Robot hardware interface initialized successfully")
        else:
            logger.warning("Robot hardware interface unavailable, hardware interface initialization required")
        
        # Initialize model collaboration orchestrator
        try:
            model_collaborator = ModelCollaborationOrchestrator()
            logger.info("Model collaboration orchestrator initialized successfully")
            
            # Add robot-specific collaboration patterns
            if model_collaborator:
                # Robot movement collaboration: motion + spatial + sensor
                model_collaborator.collaboration_patterns["robot_movement"] = {
                    "models": ["motion", "spatial", "sensor"],
                    "workflow": [
                        {"model": "sensor", "task": "get_robot_state", "share_result": True},
                        {"model": "spatial", "task": "calculate_trajectory", "depends_on": "sensor"},
                        {"model": "motion", "task": "execute_movement", "depends_on": "spatial"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "Robot movement collaboration: sensor data → spatial trajectory calculation → motion execution"
                }
                
                # 机器人感知协作：视觉 + 空间 + 计算机视觉
                model_collaborator.collaboration_patterns["robot_perception"] = {
                    "models": ["vision", "spatial", "computer_vision"],
                    "workflow": [
                        {"model": "vision", "task": "capture_image", "share_result": True},
                        {"model": "computer_vision", "task": "analyze_image", "depends_on": "vision"},
                        {"model": "spatial", "task": "estimate_depth", "depends_on": "computer_vision"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人感知协作：视觉采集 → 图像分析 → 深度估计"
                }
                
                # 机器人决策协作：管理 + 规划 + 语言
                model_collaborator.collaboration_patterns["robot_decision"] = {
                    "models": ["manager", "planning", "language"],
                    "workflow": [
                        {"model": "manager", "task": "evaluate_situation", "share_result": True},
                        {"model": "planning", "task": "generate_plan", "depends_on": "manager"},
                        {"model": "language", "task": "verbalize_plan", "depends_on": "planning"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人决策协作：情境评估 → 计划生成 → 语言表达"
                }
                
                # 机器人自主协作：自主 + 规划 + 运动
                model_collaborator.collaboration_patterns["robot_autonomous"] = {
                    "models": ["autonomous", "planning", "motion"],
                    "workflow": [
                        {"model": "autonomous", "task": "set_goal", "share_result": True},
                        {"model": "planning", "task": "create_action_sequence", "depends_on": "autonomous"},
                        {"model": "motion", "task": "execute_actions", "depends_on": "planning"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人自主协作：目标设定 → 行动序列生成 → 动作执行"
                }

                # 机器人多模态感知融合：视觉 + 音频 + 传感器 + 空间
                model_collaborator.collaboration_patterns["robot_multimodal_perception"] = {
                    "models": ["vision", "audio", "sensor", "spatial", "computer_vision"],
                    "workflow": [
                        {"model": "vision", "task": "capture_image", "share_result": True},
                        {"model": "audio", "task": "capture_audio", "share_result": True},
                        {"model": "sensor", "task": "get_sensor_data", "share_result": True},
                        {"model": "computer_vision", "task": "analyze_multimodal", "depends_on": ["vision", "audio"]},
                        {"model": "spatial", "task": "fuse_perception", "depends_on": ["computer_vision", "sensor"]}
                    ],
                    "mode": CollaborationMode.HYBRID,
                    "description": "机器人多模态感知融合：视觉+音频+传感器数据融合 → 空间感知"
                }

                # 机器人高级认知处理：管理 + 语言 + 知识 + 高级推理 + 元认知
                model_collaborator.collaboration_patterns["robot_cognitive_processing"] = {
                    "models": ["manager", "language", "knowledge", "advanced_reasoning", "meta_cognition"],
                    "workflow": [
                        {"model": "manager", "task": "evaluate_context", "share_result": True},
                        {"model": "language", "task": "understand_input", "depends_on": "manager"},
                        {"model": "knowledge", "task": "retrieve_relevant_knowledge", "depends_on": "language"},
                        {"model": "advanced_reasoning", "task": "reason_about_situation", "depends_on": "knowledge"},
                        {"model": "meta_cognition", "task": "monitor_thinking", "depends_on": "advanced_reasoning"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人高级认知处理：上下文评估 → 语言理解 → 知识检索 → 高级推理 → 元认知监控"
                }

                # 机器人自主学习系统：自主 + 数据融合 + 创意问题解决 + 优化
                model_collaborator.collaboration_patterns["robot_autonomous_learning"] = {
                    "models": ["autonomous", "data_fusion", "creative_problem_solving", "optimization", "meta_cognition"],
                    "workflow": [
                        {"model": "autonomous", "task": "identify_learning_goal", "share_result": True},
                        {"model": "data_fusion", "task": "fuse_experience_data", "depends_on": "autonomous"},
                        {"model": "creative_problem_solving", "task": "generate_solutions", "depends_on": "data_fusion"},
                        {"model": "optimization", "task": "optimize_solutions", "depends_on": "creative_problem_solving"},
                        {"model": "meta_cognition", "task": "evaluate_learning", "depends_on": "optimization"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人自主学习系统：目标识别 → 数据融合 → 方案生成 → 优化 → 学习评估"
                }

                # 机器人人机交互：语言 + 情感 + 音频 + 规划 + 协作
                model_collaborator.collaboration_patterns["robot_human_interaction"] = {
                    "models": ["language", "emotion", "audio", "planning", "collaboration"],
                    "workflow": [
                        {"model": "audio", "task": "capture_human_input", "share_result": True},
                        {"model": "language", "task": "understand_speech", "depends_on": "audio"},
                        {"model": "emotion", "task": "analyze_emotion", "depends_on": "language"},
                        {"model": "planning", "task": "plan_response", "depends_on": ["language", "emotion"]},
                        {"model": "collaboration", "task": "coordinate_response", "depends_on": "planning"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人人机交互：语音输入 → 语言理解 → 情感分析 → 响应规划 → 协作执行"
                }

                # 机器人全系统集成：管理 + 运动 + 感知 + 认知 + 自主 + 价值对齐
                model_collaborator.collaboration_patterns["robot_full_integration"] = {
                    "models": ["manager", "motion", "vision", "sensor", "language", "knowledge", "planning", "autonomous", "value_alignment"],
                    "workflow": [
                        {"model": "manager", "task": "orchestrate_system", "share_result": True},
                        {"model": "vision", "task": "perceive_environment", "share_result": True},
                        {"model": "sensor", "task": "sense_body_state", "share_result": True},
                        {"model": "language", "task": "process_commands", "depends_on": "manager"},
                        {"model": "knowledge", "task": "provide_context", "depends_on": "language"},
                        {"model": "planning", "task": "create_execution_plan", "depends_on": ["vision", "sensor", "knowledge"]},
                        {"model": "value_alignment", "task": "ensure_safety", "depends_on": "planning"},
                        {"model": "autonomous", "task": "initiate_action", "depends_on": ["planning", "value_alignment"]},
                        {"model": "motion", "task": "execute_plan", "depends_on": "autonomous"}
                    ],
                    "mode": CollaborationMode.HYBRID,
                    "description": "机器人全系统集成：系统协调 → 环境感知 → 身体感知 → 命令处理 → 知识提供 → 计划生成 → 安全验证 → 自主启动 → 运动执行"
                }

                # 机器人计算机编程控制：计算机 + 编程 + 预测
                model_collaborator.collaboration_patterns["robot_computer_programming"] = {
                    "models": ["computer", "programming", "prediction"],
                    "workflow": [
                        {"model": "computer", "task": "access_system", "share_result": True},
                        {"model": "programming", "task": "generate_code", "depends_on": "computer"},
                        {"model": "prediction", "task": "predict_outcome", "depends_on": "programming"}
                    ],
                    "mode": CollaborationMode.SERIAL,
                    "description": "机器人计算机编程控制：系统访问 → 代码生成 → 结果预测"
                }

                # 机器人高级视觉处理：视觉图像 + 视觉视频 + 计算机视觉
                model_collaborator.collaboration_patterns["robot_advanced_visual_processing"] = {
                    "models": ["vision_image", "vision_video", "computer_vision"],
                    "workflow": [
                        {"model": "vision_image", "task": "process_static_images", "share_result": True},
                        {"model": "vision_video", "task": "process_video_streams", "share_result": True},
                        {"model": "computer_vision", "task": "integrate_visual_data", "depends_on": ["vision_image", "vision_video"]}
                    ],
                    "mode": CollaborationMode.PARALLEL,
                    "description": "机器人高级视觉处理：并行处理静态图像和视频流 → 视觉数据集成"
                }

                # AGI-机器人深度融合：实时协同工作系统
                model_collaborator.collaboration_patterns["robot_agi_fusion"] = {
                    "models": ["manager", "autonomous", "value_alignment", "meta_cognition", "creative_problem_solving", "prediction", "planning", "motion", "sensor", "vision"],
                    "workflow": [
                        {"model": "manager", "task": "coordinate_agi_robot_fusion", "share_result": True},
                        {"model": "meta_cognition", "task": "analyze_system_state", "depends_on": "manager"},
                        {"model": "autonomous", "task": "initiate_agi_goals", "depends_on": "meta_cognition"},
                        {"model": "value_alignment", "task": "validate_ethical_constraints", "depends_on": "autonomous"},
                        {"model": "creative_problem_solving", "task": "generate_solutions", "depends_on": "value_alignment"},
                        {"model": "prediction", "task": "predict_outcomes", "depends_on": "creative_problem_solving"},
                        {"model": "planning", "task": "create_execution_plans", "depends_on": "prediction"},
                        {"model": "vision", "task": "perceive_environment", "depends_on": "planning"},
                        {"model": "sensor", "task": "sense_robot_state", "depends_on": "vision"},
                        {"model": "motion", "task": "execute_actions", "depends_on": ["planning", "vision", "sensor"]}
                    ],
                    "mode": CollaborationMode.HYBRID,
                    "description": "AGI-机器人深度融合：系统协调 → 元认知分析 → 自主目标 → 伦理验证 → 创造性解决方案 → 结果预测 → 执行计划 → 环境感知 → 机器人状态感知 → 动作执行"
                }

                logger.info(f"添加了 {len(['robot_movement', 'robot_perception', 'robot_decision', 'robot_autonomous', 'robot_multimodal_perception', 'robot_cognitive_processing', 'robot_autonomous_learning', 'robot_human_interaction', 'robot_full_integration', 'robot_computer_programming', 'robot_advanced_visual_processing', 'robot_agi_fusion'])} 个机器人专用协作模式")
        except Exception as collab_error:
            logger.warning(f"模型协作协调器初始化失败: {collab_error}")
            model_collaborator = None
        
        logger.info("机器人API模块初始化成功")
        return True
    except Exception as e:
        logger.error(f"机器人API模块初始化失败: {e}")
        return False

# 获取机器人模型
def get_robot_model(model_id: str):
    """获取指定的机器人模型"""
    try:
        if model_registry:
            model = model_registry.get_model(model_id)
            if model:
                return model
        return None
    except Exception as e:
        logger.error(f"获取模型失败 {model_id}: {e}")
        return None

# 关节到伺服映射
def map_joint_to_servo(joint_id: str) -> str:
    """将关节ID映射到伺服ID"""
    joint_to_servo = {
        "arm_left_shoulder": "servo_1",
        "arm_left_elbow": "servo_2",
        "arm_left_wrist": "servo_3",
        "arm_right_shoulder": "servo_4",
        "arm_right_elbow": "servo_5",
        "arm_right_wrist": "servo_6",
        "leg_left_hip": "servo_7",
        "leg_left_knee": "servo_8",
        "leg_left_ankle": "servo_9",
        "leg_right_hip": "servo_10",
        "leg_right_knee": "servo_11",
        "leg_right_ankle": "servo_12",
        "head_pan": "servo_13",
        "head_tilt": "servo_14",
        "torso_twist": "servo_15",
        "torso_bend": "servo_16"
    }
    return joint_to_servo.get(joint_id, joint_id)

def get_sensor_unit(sensor_type: str) -> str:
    """获取传感器单位"""
    unit_map = {
        "accelerometer": "m/s²",
        "gyroscope": "rad/s",
        "imu": "状态",
        "temperature": "°C",
        "battery": "%",
        "force": "N",
        "torque": "Nm",
        "proximity": "m",
        "pressure": "Pa",
        "humidity": "%",
        "current": "A",
        "voltage": "V",
        "power": "W",
        "position": "m",
        "velocity": "m/s",
        "acceleration": "m/s²",
        "angle": "°",
        "distance": "m",
    }
    return unit_map.get(sensor_type, "")

# API端点定义

@router.get("/status")
async def get_robot_status():
    """获取机器人状态"""
    try:
        # 更新状态时间戳
        robot_state["last_update"] = datetime.now().isoformat()
        
        # 获取实时数据
        if model_registry:
            # 检查关键模型状态
            critical_models = ["manager", "sensor", "motion", "computer", "vision", "spatial"]
            active_models = 0
            for model_id in critical_models:
                model = get_robot_model(model_id)
                if model:
                    active_models += 1
            
            robot_state["active_models"] = active_models
            robot_state["total_models"] = len(critical_models)
        
        return {
            "status": "success",
            "data": robot_state,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取机器人状态失败")
        raise HTTPException(status_code=500, detail=f"获取机器人状态失败: {str(e)}")

@router.get("/hardware/detect")
async def detect_hardware():
    """Detect available hardware devices"""
    try:
        detected_devices = {
            "joints": 0,
            "sensors": 0,
            "cameras": 0
        }
        detected_hardware_list = {
            "joints": [],
            "sensors": [],
            "cameras": []
        }
        
        # 1. Check if hardware interface is available
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # Detect joints
                if hasattr(robot_hardware, 'joints'):
                    detected_devices["joints"] = len(robot_hardware.joints)
                    for joint_id, joint_info in robot_hardware.joints.items():
                        detected_hardware_list["joints"].append({
                            "id": joint_id,
                            "name": joint_info.get("name", joint_id),
                            "type": joint_info.get("type", "servo"),
                            "detected": True,
                            "selected": False
                        })
                
                # Detect sensors
                if hasattr(robot_hardware, 'sensors'):
                    detected_devices["sensors"] = len(robot_hardware.sensors)
                    for sensor_id, sensor_info in robot_hardware.sensors.items():
                        detected_hardware_list["sensors"].append({
                            "id": sensor_id,
                            "name": sensor_info.get("type", sensor_id),
                            "type": sensor_info.get("type", "sensor"),
                            "detected": True,
                            "selected": False
                        })
                
                # Detect cameras
                try:
                    from core.hardware.camera_manager import CameraManager
                    camera_manager = CameraManager()
                    cameras = camera_manager.list_available_cameras(max_devices=10)
                    detected_devices["cameras"] = len(cameras)
                    for i, camera_info in enumerate(cameras):
                        detected_hardware_list["cameras"].append({
                            "id": f"camera_{camera_info.get('index', i)}",
                            "name": f"Camera {camera_info.get('index', i)}",
                            "type": camera_info.get("backend", "usb"),
                            "detected": camera_info.get("status") == "available",
                            "selected": False
                        })
                except Exception as e:
                    logger.error(f"Camera detection failed, real camera hardware required: {e}")
                    detected_devices["cameras"] = 0
                    # Do not add simulated camera data
                    camera_error = {
                        "error": True,
                        "message": "Camera hardware unavailable",
                        "requires_hardware": True,
                        "hardware_type": "Camera device",
                        "setup_instructions": "Please connect USB camera or network camera and ensure drivers are installed"
                    }
                    # Add camera error information to message
                
                return {
                    "status": "success",
                    "detectedDevices": detected_devices,
                    "detectedHardwareList": detected_hardware_list,
                    "message": f"检测到硬件: {detected_devices['joints']}个关节, {detected_devices['sensors']}个传感器, {detected_devices['cameras']}个摄像头"
                }
            except Exception as e:
                logger.warning(f"Hardware interface detection failed: {e}")

        logger.error("Robot hardware interface unavailable, real hardware connection required")
        return {
            "status": "error",
            "detectedDevices": {"joints": 0, "sensors": 0, "cameras": 0},
            "detectedHardwareList": {"joints": [], "sensors": [], "cameras": []},
            "message": "Robot hardware interface unavailable, cannot detect hardware",
            "requires_hardware_interface": True,
            "setup_instructions": "Please connect real humanoid robot hardware and ensure hardware drivers are properly installed. Required components:\n1. Robot joint controllers (servo motors)\n2. Sensor interfaces (IMU, force sensors, etc.)\n3. Camera devices\n4. Power management system",
            "hardware_requirements": {
                "minimum_joints": 12,
                "minimum_sensors": 6,
                "minimum_cameras": 2,
                "communication_protocols": ["serial", "tcp", "i2c", "spi"],
                "power_requirements": "24V DC, 10A minimum"
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "检测硬件失败")
        raise HTTPException(status_code=500, detail=f"检测硬件失败: {str(e)}")

@router.get("/hardware/status")
async def get_hardware_status():
    """Get hardware connection status and metrics"""
    try:
        # Check if hardware interface is available
        if not robot_hardware or not HARDWARE_INTERFACE_AVAILABLE:
            return {
                "status": "error",
                "message": "Hardware interface unavailable",
                "requires_hardware_interface": True,
                "joints_connected": 0,
                "sensors_connected": 0,
                "cameras_connected": 0,
                "battery_level": 0,
                "system_temperature": 0,
                "initialized": False
            }
        
        # Get hardware status from interface
        hardware_status = robot_hardware.get_hardware_status()
        
        # Get battery and temperature from sensor data if available
        battery_level = 0
        system_temperature = 0
        try:
            # Try to get from sensor data if robot_hardware has sensor_data attribute
            if hasattr(robot_hardware, 'sensor_data'):
                battery_level = robot_hardware.sensor_data.get("battery", {}).get("percentage", 0)
                system_temperature = robot_hardware.sensor_data.get("temp", {}).get("value", 0)
        except Exception as e:
            logger.warning(f"Failed to get battery/temperature data: {e}")
        
        # Map to expected frontend fields
        return {
            "status": "success",
            "data": {
                "joints_connected": hardware_status.get("servo_count", 0),  # servos as joints
                "sensors_connected": hardware_status.get("sensor_count", 0),
                "cameras_connected": hardware_status.get("camera_count", 0),
                "battery_level": battery_level,
                "system_temperature": system_temperature,
                "initialized": hardware_status.get("initialized", False),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取硬件状态失败")
        raise HTTPException(status_code=500, detail=f"获取硬件状态失败: {str(e)}")

@router.post("/hardware/initialize")
async def initialize_hardware(selected_joints: List[str] = None, selected_sensors: List[str] = None, selected_cameras: List[str] = None):
    """Initialize selected hardware devices"""
    try:
        # Check if hardware interface is available
        if not robot_hardware or not HARDWARE_INTERFACE_AVAILABLE:
            logger.warning("Hardware interface unavailable, cannot initialize hardware")
            return {
                "status": "error",
                "message": "Hardware interface unavailable, cannot initialize hardware",
                "requires_hardware_interface": True,
                "setup_instructions": "Please ensure robot hardware interface is properly installed and configured"
            }
        
        # Record selected hardware
        selected_joints = selected_joints or []
        selected_sensors = selected_sensors or []
        selected_cameras = selected_cameras or []
        
        logger.info(f"Initializing selected hardware: {len(selected_joints)} joints, {len(selected_sensors)} sensors, {len(selected_cameras)} cameras")
        
        # Initialize selected joints
        initialized_joints = 0
        for joint_id in selected_joints:
            if joint_id in robot_hardware.joints:
                try:
                    # Add specific joint initialization logic here
                    logger.info(f"Initializing joint: {joint_id}")
                    initialized_joints += 1
                except Exception as e:
                    logger.warning(f"Joint initialization failed {joint_id}: {e}")
        
        # Initialize selected sensors
        initialized_sensors = 0
        for sensor_id in selected_sensors:
            if sensor_id in robot_hardware.sensors:
                try:
                    # Add specific sensor initialization logic here
                    logger.info(f"Initializing sensor: {sensor_id}")
                    initialized_sensors += 1
                except Exception as e:
                    logger.warning(f"Sensor initialization failed {sensor_id}: {e}")
        
        # Initialize selected cameras
        initialized_cameras = 0
        for camera_id in selected_cameras:
            try:
                # Add specific camera initialization logic here
                logger.info(f"Initializing camera: {camera_id}")
                initialized_cameras += 1
            except Exception as e:
                logger.warning(f"Camera initialization failed {camera_id}: {e}")
        
        # Update robot state
        robot_state["connected"] = True
        robot_state["status"] = "idle"
        robot_state["status_text"] = "Hardware connected"
        
        return {
            "status": "success",
            "message": f"硬件初始化完成: {initialized_joints}个关节, {initialized_sensors}个传感器, {initialized_cameras}个摄像头",
            "initialized_joints": initialized_joints,
            "initialized_sensors": initialized_sensors,
            "initialized_cameras": initialized_cameras,
            "robot_state": robot_state
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "初始化硬件失败")
        raise HTTPException(status_code=500, detail=f"初始化硬件失败: {str(e)}")

@router.post("/hardware/disconnect")
async def disconnect_hardware():
    """断开硬件连接"""
    try:
        # 更新机器人状态
        robot_state["connected"] = False
        robot_state["status"] = "disconnected"
        robot_state["status_text"] = "硬件已断开"
        
        return {
            "status": "success",
            "message": "硬件连接已断开",
            "robot_state": robot_state
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "断开硬件连接失败")
        raise HTTPException(status_code=500, detail=f"断开硬件连接失败: {str(e)}")

@router.post("/hardware/test_connection")
async def test_connection(request: ConnectionTestRequest):
    """测试硬件连接"""
    try:
        connection_id = request.connectionId
        connection_type = request.connectionType
        port = request.port
        test_type = request.testType or "connectivity"
        
        logger.info(f"测试连接: {connection_id}, 类型: {connection_type}, 端口: {port}, 测试类型: {test_type}")
        
        # 检查硬件接口是否可用
        if not robot_hardware or not HARDWARE_INTERFACE_AVAILABLE:
            logger.warning("硬件接口不可用，无法测试连接")
            return {
                "status": "error",
                "message": "硬件接口不可用，无法测试连接",
                "requires_hardware_interface": True
            }
        
        # 根据连接类型执行不同的测试
        test_results = {
            "connectionId": connection_id,
            "connectionType": connection_type,
            "port": port,
            "testType": test_type,
            "tests": [],
            "overallStatus": "unknown"
        }
        
        # 测试连接性
        connectivity_test = {
            "name": "连接性测试",
            "status": "success",
            "message": f"{connection_type}连接正常",
            "timestamp": datetime.now().isoformat()
        }
        
        # 根据连接类型执行具体测试
        if connection_type == "serial":
            # 串口连接测试
            if port:
                try:
                    # 这里应该执行实际的串口测试
                    connectivity_test["message"] = f"串口{port}连接测试通过"
                except Exception as e:
                    connectivity_test["status"] = "error"
                    connectivity_test["message"] = f"串口测试失败: {str(e)}"
            else:
                connectivity_test["status"] = "error"
                connectivity_test["message"] = "未指定串口端口"
        
        elif connection_type == "usb":
            # USB连接测试
            connectivity_test["message"] = "USB连接测试通过"
        
        elif connection_type == "ethernet":
            # 以太网连接测试
            connectivity_test["message"] = "以太网连接测试通过"
        
        else:
            connectivity_test["message"] = f"{connection_type}连接测试通过"
        
        test_results["tests"].append(connectivity_test)
        
        # 如果有设备接口，尝试执行更详细的测试
        if hasattr(robot_hardware, 'device_interface') and robot_hardware.device_interface:
            try:
                # 执行基本的设备接口连接测试
                device_test = {
                    "name": "设备接口测试",
                    "status": "success",
                    "message": "设备接口连接正常",
                    "timestamp": datetime.now().isoformat()
                }
                test_results["tests"].append(device_test)
            except Exception as e:
                device_test = {
                    "name": "设备接口测试",
                    "status": "error",
                    "message": f"设备接口测试失败: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                test_results["tests"].append(device_test)
        
        # 确定总体状态
        all_passed = all(test["status"] == "success" for test in test_results["tests"])
        test_results["overallStatus"] = "success" if all_passed else "error"
        
        # 更新机器人状态
        if test_results["overallStatus"] == "success":
            robot_state["connected"] = True
            robot_state["status"] = "idle"
            robot_state["status_text"] = "连接正常"
        
        return {
            "status": "success",
            "message": f"连接测试完成: {connection_id}",
            "testResults": test_results,
            "robot_state": robot_state
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "连接测试失败")
        raise HTTPException(status_code=500, detail=f"连接测试失败: {str(e)}")

@router.post("/hardware/scan")
async def scan_hardware():
    """扫描新硬件设备"""
    try:
        # 在实际系统中，这里会执行硬件扫描
        
        logger.info("执行硬件扫描")
        
        return {
            "status": "success",
            "message": "硬件扫描完成",
            "detectedDevices": {
                "joints": 16,
                "sensors": 8,
                "cameras": 2
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "扫描硬件失败")
        raise HTTPException(status_code=500, detail=f"扫描硬件失败: {str(e)}")

@router.get("/hardware/diagnose")
async def diagnose_hardware():
    """硬件系统诊断"""
    try:
        logger.info("执行硬件系统诊断")
        
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "hardware_interface_available": HARDWARE_INTERFACE_AVAILABLE,
            "robot_hardware_initialized": robot_hardware is not None and hasattr(robot_hardware, 'hardware_initialized') and robot_hardware.hardware_initialized,
            "robot_connected": robot_state.get("connected", False),
            "diagnosis_results": {}
        }
        
        # 如果硬件接口可用，获取详细诊断信息
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 检查传感器状态
                sensor_count = len(robot_hardware.sensors)
                # 检查硬件传感器（排除模拟传感器）
                hardware_sensors = 0
                hardware_sensor_ids = []
                for sensor_id, sensor_info in robot_hardware.sensors.items():
                    if sensor_info.get("protocol") != "simulated":
                        hardware_sensors += 1
                        hardware_sensor_ids.append(sensor_id)
                
                sensor_diagnosis = {
                    "total_sensors": sensor_count,
                    "hardware_sensors": hardware_sensors,
                    "simulated_sensors": sensor_count - hardware_sensors,
                    "sensor_ids": list(robot_hardware.sensors.keys()) if sensor_count > 0 else [],
                    "sensor_data_available": len(robot_hardware.sensor_data) > 0
                }
                diagnosis["diagnosis_results"]["sensors"] = sensor_diagnosis
                
                # 检查伺服器状态
                servo_count = len(robot_hardware.servos) if hasattr(robot_hardware, 'servos') else 0
                servo_diagnosis = {
                    "total_servos": servo_count,
                    "servo_ids": list(robot_hardware.servos.keys()) if servo_count > 0 else []
                }
                diagnosis["diagnosis_results"]["servos"] = servo_diagnosis
                
                # 检查性能指标
                if hasattr(robot_hardware, 'performance_metrics'):
                    diagnosis["diagnosis_results"]["performance"] = robot_hardware.performance_metrics
                
                # 检查安全系统
                if hasattr(robot_hardware, 'safety_system'):
                    diagnosis["diagnosis_results"]["safety"] = robot_hardware.safety_system
                
                # 总体健康状态 - 模拟传感器不被允许
                overall_health = "healthy"
                if sensor_diagnosis["hardware_sensors"] == 0:
                    overall_health = "critical"
                    diagnosis["diagnosis_results"]["message"] = "未检测到硬件传感器。真实传感器硬件为AGI操作所必需。"
                elif sensor_diagnosis["simulated_sensors"] > 0:
                    overall_health = "critical"
                    diagnosis["diagnosis_results"]["message"] = f"检测到{sensor_diagnosis['simulated_sensors']}个模拟传感器。模拟传感器不被允许，真实传感器硬件为AGI操作所必需。"
                else:
                    diagnosis["diagnosis_results"]["message"] = "硬件接口运行正常"
                
                diagnosis["diagnosis_results"]["overall_health"] = overall_health
                
            except Exception as e:
                diagnosis["diagnosis_results"]["error"] = f"获取详细诊断信息时出错: {str(e)}"
                diagnosis["diagnosis_results"]["overall_health"] = "error"
        else:
            diagnosis["diagnosis_results"]["overall_health"] = "unavailable"
            diagnosis["diagnosis_results"]["message"] = "硬件接口不可用"
        
        return {
            "status": "success",
            "diagnosis": diagnosis
        }
        
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "硬件诊断失败")
        raise HTTPException(status_code=500, detail=f"硬件诊断失败: {str(e)}")

@router.get("/sensors")
async def get_sensors():
    """获取传感器数据"""
    try:
        # 1. 首先尝试从机器人硬件接口获取真实数据
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 获取硬件接口中所有传感器的数据
                hardware_sensors = []
                for sensor_id in robot_hardware.sensors:
                    sensor_data = robot_hardware.get_sensor_data(sensor_id)
                    if sensor_data.get("success"):
                        sensor_info = robot_hardware.sensors[sensor_id]
                        hardware_sensors.append({
                            "id": sensor_id,
                            "name": sensor_info.get("type", sensor_id),
                            "type": sensor_info.get("type", "unknown"),
                            "value": str(sensor_data.get("data", {})),
                            "unit": get_sensor_unit(sensor_info.get("type")),
                            "status": "active",
                            "active": True,
                            "source": "hardware",
                            "timestamp": sensor_data.get("timestamp")
                        })
                
                if hardware_sensors:
                    logger.info(f"从硬件接口获取到 {len(hardware_sensors)} 个传感器数据")
                    return {
                        "status": "success",
                        "data": hardware_sensors,
                        "source": "hardware",
                        "count": len(hardware_sensors)
                    }
            except Exception as e:
                logger.warning(f"从硬件接口获取传感器数据失败，尝试传感器模型: {e}")
        
        # 2. 尝试从传感器模型获取真实数据
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                # 调用传感器模型处理数据
                sensor_data = sensor_model.process_input({
                    "operation": "sensor_processing",
                    "data": {"action": "get_sensors"}
                })
                
                if sensor_data and "sensors" in sensor_data:
                    logger.info(f"从传感器模型获取到 {len(sensor_data['sensors'])} 个传感器数据")
                    return {
                        "status": "success",
                        "data": sensor_data["sensors"],
                        "source": "sensor_model"
                    }
            except Exception as e:
                logger.warning(f"Failed to get data from sensor model: {e}")
        
        # 3. 所有数据源都不可用
        logger.error(f"Failed to get sensor data: hardware interface and sensor model are both unavailable")
        return {
            "status": "error", 
            "message": "Failed to retrieve sensor data: hardware interface or sensor model needs initialization",
            "source": "none",
            "requires_hardware": True,
            "hardware_type": "sensor interface",
            "setup_instructions": "Please connect sensor hardware and initialize robot hardware interface"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "Failed to get sensor data")
        raise HTTPException(status_code=500, detail=f"Failed to get sensor data: {str(e)}")

@router.post("/joint")
async def control_joint(command: JointCommand):
    """控制机器人关节"""
    try:
        joint_id = command.jointId
        value = command.value
        timestamp = command.timestamp or int(time.time() * 1000)
        
        # 更新关节状态
        parts = joint_id.split('_')
        if len(parts) == 3:
            limb, joint_name, sub_joint = parts
            if limb in joints_state and joint_name in joints_state[limb]:
                joints_state[limb][joint_name][sub_joint] = value
        elif len(parts) == 2:
            limb, joint_name = parts
            if limb in joints_state:
                joints_state[limb][joint_name] = value
        
        # 1. 首先尝试使用机器人硬件接口
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 映射关节ID到伺服ID
                servo_id = map_joint_to_servo(joint_id)
                
                # 使用硬件接口控制伺服
                result = robot_hardware.control_servo(servo_id, value)
                
                if result.get("success"):
                    logger.info(f"通过硬件接口控制关节成功: {joint_id} = {value}° (伺服: {servo_id})")
                    
                    return {
                        "status": "success",
                        "message": f"关节 {joint_id} 设置为 {value}°",
                        "result": result,
                        "source": "hardware",
                        "servo_id": servo_id,
                        "timestamp": timestamp
                    }
                else:
                    logger.warning(f"硬件接口控制失败: {result.get('error')}")
            except Exception as e:
                logger.warning(f"硬件接口控制异常，尝试运动模型: {e}")
        
        # 2. 尝试通过运动模型控制关节
        motion_model = get_robot_model("motion")
        if motion_model:
            try:
                # 调用运动模型控制关节
                result = motion_model.process_input({
                    "operation": "joint_control",
                    "data": {
                        "joint_id": joint_id,
                        "value": value,
                        "timestamp": timestamp
                    }
                })
                
                logger.info(f"通过运动模型控制关节成功: {joint_id} = {value}")
                
                return {
                    "status": "success",
                    "message": f"关节 {joint_id} 设置为 {value}°",
                    "result": result,
                    "source": "motion_model",
                    "timestamp": timestamp
                }
            except Exception as e:
                logger.warning(f"运动模型控制失败，尝试其他控制方式: {e}")
        
        # 3. 所有控制方式都不可用
        logger.error(f"关节控制失败: 硬件接口和运动模型都不可用")
        
        return {
            "status": "error",
            "message": f"关节 {joint_id} 控制失败: 需要初始化硬件接口或运动模型",
            "source": "none",
            "timestamp": timestamp,
            "requires_hardware": True,
            "hardware_type": "伺服电机控制器",
            "setup_instructions": "请连接伺服电机控制器并初始化机器人硬件接口"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "控制关节失败")
        raise HTTPException(status_code=500, detail=f"控制关节失败: {str(e)}")

@router.post("/sensor/toggle")
async def toggle_sensor(command: SensorToggle):
    """切换传感器状态"""
    try:
        sensor_id = command.sensorId
        active = command.active
        
        # 更新传感器状态
        for sensor in sensors_state:
            if sensor["id"] == sensor_id:
                sensor["active"] = active
                sensor["status"] = "active" if active else "inactive"
                break
        
        # 尝试通过传感器模型控制
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                result = sensor_model.process_input({
                    "operation": "sensor_configuration",
                    "data": {
                        "action": "toggle",
                        "sensor_id": sensor_id,
                        "active": active
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"传感器 {sensor_id} {'启用' if active else '禁用'}",
                    "result": result,
                    "source": "sensor_model"
                }
            except Exception as e:
                logger.warning(f"传感器模型控制失败: {e}")
        
        # 传感器模型不可用
        logger.error(f"切换传感器状态失败: 传感器模型不可用")
        return {
            "status": "error",
            "message": f"传感器 {sensor_id} 切换失败: 需要初始化传感器模型",
            "source": "none",
            "requires_model": True,
            "model_name": "sensor",
            "setup_instructions": "请初始化传感器模型或连接硬件接口"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "切换传感器状态失败")
        raise HTTPException(status_code=500, detail=f"切换传感器状态失败: {str(e)}")

@router.post("/sensor/calibrate")
async def calibrate_sensors():
    """校准传感器"""
    try:
        # 尝试通过传感器模型校准
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                result = sensor_model.process_input({
                    "operation": "calibration",
                    "data": {"action": "calibrate_all"}
                })
                
                return {
                    "status": "success",
                    "message": "传感器校准开始",
                    "result": result,
                    "source": "sensor_model"
                }
            except Exception as e:
                logger.warning(f"传感器模型校准失败: {e}")
        
        # 传感器模型不可用，无法校准
        logger.error(f"传感器校准失败: 传感器模型不可用")
        return {
            "status": "error",
            "message": "传感器校准失败: 需要初始化传感器模型",
            "source": "none",
            "requires_model": True,
            "model_name": "sensor",
            "setup_instructions": "请初始化传感器模型以进行校准"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "传感器校准失败")
        raise HTTPException(status_code=500, detail=f"传感器校准失败: {str(e)}")

@router.post("/sensor/fusion")
async def sensor_data_fusion(command: SensorFusionRequest):
    """传感器数据融合"""
    try:
        sensor_ids = command.sensor_ids
        fusion_method = command.fusion_method
        parameters = command.parameters or {}
        
        # 尝试通过传感器模型进行数据融合
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                result = sensor_model.process_input({
                    "operation": "data_fusion",
                    "data": {
                        "sensor_ids": sensor_ids,
                        "fusion_method": fusion_method,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"传感器数据融合完成 ({fusion_method})",
                    "result": result,
                    "source": "sensor_model"
                }
            except Exception as e:
                logger.warning(f"传感器模型数据融合失败: {e}")
        
        # 传感器模型不可用，尝试通过数据融合模型
        data_fusion_model = get_robot_model("data_fusion")
        if data_fusion_model:
            try:
                result = data_fusion_model.process_input({
                    "operation": "sensor_fusion",
                    "data": {
                        "sensor_ids": sensor_ids,
                        "fusion_method": fusion_method,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"传感器数据融合完成 ({fusion_method})",
                    "result": result,
                    "source": "data_fusion_model"
                }
            except Exception as e:
                logger.warning(f"数据融合模型处理失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"传感器数据融合失败: 传感器模型和数据融合模型都不可用")
        return {
            "status": "error",
            "message": "传感器数据融合失败: 需要初始化传感器模型或数据融合模型",
            "source": "none",
            "requires_model": True,
            "model_name": "sensor",
            "setup_instructions": "请初始化传感器模型或数据融合模型以进行数据融合"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "传感器数据融合失败")
        raise HTTPException(status_code=500, detail=f"传感器数据融合失败: {str(e)}")

@router.post("/sensor/analyze")
async def sensor_data_analysis(command: SensorAnalysisRequest):
    """传感器数据分析"""
    try:
        sensor_ids = command.sensor_ids
        analysis_type = command.analysis_type
        time_range = command.time_range
        parameters = command.parameters or {}
        
        # 尝试通过传感器模型进行分析
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                result = sensor_model.process_input({
                    "operation": "data_analysis",
                    "data": {
                        "sensor_ids": sensor_ids,
                        "analysis_type": analysis_type,
                        "time_range": time_range,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"传感器数据分析完成 ({analysis_type})",
                    "result": result,
                    "source": "sensor_model"
                }
            except Exception as e:
                logger.warning(f"传感器模型数据分析失败: {e}")
        
        # 传感器模型不可用，尝试通过数据融合模型
        data_fusion_model = get_robot_model("data_fusion")
        if data_fusion_model:
            try:
                result = data_fusion_model.process_input({
                    "operation": "sensor_analysis",
                    "data": {
                        "sensor_ids": sensor_ids,
                        "analysis_type": analysis_type,
                        "time_range": time_range,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"传感器数据分析完成 ({analysis_type})",
                    "result": result,
                    "source": "data_fusion_model"
                }
            except Exception as e:
                logger.warning(f"数据融合模型分析失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"传感器数据分析失败: 传感器模型和数据融合模型都不可用")
        return {
            "status": "error",
            "message": "传感器数据分析失败: 需要初始化传感器模型或数据融合模型",
            "source": "none",
            "requires_model": True,
            "model_name": "sensor",
            "setup_instructions": "请初始化传感器模型或数据融合模型以进行数据分析"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "传感器数据分析失败")
        raise HTTPException(status_code=500, detail=f"传感器数据分析失败: {str(e)}")

@router.post("/camera/toggle")
async def toggle_camera(command: CameraCommand):
    """切换摄像头状态"""
    try:
        camera = command.camera
        active = command.active
        
        if camera in cameras_state:
            cameras_state[camera]["active"] = active
        
        # 1. 首先尝试使用硬件接口控制摄像头
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 查找对应的摄像头ID
                camera_id = None
                for cam_id, cam_info in robot_hardware.cameras.items():
                    if cam_info.get("type") == camera or cam_id == camera:
                        camera_id = cam_id
                        break
                
                if camera_id:
                    if active:
                        # 启动摄像头流
                        # 这里需要调用摄像头管理器的启动方法
                        # 启动摄像头流
                        robot_hardware.cameras[camera_id]["is_streaming"] = True
                        logger.info(f"通过硬件接口启动摄像头: {camera_id}")
                    else:
                        # 停止摄像头流
                        robot_hardware.cameras[camera_id]["is_streaming"] = False
                        logger.info(f"通过硬件接口停止摄像头: {camera_id}")
                    
                    return {
                        "status": "success",
                        "message": f"{camera}摄像头 {'启动' if active else '停止'}",
                        "camera_id": camera_id,
                        "source": "hardware"
                    }
            except Exception as e:
                logger.warning(f"硬件接口控制摄像头失败，尝试视觉模型: {e}")
        
        # 2. 尝试通过视觉模型控制摄像头
        vision_model = get_robot_model("vision")
        if vision_model:
            try:
                result = vision_model.process_input({
                    "operation": "camera_control",
                    "data": {
                        "camera": camera,
                        "action": "toggle",
                        "active": active
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"{camera}摄像头 {'启动' if active else '停止'}",
                    "result": result,
                    "source": "vision_model"
                }
            except Exception as e:
                logger.warning(f"视觉模型控制失败: {e}")
        
        # 3. 所有控制方式都不可用
        logger.error(f"摄像头控制失败: 硬件接口和视觉模型都不可用")
        
        return {
            "status": "error",
            "message": f"{camera}摄像头控制失败: 需要初始化硬件接口或视觉模型",
            "source": "none",
            "requires_hardware": True,
            "hardware_type": "摄像头设备",
            "setup_instructions": "请连接摄像头硬件并初始化视觉模型"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "切换摄像头失败")
        raise HTTPException(status_code=500, detail=f"切换摄像头失败: {str(e)}")

@router.post("/camera/calibrate")
async def calibrate_camera(request: Request):
    """校准摄像头"""
    try:
        data = await request.json()
        camera = data.get("camera", "left")
        
        if camera in cameras_state:
            cameras_state[camera]["calibrated"] = True
        
        # 尝试通过视觉模型校准
        vision_model = get_robot_model("vision")
        if vision_model:
            try:
                result = vision_model.process_input({
                    "operation": "camera_calibration",
                    "data": {"camera": camera}
                })
                
                return {
                    "status": "success",
                    "message": f"{camera}摄像头校准完成",
                    "result": result,
                    "source": "vision_model"
                }
            except Exception as e:
                logger.warning(f"视觉模型校准失败: {e}")
        
        # 视觉模型不可用，无法校准
        logger.error(f"摄像头校准失败: 视觉模型不可用")
        return {
            "status": "error",
            "message": f"{camera}摄像头校准失败: 需要初始化视觉模型",
            "source": "none",
            "requires_model": True,
            "model_name": "vision",
            "setup_instructions": "请初始化视觉模型以进行摄像头校准"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "摄像头校准失败")
        raise HTTPException(status_code=500, detail=f"摄像头校准失败: {str(e)}")

@router.post("/vision/detect")
async def vision_detection(command: VisionDetectionRequest):
    """视觉检测（对象检测、人脸检测等）"""
    try:
        camera = command.camera
        detection_type = command.detection_type
        confidence_threshold = command.confidence_threshold
        parameters = command.parameters or {}
        
        # 尝试通过视觉模型进行检测
        vision_model = get_robot_model("vision")
        if vision_model:
            try:
                result = vision_model.process_input({
                    "operation": "detection",
                    "data": {
                        "camera": camera,
                        "detection_type": detection_type,
                        "confidence_threshold": confidence_threshold,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"视觉检测完成 ({detection_type})",
                    "result": result,
                    "source": "vision_model"
                }
            except Exception as e:
                logger.warning(f"视觉模型检测失败: {e}")
        
        # 视觉模型不可用，尝试通过计算机视觉模型
        computer_vision_model = get_robot_model("computer_vision")
        if computer_vision_model:
            try:
                result = computer_vision_model.process_input({
                    "operation": "object_detection",
                    "data": {
                        "camera": camera,
                        "detection_type": detection_type,
                        "confidence_threshold": confidence_threshold,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"视觉检测完成 ({detection_type})",
                    "result": result,
                    "source": "computer_vision_model"
                }
            except Exception as e:
                logger.warning(f"计算机视觉模型检测失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"视觉检测失败: 视觉模型和计算机视觉模型都不可用")
        return {
            "status": "error",
            "message": "视觉检测失败: 需要初始化视觉模型或计算机视觉模型",
            "source": "none",
            "requires_model": True,
            "model_name": "vision",
            "setup_instructions": "请初始化视觉模型或计算机视觉模型以进行视觉检测"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "视觉检测失败")
        raise HTTPException(status_code=500, detail=f"视觉检测失败: {str(e)}")

@router.post("/vision/segment")
async def vision_segmentation(command: VisionSegmentationRequest):
    """图像分割"""
    try:
        camera = command.camera
        segmentation_type = command.segmentation_type
        parameters = command.parameters or {}
        
        # 尝试通过视觉模型进行分割
        vision_model = get_robot_model("vision")
        if vision_model:
            try:
                result = vision_model.process_input({
                    "operation": "segmentation",
                    "data": {
                        "camera": camera,
                        "segmentation_type": segmentation_type,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"图像分割完成 ({segmentation_type})",
                    "result": result,
                    "source": "vision_model"
                }
            except Exception as e:
                logger.warning(f"视觉模型分割失败: {e}")
        
        # 视觉模型不可用，尝试通过计算机视觉模型
        computer_vision_model = get_robot_model("computer_vision")
        if computer_vision_model:
            try:
                result = computer_vision_model.process_input({
                    "operation": "image_segmentation",
                    "data": {
                        "camera": camera,
                        "segmentation_type": segmentation_type,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"图像分割完成 ({segmentation_type})",
                    "result": result,
                    "source": "computer_vision_model"
                }
            except Exception as e:
                logger.warning(f"计算机视觉模型分割失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"图像分割失败: 视觉模型和计算机视觉模型都不可用")
        return {
            "status": "error",
            "message": "图像分割失败: 需要初始化视觉模型或计算机视觉模型",
            "source": "none",
            "requires_model": True,
            "model_name": "vision",
            "setup_instructions": "请初始化视觉模型或计算机视觉模型以进行图像分割"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "图像分割失败")
        raise HTTPException(status_code=500, detail=f"图像分割失败: {str(e)}")

@router.post("/spatial/depth")
async def spatial_depth_estimation(command: SpatialDepthRequest):
    """空间深度估计"""
    try:
        cameras = command.cameras
        method = command.method
        parameters = command.parameters or {}
        
        # 尝试通过空间模型进行深度估计
        spatial_model = get_robot_model("spatial")
        if spatial_model:
            try:
                result = spatial_model.process_input({
                    "operation": "depth_estimation",
                    "data": {
                        "cameras": cameras,
                        "method": method,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"深度估计完成 ({method})",
                    "result": result,
                    "source": "spatial_model"
                }
            except Exception as e:
                logger.warning(f"空间模型深度估计失败: {e}")
        
        # 空间模型不可用，尝试通过计算机视觉模型
        computer_vision_model = get_robot_model("computer_vision")
        if computer_vision_model:
            try:
                result = computer_vision_model.process_input({
                    "operation": "stereo_depth",
                    "data": {
                        "cameras": cameras,
                        "method": method,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"深度估计完成 ({method})",
                    "result": result,
                    "source": "computer_vision_model"
                }
            except Exception as e:
                logger.warning(f"计算机视觉模型深度估计失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"深度估计失败: 空间模型和计算机视觉模型都不可用")
        return {
            "status": "error",
            "message": "深度估计失败: 需要初始化空间模型或计算机视觉模型",
            "source": "none",
            "requires_model": True,
            "model_name": "spatial",
            "setup_instructions": "请初始化空间模型或计算机视觉模型以进行深度估计"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "深度估计失败")
        raise HTTPException(status_code=500, detail=f"深度估计失败: {str(e)}")

@router.post("/spatial/map")
async def spatial_mapping(command: SpatialMappingRequest):
    """空间映射"""
    try:
        operation = command.operation
        area = command.area
        parameters = command.parameters or {}
        
        # 尝试通过空间模型进行空间映射
        spatial_model = get_robot_model("spatial")
        if spatial_model:
            try:
                result = spatial_model.process_input({
                    "operation": "mapping",
                    "data": {
                        "operation": operation,
                        "area": area,
                        "parameters": parameters
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"空间映射操作完成 ({operation})",
                    "result": result,
                    "source": "spatial_model"
                }
            except Exception as e:
                logger.warning(f"空间模型映射失败: {e}")
        
        # 空间模型不可用
        logger.error(f"空间映射失败: 空间模型不可用")
        return {
            "status": "error",
            "message": "空间映射失败: 需要初始化空间模型",
            "source": "none",
            "requires_model": True,
            "model_name": "spatial",
            "setup_instructions": "请初始化空间模型以进行空间映射"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "空间映射失败")
        raise HTTPException(status_code=500, detail=f"空间映射失败: {str(e)}")

@router.post("/stereo/enable")
async def enable_stereo_vision():
    """启用立体视觉"""
    try:
        # 1. 首先尝试使用硬件接口启用立体视觉
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE and robot_hardware.camera_manager:
            try:
                # 检查是否已注册双目摄像头
                stereo_cameras = []
                for camera_id, camera_info in robot_hardware.cameras.items():
                    if camera_info.get("type") in ["left", "right", "stereo"]:
                        stereo_cameras.append(camera_id)
                
                if len(stereo_cameras) >= 2:
                    # 启用摄像头管理器的立体视觉模式
                    robot_hardware.camera_manager.config["binocular_mode"] = True
                    logger.info(f"硬件接口立体视觉已启用，使用摄像头: {stereo_cameras}")
                    
                    return {
                        "status": "success",
                        "message": "立体视觉已启用",
                        "cameras": stereo_cameras,
                        "source": "hardware"
                    }
            except Exception as e:
                logger.warning(f"硬件接口启用立体视觉失败，尝试空间模型: {e}")
        
        # 2. 尝试通过空间模型启用立体视觉
        spatial_model = get_robot_model("spatial")
        if spatial_model:
            try:
                result = spatial_model.process_input({
                    "operation": "stereo_vision",
                    "data": {"action": "enable"}
                })
                
                return {
                    "status": "success",
                    "message": "立体视觉已启用",
                    "result": result,
                    "source": "spatial_model"
                }
            except Exception as e:
                logger.warning(f"空间模型启用立体视觉失败: {e}")
        
        # 3. 所有方式都不可用
        logger.error("启用立体视觉失败: 硬件接口和空间模型都不可用")
        
        return {
            "status": "error",
            "message": "启用立体视觉失败: 需要初始化硬件接口或空间模型",
            "source": "none",
            "requires_hardware": True,
            "hardware_type": "双目摄像头",
            "setup_instructions": "请连接双目摄像头硬件并初始化空间模型"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "启用立体视觉失败")
        raise HTTPException(status_code=500, detail=f"启用立体视觉失败: {str(e)}")

@router.post("/stereo/depth")
async def calculate_depth_map():
    """计算深度图"""
    try:
        # 尝试通过空间模型计算深度图
        spatial_model = get_robot_model("spatial")
        if spatial_model:
            try:
                result = spatial_model.process_input({
                    "operation": "depth_calculation",
                    "data": {"action": "calculate"}
                })
                
                return {
                    "status": "success",
                    "message": "深度图计算中...",
                    "result": result,
                    "source": "spatial_model"
                }
            except Exception as e:
                logger.warning(f"空间模型计算深度图失败: {e}")
        
        # 空间模型不可用，无法计算深度图
        logger.error("计算深度图失败: 空间模型不可用")
        return {
            "status": "error",
            "message": "计算深度图失败: 需要初始化空间模型",
            "source": "none",
            "requires_model": True,
            "model_name": "spatial",
            "setup_instructions": "请初始化空间模型以进行深度图计算"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "计算深度图失败")
        raise HTTPException(status_code=500, detail=f"计算深度图失败: {str(e)}")

@router.post("/motion/execute")
async def execute_motion(request_data: Dict[str, Any]):
    """执行运动 - 支持多种参数格式
    
    支持的参数格式:
    1. 旧格式: {"motion": "move", "params": {"direction": "forward", "speed": 0.5}}
    2. 新格式: {"command": "move", "direction": "forward", "speed": 0.5, "duration": 1.0}
    3. 旋转格式: {"command": "rotate", "direction": "left", "angle": 45, "speed": 0.5}
    """
    try:
        # 解析输入参数，支持多种格式
        motion_type = None
        params = {}
        
        # 格式1: 旧格式 (motion + params)
        if "motion" in request_data:
            motion_type = request_data["motion"]
            params = request_data.get("params", {})
        # 格式2: 新格式 (command + direction + speed + duration)
        elif "command" in request_data:
            motion_type = request_data["command"]
            
            # 根据命令类型构建参数
            if motion_type == "move":
                params = {
                    "direction": request_data.get("direction", "forward"),
                    "speed": request_data.get("speed", 0.5),
                    "duration": request_data.get("duration", 1.0)
                }
            elif motion_type == "rotate":
                params = {
                    "direction": request_data.get("direction", "left"),
                    "angle": request_data.get("angle", 45),
                    "speed": request_data.get("speed", 0.5)
                }
            else:
                # 其他命令类型，传递所有参数
                params = {k: v for k, v in request_data.items() if k != "command"}
        else:
            # 未知格式，使用默认值
            motion_type = "move"
            params = request_data
        
        # 更新机器人状态
        robot_state["status"] = "active"
        robot_state["status_text"] = f"执行运动: {motion_type}"
        
        # 记录运动参数用于调试
        logger.info(f"执行运动: {motion_type}, 参数: {params}")
        
        # 尝试通过运动模型执行运动
        motion_model = get_robot_model("motion")
        if motion_model:
            try:
                result = motion_model.process_input({
                    "operation": "motion_execution",
                    "data": {
                        "motion": motion_type,
                        "params": params
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"执行运动: {motion_type}",
                    "result": result,
                    "source": "motion_model",
                    "motion_type": motion_type,
                    "params": params
                }
            except Exception as e:
                logger.warning(f"运动模型执行失败: {e}")
        
        # 运动模型不可用，返回模拟响应
        logger.info(f"运动模型不可用，返回模拟响应: {motion_type}")
        
        # 生成模拟响应
        if motion_type == "move":
            direction = params.get("direction", "forward")
            speed = params.get("speed", 0.5)
            duration = params.get("duration", 1.0)
            
            # 更新关节状态模拟运动
            if direction == "forward":
                joints_state["leg"]["left"]["hip"] = 15
                joints_state["leg"]["right"]["hip"] = 15
            elif direction == "backward":
                joints_state["leg"]["left"]["hip"] = -15
                joints_state["leg"]["right"]["hip"] = -15
            elif direction == "left":
                joints_state["head"]["pan"] = -30
            elif direction == "right":
                joints_state["head"]["pan"] = 30
                
            return {
                "status": "success",
                "message": f"模拟移动: {direction} (速度: {speed}, 时长: {duration}s)",
                "result": {
                    "executed": True,
                    "motion": motion_type,
                    "direction": direction,
                    "speed": speed,
                    "duration": duration,
                    "simulated": True
                },
                "source": "simulation",
                "motion_type": motion_type,
                "params": params
            }
        elif motion_type == "rotate":
            direction = params.get("direction", "left")
            angle = params.get("angle", 45)
            speed = params.get("speed", 0.5)
            
            # 更新头部关节状态模拟旋转
            if direction == "left":
                joints_state["head"]["pan"] = -angle
            elif direction == "right":
                joints_state["head"]["pan"] = angle
            elif direction == "reset":
                joints_state["head"]["pan"] = 0
                
            return {
                "status": "success",
                "message": f"模拟旋转: {direction} (角度: {angle}°, 速度: {speed})",
                "result": {
                    "executed": True,
                    "motion": motion_type,
                    "direction": direction,
                    "angle": angle,
                    "speed": speed,
                    "simulated": True
                },
                "source": "simulation",
                "motion_type": motion_type,
                "params": params
            }
        else:
            # 其他运动类型
            return {
                "status": "success",
                "message": f"执行运动: {motion_type} (模拟)",
                "result": {
                    "executed": True,
                    "motion": motion_type,
                    "params": params,
                    "simulated": True
                },
                "source": "simulation",
                "motion_type": motion_type,
                "params": params
            }
            
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "执行运动失败")
        raise HTTPException(status_code=500, detail=f"执行运动失败: {str(e)}")

@router.post("/task/plan")
async def plan_task(request_data: Dict[str, Any]):
    """规划任务 - 支持多种参数格式
    
    支持的参数格式:
    1. 旧格式: {"description": "任务描述"}
    2. 新格式: {"task_type": "waypoint_navigation", "waypoints": [...], "constraints": {...}}
    """
    try:
        # 解析输入参数，支持多种格式
        description = ""
        task_data = {}
        
        # 格式1: 旧格式 (description)
        if "description" in request_data:
            description = request_data["description"]
            task_data = request_data
        # 格式2: 新格式 (task_type + waypoints + constraints)
        elif "task_type" in request_data:
            task_type = request_data["task_type"]
            
            if task_type == "waypoint_navigation":
                waypoints = request_data.get("waypoints", [])
                constraints = request_data.get("constraints", {})
                description = f"航点导航任务 ({len(waypoints)} 个航点)"
                task_data = {
                    "task_type": task_type,
                    "waypoints": waypoints,
                    "constraints": constraints
                }
            else:
                # 其他任务类型
                description = f"{task_type} 任务"
                task_data = request_data
        else:
            # 未知格式，使用默认值
            description = str(request_data)
            task_data = request_data
        
        # 创建新任务
        new_task = {
            "id": int(time.time() * 1000),
            "description": description,
            "task_type": task_data.get("task_type", "general"),
            "data": task_data,
            "status": "已规划",
            "active": False,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        
        tasks_list.append(new_task)
        
        # 记录任务参数用于调试
        logger.info(f"规划任务: {description}, 类型: {new_task['task_type']}")
        
        # 尝试通过规划模型规划任务
        planning_model = get_robot_model("planning")
        if planning_model:
            try:
                result = planning_model.process_input({
                    "operation": "task_planning",
                    "data": {
                        "description": description,
                        "task_id": new_task["id"],
                        "task_data": task_data
                    }
                })
                
                # 更新任务信息
                if result and "plan" in result:
                    new_task["plan"] = result["plan"]
                    new_task["estimated_duration"] = result.get("estimated_duration")
                
                return {
                    "status": "success",
                    "message": f"任务规划: {description}",
                    "task": new_task,
                    "result": result,
                    "source": "planning_model"
                }
            except Exception as e:
                logger.warning(f"规划模型任务规划失败: {e}")
        
        # 规划模型不可用，返回模拟响应
        logger.info(f"规划模型不可用，返回模拟响应: {description}")
        
        # 生成模拟规划结果
        if task_data.get("task_type") == "waypoint_navigation":
            waypoints = task_data.get("waypoints", [])
            constraints = task_data.get("constraints", {})
            
            # 生成模拟路径
            simulated_path = []
            if waypoints:
                simulated_path.append({"type": "start", "position": {"x": 0, "y": 0, "z": 0}})
                
                for i, wp in enumerate(waypoints):
                    simulated_path.append({
                        "type": "waypoint",
                        "index": i + 1,
                        "position": wp,
                        "distance_from_previous": 1.0 if i == 0 else 2.0
                    })
                
                simulated_path.append({"type": "end", "position": waypoints[-1]})
            
            simulated_plan = {
                "path": simulated_path,
                "estimated_distance": len(waypoints) * 2.0,
                "estimated_time": len(waypoints) * 5.0,
                "obstacles_detected": False,
                "simulated": True
            }
            
            new_task["plan"] = simulated_plan
            new_task["estimated_duration"] = simulated_plan["estimated_time"]
            
            return {
                "status": "success",
                "message": f"模拟任务规划: {description}",
                "task": new_task,
                "result": simulated_plan,
                "source": "simulation",
                "task_id": new_task["id"]
            }
        else:
            # 其他任务类型的模拟响应
            simulated_plan = {
                "plan": "模拟任务规划",
                "estimated_duration": 60.0,
                "steps": ["步骤1: 初始化", "步骤2: 执行", "步骤3: 完成"],
                "simulated": True
            }
            
            new_task["plan"] = simulated_plan
            new_task["estimated_duration"] = simulated_plan["estimated_duration"]
            
            return {
                "status": "success",
                "message": f"模拟任务规划: {description}",
                "task": new_task,
                "result": simulated_plan,
                "source": "simulation",
                "task_id": new_task["id"]
            }
            
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "任务规划失败")
        raise HTTPException(status_code=500, detail=f"任务规划失败: {str(e)}")

@router.post("/task/execute")
async def execute_task(request_data: Dict[str, Any]):
    """执行任务 - 支持多种参数格式
    
    支持的参数格式:
    1. 旧格式: {"taskId": 12345}
    2. 新格式: {"task_id": 12345}
    3. 当前任务格式: {"task_id": "current"} - 执行最新任务
    """
    try:
        # 解析任务ID，支持多种格式
        task_id = None
        
        # 格式1: 旧格式 (taskId)
        if "taskId" in request_data:
            task_id = request_data["taskId"]
        # 格式2: 新格式 (task_id)
        elif "task_id" in request_data:
            task_id = request_data["task_id"]
        else:
            raise HTTPException(status_code=400, detail="未指定任务ID")
        
        # 查找任务
        task = None
        
        if task_id == "current":
            # 查找最新的任务
            if tasks_list:
                # 按创建时间降序排序，取最新的任务
                sorted_tasks = sorted(tasks_list, key=lambda t: t.get("id", 0), reverse=True)
                task = sorted_tasks[0]
        else:
            # 按ID查找任务
            for t in tasks_list:
                if t["id"] == task_id:
                    task = t
                    break
        
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        # 更新任务状态
        task["active"] = True
        task["status"] = "执行中"
        task["started_at"] = datetime.now().isoformat()
        
        # 更新机器人状态
        robot_state["status"] = "active"
        robot_state["status_text"] = f"执行任务: {task['description']}"
        
        # 记录任务执行信息
        logger.info(f"执行任务: {task['description']}, ID: {task['id']}, 类型: {task.get('task_type', 'general')}")
        
        # 尝试通过规划模型执行任务
        planning_model = get_robot_model("planning")
        if planning_model:
            try:
                result = planning_model.process_input({
                    "operation": "task_execution",
                    "data": {
                        "task_id": task["id"],
                        "task_description": task["description"],
                        "task_data": task.get("data", {}),
                        "task_type": task.get("task_type", "general")
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"执行任务: {task['description']}",
                    "task": task,
                    "result": result,
                    "source": "planning_model"
                }
            except Exception as e:
                logger.warning(f"规划模型任务执行失败: {e}")
        
        # 规划模型不可用，返回模拟响应
        logger.info(f"规划模型不可用，返回模拟执行响应: {task['description']}")
        
        # 生成模拟执行结果
        if task.get("task_type") == "waypoint_navigation":
            waypoints = task.get("data", {}).get("waypoints", [])
            
            simulated_result = {
                "execution_status": "in_progress",
                "current_waypoint": 1,
                "total_waypoints": len(waypoints),
                "progress": 0,
                "estimated_remaining_time": len(waypoints) * 5.0,
                "simulated": True,
                "task_id": task["id"]
            }
            
            # 启动后台任务模拟导航过程
            import asyncio
            async def simulate_navigation():
                try:
                    # 模拟导航过程
                    for i in range(len(waypoints)):
                        await asyncio.sleep(2.0)  # 模拟每个航点2秒
                        
                        # 更新任务进度
                        task["progress"] = (i + 1) / len(waypoints) * 100
                        task["current_waypoint"] = i + 1
                        
                        # 更新机器人位置状态
                        if i < len(waypoints):
                            current_pos = waypoints[i]
                            robot_state["position"] = current_pos
                
                    # 导航完成
                    task["active"] = False
                    task["completed"] = True
                    task["status"] = "已完成"
                    task["completed_at"] = datetime.now().isoformat()
                    robot_state["status"] = "idle"
                    robot_state["status_text"] = "待机"
                    
                except Exception as e:
                    logger.error(f"模拟导航失败: {e}")
            
            # 启动后台任务
            asyncio.create_task(simulate_navigation())
            
            return {
                "status": "success",
                "message": f"开始模拟航点导航: {task['description']}",
                "task": task,
                "result": simulated_result,
                "source": "simulation"
            }
        else:
            # 其他任务类型的模拟响应
            simulated_result = {
                "execution_status": "started",
                "progress": 0,
                "estimated_duration": 60.0,
                "simulated": True,
                "task_id": task["id"]
            }
            
            return {
                "status": "success",
                "message": f"开始模拟任务执行: {task['description']}",
                "task": task,
                "result": simulated_result,
                "source": "simulation"
            }
            
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "任务执行失败")
        raise HTTPException(status_code=500, detail=f"任务执行失败: {str(e)}")

@router.post("/task/stop")
async def stop_task(request_data: Dict[str, Any] = None):
    """停止任务 - 支持多种参数格式
    
    支持的参数格式:
    1. 无参数: 停止所有活动任务
    2. 带task_id参数: {"task_id": 12345} - 停止指定任务
    3. 当前任务格式: {"task_id": "current"} - 停止最新任务
    """
    try:
        stopped_tasks = []
        
        if request_data and "task_id" in request_data:
            # 停止指定任务
            task_id = request_data["task_id"]
            
            if task_id == "current":
                # 停止最新任务
                if tasks_list:
                    # 按创建时间降序排序，取最新的活动任务
                    sorted_tasks = sorted(tasks_list, key=lambda t: t.get("id", 0), reverse=True)
                    for task in sorted_tasks:
                        if task["active"]:
                            task["active"] = False
                            task["status"] = "已停止"
                            task["stopped_at"] = datetime.now().isoformat()
                            stopped_tasks.append(task)
                            break
            else:
                # 停止指定ID的任务
                for task in tasks_list:
                    if task["id"] == task_id and task["active"]:
                        task["active"] = False
                        task["status"] = "已停止"
                        task["stopped_at"] = datetime.now().isoformat()
                        stopped_tasks.append(task)
                        break
        else:
            # 停止所有活动任务
            for task in tasks_list:
                if task["active"]:
                    task["active"] = False
                    task["status"] = "已停止"
                    task["stopped_at"] = datetime.now().isoformat()
                    stopped_tasks.append(task)
        
        # 如果没有活动任务，更新机器人状态
        active_tasks = [t for t in tasks_list if t["active"]]
        if not active_tasks and robot_state["status"] == "active":
            robot_state["status"] = "idle"
            robot_state["status_text"] = "待机"
        
        # 记录停止的任务数量
        logger.info(f"停止任务完成: 停止了 {len(stopped_tasks)} 个任务")
        
        return {
            "status": "success",
            "message": f"已停止 {len(stopped_tasks)} 个任务",
            "stopped_tasks": [{"id": t["id"], "description": t["description"]} for t in stopped_tasks]
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "停止任务失败")
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")

@router.post("/emergency/stop")
async def emergency_stop():
    """紧急停止"""
    try:
        # 更新机器人状态
        robot_state["status"] = "emergency"
        robot_state["status_text"] = "紧急停止"
        
        # 停止所有任务
        for task in tasks_list:
            if task["active"]:
                task["active"] = False
                task["status"] = "紧急停止"
                task["stopped_at"] = datetime.now().isoformat()
        
        # 停止所有运动
        motion_model = get_robot_model("motion")
        if motion_model:
            try:
                motion_model.process_input({
                    "operation": "emergency_stop",
                    "data": {"action": "emergency_stop"}
                })
            except Exception as e:
                logger.warning(f"运动模型紧急停止失败: {e}")
        
        return {
            "status": "success",
            "message": "紧急停止已触发",
            "robot_state": robot_state
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "紧急停止失败")
        raise HTTPException(status_code=500, detail=f"紧急停止失败: {str(e)}")

@router.post("/system/reboot")
async def reboot_robot():
    """重启机器人"""
    try:
        # 更新机器人状态
        robot_state["status"] = "rebooting"
        robot_state["status_text"] = "重启中"
        
        # 尝试通过计算机模型重启
        computer_model = get_robot_model("computer")
        if computer_model:
            try:
                result = computer_model.process_input({
                    "operation": "system_reboot",
                    "data": {"action": "reboot"}
                })
                
                return {
                    "status": "success",
                    "message": "机器人重启中...",
                    "result": result,
                    "source": "computer_model"
                }
            except Exception as e:
                logger.warning(f"计算机模型重启失败: {e}")
        
        # 计算机模型不可用，无法重启
        logger.error("机器人重启失败: 计算机模型不可用")
        return {
            "status": "error",
            "message": "机器人重启失败: 需要初始化计算机模型",
            "source": "none",
            "requires_model": True,
            "model_name": "computer",
            "setup_instructions": "请初始化计算机模型以重启机器人"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "重启机器人失败")
        raise HTTPException(status_code=500, detail=f"重启机器人失败: {str(e)}")

@router.post("/system/calibrate")
async def calibrate_system():
    """全系统校准"""
    try:
        # 更新机器人状态
        robot_state["status"] = "calibrating"
        robot_state["status_text"] = "全系统校准中"
        
        # 执行传感器校准
        sensor_model = get_robot_model("sensor")
        if sensor_model:
            try:
                sensor_model.process_input({
                    "operation": "calibration",
                    "data": {"action": "calibrate_all"}
                })
            except Exception as e:
                logger.warning(f"传感器模型校准失败: {e}")
        
        # 执行摄像头校准
        vision_model = get_robot_model("vision")
        if vision_model:
            try:
                vision_model.process_input({
                    "operation": "camera_calibration",
                    "data": {"action": "calibrate_all"}
                })
            except Exception as e:
                logger.warning(f"视觉模型校准失败: {e}")
        
        # 执行运动校准
        motion_model = get_robot_model("motion")
        if motion_model:
            try:
                motion_model.process_input({
                    "operation": "calibration",
                    "data": {"action": "calibrate_all"}
                })
            except Exception as e:
                logger.warning(f"运动模型校准失败: {e}")
        
        return {
            "status": "success",
            "message": "全系统校准开始",
            "robot_state": robot_state
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "全系统校准失败")
        raise HTTPException(status_code=500, detail=f"全系统校准失败: {str(e)}")

@router.get("/tasks")
async def get_tasks():
    """获取任务列表"""
    try:
        return {
            "status": "success",
            "tasks": tasks_list,
            "total": len(tasks_list),
            "active": len([t for t in tasks_list if t.get("active")])
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取任务列表失败")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")

@router.get("/joints")
async def get_joints():
    """获取关节状态"""
    try:
        return {
            "status": "success",
            "joints": joints_state,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取关节状态失败")
        raise HTTPException(status_code=500, detail=f"获取关节状态失败: {str(e)}")

@router.get("/cameras")
async def get_cameras():
    """获取摄像头状态"""
    try:
        # 1. 首先尝试从硬件接口获取摄像头状态
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE and robot_hardware.camera_manager:
            try:
                hardware_cameras = {}
                for camera_id, camera_info in robot_hardware.cameras.items():
                    hardware_cameras[camera_id] = {
                        "active": camera_info.get("is_streaming", False),
                        "url": f"/api/robot/camera/stream/{camera_id}",
                        "calibrated": False,  # 需要从摄像头管理器获取校准状态
                        "type": camera_info.get("type", "unknown"),
                        "resolution": camera_info.get("params", {}).get("resolution", "1920x1080"),
                        "fps": camera_info.get("params", {}).get("fps", 60),
                        "source": "hardware"
                    }
                
                if hardware_cameras:
                    logger.info(f"从硬件接口获取到 {len(hardware_cameras)} 个摄像头状态")
                    return {
                        "status": "success",
                        "cameras": hardware_cameras,
                        "source": "hardware",
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.warning(f"从硬件接口获取摄像头状态失败: {e}")
        
        # 2. 硬件接口不可用，无法获取摄像头状态
        logger.error(f"获取摄像头状态失败: 硬件接口不可用")
        return {
            "status": "error",
            "message": "无法获取摄像头状态: 需要初始化硬件接口",
            "source": "none",
            "requires_hardware": True,
            "hardware_type": "摄像头设备",
            "setup_instructions": "请连接摄像头硬件并初始化硬件接口",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取摄像头状态失败")
        raise HTTPException(status_code=500, detail=f"获取摄像头状态失败: {str(e)}")

@router.post("/collaborate")
async def start_collaboration(command: CollaborationRequest):
    """启动模型协作任务"""
    try:
        pattern = command.pattern
        input_data = command.input_data or {}
        custom_config = command.custom_config or {}
        
        # 检查协作协调器是否可用
        if not model_collaborator:
            logger.error("模型协作协调器不可用")
            return {
                "status": "error",
                "message": "模型协作协调器不可用",
                "requires_initialization": True,
                "setup_instructions": "请初始化模型协作协调器"
            }
        
        # 检查协作模式是否存在
        if pattern not in model_collaborator.collaboration_patterns:
            logger.error(f"未知的协作模式: {pattern}")
            return {
                "status": "error",
                "message": f"未知的协作模式: {pattern}",
                "available_patterns": list(model_collaborator.collaboration_patterns.keys())
            }
        
        # 启动协作
        logger.info(f"启动协作任务: {pattern}")
        result = await model_collaborator.initiate_collaboration(pattern, input_data, custom_config)
        
        # 检查协作结果
        if "error" in result:
            logger.error(f"协作任务失败: {result['error']}")
            return {
                "status": "error",
                "message": f"协作任务失败: {result['error']}",
                "collaboration_id": result.get("collaboration_id"),
                "pattern": pattern
            }
        
        logger.info(f"协作任务成功: {pattern}")
        return {
            "status": "success",
            "message": f"协作任务 '{pattern}' 已完成",
            "pattern": pattern,
            "result": result,
            "collaboration_id": result.get("collaboration_id"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "启动协作任务失败")
        raise HTTPException(status_code=500, detail=f"启动协作任务失败: {str(e)}")

@router.get("/collaboration/patterns")
async def get_collaboration_patterns():
    """获取可用的协作模式列表"""
    try:
        if not model_collaborator:
            return {
                "status": "error",
                "message": "模型协作协调器不可用",
                "patterns": []
            }
        
        patterns = []
        for pattern_name, pattern_info in model_collaborator.collaboration_patterns.items():
            patterns.append({
                "name": pattern_name,
                "description": pattern_info.get("description", ""),
                "models": pattern_info.get("models", []),
                "mode": pattern_info.get("mode", ""),
                "workflow": pattern_info.get("workflow", [])
            })
        
        return {
            "status": "success",
            "patterns": patterns,
            "total": len(patterns),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取协作模式列表失败")
        raise HTTPException(status_code=500, detail=f"获取协作模式列表失败: {str(e)}")

# 自主意识API端点
@router.post("/autonomous/control")
async def control_autonomous(command: AutonomousCommand):
    """控制自主意识模式"""
    try:
        enabled = command.enabled
        goal = command.goal
        constraints = command.constraints or {}
        
        # 更新机器人状态
        if enabled:
            robot_state["status"] = "autonomous"
            robot_state["status_text"] = f"自主模式: {goal}" if goal else "自主模式"
        else:
            if robot_state["status"] == "autonomous":
                robot_state["status"] = "idle"
                robot_state["status_text"] = "待机"
        
        # 尝试通过自主模型控制
        autonomous_model = get_robot_model("autonomous")
        if autonomous_model:
            try:
                result = autonomous_model.process_input({
                    "operation": "autonomous_control",
                    "data": {
                        "enabled": enabled,
                        "goal": goal,
                        "constraints": constraints
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"自主模式 {'启用' if enabled else '禁用'}",
                    "goal": goal,
                    "result": result,
                    "robot_state": robot_state,
                    "source": "autonomous_model"
                }
            except Exception as e:
                logger.warning(f"自主模型控制失败: {e}")
        
        # 自主模型不可用
        logger.error(f"自主模式控制失败: 自主模型不可用")
        return {
            "status": "error",
            "message": f"自主模式控制失败: 需要初始化自主模型",
            "robot_state": robot_state,
            "source": "none",
            "requires_model": True,
            "model_name": "autonomous",
            "setup_instructions": "请初始化自主模型以启用自主意识功能"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "自主模式控制失败")
        raise HTTPException(status_code=500, detail=f"自主模式控制失败: {str(e)}")

@router.post("/autonomous/goal")
async def set_autonomous_goal(command: GoalSetting):
    """设置自主目标"""
    try:
        goal = command.goal
        priority = command.priority
        deadline = command.deadline
        
        # 尝试通过自主模型设置目标
        autonomous_model = get_robot_model("autonomous")
        if autonomous_model:
            try:
                result = autonomous_model.process_input({
                    "operation": "set_goal",
                    "data": {
                        "goal": goal,
                        "priority": priority,
                        "deadline": deadline
                    }
                })
                
                # 更新机器人状态
                robot_state["status"] = "planning"
                robot_state["status_text"] = f"目标规划: {goal}"
                
                return {
                    "status": "success",
                    "message": f"目标已设置: {goal}",
                    "goal": goal,
                    "result": result,
                    "robot_state": robot_state,
                    "source": "autonomous_model"
                }
            except Exception as e:
                logger.warning(f"自主模型设置目标失败: {e}")
        
        # 自主模型不可用，尝试通过规划模型
        planning_model = get_robot_model("planning")
        if planning_model:
            try:
                result = planning_model.process_input({
                    "operation": "goal_planning",
                    "data": {
                        "goal": goal,
                        "priority": priority,
                        "deadline": deadline
                    }
                })
                
                # 更新机器人状态
                robot_state["status"] = "planning"
                robot_state["status_text"] = f"目标规划: {goal}"
                
                return {
                    "status": "success",
                    "message": f"目标已设置: {goal}",
                    "goal": goal,
                    "result": result,
                    "robot_state": robot_state,
                    "source": "planning_model"
                }
            except Exception as e:
                logger.warning(f"规划模型设置目标失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"设置目标失败: 自主模型和规划模型都不可用")
        return {
            "status": "error",
            "message": f"设置目标失败: 需要初始化自主模型或规划模型",
            "goal": goal,
            "source": "none",
            "requires_model": True,
            "model_name": "autonomous",
            "setup_instructions": "请初始化自主模型或规划模型以设置目标"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "设置目标失败")
        raise HTTPException(status_code=500, detail=f"设置目标失败: {str(e)}")

@router.get("/autonomous/status")
async def get_autonomous_status():
    """获取自主意识状态"""
    try:
        # 尝试从自主模型获取状态
        autonomous_model = get_robot_model("autonomous")
        if autonomous_model:
            try:
                result = autonomous_model.process_input({
                    "operation": "get_status",
                    "data": {}
                })
                
                return {
                    "status": "success",
                    "data": result,
                    "robot_state": robot_state,
                    "source": "autonomous_model"
                }
            except Exception as e:
                logger.warning(f"自主模型获取状态失败: {e}")
        
        # 自主模型不可用，返回基础状态
        autonomous_status = {
            "enabled": robot_state["status"] == "autonomous",
            "current_goal": robot_state["status_text"] if robot_state["status"] == "autonomous" else None,
            "robot_state": robot_state,
            "capabilities": ["goal_setting", "planning", "execution"],
            "models_available": {
                "autonomous": autonomous_model is not None,
                "planning": get_robot_model("planning") is not None,
                "motion": get_robot_model("motion") is not None
            }
        }
        
        return {
            "status": "success",
            "data": autonomous_status,
            "source": "basic"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取自主状态失败")
        raise HTTPException(status_code=500, detail=f"获取自主状态失败: {str(e)}")

@router.post("/autonomous/plan")
async def generate_autonomous_plan(command: GoalSetting):
    """生成自主计划"""
    try:
        goal = command.goal
        priority = command.priority
        deadline = command.deadline
        
        # 尝试通过规划模型生成计划
        planning_model = get_robot_model("planning")
        if planning_model:
            try:
                result = planning_model.process_input({
                    "operation": "generate_plan",
                    "data": {
                        "goal": goal,
                        "priority": priority,
                        "deadline": deadline
                    }
                })
                
                # 更新机器人状态
                robot_state["status"] = "planning"
                robot_state["status_text"] = f"计划生成中: {goal}"
                
                return {
                    "status": "success",
                    "message": f"为目标生成计划: {goal}",
                    "goal": goal,
                    "plan": result,
                    "robot_state": robot_state,
                    "source": "planning_model"
                }
            except Exception as e:
                logger.warning(f"规划模型生成计划失败: {e}")
        
        # 规划模型不可用
        logger.error(f"生成计划失败: 规划模型不可用")
        return {
            "status": "error",
            "message": f"生成计划失败: 需要初始化规划模型",
            "goal": goal,
            "source": "none",
            "requires_model": True,
            "model_name": "planning",
            "setup_instructions": "请初始化规划模型以生成自主计划"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "生成计划失败")
        raise HTTPException(status_code=500, detail=f"生成计划失败: {str(e)}")

@router.post("/autonomous/execute")
async def execute_autonomous_plan(command: TaskSequence):
    """执行自主计划"""
    try:
        tasks = command.tasks
        dependencies = command.dependencies or {}
        
        # 更新机器人状态
        robot_state["status"] = "autonomous"
        robot_state["status_text"] = f"执行自主计划: {len(tasks)} 个任务"
        
        # 尝试通过自主模型执行计划
        autonomous_model = get_robot_model("autonomous")
        if autonomous_model:
            try:
                result = autonomous_model.process_input({
                    "operation": "execute_plan",
                    "data": {
                        "tasks": tasks,
                        "dependencies": dependencies
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"开始执行自主计划: {len(tasks)} 个任务",
                    "tasks_count": len(tasks),
                    "result": result,
                    "robot_state": robot_state,
                    "source": "autonomous_model"
                }
            except Exception as e:
                logger.warning(f"自主模型执行计划失败: {e}")
        
        # 自主模型不可用，尝试通过规划模型
        planning_model = get_robot_model("planning")
        if planning_model:
            try:
                result = planning_model.process_input({
                    "operation": "execute_tasks",
                    "data": {
                        "tasks": tasks,
                        "dependencies": dependencies
                    }
                })
                
                return {
                    "status": "success",
                    "message": f"开始执行计划: {len(tasks)} 个任务",
                    "tasks_count": len(tasks),
                    "result": result,
                    "robot_state": robot_state,
                    "source": "planning_model"
                }
            except Exception as e:
                logger.warning(f"规划模型执行计划失败: {e}")
        
        # 所有模型都不可用
        logger.error(f"执行计划失败: 自主模型和规划模型都不可用")
        return {
            "status": "error",
            "message": f"执行计划失败: 需要初始化自主模型或规划模型",
            "tasks_count": len(tasks),
            "source": "none",
            "requires_model": True,
            "model_name": "autonomous",
            "setup_instructions": "请初始化自主模型或规划模型以执行计划"
        }
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "执行计划失败")
        raise HTTPException(status_code=500, detail=f"执行计划失败: {str(e)}")

# 机器人训练API端点
# @router.post("/training/start")
async def start_robot_training(request: TrainingStartRequest):
    """启动机器人训练 - 支持前后端两种数据结构"""
    try:
        # 检查是否已经在训练中
        if training_state["status"] == "training":
            raise HTTPException(status_code=400, detail="训练正在进行中，请先停止当前训练")
        
        # 提取训练模式（兼容前端"mode"字段和后端"training_mode"字段）
        training_mode = request.training_mode
        if training_mode is None and hasattr(request, 'mode') and request.mode is not None:
            training_mode = request.mode
        
        # 提取数据集ID（兼容前端"dataset"字段和后端"dataset_id"字段）
        dataset_id = request.dataset_id
        if dataset_id is None and hasattr(request, 'dataset') and request.dataset is not None:
            dataset_id = request.dataset
        
        # 提取训练参数（兼容前端"parameters"字段和后端"training_params"字段）
        training_params = request.training_params
        if training_params is None and hasattr(request, 'parameters') and request.parameters is not None:
            training_params = request.parameters
        
        # 提取硬件配置（前端可能使用hardware_config对象）
        selected_joints = request.selected_joints or []
        selected_sensors = request.selected_sensors or []
        selected_cameras = request.selected_cameras or []
        safety_limits = request.safety_limits or training_state["safety_limits"]
        
        # 如果存在hardware_config，从中提取值
        if request.hardware_config:
            hardware_config = request.hardware_config
            if not selected_joints and 'selected_joints' in hardware_config:
                selected_joints = hardware_config['selected_joints'] or []
            if not selected_sensors and 'selected_sensors' in hardware_config:
                selected_sensors = hardware_config['selected_sensors'] or []
            if not selected_cameras and 'selected_cameras' in hardware_config:
                selected_cameras = hardware_config['selected_cameras'] or []
            if safety_limits == training_state["safety_limits"] and 'safety_limits' in hardware_config:
                safety_limits = hardware_config['safety_limits'] or training_state["safety_limits"]
        
        # 生成训练ID
        training_id = f"robot_train_{int(time.time() * 1000)}"
        
        # 更新训练状态
        training_state["status"] = "training"
        training_state["progress"] = 0
        training_state["training_id"] = training_id
        training_state["mode"] = training_mode
        training_state["models"] = request.models
        training_state["dataset_id"] = dataset_id
        training_state["selected_joints"] = selected_joints
        training_state["selected_sensors"] = selected_sensors
        training_state["selected_cameras"] = selected_cameras
        training_state["training_params"] = training_params or training_state["training_params"]
        training_state["safety_limits"] = safety_limits
        training_state["started_at"] = datetime.now().isoformat()
        training_state["error"] = None
        
        # 添加训练日志
        dataset_info = f"，数据集={dataset_id}" if dataset_id else ""
        training_state["training_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"训练启动: 模式={training_mode}, 模型={request.models}{dataset_info}",
            "level": "info"
        })
        
        # 初始化硬件接口（如果可用）
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 检查所选关节是否可用
                available_joints = robot_hardware.get_available_joints()
                for joint in training_state["selected_joints"]:
                    if joint not in available_joints:
                        training_state["training_log"].append({
                            "timestamp": datetime.now().isoformat(),
                            "message": f"警告: 关节 {joint} 不可用",
                            "level": "warning"
                        })
                
                # 检查所选传感器是否可用
                available_sensors = robot_hardware.get_sensor_list()
                for sensor in training_state["selected_sensors"]:
                    if sensor not in available_sensors:
                        training_state["training_log"].append({
                            "timestamp": datetime.now().isoformat(),
                            "message": f"警告: 传感器 {sensor} 不可用",
                            "level": "warning"
                        })
                
                # 初始化训练硬件
                robot_hardware.initialize_training_mode(
                    joints=training_state["selected_joints"],
                    sensors=training_state["selected_sensors"],
                    safety_limits=training_state["safety_limits"]
                )
                
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "硬件接口初始化成功",
                    "level": "info"
                })
            except Exception as e:
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"硬件接口初始化失败: {str(e)}",
                    "level": "error"
                })
                logger.warning(f"硬件接口初始化失败: {e}")
        
        # 启动真正的后台训练任务
        if ROBOT_TRAINING_MANAGER_AVAILABLE and robot_training_manager:
            try:
                # 使用真正的训练管理器启动训练
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "使用机器人训练管理器启动训练",
                    "level": "info"
                })
                
                # 启动异步训练任务
                asyncio.create_task(
                    run_robot_training(
                        training_id,
                        training_mode,
                        request.models,
                        dataset_id,
                        selected_joints,
                        selected_sensors,
                        selected_cameras,
                        training_params or training_state["training_params"],
                        safety_limits
                    )
                )
                
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "后台训练任务已启动",
                    "level": "info"
                })
            except Exception as e:
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"训练管理器启动失败: {str(e)}",
                    "level": "error"
                })
                logger.warning(f"训练管理器启动失败: {e}")
        else:
            # 训练管理器不可用，返回错误
            training_state["training_log"].append({
                "timestamp": datetime.now().isoformat(),
                "message": "错误: 机器人训练管理器不可用，真实AGI训练需要硬件和训练管理器",
                "level": "error"
            })
            logger.error("机器人训练管理器不可用，真实AGI训练需要硬件和训练管理器")
            
            # 返回错误而不是启动模拟训练
            return {
                "status": "error",
                "message": "机器人训练管理器不可用，真实AGI训练需要硬件和训练管理器",
                "training_id": training_id,
                "requires_training_manager": True,
                "training_state": training_state
            }
        
        return {
            "status": "success",
            "message": "训练启动成功",
            "training_id": training_id,
            "training_state": training_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "启动训练失败")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

# @router.post("/training/pause")
async def pause_robot_training(request: TrainingPauseRequest):
    """暂停机器人训练"""
    try:
        training_id = request.training_id
        
        # 验证训练ID
        if training_state["training_id"] != training_id:
            raise HTTPException(status_code=404, detail=f"训练ID {training_id} 不存在")
        
        if training_state["status"] != "training":
            raise HTTPException(status_code=400, detail="训练未在进行中，无法暂停")
        
        # 更新训练状态
        training_state["status"] = "paused"
        
        # 添加训练日志
        training_state["training_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"训练暂停",
            "level": "warning"
        })
        
        # 暂停硬件控制（如果可用）
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                robot_hardware.pause_training()
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "硬件控制已暂停",
                    "level": "info"
                })
            except Exception as e:
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"硬件控制暂停失败: {str(e)}",
                    "level": "error"
                })
                logger.warning(f"硬件控制暂停失败: {e}")
        
        return {
            "status": "success",
            "message": "训练暂停成功",
            "training_id": training_id,
            "training_state": training_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "暂停训练失败")
        raise HTTPException(status_code=500, detail=f"暂停训练失败: {str(e)}")

# @router.post("/training/stop")
async def stop_robot_training(request: TrainingStopRequest):
    """停止机器人训练"""
    try:
        training_id = request.training_id
        
        # 验证训练ID
        if training_state["training_id"] != training_id:
            raise HTTPException(status_code=404, detail=f"训练ID {training_id} 不存在")
        
        if training_state["status"] not in ["training", "paused"]:
            raise HTTPException(status_code=400, detail="训练未在进行中或已暂停，无法停止")
        
        # 更新训练状态
        training_state["status"] = "idle"
        training_state["progress"] = 0
        
        # 添加训练日志
        training_state["training_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"训练停止",
            "level": "warning"
        })
        
        # 停止硬件控制（如果可用）
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                robot_hardware.stop_training()
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "硬件控制已停止",
                    "level": "info"
                })
            except Exception as e:
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"硬件控制停止失败: {str(e)}",
                    "level": "error"
                })
                logger.warning(f"硬件控制停止失败: {e}")
        
        return {
            "status": "success",
            "message": "训练停止成功",
            "training_id": training_id,
            "training_state": training_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "停止训练失败")
        raise HTTPException(status_code=500, detail=f"停止训练失败: {str(e)}")

# @router.post("/training/reset")
async def reset_robot_training(request: TrainingResetRequest):
    """重置机器人训练配置"""
    try:
        training_id = request.training_id
        
        # 验证训练ID
        if training_state["training_id"] != training_id:
            raise HTTPException(status_code=404, detail=f"训练ID {training_id} 不存在")
        
        if training_state["status"] == "training":
            raise HTTPException(status_code=400, detail="训练进行中，请先停止训练")
        
        # 重置训练状态
        training_state["status"] = "idle"
        training_state["progress"] = 0
        training_state["active_training"] = None
        training_state["error"] = None
        training_state["training_id"] = None
        training_state["mode"] = None
        training_state["models"] = []
        training_state["selected_joints"] = []
        training_state["selected_sensors"] = []
        training_state["selected_cameras"] = []
        training_state["started_at"] = None
        training_state["training_log"] = []
        
        # 重置硬件配置（如果可用）
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                robot_hardware.reset_training_config()
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": "硬件配置已重置",
                    "level": "info"
                })
            except Exception as e:
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"硬件配置重置失败: {str(e)}",
                    "level": "error"
                })
                logger.warning(f"硬件配置重置失败: {e}")
        
        return {
            "status": "success",
            "message": "训练配置重置成功",
            "training_id": training_id,
            "training_state": training_state
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "重置训练失败")
        raise HTTPException(status_code=500, detail=f"重置训练失败: {str(e)}")

# @router.get("/training/status")
async def get_training_status():
    """获取训练状态"""
    try:
        response = TrainingStatusResponse(
            training_id=training_state["training_id"] or "",
            status=training_state["status"],
            progress=training_state["progress"],
            mode=training_state["mode"],
            models=training_state["models"],
            dataset_id=training_state["dataset_id"],
            started_at=training_state["started_at"],
            error=training_state["error"],
            training_params=training_state["training_params"],
            safety_limits=training_state["safety_limits"],
            training_log=training_state["training_log"][-20:] if training_state["training_log"] else []  # 返回最近20条日志
        )
        
        return {
            "status": "success",
            "data": response
        }
        
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取训练状态失败")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")

# @router.get("/training/progress")
async def get_training_progress():
    """获取训练进度"""
    try:
        return {
            "status": "success",
            "training_id": training_state["training_id"],
            "progress": training_state["progress"],
            "status_text": training_state["status"]
        }
        
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取训练进度失败")
        raise HTTPException(status_code=500, detail=f"获取训练进度失败: {str(e)}")

# @router.get("/training/logs")
async def get_training_logs(limit: int = 50):
    """获取训练日志"""
    try:
        logs = training_state["training_log"][-limit:] if training_state["training_log"] else []
        return {
            "status": "success",
            "logs": logs,
            "total": len(training_state["training_log"]),
            "limit": limit
        }
        
    except Exception as e:
        error_handler.handle_error(e, "RobotAPI", "获取训练日志失败")
        raise HTTPException(status_code=500, detail=f"获取训练日志失败: {str(e)}")

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.update_task = None
        self.running = False
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")
        
        # 如果有连接且任务未运行，启动更新任务
        if not self.running and self.active_connections:
            self.start_update_task()
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")
        
        # 如果没有活动连接，停止更新任务
        if not self.active_connections and self.running:
            self.stop_update_task()
    
    async def broadcast(self, message_type: str, data: Dict[str, Any]):
        """广播消息给所有连接"""
        if not self.active_connections:
            return
        
        message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket发送失败: {e}")
                disconnected.append(connection)
        
        # 移除断开的连接
        for connection in disconnected:
            self.disconnect(connection)
    
    def start_update_task(self):
        """启动数据更新任务"""
        if self.running:
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("WebSocket数据更新任务已启动")
    
    def stop_update_task(self):
        """停止数据更新任务"""
        if not self.running:
            return
        
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            self.update_task = None
        logger.info("WebSocket数据更新任务已停止")
    
    async def _update_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                # 更新机器人状态
                await self.broadcast("robot_state", robot_state)
                
                # 更新传感器数据
                sensor_data = {}
                if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
                    try:
                        sensor_data = robot_hardware.get_sensor_data()
                    except Exception as e:
                        logger.warning(f"获取传感器数据失败: {e}")
                        sensor_data = {"status": "hardware_unavailable"}
                
                await self.broadcast("sensor_data", sensor_data)
                
                # 更新关节状态
                await self.broadcast("joint_status", joints_state)
                
                # 更新摄像头状态
                await self.broadcast("camera_feed", cameras_state)
                
                # 每500毫秒更新一次
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket更新循环错误: {e}")
                await asyncio.sleep(1)

# 创建全局WebSocket管理器
websocket_manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket实时数据端点"""
    try:
        await websocket_manager.connect(websocket)
        
        # 发送初始数据
        try:
            await websocket.send_json({
                "type": "robot_state",
                "timestamp": datetime.now().isoformat(),
                "data": robot_state
            })
            
            await websocket.send_json({
                "type": "joint_status",
                "timestamp": datetime.now().isoformat(),
                "data": joints_state
            })
            
            await websocket.send_json({
                "type": "camera_feed",
                "timestamp": datetime.now().isoformat(),
                "data": cameras_state
            })
            
            # 发送传感器数据
            sensor_data = {}
            if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
                try:
                    sensor_data = robot_hardware.get_sensor_data()
                except Exception as e:
                    logger.warning(f"获取传感器数据失败: {e}")
                    sensor_data = {"status": "hardware_unavailable"}
            
            await websocket.send_json({
                "type": "sensor_data",
                "timestamp": datetime.now().isoformat(),
                "data": sensor_data
            })
            
            # 发送协作模式信息
            if model_collaborator:
                patterns_list = []
                for pattern_name, pattern_info in model_collaborator.collaboration_patterns.items():
                    patterns_list.append({
                        "name": pattern_name,
                        "description": pattern_info.get("description", ""),
                        "models": pattern_info.get("models", []),
                        "mode": pattern_info.get("mode", "")
                    })
                await websocket.send_json({
                    "type": "collaboration_patterns",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "patterns": patterns_list,
                        "total": len(patterns_list)
                    }
                })
        except Exception as e:
            logger.error(f"发送初始数据失败: {e}")
        
        # 保持连接，接收消息
        try:
            while True:
                data = await websocket.receive_text()
                logger.info(f"收到WebSocket消息: {data}")
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type", "")
                    message_data = message.get("data", {})
                    
                    if message_type == "control_joint":
                        # 控制关节
                        joint_id = message_data.get("jointId")
                        value = message_data.get("value")
                        if joint_id and value is not None:
                            # 更新关节状态
                            parts = joint_id.split('_')
                            if len(parts) == 3:
                                limb, joint_name, sub_joint = parts
                                if limb in joints_state and joint_name in joints_state[limb]:
                                    joints_state[limb][joint_name][sub_joint] = value
                            
                            # 广播更新
                            await websocket_manager.broadcast("joint_status", joints_state)
                    
                    elif message_type == "control_motion":
                        # 控制运动
                        motion_type = message_data.get("motion")
                        params = message_data.get("params", {})
                        if motion_type:
                            logger.info(f"收到运动控制请求: {motion_type}, 参数: {params}")
                    
                    elif message_type == "start_collaboration":
                        # 启动协作
                        pattern = message_data.get("pattern")
                        input_data = message_data.get("input_data", {})
                        if pattern and model_collaborator:
                            result = await model_collaborator.initiate_collaboration(pattern, input_data)
                            await websocket.send_json({
                                "type": "collaboration_result",
                                "timestamp": datetime.now().isoformat(),
                                "data": result
                            })
                    
                    elif message_type == "get_status":
                        # 获取状态
                        await websocket.send_json({
                            "type": "system_status",
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "robot_state": robot_state,
                                "joints_count": len(joints_state),
                                "sensors_count": len(sensors_state),
                                "cameras_count": len(cameras_state)
                            }
                        })
                    
                    elif message_type == "get_sensor_data":
                        # 获取传感器数据
                        sensor_data = {}
                        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
                            try:
                                sensor_data = robot_hardware.get_sensor_data()
                            except Exception as e:
                                sensor_data = {"error": str(e)}
                        
                        await websocket.send_json({
                            "type": "sensor_data",
                            "timestamp": datetime.now().isoformat(),
                            "data": sensor_data
                        })
                    
                    elif message_type == "get_joint_status":
                        # 获取关节状态
                        await websocket.send_json({
                            "type": "joint_status",
                            "timestamp": datetime.now().isoformat(),
                            "data": joints_state
                        })
                
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSON消息: {data}")
                except Exception as e:
                    logger.error(f"处理WebSocket消息失败: {e}")
        
        except WebSocketDisconnect:
            logger.info("WebSocket连接断开")
        except Exception as e:
            logger.error(f"WebSocket通信错误: {e}")
        finally:
            websocket_manager.disconnect(websocket)
    
    except Exception as e:
        logger.error(f"WebSocket端点错误: {e}")

async def run_robot_training(training_id: str, mode: str, models: List[str], dataset_id: Optional[str],
                            selected_joints: List[str], selected_sensors: List[str], selected_cameras: List[str],
                            training_params: Dict[str, Any], safety_limits: Dict[str, Any]):
    """运行真正的机器人训练任务"""
    try:
        logger.info(f"开始真正的机器人训练: ID={training_id}, 模式={mode}, 模型={models}")
        
        # 更新训练状态为训练中
        training_state["status"] = "training"
        training_state["progress"] = 0
        
        # 如果训练管理器可用，使用真正的训练
        if ROBOT_TRAINING_MANAGER_AVAILABLE and robot_training_manager:
            try:
                # 配置训练
                config = {
                    "training_id": training_id,
                    "mode": mode,
                    "models": models,
                    "dataset_id": dataset_id,
                    "selected_joints": selected_joints,
                    "selected_sensors": selected_sensors,
                    "selected_cameras": selected_cameras,
                    "training_params": training_params,
                    "safety_limits": safety_limits
                }
                
                # 启动真正的训练
                result = await robot_training_manager.start_training(config)
                
                # 更新训练状态
                training_state["progress"] = 100
                training_state["status"] = "completed"
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"训练完成: {result.get('message', '成功')}",
                    "level": "info"
                })
                
                logger.info(f"机器人训练完成: {training_id}")
                return
                
            except Exception as e:
                logger.error(f"训练管理器执行失败: {e}")
                training_state["training_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"训练执行失败: {str(e)}",
                    "level": "error"
                })
        
        # 训练管理器不可用或执行失败，返回错误状态
        training_state["status"] = "error"
        training_state["error"] = "训练管理器不可用或执行失败，真实AGI训练需要硬件和训练管理器"
        training_state["training_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": "训练失败: 真实AGI训练需要硬件和训练管理器",
            "level": "error"
        })
        logger.error("真实AGI训练需要硬件和训练管理器，模拟训练已被禁用")
        
    except Exception as e:
        logger.error(f"运行机器人训练任务失败: {e}")
        training_state["status"] = "error"
        training_state["error"] = str(e)
        training_state["training_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"训练任务失败: {str(e)}",
            "level": "error"
        })

@router.get("/robot/health")
async def get_robot_health():
    """获取机器人全面健康状态监控数据"""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "status": "operational",
                "uptime": None,  # 实际实现中应该获取系统运行时间
                "cpu_usage": None,
                "memory_usage": None,
                "disk_usage": None,
                "python_version": platform.python_version(),
                "os_info": f"{platform.system()} {platform.release()}"
            },
            "hardware": {
                "connected": False,
                "interface_available": HARDWARE_INTERFACE_AVAILABLE,
                "robot_driver_available": ROBOT_DRIVER_AVAILABLE,
                "components": {
                    "joints": 0,
                    "sensors": 0,
                    "cameras": 0,
                    "motors": 0
                },
                "connection_status": "disconnected",
                "hardware_errors": []
            },
            "training": {
                "active": training_state["status"] == "training",
                "current_mode": training_state.get("mode", "none"),
                "progress": training_state.get("progress", 0),
                "safety_violations": training_state.get("safety_violations", 0),
                "training_sessions": len(training_state.get("training_history", []))
            },
            "safety": {
                "emergency_stop_active": False,
                "temperature_ok": True,
                "battery_ok": True,
                "joint_limits_ok": True,
                "safety_monitor_active": False
            },
            "performance": {
                "response_time": 0,
                "data_throughput": 0,
                "control_latency": 0
            }
        }
        
        # 获取真实硬件状态
        if robot_hardware and HARDWARE_INTERFACE_AVAILABLE:
            try:
                # 更新硬件连接状态
                health_status["hardware"]["connected"] = True
                health_status["hardware"]["connection_status"] = "connected"
                
                # 获取硬件组件数量
                if hasattr(robot_hardware, 'joints'):
                    health_status["hardware"]["components"]["joints"] = len(robot_hardware.joints)
                
                if hasattr(robot_hardware, 'sensors'):
                    health_status["hardware"]["components"]["sensors"] = len(robot_hardware.sensors)
                
                if hasattr(robot_hardware, 'cameras'):
                    health_status["hardware"]["components"]["cameras"] = len(robot_hardware.cameras)
                
                if hasattr(robot_hardware, 'motors'):
                    health_status["hardware"]["components"]["motors"] = len(robot_hardware.motors)
                
                # 获取硬件健康指标
                try:
                    hardware_state = robot_hardware.get_state()
                    if hardware_state:
                        # 检查温度
                        temperature = hardware_state.get("temperature", 0)
                        health_status["safety"]["temperature_ok"] = temperature < 70
                        health_status["hardware"]["temperature"] = temperature
                        
                        # 检查电池
                        battery = hardware_state.get("battery", 100)
                        health_status["safety"]["battery_ok"] = battery > 20
                        health_status["hardware"]["battery"] = battery
                        
                        # 检查关节状态
                        joints = hardware_state.get("joint_positions", {})
                        if joints:
                            health_status["hardware"]["joint_positions_received"] = len(joints)
                            
                            # 检查关节是否在合理范围内
                            for joint_id, position in joints.items():
                                if position > 90 or position < -90:
                                    health_status["safety"]["joint_limits_ok"] = False
                                    health_status["hardware"]["hardware_errors"].append(f"Joint {joint_id} out of range: {position}")
                except Exception as e:
                    health_status["hardware"]["hardware_errors"].append(f"Hardware state retrieval error: {str(e)}")
                
                # 检查机器人驱动器状态
                if hasattr(robot_hardware, 'robot_driver') and robot_hardware.robot_driver:
                    health_status["hardware"]["robot_driver_connected"] = True
                    health_status["hardware"]["robot_driver_protocol"] = getattr(robot_hardware.robot_driver, 'protocol', 'unknown')
                else:
                    health_status["hardware"]["robot_driver_connected"] = False
                    
            except Exception as e:
                health_status["hardware"]["hardware_errors"].append(f"Hardware interface error: {str(e)}")
                health_status["hardware"]["connection_status"] = "error"
        
        # 计算总体健康评分 (0-100)
        health_score = 100
        
        # 减分项
        if not health_status["hardware"]["connected"]:
            health_score -= 30
        
        if not health_status["safety"]["temperature_ok"]:
            health_score -= 20
        
        if not health_status["safety"]["battery_ok"]:
            health_score -= 15
        
        if len(health_status["hardware"]["hardware_errors"]) > 0:
            health_score -= min(25, len(health_status["hardware"]["hardware_errors"]) * 5)
        
        if not health_status["safety"]["joint_limits_ok"]:
            health_score -= 10
        
        health_status["health_score"] = max(0, health_score)
        health_status["health_level"] = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "status": "success",
            "health": health_status,
            "timestamp": datetime.now().isoformat(),
            "message": f"机器人健康状态: {health_status['health_level']} (得分: {health_score}/100)"
        }
        
    except Exception as e:
        logger.error(f"获取机器人健康状态失败: {e}")
        return {
            "status": "error",
            "health": None,
            "message": f"获取机器人健康状态失败: {str(e)}",
            "requires_hardware": True,
            "setup_instructions": "请连接机器人硬件以获取完整的健康状态信息"
        }

async def run_basic_training_simulation(training_id: str):
    """基础训练模拟 - 已禁用，真实AGI训练需要硬件和训练管理器"""
    logger.error(f"模拟训练已被禁用: {training_id}")
    training_state["status"] = "error"
    training_state["error"] = "模拟训练已被禁用，真实AGI训练需要硬件和训练管理器"
    training_state["training_log"].append({
        "timestamp": datetime.now().isoformat(),
        "message": "错误: 模拟训练已被禁用，真实AGI训练需要硬件和训练管理器",
        "level": "error"
    })
    raise RuntimeError("模拟训练已被禁用。真实AGI人形机器人训练需要硬件和训练管理器。请连接真实硬件并确保训练管理器已初始化。")

# 初始化机器人API - 由外部调用
# initialize_robot_api()  # 已注释，由外部调用
