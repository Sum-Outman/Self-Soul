"""
运动控制系统

提供视觉-运动协同控制、语音-动作映射、触觉反馈-动作调整等高级运动控制功能。
支持多模态输入到机器人动作的实时映射和转换。
将原有Self-Soul系统的robot_motion_control.py功能迁移到统一认知架构中。
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class MotionType(Enum):
    """运动类型枚举"""
    WALKING = "walking"           # 行走
    TURNING = "turning"           # 转向
    REACHING = "reaching"         # 伸手
    GRASPING = "grasping"         # 抓取
    BALANCING = "balancing"       # 平衡
    GESTURE = "gesture"           # 手势
    NAVIGATION = "navigation"     # 导航
    MANIPULATION = "manipulation" # 操作


class ControlMode(Enum):
    """控制模式枚举"""
    POSITION_CONTROL = "position"      # 位置控制
    VELOCITY_CONTROL = "velocity"      # 速度控制
    TORQUE_CONTROL = "torque"          # 扭矩控制
    IMPEDANCE_CONTROL = "impedance"    # 阻抗控制
    ADMITTANCE_CONTROL = "admittance"  # 导纳控制


@dataclass
class MotionCommand:
    """运动命令数据结构"""
    motion_type: MotionType
    target: Dict[str, Any]           # 目标参数
    constraints: Dict[str, Any]      # 约束条件
    control_mode: ControlMode        # 控制模式
    priority: int = 1                # 优先级 (1-10)
    duration: float = 1.0            # 持续时间(秒)
    blend_radius: float = 0.0        # 混合半径


@dataclass
class JointTrajectory:
    """关节轨迹数据结构"""
    joint_names: List[str]
    positions: List[List[float]]     # 位置序列
    times: List[float]               # 时间序列
    velocities: Optional[List[List[float]]] = None  # 速度序列
    accelerations: Optional[List[List[float]]] = None  # 加速度序列


class MotionControlSystem:
    """运动控制系统"""
    
    def __init__(self, communication):
        """
        初始化运动控制系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 运动规划参数
        self.planning_params = {
            "max_velocity": 5.0,          # 最大关节速度 (度/秒)
            "max_acceleration": 10.0,     # 最大关节加速度 (度/秒²)
            "max_jerk": 50.0,             # 最大关节加加速度 (度/秒³)
            "sampling_time": 0.01,        # 采样时间 (秒)
            "safety_margin": 0.1,         # 安全边界
        }
        
        # 多模态映射规则
        self.mapping_rules = {
            # 语音命令到动作映射
            "voice_to_action": {
                "前进": {"motion_type": MotionType.WALKING, "params": {"direction": "forward", "steps": 3}},
                "后退": {"motion_type": MotionType.WALKING, "params": {"direction": "backward", "steps": 2}},
                "左转": {"motion_type": MotionType.TURNING, "params": {"direction": "left", "angle": 90}},
                "右转": {"motion_type": MotionType.TURNING, "params": {"direction": "right", "angle": 90}},
                "停止": {"motion_type": MotionType.BALANCING, "params": {"action": "stop"}},
                "坐下": {"motion_type": MotionType.BALANCING, "params": {"action": "sit"}},
                "站立": {"motion_type": MotionType.BALANCING, "params": {"action": "stand"}},
                "挥手": {"motion_type": MotionType.GESTURE, "params": {"gesture": "wave"}},
                "点头": {"motion_type": MotionType.GESTURE, "params": {"gesture": "nod"}},
            },
            
            # 视觉输入到动作映射
            "vision_to_action": {
                "检测到障碍物": {"motion_type": MotionType.NAVIGATION, "params": {"action": "avoid", "obstacle_type": "static"}},
                "识别到目标": {"motion_type": MotionType.NAVIGATION, "params": {"action": "approach", "target_type": "object"}},
                "检测到人脸": {"motion_type": MotionType.GESTURE, "params": {"action": "greet", "person_distance": "near"}},
                "识别到手势": {"motion_type": MotionType.GESTURE, "params": {"action": "respond", "gesture_type": "recognized"}},
                "检测到跌落风险": {"motion_type": MotionType.BALANCING, "params": {"action": "recover", "risk_level": "high"}},
            },
            
            # 传感器输入到动作映射
            "sensor_to_action": {
                "失去平衡": {"motion_type": MotionType.BALANCING, "params": {"action": "recover", "perturbation": "detected"}},
                "接触力过大": {"motion_type": MotionType.MANIPULATION, "params": {"action": "comply", "force_threshold": "exceeded"}},
                "接近边界": {"motion_type": MotionType.NAVIGATION, "params": {"action": "retreat", "boundary_type": "workspace"}},
                "温度过高": {"motion_type": MotionType.BALANCING, "params": {"action": "cool", "temperature": "critical"}},
            }
        }
        
        # 运动规划器缓存
        self.trajectory_cache = {}
        
        # 运动学模型
        self.kinematics = None
        
        # 性能指标
        self.performance_metrics = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info("运动控制系统已初始化")
    
    async def initialize(self):
        """初始化运动控制系统"""
        if self.initialized:
            return
        
        logger.info("初始化运动控制系统...")
        
        # 初始化运动学模型
        self._initialize_kinematics()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="motion_control",
            component_type="control"
        )
        
        self.initialized = True
        logger.info("运动控制系统初始化完成")
    
    def _initialize_kinematics(self):
        """初始化运动学模型"""
        # 简化的机器人运动学参数
        self.kinematics = {
            "joint_limits": {
                "hip": {"min": -45, "max": 45},
                "knee": {"min": 0, "max": 120},
                "ankle": {"min": -30, "max": 30},
                "shoulder": {"min": -90, "max": 90},
                "elbow": {"min": 0, "max": 135},
                "wrist": {"min": -90, "max": 90},
            },
            "link_lengths": {
                "thigh": 0.3,      # 大腿长度 (米)
                "shin": 0.3,       # 小腿长度 (米)
                "foot": 0.1,       # 脚长 (米)
                "upper_arm": 0.25, # 上臂长度 (米)
                "forearm": 0.25,   # 前臂长度 (米)
                "hand": 0.1,       # 手长 (米)
            },
            "mass_properties": {
                "total_mass": 50.0,  # 总质量 (kg)
                "center_of_mass": [0, 0, 0.5],  # 质心位置
            }
        }
    
    async def process_multimodal_input(self, 
                                     voice_input: Optional[str] = None,
                                     vision_input: Optional[Dict[str, Any]] = None,
                                     sensor_input: Optional[Dict[str, Any]] = None) -> List[MotionCommand]:
        """处理多模态输入，生成运动命令
        
        参数:
            voice_input: 语音输入文本
            vision_input: 视觉输入数据
            sensor_input: 传感器输入数据
            
        返回:
            运动命令列表
        """
        start_time = time.time()
        
        try:
            motion_commands = []
            
            # 处理语音输入
            if voice_input:
                voice_commands = await self._process_voice_input(voice_input)
                motion_commands.extend(voice_commands)
            
            # 处理视觉输入
            if vision_input:
                vision_commands = await self._process_vision_input(vision_input)
                motion_commands.extend(vision_commands)
            
            # 处理传感器输入
            if sensor_input:
                sensor_commands = await self._process_sensor_input(sensor_input)
                motion_commands.extend(sensor_commands)
            
            # 按优先级排序
            motion_commands.sort(key=lambda cmd: cmd.priority, reverse=True)
            
            # 更新性能指标
            processing_time = time.time() - start_time
            self._update_performance_metrics(len(motion_commands), processing_time)
            
            logger.info(f"生成 {len(motion_commands)} 个运动命令，处理时间: {processing_time:.3f}秒")
            return motion_commands
            
        except Exception as e:
            logger.error(f"处理多模态输入失败: {e}")
            return []
    
    async def _process_voice_input(self, voice_text: str) -> List[MotionCommand]:
        """处理语音输入"""
        commands = []
        
        try:
            # 转换为小写以便匹配
            voice_lower = voice_text.lower()
            
            # 查找匹配的语音命令
            for pattern, action_config in self.mapping_rules["voice_to_action"].items():
                if pattern in voice_lower:
                    motion_type = MotionType(action_config["motion_type"])
                    params = action_config["params"]
                    
                    # 创建运动命令
                    command = MotionCommand(
                        motion_type=motion_type,
                        target=params,
                        constraints={"velocity_limit": self.planning_params["max_velocity"]},
                        control_mode=ControlMode.POSITION_CONTROL,
                        priority=5  # 中等优先级
                    )
                    commands.append(command)
            
            return commands
            
        except Exception as e:
            logger.error(f"处理语音输入失败: {e}")
            return []
    
    async def _process_vision_input(self, vision_data: Dict[str, Any]) -> List[MotionCommand]:
        """处理视觉输入"""
        commands = []
        
        try:
            # 提取视觉信息
            detection_result = vision_data.get("detection", "")
            
            # 查找匹配的视觉规则
            for pattern, action_config in self.mapping_rules["vision_to_action"].items():
                if pattern in detection_result:
                    motion_type = MotionType(action_config["motion_type"])
                    params = action_config["params"]
                    
                    # 创建运动命令
                    command = MotionCommand(
                        motion_type=motion_type,
                        target=params,
                        constraints={"safety_margin": self.planning_params["safety_margin"]},
                        control_mode=ControlMode.POSITION_CONTROL,
                        priority=7  # 较高优先级
                    )
                    commands.append(command)
            
            return commands
            
        except Exception as e:
            logger.error(f"处理视觉输入失败: {e}")
            return []
    
    async def _process_sensor_input(self, sensor_data: Dict[str, Any]) -> List[MotionCommand]:
        """处理传感器输入"""
        commands = []
        
        try:
            # 提取传感器信息
            sensor_status = sensor_data.get("status", "")
            
            # 查找匹配的传感器规则
            for pattern, action_config in self.mapping_rules["sensor_to_action"].items():
                if pattern in sensor_status:
                    motion_type = MotionType(action_config["motion_type"])
                    params = action_config["params"]
                    
                    # 创建运动命令
                    command = MotionCommand(
                        motion_type=motion_type,
                        target=params,
                        constraints={"emergency_response": True},
                        control_mode=ControlMode.IMPEDANCE_CONTROL,
                        priority=9  # 高优先级
                    )
                    commands.append(command)
            
            return commands
            
        except Exception as e:
            logger.error(f"处理传感器输入失败: {e}")
            return []
    
    async def plan_trajectory(self, command: MotionCommand) -> Optional[JointTrajectory]:
        """规划关节轨迹"""
        try:
            # 检查缓存
            cache_key = self._create_cache_key(command)
            if cache_key in self.trajectory_cache:
                logger.debug(f"从缓存获取轨迹: {cache_key}")
                return self.trajectory_cache[cache_key]
            
            # 根据运动类型规划轨迹
            trajectory = None
            if command.motion_type == MotionType.WALKING:
                trajectory = await self._plan_walking_trajectory(command)
            elif command.motion_type == MotionType.TURNING:
                trajectory = await self._plan_turning_trajectory(command)
            elif command.motion_type == MotionType.BALANCING:
                trajectory = await self._plan_balancing_trajectory(command)
            elif command.motion_type == MotionType.GESTURE:
                trajectory = await self._plan_gesture_trajectory(command)
            
            # 缓存轨迹
            if trajectory:
                self.trajectory_cache[cache_key] = trajectory
            
            return trajectory
            
        except Exception as e:
            logger.error(f"规划轨迹失败: {e}")
            return None
    
    async def _plan_walking_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划行走轨迹"""
        # 简化实现 - 实际系统应该使用运动学求解器
        direction = command.target.get("direction", "forward")
        steps = command.target.get("steps", 1)
        
        # 生成腿部关节轨迹
        joint_names = ["left_hip", "left_knee", "left_ankle", 
                      "right_hip", "right_knee", "right_ankle"]
        
        positions = []
        times = []
        
        # 生成步态周期
        for step in range(steps):
            for phase in np.linspace(0, 2*np.pi, 20):
                # 简化正弦步态
                left_hip = 15 * np.sin(phase)
                left_knee = 30 * np.sin(phase + np.pi/2)
                left_ankle = 10 * np.sin(phase + np.pi)
                
                right_hip = 15 * np.sin(phase + np.pi)
                right_knee = 30 * np.sin(phase + 3*np.pi/2)
                right_ankle = 10 * np.sin(phase)
                
                positions.append([left_hip, left_knee, left_ankle, 
                                right_hip, right_knee, right_ankle])
                times.append(step * 2.0 + phase * 0.1)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    async def _plan_turning_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划转向轨迹"""
        direction = command.target.get("direction", "left")
        angle = command.target.get("angle", 90)
        
        # 简化实现
        joint_names = ["hip_yaw"]
        positions = []
        times = []
        
        for t in np.linspace(0, 1.0, 20):
            if direction == "left":
                pos = -angle * t
            else:
                pos = angle * t
            
            positions.append([pos])
            times.append(t * 2.0)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    async def _plan_balancing_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划平衡轨迹"""
        action = command.target.get("action", "stand")
        
        # 根据动作类型生成轨迹
        if action == "stand":
            return await self._generate_standing_trajectory()
        elif action == "sit":
            return await self._generate_sitting_trajectory()
        elif action == "stop":
            return await self._generate_stopping_trajectory()
        elif action == "recover":
            return await self._generate_recovery_trajectory()
        else:
            return await self._generate_default_trajectory()
    
    async def _plan_gesture_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划手势轨迹"""
        gesture = command.target.get("gesture", "wave")
        
        if gesture == "wave":
            return await self._generate_wave_trajectory()
        elif gesture == "nod":
            return await self._generate_nod_trajectory()
        else:
            return await self._generate_default_trajectory()
    
    async def _generate_standing_trajectory(self) -> JointTrajectory:
        """生成站立轨迹"""
        joint_names = ["left_hip", "left_knee", "left_ankle",
                      "right_hip", "right_knee", "right_ankle"]
        
        positions = [[0, 0, 0, 0, 0, 0]]
        times = [0.0]
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    async def _generate_wave_trajectory(self) -> JointTrajectory:
        """生成挥手轨迹"""
        joint_names = ["right_shoulder", "right_elbow", "right_wrist"]
        
        positions = []
        times = []
        
        for t in np.linspace(0, 2*np.pi, 20):
            shoulder = 30 * np.sin(t)
            elbow = 60 * np.sin(t + np.pi/2)
            wrist = 30 * np.sin(t + np.pi)
            
            positions.append([shoulder, elbow, wrist])
            times.append(t * 0.1)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    def _create_cache_key(self, command: MotionCommand) -> str:
        """创建缓存键"""
        return f"{command.motion_type.value}_{hash(str(command.target))}"
    
    def _update_performance_metrics(self, num_commands: int, processing_time: float):
        """更新性能指标"""
        self.performance_metrics['total_commands'] += num_commands
        self.performance_metrics['successful_commands'] += num_commands  # 简化
        self.performance_metrics['total_processing_time'] += processing_time
        
        if self.performance_metrics['total_commands'] > 0:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['total_commands']
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    async def shutdown(self):
        """关闭运动控制系统"""
        if not self.initialized:
            return
        
        logger.info("关闭运动控制系统...")
        
        # 清理缓存
        self.trajectory_cache.clear()
        
        # 注销组件
        try:
            await self.communication.unregister_component("motion_control")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        self.initialized = False
        logger.info("运动控制系统已关闭")