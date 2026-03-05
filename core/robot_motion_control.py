"""
多模态机器人运动控制模块

提供视觉-运动协同控制、语音-动作映射、触觉反馈-动作调整等高级运动控制功能。
支持多模态输入到机器人动作的实时映射和转换。
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotMotionControl")

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

class MultimodalMotionController:
    """多模态运动控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化多模态运动控制器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
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
        
        # 初始化运动学模型
        self._initialize_kinematics()
        
        logger.info("多模态运动控制器初始化完成")
    
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
    
    def process_multimodal_input(self, 
                                voice_input: Optional[str] = None,
                                vision_input: Optional[Dict[str, Any]] = None,
                                sensor_input: Optional[Dict[str, Any]] = None) -> List[MotionCommand]:
        """处理多模态输入，生成运动命令
        
        Args:
            voice_input: 语音输入文本
            vision_input: 视觉输入数据
            sensor_input: 传感器输入数据
            
        Returns:
            运动命令列表
        """
        motion_commands = []
        
        # 处理语音输入
        if voice_input:
            voice_commands = self._process_voice_input(voice_input)
            motion_commands.extend(voice_commands)
        
        # 处理视觉输入
        if vision_input:
            vision_commands = self._process_vision_input(vision_input)
            motion_commands.extend(vision_commands)
        
        # 处理传感器输入
        if sensor_input:
            sensor_commands = self._process_sensor_input(sensor_input)
            motion_commands.extend(sensor_commands)
        
        # 按优先级排序
        motion_commands.sort(key=lambda cmd: cmd.priority, reverse=True)
        
        logger.info(f"生成 {len(motion_commands)} 个运动命令")
        return motion_commands
    
    def _process_voice_input(self, voice_text: str) -> List[MotionCommand]:
        """处理语音输入"""
        commands = []
        
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
                    priority=5,  # 中等优先级
                    duration=2.0  # 默认2秒
                )
                commands.append(command)
                logger.info(f"语音命令匹配: '{voice_text}' -> {motion_type.value}")
                break
        
        return commands
    
    def _process_vision_input(self, vision_data: Dict[str, Any]) -> List[MotionCommand]:
        """处理视觉输入"""
        commands = []
        
        # 提取视觉信息
        vision_description = vision_data.get("description", "")
        object_info = vision_data.get("objects", [])
        obstacle_info = vision_data.get("obstacles", [])
        
        # 检查障碍物
        if obstacle_info:
            command = MotionCommand(
                motion_type=MotionType.NAVIGATION,
                target={"action": "avoid", "obstacles": obstacle_info},
                constraints={"safety_distance": 0.5},
                control_mode=ControlMode.VELOCITY_CONTROL,
                priority=8,  # 高优先级（安全相关）
                duration=3.0
            )
            commands.append(command)
        
        # 检查目标物体
        if object_info and "目标" in vision_description:
            command = MotionCommand(
                motion_type=MotionType.NAVIGATION,
                target={"action": "approach", "objects": object_info},
                constraints={"precision": 0.05},
                control_mode=ControlMode.POSITION_CONTROL,
                priority=6,
                duration=2.0
            )
            commands.append(command)
        
        # 检查人脸或手势
        if "人脸" in vision_description or "手势" in vision_description:
            command = MotionCommand(
                motion_type=MotionType.GESTURE,
                target={"action": "respond", "interaction": "social"},
                constraints={"naturalness": 0.8},
                control_mode=ControlMode.IMPEDANCE_CONTROL,
                priority=4,
                duration=1.5
            )
            commands.append(command)
        
        return commands
    
    def _process_sensor_input(self, sensor_data: Dict[str, Any]) -> List[MotionCommand]:
        """处理传感器输入"""
        commands = []
        
        # 检查平衡状态
        balance_status = sensor_data.get("balance", {})
        if balance_status.get("unstable", False):
            command = MotionCommand(
                motion_type=MotionType.BALANCING,
                target={"action": "recover", "perturbation": balance_status.get("magnitude", 0.0)},
                constraints={"stability_margin": 0.2},
                control_mode=ControlMode.IMPEDANCE_CONTROL,
                priority=9,  # 非常高优先级（安全关键）
                duration=1.0
            )
            commands.append(command)
        
        # 检查接触力
        contact_forces = sensor_data.get("contact_forces", {})
        if any(force > 50.0 for force in contact_forces.values() if isinstance(force, (int, float))):
            command = MotionCommand(
                motion_type=MotionType.MANIPULATION,
                target={"action": "comply", "force_reduction": 0.5},
                constraints={"max_force": 30.0},
                control_mode=ControlMode.ADMITTANCE_CONTROL,
                priority=7,
                duration=0.5
            )
            commands.append(command)
        
        return commands
    
    def plan_trajectory(self, motion_command: MotionCommand) -> Optional[JointTrajectory]:
        """规划运动轨迹
        
        Args:
            motion_command: 运动命令
            
        Returns:
            关节轨迹，如果规划失败则返回None
        """
        try:
            # 检查缓存
            cache_key = self._get_cache_key(motion_command)
            if cache_key in self.trajectory_cache:
                logger.info(f"使用缓存的轨迹: {cache_key}")
                return self.trajectory_cache[cache_key]
            
            # 根据运动类型规划轨迹
            if motion_command.motion_type == MotionType.WALKING:
                trajectory = self._plan_walking_trajectory(motion_command)
            elif motion_command.motion_type == MotionType.TURNING:
                trajectory = self._plan_turning_trajectory(motion_command)
            elif motion_command.motion_type == MotionType.BALANCING:
                trajectory = self._plan_balancing_trajectory(motion_command)
            elif motion_command.motion_type == MotionType.GESTURE:
                trajectory = self._plan_gesture_trajectory(motion_command)
            elif motion_command.motion_type == MotionType.NAVIGATION:
                trajectory = self._plan_navigation_trajectory(motion_command)
            else:
                trajectory = self._plan_general_trajectory(motion_command)
            
            if trajectory:
                # 缓存轨迹
                self.trajectory_cache[cache_key] = trajectory
                logger.info(f"轨迹规划成功: {motion_command.motion_type.value}, {len(trajectory.positions)} 个路径点")
            
            return trajectory
            
        except Exception as e:
            logger.error(f"轨迹规划失败: {e}")
            return None
    
    def _plan_walking_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划行走轨迹"""
        direction = command.target.get("direction", "forward")
        steps = command.target.get("steps", 1)
        
        # 简化的行走轨迹
        joint_names = [
            "left_hip", "left_knee", "left_ankle",
            "right_hip", "right_knee", "right_ankle"
        ]
        
        # 生成步态周期
        positions = []
        times = []
        
        step_duration = command.duration / steps
        for step in range(steps):
            for phase in range(5):  # 5个相位
                time_point = step * step_duration + phase * (step_duration / 5)
                phase_ratio = phase / 5
                
                # 简化的步态模式
                if direction == "forward":
                    left_leg = [10 * phase_ratio, 30 * phase_ratio, -5 * phase_ratio]
                    right_leg = [10 * (1 - phase_ratio), 30 * (1 - phase_ratio), -5 * (1 - phase_ratio)]
                else:  # backward
                    left_leg = [-10 * phase_ratio, 20 * phase_ratio, -5 * phase_ratio]
                    right_leg = [-10 * (1 - phase_ratio), 20 * (1 - phase_ratio), -5 * (1 - phase_ratio)]
                
                positions.append(left_leg + right_leg)
                times.append(time_point)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    def _plan_turning_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划转向轨迹"""
        direction = command.target.get("direction", "left")
        angle = command.target.get("angle", 90)
        
        joint_names = [
            "left_hip", "left_knee", "left_ankle",
            "right_hip", "right_knee", "right_ankle"
        ]
        
        positions = []
        times = []
        
        # 简化的转向轨迹
        for i in range(10):
            time_point = i * (command.duration / 10)
            ratio = i / 10
            
            if direction == "left":
                left_leg = [15 * ratio, 25, -5]
                right_leg = [-15 * ratio, 25, 5 * ratio]
            else:  # right
                left_leg = [-15 * ratio, 25, 5 * ratio]
                right_leg = [15 * ratio, 25, -5]
            
            positions.append(left_leg + right_leg)
            times.append(time_point)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    def _plan_balancing_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划平衡恢复轨迹"""
        action = command.target.get("action", "stand")
        
        joint_names = [
            "left_hip", "left_knee", "left_ankle",
            "right_hip", "right_knee", "right_ankle",
            "torso_twist", "torso_bend"
        ]
        
        positions = []
        times = []
        
        if action == "stand":
            # 站立姿势
            for i in range(5):
                time_point = i * (command.duration / 5)
                ratio = i / 5
                
                base_position = [0, 0, 0, 0, 0, 0, 0, 0]
                target_position = [5, 10, -2, 5, 10, -2, 0, 0]
                
                # 插值
                position = [base + (target - base) * ratio 
                           for base, target in zip(base_position, target_position)]
                positions.append(position)
                times.append(time_point)
        
        elif action == "recover":
            # 平衡恢复
            for i in range(8):
                time_point = i * (command.duration / 8)
                ratio = i / 8
                
                # 振荡恢复模式
                oscillation = np.sin(ratio * np.pi * 2) * 0.1
                position = [
                    5 + oscillation, 10, -2,
                    5 - oscillation, 10, -2,
                    oscillation * 5, oscillation * 2
                ]
                positions.append(position)
                times.append(time_point)
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    def _plan_gesture_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划手势轨迹"""
        gesture = command.target.get("gesture", "wave")
        
        if gesture == "wave":
            joint_names = ["right_shoulder", "right_elbow", "right_wrist"]
            positions = []
            times = []
            
            for i in range(10):
                time_point = i * (command.duration / 10)
                ratio = i / 10
                
                # 挥手动作
                position = [
                    30 + 30 * np.sin(ratio * np.pi * 2),  # 肩部
                    60 + 20 * np.sin(ratio * np.pi * 2 + 0.5),  # 肘部
                    0 + 45 * np.sin(ratio * np.pi * 2 + 1.0)   # 腕部
                ]
                positions.append(position)
                times.append(time_point)
            
            return JointTrajectory(
                joint_names=joint_names,
                positions=positions,
                times=times
            )
        
        # 默认返回空轨迹
        return JointTrajectory(joint_names=[], positions=[], times=[])
    
    def _plan_navigation_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划导航轨迹"""
        # 简化的导航轨迹
        return self._plan_walking_trajectory(command)
    
    def _plan_general_trajectory(self, command: MotionCommand) -> JointTrajectory:
        """规划通用轨迹"""
        # 返回默认的站立姿势
        joint_names = [
            "left_hip", "left_knee", "left_ankle",
            "right_hip", "right_knee", "right_ankle"
        ]
        
        positions = [[5, 10, -2, 5, 10, -2]]
        times = [0.0]
        
        return JointTrajectory(
            joint_names=joint_names,
            positions=positions,
            times=times
        )
    
    def _get_cache_key(self, command: MotionCommand) -> str:
        """生成缓存键"""
        return f"{command.motion_type.value}_{hash(str(command.target))}_{command.duration}"
    
    def execute_trajectory(self, trajectory: JointTrajectory, 
                          real_time: bool = True) -> Dict[str, Any]:
        """执行轨迹
        
        Args:
            trajectory: 关节轨迹
            real_time: 是否实时执行
            
        Returns:
            执行结果
        """
        try:
            logger.info(f"开始执行轨迹: {len(trajectory.positions)} 个路径点")
            
            if real_time:
                # 实时执行（模拟）
                for i, (position, time_point) in enumerate(zip(trajectory.positions, trajectory.times)):
                    logger.debug(f"路径点 {i+1}/{len(trajectory.positions)}: 位置={position[:3]}..., 时间={time_point:.2f}s")
                    # 这里应该发送到硬件控制器
                    time.sleep(0.01)  # 模拟执行延迟
            
            return {
                "success": True,
                "executed_points": len(trajectory.positions),
                "duration": trajectory.times[-1] if trajectory.times else 0,
                "message": "轨迹执行完成"
            }
            
        except Exception as e:
            logger.error(f"轨迹执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "轨迹执行失败"
            }
    
    def emergency_stop(self) -> Dict[str, Any]:
        """紧急停止"""
        logger.warning("执行紧急停止")
        
        # 发送停止命令到所有关节
        # 这里应该实现硬件的紧急停止
        
        return {
            "success": True,
            "message": "紧急停止已执行",
            "timestamp": time.time()
        }


# 全局运动控制器实例
_motion_controller = None

def get_motion_controller(config: Optional[Dict[str, Any]] = None) -> MultimodalMotionController:
    """获取运动控制器实例（单例模式）"""
    global _motion_controller
    if _motion_controller is None:
        _motion_controller = MultimodalMotionController(config)
    return _motion_controller


# 测试函数
def test_motion_controller():
    """测试运动控制器"""
    print("=" * 80)
    print("测试多模态运动控制器")
    print("=" * 80)
    
    controller = get_motion_controller()
    
    # 测试语音输入处理
    print("\n1. 测试语音输入处理:")
    voice_commands = ["前进三步", "左转", "挥手", "停止"]
    for voice in voice_commands:
        commands = controller.process_multimodal_input(voice_input=voice)
        print(f"  语音: '{voice}' -> {len(commands)} 个运动命令")
        for cmd in commands:
            print(f"    - {cmd.motion_type.value}: {cmd.target}")
    
    # 测试视觉输入处理
    print("\n2. 测试视觉输入处理:")
    vision_input = {
        "description": "检测到前方障碍物",
        "obstacles": [{"position": [1.0, 0.5, 0], "size": [0.3, 0.3, 0.5]}]
    }
    commands = controller.process_multimodal_input(vision_input=vision_input)
    print(f"  视觉输入 -> {len(commands)} 个运动命令")
    
    # 测试轨迹规划
    print("\n3. 测试轨迹规划:")
    if commands:
        trajectory = controller.plan_trajectory(commands[0])
        if trajectory:
            print(f"  规划成功: {len(trajectory.positions)} 个路径点")
            print(f"  关节数量: {len(trajectory.joint_names)}")
    
    # 测试执行
    print("\n4. 测试轨迹执行:")
    if commands and trajectory:
        result = controller.execute_trajectory(trajectory, real_time=False)
        print(f"  执行结果: {result['success']}")
        print(f"  消息: {result['message']}")
    
    print("\n✅ 运动控制器测试完成")


if __name__ == "__main__":
    test_motion_controller()