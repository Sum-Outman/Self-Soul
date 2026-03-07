"""
双足行走步态系统

为人形机器人提供双足行走功能，包括步态生成、脚步规划、重心转移和动态平衡。
基于经典的零力矩点（ZMP）控制和倒立摆模型。
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


class GaitPhase(Enum):
    """步态相位枚举"""
    DOUBLE_SUPPORT = "double_support"      # 双足支撑期
    LEFT_SWING = "left_swing"              # 左腿摆动期
    RIGHT_SWING = "right_swing"            # 右腿摆动期
    TRANSITION = "transition"              # 过渡期


class WalkingDirection(Enum):
    """行走方向枚举"""
    FORWARD = "forward"      # 前进
    BACKWARD = "backward"    # 后退
    LEFT = "left"            # 左移
    RIGHT = "right"          # 右移
    TURN_LEFT = "turn_left"  # 左转
    TURN_RIGHT = "turn_right" # 右转


@dataclass
class Footstep:
    """脚步数据结构"""
    foot: str                     # 左脚或右脚
    position: np.ndarray          # 脚的位置 [x, y, z]
    orientation: np.ndarray       # 脚的朝向 [roll, pitch, yaw]
    timestamp: float              # 时间戳
    duration: float               # 持续时间
    contact: bool = True          # 是否接触地面


@dataclass
class GaitParameters:
    """步态参数"""
    step_length: float = 0.3           # 步长 (米)
    step_width: float = 0.2            # 步宽 (米)
    step_height: float = 0.05          # 步高 (米)
    step_duration: float = 0.8         # 单步持续时间 (秒)
    double_support_ratio: float = 0.2  # 双足支撑期比例
    com_height: float = 0.8            # 质心高度 (米)
    max_step_length: float = 0.5       # 最大步长 (米)
    max_step_width: float = 0.3        # 最大步宽 (米)
    max_turn_angle: float = 30.0       # 最大转弯角度 (度)
    stability_margin: float = 0.05     # 稳定裕度 (米)


class WalkingGaitSystem:
    """双足行走步态系统"""
    
    def __init__(self, communication):
        """
        初始化双足行走步态系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 步态状态
        self.gait_phase = GaitPhase.DOUBLE_SUPPORT
        self.walking_direction = WalkingDirection.FORWARD
        self.is_walking = False
        
        # 步态参数
        self.params = GaitParameters()
        
        # 脚步序列
        self.footstep_sequence: List[Footstep] = []
        self.current_footstep_index = 0
        
        # 轨迹生成器
        self.trajectory_generator = TrajectoryGenerator()
        
        # 零力矩点 (ZMP) 规划器
        self.zmp_planner = ZMPPlanner()
        
        # 倒立摆模型
        self.pendulum_model = InvertedPendulumModel()
        
        # 性能指标
        self.performance_metrics = {
            'total_steps': 0,
            'walking_distance': 0.0,
            'walking_time': 0.0,
            'step_success_rate': 0.0,
            'balance_violations': 0,
            'average_speed': 0.0
        }
        
        # 实时状态
        self.current_state = {
            'com_position': np.array([0.0, 0.0, self.params.com_height]),
            'com_velocity': np.array([0.0, 0.0, 0.0]),
            'left_foot_position': np.array([0.0, self.params.step_width/2, 0.0]),
            'right_foot_position': np.array([0.0, -self.params.step_width/2, 0.0]),
            'zmp_reference': np.array([0.0, 0.0]),
            'phase_progress': 0.0
        }
        
        # 行走模式
        self.walking_mode = 'normal'  # normal, cautious, fast, adaptive
        
        logger.info("双足行走步态系统已初始化")
    
    async def initialize(self):
        """初始化双足行走步态系统"""
        if self.initialized:
            return
        
        logger.info("初始化双足行走步态系统...")
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="walking_gait",
            component_type="humanoid"
        )
        
        # 初始化子模块
        self.trajectory_generator.initialize(self.params)
        self.zmp_planner.initialize(self.params)
        self.pendulum_model.initialize(self.params.com_height)
        
        self.initialized = True
        logger.info("双足行走步态系统初始化完成")
    
    async def start_walking(self, direction: WalkingDirection, distance: Optional[float] = None):
        """
        开始行走。
        
        参数:
            direction: 行走方向
            distance: 行走距离（如果为None则持续行走）
        """
        try:
            if self.is_walking:
                logger.warning("已经在行走中")
                return {'success': False, 'message': '已经在行走中'}
            
            logger.info(f"开始{direction.value}行走" + (f"，距离: {distance}米" if distance else ""))
            
            self.walking_direction = direction
            self.is_walking = True
            self.gait_phase = GaitPhase.DOUBLE_SUPPORT
            self.current_footstep_index = 0
            
            # 生成脚步序列
            await self._generate_footstep_sequence(direction, distance)
            
            # 初始化轨迹
            await self._initialize_trajectory()
            
            # 更新性能指标
            self.performance_metrics['walking_time'] = 0.0
            
            return {
                'success': True,
                'message': f'开始{direction.value}行走',
                'footstep_count': len(self.footstep_sequence),
                'estimated_duration': len(self.footstep_sequence) * self.params.step_duration
            }
            
        except Exception as e:
            logger.error(f"开始行走失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def stop_walking(self):
        """停止行走"""
        try:
            if not self.is_walking:
                return {'success': False, 'message': '未在行走中'}
            
            logger.info("停止行走")
            
            self.is_walking = False
            
            # 平滑停止
            await self._smooth_stop()
            
            # 更新性能指标
            if self.performance_metrics['total_steps'] > 0:
                self.performance_metrics['step_success_rate'] = (
                    (self.performance_metrics['step_success_rate'] * 
                     (self.performance_metrics['total_steps'] - 1) + 1) / 
                    self.performance_metrics['total_steps']
                )
            
            return {'success': True, 'message': '行走已停止'}
            
        except Exception as e:
            logger.error(f"停止行走失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def update_walking(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新行走状态。
        
        参数:
            sensor_data: 传感器数据
            
        返回:
            行走控制输出
        """
        try:
            if not self.is_walking:
                return {'success': False, 'message': '未在行走中'}
            
            start_time = time.time()
            
            # 更新步态相位
            await self._update_gait_phase()
            
            # 更新当前状态
            await self._update_current_state(sensor_data)
            
            # 生成控制输出
            control_output = await self._generate_control_output()
            
            # 更新性能指标
            self.performance_metrics['walking_time'] += time.time() - start_time
            self.performance_metrics['walking_distance'] += self._calculate_step_distance()
            
            if self.gait_phase == GaitPhase.DOUBLE_SUPPORT:
                self.performance_metrics['total_steps'] += 1
            
            return control_output
            
        except Exception as e:
            logger.error(f"更新行走状态失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_footstep_sequence(self, direction: WalkingDirection, distance: Optional[float]):
        """生成脚步序列"""
        try:
            self.footstep_sequence = []
            
            # 根据方向生成脚步
            if direction == WalkingDirection.FORWARD:
                steps = self._generate_forward_steps(distance)
            elif direction == WalkingDirection.BACKWARD:
                steps = self._generate_backward_steps(distance)
            elif direction == WalkingDirection.LEFT:
                steps = self._generate_lateral_steps('left', distance)
            elif direction == WalkingDirection.RIGHT:
                steps = self._generate_lateral_steps('right', distance)
            elif direction == WalkingDirection.TURN_LEFT:
                steps = self._generate_turn_steps('left', distance)
            elif direction == WalkingDirection.TURN_RIGHT:
                steps = self._generate_turn_steps('right', distance)
            else:
                steps = []
            
            self.footstep_sequence = steps
            logger.info(f"生成了 {len(steps)} 个脚步")
            
        except Exception as e:
            logger.error(f"生成脚步序列失败: {e}")
            self.footstep_sequence = []
    
    def _generate_forward_steps(self, distance: Optional[float]) -> List[Footstep]:
        """生成前进脚步"""
        steps = []
        current_time = time.time()
        
        # 初始脚步
        left_foot = Footstep(
            foot='left',
            position=np.array([0.0, self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        right_foot = Footstep(
            foot='right',
            position=np.array([0.0, -self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        steps.extend([left_foot, right_foot])
        
        # 生成前进脚步
        if distance is None:
            # 持续行走，生成10步
            step_count = 10
        else:
            # 根据距离计算步数
            step_count = int(np.ceil(distance / self.params.step_length))
            step_count = max(2, step_count)  # 至少2步
        
        current_pos_left = left_foot.position.copy()
        current_pos_right = right_foot.position.copy()
        
        for i in range(step_count):
            # 交替脚步
            if i % 2 == 0:
                # 右脚步进
                current_pos_right[0] += self.params.step_length
                step = Footstep(
                    foot='right',
                    position=current_pos_right.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            else:
                # 左脚步进
                current_pos_left[0] += self.params.step_length
                step = Footstep(
                    foot='left',
                    position=current_pos_left.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            
            steps.append(step)
        
        return steps
    
    def _generate_backward_steps(self, distance: Optional[float]) -> List[Footstep]:
        """生成后退脚步"""
        steps = []
        current_time = time.time()
        
        # 初始脚步（与前进相同）
        left_foot = Footstep(
            foot='left',
            position=np.array([0.0, self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        right_foot = Footstep(
            foot='right',
            position=np.array([0.0, -self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        steps.extend([left_foot, right_foot])
        
        # 生成后退脚步（负方向）
        if distance is None:
            step_count = 10
        else:
            step_count = int(np.ceil(abs(distance) / self.params.step_length))
            step_count = max(2, step_count)
        
        current_pos_left = left_foot.position.copy()
        current_pos_right = right_foot.position.copy()
        
        for i in range(step_count):
            if i % 2 == 0:
                # 右脚步退
                current_pos_right[0] -= self.params.step_length
                step = Footstep(
                    foot='right',
                    position=current_pos_right.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            else:
                # 左脚步退
                current_pos_left[0] -= self.params.step_length
                step = Footstep(
                    foot='left',
                    position=current_pos_left.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            
            steps.append(step)
        
        return steps
    
    def _generate_lateral_steps(self, direction: str, distance: Optional[float]) -> List[Footstep]:
        """生成侧移脚步"""
        steps = []
        current_time = time.time()
        
        # 初始脚步
        left_foot = Footstep(
            foot='left',
            position=np.array([0.0, self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        right_foot = Footstep(
            foot='right',
            position=np.array([0.0, -self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        steps.extend([left_foot, right_foot])
        
        # 生成侧移脚步
        if distance is None:
            step_count = 10
        else:
            step_count = int(np.ceil(abs(distance) / self.params.step_width))
            step_count = max(2, step_count)
        
        current_pos_left = left_foot.position.copy()
        current_pos_right = right_foot.position.copy()
        
        sign = 1 if direction == 'right' else -1
        
        for i in range(step_count):
            if i % 2 == 0:
                # 移动右脚
                current_pos_right[1] += sign * self.params.step_width
                step = Footstep(
                    foot='right',
                    position=current_pos_right.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            else:
                # 移动左脚
                current_pos_left[1] += sign * self.params.step_width
                step = Footstep(
                    foot='left',
                    position=current_pos_left.copy(),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            
            steps.append(step)
        
        return steps
    
    def _generate_turn_steps(self, direction: str, angle: Optional[float]) -> List[Footstep]:
        """生成转弯脚步"""
        steps = []
        current_time = time.time()
        
        # 初始脚步
        left_foot = Footstep(
            foot='left',
            position=np.array([0.0, self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        right_foot = Footstep(
            foot='right',
            position=np.array([0.0, -self.params.step_width/2, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        steps.extend([left_foot, right_foot])
        
        # 生成转弯脚步
        if angle is None:
            # 默认转90度
            angle = 90.0
        
        turn_angle_per_step = 15.0  # 每步转角（度）
        step_count = int(np.ceil(abs(angle) / turn_angle_per_step))
        step_count = max(2, step_count)
        
        current_yaw_left = 0.0
        current_yaw_right = 0.0
        sign = 1 if direction == 'right' else -1
        
        for i in range(step_count):
            turn_angle = sign * min(turn_angle_per_step, abs(angle) - i * turn_angle_per_step)
            
            if i % 2 == 0:
                # 转动右脚
                current_yaw_right += turn_angle
                step = Footstep(
                    foot='right',
                    position=right_foot.position.copy(),
                    orientation=np.array([0.0, 0.0, math.radians(current_yaw_right)]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            else:
                # 转动左脚
                current_yaw_left += turn_angle
                step = Footstep(
                    foot='left',
                    position=left_foot.position.copy(),
                    orientation=np.array([0.0, 0.0, math.radians(current_yaw_left)]),
                    timestamp=current_time + (i+1) * self.params.step_duration,
                    duration=self.params.step_duration
                )
            
            steps.append(step)
        
        return steps
    
    async def _initialize_trajectory(self):
        """初始化轨迹"""
        try:
            if not self.footstep_sequence:
                return
            
            # 设置初始状态
            self.current_state['left_foot_position'] = self.footstep_sequence[0].position.copy()
            self.current_state['right_foot_position'] = self.footstep_sequence[1].position.copy()
            
            # 初始化ZMP轨迹
            await self.zmp_planner.generate_trajectory(self.footstep_sequence)
            
            # 初始化COM轨迹
            await self.pendulum_model.generate_com_trajectory(
                self.zmp_planner.zmp_trajectory,
                self.params.step_duration
            )
            
        except Exception as e:
            logger.error(f"初始化轨迹失败: {e}")
    
    async def _update_gait_phase(self):
        """更新步态相位"""
        try:
            if not self.footstep_sequence:
                return
            
            # 计算相位进度
            current_time = time.time()
            current_step = self.footstep_sequence[self.current_footstep_index]
            time_elapsed = current_time - current_step.timestamp
            
            if time_elapsed < 0:
                # 还未开始当前脚步
                self.gait_phase = GaitPhase.DOUBLE_SUPPORT
                self.current_state['phase_progress'] = 0.0
                return
            
            phase_progress = time_elapsed / current_step.duration
            
            if phase_progress < self.params.double_support_ratio:
                # 双足支撑期
                self.gait_phase = GaitPhase.DOUBLE_SUPPORT
            elif phase_progress < 1.0:
                # 单腿摆动期
                if current_step.foot == 'left':
                    self.gait_phase = GaitPhase.LEFT_SWING
                else:
                    self.gait_phase = GaitPhase.RIGHT_SWING
            else:
                # 当前脚步完成，进入过渡期
                self.gait_phase = GaitPhase.TRANSITION
                
                # 切换到下一个脚步
                if self.current_footstep_index < len(self.footstep_sequence) - 1:
                    self.current_footstep_index += 1
                else:
                    # 所有脚步完成
                    self.is_walking = False
            
            self.current_state['phase_progress'] = phase_progress
            
        except Exception as e:
            logger.error(f"更新步态相位失败: {e}")
    
    async def _update_current_state(self, sensor_data: Dict[str, Any]):
        """更新当前状态"""
        try:
            # 更新COM状态（简化的模型）
            com_trajectory = self.pendulum_model.get_com_trajectory()
            if com_trajectory:
                progress = self.current_state['phase_progress']
                idx = min(int(progress * len(com_trajectory)), len(com_trajectory) - 1)
                self.current_state['com_position'] = com_trajectory[idx]
            
            # 更新ZMP参考
            zmp_trajectory = self.zmp_planner.get_zmp_trajectory()
            if zmp_trajectory:
                progress = self.current_state['phase_progress']
                idx = min(int(progress * len(zmp_trajectory)), len(zmp_trajectory) - 1)
                self.current_state['zmp_reference'] = zmp_trajectory[idx]
            
            # 更新脚的位置（根据轨迹生成器）
            if self.current_footstep_index < len(self.footstep_sequence):
                current_step = self.footstep_sequence[self.current_footstep_index]
                if self.gait_phase in [GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING]:
                    swing_trajectory = self.trajectory_generator.generate_swing_trajectory(
                        current_step,
                        self.current_state['phase_progress']
                    )
                    
                    if current_step.foot == 'left':
                        self.current_state['left_foot_position'] = swing_trajectory
                    else:
                        self.current_state['right_foot_position'] = swing_trajectory
            
        except Exception as e:
            logger.error(f"更新当前状态失败: {e}")
    
    async def _generate_control_output(self) -> Dict[str, Any]:
        """生成控制输出"""
        try:
            # 生成关节角度
            joint_angles = await self._calculate_joint_angles()
            
            # 生成ZMP控制命令
            zmp_control = await self._calculate_zmp_control()
            
            return {
                'success': True,
                'gait_phase': self.gait_phase.value,
                'phase_progress': self.current_state['phase_progress'],
                'joint_angles': joint_angles,
                'zmp_control': zmp_control,
                'com_position': self.current_state['com_position'].tolist(),
                'zmp_reference': self.current_state['zmp_reference'].tolist(),
                'foot_positions': {
                    'left': self.current_state['left_foot_position'].tolist(),
                    'right': self.current_state['right_foot_position'].tolist()
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"生成控制输出失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'joint_angles': {},
                'zmp_control': {}
            }
    
    async def _calculate_joint_angles(self) -> Dict[str, float]:
        """计算关节角度（简化版本）"""
        try:
            # 简化的逆运动学计算
            # 实际系统应该使用完整的逆运动学求解器
            
            com = self.current_state['com_position']
            left_foot = self.current_state['left_foot_position']
            right_foot = self.current_state['right_foot_position']
            
            # 计算腿部向量
            left_leg_vector = left_foot - com
            right_leg_vector = right_foot - com
            
            # 简化的角度计算（仅用于演示）
            angles = {
                # 左腿
                'left_hip_pitch': float(math.atan2(left_leg_vector[0], left_leg_vector[2])),
                'left_hip_roll': float(math.atan2(left_leg_vector[1], left_leg_vector[2])),
                'left_knee': float(0.5),  # 固定值
                'left_ankle_pitch': float(-0.2),
                'left_ankle_roll': float(0.0),
                
                # 右腿
                'right_hip_pitch': float(math.atan2(right_leg_vector[0], right_leg_vector[2])),
                'right_hip_roll': float(math.atan2(right_leg_vector[1], right_leg_vector[2])),
                'right_knee': float(0.5),
                'right_ankle_pitch': float(-0.2),
                'right_ankle_roll': float(0.0),
            }
            
            return angles
            
        except Exception as e:
            logger.error(f"计算关节角度失败: {e}")
            return {}
    
    async def _calculate_zmp_control(self) -> Dict[str, Any]:
        """计算ZMP控制"""
        try:
            # 简化的ZMP控制（PD控制）
            com = self.current_state['com_position'][:2]
            zmp_ref = self.current_state['zmp_reference']
            
            error = com - zmp_ref
            
            # PD控制参数
            kp = 100.0
            kd = 10.0
            
            # 计算控制力（简化）
            control_force = kp * error + kd * np.array([0.0, 0.0])  # 无速度反馈
            
            return {
                'zmp_reference': zmp_ref.tolist(),
                'com_position': com.tolist(),
                'error': error.tolist(),
                'control_force': control_force.tolist(),
                'stability_margin': self.params.stability_margin
            }
            
        except Exception as e:
            logger.error(f"计算ZMP控制失败: {e}")
            return {}
    
    async def _smooth_stop(self):
        """平滑停止"""
        try:
            # 生成停止脚步
            stop_steps = self._generate_stop_steps()
            
            # 执行停止脚步
            for step in stop_steps:
                # 简化的停止逻辑
                await asyncio.sleep(0.1)
            
            logger.info("平滑停止完成")
            
        except Exception as e:
            logger.error(f"平滑停止失败: {e}")
    
    def _generate_stop_steps(self) -> List[Footstep]:
        """生成停止脚步"""
        # 简化的停止脚步：将COM移回支撑多边形中心
        steps = []
        current_time = time.time()
        
        # 最后一步回到中心
        center_step = Footstep(
            foot='both',
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            timestamp=current_time,
            duration=self.params.step_duration
        )
        
        steps.append(center_step)
        return steps
    
    def _calculate_step_distance(self) -> float:
        """计算步进距离"""
        if self.current_footstep_index == 0:
            return 0.0
        
        try:
            current_step = self.footstep_sequence[self.current_footstep_index]
            previous_step = self.footstep_sequence[self.current_footstep_index - 1]
            
            distance = np.linalg.norm(current_step.position - previous_step.position)
            return float(distance)
            
        except Exception:
            return 0.0
    
    async def get_walking_report(self) -> Dict[str, Any]:
        """获取行走报告"""
        return {
            'is_walking': self.is_walking,
            'gait_phase': self.gait_phase.value,
            'walking_direction': self.walking_direction.value,
            'current_step': self.current_footstep_index,
            'total_steps': len(self.footstep_sequence),
            'performance_metrics': self.performance_metrics,
            'current_state': {
                'com_position': self.current_state['com_position'].tolist(),
                'phase_progress': self.current_state['phase_progress'],
                'foot_positions': {
                    'left': self.current_state['left_foot_position'].tolist(),
                    'right': self.current_state['right_foot_position'].tolist()
                }
            },
            'params': {
                'step_length': self.params.step_length,
                'step_width': self.params.step_width,
                'step_duration': self.params.step_duration,
                'com_height': self.params.com_height
            },
            'timestamp': time.time()
        }
    
    async def adjust_parameters(self, new_params: Dict[str, Any]):
        """调整步态参数"""
        try:
            for key, value in new_params.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
                    logger.info(f"调整步态参数 {key} = {value}")
            
            # 更新子模块参数
            self.trajectory_generator.update_parameters(self.params)
            self.zmp_planner.update_parameters(self.params)
            
            return {'success': True, 'message': '步态参数调整成功'}
            
        except Exception as e:
            logger.error(f"调整步态参数失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def emergency_stop(self):
        """紧急停止行走"""
        try:
            logger.warning("紧急停止行走")
            
            self.is_walking = False
            self.gait_phase = GaitPhase.DOUBLE_SUPPORT
            
            return {
                'success': True,
                'action': 'emergency_stop',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"紧急停止行走失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def shutdown(self):
        """关闭双足行走步态系统"""
        if not self.initialized:
            return
        
        logger.info("关闭双足行走步态系统...")
        
        # 停止行走（如果正在行走）
        if self.is_walking:
            await self.emergency_stop()
        
        # 注销组件
        try:
            await self.communication.unregister_component("walking_gait")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        self.initialized = False
        logger.info("双足行走步态系统已关闭")


# 辅助类（简化实现）

class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self):
        self.params = None
    
    def initialize(self, params):
        self.params = params
    
    def update_parameters(self, params):
        self.params = params
    
    def generate_swing_trajectory(self, footstep: Footstep, progress: float) -> np.ndarray:
        """生成摆动腿轨迹"""
        if progress < 0 or progress > 1:
            return footstep.position
        
        # 简化的摆动轨迹（抛物线）
        start_pos = np.array([0.0, 0.0, 0.0])  # 起始位置
        end_pos = footstep.position
        
        # 计算中间点
        mid_pos = (start_pos + end_pos) / 2
        mid_pos[2] = self.params.step_height  # 抬腿高度
        
        # 抛物线插值
        if progress < 0.5:
            t = progress * 2
            pos = start_pos + (mid_pos - start_pos) * t
            pos[2] = 4 * self.params.step_height * t * (1 - t)
        else:
            t = (progress - 0.5) * 2
            pos = mid_pos + (end_pos - mid_pos) * t
            pos[2] = 4 * self.params.step_height * (1 - t) * t
        
        return pos


class ZMPPlanner:
    """ZMP规划器"""
    
    def __init__(self):
        self.params = None
        self.zmp_trajectory = []
    
    def initialize(self, params):
        self.params = params
    
    def update_parameters(self, params):
        self.params = params
    
    async def generate_trajectory(self, footsteps: List[Footstep]):
        """生成ZMP轨迹"""
        self.zmp_trajectory = []
        
        for i, footstep in enumerate(footsteps):
            # 简化的ZMP轨迹：在支撑脚之间移动
            if footstep.foot == 'left':
                zmp = np.array([footstep.position[0], footstep.position[1] + 0.05])
            elif footstep.foot == 'right':
                zmp = np.array([footstep.position[0], footstep.position[1] - 0.05])
            else:
                zmp = np.array([footstep.position[0], footstep.position[1]])
            
            # 添加多个点以形成连续轨迹
            points_per_step = 10
            for j in range(points_per_step):
                self.zmp_trajectory.append(zmp)
    
    def get_zmp_trajectory(self):
        return self.zmp_trajectory


class InvertedPendulumModel:
    """倒立摆模型"""
    
    def __init__(self):
        self.com_height = 0.8
        self.com_trajectory = []
    
    def initialize(self, com_height):
        self.com_height = com_height
    
    async def generate_com_trajectory(self, zmp_trajectory: List[np.ndarray], step_duration: float):
        """生成COM轨迹"""
        self.com_trajectory = []
        
        # 简化的COM轨迹：跟随ZMP但有相位滞后
        for i, zmp in enumerate(zmp_trajectory):
            # COM在ZMP上方，有轻微滞后
            com_x = zmp[0] * 0.9
            com_y = zmp[1] * 0.9
            com_z = self.com_height
            
            self.com_trajectory.append(np.array([com_x, com_y, com_z]))
    
    def get_com_trajectory(self):
        return self.com_trajectory