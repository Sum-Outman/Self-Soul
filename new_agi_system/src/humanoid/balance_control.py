"""
平衡控制系统

为人形机器人提供平衡控制功能，包括姿态稳定、防跌倒算法和重心控制。
基于原有Self-Soul系统的generic_robot_driver.py中的平衡控制逻辑。
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BalanceState(Enum):
    """平衡状态枚举"""
    STABLE = "stable"               # 稳定
    WARNING = "warning"             # 警告
    CRITICAL = "critical"           # 临界
    FALLING = "falling"             # 跌倒中
    RECOVERING = "recovering"       # 恢复中


class BalanceStrategy(Enum):
    """平衡策略枚举"""
    ANKLE_STRATEGY = "ankle_strategy"      # 踝关节策略
    HIP_STRATEGY = "hip_strategy"          # 髋关节策略
    STEPPING_STRATEGY = "stepping_strategy"  # 踏步策略
    COMPOSITE_STRATEGY = "composite_strategy"  # 复合策略


@dataclass
class BalanceMetrics:
    """平衡指标"""
    center_of_pressure: np.ndarray        # 压力中心 (COP)
    center_of_mass: np.ndarray            # 质心 (COM)
    base_of_support: np.ndarray           # 支撑基座 (BOS)
    stability_margin: float               # 稳定裕度
    tilt_angles: Tuple[float, float]      # 倾斜角度 (pitch, roll)
    support_polygon: List[np.ndarray]     # 支撑多边形
    timestamp: float                      # 时间戳


@dataclass
class BalanceControlParams:
    """平衡控制参数"""
    kp_ankle: float = 100.0               # 踝关节比例增益
    kd_ankle: float = 10.0                # 踝关节微分增益
    kp_hip: float = 80.0                  # 髋关节比例增益
    kd_hip: float = 8.0                   # 髋关节微分增益
    com_height: float = 0.8               # 质心高度 (米)
    max_cop_displacement: float = 0.05    # 最大COP位移 (米)
    tilt_threshold_warning: float = 0.3   # 倾斜警告阈值 (弧度)
    tilt_threshold_critical: float = 0.5  # 倾斜临界阈值 (弧度)
    recovery_timeout: float = 2.0         # 恢复超时 (秒)


class BalanceControlSystem:
    """平衡控制系统"""
    
    def __init__(self, communication):
        """
        初始化平衡控制系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 平衡状态
        self.balance_state = BalanceState.STABLE
        self.active_strategy = BalanceStrategy.ANKLE_STRATEGY
        
        # 平衡指标
        self.current_metrics = BalanceMetrics(
            center_of_pressure=np.array([0.0, 0.0]),
            center_of_mass=np.array([0.0, 0.0, 0.8]),
            base_of_support=np.array([-0.1, 0.1, -0.05, 0.05]),
            stability_margin=0.2,
            tilt_angles=(0.0, 0.0),
            support_polygon=[],
            timestamp=time.time()
        )
        
        # 控制参数
        self.params = BalanceControlParams()
        
        # 历史数据
        self.metrics_history: List[BalanceMetrics] = []
        
        # 性能指标
        self.performance_metrics = {
            'total_balance_cycles': 0,
            'stabilization_success': 0,
            'fall_prevention_success': 0,
            'recovery_success': 0,
            'average_response_time': 0.0,
            'stability_maintenance': 0.0
        }
        
        # 跌倒检测
        self.fall_detection = {
            'last_fall_time': 0.0,
            'fall_count': 0,
            'recovery_in_progress': False,
            'recovery_start_time': 0.0
        }
        
        logger.info("平衡控制系统已初始化")
    
    async def initialize(self):
        """初始化平衡控制系统"""
        if self.initialized:
            return
        
        logger.info("初始化平衡控制系统...")
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="balance_control",
            component_type="humanoid"
        )
        
        self.initialized = True
        logger.info("平衡控制系统初始化完成")
    
    async def update_balance(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新平衡状态。
        
        参数:
            sensor_data: 传感器数据，包括IMU、力传感器等
            
        返回:
            平衡控制输出
        """
        try:
            start_time = time.time()
            
            # 更新平衡指标
            await self._update_balance_metrics(sensor_data)
            
            # 检测平衡状态
            await self._detect_balance_state()
            
            # 根据状态采取控制措施
            control_output = await self._apply_balance_control()
            
            # 更新性能指标
            processing_time = time.time() - start_time
            self.performance_metrics['total_balance_cycles'] += 1
            self.performance_metrics['average_response_time'] = (
                (self.performance_metrics['average_response_time'] * 
                 (self.performance_metrics['total_balance_cycles'] - 1) + 
                 processing_time) / self.performance_metrics['total_balance_cycles']
            )
            
            return control_output
            
        except Exception as e:
            logger.error(f"更新平衡状态失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'balance_state': self.balance_state.value,
                'control_action': 'emergency_stop'
            }
    
    async def _update_balance_metrics(self, sensor_data: Dict[str, Any]):
        """更新平衡指标"""
        try:
            # 提取IMU数据
            imu_data = sensor_data.get('imu', {})
            pitch = imu_data.get('pitch', 0.0)
            roll = imu_data.get('roll', 0.0)
            
            # 提取力传感器数据
            force_data = sensor_data.get('force_sensors', {})
            left_force = force_data.get('left_foot', 0.0)
            right_force = force_data.get('right_foot', 0.0)
            
            # 计算压力中心 (COP)
            total_force = left_force + right_force
            if total_force > 0:
                # 简化的COP计算（假设双脚对称分布）
                cop_x = (right_force - left_force) / (2 * total_force) * 0.2  # 假设双脚间距0.2米
                cop_y = 0.0  # 前后方向
            else:
                cop_x, cop_y = 0.0, 0.0
            
            # 计算质心 (COM) - 简化的模型
            com_x = cop_x * 0.8  # COM滞后于COP
            com_y = cop_y * 0.8
            com_z = self.params.com_height - abs(pitch) * 0.1 - abs(roll) * 0.1  # 高度随倾斜减小
            
            # 计算稳定裕度
            stability_margin = self._calculate_stability_margin(
                np.array([com_x, com_y]), 
                self.current_metrics.base_of_support
            )
            
            # 更新指标
            self.current_metrics = BalanceMetrics(
                center_of_pressure=np.array([cop_x, cop_y]),
                center_of_mass=np.array([com_x, com_y, com_z]),
                base_of_support=self.current_metrics.base_of_support,
                stability_margin=stability_margin,
                tilt_angles=(pitch, roll),
                support_polygon=self._calculate_support_polygon(left_force, right_force),
                timestamp=time.time()
            )
            
            # 保存历史数据（限制大小）
            self.metrics_history.append(self.current_metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
        except Exception as e:
            logger.error(f"更新平衡指标失败: {e}")
    
    def _calculate_stability_margin(self, com: np.ndarray, bos: np.ndarray) -> float:
        """
        计算稳定裕度。
        
        参数:
            com: 质心位置 [x, y]
            bos: 支撑基座 [x_min, x_max, y_min, y_max]
            
        返回:
            稳定裕度（距离支撑边界的最小距离）
        """
        try:
            # 计算到各边界的距离
            dist_to_x_min = com[0] - bos[0] if com[0] >= bos[0] else 0
            dist_to_x_max = bos[1] - com[0] if com[0] <= bos[1] else 0
            dist_to_y_min = com[1] - bos[2] if com[1] >= bos[2] else 0
            dist_to_y_max = bos[3] - com[1] if com[1] <= bos[3] else 0
            
            # 如果质心在支撑基座外，返回负值
            if dist_to_x_min == 0 or dist_to_x_max == 0 or dist_to_y_min == 0 or dist_to_y_max == 0:
                return -min(
                    abs(com[0] - bos[0]), abs(com[0] - bos[1]),
                    abs(com[1] - bos[2]), abs(com[1] - bos[3])
                )
            
            # 返回最小距离
            return min(dist_to_x_min, dist_to_x_max, dist_to_y_min, dist_to_y_max)
            
        except Exception as e:
            logger.error(f"计算稳定裕度失败: {e}")
            return 0.0
    
    def _calculate_support_polygon(self, left_force: float, right_force: float) -> List[np.ndarray]:
        """计算支撑多边形"""
        try:
            # 简化的支撑多边形（双脚矩形）
            foot_length = 0.2  # 脚长
            foot_width = 0.1   # 脚宽
            foot_spacing = 0.2  # 双脚间距
            
            left_foot_vertices = [
                np.array([-foot_spacing/2 - foot_width/2, -foot_length/2]),
                np.array([-foot_spacing/2 + foot_width/2, -foot_length/2]),
                np.array([-foot_spacing/2 + foot_width/2, foot_length/2]),
                np.array([-foot_spacing/2 - foot_width/2, foot_length/2])
            ]
            
            right_foot_vertices = [
                np.array([foot_spacing/2 - foot_width/2, -foot_length/2]),
                np.array([foot_spacing/2 + foot_width/2, -foot_length/2]),
                np.array([foot_spacing/2 + foot_width/2, foot_length/2]),
                np.array([foot_spacing/2 - foot_width/2, foot_length/2])
            ]
            
            # 根据受力调整支撑多边形
            if left_force > 0 and right_force > 0:
                # 双脚支撑
                return left_foot_vertices + right_foot_vertices
            elif left_force > 0:
                # 左脚单脚支撑
                return left_foot_vertices
            elif right_force > 0:
                # 右脚单脚支撑
                return right_foot_vertices
            else:
                # 无支撑（跌倒）
                return []
                
        except Exception as e:
            logger.error(f"计算支撑多边形失败: {e}")
            return []
    
    async def _detect_balance_state(self):
        """检测平衡状态"""
        try:
            metrics = self.current_metrics
            
            # 检查倾斜角度
            pitch, roll = metrics.tilt_angles
            max_tilt = max(abs(pitch), abs(roll))
            
            if max_tilt > self.params.tilt_threshold_critical:
                if self.balance_state != BalanceState.FALLING:
                    logger.warning(f"检测到临界倾斜: {max_tilt:.3f} rad")
                    self.balance_state = BalanceState.FALLING
                    self.fall_detection['fall_count'] += 1
                    self.fall_detection['last_fall_time'] = time.time()
            elif max_tilt > self.params.tilt_threshold_warning:
                if self.balance_state == BalanceState.STABLE:
                    logger.info(f"检测到警告倾斜: {max_tilt:.3f} rad")
                    self.balance_state = BalanceState.WARNING
            else:
                # 检查稳定裕度
                if metrics.stability_margin < 0:
                    if self.balance_state != BalanceState.CRITICAL:
                        logger.warning(f"质心超出支撑基座: 裕度={metrics.stability_margin:.3f}")
                        self.balance_state = BalanceState.CRITICAL
                elif metrics.stability_margin < 0.02:  # 2cm裕度
                    if self.balance_state == BalanceState.STABLE:
                        logger.info(f"稳定裕度低: {metrics.stability_margin:.3f}")
                        self.balance_state = BalanceState.WARNING
                else:
                    # 检查是否在恢复中
                    if self.balance_state in [BalanceState.WARNING, BalanceState.CRITICAL, BalanceState.RECOVERING]:
                        if time.time() - self.fall_detection.get('last_fall_time', 0) > 1.0:
                            self.balance_state = BalanceState.STABLE
                            if self.fall_detection['recovery_in_progress']:
                                self.fall_detection['recovery_in_progress'] = False
                                self.performance_metrics['recovery_success'] += 1
                                logger.info("平衡恢复成功")
                    elif self.balance_state == BalanceState.FALLING:
                        # 尝试恢复
                        if not self.fall_detection['recovery_in_progress']:
                            self.fall_detection['recovery_in_progress'] = True
                            self.fall_detection['recovery_start_time'] = time.time()
                            self.balance_state = BalanceState.RECOVERING
                            logger.info("开始平衡恢复")
            
            # 检查恢复超时
            if (self.fall_detection['recovery_in_progress'] and 
                time.time() - self.fall_detection['recovery_start_time'] > self.params.recovery_timeout):
                logger.warning("平衡恢复超时")
                self.balance_state = BalanceState.CRITICAL
                self.fall_detection['recovery_in_progress'] = False
            
        except Exception as e:
            logger.error(f"检测平衡状态失败: {e}")
    
    async def _apply_balance_control(self) -> Dict[str, Any]:
        """应用平衡控制"""
        try:
            metrics = self.current_metrics
            control_action = {}
            
            # 根据平衡状态选择控制策略
            if self.balance_state == BalanceState.STABLE:
                # 稳定状态，微调保持
                control_action = await self._apply_ankle_strategy(metrics)
                self.active_strategy = BalanceStrategy.ANKLE_STRATEGY
                self.performance_metrics['stability_maintenance'] = 0.95 * \
                    self.performance_metrics['stability_maintenance'] + 0.05
                
            elif self.balance_state == BalanceState.WARNING:
                # 警告状态，增强控制
                control_action = await self._apply_composite_strategy(metrics)
                self.active_strategy = BalanceStrategy.COMPOSITE_STRATEGY
                self.performance_metrics['fall_prevention_success'] += 1
                
            elif self.balance_state == BalanceState.CRITICAL:
                # 临界状态，使用踏步策略
                control_action = await self._apply_stepping_strategy(metrics)
                self.active_strategy = BalanceStrategy.STEPPING_STRATEGY
                
            elif self.balance_state == BalanceState.RECOVERING:
                # 恢复状态，复合策略
                control_action = await self._apply_composite_strategy(metrics, recovery=True)
                self.active_strategy = BalanceStrategy.COMPOSITE_STRATEGY
                
            elif self.balance_state == BalanceState.FALLING:
                # 跌倒状态，紧急保护
                control_action = await self._apply_fall_protection_strategy()
                self.active_strategy = BalanceStrategy.COMPOSITE_STRATEGY
            
            # 构建响应
            return {
                'success': True,
                'balance_state': self.balance_state.value,
                'active_strategy': self.active_strategy.value,
                'stability_margin': metrics.stability_margin,
                'tilt_angles': metrics.tilt_angles,
                'control_action': control_action,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"应用平衡控制失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'balance_state': self.balance_state.value,
                'control_action': {}
            }
    
    async def _apply_ankle_strategy(self, metrics: BalanceMetrics) -> Dict[str, Any]:
        """应用踝关节策略"""
        try:
            # 计算COP到COM的误差
            cop_error = metrics.center_of_mass[:2] - metrics.center_of_pressure
            
            # PD控制计算踝关节扭矩
            ankle_torque_x = self.params.kp_ankle * cop_error[0] + self.params.kd_ankle * 0  # 简化，无速度反馈
            ankle_torque_y = self.params.kp_ankle * cop_error[1] + self.params.kd_ankle * 0
            
            # 限制扭矩
            max_torque = 50.0  # Nm
            ankle_torque_x = np.clip(ankle_torque_x, -max_torque, max_torque)
            ankle_torque_y = np.clip(ankle_torque_y, -max_torque, max_torque)
            
            return {
                'strategy': 'ankle',
                'ankle_torque': {'x': float(ankle_torque_x), 'y': float(ankle_torque_y)},
                'hip_torque': {'x': 0.0, 'y': 0.0},
                'stepping_command': None
            }
            
        except Exception as e:
            logger.error(f"应用踝关节策略失败: {e}")
            return {}
    
    async def _apply_composite_strategy(self, metrics: BalanceMetrics, recovery: bool = False) -> Dict[str, Any]:
        """应用复合策略（踝关节+髋关节）"""
        try:
            # 踝关节控制
            ankle_action = await self._apply_ankle_strategy(metrics)
            
            # 髋关节控制（补偿大角度倾斜）
            pitch, roll = metrics.tilt_angles
            hip_torque_x = self.params.kp_hip * roll + self.params.kd_hip * 0
            hip_torque_y = self.params.kp_hip * pitch + self.params.kd_hip * 0
            
            # 限制扭矩
            max_torque = 80.0  # Nm
            hip_torque_x = np.clip(hip_torque_x, -max_torque, max_torque)
            hip_torque_y = np.clip(hip_torque_y, -max_torque, max_torque)
            
            # 如果是恢复模式，增加增益
            if recovery:
                hip_torque_x *= 1.5
                hip_torque_y *= 1.5
            
            return {
                'strategy': 'composite',
                'ankle_torque': ankle_action.get('ankle_torque', {'x': 0.0, 'y': 0.0}),
                'hip_torque': {'x': float(hip_torque_x), 'y': float(hip_torque_y)},
                'stepping_command': None
            }
            
        except Exception as e:
            logger.error(f"应用复合策略失败: {e}")
            return {}
    
    async def _apply_stepping_strategy(self, metrics: BalanceMetrics) -> Dict[str, Any]:
        """应用踏步策略"""
        try:
            # 判断需要踏步的方向
            cop = metrics.center_of_pressure
            bos = metrics.base_of_support
            
            # 计算COP到支撑边界的距离
            dist_to_left = cop[0] - bos[0]
            dist_to_right = bos[1] - cop[0]
            dist_to_front = cop[1] - bos[2]
            dist_to_back = bos[3] - cop[1]
            
            # 找到最近边界
            min_dist = min(dist_to_left, dist_to_right, dist_to_front, dist_to_back)
            
            stepping_command = None
            if min_dist < 0.01:  # 1cm阈值
                # 需要踏步
                if min_dist == dist_to_left:
                    stepping_command = {'leg': 'right', 'direction': 'left', 'distance': 0.1}
                elif min_dist == dist_to_right:
                    stepping_command = {'leg': 'left', 'direction': 'right', 'distance': 0.1}
                elif min_dist == dist_to_front:
                    stepping_command = {'leg': 'right', 'direction': 'back', 'distance': 0.1}
                elif min_dist == dist_to_back:
                    stepping_command = {'leg': 'left', 'direction': 'forward', 'distance': 0.1}
            
            return {
                'strategy': 'stepping',
                'ankle_torque': {'x': 0.0, 'y': 0.0},
                'hip_torque': {'x': 0.0, 'y': 0.0},
                'stepping_command': stepping_command
            }
            
        except Exception as e:
            logger.error(f"应用踏步策略失败: {e}")
            return {}
    
    async def _apply_fall_protection_strategy(self) -> Dict[str, Any]:
        """应用跌倒保护策略"""
        try:
            # 紧急保护动作
            # 1. 降低重心
            # 2. 准备接触地面
            # 3. 限制关节速度
            
            return {
                'strategy': 'fall_protection',
                'action': 'prepare_for_impact',
                'joint_commands': {
                    'knee_angle': 30.0,  # 弯曲膝盖
                    'hip_angle': 15.0,   # 弯曲髋关节
                    'elbow_angle': 90.0, # 弯曲肘关节
                    'wrist_angle': 0.0   # 伸直手腕
                },
                'stiffness': 0.3,  # 降低刚度
                'damping': 0.8     # 增加阻尼
            }
            
        except Exception as e:
            logger.error(f"应用跌倒保护策略失败: {e}")
            return {}
    
    async def get_balance_report(self) -> Dict[str, Any]:
        """获取平衡报告"""
        return {
            'balance_state': self.balance_state.value,
            'active_strategy': self.active_strategy.value,
            'current_metrics': {
                'center_of_pressure': self.current_metrics.center_of_pressure.tolist(),
                'center_of_mass': self.current_metrics.center_of_mass.tolist(),
                'stability_margin': self.current_metrics.stability_margin,
                'tilt_angles': self.current_metrics.tilt_angles,
                'timestamp': self.current_metrics.timestamp
            },
            'performance_metrics': self.performance_metrics,
            'fall_detection': self.fall_detection,
            'params': {
                'tilt_threshold_warning': self.params.tilt_threshold_warning,
                'tilt_threshold_critical': self.params.tilt_threshold_critical,
                'com_height': self.params.com_height
            },
            'timestamp': time.time()
        }
    
    async def adjust_parameters(self, new_params: Dict[str, Any]):
        """调整控制参数"""
        try:
            for key, value in new_params.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
                    logger.info(f"调整参数 {key} = {value}")
            
            return {'success': True, 'message': '参数调整成功'}
            
        except Exception as e:
            logger.error(f"调整参数失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def emergency_stop(self):
        """紧急停止"""
        try:
            self.balance_state = BalanceState.CRITICAL
            logger.warning("平衡控制系统紧急停止")
            
            return {
                'success': True,
                'action': 'emergency_stop',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def shutdown(self):
        """关闭平衡控制系统"""
        if not self.initialized:
            return
        
        logger.info("关闭平衡控制系统...")
        
        # 注销组件
        try:
            await self.communication.unregister_component("balance_control")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        self.initialized = False
        logger.info("平衡控制系统已关闭")