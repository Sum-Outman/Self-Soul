"""
电机控制系统

提供电机控制功能，包括位置控制、速度控制、扭矩控制等。
将原有Self-Soul系统的电机控制功能迁移到统一认知架构中。
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ControlAlgorithm(Enum):
    """控制算法枚举"""
    PID = "pid"                  # PID控制
    LQR = "lqr"                  # 线性二次调节器
    MPC = "mpc"                  # 模型预测控制
    SLIDING_MODE = "sliding_mode"  # 滑模控制
    ADAPTIVE = "adaptive"        # 自适应控制
    NEURAL_NETWORK = "neural_network"  # 神经网络控制


class MotorType(Enum):
    """电机类型枚举"""
    DC_BRUSHED = "dc_brushed"      # 直流有刷电机
    DC_BRUSHLESS = "dc_brushless"  # 直流无刷电机
    STEPPER = "stepper"            # 步进电机
    SERVO = "servo"                # 伺服电机
    LINEAR = "linear"              # 线性电机


@dataclass
class MotorState:
    """电机状态数据结构"""
    motor_id: str                   # 电机ID
    position: float                 # 当前位置
    velocity: float                 # 当前速度
    torque: float                   # 当前扭矩
    current: float                  # 当前电流
    temperature: float              # 当前温度
    timestamp: float                # 时间戳
    status: str = "idle"           # 状态
    error_code: int = 0            # 错误代码


@dataclass
class ControlTarget:
    """控制目标数据结构"""
    position: Optional[float] = None      # 目标位置
    velocity: Optional[float] = None      # 目标速度
    torque: Optional[float] = None        # 目标扭矩
    trajectory: Optional[List[float]] = None  # 轨迹点
    duration: float = 1.0                 # 持续时间


class MotorControlSystem:
    """电机控制系统"""
    
    def __init__(self, communication):
        """
        初始化电机控制系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 控制配置
        self.config = {
            "control_algorithm": ControlAlgorithm.PID,
            "control_rate": 200,           # 控制频率 (Hz)
            "safety_timeout": 5.0,         # 安全超时 (秒)
            "max_position_error": 5.0,     # 最大位置误差 (度)
            "max_velocity": 10.0,          # 最大速度 (度/秒)
            "max_torque": 5.0,             # 最大扭矩 (Nm)
            "max_current": 3.0,            # 最大电流 (A)
            "overheat_threshold": 80.0,    # 过热阈值 (°C)
            "enable_feedforward": True,    # 启用前馈控制
            "enable_friction_compensation": True,  # 启用摩擦补偿
        }
        
        # 电机注册表
        self.motors: Dict[str, Dict[str, Any]] = {}
        
        # 电机状态
        self.motor_states: Dict[str, MotorState] = {}
        
        # 控制目标
        self.control_targets: Dict[str, ControlTarget] = {}
        
        # 控制器实例
        self.controllers: Dict[str, Any] = {}
        
        # 数据锁
        self.data_lock = threading.RLock()
        
        # 控制任务
        self.control_task = None
        self.control_active = False
        
        # 性能指标
        self.performance_metrics = {
            'total_control_cycles': 0,
            'successful_controls': 0,
            'failed_controls': 0,
            'emergency_stops': 0,
            'average_control_latency': 0.0,
            'position_error_sum': 0.0,
            'average_position_error': 0.0,
            'overheat_events': 0
        }
        
        logger.info("电机控制系统已初始化")
    
    async def initialize(self):
        """初始化电机控制系统"""
        if self.initialized:
            return
        
        logger.info("初始化电机控制系统...")
        
        # 扫描可用电机
        await self._scan_motors()
        
        # 初始化控制器
        await self._initialize_controllers()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="motor_control",
            component_type="control"
        )
        
        # 启动控制任务
        await self._start_control_task()
        
        self.initialized = True
        logger.info("电机控制系统初始化完成")
    
    async def _scan_motors(self):
        """扫描电机"""
        try:
            logger.info("扫描电机...")
            
            # 模拟电机发现 - 实际系统应该进行实际的硬件扫描
            self._add_simulated_motors()
            
            logger.info(f"发现 {len(self.motors)} 个电机")
            
        except Exception as e:
            logger.error(f"扫描电机失败: {e}")
    
    def _add_simulated_motors(self):
        """添加模拟电机"""
        # 腿部电机
        self.motors["motor_hip_left"] = {
            "motor_id": "motor_hip_left",
            "motor_type": MotorType.DC_BRUSHLESS.value,
            "name": "Left Hip Motor",
            "position_range": (-45, 45),      # 度
            "velocity_range": (-10, 10),      # 度/秒
            "torque_range": (-5, 5),          # Nm
            "current_range": (-3, 3),         # A
            "gear_ratio": 100,                # 减速比
            "encoder_resolution": 4096,       # 编码器分辨率
            "calibrated": False,
            "parameters": {
                "resistance": 2.4,            # 电阻 (Ω)
                "inductance": 0.001,          # 电感 (H)
                "torque_constant": 0.1,       # 扭矩常数 (Nm/A)
                "back_emf_constant": 0.1,     # 反电动势常数 (V/(rad/s))
            }
        }
        
        self.motors["motor_knee_left"] = {
            "motor_id": "motor_knee_left",
            "motor_type": MotorType.DC_BRUSHLESS.value,
            "name": "Left Knee Motor",
            "position_range": (0, 120),       # 度
            "velocity_range": (-10, 10),      # 度/秒
            "torque_range": (-5, 5),          # Nm
            "current_range": (-3, 3),         # A
            "gear_ratio": 100,
            "encoder_resolution": 4096,
            "calibrated": False,
            "parameters": self.motors["motor_hip_left"]["parameters"]
        }
        
        # 手臂电机
        self.motors["motor_shoulder_right"] = {
            "motor_id": "motor_shoulder_right",
            "motor_type": MotorType.SERVO.value,
            "name": "Right Shoulder Servo",
            "position_range": (-90, 90),      # 度
            "velocity_range": (-5, 5),        # 度/秒
            "torque_range": (-10, 10),        # Nm
            "current_range": (-2, 2),         # A
            "gear_ratio": 200,
            "encoder_resolution": 2048,
            "calibrated": False,
            "parameters": {
                "resistance": 5.0,
                "inductance": 0.002,
                "torque_constant": 0.2,
                "back_emf_constant": 0.15,
            }
        }
        
        # 初始化电机状态
        for motor_id, motor_info in self.motors.items():
            self.motor_states[motor_id] = MotorState(
                motor_id=motor_id,
                position=0.0,
                velocity=0.0,
                torque=0.0,
                current=0.0,
                temperature=25.0,
                timestamp=time.time(),
                status="idle"
            )
    
    async def _initialize_controllers(self):
        """初始化控制器"""
        try:
            for motor_id, motor_info in self.motors.items():
                # 根据电机类型选择控制器
                motor_type = motor_info.get("motor_type", MotorType.DC_BRUSHLESS.value)
                
                if motor_type == MotorType.SERVO.value:
                    controller = await self._create_pid_controller(motor_id)
                else:
                    controller = await self._create_pid_controller(motor_id)
                
                self.controllers[motor_id] = controller
            
            logger.info(f"为 {len(self.controllers)} 个电机初始化控制器")
            
        except Exception as e:
            logger.error(f"初始化控制器失败: {e}")
    
    async def _create_pid_controller(self, motor_id: str) -> Dict[str, Any]:
        """创建PID控制器"""
        motor_info = self.motors.get(motor_id, {})
        
        # 根据电机类型调整PID参数
        if motor_info.get("motor_type") == MotorType.SERVO.value:
            # 伺服电机参数
            kp, ki, kd = 2.5, 0.1, 0.05
        else:
            # 普通电机参数
            kp, ki, kd = 1.5, 0.05, 0.02
        
        return {
            "type": "pid",
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "integral_limit": 10.0,
            "output_limit": motor_info.get("torque_range", (-5, 5))[1],
            "error_history": [],
            "integral_sum": 0.0,
            "last_error": 0.0,
            "last_time": time.time()
        }
    
    async def _start_control_task(self):
        """启动控制任务"""
        if self.control_active:
            return
        
        self.control_active = True
        self.control_task = asyncio.create_task(self._control_loop())
        logger.info("电机控制任务已启动")
    
    async def _control_loop(self):
        """控制循环"""
        while self.control_active:
            try:
                start_time = time.time()
                
                # 执行控制
                control_results = await self._execute_control()
                
                # 更新性能指标
                control_time = time.time() - start_time
                self._update_performance_metrics(control_results, control_time)
                
                # 安全检查
                await self._safety_check()
                
                # 等待下一个控制周期
                await asyncio.sleep(1.0 / self.config["control_rate"])
                
            except Exception as e:
                logger.error(f"控制循环错误: {e}")
                self.performance_metrics['failed_controls'] += 1
                await asyncio.sleep(0.1)
    
    async def _execute_control(self) -> Dict[str, Dict[str, Any]]:
        """执行控制"""
        results = {}
        
        with self.data_lock:
            for motor_id, motor_state in self.motor_states.items():
                try:
                    # 检查是否有控制目标
                    control_target = self.control_targets.get(motor_id)
                    
                    if not control_target:
                        # 无目标，保持当前位置
                        continue
                    
                    # 获取控制器
                    controller = self.controllers.get(motor_id)
                    if not controller:
                        logger.warning(f"电机 {motor_id} 无控制器")
                        continue
                    
                    # 执行控制算法
                    control_output = await self._apply_control_algorithm(
                        motor_id, motor_state, control_target, controller
                    )
                    
                    # 更新电机状态
                    updated_state = await self._update_motor_state(
                        motor_id, motor_state, control_output
                    )
                    
                    # 存储结果
                    results[motor_id] = {
                        "success": True,
                        "control_output": control_output,
                        "updated_state": updated_state,
                        "position_error": abs(control_output.get("position_error", 0.0))
                    }
                    
                    self.performance_metrics['successful_controls'] += 1
                    
                except Exception as e:
                    logger.error(f"控制电机 {motor_id} 失败: {e}")
                    results[motor_id] = {
                        "success": False,
                        "error": str(e)
                    }
                    self.performance_metrics['failed_controls'] += 1
        
        return results
    
    async def _apply_control_algorithm(self, motor_id: str, 
                                      motor_state: MotorState,
                                      control_target: ControlTarget,
                                      controller: Dict[str, Any]) -> Dict[str, Any]:
        """应用控制算法"""
        algorithm = self.config["control_algorithm"]
        
        if algorithm == ControlAlgorithm.PID:
            return await self._apply_pid_control(motor_id, motor_state, control_target, controller)
        elif algorithm == ControlAlgorithm.ADAPTIVE:
            return await self._apply_adaptive_control(motor_id, motor_state, control_target, controller)
        elif algorithm == ControlAlgorithm.NEURAL_NETWORK:
            return await self._apply_neural_control(motor_id, motor_state, control_target, controller)
        else:
            # 默认使用PID
            return await self._apply_pid_control(motor_id, motor_state, control_target, controller)
    
    async def _apply_pid_control(self, motor_id: str,
                                motor_state: MotorState,
                                control_target: ControlTarget,
                                controller: Dict[str, Any]) -> Dict[str, Any]:
        """应用PID控制"""
        try:
            current_time = time.time()
            dt = current_time - controller.get("last_time", current_time)
            
            if dt <= 0:
                dt = 0.001
            
            # 计算位置误差
            target_position = control_target.position
            if target_position is None:
                target_position = motor_state.position
            
            position_error = target_position - motor_state.position
            
            # PID计算
            kp = controller.get("kp", 1.0)
            ki = controller.get("ki", 0.0)
            kd = controller.get("kd", 0.0)
            
            # 积分项
            integral_sum = controller.get("integral_sum", 0.0)
            integral_sum += position_error * dt
            
            # 积分限幅
            integral_limit = controller.get("integral_limit", 10.0)
            integral_sum = max(-integral_limit, min(integral_limit, integral_sum))
            
            # 微分项
            last_error = controller.get("last_error", 0.0)
            derivative = (position_error - last_error) / dt if dt > 0 else 0.0
            
            # 计算控制输出
            control_output = (
                kp * position_error +
                ki * integral_sum +
                kd * derivative
            )
            
            # 输出限幅
            output_limit = controller.get("output_limit", 5.0)
            control_output = max(-output_limit, min(output_limit, control_output))
            
            # 更新控制器状态
            controller["integral_sum"] = integral_sum
            controller["last_error"] = position_error
            controller["last_time"] = current_time
            
            # 记录误差历史
            error_history = controller.get("error_history", [])
            error_history.append(position_error)
            if len(error_history) > 100:
                error_history.pop(0)
            controller["error_history"] = error_history
            
            return {
                "torque": control_output,
                "position_error": position_error,
                "integral_sum": integral_sum,
                "derivative": derivative,
                "dt": dt
            }
            
        except Exception as e:
            logger.error(f"应用PID控制失败: {e}")
            return {"torque": 0.0, "position_error": 0.0}
    
    async def _apply_adaptive_control(self, motor_id: str,
                                     motor_state: MotorState,
                                     control_target: ControlTarget,
                                     controller: Dict[str, Any]) -> Dict[str, Any]:
        """应用自适应控制"""
        # 简化实现
        try:
            # 使用基本的自适应PID
            target_position = control_target.position or motor_state.position
            position_error = target_position - motor_state.position
            
            # 自适应增益调整
            error_abs = abs(position_error)
            
            if error_abs > 10.0:
                kp, ki, kd = 3.0, 0.2, 0.1  # 大误差，高增益
            elif error_abs > 1.0:
                kp, ki, kd = 2.0, 0.1, 0.05  # 中等误差，中等增益
            else:
                kp, ki, kd = 1.0, 0.05, 0.02  # 小误差，低增益
            
            # 简化PID计算
            control_output = kp * position_error
            
            # 输出限幅
            output_limit = controller.get("output_limit", 5.0)
            control_output = max(-output_limit, min(output_limit, control_output))
            
            return {
                "torque": control_output,
                "position_error": position_error,
                "adaptive_gains": {"kp": kp, "ki": ki, "kd": kd}
            }
            
        except Exception as e:
            logger.error(f"应用自适应控制失败: {e}")
            return {"torque": 0.0, "position_error": 0.0}
    
    async def _apply_neural_control(self, motor_id: str,
                                   motor_state: MotorState,
                                   control_target: ControlTarget,
                                   controller: Dict[str, Any]) -> Dict[str, Any]:
        """应用神经网络控制"""
        # 简化实现 - 实际系统需要神经网络模型
        try:
            target_position = control_target.position or motor_state.position
            position_error = target_position - motor_state.position
            
            # 简化的"神经网络"输出
            control_output = np.tanh(position_error * 0.5) * 3.0
            
            # 输出限幅
            output_limit = controller.get("output_limit", 5.0)
            control_output = max(-output_limit, min(output_limit, control_output))
            
            return {
                "torque": control_output,
                "position_error": position_error,
                "neural_confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"应用神经网络控制失败: {e}")
            return {"torque": 0.0, "position_error": 0.0}
    
    async def _update_motor_state(self, motor_id: str,
                                 motor_state: MotorState,
                                 control_output: Dict[str, Any]) -> MotorState:
        """更新电机状态"""
        try:
            # 模拟电机动力学
            torque = control_output.get("torque", 0.0)
            
            # 简单的一阶系统模型
            dt = 1.0 / self.config["control_rate"]
            
            # 更新速度 (简化的动力学)
            acceleration = torque / 0.1  # 假设惯性矩为0.1 kg·m²
            new_velocity = motor_state.velocity + acceleration * dt
            
            # 速度限幅
            max_velocity = self.config["max_velocity"]
            new_velocity = max(-max_velocity, min(max_velocity, new_velocity))
            
            # 更新位置
            new_position = motor_state.position + new_velocity * dt
            
            # 位置限幅
            motor_info = self.motors.get(motor_id, {})
            position_range = motor_info.get("position_range", (-180, 180))
            new_position = max(position_range[0], min(position_range[1], new_position))
            
            # 更新电流 (简化的电机模型)
            torque_constant = motor_info.get("parameters", {}).get("torque_constant", 0.1)
            new_current = torque / torque_constant if torque_constant != 0 else 0.0
            
            # 电流限幅
            max_current = self.config["max_current"]
            new_current = max(-max_current, min(max_current, new_current))
            
            # 更新温度 (简化的热模型)
            resistance = motor_info.get("parameters", {}).get("resistance", 2.4)
            power_loss = new_current**2 * resistance
            temp_increase = power_loss * dt / 100.0  # 简化的热容
            new_temperature = motor_state.temperature + temp_increase
            
            # 创建新状态
            updated_state = MotorState(
                motor_id=motor_id,
                position=new_position,
                velocity=new_velocity,
                torque=torque,
                current=new_current,
                temperature=new_temperature,
                timestamp=time.time(),
                status="active"
            )
            
            # 更新状态
            self.motor_states[motor_id] = updated_state
            
            return updated_state
            
        except Exception as e:
            logger.error(f"更新电机状态失败: {e}")
            return motor_state
    
    def _update_performance_metrics(self, control_results: Dict[str, Dict[str, Any]], 
                                   control_time: float):
        """更新性能指标"""
        self.performance_metrics['total_control_cycles'] += 1
        
        # 计算平均控制延迟
        total_cycles = self.performance_metrics['total_control_cycles']
        current_avg = self.performance_metrics['average_control_latency']
        self.performance_metrics['average_control_latency'] = (
            current_avg * (total_cycles - 1) + control_time
        ) / total_cycles
        
        # 计算位置误差
        position_error_sum = 0
        error_count = 0
        
        for result in control_results.values():
            if result.get("success"):
                position_error = result.get("position_error", 0)
                position_error_sum += abs(position_error)
                error_count += 1
        
        if error_count > 0:
            self.performance_metrics['position_error_sum'] += position_error_sum
            self.performance_metrics['average_position_error'] = (
                self.performance_metrics['position_error_sum'] / 
                (self.performance_metrics['successful_controls'] + 1e-6)
            )
    
    async def _safety_check(self):
        """安全检查"""
        try:
            for motor_id, motor_state in self.motor_states.items():
                # 检查温度
                if motor_state.temperature > self.config["overheat_threshold"]:
                    logger.warning(f"电机 {motor_id} 过热: {motor_state.temperature}°C")
                    self.performance_metrics['overheat_events'] += 1
                    
                    # 降低输出
                    control_target = self.control_targets.get(motor_id)
                    if control_target:
                        control_target.torque = control_target.torque * 0.5 if control_target.torque else 0.0
                
                # 检查电流
                if abs(motor_state.current) > self.config["max_current"]:
                    logger.warning(f"电机 {motor_id} 电流超标: {motor_state.current}A")
                    await self._emergency_stop_motor(motor_id)
                
                # 检查位置误差
                control_target = self.control_targets.get(motor_id)
                if control_target and control_target.position is not None:
                    position_error = abs(control_target.position - motor_state.position)
                    if position_error > self.config["max_position_error"]:
                        logger.warning(f"电机 {motor_id} 位置误差过大: {position_error}度")
                        await self._recover_position_error(motor_id, position_error)
            
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
    
    async def _emergency_stop_motor(self, motor_id: str):
        """紧急停止电机"""
        try:
            with self.data_lock:
                # 清除控制目标
                if motor_id in self.control_targets:
                    del self.control_targets[motor_id]
                
                # 停止电机
                motor_state = self.motor_states.get(motor_id)
                if motor_state:
                    motor_state.status = "emergency_stop"
                    motor_state.velocity = 0.0
                    motor_state.torque = 0.0
                    motor_state.current = 0.0
                
                self.performance_metrics['emergency_stops'] += 1
                logger.warning(f"电机 {motor_id} 紧急停止")
                
        except Exception as e:
            logger.error(f"紧急停止电机 {motor_id} 失败: {e}")
    
    async def _recover_position_error(self, motor_id: str, position_error: float):
        """恢复位置误差"""
        try:
            # 简化的恢复策略
            motor_state = self.motor_states.get(motor_id)
            if not motor_state:
                return
            
            # 缓慢返回零位
            control_target = self.control_targets.get(motor_id)
            if control_target:
                control_target.position = 0.0
                control_target.velocity = 1.0  # 慢速
                
            logger.info(f"电机 {motor_id} 开始位置恢复，误差: {position_error}度")
            
        except Exception as e:
            logger.error(f"恢复位置误差失败: {e}")
    
    async def set_control_target(self, motor_id: str, target: ControlTarget) -> Dict[str, Any]:
        """设置控制目标"""
        start_time = time.time()
        
        try:
            with self.data_lock:
                # 检查电机是否存在
                if motor_id not in self.motors:
                    return {"success": False, "error": f"电机 {motor_id} 未注册"}
                
                # 验证目标值
                validation_result = await self._validate_control_target(motor_id, target)
                if not validation_result["valid"]:
                    return {"success": False, "error": validation_result["reason"]}
                
                # 设置控制目标
                self.control_targets[motor_id] = target
                
                # 更新电机状态
                motor_state = self.motor_states.get(motor_id)
                if motor_state:
                    motor_state.status = "target_set"
                
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "motor_id": motor_id,
                    "target": target,
                    "latency": latency,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"设置控制目标失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_control_target(self, motor_id: str, target: ControlTarget) -> Dict[str, Any]:
        """验证控制目标"""
        motor_info = self.motors.get(motor_id, {})
        position_range = motor_info.get("position_range", (-180, 180))
        
        valid = True
        reasons = []
        
        # 检查位置
        if target.position is not None:
            if target.position < position_range[0] or target.position > position_range[1]:
                valid = False
                reasons.append(f"位置 {target.position} 超出范围 {position_range}")
        
        # 检查速度
        if target.velocity is not None:
            max_velocity = self.config["max_velocity"]
            if abs(target.velocity) > max_velocity:
                valid = False
                reasons.append(f"速度 {target.velocity} 超过限制 {max_velocity}")
        
        # 检查扭矩
        if target.torque is not None:
            max_torque = self.config["max_torque"]
            if abs(target.torque) > max_torque:
                valid = False
                reasons.append(f"扭矩 {target.torque} 超过限制 {max_torque}")
        
        return {
            "valid": valid,
            "reason": "; ".join(reasons) if reasons else "有效"
        }
    
    async def get_motor_state(self, motor_id: str) -> Optional[MotorState]:
        """获取电机状态"""
        with self.data_lock:
            return self.motor_states.get(motor_id)
    
    async def get_all_motor_states(self) -> Dict[str, MotorState]:
        """获取所有电机状态"""
        with self.data_lock:
            return self.motor_states.copy()
    
    async def emergency_stop_all(self) -> Dict[str, Any]:
        """紧急停止所有电机"""
        try:
            with self.data_lock:
                # 清除所有控制目标
                self.control_targets.clear()
                
                # 停止所有电机
                for motor_id, motor_state in self.motor_states.items():
                    motor_state.status = "emergency_stop"
                    motor_state.velocity = 0.0
                    motor_state.torque = 0.0
                    motor_state.current = 0.0
                
                self.performance_metrics['emergency_stops'] += 1
                
                return {
                    "success": True,
                    "stopped_motors": len(self.motor_states),
                    "message": "所有电机已紧急停止"
                }
                
        except Exception as e:
            logger.error(f"紧急停止所有电机失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.data_lock:
            return self.performance_metrics.copy()
    
    async def calibrate_motor(self, motor_id: str) -> Dict[str, Any]:
        """校准电机"""
        try:
            with self.data_lock:
                if motor_id not in self.motors:
                    return {"success": False, "error": f"电机 {motor_id} 未注册"}
                
                # 模拟校准过程
                await asyncio.sleep(0.5)  # 模拟校准时间
                
                # 标记为已校准
                self.motors[motor_id]["calibrated"] = True
                
                # 重置位置为零
                motor_state = self.motor_states.get(motor_id)
                if motor_state:
                    motor_state.position = 0.0
                    motor_state.velocity = 0.0
                
                return {
                    "success": True,
                    "motor_id": motor_id,
                    "calibration_time": 0.5,
                    "position_reset": True
                }
                
        except Exception as e:
            logger.error(f"校准电机失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """关闭电机控制系统"""
        if not self.initialized:
            return
        
        logger.info("关闭电机控制系统...")
        
        # 停止控制任务
        self.control_active = False
        if self.control_task:
            self.control_task.cancel()
            try:
                await self.control_task
            except asyncio.CancelledError:
                pass
        
        # 安全停止所有电机
        await self.emergency_stop_all()
        
        # 注销组件
        try:
            await self.communication.unregister_component("motor_control")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        # 清理数据
        with self.data_lock:
            self.motors.clear()
            self.motor_states.clear()
            self.control_targets.clear()
            self.controllers.clear()
        
        self.initialized = False
        logger.info("电机控制系统已关闭")