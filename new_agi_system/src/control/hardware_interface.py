"""
硬件控制系统

提供统一的硬件接口，包括传感器、电机、伺服等设备的控制。
将原有Self-Soul系统的robot_hardware_interface.py功能迁移到统一认知架构中。
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """设备类型枚举"""
    SENSOR = "sensor"            # 传感器
    MOTOR = "motor"              # 电机
    SERVO = "servo"              # 伺服
    CAMERA = "camera"            # 相机
    CONTROLLER = "controller"    # 控制器
    BATTERY = "battery"          # 电池


class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"                  # 惯性测量单元
    FORCE_TORQUE = "force_torque"  # 力/扭矩传感器
    ENCODER = "encoder"          # 编码器
    PROXIMITY = "proximity"      # 接近传感器
    TEMPERATURE = "temperature"  # 温度传感器
    PRESSURE = "pressure"        # 压力传感器
    CURRENT = "current"          # 电流传感器
    VOLTAGE = "voltage"          # 电压传感器


@dataclass
class DeviceInfo:
    """设备信息数据结构"""
    device_id: str               # 设备ID
    device_type: DeviceType      # 设备类型
    subtype: Optional[str] = None  # 子类型
    name: str = ""              # 设备名称
    version: str = "1.0"        # 版本
    manufacturer: str = ""      # 制造商
    parameters: Dict[str, Any] = field(default_factory=dict)  # 参数
    capabilities: List[str] = field(default_factory=list)     # 能力


@dataclass
class SensorData:
    """传感器数据结构"""
    sensor_id: str               # 传感器ID
    sensor_type: SensorType      # 传感器类型
    value: Any                   # 传感器值
    timestamp: float             # 时间戳
    unit: str = ""               # 单位
    accuracy: float = 1.0        # 精度
    status: str = "normal"       # 状态


class HardwareControlSystem:
    """硬件控制系统"""
    
    def __init__(self, communication):
        """
        初始化硬件控制系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 硬件配置
        self.config = {
            "sensor_update_rate": 100,      # 传感器更新频率 (Hz)
            "motor_control_rate": 200,      # 电机控制频率 (Hz)
            "camera_fps": 60,               # 相机帧率
            "max_motor_current": 5.0,       # 最大电机电流 (A)
            "max_servo_torque": 10.0,       # 最大伺服扭矩 (Nm)
            "emergency_stop_timeout": 0.1,  # 急停超时 (秒)
            "battery_monitoring": True,     # 电池监控
            "temperature_monitoring": True, # 温度监控
            "hardware_safety": True,        # 硬件安全保护
        }
        
        # 设备注册表
        self.sensors: Dict[str, DeviceInfo] = {}
        self.motors: Dict[str, DeviceInfo] = {}
        self.servos: Dict[str, DeviceInfo] = {}
        self.cameras: Dict[str, DeviceInfo] = {}
        self.controllers: Dict[str, DeviceInfo] = {}
        
        # 实时数据缓存
        self.sensor_data: Dict[str, SensorData] = {}
        self.motor_states: Dict[str, Dict[str, Any]] = {}
        self.servo_positions: Dict[str, float] = {}
        
        # 硬件锁
        self.data_lock = threading.RLock()
        
        # 监控任务
        self.monitoring_task = None
        self.monitoring_active = False
        
        # 性能指标
        self.performance_metrics = {
            'total_sensor_reads': 0,
            'total_motor_commands': 0,
            'total_servo_commands': 0,
            'hardware_errors': 0,
            'emergency_stops': 0,
            'average_latency': 0.0
        }
        
        logger.info("硬件控制系统已初始化")
    
    async def initialize(self):
        """初始化硬件控制系统"""
        if self.initialized:
            return
        
        logger.info("初始化硬件控制系统...")
        
        # 扫描可用硬件设备
        await self._scan_hardware_devices()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="hardware_control",
            component_type="control"
        )
        
        # 启动监控任务
        await self._start_monitoring()
        
        self.initialized = True
        logger.info("硬件控制系统初始化完成")
    
    async def _scan_hardware_devices(self):
        """扫描硬件设备"""
        try:
            logger.info("扫描硬件设备...")
            
            # 模拟设备发现 - 实际系统应该进行实际的硬件扫描
            self._add_simulated_devices()
            
            # 记录发现的设备
            total_devices = (
                len(self.sensors) + len(self.motors) + 
                len(self.servos) + len(self.cameras) + len(self.controllers)
            )
            logger.info(f"发现 {total_devices} 个硬件设备")
            
        except Exception as e:
            logger.error(f"扫描硬件设备失败: {e}")
    
    def _add_simulated_devices(self):
        """添加模拟设备（用于测试）"""
        # 添加模拟传感器
        self.sensors["imu_001"] = DeviceInfo(
            device_id="imu_001",
            device_type=DeviceType.SENSOR,
            subtype=SensorType.IMU.value,
            name="IMU Sensor",
            manufacturer="AGI Robotics",
            parameters={"range": "±16g", "resolution": "0.1°"},
            capabilities=["orientation", "acceleration", "angular_velocity"]
        )
        
        self.sensors["force_001"] = DeviceInfo(
            device_id="force_001",
            device_type=DeviceType.SENSOR,
            subtype=SensorType.FORCE_TORQUE.value,
            name="Force/Torque Sensor",
            manufacturer="AGI Robotics",
            parameters={"range": "±100N", "accuracy": "0.5%"},
            capabilities=["force_measurement", "torque_measurement"]
        )
        
        # 添加模拟电机
        self.motors["motor_001"] = DeviceInfo(
            device_id="motor_001",
            device_type=DeviceType.MOTOR,
            subtype="DC Motor",
            name="Main Joint Motor",
            manufacturer="AGI Robotics",
            parameters={"voltage": "24V", "max_current": "5A"},
            capabilities=["position_control", "velocity_control", "torque_control"]
        )
        
        # 添加模拟伺服
        self.servos["servo_001"] = DeviceInfo(
            device_id="servo_001",
            device_type=DeviceType.SERVO,
            subtype="Digital Servo",
            name="Shoulder Servo",
            manufacturer="AGI Robotics",
            parameters={"torque": "10Nm", "speed": "0.1s/60°"},
            capabilities=["position_control", "torque_limit"]
        )
    
    async def _start_monitoring(self):
        """启动监控任务"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("硬件监控任务已启动")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 更新传感器数据
                await self._update_sensor_data()
                
                # 检查硬件状态
                await self._check_hardware_status()
                
                # 更新性能指标
                await self._update_performance_metrics()
                
                # 等待下一个周期
                await asyncio.sleep(1.0 / self.config["sensor_update_rate"])
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_sensor_data(self):
        """更新传感器数据"""
        with self.data_lock:
            for sensor_id, sensor_info in self.sensors.items():
                try:
                    # 模拟传感器数据 - 实际系统应该从硬件读取
                    if sensor_info.subtype == SensorType.IMU.value:
                        value = {
                            "acceleration": [0.0, 0.0, 9.8],  # 重力加速度
                            "gyroscope": [0.0, 0.0, 0.0],
                            "orientation": [0.0, 0.0, 0.0]
                        }
                    elif sensor_info.subtype == SensorType.FORCE_TORQUE.value:
                        value = {
                            "force": [0.0, 0.0, 0.0],
                            "torque": [0.0, 0.0, 0.0]
                        }
                    else:
                        value = 0.0
                    
                    # 创建传感器数据
                    sensor_data = SensorData(
                        sensor_id=sensor_id,
                        sensor_type=SensorType(sensor_info.subtype),
                        value=value,
                        timestamp=time.time(),
                        unit="N" if sensor_info.subtype == SensorType.FORCE_TORQUE.value else "g",
                        accuracy=0.95
                    )
                    
                    # 存储数据
                    self.sensor_data[sensor_id] = sensor_data
                    
                    # 更新性能指标
                    self.performance_metrics['total_sensor_reads'] += 1
                    
                except Exception as e:
                    logger.error(f"更新传感器 {sensor_id} 数据失败: {e}")
    
    async def _check_hardware_status(self):
        """检查硬件状态"""
        # 检查传感器状态
        for sensor_id in self.sensors:
            if sensor_id not in self.sensor_data:
                logger.warning(f"传感器 {sensor_id} 无数据")
        
        # 检查电机状态
        for motor_id in self.motors:
            if motor_id not in self.motor_states:
                logger.debug(f"电机 {motor_id} 未激活")
        
        # 检查伺服状态
        for servo_id in self.servos:
            if servo_id not in self.servo_positions:
                logger.debug(f"伺服 {servo_id} 未激活")
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        # 计算平均延迟（模拟）
        if self.performance_metrics['total_sensor_reads'] > 0:
            self.performance_metrics['average_latency'] = (
                1.0 / self.config["sensor_update_rate"]
            )
    
    async def get_sensor_data(self, sensor_id: str) -> Optional[SensorData]:
        """获取传感器数据"""
        with self.data_lock:
            return self.sensor_data.get(sensor_id)
    
    async def get_all_sensor_data(self) -> Dict[str, SensorData]:
        """获取所有传感器数据"""
        with self.data_lock:
            return self.sensor_data.copy()
    
    async def control_motor(self, motor_id: str, 
                          position: Optional[float] = None,
                          velocity: Optional[float] = None,
                          torque: Optional[float] = None) -> Dict[str, Any]:
        """控制电机"""
        start_time = time.time()
        
        try:
            with self.data_lock:
                # 检查电机是否存在
                if motor_id not in self.motors:
                    return {"success": False, "error": f"电机 {motor_id} 未注册"}
                
                # 安全检查
                safety_check = await self._check_motor_safety(motor_id, position, velocity, torque)
                if not safety_check["safe"]:
                    self.performance_metrics['hardware_errors'] += 1
                    return {"success": False, "error": safety_check["reason"]}
                
                # 更新目标值
                if motor_id not in self.motor_states:
                    self.motor_states[motor_id] = {}
                
                if position is not None:
                    self.motor_states[motor_id]["target_position"] = position
                if velocity is not None:
                    self.motor_states[motor_id]["target_velocity"] = velocity
                if torque is not None:
                    self.motor_states[motor_id]["target_torque"] = torque
                
                # 模拟执行
                await asyncio.sleep(0.001)  # 模拟控制延迟
                
                # 更新状态
                self.motor_states[motor_id]["last_update"] = time.time()
                self.motor_states[motor_id]["status"] = "active"
                
                # 更新性能指标
                self.performance_metrics['total_motor_commands'] += 1
                
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "motor_id": motor_id,
                    "position": position,
                    "velocity": velocity,
                    "torque": torque,
                    "latency": latency
                }
                
        except Exception as e:
            logger.error(f"控制电机 {motor_id} 失败: {e}")
            self.performance_metrics['hardware_errors'] += 1
            return {"success": False, "error": str(e)}
    
    async def control_servo(self, servo_id: str, position: float) -> Dict[str, Any]:
        """控制伺服"""
        start_time = time.time()
        
        try:
            with self.data_lock:
                # 检查伺服是否存在
                if servo_id not in self.servos:
                    return {"success": False, "error": f"伺服 {servo_id} 未注册"}
                
                # 安全检查
                if position < 0 or position > 180:
                    self.performance_metrics['hardware_errors'] += 1
                    return {"success": False, "error": "位置超出范围 (0-180度)"}
                
                # 更新位置
                self.servo_positions[servo_id] = position
                
                # 模拟执行
                await asyncio.sleep(0.001)  # 模拟控制延迟
                
                # 更新性能指标
                self.performance_metrics['total_servo_commands'] += 1
                
                latency = time.time() - start_time
                
                return {
                    "success": True,
                    "servo_id": servo_id,
                    "position": position,
                    "latency": latency
                }
                
        except Exception as e:
            logger.error(f"控制伺服 {servo_id} 失败: {e}")
            self.performance_metrics['hardware_errors'] += 1
            return {"success": False, "error": str(e)}
    
    async def _check_motor_safety(self, motor_id: str,
                                position: Optional[float],
                                velocity: Optional[float],
                                torque: Optional[float]) -> Dict[str, Any]:
        """检查电机安全性"""
        safe = True
        reasons = []
        
        # 检查电流限制
        if torque is not None:
            max_torque = self.config.get("max_servo_torque", 10.0)
            if abs(torque) > max_torque:
                safe = False
                reasons.append(f"扭矩 {torque} 超过限制 {max_torque}")
        
        # 检查位置限制（如果适用）
        if position is not None:
            # 简化的位置检查
            if abs(position) > 180:
                safe = False
                reasons.append(f"位置 {position} 超出合理范围")
        
        return {
            "safe": safe,
            "reason": "; ".join(reasons) if reasons else "安全"
        }
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """紧急停止"""
        try:
            with self.data_lock:
                logger.warning("执行紧急停止!")
                
                # 停止所有电机
                for motor_id in self.motors:
                    self.motor_states[motor_id] = {"status": "stopped"}
                
                # 停止所有伺服
                for servo_id in self.servos:
                    self.servo_positions[servo_id] = 0.0
                
                # 更新指标
                self.performance_metrics['emergency_stops'] += 1
                
                return {
                    "success": True,
                    "message": "紧急停止已执行",
                    "stopped_motors": len(self.motors),
                    "stopped_servos": len(self.servos)
                }
                
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.data_lock:
            return self.performance_metrics.copy()
    
    def get_device_list(self) -> Dict[str, List[str]]:
        """获取设备列表"""
        with self.data_lock:
            return {
                "sensors": list(self.sensors.keys()),
                "motors": list(self.motors.keys()),
                "servos": list(self.servos.keys()),
                "cameras": list(self.cameras.keys()),
                "controllers": list(self.controllers.keys())
            }
    
    async def shutdown(self):
        """关闭硬件控制系统"""
        if not self.initialized:
            return
        
        logger.info("关闭硬件控制系统...")
        
        # 停止监控任务
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 安全关闭所有设备
        await self.emergency_stop()
        
        # 注销组件
        try:
            await self.communication.unregister_component("hardware_control")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        # 清理数据
        with self.data_lock:
            self.sensors.clear()
            self.motors.clear()
            self.servos.clear()
            self.cameras.clear()
            self.controllers.clear()
            self.sensor_data.clear()
            self.motor_states.clear()
            self.servo_positions.clear()
        
        self.initialized = False
        logger.info("硬件控制系统已关闭")