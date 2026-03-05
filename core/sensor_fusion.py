"""
传感器融合引擎

集成来自机器人各种传感器的数据，包括：
1. IMU（加速度计、陀螺仪、磁力计）
2. 力/力矩传感器
3. 足底压力传感器
4. 关节编码器
5. 视觉传感器
6. 接近传感器

提供多模态传感器数据融合，用于状态估计、平衡控制、步态分析等。
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SensorFusion")

class SensorType(Enum):
    """传感器类型枚举"""
    IMU = "imu"                      # 惯性测量单元
    FORCE_TORQUE = "force_torque"    # 力/力矩传感器
    FOOT_PRESSURE = "foot_pressure"  # 足底压力传感器
    JOINT_ENCODER = "joint_encoder"  # 关节编码器
    VISUAL = "visual"                # 视觉传感器
    PROXIMITY = "proximity"          # 接近传感器
    BATTERY = "battery"              # 电池传感器

class FusionState(Enum):
    """融合状态枚举"""
    IDLE = "idle"                    # 空闲
    INITIALIZING = "initializing"    # 初始化
    CALIBRATING = "calibrating"      # 校准
    RUNNING = "running"              # 运行中
    ERROR = "error"                  # 错误

@dataclass
class SensorData:
    """传感器数据基类"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    quality: float = 1.0  # 数据质量 (0-1)
    fused: bool = False   # 是否已融合

@dataclass
class IMUData(SensorData):
    """IMU传感器数据"""
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 9.81)  # 加速度 (m/s²)
    gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)     # 角速度 (rad/s)
    magnetometer: Optional[Tuple[float, float, float]] = None  # 磁力计数据
    temperature: Optional[float] = None       # 温度

@dataclass
class ForceTorqueData(SensorData):
    """力/力矩传感器数据"""
    forces: Tuple[float, float, float] = (0.0, 0.0, 0.0)        # 力 (N)
    torques: Tuple[float, float, float] = (0.0, 0.0, 0.0)       # 力矩 (Nm)
    center_of_pressure: Optional[Tuple[float, float]] = None  # 压力中心

@dataclass
class FootPressureData(SensorData):
    """足底压力传感器数据"""
    pressures: List[float] = None                    # 各点压力值
    total_force: float = 0.0                        # 总压力
    center_of_pressure: Tuple[float, float] = (0.0, 0.0)   # 压力中心

@dataclass
class FusionResult:
    """融合结果"""
    timestamp: float
    state_estimate: Dict[str, Any]           # 状态估计
    confidence: Dict[str, float]             # 置信度
    fused_sensors: List[str]                 # 参与融合的传感器
    cycle_time: float                        # 融合周期时间

class SensorFusionEngine:
    """传感器融合引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化传感器融合引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 融合参数
        self.fusion_params = {
            "update_rate": 100,               # 更新频率 (Hz)
            "buffer_size": 100,               # 缓冲区大小
            "calibration_samples": 1000,      # 校准样本数
            "filter_cutoff": 10.0,            # 滤波器截止频率 (Hz)
            "outlier_threshold": 3.0,         # 离群值阈值 (标准差)
        }
        
        # 传感器缓冲区
        self.sensor_buffers = {
            SensorType.IMU: [],
            SensorType.FORCE_TORQUE: [],
            SensorType.FOOT_PRESSURE: [],
            SensorType.JOINT_ENCODER: [],
        }
        
        # 状态估计
        self.state_estimate = {
            "pose": {
                "position": np.zeros(3),      # 位置 (x, y, z)
                "orientation": np.array([0, 0, 0, 1]),  # 四元数 (x, y, z, w)
                "velocity": np.zeros(3),      # 线速度
                "angular_velocity": np.zeros(3),  # 角速度
            },
            "balance": {
                "center_of_mass": np.zeros(3),  # 质心位置
                "center_of_pressure": np.zeros(2),  # 压力中心
                "margin_of_stability": 0.0,    # 稳定裕度
                "unstable": False,             # 是否不稳定
            },
            "gait": {
                "phase": "stance",             # 步态相位
                "stride_length": 0.6,          # 步幅长度
                "cadence": 90,                 # 步频 (步/分钟)
                "step_time": 0.5,              # 步时 (秒)
            },
            "contact": {
                "left_foot": False,            # 左脚接触
                "right_foot": False,           # 右脚接触
                "force_ratio": 0.5,            # 力分布比例
            }
        }
        
        # 卡尔曼滤波器参数
        self.kalman_params = {
            "process_noise": np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]),
            "measurement_noise": np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
        }
        
        # 融合状态
        self.fusion_state = FusionState.IDLE
        
        # 线程锁
        self.buffer_lock = threading.Lock()
        
        # 初始化滤波器
        self._initialize_filters()
        
        logger.info("传感器融合引擎初始化完成")
    
    def _initialize_filters(self):
        """初始化滤波器"""
        # 低通滤波器参数
        self.filters = {
            "imu_acceleration": self._create_lowpass_filter(cutoff=20.0),
            "imu_gyroscope": self._create_lowpass_filter(cutoff=20.0),
            "force": self._create_lowpass_filter(cutoff=10.0),
            "pressure": self._create_lowpass_filter(cutoff=5.0),
        }
        
        # 卡尔曼滤波器状态
        self.kalman_state = {
            "x": np.zeros(6),  # 状态向量: [位置, 速度]
            "P": np.eye(6),    # 协方差矩阵
            "Q": self.kalman_params["process_noise"],  # 过程噪声
            "R": self.kalman_params["measurement_noise"],  # 测量噪声
        }
    
    def _create_lowpass_filter(self, cutoff: float, fs: float = 100.0) -> Dict[str, Any]:
        """创建低通滤波器
        
        Args:
            cutoff: 截止频率 (Hz)
            fs: 采样频率 (Hz)
            
        Returns:
            滤波器参数
        """
        # 简化的IIR低通滤波器
        dt = 1.0 / fs
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (rc + dt)
        
        return {
            "alpha": alpha,
            "previous_value": 0.0,
            "initialized": False
        }
    
    def _lowpass_filter(self, value: float, filter_config: Dict[str, Any]) -> float:
        """应用低通滤波器
        
        Args:
            value: 输入值
            filter_config: 滤波器配置
            
        Returns:
            滤波后的值
        """
        if not filter_config["initialized"]:
            filter_config["previous_value"] = value
            filter_config["initialized"] = True
            return value
        
        filtered = filter_config["alpha"] * value + (1 - filter_config["alpha"]) * filter_config["previous_value"]
        filter_config["previous_value"] = filtered
        return filtered
    
    def start(self):
        """启动融合引擎"""
        if self.fusion_state == FusionState.RUNNING:
            logger.warning("融合引擎已在运行")
            return False
        
        logger.info("启动传感器融合引擎")
        self.fusion_state = FusionState.INITIALIZING
        
        # 执行校准
        if self._perform_calibration():
            self.fusion_state = FusionState.RUNNING
            logger.info("传感器融合引擎启动成功")
            return True
        else:
            self.fusion_state = FusionState.ERROR
            logger.error("传感器融合引擎启动失败")
            return False
    
    def stop(self):
        """停止融合引擎"""
        logger.info("停止传感器融合引擎")
        self.fusion_state = FusionState.IDLE
        return True
    
    def _perform_calibration(self) -> bool:
        """执行传感器校准"""
        logger.info("开始传感器校准...")
        
        # 这里应该实现实际的校准逻辑
        # 目前仅返回成功
        time.sleep(0.1)  # 模拟校准时间
        
        logger.info("传感器校准完成")
        return True
    
    def add_sensor_data(self, sensor_data: SensorData) -> bool:
        """添加传感器数据到缓冲区
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            是否成功添加
        """
        try:
            with self.buffer_lock:
                buffer = self.sensor_buffers.get(sensor_data.sensor_type)
                if buffer is not None:
                    buffer.append(sensor_data)
                    
                    # 保持缓冲区大小
                    if len(buffer) > self.fusion_params["buffer_size"]:
                        buffer.pop(0)
                    
                    logger.debug(f"添加 {sensor_data.sensor_type.value} 数据到缓冲区: {sensor_data.sensor_id}")
                    return True
                else:
                    logger.warning(f"未知传感器类型: {sensor_data.sensor_type}")
                    return False
                    
        except Exception as e:
            logger.error(f"添加传感器数据失败: {e}")
            return False
    
    def fuse_sensor_data(self, sensor_data: Optional[Dict[str, Any]] = None) -> FusionResult:
        """融合传感器数据
        
        Args:
            sensor_data: 可选的原始传感器数据字典
            
        Returns:
            融合结果
        """
        if self.fusion_state != FusionState.RUNNING:
            logger.warning("融合引擎未运行，返回默认融合结果")
            return self._get_default_fusion_result()
        
        start_time = time.time()
        
        try:
            # 如果有传入的原始数据，先添加到缓冲区
            if sensor_data:
                self._process_raw_sensor_data(sensor_data)
            
            # 从缓冲区获取最新数据
            latest_data = self._get_latest_sensor_data()
            
            # 执行数据融合
            fused_state = self._perform_fusion(latest_data)
            
            # 更新状态估计
            self._update_state_estimate(fused_state)
            
            # 计算融合周期时间
            cycle_time = time.time() - start_time
            
            # 创建融合结果
            result = FusionResult(
                timestamp=time.time(),
                state_estimate=self.state_estimate.copy(),
                confidence=self._calculate_confidence(latest_data),
                fused_sensors=list(latest_data.keys()),
                cycle_time=cycle_time
            )
            
            logger.debug(f"传感器数据融合完成: {len(result.fused_sensors)} 个传感器, 周期时间: {cycle_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"传感器数据融合失败: {e}")
            return self._get_default_fusion_result()
    
    def _process_raw_sensor_data(self, raw_data: Dict[str, Any]):
        """处理原始传感器数据"""
        # IMU数据
        if "imu" in raw_data:
            imu_data = raw_data["imu"]
            sensor_data = IMUData(
                sensor_id="imu_9dof",
                sensor_type=SensorType.IMU,
                timestamp=time.time(),
                data=imu_data,
                acceleration=imu_data.get("acceleration", [0, 0, 9.81]),
                gyroscope=imu_data.get("gyroscope", [0, 0, 0]),
                magnetometer=imu_data.get("magnetometer", None),
                temperature=imu_data.get("temperature", None)
            )
            self.add_sensor_data(sensor_data)
        
        # 力传感器数据
        if "force_sensors" in raw_data:
            force_data = raw_data["force_sensors"]
            
            # 左脚力传感器
            if "left_foot" in force_data:
                pressures = force_data["left_foot"]
                if isinstance(pressures, list) and len(pressures) >= 4:
                    total_force = sum(pressures[:4])
                    sensor_data = FootPressureData(
                        sensor_id="foot_pressure_left",
                        sensor_type=SensorType.FOOT_PRESSURE,
                        timestamp=time.time(),
                        data={"pressures": pressures},
                        pressures=pressures[:4],
                        total_force=total_force,
                        center_of_pressure=self._calculate_cop(pressures[:4])
                    )
                    self.add_sensor_data(sensor_data)
            
            # 右脚力传感器
            if "right_foot" in force_data:
                pressures = force_data["right_foot"]
                if isinstance(pressures, list) and len(pressures) >= 4:
                    total_force = sum(pressures[:4])
                    sensor_data = FootPressureData(
                        sensor_id="foot_pressure_right",
                        sensor_type=SensorType.FOOT_PRESSURE,
                        timestamp=time.time(),
                        data={"pressures": pressures},
                        pressures=pressures[:4],
                        total_force=total_force,
                        center_of_pressure=self._calculate_cop(pressures[:4])
                    )
                    self.add_sensor_data(sensor_data)
    
    def _calculate_cop(self, pressures: List[float]) -> Tuple[float, float]:
        """计算压力中心
        
        Args:
            pressures: 四个点的压力值 [前左, 前右, 后左, 后右]
            
        Returns:
            (x, y) 压力中心坐标
        """
        if len(pressures) != 4:
            return (0.0, 0.0)
        
        total = sum(pressures)
        if total == 0:
            return (0.0, 0.0)
        
        # 假设传感器位置（相对坐标系）
        positions = [
            (-0.05, 0.08),   # 前左
            (0.05, 0.08),    # 前右
            (-0.05, -0.08),  # 后左
            (0.05, -0.08)    # 后右
        ]
        
        x = sum(p * pos[0] for p, pos in zip(pressures, positions)) / total
        y = sum(p * pos[1] for p, pos in zip(pressures, positions)) / total
        
        return (x, y)
    
    def _get_latest_sensor_data(self) -> Dict[str, SensorData]:
        """获取最新的传感器数据"""
        latest_data = {}
        
        with self.buffer_lock:
            for sensor_type, buffer in self.sensor_buffers.items():
                if buffer:
                    latest_data[sensor_type.value] = buffer[-1]
        
        return latest_data
    
    def _perform_fusion(self, sensor_data: Dict[str, SensorData]) -> Dict[str, Any]:
        """执行数据融合"""
        fused_state = {
            "pose": self.state_estimate["pose"].copy(),
            "balance": self.state_estimate["balance"].copy(),
            "gait": self.state_estimate["gait"].copy(),
            "contact": self.state_estimate["contact"].copy(),
        }
        
        # IMU数据融合
        if "imu" in sensor_data:
            imu_data = sensor_data["imu"]
            if isinstance(imu_data, IMUData):
                fused_state = self._fuse_imu_data(imu_data, fused_state)
        
        # 足底压力数据融合
        if "foot_pressure" in sensor_data:
            pressure_data = sensor_data["foot_pressure"]
            if isinstance(pressure_data, FootPressureData):
                fused_state = self._fuse_foot_pressure_data(pressure_data, fused_state)
        
        return fused_state
    
    def _fuse_imu_data(self, imu_data: IMUData, state: Dict[str, Any]) -> Dict[str, Any]:
        """融合IMU数据"""
        # 应用低通滤波
        filtered_acc = []
        for i, acc in enumerate(imu_data.acceleration):
            filtered = self._lowpass_filter(acc, self.filters["imu_acceleration"])
            filtered_acc.append(filtered)
        
        filtered_gyro = []
        for i, gyro in enumerate(imu_data.gyroscope):
            filtered = self._lowpass_filter(gyro, self.filters["imu_gyroscope"])
            filtered_gyro.append(filtered)
        
        # 更新状态估计
        state["pose"]["velocity"] = np.array(filtered_acc)  # 简化的速度估计
        state["pose"]["angular_velocity"] = np.array(filtered_gyro)
        
        # 简单的姿态估计（基于加速度计）
        acc_norm = np.linalg.norm(filtered_acc)
        if acc_norm > 0.1:  # 避免除零
            # 计算重力方向
            gravity_dir = np.array(filtered_acc) / acc_norm
            
            # 简化的姿态估计（pitch和roll）
            pitch = np.arctan2(-gravity_dir[0], np.sqrt(gravity_dir[1]**2 + gravity_dir[2]**2))
            roll = np.arctan2(gravity_dir[1], gravity_dir[2])
            
            # 更新四元数（简化的转换）
            # 这里应该使用更精确的姿态解算算法
            state["pose"]["orientation"] = self._euler_to_quaternion(pitch, roll, 0)
        
        return state
    
    def _fuse_foot_pressure_data(self, pressure_data: FootPressureData, state: Dict[str, Any]) -> Dict[str, Any]:
        """融合足底压力数据"""
        # 检测接触状态
        left_contact = "left" in pressure_data.sensor_id and pressure_data.total_force > 10.0
        right_contact = "right" in pressure_data.sensor_id and pressure_data.total_force > 10.0
        
        # 更新接触状态
        if "left" in pressure_data.sensor_id:
            state["contact"]["left_foot"] = left_contact
            state["balance"]["center_of_pressure"] = pressure_data.center_of_pressure
        
        if "right" in pressure_data.sensor_id:
            state["contact"]["right_foot"] = right_contact
            # 可以融合左右脚的压力中心
        
        # 更新力分布比例
        # 这里需要左右脚的数据才能计算
        
        # 更新稳定裕度（简化的计算）
        cop_distance = np.linalg.norm(pressure_data.center_of_pressure)
        state["balance"]["margin_of_stability"] = max(0.1 - cop_distance, 0.0)
        
        # 检测不稳定状态
        state["balance"]["unstable"] = (cop_distance > 0.08) or (pressure_data.total_force < 20.0)
        
        return state
    
    def _euler_to_quaternion(self, pitch: float, roll: float, yaw: float) -> np.ndarray:
        """欧拉角转四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        return np.array([x, y, z, w])
    
    def _update_state_estimate(self, fused_state: Dict[str, Any]):
        """更新状态估计"""
        # 简单的状态更新，可以加入卡尔曼滤波
        alpha = 0.1  # 融合因子
        
        for key in self.state_estimate:
            if key in fused_state:
                if isinstance(self.state_estimate[key], dict):
                    for sub_key in self.state_estimate[key]:
                        if sub_key in fused_state[key]:
                            if isinstance(self.state_estimate[key][sub_key], np.ndarray):
                                self.state_estimate[key][sub_key] = (
                                    alpha * np.array(fused_state[key][sub_key]) + 
                                    (1 - alpha) * self.state_estimate[key][sub_key]
                                )
                            else:
                                self.state_estimate[key][sub_key] = fused_state[key][sub_key]
                else:
                    self.state_estimate[key] = fused_state[key]
    
    def _calculate_confidence(self, sensor_data: Dict[str, SensorData]) -> Dict[str, float]:
        """计算置信度"""
        confidence = {
            "pose": 0.7,
            "balance": 0.8,
            "gait": 0.6,
            "contact": 0.9,
        }
        
        # 根据传感器数据调整置信度
        if "imu" in sensor_data:
            confidence["pose"] = min(confidence["pose"] + 0.2, 1.0)
        
        if "foot_pressure" in sensor_data:
            confidence["balance"] = min(confidence["balance"] + 0.1, 1.0)
            confidence["contact"] = min(confidence["contact"] + 0.1, 1.0)
        
        return confidence
    
    def _get_default_fusion_result(self) -> FusionResult:
        """获取默认融合结果（当融合失败时）"""
        return FusionResult(
            timestamp=time.time(),
            state_estimate=self.state_estimate.copy(),
            confidence={
                "pose": 0.5,
                "balance": 0.5,
                "gait": 0.5,
                "contact": 0.5,
            },
            fused_sensors=[],
            cycle_time=0.001
        )
    
    def get_state_estimate(self) -> Dict[str, Any]:
        """获取当前状态估计"""
        return self.state_estimate.copy()
    
    def get_fusion_state(self) -> FusionState:
        """获取融合状态"""
        return self.fusion_state
    
    def reset(self):
        """重置融合引擎"""
        logger.info("重置传感器融合引擎")
        
        with self.buffer_lock:
            for buffer in self.sensor_buffers.values():
                buffer.clear()
        
        self._initialize_filters()
        self.fusion_state = FusionState.IDLE
        
        logger.info("传感器融合引擎重置完成")


# 全局融合引擎实例
_fusion_engine = None

def get_fusion_engine(config: Optional[Dict[str, Any]] = None) -> SensorFusionEngine:
    """获取融合引擎实例（单例模式）"""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = SensorFusionEngine(config)
    return _fusion_engine


# 测试函数
def test_sensor_fusion():
    """测试传感器融合引擎"""
    print("=" * 80)
    print("测试传感器融合引擎")
    print("=" * 80)
    
    # 创建融合引擎
    engine = get_fusion_engine()
    
    # 启动引擎
    print("1. 启动融合引擎...")
    started = engine.start()
    print(f"   启动结果: {'成功' if started else '失败'}")
    
    # 创建模拟传感器数据
    print("\n2. 添加模拟传感器数据...")
    
    # IMU数据
    imu_data = IMUData(
        sensor_id="imu_9dof",
        sensor_type=SensorType.IMU,
        timestamp=time.time(),
        data={"simulated": True},
        acceleration=(0.05, 0.1, 9.81),
        gyroscope=(0.005, 0.01, 0.002),
        magnetometer=(20.5, 15.3, 45.7),
        temperature=25.0
    )
    added = engine.add_sensor_data(imu_data)
    print(f"   IMU数据添加: {'成功' if added else '失败'}")
    
    # 足底压力数据（左脚）
    pressure_data_left = FootPressureData(
        sensor_id="foot_pressure_left",
        sensor_type=SensorType.FOOT_PRESSURE,
        timestamp=time.time(),
        data={"simulated": True},
        pressures=[25.2, 24.7, 26.5, 25.1],
        total_force=101.5,
        center_of_pressure=(-0.002, 0.003)
    )
    added = engine.add_sensor_data(pressure_data_left)
    print(f"   左脚压力数据添加: {'成功' if added else '失败'}")
    
    # 执行融合
    print("\n3. 执行传感器数据融合...")
    fusion_result = engine.fuse_sensor_data()
    print(f"   融合完成，使用传感器: {len(fusion_result.fused_sensors)} 个")
    print(f"   融合周期时间: {fusion_result.cycle_time*1000:.2f}ms")
    
    # 显示融合结果
    print("\n4. 融合结果:")
    pose = fusion_result.state_estimate["pose"]
    balance = fusion_result.state_estimate["balance"]
    
    print(f"   姿态估计:")
    print(f"     位置: {pose['position'][0]:.3f}, {pose['position'][1]:.3f}, {pose['position'][2]:.3f}")
    print(f"     线速度: {pose['velocity'][0]:.3f}, {pose['velocity'][1]:.3f}, {pose['velocity'][2]:.3f} m/s")
    
    print(f"   平衡状态:")
    print(f"     压力中心: {balance['center_of_pressure'][0]:.4f}, {balance['center_of_pressure'][1]:.4f}")
    print(f"     稳定裕度: {balance['margin_of_stability']:.3f}")
    print(f"     不稳定: {'是' if balance['unstable'] else '否'}")
    
    print(f"   置信度:")
    for key, value in fusion_result.confidence.items():
        print(f"     {key}: {value:.2f}")
    
    # 获取状态估计
    print("\n5. 获取当前状态估计...")
    state = engine.get_state_estimate()
    print(f"   状态获取成功，包含 {len(state)} 个状态类别")
    
    # 停止引擎
    print("\n6. 停止融合引擎...")
    stopped = engine.stop()
    print(f"   停止结果: {'成功' if stopped else '失败'}")
    
    print("\n✅ 传感器融合引擎测试完成")


if __name__ == "__main__":
    test_sensor_fusion()