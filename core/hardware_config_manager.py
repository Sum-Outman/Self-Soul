"""
硬件配置管理器 - AGI机器人硬件配置管理

提供机器人硬件配置的动态管理、自动发现、参数优化和自适应调整功能。
支持多种机器人平台和硬件配置，实现硬件配置的实时更新和优化。

主要功能：
1. 硬件配置动态加载和保存
2. 硬件自动发现和识别
3. 配置参数优化和调优
4. 硬件兼容性检查
5. 实时配置更新
6. 配置版本管理
"""

import json
import yaml
import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareConfigManager")

@dataclass
class JointConfig:
    """关节配置数据类"""
    id: str
    name: str
    type: str  # servo, motor, stepper
    min_angle: float = -180.0
    max_angle: float = 180.0
    initial_angle: float = 0.0
    max_speed: float = 100.0  # 度/秒
    max_torque: float = 5.0   # Nm
    protocol: str = "pwm"     # pwm, i2c, spi, can
    address: str = ""         # 硬件地址
    calibration_offset: float = 0.0
    safety_margin: float = 5.0
    
@dataclass
class SensorConfig:
    """传感器配置数据类"""
    id: str
    name: str
    type: str  # imu, accelerometer, gyroscope, temperature, etc.
    protocol: str = "i2c"  # i2c, spi, serial, analog, digital
    address: str = ""
    sampling_rate: float = 100.0  # Hz
    precision: float = 0.01
    calibration_data: Dict[str, Any] = None
    
@dataclass
class CameraConfig:
    """摄像头配置数据类"""
    id: str
    name: str
    type: str  # left, right, depth, rgb
    index: int = 0
    resolution: str = "1920x1080"
    fps: int = 60
    exposure: float = -1.0  # 自动曝光
    gain: float = 0.0
    calibration_matrix: List[List[float]] = None
    distortion_coefficients: List[float] = None
    
@dataclass
class HardwarePlatformConfig:
    """硬件平台配置数据类"""
    platform_id: str
    platform_name: str
    manufacturer: str = ""
    model: str = ""
    description: str = ""
    joints: List[JointConfig] = None
    sensors: List[SensorConfig] = None
    cameras: List[CameraConfig] = None
    communication_protocols: List[str] = None
    power_requirements: Dict[str, Any] = None
    dimensions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.joints is None:
            self.joints = []
        if self.sensors is None:
            self.sensors = []
        if self.cameras is None:
            self.cameras = []
        if self.communication_protocols is None:
            self.communication_protocols = ["serial", "tcp"]
        if self.power_requirements is None:
            self.power_requirements = {"voltage": 12.0, "current": 5.0}
        if self.dimensions is None:
            self.dimensions = {"height": 1.8, "width": 0.5, "depth": 0.3}

class HardwareConfigManager:
    """硬件配置管理器"""
    
    def __init__(self, config_dir: str = "config/hardware"):
        """初始化硬件配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self.platform_configs: Dict[str, HardwarePlatformConfig] = {}
        self.active_platform: Optional[str] = None
        self.custom_configs: Dict[str, Dict[str, Any]] = {}
        
        # 配置锁
        self.config_lock = threading.RLock()
        
        # 硬件自动发现
        self.auto_discovery_enabled = True
        
        # 加载默认配置
        self._load_default_configs()
        
        # 加载保存的配置
        self._load_saved_configs()
        
        logger.info("硬件配置管理器初始化完成")
    
    def _load_default_configs(self):
        """加载默认硬件配置"""
        try:
            # 默认人形机器人配置
            humanoid_config = HardwarePlatformConfig(
                platform_id="humanoid_v1",
                platform_name="标准人形机器人",
                manufacturer="AGI Robotics",
                model="Humanoid v1.0",
                description="标准人形机器人平台，16个自由度，双目视觉，IMU传感器",
                joints=self._create_default_joints(),
                sensors=self._create_default_sensors(),
                cameras=self._create_default_cameras(),
                communication_protocols=["serial", "tcp", "i2c", "spi"],
                power_requirements={"voltage": 24.0, "current": 10.0, "battery_capacity": 10000},
                dimensions={"height": 1.75, "width": 0.45, "depth": 0.25, "weight": 25.0}
            )
            
            self.platform_configs[humanoid_config.platform_id] = humanoid_config
            
            # 四足机器人配置
            quadruped_config = HardwarePlatformConfig(
                platform_id="quadruped_v1",
                platform_name="四足机器人",
                manufacturer="AGI Robotics",
                model="Quadruped v1.0",
                description="四足机器人平台，12个自由度，单目视觉，IMU传感器",
                joints=self._create_quadruped_joints(),
                sensors=self._create_quadruped_sensors(),
                cameras=self._create_quadruped_cameras(),
                communication_protocols=["serial", "tcp", "can"],
                power_requirements={"voltage": 12.0, "current": 8.0, "battery_capacity": 8000},
                dimensions={"height": 0.6, "width": 0.4, "depth": 0.8, "weight": 15.0}
            )
            
            self.platform_configs[quadruped_config.platform_id] = quadruped_config
            
            # 机械臂配置
            arm_config = HardwarePlatformConfig(
                platform_id="robotic_arm_v1",
                platform_name="六轴机械臂",
                manufacturer="AGI Robotics",
                model="Robotic Arm v1.0",
                description="六轴工业机械臂，6个自由度，力传感器，视觉引导",
                joints=self._create_arm_joints(),
                sensors=self._create_arm_sensors(),
                cameras=self._create_arm_cameras(),
                communication_protocols=["tcp", "modbus", "profibus"],
                power_requirements={"voltage": 220.0, "current": 5.0, "battery_capacity": 0},
                dimensions={"height": 1.2, "width": 0.8, "depth": 0.8, "weight": 45.0}
            )
            
            self.platform_configs[arm_config.platform_id] = arm_config
            
            logger.info(f"加载了 {len(self.platform_configs)} 个默认硬件平台配置")
            
        except Exception as e:
            logger.error(f"加载默认配置失败: {e}")
    
    def _create_default_joints(self) -> List[JointConfig]:
        """创建默认人形机器人关节配置"""
        joints = []
        
        # 左臂
        joints.append(JointConfig(
            id="arm_left_shoulder",
            name="左肩关节",
            type="servo",
            min_angle=-180,
            max_angle=180,
            initial_angle=0,
            max_speed=120,
            max_torque=8.0,
            protocol="pwm",
            address="0x01"
        ))
        
        joints.append(JointConfig(
            id="arm_left_elbow",
            name="左肘关节",
            type="servo",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=150,
            max_torque=6.0,
            protocol="pwm",
            address="0x02"
        ))
        
        joints.append(JointConfig(
            id="arm_left_wrist",
            name="左手腕关节",
            type="servo",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=180,
            max_torque=3.0,
            protocol="pwm",
            address="0x03"
        ))
        
        # 右臂
        joints.append(JointConfig(
            id="arm_right_shoulder",
            name="右肩关节",
            type="servo",
            min_angle=-180,
            max_angle=180,
            initial_angle=0,
            max_speed=120,
            max_torque=8.0,
            protocol="pwm",
            address="0x04"
        ))
        
        joints.append(JointConfig(
            id="arm_right_elbow",
            name="右肘关节",
            type="servo",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=150,
            max_torque=6.0,
            protocol="pwm",
            address="0x05"
        ))
        
        joints.append(JointConfig(
            id="arm_right_wrist",
            name="右手腕关节",
            type="servo",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=180,
            max_torque=3.0,
            protocol="pwm",
            address="0x06"
        ))
        
        # 左腿
        joints.append(JointConfig(
            id="leg_left_hip",
            name="左髋关节",
            type="servo",
            min_angle=-45,
            max_angle=45,
            initial_angle=0,
            max_speed=100,
            max_torque=15.0,
            protocol="pwm",
            address="0x07"
        ))
        
        joints.append(JointConfig(
            id="leg_left_knee",
            name="左膝关节",
            type="servo",
            min_angle=0,
            max_angle=90,
            initial_angle=0,
            max_speed=120,
            max_torque=12.0,
            protocol="pwm",
            address="0x08"
        ))
        
        joints.append(JointConfig(
            id="leg_left_ankle",
            name="左踝关节",
            type="servo",
            min_angle=-30,
            max_angle=30,
            initial_angle=0,
            max_speed=150,
            max_torque=10.0,
            protocol="pwm",
            address="0x09"
        ))
        
        # 右腿
        joints.append(JointConfig(
            id="leg_right_hip",
            name="右髋关节",
            type="servo",
            min_angle=-45,
            max_angle=45,
            initial_angle=0,
            max_speed=100,
            max_torque=15.0,
            protocol="pwm",
            address="0x0A"
        ))
        
        joints.append(JointConfig(
            id="leg_right_knee",
            name="右膝关节",
            type="servo",
            min_angle=0,
            max_angle=90,
            initial_angle=0,
            max_speed=120,
            max_torque=12.0,
            protocol="pwm",
            address="0x0B"
        ))
        
        joints.append(JointConfig(
            id="leg_right_ankle",
            name="右踝关节",
            type="servo",
            min_angle=-30,
            max_angle=30,
            initial_angle=0,
            max_speed=150,
            max_torque=10.0,
            protocol="pwm",
            address="0x0C"
        ))
        
        # 头部
        joints.append(JointConfig(
            id="head_pan",
            name="头部水平转动",
            type="servo",
            min_angle=-180,
            max_angle=180,
            initial_angle=0,
            max_speed=200,
            max_torque=2.0,
            protocol="pwm",
            address="0x0D"
        ))
        
        joints.append(JointConfig(
            id="head_tilt",
            name="头部垂直转动",
            type="servo",
            min_angle=-45,
            max_angle=45,
            initial_angle=0,
            max_speed=180,
            max_torque=2.0,
            protocol="pwm",
            address="0x0E"
        ))
        
        # 躯干
        joints.append(JointConfig(
            id="torso_twist",
            name="躯干扭转",
            type="servo",
            min_angle=-30,
            max_angle=30,
            initial_angle=0,
            max_speed=80,
            max_torque=20.0,
            protocol="pwm",
            address="0x0F"
        ))
        
        joints.append(JointConfig(
            id="torso_bend",
            name="躯干弯曲",
            type="servo",
            min_angle=-15,
            max_angle=15,
            initial_angle=0,
            max_speed=60,
            max_torque=25.0,
            protocol="pwm",
            address="0x10"
        ))
        
        return joints
    
    def _load_saved_configs(self):
        """加载保存的配置文件"""
        try:
            config_files = list(self.config_dir.glob("*.json")) + list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            
            for config_file in config_files:
                try:
                    if config_file.suffix == ".json":
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                    else:  # .yaml or .yml
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)
                    
                    # 提取配置ID
                    config_id = config_data.get("platform_id") or config_data.get("id") or config_file.stem
                    
                    # 转换为硬件平台配置
                    if "joints" in config_data or "sensors" in config_data or "cameras" in config_data:
                        # 这看起来像硬件平台配置
                        platform_config = HardwarePlatformConfig(
                            platform_id=config_id,
                            platform_name=config_data.get("platform_name", config_id),
                            manufacturer=config_data.get("manufacturer", ""),
                            model=config_data.get("model", ""),
                            description=config_data.get("description", ""),
                            joints=self._parse_joints_config(config_data.get("joints", [])),
                            sensors=self._parse_sensors_config(config_data.get("sensors", [])),
                            cameras=self._parse_cameras_config(config_data.get("cameras", [])),
                            communication_protocols=config_data.get("communication_protocols", ["serial", "tcp"]),
                            power_requirements=config_data.get("power_requirements", {"voltage": 12.0, "current": 5.0}),
                            dimensions=config_data.get("dimensions", {"height": 1.8, "width": 0.5, "depth": 0.3})
                        )
                        self.platform_configs[config_id] = platform_config
                        logger.info(f"从 {config_file.name} 加载硬件平台配置: {config_id}")
                    else:
                        # 自定义配置
                        self.custom_configs[config_id] = config_data
                        logger.info(f"从 {config_file.name} 加载自定义配置: {config_id}")
                        
                except Exception as e:
                    logger.warning(f"加载配置文件 {config_file} 失败: {e}")
            
            logger.info(f"从 {len(config_files)} 个文件加载了 {len(self.platform_configs)} 个平台配置和 {len(self.custom_configs)} 个自定义配置")
            
        except Exception as e:
            logger.error(f"加载保存的配置失败: {e}")
    
    def _parse_joints_config(self, joints_data: List[Dict[str, Any]]) -> List[JointConfig]:
        """解析关节配置数据"""
        joints = []
        for joint_data in joints_data:
            joint = JointConfig(
                id=joint_data.get("id", ""),
                name=joint_data.get("name", ""),
                type=joint_data.get("type", "servo"),
                min_angle=float(joint_data.get("min_angle", -180.0)),
                max_angle=float(joint_data.get("max_angle", 180.0)),
                initial_angle=float(joint_data.get("initial_angle", 0.0)),
                max_speed=float(joint_data.get("max_speed", 100.0)),
                max_torque=float(joint_data.get("max_torque", 5.0)),
                protocol=joint_data.get("protocol", "pwm"),
                address=joint_data.get("address", ""),
                calibration_offset=float(joint_data.get("calibration_offset", 0.0)),
                safety_margin=float(joint_data.get("safety_margin", 5.0))
            )
            joints.append(joint)
        return joints
    
    def _parse_sensors_config(self, sensors_data: List[Dict[str, Any]]) -> List[SensorConfig]:
        """解析传感器配置数据"""
        sensors = []
        for sensor_data in sensors_data:
            sensor = SensorConfig(
                id=sensor_data.get("id", ""),
                name=sensor_data.get("name", ""),
                type=sensor_data.get("type", "unknown"),
                protocol=sensor_data.get("protocol", "i2c"),
                address=sensor_data.get("address", ""),
                sampling_rate=float(sensor_data.get("sampling_rate", 100.0)),
                precision=float(sensor_data.get("precision", 0.01)),
                calibration_data=sensor_data.get("calibration_data")
            )
            sensors.append(sensor)
        return sensors
    
    def _parse_cameras_config(self, cameras_data: List[Dict[str, Any]]) -> List[CameraConfig]:
        """解析摄像头配置数据"""
        cameras = []
        for camera_data in cameras_data:
            camera = CameraConfig(
                id=camera_data.get("id", ""),
                name=camera_data.get("name", ""),
                type=camera_data.get("type", "rgb"),
                index=int(camera_data.get("index", 0)),
                resolution=camera_data.get("resolution", "1920x1080"),
                fps=int(camera_data.get("fps", 60)),
                exposure=float(camera_data.get("exposure", -1.0)),
                gain=float(camera_data.get("gain", 0.0)),
                calibration_matrix=camera_data.get("calibration_matrix"),
                distortion_coefficients=camera_data.get("distortion_coefficients")
            )
            cameras.append(camera)
        return cameras
    
    def _create_arm_sensors(self) -> List[SensorConfig]:
        """创建机械臂传感器配置"""
        sensors = []
        
        sensors.append(SensorConfig(
            id="force_sensor_gripper",
            name="夹爪力传感器",
            type="force",
            protocol="i2c",
            address="0x30",
            sampling_rate=100.0,
            precision=0.01
        ))
        
        sensors.append(SensorConfig(
            id="torque_sensor_wrist",
            name="手腕扭矩传感器",
            type="torque",
            protocol="analog",
            address="A1",
            sampling_rate=50.0,
            precision=0.1
        ))
        
        sensors.append(SensorConfig(
            id="position_sensor_base",
            name="基座位置传感器",
            type="position",
            protocol="digital",
            address="D3",
            sampling_rate=10.0,
            precision=0.5
        ))
        
        sensors.append(SensorConfig(
            id="current_sensor_motor",
            name="电机电流传感器",
            type="current",
            protocol="analog",
            address="A2",
            sampling_rate=20.0,
            precision=0.05
        ))
        
        return sensors
    
    def _create_arm_cameras(self) -> List[CameraConfig]:
        """创建机械臂摄像头配置"""
        cameras = []
        
        cameras.append(CameraConfig(
            id="camera_gripper",
            name="夹爪摄像头",
            type="rgb",
            index=2,
            resolution="1280x720",
            fps=30,
            exposure=-1.0,
            gain=0.0
        ))
        
        cameras.append(CameraConfig(
            id="camera_overview",
            name="工作区监控摄像头",
            type="rgb",
            index=3,
            resolution="1920x1080",
            fps=30,
            exposure=-1.0,
            gain=0.0
        ))
        
        return cameras
    
    def get_available_platforms(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的硬件平台配置
        
        Returns:
            平台配置字典
        """
        platforms = {}
        for platform_id, platform_config in self.platform_configs.items():
            platforms[platform_id] = {
                "platform_id": platform_id,
                "platform_name": platform_config.platform_name,
                "manufacturer": platform_config.manufacturer,
                "model": platform_config.model,
                "description": platform_config.description,
                "joint_count": len(platform_config.joints),
                "sensor_count": len(platform_config.sensors),
                "camera_count": len(platform_config.cameras)
            }
        return platforms
    
    def save_platform_config(self, platform_id: str, file_format: str = "json") -> bool:
        """保存平台配置到文件
        
        Args:
            platform_id: 平台ID
            file_format: 文件格式 (json, yaml, yml)
            
        Returns:
            保存是否成功
        """
        try:
            if platform_id not in self.platform_configs:
                logger.error(f"平台配置不存在: {platform_id}")
                return False
            
            platform_config = self.platform_configs[platform_id]
            
            # 转换为字典
            config_dict = asdict(platform_config)
            
            # 确定文件路径
            if file_format == "json":
                file_path = self.config_dir / f"{platform_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:  # yaml or yml
                file_path = self.config_dir / f"{platform_id}.{file_format}"
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"平台配置保存到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存平台配置失败: {e}")
            return False
    
    def _create_default_sensors(self) -> List[SensorConfig]:
        """创建默认传感器配置"""
        sensors = []
        
        sensors.append(SensorConfig(
            id="imu_1",
            name="IMU传感器",
            type="imu",
            protocol="i2c",
            address="0x68",
            sampling_rate=100.0,
            precision=0.01,
            calibration_data={
                "accel_bias": [0.0, 0.0, 0.0],
                "gyro_bias": [0.0, 0.0, 0.0],
                "mag_bias": [0.0, 0.0, 0.0]
            }
        ))
        
        sensors.append(SensorConfig(
            id="temp_1",
            name="温度传感器",
            type="temperature",
            protocol="i2c",
            address="0x48",
            sampling_rate=10.0,
            precision=0.1
        ))
        
        sensors.append(SensorConfig(
            id="battery_monitor",
            name="电池监控",
            type="voltage",
            protocol="analog",
            address="A0",
            sampling_rate=5.0,
            precision=0.01
        ))
        
        sensors.append(SensorConfig(
            id="force_left_foot",
            name="左脚力传感器",
            type="force",
            protocol="i2c",
            address="0x20",
            sampling_rate=50.0,
            precision=0.1
        ))
        
        sensors.append(SensorConfig(
            id="force_right_foot",
            name="右脚力传感器",
            type="force",
            protocol="i2c",
            address="0x21",
            sampling_rate=50.0,
            precision=0.1
        ))
        
        sensors.append(SensorConfig(
            id="proximity_front",
            name="前接近传感器",
            type="proximity",
            protocol="digital",
            address="D2",
            sampling_rate=20.0,
            precision=1.0
        ))
        
        return sensors
    
    def _create_default_cameras(self) -> List[CameraConfig]:
        """创建默认摄像头配置"""
        cameras = []
        
        cameras.append(CameraConfig(
            id="camera_left",
            name="左摄像头",
            type="left",
            index=0,
            resolution="1920x1080",
            fps=60,
            exposure=-1.0,
            gain=0.0,
            calibration_matrix=[
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0]
            ],
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0]
        ))
        
        cameras.append(CameraConfig(
            id="camera_right",
            name="右摄像头",
            type="right",
            index=1,
            resolution="1920x1080",
            fps=60,
            exposure=-1.0,
            gain=0.0,
            calibration_matrix=[
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0]
            ],
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0]
        ))
        
        return cameras
    
    def _create_quadruped_joints(self) -> List[JointConfig]:
        """创建四足机器人关节配置"""
        joints = []
        
        # 每条腿3个关节 * 4条腿 = 12个关节
        leg_names = ["front_left", "front_right", "rear_left", "rear_right"]
        
        for leg in leg_names:
            # 髋关节
            joints.append(JointConfig(
                id=f"leg_{leg}_hip",
                name=f"{leg}髋关节",
                type="servo",
                min_angle=-60,
                max_angle=60,
                initial_angle=0,
                max_speed=150,
                max_torque=8.0,
                protocol="pwm"
            ))
            
            # 大腿关节
            joints.append(JointConfig(
                id=f"leg_{leg}_thigh",
                name=f"{leg}大腿关节",
                type="servo",
                min_angle=-45,
                max_angle=45,
                initial_angle=0,
                max_speed=120,
                max_torque=10.0,
                protocol="pwm"
            ))
            
            # 小腿关节
            joints.append(JointConfig(
                id=f"leg_{leg}_calf",
                name=f"{leg}小腿关节",
                type="servo",
                min_angle=0,
                max_angle=90,
                initial_angle=45,
                max_speed=100,
                max_torque=12.0,
                protocol="pwm"
            ))
        
        return joints
    
    def _create_quadruped_sensors(self) -> List[SensorConfig]:
        """创建四足机器人传感器配置"""
        sensors = []
        
        sensors.append(SensorConfig(
            id="imu_quadruped",
            name="IMU传感器",
            type="imu",
            protocol="i2c",
            address="0x68",
            sampling_rate=100.0
        ))
        
        # 每条腿的力传感器
        leg_names = ["front_left", "front_right", "rear_left", "rear_right"]
        for leg in leg_names:
            sensors.append(SensorConfig(
                id=f"force_{leg}",
                name=f"{leg}力传感器",
                type="force",
                protocol="i2c",
                address=f"0x2{leg_names.index(leg)}",
                sampling_rate=50.0
            ))
        
        return sensors
    
    def _create_quadruped_cameras(self) -> List[CameraConfig]:
        """创建四足机器人摄像头配置"""
        cameras = []
        
        cameras.append(CameraConfig(
            id="camera_front",
            name="前向摄像头",
            type="rgb",
            index=0,
            resolution="1280x720",
            fps=30
        ))
        
        return cameras
    
    def _create_arm_joints(self) -> List[JointConfig]:
        """创建机械臂关节配置"""
        joints = []
        
        # 6轴机械臂
        joints.append(JointConfig(
            id="arm_base",
            name="基座旋转",
            type="stepper",
            min_angle=-180,
            max_angle=180,
            initial_angle=0,
            max_speed=60,
            max_torque=50.0,
            protocol="modbus"
        ))
        
        joints.append(JointConfig(
            id="arm_shoulder",
            name="肩关节",
            type="stepper",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=45,
            max_torque=80.0,
            protocol="modbus"
        ))
        
        joints.append(JointConfig(
            id="arm_elbow",
            name="肘关节",
            type="stepper",
            min_angle=-135,
            max_angle=135,
            initial_angle=0,
            max_speed=60,
            max_torque=60.0,
            protocol="modbus"
        ))
        
        joints.append(JointConfig(
            id="arm_wrist_roll",
            name="手腕旋转",
            type="stepper",
            min_angle=-180,
            max_angle=180,
            initial_angle=0,
            max_speed=90,
            max_torque=20.0,
            protocol="modbus"
        ))
        
        joints.append(JointConfig(
            id="arm_wrist_pitch",
            name="手腕俯仰",
            type="stepper",
            min_angle=-90,
            max_angle=90,
            initial_angle=0,
            max_speed=90,
            max_torque=20.0,
            protocol="modbus"
        ))
        
        joints.append(JointConfig(
            id="arm_gripper",
            name="夹爪",
            type="stepper",
            min_angle=0,
            max_angle=180,
            initial_angle=90,
            max_speed=45,
            max_torque=30.0,
            protocol="modbus"
        ))
        
        return joints
