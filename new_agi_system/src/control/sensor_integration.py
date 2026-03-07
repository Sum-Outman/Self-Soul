"""
传感器集成系统

提供传感器数据接入、融合和处理功能。
将原有Self-Soul系统的传感器数据接入功能迁移到统一认知架构中。
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


class SensorFusionMethod(Enum):
    """传感器融合方法枚举"""
    KALMAN_FILTER = "kalman_filter"      # 卡尔曼滤波
    COMPLEMENTARY_FILTER = "complementary_filter"  # 互补滤波
    PARTICLE_FILTER = "particle_filter"  # 粒子滤波
    BAYESIAN_FUSION = "bayesian_fusion"  # 贝叶斯融合
    DEEP_LEARNING = "deep_learning"      # 深度学习融合


@dataclass
class FusedSensorData:
    """融合传感器数据结构"""
    timestamp: float                     # 时间戳
    position: List[float]                # 位置估计
    velocity: List[float]                # 速度估计
    orientation: List[float]             # 方向估计
    confidence: float                    # 置信度
    raw_sensor_ids: List[str]            # 原始传感器ID
    fusion_method: SensorFusionMethod    # 融合方法
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class SensorIntegrationSystem:
    """传感器集成系统"""
    
    def __init__(self, communication):
        """
        初始化传感器集成系统。
        
        参数:
            communication: 神经通信系统
        """
        self.communication = communication
        self.initialized = False
        
        # 传感器配置
        self.config = {
            "fusion_method": SensorFusionMethod.KALMAN_FILTER,
            "fusion_rate": 100,           # 融合频率 (Hz)
            "data_buffer_size": 100,      # 数据缓冲区大小
            "outlier_threshold": 3.0,     # 异常值阈值
            "sensor_calibration": True,   # 传感器校准
            "adaptive_fusion": True,      # 自适应融合
        }
        
        # 传感器数据缓冲区
        self.sensor_buffers: Dict[str, List[Tuple[float, Any]]] = {}
        
        # 融合数据
        self.fused_data_history: List[FusedSensorData] = []
        self.latest_fused_data: Optional[FusedSensorData] = None
        
        # 传感器模型
        self.sensor_models: Dict[str, Dict[str, Any]] = {}
        
        # 融合滤波器
        self.fusion_filters: Dict[str, Any] = {}
        
        # 数据锁
        self.data_lock = threading.RLock()
        
        # 融合任务
        self.fusion_task = None
        self.fusion_active = False
        
        # 性能指标
        self.performance_metrics = {
            'total_sensor_reads': 0,
            'total_fusion_cycles': 0,
            'fusion_errors': 0,
            'average_fusion_time': 0.0,
            'data_loss_rate': 0.0,
            'outlier_detected': 0
        }
        
        logger.info("传感器集成系统已初始化")
    
    async def initialize(self):
        """初始化传感器集成系统"""
        if self.initialized:
            return
        
        logger.info("初始化传感器集成系统...")
        
        # 初始化传感器模型
        await self._initialize_sensor_models()
        
        # 初始化融合滤波器
        await self._initialize_fusion_filters()
        
        # 在通信系统中注册
        await self.communication.register_component(
            component_name="sensor_integration",
            component_type="control"
        )
        
        # 启动融合任务
        await self._start_fusion_task()
        
        self.initialized = True
        logger.info("传感器集成系统初始化完成")
    
    async def _initialize_sensor_models(self):
        """初始化传感器模型"""
        try:
            # IMU传感器模型
            self.sensor_models["imu"] = {
                "type": "imu",
                "noise_covariance": np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]),
                "bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "calibration_matrix": np.eye(6),
                "update_rate": 100,
            }
            
            # 力/扭矩传感器模型
            self.sensor_models["force_torque"] = {
                "type": "force_torque",
                "noise_covariance": np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
                "bias": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "calibration_matrix": np.eye(6),
                "update_rate": 100,
            }
            
            # 接近传感器模型
            self.sensor_models["proximity"] = {
                "type": "proximity",
                "noise_variance": 0.01,
                "range": [0.0, 5.0],
                "update_rate": 50,
            }
            
            logger.info("传感器模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化传感器模型失败: {e}")
    
    async def _initialize_fusion_filters(self):
        """初始化融合滤波器"""
        try:
            if self.config["fusion_method"] == SensorFusionMethod.KALMAN_FILTER:
                await self._initialize_kalman_filter()
            elif self.config["fusion_method"] == SensorFusionMethod.COMPLEMENTARY_FILTER:
                await self._initialize_complementary_filter()
            elif self.config["fusion_method"] == SensorFusionMethod.DEEP_LEARNING:
                await self._initialize_deep_learning_fusion()
            
            logger.info(f"{self.config['fusion_method'].value} 融合滤波器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化融合滤波器失败: {e}")
    
    async def _initialize_kalman_filter(self):
        """初始化卡尔曼滤波器"""
        # 状态维度: [x, y, z, vx, vy, vz, qw, qx, qy, qz] (位置,速度,四元数)
        state_dim = 10
        
        # 简化实现 - 实际系统需要完整的卡尔曼滤波器
        self.fusion_filters["kalman"] = {
            "state_dim": state_dim,
            "measurement_dim": 9,  # 加速度(3) + 角速度(3) + 磁场(3)
            "state": np.zeros(state_dim),
            "covariance": np.eye(state_dim) * 0.1,
            "process_noise": np.eye(state_dim) * 0.01,
            "measurement_noise": np.eye(9) * 0.1,
        }
    
    async def _initialize_complementary_filter(self):
        """初始化互补滤波器"""
        self.fusion_filters["complementary"] = {
            "alpha": 0.98,  # 加速度计权重
            "beta": 0.02,   # 陀螺仪权重
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # 四元数
            "gyro_bias": np.array([0.0, 0.0, 0.0]),
        }
    
    async def _initialize_deep_learning_fusion(self):
        """初始化深度学习融合"""
        # 简化实现 - 实际系统需要神经网络模型
        self.fusion_filters["deep_learning"] = {
            "model_loaded": False,
            "input_dim": 12,  # 6轴IMU + 6轴力/扭矩
            "output_dim": 7,  # 位置(3) + 四元数(4)
            "confidence_threshold": 0.8,
        }
    
    async def _start_fusion_task(self):
        """启动融合任务"""
        if self.fusion_active:
            return
        
        self.fusion_active = True
        self.fusion_task = asyncio.create_task(self._fusion_loop())
        logger.info("传感器融合任务已启动")
    
    async def _fusion_loop(self):
        """融合循环"""
        while self.fusion_active:
            try:
                start_time = time.time()
                
                # 执行传感器融合
                fused_data = await self._perform_sensor_fusion()
                
                if fused_data:
                    # 存储融合数据
                    with self.data_lock:
                        self.latest_fused_data = fused_data
                        self.fused_data_history.append(fused_data)
                        
                        # 保持历史数据大小
                        if len(self.fused_data_history) > self.config["data_buffer_size"]:
                            self.fused_data_history.pop(0)
                    
                    # 更新性能指标
                    fusion_time = time.time() - start_time
                    self._update_performance_metrics(fusion_time)
                
                # 等待下一个融合周期
                await asyncio.sleep(1.0 / self.config["fusion_rate"])
                
            except Exception as e:
                logger.error(f"融合循环错误: {e}")
                self.performance_metrics['fusion_errors'] += 1
                await asyncio.sleep(1.0)
    
    async def _perform_sensor_fusion(self) -> Optional[FusedSensorData]:
        """执行传感器融合"""
        try:
            # 收集传感器数据
            sensor_data = await self._collect_sensor_data()
            
            if not sensor_data:
                logger.debug("无传感器数据可用")
                return None
            
            # 检查数据质量
            valid_data = await self._validate_sensor_data(sensor_data)
            
            if not valid_data:
                logger.warning("传感器数据验证失败")
                self.performance_metrics['outlier_detected'] += 1
                return None
            
            # 执行融合
            fusion_method = self.config["fusion_method"]
            
            if fusion_method == SensorFusionMethod.KALMAN_FILTER:
                fused_state = await self._apply_kalman_filter(sensor_data)
            elif fusion_method == SensorFusionMethod.COMPLEMENTARY_FILTER:
                fused_state = await self._apply_complementary_filter(sensor_data)
            elif fusion_method == SensorFusionMethod.DEEP_LEARNING:
                fused_state = await self._apply_deep_learning_fusion(sensor_data)
            else:
                fused_state = await self._apply_basic_fusion(sensor_data)
            
            # 创建融合数据对象
            fused_data = FusedSensorData(
                timestamp=time.time(),
                position=fused_state.get("position", [0.0, 0.0, 0.0]),
                velocity=fused_state.get("velocity", [0.0, 0.0, 0.0]),
                orientation=fused_state.get("orientation", [1.0, 0.0, 0.0, 0.0]),
                confidence=fused_state.get("confidence", 0.8),
                raw_sensor_ids=list(sensor_data.keys()),
                fusion_method=fusion_method,
                metadata={
                    "processing_time": time.time() - fused_state.get("timestamp", time.time()),
                    "sensor_count": len(sensor_data)
                }
            )
            
            return fused_data
            
        except Exception as e:
            logger.error(f"执行传感器融合失败: {e}")
            return None
    
    async def _collect_sensor_data(self) -> Dict[str, Dict[str, Any]]:
        """收集传感器数据"""
        try:
            # 从硬件控制系统获取传感器数据
            # 这里需要与硬件控制系统通信
            # 简化实现 - 返回模拟数据
            
            sensor_data = {}
            
            # 模拟IMU数据
            sensor_data["imu_001"] = {
                "type": "imu",
                "acceleration": [0.0, 0.0, 9.8 + np.random.normal(0, 0.1)],
                "angular_velocity": [0.0, 0.0, 0.0],
                "magnetic_field": [0.0, 0.0, 1.0],
                "timestamp": time.time(),
                "confidence": 0.9
            }
            
            # 模拟力/扭矩数据
            sensor_data["force_001"] = {
                "type": "force_torque",
                "force": [0.0, 0.0, 50.0 + np.random.normal(0, 1.0)],
                "torque": [0.0, 0.0, 0.0],
                "timestamp": time.time(),
                "confidence": 0.8
            }
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"收集传感器数据失败: {e}")
            return {}
    
    async def _validate_sensor_data(self, sensor_data: Dict[str, Dict[str, Any]]) -> bool:
        """验证传感器数据"""
        try:
            for sensor_id, data in sensor_data.items():
                # 检查时间戳
                timestamp = data.get("timestamp", 0)
                current_time = time.time()
                
                if current_time - timestamp > 1.0:  # 1秒超时
                    logger.warning(f"传感器 {sensor_id} 数据过期")
                    return False
                
                # 检查置信度
                confidence = data.get("confidence", 0.0)
                if confidence < 0.5:
                    logger.warning(f"传感器 {sensor_id} 置信度过低: {confidence}")
                    return False
                
                # 检查异常值
                if not await self._check_outliers(sensor_id, data):
                    logger.warning(f"传感器 {sensor_id} 检测到异常值")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证传感器数据失败: {e}")
            return False
    
    async def _check_outliers(self, sensor_id: str, data: Dict[str, Any]) -> bool:
        """检查异常值"""
        try:
            threshold = self.config["outlier_threshold"]
            
            # 根据传感器类型检查异常值
            sensor_type = data.get("type", "")
            
            if sensor_type == "imu":
                acceleration = data.get("acceleration", [0, 0, 0])
                acc_magnitude = np.linalg.norm(acceleration)
                
                # 检查加速度大小是否合理
                if acc_magnitude > 20.0 or acc_magnitude < 8.0:
                    return False
                    
            elif sensor_type == "force_torque":
                force = data.get("force", [0, 0, 0])
                force_magnitude = np.linalg.norm(force)
                
                # 检查力大小是否合理
                if force_magnitude > 200.0:  # 最大200N
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查异常值失败: {e}")
            return False
    
    async def _apply_kalman_filter(self, sensor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """应用卡尔曼滤波器"""
        try:
            # 简化实现 - 实际系统需要完整的卡尔曼滤波
            filter_state = self.fusion_filters.get("kalman", {})
            
            # 提取测量值
            measurements = []
            
            for data in sensor_data.values():
                if data.get("type") == "imu":
                    acc = data.get("acceleration", [0, 0, 0])
                    gyro = data.get("angular_velocity", [0, 0, 0])
                    mag = data.get("magnetic_field", [0, 0, 1.0])
                    measurements.extend(acc + gyro + mag)
            
            if len(measurements) < 9:
                measurements = [0.0] * 9
            
            # 更新状态
            position = [0.0, 0.0, 0.0]
            velocity = [0.0, 0.0, 0.0]
            orientation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
            
            return {
                "position": position,
                "velocity": velocity,
                "orientation": orientation,
                "confidence": 0.85,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"应用卡尔曼滤波器失败: {e}")
            return self._get_default_fusion_state()
    
    async def _apply_complementary_filter(self, sensor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """应用互补滤波器"""
        try:
            filter_state = self.fusion_filters.get("complementary", {})
            alpha = filter_state.get("alpha", 0.98)
            beta = filter_state.get("beta", 0.02)
            
            # 提取IMU数据
            imu_data = None
            for data in sensor_data.values():
                if data.get("type") == "imu":
                    imu_data = data
                    break
            
            if not imu_data:
                return self._get_default_fusion_state()
            
            # 简化实现 - 实际系统需要四元数运算
            position = [0.0, 0.0, 0.0]
            velocity = [0.0, 0.0, 0.0]
            orientation = filter_state.get("orientation", [1.0, 0.0, 0.0, 0.0])
            
            return {
                "position": position,
                "velocity": velocity,
                "orientation": orientation,
                "confidence": 0.8,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"应用互补滤波器失败: {e}")
            return self._get_default_fusion_state()
    
    async def _apply_deep_learning_fusion(self, sensor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """应用深度学习融合"""
        try:
            # 简化实现 - 实际系统需要神经网络推理
            filter_state = self.fusion_filters.get("deep_learning", {})
            
            # 准备输入数据
            input_data = []
            for data in sensor_data.values():
                if data.get("type") == "imu":
                    acc = data.get("acceleration", [0, 0, 0])
                    gyro = data.get("angular_velocity", [0, 0, 0])
                    input_data.extend(acc + gyro)
                elif data.get("type") == "force_torque":
                    force = data.get("force", [0, 0, 0])
                    torque = data.get("torque", [0, 0, 0])
                    input_data.extend(force + torque)
            
            # 填充不足的维度
            input_dim = filter_state.get("input_dim", 12)
            while len(input_data) < input_dim:
                input_data.append(0.0)
            
            # 简化的"推理"
            position = [0.0, 0.0, 0.0]
            orientation = [1.0, 0.0, 0.0, 0.0]
            confidence = filter_state.get("confidence_threshold", 0.8)
            
            return {
                "position": position,
                "velocity": [0.0, 0.0, 0.0],
                "orientation": orientation,
                "confidence": confidence,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"应用深度学习融合失败: {e}")
            return self._get_default_fusion_state()
    
    async def _apply_basic_fusion(self, sensor_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """应用基本融合"""
        # 简单的加权平均
        try:
            total_weight = 0.0
            weighted_position = [0.0, 0.0, 0.0]
            weighted_orientation = [0.0, 0.0, 0.0, 0.0]
            
            for data in sensor_data.values():
                confidence = data.get("confidence", 0.5)
                
                # 位置估计
                position = data.get("position", [0.0, 0.0, 0.0])
                for i in range(3):
                    weighted_position[i] += position[i] * confidence
                
                # 方向估计
                orientation = data.get("orientation", [1.0, 0.0, 0.0, 0.0])
                for i in range(4):
                    weighted_orientation[i] += orientation[i] * confidence
                
                total_weight += confidence
            
            # 归一化
            if total_weight > 0:
                for i in range(3):
                    weighted_position[i] /= total_weight
                for i in range(4):
                    weighted_orientation[i] /= total_weight
            
            # 归一化四元数
            norm = np.linalg.norm(weighted_orientation)
            if norm > 0:
                weighted_orientation = [q / norm for q in weighted_orientation]
            
            return {
                "position": weighted_position,
                "velocity": [0.0, 0.0, 0.0],
                "orientation": weighted_orientation,
                "confidence": min(1.0, total_weight / len(sensor_data)),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"应用基本融合失败: {e}")
            return self._get_default_fusion_state()
    
    def _get_default_fusion_state(self) -> Dict[str, Any]:
        """获取默认融合状态"""
        return {
            "position": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "confidence": 0.5,
            "timestamp": time.time()
        }
    
    def _update_performance_metrics(self, fusion_time: float):
        """更新性能指标"""
        self.performance_metrics['total_fusion_cycles'] += 1
        self.performance_metrics['average_fusion_time'] = (
            self.performance_metrics['average_fusion_time'] * 
            (self.performance_metrics['total_fusion_cycles'] - 1) + 
            fusion_time
        ) / self.performance_metrics['total_fusion_cycles']
    
    async def get_latest_fused_data(self) -> Optional[FusedSensorData]:
        """获取最新的融合数据"""
        with self.data_lock:
            return self.latest_fused_data
    
    async def get_fused_data_history(self, count: int = 10) -> List[FusedSensorData]:
        """获取融合数据历史"""
        with self.data_lock:
            return self.fused_data_history[-count:] if self.fused_data_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.data_lock:
            return self.performance_metrics.copy()
    
    async def set_fusion_method(self, method: SensorFusionMethod):
        """设置融合方法"""
        try:
            self.config["fusion_method"] = method
            
            # 重新初始化滤波器
            await self._initialize_fusion_filters()
            
            logger.info(f"融合方法已设置为: {method.value}")
            
        except Exception as e:
            logger.error(f"设置融合方法失败: {e}")
    
    async def calibrate_sensors(self, sensor_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """校准传感器"""
        try:
            sensors_to_calibrate = sensor_ids or list(self.sensor_models.keys())
            
            results = {}
            for sensor_id in sensors_to_calibrate:
                # 简化校准过程
                await asyncio.sleep(0.1)  # 模拟校准时间
                
                results[sensor_id] = {
                    "success": True,
                    "calibration_time": 0.1,
                    "bias_correction": [0.0, 0.0, 0.0]
                }
            
            return {
                "success": True,
                "calibrated_sensors": list(results.keys()),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"校准传感器失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def shutdown(self):
        """关闭传感器集成系统"""
        if not self.initialized:
            return
        
        logger.info("关闭传感器集成系统...")
        
        # 停止融合任务
        self.fusion_active = False
        if self.fusion_task:
            self.fusion_task.cancel()
            try:
                await self.fusion_task
            except asyncio.CancelledError:
                pass
        
        # 注销组件
        try:
            await self.communication.unregister_component("sensor_integration")
        except Exception as e:
            logger.warning(f"注销组件失败: {e}")
        
        # 清理数据
        with self.data_lock:
            self.sensor_buffers.clear()
            self.fused_data_history.clear()
            self.latest_fused_data = None
        
        self.initialized = False
        logger.info("传感器集成系统已关闭")