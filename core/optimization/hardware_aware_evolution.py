#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件感知演化模块 - Hardware-Aware Evolution Module

基于当前硬件环境（GPU/TPU/CPU/边缘设备）自动调整演化策略，
针对不同硬件特性优化网络架构设计和演化参数。

主要功能：
1. 硬件检测和特征提取
2. 硬件感知演化参数调整
3. 针对硬件的网络架构优化
4. 动态硬件适配和降级策略
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """硬件类型枚举"""
    HIGH_END_GPU = "high_end_gpu"      # 高端GPU (如NVIDIA A100, H100)
    MID_RANGE_GPU = "mid_range_gpu"    # 中端GPU (如RTX 3080, 4090)
    LOW_END_GPU = "low_end_gpu"        # 低端GPU (如GTX 1060, RTX 3050)
    APPLE_SILICON = "apple_silicon"    # Apple Silicon (M系列芯片)
    AMD_GPU = "amd_gpu"                # AMD GPU
    INTEL_GPU = "intel_gpu"            # Intel GPU
    TPU = "tpu"                        # Google TPU
    HIGH_END_CPU = "high_end_cpu"      # 高端CPU (服务器级)
    MID_RANGE_CPU = "mid_range_cpu"    # 中端CPU (桌面级)
    LOW_END_CPU = "low_end_cpu"        # 低端CPU (移动设备)
    EDGE_DEVICE = "edge_device"        # 边缘设备 (如Jetson, Raspberry Pi)


class HardwareOptimizationLevel(Enum):
    """硬件优化级别"""
    MAX_PERFORMANCE = "max_performance"      # 最大性能（忽略功耗）
    BALANCED = "balanced"                    # 平衡性能与效率
    POWER_EFFICIENT = "power_efficient"      # 功耗优先
    EDGE_OPTIMIZED = "edge_optimized"        # 边缘设备优化


@dataclass
class HardwareFeatures:
    """硬件特征"""
    hardware_type: HardwareType
    memory_gb: float                         # 内存容量 (GB)
    compute_units: int                       # 计算单元数 (GPU核心/CPU核心)
    has_tensor_cores: bool = False           # 是否有张量核心
    has_fp16_acceleration: bool = False      # 是否支持FP16加速
    has_int8_acceleration: bool = False      # 是否支持INT8加速
    is_edge_device: bool = False             # 是否为边缘设备
    power_limit_watts: Optional[float] = None  # 功率限制 (瓦特)
    thermal_limit_celsius: Optional[float] = None  # 温度限制 (摄氏度)


@dataclass
class HardwareAwareEvolutionConfig:
    """硬件感知演化配置"""
    # 硬件优化级别
    optimization_level: HardwareOptimizationLevel = HardwareOptimizationLevel.BALANCED
    
    # 架构限制
    max_model_size_mb: float = 500.0          # 最大模型大小 (MB)
    max_parameters_millions: float = 100.0    # 最大参数量 (百万)
    max_memory_usage_gb: float = 8.0          # 最大内存使用量 (GB)
    
    # 演化参数调整范围
    population_size_range: Tuple[int, int] = (20, 200)      # 种群大小范围
    mutation_rate_range: Tuple[float, float] = (0.01, 0.3)  # 突变率范围
    crossover_rate_range: Tuple[float, float] = (0.5, 0.9)  # 交叉率范围
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)  # 学习率范围
    
    # 硬件特定参数
    gpu_optimization_boost: float = 1.2       # GPU优化提升系数
    cpu_optimization_reduction: float = 0.7   # CPU优化降低系数
    edge_device_optimization_reduction: float = 0.4  # 边缘设备优化降低系数
    
    # 架构特征限制
    max_layers_range: Tuple[int, int] = (3, 20)      # 最大层数范围
    max_layer_size_range: Tuple[int, int] = (32, 1024)  # 最大层大小范围
    allowed_operation_types: List[str] = field(default_factory=lambda: [
        "conv1d", "conv2d", "conv3d", "linear", "lstm", "gru", "attention"
    ])


class HardwareAwareEvolutionModule:
    """硬件感知演化模块"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化硬件感知演化模块
        
        Args:
            config: 配置字典
        """
        # 处理配置，转换字符串为枚举
        processed_config = config.copy() if config else {}
        
        # 转换优化级别字符串为枚举
        if "optimization_level" in processed_config and isinstance(processed_config["optimization_level"], str):
            optimization_str = processed_config["optimization_level"].upper()
            try:
                processed_config["optimization_level"] = HardwareOptimizationLevel[optimization_str]
            except KeyError:
                logger.warning(f"未知优化级别: {optimization_str}，使用默认BALANCED")
                processed_config["optimization_level"] = HardwareOptimizationLevel.BALANCED
        
        self.config = HardwareAwareEvolutionConfig(**processed_config)
        self.hardware_features: Optional[HardwareFeatures] = None
        self.hardware_type: Optional[HardwareType] = None
        
        # 检测硬件
        self._detect_hardware()
        
        logger.info(f"硬件感知演化模块初始化完成，硬件类型: {self.hardware_type.value if self.hardware_type else 'unknown'}")
    
    def _detect_hardware(self) -> None:
        """检测硬件并提取特征"""
        try:
            # 尝试使用现有的GPU管理器
            from core.gpu_manager import GPUMemoryManager
            
            gpu_manager = GPUMemoryManager()
            gpu_devices = gpu_manager.gpu_devices
            
            # 分析检测到的设备
            if gpu_devices and len(gpu_devices) > 0:
                # 有GPU设备
                best_gpu = self._select_best_gpu(gpu_devices)
                self.hardware_features = self._extract_gpu_features(best_gpu)
                self.hardware_type = self._classify_hardware_type(self.hardware_features)
                
                logger.info(f"检测到GPU: {best_gpu.get('name', 'unknown')}")
                logger.info(f"硬件特征: {self.hardware_features}")
            
            else:
                # 没有GPU，使用CPU
                self.hardware_features = self._extract_cpu_features()
                self.hardware_type = self._classify_hardware_type(self.hardware_features)
                
                logger.info(f"未检测到GPU，使用CPU")
                logger.info(f"硬件特征: {self.hardware_features}")
        
        except ImportError as e:
            logger.warning(f"GPU管理器不可用: {e}")
            # 回退到基本硬件检测
            self.hardware_features = self._extract_cpu_features()
            self.hardware_type = HardwareType.MID_RANGE_CPU
            
            logger.info(f"使用基本硬件检测，硬件类型: {self.hardware_type.value}")
    
    def _select_best_gpu(self, gpu_devices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """选择最佳GPU设备
        
        Args:
            gpu_devices: GPU设备列表
            
        Returns:
            最佳GPU设备信息
        """
        if not gpu_devices:
            return {}
        
        # 简单评分算法：基于内存大小和计算能力
        best_gpu = None
        best_score = 0.0
        
        for gpu in gpu_devices:
            if gpu.get('device_type') != 'gpu':
                continue
            
            # 基础分
            score = 0.0
            
            # 内存大小分数
            memory_gb = gpu.get('total_memory_gb', 0)
            if memory_gb > 0:
                score += min(memory_gb / 32.0, 1.0)  # 32GB为满分
            
            # 计算能力分数
            compute_capability = gpu.get('compute_capability', 0)
            if compute_capability > 0:
                score += min(compute_capability / 10.0, 1.0)  # 计算能力10.0为满分
            
            # Tensor Core加分
            if gpu.get('tensor_cores', False):
                score += 0.3
            
            # 当前利用率减分
            utilization = gpu.get('utilization', 0)
            if utilization > 0:
                score -= utilization / 100.0 * 0.2  # 最高减0.2分
            
            if score > best_score:
                best_score = score
                best_gpu = gpu
        
        return best_gpu or gpu_devices[0]
    
    def _extract_gpu_features(self, gpu_device: Dict[str, Any]) -> HardwareFeatures:
        """提取GPU特征
        
        Args:
            gpu_device: GPU设备信息
            
        Returns:
            硬件特征
        """
        # 默认值
        memory_gb = gpu_device.get('total_memory_gb', 8.0)
        compute_units = gpu_device.get('multi_processor_count', 100)
        has_tensor_cores = gpu_device.get('tensor_cores', False)
        has_fp16_acceleration = gpu_device.get('fp16', False)
        has_int8_acceleration = gpu_device.get('int8', False)
        
        # 根据GPU名称判断硬件类型
        gpu_name = gpu_device.get('name', '').lower()
        
        return HardwareFeatures(
            hardware_type=self._classify_gpu_type(gpu_name, memory_gb),
            memory_gb=memory_gb,
            compute_units=compute_units,
            has_tensor_cores=has_tensor_cores,
            has_fp16_acceleration=has_fp16_acceleration,
            has_int8_acceleration=has_int8_acceleration,
            is_edge_device=False
        )
    
    def _extract_cpu_features(self) -> HardwareFeatures:
        """提取CPU特征
        
        Returns:
            硬件特征
        """
        import psutil
        import multiprocessing
        
        # 获取CPU信息
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 判断CPU类型
        cpu_type = self._classify_cpu_type(cpu_count, memory_gb)
        
        return HardwareFeatures(
            hardware_type=cpu_type,
            memory_gb=memory_gb,
            compute_units=cpu_count,
            has_tensor_cores=False,
            has_fp16_acceleration=False,
            has_int8_acceleration=True,  # CPU通常支持INT8
            is_edge_device=False
        )
    
    def _classify_gpu_type(self, gpu_name: str, memory_gb: float) -> HardwareType:
        """根据GPU名称和内存分类GPU类型
        
        Args:
            gpu_name: GPU名称
            memory_gb: 内存大小 (GB)
            
        Returns:
            硬件类型
        """
        gpu_name_lower = gpu_name.lower()
        
        # 高端GPU (服务器级)
        high_end_indicators = ['a100', 'h100', 'v100', 'a6000', 'a40', 'a10']
        if any(indicator in gpu_name_lower for indicator in high_end_indicators):
            return HardwareType.HIGH_END_GPU
        
        # 中端GPU (消费级高端)
        mid_range_indicators = ['rtx 4', 'rtx 3', 'rtx 2080', 'rtx 2070', 'rtx 3060', 'rtx 3070', 'rtx 3080', 'rtx 3090', 'rtx 4080', 'rtx 4090']
        if any(indicator in gpu_name_lower for indicator in mid_range_indicators):
            return HardwareType.MID_RANGE_GPU
        
        # 低端GPU
        low_end_indicators = ['gtx', 'rtx 2', 'rtx 3', 'rtx 3050', 'mx', 'integrated']
        if any(indicator in gpu_name_lower for indicator in low_end_indicators):
            return HardwareType.LOW_END_GPU
        
        # Apple Silicon
        if 'mps' in gpu_name_lower or 'apple' in gpu_name_lower or 'm1' in gpu_name_lower or 'm2' in gpu_name_lower or 'm3' in gpu_name_lower:
            return HardwareType.APPLE_SILICON
        
        # AMD GPU
        if 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower:
            return HardwareType.AMD_GPU
        
        # Intel GPU
        if 'intel' in gpu_name_lower:
            return HardwareType.INTEL_GPU
        
        # 根据内存大小判断
        if memory_gb >= 24:
            return HardwareType.HIGH_END_GPU
        elif memory_gb >= 8:
            return HardwareType.MID_RANGE_GPU
        else:
            return HardwareType.LOW_END_GPU
    
    def _classify_cpu_type(self, cpu_cores: int, memory_gb: float) -> HardwareType:
        """根据CPU核心数和内存分类CPU类型
        
        Args:
            cpu_cores: CPU核心数
            memory_gb: 内存大小 (GB)
            
        Returns:
            硬件类型
        """
        # 检查是否为边缘设备
        import platform
        machine = platform.machine().lower()
        edge_architectures = ['arm', 'aarch64', 'armv7l', 'armv8l', 'riscv']
        
        if any(arch in machine for arch in edge_architectures) or memory_gb < 4:
            return HardwareType.EDGE_DEVICE
        
        # 根据核心数和内存分类
        if cpu_cores >= 32 and memory_gb >= 64:
            return HardwareType.HIGH_END_CPU
        elif cpu_cores >= 8 and memory_gb >= 16:
            return HardwareType.MID_RANGE_CPU
        else:
            return HardwareType.LOW_END_CPU
    
    def _classify_hardware_type(self, features: HardwareFeatures) -> HardwareType:
        """根据硬件特征分类硬件类型
        
        Args:
            features: 硬件特征
            
        Returns:
            硬件类型
        """
        return features.hardware_type
    
    def get_hardware_aware_evolution_parameters(self, task_complexity: float = 0.5) -> Dict[str, Any]:
        """获取硬件感知的演化参数
        
        Args:
            task_complexity: 任务复杂性 (0.0-1.0)
            
        Returns:
            演化参数字典
        """
        if not self.hardware_features:
            self._detect_hardware()
        
        # 基础参数（基于硬件类型）
        base_params = self._get_base_evolution_parameters()
        
        # 根据优化级别调整
        adjusted_params = self._adjust_for_optimization_level(base_params)
        
        # 根据任务复杂性调整
        final_params = self._adjust_for_task_complexity(adjusted_params, task_complexity)
        
        # 确保参数在有效范围内
        final_params = self._clamp_parameters(final_params)
        
        logger.info(f"硬件感知演化参数: {final_params}")
        
        return final_params
    
    def _get_base_evolution_parameters(self) -> Dict[str, Any]:
        """获取基于硬件类型的基础演化参数
        
        Returns:
            基础演化参数
        """
        if not self.hardware_type:
            return self._get_default_parameters()
        
        # 不同硬件类型的参数模板
        hardware_templates = {
            HardwareType.HIGH_END_GPU: {
                "population_size": 150,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "learning_rate": 1e-3,
                "max_layers": 15,
                "max_layer_size": 1024,
                "model_size_limit_mb": 500.0,
                "memory_limit_gb": 32.0,
            },
            HardwareType.MID_RANGE_GPU: {
                "population_size": 100,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "learning_rate": 3e-4,
                "max_layers": 12,
                "max_layer_size": 512,
                "model_size_limit_mb": 250.0,
                "memory_limit_gb": 16.0,
            },
            HardwareType.LOW_END_GPU: {
                "population_size": 50,
                "mutation_rate": 0.08,
                "crossover_rate": 0.6,
                "learning_rate": 1e-4,
                "max_layers": 8,
                "max_layer_size": 256,
                "model_size_limit_mb": 100.0,
                "memory_limit_gb": 8.0,
            },
            HardwareType.APPLE_SILICON: {
                "population_size": 80,
                "mutation_rate": 0.12,
                "crossover_rate": 0.75,
                "learning_rate": 5e-4,
                "max_layers": 10,
                "max_layer_size": 384,
                "model_size_limit_mb": 150.0,
                "memory_limit_gb": 16.0,
            },
            HardwareType.HIGH_END_CPU: {
                "population_size": 80,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "learning_rate": 1e-4,
                "max_layers": 10,
                "max_layer_size": 256,
                "model_size_limit_mb": 100.0,
                "memory_limit_gb": 16.0,
            },
            HardwareType.MID_RANGE_CPU: {
                "population_size": 50,
                "mutation_rate": 0.08,
                "crossover_rate": 0.6,
                "learning_rate": 5e-5,
                "max_layers": 8,
                "max_layer_size": 128,
                "model_size_limit_mb": 50.0,
                "memory_limit_gb": 8.0,
            },
            HardwareType.LOW_END_CPU: {
                "population_size": 30,
                "mutation_rate": 0.05,
                "crossover_rate": 0.5,
                "learning_rate": 1e-5,
                "max_layers": 5,
                "max_layer_size": 64,
                "model_size_limit_mb": 25.0,
                "memory_limit_gb": 4.0,
            },
            HardwareType.EDGE_DEVICE: {
                "population_size": 20,
                "mutation_rate": 0.03,
                "crossover_rate": 0.4,
                "learning_rate": 5e-6,
                "max_layers": 3,
                "max_layer_size": 32,
                "model_size_limit_mb": 10.0,
                "memory_limit_gb": 2.0,
            }
        }
        
        return hardware_templates.get(self.hardware_type, self._get_default_parameters())
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认演化参数
        
        Returns:
            默认演化参数
        """
        return {
            "population_size": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "learning_rate": 1e-4,
            "max_layers": 8,
            "max_layer_size": 256,
            "model_size_limit_mb": 100.0,
            "memory_limit_gb": 8.0,
        }
    
    def _adjust_for_optimization_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """根据优化级别调整参数
        
        Args:
            params: 基础参数
            
        Returns:
            调整后的参数
        """
        adjustment_factors = {
            HardwareOptimizationLevel.MAX_PERFORMANCE: {
                "population_size": 1.5,
                "mutation_rate": 1.2,
                "max_layers": 1.3,
                "max_layer_size": 1.5,
                "model_size_limit_mb": 2.0,
                "memory_limit_gb": 2.0,
            },
            HardwareOptimizationLevel.BALANCED: {
                "population_size": 1.0,
                "mutation_rate": 1.0,
                "max_layers": 1.0,
                "max_layer_size": 1.0,
                "model_size_limit_mb": 1.0,
                "memory_limit_gb": 1.0,
            },
            HardwareOptimizationLevel.POWER_EFFICIENT: {
                "population_size": 0.7,
                "mutation_rate": 0.8,
                "max_layers": 0.8,
                "max_layer_size": 0.7,
                "model_size_limit_mb": 0.7,
                "memory_limit_gb": 0.7,
            },
            HardwareOptimizationLevel.EDGE_OPTIMIZED: {
                "population_size": 0.5,
                "mutation_rate": 0.6,
                "max_layers": 0.6,
                "max_layer_size": 0.5,
                "model_size_limit_mb": 0.5,
                "memory_limit_gb": 0.5,
            }
        }
        
        factors = adjustment_factors.get(self.config.optimization_level, adjustment_factors[HardwareOptimizationLevel.BALANCED])
        
        adjusted_params = params.copy()
        for key, factor in factors.items():
            if key in adjusted_params:
                if isinstance(adjusted_params[key], (int, float)):
                    adjusted_params[key] = adjusted_params[key] * factor
        
        return adjusted_params
    
    def _adjust_for_task_complexity(self, params: Dict[str, Any], task_complexity: float) -> Dict[str, Any]:
        """根据任务复杂性调整参数
        
        Args:
            params: 当前参数
            task_complexity: 任务复杂性 (0.0-1.0)
            
        Returns:
            调整后的参数
        """
        # 确保任务复杂性在有效范围内
        task_complexity = max(0.0, min(1.0, task_complexity))
        
        adjusted_params = params.copy()
        
        # 根据任务复杂性调整参数
        # 高复杂性任务需要更大的种群和更多的探索
        complexity_factor = 0.5 + task_complexity  # 范围: 0.5-1.5
        
        if "population_size" in adjusted_params:
            adjusted_params["population_size"] = int(adjusted_params["population_size"] * complexity_factor)
        
        if "mutation_rate" in adjusted_params:
            adjusted_params["mutation_rate"] = adjusted_params["mutation_rate"] * (0.8 + 0.4 * task_complexity)
        
        if "max_layers" in adjusted_params:
            adjusted_params["max_layers"] = int(adjusted_params["max_layers"] * (0.8 + 0.4 * task_complexity))
        
        if "max_layer_size" in adjusted_params:
            adjusted_params["max_layer_size"] = int(adjusted_params["max_layer_size"] * (0.8 + 0.4 * task_complexity))
        
        return adjusted_params
    
    def _clamp_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """确保参数在有效范围内
        
        Args:
            params: 参数字典
            
        Returns:
            裁剪后的参数
        """
        clamped_params = params.copy()
        
        # 种群大小
        if "population_size" in clamped_params:
            clamped_params["population_size"] = int(max(
                self.config.population_size_range[0],
                min(self.config.population_size_range[1], clamped_params["population_size"])
            ))
        
        # 突变率
        if "mutation_rate" in clamped_params:
            clamped_params["mutation_rate"] = max(
                self.config.mutation_rate_range[0],
                min(self.config.mutation_rate_range[1], clamped_params["mutation_rate"])
            )
        
        # 交叉率
        if "crossover_rate" in clamped_params:
            clamped_params["crossover_rate"] = max(
                self.config.crossover_rate_range[0],
                min(self.config.crossover_rate_range[1], clamped_params["crossover_rate"])
            )
        
        # 学习率
        if "learning_rate" in clamped_params:
            clamped_params["learning_rate"] = max(
                self.config.learning_rate_range[0],
                min(self.config.learning_rate_range[1], clamped_params["learning_rate"])
            )
        
        # 最大层数
        if "max_layers" in clamped_params:
            clamped_params["max_layers"] = int(max(
                self.config.max_layers_range[0],
                min(self.config.max_layers_range[1], clamped_params["max_layers"])
            ))
        
        # 最大层大小
        if "max_layer_size" in clamped_params:
            clamped_params["max_layer_size"] = int(max(
                self.config.max_layer_size_range[0],
                min(self.config.max_layer_size_range[1], clamped_params["max_layer_size"])
            ))
        
        # 模型大小限制
        if "model_size_limit_mb" in clamped_params:
            clamped_params["model_size_limit_mb"] = min(
                self.config.max_model_size_mb,
                clamped_params["model_size_limit_mb"]
            )
        
        # 内存限制
        if "memory_limit_gb" in clamped_params:
            clamped_params["memory_limit_gb"] = min(
                self.config.max_memory_usage_gb,
                clamped_params["memory_limit_gb"]
            )
        
        return clamped_params
    
    def optimize_architecture_for_hardware(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """根据硬件特性优化网络架构
        
        Args:
            architecture: 原始网络架构
            
        Returns:
            优化后的网络架构
        """
        if not self.hardware_features:
            return architecture
        
        optimized_architecture = architecture.copy()
        
        # 根据硬件类型调整架构
        if self.hardware_type in [HardwareType.HIGH_END_GPU, HardwareType.MID_RANGE_GPU]:
            # GPU优化：可以使用更大的层和更复杂的结构
            optimized_architecture = self._optimize_for_gpu(optimized_architecture)
        
        elif self.hardware_type in [HardwareType.EDGE_DEVICE, HardwareType.LOW_END_CPU]:
            # 边缘设备优化：使用轻量级结构
            optimized_architecture = self._optimize_for_edge(optimized_architecture)
        
        elif self.hardware_type == HardwareType.APPLE_SILICON:
            # Apple Silicon优化：使用Metal支持的层
            optimized_architecture = self._optimize_for_apple_silicon(optimized_architecture)
        
        return optimized_architecture
    
    def _optimize_for_gpu(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """为GPU优化架构
        
        Args:
            architecture: 原始架构
            
        Returns:
            优化后的架构
        """
        optimized = architecture.copy()
        
        # GPU优化策略
        if "layers" in optimized:
            layers = optimized["layers"]
            
            # 对于GPU，可以增加层大小，使用更复杂的激活函数
            for layer in layers:
                if "size" in layer and layer["size"] < 1024:
                    # 适当增加层大小
                    layer["size"] = min(layer["size"] * 1.5, 1024)
                
                # 使用GPU友好的激活函数
                if "activation" in layer and layer["activation"] == "sigmoid":
                    layer["activation"] = "relu"  # ReLU在GPU上更快
        
        return optimized
    
    def _optimize_for_edge(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """为边缘设备优化架构
        
        Args:
            architecture: 原始架构
            
        Returns:
            优化后的架构
        """
        optimized = architecture.copy()
        
        # 边缘设备优化策略
        if "layers" in optimized:
            layers = optimized["layers"]
            
            # 减少层数和层大小
            if len(layers) > 5:
                optimized["layers"] = layers[:5]  # 限制最多5层
            
            for layer in optimized["layers"]:
                if "size" in layer and layer["size"] > 128:
                    layer["size"] = 128  # 限制层大小
                
                # 使用轻量级激活函数
                if "activation" in layer and layer["activation"] in ["gelu", "swish"]:
                    layer["activation"] = "relu"  # ReLU更轻量
        
        # 添加轻量级优化标志
        optimized["optimizations"] = optimized.get("optimizations", [])
        if "edge_optimized" not in optimized["optimizations"]:
            optimized["optimizations"].append("edge_optimized")
        
        return optimized
    
    def _optimize_for_apple_silicon(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """为Apple Silicon优化架构
        
        Args:
            architecture: 原始架构
            
        Returns:
            优化后的架构
        """
        optimized = architecture.copy()
        
        # Apple Silicon优化策略
        if "layers" in optimized:
            layers = optimized["layers"]
            
            for layer in layers:
                # 使用Metal友好的数据类型
                if "dtype" in layer and layer["dtype"] == "float64":
                    layer["dtype"] = "float32"  # float32在Apple Silicon上更快
                
                # 使用Metal友好的操作
                if layer.get("type") == "conv2d" and "groups" in layer:
                    # Metal对分组卷积支持有限，避免使用
                    if layer["groups"] > 1:
                        layer["groups"] = 1
        
        # 添加Apple Silicon优化标志
        optimized["optimizations"] = optimized.get("optimizations", [])
        if "apple_silicon_optimized" not in optimized["optimizations"]:
            optimized["optimizations"].append("apple_silicon_optimized")
        
        return optimized
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息
        
        Returns:
            硬件信息字典
        """
        if not self.hardware_features:
            self._detect_hardware()
        
        return {
            "hardware_type": self.hardware_type.value if self.hardware_type else "unknown",
            "features": {
                "memory_gb": self.hardware_features.memory_gb if self.hardware_features else 0.0,
                "compute_units": self.hardware_features.compute_units if self.hardware_features else 0,
                "has_tensor_cores": self.hardware_features.has_tensor_cores if self.hardware_features else False,
                "has_fp16_acceleration": self.hardware_features.has_fp16_acceleration if self.hardware_features else False,
                "has_int8_acceleration": self.hardware_features.has_int8_acceleration if self.hardware_features else False,
                "is_edge_device": self.hardware_features.is_edge_device if self.hardware_features else False,
            },
            "config": {
                "optimization_level": self.config.optimization_level.value,
                "max_model_size_mb": self.config.max_model_size_mb,
                "max_memory_usage_gb": self.config.max_memory_usage_gb,
            }
        }


# 工厂函数
def create_hardware_aware_evolution_module(config: Optional[Dict[str, Any]] = None) -> HardwareAwareEvolutionModule:
    """创建硬件感知演化模块实例
    
    Args:
        config: 配置字典
        
    Returns:
        硬件感知演化模块实例
    """
    return HardwareAwareEvolutionModule(config)


if __name__ == "__main__":
    # 演示硬件感知演化模块
    import sys
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("硬件感知演化模块演示")
    print("=" * 80)
    
    try:
        # 创建硬件感知演化模块
        module = create_hardware_aware_evolution_module({
            "optimization_level": "balanced"
        })
        
        # 获取硬件信息
        hardware_info = module.get_hardware_info()
        print(f"\n1. 硬件检测结果:")
        print(f"   硬件类型: {hardware_info['hardware_type']}")
        print(f"   内存: {hardware_info['features']['memory_gb']:.1f} GB")
        print(f"   计算单元: {hardware_info['features']['compute_units']}")
        print(f"   Tensor Cores: {hardware_info['features']['has_tensor_cores']}")
        print(f"   FP16加速: {hardware_info['features']['has_fp16_acceleration']}")
        
        # 获取演化参数
        print(f"\n2. 硬件感知演化参数:")
        for complexity in [0.3, 0.5, 0.8]:
            params = module.get_hardware_aware_evolution_parameters(complexity)
            print(f"\n   任务复杂性: {complexity}")
            print(f"     种群大小: {params['population_size']}")
            print(f"     突变率: {params['mutation_rate']:.3f}")
            print(f"     交叉率: {params['crossover_rate']:.3f}")
            print(f"     学习率: {params['learning_rate']:.6f}")
            print(f"     最大层数: {params['max_layers']}")
            print(f"     最大层大小: {params['max_layer_size']}")
        
        # 测试架构优化
        print(f"\n3. 架构优化测试:")
        sample_architecture = {
            "type": "classification",
            "layers": [
                {"type": "linear", "size": 256, "activation": "relu"},
                {"type": "linear", "size": 128, "activation": "relu"},
                {"type": "linear", "size": 64, "activation": "sigmoid"}
            ]
        }
        
        optimized_architecture = module.optimize_architecture_for_hardware(sample_architecture)
        print(f"   原始架构: {sample_architecture}")
        print(f"   优化后架构: {optimized_architecture}")
        
        print("\n✓ 硬件感知演化模块演示完成")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()