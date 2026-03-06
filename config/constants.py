#!/usr/bin/env python3
"""
统一配置常量管理
集中管理所有魔法数字、端口号、延迟、种子等配置
避免硬编码，提高代码可维护性和可配置性
"""

from enum import Enum
from typing import Dict, Any
import json
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 端口配置 ====================

class Ports(Enum):
    """所有服务端口定义"""
    # 主API端口
    MAIN_API = 8000
    
    # 模型服务端口 (与 model_services_config.json 保持一致)
    MANAGER = 8001
    LANGUAGE = 8002
    KNOWLEDGE = 8003
    VISION = 8004
    AUDIO = 8005
    AUTONOMOUS = 8006
    PROGRAMMING = 8007
    PLANNING = 8008
    EMOTION = 8009
    SPATIAL = 8010
    COMPUTER_VISION = 8011
    SENSOR = 8012
    MOTION = 8013
    PREDICTION = 8014
    ADVANCED_REASONING = 8015
    MULTI_MODEL_COLLABORATION = 8016
    CREATIVE_PROBLEM_SOLVING = 8017
    META_COGNITION = 8018
    VALUE_ALIGNMENT = 8019
    VISION_IMAGE = 8020
    VISION_VIDEO = 8021
    FINANCE = 8022
    MEDICAL = 8023
    COLLABORATION = 8024
    OPTIMIZATION = 8025
    COMPUTER = 8026
    MATHEMATICS = 8027
    DATA_FUSION = 8028
    TRANSLATION = 8029
    
    # 其他服务端口
    REALTIME_STREAM_MANAGER = 8766
    PERFORMANCE_MONITORING = 8080

# 端口映射字典 (便于快速查找)
PORT_MAPPING: Dict[str, int] = {
    'manager': Ports.MANAGER.value,
    'language': Ports.LANGUAGE.value,
    'knowledge': Ports.KNOWLEDGE.value,
    'vision': Ports.VISION.value,
    'audio': Ports.AUDIO.value,
    'autonomous': Ports.AUTONOMOUS.value,
    'programming': Ports.PROGRAMMING.value,
    'planning': Ports.PLANNING.value,
    'emotion': Ports.EMOTION.value,
    'spatial': Ports.SPATIAL.value,
    'computer_vision': Ports.COMPUTER_VISION.value,
    'sensor': Ports.SENSOR.value,
    'motion': Ports.MOTION.value,
    'prediction': Ports.PREDICTION.value,
    'advanced_reasoning': Ports.ADVANCED_REASONING.value,
    'multi_model_collaboration': Ports.MULTI_MODEL_COLLABORATION.value,
    'creative_problem_solving': Ports.CREATIVE_PROBLEM_SOLVING.value,
    'meta_cognition': Ports.META_COGNITION.value,
    'value_alignment': Ports.VALUE_ALIGNMENT.value,
    'vision_image': Ports.VISION_IMAGE.value,
    'vision_video': Ports.VISION_VIDEO.value,
    'finance': Ports.FINANCE.value,
    'medical': Ports.MEDICAL.value,
    'collaboration': Ports.COLLABORATION.value,
    'optimization': Ports.OPTIMIZATION.value,
    'computer': Ports.COMPUTER.value,
    'mathematics': Ports.MATHEMATICS.value,
    'data_fusion': Ports.DATA_FUSION.value,
    'performance_monitoring': Ports.PERFORMANCE_MONITORING.value,
}

# ==================== 随机种子配置 ====================

class Seeds(Enum):
    """确定性随机种子配置"""
    DATASET_GENERATION = 42  # 数据集生成种子
    MODEL_INITIALIZATION = 12345  # 模型初始化种子
    TRAINING_VALIDATION = 777  # 训练验证种子
    
# ==================== 时间延迟配置 ====================

class Delays(Enum):
    """各类延迟配置（单位：秒）"""
    # 训练相关
    TRAINING_MONITOR_CHECK = 5  # 训练监控检查间隔
    TRAINING_MAX_WAIT = 3600  # 训练最大等待时间（1小时）
    
    # 重试机制
    RETRY_INITIAL_DELAY = 1.0  # 重试初始延迟
    RETRY_BACKOFF_FACTOR = 2.0  # 指数退避因子
    
    # 服务启动
    SERVICE_STARTUP_WAIT = 30  # 服务启动等待时间
    SERVICE_HEALTH_CHECK = 10  # 服务健康检查间隔
    
    # 硬件/设备
    HARDWARE_RESPONSE = 0.1  # 硬件响应延迟
    CAMERA_FRAME_INTERVAL = 0.05  # 相机帧间隔
    
# ==================== 数据规模配置 ====================

class DataSizes(Enum):
    """数据集规模配置"""
    STANDARD_DATASET = 5000  # 标准数据集规模
    SUPER_LARGE_DATASET = 50000  # 超大数据集规模
    DIALOGUE_VARIANTS = 30  # 对话变体数量
    MIN_DIALOGUES = 20000  # 最小对话数量
    MIN_SUPER_DIALOGUES = 100000  # 超大数据集最小对话数量
    
# ==================== 训练参数配置 ====================

class TrainingParams(Enum):
    """训练参数配置"""
    DEFAULT_EPOCHS = 10
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_VALIDATION_SPLIT = 0.2
    
# ==================== 服务配置 ====================

class ServiceConfig(Enum):
    """服务相关配置"""
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 10
    WORKER_COUNT = 4
    HEALTH_CHECK_INTERVAL = 30
    
# ==================== 辅助函数 ====================

def get_model_port(model_id: str) -> int:
    """根据模型ID获取端口号"""
    return PORT_MAPPING.get(model_id.lower(), Ports.MAIN_API.value)

def load_config_from_json(file_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    full_path = os.path.join(PROJECT_ROOT, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"警告: 无法加载配置文件 {file_path}: {e}")
        return {}

def get_model_ports_config() -> Dict[str, int]:
    """获取模型端口配置"""
    config = load_config_from_json('config/model_services_config.json')
    return config.get('model_ports', {})

# ==================== 导出常用常量 ====================

# 常用种子
DATASET_SEED = Seeds.DATASET_GENERATION.value

# 常用延迟
TRAINING_CHECK_INTERVAL = Delays.TRAINING_MONITOR_CHECK.value
MAX_TRAINING_WAIT = Delays.TRAINING_MAX_WAIT.value

# 常用端口
MAIN_API_PORT = Ports.MAIN_API.value

# 数据规模
STANDARD_DATA_SIZE = DataSizes.STANDARD_DATASET.value
SUPER_LARGE_DATA_SIZE = DataSizes.SUPER_LARGE_DATASET.value