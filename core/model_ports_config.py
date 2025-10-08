"""
模型端口配置文件
存储每个模型的独立运行端口
"""
import os
import json
import logging
from typing import Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelPortsConfig")

# 配置文件路径
CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "model_services_config.json"
)

# 模型端口配置字典
MODEL_PORTS = {}

# 主API网关端口
MAIN_API_PORT = 8000

# 实时数据流管理器端口
REALTIME_STREAM_MANAGER_PORT = 8765

# 性能监控服务端口
PERFORMANCE_MONITORING_PORT = 8081

# 服务设置
SERVICE_SETTINGS = {
    "auto_start": True,
    "max_workers": 5,
    "health_check_interval": 30,
    "timeout": 10,
    "retry_count": 3
}

# 从配置文件加载设置
def load_config_from_file() -> bool:
    """从配置文件加载模型端口配置"""
    global MODEL_PORTS, MAIN_API_PORT, REALTIME_STREAM_MANAGER_PORT, PERFORMANCE_MONITORING_PORT, SERVICE_SETTINGS
    
    try:
        # 检查配置文件是否存在
        if not os.path.exists(CONFIG_FILE_PATH):
            logger.warning(f"配置文件不存在: {CONFIG_FILE_PATH}")
            # 创建默认配置
            create_default_config()
            return False
        
        # 读取配置文件
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 加载模型端口配置
        if "model_ports" in config:
            MODEL_PORTS = config["model_ports"]
            logger.info(f"成功加载 {len(MODEL_PORTS)} 个模型端口配置")
        else:
            logger.warning("配置文件中未找到模型端口配置")
            create_default_config()
        
        # 加载主API端口配置
        if "main_api" in config and "port" in config["main_api"]:
            MAIN_API_PORT = config["main_api"]["port"]
        
        # 加载实时数据流管理器端口配置
        if "realtime_stream_manager" in config and "port" in config["realtime_stream_manager"]:
            REALTIME_STREAM_MANAGER_PORT = config["realtime_stream_manager"]["port"]
        
        # 加载性能监控端口配置
        if "performance_monitoring" in config and "metrics_port" in config["performance_monitoring"]:
            PERFORMANCE_MONITORING_PORT = config["performance_monitoring"]["metrics_port"]
        
        # 加载服务设置
        if "service_settings" in config:
            SERVICE_SETTINGS = config["service_settings"]
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"配置文件解析错误: {str(e)}")
        create_default_config()
        return False
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        create_default_config()
        return False

# 创建默认配置
def create_default_config():
    """创建默认配置文件"""
    global MODEL_PORTS, MAIN_API_PORT, REALTIME_STREAM_MANAGER_PORT, PERFORMANCE_MONITORING_PORT, SERVICE_SETTINGS
    
    try:
        # 默认模型端口配置 - 使用原始配置中的模型和端口
        default_model_ports = {
            'manager': 8001,               # 管理模型端口
            'language': 8002,              # 语言模型端口
            'knowledge': 8003,             # 知识库专家模型端口
            'vision': 8004,                # 视觉模型端口
            'audio': 8005,                 # 音频处理模型端口
            'autonomous': 8006,            # 自主模型端口
            'programming': 8007,           # 编程模型端口
            'planning': 8008,              # 规划模型端口
            'emotion': 8009,               # 情感分析模型端口
            'spatial': 8010,               # 空间感知模型端口
            'computer_vision': 8011,       # 计算机视觉模型端口
            'sensor': 8012,                # 传感器模型端口
            'motion': 8013,                # 运动模型端口
            'prediction': 8014,            # 预测模型端口
            'advanced_reasoning': 8015,    # 高级推理模型端口
            'data_fusion': 8016,           # 数据融合模型端口
            'creative_problem_solving': 8017, # 创造性问题解决模型端口
            'meta_cognition': 8018,        # 元认知模型端口
            'value_alignment': 8019        # 值对齐模型端口
        }
        
        # 默认配置
        default_config = {
            "model_ports": default_model_ports,
            "main_api": {
                "port": MAIN_API_PORT,
                "host": "0.0.0.0",
                "workers": 1
            },
            "realtime_stream_manager": {
                "port": REALTIME_STREAM_MANAGER_PORT,
                "host": "0.0.0.0"
            },
            "service_settings": SERVICE_SETTINGS,
            "logging": {
                "level": "INFO",
                "file_path": "logs/model_services.log",
                "rotation": "1 day",
                "retention": "7 days"
            },
            "performance_monitoring": {
                "enabled": True,
                "metrics_port": PERFORMANCE_MONITORING_PORT,
                "collection_interval": 15
            }
        }
        
        # 确保配置目录存在
        os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
        
        # 写入默认配置文件
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已创建默认配置文件: {CONFIG_FILE_PATH}")
        
        # 更新全局变量
        MODEL_PORTS = default_model_ports
        
    except Exception as e:
        logger.error(f"创建默认配置文件失败: {str(e)}")

# 初始化时加载配置
load_config_from_file()

# 获取模型端口的辅助函数
def get_model_port(model_id: str) -> Optional[int]:
    """获取指定模型的端口号"""
    return MODEL_PORTS.get(model_id)

# 获取所有模型端口的辅助函数
def get_all_model_ports() -> Dict[str, int]:
    """获取所有模型的端口配置"""
    return MODEL_PORTS.copy()

# 更新模型端口配置
def update_model_port(model_id: str, port: int) -> bool:
    """更新指定模型的端口配置"""
    try:
        MODEL_PORTS[model_id] = port
        # 保存配置到文件
        save_config_to_file()
        return True
    except Exception as e:
        logger.error(f"更新模型端口配置失败: {str(e)}")
        return False

# 保存配置到文件
def save_config_to_file() -> bool:
    """保存配置到文件"""
    try:
        # 读取现有配置
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # 更新配置
        config["model_ports"] = MODEL_PORTS
        config["main_api"] = {
            "port": MAIN_API_PORT,
            "host": "0.0.0.0",
            "workers": 1
        }
        config["realtime_stream_manager"] = {
            "port": REALTIME_STREAM_MANAGER_PORT,
            "host": "0.0.0.0"
        }
        config["service_settings"] = SERVICE_SETTINGS
        
        # 写入配置文件
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已保存到文件: {CONFIG_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"保存配置到文件失败: {str(e)}")
        return False