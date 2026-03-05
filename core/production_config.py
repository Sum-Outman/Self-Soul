"""
生产环境配置和性能优化模块
"""

import os
import psutil
import logging
from typing import Dict, Any
from core.error_handling import error_handler

logger = logging.getLogger("ProductionConfig")


class ProductionConfig:
    """生产环境配置类"""
    
    def __init__(self):
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载生产环境配置"""
        return {
            # 性能优化配置
            "performance": {
                "max_memory_usage": int(os.getenv("MAX_MEMORY_USAGE", "85")),
                "worker_count": int(os.getenv("WORKER_COUNT", "1")),
                "model_load_threshold": int(os.getenv("MODEL_LOAD_THRESHOLD", "75")),
                "batch_processing_limit": int(os.getenv("BATCH_PROCESSING_LIMIT", "10")),
                "enable_gzip": os.getenv("ENABLE_GZIP", "true").lower() == "true",
                "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
            },
            
            # 安全配置
            "security": {
                "cors_allowed_origins": os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
                "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
                "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
                "enable_https": os.getenv("ENABLE_HTTPS", "false").lower() == "true",
            },
            
            # 监控配置
            "monitoring": {
                "enable_health_checks": os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true",
                "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
                "log_level": os.getenv("LOG_LEVEL", "WARNING"),
                "enable_file_logging": os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
            },
            
            # 数据库配置
            "database": {
                "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
                "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
                "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
            }
        }
    
    def _validate_config(self):
        """验证配置有效性"""
        perf = self.config["performance"]
        
        if perf["max_memory_usage"] > 95:
            error_handler.log_warning("Memory usage threshold set too high (>95%), may cause system instability", "ProductionConfig")
        
        if perf["worker_count"] > psutil.cpu_count():
            error_handler.log_warning(f"Worker count ({perf['worker_count']}) exceeds CPU cores ({psutil.cpu_count()})", "ProductionConfig")
    
    def get(self, section: str, key: str, default=None):
        """获取配置值"""
        return self.config.get(section, {}).get(key, default)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.config["performance"]
    
    def get_security_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.config["security"]
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.config["monitoring"]


def optimize_system_performance():
    """优化系统性能"""
    logger.info("Applying production performance optimizations...")
    
    # 设置Python环境变量
    os.environ["PYTHONOPTIMIZE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # 禁用调试模式
    os.environ["DEBUG"] = "False"
    
    # 设置日志级别
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # 优化内存使用
    import gc
    gc.set_threshold(700, 10, 10)
    
    logger.info("Production performance optimizations applied")


def get_uvicorn_config() -> Dict[str, Any]:
    """获取优化的uvicorn配置"""
    config = ProductionConfig()
    perf_config = config.get_performance_config()
    
    return {
        "host": "0.0.0.0",
        "port": int(os.getenv("MAIN_API_PORT", "8000")),
        "workers": int(os.getenv("WORKER_COUNT", "1")),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": config.get("monitoring", "log_level", "warning").lower(),
        "access_log": False,  # 生产环境禁用访问日志
        "timeout_keep_alive": 5,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }


def configure_production_logging():
    """配置生产环境日志"""
    import logging
    from logging.handlers import RotatingFileHandler
    
    # 创建日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "production.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器（仅错误级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info("Production logging configured")


def check_system_resources() -> Dict[str, Any]:
    """检查系统资源"""
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        }
    except Exception as e:
        logger.error(f"Failed to check system resources: {e}")
        return {}


if __name__ == "__main__":
    # 测试配置
    config = ProductionConfig()
    print("Performance Config:", config.get_performance_config())
    print("Security Config:", config.get_security_config())
    print("System Resources:", check_system_resources())
