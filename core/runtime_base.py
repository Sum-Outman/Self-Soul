"""
统一运行底座 - Unified Runtime Base

实现30天变强版本计划的第一优先级：
2. 稳定运行底座
   - 日志、错误捕获、重试机制
   - 配置中心化，不写死在代码里
   - 支持长期挂机不崩

核心特性：
- 统一的配置管理
- 增强的日志系统
- 完善的错误处理
- 自动重试机制
- 健康检查和监控
- 资源管理和清理
- 长期运行稳定性
"""

import os
import sys
import json
import logging
import threading
import time
import traceback
import signal
import atexit
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc
from pathlib import Path

# 导入现有模块
try:
    from core.error_handling import error_handler, configure_enhanced_logging
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    error_handler = None

try:
    from core.config_manager import ConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    ConfigManager = None

try:
    from core.self_identity import get_identity_manager, get_active_identity
    SELF_IDENTITY_AVAILABLE = True
except ImportError:
    SELF_IDENTITY_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RuntimeMode(Enum):
    """运行模式"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class RuntimeMetrics:
    """运行时指标"""
    start_time: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_count: int = 0
    warning_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    gc_collections: int = 0
    last_gc_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "thread_count": self.thread_count,
            "gc_collections": self.gc_collections,
            "last_gc_time": self.last_gc_time.isoformat()
        }


class RetryPolicy:
    """重试策略"""
    
    def __init__(self,
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: float = 0.1):
        """
        初始化重试策略
        
        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟（秒）
            max_delay: 最大延迟（秒）
            backoff_factor: 退避因子
            jitter: 抖动因子（0.0-1.0）
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, retry_count: int) -> float:
        """获取重试延迟"""
        if retry_count <= 0:
            return 0.0
        
        # 指数退避
        delay = self.initial_delay * (self.backoff_factor ** (retry_count - 1))
        
        # 应用最大延迟限制
        delay = min(delay, self.max_delay)
        
        # 添加抖动
        if self.jitter > 0:
            import random
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)
        
        return delay


class RuntimeBase:
    """运行底座核心类"""
    
    def __init__(self,
                 mode: RuntimeMode = RuntimeMode.DEVELOPMENT,
                 config_dir: str = "config",
                 data_dir: str = "data",
                 log_dir: str = "logs"):
        """
        初始化运行底座
        
        Args:
            mode: 运行模式
            config_dir: 配置目录
            data_dir: 数据目录
            log_dir: 日志目录
        """
        self.mode = mode
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        
        # 创建目录
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 配置管理器
        self.config_manager = None
        if CONFIG_MANAGER_AVAILABLE:
            self.config_manager = ConfigManager()
        
        # 身份管理器
        self.identity_manager = None
        if SELF_IDENTITY_AVAILABLE:
            self.identity_manager = get_identity_manager(data_dir=str(self.data_dir / "identity"))
        
        # 运行时指标
        self.metrics = RuntimeMetrics()
        
        # 健康状态
        self.health_status = HealthStatus.HEALTHY
        self.health_checks: List[Callable[[], bool]] = []
        
        # 重试策略
        self.default_retry_policy = RetryPolicy()
        
        # 清理钩子
        self.cleanup_hooks: List[Callable[[], None]] = []
        
        # 信号处理
        self._setup_signal_handlers()
        
        # 注册退出清理
        atexit.register(self.cleanup)
        
        # 初始化日志系统
        self._setup_logging()
        
        # 初始化配置
        self._load_configuration()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"运行底座初始化完成，模式: {mode.value}")
    
    def _setup_logging(self):
        """设置日志系统"""
        try:
            if ERROR_HANDLING_AVAILABLE:
                # 使用增强日志系统
                configure_enhanced_logging(environment=self.mode.value)
                logger.info("增强日志系统配置完成")
            else:
                # 基本日志配置
                log_file = self.log_dir / f"{self.mode.value}.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
                )
                file_handler.setFormatter(formatter)
                
                # 添加到根日志记录器
                root_logger = logging.getLogger()
                root_logger.addHandler(file_handler)
                
                logger.info("基本日志系统配置完成")
                
        except Exception as e:
            logger.error(f"日志系统配置失败: {e}")
    
    def _load_configuration(self):
        """加载配置"""
        try:
            # 加载环境变量
            self._load_environment_variables()
            
            # 加载配置文件
            if self.config_manager:
                config_files = [
                    self.config_dir / "runtime_config.json",
                    self.config_dir / "system_config.json",
                    self.config_dir / "application_config.json"
                ]
                
                for config_file in config_files:
                    if config_file.exists():
                        self.config_manager.load_from_file(str(config_file))
                        logger.info(f"加载配置文件: {config_file}")
            
            # 创建默认配置（如果不存在）
            self._create_default_configs()
            
            logger.info("配置加载完成")
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
    
    def _load_environment_variables(self):
        """加载环境变量"""
        # 设置运行模式
        env_mode = os.getenv("RUNTIME_MODE")
        if env_mode:
            try:
                self.mode = RuntimeMode(env_mode.lower())
                logger.info(f"从环境变量设置运行模式: {self.mode.value}")
            except ValueError:
                logger.warning(f"无效的运行模式: {env_mode}")
        
        # 其他环境变量
        env_vars = {
            "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "MAX_MEMORY_MB": int(os.getenv("MAX_MEMORY_MB", "1024")),
            "ENABLE_MONITORING": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
        }
        
        logger.info(f"环境变量: {env_vars}")
    
    def _create_default_configs(self):
        """创建默认配置"""
        default_configs = {
            "runtime_config.json": {
                "mode": self.mode.value,
                "logging": {
                    "level": "INFO",
                    "file": f"{self.mode.value}.log",
                    "max_size_mb": 10,
                    "backup_count": 5
                },
                "monitoring": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "memory_threshold_mb": 512,
                    "cpu_threshold_percent": 80.0
                },
                "retry": {
                    "max_retries": 3,
                    "initial_delay_seconds": 1.0,
                    "max_delay_seconds": 60.0
                },
                "cleanup": {
                    "gc_interval_seconds": 300,
                    "log_retention_days": 30
                }
            },
            "system_config.json": {
                "identity": {
                    "enabled": True,
                    "data_dir": "data/identity",
                    "auto_create": True
                },
                "memory": {
                    "max_working_memory": 100,
                    "max_long_term_memory": 1000,
                    "cleanup_interval_hours": 24
                },
                "performance": {
                    "max_threads": 10,
                    "task_timeout_seconds": 300,
                    "queue_size": 100
                }
            }
        }
        
        for filename, config in default_configs.items():
            config_file = self.config_dir / filename
            if not config_file.exists():
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info(f"创建默认配置文件: {filename}")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            self.cleanup()
            sys.exit(0)
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("启动监控线程")
        
        while True:
            try:
                # 更新指标
                self._update_metrics()
                
                # 检查健康状态
                self._check_health()
                
                # 执行清理
                self._perform_cleanup()
                
                # 休眠
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10)  # 错误后短暂休眠
    
    def _update_metrics(self):
        """更新运行时指标"""
        with self.lock:
            # 更新运行时间
            self.metrics.uptime_seconds = (datetime.now() - self.metrics.start_time).total_seconds()
            
            # 更新系统指标
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_percent = process.cpu_percent()
            self.metrics.thread_count = process.num_threads()
            
            # 更新GC统计
            gc_collect_count = gc.get_count()
            self.metrics.gc_collections = sum(gc_collect_count)
            self.metrics.last_gc_time = datetime.now()
    
    def _check_health(self):
        """检查健康状态"""
        with self.lock:
            previous_status = self.health_status
            
            # 执行健康检查
            all_healthy = True
            any_critical = False
            
            for health_check in self.health_checks:
                try:
                    if not health_check():
                        all_healthy = False
                except Exception as e:
                    logger.error(f"健康检查失败: {e}")
                    all_healthy = False
            
            # 检查系统资源
            if self.metrics.memory_usage_mb > 1024:  # 超过1GB
                any_critical = True
                logger.warning(f"内存使用过高: {self.metrics.memory_usage_mb:.1f}MB")
            
            if self.metrics.cpu_percent > 90.0:  # CPU使用率超过90%
                any_critical = True
                logger.warning(f"CPU使用率过高: {self.metrics.cpu_percent:.1f}%")
            
            # 确定健康状态
            if any_critical:
                self.health_status = HealthStatus.CRITICAL
            elif not all_healthy:
                self.health_status = HealthStatus.UNHEALTHY
            elif self.metrics.error_count > 10:
                self.health_status = HealthStatus.DEGRADED
            else:
                self.health_status = HealthStatus.HEALTHY
            
            # 记录状态变化
            if self.health_status != previous_status:
                logger.info(f"健康状态变化: {previous_status.value} -> {self.health_status.value}")
    
    def _perform_cleanup(self):
        """执行清理"""
        try:
            # 触发垃圾回收
            if self.metrics.uptime_seconds % 300 < 60:  # 每5分钟一次
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"垃圾回收清理了 {collected} 个对象")
            
            # 执行清理钩子
            for cleanup_hook in self.cleanup_hooks:
                try:
                    cleanup_hook()
                except Exception as e:
                    logger.error(f"清理钩子执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"清理过程错误: {e}")
    
    def add_health_check(self, check_func: Callable[[], bool]):
        """添加健康检查"""
        with self.lock:
            self.health_checks.append(check_func)
            logger.info(f"添加健康检查: {check_func.__name__}")
    
    def add_cleanup_hook(self, cleanup_func: Callable[[], None]):
        """添加清理钩子"""
        with self.lock:
            self.cleanup_hooks.append(cleanup_func)
            logger.info(f"添加清理钩子: {cleanup_func.__name__}")
    
    def execute_with_retry(self,
                          func: Callable,
                          args: tuple = (),
                          kwargs: Dict[str, Any] = None,
                          retry_policy: RetryPolicy = None,
                          on_retry: Callable[[int, Exception], None] = None) -> Any:
        """
        带重试的执行
        
        Args:
            func: 要执行的函数
            args: 位置参数
            kwargs: 关键字参数
            retry_policy: 重试策略，None则使用默认策略
            on_retry: 重试回调函数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 重试后仍然失败
        """
        if kwargs is None:
            kwargs = {}
        
        if retry_policy is None:
            retry_policy = self.default_retry_policy
        
        last_exception = None
        
        for retry_count in range(retry_policy.max_retries + 1):
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录成功
                with self.lock:
                    self.metrics.successful_requests += 1
                    self.metrics.total_requests += 1
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # 记录失败
                with self.lock:
                    self.metrics.failed_requests += 1
                    self.metrics.total_requests += 1
                    self.metrics.error_count += 1
                
                # 如果是最后一次重试，抛出异常
                if retry_count == retry_policy.max_retries:
                    logger.error(f"函数执行失败，已达到最大重试次数: {func.__name__}")
                    raise
                
                # 计算延迟并等待
                delay = retry_policy.get_delay(retry_count + 1)
                logger.warning(f"函数执行失败，第 {retry_count + 1} 次重试，等待 {delay:.1f} 秒: {func.__name__}")
                
                # 调用重试回调
                if on_retry:
                    try:
                        on_retry(retry_count + 1, e)
                    except Exception as callback_error:
                        logger.error(f"重试回调失败: {callback_error}")
                
                # 等待
                time.sleep(delay)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if self.config_manager:
            return self.config_manager.get(key, default)
        return default
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        if self.config_manager:
            self.config_manager.set(key, value)
    
    def get_identity(self):
        """获取当前身份"""
        if self.identity_manager:
            return self.identity_manager.get_active_identity()
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取运行时指标"""
        with self.lock:
            return self.metrics.to_dict()
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        with self.lock:
            return {
                "status": self.health_status.value,
                "metrics": self.metrics.to_dict(),
                "health_check_count": len(self.health_checks),
                "cleanup_hook_count": len(self.cleanup_hooks)
            }
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        error_msg = str(error)
        error_traceback = traceback.format_exc()
        
        logger.error(f"{context}: {error_msg}\n{error_traceback}")
        
        with self.lock:
            self.metrics.error_count += 1
        
        # 调用错误处理器（如果可用）
        if ERROR_HANDLING_AVAILABLE and error_handler:
            error_handler.handle_error(error, context)
    
    def log_warning(self, message: str, context: str = ""):
        """记录警告"""
        logger.warning(f"{context}: {message}")
        
        with self.lock:
            self.metrics.warning_count += 1
    
    def log_info(self, message: str, context: str = ""):
        """记录信息"""
        if context:
            logger.info(f"{context}: {message}")
        else:
            logger.info(message)
    
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理运行底座资源...")
        
        try:
            # 执行所有清理钩子
            for cleanup_hook in self.cleanup_hooks:
                try:
                    cleanup_hook()
                except Exception as e:
                    logger.error(f"清理钩子执行失败: {e}")
            
            # 保存配置（如果配置管理器可用）
            if self.config_manager:
                try:
                    self.config_manager.save()
                    logger.info("配置保存完成")
                except Exception as e:
                    logger.error(f"配置保存失败: {e}")
            
            # 保存身份（如果身份管理器可用）
            if self.identity_manager:
                try:
                    identity = self.identity_manager.get_active_identity()
                    if identity:
                        identity.save()
                        logger.info("身份保存完成")
                except Exception as e:
                    logger.error(f"身份保存失败: {e}")
            
            # 强制垃圾回收
            collected = gc.collect()
            if collected > 0:
                logger.info(f"垃圾回收清理了 {collected} 个对象")
            
            logger.info("运行底座清理完成")
            
        except Exception as e:
            logger.error(f"清理过程错误: {e}")
    
    def shutdown(self):
        """关闭运行底座"""
        logger.info("开始关闭运行底座...")
        
        # 执行清理
        self.cleanup()
        
        # 记录最终指标
        final_metrics = self.get_metrics()
        logger.info(f"最终运行指标: {json.dumps(final_metrics, indent=2)}")
        
        logger.info("运行底座关闭完成")


# 全局运行底座实例
_runtime_base: Optional[RuntimeBase] = None


def get_runtime_base() -> RuntimeBase:
    """获取全局运行底座"""
    global _runtime_base
    if _runtime_base is None:
        # 确定运行模式
        mode = RuntimeMode.DEVELOPMENT
        env_mode = os.getenv("RUNTIME_MODE", "").lower()
        if env_mode in ["production", "prod"]:
            mode = RuntimeMode.PRODUCTION
        elif env_mode in ["staging", "stage"]:
            mode = RuntimeMode.STAGING
        elif env_mode in ["testing", "test"]:
            mode = RuntimeMode.TESTING
        
        _runtime_base = RuntimeBase(mode=mode)
    
    return _runtime_base


def execute_with_retry(func: Callable, *args, **kwargs) -> Any:
    """带重试的执行（便捷函数）"""
    runtime = get_runtime_base()
    return runtime.execute_with_retry(func, args, kwargs)


def log_error(error: Exception, context: str = ""):
    """记录错误（便捷函数）"""
    runtime = get_runtime_base()
    runtime.log_error(error, context)


def log_warning(message: str, context: str = ""):
    """记录警告（便捷函数）"""
    runtime = get_runtime_base()
    runtime.log_warning(message, context)


def log_info(message: str, context: str = ""):
    """记录信息（便捷函数）"""
    runtime = get_runtime_base()
    runtime.log_info(message, context)


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值（便捷函数）"""
    runtime = get_runtime_base()
    return runtime.get_config(key, default)


def test_runtime_base():
    """测试运行底座"""
    print("=== 测试运行底座 ===")
    
    try:
        # 创建运行底座
        runtime = RuntimeBase(
            mode=RuntimeMode.DEVELOPMENT,
            config_dir="./test_config",
            data_dir="./test_data",
            log_dir="./test_logs"
        )
        
        print("1. 运行底座初始化成功")
        
        # 测试配置管理
        runtime.set_config("test.key", "test_value")
        value = runtime.get_config("test.key", "default")
        print(f"2. 配置管理测试: {value}")
        
        # 测试带重试的执行
        def failing_function(attempts_to_succeed: int):
            nonlocal call_count
            call_count += 1
            if call_count < attempts_to_succeed:
                raise ValueError(f"模拟失败，尝试次数: {call_count}")
            return "success"
        
        call_count = 0
        result = runtime.execute_with_retry(
            lambda: failing_function(3),
            retry_policy=RetryPolicy(max_retries=5, initial_delay=0.1)
        )
        print(f"3. 带重试的执行测试: {result} (调用次数: {call_count})")
        
        # 测试健康检查
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        runtime.add_health_check(healthy_check)
        runtime.add_health_check(unhealthy_check)
        print(f"4. 健康检查添加成功")
        
        # 测试清理钩子
        cleanup_called = [False]
        
        def cleanup_hook():
            cleanup_called[0] = True
        
        runtime.add_cleanup_hook(cleanup_hook)
        print(f"5. 清理钩子添加成功")
        
        # 测试错误日志
        try:
            raise RuntimeError("测试错误")
        except Exception as e:
            runtime.log_error(e, "测试上下文")
        print(f"6. 错误日志测试完成")
        
        # 获取指标
        metrics = runtime.get_metrics()
        print(f"7. 获取指标: {metrics['total_requests']} 个请求")
        
        # 获取健康状态
        health = runtime.get_health_status()
        print(f"8. 健康状态: {health['status']}")
        
        # 执行清理
        runtime.cleanup()
        print(f"9. 清理完成，清理钩子调用: {cleanup_called[0]}")
        
        # 关闭运行底座
        runtime.shutdown()
        print(f"10. 运行底座关闭完成")
        
        # 清理测试目录
        import shutil
        for dir_path in ["./test_config", "./test_data", "./test_logs"]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        print(f"11. 测试目录清理完成")
        
        print("✅ 所有测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_runtime_base()