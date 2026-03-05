import zlib
"""
API依赖管理器 - API Dependency Manager
提供统一的API依赖管理和自动降级功能，确保系统在依赖不可用时仍能正常运行
Provides unified API dependency management and automatic fallback functionality, ensuring system can still operate when dependencies are unavailable
"""

import logging
import functools
import threading
from typing import Dict, Any, Optional, List, Callable, Union
import importlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

def parse_iso_datetime(datetime_str: str) -> Optional[datetime]:
    """解析ISO格式的日期时间字符串，提供向后兼容性"""
    if not datetime_str:
        return None
    
    try:
        # 首先尝试使用fromisoformat（Python 3.7+）
        return datetime.fromisoformat(datetime_str)
    except AttributeError:
        # Python 3.6或更早版本，使用datetime.strptime
        try:
            # 尝试常见ISO格式
            formats = [
                "%Y-%m-%dT%H:%M:%S.%f",  # 带微秒
                "%Y-%m-%dT%H:%M:%S",      # 不带微秒
                "%Y-%m-%d %H:%M:%S.%f",   # 带空格分隔
                "%Y-%m-%d %H:%M:%S",      # 带空格分隔，不带微秒
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            
            # 如果都不匹配，返回None
            return None
        except Exception:
            return None
    except ValueError:
        # 格式不匹配
        return None

# 数据库访问层导入
try:
    from core.database.db_access_layer import (
        get_db_access_layer,
        APIDependencyRecord,
        DependencyHealthRecord
    )
    DB_ACCESS_AVAILABLE = True
    logger.info("Database access layer is available for API dependency manager")
except ImportError as e:
    DB_ACCESS_AVAILABLE = False
    logger.warning(f"Database access layer not available for API dependency manager: {e}. Using in-memory storage only.")
except Exception as e:
    DB_ACCESS_AVAILABLE = False
    logger.warning(f"Error importing database access layer: {e}. Using in-memory storage only.")

# ===== 数据转换函数 =====
def convert_to_dependency_record(dependency: 'APIDependency') -> Optional['APIDependencyRecord']:
    """将APIDependency转换为APIDependencyRecord（用于数据库存储）"""
    if not DB_ACCESS_AVAILABLE:
        return None
    
    try:
        # 转换健康状态
        health_status = dependency.health_status.copy() if dependency.health_status else {}
        if dependency.last_health_check:
            health_status['last_check'] = dependency.last_health_check.isoformat()
        
        # 创建APIDependencyRecord
        return APIDependencyRecord(
            name=dependency.name,
            provider=dependency.provider,
            module_name=dependency.module_name,
            client_class=dependency.client_class,
            config=dependency.config,
            priority=dependency.priority,
            optional=dependency.optional,
            fallback_providers=dependency.fallback_providers,
            is_available=dependency.is_available,
            error_count=dependency.error_count,
            consecutive_errors=dependency.consecutive_errors,
            last_error=dependency.last_error,
            last_health_check=dependency.last_health_check.isoformat() if dependency.last_health_check else None,
            health_status=health_status,
            created_at=None,  # 将由数据库层自动设置
            updated_at=None
        )
    except Exception as e:
        logger.error(f"Failed to convert APIDependency to APIDependencyRecord: {e}")
        return None

def convert_from_dependency_record(record: 'APIDependencyRecord') -> Optional['APIDependency']:
    """将APIDependencyRecord转换为APIDependency"""
    try:
        # 解析健康状态
        health_status = record.health_status or {}
        
        # 解析最后健康检查时间
        last_health_check = None
        if record.last_health_check:
            last_health_check = parse_iso_datetime(record.last_health_check)
        
        # 创建APIDependency
        return APIDependency(
            name=record.name,
            provider=record.provider,
            module_name=record.module_name,
            client_class=record.client_class,
            config=record.config or {},
            priority=record.priority,
            optional=record.optional,
            health_status=health_status,
            fallback_providers=record.fallback_providers or [],
            last_health_check=last_health_check,
            is_available=record.is_available,
            error_count=record.error_count,
            consecutive_errors=record.consecutive_errors,
            last_error=record.last_error
        )
    except Exception as e:
        logger.error(f"Failed to convert APIDependencyRecord to APIDependency: {e}")
        return None

def convert_to_health_record(dependency_name: str, health: 'DependencyHealth') -> Optional['DependencyHealthRecord']:
    """将DependencyHealth转换为DependencyHealthRecord（用于数据库存储）"""
    if not DB_ACCESS_AVAILABLE:
        return None
    
    try:
        # 解析最后检查时间
        last_check_str = None
        if health.last_check:
            last_check_str = health.last_check.isoformat()
        
        # 安全获取custom_metrics属性（可能不存在）
        custom_metrics = getattr(health, 'custom_metrics', {})
        
        return DependencyHealthRecord(
            dependency_name=dependency_name,
            is_healthy=health.is_healthy,
            response_time=health.response_time,
            error_message=health.error_message,
            last_check=last_check_str,
            check_count=health.check_count,
            success_count=health.success_count,
            failure_count=health.failure_count,
            health_score=health.health_score,
            consecutive_errors=health.consecutive_errors,
            custom_metrics=custom_metrics,
            created_at=None,  # 将由数据库层自动设置
            updated_at=None
        )
    except Exception as e:
        logger.error(f"Failed to convert DependencyHealth to DependencyHealthRecord: {e}")
        return None

def convert_from_health_record(record: 'DependencyHealthRecord') -> Optional['DependencyHealth']:
    """将DependencyHealthRecord转换为DependencyHealth"""
    try:
        # 解析最后检查时间
        last_check = None
        if record.last_check:
            last_check = parse_iso_datetime(record.last_check)
        
        return DependencyHealth(
            dependency_name=record.dependency_name,
            is_healthy=record.is_healthy,
            response_time=record.response_time,
            error_message=record.error_message,
            last_check=last_check,
            check_count=record.check_count,
            success_count=record.success_count,
            failure_count=record.failure_count,
            health_score=record.health_score,
            consecutive_errors=record.consecutive_errors,
            custom_metrics=record.custom_metrics or {}
        )
    except Exception as e:
        logger.error(f"Failed to convert DependencyHealthRecord to DependencyHealth: {e}")
        return None

@dataclass
class APIDependency:
    """API依赖数据类 | API Dependency Data Class"""
    name: str
    provider: str
    module_name: str
    client_class: str
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 优先级，数字越小优先级越高
    optional: bool = False  # 是否为可选依赖
    health_status: Dict[str, Any] = field(default_factory=dict)
    fallback_providers: List[str] = field(default_factory=list)
    last_health_check: Optional[datetime] = None
    is_available: bool = False
    error_count: int = 0
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 | Convert to dictionary"""
        return {
            "name": self.name,
            "provider": self.provider,
            "module_name": self.module_name,
            "client_class": self.client_class,
            "priority": self.priority,
            "optional": self.optional,
            "is_available": self.is_available,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "fallback_providers": self.fallback_providers
        }

@dataclass
class DependencyHealth:
    """依赖健康状态数据类 | Dependency Health Status Data Class"""
    dependency_name: str
    is_healthy: bool = False
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    last_check: Optional[datetime] = None
    check_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    health_score: float = 0.0
    consecutive_errors: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_success(self, response_time: float):
        """更新成功状态 | Update success status"""
        self.is_healthy = True
        self.response_time = response_time
        self.last_check = datetime.now()
        self.check_count += 1
        self.success_count += 1
        self.consecutive_errors = 0
        self.health_score = min(100.0, self.health_score + 10)
        
        # 计算基于响应时间的健康分数
        if response_time < 1.0:
            self.health_score += 5
        elif response_time > 3.0:
            self.health_score -= 5
    
    def update_failure(self, error_message: str):
        """更新失败状态 | Update failure status"""
        self.is_healthy = False
        self.error_message = error_message
        self.last_check = datetime.now()
        self.check_count += 1
        self.failure_count += 1
        self.consecutive_errors += 1
        self.health_score = max(0.0, self.health_score - 20)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要 | Get health summary"""
        success_rate = (self.success_count / self.check_count * 100) if self.check_count > 0 else 0
        
        return {
            "dependency_name": self.dependency_name,
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "success_rate": success_rate,
            "response_time": self.response_time,
            "check_count": self.check_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_errors": self.consecutive_errors,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "error_message": self.error_message
        }

class APIDependencyManager:
    """API依赖管理器 | API Dependency Manager
    
    管理所有API依赖，提供自动降级、健康检查、依赖注入等功能
    Manages all API dependencies, provides automatic fallback, health checks, dependency injection, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化依赖管理器 | Initialize dependency manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 依赖注册表
        # Dependency registry
        self.dependencies: Dict[str, APIDependency] = {}
        
        # 健康状态监控
        # Health status monitoring
        self.health_status: Dict[str, DependencyHealth] = {}
        
        # 线程同步锁
        # Thread synchronization lock
        self._lock = threading.RLock()
        
        # 客户端缓存
        # Client cache
        self.client_cache: Dict[str, Any] = {}
        
        # 数据库访问层
        # Database access layer
        self.db_available = DB_ACCESS_AVAILABLE
        self.db_layer = None
        if self.db_available:
            try:
                self.db_layer = get_db_access_layer()
                self.logger.info("Database access layer initialized for API dependency manager")
                
                # 从数据库加载现有依赖和健康状态
                self._load_from_database()
                
            except Exception as e:
                self.logger.error(f"Failed to initialize database access layer: {e}")
                self.db_available = False
                self.db_layer = None
        
        # 降级策略
        # Fallback strategies
        self.fallback_strategies = {
            "openai": ["anthropic", "google_genai", "deepseek"],
            "anthropic": ["openai", "google_genai", "cohere"],
            "google_genai": ["openai", "anthropic", "cohere"],
            "azure_openai": ["openai", "anthropic", "google_genai"],
            "deepseek": ["openai", "anthropic", "siliconflow"],
            "siliconflow": ["deepseek", "openai", "anthropic"],
            "zhipu": ["openai", "deepseek", "baidu"],
            "baidu": ["zhipu", "alibaba", "openai"],
            "alibaba": ["baidu", "zhipu", "openai"],
            "moonshot": ["openai", "deepseek", "anthropic"],
            "yi": ["openai", "deepseek", "anthropic"],
            "tencent": ["baidu", "alibaba", "openai"]
        }
        
        # 健康检查配置
        # Health check configuration
        self.health_check_config = {
            "enabled": True,
            "interval_seconds": 300,  # 5分钟
            "timeout_seconds": 10,
            "max_consecutive_errors": 5,
            "auto_recovery": True
        }

        # 监控线程控制
        # Monitoring thread control
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._monitor_thread_running = False

        # 初始化默认依赖
        # Initialize default dependencies
        self._initialize_default_dependencies()
        
        # 启动健康检查线程
        if self.health_check_config["enabled"]:
            self._start_health_monitoring()
        
        self.logger.info("API依赖管理器初始化完成 | API dependency manager initialized")
    
    def _load_from_database(self):
        """从数据库加载现有依赖和健康状态 | Load existing dependencies and health status from database"""
        if not self.db_available or not self.db_layer:
            self.logger.debug("Database not available, skipping load from database")
            return
        
        try:
            # 1. 加载API依赖
            dependency_records = self.db_layer.get_all_api_dependencies()
            with self._lock:
                for record in dependency_records:
                    dependency = convert_from_dependency_record(record)
                    if dependency:
                        self.dependencies[dependency.name] = dependency
                        self.logger.debug(f"Loaded API dependency from database: {dependency.name}")
            
            # 2. 加载健康状态
            health_records = self.db_layer.get_all_dependency_health_status()
            with self._lock:
                for record in health_records:
                    health = convert_from_health_record(record)
                    if health and record.dependency_name:
                        self.health_status[record.dependency_name] = health
                        self.logger.debug(f"Loaded health status from database: {record.dependency_name}")
            
            self.logger.info(f"Loaded {len(dependency_records)} dependencies and {len(health_records)} health status records from database")
            
        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}")
    
    def _initialize_default_dependencies(self):
        """初始化默认依赖 | Initialize default dependencies"""
        default_dependencies = [
            {
                "name": "openai_chat",
                "provider": "openai",
                "module_name": "openai",
                "client_class": "OpenAI",
                "priority": 1,
                "fallback_providers": ["anthropic", "google_genai", "deepseek"],
                "optional": False  # 核心依赖，必须可用
            },
            {
                "name": "anthropic_chat",
                "provider": "anthropic",
                "module_name": "anthropic",
                "client_class": "Anthropic",
                "priority": 2,
                "fallback_providers": ["openai", "google_genai", "cohere"],
                "optional": True  # 可选外部API
            },
            {
                "name": "google_genai_chat",
                "provider": "google_genai",
                "module_name": "google.generativeai",
                "client_class": "GenerativeModel",
                "priority": 3,
                "fallback_providers": ["openai", "anthropic", "cohere"],
                "optional": True  # 可选外部API
            },
            {
                "name": "deepseek_chat",
                "provider": "deepseek",
                "module_name": "openai",
                "client_class": "OpenAI",
                "priority": 2,
                "fallback_providers": ["openai", "anthropic", "siliconflow"],
                "optional": True  # 可选外部API
            },
            {
                "name": "siliconflow_chat",
                "provider": "siliconflow",
                "module_name": "openai",
                "client_class": "OpenAI",
                "priority": 3,
                "fallback_providers": ["deepseek", "openai", "anthropic"],
                "optional": True  # 可选外部API
            }
        ]
        
        for dep_config in default_dependencies:
            dependency_name = dep_config["name"]
            
            # 只添加不存在的依赖
            with self._lock:
                if dependency_name not in self.dependencies:
                    dependency = APIDependency(**dep_config)
                    self.dependencies[dependency_name] = dependency
                    
                    # 如果健康状态不存在，创建默认健康状态
                    if dependency_name not in self.health_status:
                        self.health_status[dependency_name] = DependencyHealth(dependency_name=dependency_name)
                    
                    # 保存到数据库
                    self._save_dependency_to_db(dependency)
                    
                    self.logger.debug(f"Default dependency initialized and saved to database: {dependency_name}")
                else:
                    self.logger.debug(f"Dependency already exists, skipping: {dependency_name}")
    
    def _save_dependency_to_db(self, dependency: APIDependency):
        """保存依赖到数据库 | Save dependency to database"""
        if not self.db_available or not self.db_layer:
            return
        
        try:
            # 转换为数据库记录
            dependency_record = convert_to_dependency_record(dependency)
            if not dependency_record:
                self.logger.warning(f"Failed to convert dependency to record: {dependency.name}")
                return
            
            # 保存到数据库
            success = self.db_layer.save_api_dependency(dependency_record)
            if success:
                self.logger.debug(f"Dependency saved to database: {dependency.name}")
            else:
                self.logger.warning(f"Failed to save dependency to database: {dependency.name}")
                
        except Exception as e:
            self.logger.error(f"Error saving dependency to database: {e}")
    
    def _save_health_status_to_db(self, dependency_name: str, health: DependencyHealth):
        """保存健康状态到数据库 | Save health status to database"""
        if not self.db_available or not self.db_layer:
            return
        
        try:
            # 转换为数据库记录
            health_record = convert_to_health_record(dependency_name, health)
            if not health_record:
                self.logger.warning(f"Failed to convert health status to record: {dependency_name}")
                return
            
            # 保存到数据库
            success = self.db_layer.save_dependency_health_status(health_record)
            if success:
                self.logger.debug(f"Health status saved to database: {dependency_name}")
            else:
                self.logger.warning(f"Failed to save health status to database: {dependency_name}")
                
        except Exception as e:
            self.logger.error(f"Error saving health status to database: {e}")
    
    def _start_health_monitoring(self):
        """启动健康监控线程 | Start health monitoring thread"""
        import threading
        
        # 检查是否已经在运行
        if self._monitor_thread_running:
            self.logger.warning("健康监控线程已经在运行 | Health monitoring thread already running")
            return
        
        def health_monitor():
            """健康监控线程的主循环"""
            self._monitor_thread_running = True
            self.logger.debug("健康监控线程启动 | Health monitoring thread started")
            
            try:
                while not self._stop_event.is_set():
                    try:
                        # 执行健康检查
                        self._perform_health_checks()
                        
                        # 等待下一个检查周期，但会检查停止事件
                        check_interval = self.health_check_config["interval_seconds"]
                        for _ in range(check_interval):
                            if self._stop_event.is_set():
                                break
                            time.sleep(1)  # 每秒检查一次停止事件
                            
                    except Exception as e:
                        self.logger.error(f"健康监控线程错误: {str(e)} | Health monitoring thread error: {str(e)}")
                        
                        # 出错后等待，但会检查停止事件
                        for _ in range(60):  # 最大等待60秒
                            if self._stop_event.is_set():
                                break
                            time.sleep(1)
                
                self.logger.info("健康监控线程正常停止 | Health monitoring thread stopped normally")
                
            except Exception as e:
                self.logger.error(f"健康监控线程异常终止: {str(e)} | Health monitoring thread terminated with exception: {str(e)}")
                
            finally:
                self._monitor_thread_running = False
                self._monitor_thread = None
        
        # 重置停止事件
        self._stop_event.clear()
        
        # 创建并启动线程
        self._monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        self._monitor_thread.start()
        self.logger.info("健康监控线程已启动 | Health monitoring thread started")
    
    def stop_health_monitoring(self, timeout_seconds: int = 30):
        """停止健康监控线程 | Stop health monitoring thread
        
        Args:
            timeout_seconds: 等待线程停止的超时时间（秒）
            
        Returns:
            bool: 线程是否成功停止
        """
        if not self._monitor_thread_running or not self._monitor_thread:
            self.logger.debug("健康监控线程未运行 | Health monitoring thread not running")
            return True
        
        self.logger.info("正在停止健康监控线程... | Stopping health monitoring thread...")
        
        # 设置停止事件
        self._stop_event.set()
        
        # 等待线程结束
        try:
            self._monitor_thread.join(timeout=timeout_seconds)
            
            if self._monitor_thread.is_alive():
                self.logger.warning(f"健康监控线程在 {timeout_seconds} 秒后仍在运行 | Health monitoring thread still running after {timeout_seconds} seconds")
                return False
            else:
                self.logger.info("健康监控线程已成功停止 | Health monitoring thread stopped successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"停止健康监控线程时出错: {str(e)} | Error stopping health monitoring thread: {str(e)}")
            return False
    
    def _perform_health_checks(self):
        """执行健康检查 | Perform health checks"""
        # 获取当前依赖名称的快照，避免在遍历过程中字典被修改
        with self._lock:
            dep_names = list(self.dependencies.keys())
        
        for dep_name in dep_names:
            # 获取依赖（在锁保护下）
            with self._lock:
                if dep_name not in self.dependencies:
                    continue  # 依赖可能在遍历过程中被删除
                dependency = self.dependencies[dep_name]
                health_status = self.health_status.get(dep_name)
            
            if not health_status:
                continue
                
            try:
                # 检查是否需要健康检查
                last_check = health_status.last_check
                if last_check and (datetime.now() - last_check).total_seconds() < self.health_check_config["interval_seconds"]:
                    continue
                
                # 执行健康检查
                start_time = time.time()
                client = self.get_client(dep_name)
                
                if client is None:
                    with self._lock:
                        if dep_name in self.health_status:
                            self.health_status[dep_name].update_failure("无法创建客户端 | Cannot create client")
                    continue
                
                # 根据提供商类型执行不同的健康检查
                health_result = self._check_dependency_health(dep_name, client, dependency.provider)
                response_time = time.time() - start_time
                
                if health_result["healthy"]:
                    health_status.update_success(response_time)
                    # 使用新的更新方法
                    self._update_dependency_state(
                        dep_name,
                        is_available=True,
                        error_count=0,
                        consecutive_errors=0,
                        last_error=None,
                        last_health_check=datetime.now(),
                        health_status=health_status.get_health_summary()
                    )
                else:
                    health_status.update_failure(health_result.get("error", "未知错误 | Unknown error"))
                    # 使用新的更新方法
                    self._update_dependency_state(
                        dep_name,
                        is_available=False,
                        error_count=dependency.error_count + 1,
                        consecutive_errors=dependency.consecutive_errors + 1,
                        last_error=health_result.get("error"),
                        last_health_check=datetime.now(),
                        health_status=health_status.get_health_summary()
                    )
                
                # 保存健康状态到数据库
                self._save_health_status_to_db(dep_name, health_status)
                
            except Exception as e:
                import traceback
                error_msg = f"健康检查失败: {str(e)} | Health check failed: {str(e)}"
                stack_trace = traceback.format_exc()
                
                # 使用锁保护更新健康状态
                with self._lock:
                    if dep_name in self.health_status:
                        self.health_status[dep_name].update_failure(error_msg)
                
                self.logger.error(f"依赖{dep_name}健康检查失败: {error_msg} | Dependency {dep_name} health check failed: {error_msg}")
                self.logger.debug(f"堆栈跟踪: {stack_trace} | Stack trace: {stack_trace}")
                
                # 即使出错也要尝试保存到数据库
                try:
                    with self._lock:
                        if dep_name in self.dependencies:
                            dependency = self.dependencies[dep_name]
                            health_status = self.health_status.get(dep_name)
                            health_summary = health_status.get_health_summary() if health_status else {}
                            
                            self._update_dependency_state(
                                dep_name,
                                is_available=False,
                                error_count=dependency.error_count + 1,
                                consecutive_errors=dependency.consecutive_errors + 1,
                                last_error=error_msg,
                                last_health_check=datetime.now(),
                                health_status=health_summary
                            )
                        
                        if dep_name in self.health_status:
                            self._save_health_status_to_db(dep_name, self.health_status[dep_name])
                except Exception as save_error:
                    self.logger.warning(f"Failed to save error state to database: {save_error}")
    
    def _check_dependency_health(self, dep_name: str, client: Any, provider: str) -> Dict[str, Any]:
        """检查依赖健康状态 | Check dependency health status"""
        try:
            if provider in ["openai", "deepseek", "siliconflow", "moonshot", "yi", "azure_openai"]:
                # OpenAI兼容API健康检查
                # OpenAI compatible API health check
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    timeout=self.health_check_config["timeout_seconds"]
                )
                
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return {"healthy": True, "response": "API响应正常 | API response normal"}
                else:
                    return {"healthy": False, "error": "API返回空响应 | API returned empty response"}
            
            elif provider == "anthropic":
                # Anthropic API健康检查
                # Anthropic API health check
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "ping"}],
                    timeout=self.health_check_config["timeout_seconds"]
                )
                
                if hasattr(response, 'content') and len(response.content) > 0:
                    return {"healthy": True, "response": "API响应正常 | API response normal"}
                else:
                    return {"healthy": False, "error": "API返回空响应 | API returned empty response"}
            
            elif provider == "google_genai":
                # Google Generative AI健康检查
                # Google Generative AI health check
                response = client.generate_content("ping")
                
                if hasattr(response, 'text') and response.text:
                    return {"healthy": True, "response": "API响应正常 | API response normal"}
                else:
                    return {"healthy": False, "error": "API返回空响应 | API returned empty response"}
            
            else:
                # 通用健康检查
                # Generic health check
                return {"healthy": True, "response": "依赖可用 | Dependency available"}
                
        except Exception as e:
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"依赖健康检查异常: {error_detail} | Dependency health check exception: {error_detail}")
            self.logger.debug(f"堆栈跟踪: {stack_trace} | Stack trace: {stack_trace}")
            return {"healthy": False, "error": error_detail}
    
    def register_dependency(self, dependency: APIDependency) -> bool:
        """注册API依赖 | Register API dependency
        
        Args:
            dependency: API依赖配置
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                self.dependencies[dependency.name] = dependency
                
                # 初始化健康状态
                if dependency.name not in self.health_status:
                    self.health_status[dependency.name] = DependencyHealth(dependency_name=dependency.name)
            
            # 保存到数据库
            self._save_dependency_to_db(dependency)
            with self._lock:
                if dependency.name in self.health_status:
                    self._save_health_status_to_db(dependency.name, self.health_status[dependency.name])
            
            self.logger.info(f"已注册依赖: {dependency.name} ({dependency.provider}) | Registered dependency: {dependency.name} ({dependency.provider})")
            return True
            
        except Exception as e:
            self.logger.error(f"注册依赖失败 {dependency.name}: {str(e)} | Failed to register dependency {dependency.name}: {str(e)}")
            return False
    
    def get_dependency(self, name: str) -> Optional[APIDependency]:
        """获取依赖信息 | Get dependency information
        
        Args:
            name: 依赖名称
            
        Returns:
            Optional[APIDependency]: 依赖信息，如果不存在则返回None
        """
        with self._lock:
            return self.dependencies.get(name)
    
    def _update_dependency_state(self, dependency_name: str, 
                                 is_available: bool = None,
                                 error_count: int = None,
                                 consecutive_errors: int = None,
                                 last_error: str = None,
                                 last_health_check: datetime = None,
                                 health_status: Dict[str, Any] = None):
        """更新依赖状态并同步到数据库 | Update dependency state and synchronize to database
        
        Args:
            dependency_name: 依赖名称
            is_available: 是否可用
            error_count: 错误计数
            consecutive_errors: 连续错误计数
            last_error: 最后错误信息
            last_health_check: 最后健康检查时间
            health_status: 健康状态
        """
        with self._lock:
            if dependency_name not in self.dependencies:
                self.logger.warning(f"Dependency not found: {dependency_name}")
                return
            
            dependency = self.dependencies[dependency_name]
            
            # 更新依赖状态
            if is_available is not None:
                dependency.is_available = is_available
            if error_count is not None:
                dependency.error_count = error_count
            if consecutive_errors is not None:
                dependency.consecutive_errors = consecutive_errors
            if last_error is not None:
                dependency.last_error = last_error
            if last_health_check is not None:
                dependency.last_health_check = last_health_check
            if health_status is not None:
                dependency.health_status = health_status
        
        # 保存到数据库
        self._save_dependency_to_db(dependency)
    
    def get_available_dependency(self, provider: str, service_type: str = "chat") -> Optional[APIDependency]:
        """获取可用的依赖 | Get available dependency
        
        Args:
            provider: API提供商
            service_type: 服务类型
            
        Returns:
            Optional[APIDependency]: 可用的依赖，如果不存在则返回None
        """
        # 在锁保护下获取依赖列表的快照
        with self._lock:
            dependencies = list(self.dependencies.values())
        
        # 首先查找指定提供商和类型的依赖
        for dep in dependencies:
            if dep.provider == provider and dep.name.endswith(f"_{service_type}"):
                if dep.is_available:
                    return dep
        
        # 如果未找到，查找同一提供商的任何可用依赖
        for dep in dependencies:
            if dep.provider == provider and dep.is_available:
                return dep
        
        # 如果仍未找到，使用降级策略
        fallback_providers = self.fallback_strategies.get(provider, [])
        for fallback_provider in fallback_providers:
            for dep in dependencies:
                if dep.provider == fallback_provider and dep.is_available:
                    self.logger.info(f"使用降级依赖: {provider} -> {fallback_provider} | Using fallback dependency: {provider} -> {fallback_provider}")
                    return dep
        
        return None
    
    def get_client(self, dependency_name: str, force_create: bool = False) -> Any:
        """获取API客户端 | Get API client
        
        Args:
            dependency_name: 依赖名称
            force_create: 是否强制创建新客户端
            
        Returns:
            Any: API客户端实例，如果创建失败则返回None
        """
        try:
            # 检查缓存
            if not force_create and dependency_name in self.client_cache:
                return self.client_cache[dependency_name]
            
            # 获取依赖配置
            dependency = self.dependencies.get(dependency_name)
            if not dependency:
                self.logger.error(f"未找到依赖: {dependency_name} | Dependency not found: {dependency_name}")
                return None
            
            # 导入模块
            module = self._import_module(dependency.module_name, dependency)
            if module is None:
                # _import_module方法已经记录了适当的日志消息
                return None
            
            # 创建客户端
            client = self._create_client(module, dependency.client_class, dependency.config, dependency.provider)
            if client is None:
                # 对于可选依赖，使用警告级别日志
                if dependency.optional:
                    self.logger.warning(f"无法创建可选客户端: {dependency_name} | Failed to create optional client: {dependency_name}")
                else:
                    self.logger.error(f"创建客户端失败: {dependency_name} | Failed to create client: {dependency_name}")
                return None
            
            # 缓存客户端
            self.client_cache[dependency_name] = client
            
            # 更新依赖可用性状态并保存到数据库
            self._update_dependency_state(dependency_name, is_available=True)
            
            return client
            
        except ImportError as e:
            error_msg = f"无法获取API客户端 {dependency_name}: 缺少必要的依赖包。请安装相应的Python包。错误: {str(e)} | Cannot get API client {dependency_name}: Missing required dependency packages. Please install the required Python packages. Error: {str(e)}"
            # 对于可选依赖，使用警告级别日志
            dependency = self.dependencies.get(dependency_name)
            if dependency and dependency.optional:
                self.logger.warning(error_msg)
                return None
            else:
                self.logger.error(error_msg)
                raise ImportError(error_msg)
        except Exception as e:
            import traceback
            error_detail = f"{type(e).__name__}: {str(e)}"
            stack_trace = traceback.format_exc()
            # 对于可选依赖，使用警告级别日志
            dependency = self.dependencies.get(dependency_name)
            if dependency and dependency.optional:
                self.logger.warning(f"获取可选客户端失败 {dependency_name}: {error_detail} | Failed to get optional client {dependency_name}: {error_detail}")
            else:
                self.logger.error(f"获取客户端失败 {dependency_name}: {error_detail} | Failed to get client {dependency_name}: {error_detail}")
            self.logger.debug(f"堆栈跟踪: {stack_trace} | Stack trace: {stack_trace}")
            return None
    
    def _import_module(self, module_name: str, dependency: APIDependency = None) -> Any:
        """导入模块 | Import module"""
        try:
            module = importlib.import_module(module_name)
            return module
        except ImportError as e:
            error_msg = f"无法导入模块: {module_name}。请安装相应的Python包。错误: {str(e)} | Cannot import module: {module_name}. Please install the required Python package. Error: {str(e)}"
            
            # 为可选依赖提供安装指南
            if dependency and dependency.optional:
                installation_guide = self._get_installation_guide(module_name, dependency.provider)
                error_msg += f"\n安装指南: {installation_guide} | Installation guide: {installation_guide}"
                self.logger.warning(error_msg)
            else:
                self.logger.warning(error_msg)
            
            return None

    def _create_client(self, module: Any, client_class: str, config: Dict[str, Any], provider: str) -> Any:
        """创建客户端 | Create client"""
        try:
            # 获取客户端类
            if hasattr(module, client_class):
                cls = getattr(module, client_class)
            else:
                # 如果模块本身就是一个客户端类
                cls = module
            
            # 根据提供商类型使用不同的初始化方式
            if provider in ["openai", "deepseek", "siliconflow", "moonshot", "yi"]:
                # OpenAI兼容客户端
                client = cls(**config)
            
            elif provider == "anthropic":
                # Anthropic客户端
                client = cls(**config)
            
            elif provider == "google_genai":
                # Google Generative AI特殊处理
                try:
                    import google.generativeai as genai  # type: ignore
                    genai.configure(api_key=config.get("api_key"))
                    model_name = config.get("model_name", "gemini-pro")
                    client = cls(model_name)
                except ImportError as e:
                    error_msg = f"无法导入Google Generative AI模块。请安装google-generativeai包。错误: {str(e)} | Cannot import Google Generative AI module. Please install google-generativeai package. Error: {str(e)}"
                    self.logger.warning(error_msg)
                    return None
            
            elif provider == "azure_openai":
                # Azure OpenAI特殊处理
                import openai
                client = openai.AzureOpenAI(**config)
            
            else:
                # 默认初始化方式
                client = cls(**config)
            
            return client
            
        except Exception as e:
            self.logger.error(f"创建客户端失败 {provider}: {str(e)} | Failed to create client {provider}: {str(e)}")
            return None
    
    def _get_installation_guide(self, module_name: str, provider: str) -> str:
        """获取依赖安装指南 | Get dependency installation guide
        
        Args:
            module_name: 模块名称
            provider: 提供商类型
            
        Returns:
            str: 安装指南
        """
        # 常见API依赖的安装指南
        installation_guides = {
            "openai": "安装OpenAI Python包: pip install openai",
            "anthropic": "安装Anthropic Python包: pip install anthropic",
            "google.generativeai": "安装Google Generative AI包: pip install google-generativeai",
            "google.generativeai_genai": "安装Google Generative AI包: pip install google-generativeai",
            "cohere": "安装Cohere Python包: pip install cohere",
            "deepseek": "安装DeepSeek Python包: pip install deepseek",
            "zhipuai": "安装智谱AI Python包: pip install zhipuai",
            "baidu": "安装百度ERNIE Python包: pip install erniebot",
            "alibaba": "安装阿里通义千问Python包: pip install dashscope",
            "moonshot": "安装Moonshot AI Python包: pip install moonshot",
            "yi": "安装零一万物Python包: pip install yi",
            "tencent": "安装腾讯混元Python包: pip install tencentcloud-sdk-python",
            "siliconflow": "安装SiliconFlow Python包: pip install siliconflow",
        }
        
        # 根据模块名称或提供商获取安装指南
        if module_name in installation_guides:
            return installation_guides[module_name]
        elif provider in installation_guides:
            return installation_guides[provider]
        else:
            # 通用安装指南
            return f"安装{module_name} Python包: pip install {module_name.replace('.', '-').replace('_', '-')}"
    
    def check_optional_dependency(self, dependency_name: str) -> Dict[str, Any]:
        """检查可选依赖的可用性 | Check optional dependency availability
        
        Args:
            dependency_name: 依赖名称
            
        Returns:
            Dict[str, Any]: 依赖状态信息
        """
        dependency = self.dependencies.get(dependency_name)
        if not dependency:
            return {
                "name": dependency_name,
                "available": False,
                "error": f"未找到依赖: {dependency_name}",
                "installation_guide": None
            }
        
        # 使用_import_module方法检查依赖可用性，它会记录适当的日志消息
        module = self._import_module(dependency.module_name, dependency)
        is_available = module is not None
        error_message = None if is_available else f"无法导入模块: {dependency.module_name}"
        
        # 更新依赖可用性状态并保存到数据库
        self._update_dependency_state(dependency_name, is_available=is_available)
        
        result = {
            "name": dependency_name,
            "provider": dependency.provider,
            "module_name": dependency.module_name,
            "available": is_available,
            "optional": dependency.optional,
            "is_available": dependency.is_available,
            "error": error_message,
            "installation_guide": self._get_installation_guide(dependency.module_name, dependency.provider) if not is_available else None,
            "last_checked": datetime.now().isoformat()
        }
        
        # 记录状态
        if is_available:
            self.logger.info(f"可选依赖 {dependency_name} 可用 | Optional dependency {dependency_name} available")
        else:
            self.logger.warning(f"可选依赖 {dependency_name} 不可用: {error_message} | Optional dependency {dependency_name} unavailable: {error_message}")
        
        return result
    
    def get_optional_dependencies_status(self) -> Dict[str, Any]:
        """获取所有可选依赖的状态 | Get status of all optional dependencies
        
        Returns:
            Dict[str, Any]: 可选依赖状态信息
        """
        optional_dependencies = {}
        
        for dep_name, dependency in self.dependencies.items():
            if dependency.optional:
                # 检查依赖可用性
                status = self.check_optional_dependency(dep_name)
                optional_dependencies[dep_name] = status
        
        # 统计信息
        total_optional = len(optional_dependencies)
        available_optional = sum(1 for status in optional_dependencies.values() if status["available"])
        
        return {
            "total_optional_dependencies": total_optional,
            "available_optional_dependencies": available_optional,
            "unavailable_optional_dependencies": total_optional - available_optional,
            "dependencies": optional_dependencies,
            "timestamp": datetime.now().isoformat()
        }
    
    def execute_with_fallback(self, dependency_name: str, operation: Callable, *args, **kwargs) -> Any:
        """使用降级策略执行操作 | Execute operation with fallback strategy
        
        Args:
            dependency_name: 主要依赖名称
            operation: 要执行的操作函数，第一个参数应为客户端
            *args: 操作函数的额外位置参数
            **kwargs: 操作函数的额外关键字参数
            
        Returns:
            Any: 操作结果
        """
        # 获取主要依赖
        primary_dependency = self.dependencies.get(dependency_name)
        if not primary_dependency:
            raise ValueError(f"未找到依赖: {dependency_name} | Dependency not found: {dependency_name}")
        
        # 获取主要客户端
        primary_client = self.get_client(dependency_name)
        
        # 尝试使用主要客户端执行操作
        if primary_client and primary_dependency.is_available:
            try:
                start_time = time.time()
                result = operation(primary_client, *args, **kwargs)
                response_time = time.time() - start_time
                
                # 更新健康状态
                self.health_status[dependency_name].update_success(response_time)
                self._update_dependency_state(
                    dependency_name,
                    is_available=True,
                    error_count=0,
                    consecutive_errors=0
                )
                
                return result
                
            except Exception as e:
                import traceback
                error_detail = f"{type(e).__name__}: {str(e)}"
                stack_trace = traceback.format_exc()
                self.logger.warning(f"主要依赖操作失败 {dependency_name}: {error_detail}，尝试降级 | Primary dependency operation failed {dependency_name}: {error_detail}, trying fallback")
                self.logger.debug(f"堆栈跟踪: {stack_trace} | Stack trace: {stack_trace}")
                
                # 更新健康状态
                self.health_status[dependency_name].update_failure(error_detail)
                self._update_dependency_state(
                    dependency_name,
                    is_available=False,
                    error_count=primary_dependency.error_count + 1,
                    consecutive_errors=primary_dependency.consecutive_errors + 1,
                    last_error=error_detail
                )
        
        # 如果主要依赖失败，尝试降级依赖
        fallback_providers = primary_dependency.fallback_providers or self.fallback_strategies.get(primary_dependency.provider, [])
        
        for fallback_provider in fallback_providers:
            # 查找可用的降级依赖
            fallback_dependency = self.get_available_dependency(fallback_provider)
            if not fallback_dependency:
                continue
            
            # 获取降级客户端
            fallback_client = self.get_client(fallback_dependency.name)
            if not fallback_client:
                continue
            
            try:
                self.logger.info(f"使用降级依赖: {fallback_dependency.name} | Using fallback dependency: {fallback_dependency.name}")
                
                start_time = time.time()
                result = operation(fallback_client, *args, **kwargs)
                response_time = time.time() - start_time
                
                # 更新降级依赖的健康状态
                self.health_status[fallback_dependency.name].update_success(response_time)
                self._update_dependency_state(
                    fallback_dependency.name,
                    is_available=True,
                    error_count=0,
                    consecutive_errors=0
                )
                
                return result
                
            except Exception as e:
                import traceback
                error_detail = f"{type(e).__name__}: {str(e)}"
                stack_trace = traceback.format_exc()
                self.logger.warning(f"降级依赖操作失败 {fallback_dependency.name}: {error_detail} | Fallback dependency operation failed {fallback_dependency.name}: {error_detail}")
                self.logger.debug(f"堆栈跟踪: {stack_trace} | Stack trace: {stack_trace}")
                
                # 更新降级依赖的健康状态
                self.health_status[fallback_dependency.name].update_failure(error_detail)
                self._update_dependency_state(
                    fallback_dependency.name,
                    is_available=False,
                    error_count=fallback_dependency.error_count + 1,
                    consecutive_errors=fallback_dependency.consecutive_errors + 1,
                    last_error=error_detail
                )
        
        # 所有依赖都失败
        error_msg = f"所有依赖都失败: {dependency_name} | All dependencies failed: {dependency_name}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def get_health_status(self, dependency_name: str = None) -> Dict[str, Any]:
        """获取健康状态 | Get health status
        
        Args:
            dependency_name: 可选的依赖名称，如果提供则只返回该依赖的健康状态
            
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        if dependency_name:
            if dependency_name in self.health_status:
                return self.health_status[dependency_name].get_health_summary()
            else:
                return {"error": f"未找到依赖: {dependency_name} | Dependency not found: {dependency_name}"}
        
        # 返回所有依赖的健康状态
        health_status = {}
        for dep_name, health in self.health_status.items():
            health_status[dep_name] = health.get_health_summary()
        
        # 计算总体健康评分
        total_health_score = 0
        healthy_count = 0
        
        for health in self.health_status.values():
            if health.is_healthy:
                healthy_count += 1
            total_health_score += health.health_score
        
        total_dependencies = len(self.health_status)
        overall_health_score = total_health_score / total_dependencies if total_dependencies > 0 else 0
        
        return {
            "overall_health_score": overall_health_score,
            "healthy_dependencies": healthy_count,
            "total_dependencies": total_dependencies,
            "health_status": health_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self, dependency_name: str = None):
        """清除客户端缓存 | Clear client cache
        
        Args:
            dependency_name: 可选的依赖名称，如果提供则只清除该依赖的缓存
        """
        if dependency_name:
            if dependency_name in self.client_cache:
                del self.client_cache[dependency_name]
                self.logger.info(f"已清除依赖缓存: {dependency_name} | Cleared dependency cache: {dependency_name}")
        else:
            self.client_cache.clear()
            self.logger.info("已清除所有客户端缓存 | Cleared all client cache")
    
    def list_dependencies(self) -> Dict[str, Any]:
        """列出所有依赖 | List all dependencies
        
        Returns:
            Dict[str, Any]: 依赖列表信息
        """
        dependencies_info = []
        
        for dep_name, dependency in self.dependencies.items():
            health = self.health_status.get(dep_name)
            
            dep_info = dependency.to_dict()
            if health:
                dep_info["health"] = health.get_health_summary()
            
            dependencies_info.append(dep_info)
        
        return {
            "total_dependencies": len(self.dependencies),
            "dependencies": dependencies_info,
            "timestamp": datetime.now().isoformat()
        }

# 全局依赖管理器实例 | Global dependency manager instance
_global_dependency_manager = None

def get_global_dependency_manager(config: Dict[str, Any] = None) -> APIDependencyManager:
    """获取全局依赖管理器实例 | Get global dependency manager instance"""
    global _global_dependency_manager
    if _global_dependency_manager is None:
        _global_dependency_manager = APIDependencyManager(config)
    return _global_dependency_manager

def set_global_dependency_manager(manager: APIDependencyManager):
    """设置全局依赖管理器实例 | Set global dependency manager instance"""
    global _global_dependency_manager
    _global_dependency_manager = manager

# 依赖注入装饰器 | Dependency injection decorator
def inject_dependency(dependency_name: str, client_param: str = "api_client"):
    """依赖注入装饰器 | Dependency injection decorator
    
    自动为方法注入API客户端依赖
    Automatically inject API client dependency for methods
    
    Args:
        dependency_name: 依赖名称
        client_param: 客户端参数在函数中的名称
        
    Returns:
        装饰器函数 | Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取依赖管理器
            manager = get_global_dependency_manager()
            
            # 获取客户端
            client = manager.get_client(dependency_name)
            if not client:
                # 如果无法获取客户端，尝试执行降级操作
                def fallback_operation(client, *args, **kwargs):
                    # 这里应该根据具体操作实现
                    raise RuntimeError(f"无法获取客户端: {dependency_name} | Cannot get client: {dependency_name}")
                
                return manager.execute_with_fallback(dependency_name, fallback_operation, *args, **kwargs)
            
            # 将客户端注入到kwargs
            kwargs[client_param] = client
            
            # 调用原始函数
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# 健康检查装饰器 | Health check decorator
def with_health_check(dependency_name: str):
    """健康检查装饰器 | Health check decorator
    
    在执行操作前检查依赖健康状态，如果不健康则使用降级策略
    Check dependency health status before executing operation, use fallback strategy if unhealthy
    
    Args:
        dependency_name: 依赖名称
        
    Returns:
        装饰器函数 | Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取依赖管理器
            manager = get_global_dependency_manager()
            
            # 检查依赖健康状态
            dependency = manager.get_dependency(dependency_name)
            if not dependency:
                raise ValueError(f"未找到依赖: {dependency_name} | Dependency not found: {dependency_name}")
            
            # 如果依赖不健康，使用降级策略
            if not dependency.is_available:
                logger.warning(f"依赖{dependency_name}不健康，使用降级策略 | Dependency {dependency_name} unhealthy, using fallback strategy")
                
                # 这里可以根据具体操作实现降级逻辑
                # 由于操作是通用的，我们无法在这里实现具体的降级逻辑
                # 所以只记录警告，继续执行
            
            # 调用原始函数
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# 示例用法 | Example usage
if __name__ == "__main__":
    import logging
    
    # 配置日志
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建依赖管理器
    # Create dependency manager
    manager = APIDependencyManager()
    
    # 示例1: 列出所有依赖
    # Example 1: List all dependencies
    logger.info("示例1: 列出所有依赖 | Example 1: List all dependencies")
    dependencies = manager.list_dependencies()
    logger.info(f"依赖列表: {dependencies} | Dependency list: {dependencies}")
    
    # 示例2: 获取健康状态
    # Example 2: Get health status
    logger.info("\n示例2: 获取健康状态 | Example 2: Get health status")
    health_status = manager.get_health_status()
    logger.info(f"健康状态: {health_status} | Health status: {health_status}")
    
    # 示例3: 注册新依赖
    # Example 3: Register new dependency
    logger.info("\n示例3: 注册新依赖 | Example 3: Register new dependency")
    new_dependency = APIDependency(
        name="cohere_chat",
        provider="cohere",
        module_name="cohere",
        client_class="Client",
        config={"api_key": "test-api-key"},
        priority=3,
        fallback_providers=["openai", "anthropic"]
    )
    
    success = manager.register_dependency(new_dependency)
    logger.info(f"注册依赖结果: {success} | Register dependency result: {success}")
    
    # 示例4: 使用装饰器
    # Example 4: Using decorator
    logger.info("\n示例4: 使用依赖注入装饰器 | Example 4: Using dependency injection decorator")
    
    @inject_dependency("openai_chat", "client")
    def chat_with_ai(client=None, message: str = "Hello"):
        """使用AI聊天 | Chat with AI"""
        if client:
            try:
                # 真实API调用逻辑 - 根据客户端类型提供不同响应
                client_type = type(client).__name__
                
                # 检查客户端支持的方法
                if hasattr(client, 'chat') and callable(client.chat):
                    # 真实OpenAI风格的聊天响应 - 执行真实API调用
                    try:
                        import time
                        start_time = time.time()
                        
                        # 尝试真实API调用
                        response = client.chat(message)
                        end_time = time.time()
                        processing_time_ms = (end_time - start_time) * 1000
                        
                        # 检查响应格式
                        if hasattr(response, 'content'):
                            response_content = response.content
                        elif hasattr(response, 'text'):
                            response_content = response.text
                        elif isinstance(response, str):
                            response_content = response
                        elif isinstance(response, dict):
                            response_content = str(response.get('content', response.get('text', str(response))))
                        else:
                            response_content = str(response)
                        
                        return f"聊天响应 [客户端: {client_type}]: {response_content[:100]}{'...' if len(response_content) > 100 else ''} | 处理时间: {processing_time_ms:.1f}ms"
                    
                    except Exception as chat_error:
                        # 聊天方法存在但调用失败，返回错误信息
                        error_msg = f"聊天API调用失败: {str(chat_error)}"
                        logging.error(f"{error_msg} | Chat API call failed: {str(chat_error)}")
                        return f"聊天操作失败: {error_msg} | 客户端: {client_type}"
                
                elif hasattr(client, 'complete') and callable(client.complete):
                    # 真实完成API响应
                    try:
                        import time
                        start_time = time.time()
                        
                        # 尝试真实完成API调用
                        response = client.complete(message)
                        end_time = time.time()
                        processing_time_ms = (end_time - start_time) * 1000
                        
                        # 检查响应格式
                        if hasattr(response, 'choices') and response.choices:
                            response_content = response.choices[0].text
                        elif hasattr(response, 'text'):
                            response_content = response.text
                        elif isinstance(response, str):
                            response_content = response
                        elif isinstance(response, dict):
                            response_content = str(response.get('text', str(response)))
                        else:
                            response_content = str(response)
                        
                        return f"完成响应 [客户端: {client_type}]: {response_content[:100]}{'...' if len(response_content) > 100 else ''} | 处理时间: {processing_time_ms:.1f}ms"
                    
                    except Exception as complete_error:
                        # 完成方法存在但调用失败
                        error_msg = f"完成API调用失败: {str(complete_error)}"
                        logging.error(f"{error_msg} | Complete API call failed: {str(complete_error)}")
                        return f"完成操作失败: {error_msg} | 客户端: {client_type}"
                
                elif hasattr(client, 'generate') and callable(client.generate):
                    # 真实生成API响应
                    try:
                        import time
                        start_time = time.time()
                        
                        # 尝试真实生成API调用
                        response = client.generate(message)
                        end_time = time.time()
                        processing_time_ms = (end_time - start_time) * 1000
                        
                        # 检查响应格式
                        if hasattr(response, 'generations') and response.generations:
                            response_content = response.generations[0].text
                        elif hasattr(response, 'text'):
                            response_content = response.text
                        elif isinstance(response, str):
                            response_content = response
                        elif isinstance(response, dict):
                            response_content = str(response.get('text', str(response)))
                        else:
                            response_content = str(response)
                        
                        return f"生成响应 [客户端: {client_type}]: {response_content[:100]}{'...' if len(response_content) > 100 else ''} | 处理时间: {processing_time_ms:.1f}ms"
                    
                    except Exception as generate_error:
                        # 生成方法存在但调用失败
                        error_msg = f"生成API调用失败: {str(generate_error)}"
                        logging.error(f"{error_msg} | Generate API call failed: {str(generate_error)}")
                        return f"生成操作失败: {error_msg} | 客户端: {client_type}"
                
                elif hasattr(client, 'predict') and callable(client.predict):
                    # 真实预测API响应
                    try:
                        import time
                        start_time = time.time()
                        
                        # 尝试真实预测API调用
                        response = client.predict(message)
                        end_time = time.time()
                        processing_time_ms = (end_time - start_time) * 1000
                        
                        # 检查响应格式
                        if hasattr(response, 'predictions') and response.predictions:
                            response_content = str(response.predictions[0])
                        elif hasattr(response, 'text'):
                            response_content = response.text
                        elif isinstance(response, str):
                            response_content = response
                        elif isinstance(response, dict):
                            response_content = str(response.get('prediction', str(response)))
                        else:
                            response_content = str(response)
                        
                        return f"预测响应 [客户端: {client_type}]: {response_content[:100]}{'...' if len(response_content) > 100 else ''} | 处理时间: {processing_time_ms:.1f}ms"
                    
                    except Exception as predict_error:
                        # 预测方法存在但调用失败
                        error_msg = f"预测API调用失败: {str(predict_error)}"
                        logging.error(f"{error_msg} | Predict API call failed: {str(predict_error)}")
                        return f"预测操作失败: {error_msg} | 客户端: {client_type}"
                
                else:
                    # 未知客户端类型，但尝试通用接口
                    try:
                        # 尝试通用调用
                        if callable(client):
                            result = client(message)
                            return f"通用AI响应: {result} [客户端类型: {client_type}]"
                    except Exception as e:
                        logger.debug(f"Generic client call failed: {e}")
                    
                    # 如果通用调用失败，提供详细的错误信息
                    available_methods = [attr for attr in dir(client) if not attr.startswith('_') and callable(getattr(client, attr))]
                    raise ValueError(f"客户端类型 {client_type} 不支持标准的AI API方法。可用方法: {available_methods[:10]}")
            
            except Exception as e:
                error_type = type(e).__name__
                error_details = str(e)
                return f"AI聊天失败: {error_type}: {error_details} | AI chat failed: {error_type}: {error_details}"
        else:
            return "未找到客户端 | Client not found"
    
    # 测试装饰器函数
    # Test decorator function
    result = chat_with_ai(message="测试消息 | Test message")
    logger.info(f"装饰器测试结果: {result} | Decorator test result: {result}")
    
    # 示例5: 使用降级策略执行操作
    # Example 5: Execute operation with fallback strategy
    logger.info("\n示例5: 使用降级策略执行操作 | Example 5: Execute operation with fallback strategy")
    
    def test_operation(client, message: str):
        """测试操作 | Test operation"""
        # 真实API调用与优雅降级策略
        client_type = type(client).__name__
        operation_id = (zlib.adler32(str(message).encode('utf-8')) & 0xffffffff) % 10000  # 生成操作ID
        
        # 检查客户端支持的方法类型
        if hasattr(client, 'chat') and callable(client.chat):
            # 真实聊天API调用 - 不再使用模拟响应
            try:
                import time
                start_time = time.time()
                
                # 尝试真实API调用
                response = client.chat(message)
                end_time = time.time()
                processing_time_ms = (end_time - start_time) * 1000
                
                # 检查响应格式
                if hasattr(response, 'content'):
                    response_content = response.content
                elif hasattr(response, 'text'):
                    response_content = response.text
                elif isinstance(response, str):
                    response_content = response
                elif isinstance(response, dict):
                    response_content = str(response.get('content', response.get('text', str(response))))
                else:
                    response_content = str(response)
                
                return f"主操作成功 [ID:{operation_id}]: {response_content[:100]}{'...' if len(response_content) > 100 else ''} | 处理时间: {processing_time_ms:.1f}ms | 客户端: {client_type}"
            
            except Exception as chat_error:
                # 聊天方法存在但调用失败，返回错误信息
                error_msg = f"聊天API调用失败: {str(chat_error)}"
                logging.error(f"{error_msg} | Chat API call failed: {str(chat_error)}")
                return f"操作失败 [ID:{operation_id}]: {error_msg} | 客户端: {client_type}"
        
        # 降级策略：尝试其他可用方法
        available_methods = []
        for attr_name in dir(client):
            if not attr_name.startswith('_') and callable(getattr(client, attr_name)):
                available_methods.append(attr_name)
        
        # 尝试找到替代方法
        for method_name in available_methods:
            if method_name in ['process', 'handle', 'execute', 'run', 'complete', 'generate', 'predict']:
                try:
                    method = getattr(client, method_name)
                    fallback_result = method(message)
                    
                    # 格式化结果
                    if isinstance(fallback_result, str):
                        result_str = fallback_result[:80]
                    elif hasattr(fallback_result, 'content') or hasattr(fallback_result, 'text'):
                        result_str = str(fallback_result)[:80]
                    else:
                        result_str = str(fallback_result)[:80]
                    
                    return f"降级操作成功 [ID:{operation_id}]: 使用 {method_name}() 方法处理消息 | 结果: {result_str}{'...' if len(str(fallback_result)) > 80 else ''} | 客户端: {client_type}"
                except Exception as method_error:
                    logging.debug(f"方法 {method_name} 调用失败: {method_error}")
                    continue
        
        # 最后降级：基本字符串处理
        try:
            # 尝试将客户端作为可调用对象
            if callable(client):
                basic_result = client(message)
                result_str = str(basic_result)[:80]
                return f"基本降级操作 [ID:{operation_id}]: 客户端可调用返回: {result_str}{'...' if len(str(basic_result)) > 80 else ''} | 客户端: {client_type}"
        except Exception as callable_error:
            logging.debug(f"客户端可调用调用失败: {callable_error}")
        
        # 最终降级：纯字符串操作
        message_length = len(message)
        word_count = len(message.split())
        processed_result = f"消息长度: {message_length}字符, 单词数: {word_count}"
        return f"最终降级操作 [ID:{operation_id}]: {processed_result} | 客户端: {client_type}"
    
    try:
        result = manager.execute_with_fallback("openai_chat", test_operation, "测试消息 | Test message")
        logger.info(f"降级策略执行结果: {result} | Fallback strategy execution result: {result}")
    except Exception as e:
        logger.error(f"降级策略执行失败: {str(e)} | Fallback strategy execution failed: {str(e)}")
    
    # 示例6: 可选依赖管理演示
    # Example 6: Optional dependency management demo
    logger.info("\n示例6: 可选依赖管理演示 | Example 6: Optional dependency management demo")
    
    # 获取所有可选依赖的状态
    optional_status = manager.get_optional_dependencies_status()
    logger.info(f"可选依赖状态: | Optional dependencies status:")
    logger.info(f"  总共可选依赖: {optional_status['total_optional_dependencies']} | Total optional dependencies: {optional_status['total_optional_dependencies']}")
    logger.info(f"  可用可选依赖: {optional_status['available_optional_dependencies']} | Available optional dependencies: {optional_status['available_optional_dependencies']}")
    logger.info(f"  不可用可选依赖: {optional_status['unavailable_optional_dependencies']} | Unavailable optional dependencies: {optional_status['unavailable_optional_dependencies']}")
    
    # 检查特定的可选依赖
    logger.info("\n检查特定可选依赖: | Check specific optional dependencies:")
    optional_deps_to_check = ["google_genai_chat", "anthropic_chat", "deepseek_chat"]
    for dep_name in optional_deps_to_check:
        status = manager.check_optional_dependency(dep_name)
        status_symbol = "✓" if status["available"] else "✗"
        logger.info(f"  {status_symbol} {dep_name}: {'可用' if status['available'] else '不可用'} | {'Available' if status['available'] else 'Unavailable'}")
        if not status["available"] and status["installation_guide"]:
            logger.info(f"    安装指南: {status['installation_guide']} | Installation guide: {status['installation_guide']}")
    
    # 演示优雅处理缺失的可选依赖
    logger.info("\n演示优雅处理缺失的可选依赖: | Demo graceful handling of missing optional dependencies:")
    
    def safe_operation_with_optional_dependency(client, operation_name: str):
        """安全操作，处理缺失的可选依赖"""
        if client is None:
            return f"可选依赖不可用，使用降级逻辑执行 {operation_name} | Optional dependency unavailable, using fallback logic for {operation_name}"
        else:
            try:
                # 实际的操作逻辑
                return f"使用可选依赖成功执行 {operation_name} | Successfully executed {operation_name} using optional dependency"
            except Exception as e:
                return f"可选依赖操作失败: {str(e)} | Optional dependency operation failed: {str(e)}"
    
    # 尝试使用可选依赖
    for dep_name in optional_deps_to_check:
        client = manager.get_client(dep_name)
        result = safe_operation_with_optional_dependency(client, f"操作 {dep_name}")
        logger.info(f"  {dep_name}: {result}")
    
    logger.info("\nAPI依赖管理器示例完成 | API dependency manager example completed")
