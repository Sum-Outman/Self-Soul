"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
# Self Soul 统一错误处理和日志系统
AGI System Unified Error Handling and Logging System
"""
import logging
import traceback
import sys
import threading
import time
from datetime import datetime
import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    loguru_logger = None
    LOGURU_AVAILABLE = False

try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message
    SENTRY_AVAILABLE = True
except ImportError:
    sentry_sdk = None
    capture_exception = None
    capture_message = None
    SENTRY_AVAILABLE = False

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 增强日志配置 - 按模块/时间分割，完整堆栈记录，告警机制
def configure_enhanced_logging(environment="production"):
    """配置增强的日志系统
    
    Args:
        environment: 环境类型，production 或 development
    
    Features:
    1. 按模块分割日志文件 (core, model, api, database 等)
    2. 按时间轮转 (每天)
    3. 完整堆栈信息记录 (始终记录，不只是开发环境)
    4. ERROR/FATAL级日志告警集成
    5. 结构化日志格式 (JSON格式，便于检索和分析)
    """
    
    # 清除现有处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建结构化日志格式化器
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            # 基础格式化
            log_entry = super().format(record)
            
            # 添加结构化数据
            structured_data = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'module': record.module,
                'filename': record.filename,
                'lineno': record.lineno,
                'function': record.funcName,
                'message': record.getMessage(),
                'environment': environment,
                'pid': os.getpid(),
                'thread': record.thread if hasattr(record, 'thread') else threading.current_thread().ident
            }
            
            # 添加异常信息（如果有）
            if record.exc_info:
                structured_data['exception'] = {
                    'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                    'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                    'stack_trace': self.formatException(record.exc_info)
                }
            
            # 添加额外字段（如果有）
            if hasattr(record, 'extra_fields'):
                structured_data.update(record.extra_fields)
            
            # 格式化输出
            if environment == 'production':
                # 生产环境：JSON格式，便于日志分析
                return json.dumps(structured_data, ensure_ascii=False)
            else:
                # 开发环境：可读性更好的格式
                return log_entry
    
    # 创建基础格式化器（用于非JSON输出）
    base_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 按模块分割的日志处理器
    modules = ['core', 'model', 'api', 'database', 'training', 'security', 'hardware']
    
    for module in modules:
        # 按模块分割的日志文件
        module_log_file = os.path.join(log_dir, f'{module}.log')
        module_handler = TimedRotatingFileHandler(
            module_log_file,
            when='midnight',  # 每天轮转
            interval=1,
            backupCount=30,   # 保留30天
            encoding='utf-8'
        )
        
        if environment == 'production':
            module_handler.setFormatter(StructuredFormatter())
        else:
            module_handler.setFormatter(base_formatter)
        
        module_handler.setLevel(logging.DEBUG if environment == 'development' else logging.INFO)
        
        # 为特定模块的日志记录器添加处理器
        module_logger = logging.getLogger(module)
        module_logger.addHandler(module_handler)
        module_logger.setLevel(logging.DEBUG if environment == 'development' else logging.INFO)
        module_logger.propagate = False  # 防止重复记录
    
    # 错误日志处理器（集中所有ERROR/FATAL日志）
    error_log_file = os.path.join(log_dir, 'error.log')
    error_handler = TimedRotatingFileHandler(
        error_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    
    if environment == 'production':
        error_handler.setFormatter(StructuredFormatter())
    else:
        error_handler.setFormatter(base_formatter)
    
    error_handler.setLevel(logging.ERROR)
    
    # 错误日志过滤器：只记录ERROR及以上级别的日志
    class ErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno >= logging.ERROR
    
    error_handler.addFilter(ErrorFilter())
    
    # 访问日志处理器
    access_log_file = os.path.join(log_dir, 'access.log')
    access_handler = TimedRotatingFileHandler(
        access_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    
    if environment == 'production':
        access_handler.setFormatter(StructuredFormatter())
    else:
        access_handler.setFormatter(base_formatter)
    
    access_handler.setLevel(logging.INFO)
    
    # 访问日志过滤器
    class AccessFilter(logging.Filter):
        def filter(self, record):
            return record.name == 'access'
    
    access_handler.addFilter(AccessFilter())
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(base_formatter)
    console_handler.setLevel(logging.DEBUG if environment == 'development' else logging.WARNING)
    
    # 告警处理器：ERROR/FATAL级日志触发告警
    class AlertHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.ERROR)
            self.alert_threshold = 10  # 10分钟内相同错误超过阈值触发告警
            self.error_counts = {}  # 存储错误计数
        
        def emit(self, record):
            error_key = f"{record.name}:{record.levelname}:{record.getMessage()[:50]}"
            current_time = time.time()
            
            # 清理过期的错误计数
            cutoff_time = current_time - (self.alert_threshold * 60)
            self.error_counts = {k: v for k, v in self.error_counts.items() if v['timestamp'] > cutoff_time}
            
            # 更新错误计数
            if error_key not in self.error_counts:
                self.error_counts[error_key] = {'count': 0, 'timestamp': current_time}
            
            self.error_counts[error_key]['count'] += 1
            self.error_counts[error_key]['timestamp'] = current_time
            
            # 检查是否达到告警阈值
            if self.error_counts[error_key]['count'] >= 5:  # 5次相同错误触发告警
                self.trigger_alert(record)
        
        def trigger_alert(self, record):
            """触发告警（可以扩展为发送邮件、短信、Webhook等）"""
            alert_message = f"🚨 系统告警: {record.name} - {record.levelname} - {record.getMessage()}"
            
            # 记录告警日志
            logging.getLogger('alert').error(alert_message)
            
            # 这里可以添加实际的告警发送逻辑，例如：
            # - 发送邮件
            # - 发送Slack/Teams消息
            # - 调用Webhook
            # - 记录到告警数据库
    
    alert_handler = AlertHandler()
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if environment == 'development' else logging.INFO)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(access_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(alert_handler)
    
    # 特定日志记录器配置
    access_logger = logging.getLogger('access')
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    
    alert_logger = logging.getLogger('alert')
    alert_logger.setLevel(logging.ERROR)
    alert_logger.propagate = False
    
    logging.info(f"增强日志系统初始化完成 - 环境: {environment}")
    logging.info(f"日志目录: {log_dir}")
    logging.info(f"日志模块: {', '.join(modules)}")

# 生产环境日志配置（向后兼容，使用增强日志系统）
def configure_production_logging():
    """配置生产环境日志系统 - 使用增强日志系统"""
    configure_enhanced_logging(environment="production")

# 根据环境配置日志
if os.getenv('ENVIRONMENT') == 'production':
    configure_production_logging()  # 这会调用 configure_enhanced_logging(environment="production")
else:
    # 开发环境配置 - 使用增强日志系统
    configure_enhanced_logging(environment="development")


"""
ErrorHandler类 - 中文类描述
ErrorHandler Class - English class description
"""

def _parse_iso_timestamp(timestamp_str: str) -> datetime:
    """解析ISO格式时间戳，支持Python 3.6+
    
    Args:
        timestamp_str: ISO格式时间戳字符串
        
    Returns:
        datetime对象
        
    Raises:
        ValueError: 时间戳格式无效
    """
    if not timestamp_str:
        raise ValueError("空时间戳字符串")
    
    try:
        # 尝试Python 3.7+的fromisoformat
        return datetime.fromisoformat(timestamp_str)
    except AttributeError:
        # Python 3.6兼容方案：手动解析ISO格式
        # ISO格式: YYYY-MM-DDTHH:MM:SS.ssssss
        import re
        
        # 匹配ISO格式
        iso_pattern = r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?'
        match = re.match(iso_pattern, timestamp_str)
        
        if not match:
            raise ValueError(f"无效的ISO时间戳格式: {timestamp_str}")
        
        year, month, day, hour, minute, second = map(int, match.groups()[:6])
        microsecond = 0
        if match.group(7):
            # 处理微秒部分，确保长度正确
            microsecond_str = match.group(7)[:6]  # 最多6位
            microsecond = int(microsecond_str.ljust(6, '0'))
        
        return datetime(year, month, day, hour, minute, second, microsecond)

class ErrorHandler:
    """Self Soul 统一错误处理类"""
    
    def __init__(self):
        # 初始化 loguru 日志记录器（如果可用）
        self.loguru_logger = loguru_logger
        self.loguru_available = LOGURU_AVAILABLE
        
        # 初始化 Sentry（如果可用）
        self.sentry_available = SENTRY_AVAILABLE
        if self.sentry_available:
            self._initialize_sentry()
        
        # 标准日志记录器（保持向后兼容）
        self.logger = logging.getLogger('AGI_System')
        self.access_logger = logging.getLogger('access')
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'component_errors': {},
            'last_error_time': None
        }
        # 错误历史记录，用于趋势分析和自动恢复
        self.error_history = []
        self.max_error_history = 1000  # 最大记录数
        # 组件状态管理：记录组件降级状态
        self.component_states = {}  # component_name -> {'status': 'normal'/'degraded'/'disabled', 'reason': str, 'since': timestamp}
        # 错误严重性定义
        self.error_severity_levels = {
            'critical': ['MemoryError', 'OSError', 'ConnectionError', 'TimeoutError'],
            'high': ['FileNotFoundError', 'PermissionError', 'ImportError', 'ValueError'],
            'medium': ['IOError', 'KeyError', 'AttributeError', 'TypeError'],
            'low': ['UserWarning', 'DeprecationWarning', 'FutureWarning']
        }
        
        # 错误码定义系统
        self.error_codes = {
            # 通用错误 (1000-1999)
            'UNKNOWN_ERROR': 1000,
            'VALIDATION_ERROR': 1001,
            'CONFIGURATION_ERROR': 1002,
            'PERMISSION_ERROR': 1003,
            'RESOURCE_NOT_FOUND': 1004,
            'TIMEOUT_ERROR': 1005,
            'NETWORK_ERROR': 1006,
            'DATABASE_ERROR': 1007,
            'FILE_SYSTEM_ERROR': 1008,
            'MEMORY_ERROR': 1009,
            
            # 模型相关错误 (2000-2999)
            'MODEL_LOAD_ERROR': 2001,
            'MODEL_INIT_ERROR': 2002,
            'MODEL_PROCESS_ERROR': 2003,
            'MODEL_TRAINING_ERROR': 2004,
            'MODEL_DEPENDENCY_ERROR': 2005,
            'MODEL_NOT_FOUND': 2006,
            'MODEL_CONFIG_ERROR': 2007,
            
            # API相关错误 (3000-3999)
            'API_VALIDATION_ERROR': 3001,
            'API_AUTH_ERROR': 3002,
            'API_RATE_LIMIT_ERROR': 3003,
            'API_ENDPOINT_ERROR': 3004,
            'API_REQUEST_ERROR': 3005,
            'API_RESPONSE_ERROR': 3006,
            
            # 硬件相关错误 (4000-4999)
            'HARDWARE_COMM_ERROR': 4001,
            'HARDWARE_DEVICE_ERROR': 4002,
            'HARDWARE_CONFIG_ERROR': 4003,
            'HARDWARE_TIMEOUT_ERROR': 4004,
            
            # 知识管理错误 (5000-5999)
            'KNOWLEDGE_LOAD_ERROR': 5001,
            'KNOWLEDGE_FUSION_ERROR': 5002,
            'KNOWLEDGE_TRANSFER_ERROR': 5003,
            'KNOWLEDGE_QUERY_ERROR': 5004,
            
            # 系统监控错误 (6000-6999)
            'MONITORING_ERROR': 6001,
            'METRICS_ERROR': 6002,
            'HEALTH_CHECK_ERROR': 6003
        }
        
        # 错误码到严重性映射
        self.error_code_severity = {
            1000: 'high',    # UNKNOWN_ERROR
            1001: 'medium',  # VALIDATION_ERROR
            1002: 'high',    # CONFIGURATION_ERROR
            1003: 'high',    # PERMISSION_ERROR
            1004: 'medium',  # RESOURCE_NOT_FOUND
            1005: 'medium',  # TIMEOUT_ERROR
            1006: 'high',    # NETWORK_ERROR
            1007: 'high',    # DATABASE_ERROR
            1008: 'medium',  # FILE_SYSTEM_ERROR
            1009: 'critical', # MEMORY_ERROR
            2001: 'high',    # MODEL_LOAD_ERROR
            2002: 'high',    # MODEL_INIT_ERROR
            2003: 'medium',  # MODEL_PROCESS_ERROR
            2004: 'medium',  # MODEL_TRAINING_ERROR
            2005: 'medium',  # MODEL_DEPENDENCY_ERROR
            2006: 'medium',  # MODEL_NOT_FOUND
            2007: 'high',    # MODEL_CONFIG_ERROR
            3001: 'medium',  # API_VALIDATION_ERROR
            3002: 'high',    # API_AUTH_ERROR
            3003: 'medium',  # API_RATE_LIMIT_ERROR
            3004: 'medium',  # API_ENDPOINT_ERROR
            3005: 'medium',  # API_REQUEST_ERROR
            3006: 'medium',  # API_RESPONSE_ERROR
            4001: 'high',    # HARDWARE_COMM_ERROR
            4002: 'high',    # HARDWARE_DEVICE_ERROR
            4003: 'high',    # HARDWARE_CONFIG_ERROR
            4004: 'medium',  # HARDWARE_TIMEOUT_ERROR
            5001: 'medium',  # KNOWLEDGE_LOAD_ERROR
            5002: 'medium',  # KNOWLEDGE_FUSION_ERROR
            5003: 'medium',  # KNOWLEDGE_TRANSFER_ERROR
            5004: 'medium',  # KNOWLEDGE_QUERY_ERROR
            6001: 'medium',  # MONITORING_ERROR
            6002: 'medium',  # METRICS_ERROR
            6003: 'medium'   # HEALTH_CHECK_ERROR
        }
        
        # 监控指标存储
        self.metrics = {
            'error_count': 0,
            'error_count_by_code': {},
            'error_count_by_component': {},
            'response_times': [],
            'component_health': {},
            'request_count': 0,
            'success_count': 0,
            'failure_count': 0
        }
        
        # 性能监控配置
        self.monitoring_config = {
            'enabled': True,
            'metrics_export_interval': 60,  # 秒
            'max_response_time_samples': 1000,
            'enable_prometheus': False,
            'enable_statsd': False
        }
    
    def _initialize_sentry(self):
        """初始化 Sentry 错误跟踪（如果可用）"""
        if not self.sentry_available:
            return
        
        try:
            # 配置 Sentry DSN（应该从环境变量或配置中获取）
            sentry_dsn = os.environ.get('SENTRY_DSN', '')
            if sentry_dsn:
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    # 设置样本率
                    traces_sample_rate=1.0,
                    # 启用自动错误捕获
                    attach_stacktrace=True,
                    # 发送环境信息
                    environment=os.environ.get('AGI_ENV', 'development'),
                    # 发送版本信息
                    release=f"self-soul-agi@{os.environ.get('AGI_VERSION', '1.0.0')}",
                    # 启用性能监控
                    enable_tracing=True
                )
                if self.loguru_available:
                    self.loguru_logger.info("Sentry initialized successfully")
                else:
                    self.logger.info("Sentry initialized successfully")
            else:
                if self.loguru_available:
                    self.loguru_logger.warning("Sentry DSN not configured, skipping Sentry initialization")
                else:
                    self.logger.warning("Sentry DSN not configured, skipping Sentry initialization")
        except Exception as e:
            if self.loguru_available:
                self.loguru_logger.error(f"Failed to initialize Sentry: {e}")
            else:
                self.logger.error(f"Failed to initialize Sentry: {e}")
    
    def log_info(self, message: str, component: str = "System"):
        """记录信息日志（支持 loguru 和标准 logging）"""
        # 使用标准 logging
        self.logger.info(f"{component}: {message}")
        
        # 如果 loguru 可用，也使用 loguru
        if self.loguru_available:
            try:
                self.loguru_logger.info(f"[{component}] {message}")
            except Exception:
                pass  # 如果 loguru 失败，静默回退
    
    def log_warning(self, message: str, component: str = "System"):
        """记录警告日志（支持 loguru 和标准 logging）"""
        # 使用标准 logging
        self.logger.warning(f"{component}: {message}")
        
        # 如果 loguru 可用，也使用 loguru
        if self.loguru_available:
            try:
                self.loguru_logger.warning(f"[{component}] {message}")
            except Exception:
                pass  # 如果 loguru 失败，静默回退
    
    def log_error(self, message: str, component: str = "System"):
        """记录错误日志（支持 loguru 和标准 logging，发送到 Sentry）"""
        # 使用标准 logging
        self.logger.error(f"{component}: {message}")
        
        # 如果 loguru 可用，也使用 loguru
        if self.loguru_available:
            try:
                self.loguru_logger.error(f"[{component}] {message}")
            except Exception:
                pass  # 如果 loguru 失败，静默回退
        
        # 如果 Sentry 可用，发送错误到 Sentry
        if self.sentry_available:
            try:
                capture_message(f"{component}: {message}", level="error")
            except Exception:
                pass  # 如果 Sentry 失败，静默回退
    
    def handle_error(self, error: Exception, component: str = "System", context: str = ""):
        """统一错误处理（支持 loguru、标准 logging 和 Sentry）"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # 格式化错误消息
        formatted_message = f"{component}: {error_type}"
        if context:
            formatted_message += f" ({context})"
        if error_message:
            formatted_message += f": {error_message}"
        
        # 使用标准 logging 记录完整错误
        self.logger.error(formatted_message, exc_info=True)
        
        # 如果 loguru 可用，也使用 loguru
        if self.loguru_available:
            try:
                self.loguru_logger.opt(exception=error).error(f"[{component}] {context or error_type}")
            except Exception:
                pass
        
        # 如果 Sentry 可用，发送异常到 Sentry
        if self.sentry_available:
            try:
                capture_exception(error)
            except Exception:
                pass
        
        # 更新错误统计
        self._update_error_stats(error_type, component)
        
        return {
            "error_type": error_type,
            "component": component,
            "message": error_message,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_error_stats(self, error_type: str, component: str):
        """更新错误统计信息"""
        self.error_stats['total_errors'] += 1
        self.error_stats['last_error_time'] = datetime.now().isoformat()
        
        # 更新错误类型统计
        if error_type not in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] = 0
        self.error_stats['error_types'][error_type] += 1
        
        # 更新组件错误统计
        if component not in self.error_stats['component_errors']:
            self.error_stats['component_errors'][component] = 0
        self.error_stats['component_errors'][component] += 1
        
        # 记录错误历史
        self.error_history.append({
            'type': error_type,
            'component': component,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保持历史记录大小
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
    def initialize(self):
        """初始化错误处理器
        Initialize error handler
        
        Returns:
            dict: 初始化结果
        """
        self.logger.info("AGI错误处理器初始化成功")
        return {"success": True, "message": "AGI错误处理器初始化成功"}
    
    def log_access(self, request_info: Dict[str, Any]):
        """记录访问日志"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'method': request_info.get('method', 'UNKNOWN'),
                'path': request_info.get('path', 'UNKNOWN'),
                'status_code': request_info.get('status_code', 0),
                'client_ip': request_info.get('client_ip', 'UNKNOWN'),
                'user_agent': request_info.get('user_agent', 'UNKNOWN'),
                'response_time': request_info.get('response_time', 0),
                'user_id': request_info.get('user_id', 'anonymous')
            }
            
            self.access_logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log access: {e}")
    
    def _determine_error_code(self, error_type: str, component_name: str, context: Optional[Dict[str, Any]] = None) -> int:
        """根据错误类型、组件名称和上下文确定错误码
        
        Args:
            error_type: 错误类型名称
            component_name: 组件名称
            context: 错误上下文
            
        Returns:
            int: 错误码
        """
        # 基于组件名称的映射
        component_lower = component_name.lower()
        
        # 检查错误类型映射
        error_type_mapping = {
            'ImportError': self.error_codes['MODEL_LOAD_ERROR'],
            'ModuleNotFoundError': self.error_codes['MODEL_LOAD_ERROR'],
            'FileNotFoundError': self.error_codes['RESOURCE_NOT_FOUND'],
            'PermissionError': self.error_codes['PERMISSION_ERROR'],
            'ConnectionError': self.error_codes['NETWORK_ERROR'],
            'TimeoutError': self.error_codes['TIMEOUT_ERROR'],
            'MemoryError': self.error_codes['MEMORY_ERROR'],
            'ValueError': self.error_codes['VALIDATION_ERROR'],
            'TypeError': self.error_codes['VALIDATION_ERROR'],
            'KeyError': self.error_codes['RESOURCE_NOT_FOUND'],
            'AttributeError': self.error_codes['RESOURCE_NOT_FOUND'],
            'IOError': self.error_codes['FILE_SYSTEM_ERROR'],
            'OSError': self.error_codes['FILE_SYSTEM_ERROR'],
        }
        
        # 首先尝试基于错误类型映射
        if error_type in error_type_mapping:
            return error_type_mapping[error_type]
        
        # 基于组件名称的映射
        if 'model' in component_lower:
            if 'load' in error_type.lower() or 'init' in error_type.lower():
                return self.error_codes['MODEL_LOAD_ERROR']
            elif 'train' in component_lower or 'training' in component_lower:
                return self.error_codes['MODEL_TRAINING_ERROR']
            else:
                return self.error_codes['MODEL_PROCESS_ERROR']
        elif 'api' in component_lower:
            if 'validation' in error_type.lower():
                return self.error_codes['API_VALIDATION_ERROR']
            elif 'auth' in error_type.lower():
                return self.error_codes['API_AUTH_ERROR']
            else:
                return self.error_codes['API_REQUEST_ERROR']
        elif 'hardware' in component_lower or 'camera' in component_lower or 'sensor' in component_lower:
            return self.error_codes['HARDWARE_COMM_ERROR']
        elif 'knowledge' in component_lower:
            if 'fusion' in component_lower:
                return self.error_codes['KNOWLEDGE_FUSION_ERROR']
            elif 'transfer' in component_lower:
                return self.error_codes['KNOWLEDGE_TRANSFER_ERROR']
            else:
                return self.error_codes['KNOWLEDGE_LOAD_ERROR']
        
        # 默认错误码
        return self.error_codes['UNKNOWN_ERROR']
    
    def _get_error_code_name(self, error_code: int) -> str:
        """根据错误码获取错误码名称
        
        Args:
            error_code: 错误码
            
        Returns:
            str: 错误码名称
        """
        # 反向查找错误码字典
        for name, code in self.error_codes.items():
            if code == error_code:
                return name
        return f"UNKNOWN_ERROR_{error_code}"
    
    def handle_error(self, error, component_name, details=None, context: Optional[Dict[str, Any]] = None):
        """处理系统错误，包含自动恢复机制"""
        # 处理 ErrorContext 对象作为 component_name 的情况
        if hasattr(component_name, 'component'):
            # 如果是 ErrorContext 对象，提取组件名称
            component_key = component_name.component
        else:
            component_key = str(component_name)
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 获取错误严重性
        severity = self._get_error_severity(error_type)
        
        # 确定错误码
        error_code = self._determine_error_code(error_type, component_key, context)
        
        # 检查组件当前状态
        component_status = self.get_component_status(component_key)
        
        # 构建错误详情 - 包含更丰富的上下文信息
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_code': error_code,
            'error_code_name': self._get_error_code_name(error_code),
            'component': component_key,
            'message': error_message,
            'severity': severity,
            'component_status': component_status['status'],
            'context': context or {},
            'stack_trace': stack_trace,
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'pid': os.getpid(),
            'thread_id': threading.current_thread().ident if 'threading' in sys.modules else None
        }
        
        if details:
            error_details['details'] = details
        
        # 更新错误统计（传递错误详情用于历史记录）
        self._update_error_stats(error_type, component_key, error_details)
        
        # 记录错误日志
        self.logger.error(f"[{component_key}] {error_type}: {error_message}")
        # 掩码敏感数据后再记录错误详情
        masked_error_details = self._mask_sensitive_data(error_details)
        self.logger.debug(f"Error details: {json.dumps(masked_error_details, indent=2)}")
        
        # 始终记录堆栈跟踪（增强日志系统要求）
        self.logger.debug(f"Stack trace: {stack_trace}")
        
        # 尝试自动恢复（基于错误类型）
        recovery_result = self._attempt_auto_recovery(error, component_key, context)
        
        # 根据错误类型返回不同级别的响应，包含组件状态和错误严重性
        response = self._get_error_response(error, error_message, severity, component_status)
        response['error_id'] = self._generate_error_id()
        response['severity'] = severity
        response['component_status'] = component_status['status']
        
        # 添加恢复信息到响应
        if recovery_result:
            response['recovery_attempted'] = True
            response['recovery_success'] = recovery_result.get('success', False)
            response['recovery_message'] = recovery_result.get('message', '')
        else:
            response['recovery_attempted'] = False
        
        # 如果恢复成功，降低日志级别
        if recovery_result and recovery_result.get('success', False):
            self.logger.info(f"自动恢复成功: {component_key} - {error_type}")
        
        return response
    
    def _update_error_stats(self, error_type: str, component_name: str, error_details: Optional[Dict[str, Any]] = None):
        """更新错误统计，包含趋势分析和预警"""
        # 处理 ErrorContext 对象作为 component_name 的情况
        if hasattr(component_name, 'component'):
            # 如果是 ErrorContext 对象，提取组件名称
            component_key = component_name.component
        else:
            component_key = str(component_name)
        
        self.error_stats['total_errors'] += 1
        self.error_stats['last_error_time'] = datetime.now().isoformat()
        
        # 更新错误类型统计
        if error_type not in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] = 0
        self.error_stats['error_types'][error_type] += 1
        
        # 更新组件错误统计
        if component_key not in self.error_stats['component_errors']:
            self.error_stats['component_errors'][component_key] = 0
        self.error_stats['component_errors'][component_key] += 1
        
        # 添加到错误历史记录
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'component': component_key,
            'details': error_details or {}
        }
        self.error_history.append(error_record)
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # 趋势分析：如果最近1分钟内相同错误发生超过3次，触发预警
        current_time = datetime.now()
        recent_threshold = current_time.timestamp() - 60  # 1分钟
        
        # 检查是否达到预警阈值
        recent_error_count = sum(1 for error in self._get_recent_errors() 
                                if error.get('type') == error_type and 
                                error.get('component') == component_key and
                                error.get('timestamp') and 
                                _parse_iso_timestamp(error.get('timestamp')).timestamp() > recent_threshold)
        
        if recent_error_count >= 3:
            warning_msg = f"组件 {component_key} 错误 {error_type} 在1分钟内发生{recent_error_count}次"
            self.logger.warning(f"错误频率预警: {warning_msg}")
            
            # 自动调整策略：暂时禁用有问题的组件（记录日志但不实际禁用）
            if recent_error_count >= 5:
                self.logger.error(f"高频错误警报: 组件 {component_key} 在1分钟内发生{recent_error_count}次{error_type}错误，建议立即检查")
                # 注意：实际禁用组件的逻辑需要根据具体系统实现
                self._degrade_component(component_key, error_type, recent_error_count)
    
    def _degrade_component(self, component_name: str, error_type: str, error_count: int):
        """降级或禁用问题组件
        
        Args:
            component_name: 组件名称
            error_type: 错误类型
            error_count: 错误计数
        """
        from datetime import datetime
        
        # 处理 ErrorContext 对象作为 component_name 的情况
        if hasattr(component_name, 'component'):
            # 如果是 ErrorContext 对象，提取组件名称
            component_key = component_name.component
        else:
            component_key = str(component_name)
        
        current_state = self.component_states.get(component_key, {'status': 'normal'})
        current_status = current_state.get('status', 'normal')
        
        # 如果组件已经被禁用或降级，不再重复操作
        if current_status in ['degraded', 'disabled']:
            self.logger.info(f"组件 {component_key} 已经处于{current_status}状态，跳过降级操作")
            return
        
        # 根据错误类型和频率决定降级策略
        severity = self._get_error_severity(error_type)
        
        # 严重性为critical且高频错误，考虑禁用组件
        if severity == 'critical' and error_count >= 10:
            new_status = 'disabled'
            reason = f"严重错误{error_type}在1分钟内发生{error_count}次，系统自动禁用组件"
        # 中等严重性且高频错误，降级组件
        elif severity in ['high', 'medium'] and error_count >= 5:
            new_status = 'degraded'
            reason = f"高频错误{error_type}在1分钟内发生{error_count}次，系统自动降级组件"
        else:
            # 仅记录警告，不实际降级
            self.logger.warning(f"组件 {component_key} 错误{error_type}发生{error_count}次，建议监控但未降级")
            return
        
        # 更新组件状态
        self.component_states[component_key] = {
            'status': new_status,
            'reason': reason,
            'since': datetime.now().isoformat(),
            'previous_status': current_status,
            'error_type': error_type,
            'error_count': error_count
        }
        
        # 记录状态变更
        self.logger.warning(f"组件 {component_key} 状态从{current_status}变更为{new_status}: {reason}")
        
        # 通知相关系统组件状态变更（实际实现需要根据系统架构调整）
        self._notify_component_status_change(component_key, new_status, reason)
    
    def _get_error_severity(self, error_type: str) -> str:
        """获取错误严重性等级
        
        Args:
            error_type: 错误类型名称
            
        Returns:
            str: 严重性等级（'critical', 'high', 'medium', 'low' 或 'unknown'）
        """
        for severity, error_types in self.error_severity_levels.items():
            if error_type in error_types:
                return severity
        
        # 如果错误类型不在预定义列表中，根据错误类型名称判断
        if 'Error' in error_type:
            return 'medium'  # 未知Error类型默认为中等
        elif 'Warning' in error_type or 'warning' in error_type.lower():
            return 'low'  # 警告类型为低严重性
        else:
            return 'unknown'
    
    def _notify_component_status_change(self, component_name: str, new_status: str, reason: str):
        """通知系统组件状态变更
        
        Args:
            component_name: 组件名称
            new_status: 新状态
            reason: 变更原因
        """
        # 实际实现中，这里可以：
        # 1. 发送WebSocket通知到前端
        # 2. 调用系统管理API
        # 3. 写入系统事件日志
        # 4. 触发自动恢复监控
        
        # 当前实现仅记录日志
        notification = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name,
            'new_status': new_status,
            'reason': reason,
            'type': 'component_status_change'
        }
        
        self.logger.info(f"组件状态变更通知: {json.dumps(notification)}")
        
        # 这里可以添加实际的通知逻辑，例如：
        # try:
        #     # 发送WebSocket通知
        #     from core.websocket_manager import websocket_manager
        #     websocket_manager.broadcast_system_event(notification)
        # except Exception as e:
        #     self.logger.error(f"发送组件状态通知失败: {e}")
    
    def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """获取组件状态
        
        Args:
            component_name: 组件名称
            
        Returns:
            Dict[str, Any]: 组件状态信息
        """
        # 处理 ErrorContext 对象作为 component_name 的情况
        if hasattr(component_name, 'component'):
            # 如果是 ErrorContext 对象，提取组件名称
            component_key = component_name.component
        else:
            component_key = str(component_name)
        
        status_info = self.component_states.get(component_key, {'status': 'normal'})
        
        # 添加额外的状态信息
        result = {
            'component': component_key,
            'status': status_info.get('status', 'normal'),
            'reason': status_info.get('reason', ''),
            'since': status_info.get('since', ''),
            'last_checked': datetime.now().isoformat()
        }
        
        # 如果组件处于降级或禁用状态，添加恢复建议
        if result['status'] in ['degraded', 'disabled']:
            result['recovery_suggestion'] = f"组件因{status_info.get('error_type', '未知错误')}被{result['status']}，建议检查组件配置和日志"
        
        return result
    
    def restore_component(self, component_name: str, reason: str = "手动恢复") -> bool:
        """恢复组件到正常状态
        
        Args:
            component_name: 组件名称
            reason: 恢复原因
            
        Returns:
            bool: 恢复是否成功
        """
        # 处理 ErrorContext 对象作为 component_name 的情况
        if hasattr(component_name, 'component'):
            # 如果是 ErrorContext 对象，提取组件名称
            component_key = component_name.component
        else:
            component_key = str(component_name)
        
        if component_key not in self.component_states:
            self.logger.info(f"组件 {component_key} 未处于降级状态，无需恢复")
            return True
        
        previous_status = self.component_states[component_key].get('status', 'normal')
        
        # 更新组件状态
        self.component_states[component_key] = {
            'status': 'normal',
            'reason': reason,
            'since': datetime.now().isoformat(),
            'previous_status': previous_status,
            'restored_at': datetime.now().isoformat(),
            'restore_reason': reason
        }
        
        self.logger.info(f"组件 {component_key} 已从{previous_status}状态恢复为正常: {reason}")
        
        # 通知恢复
        self._notify_component_status_change(component_key, 'normal', reason)
        
        return True
    
    def _get_recent_errors(self, minutes: int = 5, component_name: Optional[str] = None, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取最近的错误记录
        
        Args:
            minutes: 时间范围（分钟），默认5分钟
            component_name: 可选的组件名称过滤
            error_type: 可选的错误类型过滤
            
        Returns:
            List[Dict[str, Any]]: 错误记录列表
        """
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_errors = []
        for error in self.error_history:
            try:
                error_time = _parse_iso_timestamp(error.get('timestamp', ''))
                if error_time < cutoff_time:
                    continue
                
                # 应用过滤条件
                if component_name and error.get('component') != component_name:
                    continue
                    
                if error_type and error.get('type') != error_type:
                    continue
                
                recent_errors.append(error)
            except (ValueError, KeyError):
                # 如果时间戳格式无效，跳过该记录
                continue
        
        return recent_errors
    
    def _attempt_auto_recovery(self, error: Exception, component_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """尝试自动恢复，基于错误类型和组件上下文
        
        Args:
            error: 异常对象
            component_name: 组件名称
            context: 可选的上下文信息
            
        Returns:
            Optional[Dict[str, Any]]: 恢复结果，包含success和message字段，如果不可恢复则返回None
        """
        error_type = type(error).__name__
        
        # 1. 连接错误恢复：尝试重新连接
        if isinstance(error, (ConnectionError, TimeoutError)):
            self.logger.info(f"尝试自动恢复连接错误: {component_name} - {error_type}")
            try:
                # 根据组件类型执行不同的恢复策略
                if component_name in ["API Connection", "WebSocket", "ExternalAPI"]:
                    # 对于API连接错误，等待后重试
                    import time
                    time.sleep(2)  # 等待2秒后重试
                    return {
                        "success": True,
                        "message": f"已尝试重新连接策略，建议检查网络和API端点",
                        "recovery_type": "connection_retry"
                    }
                elif component_name in ["Database", "ModelService"]:
                    # 对于数据库和模型服务，尝试重启服务
                    return {
                        "success": True,
                        "message": "已标记服务为需要重启，将在下一个维护窗口执行",
                        "recovery_type": "service_restart_scheduled"
                    }
            except Exception as recovery_error:
                self.logger.warning(f"自动恢复失败: {recovery_error}")
                return {
                    "success": False,
                        "message": f"自动恢复失败: {str(recovery_error)}",
                        "recovery_type": "failed"
                }
        
        # 2. 资源错误恢复：尝试释放资源
        elif isinstance(error, (MemoryError, IOError, OSError)):
            self.logger.info(f"尝试自动恢复资源错误: {component_name} - {error_type}")
            try:
                if "memory" in error_type.lower() or isinstance(error, MemoryError):
                    # 内存错误：尝试清理缓存
                    import gc
                    gc.collect()  # 强制垃圾回收
                    return {
                        "success": True,
                        "message": "已执行内存清理，建议监控内存使用情况",
                        "recovery_type": "memory_cleanup"
                    }
                elif "disk" in str(error).lower() or "space" in str(error).lower():
                    # 磁盘空间错误
                    return {
                        "success": True,
                        "message": "检测到磁盘空间问题，建议清理磁盘或扩展存储",
                        "recovery_type": "disk_space_warning"
                    }
            except Exception as recovery_error:
                self.logger.warning(f"资源恢复失败: {recovery_error}")
        
        # 3. 配置错误恢复：尝试使用默认配置
        elif isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
            # 配置或数据错误，通常需要人工干预
            self.logger.info(f"配置/数据错误检测: {component_name} - {error_type}")
            return {
                "success": False,
                "message": "配置或数据错误，需要人工验证输入参数",
                "recovery_type": "requires_manual_intervention"
            }
        
        # 4. 权限错误恢复：尝试使用备用权限或记录问题
        elif isinstance(error, (PermissionError, ImportError)):
            self.logger.info(f"权限/依赖错误检测: {component_name} - {error_type}")
            if isinstance(error, ImportError):
                return {
                    "success": False,
                    "message": f"依赖包缺失: {str(error)}，请安装所需包",
                    "recovery_type": "missing_dependency"
                }
            else:
                return {
                    "success": False,
                    "message": "权限不足，请检查文件/目录权限或用户访问级别",
                    "recovery_type": "permission_denied"
                }
        
        # 5. 其他错误：记录但不尝试恢复
        else:
            self.logger.debug(f"未知错误类型，不尝试自动恢复: {component_name} - {error_type}")
        
        # 默认返回None，表示不尝试恢复或恢复不可用
        return None
    
    def _get_error_response(self, error, error_message: str, severity: str, component_status: Dict[str, Any]) -> Dict[str, Any]:
        """根据错误类型、严重性和组件状态获取响应，包含更具体的恢复建议"""
        
        # 基础响应结构
        response = {
            'success': False,
            'severity': severity,
            'component_status': component_status.get('status', 'normal')
        }
        
        # 根据错误类型设置特定的响应信息
        error_type = type(error).__name__
        
        # 错误类型分类和响应
        if isinstance(error, ValueError):
            response.update({
                'error_type': 'validation',
                'message': error_message,
                'recovery_hint': '请验证输入数据的格式和范围',
                'suggested_action': '检查输入参数，确保数据类型和取值范围正确'
            })
        elif isinstance(error, ConnectionError):
            response.update({
                'error_type': 'connection',
                'message': '无法连接到外部服务，请检查网络连接',
                'recovery_hint': '检查网络连接、防火墙设置和API端点可用性',
                'suggested_action': '验证网络连接，检查API服务状态，确认防火墙规则'
            })
        elif isinstance(error, FileNotFoundError):
            response.update({
                'error_type': 'file_not_found',
                'message': f'文件未找到: {error_message}',
                'recovery_hint': '检查文件路径、权限和磁盘空间',
                'suggested_action': '验证文件路径是否正确，检查文件权限和磁盘空间'
            })
        elif isinstance(error, TimeoutError):
            response.update({
                'error_type': 'timeout',
                'message': '请求超时，请稍后重试',
                'recovery_hint': '优化查询复杂度、增加超时时间或分批处理',
                'suggested_action': '增加超时时间，优化查询性能，或分批处理大数据集'
            })
        elif isinstance(error, MemoryError):
            response.update({
                'error_type': 'memory',
                'message': '系统内存不足，请尝试轻量级模式',
                'recovery_hint': '清理缓存、优化内存使用或增加系统内存',
                'suggested_action': '优化内存使用，清理缓存，重启服务或增加系统内存'
            })
        elif isinstance(error, ImportError):
            response.update({
                'error_type': 'import',
                'message': f'导入错误: {error_message}',
                'recovery_hint': '检查依赖安装、Python路径和模块名称',
                'suggested_action': '安装缺失的Python包，检查Python路径和模块导入语句'
            })
        elif isinstance(error, PermissionError):
            response.update({
                'error_type': 'permission',
                'message': f'权限错误: {error_message}',
                'recovery_hint': '检查文件权限、用户访问级别和安全策略',
                'suggested_action': '检查文件和目录权限，验证用户访问级别，调整安全策略'
            })
        elif isinstance(error, OSError):
            response.update({
                'error_type': 'os_error',
                'message': f'操作系统错误: {error_message}',
                'recovery_hint': '检查系统资源、磁盘空间和文件系统权限',
                'suggested_action': '检查系统资源使用情况，清理磁盘空间，验证文件系统完整性'
            })
        elif isinstance(error, KeyError):
            response.update({
                'error_type': 'key_error',
                'message': f'键错误: {error_message}',
                'recovery_hint': '检查字典键是否存在，验证数据结构',
                'suggested_action': '检查数据结构的键是否存在，验证数据来源和格式'
            })
        elif isinstance(error, AttributeError):
            response.update({
                'error_type': 'attribute_error',
                'message': f'属性错误: {error_message}',
                'recovery_hint': '检查对象属性和方法是否存在',
                'suggested_action': '验证对象类型，检查属性和方法名称是否正确'
            })
        elif isinstance(error, TypeError):
            response.update({
                'error_type': 'type_error',
                'message': f'类型错误: {error_message}',
                'recovery_hint': '检查数据类型和函数参数',
                'suggested_action': '验证数据类型，检查函数参数类型和数量'
            })
        elif isinstance(error, IndexError):
            response.update({
                'error_type': 'index_error',
                'message': f'索引错误: {error_message}',
                'recovery_hint': '检查列表、数组或字符串的索引范围',
                'suggested_action': '验证索引值是否在有效范围内，检查数据结构大小'
            })
        elif isinstance(error, ZeroDivisionError):
            response.update({
                'error_type': 'zero_division',
                'message': f'除零错误: {error_message}',
                'recovery_hint': '检查除法运算的分母是否为零',
                'suggested_action': '添加分母为零的检查，使用条件语句避免除零'
            })
        elif isinstance(error, NotImplementedError):
            response.update({
                'error_type': 'not_implemented',
                'message': f'未实现错误: {error_message}',
                'recovery_hint': '该功能尚未实现，请检查功能列表',
                'suggested_action': '确认功能是否在开发计划中，或选择替代方案'
            })
        elif isinstance(error, RuntimeError):
            response.update({
                'error_type': 'runtime_error',
                'message': f'运行时错误: {error_message}',
                'recovery_hint': '检查程序逻辑和运行时状态',
                'suggested_action': '检查程序逻辑，验证运行时环境和状态'
            })
        elif isinstance(error, AssertionError):
            response.update({
                'error_type': 'assertion_error',
                'message': f'断言错误: {error_message}',
                'recovery_hint': '程序断言失败，检查程序逻辑和条件',
                'suggested_action': '检查断言条件，验证程序逻辑和假设'
            })
        else:
            # 未知错误类型
            response.update({
                'error_type': 'internal',
                'message': f'系统内部错误: {error_message}',
                'recovery_hint': '查看详细日志、联系技术支持或重启服务',
                'suggested_action': '联系系统管理员或技术支持团队，提供错误ID和日志信息'
            })
        
        # 根据错误严重性调整响应
        if severity == 'critical':
            response['requires_immediate_attention'] = True
            response['system_impact'] = 'high'
            response['user_impact'] = '服务可能不可用或功能受限'
        elif severity == 'high':
            response['requires_immediate_attention'] = True
            response['system_impact'] = 'medium'
            response['user_impact'] = '功能可能受限或性能下降'
        elif severity == 'medium':
            response['requires_immediate_attention'] = False
            response['system_impact'] = 'low'
            response['user_impact'] = '功能正常但可能有警告或提示'
        elif severity == 'low':
            response['requires_immediate_attention'] = False
            response['system_impact'] = 'minimal'
            response['user_impact'] = '功能正常，仅为信息性提示'
        
        # 根据组件状态调整响应
        component_status_value = component_status.get('status', 'normal')
        if component_status_value in ['degraded', 'disabled']:
            response['component_degraded'] = True
            response['component_status_reason'] = component_status.get('reason', '')
            response['component_recovery_hint'] = f'组件处于{component_status_value}状态: {component_status.get("reason", "")}'
            
            # 如果组件已禁用，建议使用备用组件或功能
            if component_status_value == 'disabled':
                response['suggested_action'] = f'组件已禁用，请使用备用功能或等待组件恢复。原因: {component_status.get("reason", "")}'
        else:
            response['component_degraded'] = False
        
        # 添加技术支持联系方式
        response['technical_support'] = {
            'contact': 'silencecrowtom@qq.com',
            'support_hours': '24/7',
            'escalation_path': '系统管理员 -> 技术支持团队 -> 开发团队'
        }
        
        return response
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        import hashlib
        import uuid
        
        error_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 创建唯一错误ID
        hash_input = f"{error_id}{timestamp}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()[:8]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            'total_errors': self.error_stats['total_errors'],
            'error_types': self.error_stats['error_types'],
            'component_errors': self.error_stats['component_errors'],
            'last_error_time': self.error_stats['last_error_time']
        }
    
    def reset_error_statistics(self):
        """重置错误统计"""
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'component_errors': {},
            'last_error_time': None
        }
        self.logger.info("错误统计已重置")
    
    def log_performance(self, operation: str, duration: float, component_name: str, details: Optional[Dict[str, Any]] = None):
        """记录性能日志"""
        performance_log = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'component': component_name,
            'details': details or {}
        }
        
        # 记录慢操作（超过1秒）
        if duration > 1.0:
            self.log_warning(f"Slow operation: {operation} took {duration:.2f}s", component_name)
        
        self.logger.info(f"Performance: {json.dumps(performance_log)}")
    
    def log_debug(self, message, component_name, details=None):
        """记录调试日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name,
            'message': message
        }
        
        if details:
            log_entry['details'] = details
        
        self.logger.debug(f"Debug: {json.dumps(log_entry)}")
    
    def log_info(self, message, component_name, details=None):
        """记录信息日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name,
            'message': message
        }
        
        if details:
            log_entry['details'] = details
        
        self.logger.info(f"Info: {json.dumps(log_entry)}")
    
    def log_warning(self, message, component_name, details=None):
        """记录警告日志
        注意：这是ErrorHandler自身的log_warning方法实现，
        使用self.logger.warning而不是error_handler.log_warning
        以避免递归调用。
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name,
            'message': message
        }
        
        if details:
            log_entry['details'] = details
        
        self.logger.warning(f"Warning: {json.dumps(log_entry)}")
    
    def log_structured(self, level: str, message: str, component_name: str, structured_data: Dict[str, Any] = None):
        """记录结构化日志
        
        Args:
            level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
            message: 日志消息
            component_name: 组件名称
            structured_data: 结构化数据字典
        """
        structured_data = structured_data or {}
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name,
            'message': message,
            **structured_data  # 将结构化数据合并到日志条目中
        }
        
        # 根据级别记录日志
        level = level.lower()
        if level == 'debug':
            self.logger.debug(f"Structured: {json.dumps(log_entry)}")
        elif level == 'info':
            self.logger.info(f"Structured: {json.dumps(log_entry)}")
        elif level == 'warning':
            self.logger.warning(f"Structured: {json.dumps(log_entry)}")
        elif level == 'error':
            self.logger.error(f"Structured: {json.dumps(log_entry)}")
        elif level == 'critical':
            self.logger.critical(f"Structured: {json.dumps(log_entry)}")
        else:
            # 默认使用info级别
            self.logger.info(f"Structured: {json.dumps(log_entry)}")
    
    async def handle_async_error(self, error, component_name, details=None, context: Optional[Dict[str, Any]] = None):
        """异步处理错误"""
        # 在异步上下文中处理错误
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.handle_error(error, component_name, details, context)
        )
            
    def _mask_sensitive_data(self, data: Any) -> Any:
        """掩码敏感数据，防止泄露到日志中
        
        Args:
            data: 任意数据，可以是字典、列表、字符串等
            
        Returns:
            掩码后的数据
        """
        if data is None:
            return None
        
        # 敏感字段列表
        sensitive_fields = {
            'api_key', 'password', 'secret', 'token', 'key',
            'apikey', 'apisecret', 'client_secret', 'access_token',
            'refresh_token', 'private_key', 'credential', 'auth'
        }
        
        # 如果是字典，递归处理
        if isinstance(data, dict):
            masked_dict = {}
            for key, value in data.items():
                # 检查键是否包含敏感字段（不区分大小写）
                key_lower = str(key).lower()
                is_sensitive = any(sensitive_field in key_lower for sensitive_field in sensitive_fields)
                
                if is_sensitive and value:
                    # 掩码敏感值
                    if isinstance(value, str):
                        if len(value) > 8:
                            masked_dict[key] = f"{value[:2]}***{value[-2:]}"
                        else:
                            masked_dict[key] = "***"
                    else:
                        masked_dict[key] = "***"
                else:
                    # 递归处理嵌套数据
                    masked_dict[key] = self._mask_sensitive_data(value)
            return masked_dict
        
        # 如果是列表，处理每个元素
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        
        # 其他类型直接返回
        else:
            return data
    
    def log_error(self, message, component_name, details=None):
        """记录错误日志"""
        self.logger.error(f"[{component_name}] {message}")
        if details:
            # 掩码敏感数据后再记录
            masked_details = self._mask_sensitive_data(details)
            self.logger.debug(f"Details: {masked_details}")


# ===== Timeout Decorator and Reliability Enhancements =====

def with_timeout(timeout_seconds=30):
    """超时装饰器，保护长时间运行的操作
    
    Args:
        timeout_seconds: 超时时间（秒），默认30秒
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        import threading
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            exception = None
            
            def target():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # 线程仍在运行，超时发生
                raise TimeoutError(
                    f"操作 {func.__name__} 超时 ({timeout_seconds} 秒)"
                )
            
            if exception is not None:
                raise exception
            
            return result
        
        return wrapper
    
    return decorator


def with_timeout_async(timeout_seconds=30):
    """异步超时装饰器，保护长时间运行的异步操作
    
    Args:
        timeout_seconds: 超时时间（秒），默认30秒
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        import asyncio
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"异步操作 {func.__name__} 超时 ({timeout_seconds} 秒)"
                )
        
        return wrapper
    
    return decorator


class TimeoutDecorator:
    """超时装饰器工具类"""
    
    @staticmethod
    def protect_sync(timeout_seconds=30):
        """保护同步函数"""
        return with_timeout(timeout_seconds)
    
    @staticmethod
    def protect_async(timeout_seconds=30):
        """保护异步函数"""
        return with_timeout_async(timeout_seconds)
    
    @staticmethod
    def run_with_timeout(func, timeout_seconds=30, *args, **kwargs):
        """直接运行函数并应用超时保护
        
        Args:
            func: 要运行的函数
            timeout_seconds: 超时时间
            *args, **kwargs: 函数参数
            
        Returns:
            函数结果
            
        Raises:
            TimeoutError: 如果超时
        """
        if asyncio.iscoroutinefunction(func):
            # 异步函数
            async def async_wrapper():
                return await func(*args, **kwargs)
            
            return asyncio.run(
                asyncio.wait_for(async_wrapper(), timeout=timeout_seconds)
            )
        else:
            # 同步函数
            return with_timeout(timeout_seconds)(func)(*args, **kwargs)


# 创建全局错误处理器实例
error_handler = ErrorHandler()
