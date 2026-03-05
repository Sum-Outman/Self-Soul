"""
数据库访问层
Database Access Layer

提供统一的数据库访问接口，用于替换内存存储模式
Provides unified database access interface for replacing in-memory storage patterns
"""

import os
import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict

# 导入安全模块
try:
    from core.security import security_manager
    SECURITY_AVAILABLE = True
except ImportError:
    logger = logging.getLogger("DatabaseAccessLayer")
    logger.warning("Security module not available, database encryption disabled")
    security_manager = None
    SECURITY_AVAILABLE = False

# 导入备份模块
try:
    from core.database.backup_manager import initialize_backup_system
    BACKUP_AVAILABLE = True
except ImportError:
    logger = logging.getLogger("DatabaseAccessLayer")
    logger.warning("Backup module not available, database backup disabled")
    BACKUP_AVAILABLE = False

# 先定义logger
logger = logging.getLogger("DatabaseAccessLayer")

# 安全字段名验证函数
def validate_field_name(field_name: str) -> bool:
    """
    验证字段名是否安全，只允许字母、数字和下划线
    
    Args:
        field_name: 要验证的字段名
        
    Returns:
        是否安全
    """
    if not field_name:
        return False
    
    # 只允许字母、数字和下划线
    return all(c.isalnum() or c == '_' for c in field_name)

# 尝试导入生产数据库模块，但允许失败
PRODUCTION_DB_AVAILABLE = False
try:
    from core.production_database import (
        get_production_database,
        DatabaseConnectionPool,
        StorageManager
    )
    PRODUCTION_DB_AVAILABLE = True
    logger.info("Production database module is available")
except ImportError as e:
    logger.warning(f"Production database module not available: {e}. Using SQLite only.")
except Exception as e:
    logger.warning(f"Error importing production database module: {e}. Using SQLite only.")

from core.error_handling import error_handler

# 数据库加密辅助函数
def encrypt_field(data: str) -> str:
    """加密字段数据"""
    if not SECURITY_AVAILABLE or security_manager is None:
        return data
    try:
        # 使用安全管理器的加密方法
        # security_manager.encrypt_data 返回字典，我们需要提取加密后的字符串
        encrypted_result = security_manager.encrypt_data(data)
        if isinstance(encrypted_result, dict) and "encrypted_data" in encrypted_result:
            return encrypted_result["encrypted_data"]
        else:
            logger.warning("Failed to encrypt field, returning plaintext")
            return data
    except Exception as e:
        logger.warning(f"Field encryption failed: {e}, returning plaintext")
        return data

def decrypt_field(encrypted_data: str) -> str:
    """解密字段数据"""
    if not SECURITY_AVAILABLE or security_manager is None:
        return encrypted_data
    try:
        # security_manager.decrypt_data 需要 encrypted_data 和 hmac_value
        # 简化处理：如果数据未被加密，直接返回
        if not encrypted_data or not encrypted_data.startswith("encrypted:"):
            return encrypted_data
        # 解析加密数据格式
        # 实际实现应更健壮
        decrypted_result = security_manager.decrypt_data(encrypted_data, "")
        if isinstance(decrypted_result, dict) and "decrypted_data" in decrypted_result:
            return decrypted_result["decrypted_data"]
        else:
            logger.warning("Failed to decrypt field, returning encrypted data")
            return encrypted_data
    except Exception as e:
        logger.warning(f"Field decryption failed: {e}, returning encrypted data")
        return encrypted_data

@dataclass
class ModelConfig:
    """模型配置数据类"""
    model_id: str
    model_name: str
    model_type: str
    source: str = "local"
    api_config: Optional[Dict[str, Any]] = None
    is_active: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        if result.get('api_config') is not None:
            json_str = json.dumps(result['api_config'])
            # 加密JSON字符串
            encrypted_json = encrypt_field(json_str)
            result['api_config'] = encrypted_json
        # 处理时间字段
        if result.get('created_at') is None:
            result['created_at'] = datetime.now().isoformat()
        if result.get('updated_at') is None:
            result['updated_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        if 'api_config' in filtered_data and isinstance(filtered_data['api_config'], str):
            try:
                # 尝试解密字段
                decrypted_str = decrypt_field(filtered_data['api_config'])
                filtered_data['api_config'] = json.loads(decrypted_str)
            except json.JSONDecodeError:
                # 如果解密后不是有效的JSON，尝试直接解析原始字符串
                try:
                    filtered_data['api_config'] = json.loads(filtered_data['api_config'])
                except json.JSONDecodeError:
                    filtered_data['api_config'] = {}
        
        # 处理布尔字段（SQLite可能将布尔值存储为整数）
        for bool_field in ['is_active']:
            if bool_field in filtered_data and isinstance(filtered_data[bool_field], int):
                filtered_data[bool_field] = bool(filtered_data[bool_field])
        
        # 处理时间字段的空值
        for time_field in ['created_at', 'updated_at']:
            if time_field in filtered_data and filtered_data[time_field] is None:
                filtered_data[time_field] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class TrainingRecord:
    """训练记录数据类"""
    model_id: str
    dataset_name: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str = "running"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        for field in ['training_config', 'metrics']:
            if result.get(field) is not None:
                result[field] = json.dumps(result[field])
        # 处理时间字段
        if result.get('started_at') is None:
            result['started_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingRecord':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        for field in ['training_config', 'metrics']:
            if field in filtered_data and isinstance(filtered_data[field], str):
                try:
                    filtered_data[field] = json.loads(filtered_data[field])
                except json.JSONDecodeError:
                    filtered_data[field] = {}
        
        # 处理时间字段的空值
        for time_field in ['started_at', 'completed_at']:
            if time_field in filtered_data and filtered_data[time_field] is None:
                filtered_data[time_field] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class SystemLog:
    """系统日志数据类"""
    level: str
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        if result.get('details') is not None:
            result['details'] = json.dumps(result['details'])
        # 处理时间字段
        if result.get('created_at') is None:
            result['created_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemLog':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        if 'details' in filtered_data and isinstance(filtered_data['details'], str):
            try:
                filtered_data['details'] = json.loads(filtered_data['details'])
            except json.JSONDecodeError:
                filtered_data['details'] = {}
        
        # 处理时间字段的空值
        if 'created_at' in filtered_data and filtered_data['created_at'] is None:
            filtered_data['created_at'] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class PerformanceMetrics:
    """模型性能指标数据类"""
    model_id: str
    success_rate: float = 0.0
    latency: float = 0.0
    accuracy: float = 0.0
    collaboration_score: float = 0.0
    calls: int = 0
    last_collaboration_time: Optional[float] = None
    optimization_suggestions: Optional[List[str]] = None
    custom_metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        for json_field in ['optimization_suggestions', 'custom_metrics']:
            if result.get(json_field) is not None:
                result[json_field] = json.dumps(result[json_field])
        # 处理时间字段
        if result.get('timestamp') is None:
            result['timestamp'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        for json_field in ['optimization_suggestions', 'custom_metrics']:
            if json_field in filtered_data and isinstance(filtered_data[json_field], str):
                try:
                    filtered_data[json_field] = json.loads(filtered_data[json_field])
                except json.JSONDecodeError:
                    filtered_data[json_field] = None if json_field == 'optimization_suggestions' else {}
        
        # 处理时间字段的空值
        if 'timestamp' in filtered_data and filtered_data['timestamp'] is None:
            filtered_data['timestamp'] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class TrainingStatus:
    """训练状态数据类"""
    model_id: str
    status: str  # 'preparing', 'training', 'completed', 'error', 'cancelled'
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理时间字段
        if result.get('started_at') is None:
            result['started_at'] = datetime.now().isoformat()
        if result.get('updated_at') is None:
            result['updated_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStatus':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理时间字段的空值
        for time_field in ['started_at', 'updated_at']:
            if time_field in filtered_data and filtered_data[time_field] is None:
                filtered_data[time_field] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class TrainingProgress:
    """训练进度数据类"""
    model_id: str
    epoch: int
    step: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        if result.get('metrics') is not None:
            result['metrics'] = json.dumps(result['metrics'])
        # 处理时间字段
        if result.get('timestamp') is None:
            result['timestamp'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProgress':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        if 'metrics' in filtered_data and isinstance(filtered_data['metrics'], str):
            try:
                filtered_data['metrics'] = json.loads(filtered_data['metrics'])
            except json.JSONDecodeError:
                filtered_data['metrics'] = {}
        
        # 处理时间字段的空值
        if 'timestamp' in filtered_data and filtered_data['timestamp'] is None:
            filtered_data['timestamp'] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class KnowledgeBaseIntegration:
    """知识库集成状态数据类"""
    model_id: str
    integrated: bool = False
    knowledge_loaded: int = 0
    last_integration_time: Optional[str] = None
    integration_method: Optional[str] = None
    integration_status: Optional[str] = None
    knowledge_sources: Optional[List[str]] = None
    custom_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        for json_field in ['knowledge_sources', 'custom_data']:
            if result.get(json_field) is not None:
                result[json_field] = json.dumps(result[json_field])
        # 处理时间字段
        if result.get('last_integration_time') is None:
            result['last_integration_time'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBaseIntegration':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        for json_field in ['knowledge_sources', 'custom_data']:
            if json_field in filtered_data and isinstance(filtered_data[json_field], str):
                try:
                    filtered_data[json_field] = json.loads(filtered_data[json_field])
                except json.JSONDecodeError:
                    filtered_data[json_field] = None if json_field == 'knowledge_sources' else {}
        
        # 处理时间字段的空值
        if 'last_integration_time' in filtered_data and filtered_data['last_integration_time'] is None:
            filtered_data['last_integration_time'] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class APIDependencyRecord:
    """API依赖记录数据类（用于数据库存储）"""
    name: str  # 依赖名称，唯一标识
    provider: str  # API提供商（openai、anthropic等）
    module_name: str  # Python模块名称
    client_class: str  # 客户端类名
    config: Optional[Dict[str, Any]] = None  # API配置（JSON格式）
    priority: int = 1  # 优先级，数字越小优先级越高
    optional: bool = False  # 是否为可选依赖
    fallback_providers: Optional[List[str]] = None  # 降级提供商列表
    is_available: bool = False  # 当前是否可用
    error_count: int = 0  # 错误计数
    consecutive_errors: int = 0  # 连续错误计数
    last_error: Optional[str] = None  # 最后错误信息
    last_health_check: Optional[str] = None  # 最后健康检查时间
    health_status: Optional[Dict[str, Any]] = None  # 健康状态摘要
    created_at: Optional[str] = None  # 创建时间
    updated_at: Optional[str] = None  # 更新时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        for json_field in ['config', 'fallback_providers', 'health_status']:
            if result.get(json_field) is not None:
                result[json_field] = json.dumps(result[json_field])
        # 处理时间字段
        if result.get('created_at') is None:
            result['created_at'] = datetime.now().isoformat()
        if result.get('updated_at') is None:
            result['updated_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIDependencyRecord':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        for json_field in ['config', 'fallback_providers', 'health_status']:
            if json_field in filtered_data and isinstance(filtered_data[json_field], str):
                try:
                    filtered_data[json_field] = json.loads(filtered_data[json_field])
                except json.JSONDecodeError:
                    filtered_data[json_field] = None
        
        # 处理布尔字段（SQLite可能将布尔值存储为整数）
        for bool_field in ['optional', 'is_available']:
            if bool_field in filtered_data and isinstance(filtered_data[bool_field], int):
                filtered_data[bool_field] = bool(filtered_data[bool_field])
        
        # 处理时间字段的空值
        for time_field in ['created_at', 'updated_at', 'last_health_check']:
            if time_field in filtered_data and filtered_data[time_field] is None:
                filtered_data[time_field] = datetime.now().isoformat()
                
        return cls(**filtered_data)

@dataclass
class DependencyHealthRecord:
    """依赖健康状态记录数据类（用于数据库存储）"""
    dependency_name: str  # 依赖名称
    is_healthy: bool = False  # 是否健康
    response_time: Optional[float] = None  # 响应时间（秒）
    error_message: Optional[str] = None  # 错误信息
    last_check: Optional[str] = None  # 最后检查时间
    check_count: int = 0  # 总检查次数
    success_count: int = 0  # 成功次数
    failure_count: int = 0  # 失败次数
    health_score: float = 0.0  # 健康分数（0-100）
    consecutive_errors: int = 0  # 连续错误次数
    custom_metrics: Optional[Dict[str, Any]] = None  # 自定义指标
    created_at: Optional[str] = None  # 创建时间
    updated_at: Optional[str] = None  # 更新时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 处理JSONB字段
        if result.get('custom_metrics') is not None:
            result['custom_metrics'] = json.dumps(result['custom_metrics'])
        # 处理时间字段
        if result.get('created_at') is None:
            result['created_at'] = datetime.now().isoformat()
        if result.get('updated_at') is None:
            result['updated_at'] = datetime.now().isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyHealthRecord':
        """从字典创建实例"""
        # 过滤掉类不接受的字段（如'id'）
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 处理JSONB字段
        if 'custom_metrics' in filtered_data and isinstance(filtered_data['custom_metrics'], str):
            try:
                filtered_data['custom_metrics'] = json.loads(filtered_data['custom_metrics'])
            except json.JSONDecodeError:
                filtered_data['custom_metrics'] = {}
        
        # 处理布尔字段（SQLite可能将布尔值存储为整数）
        for bool_field in ['is_healthy']:
            if bool_field in filtered_data and isinstance(filtered_data[bool_field], int):
                filtered_data[bool_field] = bool(filtered_data[bool_field])
        
        # 处理时间字段的空值
        for time_field in ['created_at', 'updated_at', 'last_check']:
            if time_field in filtered_data and filtered_data[time_field] is None:
                filtered_data[time_field] = datetime.now().isoformat()
                
        return cls(**filtered_data)

class DatabaseAccessLayer:
    """数据库访问层主类"""
    
    def __init__(self, db_type: str = None, db_path: str = None):
        """初始化数据库访问层
        
        Args:
            db_type: 数据库类型，可选 'sqlite' 或 'postgresql'
            db_path: SQLite数据库路径（仅当db_type='sqlite'时有效）
        """
        self.db_type = db_type or os.getenv("DB_TYPE", "sqlite")
        self.db_path = db_path or os.getenv("DB_PATH", "self_soul.db")
        self.db_pool = None
        self._initialize_database()
        
        # 初始化备份系统（仅适用于SQLite）
        if self.db_type == "sqlite" and BACKUP_AVAILABLE:
            try:
                # 启动自动备份（生产环境启用，开发环境可选）
                env = os.getenv("ENVIRONMENT", "development")
                start_auto_backup = (env == "production")
                initialize_backup_system(self.db_path, start_auto_backup=start_auto_backup)
                logger.info("Database backup system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize backup system: {e}")
    
    def _initialize_database(self):
        """初始化数据库连接"""
        try:
            if self.db_type == "sqlite":
                # 使用SQLite
                self._initialize_sqlite()
            elif self.db_type == "postgresql":
                # 使用PostgreSQL连接池
                if PRODUCTION_DB_AVAILABLE:
                    self.db_pool = get_production_database()
                    logger.info("Using PostgreSQL database connection pool")
                else:
                    logger.warning("PostgreSQL not available, falling back to SQLite")
                    self.db_type = "sqlite"
                    self._initialize_sqlite()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            logger.info(f"Database access layer initialized with {self.db_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database access layer: {e}")
            # 回退到SQLite
            self.db_type = "sqlite"
            self._initialize_sqlite()
    
    def _initialize_sqlite(self):
        """初始化SQLite数据库"""
        try:
            # 确保目录存在（如果db_path包含目录）
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # 只有目录不为空时才创建
                os.makedirs(db_dir, exist_ok=True)
            
            # 创建数据库连接
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # 启用外键约束
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # 创建表结构
            self._create_tables()
            
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    def _create_tables(self):
        """创建数据库表（如果不存在）"""
        try:
            cursor = self.connection.cursor()
            
            # 创建模型配置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_configs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    source TEXT DEFAULT 'local',
                    api_config TEXT,  -- JSON格式
                    is_active BOOLEAN DEFAULT false,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # 创建训练记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_records (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    dataset_name TEXT,
                    training_config TEXT,  -- JSON格式
                    metrics TEXT,  -- JSON格式
                    status TEXT DEFAULT 'running',
                    started_at TEXT,
                    completed_at TEXT,
                    created_by TEXT
                )
            """)
            
            # 创建系统日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,  -- JSON格式
                    user_id TEXT,
                    ip_address TEXT,
                    created_at TEXT
                )
            """)
            
            # 创建性能指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    latency REAL DEFAULT 0.0,
                    accuracy REAL DEFAULT 0.0,
                    collaboration_score REAL DEFAULT 0.0,
                    calls INTEGER DEFAULT 0,
                    last_collaboration_time REAL,
                    optimization_suggestions TEXT,  -- JSON格式
                    custom_metrics TEXT,  -- JSON格式
                    timestamp TEXT
                )
            """)
            
            # 创建训练状态表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_status (
                    id TEXT PRIMARY KEY,
                    model_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    current_epoch INTEGER,
                    total_epochs INTEGER,
                    current_step INTEGER,
                    total_steps INTEGER,
                    error_message TEXT,
                    started_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # 创建训练进度表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_progress (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL,
                    accuracy REAL,
                    learning_rate REAL,
                    metrics TEXT,  -- JSON格式
                    checkpoint_path TEXT,
                    timestamp TEXT
                )
            """)
            
            # 创建知识库集成表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base_integration (
                    id TEXT PRIMARY KEY,
                    model_id TEXT UNIQUE NOT NULL,
                    integrated BOOLEAN DEFAULT false,
                    knowledge_loaded INTEGER DEFAULT 0,
                    last_integration_time TEXT,
                    integration_method TEXT,
                    integration_status TEXT,
                    knowledge_sources TEXT,  -- JSON格式
                    custom_data TEXT  -- JSON格式
                )
            """)
            
            # 创建API依赖表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_dependencies (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    provider TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    client_class TEXT NOT NULL,
                    config TEXT,  -- JSON格式
                    priority INTEGER DEFAULT 1,
                    optional BOOLEAN DEFAULT false,
                    fallback_providers TEXT,  -- JSON格式
                    is_available BOOLEAN DEFAULT false,
                    error_count INTEGER DEFAULT 0,
                    consecutive_errors INTEGER DEFAULT 0,
                    last_error TEXT,
                    last_health_check TEXT,
                    health_status TEXT,  -- JSON格式
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # 创建依赖健康状态表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dependency_health_status (
                    id TEXT PRIMARY KEY,
                    dependency_name TEXT NOT NULL,
                    is_healthy BOOLEAN DEFAULT false,
                    response_time REAL,
                    error_message TEXT,
                    last_check TEXT,
                    check_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    health_score REAL DEFAULT 0.0,
                    consecutive_errors INTEGER DEFAULT 0,
                    custom_metrics TEXT,  -- JSON格式
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_configs_model_id ON model_configs(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_records_model_id ON training_records(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_records_status ON training_records(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_model_id ON performance_metrics(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_status_model_id ON training_status(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_progress_model_id ON training_progress(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_base_integration_model_id ON knowledge_base_integration(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_dependencies_name ON api_dependencies(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_dependencies_provider ON api_dependencies(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dependency_health_status_name ON dependency_health_status(dependency_name)")
            
            self.connection.commit()
            cursor.close()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    # ===== 模型配置操作方法 =====
    
    def save_model_config(self, config: ModelConfig) -> bool:
        """保存模型配置"""
        try:
            config_dict = config.to_dict()
            config_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM model_configs WHERE model_id = ?", (config.model_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有配置
                    cursor.execute("""
                        UPDATE model_configs 
                        SET model_name = ?, model_type = ?, source = ?, api_config = ?, 
                            is_active = ?, updated_at = ?
                        WHERE model_id = ?
                    """, (
                        config_dict['model_name'],
                        config_dict['model_type'],
                        config_dict['source'],
                        config_dict.get('api_config'),
                        config_dict['is_active'],
                        config_dict['updated_at'],
                        config_dict['model_id']
                    ))
                else:
                    # 插入新配置
                    cursor.execute("""
                        INSERT INTO model_configs 
                        (id, model_id, model_name, model_type, source, api_config, is_active, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        config_dict['id'],
                        config_dict['model_id'],
                        config_dict['model_name'],
                        config_dict['model_type'],
                        config_dict['source'],
                        config_dict.get('api_config'),
                        config_dict['is_active'],
                        config_dict['created_at'],
                        config_dict['updated_at']
                    ))
                
                self.connection.commit()
                cursor.close()
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        # 检查是否已存在
                        cursor.execute("""
                            SELECT id FROM model_configs WHERE model_id = %s
                        """, (config.model_id,))
                        existing = cursor.fetchone()
                        
                        if existing:
                            # 更新现有配置
                            cursor.execute("""
                                UPDATE model_configs 
                                SET model_name = %s, model_type = %s, source = %s, api_config = %s, 
                                    is_active = %s, updated_at = %s
                                WHERE model_id = %s
                            """, (
                                config_dict['model_name'],
                                config_dict['model_type'],
                                config_dict['source'],
                                config_dict.get('api_config'),
                                config_dict['is_active'],
                                config_dict['updated_at'],
                                config_dict['model_id']
                            ))
                        else:
                            # 插入新配置
                            cursor.execute("""
                                INSERT INTO model_configs 
                                (id, model_id, model_name, model_type, source, api_config, is_active, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                config_dict['id'],
                                config_dict['model_id'],
                                config_dict['model_name'],
                                config_dict['model_type'],
                                config_dict['source'],
                                config_dict.get('api_config'),
                                config_dict['is_active'],
                                config_dict['created_at'],
                                config_dict['updated_at']
                            ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_model_config failed: {pg_error}")
                        return False
                else:
                    logger.error("PostgreSQL not available for saving model config")
                    return False
            
            logger.debug(f"Model config saved: {config.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model config {config.model_id}: {e}")
            return False
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM model_configs WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    config_dict = dict(row)
                    return ModelConfig.from_dict(config_dict)
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM model_configs WHERE model_id = %s
                        """, (model_id,))
                        
                        row = cursor.fetchone()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if row:
                            config_dict = dict(row)
                            return ModelConfig.from_dict(config_dict)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_model_config failed: {pg_error}")
                        return None
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model config {model_id}: {e}")
            return None
    
    def get_all_model_configs(self) -> List[ModelConfig]:
        """获取所有模型配置"""
        try:
            configs = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("SELECT * FROM model_configs ORDER BY model_id")
                
                for row in cursor.fetchall():
                    config_dict = dict(row)
                    configs.append(ModelConfig.from_dict(config_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("SELECT * FROM model_configs ORDER BY model_id")
                        
                        for row in cursor.fetchall():
                            config_dict = dict(row)
                            configs.append(ModelConfig.from_dict(config_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_all_model_configs failed: {pg_error}")
                        return []
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get all model configs: {e}")
            return []
    
    def delete_model_config(self, model_id: str) -> bool:
        """删除模型配置"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM model_configs WHERE model_id = ?", (model_id,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Model config deleted: {model_id}")
                    return True
                else:
                    logger.warning(f"Model config not found for deletion: {model_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM model_configs WHERE model_id = %s", (model_id,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"Model config deleted: {model_id}")
                            return True
                        else:
                            logger.warning(f"Model config not found for deletion: {model_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_model_config failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete model config {model_id}: {e}")
            return False
    
    # ===== 训练记录操作方法 =====
    
    def save_training_record(self, record: TrainingRecord) -> str:
        """保存训练记录，返回记录ID"""
        try:
            record_dict = record.to_dict()
            record_id = str(uuid.uuid4())
            record_dict['id'] = record_id
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    INSERT INTO training_records 
                    (id, model_id, dataset_name, training_config, metrics, status, started_at, completed_at, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_dict['id'],
                    record_dict['model_id'],
                    record_dict.get('dataset_name'),
                    record_dict.get('training_config'),
                    record_dict.get('metrics'),
                    record_dict['status'],
                    record_dict['started_at'],
                    record_dict.get('completed_at'),
                    record_dict.get('created_by')
                ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        # 获取PostgreSQL连接
                        connection = self.db_pool.get_connection()
                        try:
                            cursor = connection.cursor()
                            
                            cursor.execute("""
                                INSERT INTO training_records 
                                (id, model_id, dataset_name, training_config, metrics, status, started_at, completed_at, created_by)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                record_dict['id'],
                                record_dict['model_id'],
                                record_dict.get('dataset_name'),
                                json.dumps(record_dict.get('training_config')) if record_dict.get('training_config') else None,
                                json.dumps(record_dict.get('metrics')) if record_dict.get('metrics') else None,
                                record_dict['status'],
                                record_dict['started_at'],
                                record_dict.get('completed_at'),
                                record_dict.get('created_by')
                            ))
                            
                            connection.commit()
                            cursor.close()
                            logger.debug(f"PostgreSQL training record saved: {record_id}")
                            
                        finally:
                            self.db_pool.return_connection(connection)
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_training_record failed: {pg_error}")
                        # 回退到SQLite或返回空字符串
                        return ""
                else:
                    logger.warning("PostgreSQL not available for saving training record")
                    return ""
            
            logger.debug(f"Training record saved: {record_id} for model {record.model_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to save training record for model {record.model_id}: {e}")
            return ""
    
    def update_training_record(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """更新训练记录"""
        try:
            if not updates:
                return True
            
            # 准备更新语句 - 使用白名单防止SQL注入
            allowed_fields = {
                'model_id', 'dataset_name', 'training_config', 'metrics', 
                'status', 'started_at', 'completed_at', 'created_by'
            }
            
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                # 只允许白名单中的字段
                if key not in allowed_fields:
                    logger.warning(f"Ignoring disallowed field in update: {key}")
                    continue
                
                # 验证字段名安全性
                if not validate_field_name(key):
                    logger.warning(f"Invalid field name in update (security check): {key}")
                    continue
                    
                if key in ['training_config', 'metrics'] and value is not None:
                    value = json.dumps(value)
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
            # 如果没有有效的更新字段，直接返回成功
            if not set_clauses:
                logger.warning("No valid fields to update")
                return True
            
            params.append(record_id)
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                sql = f"UPDATE training_records SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(sql, params)
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Training record updated: {record_id}")
                    return True
                else:
                    logger.warning(f"Training record not found for update: {record_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()

                        set_clauses_pg = []
                        # 使用相同的白名单过滤字段
                        for key, value in updates.items():
                            if key not in allowed_fields:
                                continue  # 已在前面的循环中记录警告，这里直接跳过
                            
                            # 验证字段名安全性
                            if not validate_field_name(key):
                                logger.warning(f"Invalid field name in update (security check, PostgreSQL): {key}")
                                continue
                            
                            if key in ['training_config', 'metrics'] and value is not None:
                                value = json.dumps(value)
                            set_clauses_pg.append(f"{key} = %s")
                        
                        sql = f"UPDATE training_records SET {', '.join(set_clauses_pg)} WHERE id = %s"
                        params_pg = params  # 使用相同的参数列表
                        
                        cursor.execute(sql, params_pg)
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL training record updated: {record_id}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL training record not found for update: {record_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL update_training_record failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update training record {record_id}: {e}")
            return False
    
    def get_training_records(self, model_id: str = None, status: str = None, limit: int = 100) -> List[TrainingRecord]:
        """获取训练记录"""
        try:
            records = []
            where_clauses = []
            params = []
            
            if model_id:
                where_clauses.append("model_id = ?")
                params.append(model_id)
            
            if status:
                where_clauses.append("status = ?")
                params.append(status)
            
            where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                sql = f"""
                    SELECT * FROM training_records 
                    {where_sql}
                    ORDER BY started_at DESC 
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(sql, params)
                
                for row in cursor.fetchall():
                    record_dict = dict(row)
                    records.append(TrainingRecord.from_dict(record_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()

                        sql = f"""
                            SELECT * FROM training_records 
                            {where_sql}
                            ORDER BY started_at DESC 
                            LIMIT %s
                        """
                        params.append(limit)
                        
                        # 将?替换为%s用于PostgreSQL
                        sql = sql.replace("?", "%s")
                        
                        cursor.execute(sql, params)
                        
                        for row in cursor.fetchall():
                            record_dict = dict(row)
                            # 确保JSON字段被正确解析
                            if 'training_config' in record_dict and isinstance(record_dict['training_config'], str):
                                try:
                                    record_dict['training_config'] = json.loads(record_dict['training_config'])
                                except Exception as json_error:
                                    logger.warning(f"Failed to parse training_config JSON: {json_error}, keeping as string")
                            if 'metrics' in record_dict and isinstance(record_dict['metrics'], str):
                                try:
                                    record_dict['metrics'] = json.loads(record_dict['metrics'])
                                except Exception as json_error:
                                    logger.warning(f"Failed to parse metrics JSON: {json_error}, keeping as string")
                            records.append(TrainingRecord.from_dict(record_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_training_records failed: {pg_error}")
                        return []
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get training records: {e}")
            return []
    
    # ===== 系统日志操作方法 =====
    
    def save_system_log(self, log: SystemLog) -> bool:
        """保存系统日志"""
        try:
            log_dict = log.to_dict()
            log_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    INSERT INTO system_logs 
                    (id, level, component, message, details, user_id, ip_address, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_dict['id'],
                    log_dict['level'],
                    log_dict['component'],
                    log_dict['message'],
                    log_dict.get('details'),
                    log_dict.get('user_id'),
                    log_dict.get('ip_address'),
                    log_dict['created_at']
                ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            INSERT INTO system_logs 
                            (id, level, component, message, details, user_id, ip_address, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            log_dict['id'],
                            log_dict['level'],
                            log_dict['component'],
                            log_dict['message'],
                            log_dict.get('details'),
                            log_dict.get('user_id'),
                            log_dict.get('ip_address'),
                            log_dict['created_at']
                        ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_system_log failed: {pg_error}")
                        return False
            
            logger.debug(f"System log saved: {log.component} - {log.message[:50]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save system log: {e}")
            return False
    
    def get_system_logs(self, component: str = None, level: str = None, 
                        start_time: str = None, end_time: str = None, 
                        limit: int = 1000) -> List[SystemLog]:
        """获取系统日志"""
        try:
            logs = []
            where_clauses = []
            params = []
            
            if component:
                where_clauses.append("component = ?")
                params.append(component)
            
            if level:
                where_clauses.append("level = ?")
                params.append(level)
            
            if start_time:
                where_clauses.append("created_at >= ?")
                params.append(start_time)
            
            if end_time:
                where_clauses.append("created_at <= ?")
                params.append(end_time)
            
            where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                sql = f"""
                    SELECT * FROM system_logs 
                    {where_sql}
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(sql, params)
                
                for row in cursor.fetchall():
                    log_dict = dict(row)
                    logs.append(SystemLog.from_dict(log_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()

                        sql = f"""
                            SELECT * FROM system_logs 
                            {where_sql}
                            ORDER BY created_at DESC 
                            LIMIT %s
                        """
                        params.append(limit)
                        
                        # 将?替换为%s用于PostgreSQL
                        sql = sql.replace("?", "%s")
                        
                        cursor.execute(sql, params)
                        
                        for row in cursor.fetchall():
                            log_dict = dict(row)
                            logs.append(SystemLog.from_dict(log_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_system_logs failed: {pg_error}")
                        return []
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get system logs: {e}")
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """清理旧日志"""
        try:
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    DELETE FROM system_logs WHERE created_at < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                logger.info(f"Cleaned up {deleted_count} old system logs")
                return deleted_count
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            DELETE FROM system_logs WHERE created_at < %s
                        """, (cutoff_str,))
                        
                        deleted_count = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        logger.info(f"PostgreSQL cleaned up {deleted_count} old system logs")
                        return deleted_count
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL cleanup_old_logs failed: {pg_error}")
                        return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
            return 0
    
    # ===== 性能指标操作方法 =====
    
    def save_performance_metrics(self, metrics: PerformanceMetrics) -> str:
        """保存性能指标，返回记录ID"""
        try:
            metrics_dict = metrics.to_dict()
            metrics_id = str(uuid.uuid4())
            metrics_dict['id'] = metrics_id
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (id, model_id, success_rate, latency, accuracy, collaboration_score, 
                     calls, last_collaboration_time, optimization_suggestions, 
                     custom_metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics_dict['id'],
                    metrics_dict['model_id'],
                    metrics_dict.get('success_rate', 0.0),
                    metrics_dict.get('latency', 0.0),
                    metrics_dict.get('accuracy', 0.0),
                    metrics_dict.get('collaboration_score', 0.0),
                    metrics_dict.get('calls', 0),
                    metrics_dict.get('last_collaboration_time'),
                    metrics_dict.get('optimization_suggestions'),
                    metrics_dict.get('custom_metrics'),
                    metrics_dict.get('timestamp')
                ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            INSERT INTO performance_metrics 
                            (id, model_id, success_rate, latency, accuracy, collaboration_score, 
                             calls, last_collaboration_time, optimization_suggestions, 
                             custom_metrics, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            metrics_dict['id'],
                            metrics_dict['model_id'],
                            metrics_dict.get('success_rate', 0.0),
                            metrics_dict.get('latency', 0.0),
                            metrics_dict.get('accuracy', 0.0),
                            metrics_dict.get('collaboration_score', 0.0),
                            metrics_dict.get('calls', 0),
                            metrics_dict.get('last_collaboration_time'),
                            metrics_dict.get('optimization_suggestions'),
                            metrics_dict.get('custom_metrics'),
                            metrics_dict.get('timestamp')
                        ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_performance_metrics failed: {pg_error}")
                        return ""
                else:
                    logger.warning("PostgreSQL not available for saving performance metrics")
                    return ""
            
            logger.debug(f"Performance metrics saved: {metrics_id} for model {metrics.model_id}")
            return metrics_id
            
        except Exception as e:
            logger.error(f"Failed to save performance metrics for model {metrics.model_id}: {e}")
            return ""
    
    def get_performance_metrics(self, model_id: str, limit: int = 100) -> List[PerformanceMetrics]:
        """获取模型性能指标"""
        try:
            metrics_list = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM performance_metrics 
                    WHERE model_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (model_id, limit))
                
                for row in cursor.fetchall():
                    metrics_dict = dict(row)
                    metrics_list.append(PerformanceMetrics.from_dict(metrics_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM performance_metrics 
                            WHERE model_id = %s 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        """, (model_id, limit))
                        
                        for row in cursor.fetchall():
                            metrics_dict = dict(row)
                            metrics_list.append(PerformanceMetrics.from_dict(metrics_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_performance_metrics failed: {pg_error}")
                        return []
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics for model {model_id}: {e}")
            return []
    
    def get_all_performance_metrics(self, limit: int = 1000) -> List[PerformanceMetrics]:
        """获取所有性能指标"""
        try:
            metrics_list = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM performance_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    metrics_dict = dict(row)
                    metrics_list.append(PerformanceMetrics.from_dict(metrics_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM performance_metrics 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        """, (limit,))
                        
                        for row in cursor.fetchall():
                            metrics_dict = dict(row)
                            metrics_list.append(PerformanceMetrics.from_dict(metrics_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_all_performance_metrics failed: {pg_error}")
                        return []
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to get all performance metrics: {e}")
            return []
    
    def delete_performance_metrics(self, metrics_id: str) -> bool:
        """删除性能指标记录"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM performance_metrics WHERE id = ?", (metrics_id,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Performance metrics deleted: {metrics_id}")
                    return True
                else:
                    logger.warning(f"Performance metrics not found for deletion: {metrics_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM performance_metrics WHERE id = %s", (metrics_id,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL performance metrics deleted: {metrics_id}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL performance metrics not found for deletion: {metrics_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_performance_metrics failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete performance metrics {metrics_id}: {e}")
            return False
    
    def cleanup_old_performance_metrics(self, days_to_keep: int = 30) -> int:
        """清理旧的性能指标记录"""
        try:
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    DELETE FROM performance_metrics WHERE timestamp < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                logger.info(f"Cleaned up {deleted_count} old performance metrics")
                return deleted_count
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            DELETE FROM performance_metrics WHERE timestamp < %s
                        """, (cutoff_str,))
                        
                        deleted_count = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        logger.info(f"PostgreSQL cleaned up {deleted_count} old performance metrics")
                        return deleted_count
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL cleanup_old_performance_metrics failed: {pg_error}")
                        return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old performance metrics: {e}")
            return 0
    
    # ===== 训练状态操作方法 =====
    
    def save_training_status(self, status: TrainingStatus) -> bool:
        """保存训练状态"""
        try:
            status_dict = status.to_dict()
            status_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM training_status WHERE model_id = ?", (status.model_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有状态
                    cursor.execute("""
                        UPDATE training_status 
                        SET status = ?, progress = ?, current_epoch = ?, total_epochs = ?,
                            current_step = ?, total_steps = ?, error_message = ?,
                            started_at = ?, updated_at = ?
                        WHERE model_id = ?
                    """, (
                        status_dict['status'],
                        status_dict.get('progress', 0.0),
                        status_dict.get('current_epoch'),
                        status_dict.get('total_epochs'),
                        status_dict.get('current_step'),
                        status_dict.get('total_steps'),
                        status_dict.get('error_message'),
                        status_dict.get('started_at'),
                        status_dict.get('updated_at'),
                        status_dict['model_id']
                    ))
                else:
                    # 插入新状态
                    cursor.execute("""
                        INSERT INTO training_status 
                        (id, model_id, status, progress, current_epoch, total_epochs,
                         current_step, total_steps, error_message, started_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        status_dict['id'],
                        status_dict['model_id'],
                        status_dict['status'],
                        status_dict.get('progress', 0.0),
                        status_dict.get('current_epoch'),
                        status_dict.get('total_epochs'),
                        status_dict.get('current_step'),
                        status_dict.get('total_steps'),
                        status_dict.get('error_message'),
                        status_dict.get('started_at'),
                        status_dict.get('updated_at')
                    ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        # 检查是否已存在
                        cursor.execute("SELECT id FROM training_status WHERE model_id = %s", (status.model_id,))
                        existing = cursor.fetchone()
                        
                        if existing:
                            # 更新现有状态
                            cursor.execute("""
                                UPDATE training_status 
                                SET status = %s, progress = %s, current_epoch = %s, total_epochs = %s,
                                    current_step = %s, total_steps = %s, error_message = %s,
                                    started_at = %s, updated_at = %s
                                WHERE model_id = %s
                            """, (
                                status_dict['status'],
                                status_dict.get('progress', 0.0),
                                status_dict.get('current_epoch'),
                                status_dict.get('total_epochs'),
                                status_dict.get('current_step'),
                                status_dict.get('total_steps'),
                                status_dict.get('error_message'),
                                status_dict.get('started_at'),
                                status_dict.get('updated_at'),
                                status_dict['model_id']
                            ))
                        else:
                            # 插入新状态
                            cursor.execute("""
                                INSERT INTO training_status 
                                (id, model_id, status, progress, current_epoch, total_epochs,
                                 current_step, total_steps, error_message, started_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                status_dict['id'],
                                status_dict['model_id'],
                                status_dict['status'],
                                status_dict.get('progress', 0.0),
                                status_dict.get('current_epoch'),
                                status_dict.get('total_epochs'),
                                status_dict.get('current_step'),
                                status_dict.get('total_steps'),
                                status_dict.get('error_message'),
                                status_dict.get('started_at'),
                                status_dict.get('updated_at')
                            ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_training_status failed: {pg_error}")
                        return False
                else:
                    logger.warning("PostgreSQL not available for saving training status")
                    return False
            
            logger.debug(f"Training status saved: {status.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training status for model {status.model_id}: {e}")
            return False
    
    def get_training_status(self, model_id: str) -> Optional[TrainingStatus]:
        """获取训练状态"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM training_status WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    status_dict = dict(row)
                    return TrainingStatus.from_dict(status_dict)
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM training_status WHERE model_id = %s
                        """, (model_id,))
                        
                        row = cursor.fetchone()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if row:
                            status_dict = dict(row)
                            return TrainingStatus.from_dict(status_dict)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_training_status failed: {pg_error}")
                        return None
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get training status for model {model_id}: {e}")
            return None
    
    def delete_training_status(self, model_id: str) -> bool:
        """删除训练状态"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM training_status WHERE model_id = ?", (model_id,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Training status deleted: {model_id}")
                    return True
                else:
                    logger.warning(f"Training status not found for deletion: {model_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM training_status WHERE model_id = %s", (model_id,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL training status deleted: {model_id}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL training status not found for deletion: {model_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_training_status failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete training status {model_id}: {e}")
            return False
    
    # ===== 训练进度操作方法 =====
    
    def save_training_progress(self, progress: TrainingProgress) -> str:
        """保存训练进度记录，返回记录ID"""
        try:
            progress_dict = progress.to_dict()
            progress_id = str(uuid.uuid4())
            progress_dict['id'] = progress_id
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    INSERT INTO training_progress 
                    (id, model_id, epoch, step, loss, accuracy, learning_rate, 
                     metrics, checkpoint_path, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    progress_dict['id'],
                    progress_dict['model_id'],
                    progress_dict['epoch'],
                    progress_dict['step'],
                    progress_dict.get('loss'),
                    progress_dict.get('accuracy'),
                    progress_dict.get('learning_rate'),
                    progress_dict.get('metrics'),
                    progress_dict.get('checkpoint_path'),
                    progress_dict.get('timestamp')
                ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            INSERT INTO training_progress 
                            (id, model_id, epoch, step, loss, accuracy, learning_rate, 
                             metrics, checkpoint_path, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            progress_dict['id'],
                            progress_dict['model_id'],
                            progress_dict['epoch'],
                            progress_dict['step'],
                            progress_dict.get('loss'),
                            progress_dict.get('accuracy'),
                            progress_dict.get('learning_rate'),
                            progress_dict.get('metrics'),
                            progress_dict.get('checkpoint_path'),
                            progress_dict.get('timestamp')
                        ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_training_progress failed: {pg_error}")
                        return ""
                else:
                    logger.warning("PostgreSQL not available for saving training progress")
                    return ""
            
            logger.debug(f"Training progress saved: {progress_id} for model {progress.model_id}")
            return progress_id
            
        except Exception as e:
            logger.error(f"Failed to save training progress for model {progress.model_id}: {e}")
            return ""
    
    def get_training_progress(self, model_id: str, limit: int = 100) -> List[TrainingProgress]:
        """获取训练进度记录"""
        try:
            progress_list = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM training_progress 
                    WHERE model_id = ? 
                    ORDER BY epoch DESC, step DESC 
                    LIMIT ?
                """, (model_id, limit))
                
                for row in cursor.fetchall():
                    progress_dict = dict(row)
                    progress_list.append(TrainingProgress.from_dict(progress_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM training_progress 
                            WHERE model_id = %s 
                            ORDER BY epoch DESC, step DESC 
                            LIMIT %s
                        """, (model_id, limit))
                        
                        for row in cursor.fetchall():
                            progress_dict = dict(row)
                            progress_list.append(TrainingProgress.from_dict(progress_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_training_progress failed: {pg_error}")
                        return []
            
            return progress_list
            
        except Exception as e:
            logger.error(f"Failed to get training progress for model {model_id}: {e}")
            return []
    
    def get_training_progress_by_epoch(self, model_id: str, epoch: int) -> List[TrainingProgress]:
        """获取特定训练周期的进度记录"""
        try:
            progress_list = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM training_progress 
                    WHERE model_id = ? AND epoch = ? 
                    ORDER BY step DESC
                """, (model_id, epoch))
                
                for row in cursor.fetchall():
                    progress_dict = dict(row)
                    progress_list.append(TrainingProgress.from_dict(progress_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM training_progress 
                            WHERE model_id = %s AND epoch = %s 
                            ORDER BY step DESC
                        """, (model_id, epoch))
                        
                        for row in cursor.fetchall():
                            progress_dict = dict(row)
                            progress_list.append(TrainingProgress.from_dict(progress_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_training_progress_by_epoch failed: {pg_error}")
                        return []
            
            return progress_list
            
        except Exception as e:
            logger.error(f"Failed to get training progress for model {model_id}, epoch {epoch}: {e}")
            return []
    
    def delete_training_progress(self, progress_id: str) -> bool:
        """删除训练进度记录"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM training_progress WHERE id = ?", (progress_id,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Training progress deleted: {progress_id}")
                    return True
                else:
                    logger.warning(f"Training progress not found for deletion: {progress_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM training_progress WHERE id = %s", (progress_id,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL training progress deleted: {progress_id}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL training progress not found for deletion: {progress_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_training_progress failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete training progress {progress_id}: {e}")
            return False
    
    def cleanup_old_training_progress(self, days_to_keep: int = 30) -> int:
        """清理旧的训练进度记录"""
        try:
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    DELETE FROM training_progress WHERE timestamp < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                logger.info(f"Cleaned up {deleted_count} old training progress records")
                return deleted_count
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            DELETE FROM training_progress WHERE timestamp < %s
                        """, (cutoff_str,))
                        
                        deleted_count = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        logger.info(f"PostgreSQL cleaned up {deleted_count} old training progress records")
                        return deleted_count
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL cleanup_old_training_progress failed: {pg_error}")
                        return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old training progress: {e}")
            return 0
    
    # ===== 知识库集成操作方法 =====
    
    def save_knowledge_base_integration(self, integration: KnowledgeBaseIntegration) -> bool:
        """保存知识库集成状态"""
        try:
            integration_dict = integration.to_dict()
            integration_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM knowledge_base_integration WHERE model_id = ?", (integration.model_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有集成状态
                    cursor.execute("""
                        UPDATE knowledge_base_integration 
                        SET integrated = ?, knowledge_loaded = ?, last_integration_time = ?,
                            integration_method = ?, integration_status = ?, 
                            knowledge_sources = ?, custom_data = ?
                        WHERE model_id = ?
                    """, (
                        integration_dict['integrated'],
                        integration_dict.get('knowledge_loaded', 0),
                        integration_dict.get('last_integration_time'),
                        integration_dict.get('integration_method'),
                        integration_dict.get('integration_status'),
                        integration_dict.get('knowledge_sources'),
                        integration_dict.get('custom_data'),
                        integration_dict['model_id']
                    ))
                else:
                    # 插入新集成状态
                    cursor.execute("""
                        INSERT INTO knowledge_base_integration 
                        (id, model_id, integrated, knowledge_loaded, last_integration_time,
                         integration_method, integration_status, knowledge_sources, custom_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        integration_dict['id'],
                        integration_dict['model_id'],
                        integration_dict['integrated'],
                        integration_dict.get('knowledge_loaded', 0),
                        integration_dict.get('last_integration_time'),
                        integration_dict.get('integration_method'),
                        integration_dict.get('integration_status'),
                        integration_dict.get('knowledge_sources'),
                        integration_dict.get('custom_data')
                    ))
                
                self.connection.commit()
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        # 检查是否已存在
                        cursor.execute("SELECT id FROM knowledge_base_integration WHERE model_id = %s", (integration.model_id,))
                        existing = cursor.fetchone()
                        
                        if existing:
                            # 更新现有集成状态
                            cursor.execute("""
                                UPDATE knowledge_base_integration 
                                SET integrated = %s, knowledge_loaded = %s, last_integration_time = %s,
                                    integration_method = %s, integration_status = %s, 
                                    knowledge_sources = %s, custom_data = %s
                                WHERE model_id = %s
                            """, (
                                integration_dict['integrated'],
                                integration_dict.get('knowledge_loaded', 0),
                                integration_dict.get('last_integration_time'),
                                integration_dict.get('integration_method'),
                                integration_dict.get('integration_status'),
                                integration_dict.get('knowledge_sources'),
                                integration_dict.get('custom_data'),
                                integration_dict['model_id']
                            ))
                        else:
                            # 插入新集成状态
                            cursor.execute("""
                                INSERT INTO knowledge_base_integration 
                                (id, model_id, integrated, knowledge_loaded, last_integration_time,
                                 integration_method, integration_status, knowledge_sources, custom_data)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                integration_dict['id'],
                                integration_dict['model_id'],
                                integration_dict['integrated'],
                                integration_dict.get('knowledge_loaded', 0),
                                integration_dict.get('last_integration_time'),
                                integration_dict.get('integration_method'),
                                integration_dict.get('integration_status'),
                                integration_dict.get('knowledge_sources'),
                                integration_dict.get('custom_data')
                            ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_knowledge_base_integration failed: {pg_error}")
                        return False
                else:
                    logger.warning("PostgreSQL not available for saving knowledge base integration")
                    return False
            
            logger.debug(f"Knowledge base integration saved: {integration.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base integration for model {integration.model_id}: {e}")
            return False
    
    def get_knowledge_base_integration(self, model_id: str) -> Optional[KnowledgeBaseIntegration]:
        """获取知识库集成状态"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM knowledge_base_integration WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    integration_dict = dict(row)
                    return KnowledgeBaseIntegration.from_dict(integration_dict)
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM knowledge_base_integration WHERE model_id = %s
                        """, (model_id,))
                        
                        row = cursor.fetchone()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if row:
                            integration_dict = dict(row)
                            return KnowledgeBaseIntegration.from_dict(integration_dict)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_knowledge_base_integration failed: {pg_error}")
                        return None
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base integration for model {model_id}: {e}")
            return None
    
    def get_all_knowledge_base_integrations(self) -> List[KnowledgeBaseIntegration]:
        """获取所有知识库集成状态"""
        try:
            integrations = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("SELECT * FROM knowledge_base_integration ORDER BY model_id")
                
                for row in cursor.fetchall():
                    integration_dict = dict(row)
                    integrations.append(KnowledgeBaseIntegration.from_dict(integration_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("SELECT * FROM knowledge_base_integration ORDER BY model_id")
                        
                        for row in cursor.fetchall():
                            integration_dict = dict(row)
                            integrations.append(KnowledgeBaseIntegration.from_dict(integration_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_all_knowledge_base_integrations failed: {pg_error}")
                        return []
            
            return integrations
            
        except Exception as e:
            logger.error(f"Failed to get all knowledge base integrations: {e}")
            return []
    
    def delete_knowledge_base_integration(self, model_id: str) -> bool:
        """删除知识库集成状态"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM knowledge_base_integration WHERE model_id = ?", (model_id,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"Knowledge base integration deleted: {model_id}")
                    return True
                else:
                    logger.warning(f"Knowledge base integration not found for deletion: {model_id}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM knowledge_base_integration WHERE model_id = %s", (model_id,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL knowledge base integration deleted: {model_id}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL knowledge base integration not found for deletion: {model_id}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_knowledge_base_integration failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete knowledge base integration {model_id}: {e}")
            return False
    
    # ===== API依赖状态操作方法 =====
    
    def save_api_dependency(self, dependency: APIDependencyRecord) -> bool:
        """保存API依赖配置"""
        try:
            dependency_dict = dependency.to_dict()
            dependency_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM api_dependencies WHERE name = ?", (dependency.name,))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有依赖
                    cursor.execute("""
                        UPDATE api_dependencies 
                        SET provider = ?, module_name = ?, client_class = ?, config = ?, 
                            priority = ?, optional = ?, fallback_providers = ?, is_available = ?,
                            error_count = ?, consecutive_errors = ?, last_error = ?,
                            last_health_check = ?, health_status = ?, updated_at = ?
                        WHERE name = ?
                    """, (
                        dependency_dict['provider'],
                        dependency_dict['module_name'],
                        dependency_dict['client_class'],
                        dependency_dict.get('config'),
                        dependency_dict['priority'],
                        dependency_dict['optional'],
                        dependency_dict.get('fallback_providers'),
                        dependency_dict['is_available'],
                        dependency_dict['error_count'],
                        dependency_dict['consecutive_errors'],
                        dependency_dict.get('last_error'),
                        dependency_dict.get('last_health_check'),
                        dependency_dict.get('health_status'),
                        dependency_dict['updated_at'],
                        dependency_dict['name']
                    ))
                else:
                    # 插入新依赖
                    cursor.execute("""
                        INSERT INTO api_dependencies 
                        (id, name, provider, module_name, client_class, config, priority, optional, 
                         fallback_providers, is_available, error_count, consecutive_errors, last_error,
                         last_health_check, health_status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dependency_dict['id'],
                        dependency_dict['name'],
                        dependency_dict['provider'],
                        dependency_dict['module_name'],
                        dependency_dict['client_class'],
                        dependency_dict.get('config'),
                        dependency_dict['priority'],
                        dependency_dict['optional'],
                        dependency_dict.get('fallback_providers'),
                        dependency_dict['is_available'],
                        dependency_dict['error_count'],
                        dependency_dict['consecutive_errors'],
                        dependency_dict.get('last_error'),
                        dependency_dict.get('last_health_check'),
                        dependency_dict.get('health_status'),
                        dependency_dict['created_at'],
                        dependency_dict['updated_at']
                    ))
                
                self.connection.commit()
                cursor.close()
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        # 检查是否已存在
                        cursor.execute("SELECT id FROM api_dependencies WHERE name = %s", (dependency.name,))
                        existing = cursor.fetchone()
                        
                        if existing:
                            # 更新现有依赖
                            cursor.execute("""
                                UPDATE api_dependencies 
                                SET provider = %s, module_name = %s, client_class = %s, config = %s, 
                                    priority = %s, optional = %s, fallback_providers = %s, is_available = %s,
                                    error_count = %s, consecutive_errors = %s, last_error = %s,
                                    last_health_check = %s, health_status = %s, updated_at = %s
                                WHERE name = %s
                            """, (
                                dependency_dict['provider'],
                                dependency_dict['module_name'],
                                dependency_dict['client_class'],
                                json.dumps(dependency_dict.get('config')) if dependency_dict.get('config') else None,
                                dependency_dict['priority'],
                                dependency_dict['optional'],
                                json.dumps(dependency_dict.get('fallback_providers')) if dependency_dict.get('fallback_providers') else None,
                                dependency_dict['is_available'],
                                dependency_dict['error_count'],
                                dependency_dict['consecutive_errors'],
                                dependency_dict.get('last_error'),
                                dependency_dict.get('last_health_check'),
                                dependency_dict.get('health_status'),
                                dependency_dict['updated_at'],
                                dependency_dict['name']
                            ))
                        else:
                            # 插入新依赖
                            cursor.execute("""
                                INSERT INTO api_dependencies 
                                (id, name, provider, module_name, client_class, config, priority, optional, 
                                 fallback_providers, is_available, error_count, consecutive_errors, last_error,
                                 last_health_check, health_status, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                dependency_dict['id'],
                                dependency_dict['name'],
                                dependency_dict['provider'],
                                dependency_dict['module_name'],
                                dependency_dict['client_class'],
                                json.dumps(dependency_dict.get('config')) if dependency_dict.get('config') else None,
                                dependency_dict['priority'],
                                dependency_dict['optional'],
                                json.dumps(dependency_dict.get('fallback_providers')) if dependency_dict.get('fallback_providers') else None,
                                dependency_dict['is_available'],
                                dependency_dict['error_count'],
                                dependency_dict['consecutive_errors'],
                                dependency_dict.get('last_error'),
                                dependency_dict.get('last_health_check'),
                                dependency_dict.get('health_status'),
                                dependency_dict['created_at'],
                                dependency_dict['updated_at']
                            ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_api_dependency failed: {pg_error}")
                        return False
                else:
                    logger.error("PostgreSQL not available for saving API dependency")
                    return False
            
            logger.debug(f"API dependency saved: {dependency.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API dependency {dependency.name}: {e}")
            return False
    
    def get_api_dependency(self, name: str) -> Optional[APIDependencyRecord]:
        """获取API依赖配置"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM api_dependencies WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    dependency_dict = dict(row)
                    return APIDependencyRecord.from_dict(dependency_dict)
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM api_dependencies WHERE name = %s
                        """, (name,))
                        
                        row = cursor.fetchone()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if row:
                            # PostgreSQL返回的是元组，需要转换为字典
                            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                            dependency_dict = dict(zip(column_names, row)) if column_names else {}
                            return APIDependencyRecord.from_dict(dependency_dict)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_api_dependency failed: {pg_error}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get API dependency {name}: {e}")
            return None
    
    def get_all_api_dependencies(self) -> List[APIDependencyRecord]:
        """获取所有API依赖配置"""
        try:
            dependencies = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("SELECT * FROM api_dependencies ORDER BY name")
                
                for row in cursor.fetchall():
                    dependency_dict = dict(row)
                    dependencies.append(APIDependencyRecord.from_dict(dependency_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("SELECT * FROM api_dependencies ORDER BY name")
                        
                        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                        for row in cursor.fetchall():
                            dependency_dict = dict(zip(column_names, row)) if column_names else {}
                            dependencies.append(APIDependencyRecord.from_dict(dependency_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_all_api_dependencies failed: {pg_error}")
                        return []
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to get all API dependencies: {e}")
            return []
    
    def delete_api_dependency(self, name: str) -> bool:
        """删除API依赖配置"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("DELETE FROM api_dependencies WHERE name = ?", (name,))
                
                affected_rows = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                if affected_rows > 0:
                    logger.debug(f"API dependency deleted: {name}")
                    return True
                else:
                    logger.warning(f"API dependency not found for deletion: {name}")
                    return False
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("DELETE FROM api_dependencies WHERE name = %s", (name,))
                        
                        affected_rows = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if affected_rows > 0:
                            logger.debug(f"PostgreSQL API dependency deleted: {name}")
                            return True
                        else:
                            logger.warning(f"PostgreSQL API dependency not found: {name}")
                            return False
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL delete_api_dependency failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete API dependency {name}: {e}")
            return False
    
    def save_dependency_health_status(self, health_status: DependencyHealthRecord) -> bool:
        """保存依赖健康状态"""
        try:
            health_dict = health_status.to_dict()
            health_dict['id'] = str(uuid.uuid4())
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM dependency_health_status WHERE dependency_name = ?", 
                             (health_status.dependency_name,))
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有健康状态
                    cursor.execute("""
                        UPDATE dependency_health_status 
                        SET is_healthy = ?, response_time = ?, error_message = ?, last_check = ?,
                            check_count = ?, success_count = ?, failure_count = ?, health_score = ?,
                            consecutive_errors = ?, custom_metrics = ?, updated_at = ?
                        WHERE dependency_name = ?
                    """, (
                        health_dict['is_healthy'],
                        health_dict.get('response_time'),
                        health_dict.get('error_message'),
                        health_dict.get('last_check'),
                        health_dict['check_count'],
                        health_dict['success_count'],
                        health_dict['failure_count'],
                        health_dict['health_score'],
                        health_dict['consecutive_errors'],
                        health_dict.get('custom_metrics'),
                        health_dict['updated_at'],
                        health_dict['dependency_name']
                    ))
                else:
                    # 插入新健康状态
                    cursor.execute("""
                        INSERT INTO dependency_health_status 
                        (id, dependency_name, is_healthy, response_time, error_message, last_check,
                         check_count, success_count, failure_count, health_score, consecutive_errors,
                         custom_metrics, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        health_dict['id'],
                        health_dict['dependency_name'],
                        health_dict['is_healthy'],
                        health_dict.get('response_time'),
                        health_dict.get('error_message'),
                        health_dict.get('last_check'),
                        health_dict['check_count'],
                        health_dict['success_count'],
                        health_dict['failure_count'],
                        health_dict['health_score'],
                        health_dict['consecutive_errors'],
                        health_dict.get('custom_metrics'),
                        health_dict['created_at'],
                        health_dict['updated_at']
                    ))
                
                self.connection.commit()
                cursor.close()
                
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        # 定义截止时间（24小时前）
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        # 检查记录是否存在
                        cursor.execute("""
                            SELECT id FROM dependency_health_status 
                            WHERE dependency_name = %s AND timestamp >= %s
                        """, (health_status.dependency_name, cutoff_time))
                        
                        existing_record = cursor.fetchone()
                        
                        if existing_record:
                            # 更新现有记录
                            cursor.execute("""
                                UPDATE dependency_health_status 
                                SET status = %s, response_time = %s, error_count = %s,
                                    last_checked = %s, details = %s, timestamp = %s
                                WHERE dependency_name = %s AND timestamp >= %s
                            """, (
                                health_status.status,
                                health_status.response_time,
                                health_status.error_count,
                                health_status.last_checked,
                                json.dumps(health_status.details) if health_status.details else None,
                                health_status.timestamp,
                                health_status.dependency_name,
                                cutoff_time
                            ))
                        else:
                            # 插入新记录
                            cursor.execute("""
                                INSERT INTO dependency_health_status 
                                (dependency_name, status, response_time, error_count, last_checked, details, timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                health_status.dependency_name,
                                health_status.status,
                                health_status.response_time,
                                health_status.error_count,
                                health_status.last_checked,
                                json.dumps(health_status.details) if health_status.details else None,
                                health_status.timestamp
                            ))
                        
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        return True
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL save_dependency_health_status failed: {pg_error}")
                        return False
                else:
                    logger.error("PostgreSQL not available for saving dependency health status")
                    return False
            
            logger.debug(f"Dependency health status saved: {health_status.dependency_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dependency health status {health_status.dependency_name}: {e}")
            return False
    
    def get_dependency_health_status(self, dependency_name: str) -> Optional[DependencyHealthRecord]:
        """获取依赖健康状态"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM dependency_health_status WHERE dependency_name = ?
                """, (dependency_name,))
                
                row = cursor.fetchone()
                cursor.close()
                
                if row:
                    health_dict = dict(row)
                    return DependencyHealthRecord.from_dict(health_dict)
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        # 定义截止时间（24小时前）
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM dependency_health_status 
                            WHERE dependency_name = %s AND timestamp >= %s
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (dependency_name, cutoff_time))
                        
                        row = cursor.fetchone()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        if row:
                            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                            health_dict = dict(zip(column_names, row)) if column_names else {}
                            
                            # 处理JSON字段
                            if 'details' in health_dict and health_dict['details']:
                                try:
                                    health_dict['details'] = json.loads(health_dict['details'])
                                except Exception as json_error:
                                    logger.warning(f"Failed to parse details JSON: {json_error}, using empty dict")
                                    health_dict['details'] = {}
                            
                            return DependencyHealthRecord.from_dict(health_dict)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_dependency_health_status failed: {pg_error}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get dependency health status {dependency_name}: {e}")
            return None
    
    def get_all_dependency_health_status(self) -> List[DependencyHealthRecord]:
        """获取所有依赖健康状态"""
        try:
            health_status_list = []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("SELECT * FROM dependency_health_status ORDER BY dependency_name")
                
                for row in cursor.fetchall():
                    health_dict = dict(row)
                    health_status_list.append(DependencyHealthRecord.from_dict(health_dict))
                
                cursor.close()
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        # 定义截止时间（24小时前）
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM dependency_health_status 
                            WHERE timestamp >= %s
                            ORDER BY timestamp DESC
                        """, (cutoff_time,))
                        
                        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                        for row in cursor.fetchall():
                            health_dict = dict(zip(column_names, row)) if column_names else {}
                            
                            # 处理JSON字段
                            if 'details' in health_dict and health_dict['details']:
                                try:
                                    health_dict['details'] = json.loads(health_dict['details'])
                                except Exception as json_error:
                                    logger.warning(f"Failed to parse details JSON: {json_error}, using empty dict")
                                    health_dict['details'] = {}
                            
                            health_status_list.append(DependencyHealthRecord.from_dict(health_dict))
                        
                        cursor.close()
                        self.db_pool.return_connection(connection)
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL get_all_dependency_health_status failed: {pg_error}")
                        return []
            
            return health_status_list
            
        except Exception as e:
            logger.error(f"Failed to get all dependency health status: {e}")
            return []
    
    def cleanup_old_health_status(self, days_to_keep: int = 90) -> int:
        """清理旧的健康状态记录"""
        try:
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute("""
                    DELETE FROM dependency_health_status WHERE last_check < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                self.connection.commit()
                cursor.close()
                
                logger.info(f"Cleaned up {deleted_count} old dependency health status records")
                return deleted_count
            
            else:
                # PostgreSQL实现
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        connection = self.db_pool.get_connection()
                        cursor = connection.cursor()
                        
                        cursor.execute("""
                            DELETE FROM dependency_health_status 
                            WHERE timestamp < %s
                        """, (cutoff_date,))
                        
                        deleted_count = cursor.rowcount
                        connection.commit()
                        cursor.close()
                        self.db_pool.return_connection(connection)
                        
                        logger.debug(f"PostgreSQL cleanup deleted {deleted_count} old health status records")
                        return deleted_count
                            
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL cleanup_old_health_status failed: {pg_error}")
                        return 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old health status: {e}")
            return 0
    
    # ===== 数据库维护方法 =====
    
    def backup_database(self, backup_path: str = None) -> bool:
        """备份数据库"""
        try:
            if self.db_type == "sqlite":
                if backup_path is None:
                    backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 复制数据库文件
                import shutil
                shutil.copy2(self.db_path, backup_path)
                
                logger.info(f"Database backed up to {backup_path}")
                return True
            
            else:
                # PostgreSQL备份
                if PRODUCTION_DB_AVAILABLE and self.db_pool:
                    try:
                        if backup_path is None:
                            backup_path = f"postgresql_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        
                        backup_data = {}
                        
                        # 备份模型配置
                        try:
                            model_configs = self.get_all_model_configs()
                            backup_data['model_configs'] = [config.to_dict() if hasattr(config, 'to_dict') else config 
                                                           for config in model_configs]
                        except Exception as e:
                            logger.warning(f"Failed to backup model configs: {e}")
                        
                        # 备份训练记录
                        try:
                            training_records = self.get_all_training_records()
                            backup_data['training_records'] = [record.to_dict() if hasattr(record, 'to_dict') else record 
                                                              for record in training_records]
                        except Exception as e:
                            logger.warning(f"Failed to backup training records: {e}")
                        
                        # 备份系统日志
                        try:
                            system_logs = self.get_all_system_logs()
                            backup_data['system_logs'] = system_logs
                        except Exception as e:
                            logger.warning(f"Failed to backup system logs: {e}")
                        
                        # 备份API依赖
                        try:
                            api_dependencies = self.get_all_api_dependencies()
                            backup_data['api_dependencies'] = [dep.to_dict() if hasattr(dep, 'to_dict') else dep 
                                                              for dep in api_dependencies]
                        except Exception as e:
                            logger.warning(f"Failed to backup API dependencies: {e}")
                        
                        # 备份健康状态
                        try:
                            health_status = self.get_all_health_status()
                            backup_data['health_status'] = health_status
                        except Exception as e:
                            logger.warning(f"Failed to backup health status: {e}")
                        
                        # 保存备份数据
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            json.dump(backup_data, f, indent=2, default=str)
                        
                        logger.info(f"PostgreSQL database backed up to {backup_path}")
                        return True
                        
                    except Exception as pg_error:
                        logger.error(f"PostgreSQL backup failed: {pg_error}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.db_type == "sqlite" and hasattr(self, 'connection'):
                self.connection.close()
                logger.info("SQLite connection closed")
            
            elif self.db_pool is not None:
                self.db_pool.close_all_connections()
                logger.info("Database connection pool closed")
                
        except Exception as e:
            logger.error(f"Failed to close database connection: {e}")

# 全局实例
_db_access_layer = None

def get_db_access_layer() -> DatabaseAccessLayer:
    """获取全局数据库访问层实例"""
    global _db_access_layer
    if _db_access_layer is None:
        _db_access_layer = DatabaseAccessLayer()
    return _db_access_layer

def close_db_access_layer():
    """关闭全局数据库访问层实例"""
    global _db_access_layer
    if _db_access_layer is not None:
        _db_access_layer.close()
        _db_access_layer = None
        logger.info("Database access layer closed")

# 兼容性包装器，用于逐步替换内存存储
class ModelConfigStore:
    """模型配置存储兼容层，支持字典接口"""
    
    def __init__(self):
        self.db = get_db_access_layer()
        self.memory_cache = {}  # 内存缓存，提高性能
    
    def save(self, model_id: str, config_data: Dict[str, Any]) -> bool:
        """保存模型配置"""
        try:
            # 创建ModelConfig对象
            config = ModelConfig(
                model_id=model_id,
                model_name=config_data.get('model_name', model_id),
                model_type=config_data.get('model_type', 'unknown'),
                source=config_data.get('source', 'local'),
                api_config=config_data.get('api_config'),
                is_active=config_data.get('is_active', False),
                created_at=config_data.get('created_at'),
                updated_at=config_data.get('updated_at')
            )
            
            # 保存到数据库
            success = self.db.save_model_config(config)
            
            if success:
                # 更新内存缓存
                self.memory_cache[model_id] = config_data
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save model config {model_id}: {e}")
            return False
    
    def get(self, model_id: str, default=None) -> Optional[Dict[str, Any]]:
        """获取模型配置，支持默认值"""
        # 首先检查内存缓存
        if model_id in self.memory_cache:
            return self.memory_cache[model_id]
        
        # 从数据库获取
        config = self.db.get_model_config(model_id)
        if config:
            config_dict = asdict(config)
            # 更新内存缓存
            self.memory_cache[model_id] = config_dict
            return config_dict
        
        return default
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型配置"""
        configs = {}
        
        # 从数据库获取所有配置
        db_configs = self.db.get_all_model_configs()
        for config in db_configs:
            config_dict = asdict(config)
            model_id = config.model_id
            configs[model_id] = config_dict
            # 更新内存缓存
            self.memory_cache[model_id] = config_dict
        
        return configs
    
    def delete(self, model_id: str) -> bool:
        """删除模型配置"""
        try:
            # 从数据库删除
            success = self.db.delete_model_config(model_id)
            
            if success:
                # 从内存缓存删除
                if model_id in self.memory_cache:
                    del self.memory_cache[model_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete model config {model_id}: {e}")
            return False
    
    # 字典接口支持
    def __setitem__(self, model_id: str, config_data: Dict[str, Any]):
        """支持字典赋值语法"""
        self.save(model_id, config_data)
    
    def __getitem__(self, model_id: str) -> Dict[str, Any]:
        """支持字典获取语法"""
        result = self.get(model_id)
        if result is None:
            raise KeyError(f"Model config not found: {model_id}")
        return result
    
    def __delitem__(self, model_id: str):
        """支持字典删除语法"""
        success = self.delete(model_id)
        if not success:
            raise KeyError(f"Model config not found or could not be deleted: {model_id}")
    
    def __contains__(self, model_id: str) -> bool:
        """支持in操作符"""
        return self.get(model_id) is not None
    
    def __len__(self) -> int:
        """支持len()函数"""
        configs = self.get_all()
        return len(configs)
    
    def keys(self):
        """支持keys()方法"""
        configs = self.get_all()
        return configs.keys()
    
    def values(self):
        """支持values()方法"""
        configs = self.get_all()
        return configs.values()
    
    def items(self):
        """支持items()方法"""
        configs = self.get_all()
        return configs.items()
    
    def clear(self):
        """清空所有配置（危险操作，谨慎使用）"""
        try:
            configs = self.get_all()
            success_count = 0
            for model_id in list(configs.keys()):
                if self.delete(model_id):
                    success_count += 1
            logger.info(f"Cleared {success_count} model configs")
            return success_count == len(configs)
        except Exception as e:
            logger.error(f"Failed to clear model configs: {e}")
            return False