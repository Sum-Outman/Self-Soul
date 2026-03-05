"""
生产级数据库和存储系统优化配置
Production Database and Storage System Optimization Configuration
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import sqlite3
import psycopg2  # type: ignore
from psycopg2 import pool  # type: ignore
import redis
from redis import ConnectionPool
import pickle
import zlib

from core.error_handling import error_handler

logger = logging.getLogger("ProductionDatabase")


@dataclass
class DatabaseConfig:
    """数据库配置"""
    # 数据库类型
    db_type: str  # "sqlite", "postgresql"
    
    # 连接配置
    host: str = "localhost"
    port: int = 5432
    database: str = "selfsoul"
    username: str = "selfsoul"
    password: str = ""
    
    # 连接池配置
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: int = 30
    
    # 性能配置
    query_timeout: int = 30
    idle_timeout: int = 300
    
    # 备份配置
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7


@dataclass 
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    
    # 连接池配置
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # 性能配置
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # 内存配置
    max_memory: str = "512mb"
    max_memory_policy: str = "allkeys-lru"


@dataclass
class StorageConfig:
    """存储配置"""
    # 文件存储配置
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    backups_dir: str = "backups"
    
    # 存储限制
    max_file_size_mb: int = 100
    max_total_storage_gb: int = 10
    
    # 压缩配置
    enable_compression: bool = True
    compression_level: int = 6
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    
    # 备份配置
    auto_backup: bool = True
    backup_schedule: str = "0 2 * * *"  # 每天凌晨2点


class DatabaseConnectionPool:
    """数据库连接池管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        try:
            if self.config.db_type == "postgresql":
                self.connection_pool = pool.ThreadedConnectionPool(
                    minconn=self.config.min_connections,
                    maxconn=self.config.max_connections,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password,
                    connect_timeout=self.config.connection_timeout
                )
                logger.info("PostgreSQL connection pool initialized")
            
            elif self.config.db_type == "sqlite":
                # SQLite不需要连接池，但我们可以模拟连接管理
                self.connection_pool = SQLiteConnectionManager(self.config)
                logger.info("SQLite connection manager initialized")
            
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            if self.config.db_type == "postgresql":
                return self.connection_pool.getconn()
            else:
                return self.connection_pool.get_connection()
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    def return_connection(self, connection):
        """归还数据库连接"""
        try:
            if self.config.db_type == "postgresql":
                self.connection_pool.putconn(connection)
            else:
                self.connection_pool.return_connection(connection)
        except Exception as e:
            logger.error(f"Failed to return database connection: {e}")
    
    def close_all_connections(self):
        """关闭所有连接"""
        try:
            if self.config.db_type == "postgresql":
                self.connection_pool.closeall()
            else:
                self.connection_pool.close_all()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Failed to close database connections: {e}")


class SQLiteConnectionManager:
    """SQLite连接管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.active_connections = 0
        self.max_connections = config.max_connections
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(config.database), exist_ok=True)
    
    def get_connection(self):
        """获取SQLite连接"""
        if self.active_connections >= self.max_connections:
            raise RuntimeError("Maximum SQLite connections reached")
        
        connection = sqlite3.connect(self.config.database, timeout=self.config.connection_timeout)
        connection.row_factory = sqlite3.Row
        self.active_connections += 1
        return connection
    
    def return_connection(self, connection):
        """归还SQLite连接"""
        try:
            connection.close()
            self.active_connections -= 1
        except Exception as e:
            logger.error(f"Failed to close SQLite connection: {e}")
    
    def close_all(self):
        """关闭所有连接（SQLite不需要显式关闭）"""
        self.active_connections = 0


class RedisConnectionManager:
    """Redis连接管理器"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化Redis连接池"""
        try:
            self.connection_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password or None,
                db=self.config.database,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # 测试连接
            with self.get_connection() as client:
                client.ping()
            
            logger.info("Redis connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise
    
    def get_connection(self) -> redis.Redis:
        """获取Redis连接"""
        return redis.Redis(connection_pool=self.connection_pool)
    
    def close_pool(self):
        """关闭连接池"""
        if self.connection_pool:
            self.connection_pool.disconnect()
            logger.info("Redis connection pool closed")


class StorageManager:
    """存储管理器"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._initialize_storage()
    
    def _initialize_storage(self):
        """初始化存储目录"""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.logs_dir,
            self.config.backups_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Storage directories initialized")
    
    def store_data(self, key: str, data: Any, compress: bool = None) -> bool:
        """存储数据"""
        try:
            if compress is None:
                compress = self.config.enable_compression
            
            # 序列化数据
            serialized_data = pickle.dumps(data)
            
            # 压缩数据
            if compress:
                serialized_data = zlib.compress(serialized_data, self.config.compression_level)
            
            # 检查文件大小
            if len(serialized_data) > self.config.max_file_size_mb * 1024 * 1024:
                error_handler.log_warning(f"Data too large for storage: {len(serialized_data)} bytes", "ProductionDatabase")
                return False
            
            # 写入文件
            file_path = os.path.join(self.config.data_dir, f"{key}.dat")
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            logger.debug(f"Data stored successfully: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data {key}: {e}")
            return False
    
    def retrieve_data(self, key: str, decompress: bool = None) -> Optional[Any]:
        """检索数据"""
        try:
            if decompress is None:
                decompress = self.config.enable_compression
            
            file_path = os.path.join(self.config.data_dir, f"{key}.dat")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
            
            # 解压缩数据
            if decompress:
                serialized_data = zlib.decompress(serialized_data)
            
            # 反序列化数据
            data = pickle.loads(serialized_data)
            
            logger.debug(f"Data retrieved successfully: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data {key}: {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """删除数据"""
        try:
            file_path = os.path.join(self.config.data_dir, f"{key}.dat")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Data deleted successfully: {key}")
                return True
            else:
                error_handler.log_warning(f"Data not found for deletion: {key}", "ProductionDatabase")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete data {key}: {e}")
            return False
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """获取存储使用情况"""
        try:
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(self.config.data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            usage_gb = total_size / (1024 ** 3)
            max_gb = self.config.max_total_storage_gb
            usage_percent = (usage_gb / max_gb) * 100 if max_gb > 0 else 0
            
            return {
                "total_size_gb": round(usage_gb, 2),
                "max_size_gb": max_gb,
                "usage_percent": round(usage_percent, 1),
                "file_count": file_count,
                "status": "ok" if usage_percent < 90 else "warning"
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {"error": str(e)}


class DatabaseOptimizer:
    """数据库优化器"""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        self.db_pool = db_pool
    
    def optimize_tables(self) -> bool:
        """优化数据库表"""
        try:
            connection = self.db_pool.get_connection()
            cursor = connection.cursor()
            
            if self.db_pool.config.db_type == "postgresql":
                # PostgreSQL优化
                cursor.execute("VACUUM ANALYZE;")
                cursor.execute("REINDEX DATABASE %s;", (self.db_pool.config.database,))
            
            elif self.db_pool.config.db_type == "sqlite":
                # SQLite优化
                cursor.execute("VACUUM;")
                cursor.execute("PRAGMA optimize;")
            
            connection.commit()
            cursor.close()
            self.db_pool.return_connection(connection)
            
            logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            connection = self.db_pool.get_connection()
            cursor = connection.cursor()
            
            stats = {}
            
            if self.db_pool.config.db_type == "postgresql":
                # PostgreSQL统计
                cursor.execute("""
                    SELECT 
                        schemaname, tablename, 
                        n_live_tup, n_dead_tup,
                        last_vacuum, last_autovacuum,
                        last_analyze, last_autoanalyze
                    FROM pg_stat_user_tables;
                """)
                
                tables = cursor.fetchall()
                stats["tables"] = [dict(table) for table in tables]
                
                # 连接统计
                cursor.execute("SELECT count(*) FROM pg_stat_activity;")
                stats["active_connections"] = cursor.fetchone()[0]
                
            elif self.db_pool.config.db_type == "sqlite":
                # SQLite统计
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                table_stats = []
                for table in tables:
                    table_name = table[0]
                    # 验证表名格式以防止SQL注入
                    # 只允许字母、数字、下划线，且不能以数字开头
                    if not table_name.replace('_', '').isalnum():
                        logger.warning(f"Skipping table with invalid name: {table_name}")
                        continue
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    row_count = cursor.fetchone()[0]
                    table_stats.append({"name": table_name, "row_count": row_count})
                
                stats["tables"] = table_stats
            
            cursor.close()
            self.db_pool.return_connection(connection)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


class BackupManager:
    """备份管理器"""
    
    def __init__(self, db_pool: DatabaseConnectionPool, storage_manager: StorageManager):
        self.db_pool = db_pool
        self.storage_manager = storage_manager
        self.last_backup_time = None
    
    def create_backup(self) -> bool:
        """创建数据库备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.db_pool.config.db_type == "postgresql":
                # PostgreSQL备份
                backup_file = os.path.join(self.storage_manager.config.backups_dir, f"backup_{timestamp}.sql")
                
                # 使用pg_dump进行备份（需要系统命令）
                import subprocess
                
                cmd = [
                    "pg_dump",
                    "-h", self.db_pool.config.host,
                    "-p", str(self.db_pool.config.port),
                    "-U", self.db_pool.config.username,
                    "-d", self.db_pool.config.database,
                    "-f", backup_file
                ]
                
                # 设置密码环境变量
                env = os.environ.copy()
                env["PGPASSWORD"] = self.db_pool.config.password
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"PostgreSQL backup failed: {result.stderr}")
                    return False
            
            elif self.db_pool.config.db_type == "sqlite":
                # SQLite备份
                backup_file = os.path.join(self.storage_manager.config.backups_dir, f"backup_{timestamp}.db")
                
                connection = self.db_pool.get_connection()
                
                # 创建备份连接
                backup_conn = sqlite3.connect(backup_file)
                connection.backup(backup_conn)
                backup_conn.close()
                
                self.db_pool.return_connection(connection)
            
            self.last_backup_time = datetime.now()
            logger.info(f"Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """清理旧备份"""
        try:
            deleted_count = 0
            retention_days = self.db_pool.config.backup_retention_days
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            for filename in os.listdir(self.storage_manager.config.backups_dir):
                if filename.startswith("backup_") and (filename.endswith(".sql") or filename.endswith(".db")):
                    file_path = os.path.join(self.storage_manager.config.backups_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted old backup: {filename}")
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0


# 生产环境配置函数
def create_production_database_config() -> DatabaseConfig:
    """创建生产环境数据库配置"""
    environment = os.getenv("ENVIRONMENT", "development")
    db_password = os.getenv("DB_PASSWORD", "")
    
    # 生产环境必须设置数据库密码
    if environment == "production" and not db_password:
        raise ValueError("DB_PASSWORD environment variable must be set in production environment")
    
    return DatabaseConfig(
        db_type=os.getenv("DB_TYPE", "sqlite"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "selfsoul"),
        username=os.getenv("DB_USER", "selfsoul"),
        password=db_password,
        min_connections=2,
        max_connections=20,
        connection_timeout=30,
        query_timeout=30,
        idle_timeout=300,
        backup_enabled=True,
        backup_interval_hours=24,
        backup_retention_days=7
    )


def create_production_redis_config() -> RedisConfig:
    """创建生产环境Redis配置"""
    environment = os.getenv("ENVIRONMENT", "development")
    redis_password = os.getenv("REDIS_PASSWORD", "")
    
    # 生产环境必须设置Redis密码
    if environment == "production" and not redis_password:
        raise ValueError("REDIS_PASSWORD environment variable must be set in production environment")
    
    return RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=redis_password,
        database=int(os.getenv("REDIS_DB", "0")),
        max_connections=50,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
        max_memory="512mb",
        max_memory_policy="allkeys-lru"
    )


def create_production_storage_config() -> StorageConfig:
    """创建生产环境存储配置"""
    return StorageConfig(
        data_dir=os.getenv("DATA_DIR", "data"),
        models_dir=os.getenv("MODELS_DIR", "models"),
        logs_dir=os.getenv("LOGS_DIR", "logs"),
        backups_dir=os.getenv("BACKUPS_DIR", "backups"),
        max_file_size_mb=100,
        max_total_storage_gb=10,
        enable_compression=True,
        compression_level=6,
        cache_enabled=True,
        cache_ttl_hours=24,
        auto_backup=True,
        backup_schedule="0 2 * * *"
    )


def initialize_production_database() -> DatabaseConnectionPool:
    """初始化生产环境数据库"""
    config = create_production_database_config()
    db_pool = DatabaseConnectionPool(config)
    logger.info("Production database initialized")
    return db_pool


def initialize_production_redis() -> RedisConnectionManager:
    """初始化生产环境Redis"""
    config = create_production_redis_config()
    redis_manager = RedisConnectionManager(config)
    logger.info("Production Redis initialized")
    return redis_manager


def initialize_production_storage() -> StorageManager:
    """初始化生产环境存储"""
    config = create_production_storage_config()
    storage_manager = StorageManager(config)
    logger.info("Production storage initialized")
    return storage_manager


# 全局实例
production_db_pool = None
production_redis_manager = None
production_storage_manager = None

def get_production_database() -> DatabaseConnectionPool:
    """获取全局数据库实例"""
    global production_db_pool
    if production_db_pool is None:
        production_db_pool = initialize_production_database()
    return production_db_pool

def get_production_redis() -> RedisConnectionManager:
    """获取全局Redis实例"""
    global production_redis_manager
    if production_redis_manager is None:
        production_redis_manager = initialize_production_redis()
    return production_redis_manager

def get_production_storage() -> StorageManager:
    """获取全局存储实例"""
    global production_storage_manager
    if production_storage_manager is None:
        production_storage_manager = initialize_production_storage()
    return production_storage_manager
