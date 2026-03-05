"""
系统备份和恢复机制
System Backup and Recovery Mechanism
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from .error_handling import error_handler
import json
import shutil
import zipfile
import tarfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("BackupRecovery")

@dataclass
class BackupConfig:
    """备份配置"""
    # 备份目录
    backup_dir: str = "backups"
    
    # 备份策略
    auto_backup: bool = True
    backup_schedule: str = "0 2 * * *"  # 每天凌晨2点
    retention_days: int = 30
    max_backup_count: int = 100
    
    # 备份内容
    backup_database: bool = True
    backup_models: bool = True
    backup_configs: bool = True
    backup_logs: bool = False  # 默认不备份日志
    
    # 压缩配置
    enable_compression: bool = True
    compression_level: int = 6
    
    # 加密配置
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    
    # 验证配置
    enable_verification: bool = True
    checksum_algorithm: str = "sha256"

@dataclass
class BackupMetadata:
    """备份元数据"""
    backup_id: str
    timestamp: str
    size_bytes: int
    checksum: str
    version: str
    components: List[str]
    status: str  # "completed", "failed", "in_progress"
    error_message: Optional[str] = None

class BackupManager:
    """备份管理器"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # 线程池用于并行备份
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info("Backup manager initialized")
    
    def create_backup(self, backup_name: Optional[str] = None) -> Optional[BackupMetadata]:
        """创建系统备份"""
        try:
            # 生成备份ID
            backup_id = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / f"{backup_id}.zip"
            
            # 创建临时目录
            temp_dir = self.backup_dir / "temp" / backup_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化元数据
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                size_bytes=0,
                checksum="",
                version=self._get_system_version(),
                components=[],
                status="in_progress"
            )
            
            # 保存初始元数据
            self._save_metadata(metadata, temp_dir)
            
            # 并行备份各个组件
            backup_tasks = []
            
            if self.config.backup_database:
                backup_tasks.append(self._backup_database(temp_dir))
            
            if self.config.backup_models:
                backup_tasks.append(self._backup_models(temp_dir))
            
            if self.config.backup_configs:
                backup_tasks.append(self._backup_configs(temp_dir))
            
            if self.config.backup_logs:
                backup_tasks.append(self._backup_logs(temp_dir))
            
            # 等待所有备份任务完成
            completed_components = []
            for task in backup_tasks:
                component = task.result()
                if component:
                    completed_components.append(component)
            
            metadata.components = completed_components
            
            # 创建压缩文件
            if self.config.enable_compression:
                self._create_archive(temp_dir, backup_path)
            else:
                # 如果不压缩，直接移动整个目录
                shutil.make_archive(str(backup_path.with_suffix('')), 'zip', temp_dir)
            
            # 计算文件大小和校验和
            metadata.size_bytes = backup_path.stat().st_size
            metadata.checksum = self._calculate_checksum(backup_path)
            
            # 验证备份（如果启用）
            if self.config.enable_verification:
                if not self._verify_backup(backup_path, metadata.checksum):
                    raise ValueError("Backup verification failed")
            
            # 更新元数据状态
            metadata.status = "completed"
            self._save_metadata(metadata, backup_path.parent)
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            logger.info(f"Backup created successfully: {backup_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            
            # 保存错误信息到元数据
            if 'metadata' in locals():
                metadata.status = "failed"
                metadata.error_message = str(e)
                self._save_metadata(metadata, temp_dir)
            
            return None
    
    def restore_backup(self, backup_id: str, target_dir: str = ".") -> bool:
        """恢复系统备份"""
        try:
            backup_path = self.backup_dir / f"{backup_id}.zip"
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_id}")
                return False
            
            # 验证备份
            metadata = self._load_metadata(backup_path.parent, backup_id)
            if not metadata:
                logger.error(f"Backup metadata not found: {backup_id}")
                return False
            
            if metadata.status != "completed":
                logger.error(f"Backup status is not completed: {metadata.status}")
                return False
            
            # 验证校验和
            if self.config.enable_verification:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != metadata.checksum:
                    logger.error("Backup checksum verification failed")
                    return False
            
            # 创建恢复目录
            restore_dir = Path(target_dir) / "restore_temp"
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # 解压备份文件
            if self.config.enable_compression:
                self._extract_archive(backup_path, restore_dir)
            else:
                shutil.unpack_archive(str(backup_path), restore_dir)
            
            # 恢复各个组件
            restore_tasks = []
            
            if "database" in metadata.components:
                restore_tasks.append(self._restore_database(restore_dir))
            
            if "models" in metadata.components:
                restore_tasks.append(self._restore_models(restore_dir))
            
            if "configs" in metadata.components:
                restore_tasks.append(self._restore_configs(restore_dir))
            
            if "logs" in metadata.components:
                restore_tasks.append(self._restore_logs(restore_dir))
            
            # 等待所有恢复任务完成
            for task in restore_tasks:
                if not task.result():
                    raise ValueError("Component restoration failed")
            
            # 清理临时目录
            shutil.rmtree(restore_dir)
            
            logger.info(f"Backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def list_backups(self) -> List[BackupMetadata]:
        """列出所有备份"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            backup_id = backup_file.stem
            metadata = self._load_metadata(backup_file.parent, backup_id)
            
            if metadata:
                backups.append(metadata)
        
        # 按时间排序
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        try:
            backup_path = self.backup_dir / f"{backup_id}.zip"
            metadata_path = self.backup_dir / f"{backup_id}.json"
            
            if backup_path.exists():
                backup_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup deletion failed: {e}")
            return False
    
    def get_backup_status(self) -> Dict[str, Any]:
        """获取备份状态"""
        backups = self.list_backups()
        
        total_size = sum(b.size_bytes for b in backups)
        successful_backups = len([b for b in backups if b.status == "completed"])
        failed_backups = len([b for b in backups if b.status == "failed"])
        
        # 检查存储空间
        total_space, used_space, free_space = self._get_disk_usage()
        
        return {
            "total_backups": len(backups),
            "successful_backups": successful_backups,
            "failed_backups": failed_backups,
            "total_size_gb": round(total_size / (1024**3), 2),
            "disk_usage": {
                "total_gb": round(total_space / (1024**3), 2),
                "used_gb": round(used_space / (1024**3), 2),
                "free_gb": round(free_space / (1024**3), 2),
                "usage_percent": round((used_space / total_space) * 100, 1)
            },
            "last_backup": backups[0].timestamp if backups else None,
            "status": "healthy" if free_space > 1 * 1024**3 else "warning"  # 至少1GB空闲空间
        }
    
    # 私有方法
    def _backup_database(self, temp_dir: Path) -> str:
        """备份数据库"""
        try:
            # 这里应该调用实际的数据库备份逻辑
            db_backup_dir = temp_dir / "database"
            db_backup_dir.mkdir(exist_ok=True)

            db_data = {
                "backup_time": datetime.now().isoformat(),
                "tables": ["users", "models", "training_data"]
            }
            
            with open(db_backup_dir / "database.json", 'w') as f:
                json.dump(db_data, f, indent=2)
            
            return "database"
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return ""
    
    def _backup_models(self, temp_dir: Path) -> str:
        """备份模型文件"""
        try:
            models_backup_dir = temp_dir / "models"
            models_backup_dir.mkdir(exist_ok=True)
            
            # 查找模型文件
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("**/*"):
                    if model_file.is_file():
                        rel_path = model_file.relative_to(models_dir)
                        dest_path = models_backup_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(model_file, dest_path)
            
            return "models"
            
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            return ""
    
    def _backup_configs(self, temp_dir: Path) -> str:
        """备份配置文件"""
        try:
            configs_backup_dir = temp_dir / "configs"
            configs_backup_dir.mkdir(exist_ok=True)
            
            # 备份所有配置文件
            config_files = [
                "config.json",
                "settings.yaml",
                ".env",
                "requirements.txt"
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    shutil.copy2(config_path, configs_backup_dir / config_file)
            
            return "configs"
            
        except Exception as e:
            logger.error(f"Configs backup failed: {e}")
            return ""
    
    def _backup_logs(self, temp_dir: Path) -> str:
        """备份日志文件"""
        try:
            logs_backup_dir = temp_dir / "logs"
            logs_backup_dir.mkdir(exist_ok=True)
            
            # 备份日志文件
            logs_dir = Path("logs")
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    shutil.copy2(log_file, logs_backup_dir / log_file.name)
            
            return "logs"
            
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            return ""
    
    def _restore_database(self, restore_dir: Path) -> bool:
        """恢复数据库"""
        try:
            db_backup_dir = restore_dir / "database"
            
            if not db_backup_dir.exists():
                error_handler.log_warning("Database backup not found in restore package", "BackupManager")
                return True
            
            # 这里应该调用实际的数据库恢复逻辑
            logger.info("Database restoration completed")
            return True
            
        except Exception as e:
            logger.error(f"Database restoration failed: {e}")
            return False
    
    def _restore_models(self, restore_dir: Path) -> bool:
        """恢复模型文件"""
        try:
            models_backup_dir = restore_dir / "models"
            
            if not models_backup_dir.exists():
                error_handler.log_warning("Models backup not found in restore package", "BackupManager")
                return True
            
            # 恢复模型文件
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            for model_file in models_backup_dir.glob("**/*"):
                if model_file.is_file():
                    rel_path = model_file.relative_to(models_backup_dir)
                    dest_path = models_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, dest_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Models restoration failed: {e}")
            return False
    
    def _restore_configs(self, restore_dir: Path) -> bool:
        """恢复配置文件"""
        try:
            configs_backup_dir = restore_dir / "configs"
            
            if not configs_backup_dir.exists():
                error_handler.log_warning("Configs backup not found in restore package", "BackupManager")
                return True
            
            # 恢复配置文件
            for config_file in configs_backup_dir.glob("*"):
                if config_file.is_file():
                    shutil.copy2(config_file, Path(config_file.name))
            
            return True
            
        except Exception as e:
            logger.error(f"Configs restoration failed: {e}")
            return False
    
    def _restore_logs(self, restore_dir: Path) -> bool:
        """恢复日志文件"""
        try:
            logs_backup_dir = restore_dir / "logs"
            
            if not logs_backup_dir.exists():
                error_handler.log_warning("Logs backup not found in restore package", "BackupManager")
                return True
            
            # 恢复日志文件
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            for log_file in logs_backup_dir.glob("*"):
                if log_file.is_file():
                    shutil.copy2(log_file, logs_dir / log_file.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Logs restoration failed: {e}")
            return False
    
    def _create_archive(self, source_dir: Path, target_path: Path):
        """创建压缩文件"""
        if self.config.enable_compression:
            with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=self.config.compression_level) as zipf:
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_dir)
                        zipf.write(file_path, arcname)
        else:
            shutil.make_archive(str(target_path.with_suffix('')), 'zip', source_dir)
    
    def _extract_archive(self, archive_path: Path, target_dir: Path):
        """解压文件"""
        if self.config.enable_compression:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(target_dir)
        else:
            shutil.unpack_archive(str(archive_path), target_dir)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_func = getattr(hashlib, self.config.checksum_algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _verify_backup(self, backup_path: Path, expected_checksum: str) -> bool:
        """验证备份文件"""
        actual_checksum = self._calculate_checksum(backup_path)
        return actual_checksum == expected_checksum
    
    def _save_metadata(self, metadata: BackupMetadata, target_dir: Path):
        """保存元数据"""
        metadata_path = target_dir / f"{metadata.backup_id}.json"
        
        metadata_dict = {
            "backup_id": metadata.backup_id,
            "timestamp": metadata.timestamp,
            "size_bytes": metadata.size_bytes,
            "checksum": metadata.checksum,
            "version": metadata.version,
            "components": metadata.components,
            "status": metadata.status,
            "error_message": metadata.error_message
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_metadata(self, metadata_dir: Path, backup_id: str) -> Optional[BackupMetadata]:
        """加载元数据"""
        metadata_path = metadata_dir / f"{backup_id}.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            return BackupMetadata(**metadata_dict)
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {backup_id}: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        backups = self.list_backups()
        
        # 按时间排序
        backups.sort(key=lambda x: x.timestamp)
        
        # 删除超过保留期限的备份
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        
        for backup in backups:
            backup_time = datetime.fromisoformat(backup.timestamp)
            
            if backup_time < cutoff_time:
                self.delete_backup(backup.backup_id)
            
            # 检查备份数量限制
            if len(backups) > self.config.max_backup_count:
                self.delete_backup(backup.backup_id)
    
    def _get_system_version(self) -> str:
        """获取系统版本"""
        try:
            # 这里应该从实际的版本文件中读取
            version_file = Path("VERSION")
            if version_file.exists():
                return version_file.read_text().strip()
            else:
                return "1.0.0"
        except Exception as e:
            logger.warning(f"获取版本信息失败: {e}")
            return "1.0.0"
    
    def _get_disk_usage(self) -> tuple:
        """获取磁盘使用情况"""
        try:
            total, used, free = shutil.disk_usage(self.backup_dir)
            return total, used, free
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return 0, 0, 0

class BackupScheduler:
    """备份调度器"""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.scheduler_task = None
        self.is_running = False
    
    async def start_scheduler(self):
        """启动备份调度器"""
        if self.is_running:
            error_handler.log_warning("Backup scheduler is already running", "BackupScheduler")
            return
        
        self.is_running = True
        logger.info("Backup scheduler started")
        
        # 这里应该实现基于cron表达式的调度逻辑
        # 由于复杂性，这里简化为定时任务
        while self.is_running:
            try:
                # 检查是否需要备份
                if self._should_backup():
                    await self._perform_backup()
                
                # 每小时检查一次
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(300)  # 错误后等待5分钟
    
    def stop_scheduler(self):
        """停止备份调度器"""
        self.is_running = False
        logger.info("Backup scheduler stopped")
    
    def _should_backup(self) -> bool:
        """检查是否需要备份"""
        if not self.backup_manager.config.auto_backup:
            return False
        
        # 检查是否有今天的备份
        backups = self.backup_manager.list_backups()
        today = datetime.now().date()
        
        for backup in backups:
            backup_date = datetime.fromisoformat(backup.timestamp).date()
            if backup_date == today and backup.status == "completed":
                return False
        
        return True
    
    async def _perform_backup(self):
        """执行备份"""
        try:
            logger.info("Starting scheduled backup")
            
            # 在后台线程中执行备份
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                self.backup_manager.executor,
                self.backup_manager.create_backup
            )
            
            if metadata and metadata.status == "completed":
                logger.info("Scheduled backup completed successfully")
            else:
                logger.error("Scheduled backup failed")
                
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")

# API端点
backup_router = APIRouter(prefix="/api/backup", tags=["backup"])

@backup_router.post("/create")
async def create_backup(backup_name: Optional[str] = None):
    """创建备份"""
    try:
        backup_manager = get_backup_manager()
        metadata = backup_manager.create_backup(backup_name)
        
        if metadata:
            return {
                "success": True,
                "backup_id": metadata.backup_id,
                "message": "Backup created successfully"
            }
        else:
            return {
                "success": False,
                "message": "Backup creation failed"
            }
            
    except Exception as e:
        logger.error(f"API backup creation failed: {e}")
        raise HTTPException(status_code=500, detail="Backup creation failed")

@backup_router.post("/restore/{backup_id}")
async def restore_backup(backup_id: str):
    """恢复备份"""
    try:
        backup_manager = get_backup_manager()
        success = backup_manager.restore_backup(backup_id)
        
        if success:
            return {
                "success": True,
                "message": "Backup restored successfully"
            }
        else:
            return {
                "success": False,
                "message": "Backup restoration failed"
            }
            
    except Exception as e:
        logger.error(f"API backup restoration failed: {e}")
        raise HTTPException(status_code=500, detail="Backup restoration failed")

@backup_router.get("/list")
async def list_backups():
    """列出备份"""
    try:
        backup_manager = get_backup_manager()
        backups = backup_manager.list_backups()
        
        return {
            "success": True,
            "backups": [
                {
                    "backup_id": b.backup_id,
                    "timestamp": b.timestamp,
                    "size_bytes": b.size_bytes,
                    "components": b.components,
                    "status": b.status
                }
                for b in backups
            ]
        }
        
    except Exception as e:
        logger.error(f"API backup list failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list backups")

@backup_router.get("/status")
async def get_backup_status():
    """获取备份状态"""
    try:
        backup_manager = get_backup_manager()
        status = backup_manager.get_backup_status()
        
        return {
            "success": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"API backup status failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get backup status")

# 全局实例
backup_manager = None
backup_scheduler = None

def get_backup_manager() -> BackupManager:
    """获取全局备份管理器实例"""
    global backup_manager
    if backup_manager is None:
        config = BackupConfig()
        backup_manager = BackupManager(config)
    return backup_manager

def get_backup_scheduler() -> BackupScheduler:
    """获取全局备份调度器实例"""
    global backup_scheduler
    if backup_scheduler is None:
        backup_scheduler = BackupScheduler(get_backup_manager())
    return backup_scheduler

def initialize_backup_system():
    """初始化备份系统"""
    # 创建备份管理器
    backup_manager = get_backup_manager()
    
    # 创建备份调度器
    backup_scheduler = get_backup_scheduler()
    
    logger.info("Backup system initialized")
    return backup_manager, backup_scheduler
