"""
数据库备份管理器
Database Backup Manager

提供数据库备份和恢复功能，支持自动定期备份和加密
Provides database backup and recovery functionality with automatic scheduled backups and encryption
"""

import os
import shutil
import logging
import schedule
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
import zipfile
import hashlib

logger = logging.getLogger("DatabaseBackupManager")

class DatabaseBackupManager:
    """数据库备份管理器"""
    
    def __init__(self, db_path: str, backup_dir: str = "backups", 
                 retention_days: int = 30, backup_interval_hours: int = 24):
        """
        初始化备份管理器
        
        Args:
            db_path: 数据库文件路径
            backup_dir: 备份目录
            retention_days: 备份保留天数
            backup_interval_hours: 备份间隔小时数
        """
        self.db_path = os.path.abspath(db_path)
        self.backup_dir = os.path.abspath(backup_dir)
        self.retention_days = retention_days
        self.backup_interval_hours = backup_interval_hours
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 备份线程控制
        self.backup_thread = None
        self.running = False
        
        logger.info(f"Database backup manager initialized for {self.db_path}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Retention: {retention_days} days, Interval: {backup_interval_hours} hours")
    
    def create_backup(self, backup_name: str = None, encrypt: bool = False) -> str:
        """
        创建数据库备份
        
        Args:
            backup_name: 备份名称（可选）
            encrypt: 是否加密备份文件
            
        Returns:
            备份文件路径
        """
        try:
            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if backup_name:
                base_name = f"{backup_name}_{timestamp}"
            else:
                db_name = os.path.basename(self.db_path).replace('.db', '')
                base_name = f"{db_name}_backup_{timestamp}"
            
            backup_file = os.path.join(self.backup_dir, f"{base_name}.db")
            
            logger.info(f"Creating backup of {self.db_path} to {backup_file}")
            
            # 使用SQLite备份API创建一致的备份
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_file)
            
            try:
                # 使用SQLite在线备份API
                source_conn.backup(backup_conn)
            finally:
                backup_conn.close()
                source_conn.close()
            
            # 计算文件哈希值
            file_hash = self._calculate_file_hash(backup_file)
            hash_file = backup_file + ".sha256"
            with open(hash_file, 'w') as f:
                f.write(file_hash)
            
            # 创建元数据文件
            metadata = {
                "original_db": self.db_path,
                "backup_time": timestamp,
                "backup_file": os.path.basename(backup_file),
                "file_size": os.path.getsize(backup_file),
                "file_hash": file_hash,
                "encrypted": encrypt,
                "retention_days": self.retention_days
            }
            
            metadata_file = backup_file.replace('.db', '.meta.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # 如果需要加密
            if encrypt:
                encrypted_file = self._encrypt_backup(backup_file)
                logger.info(f"Backup encrypted: {encrypted_file}")
                return encrypted_file
            
            logger.info(f"Backup created successfully: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _encrypt_backup(self, backup_file: str) -> str:
        """
        加密备份文件
        
        注意：这是一个简化实现。实际生产环境应使用更强的加密方法。
        """
        try:
            # 尝试导入安全模块进行加密
            from core.security import security_manager
            encrypted_file = backup_file + ".encrypted"
            
            with open(backup_file, 'rb') as f:
                data = f.read()
            
            # 使用安全管理器加密（简化）
            encrypted_data = security_manager.encrypt_data(data.decode('utf-8', errors='ignore'))
            if isinstance(encrypted_data, dict) and 'encrypted_data' in encrypted_data:
                encrypted_str = encrypted_data['encrypted_data']
            else:
                encrypted_str = str(encrypted_data)
            
            with open(encrypted_file, 'w') as f:
                f.write(encrypted_str)
            
            # 删除原始未加密文件
            os.remove(backup_file)
            
            return encrypted_file
            
        except Exception as e:
            logger.warning(f"Backup encryption failed: {e}")
            return backup_file
    
    def restore_backup(self, backup_file: str, target_db_path: str = None, decrypt: bool = False) -> bool:
        """
        从备份恢复数据库
        
        Args:
            backup_file: 备份文件路径
            target_db_path: 目标数据库路径（默认使用原路径）
            decrypt: 是否需要解密
            
        Returns:
            是否恢复成功
        """
        try:
            if target_db_path is None:
                target_db_path = self.db_path
            
            # 如果需要解密
            if decrypt:
                decrypted_file = self._decrypt_backup(backup_file)
                if decrypted_file:
                    backup_file = decrypted_file
                else:
                    logger.error("Failed to decrypt backup file")
                    return False
            
            # 验证备份文件完整性
            if not self._verify_backup_integrity(backup_file):
                logger.error("Backup file integrity check failed")
                return False
            
            logger.info(f"Restoring database from {backup_file} to {target_db_path}")
            
            # 备份当前数据库（如果存在）
            if os.path.exists(target_db_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_backup = f"{target_db_path}.pre_restore_{timestamp}.bak"
                shutil.copy2(target_db_path, pre_restore_backup)
                logger.info(f"Created pre-restore backup: {pre_restore_backup}")
            
            # 执行恢复
            shutil.copy2(backup_file, target_db_path)
            
            # 验证恢复后的数据库
            if self._verify_database_integrity(target_db_path):
                logger.info("Database restored successfully")
                return True
            else:
                logger.error("Restored database integrity check failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _decrypt_backup(self, encrypted_file: str) -> Optional[str]:
        """解密备份文件"""
        try:
            from core.security import security_manager
            
            decrypted_file = encrypted_file.replace('.encrypted', '')
            
            with open(encrypted_file, 'r') as f:
                encrypted_data = f.read()
            
            # 使用安全管理器解密
            decrypted_data = security_manager.decrypt_data(encrypted_data, "")
            if isinstance(decrypted_data, dict) and 'decrypted_data' in decrypted_data:
                decrypted_str = decrypted_data['decrypted_data']
            else:
                decrypted_str = str(decrypted_data)
            
            with open(decrypted_file, 'w') as f:
                f.write(decrypted_str)
            
            return decrypted_file
            
        except Exception as e:
            logger.error(f"Failed to decrypt backup: {e}")
            return None
    
    def _verify_backup_integrity(self, backup_file: str) -> bool:
        """验证备份文件完整性"""
        try:
            # 检查哈希文件
            hash_file = backup_file + ".sha256"
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    stored_hash = f.read().strip()
                
                current_hash = self._calculate_file_hash(backup_file)
                return stored_hash == current_hash
            
            # 如果没有哈希文件，尝试连接数据库验证
            conn = sqlite3.connect(backup_file)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            return True
            
        except Exception as e:
            logger.warning(f"Backup integrity check failed: {e}")
            return False
    
    def _verify_database_integrity(self, db_path: str) -> bool:
        """验证数据库完整性"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 执行完整性检查
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            
            if result and result[0] == "ok":
                return True
            else:
                logger.warning(f"Database integrity check failed: {result}")
                return False
                
        except Exception as e:
            logger.warning(f"Database integrity check error: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """
        清理过期备份
        
        Returns:
            删除的备份文件数量
        """
        try:
            deleted_count = 0
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            for filename in os.listdir(self.backup_dir):
                file_path = os.path.join(self.backup_dir, filename)
                
                # 检查文件修改时间
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted old backup: {filename}")
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete old backup {filename}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份"""
        backups = []
        try:
            for filename in os.listdir(self.backup_dir):
                if filename.endswith('.db') or filename.endswith('.encrypted'):
                    file_path = os.path.join(self.backup_dir, filename)
                    metadata_file = file_path.replace('.db', '.meta.json').replace('.encrypted', '.meta.json')
                    
                    backup_info = {
                        "filename": filename,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        "encrypted": filename.endswith('.encrypted')
                    }
                    
                    # 加载元数据（如果存在）
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            backup_info["metadata"] = metadata
                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {filename}: {e}")
                    
                    backups.append(backup_info)
            
            # 按修改时间排序（最新的在前）
            backups.sort(key=lambda x: x["modified"], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def start_auto_backup(self):
        """启动自动备份线程"""
        if self.running:
            logger.warning("Auto backup already running")
            return
        
        self.running = True
        self.backup_thread = threading.Thread(target=self._auto_backup_worker, daemon=True)
        self.backup_thread.start()
        logger.info("Auto backup started")
    
    def stop_auto_backup(self):
        """停止自动备份线程"""
        self.running = False
        if self.backup_thread:
            self.backup_thread.join(timeout=10)
        logger.info("Auto backup stopped")
    
    def _auto_backup_worker(self):
        """自动备份工作线程"""
        # 立即创建第一个备份
        try:
            self.create_backup()
        except Exception as e:
            logger.error(f"Initial auto-backup failed: {e}")
        
        # 设置定时备份
        schedule.every(self.backup_interval_hours).hours.do(
            lambda: self._safe_create_backup()
        )
        
        # 每天清理旧备份
        schedule.every().day.do(
            lambda: self.cleanup_old_backups()
        )
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    def _safe_create_backup(self):
        """安全创建备份（捕获异常）"""
        try:
            logger.info("Starting scheduled backup...")
            self.create_backup()
            logger.info("Scheduled backup completed")
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")

# 全局备份管理器实例
_backup_manager_instance = None

def get_backup_manager(db_path: str = None, backup_dir: str = "backups") -> DatabaseBackupManager:
    """获取全局备份管理器实例"""
    global _backup_manager_instance
    
    if _backup_manager_instance is None:
        if db_path is None:
            # 尝试从环境变量获取数据库路径
            db_path = os.getenv("DB_PATH", "self_soul.db")
        
        _backup_manager_instance = DatabaseBackupManager(db_path, backup_dir)
    
    return _backup_manager_instance

def initialize_backup_system(db_path: str = None, backup_dir: str = "backups", 
                            start_auto_backup: bool = True) -> DatabaseBackupManager:
    """初始化备份系统"""
    manager = get_backup_manager(db_path, backup_dir)
    
    if start_auto_backup:
        manager.start_auto_backup()
        logger.info("Backup system initialized with auto backup")
    else:
        logger.info("Backup system initialized (manual mode)")
    
    return manager