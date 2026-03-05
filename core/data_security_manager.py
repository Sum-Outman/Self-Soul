"""
数据安全管理器 - 增强数据安全和加密功能

主要功能：
1. AES-256加密存储，敏感配置加密存储
2. 数据传输签名校验
3. 数据访问记录审计日志
4. 密钥管理和轮换

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
import logging
import time
import hashlib
import hmac
import base64
import secrets
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from core.error_handling import error_handler

# 日志记录器
logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES_256_CBC = "aes-256-cbc"
    AES_256_GCM = "aes-256-gcm"
    FERNET = "fernet"
    RSA_OAEP = "rsa-oaep"


class DataCategory(Enum):
    """数据类别枚举"""
    CONFIGURATION = "configuration"  # 配置数据
    TRAINING_DATA = "training_data"  # 训练数据
    MODEL_WEIGHTS = "model_weights"  # 模型权重
    USER_DATA = "user_data"  # 用户数据
    AUDIT_LOG = "audit_log"  # 审计日志
    API_KEY = "api_key"  # API密钥
    DATABASE = "database"  # 数据库内容


@dataclass
class EncryptionKey:
    """加密密钥定义"""
    key_id: str
    key_type: EncryptionAlgorithm
    key_data: bytes
    created_at: float
    expires_at: Optional[float] = None
    description: str = ""
    is_active: bool = True
    version: int = 1


@dataclass
class EncryptedData:
    """加密数据定义"""
    data_id: str
    data_category: DataCategory
    encrypted_content: bytes
    encryption_algorithm: EncryptionAlgorithm
    key_id: str
    iv_or_nonce: Optional[bytes] = None  # 初始化向量或Nonce
    hmac_signature: Optional[bytes] = None  # HMAC签名
    encrypted_at: float = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataAccessAudit:
    """数据访问审计日志"""
    audit_id: str
    timestamp: float
    user_id: str
    data_id: str
    data_category: DataCategory
    action: str  # read, write, delete, update
    success: bool
    ip_address: str = ""
    user_agent: str = ""
    details: Optional[Dict[str, Any]] = None


class DataSecurityManager:
    """数据安全管理器
    
    主要功能：
    1. 数据加密存储（AES-256等）
    2. 数据传输签名和验证
    3. 数据访问审计
    4. 密钥管理和轮换
    """
    
    def __init__(self, data_dir: str = None, master_key: str = None):
        """初始化数据安全管理器
        
        Args:
            data_dir: 数据存储目录
            master_key: 主密钥（用于派生其他密钥）
        """
        # 设置数据目录
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "data", "encrypted_data")
        else:
            self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 密钥管理
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = master_key or os.getenv("DATA_SECURITY_MASTER_KEY", secrets.token_hex(32))
        
        # 数据存储
        self.encrypted_data: Dict[str, EncryptedData] = {}
        
        # 审计日志
        self.audit_logs: List[DataAccessAudit] = []
        
        # 互斥锁
        self.lock = threading.RLock()
        
        # 配置
        self.config = {
            'max_audit_logs': 10000,
            'key_rotation_days': 90,  # 密钥轮换天数
            'default_algorithm': EncryptionAlgorithm.AES_256_GCM.value,
            'enable_audit_logging': True,
            'enable_hmac_verification': True,
            'auto_save_interval': 300,  # 自动保存间隔（秒）
        }
        
        # 初始化密钥
        self._initialize_keys()
        
        # 加载现有数据
        self._load_data()
        
        # 启动自动保存线程
        self._start_auto_save()
        
        logger.info(f"数据安全管理器初始化完成，已加载 {len(self.keys)} 个密钥和 {len(self.encrypted_data)} 个加密数据项")
    
    def _initialize_keys(self):
        """初始化加密密钥"""
        current_time = time.time()
        
        # 生成主密钥派生的密钥
        salt = b"self_soul_data_security_salt"
        
        # 1. AES-256-GCM 密钥
        kdf_gcm = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256位
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        aes_key = kdf_gcm.derive(f"{self.master_key}_aes_gcm".encode())
        
        aes_key_obj = EncryptionKey(
            key_id="key_aes_256_gcm_1",
            key_type=EncryptionAlgorithm.AES_256_GCM,
            key_data=aes_key,
            created_at=current_time,
            expires_at=current_time + (self.config['key_rotation_days'] * 24 * 3600),
            description="AES-256-GCM 加密密钥",
            is_active=True,
            version=1
        )
        self.keys[aes_key_obj.key_id] = aes_key_obj
        
        # 2. AES-256-CBC 密钥
        kdf_cbc = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        aes_cbc_key = kdf_cbc.derive(f"{self.master_key}_aes_cbc".encode())
        
        aes_cbc_key_obj = EncryptionKey(
            key_id="key_aes_256_cbc_1",
            key_type=EncryptionAlgorithm.AES_256_CBC,
            key_data=aes_cbc_key,
            created_at=current_time,
            expires_at=current_time + (self.config['key_rotation_days'] * 24 * 3600),
            description="AES-256-CBC 加密密钥",
            is_active=True,
            version=1
        )
        self.keys[aes_cbc_key_obj.key_id] = aes_cbc_key_obj
        
        # 3. Fernet 密钥
        fernet_key = Fernet.generate_key()
        
        fernet_key_obj = EncryptionKey(
            key_id="key_fernet_1",
            key_type=EncryptionAlgorithm.FERNET,
            key_data=fernet_key,
            created_at=current_time,
            expires_at=current_time + (self.config['key_rotation_days'] * 24 * 3600),
            description="Fernet 加密密钥",
            is_active=True,
            version=1
        )
        self.keys[fernet_key_obj.key_id] = fernet_key_obj
        
        # 4. HMAC 签名密钥
        kdf_hmac = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        hmac_key = kdf_hmac.derive(f"{self.master_key}_hmac".encode())
        
        hmac_key_obj = EncryptionKey(
            key_id="key_hmac_1",
            key_type=EncryptionAlgorithm.AES_256_GCM,  # 复用，HMAC不需要特定算法
            key_data=hmac_key,
            created_at=current_time,
            expires_at=current_time + (self.config['key_rotation_days'] * 24 * 3600),
            description="HMAC 签名密钥",
            is_active=True,
            version=1
        )
        self.keys[hmac_key_obj.key_id] = hmac_key_obj
        
        logger.info(f"初始化了 {len(self.keys)} 个加密密钥")
    
    def _load_data(self):
        """加载保存的数据"""
        data_files = {
            'keys': 'encryption_keys.json',
            'encrypted_data': 'encrypted_data.pkl',
            'audit_logs': 'data_access_audit.pkl',
        }
        
        for data_key, filename in data_files.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    if filename.endswith('.json'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    else:  # .pkl files
                        import pickle
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                    
                    # 特殊处理：将字典转换为对象
                    if data_key == 'keys' and data:
                        keys_dict = {}
                        for key_id, key_data in data.items():
                            if isinstance(key_data, dict):
                                key_data['key_type'] = EncryptionAlgorithm(key_data['key_type'])
                                keys_dict[key_id] = EncryptionKey(**key_data)
                        data = keys_dict
                    
                    elif data_key == 'encrypted_data' and data:
                        encrypted_dict = {}
                        for data_id, enc_data in data.items():
                            if isinstance(enc_data, dict):
                                enc_data['data_category'] = DataCategory(enc_data['data_category'])
                                enc_data['encryption_algorithm'] = EncryptionAlgorithm(enc_data['encryption_algorithm'])
                                encrypted_dict[data_id] = EncryptedData(**enc_data)
                        data = encrypted_dict
                    
                    setattr(self, data_key, data)
                    logger.info(f"加载 {data_key}: {len(data) if isinstance(data, dict) else 'N/A'}")
            except Exception as e:
                error_handler.handle_error(e, "DataSecurityManager", 
                                          f"加载 {filename} 失败，使用默认值")
    
    def _save_data(self):
        """保存数据到文件"""
        with self.lock:
            data_files = {
                'keys': ('encryption_keys.json', 'json'),
                'encrypted_data': ('encrypted_data.pkl', 'pkl'),
                'audit_logs': ('data_access_audit.pkl', 'pkl'),
            }
            
            for data_key, (filename, format_type) in data_files.items():
                filepath = os.path.join(self.data_dir, filename)
                try:
                    data = getattr(self, data_key)
                    
                    # 特殊处理：将对象转换为字典
                    if data_key == 'keys' and data:
                        keys_dict = {}
                        for key_id, key_obj in data.items():
                            key_dict = asdict(key_obj)
                            key_dict['key_type'] = key_obj.key_type.value
                            keys_dict[key_id] = key_dict
                        data = keys_dict
                    
                    elif data_key == 'encrypted_data' and data:
                        encrypted_dict = {}
                        for data_id, enc_obj in data.items():
                            enc_dict = asdict(enc_obj)
                            enc_dict['data_category'] = enc_obj.data_category.value
                            enc_dict['encryption_algorithm'] = enc_obj.encryption_algorithm.value
                            encrypted_dict[data_id] = enc_dict
                        data = encrypted_dict
                    
                    # 限制审计日志数量
                    if data_key == 'audit_logs':
                        if len(data) > self.config['max_audit_logs']:
                            data = data[-self.config['max_audit_logs']:]
                    
                    if format_type == 'json':
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    else:  # pkl
                        import pickle
                        with open(filepath, 'wb') as f:
                            pickle.dump(data, f)
                            
                except Exception as e:
                    error_handler.handle_error(e, "DataSecurityManager", 
                                              f"保存 {filename} 失败")
    
    def _start_auto_save(self):
        """启动自动保存线程"""
        def auto_save_worker():
            while True:
                time.sleep(self.config['auto_save_interval'])
                try:
                    self._save_data()
                except Exception as e:
                    error_handler.handle_error(e, "DataSecurityManager", "自动保存失败")
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info(f"启动自动保存线程，间隔 {self.config['auto_save_interval']} 秒")
    
    def _log_audit(self, user_id: str, data_id: str, data_category: DataCategory,
                  action: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """记录数据访问审计日志"""
        if not self.config['enable_audit_logging']:
            return
        
        audit_entry = DataAccessAudit(
            audit_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            data_id=data_id,
            data_category=data_category,
            action=action,
            success=success,
            details=details or {}
        )
        
        with self.lock:
            self.audit_logs.append(audit_entry)
            
            # 限制日志数量
            if len(self.audit_logs) > self.config['max_audit_logs']:
                self.audit_logs = self.audit_logs[-self.config['max_audit_logs']:]
        
        logger.info(f"数据访问审计: {user_id} {action} {data_category.value}:{data_id} - {success}")
    
    def _get_active_key(self, algorithm: EncryptionAlgorithm) -> Optional[EncryptionKey]:
        """获取活动的加密密钥"""
        current_time = time.time()
        
        for key in self.keys.values():
            if (key.key_type == algorithm and 
                key.is_active and 
                (key.expires_at is None or key.expires_at > current_time)):
                return key
        
        return None
    
    def _generate_hmac(self, data: bytes, key_id: str = "key_hmac_1") -> bytes:
        """生成HMAC签名"""
        if not self.config['enable_hmac_verification']:
            return b""
        
        if key_id not in self.keys:
            error_handler.log_warning(f"HMAC密钥 {key_id} 不存在", "DataSecurityManager")
            return b""
        
        key = self.keys[key_id]
        h = hmac.new(key.key_data, data, hashlib.sha256)
        return h.digest()
    
    def _verify_hmac(self, data: bytes, signature: bytes, key_id: str = "key_hmac_1") -> bool:
        """验证HMAC签名"""
        if not self.config['enable_hmac_verification']:
            return True
        
        if key_id not in self.keys:
            error_handler.log_warning(f"HMAC密钥 {key_id} 不存在", "DataSecurityManager")
            return False
        
        key = self.keys[key_id]
        expected_signature = hmac.new(key.key_data, data, hashlib.sha256).digest()
        
        # 使用恒定时间比较防止时序攻击
        return hmac.compare_digest(expected_signature, signature)
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], 
                    data_category: DataCategory,
                    key_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    user_id: str = "system") -> Optional[str]:
        """加密数据
        
        Args:
            data: 要加密的数据（字符串、字节或字典）
            data_category: 数据类别
            key_id: 密钥ID，None表示使用默认密钥
            metadata: 元数据
            user_id: 执行加密的用户ID
            
        Returns:
            加密数据ID，失败返回None
        """
        try:
            with self.lock:
                # 确定密钥
                if key_id is None:
                    # 使用默认算法
                    default_algorithm = EncryptionAlgorithm(self.config['default_algorithm'])
                    key = self._get_active_key(default_algorithm)
                    if key is None:
                        error_handler.log_warning("没有活动的加密密钥", "DataSecurityManager")
                        return None
                    key_id = key.key_id
                
                if key_id not in self.keys:
                    error_handler.log_warning(f"密钥 {key_id} 不存在", "DataSecurityManager")
                    return None
                
                key_obj = self.keys[key_id]
                algorithm = key_obj.key_type
                
                # 准备数据
                if isinstance(data, dict):
                    data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
                elif isinstance(data, str):
                    data_bytes = data.encode('utf-8')
                else:
                    data_bytes = data
                
                # 根据算法加密
                encrypted_content = b""
                iv_or_nonce = None
                
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    # AES-256-GCM加密
                    iv_or_nonce = os.urandom(12)  # GCM推荐12字节nonce
                    cipher = Cipher(
                        algorithms.AES(key_obj.key_data),
                        modes.GCM(iv_or_nonce),
                        backend=default_backend()
                    )
                    encryptor = cipher.encryptor()
                    encrypted_content = encryptor.update(data_bytes) + encryptor.finalize()
                    
                    # 获取认证标签
                    auth_tag = encryptor.tag
                    encrypted_content += auth_tag  # 将认证标签附加到密文
                
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    # AES-256-CBC加密
                    iv_or_nonce = os.urandom(16)  # CBC需要16字节IV
                    cipher = Cipher(
                        algorithms.AES(key_obj.key_data),
                        modes.CBC(iv_or_nonce),
                        backend=default_backend()
                    )
                    encryptor = cipher.encryptor()
                    
                    # 填充数据
                    padder = padding.PKCS7(128).padder()
                    padded_data = padder.update(data_bytes) + padder.finalize()
                    
                    encrypted_content = encryptor.update(padded_data) + encryptor.finalize()
                
                elif algorithm == EncryptionAlgorithm.FERNET:
                    # Fernet加密
                    fernet = Fernet(key_obj.key_data)
                    encrypted_content = fernet.encrypt(data_bytes)
                
                else:
                    error_handler.log_warning(f"不支持的加密算法: {algorithm}", "DataSecurityManager")
                    return None
                
                # 生成HMAC签名
                hmac_signature = None
                if self.config['enable_hmac_verification']:
                    # 对原始数据签名（不是密文）
                    hmac_signature = self._generate_hmac(data_bytes)
                
                # 创建加密数据对象
                data_id = f"enc_data_{hashlib.md5(f'{time.time()}{secrets.token_hex(8)}'.encode()).hexdigest()[:12]}"
                
                encrypted_data = EncryptedData(
                    data_id=data_id,
                    data_category=data_category,
                    encrypted_content=encrypted_content,
                    encryption_algorithm=algorithm,
                    key_id=key_id,
                    iv_or_nonce=iv_or_nonce,
                    hmac_signature=hmac_signature,
                    encrypted_at=time.time(),
                    metadata=metadata or {}
                )
                
                # 保存
                self.encrypted_data[data_id] = encrypted_data
                
                # 记录审计日志
                self._log_audit(
                    user_id=user_id,
                    data_id=data_id,
                    data_category=data_category,
                    action="encrypt",
                    success=True,
                    details={
                        "algorithm": algorithm.value,
                        "key_id": key_id,
                        "data_size_bytes": len(data_bytes)
                    }
                )
                
                return data_id
                
        except Exception as e:
            error_handler.handle_error(e, "DataSecurityManager", "加密数据失败")
            
            # 记录失败的审计日志
            self._log_audit(
                user_id=user_id,
                data_id="unknown",
                data_category=data_category,
                action="encrypt",
                success=False,
                details={"error": str(e)}
            )
            
            return None
    
    def decrypt_data(self, data_id: str, user_id: str = "system") -> Optional[Union[str, bytes, Dict[str, Any]]]:
        """解密数据
        
        Args:
            data_id: 加密数据ID
            user_id: 执行解密的用户ID
            
        Returns:
            解密后的数据，失败返回None
        """
        try:
            with self.lock:
                if data_id not in self.encrypted_data:
                    error_handler.log_warning(f"加密数据 {data_id} 不存在", "DataSecurityManager")
                    
                    # 记录失败的审计日志
                    self._log_audit(
                        user_id=user_id,
                        data_id=data_id,
                        data_category=DataCategory.CONFIGURATION,  # 默认类别
                        action="decrypt",
                        success=False,
                        details={"error": "数据不存在"}
                    )
                    
                    return None
                
                enc_data = self.encrypted_data[data_id]
                
                if enc_data.key_id not in self.keys:
                    error_handler.log_warning(f"密钥 {enc_data.key_id} 不存在", "DataSecurityManager")
                    
                    self._log_audit(
                        user_id=user_id,
                        data_id=data_id,
                        data_category=enc_data.data_category,
                        action="decrypt",
                        success=False,
                        details={"error": "密钥不存在"}
                    )
                    
                    return None
                
                key_obj = self.keys[enc_data.key_id]
                algorithm = enc_data.encryption_algorithm
                
                # 解密数据
                decrypted_bytes = b""
                
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    # AES-256-GCM解密
                    ciphertext = enc_data.encrypted_content
                    if len(ciphertext) < 16:  # 最小长度检查
                        error_handler.log_warning("密文太短", "DataSecurityManager")
                        return None
                    
                    # 分离认证标签（最后16字节）
                    auth_tag = ciphertext[-16:]
                    ciphertext = ciphertext[:-16]
                    
                    cipher = Cipher(
                        algorithms.AES(key_obj.key_data),
                        modes.GCM(enc_data.iv_or_nonce, auth_tag),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
                
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    # AES-256-CBC解密
                    cipher = Cipher(
                        algorithms.AES(key_obj.key_data),
                        modes.CBC(enc_data.iv_or_nonce),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    padded_data = decryptor.update(enc_data.encrypted_content) + decryptor.finalize()
                    
                    # 去除填充
                    unpadder = padding.PKCS7(128).unpadder()
                    decrypted_bytes = unpadder.update(padded_data) + unpadder.finalize()
                
                elif algorithm == EncryptionAlgorithm.FERNET:
                    # Fernet解密
                    fernet = Fernet(key_obj.key_data)
                    decrypted_bytes = fernet.decrypt(enc_data.encrypted_content)
                
                else:
                    error_handler.log_warning(f"不支持的加密算法: {algorithm}", "DataSecurityManager")
                    return None
                
                # 验证HMAC签名
                if (self.config['enable_hmac_verification'] and 
                    enc_data.hmac_signature and 
                    len(enc_data.hmac_signature) > 0):
                    
                    if not self._verify_hmac(decrypted_bytes, enc_data.hmac_signature):
                        error_handler.log_warning(f"HMAC验证失败: {data_id}", "DataSecurityManager")
                        
                        self._log_audit(
                            user_id=user_id,
                            data_id=data_id,
                            data_category=enc_data.data_category,
                            action="decrypt",
                            success=False,
                            details={"error": "HMAC验证失败"}
                        )
                        
                        return None
                
                # 尝试解析为JSON，否则返回字符串
                try:
                    # 尝试解码为UTF-8字符串
                    decrypted_str = decrypted_bytes.decode('utf-8')
                    
                    # 尝试解析为JSON
                    try:
                        result = json.loads(decrypted_str)
                    except json.JSONDecodeError:
                        result = decrypted_str
                except UnicodeDecodeError:
                    # 如果不是文本，返回字节
                    result = decrypted_bytes
                
                # 记录成功的审计日志
                self._log_audit(
                    user_id=user_id,
                    data_id=data_id,
                    data_category=enc_data.data_category,
                    action="decrypt",
                    success=True,
                    details={
                        "algorithm": algorithm.value,
                        "key_id": enc_data.key_id,
                        "data_size_bytes": len(decrypted_bytes)
                    }
                )
                
                return result
                
        except Exception as e:
            error_handler.handle_error(e, "DataSecurityManager", f"解密数据失败: {data_id}")
            
            # 记录失败的审计日志
            data_category = DataCategory.CONFIGURATION
            if data_id in self.encrypted_data:
                data_category = self.encrypted_data[data_id].data_category
            
            self._log_audit(
                user_id=user_id,
                data_id=data_id,
                data_category=data_category,
                action="decrypt",
                success=False,
                details={"error": str(e)}
            )
            
            return None
    
    def encrypt_config_value(self, config_key: str, config_value: Union[str, Dict[str, Any]], 
                           user_id: str = "system") -> Optional[str]:
        """加密配置值
        
        Args:
            config_key: 配置键名
            config_value: 配置值
            user_id: 执行加密的用户ID
            
        Returns:
            加密数据ID，失败返回None
        """
        metadata = {
            "config_key": config_key,
            "original_type": type(config_value).__name__
        }
        
        return self.encrypt_data(
            data=config_value,
            data_category=DataCategory.CONFIGURATION,
            metadata=metadata,
            user_id=user_id
        )
    
    def decrypt_config_value(self, data_id: str, user_id: str = "system") -> Optional[Union[str, Dict[str, Any]]]:
        """解密配置值
        
        Args:
            data_id: 加密数据ID
            user_id: 执行解密的用户ID
            
        Returns:
            解密后的配置值，失败返回None
        """
        return self.decrypt_data(data_id, user_id)
    
    def encrypt_training_data(self, training_data: Dict[str, Any], 
                            user_id: str = "system") -> Optional[str]:
        """加密训练数据
        
        Args:
            training_data: 训练数据
            user_id: 执行加密的用户ID
            
        Returns:
            加密数据ID，失败返回None
        """
        metadata = {
            "data_type": "training_data",
            "record_count": len(training_data.get("records", [])) if isinstance(training_data, dict) else 0
        }
        
        return self.encrypt_data(
            data=training_data,
            data_category=DataCategory.TRAINING_DATA,
            metadata=metadata,
            user_id=user_id
        )
    
    def encrypt_api_key(self, api_key: str, service_name: str, 
                       user_id: str = "system") -> Optional[str]:
        """加密API密钥
        
        Args:
            api_key: API密钥
            service_name: 服务名称
            user_id: 执行加密的用户ID
            
        Returns:
            加密数据ID，失败返回None
        """
        metadata = {
            "service_name": service_name,
            "encrypted_at_iso": datetime.utcnow().isoformat()
        }
        
        return self.encrypt_data(
            data=api_key,
            data_category=DataCategory.API_KEY,
            metadata=metadata,
            user_id=user_id
        )
    
    def generate_secure_transfer_token(self, data: Union[str, bytes, Dict[str, Any]], 
                                      expires_in_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """生成安全传输令牌（用于数据传输）
        
        Args:
            data: 要传输的数据
            expires_in_seconds: 令牌过期时间（秒）
            
        Returns:
            安全令牌字典，失败返回None
        """
        try:
            current_time = time.time()
            expiry_time = current_time + expires_in_seconds
            
            # 创建令牌负载
            payload = {
                "data": data if isinstance(data, (str, dict)) else base64.b64encode(data).decode('utf-8'),
                "exp": expiry_time,
                "iat": current_time,
                "jti": secrets.token_hex(16)
            }
            
            # 使用HMAC密钥签名
            hmac_key = self.keys.get("key_hmac_1")
            if not hmac_key:
                error_handler.log_warning("HMAC密钥不存在", "DataSecurityManager")
                return None
            
            # 创建签名
            payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
            signature = self._generate_hmac(payload_str.encode('utf-8'))
            
            return {
                "payload": payload,
                "signature": base64.b64encode(signature).decode('utf-8'),
                "algorithm": "HMAC-SHA256"
            }
            
        except Exception as e:
            error_handler.handle_error(e, "DataSecurityManager", "生成安全传输令牌失败")
            return None
    
    def verify_secure_transfer_token(self, token: Dict[str, Any]) -> Optional[Union[str, bytes, Dict[str, Any]]]:
        """验证安全传输令牌
        
        Args:
            token: 安全令牌字典
            
        Returns:
            验证通过的数据，失败返回None
        """
        try:
            # 检查令牌结构
            if "payload" not in token or "signature" not in token:
                error_handler.log_warning("令牌结构无效", "DataSecurityManager")
                return None
            
            payload = token["payload"]
            signature = base64.b64decode(token["signature"])
            
            # 验证签名
            payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
            if not self._verify_hmac(payload_str.encode('utf-8'), signature):
                error_handler.log_warning("令牌签名无效", "DataSecurityManager")
                return None
            
            # 检查过期时间
            current_time = time.time()
            if payload.get("exp", 0) < current_time:
                error_handler.log_warning("令牌已过期", "DataSecurityManager")
                return None
            
            # 提取数据
            data = payload.get("data")
            if isinstance(data, str):
                # 检查是否为base64编码的二进制数据
                try:
                    decoded_data = base64.b64decode(data)
                    # 尝试解码为UTF-8字符串
                    try:
                        return decoded_data.decode('utf-8')
                    except UnicodeDecodeError:
                        return decoded_data
                except:
                    # 如果不是base64，直接返回字符串
                    return data
            elif isinstance(data, dict):
                return data
            else:
                error_handler.log_warning("令牌数据格式无效", "DataSecurityManager")
                return None
                
        except Exception as e:
            error_handler.handle_error(e, "DataSecurityManager", "验证安全传输令牌失败")
            return None
    
    def get_audit_logs(self, user_id: str = None, data_category: DataCategory = None,
                      action: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取数据访问审计日志
        
        Args:
            user_id: 过滤用户ID
            data_category: 过滤数据类别
            action: 过滤操作类型
            limit: 返回最大数量
            
        Returns:
            审计日志列表
        """
        with self.lock:
            filtered_logs = self.audit_logs
            
            # 过滤
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            if data_category:
                filtered_logs = [log for log in filtered_logs if log.data_category == data_category]
            if action:
                filtered_logs = [log for log in filtered_logs if log.action == action]
            
            # 排序（最新的在前）
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 限制数量
            filtered_logs = filtered_logs[:limit]
            
            # 转换为字典
            result = []
            for log in filtered_logs:
                log_dict = asdict(log)
                log_dict['data_category'] = log.data_category.value
                result.append(log_dict)
            
            return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            # 统计各类别数据数量
            category_stats = {}
            for enc_data in self.encrypted_data.values():
                category = enc_data.data_category.value
                category_stats[category] = category_stats.get(category, 0) + 1
            
            # 统计活动密钥
            active_keys = sum(1 for k in self.keys.values() if k.is_active)
            expired_keys = sum(1 for k in self.keys.values() if k.expires_at and k.expires_at < time.time())
            
            status = {
                'total_keys': len(self.keys),
                'active_keys': active_keys,
                'expired_keys': expired_keys,
                'total_encrypted_data': len(self.encrypted_data),
                'category_stats': category_stats,
                'total_audit_logs': len(self.audit_logs),
                'audit_enabled': self.config['enable_audit_logging'],
                'hmac_enabled': self.config['enable_hmac_verification'],
                'default_algorithm': self.config['default_algorithm'],
                'last_updated': time.time(),
            }
            
            return status
    
    def rotate_keys(self, user_id: str = "system") -> bool:
        """轮换加密密钥
        
        Args:
            user_id: 执行轮换的用户ID
            
        Returns:
            是否成功
        """
        try:
            with self.lock:
                current_time = time.time()
                rotated_count = 0
                
                for key_id, key_obj in list(self.keys.items()):
                    # 检查是否需要轮换（过期或即将过期）
                    if (key_obj.expires_at and 
                        key_obj.expires_at < current_time + (7 * 24 * 3600)):  # 7天内过期
                        
                        # 创建新密钥
                        new_key_id = f"{key_id.split('_')[0]}_{key_id.split('_')[1]}_{int(time.time())}"
                        
                        # 重新生成密钥
                        salt = b"self_soul_key_rotation_salt"
                        
                        if key_obj.key_type == EncryptionAlgorithm.AES_256_GCM:
                            kdf = PBKDF2HMAC(
                                algorithm=hashes.SHA256(),
                                length=32,
                                salt=salt,
                                iterations=100000,
                                backend=default_backend()
                            )
                            new_key_data = kdf.derive(f"{self.master_key}_{new_key_id}".encode())
                            
                        elif key_obj.key_type == EncryptionAlgorithm.AES_256_CBC:
                            kdf = PBKDF2HMAC(
                                algorithm=hashes.SHA256(),
                                length=32,
                                salt=salt,
                                iterations=100000,
                                backend=default_backend()
                            )
                            new_key_data = kdf.derive(f"{self.master_key}_{new_key_id}_cbc".encode())
                            
                        elif key_obj.key_type == EncryptionAlgorithm.FERNET:
                            new_key_data = Fernet.generate_key()
                            
                        else:
                            continue  # 跳过不支持的密钥类型
                        
                        # 创建新密钥对象
                        new_key_obj = EncryptionKey(
                            key_id=new_key_id,
                            key_type=key_obj.key_type,
                            key_data=new_key_data,
                            created_at=current_time,
                            expires_at=current_time + (self.config['key_rotation_days'] * 24 * 3600),
                            description=f"{key_obj.description} (轮换版本)",
                            is_active=True,
                            version=key_obj.version + 1
                        )
                        
                        # 停用旧密钥
                        key_obj.is_active = False
                        key_obj.description = f"{key_obj.description} (已停用)"
                        
                        # 添加新密钥
                        self.keys[new_key_id] = new_key_obj
                        
                        rotated_count += 1
                        logger.info(f"轮换密钥: {key_id} -> {new_key_id}")
                
                # 记录审计日志
                if rotated_count > 0:
                    self._log_audit(
                        user_id=user_id,
                        data_id="key_rotation",
                        data_category=DataCategory.CONFIGURATION,
                        action="key_rotation",
                        success=True,
                        details={"rotated_count": rotated_count}
                    )
                
                return rotated_count > 0
                
        except Exception as e:
            error_handler.handle_error(e, "DataSecurityManager", "密钥轮换失败")
            
            # 记录失败的审计日志
            self._log_audit(
                user_id=user_id,
                data_id="key_rotation",
                data_category=DataCategory.CONFIGURATION,
                action="key_rotation",
                success=False,
                details={"error": str(e)}
            )
            
            return False


# 全局实例
_data_security_manager_instance = None

def get_data_security_manager() -> DataSecurityManager:
    """获取全局数据安全管理器实例"""
    global _data_security_manager_instance
    if _data_security_manager_instance is None:
        _data_security_manager_instance = DataSecurityManager()
    return _data_security_manager_instance