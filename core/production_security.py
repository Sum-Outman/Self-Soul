"""
生产级安全配置和权限管理系统
Production Security Configuration and Permission Management System
"""

import os
import hashlib
import hmac
import secrets
import time
import jwt
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
import redis
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64

from core.error_handling import error_handler

logger = logging.getLogger("ProductionSecurity")


class PermissionLevel(Enum):
    """权限级别枚举"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ResourceType(Enum):
    """资源类型枚举"""
    MODEL = "model"
    DATASET = "dataset"
    TRAINING = "training"
    SYSTEM = "system"
    API = "api"
    USER = "user"
    LOG = "log"
    CONFIG = "config"


@dataclass
class UserRole:
    """用户角色定义"""
    name: str
    permissions: Dict[ResourceType, Set[str]]
    description: str
    level: PermissionLevel


@dataclass
class SecurityPolicy:
    """安全策略配置"""
    # JWT配置
    jwt_secret: str
    encryption_key: str
    
    # JWT算法和过期时间
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    jwt_refresh_expire_days: int = 30
    
    # 密码策略
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_size: int = 5
    
    # 会话管理
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 3
    
    # 速率限制
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # 加密配置
    encryption_algorithm: str = "AES-256-GCM"
    
    # 审计配置
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    
    # 网络安全
    require_https: bool = True
    enable_csp: bool = True
    enable_hsts: bool = True
    
    # 多因素认证
    enable_mfa: bool = False
    mfa_timeout_minutes: int = 10


class ProductionSecurityManager:
    """生产级安全管理器"""
    
    def __init__(self, policy: SecurityPolicy, redis_client: Optional[redis.Redis] = None):
        self.policy = policy
        self.redis_client = redis_client
        self.roles = self._initialize_roles()
        
        # 验证配置
        self._validate_policy()
        
        # 初始化加密密钥
        self._initialize_encryption()
    
    def _validate_policy(self):
        """验证安全策略"""
        if len(self.policy.jwt_secret) < 64:
            raise ValueError("JWT secret must be at least 64 characters long")
        
        if len(self.policy.encryption_key) < 64:
            raise ValueError("Encryption key must be at least 64 characters long")
        
        if self.policy.min_password_length < 8:
            raise ValueError("Minimum password length must be at least 8 characters")
    
    def _initialize_roles(self) -> Dict[str, UserRole]:
        """初始化预定义角色"""
        return {
            "viewer": UserRole(
                name="viewer",
                permissions={
                    ResourceType.MODEL: {"read"},
                    ResourceType.DATASET: {"read"},
                    ResourceType.SYSTEM: {"read"}
                },
                description="只读用户",
                level=PermissionLevel.READ_ONLY
            ),
            "user": UserRole(
                name="user",
                permissions={
                    ResourceType.MODEL: {"read", "use"},
                    ResourceType.DATASET: {"read", "create"},
                    ResourceType.TRAINING: {"read", "create"},
                    ResourceType.SYSTEM: {"read"}
                },
                description="普通用户",
                level=PermissionLevel.READ_WRITE
            ),
            "admin": UserRole(
                name="admin",
                permissions={
                    ResourceType.MODEL: {"read", "use", "create", "update", "delete"},
                    ResourceType.DATASET: {"read", "create", "update", "delete"},
                    ResourceType.TRAINING: {"read", "create", "update", "delete", "manage"},
                    ResourceType.SYSTEM: {"read", "manage"},
                    ResourceType.API: {"read", "manage"},
                    ResourceType.USER: {"read", "manage"},
                    ResourceType.LOG: {"read"},
                    ResourceType.CONFIG: {"read", "update"}
                },
                description="管理员",
                level=PermissionLevel.ADMIN
            ),
            "super_admin": UserRole(
                name="super_admin",
                permissions={
                    ResourceType.MODEL: {"read", "use", "create", "update", "delete", "manage"},
                    ResourceType.DATASET: {"read", "create", "update", "delete", "manage"},
                    ResourceType.TRAINING: {"read", "create", "update", "delete", "manage"},
                    ResourceType.SYSTEM: {"read", "manage", "configure"},
                    ResourceType.API: {"read", "manage", "configure"},
                    ResourceType.USER: {"read", "manage", "configure"},
                    ResourceType.LOG: {"read", "manage"},
                    ResourceType.CONFIG: {"read", "update", "manage"}
                },
                description="超级管理员",
                level=PermissionLevel.SUPER_ADMIN
            )
        }
    
    def _initialize_encryption(self):
        """初始化加密系统"""
        # 生成主加密密钥
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        self.encryption_key = base64.urlsafe_b64encode(kdf.derive(self.policy.encryption_key.encode()))
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """验证密码强度"""
        errors = []
        
        if len(password) < self.policy.min_password_length:
            errors.append(f"密码长度至少需要 {self.policy.min_password_length} 个字符")
        
        if self.policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("密码必须包含大写字母")
        
        if self.policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("密码必须包含小写字母")
        
        if self.policy.require_numbers and not any(c.isdigit() for c in password):
            errors.append("密码必须包含数字")
        
        if self.policy.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("密码必须包含特殊字符")
        
        return len(errors) == 0, errors
    
    def hash_password(self, password: str) -> str:
        """使用bcrypt哈希密码"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode(), hashed_password.encode())
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_jwt_tokens(self, user_id: str, role: str, additional_claims: Dict[str, Any] = None) -> Dict[str, str]:
        """生成JWT访问令牌和刷新令牌"""
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}")
        
        # 生成访问令牌
        access_payload = {
            "user_id": user_id,
            "role": role,
            "permissions": {rt.value: list(perms) for rt, perms in self.roles[role].permissions.items()},
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.policy.jwt_expire_hours),
            "jti": secrets.token_hex(16),
            "type": "access"
        }
        
        if additional_claims:
            access_payload.update(additional_claims)
        
        access_token = jwt.encode(access_payload, self.policy.jwt_secret, algorithm=self.policy.jwt_algorithm)
        
        # 生成刷新令牌
        refresh_payload = {
            "user_id": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.policy.jwt_refresh_expire_days),
            "jti": secrets.token_hex(16),
            "type": "refresh"
        }
        
        refresh_token = jwt.encode(refresh_payload, self.policy.jwt_secret, algorithm=self.policy.jwt_algorithm)
        
        # 存储刷新令牌到Redis（如果可用）
        if self.redis_client:
            refresh_key = f"refresh_token:{user_id}:{refresh_payload['jti']}"
            self.redis_client.setex(refresh_key, self.policy.jwt_refresh_expire_days * 24 * 3600, "valid")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": self.policy.jwt_expire_hours * 3600
        }
    
    def verify_jwt_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.policy.jwt_secret, algorithms=[self.policy.jwt_algorithm])
            
            # 验证令牌类型
            if payload.get("type") != token_type:
                error_handler.log_warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}", "ProductionSecurity")
                return None
            
            # 验证刷新令牌是否在Redis中（如果可用）
            if token_type == "refresh" and self.redis_client:
                refresh_key = f"refresh_token:{payload['user_id']}:{payload['jti']}"
                if not self.redis_client.exists(refresh_key):
                    error_handler.log_warning("Refresh token not found in Redis", "ProductionSecurity")
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            error_handler.log_warning("JWT token expired", "ProductionSecurity")
            return None
        except jwt.InvalidTokenError as e:
            error_handler.log_warning(f"Invalid JWT token: {e}", "ProductionSecurity")
            return None
    
    def revoke_refresh_token(self, user_id: str, token_jti: str) -> bool:
        """撤销刷新令牌"""
        if self.redis_client:
            refresh_key = f"refresh_token:{user_id}:{token_jti}"
            return bool(self.redis_client.delete(refresh_key))
        return False
    
    def revoke_all_user_tokens(self, user_id: str) -> bool:
        """撤销用户的所有刷新令牌"""
        if self.redis_client:
            pattern = f"refresh_token:{user_id}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                return bool(self.redis_client.delete(*keys))
        return False
    
    def check_permission(self, user_permissions: Dict[str, List[str]], 
                        resource_type: ResourceType, action: str) -> bool:
        """检查用户权限"""
        resource_permissions = user_permissions.get(resource_type.value, [])
        return action in resource_permissions
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        fernet = Fernet(self.encryption_key)
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        fernet = Fernet(self.encryption_key)
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        return fernet.decrypt(encrypted).decode()
    
    def get_security_headers(self) -> Dict[str, str]:
        """获取安全HTTP头"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }
        
        if self.policy.enable_hsts:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        if self.policy.enable_csp:
            headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        
        return headers
    
    def audit_log(self, user_id: str, action: str, resource: str, 
                  success: bool, details: Dict[str, Any] = None):
        """记录审计日志"""
        if not self.policy.enable_audit_logging:
            return
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {},
            "ip_address": "",  # 可以从请求中获取
            "user_agent": ""   # 可以从请求中获取
        }
        
        logger.info(f"AUDIT: {log_entry}")
        
        # 可以存储到数据库或文件系统
        if self.redis_client:
            audit_key = f"audit_log:{datetime.utcnow().strftime('%Y%m%d')}"
            self.redis_client.rpush(audit_key, str(log_entry))


class PermissionDecorator:
    """权限装饰器"""
    
    def __init__(self, security_manager: ProductionSecurityManager):
        self.security_manager = security_manager
    
    def require_permission(self, resource_type: ResourceType, action: str):
        """需要特定权限的装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 从请求中获取用户信息
                request = kwargs.get('request')
                if not request:
                    raise PermissionError("Request object not found")
                
                # 获取用户权限
                user_permissions = getattr(request.state, 'user_permissions', {})
                
                # 检查权限
                if not self.security_manager.check_permission(user_permissions, resource_type, action):
                    self.security_manager.audit_log(
                        getattr(request.state, 'user_id', 'unknown'),
                        f"{action}_{resource_type.value}",
                        "api",
                        False,
                        {"reason": "insufficient_permissions"}
                    )
                    raise PermissionError(f"Insufficient permissions for {action} on {resource_type.value}")
                
                # 记录成功的访问
                self.security_manager.audit_log(
                    getattr(request.state, 'user_id', 'unknown'),
                    f"{action}_{resource_type.value}",
                    "api",
                    True
                )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, required_role: str):
        """需要特定角色的装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get('request')
                if not request:
                    raise PermissionError("Request object not found")
                
                user_role = getattr(request.state, 'user_role', None)
                if not user_role or user_role != required_role:
                    self.security_manager.audit_log(
                        getattr(request.state, 'user_id', 'unknown'),
                        f"access_restricted_resource",
                        "api",
                        False,
                        {"reason": "insufficient_role", "required": required_role, "actual": user_role}
                    )
                    raise PermissionError(f"Role {required_role} required")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# 生产环境安全配置
def create_production_security_policy() -> SecurityPolicy:
    """创建生产环境安全策略"""
    return SecurityPolicy(
        jwt_secret=os.getenv("JWT_SECRET", secrets.token_hex(64)),
        encryption_key=os.getenv("ENCRYPTION_KEY", secrets.token_hex(64)),
        min_password_length=12,
        require_uppercase=True,
        require_lowercase=True,
        require_numbers=True,
        require_special_chars=True,
        password_history_size=5,
        session_timeout_minutes=60,
        max_concurrent_sessions=3,
        rate_limit_requests=100,
        rate_limit_window=60,
        enable_audit_logging=True,
        audit_log_retention_days=90,
        require_https=True,
        enable_csp=True,
        enable_hsts=True,
        enable_mfa=False
    )


def initialize_production_security(redis_url: str = None) -> ProductionSecurityManager:
    """初始化生产级安全系统"""
    # 创建安全策略
    policy = create_production_security_policy()
    
    # 初始化Redis客户端（如果提供URL）
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            error_handler.log_warning(f"Failed to initialize Redis client: {e}", "ProductionSecurity")
    
    # 创建安全管理器
    security_manager = ProductionSecurityManager(policy, redis_client)
    logger.info("Production security manager initialized successfully")
    
    return security_manager


# 全局安全管理器实例
production_security_manager = None

def get_production_security_manager() -> ProductionSecurityManager:
    """获取全局安全管理器实例"""
    global production_security_manager
    if production_security_manager is None:
        production_security_manager = initialize_production_security()
    return production_security_manager
