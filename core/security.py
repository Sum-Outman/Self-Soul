"""
安全配置和权限控制模块
"""

import os
import hashlib
import hmac
import secrets
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
import logging
from core.error_handling import error_handler

logger = logging.getLogger("Security")


@dataclass
class SecurityConfig:
    """安全配置数据类"""
    # JWT配置
    jwt_secret: str
    
    # 加密配置
    encryption_key: str
    
    # JWT算法和过期时间（默认参数）
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    
    # 加密算法（默认参数）
    encryption_algorithm: str = "AES"
    
    # 速率限制配置
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 秒
    
    # CORS配置
    cors_allowed_origins: List[str] = None
    cors_allowed_methods: List[str] = None
    cors_allowed_headers: List[str] = None
    
    def __post_init__(self):
        if self.cors_allowed_origins is None:
            self.cors_allowed_origins = ["*"]
        if self.cors_allowed_methods is None:
            self.cors_allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.cors_allowed_headers is None:
            self.cors_allowed_headers = ["*"]


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limits: Dict[str, List[float]] = {}
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证安全配置"""
        env = os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            if not self.config.jwt_secret or len(self.config.jwt_secret) < 32:
                raise ValueError("JWT secret must be at least 32 characters long in production")
            
            if not self.config.encryption_key or len(self.config.encryption_key) < 32:
                raise ValueError("Encryption key must be at least 32 characters long in production")
        else:
            if not self.config.jwt_secret or len(self.config.jwt_secret) < 32:
                error_handler.log_warning("JWT secret is too short, using a random key for development", "Security")
                self.config.jwt_secret = secrets.token_hex(32)
            
            if not self.config.encryption_key or len(self.config.encryption_key) < 32:
                error_handler.log_warning("Encryption key is too short, using a random key for development", "Security")
                self.config.encryption_key = secrets.token_hex(32)
    
    def generate_jwt_token(self, payload: Dict[str, Any]) -> str:
        """生成JWT令牌"""
        # 添加过期时间
        payload["exp"] = datetime.utcnow() + timedelta(hours=self.config.jwt_expire_hours)
        payload["iat"] = datetime.utcnow()
        payload["jti"] = secrets.token_hex(16)  # 唯一标识符
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            error_handler.log_warning("JWT token expired", "Security")
            return None
        except jwt.InvalidTokenError:
            error_handler.log_warning("Invalid JWT token", "Security")
            return None
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            salt, stored_hash = hashed_password.split(":")
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
            return hmac.compare_digest(computed_hash, stored_hash)
        except Exception as e:
            logging.warning(f"密码验证失败: {e}")
            return False
    
    def check_rate_limit(self, identifier: str) -> bool:
        """检查速率限制"""
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        # 获取或初始化该标识符的请求记录
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # 清理过期的请求记录
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier] 
            if req_time > window_start
        ]
        
        # 检查是否超过限制
        if len(self.rate_limits[identifier]) >= self.config.rate_limit_requests:
            return False
        
        # 记录当前请求
        self.rate_limits[identifier].append(now)
        return True
    
    def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """获取速率限制信息"""
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        if identifier not in self.rate_limits:
            return {
                "remaining": self.config.rate_limit_requests,
                "reset_time": int(window_start + self.config.rate_limit_window)
            }
        
        # 清理过期的请求记录
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier] 
            if req_time > window_start
        ]
        
        remaining = self.config.rate_limit_requests - len(self.rate_limits[identifier])
        reset_time = int(window_start + self.config.rate_limit_window)
        
        return {
            "remaining": max(0, remaining),
            "reset_time": reset_time
        }
    
    def encrypt_data(self, data: str) -> str:
        """加密数据（简化实现）"""
        # 在实际应用中，应该使用更安全的加密库如cryptography
        import base64
        from cryptography.fernet import Fernet
        
        # 从配置密钥生成Fernet密钥
        key = base64.urlsafe_b64encode(hashlib.sha256(self.config.encryption_key.encode()).digest())
        fernet = Fernet(key)
        
        return fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        import base64
        from cryptography.fernet import Fernet
        
        key = base64.urlsafe_b64encode(hashlib.sha256(self.config.encryption_key.encode()).digest())
        fernet = Fernet(key)
        
        return fernet.decrypt(encrypted_data.encode()).decode()
    
    def sanitize_input(self, input_data: str) -> str:
        """清理输入数据"""
        import html
        
        # HTML转义
        sanitized = html.escape(input_data)
        
        # 移除危险字符
        dangerous_chars = ['<script>', 'javascript:', 'onload=', 'onerror=']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized


class APIAuthentication:
    """API认证管理器"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # 加载API密钥
        self._load_api_keys()
    
    def _load_api_keys(self):
        """加载API密钥"""
        # 从环境变量加载API密钥
        model_service_key = os.getenv("MODEL_SERVICE_API_KEY")
        system_monitor_key = os.getenv("SYSTEM_MONITOR_API_KEY")
        realtime_stream_key = os.getenv("REALTIME_STREAM_API_KEY")
        
        if model_service_key:
            self.api_keys[model_service_key] = {
                "name": "model_service",
                "permissions": ["models:read", "models:write", "training:manage"]
            }
        
        if system_monitor_key:
            self.api_keys[system_monitor_key] = {
                "name": "system_monitor",
                "permissions": ["system:monitor", "metrics:read"]
            }
        
        if realtime_stream_key:
            self.api_keys[realtime_stream_key] = {
                "name": "realtime_stream",
                "permissions": ["stream:read", "stream:write"]
            }
    
    def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        if api_key in self.api_keys:
            return self.api_keys[api_key]
        return None
    
    def has_permission(self, api_key_info: Dict[str, Any], permission: str) -> bool:
        """检查权限"""
        return permission in api_key_info.get("permissions", [])


# 全局安全配置
security_config = SecurityConfig(
    jwt_secret=os.getenv("JWT_SECRET", secrets.token_hex(64)),
    encryption_key=os.getenv("ENCRYPTION_KEY", secrets.token_hex(64)),
    rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
    cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
)

security_manager = SecurityManager(security_config)
api_auth = APIAuthentication(security_manager)


def get_security_headers() -> Dict[str, str]:
    """获取安全HTTP头"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; font-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; img-src 'self' data: https:;",
    }


def validate_api_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证API请求"""
    errors = []
    
    # 检查必需字段
    required_fields = ["model_id", "input_data"]
    for field in required_fields:
        if field not in request_data or not request_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # 验证输入数据长度
    if "input_data" in request_data and len(request_data["input_data"]) > 10000:
        errors.append("Input data too long (max 10000 characters)")
    
    # 验证模型ID格式
    if "model_id" in request_data:
        model_id = request_data["model_id"]
        if not isinstance(model_id, str) or not model_id.isalnum():
            errors.append("Invalid model ID format")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


if __name__ == "__main__":
    # 测试安全功能
    test_payload = {"user_id": "test_user", "role": "admin"}
    
    # 生成JWT令牌
    token = security_manager.generate_jwt_token(test_payload)
    print("Generated JWT token:", token)
    
    # 验证JWT令牌
    verified = security_manager.verify_jwt_token(token)
    print("Verified payload:", verified)
    
    # 测试密码哈希
    password = "test_password"
    hashed = security_manager.hash_password(password)
    print("Hashed password:", hashed)
    
    # 验证密码
    is_valid = security_manager.verify_password(password, hashed)
    print("Password verification:", is_valid)
    
    # 测试速率限制
    for i in range(5):
        allowed = security_manager.check_rate_limit("test_client")
        print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")
    
    # 测试API认证
    test_key = "test_api_key"
    api_auth.api_keys[test_key] = {"name": "test", "permissions": ["read"]}
    auth_result = api_auth.authenticate_api_key(test_key)
    print("API authentication:", auth_result)
