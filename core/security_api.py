"""
安全认证和权限管理API端点
Security Authentication and Permission Management API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from core.production_security import (
    ProductionSecurityManager, PermissionLevel, ResourceType,
    get_production_security_manager, PermissionDecorator
)

logger = logging.getLogger("SecurityAPI")

router = APIRouter(prefix="/api/security", tags=["security"])
security = HTTPBearer()


# 请求和响应模型
class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str
    password: str
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('用户名至少需要3个字符')
        if len(v) > 50:
            raise ValueError('用户名不能超过50个字符')
        return v


class LoginResponse(BaseModel):
    """登录响应模型"""
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"
    user_id: str
    role: str
    permissions: Dict[str, List[str]]


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求模型"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """修改密码请求模型"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 12:
            raise ValueError('新密码至少需要12个字符')
        return v


class UserCreateRequest(BaseModel):
    """创建用户请求模型"""
    username: str
    password: str
    role: str
    email: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('用户名至少需要3个字符')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 12:
            raise ValueError('密码至少需要12个字符')
        return v


class UserResponse(BaseModel):
    """用户响应模型"""
    user_id: str
    username: str
    role: str
    email: Optional[str]
    created_at: str
    last_login: Optional[str]
    is_active: bool


class PermissionCheckRequest(BaseModel):
    """权限检查请求模型"""
    resource_type: str
    action: str


class PermissionCheckResponse(BaseModel):
    """权限检查响应模型"""
    has_permission: bool
    resource_type: str
    action: str
    user_role: str


# 依赖注入函数
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
) -> Dict[str, Any]:
    """获取当前用户"""
    token = credentials.credentials
    
    # 验证访问令牌
    payload = security_manager.verify_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的访问令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return payload


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)):
    """需要管理员权限"""
    if user.get("role") not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return user


async def require_super_admin(user: Dict[str, Any] = Depends(get_current_user)):
    """需要超级管理员权限"""
    if user.get("role") != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要超级管理员权限"
        )
    return user


# 用户存储（生产环境中应该使用数据库）
_users_db = {}
_sessions_db = {}


def _get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """根据用户名获取用户"""
    for user_id, user_data in _users_db.items():
        if user_data.get("username") == username:
            return {"user_id": user_id, **user_data}
    return None


def _create_user_id() -> str:
    """创建用户ID"""
    import uuid
    return str(uuid.uuid4())


# API端点
@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """用户登录"""
    try:
        # 在生产环境中，这里应该查询数据库验证用户凭据
        user = _get_user_by_username(request.username)
        
        if not user:
            # 记录失败的登录尝试
            security_manager.audit_log(
                "unknown",
                "login_attempt",
                "auth",
                False,
                {"username": request.username, "reason": "user_not_found"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )
        
        # 验证密码
        if not security_manager.verify_password(request.password, user["password_hash"]):
            # 记录失败的登录尝试
            security_manager.audit_log(
                user["user_id"],
                "login_attempt",
                "auth",
                False,
                {"username": request.username, "reason": "invalid_password"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误"
            )
        
        # 检查用户是否激活
        if not user.get("is_active", True):
            security_manager.audit_log(
                user["user_id"],
                "login_attempt",
                "auth",
                False,
                {"username": request.username, "reason": "account_inactive"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="账户已被禁用"
            )
        
        # 生成JWT令牌
        tokens = security_manager.generate_jwt_tokens(
            user["user_id"],
            user["role"],
            {"username": user["username"]}
        )
        
        # 更新最后登录时间
        user["last_login"] = datetime.utcnow().isoformat()
        
        # 记录成功的登录
        security_manager.audit_log(
            user["user_id"],
            "login_success",
            "auth",
            True,
            {"username": request.username}
        )
        
        # 获取用户权限
        user_permissions = {}
        if user["role"] in security_manager.roles:
            role_permissions = security_manager.roles[user["role"]].permissions
            user_permissions = {rt.value: list(perms) for rt, perms in role_permissions.items()}
        
        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens["expires_in"],
            user_id=user["user_id"],
            role=user["role"],
            permissions=user_permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录过程中发生错误"
        )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """刷新访问令牌"""
    try:
        # 验证刷新令牌
        payload = security_manager.verify_jwt_token(request.refresh_token, "refresh")
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效或过期的刷新令牌"
            )
        
        user_id = payload["user_id"]
        
        # 获取用户信息
        user = _users_db.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在"
            )
        
        # 生成新的访问令牌
        tokens = security_manager.generate_jwt_tokens(
            user_id,
            user["role"],
            {"username": user["username"]}
        )
        
        # 获取用户权限
        user_permissions = {}
        if user["role"] in security_manager.roles:
            role_permissions = security_manager.roles[user["role"]].permissions
            user_permissions = {rt.value: list(perms) for rt, perms in role_permissions.items()}
        
        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens["expires_in"],
            user_id=user_id,
            role=user["role"],
            permissions=user_permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="令牌刷新过程中发生错误"
        )


@router.post("/logout")
async def logout(
    user: Dict[str, Any] = Depends(get_current_user),
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """用户登出"""
    try:
        # 撤销所有刷新令牌
        security_manager.revoke_all_user_tokens(user["user_id"])
        
        # 记录登出
        security_manager.audit_log(
            user["user_id"],
            "logout",
            "auth",
            True
        )
        
        return {"message": "登出成功"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出过程中发生错误"
        )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """修改密码"""
    try:
        user_id = user["user_id"]
        user_data = _users_db.get(user_id)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 验证当前密码
        if not security_manager.verify_password(request.current_password, user_data["password_hash"]):
            security_manager.audit_log(
                user_id,
                "change_password_attempt",
                "auth",
                False,
                {"reason": "invalid_current_password"}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="当前密码错误"
            )
        
        # 验证新密码强度
        is_valid, errors = security_manager.validate_password_strength(request.new_password)
        if not is_valid:
            security_manager.audit_log(
                user_id,
                "change_password_attempt",
                "auth",
                False,
                {"reason": "weak_password", "errors": errors}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"密码强度不足: {', '.join(errors)}"
            )
        
        # 更新密码
        user_data["password_hash"] = security_manager.hash_password(request.new_password)
        
        # 撤销所有现有令牌（强制重新登录）
        security_manager.revoke_all_user_tokens(user_id)
        
        # 记录密码修改
        security_manager.audit_log(
            user_id,
            "change_password",
            "auth",
            True
        )
        
        return {"message": "密码修改成功，请重新登录"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码修改过程中发生错误"
        )


@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_admin)])
async def create_user(
    request: UserCreateRequest,
    admin_user: Dict[str, Any] = Depends(require_admin),
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """创建用户（需要管理员权限）"""
    try:
        # 检查用户名是否已存在
        if _get_user_by_username(request.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
        
        # 验证角色
        if request.role not in security_manager.roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的角色: {request.role}"
            )
        
        # 验证密码强度
        is_valid, errors = security_manager.validate_password_strength(request.password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"密码强度不足: {', '.join(errors)}"
            )
        
        # 创建用户
        user_id = _create_user_id()
        user_data = {
            "username": request.username,
            "password_hash": security_manager.hash_password(request.password),
            "role": request.role,
            "email": request.email,
            "created_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "is_active": True
        }
        
        _users_db[user_id] = user_data
        
        # 记录用户创建
        security_manager.audit_log(
            admin_user["user_id"],
            "create_user",
            "user_management",
            True,
            {"created_user_id": user_id, "username": request.username, "role": request.role}
        )
        
        return UserResponse(
            user_id=user_id,
            username=request.username,
            role=request.role,
            email=request.email,
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            is_active=user_data["is_active"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建用户过程中发生错误"
        )


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """获取当前用户信息"""
    try:
        user_data = _users_db.get(user["user_id"])
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        return UserResponse(
            user_id=user["user_id"],
            username=user_data["username"],
            role=user_data["role"],
            email=user_data.get("email"),
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            is_active=user_data["is_active"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息过程中发生错误"
        )


@router.post("/check-permission", response_model=PermissionCheckResponse)
async def check_permission(
    request: PermissionCheckRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """检查用户权限"""
    try:
        # 验证资源类型
        try:
            resource_type = ResourceType(request.resource_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的资源类型: {request.resource_type}"
            )
        
        # 检查权限
        has_permission = security_manager.check_permission(
            user.get("permissions", {}),
            resource_type,
            request.action
        )
        
        return PermissionCheckResponse(
            has_permission=has_permission,
            resource_type=request.resource_type,
            action=request.action,
            user_role=user.get("role", "unknown")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check permission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="权限检查过程中发生错误"
        )


@router.get("/roles")
async def get_available_roles(
    security_manager: ProductionSecurityManager = Depends(get_production_security_manager)
):
    """获取可用角色列表"""
    try:
        roles = []
        for role_name, role in security_manager.roles.items():
            roles.append({
                "name": role.name,
                "description": role.description,
                "level": role.level.value,
                "permissions": {rt.value: list(perms) for rt, perms in role.permissions.items()}
            })
        
        return {"roles": roles}
        
    except Exception as e:
        logger.error(f"Get roles error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取角色列表过程中发生错误"
        )


# 初始化默认用户（仅用于演示）
def initialize_default_users(security_manager: ProductionSecurityManager):
    """初始化默认用户"""
    # 创建默认管理员用户
    admin_user_id = _create_user_id()
    _users_db[admin_user_id] = {
        "username": "admin",
        "password_hash": security_manager.hash_password("Admin123!@#"),
        "role": "super_admin",
        "email": "admin@selfsoul.com",
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "is_active": True
    }
    
    # 创建默认普通用户
    user_id = _create_user_id()
    _users_db[user_id] = {
        "username": "user",
        "password_hash": security_manager.hash_password("User123!@#"),
        "role": "user",
        "email": "user@selfsoul.com",
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "is_active": True
    }
    
    logger.info("Default users initialized")


# 安全中间件
async def security_middleware(request: Request, call_next):
    """安全中间件"""
    security_manager = get_production_security_manager()
    
    # 添加安全头
    response = await call_next(request)
    
    # 添加安全头
    security_headers = security_manager.get_security_headers()
    
    # 如果是文档页面，使用更宽松的CSP策略
    if request.url.path.startswith('/docs') or request.url.path in ["/openapi.json", "/redoc"]:
        # 创建安全头的副本
        modified_headers = security_headers.copy()
        # 修改CSP头以允许CDN资源
        modified_headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; font-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; img-src 'self' data: https:;"
        # 使用修改后的头
        for header, value in modified_headers.items():
            response.headers[header] = value
    else:
        # 使用原始安全头
        for header, value in security_headers.items():
            response.headers[header] = value
    
    return response


# 在FastAPI应用中集成安全系统
def setup_security_system(app, security_manager: ProductionSecurityManager):
    """设置安全系统"""
    # 添加安全API路由
    app.include_router(router)
    
    # 添加安全中间件
    app.middleware("http")(security_middleware)
    
    # 初始化默认用户
    initialize_default_users(security_manager)
    
    logger.info("Security system setup completed")