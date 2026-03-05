"""
RBAC权限管理器 - 支持细粒度权限控制和按模型/任务分配权限

基于角色的访问控制（RBAC）系统，支持：
1. 细粒度权限：查看(view)、训练(train)、配置(configure)、删除(delete)
2. 按模型/任务分配权限
3. 权限变更记录审计日志
4. 支持角色继承和多租户

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
import threading
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle

from core.error_handling import error_handler

# 日志记录器
logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """权限类型枚举"""
    VIEW = "view"           # 查看
    TRAIN = "train"         # 训练
    CONFIGURE = "configure" # 配置
    DELETE = "delete"       # 删除
    MANAGE = "manage"       # 管理（包含所有权限）
    EXECUTE = "execute"     # 执行/运行


class ResourceType(Enum):
    """资源类型枚举"""
    MODEL = "model"         # 模型
    TASK = "task"           # 任务
    DATASET = "dataset"     # 数据集
    CONFIG = "config"       # 配置
    SYSTEM = "system"       # 系统
    USER = "user"           # 用户
    LOG = "log"             # 日志
    API = "api"             # API接口


class RoleType(Enum):
    """角色类型枚举"""
    VIEWER = "viewer"       # 查看者
    USER = "user"           # 普通用户
    TRAINER = "trainer"     # 训练员
    ADMIN = "admin"         # 管理员
    SUPER_ADMIN = "super_admin"  # 超级管理员
    MODEL_OWNER = "model_owner"  # 模型所有者
    TASK_OWNER = "task_owner"    # 任务所有者


@dataclass
class Role:
    """角色定义"""
    role_id: str
    role_name: str
    role_type: RoleType
    description: str
    inherited_roles: List[str]  # 继承的角色ID列表
    default_permissions: Dict[str, List[str]]  # 默认权限 {resource_type: [permission_type]}
    created_at: float
    updated_at: float
    is_system_role: bool = False


@dataclass
class ResourcePermission:
    """资源权限定义"""
    resource_id: str
    resource_type: ResourceType
    permissions: List[str]  # 权限类型列表
    granted_at: float
    granted_by: str
    expires_at: Optional[float] = None  # 过期时间戳，None表示永不过期


@dataclass
class UserRoleAssignment:
    """用户角色分配"""
    user_id: str
    role_id: str
    assigned_at: float
    assigned_by: str
    scope: Optional[Dict[str, Any]] = None  # 作用域限制，如{"model_ids": ["model1", "model2"]}
    expires_at: Optional[float] = None


@dataclass
class DirectPermission:
    """直接权限分配（绕过角色）"""
    user_id: str
    resource_id: str
    resource_type: ResourceType
    permission_type: PermissionType
    granted_at: float
    granted_by: str
    expires_at: Optional[float] = None


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    log_id: str
    timestamp: float
    user_id: str
    action: str
    resource_type: ResourceType
    resource_id: str
    details: Dict[str, Any]
    ip_address: str = ""
    user_agent: str = ""
    success: bool = True


class RBACPermissionManager:
    """RBAC权限管理器
    
    主要功能：
    1. 细粒度权限控制（查看、训练、配置、删除）
    2. 按模型/任务分配权限
    3. 权限变更记录审计日志
    4. 支持角色继承和多租户
    5. 权限缓存和性能优化
    """
    
    def __init__(self, data_dir: str = None):
        """初始化RBAC权限管理器
        
        Args:
            data_dir: 数据目录路径，用于存储权限数据
        """
        # 设置数据目录
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "data", "rbac_permissions")
        else:
            self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 核心数据结构
        self.roles: Dict[str, Role] = {}  # {role_id: Role}
        self.user_role_assignments: Dict[str, List[UserRoleAssignment]] = defaultdict(list)  # {user_id: [UserRoleAssignment]}
        self.direct_permissions: Dict[str, List[DirectPermission]] = defaultdict(list)  # {user_id: [DirectPermission]}
        self.resource_permissions: Dict[str, Dict[str, ResourcePermission]] = defaultdict(dict)  # {resource_type: {resource_id: ResourcePermission}}
        
        # 审计日志
        self.audit_logs: List[AuditLogEntry] = []
        
        # 权限缓存 {user_id: {resource_type: {resource_id: set(permissions)}}}
        self.permission_cache: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        
        # 缓存失效时间
        self.cache_invalidation_timestamps: Dict[str, float] = defaultdict(float)
        
        # 互斥锁
        self.lock = threading.RLock()
        
        # 配置
        self.config = {
            'max_audit_logs': 10000,           # 最大审计日志记录数
            'cache_ttl_seconds': 300,          # 缓存TTL（秒）
            'auto_save_interval': 300,         # 自动保存间隔（秒）
            'enable_permission_caching': True, # 启用权限缓存
            'audit_enabled': True,             # 启用审计日志
        }
        
        # 加载现有数据
        self._load_data()
        
        # 初始化系统角色
        self._initialize_system_roles()
        
        # 启动自动保存线程
        self._start_auto_save()
        
        logger.info(f"RBAC权限管理器初始化完成，已加载 {len(self.roles)} 个角色")
    
    def _load_data(self):
        """加载保存的数据"""
        data_files = {
            'roles': 'roles.json',
            'user_role_assignments': 'user_role_assignments.pkl',
            'direct_permissions': 'direct_permissions.pkl',
            'resource_permissions': 'resource_permissions.pkl',
            'audit_logs': 'audit_logs.pkl',
        }
        
        for data_key, filename in data_files.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    if filename.endswith('.json'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    else:  # .pkl files
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                    
                    # 特殊处理：将字典转换为对象
                    if data_key == 'roles' and data:
                        roles_dict = {}
                        for role_id, role_data in data.items():
                            if isinstance(role_data, dict):
                                roles_dict[role_id] = Role(**role_data)
                        data = roles_dict
                    
                    setattr(self, data_key, data)
                    logger.info(f"加载 {data_key}: {len(data) if isinstance(data, dict) else 'N/A'}")
            except Exception as e:
                error_handler.handle_error(e, "RBACPermissionManager", 
                                          f"加载 {filename} 失败，使用默认值")
    
    def _save_data(self):
        """保存数据到文件"""
        with self.lock:
            data_files = {
                'roles': ('roles.json', 'json'),
                'user_role_assignments': ('user_role_assignments.pkl', 'pkl'),
                'direct_permissions': ('direct_permissions.pkl', 'pkl'),
                'resource_permissions': ('resource_permissions.pkl', 'pkl'),
                'audit_logs': ('audit_logs.pkl', 'pkl'),
            }
            
            for data_key, (filename, format_type) in data_files.items():
                filepath = os.path.join(self.data_dir, filename)
                try:
                    data = getattr(self, data_key)
                    
                    # 特殊处理：将对象转换为字典
                    if data_key == 'roles' and data:
                        roles_dict = {}
                        for role_id, role_obj in data.items():
                            roles_dict[role_id] = asdict(role_obj)
                        data = roles_dict
                    
                    # 限制审计日志数量
                    if data_key == 'audit_logs':
                        if len(data) > self.config['max_audit_logs']:
                            data = data[-self.config['max_audit_logs']:]
                    
                    if format_type == 'json':
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    else:  # pkl
                        with open(filepath, 'wb') as f:
                            pickle.dump(data, f)
                            
                except Exception as e:
                    error_handler.handle_error(e, "RBACPermissionManager", 
                                              f"保存 {filename} 失败")
    
    def _start_auto_save(self):
        """启动自动保存线程"""
        def auto_save_worker():
            while True:
                time.sleep(self.config['auto_save_interval'])
                try:
                    self._save_data()
                except Exception as e:
                    error_handler.handle_error(e, "RBACPermissionManager", "自动保存失败")
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info(f"启动自动保存线程，间隔 {self.config['auto_save_interval']} 秒")
    
    def _initialize_system_roles(self):
        """初始化系统角色"""
        if self.roles:
            return  # 已经加载了角色
        
        current_time = time.time()
        
        # 1. 查看者角色（只读）
        viewer_role = Role(
            role_id="role_viewer",
            role_name="查看者",
            role_type=RoleType.VIEWER,
            description="只能查看模型和任务，不能进行任何修改操作",
            inherited_roles=[],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[viewer_role.role_id] = viewer_role
        
        # 2. 普通用户角色
        user_role = Role(
            role_id="role_user",
            role_name="普通用户",
            role_type=RoleType.USER,
            description="可以查看和使用模型，创建训练任务",
            inherited_roles=["role_viewer"],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value, PermissionType.EXECUTE.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value, PermissionType.TRAIN.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[user_role.role_id] = user_role
        
        # 3. 训练员角色
        trainer_role = Role(
            role_id="role_trainer",
            role_name="训练员",
            role_type=RoleType.TRAINER,
            description="可以训练和管理训练任务",
            inherited_roles=["role_user"],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value, PermissionType.EXECUTE.value, PermissionType.TRAIN.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value, PermissionType.TRAIN.value, PermissionType.CONFIGURE.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[trainer_role.role_id] = trainer_role
        
        # 4. 管理员角色
        admin_role = Role(
            role_id="role_admin",
            role_name="管理员",
            role_type=RoleType.ADMIN,
            description="可以管理模型、任务和用户",
            inherited_roles=["role_trainer"],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value, PermissionType.EXECUTE.value, PermissionType.TRAIN.value, PermissionType.CONFIGURE.value, PermissionType.DELETE.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value, PermissionType.TRAIN.value, PermissionType.CONFIGURE.value, PermissionType.DELETE.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value, PermissionType.CONFIGURE.value, PermissionType.DELETE.value],
                ResourceType.USER.value: [PermissionType.VIEW.value, PermissionType.CONFIGURE.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value, PermissionType.CONFIGURE.value],
                ResourceType.CONFIG.value: [PermissionType.VIEW.value, PermissionType.CONFIGURE.value],
                ResourceType.LOG.value: [PermissionType.VIEW.value],
                ResourceType.API.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[admin_role.role_id] = admin_role
        
        # 5. 超级管理员角色
        super_admin_role = Role(
            role_id="role_super_admin",
            role_name="超级管理员",
            role_type=RoleType.SUPER_ADMIN,
            description="拥有所有权限",
            inherited_roles=["role_admin"],
            default_permissions={
                ResourceType.MODEL.value: [pt.value for pt in PermissionType],
                ResourceType.TASK.value: [pt.value for pt in PermissionType],
                ResourceType.DATASET.value: [pt.value for pt in PermissionType],
                ResourceType.USER.value: [pt.value for pt in PermissionType],
                ResourceType.SYSTEM.value: [pt.value for pt in PermissionType],
                ResourceType.CONFIG.value: [pt.value for pt in PermissionType],
                ResourceType.LOG.value: [pt.value for pt in PermissionType],
                ResourceType.API.value: [pt.value for pt in PermissionType],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[super_admin_role.role_id] = super_admin_role
        
        # 6. 模型所有者角色
        model_owner_role = Role(
            role_id="role_model_owner",
            role_name="模型所有者",
            role_type=RoleType.MODEL_OWNER,
            description="对自己创建的模型拥有完全控制权",
            inherited_roles=["role_trainer"],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value, PermissionType.EXECUTE.value, PermissionType.TRAIN.value, PermissionType.CONFIGURE.value, PermissionType.DELETE.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value, PermissionType.TRAIN.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[model_owner_role.role_id] = model_owner_role
        
        # 7. 任务所有者角色
        task_owner_role = Role(
            role_id="role_task_owner",
            role_name="任务所有者",
            role_type=RoleType.TASK_OWNER,
            description="对自己创建的任务拥有完全控制权",
            inherited_roles=["role_user"],
            default_permissions={
                ResourceType.MODEL.value: [PermissionType.VIEW.value, PermissionType.EXECUTE.value],
                ResourceType.TASK.value: [PermissionType.VIEW.value, PermissionType.TRAIN.value, PermissionType.CONFIGURE.value, PermissionType.DELETE.value],
                ResourceType.DATASET.value: [PermissionType.VIEW.value],
                ResourceType.SYSTEM.value: [PermissionType.VIEW.value],
            },
            created_at=current_time,
            updated_at=current_time,
            is_system_role=True
        )
        self.roles[task_owner_role.role_id] = task_owner_role
        
        logger.info(f"初始化 {len(self.roles)} 个系统角色")
    
    def _log_audit(self, user_id: str, action: str, resource_type: ResourceType, 
                  resource_id: str, details: Dict[str, Any], success: bool = True):
        """记录审计日志"""
        if not self.config['audit_enabled']:
            return
        
        log_entry = AuditLogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success
        )
        
        with self.lock:
            self.audit_logs.append(log_entry)
            
            # 限制日志数量
            if len(self.audit_logs) > self.config['max_audit_logs']:
                self.audit_logs = self.audit_logs[-self.config['max_audit_logs']:]
        
        logger.info(f"审计日志: {user_id} {action} {resource_type.value}:{resource_id} - {details}")
    
    def _invalidate_user_cache(self, user_id: str):
        """使用户权限缓存失效"""
        with self.lock:
            if user_id in self.permission_cache:
                del self.permission_cache[user_id]
            self.cache_invalidation_timestamps[user_id] = time.time()
    
    def _get_role_permissions(self, role_id: str) -> Dict[str, List[str]]:
        """获取角色的所有权限（包括继承的权限）"""
        if role_id not in self.roles:
            return {}
        
        role = self.roles[role_id]
        all_permissions = defaultdict(list)
        
        # 添加当前角色的默认权限
        for resource_type, permissions in role.default_permissions.items():
            all_permissions[resource_type].extend(permissions)
        
        # 添加继承角色的权限
        for inherited_role_id in role.inherited_roles:
            inherited_permissions = self._get_role_permissions(inherited_role_id)
            for resource_type, permissions in inherited_permissions.items():
                all_permissions[resource_type].extend(permissions)
        
        # 去重
        for resource_type in all_permissions:
            all_permissions[resource_type] = list(set(all_permissions[resource_type]))
        
        return dict(all_permissions)
    
    def assign_role_to_user(self, user_id: str, role_id: str, assigned_by: str, 
                           scope: Optional[Dict[str, Any]] = None,
                           expires_at: Optional[float] = None) -> bool:
        """分配角色给用户
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            assigned_by: 分配者用户ID
            scope: 作用域限制，如{"model_ids": ["model1", "model2"]}
            expires_at: 过期时间戳
            
        Returns:
            是否分配成功
        """
        with self.lock:
            if role_id not in self.roles:
                error_handler.log_warning(f"角色 {role_id} 不存在", "RBACPermissionManager")
                return False
            
            # 检查是否已分配相同角色
            existing_assignment = next(
                (a for a in self.user_role_assignments[user_id] 
                 if a.role_id == role_id and a.scope == scope),
                None
            )
            
            if existing_assignment:
                # 更新现有分配
                existing_assignment.assigned_at = time.time()
                existing_assignment.assigned_by = assigned_by
                existing_assignment.expires_at = expires_at
                action = "update_role_assignment"
            else:
                # 创建新分配
                assignment = UserRoleAssignment(
                    user_id=user_id,
                    role_id=role_id,
                    assigned_at=time.time(),
                    assigned_by=assigned_by,
                    scope=scope,
                    expires_at=expires_at
                )
                self.user_role_assignments[user_id].append(assignment)
                action = "assign_role"
            
            # 使缓存失效
            self._invalidate_user_cache(user_id)
            
            # 记录审计日志
            self._log_audit(
                user_id=assigned_by,
                action=action,
                resource_type=ResourceType.USER,
                resource_id=user_id,
                details={
                    "role_id": role_id,
                    "scope": scope,
                    "expires_at": expires_at,
                    "assigned_by": assigned_by
                },
                success=True
            )
            
            return True
    
    def remove_role_from_user(self, user_id: str, role_id: str, removed_by: str, 
                             scope: Optional[Dict[str, Any]] = None) -> bool:
        """从用户移除角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            removed_by: 移除者用户ID
            scope: 作用域限制，与分配时一致
            
        Returns:
            是否移除成功
        """
        with self.lock:
            if user_id not in self.user_role_assignments:
                return False
            
            # 查找匹配的分配
            assignments = self.user_role_assignments[user_id]
            removed_count = 0
            
            for i in range(len(assignments) - 1, -1, -1):
                assignment = assignments[i]
                if assignment.role_id == role_id and assignment.scope == scope:
                    assignments.pop(i)
                    removed_count += 1
            
            if removed_count > 0:
                # 使缓存失效
                self._invalidate_user_cache(user_id)
                
                # 记录审计日志
                self._log_audit(
                    user_id=removed_by,
                    action="remove_role",
                    resource_type=ResourceType.USER,
                    resource_id=user_id,
                    details={
                        "role_id": role_id,
                        "scope": scope,
                        "removed_by": removed_by,
                        "removed_count": removed_count
                    },
                    success=True
                )
                
                return True
            
            return False
    
    def grant_direct_permission(self, user_id: str, resource_id: str, 
                               resource_type: ResourceType, permission_type: PermissionType,
                               granted_by: str, expires_at: Optional[float] = None) -> bool:
        """授予直接权限（绕过角色）
        
        Args:
            user_id: 用户ID
            resource_id: 资源ID
            resource_type: 资源类型
            permission_type: 权限类型
            granted_by: 授予者用户ID
            expires_at: 过期时间戳
            
        Returns:
            是否授予成功
        """
        with self.lock:
            # 检查是否已存在相同权限
            existing_permission = next(
                (p for p in self.direct_permissions[user_id] 
                 if p.resource_id == resource_id and 
                 p.resource_type == resource_type and
                 p.permission_type == permission_type),
                None
            )
            
            if existing_permission:
                # 更新现有权限
                existing_permission.granted_at = time.time()
                existing_permission.granted_by = granted_by
                existing_permission.expires_at = expires_at
                action = "update_direct_permission"
            else:
                # 创建新权限
                permission = DirectPermission(
                    user_id=user_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    permission_type=permission_type,
                    granted_at=time.time(),
                    granted_by=granted_by,
                    expires_at=expires_at
                )
                self.direct_permissions[user_id].append(permission)
                action = "grant_direct_permission"
            
            # 使缓存失效
            self._invalidate_user_cache(user_id)
            
            # 记录审计日志
            self._log_audit(
                user_id=granted_by,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details={
                    "target_user_id": user_id,
                    "permission_type": permission_type.value,
                    "expires_at": expires_at,
                    "granted_by": granted_by
                },
                success=True
            )
            
            return True
    
    def revoke_direct_permission(self, user_id: str, resource_id: str, 
                                resource_type: ResourceType, permission_type: PermissionType,
                                revoked_by: str) -> bool:
        """撤销直接权限
        
        Args:
            user_id: 用户ID
            resource_id: 资源ID
            resource_type: 资源类型
            permission_type: 权限类型
            revoked_by: 撤销者用户ID
            
        Returns:
            是否撤销成功
        """
        with self.lock:
            if user_id not in self.direct_permissions:
                return False
            
            permissions = self.direct_permissions[user_id]
            revoked_count = 0
            
            for i in range(len(permissions) - 1, -1, -1):
                permission = permissions[i]
                if (permission.resource_id == resource_id and 
                    permission.resource_type == resource_type and
                    permission.permission_type == permission_type):
                    permissions.pop(i)
                    revoked_count += 1
            
            if revoked_count > 0:
                # 使缓存失效
                self._invalidate_user_cache(user_id)
                
                # 记录审计日志
                self._log_audit(
                    user_id=revoked_by,
                    action="revoke_direct_permission",
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details={
                        "target_user_id": user_id,
                        "permission_type": permission_type.value,
                        "revoked_by": revoked_by,
                        "revoked_count": revoked_count
                    },
                    success=True
                )
                
                return True
            
            return False
    
    def get_user_permissions(self, user_id: str, use_cache: bool = True) -> Dict[str, Dict[str, Set[str]]]:
        """获取用户的所有权限
        
        Args:
            user_id: 用户ID
            use_cache: 是否使用缓存
            
        Returns:
            权限字典 {resource_type: {resource_id: set(permission_types)}}
        """
        # 检查缓存
        if (use_cache and self.config['enable_permission_caching'] and 
            user_id in self.permission_cache and
            time.time() - self.cache_invalidation_timestamps.get(user_id, 0) < self.config['cache_ttl_seconds']):
            return self.permission_cache[user_id]
        
        with self.lock:
            permissions = defaultdict(lambda: defaultdict(set))
            current_time = time.time()
            
            # 1. 从角色获取权限
            if user_id in self.user_role_assignments:
                for assignment in self.user_role_assignments[user_id]:
                    # 检查是否过期
                    if assignment.expires_at and assignment.expires_at < current_time:
                        continue
                    
                    # 获取角色权限
                    role_permissions = self._get_role_permissions(assignment.role_id)
                    
                    # 应用作用域限制
                    scope = assignment.scope
                    for resource_type_str, perm_list in role_permissions.items():
                        resource_type = ResourceType(resource_type_str)
                        
                        # 检查是否有作用域限制
                        if scope and 'model_ids' in scope and resource_type == ResourceType.MODEL:
                            # 只授予特定模型的权限
                            for model_id in scope['model_ids']:
                                for perm in perm_list:
                                    permissions[resource_type.value][model_id].add(perm)
                        elif scope and 'task_ids' in scope and resource_type == ResourceType.TASK:
                            # 只授予特定任务的权限
                            for task_id in scope['task_ids']:
                                for perm in perm_list:
                                    permissions[resource_type.value][task_id].add(perm)
                        else:
                            # 无作用域限制，授予所有资源的权限（用'*'表示）
                            for perm in perm_list:
                                permissions[resource_type.value]['*'].add(perm)
            
            # 2. 从直接权限获取
            if user_id in self.direct_permissions:
                for permission in self.direct_permissions[user_id]:
                    # 检查是否过期
                    if permission.expires_at and permission.expires_at < current_time:
                        continue
                    
                    permissions[permission.resource_type.value][permission.resource_id].add(
                        permission.permission_type.value
                    )
            
            # 转换为普通字典
            result = {}
            for resource_type, resource_perms in permissions.items():
                result[resource_type] = {}
                for resource_id, perm_set in resource_perms.items():
                    result[resource_type][resource_id] = set(perm_set)
            
            # 更新缓存
            if self.config['enable_permission_caching']:
                self.permission_cache[user_id] = result
                self.cache_invalidation_timestamps[user_id] = current_time
            
            return result
    
    def check_permission(self, user_id: str, resource_type: ResourceType, 
                        resource_id: str, permission_type: PermissionType,
                        use_cache: bool = True) -> bool:
        """检查用户是否拥有特定权限
        
        Args:
            user_id: 用户ID
            resource_type: 资源类型
            resource_id: 资源ID
            permission_type: 权限类型
            use_cache: 是否使用缓存
            
        Returns:
            是否拥有权限
        """
        permissions = self.get_user_permissions(user_id, use_cache)
        resource_type_str = resource_type.value
        permission_type_str = permission_type.value
        
        if resource_type_str not in permissions:
            return False
        
        # 检查特定资源的权限
        if resource_id in permissions[resource_type_str]:
            if permission_type_str in permissions[resource_type_str][resource_id]:
                return True
        
        # 检查通配符权限（所有资源）
        if '*' in permissions[resource_type_str]:
            if permission_type_str in permissions[resource_type_str]['*']:
                return True
        
        # 检查MANAGE权限（包含所有权限）
        if resource_id in permissions[resource_type_str]:
            if PermissionType.MANAGE.value in permissions[resource_type_str][resource_id]:
                return True
        
        if '*' in permissions[resource_type_str]:
            if PermissionType.MANAGE.value in permissions[resource_type_str]['*']:
                return True
        
        return False
    
    def create_custom_role(self, role_name: str, role_type: RoleType, description: str,
                          inherited_roles: List[str], default_permissions: Dict[str, List[str]],
                          created_by: str) -> Optional[str]:
        """创建自定义角色
        
        Args:
            role_name: 角色名称
            role_type: 角色类型
            description: 角色描述
            inherited_roles: 继承的角色ID列表
            default_permissions: 默认权限 {resource_type: [permission_type]}
            created_by: 创建者用户ID
            
        Returns:
            角色ID，如果创建失败则返回None
        """
        with self.lock:
            # 验证继承的角色
            for inherited_role_id in inherited_roles:
                if inherited_role_id not in self.roles:
                    error_handler.log_warning(f"继承的角色 {inherited_role_id} 不存在", "RBACPermissionManager")
                    return None
            
            # 生成角色ID
            role_id = f"role_custom_{hashlib.md5(f'{role_name}_{time.time()}'.encode()).hexdigest()[:8]}"
            
            # 创建角色
            role = Role(
                role_id=role_id,
                role_name=role_name,
                role_type=role_type,
                description=description,
                inherited_roles=inherited_roles,
                default_permissions=default_permissions,
                created_at=time.time(),
                updated_at=time.time(),
                is_system_role=False
            )
            
            self.roles[role_id] = role
            
            # 记录审计日志
            self._log_audit(
                user_id=created_by,
                action="create_role",
                resource_type=ResourceType.SYSTEM,
                resource_id="roles",
                details={
                    "role_id": role_id,
                    "role_name": role_name,
                    "role_type": role_type.value,
                    "inherited_roles": inherited_roles,
                    "created_by": created_by
                },
                success=True
            )
            
            return role_id
    
    def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的角色列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            角色信息列表
        """
        with self.lock:
            if user_id not in self.user_role_assignments:
                return []
            
            roles_info = []
            current_time = time.time()
            
            for assignment in self.user_role_assignments[user_id]:
                # 检查是否过期
                if assignment.expires_at and assignment.expires_at < current_time:
                    continue
                
                if assignment.role_id in self.roles:
                    role = self.roles[assignment.role_id]
                    roles_info.append({
                        'role_id': role.role_id,
                        'role_name': role.role_name,
                        'role_type': role.role_type.value,
                        'description': role.description,
                        'assigned_at': assignment.assigned_at,
                        'assigned_by': assignment.assigned_by,
                        'scope': assignment.scope,
                        'expires_at': assignment.expires_at
                    })
            
            return roles_info
    
    def get_audit_logs(self, user_id: str = None, resource_type: ResourceType = None,
                      resource_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志
        
        Args:
            user_id: 过滤用户ID
            resource_type: 过滤资源类型
            resource_id: 过滤资源ID
            limit: 返回最大数量
            
        Returns:
            审计日志列表
        """
        with self.lock:
            filtered_logs = self.audit_logs
            
            # 过滤
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            if resource_type:
                filtered_logs = [log for log in filtered_logs if log.resource_type == resource_type]
            if resource_id:
                filtered_logs = [log for log in filtered_logs if log.resource_id == resource_id]
            
            # 排序（最新的在前）
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 限制数量
            filtered_logs = filtered_logs[:limit]
            
            # 转换为字典
            result = []
            for log in filtered_logs:
                log_dict = asdict(log)
                log_dict['resource_type'] = log.resource_type.value
                result.append(log_dict)
            
            return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            total_users = len(self.user_role_assignments)
            total_direct_permissions = sum(len(perms) for perms in self.direct_permissions.values())
            total_audit_logs = len(self.audit_logs)
            
            # 统计各角色分配数量
            role_distribution = defaultdict(int)
            for assignments in self.user_role_assignments.values():
                for assignment in assignments:
                    role_distribution[assignment.role_id] += 1
            
            status = {
                'total_roles': len(self.roles),
                'total_users': total_users,
                'total_role_assignments': sum(len(assignments) for assignments in self.user_role_assignments.values()),
                'total_direct_permissions': total_direct_permissions,
                'total_audit_logs': total_audit_logs,
                'role_distribution': dict(role_distribution),
                'cache_enabled': self.config['enable_permission_caching'],
                'audit_enabled': self.config['audit_enabled'],
                'cache_size': len(self.permission_cache),
                'last_updated': time.time(),
            }
            
            return status


# 全局实例
_rbac_permission_manager_instance = None

def get_rbac_permission_manager() -> RBACPermissionManager:
    """获取全局RBAC权限管理器实例"""
    global _rbac_permission_manager_instance
    if _rbac_permission_manager_instance is None:
        _rbac_permission_manager_instance = RBACPermissionManager()
    return _rbac_permission_manager_instance