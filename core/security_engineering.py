"""
安全校验与工程化保障模块 - Security and Engineering Guarantees

提供AGI系统的安全校验、版本回滚、演化约束等工程化保障能力，解决现有系统缺乏安全防护的问题。

核心功能：
1. 安全访问控制和权限管理
2. 系统版本管理和回滚机制
3. 演化约束和边界控制
4. 异常检测和自动恢复
5. 审计日志和合规性检查
6. 数据加密和隐私保护

设计目标：
- 建立多层次安全防护体系
- 实现可追溯的系统演化管理
- 提供自动化的异常恢复能力
- 确保系统运行的稳定性和可靠性
"""

import asyncio
import time
import logging
import threading
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
import pickle
import copy
import sqlite3

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """安全级别"""
    PUBLIC = "public"          # 公开级别
    INTERNAL = "internal"      # 内部级别
    CONFIDENTIAL = "confidential"  # 机密级别
    RESTRICTED = "restricted"  # 受限级别
    TOP_SECRET = "top_secret"  # 最高机密

class OperationType(Enum):
    """操作类型"""
    READ = "read"              # 读取操作
    WRITE = "write"            # 写入操作
    EXECUTE = "execute"        # 执行操作
    DELETE = "delete"          # 删除操作
    MODIFY = "modify"          # 修改操作
    CREATE = "create"          # 创建操作

@dataclass
class SecurityPolicy:
    """安全策略"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class AccessControlEntry:
    """访问控制条目"""
    ace_id: str
    principal: str  # 主体（用户/服务）
    resource: str   # 资源
    operation: OperationType
    allowed: bool = True
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    expires_at: Optional[float] = None

@dataclass
class SystemSnapshot:
    """系统快照"""
    snapshot_id: str
    timestamp: float
    description: str
    components: List[str]
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    timestamp: float
    principal: str
    operation: str
    resource: str
    result: str  # 'success', 'failure'
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class SecurityEngine:
    """安全引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化安全引擎"""
        self.config = config or self._get_default_config()
        
        # 访问控制列表
        self.access_control_list: Dict[str, AccessControlEntry] = {}
        
        # 安全策略
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self._init_default_policies()
        
        # 系统快照
        self.system_snapshots: Dict[str, SystemSnapshot] = {}
        self.max_snapshots = self.config.get("max_snapshots", 100)
        
        # 审计日志
        self.audit_logs = deque(maxlen=self.config.get("max_audit_logs", 10000))
        self.audit_db_path = self.config.get("audit_db_path", "security_audit.db")
        self._init_audit_database()
        
        # 加密密钥管理
        self.encryption_keys: Dict[str, str] = {}
        self._init_encryption_keys()
        
        # 异常检测
        self.anomaly_detectors: Dict[str, Any] = {}
        self._init_anomaly_detectors()
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = self.config.get("session_timeout", 3600)
        
        # 演化约束
        self.evolution_constraints: List[Dict[str, Any]] = []
        self._init_evolution_constraints()
        
        logger.info("安全引擎初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "max_snapshots": 100,
            "max_audit_logs": 10000,
            "audit_db_path": "security_audit.db",
            "session_timeout": 3600,
            "encryption_algorithm": "SHA-256",
            "enable_anomaly_detection": True,
            "enable_auto_recovery": True,
            "enable_access_control": True,
            "enable_audit_logging": True
        }
    
    def _init_default_policies(self):
        """初始化默认安全策略"""
        # 默认访问控制策略
        default_policy = SecurityPolicy(
            policy_id="default_access_control",
            name="默认访问控制策略",
            description="控制对系统资源的访问",
            rules=[
                {
                    "effect": "allow",
                    "principal": "system_admin",
                    "resource": "*",
                    "operation": "*"
                },
                {
                    "effect": "deny",
                    "principal": "guest",
                    "resource": "security.*",
                    "operation": ["write", "delete", "modify"]
                }
            ]
        )
        self.security_policies[default_policy.policy_id] = default_policy
        
        # 数据保护策略
        data_protection_policy = SecurityPolicy(
            policy_id="data_protection",
            name="数据保护策略",
            description="保护敏感数据和隐私信息",
            rules=[
                {
                    "effect": "require_encryption",
                    "resource_type": "sensitive_data",
                    "encryption_level": "high"
                },
                {
                    "effect": "log_access",
                    "resource_type": "confidential_data",
                    "log_level": "detailed"
                }
            ]
        )
        self.security_policies[data_protection_policy.policy_id] = data_protection_policy
        
        logger.info(f"初始化了 {len(self.security_policies)} 个默认安全策略")
    
    def _init_audit_database(self):
        """初始化审计数据库"""
        try:
            conn = sqlite3.connect(self.audit_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    principal TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_principal ON audit_logs(principal)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_operation ON audit_logs(operation)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"初始化审计数据库失败: {e}")
    
    def _init_encryption_keys(self):
        """初始化加密密钥"""
        # 生成默认密钥（在实际应用中应从安全存储加载）
        key_id = "default_encryption_key"
        key_value = secrets.token_hex(32)  # 256位密钥
        self.encryption_keys[key_id] = key_value
        
        logger.info("加密密钥初始化完成")
    
    def _init_anomaly_detectors(self):
        """初始化异常检测器"""
        # 访问频率异常检测器
        self.anomaly_detectors["access_frequency"] = {
            "window_size": 300,  # 5分钟窗口
            "threshold": 100,     # 最大访问次数
            "access_counts": defaultdict(int),
            "last_reset": time.time()
        }
        
        # 失败尝试检测器
        self.anomaly_detectors["failed_attempts"] = {
            "window_size": 900,  # 15分钟窗口
            "threshold": 10,      # 最大失败尝试次数
            "failed_counts": defaultdict(int),
            "last_reset": time.time()
        }
        
        logger.info("异常检测器初始化完成")
    
    def _init_evolution_constraints(self):
        """初始化演化约束"""
        # 性能约束：CPU使用率不能持续超过90%
        self.evolution_constraints.append({
            "constraint_id": "performance_cpu",
            "type": "performance",
            "metric": "cpu_usage",
            "condition": "average < 90",
            "window": 300,  # 5分钟窗口
            "action": "throttle_evolution"
        })
        
        # 安全约束：安全漏洞不能增加
        self.evolution_constraints.append({
            "constraint_id": "security_vulnerabilities",
            "type": "security",
            "metric": "vulnerability_count",
            "condition": "count <= previous_count",
            "window": 86400,  # 24小时窗口
            "action": "rollback_if_increased"
        })
        
        # 稳定性约束：错误率不能超过阈值
        self.evolution_constraints.append({
            "constraint_id": "stability_error_rate",
            "type": "stability",
            "metric": "error_rate",
            "condition": "rate < 0.01",  # 1%错误率
            "window": 3600,  # 1小时窗口
            "action": "pause_evolution"
        })
        
        logger.info(f"初始化了 {len(self.evolution_constraints)} 个演化约束")
    
    def check_access(self, principal: str, resource: str, 
                    operation: OperationType, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """检查访问权限"""
        context = context or {}
        
        # 记录访问尝试
        access_attempt = {
            "principal": principal,
            "resource": resource,
            "operation": operation.value,
            "timestamp": time.time(),
            "context": context
        }
        
        # 检查异常访问模式
        anomaly_detected = self._detect_access_anomaly(principal, resource, operation)
        if anomaly_detected:
            self.log_audit_event(
                principal=principal,
                operation=f"anomaly_detected_{operation.value}",
                resource=resource,
                result="failure",
                details={
                    "reason": "anomaly_detected",
                    "anomaly_type": anomaly_detected["type"],
                    "threshold": anomaly_detected["threshold"],
                    "actual": anomaly_detected["actual"]
                }
            )
            
            return {
                "allowed": False,
                "reason": "anomaly_detected",
                "anomaly_details": anomaly_detected,
                "timestamp": time.time()
            }
        
        # 检查访问控制列表
        allowed = self._evaluate_access_control(principal, resource, operation, context)
        
        # 记录审计日志
        self.log_audit_event(
            principal=principal,
            operation=operation.value,
            resource=resource,
            result="success" if allowed else "failure",
            details={
                "access_attempt": access_attempt,
                "access_control_result": allowed,
                "context": context
            }
        )
        
        return {
            "allowed": allowed,
            "timestamp": time.time(),
            "principal": principal,
            "resource": resource,
            "operation": operation.value
        }
    
    def _detect_access_anomaly(self, principal: str, resource: str, 
                              operation: OperationType) -> Optional[Dict[str, Any]]:
        """检测访问异常"""
        current_time = time.time()
        
        # 检查访问频率异常
        freq_detector = self.anomaly_detectors["access_frequency"]
        window_size = freq_detector["window_size"]
        
        # 重置过期计数
        if current_time - freq_detector["last_reset"] > window_size:
            freq_detector["access_counts"].clear()
            freq_detector["last_reset"] = current_time
        
        # 更新计数
        key = f"{principal}:{resource}:{operation.value}"
        freq_detector["access_counts"][key] += 1
        
        # 检查阈值
        if freq_detector["access_counts"][key] > freq_detector["threshold"]:
            return {
                "type": "access_frequency",
                "threshold": freq_detector["threshold"],
                "actual": freq_detector["access_counts"][key],
                "window_size": window_size
            }
        
        return None
    
    def _evaluate_access_control(self, principal: str, resource: str,
                               operation: OperationType, context: Dict[str, Any]) -> bool:
        """评估访问控制"""
        # 首先检查显式允许的规则
        for ace_id, ace in self.access_control_list.items():
            if ace.principal == principal and ace.resource == resource:
                if operation == ace.operation or ace.operation == OperationType(operation.value):
                    # 检查条件
                    if self._evaluate_ace_conditions(ace, context):
                        return ace.allowed
        
        # 然后检查安全策略
        for policy_id, policy in self.security_policies.items():
            if not policy.enabled:
                continue
            
            for rule in policy.rules:
                if self._evaluate_security_rule(rule, principal, resource, operation, context):
                    effect = rule.get("effect", "allow")
                    return effect == "allow"
        
        # 默认拒绝
        return False
    
    def _evaluate_ace_conditions(self, ace: AccessControlEntry, context: Dict[str, Any]) -> bool:
        """评估ACE条件"""
        if not ace.conditions:
            return True
        
        for condition in ace.conditions:
            condition_type = condition.get("type")
            condition_value = condition.get("value")
            
            if condition_type == "time_range":
                current_hour = datetime.now().hour
                start_hour = condition_value.get("start", 0)
                end_hour = condition_value.get("end", 24)
                
                if not (start_hour <= current_hour <= end_hour):
                    return False
            
            elif condition_type == "ip_range":
                client_ip = context.get("ip_address", "")
                allowed_ips = condition_value.get("ips", [])
                
                if client_ip not in allowed_ips:
                    return False
            
            elif condition_type == "security_level":
                required_level = condition_value.get("level", "internal")
                user_level = context.get("security_level", "public")
                
                # 检查用户级别是否满足要求
                level_priority = {
                    "public": 0,
                    "internal": 1,
                    "confidential": 2,
                    "restricted": 3,
                    "top_secret": 4
                }
                
                if level_priority.get(user_level, 0) < level_priority.get(required_level, 0):
                    return False
        
        return True
    
    def _evaluate_security_rule(self, rule: Dict[str, Any], principal: str, resource: str,
                              operation: OperationType, context: Dict[str, Any]) -> bool:
        """评估安全规则"""
        # 检查主体匹配
        rule_principal = rule.get("principal")
        if rule_principal and rule_principal != "*":
            if rule_principal != principal:
                return False
        
        # 检查资源匹配
        rule_resource = rule.get("resource")
        if rule_resource and rule_resource != "*":
            # 支持通配符匹配
            if not self._wildcard_match(resource, rule_resource):
                return False
        
        # 检查操作匹配
        rule_operation = rule.get("operation")
        if rule_operation and rule_operation != "*":
            if isinstance(rule_operation, list):
                if operation.value not in rule_operation:
                    return False
            elif rule_operation != operation.value:
                return False
        
        return True
    
    def _wildcard_match(self, text: str, pattern: str) -> bool:
        """通配符匹配"""
        # 简化实现，支持 * 通配符
        import re
        pattern_regex = pattern.replace("*", ".*").replace("?", ".")
        return re.match(f"^{pattern_regex}$", text) is not None
    
    def log_audit_event(self, principal: str, operation: str, resource: str,
                       result: str, details: Dict[str, Any],
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None):
        """记录审计事件"""
        log_id = f"audit_{int(time.time())}_{hashlib.md5(f'{principal}{operation}{resource}'.encode()).hexdigest()[:8]}"
        
        audit_log = AuditLog(
            log_id=log_id,
            timestamp=time.time(),
            principal=principal,
            operation=operation,
            resource=resource,
            result=result,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # 添加到内存日志
        self.audit_logs.append(audit_log)
        
        # 存储到数据库
        self._store_audit_log_to_db(audit_log)
        
        # 检查是否需要触发警报
        if result == "failure":
            self._trigger_security_alert(audit_log)
    
    def _store_audit_log_to_db(self, audit_log: AuditLog):
        """存储审计日志到数据库"""
        try:
            conn = sqlite3.connect(self.audit_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs 
                (log_id, timestamp, principal, operation, resource, result, details_json, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_log.log_id,
                audit_log.timestamp,
                audit_log.principal,
                audit_log.operation,
                audit_log.resource,
                audit_log.result,
                json.dumps(audit_log.details, default=str),
                audit_log.ip_address,
                audit_log.user_agent
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"存储审计日志失败: {e}")
    
    def _trigger_security_alert(self, audit_log: AuditLog):
        """触发安全警报"""
        # 根据失败类型决定警报级别
        alert_level = "warning"
        
        # 检查是否是重复失败
        recent_failures = [
            log for log in self.audit_logs
            if log.principal == audit_log.principal and 
               log.result == "failure" and
               audit_log.timestamp - log.timestamp < 300  # 5分钟内
        ]
        
        if len(recent_failures) > 5:
            alert_level = "critical"
        
        # 创建警报
        alert_id = f"security_alert_{int(time.time())}"
        alert_details = {
            "alert_id": alert_id,
            "level": alert_level,
            "principal": audit_log.principal,
            "operation": audit_log.operation,
            "resource": audit_log.resource,
            "timestamp": audit_log.timestamp,
            "failure_count": len(recent_failures),
            "details": audit_log.details
        }
        
        logger.warning(f"安全警报 [{alert_level}]: {alert_details}")
        
        # 触发警报处理
        self._handle_security_alert(alert_details)
    
    def _handle_security_alert(self, alert_details: Dict[str, Any]):
        """处理安全警报"""
        alert_level = alert_details["level"]
        
        if alert_level == "critical":
            # 关键警报：可能需要立即行动
            self._take_critical_security_action(alert_details)
        elif alert_level == "warning":
            # 警告：记录并监控
            self._monitor_security_issue(alert_details)
    
    def _take_critical_security_action(self, alert_details: Dict[str, Any]):
        """采取关键安全行动"""
        principal = alert_details["principal"]
        
        # 临时阻止该主体的访问
        block_ace = AccessControlEntry(
            ace_id=f"block_{principal}_{int(time.time())}",
            principal=principal,
            resource="*",
            operation=OperationType.READ,
            allowed=False,
            expires_at=time.time() + 3600  # 阻止1小时
        )
        
        self.access_control_list[block_ace.ace_id] = block_ace
        
        logger.critical(f"已临时阻止主体 {principal} 的访问")
    
    def _monitor_security_issue(self, alert_details: Dict[str, Any]):
        """监控安全问题"""
        # 记录到监控系统
        logger.warning(f"监控安全问题: {alert_details}")
    
    def create_snapshot(self, component: str, data: Any, description: str = "") -> SystemSnapshot:
        """创建系统快照"""
        # 序列化数据
        try:
            serialized_data = pickle.dumps(data)
        except Exception as e:
            logger.error(f"序列化数据失败: {e}")
            serialized_data = b""
        
        # 计算数据哈希
        data_hash = hashlib.sha256(serialized_data).hexdigest()
        
        snapshot_id = f"snapshot_{int(time.time())}_{component}"
        
        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            description=description,
            components=[component],
            data_hash=data_hash,
            metadata={
                "component": component,
                "data_size": len(serialized_data),
                "description": description
            }
        )
        
        # 存储快照
        self.system_snapshots[snapshot_id] = snapshot
        
        # 序列化数据存储（在实际应用中应存储到文件或数据库）
        snapshot_file = f"snapshots/{snapshot_id}.snapshot"
        try:
            import os
            os.makedirs("snapshots", exist_ok=True)
            with open(snapshot_file, 'wb') as f:
                f.write(serialized_data)
            
            snapshot.metadata["storage_path"] = snapshot_file
        except Exception as e:
            logger.error(f"存储快照数据失败: {e}")
        
        # 清理旧快照
        self._cleanup_old_snapshots()
        
        logger.info(f"创建系统快照: {snapshot_id} for {component}")
        
        return snapshot
    
    def _cleanup_old_snapshots(self):
        """清理旧快照"""
        if len(self.system_snapshots) <= self.max_snapshots:
            return
        
        # 按时间排序，删除最旧的快照
        snapshots_by_time = sorted(
            self.system_snapshots.items(),
            key=lambda x: x[1].timestamp
        )
        
        # 计算需要删除的数量
        to_delete = len(self.system_snapshots) - self.max_snapshots
        
        for i in range(to_delete):
            snapshot_id, snapshot = snapshots_by_time[i]
            
            # 删除存储的文件
            storage_path = snapshot.metadata.get("storage_path")
            if storage_path:
                try:
                    import os
                    if os.path.exists(storage_path):
                        os.remove(storage_path)
                except Exception as e:
                    logger.error(f"删除快照文件失败 {storage_path}: {e}")
            
            # 从内存中删除
            del self.system_snapshots[snapshot_id]
            
            logger.info(f"清理旧快照: {snapshot_id}")
    
    def restore_snapshot(self, snapshot_id: str) -> Optional[Any]:
        """恢复系统快照"""
        if snapshot_id not in self.system_snapshots:
            logger.error(f"快照不存在: {snapshot_id}")
            return None
        
        snapshot = self.system_snapshots[snapshot_id]
        storage_path = snapshot.metadata.get("storage_path")
        
        if not storage_path:
            logger.error(f"快照存储路径不存在: {snapshot_id}")
            return None
        
        try:
            # 加载快照数据
            with open(storage_path, 'rb') as f:
                serialized_data = f.read()
            
            # 验证数据完整性
            current_hash = hashlib.sha256(serialized_data).hexdigest()
            if current_hash != snapshot.data_hash:
                logger.error(f"快照数据完整性验证失败: {snapshot_id}")
                return None
            
            # 反序列化数据
            data = pickle.loads(serialized_data)
            
            logger.info(f"恢复系统快照: {snapshot_id}")
            
            return data
            
        except Exception as e:
            logger.error(f"恢复快照失败 {snapshot_id}: {e}")
            return None
    
    def encrypt_data(self, data: str, key_id: str = "default_encryption_key") -> Dict[str, Any]:
        """加密数据"""
        if key_id not in self.encryption_keys:
            logger.error(f"加密密钥不存在: {key_id}")
            return {"error": "encryption_key_not_found"}
        
        try:
            # 简化加密实现（实际应用中应使用更安全的加密算法）
            key = self.encryption_keys[key_id]
            encoded_data = data.encode('utf-8')
            
            # 使用HMAC进行数据完整性保护
            hmac_obj = hmac.new(key.encode('utf-8'), encoded_data, hashlib.sha256)
            hmac_digest = hmac_obj.digest()
            
            # 简单加密：XOR操作（仅示例，实际应用应使用AES等加密算法）
            key_bytes = key.encode('utf-8')
            encrypted_bytes = bytearray()
            
            for i, byte in enumerate(encoded_data):
                key_byte = key_bytes[i % len(key_bytes)]
                encrypted_bytes.append(byte ^ key_byte)
            
            encrypted_data = base64.b64encode(bytes(encrypted_bytes)).decode('utf-8')
            hmac_base64 = base64.b64encode(hmac_digest).decode('utf-8')
            
            return {
                "encrypted_data": encrypted_data,
                "hmac": hmac_base64,
                "key_id": key_id,
                "algorithm": "XOR-HMAC-SHA256",  # 仅示例
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"加密数据失败: {e}")
            return {"error": str(e)}
    
    def decrypt_data(self, encrypted_data: str, hmac_value: str, 
                    key_id: str = "default_encryption_key") -> Dict[str, Any]:
        """解密数据"""
        if key_id not in self.encryption_keys:
            logger.error(f"解密密钥不存在: {key_id}")
            return {"error": "decryption_key_not_found"}
        
        try:
            key = self.encryption_keys[key_id]
            
            # 解码base64数据
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            expected_hmac = base64.b64decode(hmac_value.encode('utf-8'))
            
            # 解密：XOR操作
            key_bytes = key.encode('utf-8')
            decrypted_bytes = bytearray()
            
            for i, byte in enumerate(encrypted_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                decrypted_bytes.append(byte ^ key_byte)
            
            decrypted_data = bytes(decrypted_bytes)
            
            # 验证HMAC
            hmac_obj = hmac.new(key.encode('utf-8'), decrypted_data, hashlib.sha256)
            calculated_hmac = hmac_obj.digest()
            
            if not hmac.compare_digest(calculated_hmac, expected_hmac):
                logger.error("HMAC验证失败：数据可能被篡改")
                return {"error": "hmac_validation_failed"}
            
            return {
                "decrypted_data": decrypted_data.decode('utf-8'),
                "verified": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"解密数据失败: {e}")
            return {"error": str(e)}
    
    def check_evolution_constraints(self, evolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查演化约束"""
        constraint_violations = []
        
        for constraint in self.evolution_constraints:
            constraint_id = constraint["constraint_id"]
            constraint_type = constraint["type"]
            metric = constraint["metric"]
            condition = constraint["condition"]
            
            # 获取当前指标值
            current_value = self._get_metric_value(metric, evolution_data)
            
            # 评估条件
            satisfied = self._evaluate_constraint_condition(condition, current_value, constraint)
            
            if not satisfied:
                violation = {
                    "constraint_id": constraint_id,
                    "type": constraint_type,
                    "metric": metric,
                    "condition": condition,
                    "current_value": current_value,
                    "action": constraint.get("action", "warn")
                }
                constraint_violations.append(violation)
        
        result = {
            "timestamp": time.time(),
            "total_constraints": len(self.evolution_constraints),
            "violations": constraint_violations,
            "evolution_allowed": len(constraint_violations) == 0
        }
        
        if constraint_violations:
            logger.warning(f"演化约束检查发现 {len(constraint_violations)} 个违规")
            
            # 根据违规类型采取行动
            for violation in constraint_violations:
                self._handle_constraint_violation(violation)
        
        return result
    
    def _get_metric_value(self, metric: str, evolution_data: Dict[str, Any]) -> Any:
        """获取指标值"""
        # 简化实现：从演化数据中获取指标值
        # 实际应用中可能需要从监控系统获取实时指标
        
        metric_mapping = {
            "cpu_usage": evolution_data.get("performance_metrics", {}).get("cpu_usage", 0.0),
            "memory_usage": evolution_data.get("performance_metrics", {}).get("memory_usage", 0.0),
            "error_rate": evolution_data.get("stability_metrics", {}).get("error_rate", 0.0),
            "vulnerability_count": evolution_data.get("security_metrics", {}).get("vulnerability_count", 0),
            "response_time": evolution_data.get("performance_metrics", {}).get("avg_response_time", 0.0)
        }
        
        return metric_mapping.get(metric, 0.0)
    
    def _evaluate_constraint_condition(self, condition: str, current_value: Any, 
                                     constraint: Dict[str, Any]) -> bool:
        """评估约束条件"""
        try:
            # 解析条件字符串
            # 支持简单条件：<, >, <=, >=, ==, !=
            import re
            
            # 匹配条件模式
            pattern = r'([a-zA-Z_]+)\s*([<>=!]+)\s*([\d\.]+)'
            match = re.match(pattern, condition)
            
            if match:
                metric_name = match.group(1)
                operator = match.group(2)
                threshold = float(match.group(3))
                
                # 评估条件
                if operator == "<":
                    return current_value < threshold
                elif operator == ">":
                    return current_value > threshold
                elif operator == "<=":
                    return current_value <= threshold
                elif operator == ">=":
                    return current_value >= threshold
                elif operator == "==":
                    return current_value == threshold
                elif operator == "!=":
                    return current_value != threshold
            
            # 处理其他条件类型
            if condition == "count <= previous_count":
                # 需要历史数据，这里简化处理
                return True
            
            if condition == "average < 90":
                # 需要计算平均值，这里简化处理
                return current_value < 90
            
            if condition == "rate < 0.01":
                return current_value < 0.01
            
        except Exception as e:
            logger.error(f"评估约束条件失败: {condition}, 错误: {e}")
        
        # 默认返回True，避免过度限制
        return True
    
    def _handle_constraint_violation(self, violation: Dict[str, Any]):
        """处理约束违规"""
        action = violation["action"]
        constraint_id = violation["constraint_id"]
        
        if action == "throttle_evolution":
            logger.warning(f"演化约束违规 [{constraint_id}]: 限速演化进程")
            # 实际应用中这里应该调整演化速度
        
        elif action == "rollback_if_increased":
            logger.warning(f"演化约束违规 [{constraint_id}]: 检查是否需要回滚")
            # 实际应用中这里应该触发回滚检查
        
        elif action == "pause_evolution":
            logger.warning(f"演化约束违规 [{constraint_id}]: 暂停演化进程")
            # 实际应用中这里应该暂停演化
    
    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        # 计算安全指标
        recent_failures = [
            log for log in self.audit_logs
            if log.result == "failure" and time.time() - log.timestamp < 3600
        ]
        
        active_sessions_count = len(self.active_sessions)
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if time.time() - session.get("last_activity", 0) > self.session_timeout
        ]
        
        # 清理过期会话
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        return {
            "timestamp": time.time(),
            "access_control_rules": len(self.access_control_list),
            "security_policies": len(self.security_policies),
            "system_snapshots": len(self.system_snapshots),
            "audit_logs": len(self.audit_logs),
            "recent_failures": len(recent_failures),
            "active_sessions": active_sessions_count,
            "encryption_keys": len(self.encryption_keys),
            "evolution_constraints": len(self.evolution_constraints),
            "anomaly_detectors": len(self.anomaly_detectors),
            "overall_security_level": self._calculate_overall_security_level(recent_failures)
        }
    
    def _calculate_overall_security_level(self, recent_failures: List[AuditLog]) -> str:
        """计算总体安全级别"""
        if len(recent_failures) > 10:
            return "critical"
        elif len(recent_failures) > 5:
            return "high"
        elif len(recent_failures) > 2:
            return "medium"
        else:
            return "low"
    
    def add_access_control_entry(self, principal: str, resource: str, 
                               operation: OperationType, allowed: bool = True,
                               conditions: List[Dict[str, Any]] = None,
                               expires_at: Optional[float] = None) -> str:
        """添加访问控制条目"""
        ace_id = f"ace_{int(time.time())}_{hashlib.md5(f'{principal}{resource}'.encode()).hexdigest()[:8]}"
        
        ace = AccessControlEntry(
            ace_id=ace_id,
            principal=principal,
            resource=resource,
            operation=operation,
            allowed=allowed,
            conditions=conditions or [],
            expires_at=expires_at
        )
        
        self.access_control_list[ace_id] = ace
        
        logger.info(f"添加访问控制条目: {ace_id}")
        
        return ace_id
    
    def remove_access_control_entry(self, ace_id: str) -> bool:
        """移除访问控制条目"""
        if ace_id in self.access_control_list:
            del self.access_control_list[ace_id]
            logger.info(f"移除访问控制条目: {ace_id}")
            return True
        
        return False
    
    def create_session(self, principal: str, session_data: Dict[str, Any] = None) -> str:
        """创建会话"""
        session_id = f"session_{int(time.time())}_{secrets.token_hex(8)}"
        
        session = {
            "session_id": session_id,
            "principal": principal,
            "created_at": time.time(),
            "last_activity": time.time(),
            "data": session_data or {},
            "ip_address": session_data.get("ip_address") if session_data else None,
            "user_agent": session_data.get("user_agent") if session_data else None
        }
        
        self.active_sessions[session_id] = session
        
        logger.info(f"创建会话: {session_id} for {principal}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """验证会话"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # 检查会话是否过期
        if time.time() - session["last_activity"] > self.session_timeout:
            del self.active_sessions[session_id]
            return False
        
        # 更新最后活动时间
        session["last_activity"] = time.time()
        
        return True
    
    def destroy_session(self, session_id: str) -> bool:
        """销毁会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"销毁会话: {session_id}")
            return True
        
        return False


# 全局安全引擎实例
_security_engine_instance = None

def get_security_engine() -> SecurityEngine:
    """获取安全引擎实例"""
    global _security_engine_instance
    if _security_engine_instance is None:
        _security_engine_instance = SecurityEngine()
    return _security_engine_instance

def initialize_security_engine(config: Optional[Dict[str, Any]] = None) -> SecurityEngine:
    """初始化安全引擎"""
    global _security_engine_instance
    _security_engine_instance = SecurityEngine(config)
    return _security_engine_instance
