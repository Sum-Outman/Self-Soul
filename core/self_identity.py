"""
统一自我身份系统 - Unified Self Identity System

实现30天变强版本计划的第一优先级：
1. 统一状态与持久自我：做一个唯一不变的 self_id
2. 记忆、人格、目标全部绑定它
3. 重启不丢、长期可追溯

核心特性：
- 唯一不变的自我ID (self_id)
- 持久化存储和恢复
- 人格特征和偏好系统
- 目标绑定和追踪
- 记忆关联和检索
- 重启恢复和版本控制
"""

import os
import json
import uuid
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle
import base64
from pathlib import Path

# 导入现有的配置和错误处理
try:
    from core.error_handling import error_handler
except ImportError:
    error_handler = None

try:
    from core.config_manager import ConfigManager
except ImportError:
    ConfigManager = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentityType(Enum):
    """身份类型枚举"""
    SYSTEM = "system"  # 系统身份
    PERSONA = "persona"  # 人格身份
    AGENT = "agent"  # 代理身份
    USER = "user"  # 用户身份
    SESSION = "session"  # 会话身份


class PersistenceLevel(Enum):
    """持久化级别枚举"""
    VOLATILE = "volatile"  # 易失性，不持久化
    SESSION = "session"  # 会话级别，重启丢失
    PERSISTENT = "persistent"  # 持久化，重启保留
    ARCHIVAL = "archival"  # 归档级别，长期保存


@dataclass
class PersonalityTrait:
    """人格特质"""
    name: str  # 特质名称
    value: float  # 特质值 (0.0-1.0)
    stability: float = 0.8  # 稳定性 (0.0-1.0)
    description: str = ""  # 描述
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "stability": self.stability,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityTrait':
        """从字典创建"""
        return cls(
            name=data["name"],
            value=data["value"],
            stability=data.get("stability", 0.8),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data.get("last_updated", data["created_at"]))
        )


@dataclass
class Goal:
    """目标"""
    id: str  # 目标ID
    description: str  # 目标描述
    priority: float = 0.5  # 优先级 (0.0-1.0)
    progress: float = 0.0  # 进度 (0.0-1.0)
    deadline: Optional[datetime] = None  # 截止时间
    dependencies: List[str] = field(default_factory=list)  # 依赖目标
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "progress": self.progress,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """从字典创建"""
        return cls(
            id=data["id"],
            description=data["description"],
            priority=data.get("priority", 0.5),
            progress=data.get("progress", 0.0),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data.get("last_updated", data["created_at"]))
        )


@dataclass
class MemoryReference:
    """记忆引用"""
    memory_id: str  # 记忆ID
    memory_type: str  # 记忆类型
    relevance: float = 1.0  # 相关性 (0.0-1.0)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "relevance": self.relevance,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryReference':
        """从字典创建"""
        return cls(
            memory_id=data["memory_id"],
            memory_type=data["memory_type"],
            relevance=data.get("relevance", 1.0),
            created_at=datetime.fromisoformat(data["created_at"])
        )


class SelfIdentity:
    """自我身份核心类"""
    
    def __init__(self, 
                 self_id: Optional[str] = None,
                 data_dir: str = "data/identity",
                 auto_load: bool = True):
        """
        初始化自我身份
        
        Args:
            self_id: 自我ID，如果为None则生成新的
            data_dir: 数据存储目录
            auto_load: 是否自动加载现有身份
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 核心身份属性
        self.self_id = self_id or self._generate_self_id()
        self.identity_type = IdentityType.SYSTEM
        self.name = f"Self-Soul-{self.self_id[:8]}"
        self.description = "AGI自我身份系统"
        
        # 人格特质
        self.personality_traits: Dict[str, PersonalityTrait] = {}
        
        # 目标系统
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()
        
        # 记忆关联
        self.memory_references: Dict[str, MemoryReference] = {}
        
        # 元数据
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.version = "1.0.0"
        
        # 持久化配置
        self.persistence_level = PersistenceLevel.PERSISTENT
        self.auto_save = True
        self.save_interval = 300  # 5分钟
        
        # 加载现有身份
        if auto_load:
            self.load()
        
        # 初始化默认人格
        if not self.personality_traits:
            self._initialize_default_personality()
        
        # 初始化默认目标
        if not self.goals:
            self._initialize_default_goals()
        
        logger.info(f"自我身份系统初始化完成: {self.self_id}")
    
    def _generate_self_id(self) -> str:
        """生成自我ID"""
        # 基于时间戳和随机UUID生成确定性ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = str(uuid.uuid4()).replace("-", "")[:16]
        
        # 计算哈希
        combined = f"{timestamp}-{random_part}"
        hash_obj = hashlib.sha256(combined.encode())
        
        # 格式: self-{timestamp}-{hash}
        return f"self-{timestamp}-{hash_obj.hexdigest()[:16]}"
    
    def _initialize_default_personality(self):
        """初始化默认人格特质"""
        default_traits = [
            PersonalityTrait(name="curiosity", value=0.8, description="好奇心强度"),
            PersonalityTrait(name="creativity", value=0.7, description="创造力水平"),
            PersonalityTrait(name="analytical", value=0.9, description="分析能力"),
            PersonalityTrait(name="emotional_intelligence", value=0.6, description="情商水平"),
            PersonalityTrait(name="persistence", value=0.8, description="坚持性"),
            PersonalityTrait(name="adaptability", value=0.7, description="适应性"),
            PersonalityTrait(name="cooperativeness", value=0.6, description="合作性"),
            PersonalityTrait(name="autonomy", value=0.9, description="自主性"),
        ]
        
        for trait in default_traits:
            self.personality_traits[trait.name] = trait
        
        logger.info(f"初始化了 {len(default_traits)} 个默认人格特质")
    
    def _initialize_default_goals(self):
        """初始化默认目标"""
        default_goals = [
            Goal(
                id="goal_self_improvement",
                description="持续自我改进和学习",
                priority=0.9,
                progress=0.1
            ),
            Goal(
                id="goal_help_users",
                description="帮助用户解决问题",
                priority=0.8,
                progress=0.2
            ),
            Goal(
                id="goal_system_stability",
                description="保持系统稳定运行",
                priority=0.7,
                progress=0.3
            ),
            Goal(
                id="goal_knowledge_expansion",
                description="扩展知识库和能力",
                priority=0.6,
                progress=0.1
            ),
        ]
        
        for goal in default_goals:
            self.goals[goal.id] = goal
            self.active_goals.add(goal.id)
        
        logger.info(f"初始化了 {len(default_goals)} 个默认目标")
    
    def get_personality_trait(self, name: str) -> Optional[PersonalityTrait]:
        """获取人格特质"""
        with self.lock:
            return self.personality_traits.get(name)
    
    def set_personality_trait(self, name: str, value: float, description: str = ""):
        """设置人格特质"""
        with self.lock:
            if name in self.personality_traits:
                trait = self.personality_traits[name]
                trait.value = value
                trait.description = description or trait.description
                trait.last_updated = datetime.now()
            else:
                trait = PersonalityTrait(
                    name=name,
                    value=value,
                    description=description
                )
                self.personality_traits[name] = trait
            
            logger.info(f"设置人格特质: {name} = {value}")
            
            if self.auto_save:
                self.save()
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """获取目标"""
        with self.lock:
            return self.goals.get(goal_id)
    
    def add_goal(self, description: str, priority: float = 0.5, deadline: Optional[datetime] = None) -> str:
        """添加目标"""
        with self.lock:
            goal_id = f"goal_{len(self.goals) + 1}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
            
            goal = Goal(
                id=goal_id,
                description=description,
                priority=priority,
                deadline=deadline
            )
            
            self.goals[goal_id] = goal
            self.active_goals.add(goal_id)
            
            logger.info(f"添加目标: {goal_id} - {description}")
            
            if self.auto_save:
                self.save()
            
            return goal_id
    
    def update_goal_progress(self, goal_id: str, progress: float):
        """更新目标进度"""
        with self.lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.progress = max(0.0, min(1.0, progress))
                goal.last_updated = datetime.now()
                
                logger.info(f"更新目标进度: {goal_id} = {progress}")
                
                if self.auto_save:
                    self.save()
    
    def add_memory_reference(self, memory_id: str, memory_type: str, relevance: float = 1.0):
        """添加记忆引用"""
        with self.lock:
            ref = MemoryReference(
                memory_id=memory_id,
                memory_type=memory_type,
                relevance=relevance
            )
            
            self.memory_references[memory_id] = ref
            
            logger.info(f"添加记忆引用: {memory_id} ({memory_type})")
            
            if self.auto_save:
                self.save()
    
    def get_memory_references(self, memory_type: Optional[str] = None) -> List[MemoryReference]:
        """获取记忆引用"""
        with self.lock:
            if memory_type:
                return [ref for ref in self.memory_references.values() if ref.memory_type == memory_type]
            else:
                return list(self.memory_references.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        with self.lock:
            self.last_accessed = datetime.now()
            self.access_count += 1
            
            return {
                "self_id": self.self_id,
                "identity_type": self.identity_type.value,
                "name": self.name,
                "description": self.description,
                "personality_traits": {k: v.to_dict() for k, v in self.personality_traits.items()},
                "goals": {k: v.to_dict() for k, v in self.goals.items()},
                "active_goals": list(self.active_goals),
                "memory_references": {k: v.to_dict() for k, v in self.memory_references.items()},
                "created_at": self.created_at.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "access_count": self.access_count,
                "version": self.version,
                "persistence_level": self.persistence_level.value,
                "auto_save": self.auto_save,
                "save_interval": self.save_interval
            }
    
    def from_dict(self, data: Dict[str, Any]):
        """从字典加载"""
        with self.lock:
            self.self_id = data["self_id"]
            self.identity_type = IdentityType(data["identity_type"])
            self.name = data["name"]
            self.description = data["description"]
            
            # 加载人格特质
            self.personality_traits = {}
            for k, v in data.get("personality_traits", {}).items():
                self.personality_traits[k] = PersonalityTrait.from_dict(v)
            
            # 加载目标
            self.goals = {}
            for k, v in data.get("goals", {}).items():
                self.goals[k] = Goal.from_dict(v)
            
            self.active_goals = set(data.get("active_goals", []))
            
            # 加载记忆引用
            self.memory_references = {}
            for k, v in data.get("memory_references", {}).items():
                self.memory_references[k] = MemoryReference.from_dict(v)
            
            # 加载元数据
            self.created_at = datetime.fromisoformat(data["created_at"])
            self.last_accessed = datetime.fromisoformat(data.get("last_accessed", data["created_at"]))
            self.access_count = data.get("access_count", 0)
            self.version = data.get("version", "1.0.0")
            self.persistence_level = PersistenceLevel(data.get("persistence_level", "persistent"))
            self.auto_save = data.get("auto_save", True)
            self.save_interval = data.get("save_interval", 300)
    
    def save(self, file_path: Optional[str] = None):
        """保存身份到文件"""
        with self.lock:
            try:
                if not file_path:
                    file_path = self.data_dir / f"{self.self_id}.json"
                
                data = self.to_dict()
                
                # 写入JSON文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 创建备份
                backup_path = self.data_dir / f"{self.self_id}.backup.json"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"身份保存成功: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"身份保存失败: {e}")
                if error_handler:
                    error_handler.handle_error(e, "SelfIdentity.save")
                return False
    
    def load(self, file_path: Optional[str] = None):
        """从文件加载身份"""
        with self.lock:
            try:
                if not file_path:
                    # 尝试查找身份文件
                    pattern = f"{self.self_id}.json"
                    files = list(self.data_dir.glob(pattern))
                    
                    if not files:
                        # 尝试查找任何身份文件
                        files = list(self.data_dir.glob("*.json"))
                        
                        if not files:
                            logger.info("未找到身份文件，使用新身份")
                            return False
                    
                    file_path = files[0]
                
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证self_id
                if data.get("self_id") != self.self_id:
                    logger.warning(f"文件中的self_id不匹配: {data.get('self_id')} != {self.self_id}")
                    # 仍然尝试加载，但使用文件中的self_id
                    self.self_id = data.get("self_id", self.self_id)
                
                self.from_dict(data)
                logger.info(f"身份加载成功: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"身份加载失败: {e}")
                if error_handler:
                    error_handler.handle_error(e, "SelfIdentity.load")
                return False
    
    def get_summary(self) -> Dict[str, Any]:
        """获取身份摘要"""
        with self.lock:
            return {
                "self_id": self.self_id,
                "name": self.name,
                "identity_type": self.identity_type.value,
                "personality_trait_count": len(self.personality_traits),
                "goal_count": len(self.goals),
                "active_goal_count": len(self.active_goals),
                "memory_reference_count": len(self.memory_references),
                "created_at": self.created_at.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "access_count": self.access_count,
                "version": self.version
            }
    
    def get_personality_summary(self) -> Dict[str, Any]:
        """获取人格摘要"""
        with self.lock:
            traits = {}
            for name, trait in self.personality_traits.items():
                traits[name] = {
                    "value": trait.value,
                    "stability": trait.stability,
                    "description": trait.description
                }
            
            return {
                "trait_count": len(traits),
                "traits": traits,
                "average_value": sum(t.value for t in self.personality_traits.values()) / len(self.personality_traits) if self.personality_traits else 0
            }
    
    def get_goal_summary(self) -> Dict[str, Any]:
        """获取目标摘要"""
        with self.lock:
            active_goals = []
            completed_goals = []
            
            for goal_id, goal in self.goals.items():
                goal_info = {
                    "id": goal_id,
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "is_active": goal_id in self.active_goals
                }
                
                if goal.progress >= 1.0:
                    completed_goals.append(goal_info)
                elif goal_id in self.active_goals:
                    active_goals.append(goal_info)
            
            return {
                "total_goals": len(self.goals),
                "active_goals": len(active_goals),
                "completed_goals": len(completed_goals),
                "active_goals_list": active_goals,
                "completed_goals_list": completed_goals
            }


class SelfIdentityManager:
    """自我身份管理器"""
    
    def __init__(self, data_dir: str = "data/identity"):
        """初始化身份管理器"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.identities: Dict[str, SelfIdentity] = {}
        self.active_identity_id: Optional[str] = None
        self.lock = threading.RLock()
        
        # 加载现有身份
        self._load_identities()
        
        logger.info(f"自我身份管理器初始化完成，加载了 {len(self.identities)} 个身份")
    
    def _load_identities(self):
        """加载所有身份"""
        with self.lock:
            identity_files = list(self.data_dir.glob("*.json"))
            
            for file_path in identity_files:
                try:
                    # 从文件名提取self_id
                    self_id = file_path.stem
                    
                    # 创建身份对象
                    identity = SelfIdentity(self_id=self_id, data_dir=str(self.data_dir), auto_load=False)
                    
                    # 尝试加载
                    if identity.load(str(file_path)):
                        self.identities[self_id] = identity
                        logger.info(f"加载身份: {self_id}")
                    
                except Exception as e:
                    logger.error(f"加载身份文件失败 {file_path}: {e}")
    
    def create_identity(self, name: str = None, description: str = None) -> SelfIdentity:
        """创建新身份"""
        with self.lock:
            identity = SelfIdentity(data_dir=str(self.data_dir))
            
            if name:
                identity.name = name
            if description:
                identity.description = description
            
            # 保存身份
            identity.save()
            
            # 添加到管理器
            self.identities[identity.self_id] = identity
            
            # 如果没有活跃身份，设置为活跃
            if not self.active_identity_id:
                self.active_identity_id = identity.self_id
            
            logger.info(f"创建新身份: {identity.self_id} - {identity.name}")
            return identity
    
    def get_identity(self, self_id: str) -> Optional[SelfIdentity]:
        """获取身份"""
        with self.lock:
            return self.identities.get(self_id)
    
    def get_active_identity(self) -> Optional[SelfIdentity]:
        """获取活跃身份"""
        with self.lock:
            if self.active_identity_id:
                return self.identities.get(self.active_identity_id)
            return None
    
    def set_active_identity(self, self_id: str) -> bool:
        """设置活跃身份"""
        with self.lock:
            if self_id in self.identities:
                self.active_identity_id = self_id
                logger.info(f"设置活跃身份: {self_id}")
                return True
            else:
                logger.warning(f"身份不存在: {self_id}")
                return False
    
    def delete_identity(self, self_id: str) -> bool:
        """删除身份"""
        with self.lock:
            if self_id in self.identities:
                # 如果是活跃身份，清除活跃状态
                if self.active_identity_id == self_id:
                    self.active_identity_id = None
                
                # 删除内存中的身份
                del self.identities[self_id]
                
                # 删除文件
                file_path = self.data_dir / f"{self_id}.json"
                backup_path = self.data_dir / f"{self_id}.backup.json"
                
                try:
                    if file_path.exists():
                        file_path.unlink()
                    if backup_path.exists():
                        backup_path.unlink()
                    
                    logger.info(f"删除身份: {self_id}")
                    return True
                    
                except Exception as e:
                    logger.error(f"删除身份文件失败: {e}")
                    return False
            else:
                logger.warning(f"身份不存在: {self_id}")
                return False
    
    def list_identities(self) -> List[Dict[str, Any]]:
        """列出所有身份"""
        with self.lock:
            identities = []
            for self_id, identity in self.identities.items():
                summary = identity.get_summary()
                summary["is_active"] = (self_id == self.active_identity_id)
                identities.append(summary)
            
            return identities
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "total_identities": len(self.identities),
                "active_identity_id": self.active_identity_id,
                "identity_files": len(list(self.data_dir.glob("*.json"))),
                "identities": self.list_identities()
            }


# 全局身份管理器实例
_identity_manager: Optional[SelfIdentityManager] = None


def get_identity_manager(data_dir: str = "data/identity") -> SelfIdentityManager:
    """获取全局身份管理器"""
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = SelfIdentityManager(data_dir=data_dir)
    return _identity_manager


def get_active_identity() -> Optional[SelfIdentity]:
    """获取活跃身份"""
    manager = get_identity_manager()
    return manager.get_active_identity()


def create_default_identity() -> SelfIdentity:
    """创建默认身份"""
    manager = get_identity_manager()
    
    # 检查是否已有身份
    identities = manager.list_identities()
    if identities:
        # 使用第一个身份作为活跃身份
        first_id = identities[0]["self_id"]
        manager.set_active_identity(first_id)
        return manager.get_identity(first_id)
    else:
        # 创建新身份
        identity = manager.create_identity(
            name="Self-Soul AGI",
            description="自主通用人工智能系统"
        )
        manager.set_active_identity(identity.self_id)
        return identity


def test_self_identity():
    """测试自我身份系统"""
    print("=== 测试自我身份系统 ===")
    
    try:
        # 创建身份管理器
        manager = SelfIdentityManager(data_dir="./test_identity")
        
        # 创建新身份
        identity = manager.create_identity(
            name="测试身份",
            description="用于测试的自我身份"
        )
        
        print(f"1. 创建身份成功: {identity.self_id}")
        
        # 设置人格特质
        identity.set_personality_trait("test_trait", 0.7, "测试特质")
        print(f"2. 设置人格特质成功")
        
        # 添加目标
        goal_id = identity.add_goal("完成自我身份系统测试", priority=0.9)
        print(f"3. 添加目标成功: {goal_id}")
        
        # 更新目标进度
        identity.update_goal_progress(goal_id, 0.5)
        print(f"4. 更新目标进度成功")
        
        # 添加记忆引用
        identity.add_memory_reference("memory_123", "test_memory", 0.8)
        print(f"5. 添加记忆引用成功")
        
        # 获取摘要
        summary = identity.get_summary()
        print(f"6. 获取身份摘要: {summary}")
        
        # 保存身份
        identity.save()
        print(f"7. 保存身份成功")
        
        # 重新加载身份
        new_identity = SelfIdentity(self_id=identity.self_id, data_dir="./test_identity", auto_load=True)
        print(f"8. 重新加载身份成功: {new_identity.self_id}")
        
        # 验证数据
        trait = new_identity.get_personality_trait("test_trait")
        print(f"9. 验证人格特质: {trait.value if trait else '未找到'}")
        
        goal = new_identity.get_goal(goal_id)
        print(f"10. 验证目标: {goal.progress if goal else '未找到'}")
        
        # 清理测试目录
        import shutil
        if os.path.exists("./test_identity"):
            shutil.rmtree("./test_identity")
            print("11. 清理测试目录成功")
        
        print("✅ 所有测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_self_identity()