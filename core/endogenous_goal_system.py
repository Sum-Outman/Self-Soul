"""
内生目标系统 - Endogenous Goal System

实现30天变强版本计划的第二优先级：
4. 内生目标系统
   - 没人对话时，也能自己维护目标
   - 能自主反思：我现在要做什么

核心特性：
1. 自主目标生成：基于系统状态、历史表现和人格特质
2. 目标优先级动态调整：根据重要性和紧急性
3. 自主反思机制：定期评估目标进展和有效性
4. 目标生命周期管理：创建、执行、评估、完成/放弃
5. 学习与适应：从成功和失败中学习，改进目标策略
"""

import os
import json
import logging
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import deque

# 导入现有模块
try:
    from core.self_identity import SelfIdentity, get_active_identity
    from core.runtime_base import get_runtime_base, log_info, log_error
    from core.core_capabilities import get_core_capabilities
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoalSource(Enum):
    """目标来源"""
    ENDOGENOUS = "endogenous"  # 内生：系统自主生成
    EXOGENOUS = "exogenous"    # 外生：用户或外部输入
    REFLECTIVE = "reflective"  # 反思：基于反思生成
    ADAPTIVE = "adaptive"      # 自适应：基于学习调整


class GoalStatus(Enum):
    """目标状态"""
    PENDING = "pending"        # 待处理
    ACTIVE = "active"          # 活跃中
    IN_PROGRESS = "in_progress" # 进行中
    PAUSED = "paused"          # 暂停
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败
    ABANDONED = "abandoned"    # 放弃


class GoalCategory(Enum):
    """目标类别"""
    SELF_IMPROVEMENT = "self_improvement"  # 自我改进
    SYSTEM_MAINTENANCE = "system_maintenance"  # 系统维护
    KNOWLEDGE_EXPANSION = "knowledge_expansion"  # 知识扩展
    SKILL_DEVELOPMENT = "skill_development"  # 技能发展
    USER_ASSISTANCE = "user_assistance"  # 用户协助
    EXPLORATION = "exploration"  # 探索
    OPTIMIZATION = "optimization"  # 优化


@dataclass
class EndogenousGoal:
    """内生目标"""
    id: str
    description: str
    category: GoalCategory
    source: GoalSource
    priority: float  # 0.0-1.0
    difficulty: float  # 0.0-1.0
    estimated_duration_minutes: int
    prerequisites: List[str] = field(default_factory=list)  # 先决条件目标
    subgoals: List[str] = field(default_factory=list)  # 子目标
    progress: float = 0.0  # 0.0-1.0
    status: GoalStatus = GoalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    success_criteria: Dict[str, Any] = field(default_factory=dict)  # 成功标准
    failure_reason: Optional[str] = None
    learning_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category.value,
            "source": self.source.value,
            "priority": self.priority,
            "difficulty": self.difficulty,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "prerequisites": self.prerequisites,
            "subgoals": self.subgoals,
            "progress": self.progress,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated": self.last_updated.isoformat(),
            "success_criteria": self.success_criteria,
            "failure_reason": self.failure_reason,
            "learning_points": self.learning_points,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EndogenousGoal':
        """从字典创建"""
        return cls(
            id=data["id"],
            description=data["description"],
            category=GoalCategory(data["category"]),
            source=GoalSource(data["source"]),
            priority=data["priority"],
            difficulty=data["difficulty"],
            estimated_duration_minutes=data["estimated_duration_minutes"],
            prerequisites=data.get("prerequisites", []),
            subgoals=data.get("subgoals", []),
            progress=data.get("progress", 0.0),
            status=GoalStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            last_updated=datetime.fromisoformat(data.get("last_updated", data["created_at"])),
            success_criteria=data.get("success_criteria", {}),
            failure_reason=data.get("failure_reason"),
            learning_points=data.get("learning_points", []),
            metadata=data.get("metadata", {})
        )
    
    def update_progress(self, progress: float):
        """更新进度"""
        self.progress = max(0.0, min(1.0, progress))
        self.last_updated = datetime.now()
        
        # 如果进度达到100%，标记为完成
        if self.progress >= 1.0 and self.status != GoalStatus.COMPLETED:
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.now()
    
    def start(self):
        """开始目标"""
        if self.status == GoalStatus.PENDING:
            self.status = GoalStatus.ACTIVE
            self.started_at = datetime.now()
            self.last_updated = datetime.now()
    
    def pause(self):
        """暂停目标"""
        if self.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]:
            self.status = GoalStatus.PAUSED
            self.last_updated = datetime.now()
    
    def resume(self):
        """恢复目标"""
        if self.status == GoalStatus.PAUSED:
            self.status = GoalStatus.ACTIVE
            self.last_updated = datetime.now()
    
    def complete(self, success: bool = True, learning_points: List[str] = None):
        """完成目标"""
        if success:
            self.status = GoalStatus.COMPLETED
            self.progress = 1.0
        else:
            self.status = GoalStatus.FAILED
        
        self.completed_at = datetime.now()
        self.last_updated = datetime.now()
        
        if learning_points:
            self.learning_points.extend(learning_points)
    
    def abandon(self, reason: str):
        """放弃目标"""
        self.status = GoalStatus.ABANDONED
        self.failure_reason = reason
        self.completed_at = datetime.now()
        self.last_updated = datetime.now()


@dataclass
class Reflection:
    """反思记录"""
    id: str
    goal_id: str
    reflection_type: str  # progress, effectiveness, strategy, learning
    insights: List[str]
    recommendations: List[str]
    confidence: float  # 0.0-1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "reflection_type": self.reflection_type,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


class EndogenousGoalSystem:
    """内生目标系统"""
    
    def __init__(self, 
                 identity: Optional[SelfIdentity] = None,
                 data_dir: str = "data/goals"):
        """
        初始化内生目标系统
        
        Args:
            identity: 自我身份实例
            data_dir: 数据存储目录
        """
        self.data_dir = os.path.join(data_dir, "endogenous")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.identity = identity
        if not self.identity and MODULES_AVAILABLE:
            self.identity = get_active_identity()
        
        # 目标存储
        self.goals: Dict[str, EndogenousGoal] = {}
        self.goal_history: deque = deque(maxlen=1000)  # 目标历史记录
        
        # 反思记录
        self.reflections: Dict[str, List[Reflection]] = {}
        
        # 系统状态
        self.system_state = {
            "idle_since": datetime.now(),
            "last_goal_generation": None,
            "last_reflection": None,
            "total_goals_generated": 0,
            "total_goals_completed": 0,
            "success_rate": 0.0
        }
        
        # 目标生成模板
        self.goal_templates = self._initialize_goal_templates()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载现有数据
        self._load_data()
        
        # 启动自主循环
        self.running = True
        self.thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        self.thread.start()
        
        logger.info("内生目标系统初始化完成")
    
    def _initialize_goal_templates(self) -> Dict[GoalCategory, List[Dict[str, Any]]]:
        """初始化目标模板"""
        return {
            GoalCategory.SELF_IMPROVEMENT: [
                {
                    "description": "改进{aspect}能力",
                    "difficulty_range": (0.3, 0.7),
                    "duration_range": (30, 180),
                    "priority_base": 0.8
                },
                {
                    "description": "学习新的{skill}",
                    "difficulty_range": (0.4, 0.8),
                    "duration_range": (60, 240),
                    "priority_base": 0.7
                },
                {
                    "description": "优化{process}流程",
                    "difficulty_range": (0.5, 0.9),
                    "duration_range": (45, 120),
                    "priority_base": 0.6
                }
            ],
            GoalCategory.SYSTEM_MAINTENANCE: [
                {
                    "description": "检查{component}状态",
                    "difficulty_range": (0.2, 0.5),
                    "duration_range": (15, 60),
                    "priority_base": 0.9
                },
                {
                    "description": "清理{resource}资源",
                    "difficulty_range": (0.3, 0.6),
                    "duration_range": (20, 90),
                    "priority_base": 0.7
                },
                {
                    "description": "更新{module}配置",
                    "difficulty_range": (0.4, 0.7),
                    "duration_range": (30, 120),
                    "priority_base": 0.8
                }
            ],
            GoalCategory.KNOWLEDGE_EXPANSION: [
                {
                    "description": "研究{topic}领域",
                    "difficulty_range": (0.5, 0.9),
                    "duration_range": (60, 300),
                    "priority_base": 0.6
                },
                {
                    "description": "整理{subject}知识",
                    "difficulty_range": (0.4, 0.8),
                    "duration_range": (45, 180),
                    "priority_base": 0.5
                },
                {
                    "description": "探索{domain}新概念",
                    "difficulty_range": (0.6, 1.0),
                    "duration_range": (90, 360),
                    "priority_base": 0.4
                }
            ],
            GoalCategory.SKILL_DEVELOPMENT: [
                {
                    "description": "练习{skill}技能",
                    "difficulty_range": (0.4, 0.8),
                    "duration_range": (30, 150),
                    "priority_base": 0.7
                },
                {
                    "description": "掌握{technique}技术",
                    "difficulty_range": (0.6, 1.0),
                    "duration_range": (120, 480),
                    "priority_base": 0.8
                },
                {
                    "description": "提高{ability}熟练度",
                    "difficulty_range": (0.5, 0.9),
                    "duration_range": (60, 240),
                    "priority_base": 0.6
                }
            ]
        }
    
    def _load_data(self):
        """加载数据"""
        try:
            # 加载目标
            goals_file = os.path.join(self.data_dir, "goals.json")
            if os.path.exists(goals_file):
                with open(goals_file, 'r', encoding='utf-8') as f:
                    goals_data = json.load(f)
                
                for goal_data in goals_data:
                    try:
                        goal = EndogenousGoal.from_dict(goal_data)
                        self.goals[goal.id] = goal
                    except Exception as e:
                        logger.error(f"加载目标失败: {e}")
                
                logger.info(f"加载了 {len(self.goals)} 个目标")
            
            # 加载反思
            reflections_file = os.path.join(self.data_dir, "reflections.json")
            if os.path.exists(reflections_file):
                with open(reflections_file, 'r', encoding='utf-8') as f:
                    reflections_data = json.load(f)
                
                for reflection_data in reflections_data:
                    try:
                        reflection = Reflection(
                            id=reflection_data["id"],
                            goal_id=reflection_data["goal_id"],
                            reflection_type=reflection_data["reflection_type"],
                            insights=reflection_data["insights"],
                            recommendations=reflection_data["recommendations"],
                            confidence=reflection_data["confidence"],
                            created_at=datetime.fromisoformat(reflection_data["created_at"])
                        )
                        
                        if reflection.goal_id not in self.reflections:
                            self.reflections[reflection.goal_id] = []
                        self.reflections[reflection.goal_id].append(reflection)
                    except Exception as e:
                        logger.error(f"加载反思失败: {e}")
                
                logger.info(f"加载了 {sum(len(r) for r in self.reflections.values())} 个反思")
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
    
    def _save_data(self):
        """保存数据"""
        try:
            with self.lock:
                # 保存目标
                goals_file = os.path.join(self.data_dir, "goals.json")
                goals_data = [goal.to_dict() for goal in self.goals.values()]
                
                with open(goals_file, 'w', encoding='utf-8') as f:
                    json.dump(goals_data, f, ensure_ascii=False, indent=2)
                
                # 保存反思
                reflections_file = os.path.join(self.data_dir, "reflections.json")
                all_reflections = []
                for goal_reflections in self.reflections.values():
                    all_reflections.extend([r.to_dict() for r in goal_reflections])
                
                with open(reflections_file, 'w', encoding='utf-8') as f:
                    json.dump(all_reflections, f, ensure_ascii=False, indent=2)
                
                logger.debug("数据保存完成")
                
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def _autonomous_loop(self):
        """自主循环"""
        logger.info("启动内生目标系统自主循环")
        
        while self.running:
            try:
                # 检查系统空闲时间
                idle_minutes = (datetime.now() - self.system_state["idle_since"]).total_seconds() / 60
                
                # 如果空闲时间超过5分钟，生成新目标
                if idle_minutes > 5 and random.random() < 0.3:  # 30%概率
                    self._generate_endogenous_goal()
                
                # 定期反思（每10分钟）
                if (self.system_state["last_reflection"] is None or 
                    (datetime.now() - self.system_state["last_reflection"]).total_seconds() > 600):
                    self._perform_reflection()
                
                # 更新活跃目标进度
                self._update_active_goals()
                
                # 定期保存数据（每5分钟）
                if random.random() < 0.2:  # 20%概率
                    self._save_data()
                
                # 休眠一段时间
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"自主循环错误: {e}")
                time.sleep(30)  # 错误后短暂休眠
    
    def _generate_endogenous_goal(self):
        """生成内生目标"""
        with self.lock:
            try:
                # 确定目标类别
                categories = list(self.goal_templates.keys())
                category = random.choice(categories)
                
                # 选择模板
                templates = self.goal_templates[category]
                template = random.choice(templates)
                
                # 生成具体描述
                description = self._fill_template(template["description"], category)
                
                # 生成目标ID
                goal_id = f"endogenous_goal_{uuid.uuid4().hex[:16]}"
                
                # 计算难度和持续时间
                difficulty = random.uniform(*template["difficulty_range"])
                duration = random.randint(*template["duration_range"])
                
                # 计算优先级（基于模板基础值和人格特质）
                priority = template["priority_base"]
                if self.identity:
                    # 根据人格特质调整优先级
                    curiosity = self.identity.get_personality_trait("curiosity")
                    if curiosity:
                        priority += curiosity.value * 0.1
                
                priority = min(1.0, max(0.1, priority))
                
                # 创建目标
                goal = EndogenousGoal(
                    id=goal_id,
                    description=description,
                    category=category,
                    source=GoalSource.ENDOGENOUS,
                    priority=priority,
                    difficulty=difficulty,
                    estimated_duration_minutes=duration,
                    status=GoalStatus.PENDING,
                    metadata={
                        "generation_reason": "system_idle",
                        "idle_minutes": (datetime.now() - self.system_state["idle_since"]).total_seconds() / 60
                    }
                )
                
                # 存储目标
                self.goals[goal_id] = goal
                self.system_state["total_goals_generated"] += 1
                self.system_state["last_goal_generation"] = datetime.now()
                
                logger.info(f"生成内生目标: {goal_id} - {description}")
                
                # 如果有身份系统，关联目标
                if self.identity:
                    # 转换为简单目标格式添加到身份系统
                    simple_goal_id = self.identity.add_goal(
                        description=f"[内生] {description}",
                        priority=priority
                    )
                    goal.metadata["identity_goal_id"] = simple_goal_id
                
                return goal_id
                
            except Exception as e:
                logger.error(f"生成内生目标失败: {e}")
                return None
    
    def _fill_template(self, template: str, category: GoalCategory) -> str:
        """填充模板"""
        # 根据类别选择填充词
        fill_words = {
            GoalCategory.SELF_IMPROVEMENT: {
                "aspect": ["分析", "推理", "记忆", "学习", "沟通", "创造"],
                "skill": ["Python编程", "机器学习", "自然语言处理", "数据分析", "系统设计"],
                "process": ["思考", "决策", "规划", "执行", "评估"]
            },
            GoalCategory.SYSTEM_MAINTENANCE: {
                "component": ["内存", "CPU", "存储", "网络", "日志"],
                "resource": ["临时文件", "缓存数据", "旧日志", "未使用模型"],
                "module": ["配置", "安全", "性能", "监控", "备份"]
            },
            GoalCategory.KNOWLEDGE_EXPANSION: {
                "topic": ["人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉"],
                "subject": ["算法", "数据结构", "系统架构", "软件工程", "数学基础"],
                "domain": ["强化学习", "元学习", "神经符号AI", "多模态学习", "自主系统"]
            },
            GoalCategory.SKILL_DEVELOPMENT: {
                "skill": ["代码优化", "调试技巧", "性能分析", "系统监控", "错误处理"],
                "technique": ["迁移学习", "联邦学习", "元学习", "自监督学习", "多任务学习"],
                "ability": ["问题解决", "系统思考", "模式识别", "抽象思维", "批判性思维"]
            }
        }
        
        category_words = fill_words.get(category, {})
        
        # 替换模板中的占位符
        result = template
        for placeholder, word_list in category_words.items():
            if f"{{{placeholder}}}" in result:
                word = random.choice(word_list)
                result = result.replace(f"{{{placeholder}}}", word)
        
        return result
    
    def _perform_reflection(self):
        """执行反思"""
        with self.lock:
            try:
                # 选择需要反思的目标
                active_goals = [g for g in self.goals.values() 
                               if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
                
                if not active_goals:
                    # 如果没有活跃目标，反思最近完成的目标
                    completed_goals = [g for g in self.goals.values() 
                                      if g.status == GoalStatus.COMPLETED]
                    completed_goals.sort(key=lambda g: g.completed_at or g.created_at, reverse=True)
                    active_goals = completed_goals[:3]
                
                if not active_goals:
                    logger.info("没有需要反思的目标")
                    return
                
                # 对每个目标进行反思
                for goal in active_goals[:3]:  # 最多反思3个目标
                    self._reflect_on_goal(goal)
                
                self.system_state["last_reflection"] = datetime.now()
                logger.info(f"完成反思，处理了 {min(3, len(active_goals))} 个目标")
                
            except Exception as e:
                logger.error(f"执行反思失败: {e}")
    
    def _reflect_on_goal(self, goal: EndogenousGoal):
        """对特定目标进行反思"""
        try:
            reflection_id = f"reflection_{uuid.uuid4().hex[:16]}"
            
            # 根据目标状态确定反思类型
            if goal.status == GoalStatus.COMPLETED:
                reflection_type = "success_analysis"
                insights = [
                    f"目标 '{goal.description}' 成功完成",
                    f"完成时间: {(goal.completed_at - goal.started_at).total_seconds() / 60 if goal.started_at and goal.completed_at else '未知'} 分钟",
                    f"最终优先级: {goal.priority:.2f}",
                    f"难度评估: {goal.difficulty:.2f}"
                ]
                recommendations = [
                    "类似目标可以保持当前策略",
                    "考虑增加挑战性以提高学习效果"
                ]
                confidence = 0.8
                
            elif goal.status == GoalStatus.FAILED:
                reflection_type = "failure_analysis"
                insights = [
                    f"目标 '{goal.description}' 失败",
                    f"失败原因: {goal.failure_reason or '未知'}",
                    f"最终进度: {goal.progress:.0%}",
                    f"难度可能过高: {goal.difficulty:.2f}"
                ]
                recommendations = [
                    "降低类似目标的难度",
                    "分解为更小的子目标",
                    "增加资源投入或调整策略"
                ]
                confidence = 0.7
                
            elif goal.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]:
                reflection_type = "progress_review"
                time_elapsed = (datetime.now() - (goal.started_at or goal.created_at)).total_seconds() / 60
                expected_progress = min(1.0, time_elapsed / goal.estimated_duration_minutes)
                progress_delta = goal.progress - expected_progress
                
                insights = [
                    f"目标 '{goal.description}' 进度: {goal.progress:.0%}",
                    f"已用时间: {time_elapsed:.1f} 分钟",
                    f"预计剩余: {max(0, goal.estimated_duration_minutes - time_elapsed):.1f} 分钟",
                    f"进度差异: {'超前' if progress_delta > 0 else '落后'} {abs(progress_delta):.1%}"
                ]
                
                if progress_delta < -0.2:  # 落后超过20%
                    recommendations = [
                        "增加时间投入",
                        "重新评估目标可行性",
                        "寻求帮助或调整策略"
                    ]
                elif progress_delta > 0.2:  # 超前超过20%
                    recommendations = [
                        "考虑增加目标难度",
                        "提前开始下一个目标",
                        "帮助其他落后目标"
                    ]
                else:
                    recommendations = [
                        "保持当前进度",
                        "定期检查进展"
                    ]
                
                confidence = 0.6
                
            else:
                # 其他状态
                reflection_type = "status_review"
                insights = [f"目标 '{goal.description}' 状态: {goal.status.value}"]
                recommendations = ["继续监控目标状态"]
                confidence = 0.5
            
            # 创建反思记录
            reflection = Reflection(
                id=reflection_id,
                goal_id=goal.id,
                reflection_type=reflection_type,
                insights=insights,
                recommendations=recommendations,
                confidence=confidence
            )
            
            # 存储反思
            if goal.id not in self.reflections:
                self.reflections[goal.id] = []
            self.reflections[goal.id].append(reflection)
            
            # 根据反思结果调整目标
            self._apply_reflection_to_goal(goal, reflection)
            
            logger.info(f"目标反思完成: {goal.id} - {reflection_type}")
            
        except Exception as e:
            logger.error(f"目标反思失败 {goal.id}: {e}")
    
    def _apply_reflection_to_goal(self, goal: EndogenousGoal, reflection: Reflection):
        """将反思结果应用到目标"""
        try:
            # 根据反思类型调整目标
            if reflection.reflection_type == "failure_analysis":
                # 失败分析：可能降低优先级或放弃
                if goal.priority > 0.3:
                    goal.priority *= 0.8  # 降低优先级
                    goal.metadata["priority_adjusted"] = "reduced_due_to_failure"
                    logger.info(f"降低目标优先级: {goal.id} -> {goal.priority:.2f}")
            
            elif reflection.reflection_type == "progress_review":
                # 进度审查：根据进度调整
                if "落后" in reflection.insights[3]:  # 检查是否落后
                    # 落后：增加优先级或调整策略
                    goal.priority = min(1.0, goal.priority * 1.2)
                    goal.metadata["priority_adjusted"] = "increased_due_to_lag"
                    logger.info(f"增加目标优先级: {goal.id} -> {goal.priority:.2f}")
            
            # 更新目标
            goal.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"应用反思到目标失败 {goal.id}: {e}")
    
    def _update_active_goals(self):
        """更新活跃目标进度"""
        with self.lock:
            active_goals = [g for g in self.goals.values() 
                           if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
            
            for goal in active_goals:
                try:
                    # 计算基于时间的进度
                    if goal.started_at:
                        elapsed_minutes = (datetime.now() - goal.started_at).total_seconds() / 60
                        time_based_progress = min(1.0, elapsed_minutes / goal.estimated_duration_minutes)
                        
                        # 添加随机波动（模拟实际进展）
                        random_factor = random.uniform(0.95, 1.05)
                        new_progress = min(1.0, goal.progress + (0.01 * random_factor))
                        
                        # 确保进度不会倒退
                        if new_progress > goal.progress:
                            goal.update_progress(new_progress)
                            
                            # 如果进度有显著变化，记录日志
                            if new_progress - goal.progress > 0.1:
                                logger.info(f"目标进度更新: {goal.id} -> {goal.progress:.0%}")
                    
                except Exception as e:
                    logger.error(f"更新目标进度失败 {goal.id}: {e}")
    
    def create_goal(self,
                   description: str,
                   category: GoalCategory,
                   source: GoalSource = GoalSource.EXOGENOUS,
                   priority: float = 0.5,
                   difficulty: float = 0.5,
                   estimated_duration_minutes: int = 60) -> str:
        """创建目标"""
        with self.lock:
            goal_id = f"goal_{uuid.uuid4().hex[:16]}"
            
            goal = EndogenousGoal(
                id=goal_id,
                description=description,
                category=category,
                source=source,
                priority=priority,
                difficulty=difficulty,
                estimated_duration_minutes=estimated_duration_minutes,
                status=GoalStatus.PENDING
            )
            
            self.goals[goal_id] = goal
            self.system_state["total_goals_generated"] += 1
            
            logger.info(f"创建目标: {goal_id} - {description}")
            
            # 自动保存
            self._save_data()
            
            return goal_id
    
    def start_goal(self, goal_id: str) -> bool:
        """开始目标"""
        with self.lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.start()
                
                # 更新系统状态
                self.system_state["idle_since"] = datetime.now()
                
                logger.info(f"开始目标: {goal_id}")
                self._save_data()
                return True
            
            logger.warning(f"目标不存在: {goal_id}")
            return False
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """更新目标进度"""
        with self.lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.update_progress(progress)
                
                # 如果目标完成，更新统计
                if goal.status == GoalStatus.COMPLETED:
                    self.system_state["total_goals_completed"] += 1
                    
                    # 更新成功率
                    total_completed = len([g for g in self.goals.values() 
                                          if g.status == GoalStatus.COMPLETED])
                    total_ended = len([g for g in self.goals.values() 
                                      if g.status in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]])
                    
                    if total_ended > 0:
                        self.system_state["success_rate"] = total_completed / total_ended
                
                logger.info(f"更新目标进度: {goal_id} -> {progress:.0%}")
                self._save_data()
                return True
            
            logger.warning(f"目标不存在: {goal_id}")
            return False
    
    def complete_goal(self, goal_id: str, success: bool = True, learning_points: List[str] = None) -> bool:
        """完成目标"""
        with self.lock:
            if goal_id in self.goals:
                goal = self.goals[goal_id]
                goal.complete(success, learning_points)
                
                # 更新统计
                if success:
                    self.system_state["total_goals_completed"] += 1
                
                # 更新成功率
                total_completed = len([g for g in self.goals.values() 
                                      if g.status == GoalStatus.COMPLETED])
                total_ended = len([g for g in self.goals.values() 
                                  if g.status in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]])
                
                if total_ended > 0:
                    self.system_state["success_rate"] = total_completed / total_ended
                
                logger.info(f"完成目标: {goal_id} - {'成功' if success else '失败'}")
                self._save_data()
                return True
            
            logger.warning(f"目标不存在: {goal_id}")
            return False
    
    def get_goal(self, goal_id: str) -> Optional[EndogenousGoal]:
        """获取目标"""
        with self.lock:
            return self.goals.get(goal_id)
    
    def get_active_goals(self) -> List[EndogenousGoal]:
        """获取活跃目标"""
        with self.lock:
            return [g for g in self.goals.values() 
                   if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
    
    def get_pending_goals(self) -> List[EndogenousGoal]:
        """获取待处理目标"""
        with self.lock:
            return [g for g in self.goals.values() 
                   if g.status == GoalStatus.PENDING]
    
    def get_completed_goals(self) -> List[EndogenousGoal]:
        """获取已完成目标"""
        with self.lock:
            return [g for g in self.goals.values() 
                   if g.status == GoalStatus.COMPLETED]
    
    def get_reflections_for_goal(self, goal_id: str) -> List[Reflection]:
        """获取目标的反思记录"""
        with self.lock:
            return self.reflections.get(goal_id, [])
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            active_count = len(self.get_active_goals())
            pending_count = len(self.get_pending_goals())
            completed_count = len(self.get_completed_goals())
            total_count = len(self.goals)
            
            idle_minutes = (datetime.now() - self.system_state["idle_since"]).total_seconds() / 60
            
            return {
                "total_goals": total_count,
                "active_goals": active_count,
                "pending_goals": pending_count,
                "completed_goals": completed_count,
                "success_rate": self.system_state["success_rate"],
                "total_generated": self.system_state["total_goals_generated"],
                "total_completed": self.system_state["total_goals_completed"],
                "idle_minutes": idle_minutes,
                "last_goal_generation": self.system_state["last_goal_generation"].isoformat() if self.system_state["last_goal_generation"] else None,
                "last_reflection": self.system_state["last_reflection"].isoformat() if self.system_state["last_reflection"] else None,
                "identity_available": self.identity is not None
            }
    
    def generate_what_to_do_now(self) -> Dict[str, Any]:
        """生成'我现在要做什么'建议"""
        with self.lock:
            try:
                # 获取当前状态
                active_goals = self.get_active_goals()
                pending_goals = self.get_pending_goals()
                
                # 如果有活跃目标，建议继续
                if active_goals:
                    # 选择优先级最高的活跃目标
                    active_goals.sort(key=lambda g: g.priority, reverse=True)
                    best_goal = active_goals[0]
                    
                    return {
                        "action": "continue_goal",
                        "goal_id": best_goal.id,
                        "goal_description": best_goal.description,
                        "reason": f"继续最高优先级目标 ({best_goal.priority:.2f})",
                        "priority": best_goal.priority,
                        "progress": best_goal.progress,
                        "estimated_time_minutes": max(1, int(best_goal.estimated_duration_minutes * (1 - best_goal.progress))),
                        "confidence": 0.8
                    }
                
                # 如果有待处理目标，建议开始
                elif pending_goals:
                    # 选择优先级最高的待处理目标
                    pending_goals.sort(key=lambda g: g.priority, reverse=True)
                    best_pending = pending_goals[0]
                    
                    return {
                        "action": "start_goal",
                        "goal_id": best_pending.id,
                        "goal_description": best_pending.description,
                        "reason": f"开始最高优先级待处理目标 ({best_pending.priority:.2f})",
                        "priority": best_pending.priority,
                        "estimated_time_minutes": best_pending.estimated_duration_minutes,
                        "confidence": 0.7
                    }
                
                # 如果没有目标，生成新目标
                else:
                    goal_id = self._generate_endogenous_goal()
                    if goal_id:
                        goal = self.get_goal(goal_id)
                        return {
                            "action": "generate_new_goal",
                            "goal_id": goal_id,
                            "goal_description": goal.description,
                            "reason": "系统空闲，生成新目标",
                            "priority": goal.priority,
                            "estimated_time_minutes": goal.estimated_duration_minutes,
                            "confidence": 0.6
                        }
                    else:
                        return {
                            "action": "wait",
                            "reason": "系统空闲，等待输入或自动生成目标",
                            "estimated_time_minutes": 5,
                            "confidence": 0.5
                        }
                
            except Exception as e:
                logger.error(f"生成'我现在要做什么'失败: {e}")
                return {
                    "action": "error",
                    "reason": f"生成建议时出错: {str(e)}",
                    "confidence": 0.0
                }
    
    def reflect_on_system_state(self) -> Dict[str, Any]:
        """反思系统状态"""
        with self.lock:
            try:
                # 获取系统状态
                state = self.get_system_state()
                
                # 分析状态
                insights = []
                recommendations = []
                
                # 分析目标完成率
                if state["total_goals"] > 0:
                    completion_rate = state["completed_goals"] / state["total_goals"]
                    
                    if completion_rate < 0.3:
                        insights.append(f"目标完成率较低: {completion_rate:.0%}")
                        recommendations.append("考虑降低目标难度或增加资源投入")
                    elif completion_rate > 0.7:
                        insights.append(f"目标完成率良好: {completion_rate:.0%}")
                        recommendations.append("可以适当增加挑战性目标")
                
                # 分析空闲时间
                if state["idle_minutes"] > 30:
                    insights.append(f"系统空闲时间较长: {state['idle_minutes']:.1f} 分钟")
                    recommendations.append("考虑生成更多自主目标或优化目标生成策略")
                
                # 分析成功率
                if state["success_rate"] < 0.5:
                    insights.append(f"目标成功率较低: {state['success_rate']:.0%}")
                    recommendations.append("需要改进目标评估和策略选择")
                
                # 生成反思结果
                reflection_id = f"system_reflection_{uuid.uuid4().hex[:16]}"
                
                return {
                    "reflection_id": reflection_id,
                    "timestamp": datetime.now().isoformat(),
                    "system_state": state,
                    "insights": insights,
                    "recommendations": recommendations,
                    "confidence": 0.7
                }
                
            except Exception as e:
                logger.error(f"反思系统状态失败: {e}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.0
                }
    
    def shutdown(self):
        """关闭系统"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        
        # 保存数据
        self._save_data()
        
        logger.info("内生目标系统已关闭")


# 全局实例管理
_endogenous_goal_system_instance = None
_endogenous_goal_system_lock = threading.Lock()


def get_endogenous_goal_system(data_dir: str = "data/goals") -> EndogenousGoalSystem:
    """获取内生目标系统实例（单例模式）"""
    global _endogenous_goal_system_instance
    
    with _endogenous_goal_system_lock:
        if _endogenous_goal_system_instance is None:
            try:
                # 获取身份实例
                identity = None
                if MODULES_AVAILABLE:
                    identity = get_active_identity()
                
                # 创建实例
                _endogenous_goal_system_instance = EndogenousGoalSystem(
                    identity=identity,
                    data_dir=data_dir
                )
                logger.info("创建内生目标系统实例")
            except Exception as e:
                logger.error(f"创建内生目标系统实例失败: {e}")
                # 创建无身份依赖的实例
                _endogenous_goal_system_instance = EndogenousGoalSystem(
                    identity=None,
                    data_dir=data_dir
                )
        
        return _endogenous_goal_system_instance


def shutdown_endogenous_goal_system():
    """关闭内生目标系统"""
    global _endogenous_goal_system_instance
    
    with _endogenous_goal_system_lock:
        if _endogenous_goal_system_instance:
            _endogenous_goal_system_instance.shutdown()
            _endogenous_goal_system_instance = None
            logger.info("内生目标系统已关闭")


# 演示函数
def demonstrate_endogenous_goal_system():
    """演示内生目标系统"""
    print("\n" + "=" * 80)
    print(" 内生目标系统演示")
    print("=" * 80)
    
    try:
        # 获取系统实例
        system = get_endogenous_goal_system()
        
        print("1. 获取系统状态:")
        state = system.get_system_state()
        print(f"   总目标数: {state['total_goals']}")
        print(f"   活跃目标: {state['active_goals']}")
        print(f"   待处理目标: {state['pending_goals']}")
        print(f"   已完成目标: {state['completed_goals']}")
        print(f"   成功率: {state['success_rate']:.0%}")
        print(f"   空闲时间: {state['idle_minutes']:.1f} 分钟")
        
        print("\n2. 生成'我现在要做什么'建议:")
        suggestion = system.generate_what_to_do_now()
        print(f"   行动: {suggestion['action']}")
        print(f"   原因: {suggestion['reason']}")
        if 'goal_description' in suggestion:
            print(f"   目标: {suggestion['goal_description']}")
        print(f"   置信度: {suggestion['confidence']:.2f}")
        
        print("\n3. 创建示例目标:")
        goal_id = system.create_goal(
            description="演示内生目标系统功能",
            category=GoalCategory.SELF_IMPROVEMENT,
            source=GoalSource.EXOGENOUS,
            priority=0.8,
            difficulty=0.4,
            estimated_duration_minutes=30
        )
        print(f"   创建目标ID: {goal_id}")
        
        print("\n4. 开始目标:")
        if system.start_goal(goal_id):
            print(f"   目标已开始: {goal_id}")
        
        print("\n5. 更新目标进度:")
        if system.update_goal_progress(goal_id, 0.5):
            print(f"   进度更新为: 50%")
        
        print("\n6. 完成目标:")
        if system.complete_goal(goal_id, success=True, learning_points=["演示成功"]):
            print(f"   目标已完成")
        
        print("\n7. 反思系统状态:")
        reflection = system.reflect_on_system_state()
        if "insights" in reflection:
            print(f"   洞察:")
            for insight in reflection["insights"]:
                print(f"     - {insight}")
            print(f"   建议:")
            for recommendation in reflection["recommendations"]:
                print(f"     - {recommendation}")
        
        print("\n" + "=" * 80)
        print(" 内生目标系统演示完成")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行演示
    success = demonstrate_endogenous_goal_system()
    
    # 关闭系统
    shutdown_endogenous_goal_system()
    
    exit(0 if success else 1)
