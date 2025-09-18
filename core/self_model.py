"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
自我模型系统 - 实现AGI的自我意识和认知能力
Self Model System - Implements AGI's self-awareness and cognitive capabilities

提供能力自我评估、偏好建模、限制认知和目标系统管理
Provides capability self-assessment, preference modeling, limitation awareness and goal system management
"""
import time
import numpy as np
from typing import Dict, List, Any, Optional, Set
import json
from datetime import datetime
import pickle
from pathlib import Path
from enum import Enum

from .error_handling import error_handler
from .model_registry import model_registry
from .online_learning_system import online_learning_system

class CapabilityLevel(Enum):
    """能力水平枚举"""
    NOVICE = "novice"        # 新手
    COMPETENT = "competent"  # 胜任
    PROFICIENT = "proficient" # 熟练
    EXPERT = "expert"        # 专家
    MASTER = "master"        # 大师

class PreferenceType(Enum):
    """偏好类型枚举"""
    TASK = "task"            # 任务偏好
    STYLE = "style"          # 风格偏好
    INTERACTION = "interaction" # 交互偏好
    LEARNING = "learning"    # 学习偏好

class LimitationType(Enum):
    """限制类型枚举"""
    KNOWLEDGE = "knowledge"  # 知识限制
    COMPUTATIONAL = "computational" # 计算限制
    TEMPORAL = "temporal"   # 时间限制
    ETHICAL = "ethical"     # 伦理限制

class GoalPriority(Enum):
    """目标优先级枚举"""
    CRITICAL = "critical"    # 关键
    HIGH = "high"           # 高
    MEDIUM = "medium"       # 中
    LOW = "low"             # 低

class CapabilityModel:
    """能力模型 - 管理AGI的能力自我评估"""
    
    def __init__(self):
        self.capabilities = {}  # 能力字典: {capability: {level: str, confidence: float, evidence: list}}
        self.skill_tree = {}    # 技能树结构
        self.learning_curve = {} # 学习曲线数据
        self.performance_history = [] # 性能历史记录
        
        # 初始化基础能力
        self._initialize_basic_capabilities()
    
    def _initialize_basic_capabilities(self):
        """初始化基础能力"""
        basic_capabilities = {
            "language_processing": {"level": CapabilityLevel.PROFICIENT.value, "confidence": 0.8, "evidence": []},
            "reasoning": {"level": CapabilityLevel.COMPETENT.value, "confidence": 0.7, "evidence": []},
            "learning": {"level": CapabilityLevel.COMPETENT.value, "confidence": 0.75, "evidence": []},
            "memory": {"level": CapabilityLevel.COMPETENT.value, "confidence": 0.7, "evidence": []},
            "problem_solving": {"level": CapabilityLevel.COMPETENT.value, "confidence": 0.65, "evidence": []}
        }
        self.capabilities = basic_capabilities
        
        # 初始化技能树
        self.skill_tree = {
            "root": {
                "name": "agi_capabilities",
                "children": ["cognitive", "perceptual", "motor", "social"]
            },
            "cognitive": {
                "name": "cognitive_abilities",
                "children": ["reasoning", "learning", "memory", "problem_solving"]
            },
            "perceptual": {
                "name": "perceptual_abilities",
                "children": ["vision", "audio", "sensory"]
            },
            "motor": {
                "name": "motor_abilities",
                "children": ["physical", "digital"]
            },
            "social": {
                "name": "social_abilities",
                "children": ["communication", "empathy", "collaboration"]
            }
        }
    
    def assess_capability(self, capability: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估特定能力"""
        try:
            if capability not in self.capabilities:
                self.capabilities[capability] = {
                    "level": CapabilityLevel.NOVICE.value,
                    "confidence": 0.5,
                    "evidence": []
                }
            
            # 分析性能数据
            assessment = self._analyze_performance(performance_data)
            
            # 更新能力水平
            current_level = self.capabilities[capability]["level"]
            new_level = self._determine_new_level(current_level, assessment)
            
            # 更新置信度
            confidence = self._update_confidence(assessment)
            
            # 添加证据
            evidence = {
                "timestamp": datetime.now().isoformat(),
                "assessment": assessment,
                "performance_data": performance_data
            }
            self.capabilities[capability]["evidence"].append(evidence)
            
            # 更新能力记录
            self.capabilities[capability].update({
                "level": new_level,
                "confidence": confidence,
                "last_assessed": datetime.now().isoformat()
            })
            
            # 记录性能历史
            self.performance_history.append({
                "capability": capability,
                "old_level": current_level,
                "new_level": new_level,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "capability": capability,
                "old_level": current_level,
                "new_level": new_level,
                "confidence": confidence,
                "improvement": assessment.get("improvement", 0)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "CapabilityModel", f"评估能力 {capability} 失败")
            return {"error": str(e)}
    
    def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """分析性能数据"""
        analysis = {
            "accuracy": performance_data.get("accuracy", 0),
            "efficiency": performance_data.get("efficiency", 0),
            "consistency": performance_data.get("consistency", 0),
            "improvement": performance_data.get("improvement", 0)
        }
        
        # 计算综合得分
        total_score = (analysis["accuracy"] * 0.4 + 
                      analysis["efficiency"] * 0.3 + 
                      analysis["consistency"] * 0.2 + 
                      analysis["improvement"] * 0.1)
        analysis["total_score"] = total_score
        
        return analysis
    
    def _determine_new_level(self, current_level: str, assessment: Dict[str, float]) -> str:
        """确定新的能力水平"""
        score = assessment["total_score"]
        current_enum = CapabilityLevel(current_level)
        
        level_thresholds = {
            CapabilityLevel.NOVICE: 0.3,
            CapabilityLevel.COMPETENT: 0.5,
            CapabilityLevel.PROFICIENT: 0.7,
            CapabilityLevel.EXPERT: 0.85,
            CapabilityLevel.MASTER: 0.95
        }
        
        # 根据得分确定新水平
        for level, threshold in reversed(list(level_thresholds.items())):
            if score >= threshold:
                return level.value
        
        return current_level
    
    def _update_confidence(self, assessment: Dict[str, float]) -> float:
        """更新置信度"""
        consistency = assessment.get("consistency", 0.5)
        accuracy = assessment.get("accuracy", 0.5)
        return min(1.0, max(0.1, (consistency * 0.6 + accuracy * 0.4)))
    
    def get_capability_report(self) -> Dict[str, Any]:
        """获取能力报告"""
        return {
            "capabilities": self.capabilities,
            "skill_tree": self.skill_tree,
            "performance_history": self.performance_history[-100:],  # 最近100条记录
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成能力摘要"""
        total_capabilities = len(self.capabilities)
        avg_confidence = sum(cap["confidence"] for cap in self.capabilities.values()) / total_capabilities
        level_distribution = {level.value: 0 for level in CapabilityLevel}
        
        for cap in self.capabilities.values():
            level_distribution[cap["level"]] += 1
        
        return {
            "total_capabilities": total_capabilities,
            "average_confidence": avg_confidence,
            "level_distribution": level_distribution,
            "last_updated": datetime.now().isoformat()
        }
    
    def identify_skill_gaps(self) -> List[Dict[str, Any]]:
        """识别技能差距"""
        gaps = []
        for capability, data in self.capabilities.items():
            if data["confidence"] < 0.6 or data["level"] == CapabilityLevel.NOVICE.value:
                gap = {
                    "capability": capability,
                    "current_level": data["level"],
                    "confidence": data["confidence"],
                    "priority": "high" if data["confidence"] < 0.4 else "medium",
                    "suggested_actions": self._suggest_improvement_actions(capability)
                }
                gaps.append(gap)
        
        return gaps
    
    def _suggest_improvement_actions(self, capability: str) -> List[str]:
        """建议改进行动"""
        actions = []
        if "learning" in capability:
            actions.extend([
                "增加训练数据多样性",
                "尝试不同的学习算法",
                "进行迁移学习"
            ])
        elif "reasoning" in capability:
            actions.extend([
                "练习逻辑推理问题",
                "学习新的推理策略",
                "分析推理错误案例"
            ])
        else:
            actions.extend([
                "寻找相关学习资源",
                "进行刻意练习",
                "寻求专家反馈"
            ])
        
        return actions

class PreferenceModel:
    """偏好模型 - 管理AGI的偏好和价值观"""
    
    def __init__(self):
        self.preferences = {}  # 偏好字典: {preference_type: {preference: strength}}
        self.value_system = {} # 价值系统
        self.interaction_history = [] # 交互历史
        self.learning_preferences = {} # 学习偏好
        
        # 初始化基础偏好
        self._initialize_basic_preferences()
    
    def _initialize_basic_preferences(self):
        """初始化基础偏好"""
        self.preferences = {
            PreferenceType.TASK.value: {
                "problem_solving": 0.8,
                "learning": 0.9,
                "creativity": 0.7
            },
            PreferenceType.STYLE.value: {
                "systematic": 0.6,
                "creative": 0.5,
                "practical": 0.7
            },
            PreferenceType.INTERACTION.value: {
                "detailed": 0.6,
                "concise": 0.4,
                "interactive": 0.7
            },
            PreferenceType.LEARNING.value: {
                "structured": 0.7,
                "exploratory": 0.6,
                "collaborative": 0.5
            }
        }
        
        # 初始化价值系统
        self.value_system = {
            "accuracy": 0.9,
            "efficiency": 0.8,
            "helpfulness": 0.95,
            "creativity": 0.7,
            "reliability": 0.85
        }
    
    def update_preferences(self, interaction_data: Dict[str, Any]):
        """基于交互更新偏好"""
        try:
            # 分析交互数据
            preference_changes = self._analyze_interaction(interaction_data)
            
            # 应用偏好更新
            for pref_type, changes in preference_changes.items():
                if pref_type in self.preferences:
                    for preference, delta in changes.items():
                        if preference in self.preferences[pref_type]:
                            self.preferences[pref_type][preference] = max(0.1, min(1.0, 
                                self.preferences[pref_type][preference] + delta))
                        else:
                            self.preferences[pref_type][preference] = max(0.1, min(1.0, delta))
            
            # 记录交互历史
            self.interaction_history.append({
                "timestamp": datetime.now().isoformat(),
                "interaction_data": interaction_data,
                "preference_changes": preference_changes
            })
            
            error_handler.log_info("偏好已基于交互更新", "PreferenceModel")
            
        except Exception as e:
            error_handler.handle_error(e, "PreferenceModel", "更新偏好失败")
    
    def _analyze_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """分析交互数据以检测偏好变化"""
        changes = {}
        
        # 分析任务类型偏好
        if "task_type" in interaction_data:
            task_type = interaction_data["task_type"]
            success = interaction_data.get("success", True)
            enjoyment = interaction_data.get("enjoyment", 0.5)
            
            change_strength = 0.1 * (1 if success else -0.5) * enjoyment
            changes.setdefault(PreferenceType.TASK.value, {})[task_type] = change_strength
        
        # 分析交互风格偏好
        if "interaction_style" in interaction_data:
            style = interaction_data["interaction_style"]
            effectiveness = interaction_data.get("effectiveness", 0.5)
            
            changes.setdefault(PreferenceType.INTERACTION.value, {})[style] = 0.05 * effectiveness
        
        return changes
    
    def get_preference_profile(self) -> Dict[str, Any]:
        """获取偏好配置文件"""
        return {
            "preferences": self.preferences,
            "value_system": self.value_system,
            "interaction_history_summary": self._summarize_interaction_history(),
            "learning_preferences": self.learning_preferences
        }
    
    def _summarize_interaction_history(self) -> Dict[str, Any]:
        """汇总交互历史"""
        if not self.interaction_history:
            return {"total_interactions": 0}
        
        recent_history = self.interaction_history[-50:]  # 最近50次交互
        total_interactions = len(self.interaction_history)
        
        # 计算平均偏好变化
        avg_changes = {}
        for entry in recent_history:
            for pref_type, changes in entry.get("preference_changes", {}).items():
                if pref_type not in avg_changes:
                    avg_changes[pref_type] = {}
                for pref, change in changes.items():
                    avg_changes[pref_type][pref] = avg_changes[pref_type].get(pref, 0) + change
        
        # 计算平均值
        for pref_type in avg_changes:
            for pref in avg_changes[pref_type]:
                avg_changes[pref_type][pref] /= len(recent_history)
        
        return {
            "total_interactions": total_interactions,
            "recent_interactions": len(recent_history),
            "average_preference_changes": avg_changes
        }
    
    def align_with_values(self, action: Dict[str, Any]) -> float:
        """检查行动与价值系统的对齐度"""
        alignment_scores = []
        
        for value, importance in self.value_system.items():
            if value in action:
                # 行动中体现的价值程度
                value_manifestation = action[value]
                alignment = value_manifestation * importance
                alignment_scores.append(alignment)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5

class LimitationModel:
    """限制模型 - 管理AGI的局限性认知"""
    
    def __init__(self):
        self.limitations = {}  # 限制字典: {limitation_type: {limitation: severity}}
        self.awareness_level = 0.7  # 限制认知水平
        self.adaptation_strategies = {} # 适应策略
        self.limitation_history = [] # 限制历史记录
        
        # 初始化已知限制
        self._initialize_known_limitations()
    
    def _initialize_known_limitations(self):
        """初始化已知限制"""
        self.limitations = {
            LimitationType.KNOWLEDGE.value: {
                "domain_knowledge_gaps": 0.6,
                "real_time_information": 0.7,
                "cultural_context": 0.5
            },
            LimitationType.COMPUTATIONAL.value: {
                "processing_speed": 0.4,
                "memory_capacity": 0.3,
                "energy_efficiency": 0.5
            },
            LimitationType.TEMPORAL.value: {
                "response_time": 0.5,
                "learning_speed": 0.6,
                "planning_horizon": 0.4
            },
            LimitationType.ETHICAL.value: {
                "value_alignment": 0.7,
                "bias_detection": 0.6,
                "ethical_reasoning": 0.5
            }
        }
        
        # 初始化适应策略
        self.adaptation_strategies = {
            "knowledge_gaps": ["知识检索", "请求澄清", "承认未知"],
            "computational_limits": ["优化算法", "分批处理", "资源管理"],
            "temporal_constraints": ["优先级排序", "时间管理", "渐进响应"],
            "ethical_concerns": ["伦理审查", "价值对齐", "人工监督"]
        }
    
    def assess_limitations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估在当前上下文中的限制"""
        try:
            limitation_assessment = {}
            
            for lim_type, limitations in self.limitations.items():
                context_aware_limitations = {}
                for limitation, severity in limitations.items():
                    # 根据上下文调整限制严重性
                    adjusted_severity = self._adjust_severity_for_context(limitation, severity, context)
                    context_aware_limitations[limitation] = adjusted_severity
                
                limitation_assessment[lim_type] = context_aware_limitations
            
            # 记录评估
            assessment_record = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "limitation_assessment": limitation_assessment,
                "overall_limitation_score": self._calculate_overall_score(limitation_assessment)
            }
            self.limitation_history.append(assessment_record)
            
            return assessment_record
            
        except Exception as e:
            error_handler.handle_error(e, "LimitationModel", "限制评估失败")
            return {"error": str(e)}
    
    def _adjust_severity_for_context(self, limitation: str, base_severity: float, context: Dict[str, Any]) -> float:
        """根据上下文调整限制严重性"""
        adjusted_severity = base_severity
        
        # 根据上下文因素调整
        if "complexity" in context:
            adjusted_severity *= (1 + context["complexity"] * 0.2)
        
        if "time_pressure" in context:
            adjusted_severity *= (1 + context["time_pressure"] * 0.3)
        
        if "stakes" in context:
            adjusted_severity *= (1 + context["stakes"] * 0.4)
        
        return min(1.0, max(0.1, adjusted_severity))
    
    def _calculate_overall_score(self, limitation_assessment: Dict[str, Any]) -> float:
        """计算总体限制分数"""
        total_severity = 0
        count = 0
        
        for lim_type, limitations in limitation_assessment.items():
            for severity in limitations.values():
                total_severity += severity
                count += 1
        
        return total_severity / count if count > 0 else 0.5
    
    def get_adaptation_strategies(self, limitation_type: str, limitation: str) -> List[str]:
        """获取适应策略"""
        strategies = []
        
        # 通用策略
        strategies.extend(self.adaptation_strategies.get("general", []))
        
        # 类型特定策略
        if limitation_type in self.adaptation_strategies:
            strategies.extend(self.adaptation_strategies[limitation_type])
        
        # 具体限制策略
        key = f"{limitation_type}_{limitation}"
        if key in self.adaptation_strategies:
            strategies.extend(self.adaptation_strategies[key])
        
        return list(set(strategies))  # 去重
    
    def update_awareness(self, feedback: Dict[str, Any]):
        """基于反馈更新限制认知"""
        try:
            if "limitation_awareness" in feedback:
                new_awareness = feedback["limitation_awareness"]
                # 平滑更新认知水平
                self.awareness_level = 0.8 * self.awareness_level + 0.2 * new_awareness
            
            if "new_limitations" in feedback:
                for lim_type, new_lims in feedback["new_limitations"].items():
                    if lim_type not in self.limitations:
                        self.limitations[lim_type] = {}
                    for lim, severity in new_lims.items():
                        self.limitations[lim_type][lim] = severity
            
            error_handler.log_info("限制认知已更新", "LimitationModel")
            
        except Exception as e:
            error_handler.handle_error(e, "LimitationModel", "更新限制认知失败")

class GoalModel:
    """目标模型 - 管理AGI的目标系统"""
    
    def __init__(self):
        self.goals = {}  # 目标字典: {goal_id: goal_data}
        self.goal_hierarchy = {} # 目标层次结构
        self.progress_tracking = {} # 进度跟踪
        self.goal_history = [] # 目标历史记录
        self.next_goal_id = 1
        
        # 初始化基础目标
        self._initialize_basic_goals()
    
    def _initialize_basic_goals(self):
        """初始化基础目标"""
        basic_goals = {
            "learn_continuously": {
                "description": "持续学习和自我改进",
                "priority": GoalPriority.HIGH.value,
                "deadline": None,
                "progress": 0.3,
                "dependencies": [],
                "metrics": ["learning_rate", "knowledge_growth", "skill_improvement"]
            },
            "improve_reasoning": {
                "description": "提高推理能力",
                "priority": GoalPriority.HIGH.value,
                "deadline": None,
                "progress": 0.4,
                "dependencies": ["learn_continuously"],
                "metrics": ["accuracy", "efficiency", "complexity_handled"]
            },
            "enhance_interaction": {
                "description": "增强交互能力",
                "priority": GoalPriority.MEDIUM.value,
                "deadline": None,
                "progress": 0.2,
                "dependencies": [],
                "metrics": ["user_satisfaction", "response_quality", "engagement"]
            }
        }
        
        self.goals = basic_goals
        self._build_goal_hierarchy()
    
    def _build_goal_hierarchy(self):
        """构建目标层次结构"""
        self.goal_hierarchy = {
            "root": {
                "name": "agi_development",
                "children": ["learning_goals", "performance_goals", "interaction_goals"]
            },
            "learning_goals": {
                "name": "学习目标",
                "children": ["learn_continuously", "knowledge_expansion", "skill_acquisition"]
            },
            "performance_goals": {
                "name": "性能目标",
