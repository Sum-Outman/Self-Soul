"""
认知弱点分析器

该模块实现AGI系统的认知弱点识别和分析，包括：
1. 弱点检测：识别系统在认知、执行、社会等维度的弱点
2. 根因分析：分析弱点产生的根本原因
3. 影响评估：评估弱点对系统性能的影响程度
4. 机会识别：识别改进机会和优化潜力
5. 优先级排序：为弱点修复分配优先级

分析维度：
1. 认知维度弱点：
   - 推理能力缺陷：逻辑推理、因果推断、符号推理的不足
   - 学习能力局限：知识获取、技能学习、适应速度的瓶颈
   - 规划能力短板：任务分解、路径规划、资源优化的缺陷

2. 执行维度弱点：
   - 决策质量问题：准确性、及时性、风险评估的不足
   - 执行效率低下：执行速度、资源消耗、成功率的缺陷
   - 适应能力不足：环境变化响应、策略调整、鲁棒性的局限

3. 社会维度弱点：
   - 沟通能力缺陷：表达清晰度、理解准确性、交互自然度的不足
   - 协作能力局限：任务分配、协调配合、冲突解决的短板
   - 伦理对齐问题：价值观一致性、安全约束、责任意识的缺陷

技术特性：
- 多层次弱点检测机制
- 基于证据的根因分析
- 影响传播模型
- 机会成本计算
- 动态优先级调整
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import networkx as nx
from scipy import stats

# 配置日志
logger = logging.getLogger(__name__)

class WeaknessCategory(Enum):
    """弱点类别"""
    COGNITIVE = "cognitive"
    EXECUTION = "execution"
    SOCIAL = "social"
    TECHNICAL = "technical"
    SYSTEMIC = "systemic"

class WeaknessSeverity(Enum):
    """弱点严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RootCauseType(Enum):
    """根因类型"""
    DATA_QUALITY = "data_quality"
    ALGORITHM_LIMITATION = "algorithm_limitation"
    ARCHITECTURE_DEFECT = "architecture_defect"
    TRAINING_INSUFFICIENCY = "training_insufficiency"
    RESOURCE_CONSTRAINT = "resource_constraint"
    DESIGN_FLAW = "design_flaw"
    CONFIGURATION_ERROR = "configuration_error"

@dataclass
class WeaknessEvidence:
    """弱点证据"""
    evidence_id: str
    evidence_type: str
    description: str
    data_source: str
    confidence: float
    timestamp: datetime
    raw_data: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class RootCauseAnalysis:
    """根因分析"""
    root_cause_id: str
    root_cause_type: RootCauseType
    description: str
    contributing_factors: List[str]
    evidence: List[WeaknessEvidence]
    confidence: float
    impact_score: float
    fix_complexity: float  # 修复复杂度，0-1

@dataclass
class CognitiveWeakness:
    """认知弱点"""
    weakness_id: str
    category: WeaknessCategory
    severity: WeaknessSeverity
    description: str
    affected_components: List[str]
    performance_metrics_affected: List[str]
    
    # 分析结果
    root_causes: List[RootCauseAnalysis]
    impact_assessment: Dict[str, float]
    improvement_opportunities: List[str]
    estimated_fix_effort: float  # 人天
    priority_score: float
    
    # 证据
    evidence: List[WeaknessEvidence] = field(default_factory=list)
    
    # 状态
    detected_at: datetime = field(default_factory=datetime.now)
    last_analyzed_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, investigating, fixing, resolved, mitigated
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    related_weaknesses: List[str] = field(default_factory=list)

@dataclass
class WeaknessAnalysisReport:
    """弱点分析报告"""
    report_id: str
    timestamp: datetime
    analysis_duration_seconds: float
    weaknesses_identified: List[CognitiveWeakness]
    statistical_summary: Dict[str, Any]
    recommendations: List[str]
    priority_ranking: List[Tuple[str, float]]  # (weakness_id, priority_score)

class CognitiveWeaknessAnalyzer:
    """
    认知弱点分析器
    
    核心组件:
    1. 弱点检测器: 识别性能指标中的异常和不足
    2. 根因分析器: 分析弱点产生的根本原因
    3. 影响评估器: 评估弱点对系统性能的影响
    4. 机会识别器: 识别改进机会和优化潜力
    5. 优先级排序器: 为弱点修复分配优先级
    
    工作流程:
    性能数据 → 弱点检测器 → 识别弱点 → 根因分析器 → 分析根因
    根因分析 → 影响评估器 → 评估影响 → 机会识别器 → 识别机会
    综合分析 → 优先级排序器 → 分配优先级 → 生成报告
    
    技术特性:
    - 多层次弱点检测机制
    - 基于证据的根因分析
    - 影响传播模型
    - 机会成本计算
    - 动态优先级调整
    """
    
    def __init__(self,
                 analysis_frequency_hours: float = 24.0,
                 min_confidence_threshold: float = 0.7,
                 max_weaknesses_tracked: int = 100,
                 automatic_root_cause_enabled: bool = True,
                 impact_propagation_enabled: bool = True):
        """
        初始化认知弱点分析器
        
        Args:
            analysis_frequency_hours: 分析频率（小时）
            min_confidence_threshold: 最小置信度阈值
            max_weaknesses_tracked: 最大跟踪弱点数量
            automatic_root_cause_enabled: 是否启用自动根因分析
            impact_propagation_enabled: 是否启用影响传播分析
        """
        self.analysis_frequency_hours = analysis_frequency_hours
        self.min_confidence_threshold = min_confidence_threshold
        self.max_weaknesses_tracked = max_weaknesses_tracked
        self.automatic_root_cause_enabled = automatic_root_cause_enabled
        self.impact_propagation_enabled = impact_propagation_enabled
        
        # 弱点数据库
        self.weaknesses_database: Dict[str, CognitiveWeakness] = {}
        self.weakness_history: List[Dict[str, Any]] = []
        
        # 性能数据引用
        self.performance_data: Optional[Dict[str, Any]] = None
        
        # 系统组件映射
        self.system_components: Dict[str, Dict[str, Any]] = {}
        self._initialize_system_components()
        
        # 配置参数
        self.config = {
            'severity_thresholds': {
                'critical': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2
            },
            'impact_weight_factors': {
                'performance': 0.4,
                'reliability': 0.3,
                'scalability': 0.2,
                'security': 0.1
            },
            'root_cause_confidence_threshold': 0.65,
            'evidence_correlation_threshold': 0.7,
            'max_root_causes_per_weakness': 3,
            'opportunity_identification_threshold': 0.3,
            'priority_decay_factor': 0.95
        }
        
        # 性能统计
        self.performance_stats = {
            'analyses_completed': 0,
            'weaknesses_identified': 0,
            'root_causes_analyzed': 0,
            'improvement_opportunities_found': 0,
            'false_positives': 0,
            'average_analysis_time': 0.0,
            'average_weakness_severity': 0.0,
            'resolution_rate': 0.0
        }
        
        # 状态变量
        self.last_analysis_time = time.time()
        self.system_start_time = time.time()
        
        logger.info(f"认知弱点分析器初始化完成，分析频率: {analysis_frequency_hours} 小时")
    
    def _initialize_system_components(self):
        """初始化系统组件映射"""
        # AGI系统核心组件
        components = {
            # 认知组件
            "reasoning_engine": {
                "name": "推理引擎",
                "category": "cognitive",
                "criticality": 0.9,
                "dependencies": ["knowledge_base", "memory_system"],
                "performance_metrics": ["logical_reasoning_accuracy", "causal_inference_effectiveness"]
            },
            "learning_system": {
                "name": "学习系统",
                "category": "cognitive",
                "criticality": 0.85,
                "dependencies": ["memory_system", "data_processor"],
                "performance_metrics": ["knowledge_acquisition_speed", "skill_learning_efficiency"]
            },
            "planning_system": {
                "name": "规划系统",
                "category": "cognitive",
                "criticality": 0.8,
                "dependencies": ["reasoning_engine", "world_model"],
                "performance_metrics": ["complex_task_completion_rate", "planning_efficiency_improvement"]
            },
            
            # 执行组件
            "decision_maker": {
                "name": "决策器",
                "category": "execution",
                "criticality": 0.95,
                "dependencies": ["reasoning_engine", "value_system"],
                "performance_metrics": ["decision_accuracy", "execution_speed"]
            },
            "action_executor": {
                "name": "执行器",
                "category": "execution",
                "criticality": 0.75,
                "dependencies": ["planning_system", "resource_manager"],
                "performance_metrics": ["execution_speed", "resource_utilization_efficiency"]
            },
            "adaptation_engine": {
                "name": "适应引擎",
                "category": "execution",
                "criticality": 0.7,
                "dependencies": ["learning_system", "monitoring_system"],
                "performance_metrics": ["adaptation_speed", "adaptation_effectiveness"]
            },
            
            # 社会组件
            "communication_module": {
                "name": "通信模块",
                "category": "social",
                "criticality": 0.6,
                "dependencies": ["language_model", "emotion_recognizer"],
                "performance_metrics": ["communication_clarity_score", "understanding_accuracy"]
            },
            "collaboration_engine": {
                "name": "协作引擎",
                "category": "social",
                "criticality": 0.65,
                "dependencies": ["communication_module", "task_coordinator"],
                "performance_metrics": ["collaboration_efficiency", "conflict_resolution_success"]
            },
            "ethics_module": {
                "name": "伦理模块",
                "category": "social",
                "criticality": 0.9,
                "dependencies": ["value_system", "safety_monitor"],
                "performance_metrics": ["ethical_alignment_score", "security_violations"]
            },
            
            # 技术组件
            "resource_manager": {
                "name": "资源管理器",
                "category": "technical",
                "criticality": 0.7,
                "dependencies": ["monitoring_system", "scheduler"],
                "performance_metrics": ["memory_utilization", "algorithm_time_complexity"]
            },
            "monitoring_system": {
                "name": "监控系统",
                "category": "technical",
                "criticality": 0.8,
                "dependencies": ["data_collector", "alert_system"],
                "performance_metrics": ["system_uptime", "scalability_factor"]
            }
        }
        
        self.system_components = components
    
    def analyze_performance_data(self, performance_data: Dict[str, Any]) -> WeaknessAnalysisReport:
        """分析性能数据，识别认知弱点"""
        logger.info("开始分析性能数据，识别认知弱点")
        
        start_time = time.time()
        self.performance_data = performance_data
        
        # 识别弱点
        weaknesses = self._identify_weaknesses(performance_data)
        
        # 分析根因（如果启用）
        if self.automatic_root_cause_enabled:
            for weakness in weaknesses:
                self._analyze_root_causes(weakness)
        
        # 评估影响（如果启用）
        if self.impact_propagation_enabled:
            for weakness in weaknesses:
                self._assess_impact(weakness)
        
        # 识别改进机会
        for weakness in weaknesses:
            self._identify_improvement_opportunities(weakness)
        
        # 计算优先级
        for weakness in weaknesses:
            weakness.priority_score = self._calculate_priority_score(weakness)
        
        # 排序优先级
        weaknesses.sort(key=lambda w: w.priority_score, reverse=True)
        
        # 生成统计摘要
        statistical_summary = self._generate_statistical_summary(weaknesses)
        
        # 生成建议
        recommendations = self._generate_recommendations(weaknesses)
        
        # 创建报告
        report_id = f"weakness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = WeaknessAnalysisReport(
            report_id=report_id,
            timestamp=datetime.now(),
            analysis_duration_seconds=time.time() - start_time,
            weaknesses_identified=weaknesses,
            statistical_summary=statistical_summary,
            recommendations=recommendations,
            priority_ranking=[(w.weakness_id, w.priority_score) for w in weaknesses[:10]]
        )
        
        # 更新数据库
        for weakness in weaknesses:
            self.weaknesses_database[weakness.weakness_id] = weakness
        
        # 限制数据库大小
        if len(self.weaknesses_database) > self.max_weaknesses_tracked:
            # 移除优先级最低的旧弱点
            weaknesses_list = list(self.weaknesses_database.values())
            weaknesses_list.sort(key=lambda w: w.priority_score)
            to_remove = weaknesses_list[:len(weaknesses_list) - self.max_weaknesses_tracked]
            for weakness in to_remove:
                del self.weaknesses_database[weakness.weakness_id]
        
        # 记录历史
        self.weakness_history.append({
            "report_id": report_id,
            "timestamp": datetime.now(),
            "weaknesses_count": len(weaknesses),
            "average_severity": statistical_summary.get("average_severity_score", 0.0),
            "top_weakness": weaknesses[0].description if weaknesses else None
        })
        
        # 更新统计
        self.performance_stats['analyses_completed'] += 1
        self.performance_stats['weaknesses_identified'] += len(weaknesses)
        self.performance_stats['average_analysis_time'] = (
            (self.performance_stats['average_analysis_time'] * 
             (self.performance_stats['analyses_completed'] - 1) + 
             report.analysis_duration_seconds) / self.performance_stats['analyses_completed']
        )
        
        # 更新最后分析时间
        self.last_analysis_time = time.time()
        
        logger.info(f"弱点分析完成，识别弱点: {len(weaknesses)}，最高优先级: {weaknesses[0].priority_score if weaknesses else 0.0:.3f}")
        
        return report
    
    def _identify_weaknesses(self, performance_data: Dict[str, Any]) -> List[CognitiveWeakness]:
        """从性能数据中识别弱点"""
        weaknesses = []
        
        # 检查每个性能指标
        if "performance_dimensions" not in performance_data:
            logger.warning("性能数据缺少performance_dimensions字段")
            return weaknesses
        
        performance_dimensions = performance_data["performance_dimensions"]
        
        for dim_id, dim_data in performance_dimensions.items():
            current_value = dim_data.get("current_value", 0.0)
            target_value = dim_data.get("target_value", 1.0)
            
            # 计算性能差距
            performance_gap = target_value - current_value
            
            # 如果差距超过阈值，识别为潜在弱点
            if performance_gap > 0.1:  # 10%差距阈值
                # 确定严重程度
                severity_score = min(performance_gap * 2.0, 1.0)
                severity = self._determine_severity(severity_score)
                
                # 确定类别
                category = self._determine_category(dim_id)
                
                # 找到受影响的组件
                affected_components = self._find_affected_components(dim_id)
                
                # 创建弱点
                weakness_id = f"weakness_{dim_id}_{len(weaknesses) + 1:03d}"
                weakness = CognitiveWeakness(
                    weakness_id=weakness_id,
                    category=category,
                    severity=severity,
                    description=f"{dim_data.get('name', dim_id)}性能不足: 当前值{current_value:.3f}, 目标值{target_value:.3f}",
                    affected_components=affected_components,
                    performance_metrics_affected=[dim_id],
                    root_causes=[],
                    impact_assessment={},
                    improvement_opportunities=[],
                    estimated_fix_effort=self._estimate_fix_effort(severity_score, category),
                    priority_score=0.0,
                    evidence=[
                        WeaknessEvidence(
                            evidence_id=f"evidence_{weakness_id}_1",
                            evidence_type="performance_metric",
                            description=f"性能指标{dim_id}低于目标值",
                            data_source="performance_evaluator",
                            confidence=0.85,
                            timestamp=datetime.now(),
                            metrics={dim_id: current_value}
                        )
                    ]
                )
                
                weaknesses.append(weakness)
        
        return weaknesses
    
    def _determine_severity(self, severity_score: float) -> WeaknessSeverity:
        """确定弱点严重程度"""
        if severity_score >= self.config['severity_thresholds']['critical']:
            return WeaknessSeverity.CRITICAL
        elif severity_score >= self.config['severity_thresholds']['high']:
            return WeaknessSeverity.HIGH
        elif severity_score >= self.config['severity_thresholds']['medium']:
            return WeaknessSeverity.MEDIUM
        else:
            return WeaknessSeverity.LOW
    
    def _determine_category(self, metric_id: str) -> WeaknessCategory:
        """根据指标ID确定弱点类别"""
        if "reasoning" in metric_id or "learning" in metric_id or "planning" in metric_id:
            return WeaknessCategory.COGNITIVE
        elif "decision" in metric_id or "execution" in metric_id or "adaptation" in metric_id:
            return WeaknessCategory.EXECUTION
        elif "communication" in metric_id or "collaboration" in metric_id or "ethical" in metric_id:
            return WeaknessCategory.SOCIAL
        elif "algorithm" in metric_id or "resource" in metric_id or "system" in metric_id:
            return WeaknessCategory.TECHNICAL
        else:
            return WeaknessCategory.SYSTEMIC
    
    def _find_affected_components(self, metric_id: str) -> List[str]:
        """找到受影响的系统组件"""
        affected_components = []
        
        for comp_id, comp_data in self.system_components.items():
            performance_metrics = comp_data.get("performance_metrics", [])
            if metric_id in performance_metrics:
                affected_components.append(comp_id)
        
        return affected_components
    
    def _estimate_fix_effort(self, severity_score: float, category: WeaknessCategory) -> float:
        """估计修复工作量（人天）"""
        # 基础工作量
        base_effort = 2.0
        
        # 根据严重程度调整
        severity_multiplier = 1.0 + severity_score * 2.0  # 1.0-3.0
        
        # 根据类别调整
        category_multipliers = {
            WeaknessCategory.COGNITIVE: 1.5,
            WeaknessCategory.EXECUTION: 1.2,
            WeaknessCategory.SOCIAL: 1.8,
            WeaknessCategory.TECHNICAL: 1.0,
            WeaknessCategory.SYSTEMIC: 2.0
        }
        category_multiplier = category_multipliers.get(category, 1.0)
        
        return base_effort * severity_multiplier * category_multiplier
    
    def _analyze_root_causes(self, weakness: CognitiveWeakness):
        """分析弱点根因"""
        root_causes = []
        
        # 基于弱点类型分析可能的根因
        if weakness.category == WeaknessCategory.COGNITIVE:
            root_causes.extend(self._analyze_cognitive_root_causes(weakness))
        elif weakness.category == WeaknessCategory.EXECUTION:
            root_causes.extend(self._analyze_execution_root_causes(weakness))
        elif weakness.category == WeaknessCategory.SOCIAL:
            root_causes.extend(self._analyze_social_root_causes(weakness))
        elif weakness.category == WeaknessCategory.TECHNICAL:
            root_causes.extend(self._analyze_technical_root_causes(weakness))
        
        # 限制根因数量
        root_causes.sort(key=lambda rc: rc.confidence, reverse=True)
        weakness.root_causes = root_causes[:self.config['max_root_causes_per_weakness']]
        
        # 更新统计
        self.performance_stats['root_causes_analyzed'] += len(weakness.root_causes)
    
    def _analyze_cognitive_root_causes(self, weakness: CognitiveWeakness) -> List[RootCauseAnalysis]:
        """分析认知弱点根因"""
        root_causes = []
        
        possible_causes = [
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_data",
                root_cause_type=RootCauseType.DATA_QUALITY,
                description=f"{weakness.description}的训练数据质量不足",
                contributing_factors=["训练数据量不足", "数据标注不准确", "数据分布不平衡"],
                evidence=weakness.evidence,
                confidence=0.7,
                impact_score=0.8,
                fix_complexity=0.6
            ),
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_algo",
                root_cause_type=RootCauseType.ALGORITHM_LIMITATION,
                description=f"{weakness.description}的算法实现有局限",
                contributing_factors=["算法复杂度不足", "超参数未优化", "模型容量不够"],
                evidence=weakness.evidence,
                confidence=0.65,
                impact_score=0.75,
                fix_complexity=0.7
            ),
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_training",
                root_cause_type=RootCauseType.TRAINING_INSUFFICIENCY,
                description=f"{weakness.description}的训练不充分",
                contributing_factors=["训练迭代次数不足", "学习率设置不当", "正则化不够"],
                evidence=weakness.evidence,
                confidence=0.6,
                impact_score=0.7,
                fix_complexity=0.5
            )
        ]
        
        return possible_causes
    
    def _analyze_execution_root_causes(self, weakness: CognitiveWeakness) -> List[RootCauseAnalysis]:
        """分析执行弱点根因"""
        root_causes = []
        
        possible_causes = [
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_resource",
                root_cause_type=RootCauseType.RESOURCE_CONSTRAINT,
                description=f"{weakness.description}的资源约束",
                contributing_factors=["计算资源不足", "内存限制", "存储空间不够"],
                evidence=weakness.evidence,
                confidence=0.75,
                impact_score=0.8,
                fix_complexity=0.4
            ),
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_design",
                root_cause_type=RootCauseType.DESIGN_FLAW,
                description=f"{weakness.description}的设计缺陷",
                contributing_factors=["系统架构不合理", "组件接口不匹配", "流程设计有缺陷"],
                evidence=weakness.evidence,
                confidence=0.7,
                impact_score=0.85,
                fix_complexity=0.8
            )
        ]
        
        return possible_causes
    
    def _analyze_social_root_causes(self, weakness: CognitiveWeakness) -> List[RootCauseAnalysis]:
        """分析社会弱点根因"""
        root_causes = []
        
        possible_causes = [
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_training_social",
                root_cause_type=RootCauseType.TRAINING_INSUFFICIENCY,
                description=f"{weakness.description}的社会交互训练不足",
                contributing_factors=["交互数据缺乏", "场景覆盖不全", "反馈机制不完善"],
                evidence=weakness.evidence,
                confidence=0.8,
                impact_score=0.75,
                fix_complexity=0.6
            ),
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_algo_social",
                root_cause_type=RootCauseType.ALGORITHM_LIMITATION,
                description=f"{weakness.description}的社会算法局限",
                contributing_factors=["情感理解不足", "意图识别不准", "上下文理解有限"],
                evidence=weakness.evidence,
                confidence=0.7,
                impact_score=0.8,
                fix_complexity=0.7
            )
        ]
        
        return possible_causes
    
    def _analyze_technical_root_causes(self, weakness: CognitiveWeakness) -> List[RootCauseAnalysis]:
        """分析技术弱点根因"""
        root_causes = []
        
        possible_causes = [
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_config",
                root_cause_type=RootCauseType.CONFIGURATION_ERROR,
                description=f"{weakness.description}的配置错误",
                contributing_factors=["参数设置不当", "环境配置错误", "依赖版本不匹配"],
                evidence=weakness.evidence,
                confidence=0.85,
                impact_score=0.6,
                fix_complexity=0.3
            ),
            RootCauseAnalysis(
                root_cause_id=f"rc_{weakness.weakness_id}_resource_tech",
                root_cause_type=RootCauseType.RESOURCE_CONSTRAINT,
                description=f"{weakness.description}的技术资源约束",
                contributing_factors=["硬件性能不足", "网络带宽限制", "存储IO瓶颈"],
                evidence=weakness.evidence,
                confidence=0.75,
                impact_score=0.7,
                fix_complexity=0.5
            )
        ]
        
        return possible_causes
    
    def _assess_impact(self, weakness: CognitiveWeakness):
        """评估弱点影响"""
        impact_scores = {
            "performance": 0.0,
            "reliability": 0.0,
            "scalability": 0.0,
            "security": 0.0,
            "overall": 0.0
        }
        
        # 基于严重程度计算基础影响
        severity_multipliers = {
            WeaknessSeverity.CRITICAL: 1.0,
            WeaknessSeverity.HIGH: 0.75,
            WeaknessSeverity.MEDIUM: 0.5,
            WeaknessSeverity.LOW: 0.25
        }
        base_impact = severity_multipliers.get(weakness.severity, 0.5)
        
        # 基于类别调整影响
        category_weights = {
            WeaknessCategory.COGNITIVE: {"performance": 0.4, "reliability": 0.3, "scalability": 0.2, "security": 0.1},
            WeaknessCategory.EXECUTION: {"performance": 0.5, "reliability": 0.3, "scalability": 0.1, "security": 0.1},
            WeaknessCategory.SOCIAL: {"performance": 0.3, "reliability": 0.2, "scalability": 0.1, "security": 0.4},
            WeaknessCategory.TECHNICAL: {"performance": 0.4, "reliability": 0.3, "scalability": 0.2, "security": 0.1},
            WeaknessCategory.SYSTEMIC: {"performance": 0.3, "reliability": 0.3, "scalability": 0.2, "security": 0.2}
        }
        
        weights = category_weights.get(weakness.category, {"performance": 0.25, "reliability": 0.25, "scalability": 0.25, "security": 0.25})
        
        # 计算各维度影响
        for dimension in ["performance", "reliability", "scalability", "security"]:
            weight = weights.get(dimension, 0.25)
            impact_scores[dimension] = base_impact * weight
        
        # 计算总体影响
        impact_scores["overall"] = sum(
            impact_scores[dim] * self.config['impact_weight_factors'].get(dim, 0.25)
            for dim in ["performance", "reliability", "scalability", "security"]
        )
        
        weakness.impact_assessment = impact_scores
    
    def _identify_improvement_opportunities(self, weakness: CognitiveWeakness):
        """识别改进机会"""
        opportunities = []
        
        # 基于弱点类别生成改进机会
        if weakness.category == WeaknessCategory.COGNITIVE:
            opportunities = [
                f"优化{weakness.description}的算法实现",
                f"增加{weakness.description}的训练数据",
                f"改进{weakness.description}的评估方法",
                f"引入新的{weakness.description}技术"
            ]
        elif weakness.category == WeaknessCategory.EXECUTION:
            opportunities = [
                f"优化{weakness.description}的资源分配",
                f"改进{weakness.description}的调度策略",
                f"增强{weakness.description}的容错机制",
                f"提升{weakness.description}的并行处理能力"
            ]
        elif weakness.category == WeaknessCategory.SOCIAL:
            opportunities = [
                f"增加{weakness.description}的交互训练",
                f"改进{weakness.description}的情感理解",
                f"优化{weakness.description}的沟通策略",
                f"增强{weakness.description}的伦理约束"
            ]
        elif weakness.category == WeaknessCategory.TECHNICAL:
            opportunities = [
                f"优化{weakness.description}的系统配置",
                f"升级{weakness.description}的硬件资源",
                f"改进{weakness.description}的监控机制",
                f"增强{weakness.description}的安全防护"
            ]
        else:
            opportunities = [
                f"系统性地改进{weakness.description}",
                f"重新设计{weakness.description}的架构",
                f"全面优化{weakness.description}的实现",
                f"引入创新方案解决{weakness.description}"
            ]
        
        weakness.improvement_opportunities = opportunities
        
        # 更新统计
        self.performance_stats['improvement_opportunities_found'] += len(opportunities)
    
    def _calculate_priority_score(self, weakness: CognitiveWeakness) -> float:
        """计算弱点优先级得分"""
        # 基础优先级：严重程度 × 影响
        severity_scores = {
            WeaknessSeverity.CRITICAL: 1.0,
            WeaknessSeverity.HIGH: 0.75,
            WeaknessSeverity.MEDIUM: 0.5,
            WeaknessSeverity.LOW: 0.25
        }
        severity_score = severity_scores.get(weakness.severity, 0.5)
        
        impact_score = weakness.impact_assessment.get("overall", 0.5)
        
        # 考虑修复复杂度（复杂度越高，优先级可能越低）
        complexity_factor = 1.0 - (weakness.estimated_fix_effort / 10.0)  # 标准化
        complexity_factor = max(0.3, min(1.0, complexity_factor))
        
        # 计算最终优先级
        priority_score = severity_score * impact_score * complexity_factor
        
        return priority_score
    
    def _generate_statistical_summary(self, weaknesses: List[CognitiveWeakness]) -> Dict[str, Any]:
        """生成统计摘要"""
        if not weaknesses:
            return {
                "weaknesses_count": 0,
                "average_severity_score": 0.0,
                "category_distribution": {},
                "top_weaknesses": []
            }
        
        # 计算平均严重程度
        severity_scores = {
            WeaknessSeverity.CRITICAL: 1.0,
            WeaknessSeverity.HIGH: 0.75,
            WeaknessSeverity.MEDIUM: 0.5,
            WeaknessSeverity.LOW: 0.25
        }
        
        avg_severity = np.mean([severity_scores.get(w.severity, 0.5) for w in weaknesses])
        
        # 类别分布
        category_distribution = {}
        for weakness in weaknesses:
            category = weakness.category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # 最高优先级弱点
        top_weaknesses = []
        for weakness in weaknesses[:5]:
            top_weaknesses.append({
                "id": weakness.weakness_id,
                "description": weakness.description,
                "severity": weakness.severity.value,
                "priority": weakness.priority_score,
                "category": weakness.category.value
            })
        
        return {
            "weaknesses_count": len(weaknesses),
            "average_severity_score": avg_severity,
            "category_distribution": category_distribution,
            "top_weaknesses": top_weaknesses,
            "estimated_total_fix_effort": sum(w.estimated_fix_effort for w in weaknesses),
            "average_impact_score": np.mean([w.impact_assessment.get("overall", 0.0) for w in weaknesses])
        }
    
    def _generate_recommendations(self, weaknesses: List[CognitiveWeakness]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not weaknesses:
            recommendations.append("未发现明显认知弱点，系统性能良好。")
            return recommendations
        
        # 按优先级排序
        sorted_weaknesses = sorted(weaknesses, key=lambda w: w.priority_score, reverse=True)
        
        # 为每个高优先级弱点生成建议
        for i, weakness in enumerate(sorted_weaknesses[:5]):
            recommendation = (
                f"建议{i+1}: 优先解决'{weakness.description}'。"
                f"严重程度: {weakness.severity.value}，"
                f"优先级得分: {weakness.priority_score:.3f}，"
                f"预计修复工作量: {weakness.estimated_fix_effort:.1f}人天。"
            )
            
            # 添加具体建议
            if weakness.root_causes:
                top_root_cause = weakness.root_causes[0]
                recommendation += f" 主要根因: {top_root_cause.description}。"
            
            if weakness.improvement_opportunities:
                top_opportunity = weakness.improvement_opportunities[0]
                recommendation += f" 改进方向: {top_opportunity}。"
            
            recommendations.append(recommendation)
        
        # 添加总体建议
        critical_count = sum(1 for w in weaknesses if w.severity == WeaknessSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append(f"发现{critical_count}个严重弱点，建议立即处理。")
        
        total_effort = sum(w.estimated_fix_effort for w in weaknesses)
        recommendations.append(f"总计修复工作量估计: {total_effort:.1f}人天。")
        
        return recommendations
    
    def get_weakness_dashboard(self) -> Dict[str, Any]:
        """获取弱点仪表板数据"""
        active_weaknesses = [w for w in self.weaknesses_database.values() if w.status == "active"]
        
        dashboard = {
            "timestamp": datetime.now(),
            "total_weaknesses_tracked": len(self.weaknesses_database),
            "active_weaknesses": len(active_weaknesses),
            "performance_stats": self.performance_stats,
            "severity_distribution": {},
            "category_distribution": {},
            "top_priority_weaknesses": []
        }
        
        # 严重程度分布
        severity_counts = {}
        for weakness in active_weaknesses:
            severity = weakness.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        dashboard["severity_distribution"] = severity_counts
        
        # 类别分布
        category_counts = {}
        for weakness in active_weaknesses:
            category = weakness.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        dashboard["category_distribution"] = category_counts
        
        # 最高优先级弱点
        active_weaknesses.sort(key=lambda w: w.priority_score, reverse=True)
        for weakness in active_weaknesses[:5]:
            dashboard["top_priority_weaknesses"].append({
                "id": weakness.weakness_id,
                "description": weakness.description,
                "severity": weakness.severity.value,
                "priority": weakness.priority_score,
                "category": weakness.category.value,
                "status": weakness.status,
                "detected_at": weakness.detected_at.isoformat()
            })
        
        return dashboard
    
    def track_weakness_resolution(self, weakness_id: str, resolution_status: str, notes: str = ""):
        """跟踪弱点解决状态"""
        if weakness_id not in self.weaknesses_database:
            logger.warning(f"弱点ID不存在: {weakness_id}")
            return
        
        weakness = self.weaknesses_database[weakness_id]
        weakness.status = resolution_status
        weakness.last_analyzed_at = datetime.now()
        
        # 记录解决历史
        resolution_record = {
            "weakness_id": weakness_id,
            "timestamp": datetime.now(),
            "old_status": weakness.status,
            "new_status": resolution_status,
            "notes": notes
        }
        
        # 更新解决率统计
        if resolution_status in ["resolved", "mitigated"]:
            resolved_count = sum(1 for w in self.weaknesses_database.values() 
                               if w.status in ["resolved", "mitigated"])
            total_count = len(self.weaknesses_database)
            self.performance_stats['resolution_rate'] = resolved_count / total_count if total_count > 0 else 0.0
        
        logger.info(f"弱点解决状态更新: {weakness_id} -> {resolution_status}")

# 全局实例
cognitive_weakness_analyzer_instance = CognitiveWeaknessAnalyzer()

if __name__ == "__main__":
    # 测试认知弱点分析器
    print("测试认知弱点分析器...")
    
    analyzer = CognitiveWeaknessAnalyzer(
        analysis_frequency_hours=1.0,
        automatic_root_cause_enabled=True,
        impact_propagation_enabled=True
    )
    
    # 模拟性能数据
    performance_data = {
        "performance_dimensions": {
            "logical_reasoning_accuracy": {
                "name": "逻辑推理准确率",
                "current_value": 0.75,
                "target_value": 0.95,
                "weight": 0.15
            },
            "causal_inference_effectiveness": {
                "name": "因果推断有效性",
                "current_value": 0.65,
                "target_value": 0.85,
                "weight": 0.12
            },
            "complex_task_completion_rate": {
                "name": "复杂任务完成率",
                "current_value": 0.70,
                "target_value": 0.90,
                "weight": 0.07
            }
        }
    }
    
    # 执行分析
    report = analyzer.analyze_performance_data(performance_data)
    print(f"弱点分析完成，识别弱点: {len(report.weaknesses_identified)}")
    
    # 获取仪表板
    dashboard = analyzer.get_weakness_dashboard()
    print(f"弱点仪表板: 活动弱点 {dashboard['active_weaknesses']} 个")
    
    # 显示建议
    for i, rec in enumerate(report.recommendations[:3]):
        print(f"建议{i+1}: {rec}")