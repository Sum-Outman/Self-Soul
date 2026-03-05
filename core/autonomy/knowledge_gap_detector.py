#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
知识缺口检测器 - 检测和分析知识体系中的空白、不一致和不足

核心功能:
1. 知识完整性分析: 分析知识体系的完整性和覆盖度
2. 不一致性检测: 检测知识中的矛盾、冲突和逻辑不一致
3. 模糊性识别: 识别模糊、不明确或定义不清的概念
4. 依赖性分析: 分析知识间的依赖关系和缺失链接
5. 缺口重要性评估: 评估知识缺口的重要性和紧迫性
6. 学习目标生成: 生成填补知识缺口的学习目标

知识缺口类型:
1. 概念缺失 (Concept Missing):
   - 领域中的重要概念未被定义或描述
   - 关键术语缺乏明确的定义或解释
   - 基础概念缺失导致理解困难

2. 关系缺失 (Relation Missing):
   - 概念间的关键关系未被建立
   - 因果、层次、相似性关系缺失
   - 知识孤岛缺乏连接

3. 属性不完整 (Attribute Incomplete):
   - 概念的属性描述不完整
   - 缺少关键特征或参数信息
   - 属性值范围或约束未定义

4. 不一致性 (Inconsistency):
   - 不同知识源间的矛盾
   - 逻辑推理得出的冲突结论
   - 相同概念的不同定义或描述

5. 模糊性 (Ambiguity):
   - 概念定义模糊不清
   - 术语的多重含义未澄清
   - 上下文依赖性强导致不确定性

6. 证据不足 (Evidence Deficiency):
   - 知识主张缺乏足够证据支持
   - 推理链中的假设未验证
   - 统计或实验数据缺失

检测方法:
1. 模式匹配: 基于已知知识模式检测缺失
2. 推理验证: 通过逻辑推理发现不一致
3. 对比分析: 对比不同知识源发现差异
4. 依赖分析: 分析知识依赖关系发现断裂
5. 统计分析: 基于统计方法识别异常

重要性评估因素:
1. 领域相关性: 缺口在领域中的重要性
2. 影响范围: 缺口对其他知识的影响程度
3. 使用频率: 相关概念的访问和使用频率
4. 学习难度: 填补缺口的预期难度
5. 紧迫性: 填补缺口的紧迫程度

版权所有 (c) 2026 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import logging
import time
import math
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeGapType(Enum):
    """知识缺口类型枚举"""
    CONCEPT_MISSING = "concept_missing"          # 概念缺失
    RELATION_MISSING = "relation_missing"        # 关系缺失
    ATTRIBUTE_INCOMPLETE = "attribute_incomplete"  # 属性不完整
    INCONSISTENCY = "inconsistency"             # 不一致性
    AMBIGUITY = "ambiguity"                     # 模糊性
    EVIDENCE_DEFICIENCY = "evidence_deficiency"  # 证据不足


class GapSeverity(Enum):
    """缺口严重性枚举"""
    LOW = "low"         # 低严重性：轻微影响
    MEDIUM = "medium"   # 中等严重性：局部影响
    HIGH = "high"       # 高严重性：显著影响
    CRITICAL = "critical"  # 关键严重性：系统性影响


class GapDetectionMethod(Enum):
    """缺口检测方法枚举"""
    PATTERN_MATCHING = "pattern_matching"      # 模式匹配
    REASONING_VERIFICATION = "reasoning_verification"  # 推理验证
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # 对比分析
    DEPENDENCY_ANALYSIS = "dependency_analysis"  # 依赖分析
    STATISTICAL_ANALYSIS = "statistical_analysis"  # 统计分析


@dataclass
class KnowledgeGap:
    """知识缺口数据类"""
    id: str
    gap_type: KnowledgeGapType
    description: str
    affected_concepts: List[str]
    severity: GapSeverity
    detection_method: GapDetectionMethod
    
    # 重要性评估
    importance_score: float = 0.0
    domain_relevance: float = 0.5
    impact_scope: float = 0.5
    usage_frequency: float = 0.0
    learning_difficulty: float = 0.5
    urgency: float = 0.5
    
    # 元数据
    detected_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    suggested_solutions: List[str] = field(default_factory=list)
    related_gaps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.importance_score = max(0.0, min(1.0, self.importance_score))
        self.domain_relevance = max(0.0, min(1.0, self.domain_relevance))
        self.impact_scope = max(0.0, min(1.0, self.impact_scope))
        self.usage_frequency = max(0.0, min(1.0, self.usage_frequency))
        self.learning_difficulty = max(0.0, min(1.0, self.learning_difficulty))
        self.urgency = max(0.0, min(1.0, self.urgency))
        
        if not self.id:
            self.id = f"gap_{int(time.time())}_{(zlib.adler32(str(self.description).encode('utf-8')) & 0xffffffff) % 10000}"
    
    @property
    def composite_importance(self) -> float:
        """计算综合重要性分数"""
        # 综合多个因素的加权重要性
        weights = {
            'severity': 0.3,
            'domain_relevance': 0.2,
            'impact_scope': 0.2,
            'usage_frequency': 0.15,
            'urgency': 0.15
        }
        
        # 严重性映射到数值
        severity_score = {
            GapSeverity.LOW: 0.25,
            GapSeverity.MEDIUM: 0.5,
            GapSeverity.HIGH: 0.75,
            GapSeverity.CRITICAL: 1.0
        }.get(self.severity, 0.5)
        
        composite = (
            severity_score * weights['severity'] +
            self.domain_relevance * weights['domain_relevance'] +
            self.impact_scope * weights['impact_scope'] +
            self.usage_frequency * weights['usage_frequency'] +
            self.urgency * weights['urgency']
        )
        
        return min(1.0, composite)
    
    def update_importance(self, new_importance: float):
        """更新重要性分数"""
        self.importance_score = max(0.0, min(1.0, new_importance))
        self.last_updated = time.time()
    
    def add_evidence(self, evidence: str):
        """添加证据"""
        self.evidence.append(evidence)
    
    def add_solution(self, solution: str):
        """添加解决方案建议"""
        self.suggested_solutions.append(solution)


@dataclass
class GapDetectionResult:
    """缺口检测结果数据类"""
    gaps_detected: List[KnowledgeGap]
    detection_time: float = field(default_factory=time.time)
    detection_method: GapDetectionMethod = GapDetectionMethod.PATTERN_MATCHING
    coverage_score: float = 0.0  # 检测覆盖度
    confidence_score: float = 0.0  # 检测置信度
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化验证"""
        self.coverage_score = max(0.0, min(1.0, self.coverage_score))
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
    
    @property
    def total_gaps(self) -> int:
        """获取总缺口数量"""
        return len(self.gaps_detected)
    
    @property
    def average_importance(self) -> float:
        """获取平均重要性分数"""
        if not self.gaps_detected:
            return 0.0
        return sum(gap.composite_importance for gap in self.gaps_detected) / len(self.gaps_detected)
    
    def get_gaps_by_severity(self, severity: GapSeverity) -> List[KnowledgeGap]:
        """按严重性筛选缺口"""
        return [gap for gap in self.gaps_detected if gap.severity == severity]
    
    def get_top_gaps(self, n: int = 10) -> List[KnowledgeGap]:
        """获取重要性最高的n个缺口"""
        sorted_gaps = sorted(self.gaps_detected, 
                           key=lambda g: g.composite_importance, 
                           reverse=True)
        return sorted_gaps[:n]


class KnowledgeGapDetector:
    """
    知识缺口检测器
    
    核心组件:
    1. 知识分析器: 分析知识结构和内容
    2. 模式检测器: 基于模式匹配检测缺失
    3. 推理验证器: 通过逻辑推理发现不一致
    4. 对比分析器: 对比不同知识源发现差异
    5. 重要性评估器: 评估缺口的重要性和紧迫性
    6. 目标生成器: 生成填补缺口的学习目标
    
    检测流程:
    知识输入 → 知识分析器 → 结构分析 → 模式检测器 → 模式匹配
    知识关系 → 推理验证器 → 逻辑验证 → 对比分析器 → 差异检测
    检测结果 → 重要性评估器 → 评估重要性 → 目标生成器 → 生成目标
    
    技术特性:
    - 多方法检测: 结合多种检测方法提高准确性
    - 重要性评估: 基于多个维度评估缺口重要性
    - 学习目标生成: 自动生成填补缺口的具体目标
    - 持续监控: 持续监控知识体系的变化和缺口
    """
    
    def __init__(self,
                 knowledge_manager: Optional[Any] = None,
                 detection_threshold: float = 0.3,
                 min_importance_threshold: float = 0.2,
                 enable_continuous_monitoring: bool = True):
        """
        初始化知识缺口检测器
        
        Args:
            knowledge_manager: 知识管理器实例（可选）
            detection_threshold: 检测阈值
            min_importance_threshold: 最小重要性阈值
            enable_continuous_monitoring: 启用持续监控
        """
        self.knowledge_manager = knowledge_manager
        self.detection_threshold = detection_threshold
        self.min_importance_threshold = min_importance_threshold
        self.enable_continuous_monitoring = enable_continuous_monitoring
        
        # 知识缺口存储
        self.detected_gaps: Dict[str, KnowledgeGap] = {}
        self.gap_history: List[KnowledgeGap] = []
        self.max_history_size = 1000
        
        # 检测模式库
        self.detection_patterns = self._initialize_detection_patterns()
        
        # 领域知识模板
        self.domain_templates = self._initialize_domain_templates()
        
        # 持续监控状态
        self.monitoring_active = False
        self.last_monitoring_time = time.time()
        self.monitoring_interval = 3600.0  # 1小时
        
        # 性能统计
        self.performance_stats = {
            'total_detections': 0,
            'gaps_detected': 0,
            'high_severity_gaps': 0,
            'detection_time_total': 0.0,
            'average_detection_time': 0.0,
            'false_positives': 0,
            'gaps_filled': 0
        }
        
        logger.info(f"知识缺口检测器初始化完成，检测阈值: {detection_threshold}")
    
    def _initialize_detection_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化检测模式库"""
        patterns = {
            'concept_missing': {
                'description': '概念缺失检测模式',
                'indicators': [
                    '术语缺乏定义',
                    '概念引用但未定义',
                    '领域基础概念缺失',
                    '重要实体未被描述'
                ],
                'confidence_weight': 0.8
            },
            'relation_missing': {
                'description': '关系缺失检测模式',
                'indicators': [
                    '概念间缺乏连接',
                    '因果链断裂',
                    '层次关系不完整',
                    '相似性关系缺失'
                ],
                'confidence_weight': 0.7
            },
            'attribute_incomplete': {
                'description': '属性不完整检测模式',
                'indicators': [
                    '属性列表不完整',
                    '关键属性缺失',
                    '属性值范围未定义',
                    '属性约束未指定'
                ],
                'confidence_weight': 0.6
            },
            'inconsistency': {
                'description': '不一致性检测模式',
                'indicators': [
                    '不同来源的冲突描述',
                    '逻辑推理矛盾',
                    '数值或事实冲突',
                    '时序或因果矛盾'
                ],
                'confidence_weight': 0.9
            },
            'ambiguity': {
                'description': '模糊性检测模式',
                'indicators': [
                    '定义模糊不清',
                    '术语多重含义未澄清',
                    '上下文依赖性强',
                    '边界条件不明确'
                ],
                'confidence_weight': 0.5
            }
        }
        
        return patterns
    
    def _initialize_domain_templates(self) -> Dict[str, Dict[str, Any]]:
        """初始化领域知识模板"""
        templates = {
            'computer_science': {
                'required_concepts': ['algorithm', 'data_structure', 'programming_language', 'software_engineering'],
                'required_relations': ['implements', 'extends', 'depends_on', 'optimizes'],
                'common_attributes': ['complexity', 'efficiency', 'reliability', 'scalability']
            },
            'mathematics': {
                'required_concepts': ['theorem', 'proof', 'axiom', 'lemma', 'corollary'],
                'required_relations': ['proves', 'follows_from', 'equivalent_to', 'generalizes'],
                'common_attributes': ['validity', 'applicability', 'generality', 'elegance']
            },
            'physics': {
                'required_concepts': ['law', 'theory', 'experiment', 'measurement', 'model'],
                'required_relations': ['explains', 'predicts', 'contradicts', 'supports'],
                'common_attributes': ['accuracy', 'precision', 'testability', 'falsifiability']
            }
        }
        
        return templates
    
    def detect_gaps(self, 
                   knowledge_data: Optional[Dict[str, Any]] = None,
                   domain: Optional[str] = None,
                   detection_methods: Optional[List[GapDetectionMethod]] = None) -> GapDetectionResult:
        """
        检测知识缺口
        
        Args:
            knowledge_data: 知识数据（如果为None，使用知识管理器）
            domain: 领域名称（用于领域特定检测）
            detection_methods: 使用的检测方法列表
            
        Returns:
            缺口检测结果
        """
        start_time = time.time()
        
        # 确定使用的检测方法
        if detection_methods is None:
            detection_methods = [
                GapDetectionMethod.PATTERN_MATCHING,
                GapDetectionMethod.REASONING_VERIFICATION,
                GapDetectionMethod.COMPARATIVE_ANALYSIS
            ]
        
        # 获取知识数据
        if knowledge_data is None and self.knowledge_manager is not None:
            # 从知识管理器获取数据
            try:
                knowledge_data = self._extract_knowledge_from_manager()
            except Exception as e:
                logger.error(f"从知识管理器提取数据失败: {e}")
                knowledge_data = {}
        elif knowledge_data is None:
            knowledge_data = {}
        
        # 执行检测
        detected_gaps = []
        
        # 模式匹配检测
        if GapDetectionMethod.PATTERN_MATCHING in detection_methods:
            pattern_gaps = self._detect_by_pattern_matching(knowledge_data, domain)
            detected_gaps.extend(pattern_gaps)
        
        # 推理验证检测
        if GapDetectionMethod.REASONING_VERIFICATION in detection_methods:
            reasoning_gaps = self._detect_by_reasoning_verification(knowledge_data)
            detected_gaps.extend(reasoning_gaps)
        
        # 对比分析检测
        if GapDetectionMethod.COMPARATIVE_ANALYSIS in detection_methods:
            comparative_gaps = self._detect_by_comparative_analysis(knowledge_data, domain)
            detected_gaps.extend(comparative_gaps)
        
        # 依赖分析检测
        if GapDetectionMethod.DEPENDENCY_ANALYSIS in detection_methods:
            dependency_gaps = self._detect_by_dependency_analysis(knowledge_data)
            detected_gaps.extend(dependency_gaps)
        
        # 过滤重要性低的缺口
        filtered_gaps = [gap for gap in detected_gaps if gap.composite_importance >= self.min_importance_threshold]
        
        # 去除重复缺口（基于描述相似性）
        unique_gaps = self._deduplicate_gaps(filtered_gaps)
        
        # 更新存储
        for gap in unique_gaps:
            self.detected_gaps[gap.id] = gap
        
        # 创建检测结果
        result = GapDetectionResult(
            gaps_detected=unique_gaps,
            detection_method=detection_methods[0] if detection_methods else GapDetectionMethod.PATTERN_MATCHING,
            coverage_score=self._calculate_coverage_score(knowledge_data, unique_gaps),
            confidence_score=self._calculate_confidence_score(unique_gaps),
            summary={
                'domain': domain,
                'detection_methods': [m.value for m in detection_methods],
                'total_concepts_analyzed': len(knowledge_data.get('concepts', {})),
                'total_relations_analyzed': len(knowledge_data.get('relations', [])),
                'detection_time': time.time() - start_time
            }
        )
        
        # 更新性能统计
        detection_time = time.time() - start_time
        self._update_performance_stats(detection_time, len(unique_gaps))
        
        logger.info(f"知识缺口检测完成: 发现 {len(unique_gaps)} 个缺口，覆盖度: {result.coverage_score:.2f}")
        
        return result
    
    def _extract_knowledge_from_manager(self) -> Dict[str, Any]:
        """从知识管理器提取知识数据"""
        # 简化实现：返回模拟数据
        # 实际实现应该与知识管理器交互
        return {
            'concepts': {},
            'relations': [],
            'attributes': {},
            'metadata': {
                'extraction_time': time.time(),
                'source': 'knowledge_manager'
            }
        }
    
    def _detect_by_pattern_matching(self, knowledge_data: Dict[str, Any], domain: Optional[str]) -> List[KnowledgeGap]:
        """基于模式匹配检测缺口"""
        gaps = []
        
        # 概念缺失检测
        concept_gaps = self._detect_missing_concepts(knowledge_data, domain)
        gaps.extend(concept_gaps)
        
        # 关系缺失检测
        relation_gaps = self._detect_missing_relations(knowledge_data, domain)
        gaps.extend(relation_gaps)
        
        # 属性不完整检测
        attribute_gaps = self._detect_incomplete_attributes(knowledge_data, domain)
        gaps.extend(attribute_gaps)
        
        return gaps
    
    def _detect_missing_concepts(self, knowledge_data: Dict[str, Any], domain: Optional[str]) -> List[KnowledgeGap]:
        """检测缺失的概念"""
        gaps = []
        
        # 获取领域模板
        domain_template = self.domain_templates.get(domain or 'general', {})
        required_concepts = domain_template.get('required_concepts', [])
        
        # 获取现有概念
        existing_concepts = set(knowledge_data.get('concepts', {}).keys())
        
        # 检测缺失的概念
        for concept in required_concepts:
            if concept not in existing_concepts:
                gap = KnowledgeGap(
                    id=f"concept_missing_{concept}_{int(time.time())}",
                    gap_type=KnowledgeGapType.CONCEPT_MISSING,
                    description=f"概念 '{concept}' 在领域 '{domain}' 中缺失",
                    affected_concepts=[concept],
                    severity=GapSeverity.MEDIUM,
                    detection_method=GapDetectionMethod.PATTERN_MATCHING,
                    domain_relevance=0.8 if domain else 0.5,
                    impact_scope=0.6,
                    learning_difficulty=0.5
                )
                gaps.append(gap)
        
        return gaps
    
    def _detect_missing_relations(self, knowledge_data: Dict[str, Any], domain: Optional[str]) -> List[KnowledgeGap]:
        """检测缺失的关系"""
        gaps = []
        
        # 获取领域模板
        domain_template = self.domain_templates.get(domain or 'general', {})
        required_relations = domain_template.get('required_relations', [])
        
        # 获取现有关系
        existing_relations = knowledge_data.get('relations', [])
        relation_types = set()
        
        for rel in existing_relations:
            if isinstance(rel, dict) and 'type' in rel:
                relation_types.add(rel['type'])
        
        # 检测缺失的关系类型
        for rel_type in required_relations:
            if rel_type not in relation_types:
                # 查找可能相关的概念
                concepts = list(knowledge_data.get('concepts', {}).keys())[:3]
                
                gap = KnowledgeGap(
                    id=f"relation_missing_{rel_type}_{int(time.time())}",
                    gap_type=KnowledgeGapType.RELATION_MISSING,
                    description=f"关系类型 '{rel_type}' 在领域 '{domain}' 中缺失或不足",
                    affected_concepts=concepts,
                    severity=GapSeverity.LOW,
                    detection_method=GapDetectionMethod.PATTERN_MATCHING,
                    domain_relevance=0.7 if domain else 0.5,
                    impact_scope=0.5,
                    learning_difficulty=0.4
                )
                gaps.append(gap)
        
        return gaps
    
    def _detect_incomplete_attributes(self, knowledge_data: Dict[str, Any], domain: Optional[str]) -> List[KnowledgeGap]:
        """检测不完整的属性"""
        gaps = []
        
        # 获取领域模板
        domain_template = self.domain_templates.get(domain or 'general', {})
        common_attributes = domain_template.get('common_attributes', [])
        
        # 获取现有概念和属性
        concepts = knowledge_data.get('concepts', {})
        
        for concept_name, concept_data in concepts.items():
            if isinstance(concept_data, dict):
                concept_attributes = set(concept_data.keys())
                
                # 检查是否缺少常见属性
                missing_attributes = []
                for attr in common_attributes:
                    if attr not in concept_attributes:
                        missing_attributes.append(attr)
                
                if missing_attributes:
                    gap = KnowledgeGap(
                        id=f"attribute_incomplete_{concept_name}_{int(time.time())}",
                        gap_type=KnowledgeGapType.ATTRIBUTE_INCOMPLETE,
                        description=f"概念 '{concept_name}' 缺少属性: {', '.join(missing_attributes)}",
                        affected_concepts=[concept_name],
                        severity=GapSeverity.LOW,
                        detection_method=GapDetectionMethod.PATTERN_MATCHING,
                        domain_relevance=0.6 if domain else 0.5,
                        impact_scope=0.3,
                        learning_difficulty=0.3
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _detect_by_reasoning_verification(self, knowledge_data: Dict[str, Any]) -> List[KnowledgeGap]:
        """基于推理验证检测缺口"""
        gaps = []
        
        # 简化实现：检测基本的不一致性
        concepts = knowledge_data.get('concepts', {})
        relations = knowledge_data.get('relations', [])
        
        # 检查自相矛盾的属性
        for concept_name, concept_data in concepts.items():
            if isinstance(concept_data, dict):
                # 检查数值属性的矛盾（如果存在）
                numeric_attributes = {}
                
                for key, value in concept_data.items():
                    if isinstance(value, (int, float)):
                        numeric_attributes[key] = value
                
                # 简单的矛盾检测：如果有多个数值属性，检查它们是否可能矛盾
                # 这里只是示例逻辑
                if len(numeric_attributes) > 1:
                    # 检查是否存在极端值组合可能表示矛盾
                    values = list(numeric_attributes.values())
                    if max(values) > min(values) * 10:  # 简单阈值
                        gap = KnowledgeGap(
                            id=f"inconsistency_{concept_name}_{int(time.time())}",
                            gap_type=KnowledgeGapType.INCONSISTENCY,
                            description=f"概念 '{concept_name}' 的属性值可能存在矛盾",
                            affected_concepts=[concept_name],
                            severity=GapSeverity.MEDIUM,
                            detection_method=GapDetectionMethod.REASONING_VERIFICATION,
                            domain_relevance=0.5,
                            impact_scope=0.4,
                            learning_difficulty=0.6
                        )
                        gaps.append(gap)
        
        return gaps
    
    def _detect_by_comparative_analysis(self, knowledge_data: Dict[str, Any], domain: Optional[str]) -> List[KnowledgeGap]:
        """基于对比分析检测缺口"""
        gaps = []
        
        # 简化实现：对比不同概念的描述检测模糊性
        concepts = knowledge_data.get('concepts', {})
        
        # 收集所有描述
        descriptions = []
        for concept_name, concept_data in concepts.items():
            if isinstance(concept_data, dict) and 'description' in concept_data:
                descriptions.append((concept_name, concept_data['description']))
        
        # 简单的模糊性检测：检查描述是否过于简短或含糊
        for concept_name, description in descriptions:
            if isinstance(description, str):
                # 检查描述长度
                if len(description) < 20:  # 简短描述可能不够清晰
                    gap = KnowledgeGap(
                        id=f"ambiguity_{concept_name}_{int(time.time())}",
                        gap_type=KnowledgeGapType.AMBIGUITY,
                        description=f"概念 '{concept_name}' 的描述可能过于简短或模糊",
                        affected_concepts=[concept_name],
                        severity=GapSeverity.LOW,
                        detection_method=GapDetectionMethod.COMPARATIVE_ANALYSIS,
                        domain_relevance=0.5,
                        impact_scope=0.3,
                        learning_difficulty=0.4
                    )
                    gaps.append(gap)
                
                # 检查是否包含模糊词汇
                vague_terms = ['可能', '也许', '大概', '某些', '一些', '各种']
                vague_count = sum(1 for term in vague_terms if term in description)
                
                if vague_count > 2:  # 包含多个模糊词汇
                    gap = KnowledgeGap(
                        id=f"ambiguity_vague_{concept_name}_{int(time.time())}",
                        gap_type=KnowledgeGapType.AMBIGUITY,
                        description=f"概念 '{concept_name}' 的描述包含多个模糊词汇",
                        affected_concepts=[concept_name],
                        severity=GapSeverity.LOW,
                        detection_method=GapDetectionMethod.COMPARATIVE_ANALYSIS,
                        domain_relevance=0.5,
                        impact_scope=0.3,
                        learning_difficulty=0.4
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _detect_by_dependency_analysis(self, knowledge_data: Dict[str, Any]) -> List[KnowledgeGap]:
        """基于依赖分析检测缺口"""
        gaps = []
        
        # 简化实现：检测孤立的或依赖关系断裂的概念
        concepts = knowledge_data.get('concepts', {})
        relations = knowledge_data.get('relations', [])
        
        # 构建概念间的连接图
        concept_connections = defaultdict(set)
        
        for rel in relations:
            if isinstance(rel, dict):
                source = rel.get('source')
                target = rel.get('target')
                
                if source and target:
                    concept_connections[source].add(target)
                    concept_connections[target].add(source)
        
        # 检测孤立概念（没有连接的概念）
        all_concepts = set(concepts.keys())
        connected_concepts = set(concept_connections.keys())
        isolated_concepts = all_concepts - connected_concepts
        
        for concept in isolated_concepts:
            gap = KnowledgeGap(
                id=f"relation_missing_isolated_{concept}_{int(time.time())}",
                gap_type=KnowledgeGapType.RELATION_MISSING,
                description=f"概念 '{concept}' 是孤立的，缺乏与其他概念的连接",
                affected_concepts=[concept],
                severity=GapSeverity.LOW,
                detection_method=GapDetectionMethod.DEPENDENCY_ANALYSIS,
                domain_relevance=0.5,
                impact_scope=0.4,
                learning_difficulty=0.3
            )
            gaps.append(gap)
        
        return gaps
    
    def _deduplicate_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """去重相似的缺口"""
        if not gaps:
            return []
        
        unique_gaps = []
        seen_descriptions = set()
        
        for gap in gaps:
            # 基于描述的关键词去重
            description_key = self._extract_description_key(gap.description)
            
            if description_key not in seen_descriptions:
                seen_descriptions.add(description_key)
                unique_gaps.append(gap)
        
        return unique_gaps
    
    def _extract_description_key(self, description: str) -> str:
        """提取描述关键词用于去重"""
        # 提取主要名词和动词
        words = description.lower().split()
        # 保留重要词汇（过滤常见词汇）
        stop_words = {'的', '在', '是', '有', '和', '或', '与', '了', '可能', '一些', '各种'}
        key_words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # 取前5个关键词
        return ' '.join(sorted(key_words[:5]))
    
    def _calculate_coverage_score(self, knowledge_data: Dict[str, Any], gaps: List[KnowledgeGap]) -> float:
        """计算检测覆盖度分数"""
        # 简化实现：基于检测到的缺口数量和知识复杂度
        concept_count = len(knowledge_data.get('concepts', {}))
        
        if concept_count == 0:
            return 0.0
        
        # 假设理想情况下每个概念平均有0.2个缺口
        expected_gaps = concept_count * 0.2
        actual_gaps = len(gaps)
        
        # 计算覆盖度：实际检测到的缺口与期望缺口的比例
        coverage = min(1.0, actual_gaps / expected_gaps) if expected_gaps > 0 else 0.0
        
        return coverage
    
    def _calculate_confidence_score(self, gaps: List[KnowledgeGap]) -> float:
        """计算检测置信度分数"""
        if not gaps:
            return 0.0
        
        # 基于检测方法和严重性计算平均置信度
        method_weights = {
            GapDetectionMethod.PATTERN_MATCHING: 0.7,
            GapDetectionMethod.REASONING_VERIFICATION: 0.9,
            GapDetectionMethod.COMPARATIVE_ANALYSIS: 0.6,
            GapDetectionMethod.DEPENDENCY_ANALYSIS: 0.8,
            GapDetectionMethod.STATISTICAL_ANALYSIS: 0.85
        }
        
        severity_weights = {
            GapSeverity.LOW: 0.4,
            GapSeverity.MEDIUM: 0.7,
            GapSeverity.HIGH: 0.9,
            GapSeverity.CRITICAL: 1.0
        }
        
        total_score = 0.0
        
        for gap in gaps:
            method_weight = method_weights.get(gap.detection_method, 0.5)
            severity_weight = severity_weights.get(gap.severity, 0.5)
            
            # 综合权重
            gap_score = method_weight * severity_weight * gap.composite_importance
            total_score += gap_score
        
        # 平均置信度
        avg_confidence = total_score / len(gaps) if len(gaps) > 0 else 0.0
        
        return min(1.0, avg_confidence)
    
    def _update_performance_stats(self, detection_time: float, gaps_detected: int):
        """更新性能统计"""
        self.performance_stats['total_detections'] += 1
        self.performance_stats['gaps_detected'] += gaps_detected
        
        # 更新检测时间统计
        current_avg = self.performance_stats['average_detection_time']
        total_detections = self.performance_stats['total_detections']
        
        new_avg = (current_avg * (total_detections - 1) + detection_time) / total_detections
        self.performance_stats['average_detection_time'] = new_avg
        
        # 累计总检测时间
        self.performance_stats['detection_time_total'] += detection_time
    
    def generate_learning_goals(self, 
                               gap_ids: Optional[List[str]] = None,
                               max_goals: int = 5) -> List[Dict[str, Any]]:
        """
        生成填补知识缺口的学习目标
        
        Args:
            gap_ids: 要生成目标的缺口ID列表（如果为None，使用所有高重要性缺口）
            max_goals: 最大目标数量
            
        Returns:
            学习目标列表
        """
        # 确定要处理的缺口
        if gap_ids is None:
            # 选择高重要性的缺口
            high_importance_gaps = [
                gap for gap in self.detected_gaps.values() 
                if gap.composite_importance >= 0.7
            ]
            selected_gaps = sorted(high_importance_gaps, 
                                 key=lambda g: g.composite_importance, 
                                 reverse=True)[:max_goals]
        else:
            # 使用指定的缺口ID
            selected_gaps = []
            for gap_id in gap_ids:
                if gap_id in self.detected_gaps:
                    selected_gaps.append(self.detected_gaps[gap_id])
        
        # 生成学习目标
        learning_goals = []
        
        for gap in selected_gaps[:max_goals]:
            goal = self._create_learning_goal_from_gap(gap)
            learning_goals.append(goal)
        
        logger.info(f"生成了 {len(learning_goals)} 个学习目标")
        
        return learning_goals
    
    def _create_learning_goal_from_gap(self, gap: KnowledgeGap) -> Dict[str, Any]:
        """从知识缺口创建学习目标"""
        
        # 根据缺口类型确定学习目标类型
        goal_type_mapping = {
            KnowledgeGapType.CONCEPT_MISSING: "概念学习",
            KnowledgeGapType.RELATION_MISSING: "关系建立",
            KnowledgeGapType.ATTRIBUTE_INCOMPLETE: "属性完善",
            KnowledgeGapType.INCONSISTENCY: "矛盾解决",
            KnowledgeGapType.AMBIGUITY: "概念澄清",
            KnowledgeGapType.EVIDENCE_DEFICIENCY: "证据收集"
        }
        
        goal_type = goal_type_mapping.get(gap.gap_type, "知识学习")
        
        # 创建具体的学习目标
        if gap.gap_type == KnowledgeGapType.CONCEPT_MISSING:
            description = f"学习概念: {', '.join(gap.affected_concepts)}"
            steps = [
                f"查找 {gap.affected_concepts[0]} 的定义和解释",
                f"理解 {gap.affected_concepts[0]} 的基本属性和特征",
                f"学习 {gap.affected_concepts[0]} 与其他概念的关系",
                f"应用 {gap.affected_concepts[0]} 解决实际问题"
            ]
        elif gap.gap_type == KnowledgeGapType.RELATION_MISSING:
            description = f"建立概念间的关系: {gap.description}"
            steps = [
                "分析相关概念的特征和属性",
                "识别概念间的潜在联系",
                "验证关系的逻辑合理性",
                "记录新发现的关系"
            ]
        else:
            description = f"解决知识缺口: {gap.description}"
            steps = [
                "分析缺口的具体内容和影响",
                "收集相关信息和资料",
                "验证信息的准确性和可靠性",
                "整合新知识到现有体系"
            ]
        
        goal = {
            'id': f"learning_goal_{gap.id}",
            'type': goal_type,
            'description': description,
            'gap_id': gap.id,
            'gap_type': gap.gap_type.value,
            'affected_concepts': gap.affected_concepts,
            'importance_score': gap.composite_importance,
            'estimated_difficulty': gap.learning_difficulty,
            'estimated_time': gap.learning_difficulty * 60,  # 分钟
            'learning_steps': steps,
            'success_criteria': [
                f"理解 {', '.join(gap.affected_concepts)} 的核心概念",
                "能够解释和应用新学到的知识",
                "填补了识别的知识缺口"
            ],
            'resources': ["教科书", "学术论文", "在线课程", "专家咨询"],
            'created_time': time.time()
        }
        
        return goal
    
    def start_monitoring(self):
        """启动持续监控"""
        if self.enable_continuous_monitoring:
            self.monitoring_active = True
            logger.info("知识缺口持续监控已启动")
        else:
            logger.warning("持续监控未启用")
    
    def stop_monitoring(self):
        """停止持续监控"""
        self.monitoring_active = False
        logger.info("知识缺口持续监控已停止")
    
    def check_and_update(self) -> Optional[GapDetectionResult]:
        """检查并更新缺口检测（用于持续监控）"""
        if not self.monitoring_active:
            return None
        
        current_time = time.time()
        
        # 检查是否达到监控间隔
        if current_time - self.last_monitoring_time < self.monitoring_interval:
            return None
        
        # 执行检测
        logger.info("执行定期知识缺口检测...")
        result = self.detect_gaps()
        
        # 更新监控时间
        self.last_monitoring_time = current_time
        
        return result
    
    def get_gap_statistics(self) -> Dict[str, Any]:
        """获取缺口统计信息"""
        # 按类型统计
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for gap in self.detected_gaps.values():
            type_counts[gap.gap_type.value] += 1
            severity_counts[gap.severity.value] += 1
        
        # 计算平均重要性
        importance_scores = [gap.composite_importance for gap in self.detected_gaps.values()]
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0.0
        
        return {
            'total_gaps': len(self.detected_gaps),
            'gap_type_distribution': dict(type_counts),
            'severity_distribution': dict(severity_counts),
            'average_importance': avg_importance,
            'high_importance_gaps': sum(1 for gap in self.detected_gaps.values() 
                                      if gap.composite_importance >= 0.7),
            'monitoring_active': self.monitoring_active,
            'performance_stats': self.performance_stats
        }
    
    def mark_gap_filled(self, gap_id: str, filled_by: Optional[str] = None):
        """
        标记缺口已填补
        
        Args:
            gap_id: 缺口ID
            filled_by: 填补方式或资源
        """
        if gap_id in self.detected_gaps:
            gap = self.detected_gaps[gap_id]
            
            # 记录填补信息
            gap.metadata['filled'] = True
            gap.metadata['filled_time'] = time.time()
            gap.metadata['filled_by'] = filled_by or 'unknown'
            
            # 移动到历史
            self.gap_history.append(gap)
            
            # 从当前检测中移除
            del self.detected_gaps[gap_id]
            
            # 更新性能统计
            self.performance_stats['gaps_filled'] += 1
            
            logger.info(f"知识缺口已标记为填补: {gap_id}, 填补方式: {filled_by}")
        else:
            logger.warning(f"尝试标记不存在的缺口: {gap_id}")


# 示例和测试函数
def create_example_gap_detector() -> KnowledgeGapDetector:
    """创建示例知识缺口检测器"""
    detector = KnowledgeGapDetector(
        detection_threshold=0.3,
        min_importance_threshold=0.2,
        enable_continuous_monitoring=True
    )
    return detector


def test_gap_detector():
    """测试知识缺口检测器"""
    logger.info("开始测试知识缺口检测器")
    
    # 创建示例检测器
    detector = create_example_gap_detector()
    
    # 创建示例知识数据
    example_knowledge = {
        'concepts': {
            'algorithm': {
                'description': '解决问题的步骤序列',
                'category': 'computer_science'
            },
            'data_structure': {
                'description': '组织和存储数据的方式'
            }
            # 故意缺失 'programming_language' 概念
        },
        'relations': [
            {'source': 'algorithm', 'target': 'data_structure', 'type': 'uses'}
        ],
        'metadata': {
            'domain': 'computer_science'
        }
    }
    
    # 检测缺口
    logger.info("检测知识缺口...")
    result = detector.detect_gaps(
        knowledge_data=example_knowledge,
        domain='computer_science',
        detection_methods=[
            GapDetectionMethod.PATTERN_MATCHING,
            GapDetectionMethod.REASONING_VERIFICATION
        ]
    )
    
    logger.info(f"检测结果: 发现 {result.total_gaps} 个知识缺口")
    
    # 显示重要缺口
    if result.gaps_detected:
        logger.info("重要知识缺口:")
        for i, gap in enumerate(result.get_top_gaps(3)):
            logger.info(f"  {i+1}. {gap.description} (重要性: {gap.composite_importance:.2f})")
    
    # 生成学习目标
    logger.info("生成学习目标...")
    learning_goals = detector.generate_learning_goals(max_goals=2)
    
    for i, goal in enumerate(learning_goals):
        logger.info(f"  目标 {i+1}: {goal['description']}")
    
    # 获取统计信息
    stats = detector.get_gap_statistics()
    logger.info(f"缺口统计: {stats['total_gaps']} 个活跃缺口")
    
    logger.info("知识缺口检测器测试完成")
    return detector


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_gap_detector_instance = test_gap_detector()