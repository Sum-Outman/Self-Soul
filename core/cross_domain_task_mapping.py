"""
跨领域任务映射规则 - Cross-Domain Task Mapping Rules

根据报告建议实现跨领域任务映射规则，支持调度中枢进行跨领域能力协同
示例：工程问题→知识模型 + 优化模型 + 编程模型协同
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .capability_interface import CapabilityType

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """领域类型枚举"""
    ENGINEERING = "engineering"          # 工程领域
    MEDICAL = "medical"                  # 医疗领域
    FINANCIAL = "financial"              # 金融领域
    EDUCATIONAL = "educational"          # 教育领域
    CREATIVE = "creative"                # 创意领域
    SCIENTIFIC = "scientific"            # 科学领域
    TECHNICAL = "technical"              # 技术领域
    BUSINESS = "business"                # 商业领域
    LEGAL = "legal"                      # 法律领域
    AGRICULTURAL = "agricultural"        # 农业领域


@dataclass
class DomainProfile:
    """领域配置"""
    domain_type: DomainType
    keywords: List[str]                  # 关键词
    required_capabilities: List[CapabilityType]  # 必需能力
    recommended_capabilities: List[CapabilityType]  # 推荐能力
    typical_tasks: List[str]             # 典型任务
    complexity_factor: float = 1.0       # 复杂度因子


@dataclass
class CrossDomainMapping:
    """跨领域映射"""
    source_domain: DomainType
    target_domain: DomainType
    mapping_rules: List[Dict[str, Any]]  # 映射规则
    capability_translation: Dict[CapabilityType, List[CapabilityType]]  # 能力转换
    success_rate: float = 0.0            # 成功率
    usage_count: int = 0                 # 使用次数


class CrossDomainTaskMapper:
    """
    跨领域任务映射器
    实现跨领域任务映射规则，支持调度中枢进行跨领域能力协同
    """
    
    def __init__(self):
        self.domain_profiles = self._initialize_domain_profiles()
        self.cross_domain_mappings = self._initialize_cross_domain_mappings()
        self.task_patterns = self._initialize_task_patterns()
        
        logger.info("跨领域任务映射器初始化完成")
    
    def _initialize_domain_profiles(self) -> Dict[DomainType, DomainProfile]:
        """初始化领域配置"""
        profiles = {}
        
        # 工程领域
        profiles[DomainType.ENGINEERING] = DomainProfile(
            domain_type=DomainType.ENGINEERING,
            keywords=["工程", "机械", "电气", "结构", "设计", "制造", "施工", "材料"],
            required_capabilities=[
                CapabilityType.LOGICAL_REASONING,
                CapabilityType.DATA_ANALYSIS
            ],
            recommended_capabilities=[
                CapabilityType.KNOWLEDGE_REASONING,
                CapabilityType.PROGRAMMING_CODE,
                CapabilityType.SYSTEM_OPTIMIZATION
            ],
            typical_tasks=[
                "机械设计分析",
                "电气系统优化", 
                "结构强度计算",
                "制造流程规划"
            ],
            complexity_factor=1.2
        )
        
        # 医疗领域
        profiles[DomainType.MEDICAL] = DomainProfile(
            domain_type=DomainType.MEDICAL,
            keywords=["医疗", "健康", "诊断", "治疗", "药物", "疾病", "患者", "手术"],
            required_capabilities=[
                CapabilityType.KNOWLEDGE_REASONING,
                CapabilityType.DATA_ANALYSIS
            ],
            recommended_capabilities=[
                CapabilityType.LOGICAL_REASONING,
                CapabilityType.PREDICTIVE_MODELING,
                CapabilityType.DIAGNOSTIC_SUPPORT
            ],
            typical_tasks=[
                "疾病诊断支持",
                "治疗方案建议",
                "药物相互作用分析",
                "医疗数据分析"
            ],
            complexity_factor=1.5
        )
        
        # 金融领域
        profiles[DomainType.FINANCIAL] = DomainProfile(
            domain_type=DomainType.FINANCIAL,
            keywords=["金融", "财务", "投资", "经济", "市场", "股票", "风险", "交易"],
            required_capabilities=[
                CapabilityType.DATA_ANALYSIS,
                CapabilityType.PREDICTIVE_MODELING
            ],
            recommended_capabilities=[
                CapabilityType.LOGICAL_REASONING,
                CapabilityType.DECISION_MAKING,
                CapabilityType.PREDICTIVE_MODELING
            ],
            typical_tasks=[
                "投资组合分析",
                "市场趋势预测",
                "风险评估",
                "财务报告生成"
            ],
            complexity_factor=1.3
        )
        
        # 教育领域
        profiles[DomainType.EDUCATIONAL] = DomainProfile(
            domain_type=DomainType.EDUCATIONAL,
            keywords=["教育", "学习", "教学", "课程", "培训", "学生", "教师", "考试"],
            required_capabilities=[
                CapabilityType.LANGUAGE_PROCESSING,
                CapabilityType.KNOWLEDGE_REASONING
            ],
            recommended_capabilities=[
                CapabilityType.CREATIVE_GENERATION,
                CapabilityType.PLANNING_SCHEDULING,
                CapabilityType.SELF_LEARNING
            ],
            typical_tasks=[
                "课程内容生成",
                "学习计划制定",
                "知识问答",
                "学习效果评估"
            ],
            complexity_factor=1.0
        )
        
        # 创意领域
        profiles[DomainType.CREATIVE] = DomainProfile(
            domain_type=DomainType.CREATIVE,
            keywords=["创意", "创作", "艺术", "设计", "写作", "音乐", "绘画", "创新"],
            required_capabilities=[
                CapabilityType.CREATIVE_GENERATION,
                CapabilityType.LANGUAGE_PROCESSING
            ],
            recommended_capabilities=[
                CapabilityType.EMOTION_ANALYSIS,
                CapabilityType.VISION_ANALYSIS,
                CapabilityType.AUDIO_PROCESSING
            ],
            typical_tasks=[
                "创意文案生成",
                "艺术设计创作",
                "音乐作曲",
                "故事编写"
            ],
            complexity_factor=1.1
        )
        
        return profiles
    
    def _initialize_cross_domain_mappings(self) -> Dict[Tuple[DomainType, DomainType], CrossDomainMapping]:
        """初始化跨领域映射"""
        mappings = {}
        
        # 工程→医疗映射（如医疗设备设计）
        mappings[(DomainType.ENGINEERING, DomainType.MEDICAL)] = CrossDomainMapping(
            source_domain=DomainType.ENGINEERING,
            target_domain=DomainType.MEDICAL,
            mapping_rules=[
                {
                    "pattern": r"设计.*医疗设备",
                    "translation": "将工程设计要求转换为医疗设备规格"
                },
                {
                    "pattern": r"优化.*医疗流程",
                    "translation": "应用工程优化方法改进医疗流程"
                }
            ],
            capability_translation={
                CapabilityType.MECHANICAL_ENGINEERING: [
                    CapabilityType.MEDICAL_ANALYSIS,
                    CapabilityType.DIAGNOSTIC_SUPPORT
                ],
                CapabilityType.SYSTEM_OPTIMIZATION: [
                    CapabilityType.DATA_ANALYSIS,
                    CapabilityType.PREDICTIVE_MODELING
                ]
            },
            success_rate=0.85,
            usage_count=0
        )
        
        # 医疗→工程映射（如工程中的健康安全）
        mappings[(DomainType.MEDICAL, DomainType.ENGINEERING)] = CrossDomainMapping(
            source_domain=DomainType.MEDICAL,
            target_domain=DomainType.ENGINEERING,
            mapping_rules=[
                {
                    "pattern": r"健康.*安全.*工程",
                    "translation": "将医疗健康要求融入工程设计"
                },
                {
                    "pattern": r"人体工程学",
                    "translation": "应用人体医学知识优化工程设计"
                }
            ],
            capability_translation={
                CapabilityType.MEDICAL_ANALYSIS: [
                    CapabilityType.MECHANICAL_ENGINEERING,
                    CapabilityType.SYSTEM_OPTIMIZATION
                ],
                CapabilityType.DIAGNOSTIC_SUPPORT: [
                    CapabilityType.DATA_ANALYSIS,
                    CapabilityType.LOGICAL_REASONING
                ]
            },
            success_rate=0.80,
            usage_count=0
        )
        
        # 金融→工程映射（如工程项目投资分析）
        mappings[(DomainType.FINANCIAL, DomainType.ENGINEERING)] = CrossDomainMapping(
            source_domain=DomainType.FINANCIAL,
            target_domain=DomainType.ENGINEERING,
            mapping_rules=[
                {
                    "pattern": r"工程.*投资.*分析",
                    "translation": "应用金融分析方法评估工程项目"
                },
                {
                    "pattern": r"成本.*效益.*工程",
                    "translation": "将金融成本效益分析融入工程决策"
                }
            ],
            capability_translation={
                CapabilityType.DATA_ANALYSIS: [
                    CapabilityType.MECHANICAL_ENGINEERING,
                    CapabilityType.SYSTEM_OPTIMIZATION
                ],
                CapabilityType.PREDICTIVE_MODELING: [
                    CapabilityType.LOGICAL_REASONING,
                    CapabilityType.DECISION_MAKING
                ]
            },
            success_rate=0.88,
            usage_count=0
        )
        
        # 教育→技术映射（如技术培训）
        mappings[(DomainType.EDUCATIONAL, DomainType.TECHNICAL)] = CrossDomainMapping(
            source_domain=DomainType.EDUCATIONAL,
            target_domain=DomainType.TECHNICAL,
            mapping_rules=[
                {
                    "pattern": r"技术.*培训",
                    "translation": "将教育方法应用于技术培训"
                },
                {
                    "pattern": r"编程.*教学",
                    "translation": "应用教育心理学优化编程教学"
                }
            ],
            capability_translation={
                CapabilityType.LANGUAGE_PROCESSING: [
                    CapabilityType.PROGRAMMING_CODE,
                    CapabilityType.LOGICAL_REASONING
                ],
                CapabilityType.KNOWLEDGE_REASONING: [
                    CapabilityType.DATA_ANALYSIS,
                    CapabilityType.CREATIVE_GENERATION
                ]
            },
            success_rate=0.90,
            usage_count=0
        )
        
        return mappings
    
    def _initialize_task_patterns(self) -> Dict[str, List[DomainType]]:
        """初始化任务模式"""
        patterns = {}
        
        # 工程设计类任务
        patterns[r"设计.*方案"] = [DomainType.ENGINEERING, DomainType.CREATIVE]
        patterns[r"优化.*系统"] = [DomainType.ENGINEERING, DomainType.TECHNICAL]
        patterns[r"分析.*数据"] = [DomainType.SCIENTIFIC, DomainType.FINANCIAL]
        patterns[r"生成.*报告"] = [DomainType.BUSINESS, DomainType.EDUCATIONAL]
        patterns[r"评估.*风险"] = [DomainType.FINANCIAL, DomainType.LEGAL]
        patterns[r"制定.*计划"] = [DomainType.BUSINESS, DomainType.EDUCATIONAL]
        patterns[r"解决.*问题"] = [DomainType.TECHNICAL, DomainType.ENGINEERING]
        patterns[r"创新.*设计"] = [DomainType.CREATIVE, DomainType.ENGINEERING]
        
        return patterns
    
    def identify_domains(self, task_description: str) -> List[DomainType]:
        """
        识别任务涉及的领域
        
        Args:
            task_description: 任务描述
            
        Returns:
            涉及的领域类型列表
        """
        identified_domains = []
        
        # 1. 通过关键词匹配
        for domain_type, profile in self.domain_profiles.items():
            for keyword in profile.keywords:
                if keyword in task_description:
                    if domain_type not in identified_domains:
                        identified_domains.append(domain_type)
                    break
        
        # 2. 通过任务模式匹配
        for pattern, domains in self.task_patterns.items():
            if re.search(pattern, task_description, re.IGNORECASE):
                for domain in domains:
                    if domain not in identified_domains:
                        identified_domains.append(domain)
        
        # 3. 如果未识别到任何领域，使用默认领域
        if not identified_domains:
            identified_domains.append(DomainType.TECHNICAL)
        
        logger.info(f"任务领域识别: {task_description[:50]}... -> {[d.value for d in identified_domains]}")
        return identified_domains
    
    def map_cross_domain_capabilities(self, source_domain: DomainType,
                                     target_domain: DomainType,
                                     source_capabilities: List[CapabilityType]) -> List[CapabilityType]:
        """
        跨领域映射能力
        
        Args:
            source_domain: 源领域
            target_domain: 目标领域
            source_capabilities: 源能力列表
            
        Returns:
            映射后的目标能力列表
        """
        mapping_key = (source_domain, target_domain)
        
        if mapping_key not in self.cross_domain_mappings:
            # 如果没有直接映射，尝试通用映射
            return self._generic_capability_mapping(source_capabilities)
        
        mapping = self.cross_domain_mappings[mapping_key]
        mapped_capabilities = []
        
        # 应用能力转换
        for source_capability in source_capabilities:
            if source_capability in mapping.capability_translation:
                target_capabilities = mapping.capability_translation[source_capability]
                mapped_capabilities.extend(target_capabilities)
            else:
                # 保持原能力
                mapped_capabilities.append(source_capability)
        
        # 去重
        unique_capabilities = list(set(mapped_capabilities))
        
        # 更新使用统计
        mapping.usage_count += 1
        
        logger.info(f"跨领域能力映射: {source_domain.value}->{target_domain.value}, "
                   f"源能力: {[c.value for c in source_capabilities]}, "
                   f"目标能力: {[c.value for c in unique_capabilities]}")
        
        return unique_capabilities
    
    def _generic_capability_mapping(self, capabilities: List[CapabilityType]) -> List[CapabilityType]:
        """通用能力映射"""
        # 基础能力映射表
        generic_mapping = {
            CapabilityType.LANGUAGE_PROCESSING: [
                CapabilityType.CREATIVE_GENERATION,
                CapabilityType.KNOWLEDGE_REASONING
            ],
            CapabilityType.DATA_ANALYSIS: [
                CapabilityType.PREDICTIVE_MODELING,
                CapabilityType.LOGICAL_REASONING
            ],
            CapabilityType.LOGICAL_REASONING: [
                CapabilityType.DECISION_MAKING,
                CapabilityType.PLANNING_SCHEDULING
            ],
            CapabilityType.KNOWLEDGE_REASONING: [
                CapabilityType.LANGUAGE_PROCESSING,
                CapabilityType.DATA_ANALYSIS
            ]
        }
        
        mapped_capabilities = []
        for capability in capabilities:
            if capability in generic_mapping:
                mapped_capabilities.extend(generic_mapping[capability])
            else:
                mapped_capabilities.append(capability)
        
        return list(set(mapped_capabilities))
    
    def generate_cross_domain_plan(self, task_description: str,
                                 identified_domains: List[DomainType],
                                 base_capabilities: List[CapabilityType]) -> Dict[str, Any]:
        """
        生成跨领域执行计划
        
        Args:
            task_description: 任务描述
            identified_domains: 识别到的领域
            base_capabilities: 基础能力需求
            
        Returns:
            跨领域执行计划
        """
        if len(identified_domains) < 2:
            # 单一领域任务
            return {
                "cross_domain": False,
                "domains": [d.value for d in identified_domains],
                "capabilities": base_capabilities,
                "mapping_applied": False
            }
        
        # 多领域任务，应用跨领域映射
        primary_domain = identified_domains[0]
        secondary_domains = identified_domains[1:]
        
        enhanced_capabilities = base_capabilities.copy()
        mapping_applications = []
        
        # 应用跨领域映射
        for secondary_domain in secondary_domains:
            # 从主领域到次领域的映射
            mapped_capabilities = self.map_cross_domain_capabilities(
                primary_domain, secondary_domain, base_capabilities
            )
            
            # 从次领域到主领域的映射
            reverse_mapped = self.map_cross_domain_capabilities(
                secondary_domain, primary_domain, mapped_capabilities
            )
            
            enhanced_capabilities.extend(reverse_mapped)
            mapping_applications.append({
                "from_domain": primary_domain.value,
                "to_domain": secondary_domain.value,
                "mapped_capabilities": [c.value for c in mapped_capabilities]
            })
        
        # 去重并排序
        unique_capabilities = list(set(enhanced_capabilities))
        
        # 计算跨领域复杂度因子
        complexity_factor = 1.0
        for domain in identified_domains:
            if domain in self.domain_profiles:
                complexity_factor *= self.domain_profiles[domain].complexity_factor
        
        plan = {
            "cross_domain": True,
            "domains": [d.value for d in identified_domains],
            "primary_domain": primary_domain.value,
            "secondary_domains": [d.value for d in secondary_domains],
            "base_capabilities": [c.value for c in base_capabilities],
            "enhanced_capabilities": [c.value for c in unique_capabilities],
            "mapping_applications": mapping_applications,
            "complexity_factor": complexity_factor,
            "estimated_enhancement": len(unique_capabilities) / max(1, len(base_capabilities))
        }
        
        logger.info(f"跨领域计划生成: {task_description[:50]}..., "
                   f"领域: {plan['domains']}, "
                   f"能力增强: {plan['estimated_enhancement']:.2f}倍")
        
        return plan
    
    def get_domain_specific_capabilities(self, domain: DomainType) -> List[CapabilityType]:
        """
        获取领域特定能力
        
        Args:
            domain: 领域类型
            
        Returns:
            领域特定能力列表
        """
        if domain not in self.domain_profiles:
            return []
        
        profile = self.domain_profiles[domain]
        return profile.required_capabilities + profile.recommended_capabilities
    
    def analyze_task_complexity(self, task_description: str,
                              domains: List[DomainType]) -> float:
        """
        分析任务复杂度
        
        Args:
            task_description: 任务描述
            domains: 涉及领域
            
        Returns:
            复杂度评分（0-1）
        """
        if not domains:
            return 0.5
        
        # 基础复杂度
        # 对于中文，按字符计算；对于英文，按单词计算
        if any('\u4e00' <= char <= '\u9fff' for char in task_description):
            # 包含中文字符，按字符计算
            char_count = len(task_description)
            base_complexity = min(1.0, char_count / 200.0)  # 200字符为基准
        else:
            # 英文或其他语言，按单词计算
            word_count = len(task_description.split())
            base_complexity = min(1.0, word_count / 100.0)
        
        # 领域复杂度因子
        domain_factor = 1.0
        for domain in domains:
            if domain in self.domain_profiles:
                domain_factor *= self.domain_profiles[domain].complexity_factor
        
        # 跨领域复杂度加成
        cross_domain_penalty = 1.0
        if len(domains) > 1:
            cross_domain_penalty = 1.0 + (len(domains) - 1) * 0.2
        
        final_complexity = base_complexity * domain_factor * cross_domain_penalty
        
        return min(1.0, final_complexity)
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息"""
        total_mappings = len(self.cross_domain_mappings)
        total_usage = sum(mapping.usage_count for mapping in self.cross_domain_mappings.values())
        
        # 最常用的映射
        most_used = None
        max_usage = 0
        for mapping_key, mapping in self.cross_domain_mappings.items():
            if mapping.usage_count > max_usage:
                max_usage = mapping.usage_count
                most_used = mapping_key
        
        return {
            "total_domain_profiles": len(self.domain_profiles),
            "total_cross_domain_mappings": total_mappings,
            "total_mapping_usage": total_usage,
            "most_used_mapping": str(most_used) if most_used else None,
            "most_used_count": max_usage,
            "average_success_rate": sum(m.success_rate for m in self.cross_domain_mappings.values()) / max(1, total_mappings)
        }


def integrate_with_scheduling_layer(scheduling_layer) -> CrossDomainTaskMapper:
    """
    将跨领域映射器集成到调度层
    
    Args:
        scheduling_layer: 调度层实例
        
    Returns:
        跨领域任务映射器
    """
    mapper = CrossDomainTaskMapper()
    
    # 这里可以添加与调度层的集成逻辑
    # 例如：修改调度层的任务分析方法以使用跨领域映射
    
    logger.info("跨领域任务映射器已集成到调度层")
    return mapper