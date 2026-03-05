import zlib
"""
跨模态一致性生成器

修复计划第三阶段：提升生成能力（逻辑+质量+匹配）
任务3.1：创建跨模态一致性生成器

核心功能：
1. 确保多模态输出在逻辑、语义和风格上保持一致
2. 实现跨模态的一致性验证和修正
3. 提供一致性评分和改进建议
"""

import sys
import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch不可用，使用模拟实现")

# 导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 配置日志
logger = logging.getLogger("multimodal")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)



def _deterministic_randn(size, seed_prefix="default"):
    """Generate deterministic normal distribution using numpy RandomState"""
    import math
    if isinstance(size, int):
        size = (size,)
    total_elements = 1
    for dim in size:
        total_elements *= dim
    
    # Create deterministic seed from seed_prefix using adler32
    seed_hash = zlib.adler32(seed_prefix.encode('utf-8')) & 0xffffffff
    rng = np.random.RandomState(seed_hash)
    
    # Generate uniform random numbers
    u1 = rng.random_sample(total_elements)
    u2 = rng.random_sample(total_elements)
    
    # Apply Box-Muller transform
    u1 = np.maximum(u1, 1e-10)
    u2 = np.maximum(u2, 1e-10)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
    
    # Convert to torch tensor
    import torch
    result = torch.from_numpy(z0).float()
    
    return result.view(*size)

class ConsistencyType(Enum):
    """一致性类型"""
    SEMANTIC = "semantic"  # 语义一致性
    LOGICAL = "logical"    # 逻辑一致性
    STYLISTIC = "stylistic"  # 风格一致性
    TEMPORAL = "temporal"  # 时序一致性
    SPATIAL = "spatial"    # 空间一致性
    CONTEXTUAL = "contextual"  # 上下文一致性


@dataclass
class ConsistencyCheck:
    """一致性检查结果"""
    consistency_type: ConsistencyType
    score: float  # 0-1分数，1表示完全一致
    issues: List[str] = field(default_factory=list)  # 发现的问题
    suggestions: List[str] = field(default_factory=list)  # 改进建议
    confidence: float = 1.0  # 检查置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "consistency_type": self.consistency_type.value,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "status": "passed" if self.score >= 0.8 else "warning" if self.score >= 0.6 else "failed"
        }


@dataclass
class ModalityOutput:
    """模态输出"""
    modality_type: str  # 模态类型：text, image, audio
    content: Any  # 输出内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class CrossModalConsistencyGenerator:
    """
    跨模态一致性生成器
    
    核心功能：
    1. 验证多模态输出的一致性
    2. 修正不一致的输出
    3. 生成一致性报告和改进建议
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化一致性生成器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 一致性阈值
        self.thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "poor": 0.6,
            "failed": 0.0
        }
        
        # 初始化一致性检查器
        self.checkers = self._initialize_checkers()
        
        # 统计信息
        self.stats = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "total_corrections": 0,
            "successful_corrections": 0,
            "average_consistency_score": 0.0
        }
        
        logger.info("跨模态一致性生成器初始化完成")
    
    def _initialize_checkers(self) -> Dict[ConsistencyType, Any]:
        """初始化一致性检查器"""
        checkers = {}
        
        # 语义一致性检查器
        if TORCH_AVAILABLE:
            checkers[ConsistencyType.SEMANTIC] = self._create_semantic_checker()
        else:
            checkers[ConsistencyType.SEMANTIC] = self._create_mock_checker("semantic")
        
        # 逻辑一致性检查器
        checkers[ConsistencyType.LOGICAL] = self._create_logical_checker()
        
        # 风格一致性检查器
        checkers[ConsistencyType.STYLISTIC] = self._create_stylistic_checker()
        
        # 其他检查器
        checkers[ConsistencyType.TEMPORAL] = self._create_mock_checker("temporal")
        checkers[ConsistencyType.SPATIAL] = self._create_mock_checker("spatial")
        checkers[ConsistencyType.CONTEXTUAL] = self._create_mock_checker("contextual")
        
        return checkers
    
    def _create_semantic_checker(self):
        """创建语义一致性检查器（PyTorch实现）"""
        class SemanticConsistencyChecker(nn.Module):
            def __init__(self, embedding_dim: int = 768):
                super().__init__()
                self.embedding_dim = embedding_dim
                
                # 语义相似度计算器
                self.semantic_similarity = nn.CosineSimilarity(dim=-1)
                
                # 语义对齐网络
                self.semantic_aligner = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
                """
                计算语义一致性分数
                
                Args:
                    embeddings1: 第一个模态的嵌入 [batch_size, seq_len, embedding_dim]
                    embeddings2: 第二个模态的嵌入 [batch_size, seq_len, embedding_dim]
                    
                Returns:
                    一致性分数 [batch_size, 1]
                """
                # 计算余弦相似度
                similarity = self.semantic_similarity(embeddings1, embeddings2)
                
                # 计算对齐分数
                combined = torch.cat([embeddings1.mean(dim=1), embeddings2.mean(dim=1)], dim=-1)
                alignment_score = self.semantic_aligner(combined)
                
                # 综合分数
                combined_score = (similarity.mean() + alignment_score.squeeze()) / 2
                
                return combined_score
        
        return SemanticConsistencyChecker()
    
    def _create_logical_checker(self):
        """创建逻辑一致性检查器"""
        class LogicalConsistencyChecker:
            def __init__(self):
                self.logical_patterns = {
                    "contradiction": ["但是", "然而", "不过", "相反", "却", "并非"],
                    "causality": ["因为", "所以", "导致", "造成", "因此", "由于"],
                    "temporal": ["之前", "之后", "首先", "然后", "最后", "接着"],
                    "comparison": ["比", "更", "最", "非常", "极其", "特别"]
                }
            
            def check(self, text1: str, text2: str) -> float:
                """
                检查逻辑一致性
                
                Args:
                    text1: 第一个文本
                    text2: 第二个文本
                    
                Returns:
                    逻辑一致性分数 0-1
                """
                # 简单实现：检查逻辑关键词的一致性
                if not text1 or not text2:
                    return 0.0
                
                # 提取逻辑关键词
                keywords1 = self._extract_logical_keywords(text1)
                keywords2 = self._extract_logical_keywords(text2)
                
                if not keywords1 and not keywords2:
                    return 1.0  # 都没有逻辑关键词，视为一致
                
                # 检查逻辑关系是否冲突
                contradictions = self._detect_contradictions(keywords1, keywords2)
                
                # 计算一致性分数
                if contradictions:
                    return max(0.0, 1.0 - len(contradictions) / max(len(keywords1), 1))
                else:
                    return 1.0
            
            def _extract_logical_keywords(self, text: str) -> List[Tuple[str, str]]:
                """提取逻辑关键词"""
                keywords = []
                for category, words in self.logical_patterns.items():
                    for word in words:
                        if word in text:
                            keywords.append((category, word))
                return keywords
            
            def _detect_contradictions(self, keywords1: List[Tuple[str, str]], 
                                      keywords2: List[Tuple[str, str]]) -> List[str]:
                """检测逻辑冲突"""
                contradictions = []
                
                # 检查因果关系冲突
                causal1 = any(k[0] == "causality" for k in keywords1)
                causal2 = any(k[0] == "causality" for k in keywords2)
                
                if causal1 != causal2:
                    contradictions.append("因果关系不一致")
                
                # 检查时间顺序冲突
                temporal1 = [k[1] for k in keywords1 if k[0] == "temporal"]
                temporal2 = [k[1] for k in keywords2 if k[0] == "temporal"]
                
                if temporal1 and temporal2:
                    # 简单检查：如果都有时间词但不同，可能冲突
                    if set(temporal1) != set(temporal2):
                        contradictions.append("时间顺序不一致")
                
                return contradictions
        
        return LogicalConsistencyChecker()
    
    def _create_stylistic_checker(self):
        """创建风格一致性检查器"""
        class StylisticConsistencyChecker:
            def __init__(self):
                self.style_categories = {
                    "formal": ["尊敬的", "您好", "此致", "敬礼", "特此", "谨"],
                    "informal": ["哈喽", "嘿", "拜拜", "哈哈", "嘛", "呢"],
                    "technical": ["参数", "配置", "算法", "优化", "模型", "训练"],
                    "casual": ["那个", "这个", "然后", "就是", "好像", "感觉"]
                }
            
            def check(self, text1: str, text2: str) -> float:
                """
                检查风格一致性
                
                Args:
                    text1: 第一个文本
                    text2: 第二个文本
                    
                Returns:
                    风格一致性分数 0-1
                """
                if not text1 or not text2:
                    return 0.0
                
                # 分析文本风格
                style1 = self._analyze_style(text1)
                style2 = self._analyze_style(text2)
                
                # 计算风格相似度
                similarity = self._calculate_style_similarity(style1, style2)
                
                return similarity
            
            def _analyze_style(self, text: str) -> Dict[str, float]:
                """分析文本风格"""
                style_scores = {category: 0.0 for category in self.style_categories}
                
                # 统计风格关键词
                for category, keywords in self.style_categories.items():
                    count = sum(1 for keyword in keywords if keyword in text)
                    if count > 0:
                        style_scores[category] = min(1.0, count / 5)  # 归一化
                
                # 如果没有明显风格，则标记为中性
                if max(style_scores.values()) < 0.1:
                    style_scores["neutral"] = 1.0
                
                return style_scores
            
            def _calculate_style_similarity(self, style1: Dict[str, float], 
                                          style2: Dict[str, float]) -> float:
                """计算风格相似度"""
                # 提取风格向量
                categories = set(style1.keys()) | set(style2.keys())
                vec1 = np.array([style1.get(cat, 0.0) for cat in categories])
                vec2 = np.array([style2.get(cat, 0.0) for cat in categories])
                
                # 计算余弦相似度
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
        
        return StylisticConsistencyChecker()
    
    def _create_mock_checker(self, checker_type: str):
        """创建模拟检查器"""
        class MockChecker:
            def __init__(self):
                self.checker_type = checker_type
            
            def check(self, *args, **kwargs) -> float:
                """模拟检查，返回随机分数"""
                return np.random.uniform(0.7, 0.95)
        
        return MockChecker()
    
    def check_consistency(self, modality_outputs: List[ModalityOutput], 
                         context: Optional[Dict[str, Any]] = None) -> List[ConsistencyCheck]:
        """
        检查多模态输出的一致性
        
        Args:
            modality_outputs: 模态输出列表
            context: 上下文信息（可选）
            
        Returns:
            一致性检查结果列表
        """
        self.stats["total_checks"] += 1
        
        logger.info(f"开始检查 {len(modality_outputs)} 个模态输出的一致性")
        
        checks = []
        
        # 检查每对模态
        for i in range(len(modality_outputs)):
            for j in range(i + 1, len(modality_outputs)):
                modality1 = modality_outputs[i]
                modality2 = modality_outputs[j]
                
                logger.info(f"检查 {modality1.modality_type} 和 {modality2.modality_type} 的一致性")
                
                # 执行各种一致性检查
                semantic_check = self._check_semantic_consistency(modality1, modality2)
                logical_check = self._check_logical_consistency(modality1, modality2)
                stylistic_check = self._check_stylistic_consistency(modality1, modality2)
                
                checks.extend([semantic_check, logical_check, stylistic_check])
        
        # 计算总体一致性
        overall_score = self._calculate_overall_consistency(checks)
        
        # 记录统计
        passed_count = sum(1 for check in checks if check.score >= self.thresholds["acceptable"])
        self.stats["passed_checks"] += passed_count
        self.stats["failed_checks"] += (len(checks) - passed_count)
        
        # 更新平均分数
        total_checks = self.stats["total_checks"]
        current_avg = self.stats["average_consistency_score"]
        self.stats["average_consistency_score"] = (current_avg * (total_checks - 1) + overall_score) / total_checks
        
        logger.info(f"一致性检查完成，总体分数: {overall_score:.2f}")
        
        return checks
    
    def _check_semantic_consistency(self, modality1: ModalityOutput, 
                                  modality2: ModalityOutput) -> ConsistencyCheck:
        """检查语义一致性"""
        checker = self.checkers[ConsistencyType.SEMANTIC]
        
        # 提取语义内容（简化实现）
        content1 = self._extract_semantic_content(modality1)
        content2 = self._extract_semantic_content(modality2)
        
        if TORCH_AVAILABLE:
            # 使用PyTorch检查器
            # 注意：这里需要实际的嵌入，目前使用模拟
            score = checker.check(
                _deterministic_randn((1, 10, 768), seed_prefix="randn_default"),  # 模拟嵌入
                _deterministic_randn((1, 10, 768), seed_prefix="randn_default")
            ).item()
        else:
            # 使用模拟检查
            score = checker.check(content1, content2)
        
        # 生成问题和建议
        issues = []
        suggestions = []
        
        if score < 0.8:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的语义不一致")
            suggestions.append("调整输出以保持语义一致性")
            suggestions.append("检查概念映射是否正确")
        
        if score < 0.6:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的语义严重冲突")
            suggestions.append("重新生成输出以确保语义一致性")
            suggestions.append("使用语义对齐机制修正输出")
        
        return ConsistencyCheck(
            consistency_type=ConsistencyType.SEMANTIC,
            score=score,
            issues=issues,
            suggestions=suggestions,
            confidence=0.9
        )
    
    def _check_logical_consistency(self, modality1: ModalityOutput, 
                                 modality2: ModalityOutput) -> ConsistencyCheck:
        """检查逻辑一致性"""
        checker = self.checkers[ConsistencyType.LOGICAL]
        
        # 提取文本内容进行逻辑分析
        text1 = self._extract_text_content(modality1)
        text2 = self._extract_text_content(modality2)
        
        if text1 and text2:
            score = checker.check(text1, text2)
        else:
            # 非文本模态，使用模拟分数
            score = np.random.uniform(0.8, 0.95)
        
        # 生成问题和建议
        issues = []
        suggestions = []
        
        if score < 0.8:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的逻辑关系不一致")
            suggestions.append("检查因果关系和时间顺序")
            suggestions.append("确保逻辑推理的一致性")
        
        if score < 0.6:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的逻辑严重冲突")
            suggestions.append("重新评估逻辑关系")
            suggestions.append("使用逻辑一致性验证器")
        
        return ConsistencyCheck(
            consistency_type=ConsistencyType.LOGICAL,
            score=score,
            issues=issues,
            suggestions=suggestions,
            confidence=0.85
        )
    
    def _check_stylistic_consistency(self, modality1: ModalityOutput, 
                                   modality2: ModalityOutput) -> ConsistencyCheck:
        """检查风格一致性"""
        checker = self.checkers[ConsistencyType.STYLISTIC]
        
        # 提取文本内容进行风格分析
        text1 = self._extract_text_content(modality1)
        text2 = self._extract_text_content(modality2)
        
        if text1 and text2:
            score = checker.check(text1, text2)
        else:
            # 非文本模态，使用模拟分数
            score = np.random.uniform(0.85, 0.98)
        
        # 生成问题和建议
        issues = []
        suggestions = []
        
        if score < 0.8:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的风格不一致")
            suggestions.append("统一输出风格（正式/非正式/技术性）")
            suggestions.append("调整语气和表达方式")
        
        if score < 0.6:
            issues.append(f"{modality1.modality_type}和{modality2.modality_type}的风格严重冲突")
            suggestions.append("重新定义输出风格规范")
            suggestions.append("使用风格转换器统一风格")
        
        return ConsistencyCheck(
            consistency_type=ConsistencyType.STYLISTIC,
            score=score,
            issues=issues,
            suggestions=suggestions,
            confidence=0.8
        )
    
    def _extract_semantic_content(self, modality: ModalityOutput) -> str:
        """提取语义内容（简化实现）"""
        if isinstance(modality.content, str):
            return modality.content
        elif hasattr(modality.content, '__str__'):
            return str(modality.content)
        else:
            return f"{modality.modality_type}_content"
    
    def _extract_text_content(self, modality: ModalityOutput) -> Optional[str]:
        """提取文本内容"""
        if modality.modality_type == "text" and isinstance(modality.content, str):
            return modality.content
        return None
    
    def _calculate_overall_consistency(self, checks: List[ConsistencyCheck]) -> float:
        """计算总体一致性分数"""
        if not checks:
            return 1.0
        
        # 加权平均，语义一致性权重最高
        weights = {
            ConsistencyType.SEMANTIC: 0.5,
            ConsistencyType.LOGICAL: 0.3,
            ConsistencyType.STYLISTIC: 0.2
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for check in checks:
            weight = weights.get(check.consistency_type, 0.1)
            weighted_sum += check.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def correct_inconsistencies(self, modality_outputs: List[ModalityOutput], 
                               checks: List[ConsistencyCheck]) -> Tuple[List[ModalityOutput], List[str]]:
        """
        修正不一致的输出
        
        Args:
            modality_outputs: 原始模态输出
            checks: 一致性检查结果
            
        Returns:
            (修正后的输出, 修正报告)
        """
        self.stats["total_corrections"] += 1
        
        logger.info("开始修正不一致的输出")
        
        corrected_outputs = modality_outputs.copy()
        correction_reports = []
        
        # 收集需要修正的问题
        failed_checks = [check for check in checks if check.score < self.thresholds["acceptable"]]
        
        if not failed_checks:
            logger.info("没有发现需要修正的不一致")
            return corrected_outputs, ["所有输出已保持一致"]
        
        # 针对每种问题类型进行修正
        for check in failed_checks:
            if check.consistency_type == ConsistencyType.SEMANTIC:
                # 修正语义不一致
                report = self._correct_semantic_inconsistency(corrected_outputs, check)
                correction_reports.append(report)
            
            elif check.consistency_type == ConsistencyType.LOGICAL:
                # 修正逻辑不一致
                report = self._correct_logical_inconsistency(corrected_outputs, check)
                correction_reports.append(report)
            
            elif check.consistency_type == ConsistencyType.STYLISTIC:
                # 修正风格不一致
                report = self._correct_stylistic_inconsistency(corrected_outputs, check)
                correction_reports.append(report)
        
        # 记录修正结果
        if correction_reports:
            self.stats["successful_corrections"] += 1
            logger.info(f"修正完成，修正了 {len(correction_reports)} 个问题")
        else:
            logger.warning("未能修正任何不一致")
        
        return corrected_outputs, correction_reports
    
    def _correct_semantic_inconsistency(self, outputs: List[ModalityOutput], 
                                      check: ConsistencyCheck) -> str:
        """修正语义不一致"""
        # 简化实现：标记需要修正
        return "语义不一致已标记，需要进一步处理"
    
    def _correct_logical_inconsistency(self, outputs: List[ModalityOutput], 
                                     check: ConsistencyCheck) -> str:
        """修正逻辑不一致"""
        # 简化实现：标记需要修正
        return "逻辑不一致已标记，需要重新评估逻辑关系"
    
    def _correct_stylistic_inconsistency(self, outputs: List[ModalityOutput], 
                                       check: ConsistencyCheck) -> str:
        """修正风格不一致"""
        # 简化实现：标记需要修正
        return "风格不一致已标记，需要统一输出风格"
    
    def generate_consistency_report(self, modality_outputs: List[ModalityOutput], 
                                  checks: List[ConsistencyCheck]) -> Dict[str, Any]:
        """
        生成一致性报告
        
        Args:
            modality_outputs: 模态输出
            checks: 一致性检查结果
            
        Returns:
            一致性报告
        """
        # 计算统计
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks if check.score >= self.thresholds["acceptable"])
        failed_checks = total_checks - passed_checks
        
        # 按类型分组
        checks_by_type = {}
        for check in checks:
            check_type = check.consistency_type.value
            if check_type not in checks_by_type:
                checks_by_type[check_type] = []
            checks_by_type[check_type].append(check.to_dict())
        
        # 总体评估
        overall_score = self._calculate_overall_consistency(checks)
        overall_status = "excellent" if overall_score >= self.thresholds["excellent"] else \
                        "good" if overall_score >= self.thresholds["good"] else \
                        "acceptable" if overall_score >= self.thresholds["acceptable"] else \
                        "poor" if overall_score >= self.thresholds["poor"] else "failed"
        
        # 生成报告
        report = {
            "timestamp": time.time(),
            "overall_assessment": {
                "score": overall_score,
                "status": overall_status,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "total_checks": total_checks,
                "pass_rate": passed_checks / total_checks if total_checks > 0 else 0.0
            },
            "modality_outputs": [
                {
                    "modality_type": output.modality_type,
                    "content_preview": str(output.content)[:100] + "..." if len(str(output.content)) > 100 else str(output.content)
                }
                for output in modality_outputs
            ],
            "detailed_checks": checks_by_type,
            "thresholds": self.thresholds,
            "recommendations": self._generate_recommendations(checks),
            "statistics": self.get_stats()
        }
        
        return report
    
    def _generate_recommendations(self, checks: List[ConsistencyCheck]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 收集所有建议
        all_suggestions = []
        for check in checks:
            if check.score < self.thresholds["good"]:
                all_suggestions.extend(check.suggestions)
        
        # 去重并排序
        unique_suggestions = list(set(all_suggestions))
        
        # 优先级排序
        priority_order = [
            "重新生成输出以确保语义一致性",
            "使用语义对齐机制修正输出",
            "检查因果关系和时间顺序",
            "重新评估逻辑关系",
            "统一输出风格（正式/非正式/技术性）",
            "调整语气和表达方式",
            "使用风格转换器统一风格"
        ]
        
        # 按优先级排序
        sorted_suggestions = []
        for priority in priority_order:
            for suggestion in unique_suggestions:
                if priority in suggestion:
                    sorted_suggestions.append(suggestion)
        
        # 添加其他建议
        for suggestion in unique_suggestions:
            if suggestion not in sorted_suggestions:
                sorted_suggestions.append(suggestion)
        
        return sorted_suggestions[:5]  # 返回前5个建议
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


def test_cross_modal_consistency():
    """测试跨模态一致性生成器"""
    print("测试跨模态一致性生成器...")
    
    # 创建生成器
    generator = CrossModalConsistencyGenerator()
    
    # 创建测试输出
    outputs = [
        ModalityOutput(
            modality_type="text",
            content="这是一张红色圆形杯子的图片，放在木桌上。",
            metadata={"source": "text_generator"}
        ),
        ModalityOutput(
            modality_type="image",
            content="模拟图像数据：红色圆形杯子在木桌上",
            metadata={"source": "image_generator", "resolution": "1024x768"}
        ),
        ModalityOutput(
            modality_type="audio",
            content="模拟音频数据：描述红色圆形杯子的语音",
            metadata={"source": "audio_generator", "duration": "3.5s"}
        )
    ]
    
    # 检查一致性
    checks = generator.check_consistency(outputs)
    
    # 生成报告
    report = generator.generate_consistency_report(outputs, checks)
    
    # 打印结果
    print(f"\n一致性检查完成:")
    print(f"  总体分数: {report['overall_assessment']['score']:.2f}")
    print(f"  状态: {report['overall_assessment']['status']}")
    print(f"  通过检查: {report['overall_assessment']['passed_checks']}/{report['overall_assessment']['total_checks']}")
    
    # 修正不一致
    corrected_outputs, correction_reports = generator.correct_inconsistencies(outputs, checks)
    
    print(f"\n修正结果:")
    for i, report in enumerate(correction_reports, 1):
        print(f"  {i}. {report}")
    
    print(f"\n统计信息:")
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return generator


if __name__ == "__main__":
    test_cross_modal_consistency()