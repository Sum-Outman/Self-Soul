"""
自适应输出优化器

修复计划第三阶段：提升生成能力（逻辑+质量+匹配）
任务3.2：创建自适应输出优化器

核心功能：
1. 根据输入场景和用户需求自适应调整输出形式
2. 优化输出质量和匹配度
3. 提供多模态输出优化建议
"""

import sys
import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

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


class OutputQuality(Enum):
    """输出质量等级"""
    EXCELLENT = "excellent"  # 优秀：清晰、准确、完整
    GOOD = "good"          # 良好：基本清晰，有小问题
    ACCEPTABLE = "acceptable"  # 可接受：能理解，但有明显问题
    POOR = "poor"          # 差：难以理解，问题严重
    UNACCEPTABLE = "unacceptable"  # 不可接受：完全错误或无法使用


class OutputFormat(Enum):
    """输出格式"""
    TEXT = "text"          # 文本格式
    IMAGE = "image"        # 图像格式
    AUDIO = "audio"        # 音频格式
    MULTIMODAL = "multimodal"  # 多模态格式
    INTERACTIVE = "interactive"  # 交互式格式


class UserPreference(Enum):
    """用户偏好"""
    VISUAL = "visual"      # 视觉型：偏好图像和视频
    AUDITORY = "auditory"  # 听觉型：偏好音频
    TEXTUAL = "textual"    # 文本型：偏好文字
    BALANCED = "balanced"  # 平衡型：各种格式平衡
    ADAPTIVE = "adaptive"  # 自适应型：根据场景自动调整


@dataclass
class OptimizationTarget:
    """优化目标"""
    quality_target: OutputQuality
    format_preference: OutputFormat
    user_preference: UserPreference
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.3,
        "clarity": 0.25,
        "completeness": 0.2,
        "relevance": 0.15,
        "efficiency": 0.1
    })


@dataclass
class OptimizationResult:
    """优化结果"""
    optimized_output: Any
    quality_score: float  # 0-1分数
    quality_level: OutputQuality
    improvements: List[str] = field(default_factory=list)  # 改进项
    optimization_details: Dict[str, Any] = field(default_factory=dict)  # 优化详情


class AdaptiveOutputOptimizer:
    """
    自适应输出优化器
    
    核心功能：
    1. 分析输入场景和用户需求
    2. 自适应调整输出形式和质量
    3. 提供优化建议和改进方案
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 质量评估标准
        self.quality_standards = {
            OutputQuality.EXCELLENT: {"min_score": 0.9, "description": "清晰、准确、完整"},
            OutputQuality.GOOD: {"min_score": 0.8, "description": "基本清晰，有小问题"},
            OutputQuality.ACCEPTABLE: {"min_score": 0.7, "description": "能理解，但有明显问题"},
            OutputQuality.POOR: {"min_score": 0.5, "description": "难以理解，问题严重"},
            OutputQuality.UNACCEPTABLE: {"min_score": 0.0, "description": "完全错误或无法使用"}
        }
        
        # 场景分析器
        self.scene_analyzer = SceneAnalyzer()
        
        # 质量优化器
        self.quality_optimizer = QualityOptimizer()
        
        # 格式适配器
        self.format_adapter = FormatAdapter()
        
        # 统计信息
        self.stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "quality_improvements": 0,
            "average_quality_score": 0.0,
            "format_adaptations": 0
        }
        
        logger.info("自适应输出优化器初始化完成")
    
    def optimize_output(self, original_output: Any, input_context: Dict[str, Any], 
                       target: Optional[OptimizationTarget] = None) -> OptimizationResult:
        """
        优化输出
        
        Args:
            original_output: 原始输出
            input_context: 输入上下文
            target: 优化目标（可选）
            
        Returns:
            优化结果
        """
        self.stats["total_optimizations"] += 1
        
        logger.info("开始优化输出")
        
        try:
            # 1. 分析场景
            scene_analysis = self.scene_analyzer.analyze(input_context)
            logger.info(f"场景分析完成: {scene_analysis['scene_type']}")
            
            # 2. 确定优化目标
            if target is None:
                target = self._determine_optimization_target(scene_analysis, input_context)
            
            # 3. 评估原始输出质量
            original_quality = self._assess_output_quality(original_output, scene_analysis)
            logger.info(f"原始输出质量: {original_quality['quality_level'].value} ({original_quality['score']:.2f})")
            
            # 4. 执行优化
            optimized_output, optimization_details = self._execute_optimization(
                original_output, original_quality, target, scene_analysis
            )
            
            # 5. 评估优化后质量
            optimized_quality = self._assess_output_quality(optimized_output, scene_analysis)
            
            # 6. 生成改进列表
            improvements = self._generate_improvements(original_quality, optimized_quality, optimization_details)
            
            # 7. 创建优化结果
            result = OptimizationResult(
                optimized_output=optimized_output,
                quality_score=optimized_quality["score"],
                quality_level=optimized_quality["quality_level"],
                improvements=improvements,
                optimization_details=optimization_details
            )
            
            # 更新统计
            self._update_stats(original_quality, optimized_quality, optimization_details)
            
            self.stats["successful_optimizations"] += 1
            logger.info(f"输出优化成功，质量提升: {optimized_quality['score'] - original_quality['score']:.2f}")
            
            return result
            
        except Exception as e:
            self.stats["failed_optimizations"] += 1
            logger.error(f"输出优化失败: {e}")
            
            # 返回原始输出作为失败结果
            return OptimizationResult(
                optimized_output=original_output,
                quality_score=0.5,
                quality_level=OutputQuality.POOR,
                improvements=[f"优化失败: {str(e)}"],
                optimization_details={"error": str(e), "failed": True}
            )
    
    def _determine_optimization_target(self, scene_analysis: Dict[str, Any], 
                                     input_context: Dict[str, Any]) -> OptimizationTarget:
        """
        确定优化目标
        
        Args:
            scene_analysis: 场景分析结果
            input_context: 输入上下文
            
        Returns:
            优化目标
        """
        # 分析场景类型
        scene_type = scene_analysis.get("scene_type", "general")
        
        # 确定质量目标
        quality_target = self._determine_quality_target(scene_type, input_context)
        
        # 确定格式偏好
        format_preference = self._determine_format_preference(scene_type, input_context)
        
        # 确定用户偏好
        user_preference = self._determine_user_preference(input_context)
        
        # 确定优先级权重
        priority_weights = self._determine_priority_weights(scene_type, input_context)
        
        return OptimizationTarget(
            quality_target=quality_target,
            format_preference=format_preference,
            user_preference=user_preference,
            priority_weights=priority_weights
        )
    
    def _determine_quality_target(self, scene_type: str, input_context: Dict[str, Any]) -> OutputQuality:
        """确定质量目标"""
        quality_mapping = {
            "emergency": OutputQuality.EXCELLENT,
            "professional": OutputQuality.EXCELLENT,
            "educational": OutputQuality.GOOD,
            "casual": OutputQuality.ACCEPTABLE,
            "entertainment": OutputQuality.ACCEPTABLE,
            "debug": OutputQuality.GOOD
        }
        
        return quality_mapping.get(scene_type, OutputQuality.GOOD)
    
    def _determine_format_preference(self, scene_type: str, input_context: Dict[str, Any]) -> OutputFormat:
        """确定格式偏好"""
        format_mapping = {
            "visual_scene": OutputFormat.IMAGE,
            "audio_scene": OutputFormat.AUDIO,
            "text_scene": OutputFormat.TEXT,
            "complex_scene": OutputFormat.MULTIMODAL,
            "interactive_scene": OutputFormat.INTERACTIVE
        }
        
        # 检查输入模态
        input_modalities = input_context.get("input_modalities", [])
        if len(input_modalities) >= 2:
            return OutputFormat.MULTIMODAL
        elif "image" in input_modalities:
            return OutputFormat.IMAGE
        elif "audio" in input_modalities:
            return OutputFormat.AUDIO
        else:
            return OutputFormat.TEXT
    
    def _determine_user_preference(self, input_context: Dict[str, Any]) -> UserPreference:
        """确定用户偏好"""
        # 从上下文提取用户偏好信息
        user_info = input_context.get("user_info", {})
        
        if user_info.get("prefers_visual"):
            return UserPreference.VISUAL
        elif user_info.get("prefers_audio"):
            return UserPreference.AUDITORY
        elif user_info.get("prefers_text"):
            return UserPreference.TEXTUAL
        else:
            return UserPreference.ADAPTIVE
    
    def _determine_priority_weights(self, scene_type: str, input_context: Dict[str, Any]) -> Dict[str, float]:
        """确定优先级权重"""
        # 不同场景的优先级权重
        scene_priorities = {
            "emergency": {"accuracy": 0.4, "clarity": 0.3, "completeness": 0.2, "relevance": 0.1, "efficiency": 0.0},
            "professional": {"accuracy": 0.35, "clarity": 0.25, "completeness": 0.2, "relevance": 0.15, "efficiency": 0.05},
            "educational": {"accuracy": 0.3, "clarity": 0.3, "completeness": 0.25, "relevance": 0.1, "efficiency": 0.05},
            "casual": {"accuracy": 0.25, "clarity": 0.25, "completeness": 0.2, "relevance": 0.2, "efficiency": 0.1},
            "entertainment": {"accuracy": 0.2, "clarity": 0.3, "completeness": 0.2, "relevance": 0.2, "efficiency": 0.1}
        }
        
        return scene_priorities.get(scene_type, {
            "accuracy": 0.3,
            "clarity": 0.25,
            "completeness": 0.2,
            "relevance": 0.15,
            "efficiency": 0.1
        })
    
    def _assess_output_quality(self, output: Any, scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估输出质量
        
        Args:
            output: 输出内容
            scene_analysis: 场景分析结果
            
        Returns:
            质量评估结果
        """
        # 调用质量优化器进行评估
        quality_result = self.quality_optimizer.assess_quality(output, scene_analysis)
        
        # 确定质量等级
        quality_level = self._determine_quality_level(quality_result["overall_score"])
        
        return {
            "score": quality_result["overall_score"],
            "quality_level": quality_level,
            "detailed_scores": quality_result["detailed_scores"],
            "issues": quality_result["issues"],
            "strengths": quality_result["strengths"]
        }
    
    def _determine_quality_level(self, score: float) -> OutputQuality:
        """根据分数确定质量等级"""
        for quality, standard in self.quality_standards.items():
            if score >= standard["min_score"]:
                return quality
        
        return OutputQuality.UNACCEPTABLE
    
    def _execute_optimization(self, original_output: Any, original_quality: Dict[str, Any],
                            target: OptimizationTarget, scene_analysis: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        执行优化
        
        Args:
            original_output: 原始输出
            original_quality: 原始质量评估
            target: 优化目标
            scene_analysis: 场景分析
            
        Returns:
            (优化后的输出, 优化详情)
        """
        optimization_details = {
            "original_quality": original_quality["score"],
            "target_quality": self.quality_standards[target.quality_target]["min_score"],
            "format_preference": target.format_preference.value,
            "user_preference": target.user_preference.value,
            "applied_optimizations": []
        }
        
        # 应用质量优化
        if original_quality["score"] < self.quality_standards[target.quality_target]["min_score"]:
            optimized_output, quality_optimization = self.quality_optimizer.optimize_quality(
                original_output, target, scene_analysis
            )
            optimization_details["applied_optimizations"].append("quality_optimization")
            optimization_details["quality_improvement"] = quality_optimization
        else:
            optimized_output = original_output
        
        # 应用格式适配
        if target.format_preference != OutputFormat.TEXT:  # 假设默认是文本
            adapted_output, format_adaptation = self.format_adapter.adapt_format(
                optimized_output, target.format_preference, scene_analysis
            )
            if adapted_output != optimized_output:
                optimized_output = adapted_output
                optimization_details["applied_optimizations"].append("format_adaptation")
                optimization_details["format_adaptation"] = format_adaptation
                self.stats["format_adaptations"] += 1
        
        return optimized_output, optimization_details
    
    def _generate_improvements(self, original_quality: Dict[str, Any], 
                             optimized_quality: Dict[str, Any],
                             optimization_details: Dict[str, Any]) -> List[str]:
        """生成改进列表"""
        improvements = []
        
        # 质量改进
        quality_improvement = optimized_quality["score"] - original_quality["score"]
        if quality_improvement > 0:
            improvements.append(f"质量提升: +{quality_improvement:.2f}")
        
        # 解决的具体问题
        original_issues = original_quality.get("issues", [])
        optimized_issues = optimized_quality.get("issues", [])
        
        resolved_issues = set(original_issues) - set(optimized_issues)
        if resolved_issues:
            improvements.append(f"解决了 {len(resolved_issues)} 个问题")
        
        # 新增的优势
        new_strengths = set(optimized_quality.get("strengths", [])) - set(original_quality.get("strengths", []))
        if new_strengths:
            improvements.append(f"增强了 {len(new_strengths)} 个方面")
        
        # 格式优化
        if "format_adaptation" in optimization_details:
            improvements.append("输出格式已根据场景优化")
        
        # 用户偏好匹配
        if "user_preference" in optimization_details:
            improvements.append("输出已适应用户偏好")
        
        # 如果没有改进，添加通用建议
        if not improvements:
            improvements.append("输出已符合基本要求，建议进一步优化细节")
        
        return improvements
    
    def _update_stats(self, original_quality: Dict[str, Any], 
                     optimized_quality: Dict[str, Any],
                     optimization_details: Dict[str, Any]) -> None:
        """更新统计信息"""
        # 计算质量改进
        quality_improvement = optimized_quality["score"] - original_quality["score"]
        if quality_improvement > 0.05:  # 显著改进
            self.stats["quality_improvements"] += 1
        
        # 更新平均质量分数
        total_optimizations = self.stats["total_optimizations"]
        current_avg = self.stats["average_quality_score"]
        self.stats["average_quality_score"] = (current_avg * (total_optimizations - 1) + optimized_quality["score"]) / total_optimizations
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


class SceneAnalyzer:
    """场景分析器"""
    
    def analyze(self, input_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析输入场景
        
        Args:
            input_context: 输入上下文
            
        Returns:
            场景分析结果
        """
        # 提取场景信息
        scene_type = self._determine_scene_type(input_context)
        complexity = self._assess_complexity(input_context)
        urgency = self._assess_urgency(input_context)
        
        return {
            "scene_type": scene_type,
            "complexity": complexity,
            "urgency": urgency,
            "input_modalities": input_context.get("input_modalities", []),
            "user_context": input_context.get("user_info", {}),
            "environment": input_context.get("environment", {})
        }
    
    def _determine_scene_type(self, input_context: Dict[str, Any]) -> str:
        """确定场景类型"""
        # 从上下文提取信息
        content = input_context.get("content", "")
        user_intent = input_context.get("user_intent", "")
        environment = input_context.get("environment", {})
        
        # 场景类型判断
        if "紧急" in content or "urgent" in content.lower() or environment.get("emergency"):
            return "emergency"
        elif "专业" in content or "professional" in content.lower() or user_intent == "professional_work":
            return "professional"
        elif "学习" in content or "教育" in content or "educational" in content.lower():
            return "educational"
        elif "娱乐" in content or "休闲" in content or "entertainment" in content.lower():
            return "entertainment"
        elif "调试" in content or "问题" in content or "debug" in content.lower():
            return "debug"
        else:
            return "casual"
    
    def _assess_complexity(self, input_context: Dict[str, Any]) -> str:
        """评估复杂度"""
        content = input_context.get("content", "")
        input_modalities = input_context.get("input_modalities", [])
        
        # 基于内容长度和模态数量
        content_length = len(str(content))
        modality_count = len(input_modalities)
        
        complexity_score = (content_length / 1000) + (modality_count * 0.5)
        
        if complexity_score > 2.0:
            return "high"
        elif complexity_score > 1.0:
            return "medium"
        else:
            return "low"
    
    def _assess_urgency(self, input_context: Dict[str, Any]) -> str:
        """评估紧急程度"""
        environment = input_context.get("environment", {})
        content = input_context.get("content", "")
        
        if environment.get("emergency") or "立即" in content or "urgent" in content.lower():
            return "high"
        elif "尽快" in content or "asap" in content.lower():
            return "medium"
        else:
            return "low"


class QualityOptimizer:
    """质量优化器"""
    
    def assess_quality(self, output: Any, scene_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估输出质量
        
        Args:
            output: 输出内容
            scene_analysis: 场景分析结果
            
        Returns:
            质量评估结果
        """
        # 简化实现：返回模拟质量评估
        detailed_scores = {
            "accuracy": np.random.uniform(0.7, 0.95),
            "clarity": np.random.uniform(0.6, 0.9),
            "completeness": np.random.uniform(0.5, 0.85),
            "relevance": np.random.uniform(0.8, 0.95),
            "efficiency": np.random.uniform(0.7, 0.9)
        }
        
        # 计算总体分数（加权平均）
        weights = {"accuracy": 0.3, "clarity": 0.25, "completeness": 0.2, "relevance": 0.15, "efficiency": 0.1}
        overall_score = sum(detailed_scores[k] * weights[k] for k in weights)
        
        # 识别问题和优势
        issues = self._identify_issues(detailed_scores)
        strengths = self._identify_strengths(detailed_scores)
        
        return {
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "issues": issues,
            "strengths": strengths
        }
    
    def _identify_issues(self, scores: Dict[str, float]) -> List[str]:
        """识别问题"""
        issues = []
        
        if scores["accuracy"] < 0.8:
            issues.append("准确性有待提高")
        if scores["clarity"] < 0.7:
            issues.append("清晰度不足")
        if scores["completeness"] < 0.6:
            issues.append("内容不完整")
        if scores["relevance"] < 0.85:
            issues.append("相关性需要加强")
        
        return issues
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """识别优势"""
        strengths = []
        
        if scores["accuracy"] >= 0.9:
            strengths.append("准确性高")
        if scores["clarity"] >= 0.8:
            strengths.append("表达清晰")
        if scores["completeness"] >= 0.8:
            strengths.append("内容完整")
        if scores["relevance"] >= 0.9:
            strengths.append("高度相关")
        
        return strengths
    
    def optimize_quality(self, output: Any, target: OptimizationTarget, 
                        scene_analysis: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        优化输出质量
        
        Args:
            output: 原始输出
            target: 优化目标
            scene_analysis: 场景分析
            
        Returns:
            (优化后的输出, 优化详情)
        """
        # 简化实现：返回模拟优化
        optimization_details = {
            "applied_techniques": ["quality_enhancement", "error_correction", "content_refinement"],
            "quality_improvement": np.random.uniform(0.1, 0.3),
            "target_met": True
        }
        
        # 模拟优化后的输出
        optimized_output = f"优化后的输出: {str(output)[:50]}..." if len(str(output)) > 50 else f"优化: {output}"
        
        return optimized_output, optimization_details


class FormatAdapter:
    """格式适配器"""
    
    def adapt_format(self, output: Any, target_format: OutputFormat, 
                    scene_analysis: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        适配输出格式
        
        Args:
            output: 原始输出
            target_format: 目标格式
            scene_analysis: 场景分析
            
        Returns:
            (适配后的输出, 适配详情)
        """
        adaptation_details = {
            "original_format": "text",  # 假设原始是文本
            "target_format": target_format.value,
            "adaptation_success": True,
            "techniques_applied": []
        }
        
        # 根据目标格式进行适配
        if target_format == OutputFormat.IMAGE:
            adapted_output = f"图像格式: {str(output)[:30]}..."
            adaptation_details["techniques_applied"].append("text_to_image_conversion")
        
        elif target_format == OutputFormat.AUDIO:
            adapted_output = f"音频格式: {str(output)[:40]}..."
            adaptation_details["techniques_applied"].append("text_to_speech_conversion")
        
        elif target_format == OutputFormat.MULTIMODAL:
            adapted_output = f"多模态格式: {str(output)[:20]}..."
            adaptation_details["techniques_applied"].append("multimodal_integration")
        
        elif target_format == OutputFormat.INTERACTIVE:
            adapted_output = f"交互式格式: {str(output)[:25]}..."
            adaptation_details["techniques_applied"].append("interactive_element_addition")
        
        else:  # TEXT
            adapted_output = output
        
        return adapted_output, adaptation_details


def test_adaptive_output_optimizer():
    """测试自适应输出优化器"""
    print("测试自适应输出优化器...")
    
    # 创建优化器
    optimizer = AdaptiveOutputOptimizer()
    
    # 创建测试输入
    original_output = "这是一个测试输出，包含一些基本信息和描述。"
    
    input_context = {
        "content": "紧急情况下需要清晰准确的信息",
        "input_modalities": ["text", "image"],
        "user_info": {"prefers_visual": True},
        "environment": {"emergency": True},
        "user_intent": "emergency_response"
    }
    
    # 创建优化目标
    target = OptimizationTarget(
        quality_target=OutputQuality.EXCELLENT,
        format_preference=OutputFormat.MULTIMODAL,
        user_preference=UserPreference.VISUAL,
        priority_weights={
            "accuracy": 0.4,
            "clarity": 0.3,
            "completeness": 0.2,
            "relevance": 0.1,
            "efficiency": 0.0
        }
    )
    
    # 执行优化
    result = optimizer.optimize_output(original_output, input_context, target)
    
    # 打印结果
    print(f"\n优化结果:")
    print(f"  质量分数: {result.quality_score:.2f}")
    print(f"  质量等级: {result.quality_level.value}")
    print(f"  优化输出预览: {str(result.optimized_output)[:80]}...")
    
    print(f"\n改进项:")
    for i, improvement in enumerate(result.improvements, 1):
        print(f"  {i}. {improvement}")
    
    print(f"\n优化详情:")
    for key, value in result.optimization_details.items():
        print(f"  {key}: {value}")
    
    print(f"\n统计信息:")
    stats = optimizer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return optimizer


if __name__ == "__main__":
    test_adaptive_output_optimizer()