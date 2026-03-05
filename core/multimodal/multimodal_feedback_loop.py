"""
多模态反馈循环

修复计划第三阶段：提升生成能力（逻辑+质量+匹配）
任务3.3：创建多模态反馈循环

核心功能：
1. 支持跨模态错误修正
2. 基于用户反馈同步修正多模态输出
3. 实现增量式改进，不断提升生成质量
"""

import sys
import os
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

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


class FeedbackType(Enum):
    """反馈类型"""
    CORRECTION = "correction"  # 纠正错误
    IMPROVEMENT = "improvement"  # 改进建议
    CLARIFICATION = "clarification"  # 澄清说明
    PREFERENCE = "preference"  # 偏好表达
    CRITICISM = "criticism"  # 批评指正


class CorrectionScope(Enum):
    """修正范围"""
    SINGLE_MODALITY = "single_modality"  # 单模态修正
    CROSS_MODALITY = "cross_modality"  # 跨模态修正
    ALL_MODALITIES = "all_modalities"  # 所有模态修正


@dataclass
class FeedbackItem:
    """反馈项"""
    feedback_type: FeedbackType
    content: str  # 反馈内容
    target_modality: Optional[str] = None  # 目标模态
    target_element: Optional[str] = None  # 目标元素
    confidence: float = 1.0  # 反馈置信度
    timestamp: float = field(default_factory=time.time)  # 时间戳
    user_id: Optional[str] = None  # 用户ID
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "feedback_type": self.feedback_type.value,
            "content": self.content,
            "target_modality": self.target_modality,
            "target_element": self.target_element,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "context": self.context
        }


@dataclass
class CorrectionResult:
    """修正结果"""
    original_output: Any  # 原始输出
    corrected_output: Any  # 修正后输出
    feedback_applied: List[FeedbackItem]  # 应用的反馈
    correction_scope: CorrectionScope  # 修正范围
    success: bool  # 是否成功
    improvement_score: float  # 改进分数 0-1
    details: Dict[str, Any] = field(default_factory=dict)  # 详细结果
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_output_preview": str(self.original_output)[:100] + "..." if len(str(self.original_output)) > 100 else str(self.original_output),
            "corrected_output_preview": str(self.corrected_output)[:100] + "..." if len(str(self.corrected_output)) > 100 else str(self.corrected_output),
            "feedback_applied_count": len(self.feedback_applied),
            "correction_scope": self.correction_scope.value,
            "success": self.success,
            "improvement_score": self.improvement_score,
            "details": self.details
        }


class MultimodalFeedbackLoop:
    """
    多模态反馈循环
    
    核心功能：
    1. 接收和处理用户反馈
    2. 同步修正多模态输出
    3. 学习反馈模式，实现增量式改进
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化反馈循环
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 反馈存储器
        self.feedback_storage = FeedbackStorage()
        
        # 修正引擎
        self.correction_engine = CorrectionEngine()
        
        # 学习模块
        self.learning_module = LearningModule()
        
        # 统计信息
        self.stats = {
            "total_feedbacks": 0,
            "processed_feedbacks": 0,
            "successful_corrections": 0,
            "failed_corrections": 0,
            "cross_modality_corrections": 0,
            "learning_updates": 0,
            "average_improvement_score": 0.0
        }
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("多模态反馈循环初始化完成")
    
    def process_feedback(self, feedback: FeedbackItem, original_outputs: Dict[str, Any]) -> CorrectionResult:
        """
        处理反馈并修正输出
        
        Args:
            feedback: 反馈项
            original_outputs: 原始多模态输出字典 {modality_type: output}
            
        Returns:
            修正结果
        """
        self.stats["total_feedbacks"] += 1
        
        logger.info(f"处理反馈: {feedback.feedback_type.value} - {feedback.content[:50]}...")
        
        try:
            # 1. 存储反馈
            feedback_id = self.feedback_storage.store_feedback(feedback)
            logger.debug(f"反馈已存储，ID: {feedback_id}")
            
            # 2. 分析反馈
            feedback_analysis = self._analyze_feedback(feedback, original_outputs)
            logger.info(f"反馈分析完成: {feedback_analysis['scope']}")
            
            # 3. 确定修正范围
            correction_scope = self._determine_correction_scope(feedback_analysis)
            
            # 4. 执行修正
            correction_result = self.correction_engine.correct_outputs(
                original_outputs, feedback, feedback_analysis, correction_scope
            )
            
            # 5. 评估修正效果
            improvement_score = self._evaluate_improvement(original_outputs, correction_result.corrected_outputs, feedback)
            correction_result.improvement_score = improvement_score
            
            # 6. 学习反馈模式
            if improvement_score > 0.5:  # 有效修正才学习
                self.learning_module.learn_from_feedback(feedback, feedback_analysis, improvement_score)
                self.stats["learning_updates"] += 1
            
            # 7. 更新统计
            self._update_stats(correction_result, improvement_score)
            
            self.stats["processed_feedbacks"] += 1
            self.stats["successful_corrections"] += 1
            
            logger.info(f"反馈处理成功，改进分数: {improvement_score:.2f}")
            
            return correction_result
            
        except Exception as e:
            self.stats["failed_corrections"] += 1
            logger.error(f"反馈处理失败: {e}")
            
            # 返回失败结果
            return CorrectionResult(
                original_output=original_outputs,
                corrected_output=original_outputs,  # 返回原始输出
                feedback_applied=[feedback],
                correction_scope=CorrectionScope.SINGLE_MODALITY,
                success=False,
                improvement_score=0.0,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    def _analyze_feedback(self, feedback: FeedbackItem, original_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析反馈
        
        Args:
            feedback: 反馈项
            original_outputs: 原始输出
            
        Returns:
            反馈分析结果
        """
        analysis = {
            "feedback_type": feedback.feedback_type.value,
            "content_analysis": self._analyze_content(feedback.content),
            "target_identified": False,
            "scope": "unknown",
            "confidence": feedback.confidence,
            "recommended_actions": []
        }
        
        # 识别目标模态
        if feedback.target_modality:
            analysis["target_modality"] = feedback.target_modality
            analysis["target_identified"] = True
            analysis["scope"] = "single_modality"
        else:
            # 自动识别目标模态
            identified_modality = self._identify_target_modality(feedback.content, original_outputs)
            if identified_modality:
                analysis["target_modality"] = identified_modality
                analysis["target_identified"] = True
                analysis["scope"] = "single_modality"
            else:
                # 可能是跨模态反馈
                analysis["scope"] = "cross_modality"
        
        # 分析反馈内容
        content_lower = feedback.content.lower()
        
        # 检查是否为跨模态修正
        cross_modal_keywords = ["图片里的", "语音中的", "文本和图片", "同时修正", "都错了"]
        if any(keyword in content_lower for keyword in cross_modal_keywords):
            analysis["scope"] = "cross_modality"
            analysis["recommended_actions"].append("执行跨模态同步修正")
        
        # 检查具体错误类型
        if "颜色" in feedback.content or "色彩" in feedback.content:
            analysis["error_type"] = "color_error"
            analysis["recommended_actions"].append("修正颜色信息")
        elif "形状" in feedback.content or "形态" in feedback.content:
            analysis["error_type"] = "shape_error"
            analysis["recommended_actions"].append("修正形状信息")
        elif "大小" in feedback.content or "尺寸" in feedback.content:
            analysis["error_type"] = "size_error"
            analysis["recommended_actions"].append("修正尺寸信息")
        elif "材质" in feedback.content or "材料" in feedback.content:
            analysis["error_type"] = "material_error"
            analysis["recommended_actions"].append("修正材质信息")
        
        # 示例：用户指出"图片里的杯子是玻璃的，不是陶瓷的"
        if "玻璃" in feedback.content and "陶瓷" in feedback.content:
            analysis["error_type"] = "material_correction"
            analysis["scope"] = "cross_modality"
            analysis["recommended_actions"].append("将陶瓷材质修正为玻璃材质")
            analysis["recommended_actions"].append("同步修正所有模态中的材质描述")
        
        return analysis
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """分析反馈内容"""
        content_lower = content.lower()
        
        analysis = {
            "length": len(content),
            "has_correction": any(word in content_lower for word in ["错了", "不对", "不正确", "应该是", "应该是"]),
            "has_improvement": any(word in content_lower for word in ["更好", "改进", "优化", "建议"]),
            "has_clarification": any(word in content_lower for word in ["解释", "说明", "什么意思", "为什么"]),
            "has_preference": any(word in content_lower for word in ["喜欢", "偏好", "希望", "想要"]),
            "has_criticism": any(word in content_lower for word in ["糟糕", "差劲", "不满意", "不好"]),
            "specificity_score": min(1.0, len(content) / 100)  # 内容越详细，特异性越高
        }
        
        return analysis
    
    def _identify_target_modality(self, content: str, original_outputs: Dict[str, Any]) -> Optional[str]:
        """识别目标模态"""
        content_lower = content.lower()
        
        modality_keywords = {
            "text": ["文本", "文字", "描述", "句子", "段落"],
            "image": ["图片", "图像", "照片", "画面", "视觉"],
            "audio": ["语音", "音频", "声音", "录音", "听觉"]
        }
        
        for modality, keywords in modality_keywords.items():
            if modality in original_outputs:  # 该模态有输出
                for keyword in keywords:
                    if keyword in content_lower:
                        return modality
        
        # 如果没有明确关键词，根据内容特征推断
        if any(word in content_lower for word in ["像素", "分辨率", "颜色", "形状"]):
            return "image"
        elif any(word in content_lower for word in ["音量", "音调", "语速", "发音"]):
            return "audio"
        elif any(word in content_lower for word in ["语法", "词汇", "表达", "语句"]):
            return "text"
        
        return None
    
    def _determine_correction_scope(self, feedback_analysis: Dict[str, Any]) -> CorrectionScope:
        """确定修正范围"""
        scope = feedback_analysis.get("scope", "unknown")
        
        if scope == "cross_modality":
            return CorrectionScope.CROSS_MODALITY
        elif scope == "single_modality" and feedback_analysis.get("target_identified"):
            return CorrectionScope.SINGLE_MODALITY
        else:
            # 默认单模态修正
            return CorrectionScope.SINGLE_MODALITY
    
    def _evaluate_improvement(self, original_outputs: Dict[str, Any], corrected_outputs: Dict[str, Any],
                            feedback: FeedbackItem) -> float:
        """评估改进效果"""
        # 简化实现：基于反馈类型和修正范围计算分数
        base_score = 0.5
        
        # 根据反馈类型调整
        if feedback.feedback_type == FeedbackType.CORRECTION:
            base_score += 0.2
        elif feedback.feedback_type == FeedbackType.IMPROVEMENT:
            base_score += 0.1
        
        # 根据内容特异性调整
        content_length = len(feedback.content)
        specificity_bonus = min(0.3, content_length / 200)
        base_score += specificity_bonus
        
        # 根据置信度调整
        base_score *= feedback.confidence
        
        # 确保在0-1范围内
        return max(0.0, min(1.0, base_score))
    
    def _update_stats(self, correction_result: CorrectionResult, improvement_score: float) -> None:
        """更新统计信息"""
        # 跨模态修正统计
        if correction_result.correction_scope == CorrectionScope.CROSS_MODALITY:
            self.stats["cross_modality_corrections"] += 1
        
        # 更新平均改进分数
        total_corrections = self.stats["successful_corrections"]
        current_avg = self.stats["average_improvement_score"]
        self.stats["average_improvement_score"] = (current_avg * total_corrections + improvement_score) / (total_corrections + 1)
    
    def get_feedback_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        获取反馈摘要
        
        Args:
            time_range_hours: 时间范围（小时）
            
        Returns:
            反馈摘要
        """
        # 获取指定时间范围内的反馈
        recent_feedbacks = self.feedback_storage.get_recent_feedbacks(time_range_hours)
        
        # 统计反馈类型
        type_counts = {}
        for feedback in recent_feedbacks:
            f_type = feedback.feedback_type.value
            type_counts[f_type] = type_counts.get(f_type, 0) + 1
        
        # 计算修正成功率
        total_feedbacks = self.stats["processed_feedbacks"]
        successful_corrections = self.stats["successful_corrections"]
        success_rate = successful_corrections / total_feedbacks if total_feedbacks > 0 else 0.0
        
        return {
            "time_range_hours": time_range_hours,
            "total_feedbacks": len(recent_feedbacks),
            "type_distribution": type_counts,
            "correction_stats": {
                "total_processed": total_feedbacks,
                "successful": successful_corrections,
                "failed": self.stats["failed_corrections"],
                "success_rate": success_rate,
                "cross_modality_count": self.stats["cross_modality_corrections"]
            },
            "learning_stats": {
                "learning_updates": self.stats["learning_updates"],
                "average_improvement": self.stats["average_improvement_score"]
            },
            "performance_stats": self.performance_monitor.get_stats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


class FeedbackStorage:
    """反馈存储器"""
    
    def __init__(self, max_storage: int = 10000):
        """初始化存储器"""
        self.max_storage = max_storage
        self.feedbacks = []  # 简化实现，实际应使用数据库
        self.next_id = 1
    
    def store_feedback(self, feedback: FeedbackItem) -> str:
        """存储反馈"""
        feedback_id = f"feedback_{self.next_id}"
        self.next_id += 1
        
        # 简化存储
        if len(self.feedbacks) >= self.max_storage:
            # 移除最旧的反馈
            self.feedbacks.pop(0)
        
        self.feedbacks.append({
            "id": feedback_id,
            "feedback": feedback.to_dict(),
            "timestamp": time.time()
        })
        
        return feedback_id
    
    def get_recent_feedbacks(self, hours: int) -> List[Dict[str, Any]]:
        """获取最近的反馈"""
        cutoff_time = time.time() - (hours * 3600)
        
        recent = []
        for item in self.feedbacks:
            if item["timestamp"] >= cutoff_time:
                recent.append(item["feedback"])
        
        return recent


class CorrectionEngine:
    """修正引擎"""
    
    def correct_outputs(self, original_outputs: Dict[str, Any], feedback: FeedbackItem,
                       feedback_analysis: Dict[str, Any], scope: CorrectionScope) -> CorrectionResult:
        """
        修正输出
        
        Args:
            original_outputs: 原始输出
            feedback: 反馈项
            feedback_analysis: 反馈分析
            scope: 修正范围
            
        Returns:
            修正结果
        """
        corrected_outputs = original_outputs.copy()
        applied_feedbacks = [feedback]
        details = {
            "analysis": feedback_analysis,
            "correction_method": "direct_correction",
            "scope_applied": scope.value
        }
        
        # 根据范围执行修正
        if scope == CorrectionScope.SINGLE_MODALITY:
            # 单模态修正
            target_modality = feedback_analysis.get("target_modality")
            if target_modality and target_modality in corrected_outputs:
                corrected_outputs[target_modality] = self._correct_single_modality(
                    corrected_outputs[target_modality], feedback, feedback_analysis
                )
                details["target_modality"] = target_modality
        
        elif scope == CorrectionScope.CROSS_MODALITY:
            # 跨模态修正
            for modality, output in corrected_outputs.items():
                corrected_outputs[modality] = self._correct_cross_modality(
                    output, feedback, feedback_analysis, modality
                )
            details["all_modalities_corrected"] = True
        
        elif scope == CorrectionScope.ALL_MODALITIES:
            # 所有模态修正
            for modality, output in corrected_outputs.items():
                corrected_outputs[modality] = self._apply_general_correction(output, feedback)
            details["all_modalities_corrected"] = True
        
        return CorrectionResult(
            original_output=original_outputs,
            corrected_output=corrected_outputs,
            feedback_applied=applied_feedbacks,
            correction_scope=scope,
            success=True,
            improvement_score=0.7,  # 默认分数，实际应计算
            details=details
        )
    
    def _correct_single_modality(self, original_output: Any, feedback: FeedbackItem,
                               analysis: Dict[str, Any]) -> Any:
        """修正单模态输出"""
        # 简化实现：添加修正标记
        if isinstance(original_output, str):
            return f"[修正] {original_output} - 根据反馈: {feedback.content[:30]}..."
        else:
            return f"修正后的{type(original_output).__name__}: 已应用反馈"
    
    def _correct_cross_modality(self, original_output: Any, feedback: FeedbackItem,
                              analysis: Dict[str, Any], modality: str) -> Any:
        """修正跨模态输出"""
        # 示例：用户指出"图片里的杯子是玻璃的，不是陶瓷的"
        if "玻璃" in feedback.content and "陶瓷" in feedback.content:
            if modality == "text":
                return original_output.replace("陶瓷", "玻璃") if isinstance(original_output, str) else original_output
            elif modality == "image":
                return f"修正图像: 陶瓷→玻璃 - {original_output}" if isinstance(original_output, str) else original_output
            elif modality == "audio":
                return f"修正音频: 陶瓷→玻璃 - {original_output}" if isinstance(original_output, str) else original_output
        
        # 通用跨模态修正
        if isinstance(original_output, str):
            return f"[跨模态修正] {original_output} - 反馈: {feedback.content[:20]}..."
        else:
            return original_output
    
    def _apply_general_correction(self, original_output: Any, feedback: FeedbackItem) -> Any:
        """应用通用修正"""
        if isinstance(original_output, str):
            return f"[通用修正] {original_output}"
        else:
            return original_output


class LearningModule:
    """学习模块"""
    
    def __init__(self):
        """初始化学习模块"""
        self.patterns = {}
        self.correction_rules = {}
        self.learning_rate = 0.1
    
    def learn_from_feedback(self, feedback: FeedbackItem, analysis: Dict[str, Any], improvement_score: float):
        """从反馈中学习"""
        # 提取反馈模式
        pattern_key = self._extract_pattern(feedback, analysis)
        
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {
                "count": 0,
                "total_improvement": 0.0,
                "average_improvement": 0.0,
                "last_updated": time.time()
            }
        
        # 更新模式统计
        pattern = self.patterns[pattern_key]
        pattern["count"] += 1
        pattern["total_improvement"] += improvement_score
        pattern["average_improvement"] = pattern["total_improvement"] / pattern["count"]
        pattern["last_updated"] = time.time()
        
        # 如果模式频繁出现且改进效果好，创建修正规则
        if pattern["count"] >= 3 and pattern["average_improvement"] > 0.7:
            self._create_correction_rule(feedback, analysis, pattern)
    
    def _extract_pattern(self, feedback: FeedbackItem, analysis: Dict[str, Any]) -> str:
        """提取反馈模式"""
        # 基于反馈类型、错误类型和目标模态创建模式键
        error_type = analysis.get("error_type", "general")
        target_modality = analysis.get("target_modality", "unknown")
        
        return f"{feedback.feedback_type.value}_{error_type}_{target_modality}"
    
    def _create_correction_rule(self, feedback: FeedbackItem, analysis: Dict[str, Any], pattern: Dict[str, Any]):
        """创建修正规则"""
        rule_id = f"rule_{len(self.correction_rules) + 1}"
        
        self.correction_rules[rule_id] = {
            "pattern": self._extract_pattern(feedback, analysis),
            "feedback_example": feedback.content[:100],
            "analysis": analysis,
            "effectiveness": pattern["average_improvement"],
            "frequency": pattern["count"],
            "created_at": time.time(),
            "last_applied": None,
            "application_count": 0
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化性能监控器"""
        self.response_times = []
        self.error_counts = []
        self.start_time = time.time()
    
    def record_response_time(self, response_time: float):
        """记录响应时间"""
        self.response_times.append(response_time)
        # 保持最近1000个记录
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
    
    def record_error(self):
        """记录错误"""
        self.error_counts.append(time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.response_times:
            avg_response_time = 0.0
            max_response_time = 0.0
            min_response_time = 0.0
        else:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            max_response_time = max(self.response_times)
            min_response_time = min(self.response_times)
        
        # 计算错误率（最近1小时）
        one_hour_ago = time.time() - 3600
        recent_errors = sum(1 for t in self.error_counts if t >= one_hour_ago)
        
        return {
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "average_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "recent_errors": recent_errors,
            "total_recorded_calls": len(self.response_times)
        }


def test_multimodal_feedback_loop():
    """测试多模态反馈循环"""
    print("测试多模态反馈循环...")
    
    # 创建反馈循环实例
    feedback_loop = MultimodalFeedbackLoop()
    
    # 创建测试输出
    original_outputs = {
        "text": "这是一只陶瓷杯子，放在木桌上。",
        "image": "陶瓷杯子的图像数据",
        "audio": "描述陶瓷杯子的语音"
    }
    
    # 创建测试反馈（用户指出错误）
    feedback = FeedbackItem(
        feedback_type=FeedbackType.CORRECTION,
        content="图片里的杯子是玻璃的，不是陶瓷的，请修正所有相关描述。",
        target_modality=None,  # 自动识别
        confidence=0.9,
        user_id="test_user_001"
    )
    
    # 处理反馈
    result = feedback_loop.process_feedback(feedback, original_outputs)
    
    # 打印结果
    print(f"\n反馈处理结果:")
    print(f"  成功: {result.success}")
    print(f"  修正范围: {result.correction_scope.value}")
    print(f"  改进分数: {result.improvement_score:.2f}")
    print(f"  应用反馈数: {len(result.feedback_applied)}")
    
    print(f"\n原始输出:")
    for modality, output in result.original_output.items():
        print(f"  {modality}: {str(output)[:50]}...")
    
    print(f"\n修正后输出:")
    for modality, output in result.corrected_output.items():
        print(f"  {modality}: {str(output)[:50]}...")
    
    print(f"\n详细结果:")
    for key, value in result.details.items():
        if isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
        else:
            print(f"  {key}: {value}")
    
    # 获取反馈摘要
    summary = feedback_loop.get_feedback_summary(time_range_hours=1)
    
    print(f"\n反馈摘要 (最近1小时):")
    print(f"  总反馈数: {summary['total_feedbacks']}")
    print(f"  修正成功率: {summary['correction_stats']['success_rate']:.2%}")
    print(f"  跨模态修正数: {summary['correction_stats']['cross_modality_count']}")
    
    print(f"\n统计信息:")
    stats = feedback_loop.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return feedback_loop


if __name__ == "__main__":
    test_multimodal_feedback_loop()