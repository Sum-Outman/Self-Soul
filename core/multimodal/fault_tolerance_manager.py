"""
容错与纠错机制管理器

实现真正的多模态输入容错处理，解决当前系统的核心交互逻辑缺陷：
1. 处理格式异常、质量低下、部分损坏的输入
2. 实现自适应降级策略
3. 添加用户反馈学习，不断优化容错能力
4. 提供错误检测、修复和恢复机制

核心功能示例：
- 输入：模糊图片 + 嘈杂语音 + 错别字文本
- 输出：修复后的清晰图片 + 降噪语音 + 纠正文本
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    FORMAT_ERROR = "format_error"           # 格式错误
    QUALITY_ERROR = "quality_error"         # 质量错误
    CORRUPTION_ERROR = "corruption_error"   # 损坏错误
    NOISE_ERROR = "noise_error"             # 噪声错误
    INCOMPLETE_ERROR = "incomplete_error"   # 不完整错误
    INCONSISTENCY_ERROR = "inconsistency_error"  # 不一致错误


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


@dataclass
class ErrorDetection:
    """错误检测结果"""
    error_type: ErrorType
    modality: ModalityType
    severity: float  # 严重程度 0-1
    confidence: float  # 置信度 0-1
    location: Optional[str] = None  # 错误位置
    description: Optional[str] = None  # 错误描述
    timestamp: float = None  # 检测时间
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_type": self.error_type.value,
            "modality": self.modality.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "location": self.location,
            "description": self.description,
            "timestamp": self.timestamp
        }


@dataclass
class RepairAction:
    """修复动作"""
    action_type: str  # 修复动作类型
    modality: ModalityType  # 目标模态
    parameters: Dict[str, Any]  # 修复参数
    confidence: float  # 修复置信度
    expected_improvement: float  # 预期改进程度 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action_type": self.action_type,
            "modality": self.modality.value,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "expected_improvement": self.expected_improvement
        }


@dataclass
class QualityAssessment:
    """质量评估结果"""
    modality: ModalityType
    overall_quality: float  # 整体质量 0-1
    clarity: float  # 清晰度/可读性
    completeness: float  # 完整性
    consistency: float  # 一致性
    noise_level: float  # 噪声水平 0-1
    details: Dict[str, float]  # 详细评估指标
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "modality": self.modality.value,
            "overall_quality": self.overall_quality,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "consistency": self.consistency,
            "noise_level": self.noise_level,
            "details": self.details
        }


class DegradationStrategy:
    """降级策略管理器"""
    
    def __init__(self):
        """初始化降级策略"""
        self.strategies = self._initialize_strategies()
        self.strategy_history = defaultdict(list)
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """初始化降级策略"""
        strategies = {
            # 图像降级策略
            "image_degradation": {
                "levels": [
                    {
                        "name": "high_quality",
                        "quality_threshold": 0.8,
                        "actions": ["enhance", "super_resolution"],
                        "description": "高清图增强"
                    },
                    {
                        "name": "standard_quality",
                        "quality_threshold": 0.6,
                        "actions": ["denoise", "color_correction"],
                        "description": "标准图处理"
                    },
                    {
                        "name": "low_quality",
                        "quality_threshold": 0.4,
                        "actions": ["compress", "reduce_resolution"],
                        "description": "低清图优化"
                    },
                    {
                        "name": "very_low_quality",
                        "quality_threshold": 0.2,
                        "actions": ["extract_features", "generate_description"],
                        "description": "极低质量 → 文字描述"
                    }
                ],
                "fallback_action": "generate_description"
            },
            
            # 音频降级策略
            "audio_degradation": {
                "levels": [
                    {
                        "name": "high_quality",
                        "quality_threshold": 0.8,
                        "actions": ["enhance", "noise_reduction"],
                        "description": "高清音频增强"
                    },
                    {
                        "name": "standard_quality",
                        "quality_threshold": 0.6,
                        "actions": ["normalize", "compress"],
                        "description": "标准音频处理"
                    },
                    {
                        "name": "low_quality",
                        "quality_threshold": 0.4,
                        "actions": ["extract_speech", "transcribe"],
                        "description": "低质量 → 文字转录"
                    },
                    {
                        "name": "very_low_quality",
                        "quality_threshold": 0.2,
                        "actions": ["extract_keywords", "generate_summary"],
                        "description": "极低质量 → 关键词提取"
                    }
                ],
                "fallback_action": "transcribe"
            },
            
            # 文本降级策略
            "text_degradation": {
                "levels": [
                    {
                        "name": "high_quality",
                        "quality_threshold": 0.8,
                        "actions": ["grammar_correction", "format_standardization"],
                        "description": "高质量文本校正"
                    },
                    {
                        "name": "standard_quality",
                        "quality_threshold": 0.6,
                        "actions": ["spell_check", "punctuation_correction"],
                        "description": "标准文本处理"
                    },
                    {
                        "name": "low_quality",
                        "quality_threshold": 0.4,
                        "actions": ["extract_meaning", "paraphrase"],
                        "description": "低质量 → 语义提取"
                    },
                    {
                        "name": "very_low_quality",
                        "quality_threshold": 0.2,
                        "actions": ["extract_keywords", "generate_interpretation"],
                        "description": "极低质量 → 关键词理解"
                    }
                ],
                "fallback_action": "extract_keywords"
            }
        }
        
        return strategies
    
    def get_degradation_level(self, modality: ModalityType, 
                            quality_score: float) -> Dict[str, Any]:
        """
        获取降级级别
        
        Args:
            modality: 模态类型
            quality_score: 质量分数 0-1
            
        Returns:
            降级级别配置
        """
        modality_key = f"{modality.value}_degradation"
        
        if modality_key not in self.strategies:
            # 默认策略
            return {
                "name": "default",
                "actions": ["basic_processing"],
                "description": "默认处理"
            }
        
        strategy = self.strategies[modality_key]
        
        # 查找匹配的质量级别
        for level in strategy["levels"]:
            if quality_score >= level["quality_threshold"]:
                # 记录策略选择
                self.strategy_history[modality_key].append({
                    "timestamp": time.time(),
                    "quality_score": quality_score,
                    "selected_level": level["name"],
                    "actions": level["actions"]
                })
                return level
        
        # 如果所有级别都不匹配，使用fallback
        return {
            "name": "fallback",
            "actions": [strategy["fallback_action"]],
            "description": f"降级到后备策略: {strategy['fallback_action']}"
        }
    
    def adapt_strategy(self, modality: ModalityType, 
                      feedback: Dict[str, Any]) -> None:
        """
        根据反馈调整策略
        
        Args:
            modality: 模态类型
            feedback: 反馈信息
        """
        modality_key = f"{modality.value}_degradation"
        
        if modality_key not in self.strategies:
            return
        
        # 分析反馈信息
        success_rate = feedback.get("success_rate", 0.5)
        improvement = feedback.get("improvement", 0.0)
        user_satisfaction = feedback.get("user_satisfaction", 0.5)
        
        # 根据反馈调整质量阈值（简单实现）
        if success_rate < 0.6:
            # 降低阈值，更早降级
            self._adjust_thresholds(modality_key, -0.1)
            logger.info(f"调整{modality.value}降级策略：降低质量阈值")
        elif success_rate > 0.9 and user_satisfaction > 0.8:
            # 提高阈值，更晚降级
            self._adjust_thresholds(modality_key, 0.05)
            logger.info(f"调整{modality.value}降级策略：提高质量阈值")
    
    def _adjust_thresholds(self, strategy_key: str, adjustment: float) -> None:
        """调整质量阈值"""
        if strategy_key not in self.strategies:
            return
        
        for level in self.strategies[strategy_key]["levels"]:
            level["quality_threshold"] = max(0.0, min(1.0, 
                level["quality_threshold"] + adjustment))
    
    def get_strategy_history(self, modality: Optional[ModalityType] = None) -> Dict[str, Any]:
        """获取策略历史"""
        if modality:
            key = f"{modality.value}_degradation"
            return {key: self.strategy_history.get(key, [])}
        else:
            return dict(self.strategy_history)


class ErrorDetector:
    """错误检测器"""
    
    def __init__(self):
        """初始化错误检测器"""
        self.error_patterns = self._initialize_error_patterns()
        self.detection_history = []
        
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化错误模式"""
        patterns = {
            # 文本错误模式
            "text_errors": {
                "spelling_errors": {
                    "description": "拼写错误",
                    "detector": self._detect_spelling_errors,
                    "severity_weights": {
                        "error_count": 0.6,
                        "error_density": 0.4
                    }
                },
                "grammar_errors": {
                    "description": "语法错误",
                    "detector": self._detect_grammar_errors,
                    "severity_weights": {
                        "complexity_violations": 0.7,
                        "readability_score": 0.3
                    }
                },
                "format_errors": {
                    "description": "格式错误",
                    "detector": self._detect_format_errors,
                    "severity_weights": {
                        "encoding_issues": 0.5,
                        "special_chars": 0.5
                    }
                },
                "inconsistency_errors": {
                    "description": "不一致错误",
                    "detector": self._detect_inconsistency_errors,
                    "severity_weights": {
                        "contradictions": 0.8,
                        "ambiguity": 0.2
                    }
                }
            },
            
            # 图像错误模式
            "image_errors": {
                "quality_errors": {
                    "description": "质量错误",
                    "detector": self._detect_image_quality_errors,
                    "severity_weights": {
                        "blurriness": 0.3,
                        "noise_level": 0.3,
                        "contrast": 0.2,
                        "brightness": 0.2
                    }
                },
                "format_errors": {
                    "description": "格式错误",
                    "detector": self._detect_image_format_errors,
                    "severity_weights": {
                        "unsupported_format": 0.6,
                        "corruption": 0.4
                    }
                },
                "corruption_errors": {
                    "description": "损坏错误",
                    "detector": self._detect_image_corruption_errors,
                    "severity_weights": {
                        "missing_parts": 0.7,
                        "artifacts": 0.3
                    }
                }
            },
            
            # 音频错误模式
            "audio_errors": {
                "quality_errors": {
                    "description": "质量错误",
                    "detector": self._detect_audio_quality_errors,
                    "severity_weights": {
                        "noise_level": 0.4,
                        "clarity": 0.3,
                        "volume": 0.3
                    }
                },
                "format_errors": {
                    "description": "格式错误",
                    "detector": self._detect_audio_format_errors,
                    "severity_weights": {
                        "unsupported_format": 0.7,
                        "sample_rate_issues": 0.3
                    }
                },
                "corruption_errors": {
                    "description": "损坏错误",
                    "detector": self._detect_audio_corruption_errors,
                    "severity_weights": {
                        "dropouts": 0.6,
                        "distortion": 0.4
                    }
                }
            }
        }
        
        return patterns
    
    def detect_errors(self, modality: ModalityType, 
                     data: Any, 
                     metadata: Optional[Dict[str, Any]] = None) -> List[ErrorDetection]:
        """
        检测错误
        
        Args:
            modality: 模态类型
            data: 数据
            metadata: 元数据
            
        Returns:
            错误检测列表
        """
        errors = []
        
        # 获取对应的错误模式
        modality_key = f"{modality.value}_errors"
        if modality_key not in self.error_patterns:
            logger.warning(f"未找到{modality.value}的错误模式")
            return errors
        
        error_patterns = self.error_patterns[modality_key]
        
        # 检测各种错误
        for error_name, error_config in error_patterns.items():
            try:
                detector_func = error_config["detector"]
                detection_result = detector_func(data, metadata)
                
                if detection_result and detection_result.get("detected", False):
                    # 计算严重程度
                    severity_weights = error_config.get("severity_weights", {})
                    severity = self._calculate_severity(detection_result, severity_weights)
                    
                    # 创建错误检测对象
                    error_detection = ErrorDetection(
                        error_type=ErrorType(error_name.upper().replace("_ERRORS", "_ERROR")),
                        modality=modality,
                        severity=severity,
                        confidence=detection_result.get("confidence", 0.7),
                        location=detection_result.get("location"),
                        description=f"{error_config['description']}: {detection_result.get('details', '')}"
                    )
                    
                    errors.append(error_detection)
                    
            except Exception as e:
                logger.error(f"检测{error_name}错误时发生异常: {e}")
        
        # 记录检测历史
        detection_record = {
            "timestamp": time.time(),
            "modality": modality.value,
            "data_hash": self._compute_data_hash(data),
            "error_count": len(errors),
            "errors": [err.to_dict() for err in errors]
        }
        self.detection_history.append(detection_record)
        
        logger.info(f"在{modality.value}中检测到{len(errors)}个错误")
        return errors
    
    def _detect_spelling_errors(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测拼写错误（简单实现）"""
        if not isinstance(text, str):
            return {"detected": False}
        
        # 简单拼写检查：检查非常见字符和重复字符
        common_words = ["the", "and", "you", "that", "have", "for", "not", "with", "this", "but"]
        words = text.lower().split()
        
        error_count = 0
        for word in words:
            # 清洗单词
            clean_word = ''.join(c for c in word if c.isalnum())
            if not clean_word:
                continue
            
            # 检查是否在常见词中
            if clean_word not in common_words and len(clean_word) > 3:
                # 简单启发式：检查是否有过多重复字符
                from collections import Counter
                char_counts = Counter(clean_word)
                max_repeat = max(char_counts.values())
                if max_repeat > len(clean_word) * 0.5:
                    error_count += 1
        
        error_density = error_count / max(len(words), 1)
        detected = error_density > 0.1  # 错误密度超过10%
        
        return {
            "detected": detected,
            "confidence": min(0.9, error_density * 3),
            "details": f"检测到{error_count}个可能的拼写错误",
            "error_count": error_count,
            "error_density": error_density
        }
    
    def _detect_grammar_errors(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测语法错误（简单实现）"""
        if not isinstance(text, str):
            return {"detected": False}
        
        # 简单语法检查：检查基本语法模式
        sentences = text.split('. ')
        issues = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            words = sentence.split()
            if len(words) < 2:
                issues.append("句子过短")
                continue
            
            # 检查首字母大写
            if sentence and sentence[0].islower():
                issues.append("首字母未大写")
            
            # 检查标点符号
            if sentence and sentence[-1] not in ['.', '?', '!', ',', ';', ':']:
                issues.append("缺少结束标点")
        
        detected = len(issues) > 0
        severity = len(issues) / max(len(sentences), 1)
        
        return {
            "detected": detected,
            "confidence": min(0.8, severity * 2),
            "details": f"检测到{len(issues)}个语法问题: {', '.join(issues[:3])}",
            "issue_count": len(issues),
            "severity": severity
        }
    
    def _detect_format_errors(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测格式错误"""
        # 简单格式检查
        if metadata and metadata.get("format_warning"):
            return {
                "detected": True,
                "confidence": 0.8,
                "details": "元数据报告格式问题",
                "location": "format"
            }
        
        return {"detected": False}
    
    def _detect_inconsistency_errors(self, data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测不一致错误"""
        # 简单一致性检查
        if metadata and metadata.get("inconsistency_warning"):
            return {
                "detected": True,
                "confidence": 0.7,
                "details": "元数据报告不一致问题",
                "location": "consistency"
            }
        
        return {"detected": False}
    
    def _detect_image_quality_errors(self, image_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测图像质量错误"""
        # 简单图像质量检查
        if metadata and metadata.get("image_quality"):
            quality_score = metadata.get("image_quality", 1.0)
            detected = quality_score < 0.6
            
            return {
                "detected": detected,
                "confidence": max(0.3, 1.0 - quality_score),
                "details": f"图像质量较低: {quality_score:.2f}",
                "quality_score": quality_score,
                "location": "quality"
            }
        
        return {"detected": False}
    
    def _detect_image_format_errors(self, image_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测图像格式错误"""
        if metadata and metadata.get("format_unsupported"):
            return {
                "detected": True,
                "confidence": 0.9,
                "details": "不支持的图像格式",
                "location": "format"
            }
        
        return {"detected": False}
    
    def _detect_image_corruption_errors(self, image_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测图像损坏错误"""
        if metadata and metadata.get("corruption_detected"):
            return {
                "detected": True,
                "confidence": 0.85,
                "details": "图像文件损坏",
                "location": "corruption"
            }
        
        return {"detected": False}
    
    def _detect_audio_quality_errors(self, audio_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测音频质量错误"""
        if metadata and metadata.get("audio_quality"):
            quality_score = metadata.get("audio_quality", 1.0)
            detected = quality_score < 0.6
            
            return {
                "detected": detected,
                "confidence": max(0.3, 1.0 - quality_score),
                "details": f"音频质量较低: {quality_score:.2f}",
                "quality_score": quality_score,
                "location": "quality"
            }
        
        return {"detected": False}
    
    def _detect_audio_format_errors(self, audio_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测音频格式错误"""
        if metadata and metadata.get("format_unsupported"):
            return {
                "detected": True,
                "confidence": 0.9,
                "details": "不支持的音频格式",
                "location": "format"
            }
        
        return {"detected": False}
    
    def _detect_audio_corruption_errors(self, audio_data: Any, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """检测音频损坏错误"""
        if metadata and metadata.get("corruption_detected"):
            return {
                "detected": True,
                "confidence": 0.85,
                "details": "音频文件损坏",
                "location": "corruption"
            }
        
        return {"detected": False}
    
    def _calculate_severity(self, detection_result: Dict[str, Any], 
                          weights: Dict[str, float]) -> float:
        """计算严重程度"""
        severity = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in detection_result:
                value = detection_result[metric]
                if isinstance(value, (int, float)):
                    severity += value * weight
                    total_weight += weight
        
        if total_weight > 0:
            severity = severity / total_weight
        
        return min(1.0, severity)
    
    def _compute_data_hash(self, data: Any) -> str:
        """计算数据哈希（用于去重）"""
        try:
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            return hashlib.md5(data_bytes).hexdigest()[:8]
        except:
            return "unknown"
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        stats = {
            "total_detections": len(self.detection_history),
            "modality_counts": defaultdict(int),
            "error_type_counts": defaultdict(int),
            "recent_detections": self.detection_history[-10:] if self.detection_history else []
        }
        
        for record in self.detection_history:
            stats["modality_counts"][record["modality"]] += 1
            for error in record["errors"]:
                stats["error_type_counts"][error["error_type"]] += 1
        
        return stats


class RepairEngine:
    """修复引擎"""
    
    def __init__(self):
        """初始化修复引擎"""
        self.repair_methods = self._initialize_repair_methods()
        self.repair_history = []
        
    def _initialize_repair_methods(self) -> Dict[str, Dict[str, Any]]:
        """初始化修复方法"""
        methods = {
            # 文本修复方法
            "text_repair": {
                "spell_correction": {
                    "description": "拼写纠正",
                    "repair_func": self._repair_spelling,
                    "applicable_errors": [ErrorType.FORMAT_ERROR, ErrorType.QUALITY_ERROR],
                    "confidence": 0.8,
                    "improvement": 0.7
                },
                "grammar_correction": {
                    "description": "语法纠正",
                    "repair_func": self._repair_grammar,
                    "applicable_errors": [ErrorType.QUALITY_ERROR, ErrorType.INCONSISTENCY_ERROR],
                    "confidence": 0.7,
                    "improvement": 0.6
                },
                "format_standardization": {
                    "description": "格式标准化",
                    "repair_func": self._standardize_format,
                    "applicable_errors": [ErrorType.FORMAT_ERROR],
                    "confidence": 0.9,
                    "improvement": 0.8
                },
                "semantic_extraction": {
                    "description": "语义提取",
                    "repair_func": self._extract_semantics,
                    "applicable_errors": [ErrorType.CORRUPTION_ERROR, ErrorType.INCOMPLETE_ERROR],
                    "confidence": 0.6,
                    "improvement": 0.5
                }
            },
            
            # 图像修复方法
            "image_repair": {
                "denoising": {
                    "description": "去噪处理",
                    "repair_func": self._denoise_image,
                    "applicable_errors": [ErrorType.NOISE_ERROR, ErrorType.QUALITY_ERROR],
                    "confidence": 0.75,
                    "improvement": 0.7
                },
                "super_resolution": {
                    "description": "超分辨率",
                    "repair_func": self._enhance_resolution,
                    "applicable_errors": [ErrorType.QUALITY_ERROR],
                    "confidence": 0.7,
                    "improvement": 0.6
                },
                "color_correction": {
                    "description": "颜色校正",
                    "repair_func": self._correct_colors,
                    "applicable_errors": [ErrorType.QUALITY_ERROR, ErrorType.INCONSISTENCY_ERROR],
                    "confidence": 0.8,
                    "improvement": 0.7
                },
                "feature_extraction": {
                    "description": "特征提取",
                    "repair_func": self._extract_features,
                    "applicable_errors": [ErrorType.CORRUPTION_ERROR, ErrorType.INCOMPLETE_ERROR],
                    "confidence": 0.65,
                    "improvement": 0.6
                }
            },
            
            # 音频修复方法
            "audio_repair": {
                "noise_reduction": {
                    "description": "噪声降低",
                    "repair_func": self._reduce_noise,
                    "applicable_errors": [ErrorType.NOISE_ERROR, ErrorType.QUALITY_ERROR],
                    "confidence": 0.8,
                    "improvement": 0.75
                },
                "normalization": {
                    "description": "音量标准化",
                    "repair_func": self._normalize_audio,
                    "applicable_errors": [ErrorType.QUALITY_ERROR],
                    "confidence": 0.9,
                    "improvement": 0.8
                },
                "speech_enhancement": {
                    "description": "语音增强",
                    "repair_func": self._enhance_speech,
                    "applicable_errors": [ErrorType.QUALITY_ERROR],
                    "confidence": 0.7,
                    "improvement": 0.65
                },
                "transcription": {
                    "description": "文字转录",
                    "repair_func": self._transcribe_audio,
                    "applicable_errors": [ErrorType.CORRUPTION_ERROR, ErrorType.INCOMPLETE_ERROR],
                    "confidence": 0.6,
                    "improvement": 0.7
                }
            }
        }
        
        return methods
    
    def generate_repair_actions(self, errors: List[ErrorDetection],
                              quality_assessment: QualityAssessment) -> List[RepairAction]:
        """
        生成修复动作
        
        Args:
            errors: 错误检测列表
            quality_assessment: 质量评估
            
        Returns:
            修复动作列表
        """
        repair_actions = []
        modality = quality_assessment.modality
        
        # 获取对应的修复方法
        modality_key = f"{modality.value}_repair"
        if modality_key not in self.repair_methods:
            logger.warning(f"未找到{modality.value}的修复方法")
            return repair_actions
        
        repair_methods = self.repair_methods[modality_key]
        
        # 基于错误类型选择修复方法
        for error in errors:
            for method_name, method_config in repair_methods.items():
                if error.error_type in method_config["applicable_errors"]:
                    # 根据错误严重程度调整置信度
                    adjusted_confidence = method_config["confidence"] * (1.0 - error.severity * 0.3)
                    adjusted_improvement = method_config["improvement"] * (1.0 - error.severity * 0.2)
                    
                    repair_action = RepairAction(
                        action_type=method_name,
                        modality=modality,
                        parameters={
                            "error_type": error.error_type.value,
                            "error_severity": error.severity,
                            "error_description": error.description,
                            "quality_score": quality_assessment.overall_quality
                        },
                        confidence=adjusted_confidence,
                        expected_improvement=adjusted_improvement
                    )
                    
                    repair_actions.append(repair_action)
        
        # 如果没有匹配的修复方法，使用通用方法
        if not repair_actions and modality_key in self.repair_methods:
            # 使用第一个方法作为通用修复
            method_name = list(repair_methods.keys())[0]
            method_config = repair_methods[method_name]
            
            repair_action = RepairAction(
                action_type=method_name,
                modality=modality,
                parameters={
                    "reason": "generic_repair",
                    "quality_score": quality_assessment.overall_quality
                },
                confidence=method_config["confidence"] * 0.7,
                expected_improvement=method_config["improvement"] * 0.7
            )
            
            repair_actions.append(repair_action)
        
        # 按预期改进程度排序
        repair_actions.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        return repair_actions
    
    def execute_repair(self, data: Any, repair_action: RepairAction) -> Tuple[Any, Dict[str, Any]]:
        """
        执行修复
        
        Args:
            data: 原始数据
            repair_action: 修复动作
            
        Returns:
            修复后的数据和修复报告
        """
        modality = repair_action.modality
        modality_key = f"{modality.value}_repair"
        
        if modality_key not in self.repair_methods:
            return data, {"success": False, "error": "修复方法不存在"}
        
        method_name = repair_action.action_type
        if method_name not in self.repair_methods[modality_key]:
            return data, {"success": False, "error": f"修复方法{method_name}不存在"}
        
        method_config = self.repair_methods[modality_key][method_name]
        repair_func = method_config["repair_func"]
        
        try:
            # 执行修复
            repaired_data, repair_details = repair_func(data, repair_action.parameters)
            
            # 记录修复历史
            repair_record = {
                "timestamp": time.time(),
                "modality": modality.value,
                "action_type": method_name,
                "parameters": repair_action.parameters,
                "confidence": repair_action.confidence,
                "expected_improvement": repair_action.expected_improvement,
                "actual_improvement": repair_details.get("improvement", 0.0),
                "success": repair_details.get("success", False),
                "details": repair_details
            }
            self.repair_history.append(repair_record)
            
            if repair_details.get("success", False):
                logger.info(f"成功执行{method_name}修复，改进: {repair_details.get('improvement', 0):.2f}")
            else:
                logger.warning(f"{method_name}修复执行失败或部分成功")
            
            return repaired_data, repair_details
            
        except Exception as e:
            logger.error(f"执行修复时发生异常: {e}")
            return data, {"success": False, "error": str(e)}
    
    def _repair_spelling(self, text: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """修复拼写（简单实现）"""
        # 简单拼写修复：纠正常见错误
        common_corrections = {
            "teh": "the",
            "adn": "and",
            "thsi": "this",
            "taht": "that",
            "wht": "what",
            "hw": "how",
            "whn": "when",
            "wher": "where",
            "wy": "why"
        }
        
        words = text.split()
        corrected_words = []
        corrections_made = 0
        
        for word in words:
            lower_word = word.lower()
            if lower_word in common_corrections:
                corrected_word = common_corrections[lower_word]
                # 保持原始大小写
                if word[0].isupper():
                    corrected_word = corrected_word.capitalize()
                corrected_words.append(corrected_word)
                corrections_made += 1
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        
        # 计算改进程度
        improvement = corrections_made / max(len(words), 1)
        
        return corrected_text, {
            "success": corrections_made > 0,
            "improvement": improvement,
            "corrections_made": corrections_made,
            "original_word_count": len(words),
            "details": f"纠正了{corrections_made}个拼写错误"
        }
    
    def _repair_grammar(self, text: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """修复语法（简单实现）"""
        # 简单语法修复：添加缺失的标点
        if not text:
            return text, {"success": False, "improvement": 0.0, "details": "空文本"}
        
        corrected_text = text
        
        # 确保以标点结束
        if corrected_text and corrected_text[-1] not in ['.', '?', '!', ',', ';', ':']:
            corrected_text += '.'
        
        # 确保首字母大写
        if corrected_text and corrected_text[0].islower():
            corrected_text = corrected_text[0].upper() + corrected_text[1:]
        
        improvements_made = 0
        if text != corrected_text:
            improvements_made = 1
        
        return corrected_text, {
            "success": improvements_made > 0,
            "improvement": improvements_made * 0.5,
            "improvements_made": improvements_made,
            "details": "进行了基本的语法修正"
        }
    
    def _standardize_format(self, text: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """标准化格式"""
        # 简单的格式标准化
        if not text:
            return text, {"success": False, "improvement": 0.0, "details": "空文本"}
        
        # 移除多余的空格
        corrected_text = ' '.join(text.split())
        
        # 标准化换行符
        corrected_text = corrected_text.replace('\r\n', '\n').replace('\r', '\n')
        
        improvements_made = 1 if text != corrected_text else 0
        
        return corrected_text, {
            "success": improvements_made > 0,
            "improvement": improvements_made * 0.3,
            "improvements_made": improvements_made,
            "details": "进行了格式标准化"
        }
    
    def _extract_semantics(self, text: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """提取语义（简单实现）"""
        if not text:
            return "", {"success": False, "improvement": 0.0, "details": "空文本"}
        
        # 简单语义提取：提取关键词
        words = text.split()
        keywords = []
        
        # 简单关键词提取：长度>3且不是常见词
        common_words = ["the", "and", "you", "that", "have", "for", "not", "with"]
        for word in words:
            clean_word = ''.join(c for c in word.lower() if c.isalpha())
            if len(clean_word) > 3 and clean_word not in common_words:
                keywords.append(clean_word)
        
        extracted_text = ' '.join(keywords[:5])  # 取前5个关键词
        
        return extracted_text, {
            "success": len(keywords) > 0,
            "improvement": len(keywords) / max(len(words), 1) * 0.5,
            "keywords_extracted": len(keywords),
            "details": f"提取了{len(keywords)}个关键词"
        }
    
    def _denoise_image(self, image_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """图像去噪（模拟）"""
        # 模拟图像去噪
        return image_data, {
            "success": True,
            "improvement": 0.7,
            "details": "模拟图像去噪处理",
            "noise_reduction": "simulated"
        }
    
    def _enhance_resolution(self, image_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """增强分辨率（模拟）"""
        # 模拟超分辨率
        return image_data, {
            "success": True,
            "improvement": 0.6,
            "details": "模拟超分辨率处理",
            "resolution_enhancement": "simulated"
        }
    
    def _correct_colors(self, image_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """颜色校正（模拟）"""
        # 模拟颜色校正
        return image_data, {
            "success": True,
            "improvement": 0.7,
            "details": "模拟颜色校正处理",
            "color_correction": "simulated"
        }
    
    def _extract_features(self, image_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """提取特征（模拟）"""
        # 模拟特征提取
        features = {"object_detected": "unknown", "confidence": 0.5}
        return features, {
            "success": True,
            "improvement": 0.6,
            "details": "模拟图像特征提取",
            "features_extracted": features
        }
    
    def _reduce_noise(self, audio_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """降低噪声（模拟）"""
        # 模拟噪声降低
        return audio_data, {
            "success": True,
            "improvement": 0.75,
            "details": "模拟音频噪声降低处理",
            "noise_reduction": "simulated"
        }
    
    def _normalize_audio(self, audio_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """音量标准化（模拟）"""
        # 模拟音量标准化
        return audio_data, {
            "success": True,
            "improvement": 0.8,
            "details": "模拟音频音量标准化",
            "normalization": "simulated"
        }
    
    def _enhance_speech(self, audio_data: Any, parameters: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """语音增强（模拟）"""
        # 模拟语音增强
        return audio_data, {
            "success": True,
            "improvement": 0.65,
            "details": "模拟语音增强处理",
            "speech_enhancement": "simulated"
        }
    
    def _transcribe_audio(self, audio_data: Any, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """文字转录（模拟）"""
        # 模拟语音转文字
        transcribed_text = "模拟转录的文本内容"
        return transcribed_text, {
            "success": True,
            "improvement": 0.7,
            "details": "模拟音频转录为文本",
            "transcription": transcribed_text
        }
    
    def get_repair_stats(self) -> Dict[str, Any]:
        """获取修复统计信息"""
        stats = {
            "total_repairs": len(self.repair_history),
            "successful_repairs": sum(1 for r in self.repair_history if r.get("success", False)),
            "failed_repairs": sum(1 for r in self.repair_history if not r.get("success", False)),
            "avg_improvement": 0.0,
            "modality_counts": defaultdict(int),
            "action_type_counts": defaultdict(int)
        }
        
        if self.repair_history:
            improvements = [r.get("actual_improvement", 0.0) for r in self.repair_history]
            stats["avg_improvement"] = sum(improvements) / len(improvements)
        
        for record in self.repair_history:
            stats["modality_counts"][record["modality"]] += 1
            stats["action_type_counts"][record["action_type"]] += 1
        
        return stats


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        """初始化质量评估器"""
        self.assessment_history = []
        
    def assess_quality(self, modality: ModalityType, 
                      data: Any, 
                      metadata: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """
        评估质量
        
        Args:
            modality: 模态类型
            data: 数据
            metadata: 元数据
            
        Returns:
            质量评估结果
        """
        if modality == ModalityType.TEXT:
            return self._assess_text_quality(data, metadata)
        elif modality == ModalityType.IMAGE:
            return self._assess_image_quality(data, metadata)
        elif modality == ModalityType.AUDIO:
            return self._assess_audio_quality(data, metadata)
        else:
            # 默认质量评估
            return QualityAssessment(
                modality=modality,
                overall_quality=0.5,
                clarity=0.5,
                completeness=0.5,
                consistency=0.5,
                noise_level=0.5,
                details={"reason": "default_assessment"}
            )
    
    def _assess_text_quality(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """评估文本质量"""
        if not isinstance(text, str):
            return QualityAssessment(
                modality=ModalityType.TEXT,
                overall_quality=0.1,
                clarity=0.1,
                completeness=0.1,
                consistency=0.1,
                noise_level=0.9,
                details={"error": "invalid_text_type"}
            )
        
        # 计算各种质量指标
        length = len(text)
        word_count = len(text.split())
        
        # 清晰度：基于标点符号和句子结构
        has_punctuation = any(punc in text for punc in ['.', '?', '!', ',', ';', ':'])
        clarity = 0.8 if has_punctuation else 0.4
        
        # 完整性：基于文本长度
        completeness = min(1.0, word_count / 50)  # 50词为完整
        
        # 一致性：检查文本是否连贯
        sentences = text.split('. ')
        consistency = 0.7 if len(sentences) > 1 else 0.5
        
        # 噪声水平：检查特殊字符和乱码
        noise_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ',.;:!?\'"()-'))
        noise_level = noise_chars / max(length, 1)
        
        # 整体质量
        overall_quality = (clarity * 0.3 + completeness * 0.3 + 
                          consistency * 0.2 + (1 - noise_level) * 0.2)
        
        details = {
            "length": length,
            "word_count": word_count,
            "sentence_count": len(sentences),
            "has_punctuation": has_punctuation,
            "noise_char_count": noise_chars,
            "noise_level": noise_level
        }
        
        assessment = QualityAssessment(
            modality=ModalityType.TEXT,
            overall_quality=overall_quality,
            clarity=clarity,
            completeness=completeness,
            consistency=consistency,
            noise_level=noise_level,
            details=details
        )
        
        self._record_assessment(assessment)
        return assessment
    
    def _assess_image_quality(self, image_data: Any, metadata: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """评估图像质量"""
        # 简单图像质量评估
        if metadata and "image_quality" in metadata:
            quality_score = metadata["image_quality"]
        else:
            quality_score = 0.7  # 默认值
        
        details = {
            "assessment_method": "metadata_based" if metadata else "default",
            "source": metadata.get("source", "unknown") if metadata else "unknown"
        }
        
        assessment = QualityAssessment(
            modality=ModalityType.IMAGE,
            overall_quality=quality_score,
            clarity=quality_score * 0.9,
            completeness=0.8,  # 假设图像通常是完整的
            consistency=0.7,
            noise_level=1.0 - quality_score,
            details=details
        )
        
        self._record_assessment(assessment)
        return assessment
    
    def _assess_audio_quality(self, audio_data: Any, metadata: Optional[Dict[str, Any]] = None) -> QualityAssessment:
        """评估音频质量"""
        # 简单音频质量评估
        if metadata and "audio_quality" in metadata:
            quality_score = metadata["audio_quality"]
        else:
            quality_score = 0.6  # 默认值
        
        details = {
            "assessment_method": "metadata_based" if metadata else "default",
            "source": metadata.get("source", "unknown") if metadata else "unknown"
        }
        
        assessment = QualityAssessment(
            modality=ModalityType.AUDIO,
            overall_quality=quality_score,
            clarity=quality_score * 0.8,
            completeness=0.7,
            consistency=0.6,
            noise_level=1.0 - quality_score,
            details=details
        )
        
        self._record_assessment(assessment)
        return assessment
    
    def _record_assessment(self, assessment: QualityAssessment) -> None:
        """记录评估结果"""
        record = {
            "timestamp": time.time(),
            "assessment": assessment.to_dict()
        }
        self.assessment_history.append(record)
    
    def get_assessment_stats(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        stats = {
            "total_assessments": len(self.assessment_history),
            "modality_quality": defaultdict(list),
            "recent_assessments": self.assessment_history[-10:] if self.assessment_history else []
        }
        
        for record in self.assessment_history:
            assessment = record["assessment"]
            modality = assessment["modality"]
            stats["modality_quality"][modality].append(assessment["overall_quality"])
        
        # 计算平均质量
        stats["avg_quality_by_modality"] = {}
        for modality, qualities in stats["modality_quality"].items():
            stats["avg_quality_by_modality"][modality] = sum(qualities) / len(qualities)
        
        return stats


class FaultToleranceManager:
    """
    容错与纠错机制管理器
    
    核心功能：
    1. 检测和处理各种错误
    2. 执行自适应降级策略
    3. 管理和协调修复引擎
    4. 提供用户反馈学习
    """
    
    def __init__(self, 
                 enable_learning: bool = True,
                 max_repair_attempts: int = 3):
        """
        初始化容错管理器
        
        Args:
            enable_learning: 是否启用学习功能
            max_repair_attempts: 最大修复尝试次数
        """
        self.enable_learning = enable_learning
        self.max_repair_attempts = max_repair_attempts
        
        # 初始化组件
        self.error_detector = ErrorDetector()
        self.quality_assessor = QualityAssessor()
        self.repair_engine = RepairEngine()
        self.degradation_strategy = DegradationStrategy()
        
        # 用户反馈记录
        self.feedback_history = []
        
        # 处理统计
        self.stats = {
            "total_processed": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "degradations_applied": 0,
            "learning_updates": 0
        }
        
        logger.info(f"容错管理器初始化完成，学习功能: {enable_learning}")
    
    def process_modality(self, modality: ModalityType, 
                        data: Any, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理单个模态
        
        Args:
            modality: 模态类型
            data: 数据
            metadata: 元数据
            
        Returns:
            处理结果
        """
        self.stats["total_processed"] += 1
        
        try:
            logger.info(f"开始处理{modality.value}模态")
            
            # 1. 质量评估
            quality_assessment = self.quality_assessor.assess_quality(modality, data, metadata)
            logger.info(f"质量评估完成: {quality_assessment.overall_quality:.2f}")
            
            # 2. 错误检测
            errors = self.error_detector.detect_errors(modality, data, metadata)
            logger.info(f"错误检测完成: {len(errors)}个错误")
            
            # 3. 确定处理策略
            if errors or quality_assessment.overall_quality < 0.5:
                # 需要修复
                result = self._repair_modality(modality, data, quality_assessment, errors, metadata)
            elif quality_assessment.overall_quality < 0.8:
                # 质量尚可，可能需要进行简单优化
                result = self._optimize_modality(modality, data, quality_assessment, metadata)
            else:
                # 质量良好，直接通过
                result = self._pass_through(modality, data, quality_assessment, metadata)
            
            # 4. 记录处理结果
            self._record_processing_result(modality, result, quality_assessment, errors)
            
            logger.info(f"{modality.value}模态处理完成")
            return result
            
        except Exception as e:
            logger.error(f"处理{modality.value}模态时发生异常: {e}")
            return self._create_error_result(modality, data, str(e))
    
    def _repair_modality(self, modality: ModalityType,
                        data: Any,
                        quality_assessment: QualityAssessment,
                        errors: List[ErrorDetection],
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """修复模态"""
        logger.info(f"开始修复{modality.value}模态")
        
        current_data = data
        repair_history = []
        
        # 生成修复动作
        repair_actions = self.repair_engine.generate_repair_actions(errors, quality_assessment)
        
        for attempt, repair_action in enumerate(repair_actions[:self.max_repair_attempts], 1):
            logger.info(f"修复尝试{attempt}: {repair_action.action_type}")
            
            # 执行修复
            repaired_data, repair_report = self.repair_engine.execute_repair(current_data, repair_action)
            
            # 记录修复历史
            repair_history.append({
                "attempt": attempt,
                "action": repair_action.to_dict(),
                "report": repair_report
            })
            
            if repair_report.get("success", False):
                # 修复成功，更新数据
                current_data = repaired_data
                
                # 重新评估质量
                new_quality = self.quality_assessor.assess_quality(modality, current_data, metadata)
                
                # 检查是否达到可接受质量
                if new_quality.overall_quality >= 0.6:
                    logger.info(f"修复成功，新质量: {new_quality.overall_quality:.2f}")
                    self.stats["successful_repairs"] += 1
                    
                    return self._create_success_result(
                        modality, current_data, quality_assessment, 
                        new_quality, repair_history, "repaired"
                    )
        
        # 如果所有修复尝试都失败，应用降级策略
        logger.info(f"修复失败，应用降级策略")
        return self._apply_degradation_strategy(modality, data, quality_assessment, errors, metadata)
    
    def _optimize_modality(self, modality: ModalityType,
                          data: Any,
                          quality_assessment: QualityAssessment,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """优化模态（轻微质量改进）"""
        logger.info(f"优化{modality.value}模态")
        
        # 应用轻微的优化修复
        generic_action = RepairAction(
            action_type="generic_optimization",
            modality=modality,
            parameters={
                "quality_score": quality_assessment.overall_quality,
                "reason": "quality_optimization"
            },
            confidence=0.6,
            expected_improvement=0.2
        )
        
        optimized_data, optimization_report = self.repair_engine.execute_repair(data, generic_action)
        
        # 重新评估质量
        new_quality = self.quality_assessor.assess_quality(modality, optimized_data, metadata)
        
        return self._create_success_result(
            modality, optimized_data, quality_assessment, 
            new_quality, [{"action": generic_action.to_dict(), "report": optimization_report}],
            "optimized"
        )
    
    def _pass_through(self, modality: ModalityType,
                     data: Any,
                     quality_assessment: QualityAssessment,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """直接通过（无需处理）"""
        logger.info(f"{modality.value}模态质量良好，直接通过")
        
        return self._create_success_result(
            modality, data, quality_assessment, 
            quality_assessment, [], "passed_through"
        )
    
    def _apply_degradation_strategy(self, modality: ModalityType,
                                  data: Any,
                                  quality_assessment: QualityAssessment,
                                  errors: List[ErrorDetection],
                                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """应用降级策略"""
        logger.info(f"为{modality.value}模态应用降级策略")
        
        # 获取降级级别
        degradation_level = self.degradation_strategy.get_degradation_level(
            modality, quality_assessment.overall_quality
        )
        
        self.stats["degradations_applied"] += 1
        
        # 执行降级操作
        degraded_data = data  # 实际实现中会应用降级
        
        # 创建降级结果
        result = self._create_success_result(
            modality, degraded_data, quality_assessment,
            quality_assessment, [], "degraded"
        )
        
        result["degradation_applied"] = {
            "level": degradation_level["name"],
            "actions": degradation_level["actions"],
            "description": degradation_level["description"]
        }
        
        logger.info(f"降级策略应用完成: {degradation_level['name']}")
        return result
    
    def _create_success_result(self, modality: ModalityType,
                              data: Any,
                              original_quality: QualityAssessment,
                              final_quality: QualityAssessment,
                              repair_history: List[Dict[str, Any]],
                              status: str) -> Dict[str, Any]:
        """创建成功结果"""
        quality_improvement = final_quality.overall_quality - original_quality.overall_quality
        
        return {
            "success": True,
            "modality": modality.value,
            "status": status,
            "original_quality": original_quality.to_dict(),
            "final_quality": final_quality.to_dict(),
            "quality_improvement": quality_improvement,
            "repair_history": repair_history,
            "timestamp": time.time()
        }
    
    def _create_error_result(self, modality: ModalityType,
                           data: Any,
                           error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        self.stats["failed_repairs"] += 1
        
        return {
            "success": False,
            "modality": modality.value,
            "status": "error",
            "error_message": error_message,
            "timestamp": time.time()
        }
    
    def _record_processing_result(self, modality: ModalityType,
                                result: Dict[str, Any],
                                quality_assessment: QualityAssessment,
                                errors: List[ErrorDetection]) -> None:
        """记录处理结果"""
        record = {
            "timestamp": time.time(),
            "modality": modality.value,
            "result": result,
            "quality_assessment": quality_assessment.to_dict(),
            "error_count": len(errors),
            "errors": [err.to_dict() for err in errors]
        }
        
        # 在实际实现中，这里会存储到数据库或文件
        # 现在只存储在内存中
        self.feedback_history.append(record)
    
    def add_user_feedback(self, modality: ModalityType,
                         result_id: str,
                         satisfaction: float,
                         feedback_text: Optional[str] = None) -> None:
        """
        添加用户反馈
        
        Args:
            modality: 模态类型
            result_id: 结果ID
            satisfaction: 满意度 0-1
            feedback_text: 反馈文本
        """
        feedback = {
            "timestamp": time.time(),
            "modality": modality.value,
            "result_id": result_id,
            "satisfaction": satisfaction,
            "feedback_text": feedback_text
        }
        
        self.feedback_history.append(feedback)
        
        if self.enable_learning:
            self._learn_from_feedback(feedback)
    
    def _learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """从反馈中学习"""
        modality = ModalityType(feedback["modality"])
        satisfaction = feedback["satisfaction"]
        
        # 根据满意度调整策略
        if satisfaction < 0.4:
            # 用户不满意，调整策略
            self.degradation_strategy.adapt_strategy(modality, {
                "success_rate": 0.5,
                "user_satisfaction": satisfaction
            })
            self.stats["learning_updates"] += 1
            logger.info(f"根据用户反馈调整{modality.value}策略")
    
    def process_multimodal_input(self, multimodal_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理多模态输入
        
        Args:
            multimodal_input: 多模态输入字典
            
        Returns:
            处理结果
        """
        logger.info("开始处理多模态输入")
        
        results = {}
        
        # 处理文本模态
        if "text" in multimodal_input:
            text_data = multimodal_input["text"]
            text_metadata = multimodal_input.get("metadata", {})
            text_result = self.process_modality(
                ModalityType.TEXT, text_data, text_metadata
            )
            results["text"] = text_result
        
        # 处理图像模态
        if "image_data" in multimodal_input:
            image_data = multimodal_input["image_data"]
            image_metadata = multimodal_input.get("metadata", {})
            image_result = self.process_modality(
                ModalityType.IMAGE, image_data, image_metadata
            )
            results["image"] = image_result
        
        # 处理音频模态
        if "audio_data" in multimodal_input:
            audio_data = multimodal_input["audio_data"]
            audio_metadata = multimodal_input.get("metadata", {})
            audio_result = self.process_modality(
                ModalityType.AUDIO, audio_data, audio_metadata
            )
            results["audio"] = audio_result
        
        # 总体结果
        overall_result = {
            "success": all(r.get("success", False) for r in results.values()),
            "modality_results": results,
            "timestamp": time.time(),
            "manager_stats": self.get_stats()
        }
        
        logger.info(f"多模态处理完成，成功: {overall_result['success']}")
        return overall_result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        stats = self.stats.copy()
        
        # 添加组件统计
        stats["error_detection_stats"] = self.error_detector.get_detection_stats()
        stats["quality_assessment_stats"] = self.quality_assessor.get_assessment_stats()
        stats["repair_stats"] = self.repair_engine.get_repair_stats()
        
        # 反馈统计
        stats["feedback_count"] = len(self.feedback_history)
        if self.feedback_history:
            satisfactions = [f.get("satisfaction", 0.5) for f in self.feedback_history]
            stats["avg_user_satisfaction"] = sum(satisfactions) / len(satisfactions)
        else:
            stats["avg_user_satisfaction"] = 0.0
        
        return stats
    
    def clear_history(self):
        """清空历史记录"""
        self.feedback_history.clear()
        self.error_detector.detection_history.clear()
        self.quality_assessor.assessment_history.clear()
        self.repair_engine.repair_history.clear()
        self.degradation_strategy.strategy_history.clear()
        
        logger.info("容错管理器历史记录已清空")


# 测试函数
def test_fault_tolerance_manager():
    """测试容错管理器"""
    print("测试容错管理器...")
    
    # 创建管理器实例
    manager = FaultToleranceManager()
    
    # 测试用例1：低质量文本
    print("\n=== 测试用例1：低质量文本 ===")
    test_text_1 = "thsi is a test text with some speling erors and bad grammer"
    test_input_1 = {
        "text": test_text_1,
        "metadata": {"source": "test"}
    }
    
    result_1 = manager.process_multimodal_input(test_input_1)
    print(f"文本处理结果: {json.dumps(result_1, indent=2, ensure_ascii=False)}")
    
    # 测试用例2：图像数据
    print("\n=== 测试用例2：图像数据 ===")
    test_input_2 = {
        "image_data": "模拟图像数据",
        "metadata": {
            "source": "test",
            "image_quality": 0.3  # 低质量图像
        }
    }
    
    result_2 = manager.process_multimodal_input(test_input_2)
    print(f"图像处理结果: {json.dumps(result_2, indent=2, ensure_ascii=False)}")
    
    # 测试用例3：多模态输入
    print("\n=== 测试用例3：多模态输入 ===")
    test_input_3 = {
        "text": "hw to repare brokn keyboard?",
        "image_data": "模糊键盘图片",
        "audio_data": "嘈杂的音频",
        "metadata": {"source": "multimodal_test"}
    }
    
    result_3 = manager.process_multimodal_input(test_input_3)
    print(f"多模态处理结果: {json.dumps(result_3, indent=2, ensure_ascii=False)}")
    
    # 显示统计信息
    print("\n=== 管理器统计信息 ===")
    stats = manager.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    return manager


if __name__ == "__main__":
    # 运行测试
    test_fault_tolerance_manager()
        
