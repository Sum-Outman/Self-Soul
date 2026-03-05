"""
混合模态解析器

解析和处理混合模态输入，如"模糊图片+嘈杂语音+补充文本"，
实现真正的多模态容错处理和意图理解。

核心功能：
1. 混合模态输入解析和分离
2. 模态质量评估和修复
3. 噪声和失真检测
4. 自适应预处理
5. 缺失信息推断
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hybrid_modal_parser")


class InputQuality(Enum):
    """输入质量等级"""
    EXCELLENT = "excellent"    # 优秀：清晰、完整、高质量
    GOOD = "good"              # 良好：轻微噪声或失真
    FAIR = "fair"              # 一般：中等噪声或部分信息缺失
    POOR = "poor"              # 差：严重噪声或重要信息缺失
    UNUSABLE = "unusable"      # 不可用：无法解析或损坏


class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


class HybridModalParser:
    """
    混合模态解析器
    
    解析和处理混合模态输入，支持质量评估、噪声检测和自适应修复。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化混合模态解析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 支持的模态类型
        self.supported_modalities = [modality.value for modality in ModalityType]
        
        # 质量评估器
        self.quality_assessors = self._initialize_quality_assessors()
        
        # 噪声检测器
        self.noise_detectors = self._initialize_noise_detectors()
        
        # 修复处理器
        self.repair_processors = self._initialize_repair_processors()
        
        # 统计信息
        self.stats = {
            "total_inputs": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "quality_distribution": {quality.value: 0 for quality in InputQuality},
            "repair_attempts": 0,
            "repair_successes": 0
        }
        
        logger.info("混合模态解析器初始化完成")
    
    def _initialize_quality_assessors(self) -> Dict[str, Any]:
        """初始化质量评估器"""
        # 这里应该实现各模态的质量评估器
        # 暂时返回空字典，实际实现中应该包含具体的评估器
        return {
            "text": self._assess_text_quality,
            "image": self._assess_image_quality,
            "audio": self._assess_audio_quality,
            "generic": self._assess_generic_quality
        }
    
    def _initialize_noise_detectors(self) -> Dict[str, Any]:
        """初始化噪声检测器"""
        return {
            "text": self._detect_text_noise,
            "image": self._detect_image_noise,
            "audio": self._detect_audio_noise
        }
    
    def _initialize_repair_processors(self) -> Dict[str, Any]:
        """初始化修复处理器"""
        return {
            "text": self._repair_text,
            "image": self._repair_image,
            "audio": self._repair_audio
        }
    
    def parse_hybrid_input(self, hybrid_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析混合模态输入
        
        Args:
            hybrid_input: 混合输入字典，例如:
                {
                    "text": "这是一个测试文本，包含模糊图片",
                    "image_data": "模糊的JPEG图像数据",
                    "audio_data": "嘈杂的音频数据",
                    "metadata": {"source": "user_upload"}
                }
            
        Returns:
            解析结果字典
        """
        self.stats["total_inputs"] += 1
        
        logger.info("开始解析混合模态输入...")
        
        try:
            # 1. 分离和识别模态
            modality_data = self._separate_modalities(hybrid_input)
            logger.info(f"识别到 {len(modality_data)} 个模态")
            
            # 2. 评估各模态质量
            quality_assessments = self._assess_modality_quality(modality_data)
            
            # 3. 检测噪声和问题
            noise_detections = self._detect_modality_noise(modality_data, quality_assessments)
            
            # 4. 应用修复处理
            repaired_data, repair_reports = self._repair_modalities(modality_data, quality_assessments, noise_detections)
            
            # 5. 推断缺失信息
            inferred_data = self._infer_missing_information(repaired_data, quality_assessments)
            
            # 6. 生成综合解析报告
            parse_report = self._generate_parse_report(
                modality_data, quality_assessments, noise_detections,
                repair_reports, inferred_data
            )
            
            # 7. 准备最终结果
            result = {
                "success": True,
                "parsed_modalities": list(repaired_data.keys()),
                "repaired_data": repaired_data,
                "quality_assessments": quality_assessments,
                "noise_detections": noise_detections,
                "repair_reports": repair_reports,
                "inferred_data": inferred_data,
                "parse_report": parse_report,
                "original_input_keys": list(hybrid_input.keys())
            }
            
            self.stats["successful_parses"] += 1
            logger.info("混合模态输入解析成功")
            
            return result
            
        except Exception as e:
            self.stats["failed_parses"] += 1
            logger.error(f"混合模态输入解析失败: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "parsed_modalities": [],
                "message": "混合模态输入解析失败"
            }
    
    def _separate_modalities(self, hybrid_input: Dict[str, Any]) -> Dict[str, Any]:
        """分离和识别混合输入中的不同模态"""
        modality_data = {}
        
        # 文本模态识别
        text_data = self._extract_text_data(hybrid_input)
        if text_data:
            modality_data[ModalityType.TEXT.value] = text_data
        
        # 图像模态识别
        image_data = self._extract_image_data(hybrid_input)
        if image_data:
            modality_data[ModalityType.IMAGE.value] = image_data
        
        # 音频模态识别
        audio_data = self._extract_audio_data(hybrid_input)
        if audio_data:
            modality_data[ModalityType.AUDIO.value] = audio_data
        
        # 其他模态识别
        other_data = self._extract_other_modalities(hybrid_input)
        modality_data.update(other_data)
        
        logger.debug(f"分离模态: {list(modality_data.keys())}")
        return modality_data
    
    def _extract_text_data(self, hybrid_input: Dict[str, Any]) -> Optional[Any]:
        """提取文本数据"""
        # 尝试不同的文本键
        text_keys = ["text", "text_data", "content", "message", "description", "caption"]
        
        for key in text_keys:
            if key in hybrid_input and hybrid_input[key]:
                data = hybrid_input[key]
                # 验证是否为文本数据
                if isinstance(data, str) and len(data.strip()) > 0:
                    logger.debug(f"从键 '{key}' 提取文本数据，长度: {len(data)}")
                    return data
        
        # 检查是否有嵌入在元数据中的文本
        if "metadata" in hybrid_input and isinstance(hybrid_input["metadata"], dict):
            for key, value in hybrid_input["metadata"].items():
                if isinstance(value, str) and len(value.strip()) > 10:  # 至少10个字符
                    logger.debug(f"从元数据键 '{key}' 提取文本数据")
                    return value
        
        return None
    
    def _extract_image_data(self, hybrid_input: Dict[str, Any]) -> Optional[Any]:
        """提取图像数据"""
        # 尝试不同的图像键
        image_keys = ["image", "image_data", "picture", "photo", "img", "visual"]
        
        for key in image_keys:
            if key in hybrid_input and hybrid_input[key]:
                data = hybrid_input[key]
                # 这里应该验证是否为图像数据
                # 暂时假设任何非文本数据都可能是图像
                if not isinstance(data, str) or (len(data) > 100 and any(x in data[:100].lower() for x in ['data:', 'base64', 'png', 'jpg', 'jpeg'])):
                    logger.debug(f"从键 '{key}' 提取图像数据")
                    return data
        
        return None
    
    def _extract_audio_data(self, hybrid_input: Dict[str, Any]) -> Optional[Any]:
        """提取音频数据"""
        # 尝试不同的音频键
        audio_keys = ["audio", "audio_data", "sound", "voice", "speech"]
        
        for key in audio_keys:
            if key in hybrid_input and hybrid_input[key]:
                data = hybrid_input[key]
                # 这里应该验证是否为音频数据
                # 暂时假设任何非文本、非图像数据都可能是音频
                if not isinstance(data, str) or (len(data) > 100 and any(x in data[:100].lower() for x in ['data:', 'base64', 'wav', 'mp3', 'audio'])):
                    logger.debug(f"从键 '{key}' 提取音频数据")
                    return data
        
        return None
    
    def _extract_other_modalities(self, hybrid_input: Dict[str, Any]) -> Dict[str, Any]:
        """提取其他模态数据"""
        other_modalities = {}
        
        # 排除已识别的模态键
        recognized_keys = ["text", "image", "audio", "metadata", "timestamp"]
        
        for key, value in hybrid_input.items():
            if key not in recognized_keys and value is not None:
                # 尝试确定模态类型
                modality_type = self._guess_modality_type(key, value)
                if modality_type:
                    other_modalities[modality_type] = value
                    logger.debug(f"推断模态类型: {key} -> {modality_type}")
        
        return other_modalities
    
    def _guess_modality_type(self, key: str, value: Any) -> Optional[str]:
        """根据键名和值猜测模态类型"""
        key_lower = key.lower()
        
        # 基于键名猜测
        if any(word in key_lower for word in ['video', 'movie', 'clip']):
            return ModalityType.VIDEO.value
        
        if any(word in key_lower for word in ['sensor', 'imu', 'accelerometer', 'gyroscope']):
            return ModalityType.SENSOR.value
        
        if any(word in key_lower for word in ['depth', '3d', 'pointcloud']):
            return "depth"
        
        if any(word in key_lower for word in ['thermal', 'infrared']):
            return "thermal"
        
        # 基于值类型猜测
        if isinstance(value, list) and len(value) > 0:
            # 可能是传感器数据序列
            if all(isinstance(x, (int, float)) for x in value[:10]):
                return ModalityType.SENSOR.value
        
        return None
    
    def _assess_modality_quality(self, modality_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """评估各模态质量"""
        quality_assessments = {}
        
        for modality, data in modality_data.items():
            quality_info = self._assess_single_modality_quality(modality, data)
            quality_assessments[modality] = quality_info
            
            # 更新统计
            quality_level = quality_info.get("quality_level", InputQuality.UNUSABLE.value)
            self.stats["quality_distribution"][quality_level] += 1
        
        return quality_assessments
    
    def _assess_single_modality_quality(self, modality: str, data: Any) -> Dict[str, Any]:
        """评估单个模态质量"""
        # 获取对应的质量评估器
        assessor = self.quality_assessors.get(modality, self.quality_assessors["generic"])
        
        try:
            quality_info = assessor(data)
            return quality_info
        except Exception as e:
            logger.warning(f"评估 {modality} 模态质量失败: {e}")
            return {
                "quality_level": InputQuality.UNUSABLE.value,
                "confidence": 0.0,
                "issues": ["quality_assessment_failed"],
                "score": 0.0
            }
    
    def _assess_text_quality(self, text_data: Any) -> Dict[str, Any]:
        """评估文本质量"""
        if not isinstance(text_data, str):
            return {
                "quality_level": InputQuality.UNUSABLE.value,
                "confidence": 0.0,
                "issues": ["not_text_data"],
                "score": 0.0
            }
        
        text = str(text_data)
        issues = []
        score = 1.0
        
        # 1. 长度评估
        length = len(text.strip())
        if length == 0:
            issues.append("empty_text")
            score *= 0.0
        elif length < 5:
            issues.append("very_short_text")
            score *= 0.3
        elif length < 20:
            issues.append("short_text")
            score *= 0.7
        elif length > 10000:
            issues.append("very_long_text")
            score *= 0.8
        
        # 2. 字符多样性
        unique_chars = len(set(text))
        char_diversity = unique_chars / max(len(text), 1)
        if char_diversity < 0.1:
            issues.append("low_character_diversity")
            score *= 0.5
        
        # 3. 语言结构（简单检查）
        # 检查是否有句子结束符
        has_sentence_end = any(mark in text for mark in ['.', '!', '?', '。', '！', '？'])
        if not has_sentence_end and length > 50:
            issues.append("no_sentence_ending")
            score *= 0.8
        
        # 4. 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio > 0.5:
            issues.append("high_special_character_ratio")
            score *= 0.6
        
        # 确定质量等级
        if score >= 0.9:
            quality_level = InputQuality.EXCELLENT.value
        elif score >= 0.7:
            quality_level = InputQuality.GOOD.value
        elif score >= 0.5:
            quality_level = InputQuality.FAIR.value
        elif score >= 0.2:
            quality_level = InputQuality.POOR.value
        else:
            quality_level = InputQuality.UNUSABLE.value
        
        return {
            "quality_level": quality_level,
            "confidence": min(score, 1.0),
            "issues": issues,
            "score": score,
            "length": length,
            "char_diversity": char_diversity,
            "has_sentence_end": has_sentence_end,
            "special_ratio": special_ratio
        }
    
    def _assess_image_quality(self, image_data: Any) -> Dict[str, Any]:
        """评估图像质量"""
        # 这里应该实现真实的图像质量评估
        # 暂时返回模拟结果
        issues = []
        score = 0.7  # 默认中等分数
        
        # 模拟质量评估
        if isinstance(image_data, str) and len(image_data) < 100:
            issues.append("possibly_corrupted_or_small_image")
            score = 0.3
        
        # 确定质量等级
        if score >= 0.8:
            quality_level = InputQuality.EXCELLENT.value
        elif score >= 0.6:
            quality_level = InputQuality.GOOD.value
        elif score >= 0.4:
            quality_level = InputQuality.FAIR.value
        elif score >= 0.2:
            quality_level = InputQuality.POOR.value
        else:
            quality_level = InputQuality.UNUSABLE.value
        
        return {
            "quality_level": quality_level,
            "confidence": score,
            "issues": issues,
            "score": score
        }
    
    def _assess_audio_quality(self, audio_data: Any) -> Dict[str, Any]:
        """评估音频质量"""
        # 这里应该实现真实的音频质量评估
        # 暂时返回模拟结果
        issues = []
        score = 0.6  # 默认中等分数
        
        # 模拟质量评估
        if isinstance(audio_data, str) and len(audio_data) < 100:
            issues.append("possibly_corrupted_or_small_audio")
            score = 0.3
        
        # 确定质量等级
        if score >= 0.8:
            quality_level = InputQuality.EXCELLENT.value
        elif score >= 0.6:
            quality_level = InputQuality.GOOD.value
        elif score >= 0.4:
            quality_level = InputQuality.FAIR.value
        elif score >= 0.2:
            quality_level = InputQuality.POOR.value
        else:
            quality_level = InputQuality.UNUSABLE.value
        
        return {
            "quality_level": quality_level,
            "confidence": score,
            "issues": issues,
            "score": score
        }
    
    def _assess_generic_quality(self, data: Any) -> Dict[str, Any]:
        """评估通用模态质量"""
        # 通用质量评估逻辑
        score = 0.5  # 默认分数
        
        # 根据数据类型简单评估
        if data is None:
            score = 0.0
        elif isinstance(data, (int, float, bool)):
            score = 0.9
        elif isinstance(data, str):
            score = 0.7
        elif isinstance(data, (list, dict)):
            score = 0.6
        
        # 确定质量等级
        if score >= 0.8:
            quality_level = InputQuality.EXCELLENT.value
        elif score >= 0.6:
            quality_level = InputQuality.GOOD.value
        elif score >= 0.4:
            quality_level = InputQuality.FAIR.value
        elif score >= 0.2:
            quality_level = InputQuality.POOR.value
        else:
            quality_level = InputQuality.UNUSABLE.value
        
        return {
            "quality_level": quality_level,
            "confidence": score,
            "issues": [],
            "score": score
        }
    
    def _detect_modality_noise(self, modality_data: Dict[str, Any],
                              quality_assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """检测各模态噪声和问题"""
        noise_detections = {}
        
        for modality, data in modality_data.items():
            quality_info = quality_assessments.get(modality, {})
            
            # 获取对应的噪声检测器
            detector = self.noise_detectors.get(modality, lambda d, q: {})
            
            try:
                detection_info = detector(data, quality_info)
                noise_detections[modality] = detection_info
            except Exception as e:
                logger.warning(f"检测 {modality} 模态噪声失败: {e}")
                noise_detections[modality] = {
                    "noise_detected": False,
                    "noise_types": [],
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        return noise_detections
    
    def _detect_text_noise(self, text_data: Any, quality_info: Dict[str, Any]) -> Dict[str, Any]:
        """检测文本噪声"""
        if not isinstance(text_data, str):
            return {
                "noise_detected": False,
                "noise_types": ["not_text_data"],
                "confidence": 0.0
            }
        
        text = str(text_data)
        noise_types = []
        confidence = 0.0
        
        # 1. 检查是否有乱码或编码问题
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            noise_types.append("encoding_issue")
            confidence = max(confidence, 0.8)
        
        # 2. 检查是否有过多重复字符
        if len(text) > 10:
            for i in range(len(text) - 3):
                if text[i] == text[i+1] == text[i+2] == text[i+3]:
                    noise_types.append("character_repetition")
                    confidence = max(confidence, 0.6)
                    break
        
        # 3. 检查是否有过多特殊字符
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)
        if special_ratio > 0.3:
            noise_types.append("high_special_character_ratio")
            confidence = max(confidence, 0.7)
        
        # 4. 检查是否有明显的随机字符
        if len(text) > 20 and special_ratio > 0.5:
            noise_types.append("possible_random_text")
            confidence = max(confidence, 0.9)
        
        noise_detected = len(noise_types) > 0
        
        return {
            "noise_detected": noise_detected,
            "noise_types": noise_types,
            "confidence": confidence,
            "text_length": len(text),
            "special_ratio": special_ratio
        }
    
    def _detect_image_noise(self, image_data: Any, quality_info: Dict[str, Any]) -> Dict[str, Any]:
        """检测图像噪声"""
        # 这里应该实现真实的图像噪声检测
        # 暂时返回模拟结果
        return {
            "noise_detected": quality_info.get("score", 0.0) < 0.5,
            "noise_types": ["simulated_noise_detection"] if quality_info.get("score", 0.0) < 0.5 else [],
            "confidence": 0.5
        }
    
    def _detect_audio_noise(self, audio_data: Any, quality_info: Dict[str, Any]) -> Dict[str, Any]:
        """检测音频噪声"""
        # 这里应该实现真实的音频噪声检测
        # 暂时返回模拟结果
        return {
            "noise_detected": quality_info.get("score", 0.0) < 0.5,
            "noise_types": ["simulated_noise_detection"] if quality_info.get("score", 0.0) < 0.5 else [],
            "confidence": 0.5
        }
    
    def _repair_modalities(self, modality_data: Dict[str, Any],
                          quality_assessments: Dict[str, Dict[str, Any]],
                          noise_detections: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """修复各模态数据"""
        repaired_data = {}
        repair_reports = {}
        
        for modality, data in modality_data.items():
            quality_info = quality_assessments.get(modality, {})
            noise_info = noise_detections.get(modality, {})
            
            # 检查是否需要修复
            need_repair = (
                quality_info.get("quality_level") in [InputQuality.POOR.value, InputQuality.FAIR.value] or
                noise_info.get("noise_detected", False)
            )
            
            if need_repair:
                self.stats["repair_attempts"] += 1
                
                # 获取对应的修复处理器
                repair_processor = self.repair_processors.get(modality, lambda d, q, n: (d, {"repaired": False}))
                
                try:
                    repaired, repair_report = repair_processor(data, quality_info, noise_info)
                    
                    if repair_report.get("repaired", False):
                        self.stats["repair_successes"] += 1
                        repaired_data[modality] = repaired
                        repair_reports[modality] = repair_report
                        logger.info(f"成功修复 {modality} 模态数据")
                    else:
                        # 修复失败，使用原始数据
                        repaired_data[modality] = data
                        repair_reports[modality] = repair_report
                        logger.warning(f"{modality} 模态数据修复失败")
                        
                except Exception as e:
                    logger.error(f"修复 {modality} 模态数据时出错: {e}")
                    repaired_data[modality] = data
                    repair_reports[modality] = {
                        "repaired": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
            else:
                # 不需要修复，使用原始数据
                repaired_data[modality] = data
                repair_reports[modality] = {
                    "repaired": False,
                    "reason": "no_repair_needed",
                    "quality_level": quality_info.get("quality_level", "unknown")
                }
        
        return repaired_data, repair_reports
    
    def _repair_text(self, text_data: Any, quality_info: Dict[str, Any],
                    noise_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """修复文本数据"""
        if not isinstance(text_data, str):
            return text_data, {"repaired": False, "reason": "not_text_data"}
        
        text = str(text_data)
        original_text = text
        repair_actions = []
        
        # 1. 去除多余空格
        if '  ' in text:
            text = ' '.join(text.split())
            repair_actions.append("remove_extra_spaces")
        
        # 2. 修复编码问题（简单尝试）
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            # 尝试使用错误忽略策略
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            repair_actions.append("fix_encoding")
        
        # 3. 如果太短，尝试从噪声中提取有意义的部分
        if len(text.strip()) < 5 and noise_info.get("noise_detected", False):
            # 尝试提取字母数字字符
            alnum_chars = [c for c in text if c.isalnum() or c.isspace()]
            extracted_text = ''.join(alnum_chars).strip()
            if len(extracted_text) > len(text.strip()):
                text = extracted_text
                repair_actions.append("extract_alphanumeric")
        
        # 检查是否实际进行了修复
        repaired = text != original_text and len(repair_actions) > 0
        
        return text, {
            "repaired": repaired,
            "repair_actions": repair_actions,
            "original_length": len(original_text),
            "repaired_length": len(text),
            "changes_made": repaired
        }
    
    def _repair_image(self, image_data: Any, quality_info: Dict[str, Any],
                     noise_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """修复图像数据"""
        # 这里应该实现真实的图像修复
        # 暂时返回原始数据
        return image_data, {
            "repaired": False,
            "reason": "image_repair_not_implemented",
            "message": "图像修复功能待实现"
        }
    
    def _repair_audio(self, audio_data: Any, quality_info: Dict[str, Any],
                     noise_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """修复音频数据"""
        # 这里应该实现真实的音频修复
        # 暂时返回原始数据
        return audio_data, {
            "repaired": False,
            "reason": "audio_repair_not_implemented",
            "message": "音频修复功能待实现"
        }
    
    def _infer_missing_information(self, repaired_data: Dict[str, Any],
                                  quality_assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """推断缺失信息"""
        inferred_data = {}
        
        # 检查是否有模态完全缺失或质量极差
        for modality in self.supported_modalities:
            if modality not in repaired_data:
                # 该模态完全缺失，尝试推断
                inferred = self._infer_missing_modality(modality, repaired_data, quality_assessments)
                if inferred:
                    inferred_data[modality] = inferred
        
        return inferred_data
    
    def _infer_missing_modality(self, missing_modality: str, existing_data: Dict[str, Any],
                               quality_assessments: Dict[str, Dict[str, Any]]) -> Optional[Any]:
        """推断缺失的模态"""
        # 这里应该实现基于现有模态推断缺失模态的逻辑
        # 暂时返回None，表示无法推断
        return None
    
    def _generate_parse_report(self, modality_data: Dict[str, Any],
                              quality_assessments: Dict[str, Dict[str, Any]],
                              noise_detections: Dict[str, Dict[str, Any]],
                              repair_reports: Dict[str, Dict[str, Any]],
                              inferred_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合解析报告"""
        # 计算总体质量分数
        quality_scores = []
        for modality, assessment in quality_assessments.items():
            quality_scores.append(assessment.get("score", 0.0))
        
        overall_quality = sum(quality_scores) / max(len(quality_scores), 1)
        
        # 确定总体质量等级
        if overall_quality >= 0.8:
            overall_level = InputQuality.EXCELLENT.value
        elif overall_quality >= 0.6:
            overall_level = InputQuality.GOOD.value
        elif overall_quality >= 0.4:
            overall_level = InputQuality.FAIR.value
        elif overall_quality >= 0.2:
            overall_level = InputQuality.POOR.value
        else:
            overall_level = InputQuality.UNUSABLE.value
        
        # 统计修复情况
        repaired_modalities = [modality for modality, report in repair_reports.items() 
                              if report.get("repaired", False)]
        
        # 生成报告
        report = {
            "overall_quality": {
                "score": overall_quality,
                "level": overall_level,
                "description": self._get_quality_description(overall_level)
            },
            "modality_summary": {
                "total_modalities": len(modality_data),
                "modalities_found": list(modality_data.keys()),
                "modalities_repaired": repaired_modalities,
                "modalities_inferred": list(inferred_data.keys())
            },
            "quality_distribution": {
                modality: assessment.get("quality_level", "unknown")
                for modality, assessment in quality_assessments.items()
            },
            "noise_summary": {
                modality: detection.get("noise_types", [])
                for modality, detection in noise_detections.items()
                if detection.get("noise_detected", False)
            },
            "repair_summary": {
                "total_repair_attempts": len([r for r in repair_reports.values() 
                                            if r.get("repaired", False) or "error" in r]),
                "successful_repairs": len(repaired_modalities),
                "repair_success_rate": len(repaired_modalities) / max(self.stats["repair_attempts"], 1)
            },
            "recommendations": self._generate_recommendations(
                modality_data, quality_assessments, noise_detections, repair_reports
            ),
            "timestamp": time.time()
        }
        
        return report
    
    def _get_quality_description(self, quality_level: str) -> str:
        """获取质量等级描述"""
        descriptions = {
            InputQuality.EXCELLENT.value: "输入质量优秀，所有模态清晰完整",
            InputQuality.GOOD.value: "输入质量良好，有轻微噪声但不影响理解",
            InputQuality.FAIR.value: "输入质量一般，部分模态有中等噪声或缺失",
            InputQuality.POOR.value: "输入质量差，多个模态有严重问题",
            InputQuality.UNUSABLE.value: "输入质量不可用，无法有效处理"
        }
        return descriptions.get(quality_level, "未知质量等级")
    
    def _generate_recommendations(self, modality_data: Dict[str, Any],
                                 quality_assessments: Dict[str, Dict[str, Any]],
                                 noise_detections: Dict[str, Dict[str, Any]],
                                 repair_reports: Dict[str, Dict[str, Any]]) -> List[str]:
        """生成处理建议"""
        recommendations = []
        
        # 检查低质量模态
        for modality, assessment in quality_assessments.items():
            quality_level = assessment.get("quality_level", "")
            if quality_level in [InputQuality.POOR.value, InputQuality.UNUSABLE.value]:
                recommendations.append(f"{modality}模态质量较差，建议重新输入或提供补充信息")
        
        # 检查噪声问题
        for modality, detection in noise_detections.items():
            if detection.get("noise_detected", False):
                noise_types = detection.get("noise_types", [])
                if noise_types:
                    recommendations.append(f"{modality}模态检测到噪声: {', '.join(noise_types)}")
        
        # 检查模态缺失
        expected_modalities = ["text", "image", "audio"]
        for modality in expected_modalities:
            if modality not in modality_data:
                recommendations.append(f"缺少{modality}模态，可能影响理解完整性")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取解析器统计信息"""
        stats_copy = self.stats.copy()
        
        # 计算成功率
        total = stats_copy["total_inputs"]
        successful = stats_copy["successful_parses"]
        failed = stats_copy["failed_parses"]
        
        if total > 0:
            stats_copy["success_rate"] = successful / total
            stats_copy["failure_rate"] = failed / total
        else:
            stats_copy["success_rate"] = 0.0
            stats_copy["failure_rate"] = 0.0
        
        # 计算修复成功率
        repair_attempts = stats_copy["repair_attempts"]
        repair_successes = stats_copy["repair_successes"]
        
        if repair_attempts > 0:
            stats_copy["repair_success_rate"] = repair_successes / repair_attempts
        else:
            stats_copy["repair_success_rate"] = 0.0
        
        return stats_copy


def test_hybrid_modal_parser():
    """测试混合模态解析器"""
    logger.info("测试混合模态解析器...")
    
    try:
        # 创建解析器
        parser = HybridModalParser()
        
        # 测试用例1：正常混合输入
        test_input_1 = {
            "text": "这是一张红色圆形杯子的图片，旁边有一段描述",
            "image_data": "模拟图像数据",
            "audio_data": "模拟音频数据",
            "metadata": {"source": "test", "timestamp": "2026-03-02"}
        }
        
        result_1 = parser.parse_hybrid_input(test_input_1)
        assert result_1["success"], "测试用例1解析失败"
        assert len(result_1["parsed_modalities"]) >= 1, "测试用例1未解析到任何模态"
        
        # 测试用例2：有噪声的文本
        test_input_2 = {
            "text": "这是###一个%%%有噪声的文本!!!",
            "image": "小图像"
        }
        
        result_2 = parser.parse_hybrid_input(test_input_2)
        assert result_2["success"], "测试用例2解析失败"
        
        # 检查质量评估
        quality_assessments = result_2.get("quality_assessments", {})
        assert "text" in quality_assessments, "文本质量评估缺失"
        
        # 测试用例3：无效输入
        test_input_3 = {}
        
        result_3 = parser.parse_hybrid_input(test_input_3)
        # 可能成功但解析不到模态，也可能失败
        
        # 测试统计信息获取
        stats = parser.get_statistics()
        assert "total_inputs" in stats, "统计信息中缺少total_inputs"
        assert stats["total_inputs"] >= 2, f"输入统计错误: {stats['total_inputs']}"
        
        logger.info("✅ 混合模态解析器测试通过")
        
        return {
            "success": True,
            "test_cases": 3,
            "parsed_modalities_test1": result_1["parsed_modalities"],
            "quality_assessments_test2": quality_assessments,
            "statistics": stats,
            "message": "混合模态解析器测试完成"
        }
        
    except Exception as e:
        logger.error(f"❌ 混合模态解析器测试失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "混合模态解析器测试失败"
        }


# 导出主要类和方法
__all__ = [
    "HybridModalParser",
    "InputQuality",
    "ModalityType",
    "test_hybrid_modal_parser"
]

if __name__ == "__main__":
    # 运行测试
    test_result = test_hybrid_modal_parser()
    print(f"测试结果: {test_result}")

