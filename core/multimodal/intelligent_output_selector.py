"""
智能输出选择器

修复计划第五阶段：完善用户体验（自然交互+多模态反馈）
任务5.2：创建智能输出选择器

核心功能：
1. 根据用户偏好、环境、设备自动选择最佳输出模态
2. 实现用户画像学习，个性化输出策略
3. 添加A/B测试，持续优化选择算法
"""

import sys
import os
import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 配置日志
logger = logging.getLogger("intelligent_output_selector")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class OutputModality(Enum):
    """输出模态枚举"""
    TEXT = "text"          # 文本输出
    IMAGE = "image"        # 图像输出
    AUDIO = "audio"        # 音频输出
    VIDEO = "video"        # 视频输出
    MULTIMODAL = "multimodal"  # 多模态组合输出
    INTERACTIVE = "interactive"  # 交互式输出


class UserPreferenceType(Enum):
    """用户偏好类型"""
    VISUAL = "visual"      # 视觉型：偏好图像和视频
    AUDITORY = "auditory"  # 听觉型：偏好音频
    READ_WRITE = "read_write"  # 读写型：偏好文本
    KINESTHETIC = "kinesthetic"  # 动觉型：偏好交互式
    BALANCED = "balanced"  # 平衡型：无特别偏好


class EnvironmentType(Enum):
    """环境类型"""
    QUIET_INDOOR = "quiet_indoor"  # 安静室内
    NOISY_INDOOR = "noisy_indoor"  # 嘈杂室内
    OUTDOOR = "outdoor"           # 户外
    MEETING = "meeting"           # 会议中
    DRIVING = "driving"           # 驾驶中
    WALKING = "walking"           # 行走中


class DeviceType(Enum):
    """设备类型"""
    DESKTOP = "desktop"    # 桌面电脑
    LAPTOP = "laptop"      # 笔记本电脑
    TABLET = "tablet"      # 平板
    PHONE = "phone"        # 手机
    SMARTWATCH = "smartwatch"  # 智能手表
    SMART_SPEAKER = "smart_speaker"  # 智能音箱


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preference_type: UserPreferenceType = UserPreferenceType.BALANCED
    learning_score: float = 0.5  # 学习效果分数 (0.0-1.0)
    satisfaction_score: float = 0.5  # 满意度分数 (0.0-1.0)
    interaction_count: int = 0
    modality_preferences: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.5,
        "image": 0.5,
        "audio": 0.5,
        "video": 0.5,
        "multimodal": 0.5,
        "interactive": 0.5
    })
    last_updated: float = field(default_factory=time.time)
    
    def update_preference(self, modality: str, score_delta: float) -> None:
        """更新模态偏好"""
        if modality in self.modality_preferences:
            new_score = self.modality_preferences[modality] + score_delta
            self.modality_preferences[modality] = max(0.0, min(1.0, new_score))
            self.last_updated = time.time()
    
    def get_top_modality(self, n: int = 1) -> List[Tuple[str, float]]:
        """获取前N个偏好模态"""
        sorted_items = sorted(self.modality_preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "preference_type": self.preference_type.value,
            "learning_score": self.learning_score,
            "satisfaction_score": self.satisfaction_score,
            "interaction_count": self.interaction_count,
            "modality_preferences": self.modality_preferences,
            "last_updated": self.last_updated
        }


@dataclass
class ContextInfo:
    """上下文信息"""
    environment: EnvironmentType = EnvironmentType.QUIET_INDOOR
    device: DeviceType = DeviceType.DESKTOP
    time_of_day: str = "day"  # "morning", "day", "evening", "night"
    network_speed: float = 10.0  # Mbps
    battery_level: float = 1.0  # 0.0-1.0
    screen_size: Tuple[int, int] = (1920, 1080)
    audio_output_available: bool = True
    user_attention_level: float = 0.8  # 用户注意力水平 (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "environment": self.environment.value,
            "device": self.device.value,
            "time_of_day": self.time_of_day,
            "network_speed": self.network_speed,
            "battery_level": self.battery_level,
            "screen_size": self.screen_size,
            "audio_output_available": self.audio_output_available,
            "user_attention_level": self.user_attention_level
        }


@dataclass
class SelectionResult:
    """选择结果"""
    selected_modality: OutputModality
    confidence: float
    alternatives: List[Tuple[OutputModality, float]]  # 备选方案
    reasoning: str
    context_factors: Dict[str, float]
    user_factors: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "selected_modality": self.selected_modality.value,
            "confidence": self.confidence,
            "alternatives": [(mod.value, score) for mod, score in self.alternatives],
            "reasoning": self.reasoning,
            "context_factors": self.context_factors,
            "user_factors": self.user_factors,
            "timestamp": self.timestamp
        }


class IntelligentOutputSelector:
    """
    智能输出选择器
    
    核心功能：
    1. 根据用户偏好、环境、设备自动选择最佳输出模态
    2. 实现用户画像学习，个性化输出策略
    3. 添加A/B测试，持续优化选择算法
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化智能输出选择器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 用户画像存储
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 环境检测器
        self.environment_detector = EnvironmentDetector()
        
        # 设备检测器
        self.device_detector = DeviceDetector()
        
        # 规则引擎
        self.rule_engine = RuleEngine()
        
        # 机器学习模型（简化版）
        self.ml_model = MLModel()
        
        # A/B测试管理器
        self.ab_test_manager = ABTestManager()
        
        # 统计信息
        self.stats = {
            "total_selections": 0,
            "average_confidence": 0.0,
            "user_satisfaction_avg": 0.0,
            "learning_updates": 0,
            "ab_test_runs": 0,
            "success_rate": 0.0
        }
        
        # 缓存最近的选择
        self.recent_selections = []
        
        logger.info("智能输出选择器初始化完成")
    
    def select_output_modality(self, user_id: str, input_data: Dict[str, Any],
                              available_modalities: List[OutputModality]) -> SelectionResult:
        """
        选择输出模态
        
        Args:
            user_id: 用户ID
            input_data: 输入数据
            available_modalities: 可用的输出模态列表
            
        Returns:
            选择结果
        """
        self.stats["total_selections"] += 1
        
        # 获取或创建用户画像
        user_profile = self._get_or_create_user_profile(user_id)
        
        # 检测上下文
        context = self._detect_context()
        
        # 获取输入特征
        input_features = self._extract_input_features(input_data)
        
        # 计算各模态的得分
        modality_scores = self._calculate_modality_scores(
            user_profile, context, input_features, available_modalities
        )
        
        # 应用A/B测试（如果启用）
        if self.config.get("enable_ab_testing", True):
            modality_scores = self.ab_test_manager.apply_variant(
                user_id, modality_scores, self.stats["ab_test_runs"]
            )
            self.stats["ab_test_runs"] += 1
        
        # 选择最佳模态
        selected_modality, confidence = self._select_best_modality(modality_scores)
        
        # 生成解释
        reasoning = self._generate_reasoning(
            selected_modality, modality_scores, user_profile, context
        )
        
        # 提取影响因素
        context_factors = self._extract_context_factors(context, modality_scores)
        user_factors = self._extract_user_factors(user_profile, modality_scores)
        
        # 创建结果
        result = SelectionResult(
            selected_modality=selected_modality,
            confidence=confidence,
            alternatives=self._get_alternatives(modality_scores, selected_modality),
            reasoning=reasoning,
            context_factors=context_factors,
            user_factors=user_factors
        )
        
        # 缓存结果
        self.recent_selections.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "result": result.to_dict(),
            "input_features": input_features
        })
        
        # 更新统计
        self._update_stats(confidence)
        
        logger.info(f"为用户 {user_id} 选择输出模态: {selected_modality.value}, 置信度: {confidence:.2f}")
        
        return result
    
    def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # 创建新用户画像
        profile = UserProfile(user_id=user_id)
        self.user_profiles[user_id] = profile
        
        logger.debug(f"创建新用户画像: {user_id}")
        
        return profile
    
    def _detect_context(self) -> ContextInfo:
        """检测当前上下文"""
        # 使用环境检测器
        environment = self.environment_detector.detect()
        
        # 使用设备检测器
        device = self.device_detector.detect()
        
        # 创建上下文信息
        context = ContextInfo(
            environment=environment,
            device=device,
            time_of_day=self._get_time_of_day(),
            network_speed=self._estimate_network_speed(),
            battery_level=self._get_battery_level(),
            audio_output_available=self._check_audio_output(),
            user_attention_level=self._estimate_attention_level()
        )
        
        return context
    
    def _extract_input_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取输入特征"""
        features = {
            "modality_count": len(input_data.get("modalities", [])),
            "has_text": "text" in input_data,
            "has_image": "image" in input_data,
            "has_audio": "audio" in input_data,
            "complexity": self._estimate_complexity(input_data),
            "urgency": input_data.get("urgency", 0.5),
            "sensitivity": input_data.get("sensitivity", 0.5)
        }
        
        return features
    
    def _calculate_modality_scores(self, user_profile: UserProfile, context: ContextInfo,
                                 input_features: Dict[str, Any],
                                 available_modalities: List[OutputModality]) -> Dict[OutputModality, float]:
        """计算各模态的得分"""
        scores = {}
        
        for modality in available_modalities:
            # 1. 用户偏好得分
            pref_score = self._calculate_preference_score(user_profile, modality)
            
            # 2. 上下文适配得分
            context_score = self._calculate_context_score(context, modality)
            
            # 3. 输入匹配得分
            input_score = self._calculate_input_score(input_features, modality)
            
            # 4. 规则引擎得分
            rule_score = self.rule_engine.evaluate(modality, user_profile, context, input_features)
            
            # 5. 机器学习模型得分
            ml_score = self.ml_model.predict(modality, user_profile, context, input_features)
            
            # 计算总分
            total_score = (
                pref_score * 0.25 +
                context_score * 0.25 +
                input_score * 0.20 +
                rule_score * 0.15 +
                ml_score * 0.15
            )
            
            scores[modality] = total_score
        
        return scores
    
    def _calculate_preference_score(self, user_profile: UserProfile, modality: OutputModality) -> float:
        """计算用户偏好得分"""
        modality_str = modality.value
        
        if modality_str in user_profile.modality_preferences:
            return user_profile.modality_preferences[modality_str]
        
        # 默认得分
        return 0.5
    
    def _calculate_context_score(self, context: ContextInfo, modality: OutputModality) -> float:
        """计算上下文适配得分"""
        # 环境适配规则
        env_rules = {
            EnvironmentType.NOISY_INDOOR: {OutputModality.TEXT: 0.9, OutputModality.AUDIO: 0.3},
            EnvironmentType.DRIVING: {OutputModality.AUDIO: 0.9, OutputModality.TEXT: 0.2},
            EnvironmentType.WALKING: {OutputModality.AUDIO: 0.8, OutputModality.TEXT: 0.4},
            EnvironmentType.MEETING: {OutputModality.TEXT: 0.9, OutputModality.AUDIO: 0.2}
        }
        
        # 设备适配规则
        device_rules = {
            DeviceType.PHONE: {OutputModality.TEXT: 0.7, OutputModality.AUDIO: 0.8, OutputModality.IMAGE: 0.6},
            DeviceType.SMARTWATCH: {OutputModality.TEXT: 0.3, OutputModality.AUDIO: 0.9},
            DeviceType.SMART_SPEAKER: {OutputModality.AUDIO: 1.0, OutputModality.TEXT: 0.1},
            DeviceType.DESKTOP: {OutputModality.MULTIMODAL: 0.9, OutputModality.INTERACTIVE: 0.8}
        }
        
        # 计算环境得分
        env_score = 0.5
        if context.environment in env_rules:
            env_scores = env_rules[context.environment]
            env_score = env_scores.get(modality, 0.5)
        
        # 计算设备得分
        device_score = 0.5
        if context.device in device_rules:
            device_scores = device_rules[context.device]
            device_score = device_scores.get(modality, 0.5)
        
        # 考虑网络速度和电池
        network_factor = min(1.0, context.network_speed / 50.0)  # 假设50Mbps为理想速度
        battery_factor = context.battery_level
        
        # 对于高带宽需求模态，网络因素更重要
        bandwidth_modalities = {OutputModality.VIDEO, OutputModality.IMAGE, OutputModality.MULTIMODAL}
        if modality in bandwidth_modalities:
            network_weight = 0.6
        else:
            network_weight = 0.3
        
        # 组合得分
        context_score = (
            env_score * 0.3 +
            device_score * 0.3 +
            network_factor * network_weight +
            battery_factor * 0.1
        )
        
        return min(1.0, max(0.0, context_score))
    
    def _calculate_input_score(self, input_features: Dict[str, Any], modality: OutputModality) -> float:
        """计算输入匹配得分"""
        # 输入模态匹配规则
        modality_matching = {
            "text": {OutputModality.TEXT: 0.9, OutputModality.AUDIO: 0.7, OutputModality.IMAGE: 0.4},
            "image": {OutputModality.IMAGE: 0.9, OutputModality.TEXT: 0.6, OutputModality.MULTIMODAL: 0.8},
            "audio": {OutputModality.AUDIO: 0.9, OutputModality.TEXT: 0.6, OutputModality.MULTIMODAL: 0.7}
        }
        
        # 复杂度匹配规则
        complexity_rules = {
            OutputModality.TEXT: lambda c: 0.9 if c < 0.5 else 0.5,  # 低复杂度适合文本
            OutputModality.MULTIMODAL: lambda c: 0.3 if c < 0.3 else 0.8,  # 高复杂度适合多模态
            OutputModality.INTERACTIVE: lambda c: 0.2 if c < 0.4 else 0.9  # 高复杂度适合交互式
        }
        
        # 计算模态匹配得分
        modality_score = 0.5
        for input_modality, scores in modality_matching.items():
            if input_features.get(f"has_{input_modality}", False):
                modality_score = max(modality_score, scores.get(modality, 0.5))
        
        # 计算复杂度匹配得分
        complexity = input_features.get("complexity", 0.5)
        complexity_score = 0.5
        if modality in complexity_rules:
            complexity_score = complexity_rules[modality](complexity)
        
        # 计算紧急度匹配得分
        urgency = input_features.get("urgency", 0.5)
        if urgency > 0.7:  # 紧急情况
            urgency_score = 1.0 if modality == OutputModality.AUDIO else 0.6
        else:
            urgency_score = 0.7
        
        # 组合得分
        input_score = (
            modality_score * 0.4 +
            complexity_score * 0.3 +
            urgency_score * 0.3
        )
        
        return min(1.0, max(0.0, input_score))
    
    def _select_best_modality(self, modality_scores: Dict[OutputModality, float]) -> Tuple[OutputModality, float]:
        """选择最佳模态"""
        if not modality_scores:
            return OutputModality.TEXT, 0.5
        
        # 找到最高分
        best_modality = max(modality_scores.items(), key=lambda x: x[1])
        
        # 计算置信度
        scores = list(modality_scores.values())
        if len(scores) > 1:
            scores_sorted = sorted(scores, reverse=True)
            confidence = scores_sorted[0] - scores_sorted[1]  # 与第二名的差距
            confidence = min(1.0, max(0.1, confidence * 3))  # 调整范围
        else:
            confidence = 0.8
        
        return best_modality[0], confidence
    
    def _get_alternatives(self, modality_scores: Dict[OutputModality, float],
                         selected_modality: OutputModality,
                         n: int = 3) -> List[Tuple[OutputModality, float]]:
        """获取备选方案"""
        # 排序所有模态
        sorted_modalities = sorted(modality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 排除已选择的模态，取前N个
        alternatives = []
        for modality, score in sorted_modalities:
            if modality != selected_modality and len(alternatives) < n:
                alternatives.append((modality, score))
        
        return alternatives
    
    def _generate_reasoning(self, selected_modality: OutputModality,
                          modality_scores: Dict[OutputModality, float],
                          user_profile: UserProfile,
                          context: ContextInfo) -> str:
        """生成选择解释"""
        reasons = []
        
        # 用户偏好原因
        user_pref = user_profile.get_top_modality(1)
        if user_pref and user_pref[0][0] == selected_modality.value:
            reasons.append("根据您的使用偏好")
        
        # 环境原因
        if context.environment == EnvironmentType.NOISY_INDOOR and selected_modality == OutputModality.TEXT:
            reasons.append("当前环境嘈杂，文本更清晰")
        elif context.environment == EnvironmentType.DRIVING and selected_modality == OutputModality.AUDIO:
            reasons.append("驾驶中，语音更安全")
        
        # 设备原因
        if context.device == DeviceType.SMART_SPEAKER and selected_modality == OutputModality.AUDIO:
            reasons.append("智能音箱最适合语音输出")
        elif context.device == DeviceType.DESKTOP and selected_modality == OutputModality.MULTIMODAL:
            reasons.append("桌面设备支持多模态体验")
        
        # 网络原因
        if context.network_speed < 5.0 and selected_modality not in [OutputModality.VIDEO, OutputModality.MULTIMODAL]:
            reasons.append("网络较慢，避免高带宽输出")
        
        # 生成最终解释
        if reasons:
            reasoning = "选择理由：" + "；".join(reasons)
        else:
            reasoning = f"基于综合评估选择了{selected_modality.value}输出"
        
        return reasoning
    
    def _extract_context_factors(self, context: ContextInfo,
                               modality_scores: Dict[OutputModality, float]) -> Dict[str, float]:
        """提取上下文影响因素"""
        return {
            "environment": 0.3,
            "device": 0.3,
            "network_speed": 0.2,
            "battery_level": 0.1,
            "audio_available": 0.1
        }
    
    def _extract_user_factors(self, user_profile: UserProfile,
                            modality_scores: Dict[OutputModality, float]) -> Dict[str, float]:
        """提取用户影响因素"""
        return {
            "preference_type": 0.4,
            "modality_preferences": 0.3,
            "learning_score": 0.2,
            "satisfaction_score": 0.1
        }
    
    def _get_time_of_day(self) -> str:
        """获取当前时间段"""
        from datetime import datetime
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "day"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _estimate_network_speed(self) -> float:
        """估计网络速度"""
        # 简化实现，实际中需要真实检测
        return 10.0  # Mbps
    
    def _get_battery_level(self) -> float:
        """获取电池电量"""
        # 简化实现，实际中需要系统API
        return 0.8
    
    def _check_audio_output(self) -> bool:
        """检查音频输出是否可用"""
        # 简化实现
        return True
    
    def _estimate_attention_level(self) -> float:
        """估计用户注意力水平"""
        # 简化实现
        return 0.8
    
    def _estimate_complexity(self, input_data: Dict[str, Any]) -> float:
        """估计输入复杂度"""
        complexity = 0.0
        
        # 模态数量增加复杂度
        modality_count = len(input_data.get("modalities", []))
        complexity += modality_count * 0.2
        
        # 数据大小增加复杂度
        data_size = len(str(input_data)) / 10000
        complexity += min(0.5, data_size * 0.1)
        
        return min(1.0, complexity)
    
    def _update_stats(self, confidence: float) -> None:
        """更新统计信息"""
        # 更新平均置信度
        total = self.stats["total_selections"]
        self.stats["average_confidence"] = (
            self.stats["average_confidence"] * (total - 1) + confidence
        ) / total
    
    def record_user_feedback(self, user_id: str, result: SelectionResult,
                           satisfaction: float, learning_effect: float) -> None:
        """
        记录用户反馈
        
        Args:
            user_id: 用户ID
            result: 选择结果
            satisfaction: 满意度 (0.0-1.0)
            learning_effect: 学习效果 (0.0-1.0)
        """
        if user_id not in self.user_profiles:
            logger.warning(f"用户 {user_id} 不存在，创建新画像")
            self._get_or_create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # 更新用户画像
        profile.satisfaction_score = (
            profile.satisfaction_score * profile.interaction_count + satisfaction
        ) / (profile.interaction_count + 1)
        
        profile.learning_score = (
            profile.learning_score * profile.interaction_count + learning_effect
        ) / (profile.interaction_count + 1)
        
        profile.interaction_count += 1
        
        # 更新模态偏好
        selected_modality = result.selected_modality.value
        profile.update_preference(selected_modality, satisfaction * 0.1)
        
        # 更新统计
        self.stats["user_satisfaction_avg"] = (
            self.stats["user_satisfaction_avg"] * (self.stats["total_selections"] - 1) + satisfaction
        ) / self.stats["total_selections"]
        
        self.stats["learning_updates"] += 1
        
        logger.info(f"记录用户 {user_id} 反馈: 满意度 {satisfaction:.2f}, 学习效果 {learning_effect:.2f}")


# ==================== 辅助类实现 ====================

class EnvironmentDetector:
    """环境检测器"""
    
    def detect(self) -> EnvironmentType:
        """检测当前环境"""
        # 简化实现，实际中需要传感器数据
        return EnvironmentType.QUIET_INDOOR


class DeviceDetector:
    """设备检测器"""
    
    def detect(self) -> DeviceType:
        """检测当前设备"""
        # 简化实现
        return DeviceType.DESKTOP


class RuleEngine:
    """规则引擎"""
    
    def evaluate(self, modality: OutputModality, user_profile: UserProfile,
                context: ContextInfo, input_features: Dict[str, Any]) -> float:
        """评估规则得分"""
        # 简化实现
        return 0.5


class MLModel:
    """机器学习模型"""
    
    def __init__(self):
        """初始化模型"""
        pass
    
    def predict(self, modality: OutputModality, user_profile: UserProfile,
               context: ContextInfo, input_features: Dict[str, Any]) -> float:
        """预测得分"""
        # 简化实现
        return 0.5


class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self):
        """初始化A/B测试管理器"""
        self.variants = ["control", "variant_a", "variant_b"]
        self.assignment_cache = {}
    
    def apply_variant(self, user_id: str, modality_scores: Dict[OutputModality, float],
                     test_run: int) -> Dict[OutputModality, float]:
        """应用A/B测试变体"""
        # 为每个用户分配固定变体（基于用户ID哈希）
        if user_id not in self.assignment_cache:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            variant_idx = hash_val % len(self.variants)
            self.assignment_cache[user_id] = self.variants[variant_idx]
        
        variant = self.assignment_cache[user_id]
        
        if variant == "control":
            # 控制组，不改变分数
            return modality_scores
        elif variant == "variant_a":
            # 变体A：倾向于文本输出
            modified_scores = modality_scores.copy()
            if OutputModality.TEXT in modified_scores:
                modified_scores[OutputModality.TEXT] *= 1.2
            return modified_scores
        elif variant == "variant_b":
            # 变体B：倾向于多模态输出
            modified_scores = modality_scores.copy()
            if OutputModality.MULTIMODAL in modified_scores:
                modified_scores[OutputModality.MULTIMODAL] *= 1.2
            return modified_scores
        else:
            return modality_scores


# ==================== 测试函数 ====================

def test_intelligent_output_selector() -> None:
    """测试智能输出选择器"""
    print("测试智能输出选择器...")
    
    # 创建选择器实例
    selector = IntelligentOutputSelector({
        "enable_ab_testing": True
    })
    
    # 测试用户ID
    user_id = "test_user_001"
    
    # 模拟输入数据
    input_data = {
        "modalities": ["text", "image"],
        "text": "请帮我分析这张图片",
        "image": "test_image_data",
        "urgency": 0.3,
        "sensitivity": 0.2
    }
    
    # 可用输出模态
    available_modalities = [
        OutputModality.TEXT,
        OutputModality.IMAGE,
        OutputModality.AUDIO,
        OutputModality.MULTIMODAL,
        OutputModality.INTERACTIVE
    ]
    
    # 选择输出模态
    result = selector.select_output_modality(user_id, input_data, available_modalities)
    
    # 打印结果
    print(f"\n选择结果:")
    result_dict = result.to_dict()
    for key, value in result_dict.items():
        if key != "context_factors" and key != "user_factors":
            print(f"  {key}: {value}")
    
    # 模拟用户反馈
    selector.record_user_feedback(
        user_id=user_id,
        result=result,
        satisfaction=0.8,
        learning_effect=0.7
    )
    
    # 打印统计信息
    print(f"\n统计信息:")
    for key, value in selector.stats.items():
        print(f"  {key}: {value}")
    
    # 打印用户画像
    if user_id in selector.user_profiles:
        profile = selector.user_profiles[user_id]
        print(f"\n用户画像:")
        profile_dict = profile.to_dict()
        for key, value in profile_dict.items():
            if key != "modality_preferences":
                print(f"  {key}: {value}")
        
        print(f"  模态偏好:")
        for modality, score in profile.modality_preferences.items():
            print(f"    {modality}: {score:.2f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_intelligent_output_selector()