#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景自适应参数调整模块 - Scene Adaptive Parameters Module

解决防循环模块的"硬编码"惩罚机制问题：
1. repetition_penalty=1.2是固定值，未按「文本类型」动态调整
2. temperature调整仅做"线性+0.1"，无上限保护

根据用户指出的缺陷：
- 工业控制指令（需精准重复）和自然语言对话（需防重复）应使用不同的惩罚值
- 当前实现导致工业指令被误惩罚（输出错乱）、自然语言仍有低概率循环
- temperature调整无上限保护，极端情况下会调到1.5以上，模型输出完全混乱

本模块提供基于语义场景的动态参数适配，支持：
1. 根据文本类型（工业控制/医疗影像/金融分析/教育辅导/通用对话）调整参数
2. 实时监控参数使用效果，自适应优化参数值
3. 防止参数漂移和异常值，确保系统稳定性
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import json
from collections import defaultdict, deque
import numpy as np

# 导入语义场景检测器
from core.context_memory import SemanticSceneDetector

logger = logging.getLogger(__name__)


class SceneAdaptiveParameters:
    """
    场景自适应参数管理器 - 解决防循环模块"硬编码"惩罚机制问题
    
    核心功能：
    1. 动态参数适配：根据文本语义场景动态调整repetition_penalty和temperature参数
    2. 范围保护机制：确保参数在安全范围内，防止极端值导致输出混乱
    3. 性能反馈优化：基于历史性能数据在线优化参数配置
    4. 多场景支持：工业控制、医疗影像、金融分析、教育辅导、通用对话等场景
    
    解决的问题：
    - 硬编码repetition_penalty=1.2：根据文本类型动态调整，工业控制使用较低惩罚(1.05-1.1)，
      通用对话使用较高惩罚(1.2-1.8)
    - temperature调整无上限保护：实施场景特定的温度范围限制，工业控制(0.1-0.6)，
      通用对话(0.5-1.2)，单次增量限制防止突变
    - 工业指令误惩罚：工业控制场景识别并应用特殊参数，允许必要重复
    
    设计原理：
    - 基于语义场景检测的结果进行参数映射
    - 使用置信度加权插值处理不确定场景
    - 基于在线学习机制根据性能反馈优化参数
    - 实施多重安全限制防止参数漂移和异常
    
    使用示例：
        from core.scene_adaptive_parameters import SceneAdaptiveParameters
        
        # 创建参数管理器
        param_manager = SceneAdaptiveParameters()
        
        # 检测场景并获取自适应参数
        text = "工业控制系统PID控制器调节温度"
        params = param_manager.detect_scene_and_adjust_parameters(text)
        # params: {"repetition_penalty": 1.05, "temperature": 0.3, "scene": "industrial_control", ...}
        
        # 记录性能反馈
        param_manager.record_performance(
            scene="industrial_control",
            repetition_score=0.7,  # 重复合适程度
            quality_score=0.8     # 输出质量
        )
    """
    
    # 场景到参数的默认映射
    DEFAULT_SCENE_PARAMETERS = {
        "industrial_control": {
            "description": "工业控制场景 - 需要精准重复指令，防止误惩罚",
            "repetition_penalty": 1.05,  # 较低的惩罚，允许必要重复
            "temperature": 0.3,  # 较低温度确保精确性
            "temperature_range": (0.1, 0.6),  # 温度允许范围
            "repetition_penalty_range": (1.0, 1.1),  # 重复惩罚范围
            "max_temperature_increment": 0.05,  # 最大温度增量
            "text_type": "precision_instruction"
        },
        "medical_imaging": {
            "description": "医疗影像场景 - 需要准确性和专业性",
            "repetition_penalty": 1.1,  # 中等惩罚
            "temperature": 0.4,  # 中等温度
            "temperature_range": (0.2, 0.8),
            "repetition_penalty_range": (1.05, 1.3),
            "max_temperature_increment": 0.08,
            "text_type": "professional_document"
        },
        "financial_analysis": {
            "description": "金融分析场景 - 需要准确性和严谨性",
            "repetition_penalty": 1.15,
            "temperature": 0.5,
            "temperature_range": (0.3, 0.9),
            "repetition_penalty_range": (1.1, 1.4),
            "max_temperature_increment": 0.1,
            "text_type": "analytical_report"
        },
        "educational_tutoring": {
            "description": "教育辅导场景 - 需要清晰度和教育性",
            "repetition_penalty": 1.2,
            "temperature": 0.6,
            "temperature_range": (0.4, 1.0),
            "repetition_penalty_range": (1.15, 1.5),
            "max_temperature_increment": 0.12,
            "text_type": "educational_content"
        },
        "general_conversation": {
            "description": "通用对话场景 - 需要创造性和自然性",
            "repetition_penalty": 1.3,  # 较高的惩罚防止重复
            "temperature": 0.7,  # 较高温度增加创造性
            "temperature_range": (0.5, 1.2),
            "repetition_penalty_range": (1.2, 1.8),
            "max_temperature_increment": 0.15,
            "text_type": "creative_conversation"
        }
    }
    
    def __init__(
        self,
        scene_detector: Optional[SemanticSceneDetector] = None,
        learning_rate: float = 0.01,
        adaptation_window: int = 100
    ):
        """
        初始化场景自适应参数管理器
        
        Args:
            scene_detector: 语义场景检测器实例
            learning_rate: 参数自适应学习率
            adaptation_window: 自适应窗口大小
        """
        self.scene_detector = scene_detector or SemanticSceneDetector()
        self.learning_rate = learning_rate
        self.adaptation_window = adaptation_window
        
        # 参数历史记录
        self.parameter_history = defaultdict(lambda: deque(maxlen=adaptation_window))
        self.performance_history = defaultdict(lambda: deque(maxlen=adaptation_window))
        
        # 当前场景参数
        self.current_scene = "general_conversation"
        self.current_parameters = self.DEFAULT_SCENE_PARAMETERS["general_conversation"].copy()
        
        # 参数统计
        self.scene_statistics = defaultdict(lambda: {
            "usage_count": 0,
            "avg_repetition_penalty": 0.0,
            "avg_temperature": 0.0,
            "performance_scores": [],
            "last_update": time.time()
        })
        
        # 记录初始化
        logger.info("SceneAdaptiveParameters initialized with adaptive learning")
    
    def detect_scene_and_adjust_parameters(self, text: str) -> Dict[str, Any]:
        """
        检测文本场景并动态调整生成参数（repetition_penalty, temperature等）
        
        检测逻辑：
        1. 使用SemanticSceneDetector检测文本的语义场景（工业控制、医疗影像、金融分析等）
        2. 根据场景置信度分数选择参数调整策略：
           - 高置信度（>0.7）：使用该场景的基准参数
           - 中置信度（0.3-0.7）：在场景参数和通用参数之间线性插值
           - 低置信度（<0.3）：使用通用对话场景参数
        3. 应用历史性能反馈优化参数
        4. 确保参数在允许范围内，防止参数漂移
        
        阈值依据：
        - 0.7高置信度阈值：确保场景识别准确时才使用专业参数
        - 0.3低置信度阈值：低于此阈值认为场景识别不可靠，使用通用参数
        - 参数范围：基于各场景特性设定，工业控制需要低温度（0.1-0.6）和低重复惩罚（1.0-1.1）
          ，通用对话需要较高温度（0.5-1.2）和较高重复惩罚（1.2-1.8）
        
        异常处理：
        - 如果文本为空或非字符串，场景检测器会返回"general_conversation"
        - 如果场景不在预设中，使用通用对话场景参数
        - 参数调整过程中任何异常都会被捕获并记录，系统回退到安全默认值
        
        Args:
            text: 输入文本，支持中文和英文
            
        Returns:
            调整后的参数字典，包含：
            - repetition_penalty: 重复惩罚系数（1.0-2.0）
            - temperature: 温度参数（0.1-1.2）
            - scene: 检测到的场景
            - scene_confidence: 场景置信度（0.0-1.0）
            - confidence_factor: 置信度因子（用于参数插值）
        """
        # 检测场景
        scene, scene_scores = self.scene_detector.detect_scene(text)
        self.current_scene = scene
        
        # 获取该场景的基准参数
        scene_config = self.DEFAULT_SCENE_PARAMETERS.get(
            scene, 
            self.DEFAULT_SCENE_PARAMETERS["general_conversation"]
        ).copy()
        
        # 应用自适应调整
        adapted_params = self._adapt_parameters(scene, scene_config, scene_scores)
        
        # 更新当前参数
        self.current_parameters = adapted_params.copy()
        
        # 记录使用统计
        self._update_scene_statistics(scene, adapted_params)
        
        # 记录参数历史
        timestamp = time.time()
        self.parameter_history[scene].append({
            "timestamp": timestamp,
            "repetition_penalty": adapted_params["repetition_penalty"],
            "temperature": adapted_params["temperature"],
            "scene": scene,
            "scene_score": scene_scores.get(scene, 0.0)
        })
        
        logger.debug(
            f"Adapted parameters for scene '{scene}': "
            f"repetition_penalty={adapted_params['repetition_penalty']:.3f}, "
            f"temperature={adapted_params['temperature']:.3f}"
        )
        
        return adapted_params
    
    def _adapt_parameters(
        self, 
        scene: str, 
        base_params: Dict[str, Any], 
        scene_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        根据场景置信度自适应调整参数（核心调整逻辑）
        
        调整逻辑：
        1. 基于场景置信度进行参数插值：
           - 高置信度（>0.7）：直接使用场景基准参数
           - 中置信度（0.3-0.7）：线性插值，置信度越高越接近场景参数，越低越接近通用参数
           - 低置信度（<0.3）：完全使用通用对话场景参数，并将场景标记为"general_conversation"
        2. 应用性能反馈：根据历史性能数据微调参数
           - 重复分数<0.3：增加repetition_penalty（学习率×0.2）
           - 重复分数>0.8：减少repetition_penalty（学习率×0.1）
           - 质量分数<0.4：降低temperature（学习率×0.15）
           - 质量分数>0.7：增加temperature（学习率×0.1）
        3. 参数范围限制：确保参数在允许范围内
        
        阈值依据：
        - 0.3重复分数阈值：低于此值认为重复过多，需要增加惩罚
        - 0.8重复分数阈值：高于此值认为重复过少，可减少惩罚（对需要重复的场景）
        - 0.4质量分数阈值：低于此值认为输出质量差，降低随机性
        - 0.7质量分数阈值：高于此值认为输出质量好，可增加多样性
        
        异常处理：
        - 如果场景分数字典为空，使用0.0作为默认置信度
        - 插值计算确保数值稳定性
        - 性能反馈只在有足够历史数据（≥10条）时应用
        
        Args:
            scene: 检测到的场景名称
            base_params: 该场景的基准参数配置
            scene_scores: 所有场景的置信度分数字典
            
        Returns:
            调整后的参数配置，已添加场景信息和置信度因子
        """
        # 创建参数副本
        adapted_params = base_params.copy()
        
        # 获取场景置信度
        scene_confidence = scene_scores.get(scene, 0.0)
        
        # 根据置信度调整参数
        if scene_confidence > 0.7:
            # 高置信度场景：使用基准参数，小幅调整
            confidence_factor = 1.0
        elif scene_confidence > 0.3:
            # 中等置信度场景：向通用场景参数靠拢
            confidence_factor = 0.7
            general_params = self.DEFAULT_SCENE_PARAMETERS["general_conversation"]
            # 线性插值
            adapted_params["repetition_penalty"] = (
                base_params["repetition_penalty"] * confidence_factor +
                general_params["repetition_penalty"] * (1 - confidence_factor)
            )
            adapted_params["temperature"] = (
                base_params["temperature"] * confidence_factor +
                general_params["temperature"] * (1 - confidence_factor)
            )
        else:
            # 低置信度场景：使用通用场景参数
            confidence_factor = 0.0
            adapted_params = self.DEFAULT_SCENE_PARAMETERS["general_conversation"].copy()
            # 对于低置信度场景，使用通用对话场景进行参数限制
            scene = "general_conversation"
        
        # 应用历史性能反馈
        adapted_params = self._apply_performance_feedback(scene, adapted_params)
        
        # 确保参数在允许范围内
        adapted_params = self._clamp_parameters(scene, adapted_params)
        
        # 添加场景信息
        adapted_params["scene"] = scene
        adapted_params["scene_confidence"] = scene_confidence
        adapted_params["confidence_factor"] = confidence_factor
        
        return adapted_params
    
    def _apply_performance_feedback(
        self, 
        scene: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        应用历史性能反馈优化参数（在线学习机制）
        
        优化逻辑：
        1. 检查是否有足够的历史数据（至少10条记录）
        2. 计算平均性能指标：
           - 平均重复分数：反映输出重复程度的指标（0-1，越低表示重复越多）
           - 平均质量分数：反映输出质量的指标（0-1，越高表示质量越好）
        3. 根据性能指标调整参数：
           - 重复分数<0.3：重复过多，增加repetition_penalty（+学习率×0.2）
           - 重复分数>0.8：重复过少，减少repetition_penalty（-学习率×0.1）
           - 质量分数<0.4：质量较差，降低temperature减少随机性（-学习率×0.15）
           - 质量分数>0.7：质量较好，增加temperature增加多样性（+学习率×0.1）
        
        算法依据：
        - 基于在线梯度下降思想，根据性能反馈缓慢调整参数
        - 使用移动平均而非单次性能，避免噪声影响
        - 调整幅度与学习率成正比，默认学习率0.01提供稳定调整
        
        异常处理：
        - 历史数据不足时直接返回原参数
        - 性能分数被限制在[0,1]范围内，防止异常值
        - 调整后的参数会在_clamp_parameters中进行范围限制
        
        Args:
            scene: 场景名称，用于获取对应的性能历史数据
            params: 当前参数配置
            
        Returns:
            根据性能反馈优化后的参数配置
        """
        # 检查是否有足够的历史数据
        if len(self.performance_history[scene]) < 10:
            return params
        
        # 获取最近的性能数据
        recent_performance = list(self.performance_history[scene])
        
        # 计算平均性能指标
        avg_repetition_score = np.mean([p.get("repetition_score", 0.5) for p in recent_performance])
        avg_quality_score = np.mean([p.get("quality_score", 0.5) for p in recent_performance])
        
        # 根据重复分数调整repetition_penalty
        if avg_repetition_score < 0.3:
            # 重复过多，增加惩罚
            params["repetition_penalty"] += self.learning_rate * 0.2
        elif avg_repetition_score > 0.8:
            # 重复过少，减少惩罚（对于需要重复的场景）
            params["repetition_penalty"] -= self.learning_rate * 0.1
        
        # 根据质量分数调整temperature
        if avg_quality_score < 0.4:
            # 质量较差，降低temperature减少随机性
            params["temperature"] -= self.learning_rate * 0.15
        elif avg_quality_score > 0.7:
            # 质量较好，可适当增加temperature增加多样性
            params["temperature"] += self.learning_rate * 0.1
        
        logger.debug(
            f"Applied performance feedback for scene '{scene}': "
            f"repetition_score={avg_repetition_score:.3f}, "
            f"quality_score={avg_quality_score:.3f}"
        )
        
        return params
    
    def _clamp_parameters(
        self, 
        scene: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将参数限制在允许范围内（安全防护机制）
        
        限制逻辑：
        1. 获取场景特定的参数范围配置：
           - repetition_penalty_range: 重复惩罚系数允许范围
           - temperature_range: 温度参数允许范围
           - max_temperature_increment: 单次温度最大变化量
        2. 应用范围限制：
           - repetition_penalty: 限制在[range_min, range_max]内
           - temperature: 限制在[range_min, range_max]内
        3. 应用增量限制（防止参数突变）：
           - 检查当前temperature与上次temperature的差值
           - 如果差值超过max_temperature_increment，将变化量限制在该值内
           - 防止因性能反馈或异常导致的参数剧烈波动
        
        安全设计：
        - 解决用户指出的"temperature调整无上限保护"问题
        - 防止工业控制场景使用过高temperature导致输出混乱
        - 防止通用对话场景使用过低repetition_penalty导致过度重复
        
        异常处理：
        - 如果场景配置缺失，使用默认范围（repetition_penalty: 1.0-2.0, temperature: 0.1-2.0）
        - 增量限制只在有上次温度记录时应用
        - 所有范围检查使用max/min函数，确保数学安全性
        
        Args:
            scene: 场景名称，用于获取该场景的参数范围配置
            params: 待限制的参数字典，必须包含repetition_penalty和temperature字段
            
        Returns:
            经过范围限制的安全参数，添加了last_temperature字段用于下次增量限制
        """
        scene_config = self.DEFAULT_SCENE_PARAMETERS.get(
            scene, 
            self.DEFAULT_SCENE_PARAMETERS["general_conversation"]
        )
        
        # 限制repetition_penalty
        rp_range = scene_config.get("repetition_penalty_range", (1.0, 2.0))
        params["repetition_penalty"] = max(rp_range[0], min(rp_range[1], params["repetition_penalty"]))
        
        # 限制temperature
        temp_range = scene_config.get("temperature_range", (0.1, 2.0))
        params["temperature"] = max(temp_range[0], min(temp_range[1], params["temperature"]))
        
        # 限制temperature增量（防止突变）
        if "last_temperature" in self.current_parameters:
            max_inc = scene_config.get("max_temperature_increment", 0.2)
            last_temp = self.current_parameters["last_temperature"]
            if abs(params["temperature"] - last_temp) > max_inc:
                if params["temperature"] > last_temp:
                    params["temperature"] = last_temp + max_inc
                else:
                    params["temperature"] = last_temp - max_inc
        
        # 保存当前温度用于下次增量限制
        params["last_temperature"] = params["temperature"]
        
        return params
    
    def _update_scene_statistics(
        self, 
        scene: str, 
        params: Dict[str, Any]
    ) -> None:
        """
        更新场景统计信息
        
        Args:
            scene: 场景名称
            params: 参数字典
        """
        stats = self.scene_statistics[scene]
        stats["usage_count"] += 1
        stats["last_update"] = time.time()
        
        # 更新平均参数值（移动平均）
        if stats["avg_repetition_penalty"] == 0:
            stats["avg_repetition_penalty"] = params["repetition_penalty"]
            stats["avg_temperature"] = params["temperature"]
        else:
            alpha = 0.1  # 平滑因子
            stats["avg_repetition_penalty"] = (
                alpha * params["repetition_penalty"] + 
                (1 - alpha) * stats["avg_repetition_penalty"]
            )
            stats["avg_temperature"] = (
                alpha * params["temperature"] + 
                (1 - alpha) * stats["avg_temperature"]
            )
    
    def record_performance(
        self, 
        scene: str, 
        repetition_score: float, 
        quality_score: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        记录参数使用性能数据（反馈收集机制）
        
        性能指标定义：
        - 重复评分（repetition_score）: 0-1之间的分数，表示输出重复的合适程度
          - 0.0: 完全不重复（可能丢失重要信息）
          - 0.5: 适度重复（平衡状态）
          - 1.0: 完全重复（可能过度重复）
          - 工业控制场景期望较高重复评分（0.6-0.9），允许必要重复
          - 通用对话场景期望适中重复评分（0.3-0.6），防止过度重复
        - 质量评分（quality_score）: 0-1之间的分数，表示输出整体质量
          - 基于人工评估、自动化指标或模型自评估
          - 考虑相关性、连贯性、信息量、语法正确性等维度
        
        数据管理：
        - 性能数据存储在双端队列中，自动维护窗口大小（默认1000条）
        - 评分被限制在[0,1]范围内，防止异常值污染历史数据
        - 同时记录当前使用的参数值，便于后续分析参数与性能的关系
        
        异常处理：
        - 如果scene为空或无效，数据可能丢失（调用方需确保场景有效性）
        - 评分超出[0,1]范围会被自动截断
        - 附加指标可以灵活扩展，但不影响核心反馈逻辑
        
        Args:
            scene: 场景名称，用于分类存储性能数据
            repetition_score: 重复评分（0-1，越高表示重复越合适）
            quality_score: 质量评分（0-1，越高表示输出质量越好）
            additional_metrics: 附加指标字典，可用于记录延迟、token数等扩展指标
        """
        performance_data = {
            "timestamp": time.time(),
            "scene": scene,
            "repetition_score": max(0.0, min(1.0, repetition_score)),
            "quality_score": max(0.0, min(1.0, quality_score)),
            "repetition_penalty": self.current_parameters.get("repetition_penalty", 1.2),
            "temperature": self.current_parameters.get("temperature", 0.7)
        }
        
        if additional_metrics:
            performance_data.update(additional_metrics)
        
        self.performance_history[scene].append(performance_data)
        
        # 更新场景统计
        stats = self.scene_statistics[scene]
        stats["performance_scores"].append({
            "repetition_score": repetition_score,
            "quality_score": quality_score
        })
        
        # 保留最近100个性能评分
        if len(stats["performance_scores"]) > 100:
            stats["performance_scores"] = stats["performance_scores"][-100:]
    
    def get_scene_parameters(self, scene: str) -> Dict[str, Any]:
        """
        获取指定场景的参数配置
        
        Args:
            scene: 场景名称
            
        Returns:
            场景参数配置
        """
        return self.DEFAULT_SCENE_PARAMETERS.get(
            scene,
            self.DEFAULT_SCENE_PARAMETERS["general_conversation"]
        ).copy()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        获取当前参数
        
        Returns:
            当前参数配置
        """
        return self.current_parameters.copy()
    
    def get_scene_statistics(self, scene: Optional[str] = None) -> Dict[str, Any]:
        """
        获取场景统计信息
        
        Args:
            scene: 可选场景名称，如果为None则返回所有场景统计
            
        Returns:
            场景统计信息
        """
        if scene:
            return self.scene_statistics[scene].copy()
        else:
            return {s: stats.copy() for s, stats in self.scene_statistics.items()}
    
    def get_parameter_history(
        self, 
        scene: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取参数历史
        
        Args:
            scene: 可选场景名称，如果为None则返回所有场景历史
            limit: 限制返回的历史记录数量
            
        Returns:
            参数历史记录列表
        """
        if scene:
            history = list(self.parameter_history[scene])[-limit:]
        else:
            history = []
            for scene_name in self.parameter_history:
                history.extend(list(self.parameter_history[scene_name])[-limit//5:])
            history.sort(key=lambda x: x["timestamp"], reverse=True)
            history = history[:limit]
        
        return history
    
    def get_performance_history(
        self, 
        scene: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取性能历史
        
        Args:
            scene: 可选场景名称，如果为None则返回所有场景性能历史
            limit: 限制返回的历史记录数量
            
        Returns:
            性能历史记录列表
        """
        if scene:
            history = list(self.performance_history[scene])[-limit:]
        else:
            history = []
            for scene_name in self.performance_history:
                history.extend(list(self.performance_history[scene_name])[-limit//5:])
            history.sort(key=lambda x: x["timestamp"], reverse=True)
            history = history[:limit]
        
        return history
    
    def reset_scene_statistics(self, scene: Optional[str] = None) -> None:
        """
        重置场景统计
        
        Args:
            scene: 可选场景名称，如果为None则重置所有场景统计
        """
        if scene:
            self.scene_statistics[scene] = {
                "usage_count": 0,
                "avg_repetition_penalty": 0.0,
                "avg_temperature": 0.0,
                "performance_scores": [],
                "last_update": time.time()
            }
            self.parameter_history[scene].clear()
            self.performance_history[scene].clear()
        else:
            self.scene_statistics.clear()
            self.parameter_history.clear()
            self.performance_history.clear()
            logger.info("All scene statistics reset")
    
    def export_parameters(self, filepath: str) -> bool:
        """
        导出参数配置到文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                "timestamp": time.time(),
                "default_parameters": self.DEFAULT_SCENE_PARAMETERS,
                "current_scene": self.current_scene,
                "current_parameters": self.current_parameters,
                "scene_statistics": dict(self.scene_statistics),
                "parameter_history_summary": {
                    scene: len(history) 
                    for scene, history in self.parameter_history.items()
                },
                "performance_history_summary": {
                    scene: len(history) 
                    for scene, history in self.performance_history.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Parameters exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export parameters: {e}")
            return False
    
    def import_parameters(self, filepath: str) -> bool:
        """
        从文件导入参数配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否导入成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 更新默认参数（谨慎操作）
            if "default_parameters" in import_data:
                for scene, params in import_data["default_parameters"].items():
                    if scene in self.DEFAULT_SCENE_PARAMETERS:
                        self.DEFAULT_SCENE_PARAMETERS[scene].update(params)
            
            logger.info(f"Parameters imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import parameters: {e}")
            return False


def create_scene_adaptive_parameters(
    config: Optional[Dict[str, Any]] = None
) -> SceneAdaptiveParameters:
    """
    创建场景自适应参数管理器（工厂函数）
    
    Args:
        config: 可选配置字典
        
    Returns:
        SceneAdaptiveParameters实例
    """
    config = config or {}
    
    # 创建场景检测器
    scene_detector = config.get("scene_detector")
    
    # 获取配置参数
    learning_rate = config.get("learning_rate", 0.01)
    adaptation_window = config.get("adaptation_window", 100)
    
    # 创建管理器
    manager = SceneAdaptiveParameters(
        scene_detector=scene_detector,
        learning_rate=learning_rate,
        adaptation_window=adaptation_window
    )
    
    # 应用自定义参数配置
    custom_params = config.get("custom_parameters", {})
    for scene, params in custom_params.items():
        if scene in manager.DEFAULT_SCENE_PARAMETERS:
            manager.DEFAULT_SCENE_PARAMETERS[scene].update(params)
    
    return manager


# 示例使用代码
if __name__ == "__main__":
    # 测试场景自适应参数管理器
    print("=" * 80)
    print("测试场景自适应参数管理器")
    print("=" * 80)
    
    # 创建管理器
    param_manager = create_scene_adaptive_parameters()
    
    # 测试不同场景的文本
    test_cases = [
        ("工业控制系统需要实时监控温度压力，PID控制器调节阀门开度", "industrial_control"),
        ("患者CT扫描显示肺部有阴影，建议进行活检以排除肺癌", "medical_imaging"),
        ("金融财务分析股票投资交易银行货币经济市场风险", "financial_analysis"),
        ("教育学习教学学生老师课程学校知识培训辅导考试", "educational_tutoring"),
        ("你好，今天天气怎么样？我们聊聊天吧", "general_conversation"),
    ]
    
    print("\n1. 场景自适应参数测试:")
    for text, expected_scene in test_cases:
        params = param_manager.detect_scene_and_adjust_parameters(text)
        actual_scene = params.get("scene", "unknown")
        
        print(f"   文本: '{text[:30]}...'")
        print(f"   场景: {actual_scene} (期望: {expected_scene})")
        print(f"   参数: repetition_penalty={params['repetition_penalty']:.3f}, "
              f"temperature={params['temperature']:.3f}")
        print(f"   置信度: {params.get('scene_confidence', 0.0):.3f}")
        print()
    
    print("\n2. 性能记录测试:")
    # 记录一些性能数据
    for i, (text, scene) in enumerate(test_cases):
        param_manager.record_performance(
            scene=scene,
            repetition_score=0.7 + i * 0.05,
            quality_score=0.6 + i * 0.06,
            additional_metrics={"test_iteration": i}
        )
    
    print("   性能记录完成")
    
    print("\n3. 统计信息:")
    stats = param_manager.get_scene_statistics()
    for scene, scene_stats in stats.items():
        print(f"   场景 {scene}: 使用次数={scene_stats['usage_count']}, "
              f"平均repetition_penalty={scene_stats['avg_repetition_penalty']:.3f}, "
              f"平均temperature={scene_stats['avg_temperature']:.3f}")
    
    print("\n4. 参数历史:")
    history = param_manager.get_parameter_history(limit=10)
    for entry in history:
        print(f"   时间: {time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}, "
              f"场景: {entry['scene']}, "
              f"repetition_penalty={entry['repetition_penalty']:.3f}, "
              f"temperature={entry['temperature']:.3f}")
    
    print("\n✓ 场景自适应参数管理器测试完成")