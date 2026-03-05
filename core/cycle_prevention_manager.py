#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防循环管理器 - Cycle Prevention Manager
整合嵌入式思维基础防护与场景自适应高级防护

基于用户建议的大厂防循环方案 + 现有的SceneAdaptiveParameters系统
实现双层防循环防护：
1. 基础层：嵌入式思维防护（缓冲区清理+重复检测+温度调节+看门狗重置）
2. 高级层：场景自适应防护（语义场景检测+动态参数调整+性能反馈优化）

解决用户指出的"防循环模块硬编码"问题，提供：
- 简单可靠的嵌入式防护（一用就灵）
- 智能自适应的场景防护（按需调整）
- 可扩展的防护策略（支持所有功能模型）
"""

import re
import logging
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, Callable
import time

from .scene_adaptive_parameters import SceneAdaptiveParameters

logger = logging.getLogger(__name__)


class CyclePreventionManager:
    """
    防循环管理器 - 整合嵌入式基础防护与场景自适应高级防护
    
    核心设计原则：
    1. 嵌入式思维：像单片机一样可靠（缓冲区清理+重复检测+看门狗重置）
    2. 分层防护：基础防护确保基本安全，高级防护提供智能优化
    3. 统一接口：所有功能模型使用同一套防循环接口
    4. 自适应调整：根据场景和性能动态调整防护策略
    
    解决的问题：
    - 硬编码repetition_penalty：动态调整，工业控制低惩罚，通用对话高惩罚
    - temperature无上限保护：基础层限制增量，高级层限制范围
    - 工业指令误惩罚：场景识别+特殊参数
    - 循环输出问题：重复检测+看门狗重置
    """
    
    # ==================== 基础层参数（嵌入式思维配置）====================
    # 这些参数对应嵌入式系统的硬件配置，简单可靠
    BASE_CONFIG = {
        # 缓冲区清理配置（对应单片机环形缓冲区）
        "history_buffer_size": 10,           # 对话历史缓冲区大小（最多存10轮）
        "repeat_threshold": 3,               # 重复检测阈值（连续3次重复触发防护）
        "max_retry_attempts": 3,             # 最大重试次数（看门狗重置次数）
        
        # 温度调节配置（对应PID阻尼系数）
        "base_temperature": 0.7,             # 基础温度（大厂最优值）
        "temperature_increment": 0.1,        # 温度增量（每次循环增加）
        "max_temperature": 1.0,              # 最大温度限制（防止过度随机）
        
        # 重复惩罚配置（对应电机反电动势）
        "base_repetition_penalty": 1.2,      # 基础重复惩罚（抑制重复）
        "penalty_increment": 0.05,           # 惩罚增量（每次循环增加）
        "max_repetition_penalty": 1.5,       # 最大重复惩罚限制
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_adaptive_layer: bool = True
    ):
        """
        初始化防循环管理器
        
        Args:
            config: 配置字典，可覆盖BASE_CONFIG中的默认值
            enable_adaptive_layer: 是否启用场景自适应高级防护
        """
        # 合并配置
        self.config = {**self.BASE_CONFIG, **(config or {})}
        self.enable_adaptive_layer = enable_adaptive_layer
        
        # ==================== 基础层初始化（嵌入式思维）====================
        # 1. 对话历史缓冲区（环形缓冲区，防止上下文爆炸）
        self.chat_history = deque(maxlen=self.config["history_buffer_size"])
        
        # 2. 最近输出队列（用于检测循环）
        self.last_outputs = deque(maxlen=self.config["repeat_threshold"])
        
        # 3. 当前防护状态
        self.current_temperature = self.config["base_temperature"]
        self.current_repetition_penalty = self.config["base_repetition_penalty"]
        self.retry_count = 0
        
        # ==================== 高级层初始化（场景自适应）====================
        if self.enable_adaptive_layer:
            self.scene_adaptive_params = SceneAdaptiveParameters()
            logger.info("CyclePreventionManager: 场景自适应高级防护已启用")
        else:
            self.scene_adaptive_params = None
            logger.info("CyclePreventionManager: 仅使用嵌入式基础防护")
        
        # 防护统计
        self.protection_stats = {
            "total_generations": 0,
            "cycle_detections": 0,
            "retry_attempts": 0,
            "temperature_adjustments": 0,
            "penalty_adjustments": 0,
            "scene_adaptations": 0,
        }
        
        logger.info(f"CyclePreventionManager 初始化完成 (buffer_size={self.config['history_buffer_size']}, "
                   f"repeat_threshold={self.config['repeat_threshold']})")
    
    def detect_cycle_basic(self, text: str) -> bool:
        """
        基础层循环检测（嵌入式故障检测）
        
        检测逻辑：
        1. 清理文本（去标点、转小写）
        2. 检查最近输出中重复的次数
        3. 超过阈值认为陷入循环
        
        对应嵌入式系统的"故障检测中断"
        
        Args:
            text: 待检测的文本
            
        Returns:
            bool: 是否检测到循环
        """
        if not text or not isinstance(text, str):
            return False
        
        # 清理文本（去标点、空格，只看核心内容）
        # 避免格式差异导致误判，就像嵌入式去抖动滤波
        clean_text = re.sub(r'[^\w\s]', '', text).strip().lower()
        if not clean_text:
            return False
        
        # 统计最近输出中重复的次数
        repeat_count = 0
        for output in self.last_outputs:
            clean_output = re.sub(r'[^\w\s]', '', output).strip().lower()
            if clean_text == clean_output:
                repeat_count += 1
        
        # 超过阈值就是循环（连续N次重复 = 故障）
        cycle_detected = repeat_count >= self.config["repeat_threshold"]
        
        if cycle_detected:
            logger.warning(f"基础层检测到循环: 重复次数={repeat_count}, 阈值={self.config['repeat_threshold']}")
            self.protection_stats["cycle_detections"] += 1
        
        return cycle_detected
    
    def apply_basic_protection(self, cycle_detected: bool) -> Dict[str, Any]:
        """
        应用基础层防护措施（看门狗重置+参数调整）
        
        防护逻辑：
        1. 检测到循环：增加温度（加随机性），增加惩罚，记录重试次数
        2. 温度超过上限：重置温度，清理最早的历史（彻底重置）
        3. 重试次数超限：重置所有参数（看门狗触发）
        
        对应嵌入式系统的"故障处理程序"
        
        Args:
            cycle_detected: 是否检测到循环
            
        Returns:
            Dict[str, Any]: 调整后的基础参数
        """
        protection_applied = False
        
        if cycle_detected:
            # 看门狗触发：检测到故障，开始处理
            self.retry_count += 1
            self.protection_stats["retry_attempts"] += 1
            
            logger.warning(f"基础层防护触发: 重试次数={self.retry_count}, "
                          f"当前温度={self.current_temperature:.3f}, "
                          f"当前惩罚={self.current_repetition_penalty:.3f}")
            
            # 1. 增加温度（加随机性，防止PID式震荡）
            old_temp = self.current_temperature
            self.current_temperature += self.config["temperature_increment"]
            if self.current_temperature > self.config["max_temperature"]:
                # 温度超过上限，重置并清理最早的历史
                self.current_temperature = self.config["base_temperature"]
                if self.chat_history:
                    self.chat_history.pop()  # 清理最早的历史，彻底重置上下文
                logger.info("温度超过上限，重置基础温度并清理历史")
            self.protection_stats["temperature_adjustments"] += 1
            
            # 2. 增加重复惩罚（抑制重复，类似电机反电动势）
            old_penalty = self.current_repetition_penalty
            self.current_repetition_penalty += self.config["penalty_increment"]
            if self.current_repetition_penalty > self.config["max_repetition_penalty"]:
                self.current_repetition_penalty = self.config["base_repetition_penalty"]
            self.protection_stats["penalty_adjustments"] += 1
            
            # 3. 检查重试次数限制
            if self.retry_count >= self.config["max_retry_attempts"]:
                # 重试次数超限，看门狗完全重置
                self._reset_basic_protection()
                logger.warning(f"看门狗完全重置: 重试次数超限 ({self.retry_count})")
            
            protection_applied = True
            logger.info(f"基础参数调整: 温度 {old_temp:.3f}→{self.current_temperature:.3f}, "
                       f"惩罚 {old_penalty:.3f}→{self.current_repetition_penalty:.3f}")
        
        return {
            "temperature": self.current_temperature,
            "repetition_penalty": self.current_repetition_penalty,
            "cycle_detected": cycle_detected,
            "retry_count": self.retry_count,
            "protection_applied": protection_applied,
            "protection_layer": "basic"
        }
    
    def _reset_basic_protection(self) -> None:
        """重置基础层防护状态（看门狗完全重置）"""
        self.current_temperature = self.config["base_temperature"]
        self.current_repetition_penalty = self.config["base_repetition_penalty"]
        self.retry_count = 0
        self.last_outputs.clear()
        if self.chat_history:
            # 清理一半的历史，保留一些上下文
            keep_count = len(self.chat_history) // 2
            while len(self.chat_history) > keep_count:
                self.chat_history.pop()
        logger.info("基础层防护状态已完全重置")
    
    def get_adaptive_parameters(self, text: str) -> Dict[str, Any]:
        """
        获取高级层自适应参数（场景智能调整）
        
        如果启用了场景自适应层：
        1. 检测文本的语义场景
        2. 根据场景获取自适应参数
        3. 与基础层参数融合
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 自适应参数配置
        """
        if not self.enable_adaptive_layer or not self.scene_adaptive_params:
            return {
                "temperature": self.current_temperature,
                "repetition_penalty": self.current_repetition_penalty,
                "scene": "general_conversation",
                "scene_confidence": 0.0,
                "protection_layer": "basic_only"
            }
        
        try:
            # 获取场景自适应参数
            adaptive_params = self.scene_adaptive_params.detect_scene_and_adjust_parameters(text)
            self.protection_stats["scene_adaptations"] += 1
            
            # 融合策略：优先使用自适应参数，但保留基础层的安全限制
            final_params = {
                "temperature": adaptive_params.get("temperature", self.current_temperature),
                "repetition_penalty": adaptive_params.get("repetition_penalty", self.current_repetition_penalty),
                "scene": adaptive_params.get("scene", "general_conversation"),
                "scene_confidence": adaptive_params.get("scene_confidence", 0.0),
                "confidence_factor": adaptive_params.get("confidence_factor", 1.0),
                "protection_layer": "adaptive"
            }
            
            # 确保参数在安全范围内（双重保护）
            final_params["temperature"] = min(
                final_params["temperature"],
                self.config["max_temperature"]
            )
            final_params["repetition_penalty"] = min(
                final_params["repetition_penalty"],
                self.config["max_repetition_penalty"]
            )
            
            logger.debug(f"高级层自适应参数: 场景={final_params['scene']}, "
                        f"温度={final_params['temperature']:.3f}, "
                        f"惩罚={final_params['repetition_penalty']:.3f}")
            
            return final_params
            
        except Exception as e:
            logger.error(f"获取自适应参数失败: {e}, 回退到基础参数")
            return {
                "temperature": self.current_temperature,
                "repetition_penalty": self.current_repetition_penalty,
                "scene": "general_conversation",
                "scene_confidence": 0.0,
                "protection_layer": "basic_fallback"
            }
    
    def generate_safe(
        self,
        prompt: str,
        generate_func: Callable[[str, Dict[str, Any]], str],
        max_attempts: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        安全生成文本（整合双层防护）
        
        生成流程：
        1. 更新对话历史（缓冲区清理）
        2. 构建上下文（只取最近N轮）
        3. 获取防护参数（基础+高级）
        4. 生成文本
        5. 检测循环
        6. 应用防护措施（如需）
        7. 返回安全文本
        
        Args:
            prompt: 用户输入
            generate_func: 生成函数，接受(context, params)返回文本
            max_attempts: 最大生成尝试次数，默认使用配置值
            
        Returns:
            Tuple[str, Dict[str, Any]]: (生成的文本, 防护信息)
        """
        max_attempts = max_attempts or self.config["max_retry_attempts"]
        attempts = 0
        
        # 1. 更新对话历史（缓冲区清理）
        self.chat_history.append(prompt)
        context = "\n".join(list(self.chat_history))
        
        while attempts < max_attempts:
            attempts += 1
            self.protection_stats["total_generations"] += 1
            
            # 2. 获取防护参数
            # 先获取高级层参数（场景自适应）
            adaptive_params = self.get_adaptive_parameters(prompt)
            
            # 如果高级层参数是基础回退，使用当前基础参数
            if adaptive_params["protection_layer"] == "basic_fallback":
                params = {
                    "temperature": self.current_temperature,
                    "repetition_penalty": self.current_repetition_penalty,
                    "top_p": 0.9,
                }
                protection_info = {"layer": "basic_fallback", **params}
            else:
                params = {
                    "temperature": adaptive_params["temperature"],
                    "repetition_penalty": adaptive_params["repetition_penalty"],
                    "top_p": 0.9,
                }
                protection_info = adaptive_params.copy()
            
            # 3. 生成文本
            try:
                output = generate_func(context, params)
            except Exception as e:
                logger.error(f"生成文本失败: {e}")
                # 生成失败也视为一种循环，触发防护
                output = f"生成错误: {str(e)}"
                cycle_detected = True
            else:
                # 4. 基础层循环检测
                cycle_detected = self.detect_cycle_basic(output)
            
            # 5. 应用基础层防护（如果检测到循环）
            if cycle_detected:
                basic_protection = self.apply_basic_protection(cycle_detected)
                protection_info.update({
                    "basic_protection_applied": True,
                    "basic_temperature": basic_protection["temperature"],
                    "basic_penalty": basic_protection["repetition_penalty"],
                    "retry_count": basic_protection["retry_count"]
                })
                
                # 继续重试
                if attempts < max_attempts:
                    logger.info(f"检测到循环，重新生成 (尝试 {attempts}/{max_attempts})")
                    continue
                else:
                    logger.warning(f"达到最大重试次数 ({max_attempts})，返回当前输出")
            
            # 6. 无循环或达到最大尝试次数，返回结果
            if not cycle_detected or attempts >= max_attempts:
                # 更新输出队列
                self.last_outputs.append(output)
                # 更新对话历史
                self.chat_history.append(output)
                # 重置重试计数（成功生成）
                if not cycle_detected:
                    self.retry_count = 0
                
                protection_info.update({
                    "attempts": attempts,
                    "cycle_detected": cycle_detected,
                    "history_size": len(self.chat_history),
                    "last_outputs_size": len(self.last_outputs)
                })
                
                return output, protection_info
        
        # 理论上不会执行到这里
        return "生成失败：达到最大重试次数", {
            "attempts": attempts,
            "cycle_detected": True,
            "error": "max_attempts_reached",
            "protection_layer": "emergency_fallback"
        }
    
    def record_performance(
        self,
        scene: str,
        repetition_score: float,
        quality_score: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        记录性能反馈（用于高级层优化）
        
        Args:
            scene: 场景名称
            repetition_score: 重复评分（0-1，越高表示重复越合适）
            quality_score: 质量评分（0-1，越高表示输出质量越好）
            additional_metrics: 附加指标
        """
        if self.enable_adaptive_layer and self.scene_adaptive_params:
            self.scene_adaptive_params.record_performance(
                scene=scene,
                repetition_score=repetition_score,
                quality_score=quality_score,
                additional_metrics=additional_metrics
            )
            logger.debug(f"记录性能反馈: 场景={scene}, 重复分={repetition_score:.3f}, 质量分={quality_score:.3f}")
    
    def update_history(self, text: str) -> None:
        """
        更新对话历史（手动控制）
        
        Args:
            text: 要添加到历史的文本
        """
        self.chat_history.append(text)
        logger.debug(f"更新对话历史: 当前大小={len(self.chat_history)}")
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.chat_history.clear()
        self.last_outputs.clear()
        logger.info("对话历史已清空")
    
    def get_protection_stats(self) -> Dict[str, Any]:
        """
        获取防护统计信息
        
        Returns:
            Dict[str, Any]: 防护统计
        """
        return {
            **self.protection_stats,
            "current_temperature": self.current_temperature,
            "current_repetition_penalty": self.current_repetition_penalty,
            "retry_count": self.retry_count,
            "history_size": len(self.chat_history),
            "last_outputs_size": len(self.last_outputs),
            "enable_adaptive_layer": self.enable_adaptive_layer,
            "config": self.config.copy()
        }
    
    def reset_protection_state(self) -> None:
        """重置防护状态（恢复到初始状态）"""
        self._reset_basic_protection()
        self.protection_stats = {
            "total_generations": 0,
            "cycle_detections": 0,
            "retry_attempts": 0,
            "temperature_adjustments": 0,
            "penalty_adjustments": 0,
            "scene_adaptations": 0,
        }
        logger.info("防护状态已重置")


# 单例实例
_cycle_prevention_instance = None

def get_cycle_prevention_manager(
    config: Optional[Dict[str, Any]] = None,
    enable_adaptive_layer: bool = True
) -> CyclePreventionManager:
    """
    获取防循环管理器实例（单例模式）
    
    Args:
        config: 配置字典
        enable_adaptive_layer: 是否启用场景自适应高级防护
        
    Returns:
        CyclePreventionManager实例
    """
    global _cycle_prevention_instance
    
    if _cycle_prevention_instance is None:
        _cycle_prevention_instance = CyclePreventionManager(
            config=config,
            enable_adaptive_layer=enable_adaptive_layer
        )
    
    return _cycle_prevention_instance


class MultimodalCyclePreventionManager(CyclePreventionManager):
    """
    多模态防循环管理器 - 扩展支持文本、代码、图像、音频等多种数据类型
    
    在基础嵌入式防护和场景自适应防护的基础上，增加：
    1. 多模态数据检测适配器
    2. 模态专用参数配置
    3. 跨模态循环检测
    4. 统一的多模态防护接口
    """
    
    # 数据类型枚举
    class DataType:
        TEXT = "text"           # 文本数据
        CODE = "code"           # 代码/编程数据
        IMAGE = "image"         # 图像数据（路径、张量、数组）
        AUDIO = "audio"         # 音频数据
        VECTOR = "vector"       # 向量/嵌入数据
        STRUCTURED = "structured"  # 结构化数据（JSON、XML等）
        UNKNOWN = "unknown"     # 未知类型
        
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_adaptive_layer: bool = True,
        multimodal_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化多模态防循环管理器
        
        Args:
            config: 基础配置字典
            enable_adaptive_layer: 是否启用场景自适应高级防护
            multimodal_config: 多模态专用配置
        """
        # 调用父类初始化
        super().__init__(config, enable_adaptive_layer)
        
        # 多模态专用配置
        self.multimodal_config = multimodal_config or {}
        
        # 多模态历史缓冲区（按数据类型分开）
        self.multimodal_history = {
            self.DataType.TEXT: deque(maxlen=self.config["history_buffer_size"]),
            self.DataType.CODE: deque(maxlen=self.config["history_buffer_size"]),
            self.DataType.IMAGE: deque(maxlen=self.config["history_buffer_size"] // 2),  # 图像缓冲区较小
            self.DataType.AUDIO: deque(maxlen=self.config["history_buffer_size"] // 2),  # 音频缓冲区较小
        }
        
        # 多模态输出队列
        self.multimodal_last_outputs = {
            self.DataType.TEXT: deque(maxlen=self.config["repeat_threshold"]),
            self.DataType.CODE: deque(maxlen=self.config["repeat_threshold"]),
            self.DataType.IMAGE: deque(maxlen=self.config["repeat_threshold"] // 2),
            self.DataType.AUDIO: deque(maxlen=self.config["repeat_threshold"] // 2),
        }
        
        # 模态专用检测器
        self.modality_detectors = {
            self.DataType.TEXT: self._detect_text_cycle,
            self.DataType.CODE: self._detect_code_cycle,
            self.DataType.IMAGE: self._detect_image_cycle,
            self.DataType.AUDIO: self._detect_audio_cycle,
        }
        
        # 模态专用参数
        self.modality_params = {
            self.DataType.TEXT: {
                "temperature": self.current_temperature,
                "repetition_penalty": self.current_repetition_penalty,
            },
            self.DataType.CODE: {
                "temperature": max(0.5, self.current_temperature - 0.2),  # 代码生成需要更确定性
                "repetition_penalty": min(2.0, self.current_repetition_penalty + 0.3),  # 代码重复惩罚更高
            },
            self.DataType.IMAGE: {
                "temperature": min(1.2, self.current_temperature + 0.3),  # 图像生成需要更多创造性
                "repetition_penalty": max(1.0, self.current_repetition_penalty - 0.2),  # 图像允许一定重复
            },
            self.DataType.AUDIO: {
                "temperature": self.current_temperature,
                "repetition_penalty": self.current_repetition_penalty,
            }
        }
        
        logger.info("MultimodalCyclePreventionManager 初始化完成")
    
    def detect_cycle_multimodal(self, data: Any, data_type: str = DataType.TEXT) -> bool:
        """
        多模态循环检测
        
        Args:
            data: 待检测的数据（文本、代码、图像路径、音频数据等）
            data_type: 数据类型
            
        Returns:
            bool: 是否检测到循环
        """
        if data_type not in self.modality_detectors:
            logger.warning(f"未知数据类型 {data_type}，使用文本检测器")
            data_type = self.DataType.TEXT
        
        try:
            # 调用模态专用检测器
            cycle_detected = self.modality_detectors[data_type](data)
            
            if cycle_detected:
                logger.warning(f"多模态检测到循环: 类型={data_type}, 数据={str(data)[:50]}...")
                self.protection_stats["cycle_detections"] += 1
            
            return cycle_detected
            
        except Exception as e:
            logger.error(f"多模态循环检测失败: {e}, 回退到文本检测")
            if isinstance(data, str):
                return self.detect_cycle_basic(data)
            return False
    
    def _detect_text_cycle(self, text: str) -> bool:
        """文本循环检测（复用基础层）"""
        return self.detect_cycle_basic(text)
    
    def _detect_code_cycle(self, code: str) -> bool:
        """代码循环检测（检测重复代码片段）"""
        if not code or not isinstance(code, str):
            return False
        
        # 简化版代码相似性检测
        # 1. 移除注释和空白
        import re
        clean_code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # 移除Python注释
        clean_code = re.sub(r'//.*$', '', clean_code, flags=re.MULTILINE)  # 移除JS/C++注释
        clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)  # 移除多行注释
        clean_code = re.sub(r'\s+', ' ', clean_code).strip()  # 标准化空白
        
        if not clean_code:
            return False
        
        # 2. 检查最近代码输出中的重复
        repeat_count = 0
        for output in self.multimodal_last_outputs[self.DataType.CODE]:
            if isinstance(output, str):
                clean_output = re.sub(r'#.*$', '', output, flags=re.MULTILINE)
                clean_output = re.sub(r'//.*$', '', clean_output, flags=re.MULTILINE)
                clean_output = re.sub(r'/\*.*?\*/', '', clean_output, flags=re.DOTALL)
                clean_output = re.sub(r'\s+', ' ', clean_output).strip()
                
                # 简单字符串相似度（Levenshtein距离简化版）
                similarity = self._calculate_string_similarity(clean_code, clean_output)
                if similarity > 0.8:  # 80%相似度视为重复
                    repeat_count += 1
        
        # 3. 超过阈值就是循环
        return repeat_count >= self.config["repeat_threshold"]
    
    def _detect_image_cycle(self, image_data: Any) -> bool:
        """图像循环检测（检测相似图像）"""
        # 简化版图像相似性检测
        # 实际实现可能使用图像哈希或特征提取
        
        if image_data is None:
            return False
        
        # 方法1: 如果图像是文件路径，检查路径重复
        if isinstance(image_data, str) and (image_data.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))):
            # 检查路径是否重复
            repeat_count = 0
            for output in self.multimodal_last_outputs[self.DataType.IMAGE]:
                if isinstance(output, str) and output == image_data:
                    repeat_count += 1
            
            return repeat_count >= (self.config["repeat_threshold"] // 2)  # 图像阈值更低
        
        # 方法2: 如果图像是张量或数组，简化检查（占位实现）
        # 实际项目应实现真正的图像相似性检测
        return False
    
    def _detect_audio_cycle(self, audio_data: Any) -> bool:
        """音频循环检测（检测相似音频）"""
        # 简化版音频相似性检测
        # 实际实现可能使用音频特征提取
        
        if audio_data is None:
            return False
        
        # 方法1: 如果音频是文件路径，检查路径重复
        if isinstance(audio_data, str) and (audio_data.endswith(('.wav', '.mp3', '.ogg', '.flac'))):
            repeat_count = 0
            for output in self.multimodal_last_outputs[self.DataType.AUDIO]:
                if isinstance(output, str) and output == audio_data:
                    repeat_count += 1
            
            return repeat_count >= (self.config["repeat_threshold"] // 2)  # 音频阈值更低
        
        # 方法2: 其他音频数据（占位实现）
        return False
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度（简化版Levenshtein距离）"""
        if not str1 or not str2:
            return 0.0
        
        # 简单实现：Jaccard相似度
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_safe_multimodal(
        self,
        prompt: Any,
        generate_func: Callable[[Any, Dict[str, Any]], Any],
        data_type: str = DataType.TEXT,
        max_attempts: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        多模态安全生成
        
        Args:
            prompt: 输入提示（可以是文本、图像描述、代码要求等）
            generate_func: 生成函数，接受(context, params)返回数据
            data_type: 数据类型
            max_attempts: 最大生成尝试次数
            
        Returns:
            Tuple[Any, Dict[str, Any]]: (生成的数据, 防护信息)
        """
        max_attempts = max_attempts or self.config["max_retry_attempts"]
        attempts = 0
        
        # 1. 获取模态专用历史缓冲区
        history_buffer = self.multimodal_history.get(data_type, self.chat_history)
        history_buffer.append(prompt)
        
        # 构建上下文（如果是文本类型）
        context = prompt
        if data_type == self.DataType.TEXT and isinstance(prompt, str):
            context = "\n".join(list(history_buffer))
        
        while attempts < max_attempts:
            attempts += 1
            self.protection_stats["total_generations"] += 1
            
            # 2. 获取防护参数（模态专用）
            modality_param = self.modality_params.get(data_type, {})
            adaptive_params = self.get_adaptive_parameters(
                str(prompt) if isinstance(prompt, (str, int, float)) else str(type(prompt))
            )
            
            # 融合参数：模态专用 + 场景自适应 + 基础层
            params = {
                "temperature": modality_param.get("temperature", adaptive_params.get("temperature", self.current_temperature)),
                "repetition_penalty": modality_param.get("repetition_penalty", adaptive_params.get("repetition_penalty", self.current_repetition_penalty)),
                "data_type": data_type,
                "protection_layer": "multimodal",
            }
            
            # 3. 生成数据
            try:
                output = generate_func(context, params)
            except Exception as e:
                logger.error(f"多模态生成失败: {e}")
                output = f"生成错误: {str(e)}"
                cycle_detected = True
            else:
                # 4. 多模态循环检测
                cycle_detected = self.detect_cycle_multimodal(output, data_type)
            
            # 5. 应用防护措施
            if cycle_detected:
                basic_protection = self.apply_basic_protection(cycle_detected)
                
                # 更新模态专用参数
                if data_type in self.modality_params:
                    self.modality_params[data_type]["temperature"] = basic_protection["temperature"]
                    self.modality_params[data_type]["repetition_penalty"] = basic_protection["repetition_penalty"]
                
                # 继续重试
                if attempts < max_attempts:
                    logger.info(f"多模态检测到循环，重新生成 (类型={data_type}, 尝试 {attempts}/{max_attempts})")
                    continue
                else:
                    logger.warning(f"达到最大重试次数 ({max_attempts})，返回当前输出")
            
            # 6. 无循环或达到最大尝试次数，返回结果
            if not cycle_detected or attempts >= max_attempts:
                # 更新输出队列
                output_buffer = self.multimodal_last_outputs.get(data_type, self.last_outputs)
                output_buffer.append(output)
                
                # 更新历史
                history_buffer.append(output)
                
                # 重置重试计数（成功生成）
                if not cycle_detected:
                    self.retry_count = 0
                
                protection_info = {
                    "attempts": attempts,
                    "cycle_detected": cycle_detected,
                    "data_type": data_type,
                    "temperature": params["temperature"],
                    "repetition_penalty": params["repetition_penalty"],
                    "history_size": len(history_buffer),
                    "last_outputs_size": len(output_buffer),
                    "protection_layer": "multimodal",
                }
                
                return output, protection_info
        
        # 失败回退
        return None, {
            "attempts": attempts,
            "cycle_detected": True,
            "data_type": data_type,
            "error": "max_attempts_reached",
            "protection_layer": "multimodal_emergency_fallback"
        }
    
    def update_multimodal_history(self, data: Any, data_type: str = DataType.TEXT) -> None:
        """
        更新多模态历史
        
        Args:
            data: 数据
            data_type: 数据类型
        """
        history_buffer = self.multimodal_history.get(data_type)
        if history_buffer is not None:
            history_buffer.append(data)
            logger.debug(f"更新{data_type}历史: 当前大小={len(history_buffer)}")
        else:
            logger.warning(f"未知数据类型 {data_type}，无法更新历史")
    
    def get_multimodal_stats(self) -> Dict[str, Any]:
        """
        获取多模态防护统计
        
        Returns:
            Dict[str, Any]: 多模态统计信息
        """
        base_stats = self.get_protection_stats()
        
        multimodal_stats = {
            **base_stats,
            "multimodal_history_sizes": {
                data_type: len(buffer) 
                for data_type, buffer in self.multimodal_history.items()
            },
            "multimodal_output_sizes": {
                data_type: len(buffer)
                for data_type, buffer in self.multimodal_last_outputs.items()
            },
            "modality_params": self.modality_params,
        }
        
        return multimodal_stats


def get_multimodal_cycle_prevention_manager(
    config: Optional[Dict[str, Any]] = None,
    enable_adaptive_layer: bool = True,
    multimodal_config: Optional[Dict[str, Any]] = None
) -> MultimodalCyclePreventionManager:
    """
    获取多模态防循环管理器实例
    
    Args:
        config: 基础配置字典
        enable_adaptive_layer: 是否启用场景自适应高级防护
        multimodal_config: 多模态专用配置
        
    Returns:
        MultimodalCyclePreventionManager实例
    """
    return MultimodalCyclePreventionManager(
        config=config,
        enable_adaptive_layer=enable_adaptive_layer,
        multimodal_config=multimodal_config
    )


# 示例使用
if __name__ == "__main__":
    print("=" * 80)
    print("防循环管理器测试")
    print("=" * 80)
    
    # 创建模拟的生成函数
    def mock_generate_func(context, params):
        """模拟生成函数"""
        temperature = params.get("temperature", 0.7)
        repetition_penalty = params.get("repetition_penalty", 1.2)
        
        # 模拟不同参数下的输出
        if temperature > 0.8:
            return "高温度输出：创造性较强的文本内容"
        elif repetition_penalty > 1.3:
            return "高惩罚输出：重复较少的文本内容"
        else:
            return "标准输出：正常的文本内容"
    
    print("\n1. 测试基础防护层:")
    basic_manager = CyclePreventionManager(enable_adaptive_layer=False)
    
    # 模拟循环检测
    test_text = "重复文本重复文本重复文本"
    for i in range(3):
        basic_manager.last_outputs.append(test_text)
    
    cycle_detected = basic_manager.detect_cycle_basic(test_text)
    print(f"   循环检测结果: {cycle_detected} (期望: True)")
    
    # 测试防护应用
    protection = basic_manager.apply_basic_protection(cycle_detected)
    print(f"   防护应用: 温度={protection['temperature']:.3f}, 惩罚={protection['repetition_penalty']:.3f}")
    
    print("\n2. 测试高级防护层:")
    adaptive_manager = CyclePreventionManager(enable_adaptive_layer=True)
    
    # 测试自适应参数获取
    industrial_text = "工业控制系统PID控制器调节温度"
    adaptive_params = adaptive_manager.get_adaptive_parameters(industrial_text)
    print(f"   工业控制文本参数: 场景={adaptive_params.get('scene', 'unknown')}, "
          f"温度={adaptive_params.get('temperature', 0):.3f}, "
          f"惩罚={adaptive_params.get('repetition_penalty', 0):.3f}")
    
    print("\n3. 测试安全生成:")
    output, protection_info = adaptive_manager.generate_safe(
        prompt="测试提示",
        generate_func=mock_generate_func,
        max_attempts=2
    )
    print(f"   生成输出: {output}")
    print(f"   防护信息: 尝试次数={protection_info.get('attempts', 0)}, "
          f"防护层={protection_info.get('protection_layer', 'unknown')}")
    
    print("\n4. 测试防护统计:")
    stats = adaptive_manager.get_protection_stats()
    print(f"   总生成次数: {stats['total_generations']}")
    print(f"   循环检测次数: {stats['cycle_detections']}")
    
    print("\n✓ 防循环管理器测试完成")
    
    print("\n" + "=" * 80)
    print("多模态防循环管理器测试")
    print("=" * 80)
    
    print("\n5. 测试多模态防护层:")
    multimodal_manager = MultimodalCyclePreventionManager(enable_adaptive_layer=True)
    
    # 测试多模态检测器
    print("\n  5.1 测试文本循环检测:")
    text_cycle = multimodal_manager.detect_cycle_multimodal("重复文本", multimodal_manager.DataType.TEXT)
    print(f"    文本循环检测: {text_cycle} (期望: False)")
    
    print("\n  5.2 测试代码循环检测:")
    python_code = "def hello_world():\n    print('Hello World')"
    code_cycle = multimodal_manager.detect_cycle_multimodal(python_code, multimodal_manager.DataType.CODE)
    print(f"    代码循环检测: {code_cycle} (期望: False)")
    
    print("\n  5.3 测试多模态安全生成:")
    
    # 文本生成函数
    def text_generate_func(context, params):
        return f"生成的文本: {context[:20]}... (温度={params['temperature']:.2f})"
    
    # 代码生成函数
    def code_generate_func(context, params):
        return f"""# 生成的代码
print('Generated at temperature {params["temperature"]:.2f}')"""
    
    # 测试文本生成
    text_output, text_info = multimodal_manager.generate_safe_multimodal(
        prompt="生成一段文本",
        generate_func=text_generate_func,
        data_type=multimodal_manager.DataType.TEXT,
        max_attempts=2
    )
    print(f"    文本生成结果: {text_output}")
    print(f"    文本防护信息: 类型={text_info.get('data_type', 'unknown')}, "
          f"尝试次数={text_info.get('attempts', 0)}")
    
    # 测试代码生成
    code_output, code_info = multimodal_manager.generate_safe_multimodal(
        prompt="生成Python函数",
        generate_func=code_generate_func,
        data_type=multimodal_manager.DataType.CODE,
        max_attempts=2
    )
    print(f"    代码生成结果: {code_output}")
    print(f"    代码防护信息: 类型={code_info.get('data_type', 'unknown')}, "
          f"温度={code_info.get('temperature', 0):.3f}")
    
    print("\n  5.4 测试多模态统计:")
    multimodal_stats = multimodal_manager.get_multimodal_stats()
    print(f"    多模态历史大小: {multimodal_stats['multimodal_history_sizes']}")
    print(f"    模态专用参数: {list(multimodal_stats['modality_params'].keys())}")
    
    print("\n✓ 多模态防循环管理器测试完成")
    print("\n多模态扩展特点:")
    print("  1. 多模态支持: 文本、代码、图像、音频等多种数据类型")
    print("  2. 模态专用检测: 不同类型数据使用专用检测算法")
    print("  3. 参数差异化: 代码需要确定性，图像需要创造性，音频需要连贯性")
    print("  4. 统一接口: 所有模态使用同一套多模态防护接口")
    print("\n设计特点:")
    print("  1. 基础层: 嵌入式思维防护（缓冲区清理+重复检测+看门狗重置）")
    print("  2. 高级层: 场景自适应防护（语义检测+动态参数+性能反馈）")
    print("  3. 融合策略: 基础确保可靠，高级提供智能，双重保护")
    print("  4. 统一接口: 所有功能模型使用同一套防循环接口")