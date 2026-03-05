#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型上下文记忆管理器 - 支持语义级切片和漂移检测
Enhanced Context Memory Manager

管理系统的上下文记忆，提供语义级上下文理解、场景检测和记忆更新功能
解决当前系统的核心问题：
1. 从文本长度截断升级为语义级切片
2. 从长度阈值清理升级为语义漂移检测
3. 支持多场景上下文分离（工业控制、医疗影像等）
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import time
import json
import re
from collections import defaultdict, deque
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticSceneDetector:
    """语义场景检测器 - 识别上下文中的语义场景"""
    
    # 场景关键词定义（支持中文和英文）
    SCENE_KEYWORDS = {
        "industrial_control": {
            "zh": ["工业", "控制", "工厂", "自动化", "机器人", "PLC", "传感器", "制造", "生产线", "设备", "监控",
                   "温度", "压力", "流量", "阀门", "电机", "变频器", "控制器", "仪表", "执行器", "报警", "安全",
                   "工艺", "参数", "设定", "调节", "PID", "反馈", "前馈", "回路", "连锁", "联锁"],
            "en": ["industrial", "control", "factory", "automation", "robot", "PLC", "sensor", "manufacturing", "production", "equipment", "monitoring",
                   "temperature", "pressure", "flow", "valve", "motor", "inverter", "controller", "instrument", "actuator", "alarm", "safety",
                   "process", "parameter", "setting", "regulation", "PID", "feedback", "feedforward", "loop", "interlock", "safety"]
        },
        "medical_imaging": {
            "zh": ["医疗", "影像", "医学", "诊断", "CT", "MRI", "X光", "超声", "病理", "医院", "医生", "病人", 
                   "患者", "扫描", "肺部", "阴影", "活检", "癌症", "肿瘤", "治疗", "手术", "药品", "康复", 
                   "心电图", "血压", "血糖", "检查", "化验", "报告", "症状", "疾病", "健康"],
            "en": ["medical", "imaging", "diagnosis", "CT", "MRI", "X-ray", "ultrasound", "pathology", "hospital", "doctor", "patient",
                   "scan", "lung", "shadow", "biopsy", "cancer", "tumor", "treatment", "surgery", "medicine", "recovery",
                   "ecg", "blood pressure", "blood sugar", "examination", "test", "report", "symptom", "disease", "health"]
        },
        "financial_analysis": {
            "zh": ["金融", "财务", "股票", "投资", "交易", "银行", "货币", "经济", "市场", "分析", "风险",
                   "基金", "债券", "期货", "期权", "外汇", "黄金", "原油", "指数", "财报", "利润", "收入",
                   "成本", "资产", "负债", "权益", "现金流", "估值", "评级", "策略", "组合", "收益率"],
            "en": ["financial", "finance", "stock", "investment", "trading", "bank", "currency", "economy", "market", "analysis", "risk",
                   "fund", "bond", "futures", "options", "forex", "gold", "crude", "index", "earnings", "profit", "revenue",
                   "cost", "asset", "liability", "equity", "cash flow", "valuation", "rating", "strategy", "portfolio", "yield"]
        },
        "educational_tutoring": {
            "zh": ["教育", "学习", "教学", "学生", "老师", "课程", "学校", "知识", "培训", "辅导", "考试",
                   "练习", "作业", "复习", "预习", "讲解", "答疑", "教材", "教案", "课件", "课堂", "教室",
                   "升学", "毕业", "学位", "论文", "研究", "实验", "实践", "技能", "能力", "成绩"],
            "en": ["education", "learning", "teaching", "student", "teacher", "course", "school", "knowledge", "training", "tutoring", "exam",
                   "practice", "homework", "review", "preview", "explanation", "q&a", "textbook", "lesson plan", "courseware", "classroom", "lecture",
                   "admission", "graduation", "degree", "thesis", "research", "experiment", "practice", "skill", "ability", "grade"]
        },
        "general_conversation": {
            "zh": ["聊天", "对话", "交流", "问题", "回答", "帮助", "信息", "日常", "生活", "天气", "时间",
                   "建议", "意见", "想法", "讨论", "话题", "分享", "告诉", "知道", "了解", "明白", "清楚",
                   "谢谢", "感谢", "请问", "你好", "再见", "早上", "晚上", "今天", "明天", "昨天"],
            "en": ["chat", "conversation", "communication", "question", "answer", "help", "information", "daily", "life", "weather", "time",
                   "suggestion", "advice", "opinion", "idea", "discussion", "topic", "share", "tell", "know", "understand", "clear",
                   "thanks", "thank", "please", "hello", "goodbye", "morning", "evening", "today", "tomorrow", "yesterday"]
        }
    }
    
    def __init__(self):
        """初始化语义场景检测器"""
        self.scene_scores = defaultdict(float)
        self.scene_transitions = []  # 记录场景转换
        self.last_scene = None
        
    def detect_scene(self, text: str) -> Tuple[str, Dict[str, float]]:
        """检测文本中的语义场景
        
        Args:
            text: 待检测的文本（支持中文和英文）
            
        Returns:
            Tuple[str, Dict[str, float]]: (主要场景, 所有场景的置信度分数)
        """
        if not text or not isinstance(text, str):
            return "general_conversation", {"general_conversation": 1.0}
        
        # 转换为小写以进行不区分大小写的匹配
        text_lower = text.lower()
        
        # 计算每个场景的分数
        scene_scores = {}
        for scene_name, keywords in self.SCENE_KEYWORDS.items():
            match_count = 0
            
            # 检查中文关键词
            for keyword in keywords.get("zh", []):
                if keyword in text:
                    match_count += 1
            
            # 检查英文关键词
            for keyword in keywords.get("en", []):
                if keyword.lower() in text_lower:
                    match_count += 1
            
            # 计算分数：使用匹配数量的函数，避免除以总关键词数
            # score = match_count / (match_count + 2)  # 当match_count=1时score=0.33，2时0.5，3时0.6
            if match_count == 0:
                score = 0.0
            elif match_count == 1:
                score = 0.2  # 单个匹配给予较低置信度
            elif match_count == 2:
                score = 0.4
            elif match_count == 3:
                score = 0.6
            elif match_count == 4:
                score = 0.75
            else:
                score = 0.9  # 5个或更多匹配给予高置信度
            
            scene_scores[scene_name] = score
        
        # 如果所有分数都很低，默认为通用对话
        max_score = max(scene_scores.values()) if scene_scores else 0
        if max_score < 0.05:  # 降低阈值从0.1到0.05
            scene_scores["general_conversation"] = 1.0
        
        # 确定主要场景
        main_scene = max(scene_scores.items(), key=lambda x: x[1])[0]
        
        # 记录场景转换
        if self.last_scene != main_scene and self.last_scene is not None:
            self.scene_transitions.append({
                "from": self.last_scene,
                "to": main_scene,
                "timestamp": time.time()
            })
        
        self.last_scene = main_scene
        
        return main_scene, scene_scores
    
    def get_scene_transitions(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的场景转换记录"""
        return self.scene_transitions[-count:] if self.scene_transitions else []


class SemanticDriftDetector:
    """语义漂移检测器 - 检测模型输出是否与当前场景相关"""
    
    def __init__(self, window_size: int = 10, scene_detector: Optional[Any] = None):
        """初始化语义漂移检测器
        
        Args:
            window_size: 用于检测漂移的窗口大小
            scene_detector: 可选的语义场景检测器实例
        """
        self.window_size = window_size
        self.context_window = deque(maxlen=window_size)
        self.drift_scores = []
        self.drift_threshold = 0.7  # 漂移阈值（高于此值认为发生漂移）
        self.scene_detector = scene_detector
        
    def add_context(self, context: Dict[str, Any]) -> None:
        """添加上下文到检测窗口"""
        self.context_window.append(context)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """从文本中提取关键词（支持中文和英文）
        
        Args:
            text: 输入文本
            
        Returns:
            关键词集合
        """
        if not text:
            return set()
        
        keywords = set()
        
        # 方法1: 按标点符号和空格分割文本
        # 分割中文文本（使用中文标点符号和空格）
        segments = re.split(r'[，。；！？、\s]+', text)
        for segment in segments:
            if segment:
                # 添加整个分段（如果不太长）
                if len(segment) <= 6:
                    keywords.add(segment)
                # 对于长分段，添加2-4字符的子串
                if len(segment) > 2:
                    for i in range(2, min(5, len(segment))):
                        keywords.add(segment[:i])
        
        # 方法2: 提取连续的中文字符序列
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
        for word in chinese_words:
            if len(word) <= 4:
                keywords.add(word)
            else:
                keywords.add(word)
                # 添加子串
                for i in range(2, min(5, len(word))):
                    keywords.add(word[:i])
        
        # 提取英文单词（包括单字母缩写）
        english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords.update(english_words[:20])  # 限制英文单词数量
        
        # 提取数字（可能包含重要信息）
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        keywords.update(numbers[:5])
        
        # 移除空字符串
        keywords.discard('')
        
        return keywords
    
    def detect_drift(self, current_output: str, current_scene: str) -> Tuple[bool, float]:
        """检测语义漂移
        
        Args:
            current_output: 当前模型输出
            current_scene: 当前语义场景
            
        Returns:
            Tuple[bool, float]: (是否发生漂移, 漂移分数)
        """
        if len(self.context_window) < 2:
            return False, 0.0
        
        # 提取最近上下文中的关键词
        recent_keywords = set()
        for context in self.context_window:
            if isinstance(context, dict):
                # 优先使用content字段，其次是text字段
                text = context.get("content", context.get("text", ""))
                if text:
                    keywords = self._extract_keywords(text)
                    recent_keywords.update(keywords)
        
        # 提取当前输出的关键词
        output_keywords = self._extract_keywords(current_output)
        
        # 计算相关性分数（共同关键词的比例）
        keyword_relevance = 0.0
        if recent_keywords:
            overlap = len(recent_keywords & output_keywords)
            # 使用Jaccard相似度计算相关性
            union_size = len(recent_keywords | output_keywords)
            if union_size > 0:
                keyword_relevance = overlap / union_size
        
        # 基于场景的相似度
        scene_relevance = 0.0
        if self.scene_detector and current_scene:
            # 检测当前输出的场景
            output_scene, scene_scores = self.scene_detector.detect_scene(current_output)
            if output_scene == current_scene:
                scene_relevance = 0.8  # 相同场景给予高相关性
            elif current_scene in scene_scores and scene_scores[current_scene] > 0.1:
                scene_relevance = scene_scores[current_scene] * 0.8  # 部分匹配
        
        # 结合两种相关性分数（取最大值）
        relevance_score = max(keyword_relevance, scene_relevance)
        
        # 如果都没有匹配，给予最低相关性0.1（避免完全为0）
        if relevance_score == 0.0:
            relevance_score = 0.1
        
        # 计算漂移分数（1 - 相关性分数）
        drift_score = 1.0 - relevance_score
        
        # 记录漂移分数
        self.drift_scores.append({
            "timestamp": time.time(),
            "drift_score": drift_score,
            "relevance_score": relevance_score,
            "current_scene": current_scene
        })
        
        # 检查是否超过漂移阈值
        is_drift = drift_score > self.drift_threshold
        
        return is_drift, drift_score
    
    def get_drift_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """获取漂移检测历史"""
        return self.drift_scores[-count:] if self.drift_scores else []


class ContextMemoryManager:
    """
    增强型上下文记忆管理器
    负责管理系统的上下文记忆，提供语义级上下文理解、场景检测和记忆更新功能
    支持语义级切片和漂移检测
    """
    
    def __init__(self, max_history_size: int = 100):
        """初始化上下文记忆管理器
        
        Args:
            max_history_size: 最大历史记录数量
        """
        # 基础上下文存储
        self.context_history = []
        self.max_history_size = max_history_size
        
        # 语义场景管理
        self.semantic_scenes = []  # 按场景分组的上下文
        self.current_scene = "general_conversation"
        self.scene_history = []  # 场景历史记录
        
        # 初始化检测器
        self.scene_detector = SemanticSceneDetector()
        self.drift_detector = SemanticDriftDetector(scene_detector=self.scene_detector)
        
        # 语义切片配置
        self.semantic_slice_enabled = True
        self.min_scene_contexts = 3  # 场景最少上下文数量
        self.max_scene_contexts = 20  # 场景最大上下文数量
        
        # 漂移检测配置
        self.drift_detection_enabled = True
        self.auto_clean_on_drift = True  # 检测到漂移时自动清理
        
        # 状态跟踪
        self.initialized = False
        self.last_clean_time = time.time()
        self.clean_trigger_count = 0
        
        logger.info("增强型上下文记忆管理器初始化完成")
        
    def initialize(self) -> bool:
        """初始化管理器"""
        try:
            self.initialized = True
            logger.info("增强型上下文记忆管理器初始化完成")
            return True
        except Exception as e:
            logger.error(f"上下文记忆管理器初始化失败: {str(e)}")
            return False
    
    def calculate_understanding_score(self, context: Dict[str, Any]) -> float:
        """
        计算上下文理解分数（增强版：包含语义场景分析）
        
        Args:
            context: 上下文信息
            
        Returns:
            float: 理解分数(0-1)
        """
        try:
            # 提取文本内容用于语义分析
            context_text = self._extract_text_from_context(context)
            
            # 检测语义场景
            main_scene, scene_scores = self.scene_detector.detect_scene(context_text)
            
            # 基础分数：基于上下文元素数量
            context_elements = len(context)
            base_score = min(0.5, context_elements * 0.05)
            
            # 语义场景分数：场景置信度
            scene_confidence = scene_scores.get(main_scene, 0.0)
            scene_score = scene_confidence * 0.3
            
            # 历史相似度分数
            similarity_score = 0.0
            if self.context_history:
                recent_contexts = self.context_history[-5:]
                similarities = []
                
                for hist_context in recent_contexts:
                    # 语义级相似度计算
                    similarity = self._calculate_semantic_similarity(context, hist_context)
                    similarities.append(similarity)
                
                if similarities:
                    similarity_score = sum(similarities) / len(similarities) * 0.2
            
            # 总分数
            total_score = base_score + scene_score + similarity_score
            return round(min(1.0, total_score), 2)
            
        except Exception as e:
            logger.error(f"上下文理解分数计算失败: {str(e)}")
            return 0.5
    
    def _extract_text_from_context(self, context: Dict[str, Any]) -> str:
        """从上下文中提取文本内容"""
        if not context:
            return ""
        
        # 尝试从常见字段中提取文本
        text_fields = ["text", "message", "content", "query", "question", "input", "output", "response"]
        
        for field in text_fields:
            if field in context:
                value = context[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (list, dict)):
                    return str(value)
        
        # 如果没有找到文本字段，将整个上下文转换为字符串
        return str(context)
    
    def _calculate_semantic_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算两个上下文之间的语义相似度"""
        try:
            # 提取文本
            text1 = self._extract_text_from_context(context1)
            text2 = self._extract_text_from_context(context2)
            
            if not text1 or not text2:
                return 0.0
            
            # 检测场景
            scene1, scores1 = self.scene_detector.detect_scene(text1)
            scene2, scores2 = self.scene_detector.detect_scene(text2)
            
            # 场景匹配分数
            scene_match_score = 1.0 if scene1 == scene2 else 0.0
            
            # 关键词重叠分数
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if words1 and words2:
                overlap = len(words1 & words2)
                max_words = max(len(words1), len(words2))
                word_overlap_score = overlap / max_words if max_words > 0 else 0.0
            else:
                word_overlap_score = 0.0
            
            # 综合相似度分数
            similarity = scene_match_score * 0.6 + word_overlap_score * 0.4
            return similarity
            
        except Exception as e:
            logger.error(f"语义相似度计算失败: {str(e)}")
            return 0.0
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """更新上下文记忆（增强版：支持语义切片和漂移检测）"""
        try:
            # 添加时间戳
            context_with_timestamp = context.copy()
            context_with_timestamp['timestamp'] = time.time()
            
            # 提取文本用于语义分析
            context_text = self._extract_text_from_context(context)
            
            # 检测语义场景
            main_scene, scene_scores = self.scene_detector.detect_scene(context_text)
            context_with_timestamp['semantic_scene'] = main_scene
            context_with_timestamp['scene_scores'] = scene_scores
            
            # 更新当前场景
            self.current_scene = main_scene
            self.scene_history.append({
                "scene": main_scene,
                "timestamp": time.time(),
                "confidence": scene_scores.get(main_scene, 0.0)
            })
            
            # 添加到历史记录
            self.context_history.append(context_with_timestamp)
            
            # 添加到语义场景分组
            self._add_to_semantic_scene(context_with_timestamp, main_scene)
            
            # 添加到漂移检测器
            self.drift_detector.add_context(context_with_timestamp)
            
            # 检查语义漂移
            if self.drift_detection_enabled:
                is_drift, drift_score = self.drift_detector.detect_drift(context_text, main_scene)
                if is_drift:
                    logger.warning(f"检测到语义漂移，分数: {drift_score:.3f}")
                    if self.auto_clean_on_drift:
                        self._clean_context_on_drift(main_scene, drift_score)
            
            # 限制历史记录大小（语义感知的清理）
            self._semantic_aware_cleanup()
                
        except Exception as e:
            logger.error(f"上下文更新失败: {str(e)}")
    
    def _add_to_semantic_scene(self, context: Dict[str, Any], scene: str) -> None:
        """添加上下文到语义场景分组"""
        # 查找是否已存在该场景的分组
        scene_group = None
        for group in self.semantic_scenes:
            if group["scene"] == scene:
                scene_group = group
                break
        
        # 如果不存在，创建新的场景分组
        if not scene_group:
            scene_group = {
                "scene": scene,
                "contexts": [],
                "created_at": time.time(),
                "last_updated": time.time(),
                "context_count": 0
            }
            self.semantic_scenes.append(scene_group)
        
        # 添加上下文到分组
        scene_group["contexts"].append(context)
        scene_group["last_updated"] = time.time()
        scene_group["context_count"] = len(scene_group["contexts"])
        
        # 限制每个场景的上下文数量
        if len(scene_group["contexts"]) > self.max_scene_contexts:
            # 移除最旧的上下文，保留最新的
            scene_group["contexts"] = scene_group["contexts"][-self.max_scene_contexts:]
    
    def _clean_context_on_drift(self, scene: str, drift_score: float) -> None:
        """在检测到语义漂移时清理上下文"""
        try:
            logger.info(f"语义漂移触发清理，场景: {scene}, 漂移分数: {drift_score:.3f}")
            
            # 清理当前场景的上下文（保留最新的几个）
            for group in self.semantic_scenes:
                if group["scene"] == scene:
                    # 保留最新的3个上下文
                    if len(group["contexts"]) > 3:
                        group["contexts"] = group["contexts"][-3:]
                        group["context_count"] = len(group["contexts"])
                        logger.info(f"清理场景 '{scene}' 的上下文，保留 {len(group['contexts'])} 个")
            
            # 更新清理统计
            self.clean_trigger_count += 1
            self.last_clean_time = time.time()
            
        except Exception as e:
            logger.error(f"漂移清理失败: {str(e)}")
    
    def _semantic_aware_cleanup(self) -> None:
        """语义感知的上下文清理"""
        try:
            # 检查是否需要清理（基于长度阈值）
            if len(self.context_history) <= self.max_history_size:
                return
            
            # 计算需要清理的数量
            excess_count = len(self.context_history) - self.max_history_size
            
            # 语义感知的清理策略：
            # 1. 优先清理与当前场景最不相关的上下文
            # 2. 保留每个场景的最少上下文数量
            
            # 获取当前场景
            current_scene = self.current_scene
            
            # 按场景分组上下文
            scene_contexts = defaultdict(list)
            for i, context in enumerate(self.context_history):
                scene = context.get('semantic_scene', 'unknown')
                scene_contexts[scene].append((i, context))
            
            # 清理策略：优先清理非当前场景的旧上下文
            removed_indices = []
            
            # 首先清理未知场景的上下文
            if 'unknown' in scene_contexts:
                unknown_contexts = scene_contexts['unknown']
                for i, _ in unknown_contexts[:excess_count]:
                    removed_indices.append(i)
            
            # 如果还需要清理，清理非当前场景的上下文
            if len(removed_indices) < excess_count:
                remaining_needed = excess_count - len(removed_indices)
                for scene, contexts in scene_contexts.items():
                    if scene != current_scene and scene != 'unknown':
                        # 保留每个场景的最少上下文数量
                        keep_count = min(self.min_scene_contexts, len(contexts))
                        remove_count = len(contexts) - keep_count
                        
                        if remove_count > 0:
                            # 移除最旧的上下文
                            remove_now = min(remove_count, remaining_needed)
                            for i, _ in contexts[:remove_now]:
                                removed_indices.append(i)
                                remaining_needed -= 1
                                
                                if remaining_needed <= 0:
                                    break
                    
                    if remaining_needed <= 0:
                        break
            
            # 如果还需要清理，清理当前场景中最旧的上下文
            if len(removed_indices) < excess_count:
                remaining_needed = excess_count - len(removed_indices)
                if current_scene in scene_contexts:
                    current_contexts = scene_contexts[current_scene]
                    # 保留最少上下文数量
                    keep_count = min(self.min_scene_contexts, len(current_contexts))
                    remove_count = len(current_contexts) - keep_count
                    
                    if remove_count > 0:
                        remove_now = min(remove_count, remaining_needed)
                        for i, _ in current_contexts[:remove_now]:
                            removed_indices.append(i)
            
            # 按索引从大到小移除，避免索引变化问题
            removed_indices.sort(reverse=True)
            for idx in removed_indices:
                if 0 <= idx < len(self.context_history):
                    removed_context = self.context_history.pop(idx)
                    logger.debug(f"清理上下文: 场景={removed_context.get('semantic_scene', 'unknown')}, 索引={idx}")
            
            logger.info(f"语义感知清理完成，移除了 {len(removed_indices)} 个上下文")
            
        except Exception as e:
            logger.error(f"语义感知清理失败: {str(e)}")
    
    def update_visual_context(self, visual_data: Dict[str, Any], analysis_result: Dict[str, Any], 
                             learning_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """更新视觉上下文记忆（增强版：包含语义场景）"""
        try:
            # 提取文本用于语义分析
            visual_text = f"视觉分析: {analysis_result.get('description', '')} {learning_outcome.get('summary', '')}"
            
            # 检测语义场景
            main_scene, scene_scores = self.scene_detector.detect_scene(visual_text)
            
            # 创建上下文更新
            context_update = {
                'visual_context_type': 'image_analysis',
                'timestamp': time.time(),
                'has_visual_data': bool(visual_data),
                'has_analysis': bool(analysis_result),
                'has_learning': bool(learning_outcome),
                'semantic_scene': main_scene,
                'scene_scores': scene_scores,
                'text_summary': visual_text[:200]  # 截断文本摘要
            }
            
            self.update_context(context_update)
            return context_update
            
        except Exception as e:
            logger.error(f"视觉上下文更新失败: {str(e)}")
            return {'error': str(e)}
    
    def get_recent_contexts(self, count: int = 5, scene_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取最近的上下文（支持场景过滤）"""
        try:
            if scene_filter:
                # 过滤指定场景的上下文
                filtered_contexts = [
                    ctx for ctx in self.context_history
                    if ctx.get('semantic_scene') == scene_filter
                ]
                return filtered_contexts[-count:]
            else:
                return self.context_history[-count:]
        except Exception as e:
            logger.error(f"获取最近上下文失败: {str(e)}")
            return []
    
    def get_contexts_by_scene(self, scene: str, count: int = 10) -> List[Dict[str, Any]]:
        """获取指定语义场景的上下文"""
        try:
            filtered_contexts = [
                ctx for ctx in self.context_history
                if ctx.get('semantic_scene') == scene
            ]
            return filtered_contexts[-count:]
        except Exception as e:
            logger.error(f"获取场景上下文失败: {str(e)}")
            return []
    
    def clear_context(self, scene_filter: Optional[str] = None) -> None:
        """清除上下文记忆（支持按场景清理）"""
        try:
            if scene_filter:
                # 只清理指定场景的上下文
                original_count = len(self.context_history)
                self.context_history = [
                    ctx for ctx in self.context_history
                    if ctx.get('semantic_scene') != scene_filter
                ]
                removed_count = original_count - len(self.context_history)
                logger.info(f"已清除场景 '{scene_filter}' 的 {removed_count} 个上下文")
            else:
                # 清理所有上下文
                self.context_history.clear()
                self.semantic_scenes.clear()
                self.scene_history.clear()
                logger.info("所有上下文记忆已清空")
                
        except Exception as e:
            logger.error(f"清除上下文记忆失败: {str(e)}")
    
    def get_semantic_analysis(self) -> Dict[str, Any]:
        """获取语义分析结果"""
        try:
            # 统计场景分布
            scene_distribution = defaultdict(int)
            for context in self.context_history:
                scene = context.get('semantic_scene', 'unknown')
                scene_distribution[scene] += 1
            
            # 获取最近的漂移检测结果
            drift_history = self.drift_detector.get_drift_history(10)
            
            # 获取场景转换记录
            scene_transitions = self.scene_detector.get_scene_transitions(10)
            
            return {
                "current_scene": self.current_scene,
                "scene_distribution": dict(scene_distribution),
                "total_contexts": len(self.context_history),
                "semantic_scene_count": len(self.semantic_scenes),
                "drift_detection_enabled": self.drift_detection_enabled,
                "recent_drift_scores": [{"score": d["drift_score"], "time": d["timestamp"]} for d in drift_history[-5:]],
                "scene_transitions": scene_transitions[-5:],
                "clean_trigger_count": self.clean_trigger_count,
                "last_clean_time": self.last_clean_time
            }
            
        except Exception as e:
            logger.error(f"获取语义分析失败: {str(e)}")
            return {"error": str(e)}
    
    def enable_semantic_slicing(self, enabled: bool = True) -> None:
        """启用或禁用语义切片"""
        self.semantic_slice_enabled = enabled
        logger.info(f"语义切片已{'启用' if enabled else '禁用'}")
    
    def enable_drift_detection(self, enabled: bool = True) -> None:
        """启用或禁用漂移检测"""
        self.drift_detection_enabled = enabled
        logger.info(f"语义漂移检测已{'启用' if enabled else '禁用'}")

# 全局实例便于访问
context_memory_manager = ContextMemoryManager()