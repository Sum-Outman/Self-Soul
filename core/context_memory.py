#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文记忆管理器
Context Memory Manager

管理系统的上下文记忆，提供上下文理解和记忆更新功能
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import time
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextMemoryManager:
    """
    上下文记忆管理器
    负责管理系统的上下文记忆，提供上下文理解和记忆更新功能
    """
    
    def __init__(self):
        """初始化上下文记忆管理器"""
        self.context_history = []
        self.max_history_size = 100  # 最大历史记录数量
        self.initialized = False
        logger.info("上下文记忆管理器初始化")
        
    def initialize(self) -> bool:
        """初始化管理器"""
        try:
            self.initialized = True
            logger.info("上下文记忆管理器初始化完成")
            return True
        except Exception as e:
            logger.error(f"上下文记忆管理器初始化失败: {str(e)}")
            return False
    
    def calculate_understanding_score(self, context: Dict[str, Any]) -> float:
        """
        计算上下文理解分数
        
        Args:
            context: 上下文信息
            
        Returns:
            float: 理解分数(0-1)
        """
        try:
            # 简单实现：基于上下文元素数量和历史相似度计算理解分数
            context_elements = len(context)
            
            # 基本分数
            base_score = min(0.7, context_elements * 0.1)
            
            # 如果有历史记录，计算相似度分数
            similarity_score = 0.0
            if self.context_history:
                # 计算与最近5个上下文的相似度
                recent_contexts = self.context_history[-5:]
                similarities = []
                
                # 简单相似度计算：共同键的比例
                for hist_context in recent_contexts:
                    common_keys = set(context.keys()) & set(hist_context.keys())
                    max_keys = max(len(context), len(hist_context))
                    similarity = len(common_keys) / max_keys if max_keys > 0 else 0
                    similarities.append(similarity)
                
                # 计算平均相似度并加权
                if similarities:
                    similarity_score = sum(similarities) / len(similarities) * 0.3
            
            # 总分数
            total_score = base_score + similarity_score
            return round(min(1.0, total_score), 2)
            
        except Exception as e:
            logger.error(f"上下文理解分数计算失败: {str(e)}")
            return 0.5
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """更新上下文记忆"""
        try:
            # 添加时间戳
            context_with_timestamp = context.copy()
            context_with_timestamp['timestamp'] = time.time()
            
            # 添加到历史记录
            self.context_history.append(context_with_timestamp)
            
            # 限制历史记录大小
            if len(self.context_history) > self.max_history_size:
                self.context_history.pop(0)
                
        except Exception as e:
            logger.error(f"上下文更新失败: {str(e)}")
    
    def update_visual_context(self, visual_data: Dict[str, Any], analysis_result: Dict[str, Any], 
                             learning_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """更新视觉上下文记忆"""
        try:
            # 简化实现：只记录基本信息
            context_update = {
                'visual_context_type': 'image_analysis',
                'timestamp': time.time(),
                'has_visual_data': bool(visual_data),
                'has_analysis': bool(analysis_result),
                'has_learning': bool(learning_outcome)
            }
            
            self.update_context(context_update)
            return context_update
            
        except Exception as e:
            logger.error(f"视觉上下文更新失败: {str(e)}")
            return {'error': str(e)}
    
    def get_recent_contexts(self, count: int = 5) -> List[Dict[str, Any]]:
        """获取最近的上下文"""
        try:
            return self.context_history[-count:]
        except Exception as e:
            logger.error(f"获取最近上下文失败: {str(e)}")
            return []
    
    def clear_context(self) -> None:
        """清除上下文记忆"""
        try:
            self.context_history.clear()
            logger.info("上下文记忆已清空")
        except Exception as e:
            logger.error(f"清除上下文记忆失败: {str(e)}")

# 全局实例便于访问
context_memory_manager = ContextMemoryManager()