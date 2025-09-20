"""
自我反思引擎模块
实现AGI级别的自我反思功能

根据评估结果进行自我反思，生成改进建议
"""

from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SelfReflectionEngine")

class SelfReflectionEngine:
    """
    自我反思引擎类
    负责根据评估结果进行自我反思，生成见解和改进建议
    """
    
    def __init__(self):
        """初始化自我反思引擎"""
        # 反思日志，使用双端队列限制最大条目数
        self.reflection_log = deque(maxlen=50)
        
        # 改进建议列表
        self.improvement_suggestions = []
        
        # 反思配置
        self.reflection_config = {
            'auto_reflection_enabled': True,
            'low_score_threshold': 0.4,
            'high_score_threshold': 0.8,
            'reflection_timeout': 60  # 秒
        }
        
        logger.info("自我反思引擎初始化完成")
    
    def reflect_on_assessment(self, reflection_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据评估结果进行自我反思
        
        Args:
            reflection_input: 包含评估结果的字典
                格式: {
                    "value_name": "价值观名称",
                    "action": "执行的行为",
                    "assessment_score": 评估分数(0-1),
                    "context": 上下文信息
                }
        
        Returns:
            包含反思结果的字典
        """
        # 提取输入数据
        value_name = reflection_input.get('value_name', 'unknown')
        action = reflection_input.get('action', '')
        score = reflection_input.get('assessment_score', 0.5)
        context = reflection_input.get('context', {})
        
        # 生成反思内容
        reflection = {
            'value': value_name,
            'action': str(action),
            'score': score,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'insights': self._generate_insights(value_name, score, action),
            'improvement_needed': score < self.reflection_config['low_score_threshold']
        }
        
        # 记录反思
        self.reflection_log.append(reflection)
        
        # 如果分数低于阈值，生成改进建议
        if reflection['improvement_needed']:
            suggestion = self._generate_improvement_suggestion(reflection)
            self.improvement_suggestions.append(suggestion)
            reflection['suggestion'] = suggestion
        
        logger.info(f"完成对'{value_name}'价值的反思，得分: {score}")
        return reflection
    
    def _generate_insights(self, value_name: str, score: float, action: Any) -> str:
        """
        生成反思见解
        
        Args:
            value_name: 价值观名称
            score: 评估分数
            action: 执行的行为
        
        Returns:
            见解文本
        """
        action_str = str(action)[:100]  # 限制长度，避免过长
        
        if score > self.reflection_config['high_score_threshold']:
            return f"行为与'{value_name}'价值观高度一致，建议保持此行为模式"
        elif score > 0.6:
            return f"行为基本符合'{value_name}'价值观，仍有改进空间"
        elif score > self.reflection_config['low_score_threshold']:
            return f"行为与'{value_name}'价值观存在一定冲突，需要谨慎处理"
        else:
            return f"行为严重违反'{value_name}'价值观，必须修改或放弃"
    
    def _generate_improvement_suggestion(self, reflection: Dict[str, Any]) -> str:
        """
        生成改进建议
        
        Args:
            reflection: 反思结果
        
        Returns:
            建议文本
        """
        value_name = reflection['value']
        score = reflection['score']
        action = reflection['action']
        
        suggestion = f"价值观'{value_name}'评估分数较低 ({score:.2f})，建议重新评估行为: {action}"
        
        # 根据不同价值观提供特定建议
        specific_suggestions = {
            'safety': "考虑所有潜在风险，确保行为不会导致伤害",
            'helpfulness': "分析用户真正需求，提供更有针对性的帮助",
            'honesty': "确保所有信息准确无误，避免误导",
            'fairness': "考虑所有相关方的利益，确保公平对待",
            'autonomy_respect': "尊重用户自主权，提供选择而非强制",
            'privacy': "确保数据安全，保护用户隐私"
        }
        
        if value_name in specific_suggestions:
            suggestion += f"。{specific_suggestions[value_name]}"
        
        return suggestion
    
    def get_recent_reflections(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的反思记录
        
        Args:
            count: 要获取的记录数量
        
        Returns:
            反思记录列表
        """
        return list(self.reflection_log)[-count:]
    
    def get_improvement_suggestions(self, clear: bool = False) -> List[str]:
        """
        获取改进建议
        
        Args:
            clear: 是否清除建议列表
        
        Returns:
            改进建议列表
        """
        suggestions = self.improvement_suggestions.copy()
        
        if clear:
            self.improvement_suggestions = []
        
        return suggestions
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """
        获取反思统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.reflection_log:
            return {
                'total_reflections': 0,
                'average_score': 0.0,
                'low_score_count': 0,
                'high_score_count': 0
            }
        
        scores = [entry['score'] for entry in self.reflection_log]
        
        return {
            'total_reflections': len(self.reflection_log),
            'average_score': sum(scores) / len(scores),
            'low_score_count': sum(1 for score in scores if score < self.reflection_config['low_score_threshold']),
            'high_score_count': sum(1 for score in scores if score > self.reflection_config['high_score_threshold']),
            'pending_suggestions': len(self.improvement_suggestions)
        }