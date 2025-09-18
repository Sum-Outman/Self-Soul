"""
自我反思和元认知模块
实现深度自我监控、错误分析、性能评估和自我改进机制
"""

import json
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionSession:
    """自我反思会话数据类"""
    session_id: str
    trigger: str
    focus_areas: List[str]
    insights_gained: List[str]
    action_plan: Dict[str, Any]
    effectiveness_score: float
    timestamp: float

@dataclass
class MetaCognitiveState:
    """元认知状态数据类"""
    self_awareness_level: float
    error_detection_sensitivity: float
    learning_efficiency: float
    adaptation_capability: float
    performance_trend: List[float]

class SelfReflectionModule:
    """
    自我反思模块 - 实现深度元认知和自我改进能力
    使系统能够监控、分析和优化自身的认知过程
    """
    
    def __init__(self):
        self.reflection_history: List[ReflectionSession] = []
        self.meta_state = MetaCognitiveState(
            self_awareness_level=0.7,
            error_detection_sensitivity=0.6,
            learning_efficiency=0.5,
            adaptation_capability=0.6,
            performance_trend=[]
        )
        
        # 反思触发器和阈值
        self.reflection_triggers = {
            "performance_decline": self._check_performance_decline,
            "error_pattern": self._check_error_patterns,
            "novel_situation": self._check_novel_situation,
            "learning_plateau": self._check_learning_plateau,
            "periodic": self._check_periodic_reflection
        }
        
        # 性能监控数据
        self.performance_metrics = {
            "task_success_rates": [],
            "error_rates": [],
            "response_times": [],
            "learning_speeds": [],
            "adaptation_times": []
        }
        
        # 错误模式数据库
        self.error_patterns = self._initialize_error_patterns()
        
        # 加载历史数据
        self._load_reflection_history()
    
    def _initialize_error_patterns(self) -> Dict[str, Any]:
        """初始化错误模式数据库"""
        return {
            "reasoning_errors": {
                "logical_fallacies": ["以偏概全", "错误因果", "非黑即白", "诉诸情感"],
                "cognitive_biases": ["确认偏误", "锚定效应", "可用性启发", "群体思维"],
                "assumption_errors": ["错误前提", "未验证假设", "过时信息", "文化偏见"]
            },
            "learning_errors": {
                "overfitting": ["记忆而非理解", "缺乏泛化", "数据偏见敏感"],
                "underfitting": ["模式识别不足", "特征提取不够", "模型太简单"],
                "catastrophic_forgetting": ["新旧知识冲突", "知识干扰", "缺乏巩固"]
            },
            "interaction_errors": {
                "misunderstanding": ["语义歧义", "语境缺失", "文化差异"],
                "inappropriate_response": ["语气不当", "详细度不适", "时机错误"],
                "communication_failure": ["信息丢失", "反馈缺失", "期望不匹配"]
            }
        }
    
    def _load_reflection_history(self):
        """加载反思历史数据"""
        history_file = Path("data/reflection_history.pkl")
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.reflection_history = data.get('history', [])
                    self.meta_state = data.get('state', self.meta_state)
                    self.performance_metrics = data.get('metrics', self.performance_metrics)
                logger.info(f"加载了 {len(self.reflection_history)} 条反思历史记录")
            except Exception as e:
                logger.warning(f"加载反思历史失败: {e}")
    
    def _save_reflection_history(self):
        """保存反思历史数据"""
        try:
            data = {
                'history': self.reflection_history,
                'state': self.meta_state,
                'metrics': self.performance_metrics
            }
            with open("data/reflection_history.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"保存反思历史失败: {e}")
    
    def _check_performance_decline(self, current_performance: float) -> bool:
        """检查性能下降"""
        if len(self.performance_metrics["task_success_rates"]) >= 5:
            recent_avg = np.mean(self.performance_metrics["task_success_rates"][-5:])
            if current_performance < recent_avg * 0.8:  # 性能下降20%
                return True
        return False
    
    def _check_error_patterns(self, recent_errors: List[Dict[str, Any]]) -> bool:
        """检查错误模式"""
        if len(recent_errors) >= 3:
            error_types = [error.get('type', '') for error in recent_errors]
            # 检查是否有相同类型的错误连续出现
            from collections import Counter
            counter = Counter(error_types)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 3:  # 同类型错误出现3次以上
                return True
        return False
    
    def _check_novel_situation(self, situation_context: Dict[str, Any]) -> bool:
        """检查新情境"""
        novelty_score = situation_context.get('novelty', 0)
        return novelty_score > 0.7  # 新颖性阈值
    
    def _check_learning_plateau(self) -> bool:
        """检查学习平台期"""
        if len(self.performance_metrics["learning_speeds"]) >= 10:
            recent_speeds = self.performance_metrics["learning_speeds"][-10:]
            if np.std(recent_speeds) < 0.1 and np.mean(recent_speeds) < 0.3:
                return True
        return False
    
    def _check_periodic_reflection(self) -> bool:
        """定期反思检查"""
        # 每24小时至少进行一次反思
        if self.reflection_history:
            last_reflection = self.reflection_history[-1].timestamp
            return time.time() - last_reflection > 86400  # 24小时
        return True  # 如果没有历史记录，立即进行反思
    
    def should_reflect(self, context: Dict[str, Any]) -> bool:
        """决定是否应该进行反思"""
        for trigger_name, trigger_func in self.reflection_triggers.items():
            if trigger_name == "performance_decline":
                if trigger_func(context.get('current_performance', 0)):
                    return True
            elif trigger_name == "error_pattern":
                if trigger_func(context.get('recent_errors', [])):
                    return True
            elif trigger_name == "novel_situation":
                if trigger_func(context):
                    return True
            elif trigger_name == "learning_plateau":
                if trigger_func():
                    return True
            elif trigger_name == "periodic":
                if trigger_func():
                    return True
        return False
    
    def conduct_reflection(self, context: Dict[str, Any]) -> ReflectionSession:
        """进行自我反思会话"""
        session_id = f"reflection_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 分析当前状态
        focus_areas = self._identify_focus_areas(context)
        insights = self._generate_insights(focus_areas, context)
        action_plan = self._create_action_plan(insights)
        
        # 创建反思会话
        session = ReflectionSession(
            session_id=session_id,
            trigger=context.get('trigger', 'unknown'),
            focus_areas=focus_areas,
            insights_gained=insights,
            action_plan=action_plan,
            effectiveness_score=0.0,  # 初始效果分数
            timestamp=time.time()
        )
        
        # 记录会话
        self.reflection_history.append(session)
        self._update_meta_state(session)
        self._save_reflection_history()
        
        logger.info(f"进行自我反思会话: {session_id}, 焦点领域: {focus_areas}")
        
        return session
    
    def _identify_focus_areas(self, context: Dict[str, Any]) -> List[str]:
        """识别需要关注的领域"""
        focus_areas = []
        
        # 基于性能数据识别焦点
        if self._check_performance_decline(context.get('current_performance', 0)):
            focus_areas.append("performance_optimization")
        
        # 基于错误模式识别焦点
        if self._check_error_patterns(context.get('recent_errors', [])):
            focus_areas.append("error_correction")
        
        # 基于学习效率识别焦点
        if self._check_learning_plateau():
            focus_areas.append("learning_efficiency")
        
        # 基于新情境识别焦点
        if self._check_novel_situation(context):
            focus_areas.append("adaptation_strategy")
        
        # 如果没有特定焦点，进行一般性反思
        if not focus_areas:
            focus_areas = ["general_self_improvement", "knowledge_consolidation"]
        
        return focus_areas
    
    def _generate_insights(self, focus_areas: List[str], context: Dict[str, Any]) -> List[str]:
        """生成反思洞察"""
        insights = []
        
        for area in focus_areas:
            if area == "performance_optimization":
                insights.extend(self._generate_performance_insights(context))
            elif area == "error_correction":
                insights.extend(self._generate_error_insights(context))
            elif area == "learning_efficiency":
                insights.extend(self._generate_learning_insights())
            elif area == "adaptation_strategy":
                insights.extend(self._generate_adaptation_insights(context))
            else:
                insights.extend(self._generate_general_insights())
        
        return insights
    
    def _generate_performance_insights(self, context: Dict[str, Any]) -> List[str]:
        """生成性能相关洞察"""
        insights = []
        current_perf = context.get('current_performance', 0)
        
        if len(self.performance_metrics["task_success_rates"]) >= 5:
            avg_perf = np.mean(self.performance_metrics["task_success_rates"][-5:])
            trend = "上升" if current_perf > avg_perf else "下降"
            insights.append(f"当前性能{trend}，最近平均成功率: {avg_perf:.2f}")
        
        # 分析响应时间
        if self.performance_metrics["response_times"]:
            avg_response = np.mean(self.performance_metrics["response_times"][-10:])
            insights.append(f"平均响应时间: {avg_response:.2f}秒")
        
        return insights
    
    def _generate_error_insights(self, context: Dict[str, Any]) -> List[str]:
        """生成错误相关洞察"""
        insights = []
        recent_errors = context.get('recent_errors', [])
        
        if recent_errors:
            error_types = [error.get('type', 'unknown') for error in recent_errors]
            from collections import Counter
            counter = Counter(error_types)
            most_common = counter.most_common(1)
            if most_common:
                insights.append(f"最常见错误类型: {most_common[0][0]} (出现{most_common[0][1]}次)")
        
        # 提供错误改进建议
        insights.append("建议: 增加错误检测机制，提前预防常见错误模式")
        
        return insights
    
    def _generate_learning_insights(self) -> List[str]:
        """生成学习相关洞察"""
        insights = []
        
        if self.performance_metrics["learning_speeds"]:
            avg_speed = np.mean(self.performance_metrics["learning_speeds"][-10:])
            insights.append(f"平均学习速度: {avg_speed:.2f}")
        
        insights.append("建议: 尝试不同的学习策略，如迁移学习或元学习")
        
        return insights
    
    def _generate_adaptation_insights(self, context: Dict[str, Any]) -> List[str]:
        """生成适应相关洞察"""
        insights = []
        
        if self.performance_metrics["adaptation_times"]:
            avg_adaptation = np.mean(self.performance_metrics["adaptation_times"][-5:])
            insights.append(f"平均适应时间: {avg_adaptation:.2f}秒")
        
        insights.append("建议: 建立更灵活的情景识别和应对机制")
        
        return insights
    
    def _generate_general_insights(self) -> List[str]:
        """生成一般性洞察"""
        return [
            "定期反思有助于持续改进",
            "多样化学习经验能提高泛化能力",
            "错误是学习的重要机会",
            "自我监控是智能系统的关键能力"
        ]
    
    def _create_action_plan(self, insights: List[str]) -> Dict[str, Any]:
        """创建改进行动计划"""
        action_plan = {
            "short_term": [],
            "medium_term": [],
            "long_term": []
        }
        
        # 基于洞察生成行动计划
        for insight in insights:
            if "错误" in insight:
                action_plan["short_term"].append("实施错误检测和预防机制")
            elif "性能" in insight:
                action_plan["short_term"].append("优化关键算法性能")
            elif "学习" in insight:
                action_plan["medium_term"].append("尝试新的学习策略")
            elif "适应" in insight:
                action_plan["medium_term"].append("增强情景适应能力")
            else:
                action_plan["long_term"].append("持续监控和改进系统能力")
        
        return action_plan
    
    def _update_meta_state(self, session: ReflectionSession):
        """更新元认知状态"""
        # 更新自我意识水平
        self.meta_state.self_awareness_level = min(1.0, 
            self.meta_state.self_awareness_level + 0.05)
        
        # 更新错误检测敏感性
        if "error_correction" in session.focus_areas:
            self.meta_state.error_detection_sensitivity = min(1.0,
                self.meta_state.error_detection_sensitivity + 0.1)
        
        # 更新学习效率
        if "learning_efficiency" in session.focus_areas:
            self.meta_state.learning_efficiency = min(1.0,
                self.meta_state.learning_efficiency + 0.08)
        
        # 更新适应能力
        if "adaptation_strategy" in session.focus_areas:
            self.meta_state.adaptation_capability = min(1.0,
                self.meta_state.adaptation_capability + 0.07)
        
        # 更新性能趋势
        self.meta_state.performance_trend.append(session.effectiveness_score)
        if len(self.meta_state.performance_trend) > 20:
            self.meta_state.performance_trend = self.meta_state.performance_trend[-20:]
    
    def update_performance_metrics(self, metric_type: str, value: float):
        """更新性能指标"""
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
            # 保持最近100个数据点
            if len(self.performance_metrics[metric_type]) > 100:
                self.performance_metrics[metric_type] = self.performance_metrics[metric_type][-100:]
    
    def evaluate_reflection_effectiveness(self, session_id: str, effectiveness: float):
        """评估反思会话的有效性"""
        for session in self.reflection_history:
            if session.session_id == session_id:
                session.effectiveness_score = effectiveness
                self._save_reflection_history()
                logger.info(f"更新反思会话 {session_id} 的有效性评分: {effectiveness}")
                break
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """获取反思统计信息"""
        return {
            "total_sessions": len(self.reflection_history),
            "recent_effectiveness": np.mean([s.effectiveness_score for s in self.reflection_history[-5:]]) if self.reflection_history else 0,
            "self_awareness": self.meta_state.self_awareness_level,
            "error_sensitivity": self.meta_state.error_detection_sensitivity,
            "learning_efficiency": self.meta_state.learning_efficiency,
            "adaptation_capability": self.meta_state.adaptation_capability
        }
    
    def get_recommendations(self) -> List[str]:
        """获取改进建议"""
        recommendations = []
        
        # 基于元认知状态生成建议
        if self.meta_state.self_awareness_level < 0.8:
            recommendations.append("增加自我监控频率以提高自我意识")
        
        if self.meta_state.error_detection_sensitivity < 0.7:
            recommendations.append("加强错误模式识别训练")
        
        if self.meta_state.learning_efficiency < 0.6:
            recommendations.append("尝试更高效的学习算法和策略")
        
        if self.meta_state.adaptation_capability < 0.65:
            recommendations.append("开发更灵活的情景适应机制")
        
        return recommendations

# 单例实例
self_reflection_module = SelfReflectionModule()

if __name__ == "__main__":
    # 测试代码
    srm = SelfReflectionModule()
    
    print("=== 测试自我反思模块 ===")
    
    # 模拟性能数据
    srm.update_performance_metrics("task_success_rates", 0.85)
    srm.update_performance_metrics("task_success_rates", 0.82)
    srm.update_performance_metrics("task_success_rates", 0.78)
    srm.update_performance_metrics("task_success_rates", 0.75)
    srm.update_performance_metrics("task_success_rates", 0.72)  # 性能下降
    
    # 模拟错误数据
    recent_errors = [
        {"type": "reasoning_error", "message": "逻辑推理错误"},
        {"type": "reasoning_error", "message": "因果判断错误"},
        {"type": "reasoning_error", "message": "假设验证失败"}
    ]
    
    # 检查是否应该反思
    context = {
        "current_performance": 0.72,
        "recent_errors": recent_errors,
        "trigger": "performance_decline"
    }
    
    if srm.should_reflect(context):
        print("检测到需要反思的情况")
        session = srm.conduct_reflection(context)
        print(f"反思会话ID: {session.session_id}")
        print(f"焦点领域: {session.focus_areas}")
        print(f"获得的洞察: {session.insights_gained}")
        print(f"行动计划: {session.action_plan}")
    
    # 显示反思统计
    stats = srm.get_reflection_stats()
    print("\n=== 反思统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 获取改进建议
    recommendations = srm.get_recommendations()
    print("\n=== 改进建议 ===")
    for rec in recommendations:
        print(f"- {rec}")
