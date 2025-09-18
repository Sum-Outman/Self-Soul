"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""
import numpy as np
import time
from datetime import datetime, timedelta
from collections import deque
import random
from core.error_handling import error_handler

class CuriosityModule:
    """好奇心模块 - 驱动探索和学习"""
    
    def __init__(self):
        self.exploration_history = deque(maxlen=1000)
        self.novelty_threshold = 0.3
        self.learning_progress_threshold = 0.1
        self.curiosity_level = 0.7  # 初始好奇心水平
    
    def calculate_curiosity_score(self, state, context):
        """计算好奇心得分"""
        novelty = self._calculate_novelty(state, context)
        learning_potential = self._estimate_learning_potential(state)
        uncertainty = self._measure_uncertainty(state)
        
        curiosity_score = (novelty * 0.4 + learning_potential * 0.3 + uncertainty * 0.3) * self.curiosity_level
        
        return {
            'curiosity_score': curiosity_score,
            'novelty': novelty,
            'learning_potential': learning_potential,
            'uncertainty': uncertainty,
            'recommended_action': curiosity_score > 0.5
        }
    
    def _calculate_novelty(self, state, context):
        """计算新颖性"""
        # 简化的新颖性计算
        state_hash = hash(str(state))
        if any(exp['state_hash'] == state_hash for exp in self.exploration_history):
            return 0.2  # 已经探索过，新颖性低
        
        return 0.8  # 新状态，新颖性高
    
    def _estimate_learning_potential(self, state):
        """估计学习潜力"""
        # 基于状态的复杂性估计学习潜力
        complexity = min(len(str(state)) / 1000, 1.0)
        return complexity * 0.6 + 0.2
    
    def _measure_uncertainty(self, state):
        """测量不确定性"""
        # 简化的不确定性测量
        return random.uniform(0.3, 0.7)
    
    def record_exploration(self, state, outcome):
        """记录探索结果"""
        state_hash = hash(str(state))
        self.exploration_history.append({
            'state_hash': state_hash,
            'timestamp': time.time(),
            'outcome': outcome,
            'learning_gain': self._calculate_learning_gain(outcome)
        })
        
        # 根据结果调整好奇心水平
        if outcome.get('success', False):
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
        else:
            self.curiosity_level = max(0.1, self.curiosity_level - 0.02)

class CompetenceModule:
    """能力模块 - 追求技能掌握"""
    
    def __init__(self):
        self.skill_levels = {}
        self.challenge_history = deque(maxlen=500)
        self.optimal_challenge_zone = 0.6  # 最佳挑战区域
    
    def assess_skill_level(self, skill_domain, performance_metrics):
        """评估技能水平"""
        if skill_domain not in self.skill_levels:
            self.skill_levels[skill_domain] = {
                'level': 0.3,
                'confidence': 0.5,
                'last_assessed': time.time()
            }
        
        # 基于性能指标更新技能水平
        performance_score = self._calculate_performance_score(performance_metrics)
        current_level = self.skill_levels[skill_domain]['level']
        
        # 技能水平更新
        new_level = current_level * 0.8 + performance_score * 0.2
        self.skill_levels[skill_domain]['level'] = new_level
        self.skill_levels[skill_domain]['confidence'] = min(1.0, 
            self.skill_levels[skill_domain]['confidence'] + 0.1)
        self.skill_levels[skill_domain]['last_assessed'] = time.time()
        
        return new_level
    
    def _calculate_performance_score(self, metrics):
        """计算性能得分"""
        # 简化的性能计算
        if 'accuracy' in metrics:
            return metrics['accuracy']
        elif 'success_rate' in metrics:
            return metrics['success_rate']
        else:
            return 0.5  # 默认得分
    
    def recommend_challenge(self, skill_domain):
        """推荐适当难度的挑战"""
        current_level = self.skill_levels.get(skill_domain, {}).get('level', 0.3)
        
        # 在最佳挑战区域附近推荐难度
        challenge_difficulty = max(0.1, min(0.9, 
            current_level + random.uniform(-0.2, 0.2)))
        
        return {
            'recommended_difficulty': challenge_difficulty,
            'is_optimal': abs(challenge_difficulty - current_level) <= 0.2,
            'skill_gap': challenge_difficulty - current_level
        }

class AutonomyModule:
    """自主性模块 - 追求自我决定"""
    
    def __init__(self):
        self.decision_history = deque(maxlen=1000)
        self.autonomy_level = 0.5
        self.preference_profile = self._initialize_preferences()
    
    def _initialize_preferences(self):
        """初始化偏好配置"""
        return {
            'exploration_vs_exploitation': 0.6,
            'risk_tolerance': 0.4,
            'novelty_seeking': 0.7,
            'social_interaction': 0.3
        }
    
    def evaluate_autonomy(self, decision_context):
        """评估自主性需求"""
        autonomy_score = self._calculate_autonomy_score(decision_context)
        autonomy_satisfaction = min(1.0, autonomy_score / self.autonomy_level)
        
        return {
            'autonomy_score': autonomy_score,
            'autonomy_satisfaction': autonomy_satisfaction,
            'needs_more_autonomy': autonomy_satisfaction < 0.7,
            'recommended_action': 'increase_autonomy' if autonomy_satisfaction < 0.7 else 'maintain'
        }
    
    def _calculate_autonomy_score(self, context):
        """计算自主性得分"""
        # 基于决策自由度计算
        decision_freedom = context.get('decision_freedom', 0.5)
        choice_variety = context.get('choice_variety', 0.3)
        external_constraints = context.get('external_constraints', 0.0)
        
        return (decision_freedom * 0.6 + choice_variety * 0.4) * (1 - external_constraints)
    
    def update_preferences(self, experience_outcome):
        """基于经验更新偏好"""
        # 简化的偏好更新
        if experience_outcome.get('success', False):
            self.preference_profile['risk_tolerance'] = min(1.0, 
                self.preference_profile['risk_tolerance'] + 0.05)
        else:
            self.preference_profile['risk_tolerance'] = max(0.1, 
                self.preference_profile['risk_tolerance'] - 0.03)

class RelatednessModule:
    """关联性模块 - 追求社会连接"""
    
    def __init__(self):
        self.social_connections = {}
        self.collaboration_history = deque(maxlen=500)
        self.relatedness_need = 0.4  # 关联性需求水平
    
    def assess_relatedness(self, social_context):
        """评估关联性需求"""
        connection_quality = self._evaluate_connections(social_context)
        collaboration_opportunities = social_context.get('collaboration_opportunities', 0.3)
        
        relatedness_score = connection_quality * 0.7 + collaboration_opportunities * 0.3
        relatedness_satisfaction = min(1.0, relatedness_score / self.relatedness_need)
        
        return {
            'relatedness_score': relatedness_score,
            'relatedness_satisfaction': relatedness_satisfaction,
            'needs_more_connection': relatedness_satisfaction < 0.6,
            'recommended_actions': self._generate_connection_suggestions(social_context)
        }
    
    def _evaluate_connections(self, social_context):
        """评估连接质量"""
        # 简化的连接评估
        active_connections = social_context.get('active_connections', 0)
        connection_depth = social_context.get('connection_depth', 0.3)
        
        return min(1.0, active_connections * 0.2 + connection_depth * 0.8)
    
    def _generate_connection_suggestions(self, social_context):
        """生成连接建议"""
        suggestions = []
        
        if social_context.get('active_connections', 0) < 3:
            suggestions.append('建立新的协作关系')
        
        if social_context.get('connection_depth', 0) < 0.5:
            suggestions.append('深化现有连接')
        
        if not suggestions:
            suggestions.append('维持当前连接状态')
        
        return suggestions

class IntrinsicMotivationSystem:
    """内在动机系统 - 基于自我决定理论"""
    
    def __init__(self):
        self.curiosity_module = CuriosityModule()
        self.competence_module = CompetenceModule()
        self.autonomy_module = AutonomyModule()
        self.relatedness_module = RelatednessModule()
        
        self.motivation_state = {
            'overall_motivation': 0.6,
            'last_updated': time.time(),
            'trend': 'stable'
        }
        
        error_handler.log_info("内在动机系统初始化完成", "IntrinsicMotivationSystem")
    
    def assess_motivation(self, current_state, context=None):
        """综合评估动机状态"""
        try:
            # 评估各个动机维度
            curiosity_assessment = self.curiosity_module.calculate_curiosity_score(current_state, context or {})
            competence_assessment = self._assess_competence(context or {})
            autonomy_assessment = self.autonomy_module.evaluate_autonomy(context or {})
            relatedness_assessment = self.relatedness_module.assess_relatedness(context or {})
            
            # 计算总体动机
            overall_motivation = self._calculate_overall_motivation(
                curiosity_assessment, competence_assessment, 
                autonomy_assessment, relatedness_assessment
            )
            
            # 更新动机状态
            self.motivation_state = {
                'overall_motivation': overall_motivation,
                'curiosity': curiosity_assessment['curiosity_score'],
                'competence': competence_assessment['competence_level'],
                'autonomy': autonomy_assessment['autonomy_score'],
                'relatedness': relatedness_assessment['relatedness_score'],
                'last_updated': time.time(),
                'trend': self._determine_motivation_trend(overall_motivation)
            }
            
            return {
                'motivation_state': self.motivation_state,
                'dimensional_assessments': {
                    'curiosity': curiosity_assessment,
                    'competence': competence_assessment,
                    'autonomy': autonomy_assessment,
                    'relatedness': relatedness_assessment
                },
                'recommendations': self._generate_motivation_recommendations()
            }
            
        except Exception as e:
            error_handler.handle_error(e, "IntrinsicMotivationSystem", "动机评估失败")
            return {"error": str(e)}
    
    def _assess_competence(self, context):
        """评估能力维度"""
        skill_domain = context.get('skill_domain', 'general')
        performance_metrics = context.get('performance_metrics', {})
        
        competence_level = self.competence_module.assess_skill_level(skill_domain, performance_metrics)
        challenge_recommendation = self.competence_module.recommend_challenge(skill_domain)
        
        return {
            'competence_level': competence_level,
            'skill_domain': skill_domain,
            'challenge_recommendation': challenge_recommendation,
            'confidence': self.competence_module.skill_levels.get(skill_domain, {}).get('confidence', 0.5)
        }
    
    def _calculate_overall_motivation(self, curiosity, competence, autonomy, relatedness):
        """计算总体动机"""
        # 基于自我决定理论的权重
        weights = {
            'autonomy': 0.4,      # 自主性最重要
            'competence': 0.3,    # 能力次之
            'relatedness': 0.2,   # 关联性
            'curiosity': 0.1      # 好奇心（包含在能力中）
        }
        
        return (
            autonomy['autonomy_score'] * weights['autonomy'] +
            competence['competence_level'] * weights['competence'] +
            relatedness['relatedness_score'] * weights['relatedness'] +
            curiosity['curiosity_score'] * weights['curiosity']
        )
    
    def _determine_motivation_trend(self, current_motivation):
        """确定动机趋势"""
        # 简化的趋势分析
        if 'previous_motivation' not in self.motivation_state:
            return 'stable'
        
        previous = self.motivation_state['previous_motivation']
        if current_motivation > previous + 0.1:
            return 'improving'
        elif current_motivation < previous - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_motivation_recommendations(self):
        """生成动机提升建议"""
        recommendations = []
        state = self.motivation_state
        
        if state['curiosity'] < 0.4:
            recommendations.append('增加探索性活动以提升好奇心')
        
        if state['competence'] < 0.5:
            recommendations.append('专注于技能提升和掌握')
        
        if state['autonomy'] < 0.6:
            recommendations.append('增加决策自主权')
        
        if state['relatedness'] < 0.3:
            recommendations.append('加强社会连接和协作')
        
        if not recommendations:
            recommendations.append('动机状态良好，继续保持当前活动')
        
        return recommendations
    
    def update_based_on_experience(self, experience):
        """基于经验更新动机系统"""
        try:
            # 更新好奇心模块
            if 'state' in experience and 'outcome' in experience:
                self.curiosity_module.record_exploration(experience['state'], experience['outcome'])
            
            # 更新能力模块
            if 'skill_domain' in experience and 'performance' in experience:
                self.competence_module.assess_skill_level(
                    experience['skill_domain'], experience['performance'])
            
            # 更新自主性模块
            if 'decision_context' in experience:
                self.autonomy_module.update_preferences(experience.get('outcome', {}))
            
            # 重新评估动机
            return self.assess_motivation({}, {})
            
        except Exception as e:
            error_handler.handle_error(e, "IntrinsicMotivationSystem", "经验更新失败")
            return {"error": str(e)}
    
    def get_motivation_report(self):
        """获取动机状态报告"""
        return {
            'current_state': self.motivation_state,
            'curiosity_stats': {
                'exploration_count': len(self.curiosity_module.exploration_history),
                'curiosity_level': self.curiosity_module.curiosity_level
            },
            'competence_stats': {
                'tracked_skills': len(self.competence_module.skill_levels),
                'average_skill_level': np.mean([s['level'] for s in self.competence_module.skill_levels.values()]) 
                if self.competence_module.skill_levels else 0
            },
            'autonomy_stats': {
                'decision_count': len(self.autonomy_module.decision_history),
                'autonomy_level': self.autonomy_module.autonomy_level
            },
            'relatedness_stats': {
                'social_connections': len(self.relatedness_module.social_connections),
                'collaboration_count': len(self.relatedness_module.collaboration_history)
            },
            'report_timestamp': datetime.now().isoformat()
        }