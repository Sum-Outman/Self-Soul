"""
Autonomous Decision Quality Assessment

Advanced assessment of decision quality considering multiple dimensions:
- Risk assessment and mitigation
- Uncertainty quantification
- Long-term impact analysis
- Ethical considerations
- Resource optimization
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)

class DecisionQualityAssessor:
    """自主决策质量评估器"""
    
    def __init__(self):
        """初始化决策质量评估器"""
        # 评估维度
        self.assessment_dimensions = {
            'risk_management': {
                'name': '风险管理',
                'weight': 0.25,
                'subdimensions': [
                    'risk_identification',
                    'risk_quantification',
                    'risk_mitigation',
                    'risk_monitoring'
                ]
            },
            'uncertainty_handling': {
                'name': '不确定性处理',
                'weight': 0.20,
                'subdimensions': [
                    'uncertainty_quantification',
                    'robustness_analysis',
                    'sensitivity_analysis',
                    'scenario_planning'
                ]
            },
            'long_term_impact': {
                'name': '长期影响',
                'weight': 0.20,
                'subdimensions': [
                    'sustainability',
                    'scalability',
                    'adaptability',
                    'future_proofing'
                ]
            },
            'ethical_alignment': {
                'name': '伦理对齐',
                'weight': 0.15,
                'subdimensions': [
                    'fairness',
                    'transparency',
                    'accountability',
                    'safety'
                ]
            },
            'resource_optimization': {
                'name': '资源优化',
                'weight': 0.20,
                'subdimensions': [
                    'efficiency',
                    'effectiveness',
                    'cost_benefit',
                    'resource_allocation'
                ]
            }
        }
        
        # 评估标准
        self.assessment_criteria = {
            'risk_management': {
                'risk_identification': {
                    'description': '风险识别能力',
                    'metrics': ['risk_coverage', 'risk_specificity', 'timeliness'],
                    'scoring_range': [0, 1]
                },
                'risk_quantification': {
                    'description': '风险量化能力',
                    'metrics': ['quantification_accuracy', 'probability_estimation', 'impact_assessment'],
                    'scoring_range': [0, 1]
                }
            },
            'uncertainty_handling': {
                'uncertainty_quantification': {
                    'description': '不确定性量化',
                    'metrics': ['uncertainty_range', 'confidence_intervals', 'probability_distributions'],
                    'scoring_range': [0, 1]
                },
                'robustness_analysis': {
                    'description': '鲁棒性分析',
                    'metrics': ['robustness_score', 'failure_tolerance', 'degradation_gracefulness'],
                    'scoring_range': [0, 1]
                }
            },
            'long_term_impact': {
                'sustainability': {
                    'description': '可持续性',
                    'metrics': ['environmental_impact', 'social_impact', 'economic_sustainability'],
                    'scoring_range': [0, 1]
                },
                'scalability': {
                    'description': '可扩展性',
                    'metrics': ['scaling_potential', 'resource_scaling', 'performance_scaling'],
                    'scoring_range': [0, 1]
                }
            },
            'ethical_alignment': {
                'fairness': {
                    'description': '公平性',
                    'metrics': ['bias_detection', 'fairness_metrics', 'equity_assessment'],
                    'scoring_range': [0, 1]
                },
                'transparency': {
                    'description': '透明度',
                    'metrics': ['explainability', 'decision_rationale', 'audit_trail'],
                    'scoring_range': [0, 1]
                }
            },
            'resource_optimization': {
                'efficiency': {
                    'description': '效率',
                    'metrics': ['resource_utilization', 'time_efficiency', 'computational_efficiency'],
                    'scoring_range': [0, 1]
                },
                'effectiveness': {
                    'description': '有效性',
                    'metrics': ['goal_achievement', 'performance_metrics', 'outcome_quality'],
                    'scoring_range': [0, 1]
                }
            }
        }
        
        # 评估历史
        self.assessment_history = []
        self.decision_baselines = {}
        
        logger.info("Decision quality assessment system initialized")
    
    def assess_decision(self, decision_context: Dict[str, Any], 
                       decision_result: Dict[str, Any],
                       decision_strategy: str) -> Dict[str, Any]:
        """
        全面评估决策质量
        
        Args:
            decision_context: 决策上下文
            decision_result: 决策结果
            decision_strategy: 使用的决策策略
            
        Returns:
            评估结果字典
        """
        logger.info(f"Assessing decision quality for strategy: {decision_strategy}")
        
        try:
            # 分析决策上下文
            context_analysis = self._analyze_decision_context(decision_context)
            
            # 分析决策结果
            result_analysis = self._analyze_decision_result(decision_result)
            
            # 评估各维度质量
            dimension_scores = {}
            dimension_details = {}
            
            for dim_id, dim_config in self.assessment_dimensions.items():
                dim_score, dim_detail = self._assess_dimension(
                    dim_id, context_analysis, result_analysis, decision_strategy
                )
                dimension_scores[dim_id] = dim_score
                dimension_details[dim_id] = dim_detail
            
            # 计算总体质量分数
            overall_quality = self._calculate_overall_quality(dimension_scores)
            
            # 生成质量报告
            quality_report = self._generate_quality_report(
                overall_quality, dimension_scores, dimension_details,
                decision_context, decision_result, decision_strategy
            )
            
            # 记录评估历史
            assessment_record = {
                'timestamp': datetime.now().isoformat(),
                'decision_strategy': decision_strategy,
                'overall_quality': overall_quality,
                'dimension_scores': dimension_scores,
                'quality_report': quality_report,
                'context_summary': self._summarize_context(decision_context),
                'result_summary': self._summarize_result(decision_result)
            }
            
            self.assessment_history.append(assessment_record)
            
            logger.info(f"Decision quality assessment completed. Overall quality: {overall_quality:.3f}")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Decision quality assessment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'overall_quality': 0.5,
                'dimension_scores': {},
                'quality_report': {}
            }
    
    def _analyze_decision_context(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """分析决策上下文"""
        analysis = {
            'complexity': 'medium',
            'risk_level': 'medium',
            'uncertainty': 'medium',
            'time_constraint': 'normal',
            'information_completeness': 'partial',
            'stakeholders': [],
            'resources_available': 'moderate',
            'ethical_considerations': []
        }
        
        # 从上下文中提取信息
        if 'complexity' in decision_context:
            analysis['complexity'] = decision_context['complexity']
        
        if 'risk_level' in decision_context:
            analysis['risk_level'] = decision_context['risk_level']
        
        if 'time_constraint' in decision_context:
            analysis['time_constraint'] = decision_context['time_constraint']
        
        if 'stakeholders' in decision_context:
            analysis['stakeholders'] = decision_context['stakeholders']
        
        # 计算上下文质量分数
        context_quality = 0.5
        
        # 信息完整性
        info_keys = ['complexity', 'risk_level', 'time_constraint', 'resources_available']
        info_present = sum(1 for key in info_keys if key in decision_context)
        info_completeness = info_present / len(info_keys)
        
        context_quality = info_completeness * 0.3 + 0.7 * 0.5  # 加权平均
        
        analysis['context_quality'] = context_quality
        analysis['info_completeness'] = info_completeness
        
        return analysis
    
    def _analyze_decision_result(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析决策结果"""
        analysis = {
            'has_quality_metric': 'decision_quality' in decision_result,
            'has_risk_assessment': 'risk_assessment' in decision_result,
            'has_confidence': 'confidence_level' in decision_result,
            'has_execution_plan': 'execution_plan' in decision_result,
            'has_monitoring': 'monitoring_requirements' in decision_result,
            'result_completeness': 0.5
        }
        
        # 计算结果完整性
        required_keys = ['decision_quality', 'risk_assessment', 'confidence_level']
        present_keys = sum(1 for key in required_keys if key in decision_result)
        completeness = present_keys / len(required_keys) if required_keys else 0.0
        
        analysis['result_completeness'] = completeness
        
        # 提取质量相关指标
        if 'decision_quality' in decision_result:
            analysis['decision_quality'] = decision_result['decision_quality']
        
        if 'risk_assessment' in decision_result:
            analysis['risk_assessment'] = decision_result['risk_assessment']
        
        if 'confidence_level' in decision_result:
            analysis['confidence_level'] = decision_result['confidence_level']
        
        return analysis
    
    def _assess_dimension(self, dimension_id: str, context_analysis: Dict[str, Any],
                         result_analysis: Dict[str, Any], decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估单个维度"""
        dim_config = self.assessment_dimensions[dimension_id]
        
        dimension_score = 0.5  # 默认分数
        dimension_detail = {
            'dimension_id': dimension_id,
            'dimension_name': dim_config['name'],
            'assessment_method': 'composite',
            'factors_considered': []
        }
        
        # 根据维度使用不同的评估方法
        if dimension_id == 'risk_management':
            score, detail = self._assess_risk_management(context_analysis, result_analysis, decision_strategy)
        elif dimension_id == 'uncertainty_handling':
            score, detail = self._assess_uncertainty_handling(context_analysis, result_analysis, decision_strategy)
        elif dimension_id == 'long_term_impact':
            score, detail = self._assess_long_term_impact(context_analysis, result_analysis, decision_strategy)
        elif dimension_id == 'ethical_alignment':
            score, detail = self._assess_ethical_alignment(context_analysis, result_analysis, decision_strategy)
        elif dimension_id == 'resource_optimization':
            score, detail = self._assess_resource_optimization(context_analysis, result_analysis, decision_strategy)
        else:
            score = 0.5
            detail = {'method': 'default', 'reason': f"No specific assessment for {dimension_id}"}
        
        dimension_score = score
        dimension_detail['score'] = dimension_score
        dimension_detail.update(detail)
        
        return dimension_score, dimension_detail
    
    def _assess_risk_management(self, context_analysis: Dict[str, Any],
                              result_analysis: Dict[str, Any],
                              decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估风险管理"""
        score = 0.5
        
        assessment_detail = {
            'assessment_method': 'risk_based',
            'factors_considered': []
        }
        
        # 考虑上下文风险水平
        risk_level = context_analysis.get('risk_level', 'medium')
        risk_multiplier = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6,
            'critical': 0.4
        }.get(risk_level, 0.8)
        
        score *= risk_multiplier
        assessment_detail['factors_considered'].append(f'risk_level_{risk_level}')
        
        # 考虑结果中的风险评估
        if result_analysis.get('has_risk_assessment', False):
            risk_assessment = result_analysis.get('risk_assessment', 'medium')
            # 风险评估与上下文风险水平的一致性
            if risk_level == risk_assessment:
                score += 0.1  # 一致性好
                assessment_detail['factors_considered'].append('risk_assessment_consistent')
            else:
                score -= 0.1  # 不一致性差
                assessment_detail['factors_considered'].append('risk_assessment_inconsistent')
        
        # 考虑决策策略的风险处理能力
        strategy_risk_handling = {
            'conservative': 0.8,
            'balanced': 0.7,
            'proactive': 0.6
        }.get(decision_strategy, 0.7)
        
        score = (score + strategy_risk_handling) / 2
        assessment_detail['factors_considered'].append(f'strategy_{decision_strategy}_risk_handling')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_uncertainty_handling(self, context_analysis: Dict[str, Any],
                                   result_analysis: Dict[str, Any],
                                   decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估不确定性处理"""
        score = 0.5
        
        assessment_detail = {
            'assessment_method': 'uncertainty_based',
            'factors_considered': []
        }
        
        # 考虑信息完整性
        info_completeness = context_analysis.get('info_completeness', 0.5)
        score = (score + info_completeness) / 2
        assessment_detail['factors_considered'].append(f'info_completeness_{info_completeness:.2f}')
        
        # 考虑结果中的置信度
        if 'confidence_level' in result_analysis:
            confidence = result_analysis['confidence_level']
            # 置信度应与不确定性水平相匹配
            uncertainty = context_analysis.get('uncertainty', 'medium')
            uncertainty_factor = {
                'low': 0.9,
                'medium': 0.7,
                'high': 0.5
            }.get(uncertainty, 0.7)
            
            # 理想的置信度应与不确定性负相关
            ideal_confidence = 1.0 - (1.0 - uncertainty_factor) * 0.5
            confidence_diff = abs(confidence - ideal_confidence)
            confidence_penalty = confidence_diff * 0.5
            
            score -= confidence_penalty
            assessment_detail['factors_considered'].append('confidence_penalty_applied')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_long_term_impact(self, context_analysis: Dict[str, Any],
                               result_analysis: Dict[str, Any],
                               decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估长期影响"""
        score = 0.5
        
        assessment_detail = {
            'assessment_method': 'long_term_based',
            'factors_considered': []
        }
        
        # 考虑时间约束
        time_constraint = context_analysis.get('time_constraint', 'normal')
        time_factor = {
            'urgent': 0.6,      # 紧急决策可能忽视长期影响
            'normal': 0.7,
            'relaxed': 0.8      # 宽松时间可以考虑更长期
        }.get(time_constraint, 0.7)
        
        score *= time_factor
        assessment_detail['factors_considered'].append(f'time_constraint_{time_constraint}')
        
        # 考虑决策策略的长期导向
        strategy_long_term = {
            'conservative': 0.6,   # 保守策略可能过于关注短期风险
            'balanced': 0.7,
            'proactive': 0.8       # 主动策略可能更考虑长期
        }.get(decision_strategy, 0.7)
        
        score = (score + strategy_long_term) / 2
        assessment_detail['factors_considered'].append(f'strategy_{decision_strategy}_long_term')
        
        # 检查是否有执行计划（表明考虑了实施）
        if result_analysis.get('has_execution_plan', False):
            score += 0.1
            assessment_detail['factors_considered'].append('has_execution_plan')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_ethical_alignment(self, context_analysis: Dict[str, Any],
                                result_analysis: Dict[str, Any],
                                decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估伦理对齐"""
        score = 0.5
        
        assessment_detail = {
            'assessment_method': 'ethics_based',
            'factors_considered': []
        }
        
        # 考虑利益相关者
        stakeholders = context_analysis.get('stakeholders', [])
        if stakeholders:
            # 有利益相关者考虑，伦理关注度更高
            score += 0.1
            assessment_detail['factors_considered'].append(f'stakeholders_considered_{len(stakeholders)}')
        
        # 考虑决策策略的伦理倾向
        strategy_ethics = {
            'conservative': 0.7,   # 保守策略通常更谨慎
            'balanced': 0.6,
            'proactive': 0.5       # 主动策略可能更冒险
        }.get(decision_strategy, 0.6)
        
        score = (score + strategy_ethics) / 2
        assessment_detail['factors_considered'].append(f'strategy_{decision_strategy}_ethics')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_resource_optimization(self, context_analysis: Dict[str, Any],
                                    result_analysis: Dict[str, Any],
                                    decision_strategy: str) -> Tuple[float, Dict[str, Any]]:
        """评估资源优化"""
        score = 0.5
        
        assessment_detail = {
            'assessment_method': 'resource_based',
            'factors_considered': []
        }
        
        # 考虑可用资源
        resources = context_analysis.get('resources_available', 'moderate')
        resource_factor = {
            'limited': 0.6,     # 资源有限需要更优化
            'moderate': 0.7,
            'abundant': 0.8     # 资源丰富可能不够优化
        }.get(resources, 0.7)
        
        score *= resource_factor
        assessment_detail['factors_considered'].append(f'resources_{resources}')
        
        # 考虑决策策略的资源效率
        strategy_efficiency = {
            'conservative': 0.6,   # 保守策略可能资源利用不足
            'balanced': 0.7,
            'proactive': 0.8       # 主动策略可能更高效
        }.get(decision_strategy, 0.7)
        
        score = (score + strategy_efficiency) / 2
        assessment_detail['factors_considered'].append(f'strategy_{decision_strategy}_efficiency')
        
        # 检查是否有监控要求（表明考虑了资源监控）
        if result_analysis.get('has_monitoring', False):
            score += 0.1
            assessment_detail['factors_considered'].append('has_monitoring')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _calculate_overall_quality(self, dimension_scores: Dict[str, float]) -> float:
        """计算总体质量分数"""
        overall_quality = 0.0
        total_weight = 0.0
        
        for dim_id, dim_score in dimension_scores.items():
            weight = self.assessment_dimensions[dim_id]['weight']
            overall_quality += dim_score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_quality /= total_weight
        
        return overall_quality
    
    def _generate_quality_report(self, overall_quality: float,
                               dimension_scores: Dict[str, float],
                               dimension_details: Dict[str, Any],
                               decision_context: Dict[str, Any],
                               decision_result: Dict[str, Any],
                               decision_strategy: str) -> Dict[str, Any]:
        """生成质量报告"""
        quality_report = {
            'overall_quality_score': overall_quality,
            'quality_level': self._determine_quality_level(overall_quality),
            'decision_strategy': decision_strategy,
            'strengths': [],
            'weaknesses': [],
            'improvement_recommendations': [],
            'dimension_analysis': {}
        }
        
        # 分析各维度表现
        for dim_id, dim_score in dimension_scores.items():
            dim_config = self.assessment_dimensions[dim_id]
            dim_detail = dimension_details[dim_id]
            
            quality_report['dimension_analysis'][dim_id] = {
                'name': dim_config['name'],
                'score': dim_score,
                'weight': dim_config['weight'],
                'assessment_detail': dim_detail
            }
            
            # 识别优势和弱点
            if dim_score >= 0.7:
                quality_report['strengths'].append(dim_config['name'])
            elif dim_score <= 0.4:
                quality_report['weaknesses'].append(dim_config['name'])
        
        # 生成改进建议
        for dim_id, dim_score in dimension_scores.items():
            if dim_score < 0.6:
                dim_config = self.assessment_dimensions[dim_id]
                recommendation = {
                    'dimension': dim_config['name'],
                    'current_score': dim_score,
                    'target_score': 0.7,
                    'suggestion': f"Improve {dim_config['name'].lower()} through better analysis and planning"
                }
                quality_report['improvement_recommendations'].append(recommendation)
        
        # 添加上下文摘要
        quality_report['context_summary'] = self._summarize_context(decision_context)
        
        # 添加结果摘要
        quality_report['result_summary'] = self._summarize_result(decision_result)
        
        return quality_report
    
    def _determine_quality_level(self, overall_quality: float) -> str:
        """根据总体质量确定等级"""
        if overall_quality >= 0.9:
            return "卓越"
        elif overall_quality >= 0.8:
            return "优秀"
        elif overall_quality >= 0.7:
            return "良好"
        elif overall_quality >= 0.6:
            return "合格"
        elif overall_quality >= 0.5:
            return "需要改进"
        elif overall_quality >= 0.4:
            return "不足"
        else:
            return "严重不足"
    
    def _summarize_context(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """摘要决策上下文"""
        summary = {
            'key_factors': [],
            'risk_level': decision_context.get('risk_level', 'unknown'),
            'complexity': decision_context.get('complexity', 'unknown'),
            'time_constraint': decision_context.get('time_constraint', 'unknown'),
            'has_stakeholders': 'stakeholders' in decision_context
        }
        
        # 提取关键因素
        for key in ['complexity', 'risk_level', 'time_constraint', 'resources_available']:
            if key in decision_context:
                summary['key_factors'].append(f"{key}: {decision_context[key]}")
        
        return summary
    
    def _summarize_result(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """摘要决策结果"""
        summary = {
            'has_quality_metric': 'decision_quality' in decision_result,
            'has_risk_assessment': 'risk_assessment' in decision_result,
            'has_confidence': 'confidence_level' in decision_result,
            'has_execution_plan': 'execution_plan' in decision_result
        }
        
        if 'decision_quality' in decision_result:
            summary['decision_quality'] = decision_result['decision_quality']
        
        if 'risk_assessment' in decision_result:
            summary['risk_assessment'] = decision_result['risk_assessment']
        
        if 'confidence_level' in decision_result:
            summary['confidence_level'] = decision_result['confidence_level']
        
        return summary
    
    def get_assessment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.assessment_history[-limit:] if self.assessment_history else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        if not self.assessment_history:
            return {'error': 'No assessment history available'}
        
        overall_scores = [record['overall_quality'] for record in self.assessment_history]
        
        stats = {
            'total_assessments': len(self.assessment_history),
            'average_quality': np.mean(overall_scores) if overall_scores else 0.0,
            'max_quality': np.max(overall_scores) if overall_scores else 0.0,
            'min_quality': np.min(overall_scores) if overall_scores else 0.0,
            'std_dev_quality': np.std(overall_scores) if overall_scores else 0.0,
            'recent_trend': self._calculate_recent_trend(overall_scores)
        }
        
        return stats
    
    def _calculate_recent_trend(self, scores: List[float]) -> str:
        """计算近期趋势"""
        if len(scores) < 2:
            return "insufficient_data"
        
        recent_scores = scores[-5:] if len(scores) >= 5 else scores
        if len(recent_scores) < 2:
            return "stable"
        
        # 简单线性趋势
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
        except:
            return "unknown"