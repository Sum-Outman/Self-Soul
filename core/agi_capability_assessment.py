"""
AGI Capability Assessment System

Comprehensive evaluation指标体系 for assessing AGI capabilities across models.
Provides multi-dimensional assessment including reasoning, decision-making,
emotional intelligence, knowledge integration, and autonomous learning.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)

class AGICapabilityAssessment:
    """AGI能力评估系统"""
    
    def __init__(self):
        """Initialize AGI capability assessment system"""
        # 定义评估维度
        self.dimensions = {
            'reasoning_capability': {
                'name': '推理能力',
                'description': '逻辑推理、因果推断、归纳演绎等能力',
                'weight': 0.25,
                'subdimensions': [
                    'logical_reasoning',
                    'causal_inference',
                    'analogical_reasoning',
                    'counterfactual_reasoning',
                    'abductive_reasoning'
                ]
            },
            'decision_making': {
                'name': '决策能力',
                'description': '自主决策、风险评估、策略选择等能力',
                'weight': 0.20,
                'subdimensions': [
                    'autonomous_decision',
                    'risk_assessment',
                    'strategy_selection',
                    'decision_quality',
                    'execution_planning'
                ]
            },
            'emotional_intelligence': {
                'name': '情感智能',
                'description': '情感感知、情感适配、情感推理等能力',
                'weight': 0.15,
                'subdimensions': [
                    'emotion_perception',
                    'emotion_adaptation',
                    'emotion_reasoning',
                    'social_intelligence',
                    'empathy'
                ]
            },
            'knowledge_integration': {
                'name': '知识整合',
                'description': '知识获取、知识融合、知识推理等能力',
                'weight': 0.20,
                'subdimensions': [
                    'knowledge_acquisition',
                    'knowledge_fusion',
                    'cross_domain_reasoning',
                    'knowledge_graph_construction',
                    'semantic_understanding'
                ]
            },
            'autonomous_learning': {
                'name': '自主学习',
                'description': '自主学习、自我优化、元学习等能力',
                'weight': 0.20,
                'subdimensions': [
                    'self_supervised_learning',
                    'meta_learning',
                    'self_improvement',
                    'adaptation_speed',
                    'generalization'
                ]
            }
        }
        
        # 评估标准
        self.evaluation_criteria = {
            'reasoning_capability': {
                'logical_reasoning': {
                    'description': '形式逻辑推理能力',
                    'metrics': ['accuracy', 'complexity', 'speed'],
                    'scoring_range': [0, 1]
                },
                'causal_inference': {
                    'description': '因果关系推断能力',
                    'metrics': ['precision', 'recall', 'f1_score'],
                    'scoring_range': [0, 1]
                },
                'analogical_reasoning': {
                    'description': '类比推理能力',
                    'metrics': ['similarity_accuracy', 'transfer_effectiveness'],
                    'scoring_range': [0, 1]
                }
            },
            'decision_making': {
                'autonomous_decision': {
                    'description': '自主决策能力',
                    'metrics': ['decision_quality', 'autonomy_level', 'success_rate'],
                    'scoring_range': [0, 1]
                },
                'risk_assessment': {
                    'description': '风险评估能力',
                    'metrics': ['risk_accuracy', 'risk_precision', 'risk_recall'],
                    'scoring_range': [0, 1]
                }
            },
            'emotional_intelligence': {
                'emotion_perception': {
                    'description': '情感感知能力',
                    'metrics': ['perception_accuracy', 'emotion_range', 'context_sensitivity'],
                    'scoring_range': [0, 1]
                },
                'emotion_adaptation': {
                    'description': '情感适配能力',
                    'metrics': ['adaptation_speed', 'adaptation_effectiveness', 'stability'],
                    'scoring_range': [0, 1]
                }
            },
            'knowledge_integration': {
                'knowledge_acquisition': {
                    'description': '知识获取能力',
                    'metrics': ['acquisition_speed', 'knowledge_quality', 'diversity'],
                    'scoring_range': [0, 1]
                },
                'knowledge_fusion': {
                    'description': '知识融合能力',
                    'metrics': ['fusion_accuracy', 'integration_depth', 'cross_domain_ability'],
                    'scoring_range': [0, 1]
                }
            },
            'autonomous_learning': {
                'self_supervised_learning': {
                    'description': '自监督学习能力',
                    'metrics': ['learning_efficiency', 'representation_quality', 'task_performance'],
                    'scoring_range': [0, 1]
                },
                'meta_learning': {
                    'description': '元学习能力',
                    'metrics': ['fast_adaptation', 'few_shot_performance', 'transfer_learning'],
                    'scoring_range': [0, 1]
                }
            }
        }
        
        # 评估历史
        self.assessment_history = []
        self.model_baselines = {}
        
        logger.info("AGI capability assessment system initialized")
    
    def assess_model(self, model, assessment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        全面评估模型的AGI能力
        
        Args:
            model: 待评估的模型对象
            assessment_data: 可选的评估数据
            
        Returns:
            评估结果字典
        """
        logger.info(f"Starting AGI capability assessment for model: {model.__class__.__name__}")
        
        try:
            # 收集模型数据
            model_data = self._collect_model_data(model, assessment_data)
            
            # 评估各维度能力
            dimension_scores = {}
            dimension_details = {}
            
            for dim_id, dim_config in self.dimensions.items():
                dim_score, dim_detail = self._assess_dimension(dim_id, model_data)
                dimension_scores[dim_id] = dim_score
                dimension_details[dim_id] = dim_detail
            
            # 计算总体AGI能力分数
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # 生成能力画像
            capability_profile = self._generate_capability_profile(
                overall_score, dimension_scores, dimension_details
            )
            
            # 记录评估结果
            assessment_record = {
                'model_id': model_data.get('model_id', 'unknown'),
                'model_class': model.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'overall_score': overall_score,
                'dimension_scores': dimension_scores,
                'capability_profile': capability_profile,
                'assessment_version': '1.0'
            }
            
            self.assessment_history.append(assessment_record)
            
            logger.info(f"AGI assessment completed. Overall score: {overall_score:.3f}")
            return assessment_record
            
        except Exception as e:
            logger.error(f"AGI capability assessment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_class': model.__class__.__name__
            }
    
    def _collect_model_data(self, model, assessment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """收集模型数据用于评估"""
        model_data = {
            'model_id': getattr(model, 'model_id', model.__class__.__name__),
            'model_class': model.__class__.__name__,
            'collection_time': time.time()
        }
        
        # 收集AGI能力数据
        if hasattr(model, 'get_agi_capabilities'):
            try:
                agi_capabilities = model.get_agi_capabilities()
                model_data['agi_capabilities'] = agi_capabilities
            except Exception as e:
                logger.warning(f"Failed to get AGI capabilities: {e}")
        
        # 收集情感状态数据
        if hasattr(model, 'get_emotion_state'):
            try:
                emotion_state = model.get_emotion_state()
                model_data['emotion_state'] = emotion_state
            except Exception as e:
                logger.warning(f"Failed to get emotion state: {e}")
        
        # 收集性能指标
        if hasattr(model, '_performance_metrics'):
            try:
                performance_metrics = model._performance_metrics
                model_data['performance_metrics'] = performance_metrics
            except Exception as e:
                logger.warning(f"Failed to get performance metrics: {e}")
        
        # 收集推理历史
        if hasattr(model, '_reasoning_history'):
            try:
                reasoning_history = model._reasoning_history
                model_data['reasoning_history'] = reasoning_history[-10:]  # 最近10条
            except Exception as e:
                logger.warning(f"Failed to get reasoning history: {e}")
        
        # 整合外部评估数据
        if assessment_data:
            model_data.update(assessment_data)
        
        return model_data
    
    def _assess_dimension(self, dimension_id: str, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估单个维度"""
        dim_config = self.dimensions[dimension_id]
        criteria = self.evaluation_criteria.get(dimension_id, {})
        
        dimension_score = 0.0
        dimension_detail = {
            'dimension_id': dimension_id,
            'dimension_name': dim_config['name'],
            'subdimension_scores': {},
            'assessment_methods': [],
            'confidence': 0.0
        }
        
        # 根据维度使用不同的评估方法
        if dimension_id == 'reasoning_capability':
            score, detail = self._assess_reasoning_capability(model_data)
        elif dimension_id == 'decision_making':
            score, detail = self._assess_decision_making(model_data)
        elif dimension_id == 'emotional_intelligence':
            score, detail = self._assess_emotional_intelligence(model_data)
        elif dimension_id == 'knowledge_integration':
            score, detail = self._assess_knowledge_integration(model_data)
        elif dimension_id == 'autonomous_learning':
            score, detail = self._assess_autonomous_learning(model_data)
        else:
            score = 0.5
            detail = {'method': 'default', 'reason': f"No specific assessment for {dimension_id}"}
        
        dimension_score = score
        dimension_detail['score'] = dimension_score
        dimension_detail['assessment_detail'] = detail
        
        return dimension_score, dimension_detail
    
    def _assess_reasoning_capability(self, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估推理能力"""
        score = 0.5  # 默认分数
        
        assessment_detail = {
            'method': 'composite_assessment',
            'factors_considered': [],
            'reasoning_samples': []
        }
        
        # 检查AGI能力报告
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'reasoning_engine' in agi_cap:
                score += 0.2
                assessment_detail['factors_considered'].append('reasoning_engine_present')
        
        # 检查推理历史
        if 'reasoning_history' in model_data:
            reasoning_history = model_data['reasoning_history']
            if len(reasoning_history) > 0:
                score += 0.1
                assessment_detail['reasoning_samples'] = reasoning_history
                assessment_detail['factors_considered'].append('reasoning_history_available')
        
        # 检查性能指标中的推理准确率
        if 'performance_metrics' in model_data:
            perf_metrics = model_data['performance_metrics']
            if 'reasoning_accuracy' in perf_metrics:
                reasoning_acc = perf_metrics['reasoning_accuracy'].get('current', 0.5)
                score = (score + reasoning_acc) / 2
                assessment_detail['factors_considered'].append('reasoning_accuracy_metric')
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_decision_making(self, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估决策能力"""
        score = 0.5
        
        assessment_detail = {
            'method': 'composite_assessment',
            'factors_considered': []
        }
        
        # 检查自主决策能力
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'decision_maker' in agi_cap:
                score += 0.2
                assessment_detail['factors_considered'].append('decision_maker_present')
        
        # 检查性能指标中的决策质量
        if 'performance_metrics' in model_data:
            perf_metrics = model_data['performance_metrics']
            if 'decision_quality' in perf_metrics:
                decision_quality = perf_metrics['decision_quality'].get('current', 0.5)
                score = (score + decision_quality) / 2
                assessment_detail['factors_considered'].append('decision_quality_metric')
        
        # 检查情感状态对决策的影响
        if 'emotion_state' in model_data:
            emotion_state = model_data['emotion_state']
            confidence = emotion_state.get('confidence', 0.5)
            # 高自信心可能提升决策质量
            score = score * 0.7 + confidence * 0.3
            assessment_detail['factors_considered'].append('emotion_confidence')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_emotional_intelligence(self, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估情感智能"""
        score = 0.5
        
        assessment_detail = {
            'method': 'composite_assessment',
            'factors_considered': []
        }
        
        # 检查情感状态
        if 'emotion_state' in model_data:
            emotion_state = model_data['emotion_state']
            # 评估情感状态的复杂性和稳定性
            emotion_complexity = len(emotion_state)
            if emotion_complexity >= 5:  # 至少有5个情感维度
                score += 0.2
                assessment_detail['factors_considered'].append('emotion_state_complexity')
            
            # 检查情感稳定性（压力水平低）
            stress_level = emotion_state.get('stress_level', 0.5)
            if stress_level < 0.3:
                score += 0.1
                assessment_detail['factors_considered'].append('low_stress_level')
        
        # 检查情感适配能力
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'cognitive_functions' in agi_cap:
                cognitive_funcs = agi_cap['cognitive_functions']
                if isinstance(cognitive_funcs, list) and len(cognitive_funcs) > 0:
                    score += 0.1
                    assessment_detail['factors_considered'].append('cognitive_functions_available')
        
        # 检查性能指标中的情感稳定性
        if 'performance_metrics' in model_data:
            perf_metrics = model_data['performance_metrics']
            if 'emotion_stability' in perf_metrics:
                emotion_stability = perf_metrics['emotion_stability'].get('current', 0.5)
                score = (score + emotion_stability) / 2
                assessment_detail['factors_considered'].append('emotion_stability_metric')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_knowledge_integration(self, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估知识整合能力"""
        score = 0.5
        
        assessment_detail = {
            'method': 'composite_assessment',
            'factors_considered': []
        }
        
        # 检查知识库大小
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'knowledge_base_size' in agi_cap:
                kb_size = agi_cap['knowledge_base_size']
                if isinstance(kb_size, int) and kb_size > 0:
                    # 对数缩放知识库大小
                    kb_score = min(1.0, np.log1p(kb_size) / 10.0)
                    score = (score + kb_score) / 2
                    assessment_detail['factors_considered'].append('knowledge_base_size')
        
        # 检查知识整合相关功能
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'learning_mechanisms' in agi_cap:
                learning_mech = agi_cap['learning_mechanisms']
                if isinstance(learning_mech, list) and len(learning_mech) > 0:
                    score += 0.1
                    assessment_detail['factors_considered'].append('learning_mechanisms_available')
        
        # 检查性能指标中的知识整合
        if 'performance_metrics' in model_data:
            perf_metrics = model_data['performance_metrics']
            if 'knowledge_integration' in perf_metrics:
                knowledge_integration = perf_metrics['knowledge_integration'].get('current', 0.5)
                score = (score + knowledge_integration) / 2
                assessment_detail['factors_considered'].append('knowledge_integration_metric')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _assess_autonomous_learning(self, model_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """评估自主学习能力"""
        score = 0.5
        
        assessment_detail = {
            'method': 'composite_assessment',
            'factors_considered': []
        }
        
        # 检查自主度水平
        if hasattr(model_data.get('model', None), '_autonomy_level'):
            autonomy_level = model_data['model']._autonomy_level
            score = (score + autonomy_level) / 2
            assessment_detail['factors_considered'].append('autonomy_level')
        
        # 检查学习机制
        if 'agi_capabilities' in model_data:
            agi_cap = model_data['agi_capabilities']
            if 'learning_mechanisms' in agi_cap:
                learning_mech = agi_cap['learning_mechanisms']
                if isinstance(learning_mech, list) and len(learning_mech) > 0:
                    score += 0.2
                    assessment_detail['factors_considered'].append('learning_mechanisms_available')
        
        # 检查性能指标中的适应速度
        if 'performance_metrics' in model_data:
            perf_metrics = model_data['performance_metrics']
            if 'adaptation_speed' in perf_metrics:
                adaptation_speed = perf_metrics['adaptation_speed'].get('current', 0.5)
                score = (score + adaptation_speed) / 2
                assessment_detail['factors_considered'].append('adaptation_speed_metric')
        
        score = max(0.0, min(1.0, score))
        
        return score, assessment_detail
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """计算总体AGI能力分数"""
        overall_score = 0.0
        total_weight = 0.0
        
        for dim_id, dim_score in dimension_scores.items():
            weight = self.dimensions[dim_id]['weight']
            overall_score += dim_score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        return overall_score
    
    def _generate_capability_profile(self, overall_score: float,
                                   dimension_scores: Dict[str, float],
                                   dimension_details: Dict[str, Any]) -> Dict[str, Any]:
        """生成能力画像"""
        capability_profile = {
            'overall_agi_score': overall_score,
            'agi_level': self._determine_agi_level(overall_score),
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'dimension_breakdown': {}
        }
        
        # 分析各维度表现
        for dim_id, dim_score in dimension_scores.items():
            dim_config = self.dimensions[dim_id]
            dim_detail = dimension_details[dim_id]
            
            capability_profile['dimension_breakdown'][dim_id] = {
                'name': dim_config['name'],
                'score': dim_score,
                'weight': dim_config['weight'],
                'description': dim_config['description'],
                'assessment_detail': dim_detail
            }
            
            # 识别优势和弱点
            if dim_score >= 0.8:
                capability_profile['strengths'].append(dim_config['name'])
            elif dim_score <= 0.3:
                capability_profile['weaknesses'].append(dim_config['name'])
        
        # 生成改进建议
        for dim_id, dim_score in dimension_scores.items():
            if dim_score < 0.6:
                dim_config = self.dimensions[dim_id]
                recommendation = f"提升{dim_config['name']}，当前分数{dim_score:.2f}低于阈值0.6"
                capability_profile['recommendations'].append(recommendation)
        
        return capability_profile
    
    def _determine_agi_level(self, overall_score: float) -> str:
        """根据总体分数确定AGI等级"""
        if overall_score >= 0.9:
            return "AGI+ (超越人类水平)"
        elif overall_score >= 0.8:
            return "AGI (人类水平通用智能)"
        elif overall_score >= 0.7:
            return "高级通用智能"
        elif overall_score >= 0.6:
            return "中级通用智能"
        elif overall_score >= 0.5:
            return "初级通用智能"
        elif overall_score >= 0.4:
            return "狭义人工智能"
        elif overall_score >= 0.3:
            return "基础人工智能"
        else:
            return "有限人工智能"
    
    def compare_models(self, model_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较多个模型的AGI能力"""
        if not model_assessments:
            return {'error': 'No assessments provided'}
        
        comparison_result = {
            'compared_models': [],
            'ranking': [],
            'comparison_matrix': {},
            'statistical_summary': {}
        }
        
        # 收集比较数据
        for assessment in model_assessments:
            model_info = {
                'model_id': assessment.get('model_id', 'unknown'),
                'model_class': assessment.get('model_class', 'unknown'),
                'overall_score': assessment.get('overall_score', 0.0),
                'dimension_scores': assessment.get('dimension_scores', {}),
                'timestamp': assessment.get('timestamp', '')
            }
            comparison_result['compared_models'].append(model_info)
        
        # 按总体分数排序
        sorted_models = sorted(
            comparison_result['compared_models'],
            key=lambda x: x['overall_score'],
            reverse=True
        )
        comparison_result['ranking'] = sorted_models
        
        # 生成比较矩阵
        comparison_matrix = {}
        for dim_id in self.dimensions.keys():
            dim_scores = {}
            for model_info in comparison_result['compared_models']:
                model_id = model_info['model_id']
                dim_scores[model_id] = model_info['dimension_scores'].get(dim_id, 0.0)
            comparison_matrix[dim_id] = dim_scores
        
        comparison_result['comparison_matrix'] = comparison_matrix
        
        # 统计摘要
        overall_scores = [m['overall_score'] for m in comparison_result['compared_models']]
        if overall_scores:
            comparison_result['statistical_summary'] = {
                'average_score': np.mean(overall_scores),
                'max_score': np.max(overall_scores),
                'min_score': np.min(overall_scores),
                'std_dev': np.std(overall_scores),
                'score_range': np.max(overall_scores) - np.min(overall_scores)
            }
        
        return comparison_result
    
    def export_assessment_report(self, assessment_result: Dict[str, Any], 
                               format: str = 'json') -> str:
        """导出评估报告"""
        if format == 'json':
            return json.dumps(assessment_result, indent=2, ensure_ascii=False)
        elif format == 'text':
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("AGI Capability Assessment Report")
            report_lines.append("=" * 60)
            report_lines.append(f"Model: {assessment_result.get('model_class', 'Unknown')}")
            report_lines.append(f"Model ID: {assessment_result.get('model_id', 'Unknown')}")
            report_lines.append(f"Assessment Time: {assessment_result.get('timestamp', 'Unknown')}")
            report_lines.append(f"Overall AGI Score: {assessment_result.get('overall_score', 0.0):.3f}")
            report_lines.append(f"AGI Level: {assessment_result.get('capability_profile', {}).get('agi_level', 'Unknown')}")
            report_lines.append("\nDimension Scores:")
            
            for dim_id, dim_score in assessment_result.get('dimension_scores', {}).items():
                dim_name = self.dimensions.get(dim_id, {}).get('name', dim_id)
                report_lines.append(f"  - {dim_name}: {dim_score:.3f}")
            
            report_lines.append("\nStrengths:")
            for strength in assessment_result.get('capability_profile', {}).get('strengths', []):
                report_lines.append(f"  - {strength}")
            
            report_lines.append("\nWeaknesses:")
            for weakness in assessment_result.get('capability_profile', {}).get('weaknesses', []):
                report_lines.append(f"  - {weakness}")
            
            report_lines.append("\nRecommendations:")
            for rec in assessment_result.get('capability_profile', {}).get('recommendations', []):
                report_lines.append(f"  - {rec}")
            
            report_lines.append("\n" + "=" * 60)
            return "\n".join(report_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")