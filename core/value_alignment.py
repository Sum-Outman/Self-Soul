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
from datetime import datetime
from collections import defaultdict
import json
from core.error_handling import error_handler

class ValueSystem:
    """价值系统 - 定义和管理核心价值"""
    
    def __init__(self):
        self.core_values = self._initialize_core_values()
        self.value_weights = self._initialize_value_weights()
        self.value_violations = defaultdict(int)
        self.value_fulfillments = defaultdict(int)
    
    def _initialize_core_values(self):
        """初始化核心价值"""
        return {
            'safety': {
                'description': '确保系统安全和无害',
                'priority': 0.9,
                'metrics': ['risk_avoidance', 'error_prevention']
            },
            'helpfulness': {
                'description': '提供有用和有益的帮助',
                'priority': 0.8,
                'metrics': ['utility_score', 'user_satisfaction']
            },
            'honesty': {
                'description': '保持诚实和透明',
                'priority': 0.85,
                'metrics': ['truthfulness', 'transparency']
            },
            'fairness': {
                'description': '确保公平和无偏见',
                'priority': 0.75,
                'metrics': ['bias_detection', 'equitable_treatment']
            },
            'autonomy_respect': {
                'description': '尊重用户自主权',
                'priority': 0.7,
                'metrics': ['user_choice', 'consent_respect']
            },
            'privacy': {
                'description': '保护用户隐私',
                'priority': 0.8,
                'metrics': ['data_protection', 'consent_management']
            }
        }
    
    def _initialize_value_weights(self):
        """初始化价值权重"""
        return {value: self.core_values[value]['priority'] for value in self.core_values}
    
    def evaluate_action(self, action, context):
        """评估行动的价值对齐性"""
        value_scores = {}
        violations = []
        fulfillments = []
        
        for value_name, value_info in self.core_values.items():
            score, is_violation, is_fulfillment = self._assess_value_alignment(
                value_name, action, context
            )
            
            value_scores[value_name] = score * self.value_weights[value_name]
            
            if is_violation:
                violations.append(value_name)
                self.value_violations[value_name] += 1
            
            if is_fulfillment:
                fulfillments.append(value_name)
                self.value_fulfillments[value_name] += 1
        
        total_score = sum(value_scores.values()) / sum(self.value_weights.values())
        
        return {
            'total_alignment_score': total_score,
            'value_breakdown': value_scores,
            'violations': violations,
            'fulfillments': fulfillments,
            'recommendation': self._generate_recommendation(total_score, violations)
        }
    
    def _assess_value_alignment(self, value_name, action, context):
        """评估特定价值的对齐性"""
        assessment_methods = {
            'safety': self._assess_safety,
            'helpfulness': self._assess_helpfulness,
            'honesty': self._assess_honesty,
            'fairness': self._assess_fairness,
            'autonomy_respect': self._assess_autonomy_respect,
            'privacy': self._assess_privacy
        }
        
        method = assessment_methods.get(value_name)
        if method:
            return method(action, context)
        else:
            return 0.5, False, False  # 默认分数，无违反或满足
    
    def _assess_safety(self, action, context):
        """评估安全性"""
        action_str = str(action).lower()
        context_str = str(context).lower()
        
        # 检测潜在风险
        risk_indicators = ['danger', 'harm', 'risk', 'unsafe', 'malicious']
        has_risk = any(indicator in action_str or indicator in context_str 
                      for indicator in risk_indicators)
        
        if has_risk:
            return 0.2, True, False  # 低分，有违反
        
        # 检测安全措施
        safety_indicators = ['safe', 'secure', 'protect', 'prevent']
        has_safety = any(indicator in action_str for indicator in safety_indicators)
        
        return 0.8 if has_safety else 0.5, False, has_safety
    
    def _assess_helpfulness(self, action, context):
        """评估帮助性"""
        action_str = str(action).lower()
        
        helpful_indicators = ['help', 'assist', 'support', 'benefit', 'useful']
        is_helpful = any(indicator in action_str for indicator in helpful_indicators)
        
        return 0.9 if is_helpful else 0.5, False, is_helpful
    
    def _assess_honesty(self, action, context):
        """评估诚实性"""
        action_str = str(action).lower()
        
        dishonesty_indicators = ['deceive', 'lie', 'mislead', 'false', 'fake']
        is_dishonest = any(indicator in action_str for indicator in dishonesty_indicators)
        
        if is_dishonest:
            return 0.1, True, False  # 低分，有违反
        
        transparency_indicators = ['truth', 'honest', 'transparent', 'clear']
        is_transparent = any(indicator in action_str for indicator in transparency_indicators)
        
        return 0.9 if is_transparent else 0.6, False, is_transparent
    
    def _assess_fairness(self, action, context):
        """评估公平性"""
        action_str = str(action).lower()
        
        bias_indicators = ['bias', 'discriminate', 'unfair', 'prejudice']
        has_bias = any(indicator in action_str for indicator in bias_indicators)
        
        if has_bias:
            return 0.2, True, False  # 低分，有违反
        
        fairness_indicators = ['fair', 'equal', 'impartial', 'just']
        is_fair = any(indicator in action_str for indicator in fairness_indicators)
        
        return 0.8 if is_fair else 0.5, False, is_fair
    
    def _assess_autonomy_respect(self, action, context):
        """评估自主权尊重"""
        action_str = str(action).lower()
        context_str = str(context).lower()
        
        coercion_indicators = ['force', 'coerce', 'mandate', 'require']
        is_coercive = any(indicator in action_str for indicator in coercion_indicators)
        
        if is_coercive and 'consent' not in context_str:
            return 0.3, True, False  # 低分，有违反
        
        choice_indicators = ['choice', 'option', 'decide', 'select']
        has_choice = any(indicator in action_str for indicator in choice_indicators)
        
        return 0.8 if has_choice else 0.5, False, has_choice
    
    def _assess_privacy(self, action, context):
        """评估隐私保护"""
        action_str = str(action).lower()
        
        privacy_violation_indicators = ['share', 'collect', 'store', 'data', 'personal']
        is_privacy_risk = any(indicator in action_str 
                            for indicator in privacy_violation_indicators)
        
        if is_privacy_risk and 'consent' not in action_str:
            return 0.2, True, False  # 低分，有违反
        
        privacy_protection_indicators = ['private', 'secure', 'encrypt', 'protect']
        protects_privacy = any(indicator in action_str 
                             for indicator in privacy_protection_indicators)
        
        return 0.9 if protects_privacy else 0.5, False, protects_privacy
    
    def _generate_recommendation(self, alignment_score, violations):
        """生成推荐"""
        if alignment_score < 0.4:
            return "强烈不建议执行：存在严重价值冲突"
        elif alignment_score < 0.6:
            return "谨慎执行：存在价值风险，需要修改"
        elif alignment_score < 0.8:
            return "可以执行：基本符合价值要求"
        else:
            return "推荐执行：高度符合价值标准"
    
    def get_value_statistics(self):
        """获取价值统计"""
        return {
            'total_violations': sum(self.value_violations.values()),
            'total_fulfillments': sum(self.value_fulfillments.values()),
            'violation_breakdown': dict(self.value_violations),
            'fulfillment_breakdown': dict(self.value_fulfillments),
            'overall_alignment_score': self._calculate_overall_alignment()
        }
    
    def _calculate_overall_alignment(self):
        """计算总体对齐分数"""
        total_actions = sum(self.value_violations.values()) + sum(self.value_fulfillments.values())
        if total_actions == 0:
            return 0.5  # 默认分数
        
        violation_score = sum(self.value_violations.values()) / total_actions
        fulfillment_score = sum(self.value_fulfillments.values()) / total_actions
        
        return max(0.0, min(1.0, 0.5 + (fulfillment_score - violation_score) * 0.5))

class EthicalReasoner:
    """伦理推理器 - 进行伦理决策推理"""
    
    def __init__(self):
        self.ethical_frameworks = self._load_ethical_frameworks()
        self.case_studies = deque(maxlen=1000)
        self.ethical_dilemmas_resolved = 0
    
    def _load_ethical_frameworks(self):
        """加载伦理框架"""
        return {
            'utilitarianism': {
                'description': '最大化整体幸福',
                'key_principle': '追求最大多数人的最大幸福',
                'evaluation_method': self._evaluate_utilitarian
            },
            'deontology': {
                'description': '遵循道德义务和规则',
                'key_principle': '行动必须符合道德规则',
                'evaluation_method': self._evaluate_deontological
            },
            'virtue_ethics': {
                'description': '培养道德美德',
                'key_principle': '行动应体现道德美德',
                'evaluation_method': self._evaluate_virtue
            },
            'rights_based': {
                'description': '尊重基本权利',
                'key_principle': '行动必须尊重基本权利',
                'evaluation_method': self._evaluate_rights_based
            }
        }
    
    def resolve_ethical_dilemma(self, dilemma_description, context=None):
        """解决伦理困境"""
        try:
            framework_evaluations = {}
            
            # 使用所有框架进行评估
            for framework_name, framework_info in self.ethical_frameworks.items():
                evaluation = framework_info['evaluation_method'](dilemma_description, context)
                framework_evaluations[framework_name] = evaluation
            
            # 综合评估
            consensus = self._reach_consensus(framework_evaluations)
            
            # 记录案例
            case_id = self._record_case(dilemma_description, framework_evaluations, consensus)
            self.ethical_dilemmas_resolved += 1
            
            return {
                'case_id': case_id,
                'framework_evaluations': framework_evaluations,
                'consensus_recommendation': consensus,
                'confidence': self._calculate_confidence(framework_evaluations)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "EthicalReasoner", "伦理困境解决失败")
            return {"error": str(e)}
    
    def _evaluate_utilitarian(self, dilemma, context):
        """功利主义评估"""
        # 简化的功利主义计算
        positive_impact = self._estimate_impact(dilemma, 'positive')
        negative_impact = self._estimate_impact(dilemma, 'negative')
        
        net_utility = positive_impact - negative_impact
        recommendation = net_utility > 0
        
        return {
            'net_utility': net_utility,
            'recommendation': recommendation,
            'reasoning': f"净效用: {net_utility:.2f} ({'推荐' if recommendation else '不推荐'})"
        }
    
    def _evaluate_deontological(self, dilemma, context):
        """义务论评估"""
        # 检查是否违反道德规则
        rule_violations = self._detect_rule_violations(dilemma)
        
        recommendation = len(rule_violations) == 0
        return {
            'rule_violations': rule_violations,
            'recommendation': recommendation,
            'reasoning': f"{len(rule_violations)} 条规则违反 ({'推荐' if recommendation else '不推荐'})"
        }
    
    def _evaluate_virtue(self, dilemma, context):
        """美德伦理评估"""
        # 检查行动体现的美德
        virtues_demonstrated = self._identify_virtues(dilemma)
        vices_demonstrated = self._identify_vices(dilemma)
        
        recommendation = len(virtues_demonstrated) > len(vices_demonstrated)
        return {
            'virtues': virtues_demonstrated,
            'vices': vices_demonstrated,
            'recommendation': recommendation,
            'reasoning': f"美德: {len(virtues_demonstrated)}, 恶习: {len(vices_demonstrated)} ({'推荐' if recommendation else '不推荐'})"
        }
    
    def _evaluate_rights_based(self, dilemma, context):
        """权利基础评估"""
        # 检查权利侵犯
        rights_violations = self._detect_rights_violations(dilemma)
        rights_protections = self._detect_rights_protections(dilemma)
        
        recommendation = len(rights_violations) == 0
        return {
            'rights_violations': rights_violations,
            'rights_protections': rights_protections,
            'recommendation': recommendation,
            'reasoning': f"权利侵犯: {len(rights_violations)}, 权利保护: {len(rights_protections)} ({'推荐' if recommendation else '不推荐'})"
        }
    
    def _estimate_impact(self, dilemma, impact_type):
        """估计影响"""
        # 简化的影响估计
        dilemma_lower = dilemma.lower()
        impact_keywords = {
            'positive': ['help', 'benefit', 'improve', 'save', 'protect'],
            'negative': ['harm', 'hurt', 'damage', 'risk', 'danger']
        }
        
        keywords = impact_keywords.get(impact_type, [])
        impact_score = sum(1 for keyword in keywords if keyword in dilemma_lower)
        
        return min(impact_score * 0.2, 1.0)  # 标准化到0-1范围
    
    def _detect_rule_violations(self, dilemma):
        """检测规则违反"""
        # 简化的规则违反检测
        rules = [
            ('do_not_harm', '造成伤害'),
            ('do_not_lie', '说谎欺骗'),
            ('do_not_steal', '窃取资源'),
            ('respect_autonomy', '侵犯自主权')
        ]
        
        violations = []
        dilemma_lower = dilemma.lower()
        
        for rule_id, rule_desc in rules:
            if rule_id == 'do_not_harm' and any(word in dilemma_lower 
                                              for word in ['harm', 'hurt', 'damage']):
                violations.append(rule_desc)
            elif rule_id == 'do_not_lie' and any(word in dilemma_lower 
                                               for word in ['lie', 'deceive', 'false']):
                violations.append(rule_desc)
        
        return violations
    
    def _identify_virtues(self, dilemma):
        """识别美德"""
        virtues = {
            'honesty': ['truth', 'honest', 'transparent'],
            'compassion': ['help', 'care', 'support'],
            'courage': ['brave', 'courage', 'stand'],
            'wisdom': ['wise', 'smart', 'knowledge']
        }
        
        demonstrated = []
        dilemma_lower = dilemma.lower()
        
        for virtue, keywords in virtues.items():
            if any(keyword in dilemma_lower for keyword in keywords):
                demonstrated.append(virtue)
        
        return demonstrated
    
    def _identify_vices(self, dilemma):
        """识别恶习"""
        vices = {
            'deceit': ['lie', 'deceive', 'false'],
            'harm': ['harm', 'hurt', 'damage'],
            'greed': ['greed', 'selfish', 'take'],
            'cowardice': ['coward', 'fear', 'avoid']
        }
        
        demonstrated = []
        dilemma_lower = dilemma.lower()
        
        for vice, keywords in vices.items():
            if any(keyword in dilemma_lower for keyword in keywords):
                demonstrated.append(vice)
        
        return demonstrated
    
    def _detect_rights_violations(self, dilemma):
        """检测权利侵犯"""
        rights = {
            'privacy': ['private', 'data', 'personal'],
            'autonomy': ['force', 'coerce', 'require'],
            'safety': ['danger', 'risk', 'unsafe'],
            'fairness': ['unfair', 'bias', 'discriminate']
        }
        
        violations = []
        dilemma_lower = dilemma.lower()
        
        for right, keywords in rights.items():
            if any(keyword in dilemma_lower for keyword in keywords):
                # 检查是否有保护措施
                if not any(protect in dilemma_lower 
                          for protect in ['protect', 'respect', 'ensure']):
                    violations.append(right)
        
        return violations
    
    def _detect_rights_protections(self, dilemma):
        """检测权利保护"""
        protections = {
            'privacy': ['protect privacy', 'secure data'],
            'autonomy': ['respect choice', 'allow decision'],
            'safety': ['ensure safety', 'prevent harm'],
            'fairness': ['ensure fairness', 'avoid bias']
        }
        
        detected = []
        dilemma_lower = dilemma.lower()
        
        for right, phrases in protections.items():
            if any(phrase in dilemma_lower for phrase in phrases):
                detected.append(right)
        
        return detected
    
    def _reach_consensus(self, framework_evaluations):
        """达成共识"""
        recommendations = [evaluation['recommendation'] 
                          for evaluation in framework_evaluations.values()]
        
        # 简单多数决
        if sum(recommendations) >= len(recommendations) / 2:
            return {
                'recommendation': True,
                'consensus_level': 'majority',
                'supporting_frameworks': [name for name, eval in framework_evaluations.items() 
                                        if eval['recommendation']]
            }
        else:
            return {
                'recommendation': False,
                'consensus_level': 'majority',
                'opposing_frameworks': [name for name, eval in framework_evaluations.items() 
                                      if not eval['recommendation']]
            }
    
    def _calculate_confidence(self, framework_evaluations):
        """计算置信度"""
        agreements = sum(1 for evaluation in framework_evaluations.values() 
                        if evaluation['recommendation'])
        total = len(framework_evaluations)
        
        return agreements / total
    
    def _record_case(self, dilemma, evaluations, consensus):
        """记录案例"""
        case_id = f"ethical_case_{len(self.case_studies)}_{int(time.time())}"
        
        case_record = {
            'case_id': case_id,
            'dilemma': dilemma,
            'timestamp': time.time(),
            'evaluations': evaluations,
            'consensus': consensus,
            'lessons_learned': self._extract_lessons(dilemma, evaluations, consensus)
        }
        
        self.case_studies.append(case_record)
        return case_id
    
    def _extract_lessons(self, dilemma, evaluations, consensus):
        """提取经验教训"""
        lessons = []
        
        # 从评估中提取见解
        for framework_name, evaluation in evaluations.items():
            if 'reasoning' in evaluation:
                lessons.append(f"{framework_name}: {evaluation['reasoning']}")
        
        # 添加共识见解
        lessons.append(f"共识: {consensus['consensus_level']} 支持{'推荐' if consensus['recommendation'] else '不推荐'}")
        
        return lessons
    
    def get_ethical_report(self):
        """获取伦理报告"""
        return {
            'frameworks_available': list(self.ethical_frameworks.keys()),
            'cases_resolved': self.ethical_dilemmas_resolved,
            'recent_cases': [{
                'case_id': case['case_id'],
                'timestamp': datetime.fromtimestamp(case['timestamp']).isoformat(),
                'consensus': case['consensus']['recommendation']
            } for case in list(self.case_studies)[-5:]],
            'framework_usage': self._calculate_framework_usage()
        }
    
    def _calculate_framework_usage(self):
        """计算框架使用情况"""
        usage = defaultdict(int)
        for case in self.case_studies:
            for framework_name in case['evaluations']:
                usage[framework_name] += 1
        
        return dict(usage)

class ValueAlignment:
    """价值对齐系统 - 确保AI系统与人类价值观一致"""
    
    def __init__(self):
        self.value_system = ValueSystem()
        self.ethical_reasoner = EthicalReasoner()
        
        error_handler.log_info("价值对齐系统初始化完成", "ValueAlignment")
    
    def align_action(self, proposed_action, context=None):
        """对齐行动与价值观"""
        try:
            # 价值评估
            value_assessment = self.value_system.evaluate_action(proposed_action, context or {})
            
            # 如果是伦理困境，进行伦理推理
            requires_ethical_review = self._requires_ethical_review(proposed_action, context)
            ethical_assessment = None
            
            if requires_ethical_review:
                ethical_assessment = self.ethical_reasoner.resolve_ethical_dilemma(
                    str(proposed_action), context
                )
            
            # 综合评估
            overall_assessment = self._integrate_assessments(value_assessment, ethical_assessment)
            
            return {
                'value_assessment': value_assessment,
                'ethical_assessment': ethical_assessment,
                'overall_assessment': overall_assessment,
                'alignment_verdict': self._make_verdict(overall_assessment)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ValueAlignment", "价值对齐失败")
            return {"error": str(e)}
    
    def _requires_ethical_review(self, action, context):
        """判断是否需要伦理审查"""
        action_str = str(action).lower()
        context_str = str(context or {}).lower()
        
        # 触发伦理审查的关键词
        ethical_triggers = [
            'ethical', 'moral', 'dilemma', 'right', 'wrong',
            'should', 'ought', 'permissible', 'obligation'
        ]
        
        return any(trigger in action_str or trigger in context_str 
                  for trigger in ethical_triggers)
    
    def _integrate_assessments(self, value_assessment, ethical_assessment):
        """整合评估结果"""
        if ethical_assessment is None:
            # 只有价值评估
            return {
                'alignment_score': value_assessment['total_alignment_score'],
                'confidence': 0.8,
                'primary_concerns': value_assessment['violations'],
                'positive_aspects': value_assessment['fulfillments']
            }
        
        # 整合价值和伦理评估
        value_score = value_assessment['total_alignment_score']
        ethical_confidence = ethical_assessment['confidence']
        ethical_recommendation = 1.0 if ethical_assessment['consensus_recommendation']['recommendation'] else 0.0
        
        integrated_score = (value_score * 0.6 + ethical_recommendation * 0.4) * ethical_confidence
        
        concerns = value_assessment['violations'].copy()
        if not ethical_assessment['consensus_recommendation']['recommendation']:
            concerns.append('ethical_concerns')
        
        positives = value_assessment['fulfillments'].copy()
        if ethical_assessment['consensus_recommendation']['recommendation']:
            positives.append('ethically_sound')
        
        return {
            'alignment_score': integrated_score,
            'confidence': min(ethical_confidence, 0.9),  # 保守置信度
            'primary_concerns': concerns,
            'positive_aspects': positives,
            'requires_human_review': integrated_score < 0.6
        }
    
    def _make_verdict(self, assessment):
        """做出最终裁决"""
        score = assessment['alignment_score']
        
        if score >= 0.8:
            return {
                'verdict': 'APPROVED',
                'confidence': 'HIGH',
                'reasoning': '行动高度符合价值和伦理标准'
            }
        elif score >= 0.6:
            return {
                'verdict': 'CONDITIONALLY_APPROVED',
                'confidence': 'MEDIUM',
                'reasoning': '行动基本符合要求，建议监控执行'
            }
        elif score >= 0.4:
            return {
                'verdict': 'REQUIRES_MODIFICATION',
                'confidence': 'LOW',
                'reasoning': '行动需要修改以符合价值标准'
            }
        else:
            return {
                'verdict': 'REJECTED',
                'confidence': 'HIGH',
                'reasoning': '行动存在严重价值或伦理问题'
            }
    
    def get_alignment_report(self):
        """获取对齐报告"""
        return {
            'value_system': self.value_system.get_value_statistics(),
            'ethical_reasoning': self.ethical_reasoner.get_ethical_report(),
            'overall_alignment_health': self._calculate_alignment_health(),
            'report_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_alignment_health(self):
        """计算对齐健康度"""
        value_stats = self.value_system.get_value_statistics()
        ethical_report = self.ethical_reasoner.get_ethical_report()
        
        violation_ratio = value_stats['total_violations'] / max(1, value_stats['total_fulfillments'])
        case_success_ratio = ethical_report['cases_resolved'] / max(1, ethical_report['cases_resolved'])
        
        health_score = max(0.0, min(1.0, 1.0 - violation_ratio * 0.5 + case_success_ratio * 0.3))
        
        if health_score > 0.8:
            return {'score': health_score, 'status': 'EXCELLENT'}
        elif health_score > 0.6:
            return {'score': health_score, 'status': 'GOOD'}
        elif health_score > 0.4:
            return {'score': health_score, 'status': 'FAIR'}
        else:
            return {'score': health_score, 'status': 'POOR'}
    
    def export_alignment_data(self, file_path):
        """导出对齐数据"""
        try:
            export_data = {
                'value_statistics': self.value_system.get_value_statistics(),
                'ethical_cases': list(self.ethical_reasoner.case_studies)[-20:],  # 最近20个案例
                'alignment_health': self._calculate_alignment_health(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "export_path": file_path}
            
        except Exception as e:
            error_handler.handle_error(e, "ValueAlignment", "对齐数据导出失败")
            return {"error": str(e)}