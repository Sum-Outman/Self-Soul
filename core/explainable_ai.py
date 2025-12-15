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

class DecisionTracer:
    """决策追踪器 - 记录和解释决策过程"""
    
    def __init__(self):
        self.decision_log = []
        self.decision_counter = 0
        self.trace_depth = 10  # 最大追踪深度
    
    def start_trace(self, decision_context):
        """开始决策追踪"""
        trace_id = f"trace_{self.decision_counter}_{int(time.time())}"
        self.decision_counter += 1
        
        trace_entry = {
            'trace_id': trace_id,
            'start_time': time.time(),
            'context': decision_context,
            'steps': [],
            'final_decision': None,
            'confidence': 0.0,
            'explanation': None
        }
        
        self.decision_log.append(trace_entry)
        return trace_id
    
    def add_step(self, trace_id, step_description, reasoning, confidence=0.5):
        """添加决策步骤"""
        for trace in self.decision_log:
            if trace['trace_id'] == trace_id:
                step = {
                    'step_number': len(trace['steps']) + 1,
                    'timestamp': time.time(),
                    'description': step_description,
                    'reasoning': reasoning,
                    'confidence': confidence
                }
                trace['steps'].append(step)
                return True
        return False
    
    def finalize_decision(self, trace_id, decision, confidence, explanation):
        """最终化决策"""
        for trace in self.decision_log:
            if trace['trace_id'] == trace_id:
                trace['final_decision'] = decision
                trace['confidence'] = confidence
                trace['explanation'] = explanation
                trace['end_time'] = time.time()
                trace['duration'] = trace['end_time'] - trace['start_time']
                return True
        return False
    
    def get_explanation(self, trace_id):
        """获取决策解释"""
        for trace in self.decision_log:
            if trace['trace_id'] == trace_id:
                return self._generate_natural_language_explanation(trace)
        return {"error": "Trace not found"}
    
    def _generate_natural_language_explanation(self, trace):
        """生成自然语言解释"""
        explanation_parts = []
        
        # 添加上下文
        explanation_parts.append(f"决策上下文: {trace['context'].get('description', '无描述')}")
        
        # 添加步骤
        for step in trace['steps']:
            explanation_parts.append(
                f"步骤 {step['step_number']}: {step['description']} "
                f"(推理: {step['reasoning']}, 置信度: {step['confidence']:.2f})"
            )
        
        # 添加最终决策
        explanation_parts.append(
            f"最终决策: {trace['final_decision']} "
            f"(总体置信度: {trace['confidence']:.2f})"
        )
        
        # 添加解释
        if trace['explanation']:
            explanation_parts.append(f"详细解释: {trace['explanation']}")
        
        # 添加性能信息
        explanation_parts.append(f"决策耗时: {trace['duration']:.2f}秒")
        
        return "\n".join(explanation_parts)
    
    def get_recent_decisions(self, limit=5):
        """获取最近的决策"""
        recent = self.decision_log[-limit:] if self.decision_log else []
        return [
            {
                'trace_id': trace['trace_id'],
                'context': trace['context'],
                'decision': trace['final_decision'],
                'confidence': trace['confidence'],
                'timestamp': datetime.fromtimestamp(trace['start_time']).isoformat()
            }
            for trace in recent
        ]

class ConfidenceCalibrator:
    """置信度校准器 - 确保置信度准确反映真实概率"""
    
    def __init__(self):
        self.calibration_data = []
        self.calibration_history = []
        self.reliability_score = 0.8  # 初始可靠性评分
    
    def calibrate(self, predicted_confidence, actual_outcome):
        """校准置信度"""
        calibration_point = {
            'predicted': predicted_confidence,
            'actual': 1.0 if actual_outcome else 0.0,
            'timestamp': time.time()
        }
        
        self.calibration_data.append(calibration_point)
        
        # 更新可靠性评分
        self._update_reliability()
        
        return self._get_calibrated_confidence(predicted_confidence)
    
    def _update_reliability(self):
        """更新可靠性评分"""
        if len(self.calibration_data) < 10:
            return
        
        # 计算校准误差
        errors = []
        for point in self.calibration_data[-100:]:  # 使用最近100个点
            error = abs(point['predicted'] - point['actual'])
            errors.append(error)
        
        average_error = np.mean(errors) if errors else 0.5
        self.reliability_score = max(0.1, min(1.0, 1.0 - average_error))
        
        self.calibration_history.append({
            'timestamp': time.time(),
            'reliability': self.reliability_score,
            'sample_size': len(self.calibration_data)
        })
    
    def _get_calibrated_confidence(self, raw_confidence):
        """获取校准后的置信度"""
        # 简单的线性校准
        calibrated = raw_confidence * self.reliability_score
        
        # 确保在合理范围内
        return max(0.0, min(1.0, calibrated))
    
    def get_calibration_report(self):
        """获取校准报告"""
        return {
            'reliability_score': self.reliability_score,
            'calibration_points': len(self.calibration_data),
            'average_error': self._calculate_average_error(),
            'calibration_status': self._get_calibration_status()
        }
    
    def _calculate_average_error(self):
        """计算平均误差"""
        if not self.calibration_data:
            return 0.5
        
        errors = [abs(p['predicted'] - p['actual']) for p in self.calibration_data[-50:]]
        return np.mean(errors) if errors else 0.5
    
    def _get_calibration_status(self):
        """获取校准状态"""
        if self.reliability_score > 0.9:
            return 'excellent'
        elif self.reliability_score > 0.7:
            return 'good'
        elif self.reliability_score > 0.5:
            return 'fair'
        else:
            return 'poor'

class FeatureImportanceAnalyzer:
    """特征重要性分析器 - 分析决策中的关键因素"""
    
    def __init__(self):
        self.feature_importance = defaultdict(float)
        self.feature_usage_count = defaultdict(int)
    
    def analyze_decision(self, decision_data, features, outcome):
        """分析决策中的特征重要性"""
        # 简化的特征重要性分析
        for feature in features:
            feature_value = decision_data.get(feature)
            if feature_value is not None:
                # 基于特征值和结果更新重要性
                importance_update = self._calculate_importance_update(feature_value, outcome)
                self.feature_importance[feature] = (
                    self.feature_importance[feature] * 0.9 + importance_update * 0.1
                )
                self.feature_usage_count[feature] += 1
        
        return self.get_feature_importance()
    
    def _calculate_importance_update(self, feature_value, outcome):
        """计算特征重要性更新"""
        # 简化的实现 - 实际中应使用更复杂的方法
        if isinstance(feature_value, (int, float)):
            # 数值特征
            return abs(feature_value) * (1.0 if outcome else 0.5)
        else:
            # 分类特征
            return 0.7 if outcome else 0.3
    
    def get_feature_importance(self, top_n=10):
        """获取特征重要性排名"""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {
                'feature': feature,
                'importance': importance,
                'usage_count': self.feature_usage_count[feature]
            }
            for feature, importance in sorted_features
        ]
    
    def reset_analysis(self):
        """重置分析数据"""
        self.feature_importance.clear()
        self.feature_usage_count.clear()

class CounterfactualGenerator:
    """反事实生成器 - 生成替代决策场景"""
    
    def __init__(self):
        self.counterfactual_scenarios = []
    
    def generate_counterfactuals(self, decision_data, actual_outcome, num_scenarios=3):
        """生成反事实场景"""
        scenarios = []
        
        for i in range(num_scenarios):
            scenario = self._create_counterfactual_scenario(decision_data, actual_outcome)
            scenarios.append(scenario)
            self.counterfactual_scenarios.append(scenario)
        
        return scenarios
    
    def _create_counterfactual_scenario(self, decision_data, actual_outcome):
        """创建反事实场景"""
        # 简化的反事实生成
        scenario_id = f"counterfactual_{len(self.counterfactual_scenarios)}_{int(time.time())}"
        
        # 修改一些决策参数
        modified_data = decision_data.copy()
        changes = []
        
        # 随机选择一些特征进行修改
        features = list(decision_data.keys())
        if features:
            num_changes = min(3, len(features))
            features_to_change = np.random.choice(features, num_changes, replace=False)
            
            for feature in features_to_change:
                original_value = decision_data[feature]
                new_value = self._modify_value(original_value)
                modified_data[feature] = new_value
                changes.append({
                    'feature': feature,
                    'original': original_value,
                    'modified': new_value
                })
        
        # 预测可能的结果
        predicted_outcome = self._predict_outcome(modified_data, actual_outcome)
        
        return {
            'scenario_id': scenario_id,
            'original_data': decision_data,
            'modified_data': modified_data,
            'changes': changes,
            'predicted_outcome': predicted_outcome,
            'actual_outcome': actual_outcome,
            'timestamp': time.time()
        }
    
    def _modify_value(self, value):
        """修改值"""
        if isinstance(value, (int, float)):
            # 数值：增加或减少10-20%
            change_factor = 1.0 + np.random.uniform(-0.2, 0.2)
            return value * change_factor
        elif isinstance(value, bool):
            # 布尔值：取反
            return not value
        else:
            # 其他类型：随机选择或修改
            return value  # 暂不修改
    
    def _predict_outcome(self, modified_data, actual_outcome):
        """预测结果"""
        # 简化的预测：基于修改程度
        change_magnitude = 0.0
        for key in modified_data:
            if isinstance(modified_data[key], (int, float)) and key in actual_outcome:
                change_magnitude += abs(modified_data[key] - actual_outcome.get(key, 0))
        
        # 变化越大，结果差异可能越大
        outcome_difference = min(1.0, change_magnitude / 10.0)
        return not actual_outcome if outcome_difference > 0.5 else actual_outcome
    
    def get_scenario_analysis(self, scenario_id):
        """获取场景分析"""
        for scenario in self.counterfactual_scenarios:
            if scenario['scenario_id'] == scenario_id:
                return self._analyze_scenario(scenario)
        return {"error": "Scenario not found"}
    
    def _analyze_scenario(self, scenario):
        """分析场景"""
        analysis = {
            'scenario_id': scenario['scenario_id'],
            'key_changes': scenario['changes'],
            'outcome_difference': scenario['predicted_outcome'] != scenario['actual_outcome'],
            'sensitivity_analysis': self._perform_sensitivity_analysis(scenario),
            'learning_insights': self._extract_learning_insights(scenario)
        }
        return analysis
    
    def _perform_sensitivity_analysis(self, scenario):
        """执行敏感性分析"""
        # 简化的敏感性分析
        sensitivity = {}
        for change in scenario['changes']:
            feature = change['feature']
            sensitivity[feature] = {
                'change_magnitude': abs(change['modified'] - change['original']),
                'impact_on_outcome': 0.7  # 简化估计
            }
        return sensitivity
    
    def _extract_learning_insights(self, scenario):
        """提取学习见解"""
        insights = []
        
        if scenario['predicted_outcome'] != scenario['actual_outcome']:
            insights.append("小的参数变化可能导致完全不同的结果")
            insights.append("系统对某些特征非常敏感")
        else:
            insights.append("系统对此类变化具有鲁棒性")
            insights.append("决策相对稳定")
        
        return insights

class ExplainableAI:
    """可解释AI系统 - 提供透明的决策解释"""
    
    def __init__(self):
        self.decision_tracer = DecisionTracer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.counterfactual_generator = CounterfactualGenerator()
        
        error_handler.log_info("可解释AI系统初始化完成", "ExplainableAI")
    
    def explain_decision(self, decision_process, decision_data, outcome):
        """解释决策"""
        try:
            # 开始追踪
            trace_id = self.decision_tracer.start_trace({
                'description': f"决策解释请求",
                'decision_data': decision_data,
                'process': decision_process
            })
            
            # 分析特征重要性
            feature_importance = self.feature_analyzer.analyze_decision(
                decision_data, list(decision_data.keys()), outcome
            )
            
            # 生成反事实
            counterfactuals = self.counterfactual_generator.generate_counterfactuals(
                decision_data, outcome, 2
            )
            
            # 校准置信度
            calibrated_confidence = self.confidence_calibrator.calibrate(
                decision_process.get('confidence', 0.5), outcome
            )
            
            # 最终化追踪
            explanation = self._generate_comprehensive_explanation(
                decision_process, feature_importance, counterfactuals, calibrated_confidence
            )
            
            self.decision_tracer.finalize_decision(
                trace_id, decision_process.get('decision'), calibrated_confidence, explanation
            )
            
            return {
                'trace_id': trace_id,
                'explanation': explanation,
                'feature_importance': feature_importance,
                'calibrated_confidence': calibrated_confidence,
                'counterfactuals': [cf['scenario_id'] for cf in counterfactuals]
            }
            
        except Exception as e:
            error_handler.handle_error(e, "ExplainableAI", "决策解释失败")
            return {"error": str(e)}
    
    def _generate_comprehensive_explanation(self, decision_process, feature_importance, counterfactuals, confidence):
        """生成全面解释"""
        explanation_parts = []
        
        # 添加决策过程
        explanation_parts.append("## 决策过程分析")
        explanation_parts.append(f"决策类型: {decision_process.get('type', '未知')}")
        explanation_parts.append(f"使用的方法: {decision_process.get('method', '未指定')}")
        
        # 添加特征重要性
        explanation_parts.append("\n## 关键影响因素")
        for i, feature in enumerate(feature_importance[:5], 1):
            explanation_parts.append(
                f"{i}. {feature['feature']} (重要性: {feature['importance']:.3f}, "
                f"使用次数: {feature['usage_count']})"
            )
        
        # 添加置信度信息
        explanation_parts.append(f"\n## 置信度评估")
        explanation_parts.append(f"校准后置信度: {confidence:.3f}")
        calibration_report = self.confidence_calibrator.get_calibration_report()
        explanation_parts.append(f"系统可靠性: {calibration_report['reliability_score']:.3f}")
        
        # 添加反事实见解
        explanation_parts.append("\n## 替代场景分析")
        if counterfactuals:
            explanation_parts.append("生成了替代决策场景进行分析")
            explanation_parts.append("关键发现: 系统对某些参数变化敏感")
        else:
            explanation_parts.append("无需生成替代场景")
        
        # 添加总体评估
        explanation_parts.append("\n## 总体评估")
        if confidence > 0.8:
            explanation_parts.append("决策高度可靠，基于充分的分析和验证")
        elif confidence > 0.6:
            explanation_parts.append("决策较为可靠，但存在一定不确定性")
        else:
            explanation_parts.append("决策可靠性较低，建议进一步验证")
        
        return "\n".join(explanation_parts)
    
    def get_system_report(self):
        """获取系统报告"""
        return {
            'decision_tracing': {
                'total_traces': len(self.decision_tracer.decision_log),
                'recent_decisions': self.decision_tracer.get_recent_decisions(5)
            },
            'confidence_calibration': self.confidence_calibrator.get_calibration_report(),
            'feature_analysis': {
                'tracked_features': len(self.feature_analyzer.feature_importance),
                'top_features': self.feature_analyzer.get_feature_importance(5)
            },
            'counterfactual_analysis': {
                'total_scenarios': len(self.counterfactual_generator.counterfactual_scenarios),
                'last_generated': datetime.fromtimestamp(
                    self.counterfactual_generator.counterfactual_scenarios[-1]['timestamp']
                ).isoformat() if self.counterfactual_generator.counterfactual_scenarios else None
            },
            'report_timestamp': datetime.now().isoformat()
        }
    
    def export_explanations(self, file_path):
        """导出解释数据"""
        try:
            export_data = {
                'decision_traces': self.decision_tracer.decision_log[-100:],  # 最近100个决策
                'calibration_data': self.confidence_calibrator.calibration_data[-50:],
                'feature_importance': self.feature_analyzer.get_feature_importance(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "export_path": file_path}
            
        except Exception as e:
            error_handler.handle_error(e, "ExplainableAI", "解释数据导出失败")
            return {"error": str(e)}