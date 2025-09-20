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

"""
基础模型类 - 所有模型的基类
Base Model Class - Base class for all models

提供通用接口和功能，确保所有模型的一致性
Provides common interfaces and functionality to ensure consistency across all models
"""
"""
预测模型：负责时间序列预测、趋势分析和概率预测
Prediction Model: Responsible for time series forecasting, trend analysis, and probabilistic prediction
"""
import time
import numpy as np
from typing import Dict, List, Any
from core.error_handling import error_handler


"""
PredictionModel类 - 中文类描述
PredictionModel Class - English class description
"""
class PredictionModel:
    """预测模型类
    Prediction Model Class
    """
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self):
        # 预测方法库
        # Prediction methods library
        self.prediction_methods = {
            'time_series': self._time_series_forecast,
            'trend_analysis': self._trend_analysis,
            'probabilistic': self._probabilistic_prediction,
            'pattern_matching': self._pattern_matching
        }
        
        # 预测历史记录
        # Prediction prediction_history
        self.prediction_history = []
        
        # 训练历史记录
        # Training history
        self.training_history = []
        
        # 模型配置
        # Model configuration
        self.config = {
            'confidence_threshold': 0.7,
            'max_history_size': 1000,
            'default_horizon': 5,  # 默认预测步长 Default prediction steps
            'training_epochs': 10,  # 默认训练轮数 Default training epochs
            'learning_rate': 0.001,  # 默认学习率 Default learning rate
            'max_training_history': 50  # 最大训练历史记录 Max training history
        }
    
    
"""
train函数 - 中文函数描述
train Function - English function description

Args:
    training_data: 训练数据 (Training data)
    parameters: 训练参数，可选 (Training parameters, optional)
    callback: 进度回调函数，可选 (Progress callback function, optional)
    
Returns:
    训练结果字典 (Training result dictionary)
"""
def train(self, training_data: Any = None, parameters: Dict[str, Any] = None, 
          callback: Callable[[int, Dict], None] = None) -> Dict[str, Any]:
        """训练预测模型
        Train prediction model
        """
        try:
            error_handler.log_info("开始训练预测模型", "PredictionModel")
            
            # 设置训练参数
            # Set training parameters
            params = parameters or {}
            epochs = params.get("epochs", self.config['training_epochs'])
            learning_rate = params.get("learning_rate", self.config['learning_rate'])
            
            # 模拟训练过程
            # Simulate training process
            start_time = time.time()
            training_metrics = {
                'loss': [],
                'accuracy': [],
                'confidence_improvement': []
            }
            
            for epoch in range(epochs):
                # 模拟训练进度
                # Simulate training progress
                progress = int((epoch + 1) * 100 / epochs)
                
                # 计算模拟指标
                # Calculate simulated metrics
                loss = 0.8 - (epoch * 0.07)  # 损失逐渐减少 Loss gradually decreases
                accuracy = 60 + (epoch * 3)   # 准确率逐渐提高 Accuracy gradually improves
                confidence_improvement = 0.1 + (epoch * 0.08)  # 置信度改进 Confidence improvement
                
                # 记录指标
                # Record metrics
                training_metrics['loss'].append(loss)
                training_metrics['accuracy'].append(accuracy)
                training_metrics['confidence_improvement'].append(confidence_improvement)
                
                # 调用进度回调
                # Call progress callback
                if callback:
                    callback(progress, {
                        'epoch': epoch + 1,
                        'loss': loss,
                        'accuracy': accuracy,
                        'confidence_improvement': confidence_improvement
                    })
                
                # 模拟训练时间
                # Simulate training time
                time.sleep(0.3)
            
            # 计算最终指标
            # Calculate final metrics
            training_time = time.time() - start_time
            final_loss = training_metrics['loss'][-1]
            final_accuracy = training_metrics['accuracy'][-1]
            final_confidence = training_metrics['confidence_improvement'][-1]
            
            # 记录训练历史
            # Record training history
            training_record = {
                'timestamp': time.time(),
                'training_time': training_time,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'final_metrics': {
                    'loss': final_loss,
                    'accuracy': final_accuracy,
                    'confidence_improvement': final_confidence
                },
                'training_metrics': training_metrics
            }
            
            self.training_history.append(training_record)
            
            # 保持训练历史记录大小
            # Keep training history size within limit
            if len(self.training_history) > self.config['max_training_history']:
                self.training_history.pop(0)
            
            # 更新模型配置（模拟训练效果）
            # Update model configuration (simulate training effect)
            self.config['confidence_threshold'] = min(0.9, self.config['confidence_threshold'] + 0.05)
            
            error_handler.log_info(f"预测模型训练完成，耗时: {training_time:.2f}秒", "PredictionModel")
            
            return {
                'status': 'completed',
                'training_time': training_time,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'final_metrics': {
                    'loss': final_loss,
                    'accuracy': final_accuracy,
                    'confidence_improvement': final_confidence
                },
                'training_history_size': len(self.training_history)
            }
            
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "训练失败")
            return {"error": str(e)}


"""
predict函数 - 中文函数描述
predict Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def predict(self, data: Any, method: str = 'auto', horizon: int = None, **kwargs):
        """执行预测
        Execute prediction
        """
        try:
            error_handler.log_info(f"开始预测，方法: {method}", "PredictionModel")
            
            # 自动选择预测方法
            # Automatically select prediction method
            if method == 'auto':
                method = self._select_prediction_method(data)
            
            # 获取预测方法
            # Get prediction method
            if method not in self.prediction_methods:
                error_handler.log_warning(f"未知预测方法: {method}", "PredictionModel")
                method = 'trend_analysis'  # 默认回退方法 Default fallback method
            
            # 设置预测步长
            # Set prediction horizon
            if horizon is None:
                horizon = self.config['default_horizon']
            
            # 执行预测
            # Execute prediction
            prediction_result = self.prediction_methods[method](data, horizon, **kwargs)
            
            # 记录预测历史
            # Record prediction history
            self._record_prediction({
                'method': method,
                'data': data,
                'result': prediction_result,
                'timestamp': time.time(),
                'horizon': horizon
            })
            
            return prediction_result
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "预测失败")
            return {"error": str(e)}
    
    
"""
_select_prediction_method函数 - 中文函数描述
_select_prediction_method Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _select_prediction_method(self, data):
        """自动选择最适合的预测方法
        Automatically select the most suitable prediction method
        """
        # 简单实现：基于数据类型选择方法
        # Simple implementation: select method based on data type
        if isinstance(data, (list, np.ndarray)):
            if len(data) > 10:  # 足够的时间序列数据 Enough time series data
                return 'time_series'
            else:
                return 'trend_analysis'
        elif isinstance(data, dict):
            if 'probabilities' in data or 'uncertainty' in data:
                return 'probabilistic'
            else:
                return 'pattern_matching'
        else:
            return 'trend_analysis'  # 默认方法 Default method
    
    
"""
_time_series_forecast函数 - 中文函数描述
_time_series_forecast Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _time_series_forecast(self, data, horizon, **kwargs):
        """时间序列预测
        Time series forecasting
        """
        # 简单实现：使用移动平均进行预测
        # Simple implementation: use moving average for forecasting
        try:
            if not isinstance(data, (list, np.ndarray)):
                data = list(data) if hasattr(data, '__iter__') else [data]
            
            data = np.array(data)
            
            # 计算移动平均
            # Calculate moving average
            window_size = min(len(data), 3)  # 简单的窗口大小 Simple window size
            if window_size == 0:
                return {"forecast": [], "confidence": 0.0}
            
            # 简单移动平均预测
            # Simple moving average forecast
            last_values = data[-window_size:]
            forecast = []
            
            for i in range(horizon):
                next_value = np.mean(last_values)
                forecast.append(float(next_value))
                last_values = np.append(last_values[1:], next_value)
            
            # 计算置信度（基于数据波动性）
            # Calculate confidence (based on data volatility)
            volatility = np.std(data) / (np.mean(data) + 1e-10)
            confidence = max(0.1, 1.0 - min(volatility, 1.0))
            
            return {
                "forecast": forecast,
                "confidence": float(confidence),
                "method": "moving_average",
                "window_size": window_size
            }
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "时间序列预测失败")
            return {"forecast": [], "confidence": 0.0, "error": str(e)}
    
    
"""
_trend_analysis函数 - 中文函数描述
_trend_analysis Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _trend_analysis(self, data, horizon, **kwargs):
        """趋势分析预测
        Trend analysis prediction
        """
        # 简单实现：线性趋势外推
        # Simple implementation: linear trend extrapolation
        try:
            if not isinstance(data, (list, np.ndarray)):
                data = list(data) if hasattr(data, '__iter__') else [data]
            
            data = np.array(data)
            
            if len(data) < 2:
                # 数据不足，使用最后值
                # Insufficient data, use last value
                forecast = [float(data[-1])] * horizon if len(data) > 0 else [0.0] * horizon
                return {
                    "forecast": forecast,
                    "confidence": 0.3,
                    "method": "constant"
                }
            
            # 简单线性回归
            # Simple linear regression
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # 生成预测
            # Generate forecast
            forecast = []
            for i in range(1, horizon + 1):
                next_value = slope * (len(data) + i) + intercept
                forecast.append(float(next_value))
            
            # 计算置信度（基于拟合优度）
            # Calculate confidence (based on goodness of fit)
            residuals = data - (slope * x + intercept)
            rss = np.sum(residuals ** 2)
            tss = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (rss / tss) if tss > 0 else 0
            confidence = max(0.1, min(r_squared, 1.0))
            
            return {
                "forecast": forecast,
                "confidence": float(confidence),
                "method": "linear_trend",
                "slope": float(slope),
                "intercept": float(intercept)
            }
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "趋势分析失败")
            return {"forecast": [], "confidence": 0.0, "error": str(e)}
    
    
"""
_probabilistic_prediction函数 - 中文函数描述
_probabilistic_prediction Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _probabilistic_prediction(self, data, horizon, **kwargs):
        """概率预测
        Probabilistic prediction
        """
        # 简单实现：基于概率分布的预测
        # Simple implementation: prediction based on probability distribution
        try:
            probabilities = data.get('probabilities', {})
            uncertainty = data.get('uncertainty', 0.5)
            
            # 如果没有概率数据，回退到趋势分析
            # If no probability data, fall back to trend analysis
            if not probabilities:
                return self._trend_analysis(data, horizon, **kwargs)
            
            # 简单概率预测
            # Simple probabilistic prediction
            forecast = []
            confidence = 1.0 - uncertainty
            
            # 对于每个预测步长，选择最可能的值
            # For each prediction step, select the most likely value
            for i in range(horizon):
                if probabilities:
                    # 简单实现：取概率最高的值
                    # Simple implementation: take the value with highest probability
                    best_value = max(probabilities.items(), key=lambda x: x[1])[0]
                    forecast.append(best_value)
                else:
                    forecast.append(None)
            
            return {
                "forecast": forecast,
                "confidence": float(confidence),
                "method": "probabilistic",
                "uncertainty": float(uncertainty)
            }
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "概率预测失败")
            return {"forecast": [], "confidence": 0.0, "error": str(e)}
    
    
"""
_pattern_matching函数 - 中文函数描述
_pattern_matching Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _pattern_matching(self, data, horizon, **kwargs):
        """模式匹配预测
        Pattern matching prediction
        """
        # 简单实现：基于历史模式匹配
        # Simple implementation: pattern matching based on history
        try:
            # 如果没有足够历史数据，回退到趋势分析
            # If not enough historical data, fall back to trend analysis
            if len(self.prediction_history) < 3:
                return self._trend_analysis(data, horizon, **kwargs)
            
            # 简单模式匹配：查找相似历史模式
            # Simple pattern matching: find similar historical patterns
            similar_patterns = []
            
            for history in self.prediction_history[-10:]:  # 检查最近10条历史 Check last 10 histories
                if 'result' in history and 'forecast' in history['result']:
                    # 简单相似度计算（基于数据长度和类型）
                    # Simple similarity calculation (based on data length and type)
                    similarity = self._calculate_similarity(data, history['data'])
                    if similarity > 0.5:  # 相似度阈值 Similarity threshold
                        similar_patterns.append((history, similarity))
            
            if similar_patterns:
                # 使用最相似的历史模式进行预测
                # Use the most similar historical pattern for prediction
                best_match = max(similar_patterns, key=lambda x: x[1])
                historical_result = best_match[0]['result']
                
                # 简单外推：使用历史预测模式
                # Simple extrapolation: use historical forecast pattern
                if 'forecast' in historical_result and len(historical_result['forecast']) >= horizon:
                    forecast = historical_result['forecast'][:horizon]
                else:
                    forecast = [historical_result.get('forecast', [0])[-1]] * horizon
                
                confidence = best_match[1] * historical_result.get('confidence', 0.5)
                
                return {
                    "forecast": forecast,
                    "confidence": float(confidence),
                    "method": "pattern_matching",
                    "similar_patterns_count": len(similar_patterns)
                }
            else:
                # 没有找到相似模式，回退到趋势分析
                # No similar patterns found, fall back to trend analysis
                return self._trend_analysis(data, horizon, **kwargs)
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "模式匹配失败")
            return {"forecast": [], "confidence": 0.0, "error": str(e)}
    
    
"""
_calculate_similarity函数 - 中文函数描述
_calculate_similarity Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_similarity(self, data1, data2):
        """计算两个数据模式的相似度
        Calculate similarity between two data patterns
        """
        # 简单实现：基于数据类型和长度的相似度
        # Simple implementation: similarity based on data type and length
        try:
            if type(data1) != type(data2):
                return 0.0
            
            if isinstance(data1, (list, np.ndarray)) and isinstance(data2, (list, np.ndarray)):
                # 对于数值序列，计算相关系数
                # For numerical sequences, calculate correlation coefficient
                if len(data1) > 0 and len(data2) > 0:
                    min_len = min(len(data1), len(data2))
                    data1_sub = data1[:min_len]
                    data2_sub = data2[:min_len]
                    
                    # 简单相关系数
                    # Simple correlation coefficient
                    correlation = np.corrcoef(data1_sub, data2_sub)[0, 1]
                    return max(0.0, float(correlation))
                else:
                    return 0.0
            else:
                # 对于其他类型，简单类型匹配
                # For other types, simple type matching
                return 0.5 if data1 == data2 else 0.0
        except:
            return 0.0
    
    
"""
_record_prediction函数 - 中文函数描述
_record_prediction Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _record_prediction(self, prediction_record):
        """记录预测历史
        Record prediction history
        """
        self.prediction_history.append(prediction_record)
        
        # 保持历史记录不超过最大大小
        # Keep history size within limit
        if len(self.prediction_history) > self.config['max_history_size']:
            self.prediction_history.pop(0)
    
    
"""
get_prediction_history函数 - 中文函数描述
get_prediction_history Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def get_prediction_history(self, limit=10):
        """获取预测历史
        Get prediction history
        """
        return self.prediction_history[-limit:]
    
    
"""
clear_history函数 - 中文函数描述
clear_history Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def clear_history(self):
        """清空预测历史
        Clear prediction history
        """
        self.prediction_history = []
    
    
"""
update_config函数 - 中文函数描述
update_config Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def update_config(self, new_config):
        """更新模型配置
        Update model configuration
        """
        self.config.update(new_config)
        error_handler.log_info(f"预测模型配置已更新: {self.config}", "PredictionModel")
    
    
"""
get_status函数 - 中文函数描述
get_status Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def get_status(self):
        """获取模型状态
        Get model status
        """
        return {
            "status": "active",
            "history_size": len(self.prediction_history),
            "config": self.config,
            "methods_available": list(self.prediction_methods.keys())
        }

    
"""
predictive_decision_making函数 - 中文函数描述
predictive_decision_making Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def predictive_decision_making(self, current_state, goals, constraints=None, horizon=5):
        """基于预测的前瞻性决策
        Predictive decision making based on forecasts
        """
        try:
            error_handler.log_info("开始前瞻性决策制定", "PredictionModel")
            
            # 预测未来状态
            future_predictions = self.predict(current_state, horizon=horizon)
            
            if 'error' in future_predictions:
                return {"error": future_predictions['error']}
            
            # 评估预测结果与目标的匹配度
            goal_assessment = self._assess_goal_alignment(future_predictions, goals)
            
            # 生成决策选项
            decision_options = self._generate_decision_options(current_state, future_predictions, goals, constraints)
            
            # 选择最优决策
            optimal_decision = self._select_optimal_decision(decision_options, goal_assessment)
            
            # 记录决策过程
            self._record_decision_making({
                'current_state': current_state,
                'predictions': future_predictions,
                'goals': goals,
                'decision': optimal_decision,
                'timestamp': time.time()
            })
            
            error_handler.log_info("前瞻性决策制定完成", "PredictionModel")
            return optimal_decision
            
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "前瞻性决策制定失败")
            return {"error": str(e)}
    
    
"""
_assess_goal_alignment函数 - 中文函数描述
_assess_goal_alignment Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _assess_goal_alignment(self, predictions, goals):
        """评估预测结果与目标的匹配度"""
        alignment_scores = {}
        
        if isinstance(goals, dict) and 'targets' in goals:
            for target_name, target_value in goals['targets'].items():
                if 'forecast' in predictions:
                    # 简单匹配度计算：基于预测值与目标值的接近程度
                    forecast_values = predictions['forecast']
                    if forecast_values and isinstance(target_value, (int, float)):
                        # 计算平均绝对误差的倒数作为匹配度
                        mae = np.mean([abs(fv - target_value) for fv in forecast_values if isinstance(fv, (int, float))])
                        alignment_score = 1.0 / (1.0 + mae) if mae > 0 else 1.0
                        alignment_scores[target_name] = min(max(alignment_score, 0), 1)
        
        return alignment_scores
    
    
"""
_generate_decision_options函数 - 中文函数描述
_generate_decision_options Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _generate_decision_options(self, current_state, predictions, goals, constraints):
        """生成可能的决策选项"""
        decision_options = []
        
        # 基于预测生成决策选项
        if 'forecast' in predictions and predictions['forecast']:
            forecast_trend = self._analyze_forecast_trend(predictions['forecast'])
            
            # 根据趋势生成决策
            if forecast_trend == 'increasing':
                decision_options.append({
                    'action': 'increase_investment',
                    'description': '增加投资以利用上升趋势',
                    'expected_impact': 0.7,
                    'risk_level': 0.3
                })
                decision_options.append({
                    'action': 'maintain_current',
                    'description': '维持当前策略',
                    'expected_impact': 0.5,
                    'risk_level': 0.2
                })
            elif forecast_trend == 'decreasing':
                decision_options.append({
                    'action': 'reduce_exposure',
                    'description': '减少暴露以降低风险',
                    'expected_impact': 0.6,
                    'risk_level': 0.4
                })
                decision_options.append({
                    'action': 'hedge_position',
                    'description': '对冲头寸',
                    'expected_impact': 0.4,
                    'risk_level': 0.3
                })
            else:  # stable or unknown
                decision_options.append({
                    'action': 'monitor_closely',
                    'description': '密切监控情况',
                    'expected_impact': 0.3,
                    'risk_level': 0.1
                })
        
        # 添加基于约束的选项
        if constraints:
            for constraint in constraints.get('limitations', []):
                decision_options.append({
                    'action': f'respect_{constraint}',
                    'description': f'遵守约束: {constraint}',
                    'expected_impact': 0.5,
                    'risk_level': 0.1
                })
        
        return decision_options
    
    
"""
_analyze_forecast_trend函数 - 中文函数描述
_analyze_forecast_trend Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _analyze_forecast_trend(self, forecast):
        """分析预测趋势"""
        if not forecast or len(forecast) < 2:
            return 'unknown'
        
        # 简单趋势分析
        first_half = forecast[:len(forecast)//2]
        second_half = forecast[len(forecast)//2:]
        
        if not all(isinstance(f, (int, float)) for f in forecast):
            return 'unknown'
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        if avg_second > avg_first * 1.1:
            return 'increasing'
        elif avg_second < avg_first * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    
"""
_select_optimal_decision函数 - 中文函数描述
_select_optimal_decision Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _select_optimal_decision(self, decision_options, goal_assessment):
        """选择最优决策"""
        if not decision_options:
            return {"error": "没有可用的决策选项"}
        
        # 计算每个选项的得分
        scored_options = []
        for option in decision_options:
            score = self._calculate_decision_score(option, goal_assessment)
            scored_options.append({
                **option,
                'score': score
            })
        
        # 选择得分最高的选项
        best_option = max(scored_options, key=lambda x: x['score'])
        
        return {
            'selected_decision': best_option,
            'alternative_options': scored_options,
            'selection_time': time.time()
        }
    
    
"""
_calculate_decision_score函数 - 中文函数描述
_calculate_decision_score Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_decision_score(self, decision, goal_assessment):
        """计算决策得分"""
        base_score = decision.get('expected_impact', 0.5) * (1 - decision.get('risk_level', 0.5))
        
        # 如果有关联的目标，调整得分
        if goal_assessment:
            avg_alignment = np.mean(list(goal_assessment.values())) if goal_assessment else 0.5
            base_score *= (0.3 + 0.7 * avg_alignment)  # 加权调整
        
        return base_score
    
    
"""
_record_decision_making函数 - 中文函数描述
_record_decision_making Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _record_decision_making(self, decision_record):
        """记录决策过程"""
        if not hasattr(self, 'decision_history'):
            self.decision_history = []
        
        self.decision_history.append(decision_record)
        
        # 保持历史记录大小
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    
"""
get_decision_history函数 - 中文函数描述
get_decision_history Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def get_decision_history(self, limit=10):
        """获取决策历史"""
        if hasattr(self, 'decision_history'):
            return self.decision_history[-limit:]
        return []
    
    
"""
integrate_with_planning函数 - 中文函数描述
integrate_with_planning Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def integrate_with_planning(self, planning_model, current_state, goals):
        """与规划模型集成，提供预测输入"""
        try:
            error_handler.log_info("开始与规划模型集成", "PredictionModel")
            
            # 生成预测
            predictions = self.predict(current_state, horizon=5)
            
            if 'error' in predictions:
                return {"error": predictions['error']}
            
            # 创建增强的目标信息，包含预测
            enhanced_goals = {
                'original_goals': goals,
                'predictions': predictions,
                'recommended_actions': self.predictive_decision_making(current_state, goals)
            }
            
            # 调用规划模型创建计划
            plan = planning_model.create_plan(enhanced_goals, available_models=[])
            
            error_handler.log_info("与规划模型集成完成", "PredictionModel")
            return plan
            
        except Exception as e:
            error_handler.handle_error(e, "PredictionModel", "与规划模型集成失败")
            return {"error": str(e)}
