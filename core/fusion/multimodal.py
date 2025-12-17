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
multimodal.py - 中文描述
multimodal.py - English description

版权所有 (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""
import time
import numpy as np
from core.error_handling import error_handler


"""
MultimodalFusion类 - 中文类描述
MultimodalFusion Class - English class description
"""
class MultimodalFusion:
    """多模态数据融合引擎 / Multimodal Data Fusion Engine"""
    
    """
    __init__函数 - 中文函数描述
    __init__ Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def __init__(self):
        # 融合权重
        self.fusion_weights = {
            'text': 0.3,
            'audio': 0.25,
            'image': 0.3,
            'sensor': 0.15
        }
        
        # 新增：上下文感知融合配置
        self.context_aware_fusion = {
            'enabled': True,
            'memory_window': 5,  # 记忆窗口大小
            'context_history': []  # 上下文历史
        }
        
        # 新增：自适应融合学习器
        self.adaptive_learner = {
            'enabled': False,
            'learning_rate': 0.01,
            'feedback_history': []
        }

        # 增强：多模态融合方法
    
    """
    fuse_results函数 - 中文函数描述
    fuse_results Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def fuse_results(self, results, modalities=None, context=None):
        """融合多个模型的结果，支持上下文感知"""
        try:
            error_handler.log_info(f"开始多模态融合，模态: {list(results.keys())}", "MultimodalFusion")
            
            # 如果未指定模态，则使用所有可用模态
            if modalities is None:
                modalities = list(results.keys())
            
            # 过滤出要融合的结果
            filtered_results = {m: results[m] for m in modalities if m in results}
            if not filtered_results:
                error_handler.log_warning("没有可融合的结果", "MultimodalFusion")
                return {}
            
            # 检查是否应该使用上下文感知融合
            if self.context_aware_fusion['enabled'] and context:
                # 更新上下文历史
                self._update_context_history(context)
                
                # 根据上下文调整融合权重
                context_adjusted_weights = self._adjust_weights_based_on_context(context)
                original_weights = self.fusion_weights.copy()
                
                # 应用上下文调整的权重
                self.fusion_weights.update(context_adjusted_weights)
            
            # 根据模态类型选择融合策略
            fusion_strategy = self._select_fusion_strategy(modalities)
            
            # 执行融合
            fused_result = fusion_strategy(filtered_results)
            
            # 如果使用了上下文感知融合，恢复原始权重
            if self.context_aware_fusion['enabled'] and context:
                self.fusion_weights = original_weights
            
            error_handler.log_info("多模态融合完成", "MultimodalFusion")
            return fused_result
        except Exception as e:
            error_handler.handle_error(e, "MultimodalFusion", "融合结果失败")
            return {}

    # 新增：基于上下文调整融合权重
    
    """
    _adjust_weights_based_on_context函数 - 中文函数描述
    _adjust_weights_based_on_context Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _adjust_weights_based_on_context(self, context):
        """根据上下文调整融合权重"""
        # 简单实现：基于上下文类型调整权重
        adjusted_weights = {}
        
        # 分析上下文
        context_type = context.get('type', '')
        context_importance = context.get('importance', 1.0)
        
        # 根据上下文类型调整权重
        if context_type == 'visual_task':
            # 视觉任务中增加视觉模态权重
            adjusted_weights['image'] = self.fusion_weights.get('image', 0.25) * 1.5
            adjusted_weights['video'] = self.fusion_weights.get('video', 0.25) * 1.5
        elif context_type == 'audio_task':
            # 音频任务中增加音频模态权重
            adjusted_weights['audio'] = self.fusion_weights.get('audio', 0.25) * 1.5
        elif context_type == 'text_task':
            # 文本任务中增加文本模态权重
            adjusted_weights['text'] = self.fusion_weights.get('text', 0.25) * 1.5
        
        # 根据上下文重要性调整权重缩放
        for modality, weight in adjusted_weights.items():
            adjusted_weights[modality] *= context_importance
        
        # 归一化调整后的权重
        if adjusted_weights:
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for modality in adjusted_weights:
                    adjusted_weights[modality] /= total_weight
        
        return adjusted_weights

    # 新增：更新上下文历史
    
    """
    _update_context_history函数 - 中文函数描述
    _update_context_history Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _update_context_history(self, context):
        """更新上下文历史"""
        # 添加新上下文
        self.context_aware_fusion['context_history'].append({
            'context': context,
            'timestamp': time.time()
        })
        
        # 保持历史记录不超过窗口大小
        if len(self.context_aware_fusion['context_history']) > self.context_aware_fusion['memory_window']:
            self.context_aware_fusion['context_history'].pop(0)

    # 增强：加权融合策略
    
    """
    _weighted_fusion函数 - 中文函数描述
    _weighted_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _weighted_fusion(self, results):
        """高级加权融合策略，考虑模型置信度和可靠性"""
        # 简单实现，实际应用中需要根据具体数据类型和模型结果调整
        fused_result = {}
        
        # 计算所有权重的总和
        total_weight = sum(self.fusion_weights.get(m, 0.25) for m in results)
        
        # 对每个模态的结果进行加权融合
        for modality, result in results.items():
            # 获取基础权重
            base_weight = self.fusion_weights.get(modality, 0.25) / total_weight
            
            # 如果结果包含置信度分数，使用它来调整权重
            confidence = result.get('confidence', 1.0)
            weight = base_weight * confidence
            
            # 根据结果类型执行不同的融合操作
            if isinstance(result, dict):
                for key, value in result.items():
                    # 跳过元数据字段
                    if key in ['confidence', 'timestamp', 'source']:
                        continue
                        
                    if key not in fused_result:
                        fused_result[key] = 0
                        
                    # 如果是数值类型，进行加权求和
                    if isinstance(value, (int, float)):
                        fused_result[key] += value * weight
                    # 如果是字符串类型，处理文本融合
                    elif isinstance(value, str):
                        if key not in fused_result:
                            fused_result[key] = value
                        else:
                            # 简单文本融合：添加新信息
                            if value not in fused_result[key]:
                                fused_result[key] += f"; {value}"
                    # 对于数组类型，取加权平均
                    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                        if key not in fused_result:
                            fused_result[key] = [0] * len(value)
                        for i, val in enumerate(value):
                            fused_result[key][i] += val * weight
                    # 其他类型，可以根据具体需求扩展
                    elif isinstance(value, dict):
                        if key not in fused_result:
                            fused_result[key] = {}
                        # 递归融合字典
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                if sub_key not in fused_result[key]:
                                    fused_result[key][sub_key] = 0
                                fused_result[key][sub_key] += sub_value * weight
                            elif isinstance(sub_value, str) and sub_key not in fused_result[key]:
                                fused_result[key][sub_key] = sub_value
        
        # 添加融合元数据
        fused_result['fusion_metadata'] = {
            'modalities': list(results.keys()),
            'weights_used': self.fusion_weights.copy(),
            'timestamp': time.time()
        }
        
        return fused_result

    # 增强：文本和视觉融合策略
    
    """
    _text_visual_fusion函数 - 中文函数描述
    _text_visual_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _text_visual_fusion(self, results):
        """增强的文本和视觉融合策略"""
        # 实现文本和视觉数据的融合
        # 例如，将图像识别的结果与文本描述结合起来
        text_result = results.get('text', {})
        visual_result = results.get('image', {}) or results.get('video', {})
        
        # 基础融合
        fused_result = {**text_result, **visual_result}
        
        # 文本描述和视觉描述的智能融合
        if 'description' in text_result and 'description' in visual_result:
            # 分析两个描述的相关性
            text_desc = text_result['description']
            visual_desc = visual_result['description']
            
            # 简单相关性分析：关键词重叠
            text_words = set(text_desc.lower().split())
            visual_words = set(visual_desc.lower().split())
            overlap = text_words.intersection(visual_words)
            
            # 根据相关性决定融合方式
            if len(overlap) > 0:
                # 高相关性：合并描述
                fused_result['description'] = f"{visual_desc} {text_desc}"
            else:
                # 低相关性：分别保留
                fused_result['visual_description'] = visual_desc
                fused_result['text_description'] = text_desc
        
        # 融合情感分析结果
        if 'emotion' in text_result and 'emotion' in visual_result:
            text_emotion = text_result['emotion']
            visual_emotion = visual_result['emotion']
            
            # 情感一致性检查
            if text_emotion == visual_emotion:
                # 情感一致，增强置信度
                fused_result['emotion'] = text_emotion
                fused_result['emotion_confidence'] = (text_result.get('confidence', 1.0) + 
                                                      visual_result.get('confidence', 1.0)) / 2
            else:
                # 情感不一致，分别保留
                fused_result['text_emotion'] = text_emotion
                fused_result['visual_emotion'] = visual_emotion
        
        return fused_result

    # 增强：音频和视觉融合策略
    
    """
    _audio_visual_fusion函数 - 中文函数描述
    _audio_visual_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _audio_visual_fusion(self, results):
        """增强的音频和视觉融合策略"""
        # 实现音频和视觉数据的融合
        audio_result = results.get('audio', {})
        visual_result = results.get('image', {}) or results.get('video', {})
        
        # 基础融合
        fused_result = {**audio_result, **visual_result}
        
        # 融合语音转文本和图像识别结果
        if 'transcript' in audio_result and 'objects' in visual_result:
            transcript = audio_result['transcript']
            objects = visual_result['objects']
            
            # 检查语音中是否提到了图像中的对象
            mentioned_objects = []
            for obj in objects:
                if isinstance(obj, dict) and 'name' in obj:
                    obj_name = obj['name'].lower()
                    if obj_name in transcript.lower():
                        mentioned_objects.append(obj)
                elif isinstance(obj, str):
                    if obj.lower() in transcript.lower():
                        mentioned_objects.append(obj)
            
            if mentioned_objects:
                # 添加关联信息
                fused_result['audio_visual_correlation'] = {
                    'mentioned_objects': mentioned_objects,
                    'transcript_reference': transcript
                }
        
        # 添加视听摘要
        if 'transcript' in audio_result and 'description' in visual_result:
            fused_result['audio_visual_summary'] = f"视觉内容: {visual_result['description']} 音频内容: {audio_result['transcript']}"
        
        return fused_result

    # 新增：基于深度学习的融合
    
    """
    deep_learning_fusion函数 - 中文函数描述
    deep_learning_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def deep_learning_fusion(self, results, model=None):
        """使用深度学习模型进行多模态融合"""
        # 简单实现，实际应使用预训练的多模态融合模型
        # 例如 CLIP、ViLBERT、ALIGN 等
        
        # 检查是否提供了自定义模型
        if model and hasattr(model, 'fuse'):
            try:
                return model.fuse(results)
            except Exception as e:
                error_handler.handle_error(e, "MultimodalFusion", "深度学习融合失败")
                # 回退到加权融合
                return self._weighted_fusion(results)
        
        # 提取特征向量
        feature_vectors = []
        modalities = []
        
        for modality, result in results.items():
            if 'features' in result and isinstance(result['features'], list):
                feature_vectors.append(np.array(result['features']))
                modalities.append(modality)
        
        if not feature_vectors:
            # 没有特征向量，回退到加权融合
            return self._weighted_fusion(results)
        
        # 简单特征融合：连接特征向量
        concatenated_features = np.concatenate(feature_vectors) if len(feature_vectors) > 0 else np.array([])
        
        # 返回融合结果
        return {
            'fused_features': concatenated_features.tolist(),
            'modalities_used': modalities,
            'fusion_method': 'deep_learning_concat'
        }

    # 新增：基于反馈的自适应融合
    
    """
    update_based_on_feedback函数 - 中文函数描述
    update_based_on_feedback Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def update_based_on_feedback(self, feedback, fused_result):
        """基于用户或系统反馈更新融合策略"""
        if not self.adaptive_learner['enabled']:
            return
        
        # 记录反馈
        self.adaptive_learner['feedback_history'].append({
            'feedback': feedback,
            'fused_result': fused_result,
            'timestamp': time.time()
        })
        
        # 简单实现：基于反馈调整权重
        # 实际应用中应使用更复杂的学习算法
        if 'preferred_modality' in feedback:
            preferred_modality = feedback['preferred_modality']
            if preferred_modality in self.fusion_weights:
                # 增加首选模态的权重
                self.fusion_weights[preferred_modality] *= (1 + self.adaptive_learner['learning_rate'])
                
                # 归一化权重
                total_weight = sum(self.fusion_weights.values())
                for modality in self.fusion_weights:
                    self.fusion_weights[modality] /= total_weight
                    
                error_handler.log_info(f"基于反馈更新融合权重: {self.fusion_weights}", "MultimodalFusion")

    # 内部方法：选择融合策略
    
    """
    _select_fusion_strategy函数 - 中文函数描述
    _select_fusion_strategy Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _select_fusion_strategy(self, modalities):
        """根据模态组合选择融合策略"""
        # 检查是否包含文本和视觉
        if 'text' in modalities and ('image' in modalities or 'video' in modalities):
            return self._text_visual_fusion
        # 检查是否包含音频和视觉
        elif 'audio' in modalities and ('image' in modalities or 'video' in modalities):
            return self._audio_visual_fusion
        else:
            # 默认使用加权融合
            return self._weighted_fusion

    # 新增：自适应融合策略
    
    """
    adaptive_fusion函数 - 中文函数描述
    adaptive_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def adaptive_fusion(self, results, context=None):
        """基于上下文的自适应融合策略
        Context-based adaptive fusion strategy
        """
        try:
            error_handler.log_info("开始自适应多模态融合", "MultimodalFusion")
            
            # 简单实现：根据上下文和结果类型自动选择最佳融合策略
            if not results:
                return {}
            
            modalities = list(results.keys())
            
            # 分析结果质量
            result_quality = self._assess_result_quality(results)
            
            # 根据结果质量调整权重
            adjusted_weights = self._adjust_weights_based_on_quality(result_quality)
            
            # 执行加权融合
            fused_result = self._weighted_fusion_simple(results, adjusted_weights)
            
            return fused_result
        except Exception as e:
            error_handler.handle_error(e, "MultimodalFusion", "自适应融合失败")
            # 回退到标准融合
            return self._weighted_fusion(results)
    
    """
    _assess_result_quality函数 - 中文函数描述
    _assess_result_quality Function - English function description
    
    Args:
        results: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _assess_result_quality(self, results):
        """评估各模态结果的质量"""
        quality_scores = {}
        
        for modality, result in results.items():
            quality = 1.0
            
            # 基于置信度评分
            if 'confidence' in result:
                quality = result['confidence']
            
            # 基于完整性评分
            if isinstance(result, dict):
                # 检查关键字段是否存在
                required_fields = {'text': ['content'], 'image': ['description'], 'audio': ['transcript']}
                if modality in required_fields:
                    missing_fields = [field for field in required_fields[modality] if field not in result]
                    if missing_fields:
                        quality *= 0.7  # 减少质量评分
            
            quality_scores[modality] = quality
        
        return quality_scores
    
    """
    _adjust_weights_based_on_quality函数 - 中文函数描述
    _adjust_weights_based_on_quality Function - English function description
    
    Args:
        quality_scores: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _adjust_weights_based_on_quality(self, quality_scores):
        """根据结果质量调整融合权重"""
        adjusted_weights = {}
        
        for modality, quality in quality_scores.items():
            # 获取基础权重
            base_weight = self.fusion_weights.get(modality, 0.25)
            
            # 根据质量调整权重
            adjusted_weights[modality] = base_weight * quality
        
        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for modality in adjusted_weights:
                adjusted_weights[modality] /= total_weight
        
        return adjusted_weights
    
    """
    _weighted_fusion_simple函数 - 中文函数描述
    _weighted_fusion_simple Function - English function description
    
    Args:
        results: 参数描述 (Parameter description)
        weights: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _weighted_fusion_simple(self, results, weights):
        """简单加权融合策略"""
        fused_result = {}
        
        for modality, result in results.items():
            weight = weights.get(modality, 0.25)
            
            if isinstance(result, dict):
                for key, value in result.items():
                    if key in ['confidence', 'timestamp', 'source']:
                        continue
                        
                    if key not in fused_result:
                        fused_result[key] = 0
                        
                    if isinstance(value, (int, float)):
                        fused_result[key] += value * weight
                    elif isinstance(value, str):
                        if key not in fused_result:
                            fused_result[key] = value
                        else:
                            if value not in fused_result[key]:
                                fused_result[key] += f"; {value}"
        
        return fused_result
    
    """
    advanced_adaptive_fusion函数 - 中文函数描述
    advanced_adaptive_fusion Function - English function description
    
    Args:
        results: 多模态结果
        context: 上下文信息
        task_type: 任务类型
        
    Returns:
        融合后的结果
    """
    def advanced_adaptive_fusion(self, results, context=None, task_type=None):
        """高级自适应融合策略，考虑上下文和任务类型"""
        try:
            error_handler.log_info("开始高级自适应多模态融合", "MultimodalFusion")
            
            if not results:
                return {}
            
            # 分析融合输入
            fusion_analysis = self._analyze_fusion_inputs(results, context, task_type)
            
            # 选择最优融合策略
            selected_strategy = self._select_optimal_fusion_strategy(fusion_analysis)
            
            # 动态调整融合参数
            fusion_params = self._adjust_fusion_parameters(fusion_analysis)
            
            # 执行融合
            fused_result = selected_strategy(results, fusion_params)
            
            # 记录融合决策
            self._record_fusion_decision({
                'strategy': selected_strategy.__name__,
                'parameters': fusion_params,
                'context': context,
                'task_type': task_type,
                'timestamp': time.time()
            })
            
            error_handler.log_info("高级自适应融合完成", "MultimodalFusion")
            return fused_result
            
        except Exception as e:
            error_handler.handle_error(e, "MultimodalFusion", "高级自适应融合失败")
            # 回退到标准自适应融合
            return self.adaptive_fusion(results, context)
    
    """
    _analyze_fusion_inputs函数 - 中文函数描述
    _analyze_fusion_inputs Function - English function description
    
    Args:
        results: 参数描述 (Parameter description)
        context: 上下文信息
        task_type: 任务类型
        
    Returns:
        返回值描述 (Return value description)
    """
    def _analyze_fusion_inputs(self, results, context, task_type):
        """深度分析融合输入数据、上下文和任务类型"""
        analysis = {
            'modality_characteristics': {},
            'quality_assessment': {},
            'context_relevance': {},
            'task_requirements': {},
            'temporal_consistency': {}
        }
        
        # 分析各模态特征
        for modality, result in results.items():
            analysis['modality_characteristics'][modality] = self._analyze_modality_characteristics(result, modality)
        
        # 评估结果质量
        analysis['quality_assessment'] = self._comprehensive_quality_assessment(results)
        
        # 分析上下文相关性
        analysis['context_relevance'] = self._assess_context_relevance(results, context)
        
        # 分析任务需求
        analysis['task_requirements'] = self._analyze_task_requirements(task_type)
        
        # 检查时间一致性
        analysis['temporal_consistency'] = self._check_temporal_consistency(results)
        
        return analysis
    
    """
    _analyze_modality_characteristics函数 - 中文函数描述
    _analyze_modality_characteristics Function - English function description
    
    Args:
        result: 模态结果
        modality: 模态类型
        
    Returns:
        模态特征分析结果
    """
    def _analyze_modality_characteristics(self, result, modality):
        """分析特定模态数据的特征"""
        characteristics = {
            'data_type': self._determine_data_type(result),
            'completeness': self._assess_completeness(result),
            'consistency': self._assess_internal_consistency(result),
            'novelty': self._assess_novelty(result, modality),
            'reliability': self._assess_reliability(modality)
        }
        return characteristics
    
    """
    _determine_data_type函数 - 中文函数描述
    _determine_data_type Function - English function description
    
    Args:
        result: 要分析的结果
        
    Returns:
        数据类型
    """
    def _determine_data_type(self, result):
        """确定数据类型（数值、分类、文本、视觉等）"""
        if isinstance(result, dict):
            if any(isinstance(v, (int, float)) for v in result.values()):
                return 'numeric'
            elif any(isinstance(v, str) for v in result.values()):
                return 'textual'
            elif any(isinstance(v, list) for v in result.values()):
                return 'structured'
        elif isinstance(result, (int, float)):
            return 'numeric'
        elif isinstance(result, str):
            return 'textual'
        return 'unknown'
    
    """
    _assess_completeness函数 - 中文函数描述
    _assess_completeness Function - English function description
    
    Args:
        result: 要评估的数据
        
    Returns:
        完整性分数
    """
    def _assess_completeness(self, result):
        """评估数据完整性"""
        if isinstance(result, dict):
            # 检查关键字段是否存在
            required_fields = {'confidence', 'value', 'timestamp'}
            present_fields = set(result.keys())
            completeness = len(present_fields.intersection(required_fields)) / len(required_fields)
            return completeness
        return 0.7  # 默认完整性分数
    
    """
    _assess_internal_consistency函数 - 中文函数描述
    _assess_internal_consistency Function - English function description
    
    Args:
        result: 要评估的数据
        
    Returns:
        内部一致性分数
    """
    def _assess_internal_consistency(self, result):
        """评估内部一致性"""
        if isinstance(result, dict):
            # 检查置信度与值的一致性
            if 'confidence' in result and 'value' in result:
                confidence = result['confidence']
                value = result['value']
                # 简单一致性检查
                if isinstance(value, (int, float)) and confidence > 0.8 and abs(value) > 1.0:
                    return 0.9
                elif isinstance(value, str) and confidence > 0.8 and len(value) > 5:
                    return 0.8
            return 0.6
        return 0.5


    """
    _assess_novelty函数 - 中文函数描述
    _assess_novelty Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _assess_novelty(self, result, modality):
        """评估数据新颖性（与历史数据比较）"""
        # 简单实现：检查是否与最近结果相似
        novelty_score = 0.7  # 默认新颖性
        
        if hasattr(self, 'recent_results') and modality in self.recent_results:
            recent = self.recent_results[modality]
            if isinstance(result, dict) and isinstance(recent, dict):
                # 比较关键字段
                key_diff = len(set(result.keys()).symmetric_difference(set(recent.keys())))
                novelty_score = max(0.3, 1.0 - (key_diff / max(len(result), len(recent))))
        
        return novelty_score
    
    
    """
    _assess_reliability函数 - 中文函数描述
    _assess_reliability Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _assess_reliability(self, modality):
        """评估模态的可靠性（基于历史性能）"""
        reliability_scores = {
            'text': 0.8,
            'audio': 0.7,
            'image': 0.75,
            'video': 0.7,
            'sensor': 0.6
        }
        return reliability_scores.get(modality, 0.5)
    
    
    """
    _comprehensive_quality_assessment函数 - 中文函数描述
    _comprehensive_quality_assessment Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _comprehensive_quality_assessment(self, results):
        """综合质量评估"""
        quality_scores = {}
        
        for modality, result in results.items():
            # 基于多个因素评估质量
            score = 0.5  # 基础分数
            
            # 置信度评估
            if isinstance(result, dict) and 'confidence' in result:
                score = result['confidence'] * 0.6
            
            # 完整性加分
            completeness = self._assess_completeness(result)
            score += completeness * 0.2
            
            # 一致性加分
            consistency = self._assess_internal_consistency(result)
            score += consistency * 0.2
            
            quality_scores[modality] = min(max(score, 0), 1)
        
        return quality_scores
    
    
    """
    _assess_context_relevance函数 - 中文函数描述
    _assess_context_relevance Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _assess_context_relevance(self, results, context):
        """评估各模态结果与上下文的相关性"""
        relevance_scores = {}
        
        if not context:
            # 无上下文，默认相关性
            for modality in results:
                relevance_scores[modality] = 0.7
            return relevance_scores
        
        # 分析上下文类型
        context_type = context.get('type', 'general')
        
        # 根据上下文类型分配相关性分数
        context_relevance_map = {
            'visual_task': {'image': 0.9, 'video': 0.9, 'text': 0.6, 'audio': 0.4, 'sensor': 0.3},
            'audio_task': {'audio': 0.9, 'text': 0.7, 'image': 0.4, 'video': 0.4, 'sensor': 0.3},
            'text_task': {'text': 0.9, 'audio': 0.6, 'image': 0.4, 'video': 0.4, 'sensor': 0.2},
            'sensor_task': {'sensor': 0.9, 'text': 0.5, 'audio': 0.4, 'image': 0.3, 'video': 0.3},
            'multimodal_task': {'text': 0.8, 'audio': 0.7, 'image': 0.8, 'video': 0.8, 'sensor': 0.6},
            'general': {'text': 0.7, 'audio': 0.6, 'image': 0.6, 'video': 0.6, 'sensor': 0.5}
        }
        
        relevance_template = context_relevance_map.get(context_type, context_relevance_map['general'])
        
        for modality in results:
            relevance_scores[modality] = relevance_template.get(modality, 0.5)
        
        return relevance_scores
    
    
    """
    _analyze_task_requirements函数 - 中文函数描述
    _analyze_task_requirements Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _analyze_task_requirements(self, task_type):
        """分析任务需求"""
        task_requirements = {
            'precision_required': 0.7,
            'recall_required': 0.6,
            'speed_required': 0.8,
            'explainability_required': 0.5
        }
        
        # 根据任务类型调整需求
        task_specific_requirements = {
            'classification': {'precision_required': 0.9, 'recall_required': 0.8, 'speed_required': 0.7},
            'detection': {'precision_required': 0.8, 'recall_required': 0.9, 'speed_required': 0.8},
            'generation': {'precision_required': 0.6, 'recall_required': 0.5, 'explainability_required': 0.8},
            'analysis': {'precision_required': 0.9, 'recall_required': 0.7, 'explainability_required': 0.9},
            'real_time': {'speed_required': 0.9, 'precision_required': 0.6, 'recall_required': 0.5}
        }
        
        if task_type in task_specific_requirements:
            task_requirements.update(task_specific_requirements[task_type])
        
        return task_requirements
    
    
    """
    _check_temporal_consistency函数 - 中文函数描述
    _check_temporal_consistency Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _check_temporal_consistency(self, results):
        """检查时间一致性（与历史结果比较）"""
        consistency_scores = {}
        
        # 初始化历史记录
        if not hasattr(self, 'recent_results'):
            self.recent_results = {}
        
        for modality, result in results.items():
            if modality in self.recent_results:
                # 比较当前结果与历史结果
                old_result = self.recent_results[modality]
                similarity = self._calculate_result_similarity(result, old_result)
                consistency_scores[modality] = similarity
            else:
                consistency_scores[modality] = 0.7  # 无历史数据，默认一致性
            
            # 更新历史记录
            self.recent_results[modality] = result
        
        return consistency_scores
    
    
    """
    _calculate_result_similarity函数 - 中文函数描述
    _calculate_result_similarity Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _calculate_result_similarity(self, result1, result2):
        """计算两个结果的相似度"""
        if type(result1) != type(result2):
            return 0.3
        
        if isinstance(result1, dict) and isinstance(result2, dict):
            # 计算字典相似度
            common_keys = set(result1.keys()).intersection(set(result2.keys()))
            if not common_keys:
                return 0.3
            
            similarities = []
            for key in common_keys:
                if isinstance(result1[key], (int, float)) and isinstance(result2[key], (int, float)):
                    # 数值相似度
                    diff = abs(result1[key] - result2[key])
                    max_val = max(abs(result1[key]), abs(result2[key]), 1.0)
                    similarity = 1.0 - (diff / max_val)
                    similarities.append(similarity)
                elif isinstance(result1[key], str) and isinstance(result2[key], str):
                    # 文本相似度（简单实现）
                    words1 = set(result1[key].lower().split())
                    words2 = set(result2[key].lower().split())
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        similarities.append(similarity)
            
            if similarities:
                return sum(similarities) / len(similarities)
            else:
                return 0.5
        
        elif isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
            # 数值相似度
            diff = abs(result1 - result2)
            max_val = max(abs(result1), abs(result2), 1.0)
            return 1.0 - (diff / max_val)
        
        return 0.4  # 默认相似度
    
    
    """
    _select_optimal_fusion_strategy函数 - 中文函数描述
    _select_optimal_fusion_strategy Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _select_optimal_fusion_strategy(self, fusion_analysis):
        """选择最优融合策略"""
        # 基于分析结果选择策略
        quality_scores = fusion_analysis['quality_assessment']
        context_relevance = fusion_analysis['context_relevance']
        task_requirements = fusion_analysis['task_requirements']
        
        # 计算各模态的综合得分
        modality_scores = {}
        for modality in quality_scores:
            modality_scores[modality] = (
                quality_scores[modality] * 0.4 +
                context_relevance.get(modality, 0.5) * 0.3 +
                fusion_analysis['temporal_consistency'].get(modality, 0.5) * 0.3
            )
        
        # 根据任务需求选择策略
        if task_requirements['speed_required'] > 0.8:
            # 高速度需求，选择简单融合
            return self._weighted_fusion_simple
        elif task_requirements['precision_required'] > 0.8:
            # 高精度需求，选择精细融合
            return self._precision_optimized_fusion
        elif task_requirements['explainability_required'] > 0.7:
            # 高可解释性需求，选择透明融合
            return self._explainable_fusion
        else:
            # 默认选择自适应融合
            return self.adaptive_fusion
    
    
    """
    _precision_optimized_fusion函数 - 中文函数描述
    _precision_optimized_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _precision_optimized_fusion(self, results, params=None):
        """精度优化的融合策略"""
        # 实现高精度需求的融合逻辑
        fused_result = {}
        
        for modality, result in results.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in fused_result:
                        fused_result[key] = {'values': [], 'weights': []}
                    
                    if isinstance(value, (int, float)):
                        # 对于数值数据，记录值和权重
                        weight = params.get('weights', {}).get(modality, 0.5) if params else 0.5
                        fused_result[key]['values'].append(value)
                        fused_result[key]['weights'].append(weight)
        
        # 计算加权平均值
        for key, data in fused_result.items():
            if data['values'] and data['weights']:
                total_weight = sum(data['weights'])
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in zip(data['values'], data['weights']))
                    fused_result[key] = weighted_sum / total_weight
        
        return fused_result
    
    
    """
    _explainable_fusion函数 - 中文函数描述
    _explainable_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _explainable_fusion(self, results, params=None):
        """可解释性优化的融合策略"""
        fused_result = {}
        fusion_explanation = {}
        
        for modality, result in results.items():
            fusion_explanation[modality] = {
                'contribution': params.get('weights', {}).get(modality, 0.5) if params else 0.5,
                'quality_score': self._assess_completeness(result),
                'relevance_score': 0.7  # 默认相关性
            }
            
            # 融合结果
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in fused_result:
                        fused_result[key] = value
                    elif isinstance(value, str) and isinstance(fused_result[key], str):
                        # 文本融合：添加新信息
                        fused_result[key] += f" | {value}"
        
        # 添加解释信息
        fused_result['fusion_explanation'] = fusion_explanation
        fused_result['fusion_strategy'] = 'explainable_fusion'
        
        return fused_result
    
    
    """
    _adjust_fusion_parameters函数 - 中文函数描述
    _adjust_fusion_parameters Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _adjust_fusion_parameters(self, fusion_analysis):
        """动态调整融合参数"""
        params = {
            'weights': {},
            'strategy_params': {},
            'quality_threshold': 0.6,
            'consistency_threshold': 0.5
        }
        
        # 基于质量评估调整权重
        quality_scores = fusion_analysis['quality_assessment']
        total_quality = sum(quality_scores.values())
        
        if total_quality > 0:
            for modality, score in quality_scores.items():
                params['weights'][modality] = score / total_quality
        
        # 基于任务需求调整参数
        task_reqs = fusion_analysis['task_requirements']
        if task_reqs['precision_required'] > 0.8:
            params['quality_threshold'] = 0.7
            params['consistency_threshold'] = 0.6
        
        if task_reqs['speed_required'] > 0.8:
            params['quality_threshold'] = 0.5
            params['consistency_threshold'] = 0.4
        
        return params
    
    
    """
    _record_fusion_decision函数 - 中文函数描述
    _record_fusion_decision Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _record_fusion_decision(self, decision_info):
        """记录融合决策信息"""
        if not hasattr(self, 'fusion_decisions_history'):
            self.fusion_decisions_history = []
        
        self.fusion_decisions_history.append(decision_info)
        
        # 保持历史记录大小
        if len(self.fusion_decisions_history) > 100:
            self.fusion_decisions_history = self.fusion_decisions_history[-100:]
    
    
    """
    get_fusion_metrics函数 - 中文函数描述
    get_fusion_metrics Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def get_fusion_metrics(self):
        """获取融合性能指标"""
        metrics = {
            'total_fusions': len(self.fusion_decisions_history) if hasattr(self, 'fusion_decisions_history') else 0,
            'average_decision_time': 0.1,  # 模拟值
            'success_rate': 0.95,  # 模拟值
            'strategy_distribution': {}
        }
        
        # 计算策略分布
        if hasattr(self, 'fusion_decisions_history'):
            strategies = [d['strategy'] for d in self.fusion_decisions_history]
            from collections import Counter
            strategy_counts = Counter(strategies)
            total = len(strategies)
            if total > 0:
                metrics['strategy_distribution'] = {k: v/total for k, v in strategy_counts.items()}
        
        return metrics
    
    
    """
    enable_adaptive_learning函数 - 中文函数描述
    enable_adaptive_learning Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def enable_adaptive_learning(self, enabled=True):
        """启用或禁用自适应学习"""
        self.adaptive_learner['enabled'] = enabled
        error_handler.log_info(f"自适应学习 {'启用' if enabled else '禁用'}", "MultimodalFusion")
    
    
    """
    update_from_feedback函数 - 中文函数描述
    update_from_feedback Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def update_from_feedback(self, feedback_data):
        """基于反馈数据更新融合策略"""
        if not self.adaptive_learner['enabled']:
            return
        
        # 分析反馈数据
        feedback_analysis = self._analyze_feedback(feedback_data)
        
        # 更新融合权重
        self._update_weights_from_feedback(feedback_analysis)
        
        # 更新策略选择逻辑
        self._update_strategy_selection(feedback_analysis)
        
        error_handler.log_info("基于反馈更新融合策略完成", "MultimodalFusion")
    
    
    """
    _analyze_feedback函数 - 中文函数描述
    _analyze_feedback Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _analyze_feedback(self, feedback_data):
        """分析反馈数据"""
        analysis = {
            'preferred_modalities': [],
            'rejected_modalities': [],
            'quality_ratings': {},
            'suggested_improvements': []
        }
        
        # 简单反馈分析
        if 'preferred_modality' in feedback_data:
            analysis['preferred_modalities'].append(feedback_data['preferred_modality'])
        
        if 'quality_ratings' in feedback_data:
            analysis['quality_ratings'] = feedback_data['quality_ratings']
        
        return analysis
    
    
    """
    _update_weights_from_feedback函数 - 中文函数描述
    _update_weights_from_feedback Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _update_weights_from_feedback(self, feedback_analysis):
        """基于反馈更新融合权重"""
        # 增加首选模态的权重
        for modality in feedback_analysis['preferred_modalities']:
            if modality in self.fusion_weights:
                self.fusion_weights[modality] *= 1.1  # 增加10%
        
        # 根据质量评分调整权重
        for modality, rating in feedback_analysis['quality_ratings'].items():
            if modality in self.fusion_weights:
                # 质量评分越高，权重增加越多
                adjustment = 1.0 + (rating - 0.5) * 0.2  # 评分0.5不变，1.0增加10%，0.0减少10%
                self.fusion_weights[modality] *= adjustment
        
        # 归一化权重
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            for modality in self.fusion_weights:
                self.fusion_weights[modality] /= total_weight
    
    
    """
    _update_strategy_selection函数 - 中文函数描述
    _update_strategy_selection Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _update_strategy_selection(self, feedback_analysis):
        """基于反馈更新策略选择逻辑"""
        # 这里可以实现更复杂的策略选择逻辑更新
        # 目前简单记录反馈信息
        if not hasattr(self, 'feedback_learning_data'):
            self.feedback_learning_data = []
        
        self.feedback_learning_data.append({
            'feedback': feedback_analysis,
            'timestamp': time.time()
        })


"""
MultimodalFusionEngine类 - 中文类描述
MultimodalFusionEngine Class - English class description
"""
class MultimodalFusionEngine:
    """多模态融合引擎，整合来自不同模型的信息"""
    
    """
    __init__函数 - 中文函数描述
    __init__ Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def __init__(self):
        self.fusion_strategies = {
            'early': self._early_fusion,
            'late': self._late_fusion,
            'hybrid': self._hybrid_fusion
        }
        self.default_strategy = 'hybrid'
        self.context_store = {}

    
    """
    fuse函数 - 中文函数描述
    fuse Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def fuse(self, inputs, strategy=None, context_id=None):
        """融合多模态输入
        Args:
            inputs: 包含不同模态数据的字典
            strategy: 融合策略
            context_id: 上下文ID，用于持续对话
        """
        strategy = strategy or self.default_strategy
        fusion_func = self.fusion_strategies.get(strategy, self._hybrid_fusion)

        # 加载上下文（如果有）
        context = self.context_store.get(context_id, {})

        # 执行融合
        result = fusion_func(inputs, context)

        # 更新上下文
        if context_id:
            self.context_store[context_id] = self._update_context(context, inputs, result)

        return result

    
    """
    _early_fusion函数 - 中文函数描述
    _early_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _early_fusion(self, inputs, context):
        """早期融合：在特征提取阶段融合多模态数据
        Early fusion: fuse multimodal data at feature extraction stage
        """
        fused_features = {}
        feature_dimensions = {}
        
        # 收集所有模态的特征
        for modality, data in inputs.items():
            if isinstance(data, dict) and 'features' in data:
                features = data['features']
                fused_features[modality] = features
                feature_dimensions[modality] = len(features)
        
        # 对齐特征维度
        max_dim = max(feature_dimensions.values()) if feature_dimensions else 0
        aligned_features = []
        
        for modality, features in fused_features.items():
            # 对特征进行填充或截断以对齐维度
            if len(features) < max_dim:
                # 填充零值
                padded = features + [0.0] * (max_dim - len(features))
                aligned_features.append(padded)
            elif len(features) > max_dim:
                # 截断特征
                truncated = features[:max_dim]
                aligned_features.append(truncated)
            else:
                aligned_features.append(features)
        
        if not aligned_features:
            return {'fused_result': 'No features to fuse', 'fusion_strategy': 'early'}
        
        # 简单的特征拼接作为早期融合
        concatenated = [item for sublist in aligned_features for item in sublist]
        
        return {
            'fused_features': concatenated,
            'fusion_strategy': 'early',
            'feature_dimensions': feature_dimensions,
            'context_used': context
        }

    
    """
    _late_fusion函数 - 中文函数描述
    _late_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _late_fusion(self, inputs, context):
        """晚期融合：在决策阶段融合多模态数据结果
        Late fusion: fuse multimodal data results at decision stage
        """
        fused_results = {}
        confidence_scores = {}
        
        # 收集所有模态的结果和置信度
        for modality, data in inputs.items():
            if isinstance(data, dict):
                # 提取结果数据
                result_data = {k: v for k, v in data.items() if k != 'confidence'}
                
                # 提取置信度
                confidence = data.get('confidence', 0.5)
                confidence_scores[modality] = confidence
                
                # 根据置信度加权融合
                for key, value in result_data.items():
                    if key not in fused_results:
                        fused_results[key] = []
                    
                    fused_results[key].append((value, confidence))
        
        # 对每个键进行融合
        final_result = {}
        for key, weighted_values in fused_results.items():
            # 根据数据类型选择融合方式
            if all(isinstance(v[0], (int, float)) for v in weighted_values):
                # 数值类型：加权平均
                total_weight = sum(w for v, w in weighted_values)
                if total_weight > 0:
                    weighted_avg = sum(v * w for v, w in weighted_values) / total_weight
                    final_result[key] = weighted_avg
            elif all(isinstance(v[0], str) for v in weighted_values):
                # 字符串类型：根据置信度选择最可靠的
                sorted_values = sorted(weighted_values, key=lambda x: x[1], reverse=True)
                final_result[key] = sorted_values[0][0]
                # 记录所有结果作为参考
                final_result[f"{key}_all"] = [v for v, w in weighted_values]
                final_result[f"{key}_confidences"] = [w for v, w in weighted_values]
            else:
                # 其他类型：简单合并
                final_result[key] = [v for v, w in weighted_values]
        
        # 添加融合元数据
        final_result.update({
            'fusion_strategy': 'late',
            'confidence_scores': confidence_scores,
            'context_used': context
        })
        
        return final_result

    
    """
    _hybrid_fusion函数 - 中文函数描述
    _hybrid_fusion Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _hybrid_fusion(self, inputs, context):
        """混合融合：结合早期融合和晚期融合的优点
        Hybrid fusion: combine the advantages of both early and late fusion
        """
        # 分离特征数据和结果数据
        feature_data = {}
        result_data = {}
        
        for modality, data in inputs.items():
            if isinstance(data, dict):
                # 检查是否包含特征数据（用于早期融合）
                if 'features' in data:
                    feature_data[modality] = data
                # 检查是否包含结果数据（用于晚期融合）
                elif any(k not in ['features', 'confidence'] for k in data.keys()):
                    result_data[modality] = data
            else:
                result_data[modality] = data
        
        # 执行早期融合（如果有特征数据）
        early_fusion_result = {}
        if feature_data:
            early_fusion_result = self._early_fusion(feature_data, context)
        
        # 执行晚期融合（如果有结果数据）
        late_fusion_result = {}
        if result_data:
            late_fusion_result = self._late_fusion(result_data, context)
        
        # 合并两种融合结果
        hybrid_result = {**early_fusion_result, **late_fusion_result}
        
        # 更新融合策略标识
        hybrid_result['fusion_strategy'] = 'hybrid'
        
        # 如果两种融合都执行了，添加融合策略说明
        if feature_data and result_data:
            hybrid_result['fusion_approach'] = {
                'early_fusion_applied': True,
                'late_fusion_applied': True,
                'early_fusion_modalities': list(feature_data.keys()),
                'late_fusion_modalities': list(result_data.keys())
            }
        elif feature_data:
            hybrid_result['fusion_approach'] = {
                'early_fusion_applied': True,
                'late_fusion_applied': False,
                'early_fusion_modalities': list(feature_data.keys())
            }
        elif result_data:
            hybrid_result['fusion_approach'] = {
                'early_fusion_applied': False,
                'late_fusion_applied': True,
                'late_fusion_modalities': list(result_data.keys())
            }
        
        return hybrid_result

    
    """
    _update_context函数 - 中文函数描述
    _update_context Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def _update_context(self, context, inputs, result):
        """更新上下文信息，用于持续对话
        Update context information for continuous dialogue
        """
        updated_context = context.copy()
        
        # 添加当前输入的模态信息
        updated_context['last_modalities'] = list(inputs.keys())
        updated_context['last_update_time'] = time.time()
        
        # 保存关键融合结果
        if isinstance(result, dict):
            # 提取关键信息
            key_info = {}
            
            # 保存融合策略
            if 'fusion_strategy' in result:
                key_info['last_strategy'] = result['fusion_strategy']
            
            # 保存置信度信息
            if 'confidence_scores' in result:
                key_info['last_confidences'] = result['confidence_scores']
            
            # 保存主要结果数据
            main_results = {k: v for k, v in result.items() 
                          if k not in ['fusion_strategy', 'confidence_scores', 'context_used', 
                                      'fusion_approach', 'feature_dimensions', 'fused_features']}
            
            if main_results:
                key_info['last_results'] = main_results
            
            # 更新上下文历史
            if 'history' not in updated_context:
                updated_context['history'] = []
            
            updated_context['history'].append(key_info)
            
            # 限制历史记录长度
            max_history = 10
            if len(updated_context['history']) > max_history:
                updated_context['history'] = updated_context['history'][-max_history:]
        
        return updated_context


"""
AdvancedMultimodalFusion类 - 中文类描述
AdvancedMultimodalFusion Class - English class description
"""
class AdvancedMultimodalFusion(MultimodalFusion):
    """高级多模态融合类，用于管理模型"""
    
    
    """
    fuse函数 - 中文函数描述
    fuse Function - English function description

    Args:
        params: 参数描述 (Parameter description)
        
    Returns:
        返回值描述 (Return value description)
    """
    def fuse(self, results, emotion=None):
        """融合结果，考虑情感状态"""
        context = {'emotion': emotion} if emotion else {}
        return self.adaptive_fusion(results, context)
