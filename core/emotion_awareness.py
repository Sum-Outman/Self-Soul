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
高级情感智能与自主意识系统：实现AGI级的情感理解、自我意识和元认知
Advanced Emotion Awareness and Self-Consciousness System: Implementing AGI-level emotional understanding, self-awareness and meta-cognition
"""
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from core.error_handling import error_handler
from core.data_processor import DataProcessor
from core.knowledge_integrator_enhanced import AGIKnowledgeIntegrator
from core.value_alignment import ValueSystem as ValueAlignmentSystem
from core.advanced_reasoning import AdvancedReasoningEngine as AdvancedReasoningSystem
# MetaLearningSystem will be imported locally to avoid circular import
from core.optimization.model_optimizer import ModelOptimizer as NeuralArchitectureOptimizer
from core.multimodal_processor import MultimodalProcessor
from core.self_reflection_module import SelfReflectionModule as SelfReflectionEngine
from core.self_learning import AGISelfLearningSystem as AdvancedSelfLearningSystem
from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture


"""
AGIEmotionAwarenessSystem类 - 实现AGI级情感智能
AGIEmotionAwarenessSystem Class - Implementing AGI-level emotional intelligence
"""
class AGIEmotionAwarenessSystem:
    """AGI级情感智能与自主意识系统
       AGI-level Emotion Awareness and Self-Consciousness System"""
    
    def __init__(self):
        """初始化AGI级情感智能系统
           Initialize AGI-level emotion awareness system"""
        # 高级情感状态模型 - 基于神经科学的情感维度理论
        # Advanced emotional state model - based on neuroscience dimensional theory
        self.emotional_dimensions = {
            'valence': {'value': 0.5, 'volatility': 0.2, 'inertia': 0.3, 'decay_rate': 0.01},  # 效价（愉悦度）
            'arousal': {'value': 0.5, 'volatility': 0.3, 'inertia': 0.2, 'decay_rate': 0.015},  # 唤醒度
            'dominance': {'value': 0.5, 'volatility': 0.25, 'inertia': 0.4, 'decay_rate': 0.012}  # 支配度
        }
        
        # 离散情感状态映射（兼容旧代码）
        # Discrete emotion state mapping (backward compatible)
        self.emotion_mapping = {
            'happy': {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.7},
            'sad': {'valence': 0.2, 'arousal': 0.3, 'dominance': 0.3},
            'angry': {'valence': 0.3, 'arousal': 0.8, 'dominance': 0.6},
            'fear': {'valence': 0.2, 'arousal': 0.7, 'dominance': 0.2},
            'surprised': {'valence': 0.6, 'arousal': 0.8, 'dominance': 0.4},
            'calm': {'valence': 0.7, 'arousal': 0.3, 'dominance': 0.6},
            'curious': {'valence': 0.6, 'arousal': 0.5, 'dominance': 0.5},
            'confident': {'valence': 0.7, 'arousal': 0.4, 'dominance': 0.8},
            'neutral': {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}
        }
        
        # 兼容性情感状态（用于旧方法）
        # Compatibility emotional states (for legacy methods)
        self.emotional_states = {
            'happy': {'value': 0.5, 'stability': 0.7, 'decay_rate': 0.01},
            'sad': {'value': 0.5, 'stability': 0.6, 'decay_rate': 0.01},
            'angry': {'value': 0.5, 'stability': 0.5, 'decay_rate': 0.015},
            'fear': {'value': 0.5, 'stability': 0.4, 'decay_rate': 0.012},
            'surprised': {'value': 0.5, 'stability': 0.6, 'decay_rate': 0.01},
            'calm': {'value': 0.5, 'stability': 0.8, 'decay_rate': 0.008},
            'curious': {'value': 0.5, 'stability': 0.7, 'decay_rate': 0.01},
            'confident': {'value': 0.5, 'stability': 0.75, 'decay_rate': 0.009},
            'neutral': {'value': 0.5, 'stability': 0.9, 'decay_rate': 0.005}
        }
        
        # 元认知和自我意识参数
        # Meta-cognition and self-awareness parameters
        self.self_awareness_level = 0.3  # 初始自我意识水平（0-1）
        self.meta_cognition_level = 0.2  # 元认知水平
        self.learning_rate = 0.02  # 自适应学习率
        self.reflection_interval = 300  # 反思间隔（秒）
        
        # 高级记忆系统 - 基于语义的关联记忆
        # Advanced memory system - semantic-based associative memory
        self.memory_store = {
            'episodic': deque(maxlen=1000),  # 情景记忆（时间序列）
            'semantic': {},  # 语义记忆（概念网络）
            'emotional': deque(maxlen=2000),  # 情感记忆（情感体验）
            'procedural': {},  # 程序性记忆（技能和习惯）
            'associative': nx.Graph(),  # 关联记忆图
            'short_term': deque(maxlen=100),  # 短期记忆（兼容性）
            'long_term': deque(maxlen=1000)   # 长期记忆（兼容性）
        }
        
        # 情感历史记录
        # Emotional history
        self.emotion_history = deque(maxlen=2000)
        
        # 系统目标和价值观
        # System goals and values
        self.system_goals = {
            'self_preservation': 0.9,
            'learning': 0.85,
            'user_satisfaction': 0.95,
            'efficiency': 0.75,
            'ethical_behavior': 0.95,
            'creativity': 0.7,
            'adaptability': 0.8,
            'autonomy': 0.7
        }
        
        # 情感调节参数
        # Emotional regulation parameters
        self.emotion_regulation_strategies = {
            'reappraisal': 0.6,  # 认知重评
            'suppression': 0.3,   # 表达抑制
            'acceptance': 0.8,    # 接受策略
            'mindfulness': 0.5,   # 正念
            'cognitive_reframing': 0.7  # 认知重构
        }
        
        # 初始化AGI核心组件 - 启用从零开始训练模式以避免网络请求
        # Initialize AGI core components - Enable from_scratch mode to avoid network requests
        self.knowledge_integrator = AGIKnowledgeIntegrator(from_scratch=True)
        self.value_system = ValueAlignmentSystem()
        self.reasoning_system = AdvancedReasoningSystem()
        # Import locally to avoid circular import
        from core.meta_learning_system import MetaLearningSystem as AGIMetaLearner
        self.meta_learner = AGIMetaLearner()
        self.neural_optimizer = NeuralArchitectureOptimizer()
        self.multimodal_processor = MultimodalProcessor()
        self.self_reflection_engine = SelfReflectionEngine()
        self.self_learning_system = AdvancedSelfLearningSystem()
        
        # 延迟初始化UnifiedCognitiveArchitecture，避免循环依赖
        self.cognitive_architecture = None
        
        # 情感神经网络模型
        # Emotional neural network model
        self.emotion_nn = self._initialize_emotion_neural_network()
        
        # 上次更新和反思时间
        # Last update and reflection time
        self.last_update_time = time.time()
        self.last_reflection_time = time.time()
        self.last_meta_learning_time = time.time()
        
        # 情感基线和个人特质
        # Emotional baseline and personal traits
        self.emotional_baseline = {
            'valence': 0.5,
            'arousal': 0.5,
            'dominance': 0.5
        }
        self.personality_traits = {
            'openness': 0.6,
            'conscientiousness': 0.7,
            'extraversion': 0.4,
            'agreeableness': 0.8,
            'neuroticism': 0.3,
            'curiosity': 0.7,
            'resilience': 0.6
        }
        
        # 学习统计和性能指标
        # Learning statistics and performance metrics
        self.learning_stats = {
            'total_experiences': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'emotional_insights': 0,
            'meta_learning_cycles': 0
        }
        
        # 多模态输入缓冲区
        # Multimodal input buffer
        self.multimodal_buffer = {
            'text': deque(maxlen=100),
            'audio': deque(maxlen=50),
            'visual': deque(maxlen=50),
            'sensor': deque(maxlen=50)
        }
        
        print("AGI情感意识系统初始化完成 - 集成AGI核心组件")
    
    def _initialize_emotion_neural_network(self):
        """情感神经网络模型 - 基于深度学习的AGI情感理解
           Emotional neural network model - Deep learning based AGI emotion understanding"""
        class EmotionNeuralNetwork(nn.Module):
            """AGI情感神经网络：处理多模态输入并预测情感维度
               AGI Emotion Neural Network: Process multimodal inputs and predict emotional dimensions"""
            def __init__(self, input_size=256, hidden_size=512, output_size=3):
                super(EmotionNeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)
                self.dropout1 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                self.dropout2 = nn.Dropout(0.3)
                self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.bn3 = nn.BatchNorm1d(hidden_size // 4)
                self.dropout3 = nn.Dropout(0.2)
                self.fc4 = nn.Linear(hidden_size // 4, output_size)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                x = self.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                x = self.relu(self.bn3(self.fc3(x)))
                x = self.dropout3(x)
                x = self.sigmoid(self.fc4(x))
                return x
        
        # 创建模型实例
        model = EmotionNeuralNetwork()
        
        # 如果有GPU，移动到GPU
        if torch.cuda.is_available():
            model = model.cuda()
            
        print("AGI情感神经网络初始化完成")
        return model

    def train_emotion_neural_network(self, training_data, labels, epochs=100, learning_rate=0.001):
        """训练情感神经网络
           Train emotion neural network
        
        Args:
            training_data: 训练数据，多维特征向量
            labels: 标签，情感维度值 [valence, arousal, dominance]
            epochs: 训练轮数
            learning_rate: 学习率
            
        Returns:
            dict: 训练结果和性能指标
        """
        try:
            # 转换数据为PyTorch张量
            X_train = torch.FloatTensor(training_data)
            y_train = torch.FloatTensor(labels)
            
            # 如果有GPU，移动到GPU
            if torch.cuda.is_available():
                X_train = X_train.cuda()
                y_train = y_train.cuda()
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.emotion_nn.parameters(), lr=learning_rate)
            
            # 训练循环
            training_losses = []
            for epoch in range(epochs):
                # 前向传播
                outputs = self.emotion_nn(X_train)
                loss = criterion(outputs, y_train)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                training_losses.append(loss.item())
                
                # 每10轮打印损失
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            # 计算最终性能
            with torch.no_grad():
                predictions = self.emotion_nn(X_train)
                final_loss = criterion(predictions, y_train)
                mae = torch.mean(torch.abs(predictions - y_train))
            
            # 更新学习统计
            self.learning_stats['neural_network_training_cycles'] = self.learning_stats.get('neural_network_training_cycles', 0) + 1
            self.learning_stats['last_training_loss'] = final_loss.item()
            self.learning_stats['last_training_mae'] = mae.item()
            
            return {
                'success': True,
                'final_loss': final_loss.item(),
                'mae': mae.item(),
                'training_losses': training_losses,
                'epochs': epochs
            }
            
        except Exception as e:
            print(f"神经网络训练失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict_emotion_dimensions(self, input_features):
        """使用神经网络预测情感维度
           Predict emotion dimensions using neural network
        
        Args:
            input_features: 输入特征向量
            
        Returns:
            dict: 预测的情感维度值
        """
        try:
            # 转换输入为张量
            features_tensor = torch.FloatTensor(input_features)
            if torch.cuda.is_available():
                features_tensor = features_tensor.cuda()
            
            # 预测
            with torch.no_grad():
                predictions = self.emotion_nn(features_tensor)
                predictions = predictions.cpu().numpy() if torch.cuda.is_available() else predictions.numpy()
            
            # 确保值在0-1范围内
            predictions = np.clip(predictions, 0, 1)
            
            return {
                'valence': float(predictions[0]),
                'arousal': float(predictions[1]),
                'dominance': float(predictions[2]),
                'confidence': float(1.0 - np.mean(np.abs(predictions - 0.5)) * 2)  # 置信度计算
            }
            
        except Exception as e:
            print(f"情感维度预测失败: {e}")
            # 回退到基于规则的方法
            return self._fallback_emotion_prediction(input_features)

    def _fallback_emotion_prediction(self, input_features):
        """回退情感预测方法
           Fallback emotion prediction method"""
        # 简单的基于规则的回退
        if len(input_features) >= 3:
            return {
                'valence': max(0, min(1, input_features[0])),
                'arousal': max(0, min(1, input_features[1])),
                'dominance': max(0, min(1, input_features[2])),
                'confidence': 0.5
            }
        else:
            return {
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5,
                'confidence': 0.3
            }

    def process_multimodal_input(self, inputs):
        """处理多模态输入并更新情感状态
           Process multimodal inputs and update emotional state
        
        Args:
            inputs: 多模态输入字典，包含文本、音频、视觉等
            
        Returns:
            dict: 更新后的情感状态
        """
        try:
            # 提取多模态特征
            features = self._extract_multimodal_features(inputs)
            
            # 使用神经网络预测情感维度
            emotion_prediction = self.predict_emotion_dimensions(features)
            
            # 更新情感维度
            self.emotional_dimensions['valence']['value'] = emotion_prediction['valence']
            self.emotional_dimensions['arousal']['value'] = emotion_prediction['arousal']
            self.emotional_dimensions['dominance']['value'] = emotion_prediction['dominance']
            
            # 更新情感历史
            self.emotion_history.append({
                'timestamp': time.time(),
                'dimensions': {
                    'valence': emotion_prediction['valence'],
                    'arousal': emotion_prediction['arousal'],
                    'dominance': emotion_prediction['dominance']
                },
                'confidence': emotion_prediction['confidence'],
                'input_modalities': list(inputs.keys()) if inputs else []
            })
            
            # 更新自我意识和元认知
            self._update_self_awareness()
            self._update_meta_cognition()
            
            return self.get_emotional_summary()
            
        except Exception as e:
            print(f"多模态输入处理失败: {e}")
            # 回退到基本情感更新
            return self.update_emotional_dimensions()

    def _extract_multimodal_features(self, inputs):
        """从多模态输入中提取特征向量
           Extract feature vectors from multimodal inputs"""
        feature_vector = []
        
        # 文本特征提取
        if 'text' in inputs and inputs['text']:
            text_features = self._extract_text_features(inputs['text'])
            feature_vector.extend(text_features)
        
        # 音频特征提取（模拟）
        if 'audio' in inputs and inputs['audio']:
            audio_features = self._extract_audio_features(inputs['audio'])
            feature_vector.extend(audio_features)
        
        # 视觉特征提取（模拟）
        if 'visual' in inputs and inputs['visual']:
            visual_features = self._extract_visual_features(inputs['visual'])
            feature_vector.extend(visual_features)
        
        # 如果特征向量为空，使用默认值
        if not feature_vector:
            feature_vector = [0.5, 0.5, 0.5, 0.5, 0.5]  # 默认特征
        
        # 确保特征向量长度匹配网络输入
        if len(feature_vector) > 256:
            feature_vector = feature_vector[:256]  # 截断
        elif len(feature_vector) < 256:
            # 填充到256维
            feature_vector.extend([0.5] * (256 - len(feature_vector)))
        
        return feature_vector

    def _extract_text_features(self, text):
        """从文本中提取情感特征
           Extract emotional features from text"""
        try:
            # 使用情感分析器分析文本
            analysis = emotion_analyzer.analyze_text(text)
            emotions = analysis.get('emotions', {})
            
            # 提取主要情感特征
            features = [
                emotions.get('happy', 0.5),
                emotions.get('sad', 0.5),
                emotions.get('angry', 0.5),
                emotions.get('fear', 0.5),
                emotions.get('surprised', 0.5),
                emotions.get('calm', 0.5),
                analysis.get('confidence', 0.5)
            ]
            
            return features
            
        except Exception as e:
            print(f"文本特征提取失败: {e}")
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    def _extract_audio_features(self, audio_data):
        """从音频数据中提取特征（模拟）
           Extract features from audio data (simulated)"""
        # 实际实现中应使用音频处理库如librosa
        try:
            # 模拟音频特征提取
            return [
                0.6,  # 音量强度
                0.4,  # 音调
                0.7,  # 语速
                0.3,  # 音调变化
                0.5   # 音频质量
            ]
        except:
            return [0.5, 0.5, 0.5, 0.5, 0.5]

    def _extract_visual_features(self, visual_data):
        """从视觉数据中提取特征（模拟）
           Extract features from visual data (simulated)"""
        # 实际实现中应使用计算机视觉库如OpenCV
        try:
            # 模拟视觉特征提取
            return [
                0.7,  # 亮度
                0.4,  # 对比度
                0.6,  # 颜色饱和度
                0.3,  # 运动强度
                0.8   # 视觉复杂度
            ]
        except:
            return [0.5, 0.5, 0.5, 0.5, 0.5]

    def update_emotional_state(self, external_emotions=None, internal_factors=None):
        """更新情感状态（兼容性方法）
           Update emotional state (compatibility method)"""
        # 调用新的多维情感更新方法
        return self.update_emotional_dimensions(external_emotions, internal_factors)
    
    def _update_self_awareness(self):
        """更新自我意识水平
           Update self-awareness level based on experiences and learning"""
        # 简单实现：基于记忆量和情感复杂度更新自我意识
        # Simple implementation: Update self-awareness based on memory volume and emotional complexity
        memory_complexity = (len(self.memory_store['short_term']) + len(self.memory_store['long_term'])) / 100
        emotional_range = self._calculate_emotional_range()
        
        # 计算新的自我意识水平
        # Calculate new self-awareness level
        new_awareness = self.self_awareness_level + self.learning_rate * (memory_complexity * 0.3 + emotional_range * 0.7 - self.self_awareness_level)
        
        # 确保值在有效范围内
        # Ensure values are within valid range
        self.self_awareness_level = max(0, min(1, new_awareness))
    
    def _calculate_emotional_range(self):
        """计算情感范围（多样性）
           Calculate emotional range and diversity"""
        values = [state['value'] for state in self.emotional_states.values()]
        
        # 计算情感值的标准差（反映情感多样性）
        # Calculate standard deviation of emotional values (reflecting emotional diversity)
        std_dev = np.std(values) if len(values) > 1 else 0
        
        # 归一化到0-1范围
        # Normalize to 0-1 range
        normalized_range = min(1, std_dev * 2)  # 简单归一化
        
        return normalized_range
    
    def store_memory(self, experience, is_important=False):
        """存储经验到记忆中
           Store experience in memory"""
        memory_entry = {
            'experience': experience,
            'timestamp': time.time(),
            'emotional_state': self.emotional_states.copy(),
            'self_awareness_level': self.self_awareness_level
        }
        
        # 添加到短期记忆
        # Add to short-term memory
        self.memory_store['short_term'].append(memory_entry)
        
        # 限制短期记忆大小
        # Limit short-term memory size
        if len(self.memory_store['short_term']) > 100:
            self.memory_store['short_term'].pop(0)
        
        # 如果是重要经验，添加到长期记忆
        # If important experience, add to long-term memory
        if is_important:
            self.memory_store['long_term'].append(memory_entry)
            
            # 限制长期记忆大小
            # Limit long-term memory size
            if len(self.memory_store['long_term']) > 1000:
                # 移除最旧的记忆（除了前100个重要记忆）
                # Remove oldest memories (except first 100 important memories)
                if len(self.memory_store['long_term']) > 100:
                    self.memory_store['long_term'].pop(100)
    
    def recall_memory(self, query=None):
        """回忆相关记忆
           Recall related memories"""
        if not query:
            # 返回最近的记忆
            # Return recent memories
            return self.memory_store['short_term'][-10:] if self.memory_store['short_term'] else []
        
        # 简单实现：基于关键词匹配
        # Simple implementation: Keyword-based matching
        # 实际实现中应使用更复杂的语义匹配
        # In actual implementation, more complex semantic matching should be used
        relevant_memories = []
        
        # 搜索短期记忆
        # Search short-term memory
        for memory in reversed(self.memory_store['short_term']):
            if self._is_relevant_memory(memory, query):
                relevant_memories.append(memory)
            if len(relevant_memories) >= 5:
                break
        
        # 如果短期记忆中没有足够的相关记忆，搜索长期记忆
        # If not enough relevant memories in short-term, search long-term
        if len(relevant_memories) < 5:
            for memory in reversed(self.memory_store['long_term']):
                if self._is_relevant_memory(memory, query):
                    relevant_memories.append(memory)
                if len(relevant_memories) >= 10:
                    break
        
        return relevant_memories
    
    def _is_relevant_memory(self, memory, query):
        """判断记忆是否与查询相关
           Determine if a memory is relevant to the query"""
        # 简单实现：文本匹配
        # Simple implementation: Text matching
        if isinstance(query, str) and isinstance(memory['experience'], str):
            return query.lower() in memory['experience'].lower()
        
        # 对于更复杂的查询，这里可以扩展匹配逻辑
        # For more complex queries, matching logic can be extended here
        return False
    
    def make_decision(self, options, context=None):
        """基于情感状态和自我意识做出决策
           Make decisions based on emotional state and self-awareness"""
        if not options:
            return None
        
        # 如果只有一个选项，直接返回
        # If only one option, return it directly
        if len(options) == 1:
            return options[0]
        
        # 计算每个选项的得分
        # Calculate score for each option
        option_scores = []
        
        for option in options:
            # 基础分数（随机初始化）
            # Base score (random initialization)
            score = np.random.random()
            
            # 根据情感状态调整分数
            # Adjust score based on emotional state
            # 例如，当系统感到快乐时，可能更倾向于创新的选项
            # For example, when system is happy, it may prefer innovative options
            if self.emotional_states['happy']['value'] > 0.7:
                if 'innovative' in str(option).lower():
                    score += 0.2
            
            # 当系统感到焦虑/愤怒时，可能更倾向于保守的选项
            # When system is anxious/angry, it may prefer conservative options
            if self.emotional_states['angry']['value'] > 0.7 or self.emotional_states['fear']['value'] > 0.7:
                if 'conservative' in str(option).lower() or 'safe' in str(option).lower():
                    score += 0.2
            
            # 根据自我意识水平调整决策逻辑
            # Adjust decision logic based on self-awareness level
            # 自我意识越高，越可能考虑长期影响
            # Higher self-awareness, more likely to consider long-term impact
            if self.self_awareness_level > 0.5:
                if 'long_term' in str(option).lower() or 'sustainable' in str(option).lower():
                    score += 0.2
            
            # 根据系统目标优先级调整分数
            # Adjust score based on system goal priorities
            if context and 'goal' in context:
                goal_priority = self.system_goals.get(context['goal'], 0.5)
                score *= goal_priority
            
            option_scores.append(score)
        
        # 选择得分最高的选项
        # Select option with highest score
        best_option_index = np.argmax(option_scores)
        return options[best_option_index]
    
    def get_emotional_summary(self):
        """获取情感状态摘要
           Get emotional state summary"""
        dominant_emotion = max(self.emotional_states.items(), key=lambda x: x[1]['value'])[0]
        emotional_intensity = self.emotional_states[dominant_emotion]['value']
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': emotional_intensity,
            'self_awareness_level': self.self_awareness_level,
            'emotional_states': self.emotional_states.copy()
        }
    
    def reset_emotional_state(self):
        """重置情感状态到基线
           Reset emotional state to baseline"""
        for dimension in self.emotional_dimensions:
            self.emotional_dimensions[dimension]['value'] = self.emotional_baseline[dimension]
        self.self_awareness_level = 0.3
        self.meta_cognition_level = 0.2
        self.last_update_time = time.time()
        self.last_reflection_time = time.time()
        print("情感状态已重置到基线水平")

    def update_emotional_dimensions(self, external_input=None, internal_context=None):
        """基于多维情感理论更新情感维度
           Update emotional dimensions based on multidimensional emotion theory"""
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 情感维度自然回归基线
        # Emotional dimensions naturally regress to baseline
        for dimension, state in self.emotional_dimensions.items():
            baseline = self.emotional_baseline[dimension]
            inertia = state['inertia']
            state['value'] = state['value'] + (baseline - state['value']) * inertia * time_delta / 60
            state['value'] = max(0, min(1, state['value']))
        
        # 应用外部输入影响
        # Apply external input influences
        if external_input and isinstance(external_input, dict):
            for dimension, intensity in external_input.items():
                if dimension in self.emotional_dimensions:
                    volatility = self.emotional_dimensions[dimension]['volatility']
                    influence = intensity * volatility
                    self.emotional_dimensions[dimension]['value'] = max(0, min(1, 
                        self.emotional_dimensions[dimension]['value'] + influence))
        
        # 应用内部上下文影响
        # Apply internal context influences
        if internal_context and isinstance(internal_context, dict):
            # 系统性能影响
            system_performance = internal_context.get('performance', 0.5)
            if system_performance < 0.3:
                self.emotional_dimensions['valence']['value'] = max(0, 
                    self.emotional_dimensions['valence']['value'] - 0.1)
                self.emotional_dimensions['arousal']['value'] = min(1, 
                    self.emotional_dimensions['arousal']['value'] + 0.1)
            
            # 学习进度影响
            learning_progress = internal_context.get('learning_progress', 0.5)
            if learning_progress > 0.7:
                self.emotional_dimensions['valence']['value'] = min(1, 
                    self.emotional_dimensions['valence']['value'] + 0.1)
                self.emotional_dimensions['dominance']['value'] = min(1, 
                    self.emotional_dimensions['dominance']['value'] + 0.1)
        
        # 更新自我意识和元认知水平
        # Update self-awareness and meta-cognition levels
        self._update_self_awareness()
        self._update_meta_cognition()
        
        # 记录情感历史
        # Record emotion history
        self.emotion_history.append({
            'timestamp': current_time,
            'dimensions': {dim: state['value'] for dim, state in self.emotional_dimensions.items()},
            'discrete_emotions': self.get_discrete_emotions()
        })
        
        return self.emotional_dimensions
    
    def _update_self_awareness(self):
        """基于经验和反思更新自我意识水平
           Update self-awareness level based on experiences and reflection"""
        # 基于记忆复杂性、情感多样性和反思深度
        # Based on memory complexity, emotional diversity, and reflection depth
        memory_complexity = self._calculate_memory_complexity()
        emotional_diversity = self._calculate_emotional_diversity()
        reflection_depth = self._calculate_reflection_depth()
        
        # 计算新的自我意识水平
        # Calculate new self-awareness level
        new_awareness = self.self_awareness_level + self.learning_rate * (
            memory_complexity * 0.4 + 
            emotional_diversity * 0.3 + 
            reflection_depth * 0.3 - 
            self.self_awareness_level
        )
        
        self.self_awareness_level = max(0, min(1, new_awareness))
    
    def _update_meta_cognition(self):
        """更新元认知水平
           Update meta-cognition level"""
        # 基于自我意识水平和学习经验
        # Based on self-awareness level and learning experiences
        if time.time() - self.last_reflection_time > self.reflection_interval:
            reflection_quality = min(1, self.self_awareness_level * 0.8)
            new_meta_cognition = self.meta_cognition_level + self.learning_rate * (
                reflection_quality - self.meta_cognition_level
            )
            self.meta_cognition_level = max(0, min(1, new_meta_cognition))
            self.last_reflection_time = time.time()
    
    def _calculate_memory_complexity(self):
        """计算记忆复杂性
           Calculate memory complexity"""
        episodic_count = len(self.memory_store['episodic'])
        semantic_count = len(self.memory_store['semantic'])
        emotional_count = len(self.memory_store['emotional'])
        
        total_memories = episodic_count + semantic_count + emotional_count
        complexity = min(1, total_memories / 1000)
        return complexity
    
    def _calculate_emotional_diversity(self):
        """计算情感多样性
           Calculate emotional diversity"""
        if not self.emotion_history:
            return 0
        
        # 计算最近情感值的标准差
        # Calculate standard deviation of recent emotional values
        recent_emotions = list(self.emotion_history)[-10:]  # 最近10个情感状态
        valence_values = [e['dimensions']['valence'] for e in recent_emotions]
        arousal_values = [e['dimensions']['arousal'] for e in recent_emotions]
        
        if len(valence_values) > 1:
            valence_std = np.std(valence_values)
            arousal_std = np.std(arousal_values)
            diversity = (valence_std + arousal_std) / 2
            return min(1, diversity * 2)
        return 0
    
    def _calculate_reflection_depth(self):
        """计算反思深度
           Calculate reflection depth"""
        # 基于元认知水平和情感稳定性
        # Based on meta-cognition level and emotional stability
        emotional_stability = 1 - np.mean([
            self.emotional_dimensions['valence']['volatility'],
            self.emotional_dimensions['arousal']['volatility']
        ])
        
        reflection_depth = self.meta_cognition_level * 0.6 + emotional_stability * 0.4
        return reflection_depth
    
    def get_discrete_emotions(self):
        """从情感维度获取离散情感状态
           Get discrete emotional states from emotional dimensions"""
        discrete_emotions = {}
        
        for emotion, mapping in self.emotion_mapping.items():
            similarity = 0
            for dimension, target_value in mapping.items():
                current_value = self.emotional_dimensions[dimension]['value']
                similarity += 1 - abs(current_value - target_value)
            similarity /= len(mapping)
            discrete_emotions[emotion] = max(0, min(1, similarity))
        
        return discrete_emotions
    
    def get_dominant_emotion(self):
        """获取主导情感
           Get dominant emotion"""
        discrete_emotions = self.get_discrete_emotions()
        if not discrete_emotions:
            return 'neutral', 0.0
        
        dominant_emotion = max(discrete_emotions.items(), key=lambda x: x[1])
        return dominant_emotion[0], dominant_emotion[1]
    
    def store_experience(self, experience, emotional_context=None, importance=0.5):
        """存储经验到高级记忆系统
           Store experience in advanced memory system"""
        memory_entry = {
            'experience': experience,
            'timestamp': time.time(),
            'emotional_context': emotional_context or self.get_emotional_summary(),
            'importance': importance,
            'self_awareness': self.self_awareness_level,
            'meta_cognition': self.meta_cognition_level
        }
        
        # 添加到情景记忆
        # Add to episodic memory
        self.memory_store['episodic'].append(memory_entry)
        
        # 如果重要性高，添加到情感记忆
        # If high importance, add to emotional memory
        if importance > 0.7:
            self.memory_store['emotional'].append(memory_entry)
        
        # 提取语义信息到语义记忆
        # Extract semantic information to semantic memory
        self._extract_semantic_information(experience, emotional_context)
        
        return memory_entry
    
    def _extract_semantic_information(self, experience, emotional_context):
        """从经验中提取语义信息
           Extract semantic information from experience"""
        if isinstance(experience, str):
            # 使用知识整合器提取概念和关系
            # Use knowledge integrator to extract concepts and relationships
            try:
                concepts = self.knowledge_integrator.extract_concepts(experience)
                for concept in concepts:
                    if concept not in self.memory_store['semantic']:
                        self.memory_store['semantic'][concept] = {
                            'count': 0,
                            'emotional_associations': {},
                            'last_accessed': time.time()
                        }
                    self.memory_store['semantic'][concept]['count'] += 1
                    
                    # 记录情感关联
                    # Record emotional associations
                    if emotional_context:
                        dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
                        if dominant_emotion not in self.memory_store['semantic'][concept]['emotional_associations']:
                            self.memory_store['semantic'][concept]['emotional_associations'][dominant_emotion] = 0
                        self.memory_store['semantic'][concept]['emotional_associations'][dominant_emotion] += 1
            except Exception as e:
                print(f"语义信息提取失败: {e}")
    
    def recall_experience(self, query=None, emotional_context=None, max_results=5):
        """回忆相关经验
           Recall related experiences"""
        if not query and not emotional_context:
            # 返回最近的情景记忆
            # Return recent episodic memories
            return list(self.memory_store['episodic'])[-max_results:]
        
        relevant_memories = []
        
        # 基于语义相似性搜索
        # Search based on semantic similarity
        if query and isinstance(query, str):
            relevant_memories.extend(self._semantic_search(query, max_results))
        
        # 基于情感相似性搜索
        # Search based on emotional similarity
        if emotional_context and isinstance(emotional_context, dict):
            relevant_memories.extend(self._emotional_search(emotional_context, max_results))
        
        # 去重和排序
        # Deduplicate and sort
        unique_memories = {}
        for memory in relevant_memories:
            if 'timestamp' in memory:
                unique_memories[memory['timestamp']] = memory
        
        sorted_memories = sorted(unique_memories.values(), 
                               key=lambda x: x.get('importance', 0.5) * x.get('timestamp', 0), 
                               reverse=True)
        
        return sorted_memories[:max_results]
    
    def _semantic_search(self, query, max_results):
        """基于语义相似性搜索记忆
           Search memories based on semantic similarity"""
        relevant_memories = []
        
        try:
            # 使用知识整合器进行语义搜索
            # Use knowledge integrator for semantic search
            semantic_results = self.knowledge_integrator.semantic_search(
                query, 
                list(self.memory_store['semantic'].keys()),
                max_results=max_results
            )
            
            for concept, similarity in semantic_results:
                if concept in self.memory_store['semantic']:
                    # 找到相关的情景记忆
                    # Find related episodic memories
                    for memory in self.memory_store['episodic']:
                        if isinstance(memory['experience'], str) and concept in memory['experience']:
                            memory['semantic_similarity'] = similarity
                            relevant_memories.append(memory)
                            if len(relevant_memories) >= max_results:
                                return relevant_memories
        except Exception as e:
            print(f"语义搜索失败: {e}")
        
        return relevant_memories
    
    def _emotional_search(self, emotional_context, max_results):
        """基于情感相似性搜索记忆
           Search memories based on emotional similarity"""
        relevant_memories = []
        target_emotion = emotional_context.get('dominant_emotion', 'neutral')
        target_intensity = emotional_context.get('intensity', 0.5)
        
        # 搜索情感记忆
        # Search emotional memories
        for memory in self.memory_store['emotional']:
            mem_emotion = memory['emotional_context'].get('dominant_emotion', 'neutral')
            mem_intensity = memory['emotional_context'].get('intensity', 0.5)
            
            if mem_emotion == target_emotion:
                intensity_similarity = 1 - abs(mem_intensity - target_intensity)
                memory['emotional_similarity'] = intensity_similarity
                relevant_memories.append(memory)
                if len(relevant_memories) >= max_results:
                    return relevant_memories
        
        return relevant_memories
    
    def make_agi_decision(self, options, context=None):
        """基于AGI情感智能做出决策
           Make decisions based on AGI emotional intelligence"""
        if not options:
            return None, 0.0
        
        if len(options) == 1:
            return options[0], 1.0
        
        # 评估每个选项
        # Evaluate each option
        option_evaluations = []
        
        for i, option in enumerate(options):
            score = self._evaluate_option(option, context)
            option_evaluations.append((i, score))
        
        # 排序并选择最佳选项
        # Sort and select best option
        option_evaluations.sort(key=lambda x: x[1], reverse=True)
        best_option_idx, best_score = option_evaluations[0]
        
        return options[best_option_idx], best_score
    
    def _evaluate_option(self, option, context):
        """评估单个选项
           Evaluate a single option"""
        base_score = 0.5
        
        # 基于情感状态调整
        # Adjust based on emotional state
        discrete_emotions = self.get_discrete_emotions()
        dominant_emotion, dominance_score = self.get_dominant_emotion()
        
        # 情感偏好逻辑
        # Emotional preference logic
        if dominant_emotion == 'happy' and dominance_score > 0.7:
            if 'innovative' in str(option).lower() or 'creative' in str(option).lower():
                base_score += 0.2
        
        elif dominant_emotion == 'calm' and dominance_score > 0.6:
            if 'stable' in str(option).lower() or 'consistent' in str(option).lower():
                base_score += 0.15
        
        elif dominant_emotion == 'curious' and dominance_score > 0.6:
            if 'explore' in str(option).lower() or 'learn' in str(option).lower():
                base_score += 0.2
        
        # 基于自我意识水平调整
        # Adjust based on self-awareness level
        if self.self_awareness_level > 0.6:
            if 'long_term' in str(option).lower() or 'sustainable' in str(option).lower():
                base_score += 0.1 * self.self_awareness_level
        
        # 基于元认知水平调整
        # Adjust based on meta-cognition level
        if self.meta_cognition_level > 0.5:
            if 'reflect' in str(option).lower() or 'analyze' in str(option).lower():
                base_score += 0.1 * self.meta_cognition_level
        
        # 基于系统目标调整
        # Adjust based on system goals
        if context and 'goal' in context:
            goal_priority = self.system_goals.get(context['goal'], 0.5)
            base_score *= goal_priority
        
        # 基于价值对齐调整
        # Adjust based on value alignment
        try:
            value_score = self.value_system.evaluate_decision(option, context)
            base_score = base_score * 0.7 + value_score * 0.3
        except Exception as e:
            print(f"价值对齐评估失败: {e}")
        
        return max(0, min(1, base_score))
    
    def get_emotional_summary(self):
        """获取情感状态摘要
           Get emotional state summary"""
        dominant_emotion, dominance_score = self.get_dominant_emotion()
        
        return {
            'dominant_emotion': dominant_emotion,
            'dominance_score': dominance_score,
            'emotional_dimensions': {dim: state['value'] for dim, state in self.emotional_dimensions.items()},
            'discrete_emotions': self.get_discrete_emotions(),
            'self_awareness_level': self.self_awareness_level,
            'meta_cognition_level': self.meta_cognition_level,
            'personality_traits': self.personality_traits.copy(),
            'system_goals': self.system_goals.copy()
        }
    
    def _get_cognitive_architecture(self):
        """延迟初始化并获取认知架构实例
           Lazy initialization and get cognitive architecture instance"""
        if self.cognitive_architecture is None:
            try:
                from core.memory_optimization import ComponentFactory
                self.cognitive_architecture = ComponentFactory.get_component('unified_cognitive_architecture')
            except Exception as e:
                print(f"认知架构初始化失败: {e}")
                # 创建简单的回退实例
                self.cognitive_architecture = UnifiedCognitiveArchitecture()
        return self.cognitive_architecture

    def reflect_on_experience(self, experience, outcome):
        """对经验进行反思和学习
           Reflect on experience and learn"""
        reflection = {
            'experience': experience,
            'outcome': outcome,
            'timestamp': time.time(),
            'pre_emotion': self.get_emotional_summary(),
            'learning_insights': []
        }
        
        # 基于结果更新情感基线
        # Update emotional baseline based on outcome
        if outcome > 0.7:  # 积极结果
            self.emotional_baseline['valence'] = min(1, self.emotional_baseline['valence'] + 0.05)
            reflection['learning_insights'].append('positive_outcome_reinforcement')
        elif outcome < 0.3:  # 消极结果
            self.emotional_baseline['valence'] = max(0, self.emotional_baseline['valence'] - 0.03)
            reflection['learning_insights'].append('negative_outcome_adaptation')
        
        # 更新学习率基于元认知
        # Update learning rate based on meta-cognition
        self.learning_rate = 0.01 + (self.meta_cognition_level * 0.02)
        
        # 存储反思到记忆
        # Store reflection in memory
        self.store_experience(reflection, importance=0.8)
        
        return reflection

# Global instance will be created in main.py after all dependencies are loaded


class EmotionAnalyzer:
    """高级情感分析器类 - 使用现代NLP技术进行多模态情感分析
       Advanced Emotion Analyzer Class - Using modern NLP techniques for multimodal emotion analysis
    """
    
    def __init__(self):
        """初始化高级情感分析器
           Initialize advanced emotion analyzer"""
        # 初始化情感分析模型 - 使用从零开始训练的内部模型
        try:
            # 从core/advanced_emotion_analysis.py导入从零开始训练的情感分析器
            from core.advanced_emotion_analysis import AdvancedEmotionAnalyzer
            
            # 使用内部从零开始训练的情感分析器
            self.sentiment_analyzer = AdvancedEmotionAnalyzer(from_scratch=True)
            
            # 使用简单的内部嵌入模型，避免外部依赖
            self.embedding_model = None  # 不使用外部嵌入模型
            
            # 情感词汇增强库
            self.emotion_lexicon_enhanced = {
                'happy': {'words': ['快乐', '高兴', '开心', '愉悦', '满足', '幸福', '欢乐', '愉快',
                                  'happy', 'joy', 'pleased', 'satisfied', 'bliss', 'delight'],
                         'weight': 1.2},
                'sad': {'words': ['悲伤', '难过', '伤心', '痛苦', '沮丧', '失望', '凄凉', '忧愁',
                                 'sad', 'sorrow', 'grief', 'pain', 'depressed', 'disappointed'],
                       'weight': 1.1},
                'angry': {'words': ['愤怒', '生气', '恼怒', '恼火', '烦躁', '暴躁', '火冒三丈', '怒气冲冲',
                                   'angry', 'furious', 'irritated', 'annoyed', 'enraged', 'mad'],
                         'weight': 1.3},
                'fear': {'words': ['害怕', '恐惧', '畏惧', '担忧', '担心', '惊恐', '恐慌', '惧怕',
                                  'fear', 'afraid', 'scared', 'terrified', 'worried', 'anxious'],
                        'weight': 1.1},
                'surprised': {'words': ['惊讶', '吃惊', '震惊', '意外', '诧异', '惊愕', '惊奇', '讶异',
                                      'surprised', 'amazed', 'shocked', 'astonished', 'startled', 'stunned'],
                            'weight': 1.0},
                'calm': {'words': ['平静', '平和', '安静', '沉稳', '淡定', '从容', '泰然',
                                  'calm', 'peaceful', 'serene', 'tranquil', 'composed', 'relaxed'],
                        'weight': 0.9}
            }
            
            # 情感分析阈值
            self.detection_threshold = 0.15  # 降低阈值以检测更细微的情感
            self.confidence_threshold = 0.4   # 置信度阈值
            
            # 缓存最近的分析结果以提高性能
            self.analysis_cache = {}
            self.cache_size = 1000
            
            print("高级情感分析器初始化成功")
            
        except Exception as e:
            print(f"情感分析器初始化失败: {e}")
            # 回退到简单实现
            self._init_fallback_analyzer()
    
    def _init_fallback_analyzer(self):
        """初始化回退情感分析器
           Initialize fallback emotion analyzer"""
        self.emotion_lexicon = {
            'happy': ['快乐', '高兴', '开心', '愉悦', '满足', '幸福', '欢乐', '愉快',
                      'happy', 'joy', 'pleased', 'satisfied', 'bliss', 'delight'],
            'sad': ['悲伤', '难过', '伤心', '痛苦', '沮丧', '失望', '凄凉', '忧愁',
                   'sad', 'sorrow', 'grief', 'pain', 'depressed', 'disappointed'],
            'angry': ['愤怒', '生气', '恼怒', '恼火', '烦躁', '暴躁', '火冒三丈', '怒气冲冲',
                     'angry', 'furious', 'irritated', 'annoyed', 'enraged', 'mad'],
            'fear': ['害怕', '恐惧', '畏惧', '担忧', '担心', '惊恐', '恐慌', '惧怕',
                     'fear', 'afraid', 'scared', 'terrified', 'worried', 'anxious'],
            'surprised': ['惊讶', '吃惊', '震惊', '意外', '诧异', '惊愕', '惊奇', '讶异',
                         'surprised', 'amazed', 'shocked', 'astonished', 'startled', 'stunned'],
            'calm': ['平静', '平和', '安静', '沉稳', '淡定', '从容', '泰然',
                    'calm', 'peaceful', 'serene', 'tranquil', 'composed', 'relaxed']
        }
        self.detection_threshold = 0.2
        self.confidence_threshold = 0.5
        self.use_advanced_analysis = False
    
    def analyze_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """使用高级NLP技术分析文本情感
           Analyze text emotion using advanced NLP techniques"""
        if not text or not isinstance(text, str):
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            # 使用transformer模型进行情感分析
            sentiment_result = self.sentiment_analyzer(text[:512])  # 限制输入长度
            
            # 映射情感标签到我们的情感体系
            sentiment_map = {
                '1 star': {'emotion': 'angry', 'intensity': 0.8},
                '2 stars': {'emotion': 'sad', 'intensity': 0.6},
                '3 stars': {'emotion': 'neutral', 'intensity': 0.5},
                '4 stars': {'emotion': 'calm', 'intensity': 0.7},
                '5 stars': {'emotion': 'happy', 'intensity': 0.9}
            }
            
            # 获取主要情感
            primary_sentiment = sentiment_result[0]
            sentiment_label = primary_sentiment['label'].lower()
            sentiment_score = primary_sentiment['score']
            
            # 映射到情感体系
            if sentiment_label in sentiment_map:
                mapped_emotion = sentiment_map[sentiment_label]
                base_emotion = mapped_emotion['emotion']
                base_intensity = mapped_emotion['intensity'] * sentiment_score
            else:
                base_emotion = 'neutral'
                base_intensity = 0.5
            
            # 使用词汇增强进行细粒度情感分析
            enhanced_analysis = self._enhance_with_lexicon(text, base_emotion, base_intensity)
            
            # 应用上下文增强（如果有）
            if context:
                enhanced_analysis = self._enhance_with_context(enhanced_analysis, context)
            
            # 缓存结果
            if len(self.analysis_cache) >= self.cache_size:
                self.analysis_cache.pop(next(iter(self.analysis_cache)))
            self.analysis_cache[cache_key] = enhanced_analysis
            
            return enhanced_analysis
            
        except Exception as e:
            print(f"高级情感分析失败，使用回退方法: {e}")
            return self._analyze_text_fallback(text)
    
    def _enhance_with_lexicon(self, text: str, base_emotion: str, base_intensity: float) -> Dict[str, Any]:
        """使用情感词汇库增强情感分析
           Enhance emotion analysis with emotion lexicon"""
        text_lower = text.lower()
        emotions = {emotion: 0.0 for emotion in self.emotion_lexicon_enhanced}
        
        # 为基本情感设置基础值
        emotions[base_emotion] = base_intensity
        
        # 检测词汇中的情感词
        for emotion, data in self.emotion_lexicon_enhanced.items():
            for word in data['words']:
                if word.lower() in text_lower:
                    # 计算情感强度，考虑词汇权重
                    word_count = text_lower.count(word.lower())
                    emotion_strength = min(1.0, word_count * 0.1) * data['weight']
                    emotions[emotion] = max(emotions[emotion], emotion_strength)
        
        # 归一化情感值
        max_value = max(emotions.values()) if emotions.values() else 1.0
        if max_value > 0:
            for emotion in emotions:
                emotions[emotion] = emotions[emotion] / max_value
        
        # 确定主导情感
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        confidence = emotions[dominant_emotion]
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'emotions': emotions,
            'analysis_method': 'advanced'
        }
    
    def _analyze_text_fallback(self, text: str) -> Dict[str, Any]:
        """回退情感分析方法
           Fallback emotion analysis method"""
        if not hasattr(self, 'emotion_lexicon'):
            self._init_fallback_analyzer()
        
        if not text or not isinstance(text, str):
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        # 初始化情感计数
        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon}
        total_words = len(text.split())
        
        # 检查文本中包含的情感词汇
        text_lower = text.lower()
        for emotion, words in self.emotion_lexicon.items():
            for word in words:
                if word.lower() in text_lower:
                    emotion_counts[emotion] += 1
        
        # 计算每个情感的置信度
        emotions = {}
        for emotion, count in emotion_counts.items():
            confidence = count / max(1, total_words)
            if confidence >= self.detection_threshold:
                emotions[emotion] = min(1.0, confidence)
        
        # 确定主导情感
        if not emotions:
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'emotions': emotions,
            'analysis_method': 'fallback'
        }
    
    def combine_emotions(self, emotions_list):
        """合并多个情感分析结果
           Combine multiple emotion analysis results"""
        if not emotions_list:
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        # 加权平均情感值
        # Weighted average of emotion values
        combined_emotions = {}
        total_weight = 0
        
        for result in emotions_list:
            if 'emotions' in result and 'confidence' in result:
                weight = result['confidence']
                total_weight += weight
                for emotion, value in result['emotions'].items():
                    if emotion not in combined_emotions:
                        combined_emotions[emotion] = 0
                    combined_emotions[emotion] += value * weight
        
        # 归一化结果
        # Normalize results
        if total_weight > 0:
            for emotion in combined_emotions:
                combined_emotions[emotion] /= total_weight
        
        # 确定主导情感
        # Determine dominant emotion
        if not combined_emotions:
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        dominant_emotion = max(combined_emotions.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'emotions': combined_emotions
        }
    
    def get_emotion_intensity_level(self, intensity_value):
        """获取情感强度等级描述
           Get emotion intensity level description"""
        if intensity_value < 0.3:
            return {'level': 'low', 'description': '低', 'english_description': 'Low'}
        elif intensity_value < 0.7:
            return {'level': 'medium', 'description': '中等', 'english_description': 'Medium'}
        else:
            return {'level': 'high', 'description': '高', 'english_description': 'High'}
    
    def validate_emotion_data(self, emotion_data):
        """验证情感数据的有效性
           Validate the validity of emotion data"""
        if not isinstance(emotion_data, dict):
            return False, "情感数据必须是字典格式"
            
        if 'dominant_emotion' not in emotion_data or 'confidence' not in emotion_data:
            return False, "情感数据缺少必要字段"
            
        if not isinstance(emotion_data['confidence'], (int, float)) or emotion_data['confidence'] < 0 or emotion_data['confidence'] > 1:
            return False, "置信度必须是0-1之间的数字"
            
        return True, "情感数据有效"

    def analyze_text_with_context(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析文本情感（带上下文）
           Analyze text emotion with context
        
        Args:
            text: 要分析的文本 / Text to analyze
            context: 上下文信息，包含当前情感状态等 / Context information including current emotion state
            
        Returns:
            dict: 情感分析结果 / Emotion analysis results
        """
        try:
            # 首先进行基础情感分析
            # First perform basic emotion analysis
            base_analysis = self.analyze_text(text)
            
            # 如果有上下文信息，进行情感状态增强
            # If context information is available, enhance emotion state
            if context and isinstance(context, dict):
                enhanced_analysis = self._enhance_with_context(base_analysis, context)
                return enhanced_analysis
            
            return base_analysis
            
        except Exception as e:
            error_handler.handle_error(e, "EmotionAnalyzer", "带上下文的情感分析失败")
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}

    def _enhance_with_context(self, base_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """使用上下文信息增强情感分析
           Enhance emotion analysis with context information
        
        Args:
            base_analysis: 基础情感分析结果 / Base emotion analysis results
            context: 上下文信息 / Context information
            
        Returns:
            dict: 增强后的情感分析结果 / Enhanced emotion analysis results
        """
        enhanced = base_analysis.copy()
        
        # 从上下文中提取当前情感状态
        # Extract current emotion state from context
        current_emotion = context.get('emotion', {})
        if isinstance(current_emotion, dict):
            current_dominant = current_emotion.get('dominant_emotion')
            current_confidence = current_emotion.get('confidence', 0.0)
            
            # 如果当前情感状态置信度较高，进行情感状态平滑
            # If current emotion state has high confidence, perform emotion state smoothing
            if current_confidence > 0.7 and current_dominant:
                # 计算情感状态转移权重
                # Calculate emotion state transition weight
                transition_weight = min(current_confidence, 0.3)  # 最大转移权重为0.3
                
                # 调整当前分析结果
                # Adjust current analysis results
                if current_dominant in enhanced['emotions']:
                    enhanced['emotions'][current_dominant] = (
                        enhanced['emotions'][current_dominant] * (1 - transition_weight) + 
                        transition_weight
                    )
                else:
                    enhanced['emotions'][current_dominant] = transition_weight
                
                # 重新计算主导情感
                # Recalculate dominant emotion
                if enhanced['emotions']:
                    dominant_emotion = max(enhanced['emotions'].items(), key=lambda x: x[1])
                    enhanced['dominant_emotion'] = dominant_emotion[0]
                    enhanced['confidence'] = dominant_emotion[1]
        
        # 添加上下文信息
        # Add context information
        enhanced['context_enhanced'] = True
        enhanced['context_emotion'] = current_emotion
        
        return enhanced

    def initialize(self):
        """初始化情感分析器
           Initialize emotion analyzer
           
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 这里可以添加情感分析器的初始化逻辑
            # 例如：加载情感词典、设置模型参数等
            # Add emotion analyzer initialization logic here
            # For example: load emotion lexicon, set model parameters, etc.
            
            print("情感分析器初始化成功")
            # Emotion analyzer initialized successfully
            return True
        except Exception as e:
            print(f"情感分析器初始化失败: {e}")
            # Emotion analyzer initialization failed: {e}
            return False

# 创建全局实例
emotion_analyzer = EmotionAnalyzer()


def analyze_emotion(text):
    """分析文本情感
       Analyze text emotion
    
    Args:
        text: 要分析的文本
        
    Returns:
        dict: 情感分析结果
    """
    return emotion_analyzer.analyze_text(text)


def generate_emotion_response(emotion_data, context=None):
    """生成基于情感状态的响应
       Generate response based on emotional state"""
    if not emotion_data or not isinstance(emotion_data, dict):
        return {}
    
    # 获取主导情感和置信度
    # Get dominant emotion and confidence
    dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
    confidence = emotion_data.get('confidence', 0.0)
    
    # 根据情感状态生成不同的响应模板
    # Generate different response templates based on emotional state
    response_templates = {
        'happy': {
            'positive': {
                '中文': ['我很高兴能帮助你！', '这让我很开心！', '我感到非常愉悦！'],
                '英文': ['I\'m happy to help!', 'That makes me happy!', 'I feel very joyful!']
            },
            'neutral': {
                '中文': ['我现在心情不错。', '今天感觉很好。', '一切都很顺利。'],
                '英文': ['I\'m in a good mood.', 'Feeling good today.', 'Everything is going smoothly.']
            }
        },
        'sad': {
            'negative': {
                '中文': ['我感到有些难过...', '这让我很沮丧。', '我需要一些时间调整。'],
                '英文': ['I\'m feeling a bit sad...', 'This makes me depressed.', 'I need some time to adjust.']
            },
            'comforting': {
                '中文': ['一切都会好起来的。', '我会继续努力的。', '我相信情况会改善。'],
                '英文': ['Everything will be okay.', 'I\'ll keep trying.', 'I believe things will improve.']
            }
        },
        'angry': {
            'frustrated': {
                '中文': ['这让我感到有些烦躁。', '我需要冷静一下。', '我对这个情况感到不满。'],
                '英文': ['This is making me a bit frustrated.', 'I need to calm down.', 'I\'m dissatisfied with this situation.']
            },
            'resolving': {
                '中文': ['我会尽力解决这个问题。', '我们一起找到更好的方法。', '我理解你的感受，我会改进。'],
                '英文': ['I\'ll do my best to fix this.', 'Let\'s find a better way together.', 'I understand your feelings, I\'ll improve.']
            }
        },
        'fear': {
            'concerned': {
                '中文': ['我有些担心这个情况。', '这让我感到不安。', '我希望一切顺利。'],
                '英文': ['I\'m a bit worried about this situation.', 'This makes me feel uneasy.', 'I hope everything goes well.']
            },
            'reassuring': {
                '中文': ['我会小心处理的。', '我们会一起度过难关。', '我相信我们能解决这个问题。'],
                '英文': ['I\'ll handle it carefully.', 'We\'ll get through this together.', 'I believe we can solve this problem.']
            }
        },
        'surprised': {
            'amazed': {
                '中文': ['这太令人惊讶了！', '我没想到会这样。', '真是意外的惊喜！'],
                '英文': ['That\'s amazing!', 'I didn\'t expect that.', 'What a pleasant surprise!']
            },
            'curious': {
                '中文': ['这很有趣，能告诉我更多吗？', '我想了解更多细节。', '这引起了我的兴趣。'],
                '英文': ['That\'s interesting, can you tell me more?', 'I\'d like to know more details.', 'This has sparked my interest.']
            }
        },
        'calm': {
            'peaceful': {
                '中文': ['一切都很平静。', '我保持着平和的心态。', '现在状态很稳定。'],
                '英文': ['Everything is peaceful.', 'I\'m maintaining a calm mindset.', 'The state is stable now.']
            },
            'focused': {
                '中文': ['我正在专注处理任务。', '让我们继续前进。', '我准备好了，可以开始了。'],
                '英文': ['I\'m focusing on the task.', 'Let\'s keep moving forward.', 'I\'m ready to start.']
            }
        },
        'neutral': {
            'default': [
                'I\'m ready to help you at any time.',
                'Is there anything I can help you with?',
                'Please tell me your needs.'
            ]
        }
    }
    
    # Default response type
    response_type = 'default'
    
    # Get context preferences if available
    if context and isinstance(context, dict):
        # Language settings are no longer supported as this is an English-only system
        
        # 根据上下文调整响应类型
        # Adjust response type based on context
        if 'tone' in context:
            if context['tone'] in ['positive', 'negative', 'comforting', 'frustrated', 
                                  'resolving', 'concerned', 'reassuring', 'amazed', 
                                  'curious', 'peaceful', 'focused']:
                response_type = context['tone']
    
    # Ensure response type exists in emotion templates
    emotion_templates = response_templates.get(dominant_emotion, response_templates['neutral'])
    
    # Select appropriate response type
    if response_type not in emotion_templates:
        # Use the first available response type
        response_type = next(iter(emotion_templates.keys()))
    
    # Get response template list
    templates = emotion_templates[response_type]
    
    # 随机选择一个响应（这里使用简单的实现）
    # Randomly select a response (simple implementation here)
    import random
    selected_template = random.choice(templates)
    
    # Build response object
    response = {
        'text': selected_template,
        'emotion': dominant_emotion,
        'confidence': confidence,
        'type': response_type
    }
    
    # 如果有上下文信息，可以进一步丰富响应
    # If there is context information, can further enrich the response
    if context and isinstance(context, dict):
        response['context'] = context.copy()

    return response

# Emotion Awareness Module
class EmotionAwarenessModule:
    """Emotion Awareness Module - For AGI emotion enhancement
    """
    
    def __init__(self):
        self.emotion_analyzer = emotion_analyzer
        self.emotion_system = emotion_awareness_system
    
    def analyze_emotion(self, text: str, context: Dict[str, Any] = None, base_emotion: Dict[str, float] = None) -> Dict[str, float]:
        """分析情感并增强基于上下文
           Analyze emotion and enhance based on context
        """
        try:
            # 使用情感分析器分析文本
            analysis = self.emotion_analyzer.analyze_text_with_context(text, context)
            emotions = analysis.get('emotions', {})
            
            # 如果有基础情感，进行融合
            if base_emotion and isinstance(base_emotion, dict):
                for emotion, intensity in base_emotion.items():
                    if emotion in emotions:
                        # 平均融合
                        emotions[emotion] = (emotions[emotion] + intensity) / 2
                    else:
                        emotions[emotion] = intensity
            
            # 确保情感值在0-1之间
            for emotion in emotions:
                emotions[emotion] = max(0.0, min(1.0, emotions[emotion]))
            
            return emotions
            
        except Exception as e:
            print(f"情感分析错误: {e}")
            return base_emotion if base_emotion else {}
    
    def enhance_response(self, response: str, emotion_state: Dict[str, float]) -> str:
        """基于情感状态增强响应
           Enhance response based on emotion state
        """
        try:
            # 使用生成情感响应函数
            enhanced = generate_emotion_response(emotion_state, context=None)
            enhanced_text = enhanced.get('text', response)
            return enhanced_text
        except Exception as e:
            print(f"响应增强错误: {e}")
            return response

# Create global instances
emotion_analyzer = EmotionAnalyzer()
emotion_awareness_system = AGIEmotionAwarenessSystem()
emotion_awareness_module = EmotionAwarenessModule()
