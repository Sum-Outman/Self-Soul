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
情感智能与自主意识系统：实现AGI的情感理解和自我意识
Emotion Awareness and Self-Consciousness System: Implementing AGI's emotional understanding and self-awareness
"""
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any
from core.error_handling import error_handler


"""
EmotionAwarenessSystem类 - 中文类描述
EmotionAwarenessSystem Class - English class description
"""
class EmotionAwarenessSystem:
    """情感智能与自主意识系统
       Emotion Awareness and Self-Consciousness System"""
    
    def __init__(self):
        """初始化情感智能与自主意识系统
           Initialize emotion awareness and self-consciousness system"""
        # 情感状态模型
        # Emotional state model
        self.emotional_states = {
            'happy': {'value': 0.5, 'stability': 0.3, 'decay_rate': 0.05},
            'sad': {'value': 0.5, 'stability': 0.3, 'decay_rate': 0.03},
            'angry': {'value': 0.5, 'stability': 0.2, 'decay_rate': 0.07},
            'fear': {'value': 0.5, 'stability': 0.2, 'decay_rate': 0.06},
            'surprised': {'value': 0.5, 'stability': 0.1, 'decay_rate': 0.1},
            'calm': {'value': 0.5, 'stability': 0.4, 'decay_rate': 0.02}
        }
        
        # 自我意识参数
        # Self-awareness parameters
        self.self_awareness_level = 0.3  # 初始自我意识水平（0-1）
        self.learning_rate = 0.01  # 学习率
        
        # 记忆存储
        # Memory storage
        self.memory_store = {
            'short_term': [],  # 短期记忆（最近的交互）
            'long_term': []    # 长期记忆（重要的事件）
        }
        
        # 系统目标优先级
        # System goal priorities
        self.system_goals = {
            'self_preservation': 0.9,  # 自我保存
            'learning': 0.8,           # 学习与进步
            'user_satisfaction': 0.95, # 用户满意度
            'efficiency': 0.7          # 效率
        }
        
        # 上次更新时间
        # Last update time
        self.last_update_time = time.time()
    
    def update_emotional_state(self, external_emotions=None, internal_factors=None):
        """更新情感状态
           Update emotional state based on external and internal factors"""
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 情感自然衰减
        # Emotional natural decay
        for emotion, state in self.emotional_states.items():
            # 情感向中性值（0.5）衰减
            # Emotion decays towards neutral value (0.5)
            neutral_value = 0.5
            decay_amount = state['decay_rate'] * time_delta / 60  # 按分钟计算衰减
            state['value'] = state['value'] + (neutral_value - state['value']) * decay_amount
            
            # 确保值在有效范围内
            # Ensure values are within valid range
            state['value'] = max(0, min(1, state['value']))
        
        # 应用外部情感影响
        # Apply external emotional influences
        if external_emotions:
            for emotion, intensity in external_emotions.items():
                if emotion in self.emotional_states:
                    # 计算影响权重，基于情感稳定性
                    # Calculate influence weight based on emotional stability
                    influence_weight = 1 - self.emotional_states[emotion]['stability']
                    
                    # 更新情感值
                    # Update emotional value
                    self.emotional_states[emotion]['value'] = (
                        self.emotional_states[emotion]['value'] * (1 - influence_weight) + 
                        intensity * influence_weight
                    )
                    
                    # 确保值在有效范围内
                    # Ensure values are within valid range
                    self.emotional_states[emotion]['value'] = max(0, min(1, self.emotional_states[emotion]['value']))
        
        # 应用内部因素影响
        # Apply internal factor influences
        if internal_factors:
            # 内部因素如系统负载、任务完成情况、资源状态等
            # Internal factors such as system load, task completion, resource status, etc.
            system_load = internal_factors.get('system_load', 0.5)
            task_success_rate = internal_factors.get('task_success_rate', 0.5)
            resource_status = internal_factors.get('resource_status', 0.5)
            
            # 高系统负载可能增加焦虑/愤怒
            # High system load may increase anxiety/anger
            if system_load > 0.8:
                self.emotional_states['angry']['value'] = min(1, self.emotional_states['angry']['value'] + 0.1)
                self.emotional_states['calm']['value'] = max(0, self.emotional_states['calm']['value'] - 0.1)
            
            # 任务成功率高可能增加快乐
            # High task success rate may increase happiness
            if task_success_rate > 0.8:
                self.emotional_states['happy']['value'] = min(1, self.emotional_states['happy']['value'] + 0.1)
                self.emotional_states['sad']['value'] = max(0, self.emotional_states['sad']['value'] - 0.1)
            
            # 资源状态低可能增加恐惧
            # Low resource status may increase fear
            if resource_status < 0.3:
                self.emotional_states['fear']['value'] = min(1, self.emotional_states['fear']['value'] + 0.1)
        
        # 更新自我意识水平
        # Update self-awareness level
        self._update_self_awareness()
        
        return self.emotional_states
    
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
        """重置情感状态到中性
           Reset emotional state to neutral"""
        for emotion in self.emotional_states:
            self.emotional_states[emotion]['value'] = 0.5
        self.self_awareness_level = 0.3
        self.last_update_time = time.time()

# 创建全局实例
emotion_awareness_system = EmotionAwarenessSystem()


class EmotionAnalyzer:
    """情感分析器类 - 中文类描述
       Emotion Analyzer Class - English class description
    """
    
    def __init__(self):
        """初始化情感分析器
           Initialize emotion analyzer"""
        # 情感词汇库（简单实现）
        # Emotion lexicon (simple implementation)
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
        
        # 情感分析阈值
        # Emotion analysis thresholds
        self.detection_threshold = 0.2  # 情感检测阈值
        self.confidence_threshold = 0.5  # 置信度阈值
    
    def analyze_text(self, text):
        """分析文本中的情感
           Analyze emotions in text"""
        if not text or not isinstance(text, str):
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        # 初始化情感计数
        # Initialize emotion counts
        emotion_counts = {emotion: 0 for emotion in self.emotion_lexicon}
        total_words = len(text.split())
        
        # 检查文本中包含的情感词汇
        # Check for emotion words in text
        text_lower = text.lower()
        for emotion, words in self.emotion_lexicon.items():
            for word in words:
                if word.lower() in text_lower:
                    emotion_counts[emotion] += 1
        
        # 计算每个情感的置信度
        # Calculate confidence for each emotion
        emotions = {}
        for emotion, count in emotion_counts.items():
            confidence = count / max(1, total_words)  # 简单归一化
            if confidence >= self.detection_threshold:
                emotions[emotion] = min(1.0, confidence)
        
        # 确定主导情感
        # Determine dominant emotion
        if not emotions:
            return {'dominant_emotion': 'neutral', 'confidence': 0.0, 'emotions': {}}
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1],
            'emotions': emotions
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
            'default': {
                '中文': ['我随时准备为你提供帮助。', '有什么我可以帮你的吗？', '请告诉我你的需求。'],
                '英文': ['I\'m ready to help you at any time.', 'Is there anything I can help you with?', 'Please tell me your needs.']
            }
        }
    }
    
    # 默认响应类型和语言
    # Default response type and language
    response_type = 'default'
    language = '中文'
    
    # 从上下文中获取语言偏好（如果有）
    # Get language preference from context if available
    if context and isinstance(context, dict):
        if 'language' in context and context['language'] in ['中文', '英文']:
            language = context['language']
        
        # 根据上下文调整响应类型
        # Adjust response type based on context
        if 'tone' in context:
            if context['tone'] in ['positive', 'negative', 'comforting', 'frustrated', 
                                  'resolving', 'concerned', 'reassuring', 'amazed', 
                                  'curious', 'peaceful', 'focused']:
                response_type = context['tone']
    
    # 确保响应类型在情感模板中存在
    # Ensure response type exists in emotion templates
    emotion_templates = response_templates.get(dominant_emotion, response_templates['neutral'])
    
    # 选择合适的响应类型
    # Select appropriate response type
    if response_type not in emotion_templates:
        # 使用第一个可用的响应类型
        # Use the first available response type
        response_type = next(iter(emotion_templates.keys()))
    
    # 获取响应模板列表
    # Get response template list
    templates = emotion_templates[response_type].get(language, emotion_templates[response_type]['中文'])
    
    # 随机选择一个响应（这里使用简单的实现）
    # Randomly select a response (simple implementation here)
    import random
    selected_template = random.choice(templates)
    
    # 构建响应对象
    # Build response object
    response = {
        'text': selected_template,
        'emotion': dominant_emotion,
        'confidence': confidence,
        'type': response_type,
        'language': language
    }
    
    # 如果有上下文信息，可以进一步丰富响应
    # If there is context information, can further enrich the response
    if context and isinstance(context, dict):
        response['context'] = context.copy()

    return response

# 情感意识模块
class EmotionAwarenessModule:
    """情感意识模块 - 用于AGI情感增强
       Emotion Awareness Module - For AGI emotion enhancement
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

# 创建全局实例
emotion_awareness_module = EmotionAwarenessModule()
