"""
Medical Model: 医疗健康专业模型
"""

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


"""
MedicalModel类 - 中文类描述
MedicalModel Class - English class description
"""
class MedicalModel:
    """医疗健康专业模型，提供健康咨询、症状分析等功能
       Medical professional model providing health consultation and symptom analysis
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
        # 初始化医学知识库
        # Initialize medical knowledge base
        self.medical_knowledge = {
            # 常见疾病和症状关系
            # Common diseases and symptom relationships
            'symptoms_to_diseases': {
                '发热': ['感冒', '流感', '肺炎', '新冠病毒感染'],
                '咳嗽': ['感冒', '流感', '肺炎', '支气管炎', '过敏'],
                '头痛': ['偏头痛', '紧张性头痛', '感冒', '高血压'],
                '腹痛': ['胃炎', '胆囊炎', '阑尾炎', '肠胃炎'],
                '腹泻': ['肠胃炎', '食物中毒', '肠道感染']
            },
            
            # 疾病描述和建议
            # Disease descriptions and recommendations
            'disease_info': {
                '感冒': {
                    'description': '由病毒引起的上呼吸道感染，通常具有自限性',
                    'recommendations': ['休息', '多喝水', '保持室内空气流通', '对症治疗']
                },
                '流感': {
                    'description': '由流感病毒引起的急性呼吸道传染病，传染性强',
                    'recommendations': ['及时就医', '休息', '多喝水', '避免前往人群密集场所']
                },
                '肺炎': {
                    'description': '肺部的炎症，可由细菌、病毒或真菌引起',
                    'recommendations': ['立即就医', '按医嘱使用抗生素', '充足休息']
                },
                '肠胃炎': {
                    'description': '胃肠道的炎症，常见症状包括腹泻、呕吐和腹痛',
                    'recommendations': ['补充水分和电解质', '饮食清淡', '避免刺激性食物']
                }
            },
            
            # 健康生活建议
            # Healthy lifestyle advice
            'health_advice': {
                '饮食': ['均衡饮食', '多吃蔬果', '控制油盐摄入', '规律进餐'],
                '运动': ['每周至少150分钟中等强度有氧运动', '每周2-3次力量训练', '避免久坐'],
                '睡眠': ['保持7-9小时充足睡眠', '建立规律的睡眠习惯', '睡前避免使用电子设备'],
                '心理': ['保持积极心态', '学会压力管理', '建立良好的社会支持网络']
            }
        }
        
        # 支持的语言
        # Supported languages
        self.supported_languages = ['zh', 'en']
        
        # 多语言医疗术语映射
        # Multilingual medical term mapping
        self.medical_terms_translation = {
            'zh': {
                'symptoms': '症状',
                'diseases': '疾病',
                'recommendations': '建议',
                'description': '描述'
            },
            'en': {
                'symptoms': 'symptoms',
                'diseases': 'diseases',
                'recommendations': 'recommendations',
                'description': 'description'
            }
        }
    
    
"""
analyze_symptoms函数 - 中文函数描述
analyze_symptoms Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def analyze_symptoms(self, symptoms, lang='zh'):
        """分析症状并提供可能的健康问题和建议
           Analyze symptoms and provide possible health issues and recommendations
        
        Args:
            symptoms (list): 症状列表
            lang (str): 语言代码
        
        Returns:
            dict: 分析结果
        """
        if not symptoms:
            return {self._translate('symptoms', lang): "请提供症状信息以获得分析"}
        
        # 找出可能的疾病
        # Find possible diseases
        possible_diseases = set()
        for symptom in symptoms:
            if symptom in self.medical_knowledge['symptoms_to_diseases']:
                possible_diseases.update(self.medical_knowledge['symptoms_to_diseases'][symptom])
        
        # 构建结果
        # Build result
        result = {
            self._translate('symptoms', lang): symptoms,
            self._translate('diseases', lang): []
        }
        
        # 添加每种疾病的信息
        # Add information for each disease
        for disease in possible_diseases:
            if disease in self.medical_knowledge['disease_info']:
                disease_info = self.medical_knowledge['disease_info'][disease]
                result[self._translate('diseases', lang)].append({
                    'name': disease,
                    self._translate('description', lang): disease_info['description'],
                    self._translate('recommendations', lang): disease_info['recommendations']
                })
            else:
                result[self._translate('diseases', lang)].append({
                    'name': disease,
                    self._translate('description', lang): "暂无详细信息",
                    self._translate('recommendations', lang): ["建议咨询专业医生"]
                })
        
        # 如果没有找到匹配的疾病，提供一般建议
        # If no matching diseases found, provide general advice
        if not result[self._translate('diseases', lang)]:
            result['general_advice'] = ["建议记录详细症状并咨询专业医生", "密切关注症状变化"]
        
        return result
    
    
"""
get_health_advice函数 - 中文函数描述
get_health_advice Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def get_health_advice(self, category=None, lang='zh'):
        """提供健康生活建议
           Provide health lifestyle advice
        
        Args:
            category (str): 建议类别
            lang (str): 语言代码
        
        Returns:
            dict: 健康建议
        """
        result = {}
        
        if category and category in self.medical_knowledge['health_advice']:
            result[category] = self.medical_knowledge['health_advice'][category]
        else:
            result = self.medical_knowledge['health_advice']
        
        return result
    
    
"""
_translate函数 - 中文函数描述
_translate Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _translate(self, term, lang):
        """翻译医学术语
           Translate medical terms
        """
        if lang in self.medical_terms_translation and term in self.medical_terms_translation[lang]:
            return self.medical_terms_translation[lang][term]
        return term
    
    
"""
train函数 - 中文函数描述
train Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def train(self, training_data, callback=None):
        """训练医疗模型
           Train the medical model
        
        Args:
            training_data: 训练数据
            callback: 进度回调函数
        
        Returns:
            dict: 训练结果
        """
        # 模拟训练过程
        # Simulate training process
        # 在实际实现中，这里应该使用真实的医疗数据训练模型
        # In actual implementation, real medical data should be used to train the model
        
        # 记录训练开始时间
        # Record training start time
        import time
        start_time = time.time()
        
        # 模拟训练进度
        # Simulate training progress
        for i in range(10):
            time.sleep(0.5)  # 模拟训练时间
            progress = (i + 1) * 10
            
            # 计算模拟指标
            # Calculate simulated metrics
            loss = 0.5 - (i * 0.04)
            accuracy = 60 + (i * 3)
            
            # 调用回调函数更新进度
            # Call callback function to update progress
            if callback:
                callback(progress, {'loss': loss, 'accuracy': accuracy})
        
        # 返回训练结果
        # Return training results
        return {
            'status': 'completed',
            'training_time': time.time() - start_time,
            'final_metrics': {
                'loss': 0.1,
                'accuracy': 87
            }
        }
    
    
"""
process函数 - 中文函数描述
process Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def process(self, input_data):
        """处理输入数据
           Process input data
        
        Args:
            input_data (dict): 输入数据，包含查询类型和参数
        
        Returns:
            dict: 处理结果
        """
        query_type = input_data.get('type', 'symptom_analysis')
        lang = input_data.get('lang', 'zh')
        
        if query_type == 'symptom_analysis':
            symptoms = input_data.get('symptoms', [])
            return self.analyze_symptoms(symptoms, lang)
        elif query_type == 'health_advice':
            category = input_data.get('category')
            return self.get_health_advice(category, lang)
        else:
            return {"error": "不支持的查询类型"}