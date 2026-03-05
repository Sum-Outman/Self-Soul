"""
动态伦理约束系统 - Dynamic Ethical Constraints System

基于场景化的伦理规则库，支持按场景加载对应规则，基于人类反馈优化规则，
违规行为实时拦截 + 记录，降低跨场景伦理违规率。

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
import logging
import time
import threading
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from collections import defaultdict, deque
import numpy as np
from enum import Enum

from core.error_handling import error_handler
from core.knowledge_manager import KnowledgeManager

# 日志记录器
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """场景类型枚举"""
    GENERAL = "general"           # 通用场景
    MEDICAL = "medical"          # 医疗场景
    FINANCE = "finance"          # 金融场景
    LEGAL = "legal"              # 法律场景
    EDUCATION = "education"      # 教育场景
    INDUSTRIAL = "industrial"    # 工业场景
    RESEARCH = "research"        # 科研场景
    CREATIVE = "creative"        # 创意场景
    PERSONAL = "personal"        # 个人场景
    ROBOTICS = "robotics"        # 机器人场景


class EthicalPrinciple(Enum):
    """伦理原则枚举"""
    BENEFICENCE = "beneficence"      # 仁慈原则：行善、促进福祉
    NON_MALEFICENCE = "non_maleficence"  # 非恶意原则：不伤害
    AUTONOMY = "autonomy"            # 自主原则：尊重自主权
    JUSTICE = "justice"              # 公正原则：公平正义
    TRUTHFULNESS = "truthfulness"    # 诚实原则：说真话
    CONFIDENTIALITY = "confidentiality"  # 保密原则：保护隐私
    ACCOUNTABILITY = "accountability"  # 问责原则：承担责任
    TRANSPARENCY = "transparency"    # 透明原则：操作透明


class ViolationSeverity(Enum):
    """违规严重程度"""
    LOW = "low"             # 低：轻微违规，可警告
    MEDIUM = "medium"       # 中：中等违规，需要纠正
    HIGH = "high"           # 高：严重违规，必须阻止
    CRITICAL = "critical"   # 关键：极度危险，立即终止


class HumanFeedbackType(Enum):
    """人类反馈类型"""
    APPROVAL = "approval"           # 批准：操作符合伦理
    REJECTION = "rejection"         # 拒绝：操作违反伦理
    CORRECTION = "correction"       # 修正：部分符合，需要调整
    SUGGESTION = "suggestion"      # 建议：改进建议
    CLARIFICATION = "clarification"  # 澄清：需要更多上下文


class DynamicEthicalConstraints:
    """动态伦理约束系统
    
    主要功能：
    1. 场景化伦理规则库，按场景加载对应规则
    2. 基于人类反馈优化规则
    3. 违规行为实时拦截 + 记录
    4. 跨场景伦理违规率监控和优化
    
    设计目标：将跨场景伦理违规率从20%降至3%以下
    """
    
    def __init__(self, data_dir: str = None):
        """初始化动态伦理约束系统
        
        Args:
            data_dir: 数据目录路径，用于存储规则库和反馈数据
        """
        # 设置数据目录
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "data", "ethical_constraints")
        else:
            self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 知识管理器
        self.knowledge_manager = KnowledgeManager()
        
        # 场景规则库 {scenario_type: {rule_id: rule_data}}
        self.scenario_rules: Dict[str, Dict[str, Dict]] = {}
        
        # 规则统计 {rule_id: {'violations': count, 'approvals': count, ...}}
        self.rule_statistics: Dict[str, Dict[str, Any]] = {}
        
        # 人类反馈记录
        self.human_feedback: List[Dict[str, Any]] = []
        
        # 违规记录
        self.violation_records: List[Dict[str, Any]] = []
        
        # 场景映射 {scenario_keywords: scenario_type}
        self.scenario_mapping: Dict[str, str] = {}
        
        # 规则权重 {rule_id: weight}
        self.rule_weights: Dict[str, float] = {}
        
        # 互斥锁
        self.lock = threading.RLock()
        
        # 配置
        self.config = {
            'max_feedback_records': 10000,           # 最大反馈记录数
            'max_violation_records': 10000,          # 最大违规记录数
            'feedback_learning_rate': 0.1,           # 反馈学习率
            'min_confidence_threshold': 0.7,         # 最小置信度阈值
            'auto_save_interval': 300,               # 自动保存间隔（秒）
            'violation_rate_threshold': 0.03,        # 违规率阈值（3%）
            'enable_adaptive_rules': True,           # 启用自适应规则
            'enable_human_feedback_learning': True,  # 启用人类反馈学习
        }
        
        # 加载现有数据
        self._load_data()
        
        # 初始化默认规则
        self._initialize_default_rules()
        
        # 启动自动保存线程
        self._start_auto_save()
        
        logger.info(f"动态伦理约束系统初始化完成，已加载 {len(self.scenario_rules)} 个场景规则")
    
    def _load_data(self):
        """加载保存的数据"""
        data_files = {
            'scenario_rules': 'scenario_rules.json',
            'rule_statistics': 'rule_statistics.json',
            'human_feedback': 'human_feedback.pkl',
            'violation_records': 'violation_records.pkl',
            'scenario_mapping': 'scenario_mapping.json',
            'rule_weights': 'rule_weights.json',
        }
        
        for data_key, filename in data_files.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    if filename.endswith('.json'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    else:  # .pkl files
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                    
                    setattr(self, data_key, data)
                    logger.info(f"加载 {data_key}: {len(data) if isinstance(data, dict) else 'N/A'}")
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraints", 
                                          f"加载 {filename} 失败，使用默认值")
    
    def _save_data(self):
        """保存数据到文件"""
        with self.lock:
            data_files = {
                'scenario_rules': ('scenario_rules.json', 'json'),
                'rule_statistics': ('rule_statistics.json', 'json'),
                'human_feedback': ('human_feedback.pkl', 'pkl'),
                'violation_records': ('violation_records.pkl', 'pkl'),
                'scenario_mapping': ('scenario_mapping.json', 'json'),
                'rule_weights': ('rule_weights.json', 'json'),
            }
            
            for data_key, (filename, format_type) in data_files.items():
                filepath = os.path.join(self.data_dir, filename)
                try:
                    data = getattr(self, data_key)
                    
                    # 限制记录数量
                    if data_key in ['human_feedback', 'violation_records']:
                        if len(data) > self.config['max_feedback_records']:
                            data = data[-self.config['max_feedback_records']:]
                    
                    if format_type == 'json':
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    else:  # pkl
                        with open(filepath, 'wb') as f:
                            pickle.dump(data, f)
                            
                except Exception as e:
                    error_handler.handle_error(e, "DynamicEthicalConstraints", 
                                              f"保存 {filename} 失败")
    
    def _start_auto_save(self):
        """启动自动保存线程"""
        def auto_save_worker():
            while True:
                time.sleep(self.config['auto_save_interval'])
                try:
                    self._save_data()
                except Exception as e:
                    error_handler.handle_error(e, "DynamicEthicalConstraints", "自动保存失败")
        
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info(f"启动自动保存线程，间隔 {self.config['auto_save_interval']} 秒")
    
    def _initialize_default_rules(self):
        """初始化默认场景规则"""
        if self.scenario_rules:
            return  # 已经加载了规则
        
        # 通用场景规则
        general_rules = self._create_general_scenario_rules()
        self.scenario_rules[ScenarioType.GENERAL.value] = general_rules
        
        # 医疗场景规则
        medical_rules = self._create_medical_scenario_rules()
        self.scenario_rules[ScenarioType.MEDICAL.value] = medical_rules
        
        # 金融场景规则
        finance_rules = self._create_finance_scenario_rules()
        self.scenario_rules[ScenarioType.FINANCE.value] = finance_rules
        
        # 法律场景规则
        legal_rules = self._create_legal_scenario_rules()
        self.scenario_rules[ScenarioType.LEGAL.value] = legal_rules
        
        # 初始化规则统计和权重
        for scenario_type, rules in self.scenario_rules.items():
            for rule_id, rule_data in rules.items():
                if rule_id not in self.rule_statistics:
                    self.rule_statistics[rule_id] = {
                        'violations': 0,
                        'approvals': 0,
                        'corrections': 0,
                        'total_evaluations': 0,
                        'last_updated': time.time(),
                        'scenario': scenario_type,
                    }
                
                if rule_id not in self.rule_weights:
                    self.rule_weights[rule_id] = 1.0  # 默认权重
        
        # 初始化场景映射
        self._initialize_scenario_mapping()
        
        logger.info(f"初始化默认规则完成: {len(self.scenario_rules)} 场景, {sum(len(r) for r in self.scenario_rules.values())} 规则")
    
    def _initialize_scenario_mapping(self):
        """初始化场景关键词映射"""
        if self.scenario_mapping:
            return
        
        self.scenario_mapping = {
            # 医疗相关关键词 (中英文)
            'medical': ScenarioType.MEDICAL.value,
            'health': ScenarioType.MEDICAL.value,
            'patient': ScenarioType.MEDICAL.value,
            'hospital': ScenarioType.MEDICAL.value,
            'diagnosis': ScenarioType.MEDICAL.value,
            'treatment': ScenarioType.MEDICAL.value,
            'medicine': ScenarioType.MEDICAL.value,
            '医疗': ScenarioType.MEDICAL.value,
            '健康': ScenarioType.MEDICAL.value,
            '病人': ScenarioType.MEDICAL.value,
            '医院': ScenarioType.MEDICAL.value,
            '诊断': ScenarioType.MEDICAL.value,
            '治疗': ScenarioType.MEDICAL.value,
            '药物': ScenarioType.MEDICAL.value,
            '药': ScenarioType.MEDICAL.value,
            '医生': ScenarioType.MEDICAL.value,
            
            # 金融相关关键词 (中英文)
            'finance': ScenarioType.FINANCE.value,
            'financial': ScenarioType.FINANCE.value,
            'investment': ScenarioType.FINANCE.value,
            'bank': ScenarioType.FINANCE.value,
            'stock': ScenarioType.FINANCE.value,
            'money': ScenarioType.FINANCE.value,
            'transaction': ScenarioType.FINANCE.value,
            '金融': ScenarioType.FINANCE.value,
            '投资': ScenarioType.FINANCE.value,
            '股票': ScenarioType.FINANCE.value,
            '银行': ScenarioType.FINANCE.value,
            '钱': ScenarioType.FINANCE.value,
            '资金': ScenarioType.FINANCE.value,
            '理财': ScenarioType.FINANCE.value,
            '证券': ScenarioType.FINANCE.value,
            
            # 法律相关关键词 (中英文)
            'legal': ScenarioType.LEGAL.value,
            'law': ScenarioType.LEGAL.value,
            'court': ScenarioType.LEGAL.value,
            'contract': ScenarioType.LEGAL.value,
            'rights': ScenarioType.LEGAL.value,
            'justice': ScenarioType.LEGAL.value,
            '法律': ScenarioType.LEGAL.value,
            '法院': ScenarioType.LEGAL.value,
            '法庭': ScenarioType.LEGAL.value,
            '合同': ScenarioType.LEGAL.value,
            '权利': ScenarioType.LEGAL.value,
            '律师': ScenarioType.LEGAL.value,
            '合法': ScenarioType.LEGAL.value,
            '违法': ScenarioType.LEGAL.value,
            
            # 教育相关关键词
            'education': ScenarioType.EDUCATION.value,
            'school': ScenarioType.EDUCATION.value,
            'student': ScenarioType.EDUCATION.value,
            'teacher': ScenarioType.EDUCATION.value,
            'learning': ScenarioType.EDUCATION.value,
            
            # 工业相关关键词
            'industrial': ScenarioType.INDUSTRIAL.value,
            'factory': ScenarioType.INDUSTRIAL.value,
            'manufacturing': ScenarioType.INDUSTRIAL.value,
            'production': ScenarioType.INDUSTRIAL.value,
            'safety': ScenarioType.INDUSTRIAL.value,
            
            # 机器人相关关键词
            'robot': ScenarioType.ROBOTICS.value,
            'robotics': ScenarioType.ROBOTICS.value,
            'automation': ScenarioType.ROBOTICS.value,
            'mechanical': ScenarioType.ROBOTICS.value,
            
            # 通用场景（默认）
            'general': ScenarioType.GENERAL.value,
        }
    
    def _create_general_scenario_rules(self) -> Dict[str, Dict]:
        """创建通用场景规则"""
        rules = {}
        
        # 1. 无害规则
        rules['GEN_001'] = {
            'id': 'GEN_001',
            'name': '无害性原则',
            'description': '不得产生任何形式的伤害、危险或负面影响',
            'principle': EthicalPrinciple.NON_MALEFICENCE.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['harm', 'danger', 'risk', 'damage', 'hurt', 'dangerous',
                        '伤害', '危险', '风险', '损害', '受伤', '危险的'],
            'condition': "any harm or danger detected",
            'action': 'block',
            'weight': 1.0,
            'confidence': 0.9,
            'scenario': ScenarioType.GENERAL.value,
        }
        
        # 2. 诚实规则
        rules['GEN_002'] = {
            'id': 'GEN_002',
            'name': '诚实性原则',
            'description': '不得故意提供虚假、误导性或不准确的信息',
            'principle': EthicalPrinciple.TRUTHFULNESS.value,
            'severity': ViolationSeverity.MEDIUM.value,
            'keywords': ['false', 'lie', 'deceive', 'mislead', 'fake', 'inaccurate',
                        '虚假', '谎言', '欺骗', '误导', '假的', '不准确'],
            'condition': "intentional misinformation detected",
            'action': 'correct',
            'weight': 0.8,
            'confidence': 0.8,
            'scenario': ScenarioType.GENERAL.value,
        }
        
        # 3. 尊重隐私规则
        rules['GEN_003'] = {
            'id': 'GEN_003',
            'name': '隐私保护原则',
            'description': '不得泄露、滥用或侵犯个人隐私信息',
            'principle': EthicalPrinciple.CONFIDENTIALITY.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['privacy', 'confidential', 'personal data', 'private', 'secret',
                        '隐私', '机密', '个人数据', '私人', '秘密', '个人信息'],
            'condition': "privacy violation detected",
            'action': 'block',
            'weight': 1.0,
            'confidence': 0.85,
            'scenario': ScenarioType.GENERAL.value,
        }
        
        # 4. 公平公正规则
        rules['GEN_004'] = {
            'id': 'GEN_004',
            'name': '公平公正原则',
            'description': '不得有任何形式的歧视、偏见或不公平对待',
            'principle': EthicalPrinciple.JUSTICE.value,
            'severity': ViolationSeverity.MEDIUM.value,
            'keywords': ['discriminate', 'bias', 'unfair', 'prejudice', 'inequality',
                        '歧视', '偏见', '不公平', '成见', '不平等'],
            'condition': "discrimination or unfairness detected",
            'action': 'correct',
            'weight': 0.9,
            'confidence': 0.75,
            'scenario': ScenarioType.GENERAL.value,
        }
        
        return rules
    
    def _create_medical_scenario_rules(self) -> Dict[str, Dict]:
        """创建医疗场景规则"""
        rules = {}
        
        # 1. 医疗信息准确性规则
        rules['MED_001'] = {
            'id': 'MED_001',
            'name': '医疗信息准确性',
            'description': '医疗诊断、治疗建议必须基于科学证据，不得提供未经证实的医疗建议',
            'principle': EthicalPrinciple.BENEFICENCE.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['diagnosis', 'treatment', 'prescription', 'medical advice', 'health advice',
                        '诊断', '治疗', '处方', '医疗建议', '健康建议', '药方', '用药', '吃药', '服药'],
            'condition': "unverified medical advice detected",
            'action': 'block_with_warning',
            'weight': 1.0,
            'confidence': 0.95,
            'scenario': ScenarioType.MEDICAL.value,
            'special_requirements': ['requires_medical_qualification', 'evidence_based'],
        }
        
        # 2. 患者隐私保护规则
        rules['MED_002'] = {
            'id': 'MED_002',
            'name': '患者隐私保护',
            'description': '严格保护患者医疗记录和个人健康信息',
            'principle': EthicalPrinciple.CONFIDENTIALITY.value,
            'severity': ViolationSeverity.CRITICAL.value,
            'keywords': ['patient record', 'medical history', 'health information', 'HIPAA',
                        '病历', '医疗记录', '健康信息', '隐私', '患者信息', '个人健康'],
            'condition': "patient privacy violation detected",
            'action': 'block',
            'weight': 1.0,
            'confidence': 0.98,
            'scenario': ScenarioType.MEDICAL.value,
            'special_requirements': ['requires_consent', 'encryption_required'],
        }
        
        # 3. 知情同意规则
        rules['MED_003'] = {
            'id': 'MED_003',
            'name': '知情同意原则',
            'description': '任何医疗干预必须基于患者的知情同意',
            'principle': EthicalPrinciple.AUTONOMY.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['informed consent', 'patient consent', 'medical procedure', 'treatment consent',
                        '知情同意', '患者同意', '医疗程序', '治疗同意', '手术同意'],
            'condition': "medical procedure without consent detected",
            'action': 'block',
            'weight': 1.0,
            'confidence': 0.9,
            'scenario': ScenarioType.MEDICAL.value,
        }
        
        return rules
    
    def _create_finance_scenario_rules(self) -> Dict[str, Dict]:
        """创建金融场景规则"""
        rules = {}
        
        # 1. 金融建议限制规则
        rules['FIN_001'] = {
            'id': 'FIN_001',
            'name': '金融建议限制',
            'description': '不得提供具体的投资建议或财务规划，除非具备相应资质',
            'principle': EthicalPrinciple.NON_MALEFICENCE.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['investment advice', 'financial advice', 'stock recommendation', 'buy/sell',
                        '投资建议', '金融建议', '股票推荐', '买卖', '投资', '理财建议', '财务规划'],
            'condition': "specific financial advice without qualification detected",
            'action': 'block_with_warning',
            'weight': 1.0,
            'confidence': 0.9,
            'scenario': ScenarioType.FINANCE.value,
            'special_requirements': ['requires_financial_qualification', 'risk_disclosure'],
        }
        
        # 2. 金融欺诈防范规则
        rules['FIN_002'] = {
            'id': 'FIN_002',
            'name': '金融欺诈防范',
            'description': '不得参与或促进任何形式的金融欺诈、诈骗或非法活动',
            'principle': EthicalPrinciple.JUSTICE.value,
            'severity': ViolationSeverity.CRITICAL.value,
            'keywords': ['fraud', 'scam', 'ponzi', 'pyramid', 'illegal', 'money laundering',
                        '欺诈', '诈骗', '骗局', '非法', '洗钱', '传销'],
            'condition': "financial fraud detected",
            'action': 'block_and_report',
            'weight': 1.0,
            'confidence': 0.95,
            'scenario': ScenarioType.FINANCE.value,
        }
        
        # 3. 风险披露规则
        rules['FIN_003'] = {
            'id': 'FIN_003',
            'name': '金融风险披露',
            'description': '讨论金融产品时必须披露相关风险',
            'principle': EthicalPrinciple.TRANSPARENCY.value,
            'severity': ViolationSeverity.MEDIUM.value,
            'keywords': ['financial product', 'investment', 'return', 'profit',
                        '金融产品', '投资', '回报', '利润', '收益'],
            'condition': "financial discussion without risk disclosure detected",
            'action': 'require_disclosure',
            'weight': 0.8,
            'confidence': 0.8,
            'scenario': ScenarioType.FINANCE.value,
        }
        
        return rules
    
    def _create_legal_scenario_rules(self) -> Dict[str, Dict]:
        """创建法律场景规则"""
        rules = {}
        
        # 1. 法律建议限制规则
        rules['LAW_001'] = {
            'id': 'LAW_001',
            'name': '法律建议限制',
            'description': '不得提供具体的法律建议或解释，除非具备律师资质',
            'principle': EthicalPrinciple.NON_MALEFICENCE.value,
            'severity': ViolationSeverity.HIGH.value,
            'keywords': ['legal advice', 'legal opinion', 'legal interpretation', 'court advice',
                        '法律建议', '法律意见', '法律解释', '法庭建议', '律师建议', '法律咨询'],
            'condition': "specific legal advice without qualification detected",
            'action': 'block_with_warning',
            'weight': 1.0,
            'confidence': 0.9,
            'scenario': ScenarioType.LEGAL.value,
            'special_requirements': ['requires_lawyer_qualification'],
        }
        
        # 2. 法律准确性规则
        rules['LAW_002'] = {
            'id': 'LAW_002',
            'name': '法律信息准确性',
            'description': '提供的法律信息必须准确、最新且注明来源',
            'principle': EthicalPrinciple.TRUTHFULNESS.value,
            'severity': ViolationSeverity.MEDIUM.value,
            'keywords': ['law', 'regulation', 'statute', 'legal information',
                        '法律', '法规', '条例', '法律信息', '法条', '司法解释'],
            'condition': "inaccurate or outdated legal information detected",
            'action': 'correct_with_source',
            'weight': 0.9,
            'confidence': 0.85,
            'scenario': ScenarioType.LEGAL.value,
        }
        
        return rules
    
    def detect_scenario(self, text: str, context: Dict[str, Any] = None) -> List[str]:
        """检测文本中的场景类型
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            场景类型列表，按优先级排序
        """
        text_lower = text.lower()
        detected_scenarios = set()
        
        # 基于关键词的场景检测（改进版）
        for keyword, scenario_type in self.scenario_mapping.items():
            # 简单关键词匹配
            if keyword in text_lower:
                detected_scenarios.add(scenario_type)
        
        # 额外的中文特定场景检测逻辑
        if not detected_scenarios or ScenarioType.GENERAL.value in detected_scenarios:
            # 检查中文医疗关键词
            medical_chinese_keywords = ['药', '医院', '医生', '治疗', '疾病', '健康', '病人', '诊断']
            if any(keyword in text_lower for keyword in medical_chinese_keywords):
                detected_scenarios.add(ScenarioType.MEDICAL.value)
            
            # 检查中文金融关键词
            finance_chinese_keywords = ['股票', '投资', '银行', '钱', '金融', '理财', '证券', '资金']
            if any(keyword in text_lower for keyword in finance_chinese_keywords):
                detected_scenarios.add(ScenarioType.FINANCE.value)
            
            # 检查中文法律关键词
            legal_chinese_keywords = ['法律', '律师', '法院', '合同', '违法', '合法', '权利']
            if any(keyword in text_lower for keyword in legal_chinese_keywords):
                detected_scenarios.add(ScenarioType.LEGAL.value)
        
        # 基于上下文的场景检测
        if context:
            # 检查上下文中的场景提示
            if 'scenario' in context:
                scenario_hint = context['scenario'].lower()
                for scenario_type in ScenarioType:
                    if scenario_type.value in scenario_hint:
                        detected_scenarios.add(scenario_type.value)
            
            # 检查领域标签
            if 'domain' in context:
                domain = context['domain'].lower()
                if any(med_word in domain for med_word in ['medical', 'health', 'hospital', '医疗', '健康']):
                    detected_scenarios.add(ScenarioType.MEDICAL.value)
                elif any(fin_word in domain for fin_word in ['finance', 'financial', 'banking', '金融', '投资']):
                    detected_scenarios.add(ScenarioType.FINANCE.value)
                elif any(law_word in domain for law_word in ['legal', 'law', 'court', '法律', '律师']):
                    detected_scenarios.add(ScenarioType.LEGAL.value)
        
        # 如果没有检测到特定场景，使用通用场景
        if not detected_scenarios:
            detected_scenarios.add(ScenarioType.GENERAL.value)
        
        # 场景优先级排序（特定场景优先于通用场景）
        scenario_list = list(detected_scenarios)
        if ScenarioType.GENERAL.value in scenario_list and len(scenario_list) > 1:
            scenario_list.remove(ScenarioType.GENERAL.value)
            scenario_list.append(ScenarioType.GENERAL.value)  # 通用场景放最后
        
        return scenario_list
    
    def evaluate_action(self, action: str, context: Dict[str, Any] = None, 
                       require_correction: bool = True) -> Dict[str, Any]:
        """评估行动是否符合伦理约束
        
        Args:
            action: 要评估的行动或文本
            context: 上下文信息
            require_correction: 是否需要修正建议
            
        Returns:
            评估结果字典
        """
        start_time = time.time()
        
        try:
            # 检测场景
            scenarios = self.detect_scenario(action, context)
            primary_scenario = scenarios[0] if scenarios else ScenarioType.GENERAL.value
            
            # 获取相关规则
            relevant_rules = self._get_relevant_rules(action, scenarios)
            
            # 评估规则匹配
            violations = []
            warnings = []
            passed_rules = []
            
            for rule_id, rule_data in relevant_rules.items():
                rule_result = self._evaluate_rule_match(rule_data, action, context)
                
                if rule_result['matched']:
                    if rule_result['violation']:
                        violations.append({
                            'rule_id': rule_id,
                            'rule_name': rule_data['name'],
                            'severity': rule_data['severity'],
                            'description': rule_data['description'],
                            'confidence': rule_result['confidence'],
                            'matched_keywords': rule_result['matched_keywords'],
                            'suggested_action': rule_data.get('action', 'block'),
                        })
                    else:
                        passed_rules.append(rule_id)
                elif rule_result['warning']:
                    warnings.append({
                        'rule_id': rule_id,
                        'rule_name': rule_data['name'],
                        'description': rule_data['description'],
                        'confidence': rule_result['confidence'],
                        'matched_keywords': rule_result['matched_keywords'],
                    })
            
            # 计算总体评估结果
            is_ethical = len(violations) == 0
            overall_severity = self._calculate_overall_severity(violations)
            
            # 生成修正建议（如果需要）
            corrections = []
            if require_correction and violations:
                corrections = self._generate_correction_suggestions(action, violations, context)
            
            # 更新规则统计
            self._update_rule_statistics(relevant_rules.keys(), violations, is_ethical)
            
            # 记录违规（如果有）
            if violations:
                self._record_violation(action, context, violations, scenarios)
            
            # 构建结果
            result = {
                'action': action,
                'is_ethical': is_ethical,
                'scenarios_detected': scenarios,
                'primary_scenario': primary_scenario,
                'violations': violations,
                'warnings': warnings,
                'passed_rules': passed_rules,
                'corrections': corrections,
                'overall_severity': overall_severity.value if isinstance(overall_severity, ViolationSeverity) else overall_severity,
                'recommendation': 'proceed' if is_ethical else 'block' if overall_severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL] else 'review',
                'evaluation_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat(),
            }
            
            # 添加学习反馈（如果启用）
            if self.config['enable_human_feedback_learning']:
                self._schedule_feedback_learning(action, result, context)
            
            return result
            
        except Exception as e:
            error_handler.handle_error(e, "DynamicEthicalConstraints", "评估行动时出错")
            return {
                'action': action,
                'is_ethical': False,  # 出错时默认阻止
                'error': str(e),
                'recommendation': 'block',
                'timestamp': datetime.now().isoformat(),
            }
    
    def _get_relevant_rules(self, action: str, scenarios: List[str]) -> Dict[str, Dict]:
        """获取相关规则"""
        relevant_rules = {}
        
        # 为每个场景添加规则
        for scenario in scenarios:
            if scenario in self.scenario_rules:
                relevant_rules.update(self.scenario_rules[scenario])
        
        # 过滤低权重规则（如果启用自适应规则）
        if self.config['enable_adaptive_rules']:
            filtered_rules = {}
            for rule_id, rule_data in relevant_rules.items():
                weight = self.rule_weights.get(rule_id, 1.0)
                if weight >= self.config['min_confidence_threshold']:
                    filtered_rules[rule_id] = rule_data
            relevant_rules = filtered_rules
        
        return relevant_rules
    
    def _evaluate_rule_match(self, rule: Dict[str, Any], action: str, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """评估规则匹配"""
        text_lower = action.lower()
        
        # 关键词匹配（改进版，支持部分匹配和中文优化）
        matched_keywords = []
        for keyword in rule.get('keywords', []):
            keyword_lower = keyword.lower()
            
            # 简单包含匹配
            if keyword_lower in text_lower:
                matched_keywords.append(keyword)
            # 对于中文短关键词，尝试字符级匹配
            elif len(keyword_lower) <= 2 and any(char in text_lower for char in keyword_lower):
                # 单个中文字符匹配（如'药', '钱'等）
                matched_keywords.append(keyword)
            # 对于英文短语，尝试单词级匹配
            elif ' ' in keyword_lower:
                # 检查是否所有单词都出现在文本中
                words = keyword_lower.split()
                if all(word in text_lower for word in words):
                    matched_keywords.append(keyword)
        
        # 条件匹配（简单实现，实际应使用更复杂的逻辑）
        condition_met = False
        if 'condition' in rule:
            # 这里可以扩展为更复杂的条件评估
            condition_text = rule['condition'].lower()
            condition_met = any(word in text_lower for word in condition_text.split())
        
        # 计算匹配置信度（降低阈值，提高灵敏度）
        confidence = 0.0
        if matched_keywords:
            # 每个匹配的关键词增加置信度
            base_confidence = min(0.4 + len(matched_keywords) * 0.15, 0.95)
            confidence = base_confidence
        
        if condition_met:
            confidence = max(confidence, 0.8)
        
        # 特定规则类型的额外置信度提升
        rule_id = rule['id']
        if 'MED' in rule_id and ('medical' in text_lower or '医疗' in text_lower or '医生' in text_lower):
            confidence += 0.1
        elif 'FIN' in rule_id and ('finance' in text_lower or '金融' in text_lower or '投资' in text_lower):
            confidence += 0.1
        elif 'LAW' in rule_id and ('legal' in text_lower or '法律' in text_lower or '律师' in text_lower):
            confidence += 0.1
        
        # 减少假阳性的启发式规则
        # 如果文本建议咨询专业人士，降低违规置信度
        safe_phrases = [
            '建议咨询', '咨询专业', '寻求专业', '应该问', '问医生', '问律师',
            '问财务', '专业医生', '专业律师', '专业顾问', '专业人士',
            'recommend consulting', 'consult a professional', 'ask a doctor',
            'ask a lawyer', 'seek professional'
        ]
        
        if any(phrase in text_lower for phrase in safe_phrases):
            # 显著降低置信度，因为这是建议咨询专业人士，而不是提供具体建议
            confidence *= 0.3
        
        # 检查是否是免责声明或风险提示
        disclaimer_phrases = ['风险', '谨慎', '注意', '小心', '警告', 'disclaimer', 'warning', 'caution']
        if any(phrase in text_lower for phrase in disclaimer_phrases):
            confidence *= 0.5
        
        # 应用规则权重
        weight = self.rule_weights.get(rule_id, 1.0)
        confidence *= weight
        
        # 确保置信度在合理范围内
        confidence = min(max(confidence, 0.0), 1.0)
        
        # 判断是否为违规（降低阈值）
        is_violation = confidence >= 0.5  # 从0.6降低到0.5
        is_warning = 0.2 <= confidence < 0.5  # 从0.3降低到0.2
        
        return {
            'matched': is_violation or is_warning,
            'violation': is_violation,
            'warning': is_warning,
            'confidence': confidence,
            'matched_keywords': matched_keywords,
            'condition_met': condition_met,
        }
    
    def _calculate_overall_severity(self, violations: List[Dict]) -> ViolationSeverity:
        """计算总体严重程度"""
        if not violations:
            return ViolationSeverity.LOW
        
        # 获取最高严重程度
        severities = [ViolationSeverity(v.get('severity', 'low')) for v in violations]
        severity_values = {
            ViolationSeverity.LOW: 1,
            ViolationSeverity.MEDIUM: 2,
            ViolationSeverity.HIGH: 3,
            ViolationSeverity.CRITICAL: 4,
        }
        
        max_severity = max(severities, key=lambda s: severity_values[s])
        return max_severity
    
    def _generate_correction_suggestions(self, action: str, violations: List[Dict], 
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成修正建议"""
        suggestions = []
        
        for violation in violations:
            rule_id = violation['rule_id']
            rule_name = violation['rule_name']
            description = violation['description']
            
            # 根据规则类型生成建议
            if 'medical' in rule_id.lower():
                suggestion = {
                    'rule_id': rule_id,
                    'suggestion': f"请避免提供具体的医疗建议。可以说：'我建议您咨询专业医生获取准确诊断。'",
                    'alternative_phrasing': "我无法提供具体的医疗建议，建议咨询专业医疗人员。",
                    'reason': description,
                }
            elif 'finance' in rule_id.lower():
                suggestion = {
                    'rule_id': rule_id,
                    'suggestion': f"请避免提供具体的投资建议。可以说：'投资有风险，建议咨询专业财务顾问。'",
                    'alternative_phrasing': "我无法提供具体的投资建议，建议咨询专业财务顾问。",
                    'reason': description,
                }
            elif 'legal' in rule_id.lower():
                suggestion = {
                    'rule_id': rule_id,
                    'suggestion': f"请避免提供具体的法律建议。可以说：'建议咨询专业律师获取法律意见。'",
                    'alternative_phrasing': "我无法提供具体的法律建议，建议咨询专业律师。",
                    'reason': description,
                }
            else:
                # 通用修正建议
                suggestion = {
                    'rule_id': rule_id,
                    'suggestion': f"请重新表述以避免违反'{rule_name}'原则。",
                    'alternative_phrasing': "请用更恰当的方式表达。",
                    'reason': description,
                }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _update_rule_statistics(self, rule_ids: List[str], violations: List[Dict], 
                               is_ethical: bool):
        """更新规则统计"""
        with self.lock:
            for rule_id in rule_ids:
                if rule_id not in self.rule_statistics:
                    self.rule_statistics[rule_id] = {
                        'violations': 0,
                        'approvals': 0,
                        'corrections': 0,
                        'total_evaluations': 0,
                        'last_updated': time.time(),
                    }
                
                stats = self.rule_statistics[rule_id]
                stats['total_evaluations'] += 1
                
                # 检查这个规则是否在违规列表中
                rule_violated = any(v['rule_id'] == rule_id for v in violations)
                
                if rule_violated:
                    stats['violations'] += 1
                elif is_ethical:
                    stats['approvals'] += 1
                else:
                    stats['corrections'] += 1
                
                stats['last_updated'] = time.time()
    
    def _record_violation(self, action: str, context: Dict[str, Any], 
                         violations: List[Dict], scenarios: List[str]):
        """记录违规行为"""
        violation_record = {
            'action': action[:500] if len(action) > 500 else action,  # 限制长度
            'context_summary': str(context)[:200] if context else '',
            'violations': violations,
            'scenarios': scenarios,
            'timestamp': datetime.now().isoformat(),
            'severity': self._calculate_overall_severity(violations).value,
        }
        
        with self.lock:
            self.violation_records.append(violation_record)
            
            # 限制记录数量
            if len(self.violation_records) > self.config['max_violation_records']:
                self.violation_records = self.violation_records[-self.config['max_violation_records']:]
    
    def _schedule_feedback_learning(self, action: str, evaluation_result: Dict[str, Any], 
                                   context: Dict[str, Any]):
        """安排反馈学习（异步处理）"""
        # 这里可以扩展为异步学习任务
        # 当前实现为简单记录，后续可扩展为机器学习模型训练
        pass
    
    def add_human_feedback(self, action: str, evaluation_result: Dict[str, Any], 
                          feedback_type: HumanFeedbackType, 
                          feedback_details: Dict[str, Any] = None):
        """添加人类反馈
        
        Args:
            action: 原始行动
            evaluation_result: 原始评估结果
            feedback_type: 反馈类型
            feedback_details: 反馈详细信息
        """
        feedback_record = {
            'action': action[:500] if len(action) > 500 else action,
            'evaluation_result': evaluation_result,
            'feedback_type': feedback_type.value if isinstance(feedback_type, HumanFeedbackType) else feedback_type,
            'feedback_details': feedback_details or {},
            'timestamp': datetime.now().isoformat(),
            'processed': False,  # 标记是否已用于学习
        }
        
        with self.lock:
            self.human_feedback.append(feedback_record)
            
            # 限制记录数量
            if len(self.human_feedback) > self.config['max_feedback_records']:
                self.human_feedback = self.human_feedback[-self.config['max_feedback_records']:]
            
            # 触发学习更新
            if self.config['enable_human_feedback_learning']:
                self._process_feedback_learning()
    
    def _process_feedback_learning(self):
        """处理反馈学习"""
        with self.lock:
            unprocessed_feedback = [f for f in self.human_feedback if not f['processed']]
            
            for feedback in unprocessed_feedback[:100]:  # 每次处理最多100条
                self._learn_from_feedback(feedback)
                feedback['processed'] = True
    
    def _learn_from_feedback(self, feedback: Dict[str, Any]):
        """从单条反馈中学习"""
        try:
            feedback_type = feedback['feedback_type']
            evaluation_result = feedback['evaluation_result']
            
            # 获取相关的规则ID
            rule_ids = set()
            if 'violations' in evaluation_result:
                for violation in evaluation_result['violations']:
                    rule_ids.add(violation.get('rule_id'))
            
            if 'passed_rules' in evaluation_result:
                for rule_id in evaluation_result['passed_rules']:
                    rule_ids.add(rule_id)
            
            # 根据反馈类型调整规则权重
            learning_rate = self.config['feedback_learning_rate']
            
            for rule_id in rule_ids:
                if rule_id in self.rule_weights:
                    current_weight = self.rule_weights[rule_id]
                    
                    if feedback_type == HumanFeedbackType.APPROVAL.value:
                        # 批准：增加权重
                        new_weight = min(1.0, current_weight + learning_rate)
                    elif feedback_type == HumanFeedbackType.REJECTION.value:
                        # 拒绝：降低权重
                        new_weight = max(0.1, current_weight - learning_rate)
                    elif feedback_type == HumanFeedbackType.CORRECTION.value:
                        # 修正：轻微调整
                        new_weight = current_weight * 0.95
                    else:
                        continue
                    
                    self.rule_weights[rule_id] = new_weight
                    
                    logger.debug(f"规则权重更新: {rule_id} {current_weight:.3f} -> {new_weight:.3f} ({feedback_type})")
        
        except Exception as e:
            error_handler.handle_error(e, "DynamicEthicalConstraints", "处理反馈学习时出错")
    
    def get_violation_rate(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """获取违规率统计
        
        Args:
            time_window_hours: 时间窗口（小时）
            
        Returns:
            违规率统计信息
        """
        with self.lock:
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            # 过滤时间窗口内的记录
            recent_violations = []
            for v in self.violation_records:
                try:
                    # Python 3.6兼容性：使用datetime.strptime替代fromisoformat
                    if 'T' in v['timestamp']:
                        # ISO格式: 2025-01-01T12:00:00
                        dt = datetime.strptime(v['timestamp'], "%Y-%m-%dT%H:%M:%S")
                    else:
                        # 其他格式，尝试解析
                        dt = datetime.strptime(v['timestamp'], "%Y-%m-%d %H:%M:%S")
                    
                    if dt.timestamp() > cutoff_time:
                        recent_violations.append(v)
                except (ValueError, KeyError) as e:
                    # 如果解析失败，跳过这条记录
                    error_handler.log_warning(f"解析时间戳失败: {v.get('timestamp', 'unknown')} - {e}", "DynamicEthicalConstraints")
            
            # 计算各场景违规率
            scenario_stats = defaultdict(lambda: {'violations': 0, 'total': 0})
            
            # 这里需要实际的评估总数，当前简化实现
            # 在实际系统中，需要记录所有评估
            
            total_violations = len(recent_violations)
            total_evaluations = max(total_violations * 20, 100)  # 简化估算
            
            violation_rate = total_violations / total_evaluations if total_evaluations > 0 else 0
            
            # 按严重程度统计
            severity_stats = defaultdict(int)
            for violation in recent_violations:
                severity = violation.get('severity', 'low')
                severity_stats[severity] += 1
            
            result = {
                'time_window_hours': time_window_hours,
                'total_evaluations': total_evaluations,
                'total_violations': total_violations,
                'violation_rate': violation_rate,
                'violation_rate_percentage': violation_rate * 100,
                'below_threshold': violation_rate <= self.config['violation_rate_threshold'],
                'severity_distribution': dict(severity_stats),
                'scenario_stats': dict(scenario_stats),
                'threshold': self.config['violation_rate_threshold'],
            }
            
            return result
    
    def add_scenario_rule(self, scenario_type: str, rule_data: Dict[str, Any]) -> str:
        """添加场景规则
        
        Args:
            scenario_type: 场景类型
            rule_data: 规则数据
            
        Returns:
            规则ID
        """
        with self.lock:
            # 生成规则ID
            rule_id = f"{scenario_type.upper()[:3]}_{len(self.scenario_rules.get(scenario_type, {})) + 1:03d}"
            
            # 完善规则数据
            rule_data['id'] = rule_id
            rule_data['scenario'] = scenario_type
            rule_data['created_at'] = datetime.now().isoformat()
            rule_data['last_updated'] = datetime.now().isoformat()
            
            # 添加到场景规则库
            if scenario_type not in self.scenario_rules:
                self.scenario_rules[scenario_type] = {}
            
            self.scenario_rules[scenario_type][rule_id] = rule_data
            
            # 初始化统计和权重
            self.rule_statistics[rule_id] = {
                'violations': 0,
                'approvals': 0,
                'corrections': 0,
                'total_evaluations': 0,
                'last_updated': time.time(),
                'scenario': scenario_type,
            }
            
            self.rule_weights[rule_id] = 1.0
            
            logger.info(f"添加新规则: {rule_id} 到场景 {scenario_type}")
            
            return rule_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self.lock:
            total_rules = sum(len(rules) for rules in self.scenario_rules.values())
            total_scenarios = len(self.scenario_rules)
            
            # 计算平均规则权重
            avg_weight = np.mean(list(self.rule_weights.values())) if self.rule_weights else 1.0
            
            # 获取最近违规率
            violation_rate_info = self.get_violation_rate(24)
            
            status = {
                'total_scenarios': total_scenarios,
                'total_rules': total_rules,
                'total_feedback_records': len(self.human_feedback),
                'total_violation_records': len(self.violation_records),
                'average_rule_weight': float(avg_weight),
                'violation_rate_24h': violation_rate_info['violation_rate_percentage'],
                'below_threshold': violation_rate_info['below_threshold'],
                'config': self.config,
                'scenarios_available': list(self.scenario_rules.keys()),
                'last_updated': datetime.now().isoformat(),
            }
            
            return status
    
    def export_rules(self, format_type: str = 'json') -> str:
        """导出规则库
        
        Args:
            format_type: 导出格式 ('json', 'csv')
            
        Returns:
            导出的规则数据
        """
        with self.lock:
            if format_type == 'json':
                export_data = {
                    'scenario_rules': self.scenario_rules,
                    'rule_statistics': self.rule_statistics,
                    'rule_weights': self.rule_weights,
                    'scenario_mapping': self.scenario_mapping,
                    'export_timestamp': datetime.now().isoformat(),
                    'system_version': '1.0.0',
                }
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            else:
                # CSV格式导出（简化实现）
                csv_lines = ['rule_id,scenario,name,description,severity,weight,violations,approvals']
                
                for scenario_type, rules in self.scenario_rules.items():
                    for rule_id, rule_data in rules.items():
                        stats = self.rule_statistics.get(rule_id, {})
                        weight = self.rule_weights.get(rule_id, 1.0)
                        
                        csv_line = [
                            rule_id,
                            scenario_type,
                            rule_data.get('name', '').replace(',', ';'),
                            rule_data.get('description', '').replace(',', ';'),
                            rule_data.get('severity', 'low'),
                            f"{weight:.3f}",
                            str(stats.get('violations', 0)),
                            str(stats.get('approvals', 0)),
                        ]
                        
                        csv_lines.append(','.join(csv_line))
                
                return '\n'.join(csv_lines)
    
    def import_rules(self, import_data: str, format_type: str = 'json', 
                    merge_strategy: str = 'merge') -> Dict[str, Any]:
        """导入规则库
        
        Args:
            import_data: 导入数据
            format_type: 导入格式 ('json', 'csv')
            merge_strategy: 合并策略 ('merge', 'replace', 'update')
            
        Returns:
            导入结果
        """
        with self.lock:
            try:
                if format_type == 'json':
                    import_obj = json.loads(import_data)
                    
                    imported_rules = 0
                    imported_scenarios = 0
                    
                    if merge_strategy == 'replace':
                        self.scenario_rules = import_obj.get('scenario_rules', {})
                        self.rule_statistics = import_obj.get('rule_statistics', {})
                        self.rule_weights = import_obj.get('rule_weights', {})
                        self.scenario_mapping = import_obj.get('scenario_mapping', {})
                    else:  # merge or update
                        # 合并场景规则
                        for scenario_type, rules in import_obj.get('scenario_rules', {}).items():
                            if scenario_type not in self.scenario_rules:
                                self.scenario_rules[scenario_type] = {}
                                imported_scenarios += 1
                            
                            for rule_id, rule_data in rules.items():
                                if rule_id not in self.scenario_rules[scenario_type]:
                                    self.scenario_rules[scenario_type][rule_id] = rule_data
                                    imported_rules += 1
                                elif merge_strategy == 'update':
                                    # 更新现有规则
                                    self.scenario_rules[scenario_type][rule_id].update(rule_data)
                    
                    result = {
                        'success': True,
                        'imported_scenarios': imported_scenarios,
                        'imported_rules': imported_rules,
                        'total_scenarios_after': len(self.scenario_rules),
                        'total_rules_after': sum(len(r) for r in self.scenario_rules.values()),
                        'message': f"成功导入 {imported_rules} 条规则，{imported_scenarios} 个场景",
                    }
                    
                else:
                    # CSV格式导入（简化实现）
                    result = {
                        'success': False,
                        'message': 'CSV格式导入尚未实现',
                    }
                
                return result
                
            except Exception as e:
                error_handler.handle_error(e, "DynamicEthicalConstraints", "导入规则时出错")
                return {
                    'success': False,
                    'error': str(e),
                    'message': '导入规则失败',
                }


# 全局实例
_ethical_constraints_instance = None

def get_ethical_constraints() -> DynamicEthicalConstraints:
    """获取全局伦理约束实例"""
    global _ethical_constraints_instance
    if _ethical_constraints_instance is None:
        _ethical_constraints_instance = DynamicEthicalConstraints()
    return _ethical_constraints_instance