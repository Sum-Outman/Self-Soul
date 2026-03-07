"""
验证系统

提供输入验证、语义验证和上下文验证功能。
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import jsonschema
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"      # 基本验证：类型、长度等
    SCHEMA = "schema"    # 模式验证：JSON Schema
    SEMANTIC = "semantic"  # 语义验证：内容意义
    CONTEXT = "context"  # 上下文验证：业务逻辑


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


class SchemaValidator:
    """模式验证器"""
    
    def __init__(self):
        # 预定义模式
        self.schemas = {
            'text_input': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'minLength': 1, 'maxLength': 10000},
                    'context': {'type': 'object', 'additionalProperties': True},
                    'options': {'type': 'object', 'additionalProperties': True}
                },
                'required': ['text']
            },
            'multimodal_input': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'minLength': 0, 'maxLength': 5000},
                    'image': {'type': 'string', 'format': 'base64'},
                    'audio': {'type': 'string', 'format': 'base64'},
                    'structured': {'type': 'object', 'additionalProperties': True},
                    'context': {'type': 'object', 'additionalProperties': True}
                },
                'minProperties': 1  # 至少一个模态
            },
            'cognitive_state': {
                'type': 'object',
                'properties': {
                    'current_focus': {'type': ['string', 'null']},
                    'working_memory': {'type': 'array', 'items': {'type': 'object'}},
                    'goal_stack': {'type': 'array', 'items': {'type': 'object'}},
                    'cognitive_load': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'attention_fatigue': {'type': 'number', 'minimum': 0, 'maximum': 1}
                }
            },
            'configuration': {
                'type': 'object',
                'properties': {
                    'embedding_dim': {'type': 'integer', 'minimum': 128, 'maximum': 4096},
                    'max_shared_memory_mb': {'type': 'integer', 'minimum': 10, 'maximum': 10240},
                    'enable_cache': {'type': 'boolean'},
                    'processing_mode': {'type': 'string', 'enum': ['sequential', 'parallel', 'hybrid']}
                },
                'required': ['embedding_dim']
            }
        }
        
        logger.info("模式验证器已初始化")
    
    def validate(self, data: Any, schema_name: str = None, 
                custom_schema: Dict = None) -> ValidationResult:
        """
        根据模式验证数据
        
        参数:
            data: 要验证的数据
            schema_name: 预定义模式名称
            custom_schema: 自定义模式
        
        返回:
            验证结果
        """
        result = ValidationResult(is_valid=True)
        
        # 确定使用哪个模式
        schema = None
        if custom_schema:
            schema = custom_schema
        elif schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
        else:
            result.add_error(f"无效的模式名称: {schema_name}")
            return result
        
        try:
            # 使用jsonschema验证
            jsonschema.validate(instance=data, schema=schema)
            
            # 额外的自定义验证
            self._additional_validation(data, schema_name, result)
            
        except jsonschema.ValidationError as e:
            result.add_error(f"模式验证失败: {e.message}")
            result.metadata['validation_path'] = list(e.path)
            result.metadata['validator'] = e.validator
            result.metadata['validator_value'] = e.validator_value
            
        except Exception as e:
            result.add_error(f"验证过程中发生错误: {str(e)}")
        
        return result
    
    def _additional_validation(self, data: Dict, schema_name: str, 
                             result: ValidationResult):
        """额外的自定义验证"""
        if schema_name == 'text_input':
            if 'text' in data:
                text = data['text']
                # 检查空白文本
                if not text.strip():
                    result.add_warning("文本内容为空或仅包含空白字符")
                
                # 检查特殊字符比例
                special_chars = re.findall(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', text)
                if len(special_chars) > len(text) * 0.2:
                    result.add_warning("特殊字符比例较高")
        
        elif schema_name == 'multimodal_input':
            # 检查模态组合的合理性
            modalities = [key for key in ['text', 'image', 'audio', 'structured'] 
                         if key in data and data[key]]
            
            if len(modalities) > 3:
                result.add_warning("多模态输入包含多个模态，可能影响处理效率")
            
            # 检查数据大小（简化版）
            total_size = 0
            for modality in modalities:
                value = data[modality]
                if isinstance(value, str):
                    total_size += len(value)
                elif isinstance(value, dict):
                    total_size += len(json.dumps(value))
            
            if total_size > 10 * 1024 * 1024:  # 10MB
                result.add_warning("输入数据较大，可能影响性能")
    
    def register_schema(self, name: str, schema: Dict):
        """注册新模式"""
        try:
            # 验证模式本身
            jsonschema.Draft7Validator.check_schema(schema)
            self.schemas[name] = schema
            logger.info(f"已注册新模式: {name}")
        except jsonschema.SchemaError as e:
            logger.error(f"无效的模式: {name} - {e.message}")
            raise
    
    def get_schema(self, name: str) -> Optional[Dict]:
        """获取模式定义"""
        return self.schemas.get(name)


class SemanticValidator:
    """语义验证器"""
    
    def __init__(self):
        # 语义规则
        self.rules = {
            'meaningful_text': {
                'description': '文本应具有语义意义',
                'validator': self._validate_meaningful_text
            },
            'appropriate_length': {
                'description': '输入长度应适中',
                'validator': self._validate_appropriate_length
            },
            'coherent_structure': {
                'description': '内容结构应连贯',
                'validator': self._validate_coherent_structure
            },
            'relevant_content': {
                'description': '内容应与上下文相关',
                'validator': self._validate_relevant_content
            }
        }
        
        logger.info("语义验证器已初始化")
    
    def validate(self, data: Any, context: Dict = None, 
                rules: List[str] = None) -> ValidationResult:
        """
        语义验证
        
        参数:
            data: 要验证的数据
            context: 上下文信息
            rules: 要应用的规则列表（如果为None，则应用所有规则）
        
        返回:
            验证结果
        """
        result = ValidationResult(is_valid=True)
        
        if context is None:
            context = {}
        
        # 确定要应用的规则
        rules_to_apply = rules or list(self.rules.keys())
        
        # 应用规则
        for rule_name in rules_to_apply:
            if rule_name in self.rules:
                rule = self.rules[rule_name]
                try:
                    rule['validator'](data, context, result)
                except Exception as e:
                    logger.error(f"规则 {rule_name} 执行错误: {e}")
                    result.add_warning(f"规则 {rule_name} 执行失败: {str(e)}")
            else:
                result.add_warning(f"未知的规则: {rule_name}")
        
        return result
    
    def _validate_meaningful_text(self, data: Any, context: Dict, 
                                result: ValidationResult):
        """验证文本是否有意义"""
        if isinstance(data, str):
            text = data.strip()
            
            # 检查重复字符
            if len(text) >= 10:
                for i in range(len(text) - 3):
                    if text[i] == text[i+1] == text[i+2] == text[i+3]:
                        result.add_warning("文本中包含重复字符序列")
                        break
            
            # 检查词汇多样性
            words = text.split()
            if len(words) >= 5:
                unique_words = set(words)
                diversity = len(unique_words) / len(words)
                if diversity < 0.3:
                    result.add_warning("文本词汇多样性较低")
            
            # 检查句子结构
            sentences = re.split(r'[.!?]+', text)
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            
            if valid_sentences:
                avg_words = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
                if avg_words > 50:
                    result.add_warning("句子过长，可能影响理解")
                elif avg_words < 3 and len(valid_sentences) > 1:
                    result.add_warning("句子过短，可能缺乏上下文")
    
    def _validate_appropriate_length(self, data: Any, context: Dict, 
                                   result: ValidationResult):
        """验证长度是否合适"""
        if isinstance(data, str):
            length = len(data)
            if length == 0:
                result.add_error("文本不能为空")
            elif length < 10:
                result.add_warning("文本可能过短")
            elif length > 10000:
                result.add_warning("文本过长，建议分段处理")
        
        elif isinstance(data, dict):
            # 对于字典，检查键值对数量
            num_items = len(data)
            if num_items == 0:
                result.add_warning("字典为空")
            elif num_items > 100:
                result.add_warning("字典包含过多项目")
        
        elif isinstance(data, list):
            # 对于列表，检查元素数量
            num_items = len(data)
            if num_items == 0:
                result.add_warning("列表为空")
            elif num_items > 1000:
                result.add_warning("列表包含过多元素")
    
    def _validate_coherent_structure(self, data: Any, context: Dict, 
                                   result: ValidationResult):
        """验证结构连贯性"""
        if isinstance(data, dict):
            # 检查嵌套深度
            max_depth = self._calculate_dict_depth(data)
            if max_depth > 10:
                result.add_warning("数据结构嵌套过深")
            
            # 检查键名一致性
            keys = list(data.keys())
            if keys:
                # 检查键名风格是否一致
                styles = []
                for key in keys:
                    if '_' in key:
                        styles.append('snake_case')
                    elif key[0].isupper():
                        styles.append('PascalCase')
                    elif key[0].islower() and any(c.isupper() for c in key[1:]):
                        styles.append('camelCase')
                    else:
                        styles.append('other')
                
                if len(set(styles)) > 1:
                    result.add_warning("键名命名风格不一致")
    
    def _validate_relevant_content(self, data: Any, context: Dict, 
                                 result: ValidationResult):
        """验证内容相关性"""
        # 简化版的相关性检查
        # 在实际实现中，这里会有更复杂的语义分析
        
        if 'expected_topic' in context and isinstance(data, str):
            expected_topic = context['expected_topic'].lower()
            text = data.lower()
            
            # 简单的关键词检查
            topic_words = expected_topic.split()
            matches = sum(1 for word in topic_words if word in text)
            
            relevance = matches / len(topic_words) if topic_words else 0
            if relevance < 0.3:
                result.add_warning("内容可能偏离预期主题")
    
    def _calculate_dict_depth(self, data: Dict, current_depth: int = 0) -> int:
        """计算字典的最大嵌套深度"""
        if not isinstance(data, dict) or not data:
            return current_depth
        
        depths = [current_depth + 1]  # 当前层
        
        for value in data.values():
            if isinstance(value, dict):
                depths.append(self._calculate_dict_depth(value, current_depth + 1))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        depths.append(self._calculate_dict_depth(item, current_depth + 1))
        
        return max(depths) if depths else current_depth + 1
    
    def register_rule(self, name: str, description: str, validator: Callable):
        """注册新规则"""
        self.rules[name] = {
            'description': description,
            'validator': validator
        }
        logger.info(f"已注册新语义规则: {name}")


class ContextValidator:
    """上下文验证器"""
    
    def __init__(self):
        # 上下文验证规则
        self.context_rules = {}
        
        logger.info("上下文验证器已初始化")
    
    def validate(self, data: Any, context: Dict, 
                validation_context: Dict = None) -> ValidationResult:
        """
        上下文验证
        
        参数:
            data: 要验证的数据
            context: 数据上下文
            validation_context: 验证上下文（规则、约束等）
        
        返回:
            验证结果
        """
        result = ValidationResult(is_valid=True)
        
        if validation_context is None:
            validation_context = {}
        
        # 应用上下文规则
        self._apply_context_rules(data, context, validation_context, result)
        
        # 检查数据与上下文的一致性
        self._check_context_consistency(data, context, result)
        
        return result
    
    def _apply_context_rules(self, data: Any, context: Dict, 
                           validation_context: Dict, result: ValidationResult):
        """应用上下文规则"""
        # 从验证上下文中获取规则
        rules = validation_context.get('rules', [])
        
        for rule in rules:
            rule_type = rule.get('type')
            
            if rule_type == 'required_fields':
                required = rule.get('fields', [])
                if isinstance(data, dict):
                    missing = [field for field in required if field not in data]
                    if missing:
                        result.add_error(f"缺少必要字段: {', '.join(missing)}")
            
            elif rule_type == 'field_constraints':
                constraints = rule.get('constraints', {})
                if isinstance(data, dict):
                    for field, constraint in constraints.items():
                        if field in data:
                            self._apply_field_constraint(data[field], field, constraint, result)
            
            elif rule_type == 'context_dependencies':
                dependencies = rule.get('dependencies', {})
                for target_field, source_fields in dependencies.items():
                    if isinstance(data, dict) and target_field in data:
                        missing_deps = [sf for sf in source_fields if sf not in context]
                        if missing_deps:
                            result.add_warning(
                                f"字段 '{target_field}' 缺少上下文依赖: {', '.join(missing_deps)}"
                            )
    
    def _apply_field_constraint(self, value: Any, field: str, 
                              constraint: Dict, result: ValidationResult):
        """应用字段约束"""
        constraint_type = constraint.get('type')
        
        if constraint_type == 'range':
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            
            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    result.add_error(f"字段 '{field}' 值 {value} 小于最小值 {min_val}")
                if max_val is not None and value > max_val:
                    result.add_error(f"字段 '{field}' 值 {value} 大于最大值 {max_val}")
        
        elif constraint_type == 'enum':
            allowed_values = constraint.get('values', [])
            if value not in allowed_values:
                result.add_error(f"字段 '{field}' 值 {value} 不在允许值列表中: {allowed_values}")
        
        elif constraint_type == 'pattern':
            pattern = constraint.get('pattern')
            if isinstance(value, str) and pattern:
                if not re.match(pattern, value):
                    result.add_error(f"字段 '{field}' 值不匹配模式: {pattern}")
        
        elif constraint_type == 'custom':
            validator = constraint.get('validator')
            if callable(validator):
                try:
                    is_valid, message = validator(value)
                    if not is_valid:
                        result.add_error(f"字段 '{field}' 验证失败: {message}")
                except Exception as e:
                    result.add_error(f"字段 '{field}' 自定义验证错误: {str(e)}")
    
    def _check_context_consistency(self, data: Any, context: Dict, 
                                 result: ValidationResult):
        """检查上下文一致性"""
        # 检查时间一致性
        if 'timestamp' in data and 'request_time' in context:
            data_time = data['timestamp']
            request_time = context['request_time']
            
            if isinstance(data_time, (int, float)) and isinstance(request_time, (int, float)):
                time_diff = abs(data_time - request_time)
                if time_diff > 3600:  # 1小时
                    result.add_warning("数据时间戳与请求时间相差较大")
        
        # 检查用户上下文
        if 'user_id' in context and 'user_id' in data:
            if context['user_id'] != data['user_id']:
                result.add_error("用户ID不一致")
        
        # 检查会话上下文
        if 'session_id' in context and 'session_id' in data:
            if context['session_id'] != data['session_id']:
                result.add_warning("会话ID不一致")
    
    def register_context_rule(self, name: str, rule: Dict):
        """注册上下文规则"""
        self.context_rules[name] = rule
        logger.info(f"已注册上下文规则: {name}")