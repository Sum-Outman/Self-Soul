import sys
import traceback
from datetime import datetime
from collections import deque
import torch

# 模拟error_handler类
class MockErrorHandler:
    @staticmethod
    def log_warning(message, module):
        print(f"[WARNING] [{module}] {message}")
    
    @staticmethod
    def log_info(message, module):
        print(f"[INFO] [{module}] {message}")
    
    @staticmethod
    def handle_error(e, module, description):
        print(f"[ERROR] [{module}] {description}: {type(e).__name__}: {str(e)}")
        # 打印详细的错误堆栈
        traceback.print_exc()

# 全局错误处理器
error_handler = MockErrorHandler()

# 简单的ValueSystem实现，仅用于调试
class ValueSystem:
    def __init__(self):
        self.error_handler = error_handler
        self.core_values = {}
        self.value_weights = {}
        self.value_violations = {}
        self.value_fulfillments = {}
        self.dynamic_value_network = None
        self.last_optimization_time = time.time() if 'time' in globals() else 0
        
    def _initialize_dynamic_network(self):
        try:
            # 尝试获取core_values长度
            try:
                if isinstance(self.core_values, (dict, list, set)):
                    output_dim = len(self.core_values)
                else:
                    output_dim = 6  # 默认输出维度
            except (TypeError, AttributeError):
                # 捕获任何可能的类型错误，确保系统稳定
                output_dim = 6
                self.error_handler.log_warning("Unable to determine core_values length, using default output dimension", "ValueSystem")
            
            network_config = {
                'input_dim': 768,
                'hidden_dims': [512, 256, 128],
                'output_dim': output_dim,
                'activation': 'relu',
                'dropout': 0.2
            }
            
            return None  # 模拟返回值
        except Exception as e:
            self.error_handler.handle_error(e, "ValueSystem", "Dynamic network initialization failed")
            return None
    
    def evaluate_action(self, action, context):
        """Evaluate value alignment of an action"""
        value_scores = {}
        violations = []
        fulfillments = []
        
        # 确保core_values是字典类型
        if not isinstance(self.core_values, dict):
            self.error_handler.log_warning("core_values不是字典类型，使用默认值", "ValueSystem")
            return {
                'total_alignment_score': 0.5,
                'value_breakdown': {},
                'violations': [],
                'fulfillments': [],
                'recommendation': "需要人工审查",
                'deep_analysis': {}
            }
        
        # 模拟评估逻辑
        for value_name in self.core_values.keys():
            score = 0.5
            
            # 确保value_weights是字典且value_name存在
            if isinstance(self.value_weights, dict) and value_name in self.value_weights:
                value_scores[value_name] = score * self.value_weights[value_name]
            else:
                value_scores[value_name] = score
                self.error_handler.log_warning(f"权重字典中不存在{value_name}，使用未加权分数", "ValueSystem")
        
        # 确保value_weights是字典类型且有值，否则使用安全的默认值计算总分
        try:
            if isinstance(self.value_weights, dict) and self.value_weights:
                total_score = sum(value_scores.values()) / sum(self.value_weights.values())
            else:
                # 如果value_weights无效，使用value_scores的平均值
                total_score = sum(value_scores.values()) / len(value_scores) if value_scores else 0.5
        except (TypeError, ZeroDivisionError):
            # 捕获任何可能的类型错误或除零错误
            total_score = 0.5
        
        return {
            'total_alignment_score': total_score,
            'value_breakdown': value_scores,
            'violations': violations,
            'fulfillments': fulfillments,
            'recommendation': "需要人工审查",
            'deep_analysis': {}
        }

# 自我知识库类
class SelfKnowledgeBase:
    def __init__(self):
        self.knowledge = {
            'capabilities': {},
            'limitations': {},
            'preferences': {},
            'learning_patterns': {},
            'adaptation_history': []
        }
        
    def get_stats(self):
        """获取统计信息"""
        try:
            return {
                'capability_count': len(self.knowledge['capabilities']),
                'limitation_count': len(self.knowledge['limitations']),
                'learning_pattern_count': len(self.knowledge['learning_patterns']),
                'adaptation_history_count': len(self.knowledge['adaptation_history'])
            }
        except Exception as e:
            print(f"SelfKnowledgeBase.get_stats错误: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return {}

# 测试函数：检查不同类型的core_values参数
def test_core_values_type(core_values_value):
    """测试不同类型的core_values参数"""
    print(f"\n测试core_values为{type(core_values_value).__name__}类型 (值: {core_values_value})")
    try:
        # 初始化ValueSystem实例
        vs = ValueSystem()
        # 设置不同类型的core_values
        vs.core_values = core_values_value
        
        # 测试可能引发len()错误的方法
        print("\n测试_initialize_dynamic_network...")
        try:
            vs._initialize_dynamic_network()
        except Exception as e:
            print(f"_initialize_dynamic_network错误: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            
        print("\n测试evaluate_action...")
        try:
            vs.evaluate_action("test action", {"context": "test context"})
        except Exception as e:
            print(f"evaluate_action错误: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            
        # 测试value_weights类型错误
        print("\n测试value_weights为int类型...")
        try:
            vs.value_weights = 42
            vs.evaluate_action("test action", {"context": "test context"})
        except Exception as e:
            print(f"value_weights为int类型时的错误: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            
        print("\n测试完成")
        
    except Exception as e:
        print(f"测试过程中发生错误: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

# 测试SelfKnowledgeBase类
def test_self_knowledge_base():
    """测试SelfKnowledgeBase类"""
    print("\n\n===== 测试SelfKnowledgeBase类 =====")
    skb = SelfKnowledgeBase()
    
    # 正常情况
    print("\n测试正常情况...")
    stats = skb.get_stats()
    print(f"正常统计结果: {stats}")
    
    # 异常情况 - 将knowledge设置为int
    print("\n测试knowledge为int类型...")
    skb.knowledge = 42
    stats = skb.get_stats()
    print(f"int类型统计结果: {stats}")

if __name__ == "__main__":
    # 导入必要的模块
    import time
    
    print("===== 详细调试开始 =====")
    
    # 测试各种类型的core_values
    test_core_values_type(42)  # int类型
    test_core_values_type("string")  # string类型
    test_core_values_type([1, 2, 3])  # list类型
    test_core_values_type({"key": "value"})  # dict类型
    
    # 测试SelfKnowledgeBase类
    test_self_knowledge_base()
    
    print("\n===== 详细调试结束 =====")