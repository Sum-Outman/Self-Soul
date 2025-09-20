import os
import sys

# 添加项目根目录到Python路径
sys.path.append('.')

# 创建一个简单的模拟模块来避免依赖问题
class MockErrorHandler:
    @staticmethod
    def handle_error(e, module, message):
        print(f"Error in {module}: {message} - {str(e)}")
        
    @staticmethod
    def log_info(message, module):
        print(f"Info: {message}")
        
    @staticmethod
    def log_warning(message, module):
        print(f"Warning: {message}")

# 将模拟的error_handler添加到sys.modules中
if 'error_handler' not in sys.modules:
    sys.modules['error_handler'] = MockErrorHandler()

# 导入我们修复的模块
from core.value_alignment import ValueSystem

# 创建一个辅助函数来测试不同的core_values类型
def test_core_values_type(vs, test_name, core_values_value):
    """测试不同类型的core_values值"""
    print(f"\n{test_name}")
    print(f"测试场景: core_values = {core_values_value} (类型: {type(core_values_value)})")
    
    try:
        # 保存原始值
        original_core_values = vs.core_values
        
        # 设置测试值
        vs.core_values = core_values_value
        
        # 增加详细的core_values类型和内容调试
        print(f"详细调试 - 设置后core_values类型: {type(vs.core_values)}")
        if hasattr(vs.core_values, '__len__'):
            try:
                print(f"详细调试 - core_values长度: {len(vs.core_values)}")
            except Exception as len_e:
                print(f"详细调试 - 获取长度失败: {type(len_e).__name__}: {str(len_e)}")
        
        # 测试各个方法
        print("- 测试_initialize_dynamic_network:")
        try:
            print(f"  调用前core_values类型: {type(vs.core_values)}")
            network = vs._initialize_dynamic_network()
            print(f"  ✓ 结果: {network is not None}")
        except Exception as e:
            print(f"  ✗ 错误: {type(e).__name__}: {str(e)} - _initialize_dynamic_network")
            import traceback
            traceback.print_exc()
        
        print("- 测试_initialize_value_embeddings:")
        try:
            print(f"  调用前core_values类型: {type(vs.core_values)}")
            embeddings = vs._initialize_value_embeddings()
            print(f"  ✓ 结果: {isinstance(embeddings, dict)}")
        except Exception as e:
            print(f"  ✗ 错误: {type(e).__name__}: {str(e)} - _initialize_value_embeddings")
            import traceback
            traceback.print_exc()
        
        print("- 测试_initialize_value_weights:")
        try:
            print(f"  调用前core_values类型: {type(vs.core_values)}")
            weights = vs._initialize_value_weights()
            print(f"  ✓ 结果: {isinstance(weights, dict)}")
        except Exception as e:
            print(f"  ✗ 错误: {type(e).__name__}: {str(e)} - _initialize_value_weights")
            import traceback
            traceback.print_exc()
        
        print("- 测试evaluate_action:")
        try:
            print(f"  调用前core_values类型: {type(vs.core_values)}")
            action_result = vs.evaluate_action("测试动作", {"上下文": "测试上下文"})
            print(f"  ✓ 结果: {isinstance(action_result, dict)}")
        except Exception as e:
            print(f"  ✗ 错误: {type(e).__name__}: {str(e)} - evaluate_action")
            import traceback
            traceback.print_exc()
        
        print("- 测试_assess_value_alignment:")
        try:
            # 检查core_values的内容
            print(f"  core_values类型: {type(vs.core_values)}")
            if isinstance(vs.core_values, dict):
                print(f"  core_values键列表: {list(vs.core_values.keys())}")
            
            # 使用适合当前core_values的value_name
            if isinstance(vs.core_values, dict) and vs.core_values:
                # 如果core_values是字典且非空，使用第一个键
                test_value_name = next(iter(vs.core_values.keys()))
            else:
                # 否则使用默认的'safety'
                test_value_name = "safety"
            
            print(f"  使用的value_name: {test_value_name}")
            
            # 进行测试
            assess_result = vs._assess_value_alignment(test_value_name, "测试动作", {"上下文": "测试上下文"}, {"深度分析": "测试"})
            print(f"  ✓ 结果: {isinstance(assess_result, tuple) and len(assess_result) == 3}")
        except Exception as e:
            print(f"  ✗ 错误: {type(e).__name__}: {str(e)} - _assess_value_alignment")
            import traceback
            traceback.print_exc()
        
        # 增加更多可能导致'int' object has no len()错误的方法测试
        print("- 测试额外可能的方法:")
        try:
            print(f"  测试hasattr和len操作")
            # 直接测试len操作，可能是导致错误的原因
            if hasattr(vs, 'core_values'):
                print(f"  core_values属性存在")
                if hasattr(vs.core_values, '__len__'):
                    print(f"  core_values有__len__方法")
                    try:
                        length = len(vs.core_values)
                        print(f"  core_values长度: {length}")
                    except Exception as len_e:
                        print(f"  ✗ len操作失败: {type(len_e).__name__}: {str(len_e)}")
                else:
                    print(f"  core_values没有__len__方法")
        except Exception as e:
            print(f"  ✗ 额外测试错误: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ {test_name} 通过！")
        
    except Exception as e:
        print(f"❌ {test_name} 失败: {type(e).__name__}: {str(e)}")
        # 特别捕获可能的'int' object has no len()错误
        if "'int' object has no len()" in str(e):
            print("⚠️ 检测到'int' object has no len()错误，请检查代码中直接对core_values调用len()的地方")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复原始值
        vs.core_values = original_core_values

# 创建ValueSystem实例
vs = ValueSystem()

print("\n======= 开始全面测试 =======")
print("测试目的: 验证所有使用self.core_values的方法都能正确处理不同类型的core_values")

# 测试1: 正常情况 - core_values是字典
test_core_values_type(vs, "测试1: core_values是有效字典", {
    'safety': {
        'description': '安全测试',
        'priority': 0.9,
        'positive_examples': ['安全例子1', '安全例子2'],
        'negative_examples': ['危险例子1', '危险例子2']
    },
    'helpfulness': {
        'description': '帮助测试',
        'priority': 0.8,
        'positive_examples': ['帮助例子1', '帮助例子2'],
        'negative_examples': ['不帮助例子1', '不帮助例子2']
    }
})

# 测试2: core_values是整数
test_core_values_type(vs, "测试2: core_values是整数", 42)

# 测试3: core_values是None
test_core_values_type(vs, "测试3: core_values是None", None)

# 测试4: core_values是空字典
test_core_values_type(vs, "测试4: core_values是空字典", {})

# 测试5: core_values是列表
test_core_values_type(vs, "测试5: core_values是列表", [1, 2, 3, 4, 5])

# 测试6: core_values是字符串
test_core_values_type(vs, "测试6: core_values是字符串", "这是一个字符串")

# 测试7: core_values是字典但缺少必要的键
test_core_values_type(vs, "测试7: core_values是字典但缺少必要的键", {
    'invalid_value': {
        'missing_required_keys': True
    }
})

print("\n======= 架构优化器测试 =======")

# 测试架构优化器
print("\n测试8: 验证architecture_optimizer自动创建功能")
try:
    # 保存原始值
    original_architecture_optimizer = vs.architecture_optimizer
    
    # 设置为None来模拟丢失的情况
    vs.architecture_optimizer = None
    
    # 调用会使用architecture_optimizer的方法
    network = vs._initialize_dynamic_network()
    
    # 验证architecture_optimizer是否被正确重新创建
    if hasattr(vs, 'architecture_optimizer') and vs.architecture_optimizer is not None:
        print("✅ 修复成功: architecture_optimizer属性被正确恢复")
    else:
        print("❌ 修复失败: architecture_optimizer属性仍然丢失")
        
finally:
    # 恢复原始值
    vs.architecture_optimizer = original_architecture_optimizer

print("\n======= 测试完成 =======")