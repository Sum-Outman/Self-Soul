import sys
import os

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
import sys
if 'error_handler' not in sys.modules:
    sys.modules['error_handler'] = MockErrorHandler()

# 导入我们修复的模块
from core.value_alignment import ValueSystem

# 测试1：验证architecture_optimizer修复
print("\n测试1: 验证architecture_optimizer修复")
try:
    # 创建ValueSystem实例
    vs = ValueSystem()
    
    # 模拟架构优化器丢失的情况
    vs.architecture_optimizer = None
    
    # 调用会使用architecture_optimizer的方法
    result = vs._initialize_dynamic_network()
    
    # 验证architecture_optimizer是否被正确重新创建
    if hasattr(vs, 'architecture_optimizer') and vs.architecture_optimizer is not None:
        print("✅ 修复成功: architecture_optimizer属性被正确恢复")
    else:
        print("❌ 修复失败: architecture_optimizer属性仍然丢失")
        
except Exception as e:
    print(f"❌ 测试失败: {str(e)}")

# 测试2：验证'int' object has no len()修复
print("\n测试2: 验证'int' object has no len()修复")
try:
    # 创建ValueSystem实例
    vs = ValueSystem()
    
    # 模拟core_values是整数的情况
    original_core_values = vs.core_values
    vs.core_values = 42  # 将core_values设置为整数
    
    # 调用会使用len(self.core_values)的方法
    result = vs._initialize_dynamic_network()
    
    # 验证没有引发'int' object has no len()错误
    print("✅ 修复成功: 当core_values是整数时没有引发错误")
    
    # 恢复原始值
    vs.core_values = original_core_values
    
except TypeError as e:
    if "'int' object has no len()" in str(e):
        print("❌ 修复失败: 仍然引发了'int' object has no len()错误")
    else:
        print(f"❌ 测试失败: {str(e)}")
        
except Exception as e:
    print(f"❌ 测试失败: {str(e)}")

print("\n测试完成！")