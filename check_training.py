import json
import os

print('=== 模型预训练情况检查 ===')

# 检查模型注册表
print('\n1. 模型注册表状态:')
try:
    from core.model_registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.get_all_models()
    if models:
        for model_id, model_info in models.items():
            print(f'  - {model_id}: {model_info.get("status", "未知")}')
    else:
        print('  未找到已注册模型')
except Exception as e:
    print(f'  模型注册表检查错误: {e}')

# 检查训练历史记录
print('\n2. 训练历史记录:')
try:
    with open('data/training_history.json', 'r', encoding='utf-8') as f:
        history = json.load(f)
        print(f'  训练记录数量: {len(history)}')
        for record in history[-5:]:
            print(f'  - {record.get("model_name", "未知")}: {record.get("status", "未知")}')
except FileNotFoundError:
    print('  训练历史文件不存在')
except Exception as e:
    print(f'  训练历史检查错误: {e}')

# 检查核心模型目录
print('\n3. 核心模型组件状态:')
core_models = [
    'language', 'audio', 'vision', 'video', 'spatial', 
    'sensor', 'computer', 'motion', 'knowledge', 'programming', 'emotion'
]
for model_type in core_models:
    try:
        module_name = f'core.models.{model_type}'
        __import__(module_name)
        print(f'  - {model_type}_model: 模块存在')
    except ImportError:
        print(f'  - {model_type}_model: 模块缺失')
    except Exception as e:
        print(f'  - {model_type}_model: 检查错误 - {e}')

# 检查训练管理器状态
print('\n4. 训练管理器状态:')
try:
    from core.training_manager import TrainingManager
    training_manager = TrainingManager()
    training_status = training_manager.get_training_status()
    print(f'  训练状态: {training_status}')
except Exception as e:
    print(f'  训练管理器检查错误: {e}')

print('\n=== 检查完成 ===')
