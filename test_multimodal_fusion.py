# -*- coding: utf-8 -*-
"""
Self Soul AGI系统多模态融合功能测试脚本
"""

import os
import sys
import time
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 导入多模态融合模块
from core.fusion.multimodal import MultimodalFusion

# 测试多模态融合核心功能
def test_multimodal_fusion_basic():
    print("=== 测试多模态融合基本功能 ===")
    
    # 创建多模态融合实例
    fusion = MultimodalFusion()
    
    # 测试简单加权融合
    test_results = {
        'text': {
            'result': {'text': '这是一个测试文本'},
            'features': [0.1, 0.2, 0.3],
            'confidence': 0.8
        },
        'image': {
            'result': {'image': '测试图像描述'},
            'features': [0.4, 0.5, 0.6],
            'confidence': 0.9
        }
    }
    
    result = fusion.fuse_results(test_results)
    print(f"✓ 简单加权融合测试完成，结果: {result}")
    print(f"  - 融合结果键: {list(result.keys())}")
    
    return fusion

# 测试上下文相似度计算
def test_context_similarity(fusion):
    print("\n=== 测试上下文相似度计算 ===")
    
    # 测试数值相似度
    context1 = {'temperature': 25.5, 'humidity': 60}
    context2 = {'temperature': 26.0, 'humidity': 58}
    similarity = fusion._calculate_context_similarity(context1, context2)
    print(f"✓ 数值上下文相似度: {similarity:.2f}")
    
    # 测试字符串相似度
    context3 = {'text': '这是一个测试句子'}
    context4 = {'text': '这是一个相似的测试句子'}
    similarity = fusion._calculate_context_similarity(context3, context4)
    print(f"✓ 字符串上下文相似度: {similarity:.2f}")
    
    # 测试列表相似度
    context5 = {'tags': ['猫', '狗', '鸟']}
    context6 = {'tags': ['猫', '狗', '鱼']}
    similarity = fusion._calculate_context_similarity(context5, context6)
    print(f"✓ 列表上下文相似度: {similarity:.2f}")
    
    # 测试混合类型相似度
    context7 = {'temperature': 25.5, 'text': '测试文本', 'tags': ['a', 'b', 'c']}
    context8 = {'temperature': 26.0, 'text': '相似测试文本', 'tags': ['a', 'b', 'd']}
    similarity = fusion._calculate_context_similarity(context7, context8)
    print(f"✓ 混合类型上下文相似度: {similarity:.2f}")

# 测试上下文历史管理
def test_context_history(fusion):
    print("\n=== 测试上下文历史管理 ===")
    
    # 更新上下文历史
    test_context = {'test': 'context1', 'value': 10}
    fusion._update_context_history(test_context)
    time.sleep(0.1)  # 确保时间戳不同
    
    test_context2 = {'test': 'context2', 'value': 20}
    fusion._update_context_history(test_context2)
    time.sleep(0.1)
    
    test_context3 = {'test': 'similar context', 'value': 15}
    fusion._update_context_history(test_context3)
    
    # 检索上下文历史
    retrieved = fusion.retrieve_context_history(max_results=3)
    print(f"✓ 检索上下文历史: {len(retrieved)} 条记录")
    
    # 测试带相似度过滤的检索
    similar_context = {'test': 'similar context', 'value': 16}
    retrieved_similar = fusion.retrieve_context_history(
        context=similar_context, 
        max_results=2, 
        similarity_threshold=0.5
    )
    print(f"✓ 带相似度过滤的检索: {len(retrieved_similar)} 条记录")
    for item in retrieved_similar:
        print(f"  - 相似度: {item.get('similarity', 0):.2f}, 上下文: {item['context']}")

# 测试融合历史管理
def test_fusion_history(fusion):
    print("\n=== 测试融合历史管理 ===")
    
    # 创建测试融合结果
    test_results = {
        'text': {
            'result': {'text': '测试文本1'},
            'features': [0.1, 0.2, 0.3],
            'confidence': 0.8
        }
    }
    
    # 执行融合并记录历史
    for i in range(3):
        result = fusion.fuse_results(test_results)
        fusion._update_fusion_history(result, {'test': f'context{i}'})
        time.sleep(0.1)
    
    # 获取融合历史
    history = fusion.get_fusion_history(max_results=2)
    print(f"✓ 获取融合历史: {len(history)} 条记录")
    
    # 清理历史记录
    fusion.cleanup_old_history(max_age=0.1, max_size=1)
    cleaned_history = fusion.get_fusion_history()
    print(f"✓ 清理历史记录后: {len(cleaned_history)} 条记录")

# 测试深度学习融合
def test_deep_learning_fusion(fusion):
    print("\n=== 测试深度学习融合 ===")
    
    # 准备测试特征
    test_results = {
        'text': {
            'result': {'text': '深度学习测试'},
            'features': np.random.rand(10).tolist(),
            'confidence': 0.8
        },
        'audio': {
            'result': {'audio': '测试音频'},
            'features': np.random.rand(10).tolist(),
            'confidence': 0.7
        }
    }
    
    # 测试特征连接融合
    result = fusion.deep_learning_fusion(test_results, fusion_type='concat')
    print(f"✓ 特征连接融合测试完成")
    print(f"  - 返回键: {list(result.keys())}")
    if 'features' in result:
        print(f"  - 融合特征形状: {len(result['features'])}")
    
    # 测试注意力融合
    result = fusion.deep_learning_fusion(test_results, fusion_type='attention')
    print(f"✓ 注意力融合测试完成")
    print(f"  - 返回键: {list(result.keys())}")
    if 'features' in result:
        print(f"  - 融合特征形状: {len(result['features'])}")

# 测试自适应融合
def test_adaptive_fusion(fusion):
    print("\n=== 测试自适应融合 ===")
    
    test_results = {
        'text': {
            'result': {'text': '自适应融合测试'},
            'features': [0.1, 0.2, 0.3],
            'confidence': 0.8
        },
        'image': {
            'result': {'image': '自适应图像描述'},
            'features': [0.4, 0.5, 0.6],
            'confidence': 0.9
        },
        'audio': {
            'result': {'audio': '自适应音频描述'},
            'features': [0.7, 0.8, 0.9],
            'confidence': 0.6
        }
    }
    
    # 测试自适应融合
    result = fusion.adaptive_fusion(test_results)
    print(f"✓ 自适应融合测试完成")
    print(f"  - 返回键: {list(result.keys())}")
    if 'fusion_method' in result:
        print(f"  - 融合方法: {result['fusion_method']}")
    if 'fusion_quality' in result:
        print(f"  - 融合质量: {result['fusion_quality']:.2f}")
    if 'modalities_used' in result:
        print(f"  - 使用的模态: {result['modalities_used']}")

# 主测试函数
if __name__ == "__main__":
    try:
        # 执行所有测试
        fusion = test_multimodal_fusion_basic()
        test_context_similarity(fusion)
        test_context_history(fusion)
        test_fusion_history(fusion)
        test_deep_learning_fusion(fusion)
        test_adaptive_fusion(fusion)
        
        print("\n=== 所有测试完成！ ===")
        print("✓ 多模态融合功能测试通过")
        
    except Exception as e:
        print(f"\n× 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)