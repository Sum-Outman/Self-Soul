"""
统一认知架构测试的Pytest配置。
"""

import pytest
import sys
import os

# 将源目录添加到路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_text_input():
    """用于测试的示例文本输入"""
    return "This is a test sentence for the unified cognitive architecture."


@pytest.fixture
def sample_image_input():
    """用于测试的示例图像输入（模拟）"""
    import numpy as np
    return np.random.randn(3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_audio_input():
    """用于测试的示例音频输入（模拟）"""
    import numpy as np
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def sample_structured_input():
    """用于测试的示例结构化输入"""
    return {
        'feature1': 0.5,
        'feature2': 1.0,
        'feature3': 0.0,
        'feature4': 0.8
    }


@pytest.fixture
def sample_multimodal_input():
    """用于测试的示例多模态输入"""
    return {
        'text': "Test multimodal input",
        'structured': {'value': 0.7, 'category': 'test'}
    }