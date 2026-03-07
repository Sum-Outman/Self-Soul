"""
真正的AGI系统 - 统一认知架构

真正的AGI系统核心包，实现了统一认知架构，
使用神经通信替代HTTP微服务。

此包包含：
- cognitive: 统一认知组件（感知、记忆、推理等）
- neural: 神经网络组件和通信
- api: API服务器和通信协议
- utils: 工具函数和监控
"""

__version__ = "0.1.0"
__author__ = "True AGI System Team"
__description__ = "真正的AGI统一认知架构"

# 导入核心组件以便访问
from .cognitive import UnifiedCognitiveArchitecture
from .neural import NeuralCommunication
from .api import app as AGIServer

# 导出主要类
__all__ = [
    'UnifiedCognitiveArchitecture',
    'NeuralCommunication',
    'AGIServer'
]