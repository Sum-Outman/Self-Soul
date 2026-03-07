"""
训练功能模块

为统一认知架构提供模型训练功能。
"""

from .trainer import TrainingManager
from .schemas import TrainingRequest, TrainingResponse, TrainingStatus

__all__ = ['TrainingManager', 'TrainingRequest', 'TrainingResponse', 'TrainingStatus']