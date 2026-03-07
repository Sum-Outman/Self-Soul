"""
Neural Components Module

Contains neural network components and communication systems:
- Networks: Neural network architectures
- Communication: Tensor-based communication (replaces HTTP)
- Optimization: Neural network optimization and compilation
- Training: Training systems for neural components
"""

from .networks import (
    TextEncoder, ImageEncoder, AudioEncoder, StructuredEncoder,
    UnifiedProjectionNetwork, AttentionNetwork, MemoryNetwork
)
from .communication import NeuralCommunication, SharedTensorManager
from .optimization import CognitiveCompiler, GPUMemoryManager
from .training import NeuralTrainer, MetaTrainer

__all__ = [
    # Networks
    'TextEncoder', 'ImageEncoder', 'AudioEncoder', 'StructuredEncoder',
    'UnifiedProjectionNetwork', 'AttentionNetwork', 'MemoryNetwork',
    
    # Communication
    'NeuralCommunication', 'SharedTensorManager',
    
    # Optimization
    'CognitiveCompiler', 'GPUMemoryManager',
    
    # Training
    'NeuralTrainer', 'MetaTrainer'
]