"""
Cognitive Components Module

Contains all cognitive components for the unified AGI architecture:
- Perception: Multimodal input processing
- Attention: Hierarchical attention mechanisms
- Memory: Episodic and semantic memory systems
- Reasoning: Universal reasoning engine
- Planning: Hierarchical planning system
- Decision: Value-based decision making
- Action: Adaptive action execution
- Learning: Meta-learning system
"""

from .architecture import UnifiedCognitiveArchitecture
from .representation import UnifiedRepresentationSpace
from .perception import MultimodalPerception
from .attention import HierarchicalAttention
from .memory import EpisodicSemanticMemory
from .reasoning import UniversalReasoningEngine
from .planning import HierarchicalPlanning
from .decision import ValueBasedDecision
from .action import AdaptiveAction
from .learning import MetaLearningSystem

__all__ = [
    'UnifiedCognitiveArchitecture',
    'UnifiedRepresentationSpace',
    'MultimodalPerception',
    'HierarchicalAttention',
    'EpisodicSemanticMemory',
    'UniversalReasoningEngine',
    'HierarchicalPlanning',
    'ValueBasedDecision',
    'AdaptiveAction',
    'MetaLearningSystem'
]