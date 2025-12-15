"""
Base Model Components - Modular AGI Foundation Architecture

This package provides a modular foundation for all AGI models, 
breaking down the monolithic base classes into focused, reusable components.

Components:
- performance_mixin.py: Performance monitoring and optimization
- external_api_mixin.py: External API integration and management  
- cache_mixin.py: Caching and optimization strategies
- error_handling_mixin.py: Error handling and recovery mechanisms
- resource_mixin.py: Resource management and monitoring
- training_mixin.py: Training lifecycle and optimization
- agi_core_mixin.py: AGI-specific cognitive capabilities
- composite_base_model.py: Complete AGI model with all capabilities
"""

from .unified_mixins import UnifiedPerformanceCacheMixin as PerformanceMixin
from .unified_mixins import UnifiedExternalAPIMixin as ExternalAPIMixin
from .unified_mixins import UnifiedPerformanceCacheMixin as CacheMixin
from .unified_mixins import UnifiedErrorResourceMixin as ErrorHandlingMixin
from .unified_mixins import UnifiedErrorResourceMixin as ResourceMixin
from .training_mixin import TrainingLifecycleMixin as TrainingMixin
from .agi_core_mixin import AGICoreCapabilitiesMixin as AGICoreMixin

# Composite base model that includes all mixins
from .composite_base_model import CompositeBaseModel

__all__ = [
    'PerformanceMixin',
    'ExternalAPIMixin', 
    'CacheMixin',
    'ErrorHandlingMixin',
    'ResourceMixin',
    'TrainingMixin',
    'AGICoreMixin',
    'CompositeBaseModel'
]
