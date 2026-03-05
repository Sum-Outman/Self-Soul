"""
Visual Video Processing Model
Specialized model for video stream analysis, enhancement, and generation with AGI capabilities
"""

from .unified_visual_video_model import UnifiedVisualVideoModel

# Backward compatibility with old imports
VideoVisionModel = UnifiedVisualVideoModel

__all__ = ['UnifiedVisualVideoModel', 'VideoVisionModel']