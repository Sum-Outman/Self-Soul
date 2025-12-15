# Video processing models package
# 视频处理模型包
# Contains models for video stream processing and analysis
# 包含视频流处理和分析模型

# Import the actual video model implementation
from .unified_video_model import UnifiedVideoModel as VideoModel

# For compatibility with existing imports
# 为了与现有导入兼容
VideoVisionModel = VideoModel

__all__ = ['VideoModel', 'VideoVisionModel']
