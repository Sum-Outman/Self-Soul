# Motion control models package
# 运动控制模型包
# Contains models for motion and actuator control
# 包含运动和执行器控制模型

# Import the actual motion model implementation
from .unified_motion_model import UnifiedMotionModel as MotionModel

__all__ = ['MotionModel']
