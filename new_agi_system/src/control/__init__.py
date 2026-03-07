"""
控制模块

提供机器人控制功能，包括运动控制、硬件接口、传感器数据接入、电机控制等功能。
将原有Self-Soul系统的控制功能迁移到统一认知架构中。
"""

from .motion_control import MotionControlSystem
from .hardware_interface import HardwareControlSystem
from .sensor_integration import SensorIntegrationSystem
from .motor_control import MotorControlSystem

__all__ = [
    'MotionControlSystem',
    'HardwareControlSystem',
    'SensorIntegrationSystem',
    'MotorControlSystem'
]