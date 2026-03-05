"""
Robot Hardware Interface Module

Provides hardware interfaces for robot control, including:
- Robot hardware interface
- External device interface
- Camera manager
- Protocol handlers
- Robot drivers
"""

from .robot_hardware_interface import RobotHardwareInterface
from .external_device_interface import ExternalDeviceInterface
from .camera_manager import CameraManager

# Protocol handlers
try:
    from .protocols.ethernet_handler import EthernetHandler
    from .protocols.serial_handler import SerialHandler
    from .protocols.usb_handler import USBHandler
    from .protocols.can_handler import CANHandler
    from .protocols.i2c_handler import I2CHandler
    from .protocols.pwm_handler import PWMHandler
    from .protocols.ros_handler import ROSHandler
    from .protocols.ros2_handler import ROS2Handler
except ImportError:
    # Protocol handlers may not be available
    pass

# Robot drivers
try:
    from .robot_driver_base import RobotDriverBase, RobotPlatform, JointType, SensorType, CommunicationProtocol
    from .generic_robot_driver import GenericRobotDriver, HumanoidJoint, HumanoidSensor
except ImportError:
    # Robot drivers may not be available
    pass

__all__ = [
    # Core interfaces
    "RobotHardwareInterface",
    "ExternalDeviceInterface", 
    "CameraManager",
    
    # Protocol handlers
    "EthernetHandler",
    "SerialHandler",
    "USBHandler",
    "CANHandler",
    "I2CHandler",
    "PWMHandler",
    "ROSHandler",
    "ROS2Handler",
    
    # Robot drivers
    "RobotDriverBase",
    "RobotPlatform",
    "JointType",
    "SensorType",
    "CommunicationProtocol",
    "GenericRobotDriver",
    "HumanoidJoint",
    "HumanoidSensor",
]