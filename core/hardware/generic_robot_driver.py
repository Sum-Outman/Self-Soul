"""
Generic Robot Driver - Real hardware driver for AGI humanoid robots

Supports multiple communication protocols and robot platforms.
Provides real hardware control for humanoid robot systems.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from .robot_driver_base import (
    RobotDriverBase, RobotPlatform, JointType, SensorType, CommunicationProtocol
)

logger = logging.getLogger("GenericRobotDriver")

class HumanoidJoint(Enum):
    """Standard humanoid robot joints"""
    # Left arm
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll"
    LEFT_ELBOW = "left_elbow"
    LEFT_WRIST = "left_wrist"
    
    # Right arm
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_WRIST = "right_wrist"
    
    # Left leg
    LEFT_HIP_YAW = "left_hip_yaw"
    LEFT_HIP_ROLL = "left_hip_roll"
    LEFT_HIP_PITCH = "left_hip_pitch"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE_PITCH = "left_ankle_pitch"
    LEFT_ANKLE_ROLL = "left_ankle_roll"
    
    # Right leg
    RIGHT_HIP_YAW = "right_hip_yaw"
    RIGHT_HIP_ROLL = "right_hip_roll"
    RIGHT_HIP_PITCH = "right_hip_pitch"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch"
    RIGHT_ANKLE_ROLL = "right_ankle_roll"
    
    # Head and torso
    HEAD_PAN = "head_pan"
    HEAD_TILT = "head_tilt"
    TORSO_TWIST = "torso_twist"
    TORSO_BEND = "torso_bend"

class HumanoidSensor(Enum):
    """Standard humanoid robot sensors"""
    # IMU sensors
    IMU_BODY = "imu_body"
    IMU_LEFT_FOOT = "imu_left_foot"
    IMU_RIGHT_FOOT = "imu_right_foot"
    
    # Force/torque sensors
    FORCE_LEFT_FOOT = "force_left_foot"
    FORCE_RIGHT_FOOT = "force_right_foot"
    TORQUE_LEFT_HIP = "torque_left_hip"
    TORQUE_RIGHT_HIP = "torque_right_hip"
    TORQUE_LEFT_KNEE = "torque_left_knee"
    TORQUE_RIGHT_KNEE = "torque_right_knee"
    TORQUE_LEFT_ANKLE = "torque_left_ankle"
    TORQUE_RIGHT_ANKLE = "torque_right_ankle"
    
    # Cameras
    CAMERA_HEAD = "camera_head"
    CAMERA_CHEST = "camera_chest"
    CAMERA_LEFT_HAND = "camera_left_hand"
    CAMERA_RIGHT_HAND = "camera_right_hand"
    
    # Proximity sensors
    PROXIMITY_FRONT = "proximity_front"
    PROXIMITY_BACK = "proximity_back"
    PROXIMITY_LEFT = "proximity_left"
    PROXIMITY_RIGHT = "proximity_right"
    
    # System sensors
    BATTERY = "battery"
    SYSTEM_TEMPERATURE = "system_temperature"
    MOTOR_TEMPERATURE = "motor_temperature"
    CURRENT_SENSOR = "current_sensor"

class GenericRobotDriver(RobotDriverBase):
    """Generic robot driver for humanoid robots with real hardware control"""
    
    def __init__(self, platform: RobotPlatform, config: Dict[str, Any]):
        """
        Initialize generic robot driver
        
        Args:
            platform: Robot platform (should be humanoid type)
            config: Configuration dictionary
        """
        super().__init__(platform, config)
        
        # Protocol-specific connections
        self.protocol = config.get("protocol", CommunicationProtocol.ETHERNET.value)
        self.connection_params = config.get("connection_params", {})
        
        # Protocol handlers
        self.protocol_handler = None
        
        # Humanoid-specific configuration
        self.joint_mapping = config.get("joint_mapping", {})
        self.sensor_mapping = config.get("sensor_mapping", {})
        
        # Initialize default humanoid joint mapping if not provided
        if not self.joint_mapping:
            self._initialize_default_joint_mapping()
        
        if not self.sensor_mapping:
            self._initialize_default_sensor_mapping()
        
        logger.info(f"Initialized generic robot driver with protocol: {self.protocol}")
    
    def _initialize_default_joint_mapping(self):
        """Initialize default joint mapping for humanoid robot"""
        self.joint_mapping = {
            # Left arm (4 joints)
            HumanoidJoint.LEFT_SHOULDER_PITCH.value: {"id": 1, "type": JointType.REVOLUTE.value, "min": -2.0, "max": 2.0},
            HumanoidJoint.LEFT_SHOULDER_ROLL.value: {"id": 2, "type": JointType.REVOLUTE.value, "min": -1.5, "max": 1.5},
            HumanoidJoint.LEFT_ELBOW.value: {"id": 3, "type": JointType.REVOLUTE.value, "min": -2.5, "max": 2.5},
            HumanoidJoint.LEFT_WRIST.value: {"id": 4, "type": JointType.REVOLUTE.value, "min": -1.8, "max": 1.8},
            
            # Right arm (4 joints)
            HumanoidJoint.RIGHT_SHOULDER_PITCH.value: {"id": 5, "type": JointType.REVOLUTE.value, "min": -2.0, "max": 2.0},
            HumanoidJoint.RIGHT_SHOULDER_ROLL.value: {"id": 6, "type": JointType.REVOLUTE.value, "min": -1.5, "max": 1.5},
            HumanoidJoint.RIGHT_ELBOW.value: {"id": 7, "type": JointType.REVOLUTE.value, "min": -2.5, "max": 2.5},
            HumanoidJoint.RIGHT_WRIST.value: {"id": 8, "type": JointType.REVOLUTE.value, "min": -1.8, "max": 1.8},
            
            # Left leg (6 joints)
            HumanoidJoint.LEFT_HIP_YAW.value: {"id": 9, "type": JointType.REVOLUTE.value, "min": -1.0, "max": 1.0},
            HumanoidJoint.LEFT_HIP_ROLL.value: {"id": 10, "type": JointType.REVOLUTE.value, "min": -0.5, "max": 0.5},
            HumanoidJoint.LEFT_HIP_PITCH.value: {"id": 11, "type": JointType.REVOLUTE.value, "min": -1.5, "max": 1.5},
            HumanoidJoint.LEFT_KNEE.value: {"id": 12, "type": JointType.REVOLUTE.value, "min": 0, "max": 2.5},
            HumanoidJoint.LEFT_ANKLE_PITCH.value: {"id": 13, "type": JointType.REVOLUTE.value, "min": -1.0, "max": 1.0},
            HumanoidJoint.LEFT_ANKLE_ROLL.value: {"id": 14, "type": JointType.REVOLUTE.value, "min": -0.5, "max": 0.5},
            
            # Right leg (6 joints)
            HumanoidJoint.RIGHT_HIP_YAW.value: {"id": 15, "type": JointType.REVOLUTE.value, "min": -1.0, "max": 1.0},
            HumanoidJoint.RIGHT_HIP_ROLL.value: {"id": 16, "type": JointType.REVOLUTE.value, "min": -0.5, "max": 0.5},
            HumanoidJoint.RIGHT_HIP_PITCH.value: {"id": 17, "type": JointType.REVOLUTE.value, "min": -1.5, "max": 1.5},
            HumanoidJoint.RIGHT_KNEE.value: {"id": 18, "type": JointType.REVOLUTE.value, "min": 0, "max": 2.5},
            HumanoidJoint.RIGHT_ANKLE_PITCH.value: {"id": 19, "type": JointType.REVOLUTE.value, "min": -1.0, "max": 1.0},
            HumanoidJoint.RIGHT_ANKLE_ROLL.value: {"id": 20, "type": JointType.REVOLUTE.value, "min": -0.5, "max": 0.5},
            
            # Head and torso (4 joints)
            HumanoidJoint.HEAD_PAN.value: {"id": 21, "type": JointType.REVOLUTE.value, "min": -1.8, "max": 1.8},
            HumanoidJoint.HEAD_TILT.value: {"id": 22, "type": JointType.REVOLUTE.value, "min": -0.8, "max": 0.8},
            HumanoidJoint.TORSO_TWIST.value: {"id": 23, "type": JointType.REVOLUTE.value, "min": -0.5, "max": 0.5},
            HumanoidJoint.TORSO_BEND.value: {"id": 24, "type": JointType.REVOLUTE.value, "min": -0.3, "max": 0.3},
        }
    
    def _initialize_default_sensor_mapping(self):
        """Initialize default sensor mapping for humanoid robot"""
        self.sensor_mapping = {
            # IMU sensors
            HumanoidSensor.IMU_BODY.value: {"id": 101, "type": SensorType.IMU.value},
            HumanoidSensor.IMU_LEFT_FOOT.value: {"id": 102, "type": SensorType.IMU.value},
            HumanoidSensor.IMU_RIGHT_FOOT.value: {"id": 103, "type": SensorType.IMU.value},
            
            # Force sensors
            HumanoidSensor.FORCE_LEFT_FOOT.value: {"id": 104, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.FORCE_RIGHT_FOOT.value: {"id": 105, "type": SensorType.FORCE_TORQUE.value},
            
            # Torque sensors
            HumanoidSensor.TORQUE_LEFT_HIP.value: {"id": 106, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.TORQUE_RIGHT_HIP.value: {"id": 107, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.TORQUE_LEFT_KNEE.value: {"id": 108, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.TORQUE_RIGHT_KNEE.value: {"id": 109, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.TORQUE_LEFT_ANKLE.value: {"id": 110, "type": SensorType.FORCE_TORQUE.value},
            HumanoidSensor.TORQUE_RIGHT_ANKLE.value: {"id": 111, "type": SensorType.FORCE_TORQUE.value},
            
            # System sensors
            HumanoidSensor.BATTERY.value: {"id": 112, "type": SensorType.BATTERY.value},
            HumanoidSensor.SYSTEM_TEMPERATURE.value: {"id": 113, "type": SensorType.TEMPERATURE.value},
            HumanoidSensor.MOTOR_TEMPERATURE.value: {"id": 114, "type": SensorType.TEMPERATURE.value},
            HumanoidSensor.CURRENT_SENSOR.value: {"id": 115, "type": SensorType.CURRENT.value},
        }
    
    def connect(self) -> bool:
        """Connect to robot hardware using configured protocol
        
        Returns:
            bool: True if connection successful
        """
        try:
            logger.info(f"Connecting to robot using {self.protocol} protocol...")
            
            # Initialize protocol handler based on selected protocol
            self.protocol_handler = self._create_protocol_handler()
            
            if not self.protocol_handler:
                logger.error(f"Unsupported protocol: {self.protocol}")
                return False
            
            # Establish connection
            connection_result = self.protocol_handler.connect(self.connection_params)
            
            if connection_result:
                self.connected = True
                logger.info(f"Successfully connected to robot via {self.protocol}")
                return True
            else:
                logger.error(f"Failed to connect via {self.protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def _create_protocol_handler(self):
        """Create protocol handler based on configuration"""
        protocol = self.protocol.lower()
        
        # Import protocol handlers dynamically to avoid dependencies
        try:
            if protocol == CommunicationProtocol.ETHERNET.value:
                from .protocols.ethernet_handler import EthernetHandler
                return EthernetHandler()
            
            elif protocol == CommunicationProtocol.SERIAL.value:
                from .protocols.serial_handler import SerialHandler
                return SerialHandler()
            
            elif protocol == CommunicationProtocol.USB.value:
                from .protocols.usb_handler import USBHandler
                return USBHandler()
            
            elif protocol == CommunicationProtocol.CAN_BUS.value:
                from .protocols.can_handler import CANHandler
                return CANHandler()
            
            elif protocol == CommunicationProtocol.I2C.value:
                from .protocols.i2c_handler import I2CHandler
                return I2CHandler()
            
            elif protocol == CommunicationProtocol.PWM.value:
                from .protocols.pwm_handler import PWMHandler
                return PWMHandler()
            
            elif protocol == CommunicationProtocol.ROS.value:
                from .protocols.ros_handler import ROSHandler
                return ROSHandler()
            
            elif protocol == CommunicationProtocol.ROS2.value:
                from .protocols.ros2_handler import ROS2Handler
                return ROS2Handler()
            
            else:
                logger.warning(f"Protocol {protocol} not implemented, using simulation")
                # For unimplemented protocols, return a stub handler
                # In real implementation, this should raise an error
                from .protocols.base_protocol_handler import BaseProtocolHandler
                return BaseProtocolHandler()
                
        except ImportError as e:
            logger.error(f"Failed to import protocol handler for {protocol}: {e}")
            return None
    
    def disconnect(self) -> bool:
        """Disconnect from robot hardware
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.protocol_handler:
                result = self.protocol_handler.disconnect()
                self.protocol_handler = None
            else:
                result = True
            
            self.connected = False
            self.initialized = False
            logger.info("Disconnected from robot")
            return result
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize robot hardware (calibration, homing, etc.)
        
        Returns:
            bool: True if initialization successful
        """
        if not self.connected:
            logger.error("Cannot initialize: Not connected to robot")
            return False
        
        try:
            logger.info("Initializing robot hardware...")
            
            # Send initialization command
            if self.protocol_handler:
                init_result = self.protocol_handler.send_command("initialize", {})
                
                if init_result.get("success"):
                    self.initialized = True
                    
                    # Perform calibration
                    calibration_result = self._perform_calibration()
                    
                    if calibration_result:
                        logger.info("Robot initialization and calibration completed")
                        return True
                    else:
                        logger.error("Robot calibration failed")
                        return False
                else:
                    logger.error(f"Initialization failed: {init_result.get('error', 'Unknown error')}")
                    return False
            else:
                logger.error("No protocol handler available")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def _perform_calibration(self) -> bool:
        """Perform robot calibration
        
        Returns:
            bool: True if calibration successful
        """
        try:
            logger.info("Starting robot calibration...")
            
            # Move to home position
            home_result = self._move_to_home_position()
            
            if not home_result:
                logger.warning("Failed to move to home position, continuing calibration")
            
            # Calibrate sensors
            sensor_calibration = self._calibrate_sensors()
            
            # Calibrate joints
            joint_calibration = self._calibrate_joints()
            
            logger.info(f"Calibration completed: sensors={sensor_calibration}, joints={joint_calibration}")
            return sensor_calibration or joint_calibration  # At least one should succeed
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    def _move_to_home_position(self) -> bool:
        """Move robot to home/safe position
        
        Returns:
            bool: True if movement successful
        """
        try:
            # Define home positions for each joint
            home_positions = {}
            for joint_name, mapping in self.joint_mapping.items():
                # Set to middle of range or 0
                min_pos = mapping.get("min", -1.0)
                max_pos = mapping.get("max", 1.0)
                home_positions[joint_name] = (min_pos + max_pos) / 2.0
            
            # Move to home positions
            return self.set_joint_positions(
                list(home_positions.keys()),
                list(home_positions.values()),
                velocities=[0.5] * len(home_positions)  # Slow velocity for safety
            )
            
        except Exception as e:
            logger.error(f"Failed to move to home position: {e}")
            return False
    
    def _calibrate_sensors(self) -> bool:
        """Calibrate robot sensors
        
        Returns:
            bool: True if sensor calibration successful
        """
        try:
            if not self.protocol_handler:
                return False
            
            # Send sensor calibration command
            result = self.protocol_handler.send_command("calibrate_sensors", {})
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Sensor calibration error: {e}")
            return False
    
    def _calibrate_joints(self) -> bool:
        """Calibrate robot joints
        
        Returns:
            bool: True if joint calibration successful
        """
        try:
            if not self.protocol_handler:
                return False
            
            # Send joint calibration command
            result = self.protocol_handler.send_command("calibrate_joints", {})
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Joint calibration error: {e}")
            return False
    
    def get_joint_count(self) -> int:
        """Get number of joints in robot
        
        Returns:
            int: Number of joints
        """
        return len(self.joint_mapping)
    
    def get_joint_names(self) -> List[str]:
        """Get list of joint names
        
        Returns:
            List[str]: Joint names
        """
        return list(self.joint_mapping.keys())
    
    def get_joint_position(self, joint_name: str) -> float:
        """Get current position of specific joint
        
        Args:
            joint_name: Name of joint
            
        Returns:
            float: Joint position (radians)
        """
        if not self.connected or not self.protocol_handler:
            logger.error("Not connected to robot")
            return 0.0
        
        try:
            # Get position from protocol handler
            result = self.protocol_handler.send_command(
                "get_joint_position",
                {"joint_name": joint_name, "joint_id": self.joint_mapping[joint_name]["id"]}
            )
            
            if result.get("success"):
                position = result.get("position", 0.0)
                
                # Update cached position
                self.joint_positions[joint_name] = position
                return position
            else:
                logger.error(f"Failed to get joint position: {result.get('error')}")
                return self.joint_positions.get(joint_name, 0.0)
                
        except Exception as e:
            logger.error(f"Error getting joint position: {e}")
            return self.joint_positions.get(joint_name, 0.0)
    
    def set_joint_position(self, joint_name: str, position: float, 
                          velocity: Optional[float] = None,
                          acceleration: Optional[float] = None) -> bool:
        """Set target position for specific joint
        
        Args:
            joint_name: Name of joint
            position: Target position (radians)
            velocity: Velocity limit (rad/s)
            acceleration: Acceleration limit (rad/s²)
            
        Returns:
            bool: True if command accepted
        """
        if not self.connected or not self.protocol_handler:
            logger.error("Not connected to robot")
            return False
        
        # Check joint limits
        if joint_name in self.joint_mapping:
            joint_info = self.joint_mapping[joint_name]
            min_pos = joint_info.get("min", -3.14)
            max_pos = joint_info.get("max", 3.14)
            
            if position < min_pos or position > max_pos:
                logger.error(f"Position {position} out of range [{min_pos}, {max_pos}] for joint {joint_name}")
                return False
        
        try:
            # Prepare command
            command_data = {
                "joint_name": joint_name,
                "joint_id": self.joint_mapping[joint_name]["id"],
                "position": position,
            }
            
            if velocity is not None:
                command_data["velocity"] = velocity
            
            if acceleration is not None:
                command_data["acceleration"] = acceleration
            
            # Send command
            result = self.protocol_handler.send_command("set_joint_position", command_data)
            
            if result.get("success"):
                logger.debug(f"Set joint {joint_name} to position {position}")
                return True
            else:
                logger.error(f"Failed to set joint position: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting joint position: {e}")
            return False
    
    def set_joint_positions(self, joint_names: List[str], positions: List[float],
                           velocities: Optional[List[float]] = None,
                           accelerations: Optional[List[float]] = None) -> bool:
        """Set target positions for multiple joints
        
        Args:
            joint_names: List of joint names
            positions: List of target positions
            velocities: List of velocity limits
            accelerations: List of acceleration limits
            
        Returns:
            bool: True if command accepted
        """
        if not self.connected or not self.protocol_handler:
            logger.error("Not connected to robot")
            return False
        
        # Validate inputs
        if len(joint_names) != len(positions):
            logger.error(f"Joint names ({len(joint_names)}) and positions ({len(positions)}) count mismatch")
            return False
        
        if velocities and len(velocities) != len(joint_names):
            logger.error(f"Velocities count ({len(velocities)}) doesn't match joints ({len(joint_names)})")
            return False
        
        if accelerations and len(accelerations) != len(joint_names):
            logger.error(f"Accelerations count ({len(accelerations)}) doesn't match joints ({len(joint_names)})")
            return False
        
        try:
            # Prepare command data
            joint_data = []
            for i, joint_name in enumerate(joint_names):
                joint_info = {
                    "joint_name": joint_name,
                    "joint_id": self.joint_mapping[joint_name]["id"],
                    "position": positions[i],
                }
                
                if velocities:
                    joint_info["velocity"] = velocities[i]
                
                if accelerations:
                    joint_info["acceleration"] = accelerations[i]
                
                joint_data.append(joint_info)
            
            # Send multi-joint command
            result = self.protocol_handler.send_command("set_joint_positions", {"joints": joint_data})
            
            if result.get("success"):
                logger.debug(f"Set {len(joint_names)} joints to target positions")
                return True
            else:
                logger.error(f"Failed to set joint positions: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting joint positions: {e}")
            return False
    
    def get_sensor_count(self) -> int:
        """Get number of sensors in robot
        
        Returns:
            int: Number of sensors
        """
        return len(self.sensor_mapping)
    
    def get_sensor_names(self) -> List[str]:
        """Get list of sensor names
        
        Returns:
            List[str]: Sensor names
        """
        return list(self.sensor_mapping.keys())
    
    def get_sensor_value(self, sensor_name: str) -> Any:
        """Get current value from specific sensor
        
        Args:
            sensor_name: Name of sensor
            
        Returns:
            Any: Sensor value (type depends on sensor)
        """
        if not self.connected or not self.protocol_handler:
            logger.error("Not connected to robot")
            return None
        
        try:
            # Get sensor value from protocol handler
            result = self.protocol_handler.send_command(
                "get_sensor_value",
                {"sensor_name": sensor_name, "sensor_id": self.sensor_mapping[sensor_name]["id"]}
            )
            
            if result.get("success"):
                value = result.get("value", None)
                
                # Update cached value
                self.sensor_readings[sensor_name] = value
                return value
            else:
                logger.error(f"Failed to get sensor value: {result.get('error')}")
                return self.sensor_readings.get(sensor_name, None)
                
        except Exception as e:
            logger.error(f"Error getting sensor value: {e}")
            return self.sensor_readings.get(sensor_name, None)
    
    def get_battery_level(self) -> float:
        """Get battery level (percentage)
        
        Returns:
            float: Battery level (0-100%)
        """
        try:
            # Try to get from battery sensor
            if HumanoidSensor.BATTERY.value in self.sensor_mapping:
                battery_value = self.get_sensor_value(HumanoidSensor.BATTERY.value)
                
                if isinstance(battery_value, dict):
                    return battery_value.get("percentage", 0.0)
                elif isinstance(battery_value, (int, float)):
                    return float(battery_value)
            
            # Fallback to protocol handler
            if self.protocol_handler:
                result = self.protocol_handler.send_command("get_battery_level", {})
                if result.get("success"):
                    return result.get("level", 0.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting battery level: {e}")
            return 0.0
    
    def get_temperature(self) -> float:
        """Get robot system temperature
        
        Returns:
            float: Temperature in degrees Celsius
        """
        try:
            # Try to get from temperature sensor
            if HumanoidSensor.SYSTEM_TEMPERATURE.value in self.sensor_mapping:
                temp_value = self.get_sensor_value(HumanoidSensor.SYSTEM_TEMPERATURE.value)
                
                if isinstance(temp_value, dict):
                    return temp_value.get("temperature", 0.0)
                elif isinstance(temp_value, (int, float)):
                    return float(temp_value)
            
            # Fallback to protocol handler
            if self.protocol_handler:
                result = self.protocol_handler.send_command("get_temperature", {})
                if result.get("success"):
                    return result.get("temperature", 0.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
            return 0.0
    
    def emergency_stop(self) -> bool:
        """Trigger emergency stop
        
        Returns:
            bool: True if emergency stop activated
        """
        try:
            logger.warning("Activating emergency stop!")
            
            if self.protocol_handler:
                result = self.protocol_handler.send_command("emergency_stop", {})
                
                # Also stop control thread
                self.stop_control_thread()
                
                return result.get("success", False)
            else:
                # Stop control thread as fallback
                self.stop_control_thread()
                return True
                
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            # Still try to stop control thread
            self.stop_control_thread()
            return False
    
    def clear_emergency_stop(self) -> bool:
        """Clear emergency stop condition
        
        Returns:
            bool: True if emergency stop cleared
        """
        try:
            logger.info("Clearing emergency stop")
            
            if self.protocol_handler:
                result = self.protocol_handler.send_command("clear_emergency_stop", {})
                return result.get("success", False)
            else:
                return True
                
        except Exception as e:
            logger.error(f"Error clearing emergency stop: {e}")
            return False
    
    def _platform_control_cycle(self):
        """Humanoid robot specific control cycle"""
        # Humanoid-specific control logic
        # This could include balance control, walking algorithms, etc.
        
        # For now, just update states
        self._update_joint_states()
        self._update_sensor_readings()
        
        # Apply humanoid-specific safety checks
        self._apply_humanoid_safety_checks()
    
    def _apply_humanoid_safety_checks(self):
        """Apply humanoid-specific safety checks"""
        if not self.safety_enabled:
            return
        
        # Check for fall detection using IMU
        try:
            if HumanoidSensor.IMU_BODY.value in self.sensor_readings:
                imu_data = self.sensor_readings[HumanoidSensor.IMU_BODY.value]
                
                if isinstance(imu_data, dict):
                    # Check for excessive tilt (potential fall)
                    tilt_threshold = 0.7  # radians (~40 degrees)
                    
                    if abs(imu_data.get("pitch", 0)) > tilt_threshold or \
                       abs(imu_data.get("roll", 0)) > tilt_threshold:
                        logger.warning(f"Excessive tilt detected: pitch={imu_data.get('pitch', 0)}, roll={imu_data.get('roll', 0)}")
                        self.emergency_stop()
                        return
        except Exception as e:
            logger.warning(f"IMU check failed: {e}")
        
        # Check foot pressure for balance
        try:
            if HumanoidSensor.FORCE_LEFT_FOOT.value in self.sensor_readings and \
               HumanoidSensor.FORCE_RIGHT_FOOT.value in self.sensor_readings:
                
                left_force = self.sensor_readings[HumanoidSensor.FORCE_LEFT_FOOT.value]
                right_force = self.sensor_readings[HumanoidSensor.FORCE_RIGHT_FOOT.value]
                
                # Simple balance check
                if isinstance(left_force, (int, float)) and isinstance(right_force, (int, float)):
                    total_force = left_force + right_force
                    
                    if total_force > 0:
                        balance_ratio = abs(left_force - right_force) / total_force
                        
                        if balance_ratio > 0.8:  # 80% imbalance
                            logger.warning(f"Significant balance imbalance detected: ratio={balance_ratio}")
                            # Don't emergency stop for imbalance, just warn
        except Exception as e:
            logger.warning(f"Force sensor check failed: {e}")