"""
Robot Hardware Interface Module - AGI-enhanced robot hardware integration and control

Provides unified interface for robot hardware devices including sensors, cameras, motor controllers, servo drivers, etc.
Integrates existing external device interfaces and camera managers to provide advanced robot-specific hardware control capabilities.

AGI-enhanced features:
- Intelligent hardware discovery and auto-configuration
- Real-time sensor data fusion and analysis
- Adaptive motor control and motion planning
- Predictive hardware behavior modeling
- Autonomous hardware management and optimization
- Multi-protocol seamless switching
- Real-time performance optimization
- Fault tolerance and self-healing
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import numpy as np

# Import existing hardware interface modules
try:
    from .external_device_interface import ExternalDeviceInterface
    from .camera_manager import CameraManager
except ImportError as e:
    logging.error(f"Failed to import hardware modules: {e}")
    ExternalDeviceInterface = None
    CameraManager = None

# Import new robot driver modules for real hardware control
try:
    from .robot_driver_base import RobotDriverBase, RobotPlatform, JointType, SensorType, CommunicationProtocol
    from .generic_robot_driver import GenericRobotDriver, HumanoidJoint, HumanoidSensor
    ROBOT_DRIVER_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import robot driver modules: {e}")
    RobotDriverBase = None
    GenericRobotDriver = None
    ROBOT_DRIVER_AVAILABLE = False

# Import robot learning system for AGI-enhanced capabilities
try:
    from ..robot_learning_system import RobotLearningSystem
    ROBOT_LEARNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Robot learning system not available: {e}")
    RobotLearningSystem = None
    ROBOT_LEARNING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobotHardwareInterface")

class RobotHardwareInterface:
    """AGI-enhanced Robot Hardware Interface"""
    
    def __init__(self, use_robot_driver: bool = True):
       """Initialize robot hardware interface
        
       Args:
           use_robot_driver: If True, use the new GenericRobotDriver for real hardware control.
                             If False, use legacy ExternalDeviceInterface for compatibility.
       """
       self.device_interface = None
       self.camera_manager = None
       self.robot_driver = None
       self.hardware_initialized = False
       self.use_robot_driver = use_robot_driver and ROBOT_DRIVER_AVAILABLE
       
       # Robot hardware configuration
       self.config = {
           "sensor_update_rate": 100,      # Sensor update frequency (Hz)
           "motor_control_rate": 200,      # Motor control frequency (Hz)
           "camera_fps": 60,               # Camera frame rate
           "max_motor_current": 5.0,       # Maximum motor current (A)
           "max_servo_torque": 10.0,       # Maximum servo torque (Nm)
           "emergency_stop_timeout": 0.1,  # Emergency stop timeout (seconds)
           "battery_monitoring": True,     # Battery monitoring
           "temperature_monitoring": True, # Temperature monitoring
           "hardware_safety": True,        # Hardware safety protection
       }
       
       # Hardware device registry
       self.sensors = {}           # Sensor devices
       self.cameras = {}           # Camera devices
       self.motors = {}            # Motor devices
       self.servos = {}            # Servo devices
       self.controllers = {}       # Controller devices
       
       # Real-time data cache
       self.sensor_data = {}       # Sensor data
       self.camera_frames = {}     # Camera frame data
       self.motor_states = {}      # Motor states
       self.servo_positions = {}   # Servo positions
       
       # Control threads and locks
       self.control_threads = {}
       self.data_lock = threading.Lock()
       self.control_lock = threading.Lock()
       
       # Performance metrics
       self.performance_metrics = {
           "sensor_latency": 0.0,
           "control_latency": 0.0,
           "data_throughput": 0.0,
           "hardware_availability": 1.0,
           "error_rate": 0.0,
       }
       
       # Safety system
       self.safety_system = {
           "emergency_stop": False,
           "overcurrent_protection": True,
           "overtemperature_protection": True,
           "position_limit": True,
           "velocity_limit": True,
       }
       
       # Learning system for AGI-enhanced capabilities
       self.learning_system = None
       if ROBOT_LEARNING_AVAILABLE and RobotLearningSystem:
           try:
               self.learning_system = RobotLearningSystem(self, None)
               logger.info("Robot learning system initialized")
           except Exception as e:
               logger.warning(f"Failed to initialize learning system: {e}")
       
       # Hardware initialization will be done lazily when needed
       # self._initialize_hardware_interfaces() is called via initialize() method
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize hardware interfaces lazily
        
        Returns:
            Dictionary with success status and optional error message
        """
        try:
            return self._initialize_hardware_interfaces()
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return {"success": False, "error": str(e), "initialized": False}
    
    def _initialize_hardware_interfaces(self) -> Dict[str, Any]:
       """Initialize hardware interfaces"""
       try:
           # Try to initialize robot driver first if enabled
           if self.use_robot_driver and GenericRobotDriver:
               try:
                   # Configuration for humanoid robot driver
                   driver_config = {
                       "protocol": "ethernet",  # Default protocol, can be configured via environment
                       "control_rate": self.config["motor_control_rate"],
                       "update_rate": self.config["sensor_update_rate"],
                       "max_velocity": 5.0,
                       "max_torque": self.config["max_servo_torque"],
                       "max_current": self.config["max_motor_current"],
                       "connection_params": {
                           "host": "localhost",  # Default, should be configured
                           "port": 5000,         # Default port for robot control
                       }
                   }
                   
                   # Check for environment configuration
                   import os
                   robot_host = os.environ.get('ROBOT_HOST', 'localhost')
                   robot_port = os.environ.get('ROBOT_PORT', '5000')
                   robot_protocol = os.environ.get('ROBOT_PROTOCOL', 'ethernet')
                   
                   driver_config["protocol"] = robot_protocol
                   driver_config["connection_params"]["host"] = robot_host
                   driver_config["connection_params"]["port"] = int(robot_port)
                   
                   # Create robot driver
                   self.robot_driver = GenericRobotDriver(
                       platform=RobotPlatform.HUMANOID,
                       config=driver_config
                   )
                   
                   # Connect to robot hardware
                   if self.robot_driver.connect():
                       logger.info("Robot driver connected successfully to real hardware")
                       # Initialize robot hardware
                       if self.robot_driver.initialize():
                           logger.info("Robot hardware initialized successfully via robot driver")
                           # Start control thread
                           self.robot_driver.start_control_thread()
                           logger.info("Robot control thread started")
                       else:
                           logger.warning("Robot driver initialization failed, falling back to legacy interface")
                           self.robot_driver = None
                   else:
                       logger.warning("Robot driver connection failed, falling back to legacy interface")
                       self.robot_driver = None
                       
               except Exception as e:
                   logger.error(f"Robot driver initialization error: {e}")
                   self.robot_driver = None
           
           # Initialize legacy device interface if robot driver not available
           if not self.robot_driver and ExternalDeviceInterface:
               self.device_interface = ExternalDeviceInterface()
               self.device_interface.initialize()
               logger.info("External device interface initialized successfully")
           
           if CameraManager:
               self.camera_manager = CameraManager()
               logger.info("Camera manager initialized successfully")
           
           self.hardware_initialized = True
           
           # Register sensors - real hardware only (simulation not supported)
           try:
               self._register_default_sensors()
           except Exception as e:
               logger.error(f"Sensor registration failed: {e}")
               self.hardware_initialized = False
               return {"success": False, "error": f"Sensor registration failed: {e}", "initialized": False}
           
           # Register servo motors - real hardware only (simulation not supported)
           try:
               self._register_default_servos()
           except Exception as e:
               logger.error(f"Servo registration failed: {e}")
               # Don't fail completely if servos fail, sensors might still work
           
           logger.info("Robot hardware interface initialization completed successfully")
           return {"success": True, "initialized": True, "hardware_available": True}
           
       except Exception as e:
           logger.error(f"Hardware interface initialization failed: {str(e)}")
           self.hardware_initialized = False
           return {"success": False, "error": str(e), "initialized": False}
    
    def _register_default_sensors(self):
       """Register default sensors - real hardware preferred, simulation allowed in development"""
       import os
       environment = os.environ.get('ENVIRONMENT', 'production').lower()
       robot_test_mode = os.environ.get('ROBOT_HARDWARE_TEST_MODE', 'false').lower() == 'true'
       allow_simulation = os.environ.get('ALLOW_ROBOT_SIMULATION', 'false').lower() == 'true'
       
       # Check if simulation is allowed in current environment
       simulation_allowed = environment != 'production' or robot_test_mode or allow_simulation
       
       if simulation_allowed:
           logger.warning(f"Simulation mode enabled for environment: {environment}")
           logger.info("Registering simulated sensors for development/testing purposes")
           return self._register_simulated_sensors_for_development()
       
       # Production environment or simulation not allowed - require real hardware
       logger.info("AGI robot requires real hardware sensors - checking hardware availability")
       
       # Check if device interface is available
       if not self.device_interface:
           raise RuntimeError("""
AGI Robot Hardware Interface Error: External device interface not initialized. 
Please ensure robot hardware drivers are installed and connect real humanoid robot hardware.

Required hardware for AGI robot control:
1. Servo motors for joints (minimum 12 for humanoid robot)
2. IMU sensors (accelerometer, gyroscope, magnetometer)
3. Force/torque sensors for balance
4. Camera systems for vision
5. Battery management system
6. Communication interfaces (USB, Serial, Ethernet, CAN bus)

For development/testing without real hardware, set environment variables:
- ENVIRONMENT=development
- ALLOW_ROBOT_SIMULATION=true
- ROBOT_HARDWARE_TEST_MODE=true

Installation steps for real hardware:
1. Install robot hardware drivers for your specific robot model
2. Configure hardware communication protocols
3. Initialize the robot hardware interface
4. Calibrate sensors and actuators
""")
       
       # Register real AGI robot sensors
       logger.info("Registering real AGI robot sensors...")
       
       # Define AGI robot sensor configuration
       agi_robot_sensors = [
           {
               "id": "imu_9dof",
               "type": "imu",
               "protocol": "i2c",
               "params": {"address": "0x68", "fusion_rate": "200Hz", "calibration": True}
           },
           {
               "id": "foot_pressure_left",
               "type": "force_sensor",
               "protocol": "i2c",
               "params": {"address": "0x40", "channels": 4, "range": "0-100kg"}
           },
           {
               "id": "foot_pressure_right",
               "type": "force_sensor",
               "protocol": "i2c",
               "params": {"address": "0x41", "channels": 4, "range": "0-100kg"}
           },
           {
               "id": "joint_torque_hip_left",
               "type": "torque_sensor",
               "protocol": "can",
               "params": {"can_id": 101, "range": "±50Nm", "update_rate": "1kHz"}
           },
           {
               "id": "joint_torque_knee_left",
               "type": "torque_sensor",
               "protocol": "can",
               "params": {"can_id": 102, "range": "±30Nm", "update_rate": "1kHz"}
           },
           {
               "id": "joint_torque_ankle_left",
               "type": "torque_sensor",
               "protocol": "can",
               "params": {"can_id": 103, "range": "±20Nm", "update_rate": "1kHz"}
           },
           {
               "id": "battery_system",
               "type": "battery",
               "protocol": "i2c",
               "params": {"address": "0x6B", "cells": 6, "capacity": "4000mAh"}
           },
           {
               "id": "system_temperature",
               "type": "temperature",
               "protocol": "i2c",
               "params": {"address": "0x48", "sensors": 4, "range": "0-100°C"}
           },
           {
               "id": "proximity_front",
               "type": "proximity",
               "protocol": "uart",
               "params": {"port": "/dev/ttyUSB0", "range": "0.1-5m", "resolution": "1mm"}
           },
           {
               "id": "proximity_rear",
               "type": "proximity",
               "protocol": "uart",
               "params": {"port": "/dev/ttyUSB1", "range": "0.1-5m", "resolution": "1mm"}
           }
       ]
       
       registered_sensors = []
       failed_sensors = []
       
       for sensor_config in agi_robot_sensors:
           try:
               result = self.register_sensor(
                   sensor_config["id"],
                   sensor_config["type"],
                   sensor_config["protocol"],
                   sensor_config["params"]
               )
               
               if result.get("success"):
                   registered_sensors.append(sensor_config["id"])
                   logger.info(f"✅ AGI robot sensor registered: {sensor_config['id']} ({sensor_config['type']})")
                   
                   # Initialize with real calibration data
                   self.sensors[sensor_config["id"]]["calibration_data"] = {
                       "offset": [0.0, 0.0, 0.0],
                       "scale": [1.0, 1.0, 1.0],
                       "calibrated_at": datetime.now().isoformat()
                   }
               else:
                   failed_sensors.append(sensor_config["id"])
                   logger.warning(f"⚠️ Failed to register AGI robot sensor {sensor_config['id']}: {result.get('error', 'Unknown error')}")
                   
           except Exception as e:
               failed_sensors.append(sensor_config["id"])
               logger.warning(f"⚠️ Exception registering AGI robot sensor {sensor_config['id']}: {e}")
       
       if not registered_sensors:
           raise RuntimeError(f"""
AGI Robot Sensor Registration Failed: Could not register any real robot sensors.
Failed sensors: {failed_sensors}

Please ensure:
1. Robot hardware is powered on and connected
2. Communication interfaces are properly configured
3. Required drivers are installed
4. Sensor addresses match hardware configuration
""")
       
       logger.info(f"✅ AGI Robot Sensor Registration Complete: {len(registered_sensors)} sensors registered, {len(failed_sensors)} failed")
       
       # Start real-time sensor data processing
       self._start_real_time_sensor_processing()
       
       return registered_sensors
    
    def _register_simulated_sensors(self):
       """Simulated sensor registration not supported - real hardware required"""
       raise RuntimeError(
           "Simulated sensor registration is not supported. This method has been disabled to ensure real hardware usage. "
           "Please connect real sensor hardware and use real hardware interface. "
           "Remove any environment variables that enable test mode (ROBOT_HARDWARE_TEST_MODE) and set ENVIRONMENT=production."
       )
    
    def _initialize_simulated_sensor_data(self, sensor_id: str, sensor_type: str):
       """Simulated sensor data initialization not supported - real hardware required"""
       raise RuntimeError(
           f"Simulated sensor data initialization is not supported. This method has been disabled to ensure real hardware usage. "
           f"Please connect real sensor hardware and use real hardware interface. "
           f"Sensor ID: {sensor_id}, Type: {sensor_type}. "
           f"Remove any environment variables that enable test mode (ROBOT_HARDWARE_TEST_MODE) and set ENVIRONMENT=production."
       )
    
    def _register_simulated_sensors_for_development(self):
       """Register simulated sensors for development/testing purposes"""
       import os
       import time
       
       logger.info("Registering simulated sensors for development/testing")
       
       # Define simulated sensor configuration
       simulated_sensors = [
           {
               "id": "imu_9dof_sim",
               "type": "imu",
               "protocol": "simulated",
               "params": {"simulation": True, "update_rate": "100Hz", "noise_level": "low"}
           },
           {
               "id": "foot_pressure_left_sim",
               "type": "force_sensor",
               "protocol": "simulated",
               "params": {"simulation": True, "channels": 4, "range": "0-100kg", "noise": 0.1}
           },
           {
               "id": "foot_pressure_right_sim",
               "type": "force_sensor",
               "protocol": "simulated",
               "params": {"simulation": True, "channels": 4, "range": "0-100kg", "noise": 0.1}
           },
           {
               "id": "joint_torque_hip_left_sim",
               "type": "torque_sensor",
               "protocol": "simulated",
               "params": {"simulation": True, "range": "±50Nm", "update_rate": "1kHz"}
           },
           {
               "id": "battery_system_sim",
               "type": "battery",
               "protocol": "simulated",
               "params": {"simulation": True, "cells": 6, "capacity": "4000mAh", "voltage": 24.0}
           },
           {
               "id": "proximity_front_sim",
               "type": "proximity",
               "protocol": "simulated",
               "params": {"simulation": True, "range": "0.1-5m", "resolution": "1mm"}
           }
       ]
       
       registered_sensors = []
       for sensor_config in simulated_sensors:
           try:
               # Add to sensors registry
               self.sensors[sensor_config["id"]] = {
                   "id": sensor_config["id"],
                   "type": sensor_config["type"],
                   "protocol": sensor_config["protocol"],
                   "params": sensor_config["params"],
                   "simulated": True,
                   "registered_at": time.time(),
                   "status": "active"
               }
               
               # Initialize simulated data
               self.sensor_data[sensor_config["id"]] = {
                   "value": 0.0,
                   "timestamp": time.time(),
                   "quality": 1.0,
                   "simulated": True
               }
               
               registered_sensors.append(sensor_config["id"])
               logger.info(f"Registered simulated sensor: {sensor_config['id']} ({sensor_config['type']})")
               
           except Exception as e:
               logger.warning(f"Failed to register simulated sensor {sensor_config['id']}: {e}")
       
       logger.info(f"Successfully registered {len(registered_sensors)} simulated sensors for development")
       return registered_sensors
    
    def _register_default_servos(self):
       """Register default servo motors - real hardware preferred, simulation allowed in development"""
       import os
       environment = os.environ.get('ENVIRONMENT', 'production').lower()
       robot_test_mode = os.environ.get('ROBOT_HARDWARE_TEST_MODE', 'false').lower() == 'true'
       allow_simulation = os.environ.get('ALLOW_ROBOT_SIMULATION', 'false').lower() == 'true'
       
       # Check if simulation is allowed in current environment
       simulation_allowed = environment != 'production' or robot_test_mode or allow_simulation
       
       if simulation_allowed:
           logger.warning(f"Simulation mode enabled for environment: {environment}")
           logger.info("Registering simulated servos for development/testing purposes")
           return self._register_simulated_servos_for_development()
       
       # Production environment or simulation not allowed - require real hardware
       logger.info("AGI robot requires real servo motors - checking hardware availability")
       
       # Check if device interface is available - simulation mode not supported
       if not self.device_interface:
           raise RuntimeError(
               "Robot hardware interface unavailable: External device interface not initialized. "
               "Please ensure robot joint controller drivers are installed and connect real hardware. "
               "Simulation mode is not supported for AGI robot operations in production environment."
           )
       
       try:
           # Check if device interface is available
           if not self.device_interface:
               raise RuntimeError("Robot hardware interface unavailable: External device interface not initialized. Please ensure robot joint controller drivers are installed and connect real hardware.")
           
           # Attempt to register real servo motors
           default_servos = [
               {"id": "servo_1", "type": "standard", "protocol": "pwm", "params": {"pin": 0, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_2", "type": "standard", "protocol": "pwm", "params": {"pin": 1, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_3", "type": "standard", "protocol": "pwm", "params": {"pin": 2, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_4", "type": "standard", "protocol": "pwm", "params": {"pin": 3, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_5", "type": "standard", "protocol": "pwm", "params": {"pin": 4, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_6", "type": "standard", "protocol": "pwm", "params": {"pin": 5, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_7", "type": "standard", "protocol": "pwm", "params": {"pin": 6, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_8", "type": "standard", "protocol": "pwm", "params": {"pin": 7, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_9", "type": "standard", "protocol": "pwm", "params": {"pin": 8, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_10", "type": "standard", "protocol": "pwm", "params": {"pin": 9, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_11", "type": "standard", "protocol": "pwm", "params": {"pin": 10, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_12", "type": "standard", "protocol": "pwm", "params": {"pin": 11, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_13", "type": "standard", "protocol": "pwm", "params": {"pin": 12, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_14", "type": "standard", "protocol": "pwm", "params": {"pin": 13, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_15", "type": "standard", "protocol": "pwm", "params": {"pin": 14, "min_pulse": 500, "max_pulse": 2500}},
               {"id": "servo_16", "type": "standard", "protocol": "pwm", "params": {"pin": 15, "min_pulse": 500, "max_pulse": 2500}}
           ]
           
           registered_count = 0
           failed_servos = []
           for servo_config in default_servos:
               try:
                   result = self.register_servo(
                       servo_config["id"],
                       servo_config["type"],
                       servo_config["protocol"],
                       servo_config["params"]
                   )
                   if result.get("success"):
                       registered_count += 1
                       logger.info(f"Successfully registered servo: {servo_config['id']}")
                   else:
                       failed_servos.append(servo_config['id'])
                       logger.warning(f"Failed to register servo {servo_config['id']}: {result.get('error', 'Unknown error')}")
               except Exception as e:
                   failed_servos.append(servo_config['id'])
                   logger.warning(f"Exception occurred while registering servo {servo_config['id']}: {e}")
           
           if registered_count == 0:
               raise RuntimeError(f"Cannot register any real servo motors. Failed servos: {failed_servos}. Please check hardware connections and drivers.")
           
           logger.info(f"Successfully registered {registered_count} real servo motors, {len(failed_servos)} servos failed to register")
           
       except Exception as e:
           logger.warning(f"Failed to register real servos: {e}")
           # Simulation mode not supported - always raise error
           raise RuntimeError(
               f"Cannot register any real servo motors. Please connect real servo motor hardware and ensure drivers are installed. "
               f"Error: {str(e)}\n"
               f"Simulation mode is not supported for AGI robot operations. "
               f"Remove any environment variables that enable test mode (ROBOT_HARDWARE_TEST_MODE) and set ENVIRONMENT=production."
           )
    
    def _register_simulated_servos(self):
        """Simulated servo motor registration not supported - real hardware required"""
        raise RuntimeError(
            "Simulated servo motor registration is not supported. This method has been disabled to ensure real hardware usage. "
            "Please connect real servo motor hardware and use real hardware interface. "
            "Remove any environment variables that enable test mode (ROBOT_HARDWARE_TEST_MODE) and set ENVIRONMENT=production."
        )
    
    def _register_simulated_servos_for_development(self):
        """Register simulated servo motors for development/testing purposes"""
        import time
        
        logger.info("Registering simulated servo motors for development/testing")
        
        # Define simulated servo configuration
        simulated_servos = [
            {"id": "servo_1_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -90, "max_angle": 90}},
            {"id": "servo_2_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -90, "max_angle": 90}},
            {"id": "servo_3_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -90, "max_angle": 90}},
            {"id": "servo_4_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -45, "max_angle": 45}},
            {"id": "servo_5_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -45, "max_angle": 45}},
            {"id": "servo_6_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -30, "max_angle": 30}},
            {"id": "servo_7_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -30, "max_angle": 30}},
            {"id": "servo_8_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -60, "max_angle": 60}},
            {"id": "servo_9_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -60, "max_angle": 60}},
            {"id": "servo_10_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -90, "max_angle": 90}},
            {"id": "servo_11_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -90, "max_angle": 90}},
            {"id": "servo_12_sim", "type": "standard", "protocol": "simulated", "params": {"simulation": True, "min_angle": -120, "max_angle": 120}}
        ]
        
        registered_servos = []
        for servo_config in simulated_servos:
            try:
                # Add to servos registry
                self.servos[servo_config["id"]] = {
                    "id": servo_config["id"],
                    "type": servo_config["type"],
                    "protocol": servo_config["protocol"],
                    "params": servo_config["params"],
                    "simulated": True,
                    "registered_at": time.time(),
                    "status": "active",
                    "position": 0.0,
                    "velocity": 0.0,
                    "torque": 0.0
                }
                
                # Initialize servo position
                self.servo_positions[servo_config["id"]] = {
                    "position": 0.0,
                    "target_position": 0.0,
                    "velocity": 0.0,
                    "torque": 0.0,
                    "timestamp": time.time(),
                    "simulated": True
                }
                
                registered_servos.append(servo_config["id"])
                logger.info(f"Registered simulated servo: {servo_config['id']}")
                
            except Exception as e:
                logger.warning(f"Failed to register simulated servo {servo_config['id']}: {e}")
        
        logger.info(f"Successfully registered {len(registered_servos)} simulated servos for development")
        return registered_servos
    
    def register_sensor(self, sensor_id: str, sensor_type: str, 
                      protocol: str, params: Dict[str, Any]) -> Dict[str, Any]:
       """Register sensor device"""
       try:
           if not self.device_interface:
               return {"success": False, "error": "Device interface not initialized"}
           
           # Connect to sensor device
           result = self.device_interface.connect_device(sensor_id, protocol, params)
           if not result["success"]:
               return result
           
           # Store sensor information
           self.sensors[sensor_id] = {
               "id": sensor_id,
               "type": sensor_type,
               "protocol": protocol,
               "params": params,
               "registered_at": datetime.now().isoformat(),
               "last_update": None,
               "data_format": self._get_sensor_data_format(sensor_type),
           }
           
           # Start data reading thread
           self._start_sensor_read_thread(sensor_id)
           
           logger.info(f"Sensor {sensor_id} ({sensor_type}) registered successfully")
           return {"success": True, "sensor_id": sensor_id}
           
       except Exception as e:
           logger.error(f"Failed to register sensor {sensor_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def register_camera(self, camera_id: str, camera_type: str, 
                      camera_index: int, params: Dict[str, Any]) -> Dict[str, Any]:
       """Register camera device"""
       try:
           if not self.camera_manager:
               return {"success": False, "error": "Camera manager not initialized"}
           
           # Use camera manager to open camera
           # Note: This needs to be adjusted according to actual camera manager API
           camera_params = {
               "camera_index": camera_index,
               "resolution": params.get("resolution", (1920, 1080)),
               "fps": params.get("fps", 60),
               "format": params.get("format", "bgr"),
           }
           
           # Store camera information
           self.cameras[camera_id] = {
               "id": camera_id,
               "type": camera_type,
               "index": camera_index,
               "params": camera_params,
               "registered_at": datetime.now().isoformat(),
               "is_streaming": False,
               "last_frame": None,
           }
           
           logger.info(f"Camera {camera_id} ({camera_type}) registered successfully")
           return {"success": True, "camera_id": camera_id}
           
       except Exception as e:
           logger.error(f"Failed to register camera {camera_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def register_motor(self, motor_id: str, motor_type: str,
                     protocol: str, params: Dict[str, Any]) -> Dict[str, Any]:
       """Register motor device"""
       try:
           if not self.device_interface:
               return {"success": False, "error": "Device interface not initialized"}
           
           # Connect to motor controller
           result = self.device_interface.connect_device(motor_id, protocol, params)
           if not result["success"]:
               return result
           
           # Store motor information
           self.motors[motor_id] = {
               "id": motor_id,
               "type": motor_type,
               "protocol": protocol,
               "params": params,
               "registered_at": datetime.now().isoformat(),
               "current_position": 0.0,
               "target_position": 0.0,
               "velocity": 0.0,
               "current": 0.0,
               "temperature": 25.0,
               "is_enabled": False,
           }
           
           # Initialize motor parameters
           self._initialize_motor_parameters(motor_id, motor_type, params)
           
           logger.info(f"Motor {motor_id} ({motor_type}) registered successfully")
           return {"success": True, "motor_id": motor_id}
           
       except Exception as e:
           logger.error(f"Failed to register motor {motor_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def register_servo(self, servo_id: str, servo_type: str,
                     protocol: str, params: Dict[str, Any]) -> Dict[str, Any]:
       """Register servo device"""
       try:
           if not self.device_interface:
               return {"success": False, "error": "Device interface not initialized"}
           
           # Connect to servo driver
           result = self.device_interface.connect_device(servo_id, protocol, params)
           if not result["success"]:
               return result
           
           # Store servo information
           self.servos[servo_id] = {
               "id": servo_id,
               "type": servo_type,
               "protocol": protocol,
               "params": params,
               "registered_at": datetime.now().isoformat(),
               "current_angle": 0.0,
               "target_angle": 0.0,
               "torque": 0.0,
               "temperature": 25.0,
               "is_enabled": False,
               "min_angle": params.get("min_angle", -180),
               "max_angle": params.get("max_angle", 180),
               "max_speed": params.get("max_speed", 100),
               "max_torque": params.get("max_torque", 5.0),
           }
           
           logger.info(f"Servo {servo_id} ({servo_type}) registered successfully")
           return {"success": True, "servo_id": servo_id}
           
       except Exception as e:
           logger.error(f"Failed to register servo {servo_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def control_servo(self, servo_id: str, angle: float, 
                    velocity: Optional[float] = None,
                    torque: Optional[float] = None) -> Dict[str, Any]:
       """Control servo angle"""
       try:
           with self.data_lock:
               if servo_id not in self.servos:
                   return {"success": False, "error": f"Servo {servo_id} not registered"}
               
               servo = self.servos[servo_id]
               
               # Safety check
               safety_check = self._check_servo_safety(servo_id, angle, velocity, torque)
               if not safety_check["safe"]:
                   return {"success": False, "error": safety_check["reason"]}
               
               # Update target angle
               servo["target_angle"] = angle
               if velocity is not None:
                   servo["target_velocity"] = velocity
               if torque is not None:
                   servo["target_torque"] = torque
               
               # Check if real hardware is available
               # Try robot driver first if available
               if self.robot_driver and self.robot_driver.connected:
                   try:
                       # Map servo ID to robot joint name
                       joint_name = self._map_servo_to_joint(servo_id)
                       if joint_name:
                           # Use robot driver for real hardware control
                           success = self.robot_driver.set_joint_position(
                               joint_name, angle, velocity, None  # acceleration not specified
                           )
                           
                           if success:
                               servo["last_command"] = datetime.now().isoformat()
                               logger.debug(f"Servo {servo_id} (mapped to {joint_name}) controlled via robot driver: angle={angle}")
                               return {"success": True, "servo_id": servo_id, "angle": angle, "method": "robot_driver"}
                           else:
                               logger.warning(f"Robot driver failed to control servo {servo_id}, falling back to legacy interface")
                       else:
                           logger.debug(f"No joint mapping for servo {servo_id}, using legacy interface")
                   except Exception as e:
                       logger.error(f"Robot driver control error for servo {servo_id}: {e}")
                       # Fall through to legacy interface
               
               # Fall back to legacy device interface
               if not self.device_interface:
                   return {
                       "success": False, 
                       "error": f"Hardware interface not initialized. Real hardware required for servo control.",
                       "servo_id": servo_id
                   }
               
               if servo.get("protocol") == "simulated":
                   return {
                       "success": False,
                       "error": f"Servo {servo_id} is configured with simulated protocol. Real hardware protocol required.",
                       "servo_id": servo_id
                   }
               
               # Build and send control command to real hardware via legacy interface
               command = self._build_servo_command(servo_id, angle, velocity, torque)
               result = self.device_interface.send_data(servo_id, command, "servo_control")
               
               if result["success"]:
                   servo["last_command"] = datetime.now().isoformat()
                   logger.debug(f"Servo {servo_id} control command sent via legacy interface: angle={angle}")
                   return {"success": True, "servo_id": servo_id, "angle": angle, "method": "legacy_interface"}
               else:
                   return result
               
       except Exception as e:
           logger.error(f"Failed to control servo {servo_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def control_motor(self, motor_id: str, position: Optional[float] = None,
                    velocity: Optional[float] = None,
                    torque: Optional[float] = None) -> Dict[str, Any]:
       """Control motor"""
       try:
           with self.data_lock:
               if motor_id not in self.motors:
                   return {"success": False, "error": f"Motor {motor_id} not registered"}
               
               motor = self.motors[motor_id]
               
               # Safety check
               safety_check = self._check_motor_safety(motor_id, position, velocity, torque)
               if not safety_check["safe"]:
                   return {"success": False, "error": safety_check["reason"]}
               
               # Update target values
               if position is not None:
                   motor["target_position"] = position
               if velocity is not None:
                   motor["target_velocity"] = velocity
               if torque is not None:
                   motor["target_torque"] = torque
               
               # Check if real hardware is available
               # Try robot driver first if available and position is specified
               # Robot driver requires position for joint control
               if self.robot_driver and self.robot_driver.connected and position is not None:
                   try:
                       # Map motor ID to robot joint name
                       joint_name = self._map_motor_to_joint(motor_id)
                       if joint_name:
                           # Use robot driver for real hardware control
                           success = self.robot_driver.set_joint_position(
                               joint_name, position, velocity, None  # acceleration not specified
                           )
                           
                           if success:
                               motor["last_command"] = datetime.now().isoformat()
                               logger.debug(f"Motor {motor_id} (mapped to {joint_name}) controlled via robot driver: position={position}")
                               return {"success": True, "motor_id": motor_id, "position": position, "method": "robot_driver"}
                           else:
                               logger.warning(f"Robot driver failed to control motor {motor_id}, falling back to legacy interface")
                       else:
                           logger.debug(f"No joint mapping for motor {motor_id}, using legacy interface")
                   except Exception as e:
                       logger.error(f"Robot driver control error for motor {motor_id}: {e}")
                       # Fall through to legacy interface
               
               # Fall back to legacy device interface
               if not self.device_interface:
                   return {
                       "success": False, 
                       "error": f"Hardware interface not initialized. Real hardware required for motor control.",
                       "motor_id": motor_id
                   }
               
               if motor.get("protocol") == "simulated":
                   return {
                       "success": False,
                       "error": f"Motor {motor_id} is configured with simulated protocol. Real hardware protocol required.",
                       "motor_id": motor_id
                   }
               
               # Build and send control command to real hardware via legacy interface
               command = self._build_motor_command(motor_id, position, velocity, torque)
               result = self.device_interface.send_data(motor_id, command, "motor_control")
               
               if result["success"]:
                   motor["last_command"] = datetime.now().isoformat()
                   logger.debug(f"Motor {motor_id} control command sent via legacy interface: position={position}")
                   return {"success": True, "motor_id": motor_id, "method": "legacy_interface"}
               else:
                   return result
               
       except Exception as e:
           logger.error(f"Failed to control motor {motor_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def _map_servo_to_joint(self, servo_id: str) -> Optional[str]:
        """Map servo ID to humanoid robot joint name
        
        Args:
            servo_id: Servo device ID (e.g., "servo_1")
            
        Returns:
            Joint name string or None if no mapping exists
        """
        # Default mapping for humanoid robot with 16 servos
        servo_to_joint_mapping = {
            # Left arm
            "servo_1": "left_shoulder_pitch",
            "servo_2": "left_shoulder_roll", 
            "servo_3": "left_elbow",
            "servo_4": "left_wrist",
            
            # Right arm
            "servo_5": "right_shoulder_pitch",
            "servo_6": "right_shoulder_roll",
            "servo_7": "right_elbow",
            "servo_8": "right_wrist",
            
            # Left leg
            "servo_9": "left_hip_yaw",
            "servo_10": "left_hip_roll",
            "servo_11": "left_hip_pitch",
            "servo_12": "left_knee",
            "servo_13": "left_ankle_pitch",
            "servo_14": "left_ankle_roll",
            
            # Right leg
            "servo_15": "right_hip_yaw",
            "servo_16": "right_hip_roll",
            "servo_17": "right_hip_pitch",
            "servo_18": "right_knee",
            "servo_19": "right_ankle_pitch",
            "servo_20": "right_ankle_roll",
            
            # Head and torso
            "servo_21": "head_pan",
            "servo_22": "head_tilt",
            "servo_23": "torso_twist",
            "servo_24": "torso_bend",
        }
        
        return servo_to_joint_mapping.get(servo_id)
    
    def _map_motor_to_joint(self, motor_id: str) -> Optional[str]:
        """Map motor ID to humanoid robot joint name
        
        Args:
            motor_id: Motor device ID
            
        Returns:
            Joint name string or None if no mapping exists
        """
        # For humanoid robots, motors may control the same joints as servos
        # Try to map motor IDs that follow servo naming pattern
        if motor_id.startswith("motor_"):
            # Convert "motor_X" to "servo_X" and try servo mapping
            servo_id = motor_id.replace("motor_", "servo_", 1)
            return self._map_servo_to_joint(servo_id)
        
        # Additional motor-specific mappings can be added here
        motor_to_joint_mapping = {
            "motor_wheel_left": "wheel_left",
            "motor_wheel_right": "wheel_right",
            "motor_gripper_left": "gripper_left",
            "motor_gripper_right": "gripper_right",
        }
        
        return motor_to_joint_mapping.get(motor_id)
    
    def _map_sensor_to_robot_sensor(self, sensor_id: str) -> Optional[str]:
        """Map sensor ID to humanoid robot sensor name
        
        Args:
            sensor_id: Sensor device ID
            
        Returns:
            Robot sensor name string or None if no mapping exists
        """
        # Default mapping for humanoid robot sensors
        sensor_to_robot_mapping = {
            # IMU sensors
            "imu_9dof": "imu_body",
            "foot_pressure_left": "force_left_foot",
            "foot_pressure_right": "force_right_foot",
            "joint_torque_hip_left": "torque_left_hip",
            "joint_torque_knee_left": "torque_left_knee",
            "joint_torque_ankle_left": "torque_left_ankle",
            "battery_system": "battery",
            "system_temperature": "system_temperature",
            "proximity_front": "proximity_front",
            "proximity_rear": "proximity_back",
            
            # Additional mappings from sensor registration
            "imu": "imu_body",
            "force_sensor": "force_left_foot",  # Default to left foot
            "torque_sensor": "torque_left_hip",  # Default to left hip
            "temperature": "system_temperature",
            "battery": "battery",
            "proximity": "proximity_front",
        }
        
        # First try direct mapping
        if sensor_id in sensor_to_robot_mapping:
            return sensor_to_robot_mapping[sensor_id]
        
        # Try to map based on sensor ID pattern
        if sensor_id.startswith("imu"):
            return "imu_body"
        elif sensor_id.startswith("foot_pressure"):
            if "left" in sensor_id:
                return "force_left_foot"
            elif "right" in sensor_id:
                return "force_right_foot"
        elif sensor_id.startswith("joint_torque"):
            if "hip" in sensor_id and "left" in sensor_id:
                return "torque_left_hip"
            elif "hip" in sensor_id and "right" in sensor_id:
                return "torque_right_hip"
            elif "knee" in sensor_id and "left" in sensor_id:
                return "torque_left_knee"
            elif "knee" in sensor_id and "right" in sensor_id:
                return "torque_right_knee"
            elif "ankle" in sensor_id and "left" in sensor_id:
                return "torque_left_ankle"
            elif "ankle" in sensor_id and "right" in sensor_id:
                return "torque_right_ankle"
        elif sensor_id.startswith("battery"):
            return "battery"
        elif sensor_id.startswith("temperature") or sensor_id.startswith("temp"):
            return "system_temperature"
        elif sensor_id.startswith("proximity"):
            if "front" in sensor_id:
                return "proximity_front"
            elif "rear" in sensor_id or "back" in sensor_id:
                return "proximity_back"
            elif "left" in sensor_id:
                return "proximity_left"
            elif "right" in sensor_id:
                return "proximity_right"
        
        return None
    
    def get_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
       """Get sensor data"""
       try:
           with self.data_lock:
               if sensor_id not in self.sensors:
                   return {"success": False, "error": f"Sensor {sensor_id} not registered"}
               
               sensor = self.sensors[sensor_id]
               
               # Try robot driver first if available
               if self.robot_driver and self.robot_driver.connected:
                   try:
                       # Map sensor ID to robot sensor name
                       robot_sensor_name = self._map_sensor_to_robot_sensor(sensor_id)
                       if robot_sensor_name:
                           # Get sensor value from robot driver
                           sensor_value = self.robot_driver.get_sensor_value(robot_sensor_name)
                           
                           if sensor_value is not None:
                               # Format sensor data
                               data = self._format_sensor_data(sensor_id, sensor_value)
                               
                               # Update cache
                               self.sensor_data[sensor_id] = data
                               sensor["last_update"] = datetime.now().isoformat()
                               
                               logger.debug(f"Sensor {sensor_id} (mapped to {robot_sensor_name}) data retrieved via robot driver")
                               return {
                                   "success": True,
                                   "sensor_id": sensor_id,
                                   "data": data,
                                   "timestamp": sensor["last_update"],
                                   "method": "robot_driver",
                               }
                           else:
                               logger.warning(f"Robot driver returned None for sensor {sensor_id}, falling back to legacy interface")
                       else:
                           logger.debug(f"No robot sensor mapping for {sensor_id}, using legacy interface")
                   except Exception as e:
                       logger.error(f"Robot driver sensor read error for {sensor_id}: {e}")
                       # Fall through to legacy interface
               
               # Fall back to legacy device interface
               if not self.device_interface:
                   return {"success": False, "error": "Hardware interface not initialized. Sensor data unavailable."}
               
               # Try to get from data cache first
               if sensor_id in self.sensor_data:
                   data = self.sensor_data[sensor_id]
                   return {
                       "success": True,
                       "sensor_id": sensor_id,
                       "data": data,
                       "timestamp": sensor.get("last_update"),
                       "method": "cache",
                   }
               else:
                   return {"success": False, "error": "Sensor data not available"}
               
       except Exception as e:
           logger.error(f"Failed to get sensor data {sensor_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def get_camera_frame(self, camera_id: str) -> Dict[str, Any]:
       """Get camera frame"""
       try:
           with self.data_lock:
               if camera_id not in self.cameras:
                   return {"success": False, "error": f"Camera {camera_id} not registered"}
               
               camera = self.cameras[camera_id]
               
               # Get from frame cache
               if camera_id in self.camera_frames:
                   frame = self.camera_frames[camera_id]
                   return {
                       "success": True,
                       "camera_id": camera_id,
                       "frame": frame,
                       "timestamp": datetime.now().isoformat(),
                   }
               else:
                   return {"success": False, "error": "Camera frame not available"}
               
       except Exception as e:
           logger.error(f"Failed to get camera frame {camera_id}: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def emergency_stop(self) -> Dict[str, Any]:
       """Emergency stop all hardware"""
       try:
           with self.control_lock:
               self.safety_system["emergency_stop"] = True
               
               # Stop all motors
               for motor_id in self.motors:
                   self._emergency_stop_motor(motor_id)
               
               # Stop all servos
               for servo_id in self.servos:
                   self._emergency_stop_servo(servo_id)
               
               logger.warning("Emergency stop activated")
               return {"success": True, "emergency_stop": True}
               
       except Exception as e:
           logger.error(f"Emergency stop failed: {str(e)}")
           return {"success": False, "error": str(e)}
    
    def resume_normal_operation(self) -> Dict[str, Any]:
       """Resume normal operation"""
       try:
           with self.control_lock:
               self.safety_system["emergency_stop"] = False
               logger.info("Normal operation resumed")
               return {"success": True, "emergency_stop": False}
               
       except Exception as e:
           logger.error(f"Failed to resume normal operation: {str(e)}")
           return {"success": False, "error": str(e)}
    
    # Private helper methods
    
    def _get_sensor_data_format(self, sensor_type: str) -> Dict[str, Any]:
       """Get sensor data format"""
       formats = {
           "imu": {"acceleration": "vector3", "gyro": "vector3", "magnetometer": "vector3"},
           "force_sensor": {"force": "vector3", "torque": "vector3"},
           "proximity": {"distance": "float", "object_detected": "bool"},
           "temperature": {"temperature": "float"},
           "current_sensor": {"current": "float", "voltage": "float", "power": "float"},
           "encoder": {"position": "float", "velocity": "float"},
           "pressure": {"pressure": "float"},
           "humidity": {"humidity": "float", "temperature": "float"},
       }
       return formats.get(sensor_type, {"raw": "binary"})
    
    def _format_sensor_data(self, sensor_id: str, raw_value: Any) -> Dict[str, Any]:
        """Format raw sensor value to standardized data format
        
        Args:
            sensor_id: Sensor device ID
            raw_value: Raw sensor value from hardware
            
        Returns:
            Formatted sensor data dictionary
        """
        sensor_type = self.sensors.get(sensor_id, {}).get("type", "unknown")
        
        # Get format specification
        format_spec = self._get_sensor_data_format(sensor_type)
        
        # Format based on sensor type
        if sensor_type == "imu":
            # IMU data should be dict with acceleration, gyro, magnetometer
            if isinstance(raw_value, dict):
                return raw_value
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 9:
                # Assume [ax, ay, az, gx, gy, gz, mx, my, mz]
                return {
                    "acceleration": {"x": float(raw_value[0]), "y": float(raw_value[1]), "z": float(raw_value[2])},
                    "gyro": {"x": float(raw_value[3]), "y": float(raw_value[4]), "z": float(raw_value[5])},
                    "magnetometer": {"x": float(raw_value[6]), "y": float(raw_value[7]), "z": float(raw_value[8])},
                }
            else:
                # Return raw value in a structured format
                return {"raw": raw_value}
        
        elif sensor_type == "force_sensor":
            if isinstance(raw_value, dict):
                return raw_value
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 6:
                # Assume [fx, fy, fz, tx, ty, tz]
                return {
                    "force": {"x": float(raw_value[0]), "y": float(raw_value[1]), "z": float(raw_value[2])},
                    "torque": {"x": float(raw_value[3]), "y": float(raw_value[4]), "z": float(raw_value[5])},
                }
            else:
                return {"raw": raw_value}
        
        elif sensor_type in ["temperature", "current_sensor", "encoder", "pressure", "humidity"]:
            # Simple scalar or dict values
            if isinstance(raw_value, dict):
                return raw_value
            else:
                # Create simple measurement
                key = list(format_spec.keys())[0] if format_spec else "value"
                return {key: float(raw_value) if isinstance(raw_value, (int, float)) else raw_value}
        
        elif sensor_type == "proximity":
            if isinstance(raw_value, dict):
                return raw_value
            else:
                return {
                    "distance": float(raw_value) if isinstance(raw_value, (int, float)) else 0.0,
                    "object_detected": bool(raw_value) if isinstance(raw_value, (int, float)) else False,
                }
        
        else:
            # Unknown sensor type - return raw value
            return {"raw": raw_value}
    
    def _start_sensor_read_thread(self, sensor_id: str):
       """Start sensor data reading thread"""
       def sensor_read_loop():
           while sensor_id in self.sensors:
               try:
                   # Try robot driver first if available
                   if self.robot_driver and self.robot_driver.connected:
                       try:
                           # Map sensor ID to robot sensor name
                           robot_sensor_name = self._map_sensor_to_robot_sensor(sensor_id)
                           if robot_sensor_name:
                               # Get sensor value from robot driver
                               sensor_value = self.robot_driver.get_sensor_value(robot_sensor_name)
                               
                               if sensor_value is not None:
                                   # Format sensor data
                                   data = self._format_sensor_data(sensor_id, sensor_value)
                                   
                                   with self.data_lock:
                                       self.sensor_data[sensor_id] = data
                                       self.sensors[sensor_id]["last_update"] = datetime.now().isoformat()
                               else:
                                   # Robot driver returned None, try legacy interface
                                   self._read_sensor_via_legacy_interface(sensor_id)
                           else:
                               # No mapping, try legacy interface
                               self._read_sensor_via_legacy_interface(sensor_id)
                       except Exception as e:
                           logger.error(f"Robot driver sensor read error in thread {sensor_id}: {e}")
                           # Fall back to legacy interface
                           self._read_sensor_via_legacy_interface(sensor_id)
                   else:
                       # Robot driver not available, use legacy interface
                       self._read_sensor_via_legacy_interface(sensor_id)
                   
                   time.sleep(1.0 / self.config["sensor_update_rate"])
                   
               except Exception as e:
                   logger.error(f"Sensor reading thread error {sensor_id}: {str(e)}")
                   time.sleep(0.1)
       
       thread = threading.Thread(target=sensor_read_loop, daemon=True)
       thread.start()
       self.control_threads[f"sensor_{sensor_id}"] = thread
    
    def _read_sensor_via_legacy_interface(self, sensor_id: str) -> None:
        """Read sensor data via legacy device interface
        
        Args:
            sensor_id: Sensor device ID
        """
        try:
            if self.device_interface and sensor_id in self.sensors:
                data = self.device_interface.receive_data(sensor_id, "sensor_data")
                if data:
                    with self.data_lock:
                        self.sensor_data[sensor_id] = data
                        self.sensors[sensor_id]["last_update"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Legacy sensor read error {sensor_id}: {e}")
    
    def _initialize_motor_parameters(self, motor_id: str, motor_type: str, params: Dict[str, Any]):
       """Initialize motor parameters"""
       motor = self.motors[motor_id]
       
       # Set default parameters
       defaults = {
           "dc_motor": {"max_rpm": 3000, "torque_constant": 0.1, "resistance": 1.0},
           "stepper": {"steps_per_rev": 200, "holding_torque": 2.0},
           "brushless": {"kv_rating": 1000, "max_current": 30.0},
       }
       
       motor_defaults = defaults.get(motor_type, {})
       for key, value in motor_defaults.items():
           if key not in motor:
               motor[key] = value
       
       # Merge user parameters
       motor.update(params)
    
    def _check_servo_safety(self, servo_id: str, angle: float, 
                          velocity: Optional[float], torque: Optional[float]) -> Dict[str, Any]:
       """Check servo safety limits"""
       servo = self.servos[servo_id]
       
       # Angle limit
       if angle < servo["min_angle"] or angle > servo["max_angle"]:
           return {"safe": False, "reason": f"Angle out of limit ({servo['min_angle']}~{servo['max_angle']})"}
       
       # Speed limit
       if velocity is not None and velocity > servo["max_speed"]:
           return {"safe": False, "reason": f"Speed exceeds limit (max {servo['max_speed']})"}
       
       # Torque limit
       if torque is not None and torque > servo["max_torque"]:
           return {"safe": False, "reason": f"Torque exceeds limit (max {servo['max_torque']})"}
       
       # Emergency stop check
       if self.safety_system["emergency_stop"]:
           return {"safe": False, "reason": "Emergency stop activated"}
       
       return {"safe": True}
    
    def _check_motor_safety(self, motor_id: str, position: Optional[float],
                          velocity: Optional[float], torque: Optional[float]) -> Dict[str, Any]:
       """Check motor safety limits"""
       motor = self.motors[motor_id]
       
       # Current limit
       if torque is not None:
           estimated_current = torque / motor.get("torque_constant", 1.0)
           if estimated_current > self.config["max_motor_current"]:
               return {"safe": False, "reason": f"Current exceeds limit (max {self.config['max_motor_current']}A)"}
       
       # Temperature limit
       if motor["temperature"] > 80.0:
           return {"safe": False, "reason": f"Temperature too high ({motor['temperature']}°C)"}
       
       # Emergency stop check
       if self.safety_system["emergency_stop"]:
           return {"safe": False, "reason": "Emergency stop activated"}
       
       return {"safe": True}
    
    def _build_servo_command(self, servo_id: str, angle: float,
                           velocity: Optional[float], torque: Optional[float]) -> Any:
       """Build servo control command"""
       servo = self.servos[servo_id]
       protocol = servo["protocol"]
       
       # Build command based on protocol
       if protocol == "serial":
           # Serial protocol command format
           cmd = f"SET SERVO {servo_id} ANGLE {angle}"
           if velocity is not None:
               cmd += f" VELOCITY {velocity}"
           if torque is not None:
               cmd += f" TORQUE {torque}"
           return cmd.encode()
       
       elif protocol == "tcp" or protocol == "udp":
           # TCP/UDP protocol command format
           command = {
               "device_id": servo_id,
               "command": "set_angle",
               "angle": angle,
               "timestamp": time.time(),
           }
           if velocity is not None:
               command["velocity"] = velocity
           if torque is not None:
               command["torque"] = torque
           return json.dumps(command).encode()
       
       else:
           # Default: return raw data
           return {"angle": angle, "velocity": velocity, "torque": torque}
    
    def _build_motor_command(self, motor_id: str, position: Optional[float],
                           velocity: Optional[float], torque: Optional[float]) -> Any:
       """Build motor control command"""
       motor = self.motors[motor_id]
       protocol = motor["protocol"]
       
       # Build command based on protocol
       if protocol == "serial":
           cmd = f"SET MOTOR {motor_id}"
           if position is not None:
               cmd += f" POSITION {position}"
           if velocity is not None:
               cmd += f" VELOCITY {velocity}"
           if torque is not None:
               cmd += f" TORQUE {torque}"
           return cmd.encode()
       
       elif protocol == "tcp" or protocol == "udp":
           command = {
               "device_id": motor_id,
               "command": "motor_control",
               "timestamp": time.time(),
           }
           if position is not None:
               command["position"] = position
           if velocity is not None:
               command["velocity"] = velocity
           if torque is not None:
               command["torque"] = torque
           return json.dumps(command).encode()
       
       else:
           # Default: return raw data
           return {"position": position, "velocity": velocity, "torque": torque}
    
    def _emergency_stop_motor(self, motor_id: str):
       """Emergency stop motor"""
       try:
           motor = self.motors[motor_id]
           motor["is_enabled"] = False
           
           # Send stop command
           if self.device_interface:
               command = f"STOP MOTOR {motor_id}"
               self.device_interface.send_data(motor_id, command.encode(), "emergency")
           
           logger.debug(f"Motor {motor_id} emergency stopped")
           
       except Exception as e:
           logger.error(f"Motor emergency stop failed {motor_id}: {str(e)}")
    
    def _emergency_stop_servo(self, servo_id: str):
       """Emergency stop servo"""
       try:
           servo = self.servos[servo_id]
           servo["is_enabled"] = False
           
           # Send stop command
           if self.device_interface:
               command = f"STOP SERVO {servo_id}"
               self.device_interface.send_data(servo_id, command.encode(), "emergency")
           
           logger.debug(f"Servo {servo_id} emergency stopped")
           
       except Exception as e:
           logger.error(f"Servo emergency stop failed {servo_id}: {str(e)}")
    
    def get_hardware_status(self) -> Dict[str, Any]:
       """Get hardware status overview"""
       with self.data_lock:
           status = {
               "initialized": self.hardware_initialized,
               "sensor_count": len(self.sensors),
               "camera_count": len(self.cameras),
               "motor_count": len(self.motors),
               "servo_count": len(self.servos),
               "emergency_stop": self.safety_system["emergency_stop"],
               "performance_metrics": self.performance_metrics,
               "sensors": list(self.sensors.keys()),
               "cameras": list(self.cameras.keys()),
               "motors": list(self.motors.keys()),
               "servos": list(self.servos.keys()),
               "timestamp": datetime.now().isoformat(),
           }
           return status
    
    # ========== Robot Training API Compatibility Methods ==========
    
    async def is_connected(self) -> bool:
        """Check if hardware is connected (async compatibility)"""
        return self.hardware_initialized
    
    async def get_device_status(self) -> Dict[str, Any]:
        """Get device status (async)"""
        status = self.get_hardware_status()
        devices = {}
        
        # Format devices
        for sensor_id in self.sensors.keys():
            devices[sensor_id] = {
                "name": f"Sensor {sensor_id}",
                "status": "connected" if self.sensors[sensor_id].get("is_enabled", False) else "disconnected",
                "details": {"type": "sensor", "protocol": self.sensors[sensor_id].get("protocol", "unknown")}
            }
        
        for motor_id in self.motors.keys():
            devices[motor_id] = {
                "name": f"Motor {motor_id}",
                "status": "connected" if self.motors[motor_id].get("is_enabled", False) else "disconnected",
                "details": {"type": "motor", "protocol": self.motors[motor_id].get("protocol", "unknown")}
            }
        
        for servo_id in self.servos.keys():
            devices[servo_id] = {
                "name": f"Servo {servo_id}",
                "status": "connected" if self.servos[servo_id].get("is_enabled", False) else "disconnected",
                "details": {"type": "servo", "protocol": self.servos[servo_id].get("protocol", "unknown")}
            }
        
        # Add joint controller if motors/servos exist
        if self.motors or self.servos:
            devices["joint_controller"] = {
                "name": "Joint Controller",
                "status": "connected" if self.hardware_initialized else "disconnected",
                "details": {"type": "controller", "device_count": len(self.motors) + len(self.servos)}
            }
        
        # Add emergency stop
        devices["emergency_stop"] = {
            "name": "Emergency Stop",
            "status": "ready" if not self.safety_system["emergency_stop"] else "triggered",
            "details": {"type": "safety", "enabled": True}
        }
        
        return devices
    
    async def connect_device(self, device_id: str, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to a hardware device (async) - real hardware required"""
        try:
            # Validate protocol - simulated protocol not allowed
            protocol = device_info.get("protocol", "unknown")
            if protocol == "simulated":
                return {
                    "success": False,
                    "error": f"Simulated protocol not allowed for device {device_id}. Real hardware protocol required."
                }
            
            # Connect hardware device based on type
            if device_id.startswith("sensor"):
                if device_id not in self.sensors:
                    self.sensors[device_id] = {
                        "is_enabled": True,
                        "protocol": protocol,
                        "type": device_info.get("type", "generic")
                    }
                else:
                    self.sensors[device_id]["is_enabled"] = True
                    
            elif device_id.startswith("motor"):
                if device_id not in self.motors:
                    self.motors[device_id] = {
                        "is_enabled": True,
                        "protocol": protocol,
                        "type": device_info.get("type", "dc")
                    }
                else:
                    self.motors[device_id]["is_enabled"] = True
                    
            elif device_id.startswith("servo"):
                if device_id not in self.servos:
                    self.servos[device_id] = {
                        "is_enabled": True,
                        "protocol": protocol,
                        "type": device_info.get("type", "position")
                    }
                else:
                    self.servos[device_id]["is_enabled"] = True
            else:
                return {
                    "success": False,
                    "error": f"Unknown device type for {device_id}. Device ID must start with 'sensor', 'motor', or 'servo'."
                }
            
            return {
                "success": True,
                "message": f"Device {device_id} connected",
                "details": {"protocol": protocol, "hardware": True}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def disconnect_device(self, device_id: str) -> Dict[str, Any]:
        """Disconnect from a hardware device (async)"""
        try:
            if device_id in self.sensors:
                self.sensors[device_id]["is_enabled"] = False
            elif device_id in self.motors:
                self.motors[device_id]["is_enabled"] = False
            elif device_id in self.servos:
                self.servos[device_id]["is_enabled"] = False
            
            return {
                "success": True,
                "message": f"Device {device_id} disconnected"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get sensor data from real hardware (async)"""
        try:
            sensor_data = {}
            with self.data_lock:
                # Check if hardware interface is available
                if not self.device_interface and not (self.robot_driver and self.robot_driver.connected):
                    logger.error("Hardware interface not initialized. Cannot read sensor data.")
                    return {}
                
                for sensor_id, sensor_info in self.sensors.items():
                    if sensor_info.get("is_enabled", False):
                        # Try robot driver first if available
                        if self.robot_driver and self.robot_driver.connected:
                            try:
                                # Map sensor ID to robot sensor name
                                robot_sensor_name = self._map_sensor_to_robot_sensor(sensor_id)
                                if robot_sensor_name:
                                    # Get sensor value from robot driver
                                    sensor_value = self.robot_driver.get_sensor_value(robot_sensor_name)
                                    
                                    if sensor_value is not None:
                                        # Format sensor data
                                        formatted_data = self._format_sensor_data(sensor_id, sensor_value)
                                        
                                        sensor_data[sensor_id] = {
                                            "name": f"Sensor {sensor_id}",
                                            "value": formatted_data,
                                            "unit": "robot_units",
                                            "status": "normal",
                                            "trend": "stable",
                                            "hardware": True,
                                            "method": "robot_driver"
                                        }
                                        continue  # Skip legacy interface
                            except Exception as e:
                                logger.error(f"Robot driver sensor read error for {sensor_id}: {e}")
                                # Fall through to legacy interface
                        
                        # Fall back to legacy device interface
                        if self.device_interface:
                            result = self.device_interface.read_data(sensor_id, "sensor_read")
                            if result.get("success", False):
                                sensor_data[sensor_id] = {
                                    "name": f"Sensor {sensor_id}",
                                    "value": result.get("value", 0.0),
                                    "unit": result.get("unit", "units"),
                                    "status": result.get("status", "normal"),
                                    "trend": result.get("trend", "stable"),
                                    "hardware": True,
                                    "method": "legacy_interface"
                                }
                            else:
                                sensor_data[sensor_id] = {
                                    "name": f"Sensor {sensor_id}",
                                    "value": None,
                                    "unit": "unknown",
                                    "status": "error",
                                    "error": result.get("error", "Read failed"),
                                    "hardware": False
                                }
                        else:
                            # No hardware interface available
                            sensor_data[sensor_id] = {
                                "name": f"Sensor {sensor_id}",
                                "value": None,
                                "unit": "unknown",
                                "status": "error",
                                "error": "No hardware interface available",
                                "hardware": False
                            }
            
            return sensor_data
        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return {}
    
    async def get_joint_positions(self) -> List[Dict[str, Any]]:
        """Get joint positions from real hardware (async)"""
        try:
            joints = []
            with self.data_lock:
                if not self.device_interface:
                    logger.error("Hardware interface not initialized. Cannot read joint positions.")
                    return []
                
                # Read servo positions
                for servo_id, servo_info in self.servos.items():
                    if servo_info.get("is_enabled", False):
                        result = self.device_interface.read_data(servo_id, "servo_read")
                        if result.get("success", False):
                            joints.append({
                                "id": servo_id,
                                "name": f"Joint {servo_id}",
                                "position": result.get("position", 0.0),
                                "actual": result.get("actual", 0.0),
                                "velocity": result.get("velocity", 0.0),
                                "torque": result.get("torque", 0.0),
                                "hardware": True
                            })
                        else:
                            joints.append({
                                "id": servo_id,
                                "name": f"Joint {servo_id}",
                                "position": 0.0,
                                "actual": 0.0,
                                "velocity": 0.0,
                                "torque": 0.0,
                                "error": result.get("error", "Read failed"),
                                "hardware": False
                            })
                
                # Read motor positions
                for motor_id, motor_info in self.motors.items():
                    if motor_info.get("is_enabled", False):
                        result = self.device_interface.read_data(motor_id, "motor_read")
                        if result.get("success", False):
                            joints.append({
                                "id": motor_id,
                                "name": f"Motor {motor_id}",
                                "position": result.get("position", 0.0),
                                "actual": result.get("actual", 0.0),
                                "velocity": result.get("velocity", 0.0),
                                "torque": result.get("torque", 0.0),
                                "hardware": True
                            })
                        else:
                            joints.append({
                                "id": motor_id,
                                "name": f"Motor {motor_id}",
                                "position": 0.0,
                                "actual": 0.0,
                                "velocity": 0.0,
                                "torque": 0.0,
                                "error": result.get("error", "Read failed"),
                                "hardware": False
                            })
            
            return joints
        except Exception as e:
            logger.error(f"Failed to get joint positions: {e}")
            return []
    
    async def set_safety_limits(self, limits: Dict[str, Any]) -> Dict[str, Any]:
        """Set safety limits (async)"""
        try:
            # Update safety system configuration
            if "max_joint_velocity" in limits:
                self.config["max_servo_torque"] = limits["max_joint_velocity"]
            if "max_joint_torque" in limits:
                self.config["max_servo_torque"] = limits["max_joint_torque"]
            if "max_temperature" in limits:
                self.config["max_motor_current"] = limits["max_temperature"]
            
            # Update safety system
            self.safety_system["velocity_limit"] = True
            self.safety_system["position_limit"] = True
            self.safety_system["overtemperature_protection"] = True
            
            return {
                "success": True,
                "message": "Safety limits updated"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """Execute emergency stop (async)"""
        try:
            self.safety_system["emergency_stop"] = True
            
            # Stop all motors and servos
            for motor_id in self.motors:
                self.motors[motor_id]["is_enabled"] = False
            for servo_id in self.servos:
                self.servos[servo_id]["is_enabled"] = False
            
            return {
                "success": True,
                "message": "Emergency stop executed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def soft_stop(self) -> Dict[str, Any]:
        """Execute soft stop (async)"""
        try:
            # Gradually reduce control signals
            # For safety, disable devices
            for motor_id in self.motors:
                if self.motors[motor_id].get("is_enabled", False):
                    self.motors[motor_id]["is_enabled"] = False
            for servo_id in self.servos:
                if self.servos[servo_id].get("is_enabled", False):
                    self.servos[servo_id]["is_enabled"] = False
            
            return {
                "success": True,
                "message": "Soft stop executed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def reset_safety(self) -> Dict[str, Any]:
        """Reset safety system (async)"""
        try:
            self.safety_system["emergency_stop"] = False
            
            return {
                "success": True,
                "message": "Safety system reset"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def pause(self) -> Dict[str, Any]:
        """Pause hardware operations (async)"""
        try:
            # Mark all devices as paused
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id, device_info in devices.items():
                    device_info["paused"] = True
            
            return {
                "success": True,
                "message": "Hardware paused"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def resume(self) -> Dict[str, Any]:
        """Resume hardware operations (async)"""
        try:
            # Mark all devices as resumed
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id, device_info in devices.items():
                    device_info["paused"] = False
            
            return {
                "success": True,
                "message": "Hardware resumed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop(self) -> Dict[str, Any]:
        """Stop hardware operations (async)"""
        try:
            # Disable all devices
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id, device_info in devices.items():
                    device_info["is_enabled"] = False
            
            return {
                "success": True,
                "message": "Hardware stopped"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a training action using real hardware (async)"""
        try:
            action_type = action.get("type", "unknown")
            
            if action_type == "joint_movement":
                joints = action.get("joints", {})
                duration = action.get("duration", 0.5)
                results = {}
                errors = []
                
                # Execute real hardware movement for each joint
                for joint_id, position in joints.items():
                    if joint_id in self.servos:
                        result = self.control_servo(joint_id, position)
                        results[joint_id] = result
                        if not result.get("success", False):
                            errors.append(f"Servo {joint_id}: {result.get('error', 'Unknown error')}")
                    elif joint_id in self.motors:
                        result = self.control_motor(joint_id, position)
                        results[joint_id] = result
                        if not result.get("success", False):
                            errors.append(f"Motor {joint_id}: {result.get('error', 'Unknown error')}")
                    else:
                        results[joint_id] = {"success": False, "error": f"Joint {joint_id} not found"}
                        errors.append(f"Joint {joint_id} not found")
                
                # Wait for movement completion (real hardware movement time)
                if duration > 0:
                    await asyncio.sleep(duration)
                
                success = len(errors) == 0
                return {
                    "success": success,
                    "action_type": action_type,
                    "duration": duration,
                    "joints_moved": list(joints.keys()),
                    "results": results,
                    "errors": errors if not success else [],
                    "message": f"Joint movement executed using real hardware" if success else f"Joint movement failed: {', '.join(errors)}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported action type: {action_type}. Only 'joint_movement' supported."
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_stage(self, stage_id: str) -> Dict[str, Any]:
        """Test a training stage with real hardware (async)"""
        try:
            if not self.hardware_initialized:
                return {
                    "success": False,
                    "stage_id": stage_id,
                    "error": "Hardware not initialized. Cannot test stage."
                }
            
            test_results = {
                "stage_id": stage_id,
                "hardware_ready": False,
                "sensors_ready": False,
                "motors_ready": False,
                "servos_ready": False,
                "cameras_ready": False,
                "errors": []
            }
            
            # Test sensor hardware
            sensor_data = await self.get_sensor_data()
            if sensor_data:
                test_results["sensors_ready"] = True
            else:
                test_results["errors"].append("No sensor data available")
            
            # Test joint hardware
            joint_positions = await self.get_joint_positions()
            if joint_positions:
                test_results["motors_ready"] = any("motor" in joint["id"] for joint in joint_positions)
                test_results["servos_ready"] = any("servo" in joint["id"] for joint in joint_positions)
            else:
                test_results["errors"].append("No joint position data available")
            
            # Test camera hardware if available
            if self.camera_manager:
                cameras = self.get_camera_frames(list(self.cameras.keys()))
                test_results["cameras_ready"] = any("error" not in frame for frame in cameras.values())
                if not test_results["cameras_ready"]:
                    test_results["errors"].append("Camera hardware test failed")
            
            # Overall hardware readiness
            test_results["hardware_ready"] = (
                test_results["sensors_ready"] and 
                (test_results["motors_ready"] or test_results["servos_ready"])
            )
            
            success = test_results["hardware_ready"] and len(test_results["errors"]) == 0
            
            return {
                "success": success,
                "stage_id": stage_id,
                **test_results,
                "message": f"Stage {stage_id} test passed" if success else f"Stage {stage_id} test failed: {', '.join(test_results['errors'])}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage_id": stage_id
            }
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize robot hardware (synchronous version)"""
        try:
            # Call existing initialization logic
            self._initialize_hardware_interfaces()
            
            if self.hardware_initialized:
                return {
                    "success": True,
                    "message": "Robot hardware initialized successfully",
                    "joints": list(self.servos.keys()) + list(self.motors.keys()),
                    "sensors": list(self.sensors.keys()),
                    "cameras": list(self.cameras.keys())
                }
            else:
                return {
                    "success": False,
                    "error": "Hardware initialization failed"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure robot hardware for specific task"""
        try:
            # Store configuration
            self.config.update(config)
            
            # Configure joints
            joints = config.get("joints", [])
            for joint in joints:
                if joint in self.servos:
                    self.servos[joint]["configured"] = True
                elif joint in self.motors:
                    self.motors[joint]["configured"] = True
            
            # Configure sensors
            sensors = config.get("sensors", [])
            for sensor in sensors:
                if sensor in self.sensors:
                    self.sensors[sensor]["configured"] = True
            
            # Configure cameras
            cameras = config.get("cameras", [])
            for camera in cameras:
                if camera in self.cameras:
                    self.cameras[camera]["configured"] = True
            
            return {
                "success": True,
                "message": f"Hardware configured: {len(joints)} joints, {len(sensors)} sensors, {len(cameras)} cameras",
                "configured_joints": joints,
                "configured_sensors": sensors,
                "configured_cameras": cameras
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current hardware state"""
        with self.data_lock:
            # Get hardware status
            status = self.get_hardware_status()
            
            # Get current joint positions
            joint_positions = {}
            for servo_id, servo in self.servos.items():
                if "current_position" in servo:
                    joint_positions[servo_id] = servo["current_position"]
            
            for motor_id, motor in self.motors.items():
                if "current_position" in motor:
                    joint_positions[motor_id] = motor["current_position"]
            
            # Get sensor data
            sensor_data = {}
            for sensor_id in self.sensors.keys():
                sensor_data[sensor_id] = self.sensor_data.get(sensor_id, {})
            
            # Get battery and temperature
            battery_level = self.sensor_data.get("battery", {}).get("percentage", 0)
            system_temperature = self.sensor_data.get("temp", {}).get("value", 0)
            
            return {
                "joints_connected": len(self.servos) + len(self.motors),
                "sensors_connected": len(self.sensors),
                "cameras_connected": len(self.cameras),
                "battery_level": battery_level,
                "system_temperature": system_temperature,
                "joint_velocities": {},  # In real implementation, this would track velocities
                "joint_torques": {},     # In real implementation, this would track torques
                "temperatures": {"system": system_temperature},
                "joint_positions": joint_positions,
                "sensor_data": sensor_data,
                "status_overview": status
            }
    
    def get_joint_positions(self) -> Dict[str, float]:
        """Get current joint positions (synchronous version)"""
        positions = {}
        with self.data_lock:
            for servo_id, servo in self.servos.items():
                if "current_position" in servo:
                    positions[servo_id] = servo["current_position"]
            
            for motor_id, motor in self.motors.items():
                if "current_position" in motor:
                    positions[motor_id] = motor["current_position"]
        
        return positions
    
    def set_joint_positions(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Set joint positions (synchronous version)"""
        try:
            results = {}
            for joint_id, position in positions.items():
                if joint_id in self.servos:
                    # Check safety
                    if not self._check_servo_safety(joint_id, position):
                        return {
                            "success": False,
                            "error": f"Safety check failed for joint {joint_id} position {position}"
                        }
                    
                    # Set position
                    result = self.control_servo(joint_id, position)
                    results[joint_id] = result
                    
                elif joint_id in self.motors:
                    # Check safety
                    if not self._check_motor_safety(joint_id, position):
                        return {
                            "success": False,
                            "error": f"Safety check failed for joint {joint_id} position {position}"
                        }
                    
                    # Set position
                    result = self.control_motor(joint_id, position)
                    results[joint_id] = result
                    
                else:
                    results[joint_id] = {"success": False, "error": f"Joint {joint_id} not found"}
            
            # Check if all operations were successful
            all_successful = all(result.get("success", False) for result in results.values())
            
            return {
                "success": all_successful,
                "results": results,
                "message": f"Set positions for {len(positions)} joints" if all_successful else "Some joints failed to move"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_sensor_data(self, sensor_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get sensor data for multiple sensors"""
        data = {}
        with self.data_lock:
            if sensor_ids is None:
                sensor_ids = list(self.sensors.keys())
            
            for sensor_id in sensor_ids:
                if sensor_id not in self.sensors:
                    data[sensor_id] = {"error": f"Sensor {sensor_id} not registered"}
                    continue
                
                # Try robot driver first if available
                if self.robot_driver and self.robot_driver.connected:
                    try:
                        # Map sensor ID to robot sensor name
                        robot_sensor_name = self._map_sensor_to_robot_sensor(sensor_id)
                        if robot_sensor_name:
                            # Get sensor value from robot driver
                            sensor_value = self.robot_driver.get_sensor_value(robot_sensor_name)
                            
                            if sensor_value is not None:
                                # Format sensor data
                                formatted_data = self._format_sensor_data(sensor_id, sensor_value)
                                
                                # Update cache
                                self.sensor_data[sensor_id] = formatted_data
                                
                                data[sensor_id] = formatted_data
                                continue  # Skip legacy interface
                    except Exception as e:
                        logger.error(f"Robot driver sensor read error for {sensor_id}: {e}")
                        # Fall through to legacy interface
                
                # Try to get from data cache
                if sensor_id in self.sensor_data:
                    data[sensor_id] = self.sensor_data[sensor_id]
                else:
                    data[sensor_id] = {"error": "No data available"}
        
        return data
    
    def get_camera_frames(self, camera_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get camera frames from real hardware"""
        frames = {}
        with self.data_lock:
            if camera_ids is None:
                camera_ids = list(self.cameras.keys())
            
            for camera_id in camera_ids:
                if camera_id in self.cameras:
                    try:
                        # Get frame from real camera hardware
                        if self.camera_manager:
                            frame_data = self.camera_manager.get_frame(camera_id)
                            if frame_data and frame_data.get("success", False):
                                frames[camera_id] = {
                                    "frame_id": frame_data.get("frame_id", f"frame_{int(time.time() * 1000)}"),
                                    "timestamp": frame_data.get("timestamp", datetime.now().isoformat()),
                                    "width": frame_data.get("width", 640),
                                    "height": frame_data.get("height", 480),
                                    "format": frame_data.get("format", "rgb"),
                                    "data": frame_data.get("data"),  # Actual frame data
                                    "hardware": True
                                }
                            else:
                                frames[camera_id] = {
                                    "error": frame_data.get("error", "Failed to capture frame"),
                                    "hardware": False
                                }
                        else:
                            frames[camera_id] = {
                                "error": "Camera manager not available",
                                "hardware": False
                            }
                    except Exception as e:
                        frames[camera_id] = {
                            "error": f"Camera error: {str(e)}",
                            "hardware": False
                        }
                else:
                    frames[camera_id] = {"error": "Camera not found"}
        
        return frames
    
    def get_sensors(self) -> Dict[str, Any]:
        """Get all sensors
        
        Returns:
            Dictionary of all registered sensors
        """
        return self.sensors
    
    def get_actuators(self) -> Dict[str, Any]:
        """Get all actuators (motors and servos)
        
        Returns:
            Dictionary of all registered actuators
        """
        actuators = {}
        actuators.update(self.motors)
        actuators.update(self.servos)
        return actuators
    
    def get_devices(self) -> Dict[str, Any]:
        """Get all devices (sensors, motors, servos, cameras)
        
        Returns:
            Dictionary of all registered devices
        """
        devices = {}
        devices.update(self.sensors)
        devices.update(self.motors)
        devices.update(self.servos)
        devices.update(self.cameras)
        return devices
    
    def pause(self) -> Dict[str, Any]:
        """Pause hardware operations (synchronous version)"""
        try:
            # Mark all devices as paused
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id, device_info in devices.items():
                    device_info["paused"] = True
            
            return {
                "success": True,
                "message": "Hardware paused"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def stop(self) -> Dict[str, Any]:
        """Stop hardware operations (synchronous version)"""
        try:
            # Disable all devices
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id, device_info in devices.items():
                    device_info["is_enabled"] = False
            
            return {
                "success": True,
                "message": "Hardware stopped"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop (synchronous version)"""
        try:
            self.safety_system["emergency_stop"] = True
            
            # Call existing emergency stop logic
            self._emergency_stop_all()
            
            return {
                "success": True,
                "message": "Emergency stop activated"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def release_resources(self) -> Dict[str, Any]:
        """Release hardware resources"""
        try:
            # Disconnect all devices
            for device_type, devices in [("sensors", self.sensors), ("motors", self.motors), ("servos", self.servos)]:
                for device_id in list(devices.keys()):
                    devices[device_id]["is_enabled"] = False
            
            # Clear data cache
            with self.data_lock:
                self.sensor_data.clear()
                self.camera_frames.clear()
                self.motor_states.clear()
                self.servo_positions.clear()
            
            return {
                "success": True,
                "message": "Hardware resources released"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown hardware interface"""
        try:
            # Release resources
            self.release_resources()
            
            # Stop control threads
            for thread_id, thread in self.control_threads.items():
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            # Clear hardware state
            self.hardware_initialized = False
            
            return {
                "success": True,
                "message": "Robot hardware interface shutdown completed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self) -> Dict[str, Any]:
        """Cleanup hardware interface (alias for shutdown)"""
        return self.shutdown()
    
    def _emergency_stop_all(self):
        """Internal emergency stop for all devices"""
        # Stop all motors
        for motor_id in self.motors.keys():
            try:
                self._emergency_stop_motor(motor_id)
            except Exception as e:
                logger.error(f"Failed to emergency stop motor {motor_id}: {str(e)}")
        
        # Stop all servos
        for servo_id in self.servos.keys():
            try:
                self._emergency_stop_servo(servo_id)
            except Exception as e:
                logger.error(f"Failed to emergency stop servo {servo_id}: {str(e)}")
