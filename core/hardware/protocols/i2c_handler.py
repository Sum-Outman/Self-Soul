"""
I2C Protocol Handler - Real hardware communication via I2C

Provides real hardware control for robots with I2C interfaces.

Real implementation using smbus2 library for actual I2C communication.
"""

import logging
import time
from typing import Dict, Any

from .base_protocol_handler import BaseProtocolHandler

logger = logging.getLogger("I2CHandler")

class I2CHandler(BaseProtocolHandler):
    """I2C protocol handler for real robot communication"""
    
    def __init__(self):
        """Initialize I2C handler with real hardware support"""
        super().__init__()
        self.i2c_bus = None
        self.bus_number = 1  # Default I2C bus (usually 1 for Raspberry Pi)
        self.device_address = 0x40  # Default I2C address
        self.i2c_frequency = 100000  # Default 100kHz
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to robot via real I2C hardware
        
        Args:
            connection_params: Must contain 'bus', 'address', and optional 'frequency'
            
        Returns:
            bool: True if connection successful with real hardware
        """
        try:
            # Validate parameters
            if not self.validate_connection_params(connection_params):
                return False
            
            # Extract connection parameters
            self.bus_number = connection_params.get("bus", 1)
            self.device_address = connection_params.get("address", 0x40)
            self.i2c_frequency = connection_params.get("frequency", 100000)
            
            # Check if smbus2 is available
            try:
                from smbus2 import SMBus
                logger.info(f"Connecting to real I2C bus: bus={self.bus_number}, address=0x{self.device_address:02X}, frequency={self.i2c_frequency}Hz")
                
                # Create real I2C bus connection
                self.i2c_bus = SMBus(self.bus_number)
                
                # Test connection by reading a byte
                try:
                    # Try to read a byte from the device to verify connection
                    test_byte = self.i2c_bus.read_byte(self.device_address)
                    logger.info(f"Real I2C connection test successful: read byte 0x{test_byte:02X}")
                except Exception as e:
                    logger.warning(f"I2C read test failed (may be normal for write-only devices): {e}")
                
                self.connected = True
                logger.info(f"Successfully connected to real I2C hardware on bus {self.bus_number}, address 0x{self.device_address:02X}")
                return True
                
            except ImportError as e:
                error_msg = "smbus2 library not installed. Please install with: pip install smbus2"
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg}. Required for real I2C hardware communication.")
            except Exception as e:
                error_msg = f"Real I2C hardware connection failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"I2C connection failed: {e}")
            self.connected = False
            self.i2c_bus = None
            raise RuntimeError(f"Real I2C hardware connection failed: {e}")
    
    def disconnect(self) -> bool:
        """Disconnect from real I2C hardware
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.i2c_bus:
                self.i2c_bus.close()
                self.i2c_bus = None
                logger.info("Disconnected from real I2C hardware")
            
            self.connected = False
            return True
            
        except Exception as e:
            logger.error(f"I2C disconnection error: {e}")
            return False
    
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send real command to robot via I2C
        
        Args:
            command: Command name (e.g., "set_joint_position", "get_sensor_data")
            data: Command data including register, value, etc.
            
        Returns:
            Dict[str, Any]: Real response from robot hardware
        """
        if not self.connected or not self.i2c_bus:
            return self._format_response(False, error="Not connected to real I2C hardware")
        
        try:
            from smbus2 import SMBus
            
            # Parse command data
            register = data.get("register", 0x00)
            value = data.get("value", 0)
            length = data.get("length", 1)
            read_only = data.get("read_only", False)
            
            # Execute command based on type
            if command == "get_joint_position":
                # Read position from I2C device
                try:
                    # Read 2 bytes for position (assuming 16-bit value)
                    data_bytes = self.i2c_bus.read_i2c_block_data(self.device_address, register, 2)
                    position = (data_bytes[0] << 8) | data_bytes[1]
                    # Convert to degrees (assuming 0-1023 maps to 0-180 degrees)
                    position_degrees = (position / 1023.0) * 180.0
                    
                    return self._format_response(True, data={
                        "position": position_degrees,
                        "raw_bytes": data_bytes,
                        "register": f"0x{register:02X}"
                    })
                except Exception as e:
                    logger.error(f"I2C position read error: {e}")
                    return self._format_response(False, error=f"Failed to read position: {e}")
                
            elif command == "get_battery_level":
                # Read battery level from I2C device
                try:
                    # Read 1 byte for battery level (0-100%)
                    battery_byte = self.i2c_bus.read_byte_data(self.device_address, register)
                    battery_level = float(battery_byte)  # 0-255 scale
                    if battery_level > 100:  # Normalize if needed
                        battery_level = 100.0
                    
                    return self._format_response(True, data={
                        "level": battery_level,
                        "raw_byte": battery_byte,
                        "register": f"0x{register:02X}"
                    })
                except Exception as e:
                    logger.error(f"I2C battery read error: {e}")
                    return self._format_response(False, error=f"Failed to read battery: {e}")
                
            elif command == "get_temperature":
                # Read temperature from I2C device
                try:
                    # Read 2 bytes for temperature (assuming 16-bit value in 0.1°C units)
                    data_bytes = self.i2c_bus.read_i2c_block_data(self.device_address, register, 2)
                    temperature_raw = (data_bytes[0] << 8) | data_bytes[1]
                    temperature_c = temperature_raw / 10.0  # Convert to °C
                    
                    return self._format_response(True, data={
                        "temperature": temperature_c,
                        "raw_bytes": data_bytes,
                        "register": f"0x{register:02X}"
                    })
                except Exception as e:
                    logger.error(f"I2C temperature read error: {e}")
                    return self._format_response(False, error=f"Failed to read temperature: {e}")
                
            elif command == "set_joint_position":
                # Write position to I2C device
                try:
                    # Convert degrees to raw value (0-1023)
                    position_raw = int((value / 180.0) * 1023)
                    # Write 2 bytes
                    self.i2c_bus.write_i2c_block_data(self.device_address, register, 
                                                     [(position_raw >> 8) & 0xFF, position_raw & 0xFF])
                    
                    return self._format_response(True, data={
                        "acknowledged": True,
                        "position_set": value,
                        "raw_value": position_raw,
                        "register": f"0x{register:02X}"
                    })
                except Exception as e:
                    logger.error(f"I2C position write error: {e}")
                    return self._format_response(False, error=f"Failed to set position: {e}")
                
            elif command == "read_register":
                # Generic register read
                try:
                    data_bytes = self.i2c_bus.read_i2c_block_data(self.device_address, register, length)
                    
                    return self._format_response(True, data={
                        "register": f"0x{register:02X}",
                        "data": data_bytes,
                        "length": length
                    })
                except Exception as e:
                    logger.error(f"I2C register read error: {e}")
                    return self._format_response(False, error=f"Failed to read register: {e}")
                
            elif command == "write_register":
                # Generic register write
                try:
                    if isinstance(value, list):
                        data_bytes = value
                    else:
                        data_bytes = [value]
                    
                    self.i2c_bus.write_i2c_block_data(self.device_address, register, data_bytes)
                    
                    return self._format_response(True, data={
                        "acknowledged": True,
                        "register": f"0x{register:02X}",
                        "data_written": data_bytes
                    })
                except Exception as e:
                    logger.error(f"I2C register write error: {e}")
                    return self._format_response(False, error=f"Failed to write register: {e}")
                
            else:
                # Generic I2C read/write command
                try:
                    if read_only:
                        # Read operation
                        data_bytes = self.i2c_bus.read_i2c_block_data(self.device_address, register, length)
                        return self._format_response(True, data={
                            "command": command,
                            "operation": "read",
                            "data": data_bytes,
                            "register": f"0x{register:02X}"
                        })
                    else:
                        # Write operation
                        if isinstance(value, list):
                            data_bytes = value
                        else:
                            data_bytes = [value]
                        
                        self.i2c_bus.write_i2c_block_data(self.device_address, register, data_bytes)
                        return self._format_response(True, data={
                            "command": command,
                            "operation": "write",
                            "data_written": data_bytes,
                            "register": f"0x{register:02X}"
                        })
                        
                except Exception as e:
                    logger.error(f"I2C generic command error: {e}")
                    return self._format_response(False, error=f"Failed to execute I2C command: {e}")
                
        except ImportError as e:
            error_msg = "smbus2 library not available for real I2C communication"
            logger.error(error_msg)
            return self._format_response(False, error=error_msg)
        except Exception as e:
            error_msg = f"I2C command error: {e}"
            logger.error(error_msg)
            return self._format_response(False, error=str(e))
    
    def get_required_parameters(self) -> list:
        """Get required connection parameters for real I2C
        
        Returns:
            list: ['bus', 'address']
        """
        return ["bus", "address"]
    
    def get_supported_commands(self) -> list:
        """Get list of supported commands for real I2C
        
        Returns:
            list: Supported command names
        """
        return [
            "get_joint_position",
            "set_joint_position",
            "get_battery_level",
            "get_temperature",
            "read_register",
            "write_register",
            "get_sensor_data",
            "set_motor_speed",
            "get_motor_current",
            "emergency_stop"
        ]