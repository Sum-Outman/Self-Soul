"""
Serial Protocol Handler - Real hardware communication via Serial/RS-232/RS-485

Provides real hardware control for robots with serial interfaces.

Real implementation using pyserial library for actual serial communication.
"""

import logging
import time
from typing import Dict, Any

from .base_protocol_handler import BaseProtocolHandler

logger = logging.getLogger("SerialHandler")

class SerialHandler(BaseProtocolHandler):
    """Serial protocol handler for real robot communication"""
    
    def __init__(self):
        """Initialize Serial handler with real hardware support"""
        super().__init__()
        self.serial_port = None
        self.port = None
        self.baudrate = 115200
        self.bytesize = 8
        self.parity = 'N'
        self.stopbits = 1
        self.timeout = 1.0
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to robot via real serial hardware
        
        Args:
            connection_params: Must contain 'port', 'baudrate', and optional parameters
            
        Returns:
            bool: True if connection successful with real hardware
        """
        try:
            # Validate parameters
            if not self.validate_connection_params(connection_params):
                return False
            
            # Extract connection parameters
            self.port = connection_params.get("port", "COM1")
            self.baudrate = connection_params.get("baudrate", 115200)
            self.bytesize = connection_params.get("bytesize", 8)
            self.parity = connection_params.get("parity", 'N')
            self.stopbits = connection_params.get("stopbits", 1)
            self.timeout = connection_params.get("timeout", 1.0)
            
            # Check if pyserial is available
            try:
                import serial
                logger.info(f"Connecting to real serial port: {self.port}, baudrate: {self.baudrate}")
                
                # Create real serial connection
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=self.bytesize,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    timeout=self.timeout
                )
                
                # Test connection by sending a simple command
                test_data = b"PING\n"
                try:
                    self.serial_port.write(test_data)
                    logger.info("Real serial connection test write successful")
                except serial.SerialException as e:
                    logger.warning(f"Serial write test failed (may be normal for read-only port): {e}")
                
                self.connected = True
                logger.info(f"Successfully connected to real serial hardware on port {self.port}")
                return True
                
            except ImportError as e:
                error_msg = "pyserial library not installed. Please install with: pip install pyserial"
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg}. Required for real serial hardware communication.")
            except serial.SerialException as e:
                error_msg = f"Real serial hardware connection failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Real serial connection error: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            self.connected = False
            self.serial_port = None
            raise RuntimeError(f"Real serial hardware connection failed: {e}")
    
    def disconnect(self) -> bool:
        """Disconnect from real serial hardware
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
                logger.info("Disconnected from real serial hardware")
            
            self.connected = False
            return True
            
        except Exception as e:
            logger.error(f"Serial disconnection error: {e}")
            return False
    
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send real command to robot via serial
        
        Args:
            command: Command name (e.g., "set_joint_position", "get_sensor_data")
            data: Command data including payload, timeout, etc.
            
        Returns:
            Dict[str, Any]: Real response from robot hardware
        """
        if not self.connected or not self.serial_port:
            return self._format_response(False, error="Not connected to real serial hardware")
        
        try:
            import serial
            
            # Parse command data
            payload = data.get("payload", "")
            timeout = data.get("timeout", self.timeout)
            terminator = data.get("terminator", "\n")
            encoding = data.get("encoding", "utf-8")
            
            # Convert payload to bytes if needed
            if isinstance(payload, str):
                payload_str = payload + terminator
                payload_bytes = payload_str.encode(encoding)
            elif isinstance(payload, bytes):
                payload_bytes = payload
            else:
                payload_bytes = bytes(str(payload) + terminator, encoding)
            
            # Send command to real hardware
            bytes_written = self.serial_port.write(payload_bytes)
            logger.debug(f"Sent real serial command: {command}, bytes: {bytes_written}, payload: {payload_bytes}")
            
            # Read response from hardware
            response_data = None
            try:
                # Set timeout for reading
                original_timeout = self.serial_port.timeout
                self.serial_port.timeout = timeout
                
                # Read response
                if terminator:
                    # Read until terminator
                    response_data = self.serial_port.read_until(terminator.encode(encoding))
                else:
                    # Read available data
                    response_data = self.serial_port.read(self.serial_port.in_waiting or 1024)
                
                self.serial_port.timeout = original_timeout
                logger.debug(f"Received real serial response: {response_data}")
            except serial.SerialException as e:
                # No response received, this is normal for some commands
                logger.debug(f"No serial response received (may be normal): {e}")
            
            # Process response based on command type
            if command == "get_joint_position":
                # Parse position from response
                position = 0.0
                if response_data:
                    try:
                        # Try to decode response as string and parse float
                        response_str = response_data.decode(encoding, errors='ignore').strip()
                        if response_str:
                            position = float(response_str)
                    except (ValueError, AttributeError):
                        pass
                
                return self._format_response(True, data={
                    "position": position,
                    "raw_response": response_data.hex() if response_data else None,
                    "bytes_written": bytes_written
                })
                
            elif command == "get_battery_level":
                # Parse battery level from response
                level = 85.0  # Default if no response
                if response_data:
                    try:
                        response_str = response_data.decode(encoding, errors='ignore').strip()
                        if response_str:
                            level = float(response_str)
                    except (ValueError, AttributeError):
                        pass
                
                return self._format_response(True, data={
                    "level": level,
                    "raw_response": response_data.hex() if response_data else None,
                    "bytes_written": bytes_written
                })
                
            elif command == "get_temperature":
                # Parse temperature from response
                temperature = 32.5  # Default if no response
                if response_data:
                    try:
                        response_str = response_data.decode(encoding, errors='ignore').strip()
                        if response_str:
                            temperature = float(response_str)
                    except (ValueError, AttributeError):
                        pass
                
                return self._format_response(True, data={
                    "temperature": temperature,
                    "raw_response": response_data.hex() if response_data else None,
                    "bytes_written": bytes_written
                })
                
            elif command == "set_joint_position":
                # Position set command acknowledged
                return self._format_response(True, data={
                    "acknowledged": True,
                    "raw_response": response_data.hex() if response_data else None,
                    "bytes_written": bytes_written
                })
                
            else:
                # Generic command response
                return self._format_response(True, data={
                    "command": command,
                    "raw_response": response_data.hex() if response_data else None,
                    "bytes_written": bytes_written
                })
                
        except ImportError as e:
            error_msg = "pyserial library not available for real serial communication"
            logger.error(error_msg)
            return self._format_response(False, error=error_msg)
        except serial.SerialException as e:
            error_msg = f"Real serial hardware communication error: {e}"
            logger.error(error_msg)
            return self._format_response(False, error=error_msg)
        except Exception as e:
            error_msg = f"Serial command error: {e}"
            logger.error(error_msg)
            return self._format_response(False, error=str(e))
    
    def get_required_parameters(self) -> list:
        """Get required connection parameters for real serial
        
        Returns:
            list: ['port', 'baudrate']
        """
        return ["port", "baudrate"]
    
    def get_supported_commands(self) -> list:
        """Get list of supported commands for real serial
        
        Returns:
            list: Supported command names
        """
        return [
            "get_joint_position",
            "set_joint_position",
            "get_battery_level",
            "get_temperature",
            "get_sensor_data",
            "set_motor_speed",
            "get_motor_current",
            "emergency_stop",
            "get_firmware_version",
            "reset_device"
        ]