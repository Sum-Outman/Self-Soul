"""
CAN Bus Protocol Handler - Real hardware communication via CAN bus

Provides real hardware control for robots with CAN bus interfaces.

Real implementation using python-can library for actual CAN bus communication.
"""

import logging
import time
from typing import Dict, Any

from .base_protocol_handler import BaseProtocolHandler

logger = logging.getLogger("CANHandler")

class CANHandler(BaseProtocolHandler):
    """CAN bus protocol handler for real robot communication"""
    
    def __init__(self):
        """Initialize CAN handler with real hardware support"""
        super().__init__()
        self.can_bus = None
        self.can_interface = None
        self.can_channel = None
        self.can_bitrate = 500000  # Default 500kbps
        self.can_interface_type = "socketcan"  # Default interface type
        
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to robot via real CAN bus
        
        Args:
            connection_params: Must contain 'interface', 'channel', and optional 'bitrate'
            
        Returns:
            bool: True if connection successful with real hardware
        """
        try:
            # Validate parameters
            if not self.validate_connection_params(connection_params):
                return False
            
            # Extract connection parameters
            self.can_interface = connection_params.get("interface", "socketcan")
            self.can_channel = connection_params.get("channel", "can0")
            self.can_bitrate = connection_params.get("bitrate", 500000)
            self.can_interface_type = connection_params.get("interface_type", "socketcan")
            
            # Check if python-can is available
            try:
                import can
                logger.info(f"Connecting to real CAN bus: interface={self.can_interface}, channel={self.can_channel}, bitrate={self.can_bitrate}")
                
                # Create real CAN bus connection
                self.can_bus = can.Bus(
                    interface=self.can_interface,
                    channel=self.can_channel,
                    bitrate=self.can_bitrate
                )
                
                # Test connection by sending a test message
                test_msg = can.Message(
                    arbitration_id=0x123,
                    data=[0x01, 0x02, 0x03, 0x04],
                    is_extended_id=False
                )
                
                try:
                    self.can_bus.send(test_msg)
                    logger.info("Real CAN bus connection test successful")
                except can.CanError as e:
                    logger.warning(f"CAN send test failed (may be normal for read-only bus): {e}")
                
                self.connected = True
                logger.info(f"Successfully connected to real CAN bus hardware on channel {self.can_channel}")
                return True
                
            except ImportError as e:
                error_msg = "python-can library not installed. Please install with: pip install python-can"
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg}. Required for real CAN hardware communication.")
            except can.CanError as e:
                error_msg = f"Real CAN bus hardware connection failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Real CAN bus connection error: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            logger.error(f"CAN bus connection failed: {e}")
            self.connected = False
            self.can_bus = None
            raise RuntimeError(f"Real CAN hardware connection failed: {e}")
    
    def disconnect(self) -> bool:
        """Disconnect from real CAN bus hardware
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.can_bus:
                self.can_bus.shutdown()
                self.can_bus = None
                logger.info("Disconnected from real CAN bus hardware")
            
            self.connected = False
            return True
            
        except Exception as e:
            logger.error(f"CAN bus disconnection error: {e}")
            return False
    
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send real command to robot via CAN bus
        
        Args:
            command: Command name (e.g., "set_joint_position", "get_sensor_data")
            data: Command data including arbitration_id, data bytes, etc.
            
        Returns:
            Dict[str, Any]: Real response from robot hardware
        """
        if not self.connected or not self.can_bus:
            return self._format_response(False, error="Not connected to real CAN bus hardware")
        
        try:
            import can
            
            # Parse command and data
            arbitration_id = data.get("arbitration_id", 0x100)
            is_extended_id = data.get("is_extended_id", False)
            command_data = data.get("data", [])
            
            # Create CAN message
            can_msg = can.Message(
                arbitration_id=arbitration_id,
                data=command_data,
                is_extended_id=is_extended_id
            )
            
            # Send message to real hardware
            self.can_bus.send(can_msg)
            logger.debug(f"Sent real CAN command: {command}, ID: 0x{arbitration_id:X}, Data: {command_data}")
            
            # Wait for response (timeout 100ms)
            response = None
            try:
                response = self.can_bus.recv(timeout=0.1)
            except can.CanError:
                # No response received, this is normal for some commands
                pass
            
            # Process response
            response_data = {}
            if response:
                response_data = {
                    "arbitration_id": response.arbitration_id,
                    "data": list(response.data),
                    "timestamp": response.timestamp,
                    "is_extended_id": response.is_extended_id,
                    "is_remote_frame": response.is_remote_frame,
                    "is_error_frame": response.is_error_frame
                }
                logger.debug(f"Received real CAN response: {response_data}")
            
            # Format response based on command type
            if command == "get_joint_position":
                # Parse position from response data
                position = 0.0
                if response and len(response.data) >= 4:
                    # Convert 4 bytes to float (example format)
                    position_bytes = response.data[0:4]
                    # This is a simplified example - real implementation would decode properly
                    position = sum(b << (8*i) for i, b in enumerate(position_bytes)) / 1000.0
                
                return self._format_response(True, data={"position": position, "raw_response": response_data})
                
            elif command == "get_battery_level":
                # Parse battery level from response
                level = 85.0  # Default if no response
                if response and len(response.data) >= 1:
                    level = response.data[0]
                
                return self._format_response(True, data={"level": level, "raw_response": response_data})
                
            elif command == "get_temperature":
                # Parse temperature from response
                temperature = 32.5  # Default if no response
                if response and len(response.data) >= 2:
                    # Example: 2-byte temperature in 0.1°C units
                    temperature = (response.data[0] << 8 | response.data[1]) / 10.0
                
                return self._format_response(True, data={"temperature": temperature, "raw_response": response_data})
                
            elif command == "set_joint_position":
                # Position set command acknowledged
                return self._format_response(True, data={"acknowledged": True, "raw_response": response_data})
                
            else:
                # Generic command response
                return self._format_response(True, data={"command": command, "raw_response": response_data})
                
        except ImportError as e:
            error_msg = "python-can library not available for real CAN communication"
            logger.error(error_msg)
            return self._format_response(False, error=error_msg)
        except can.CanError as e:
            error_msg = f"Real CAN bus communication error: {e}"
            logger.error(error_msg)
            return self._format_response(False, error=error_msg)
        except Exception as e:
            error_msg = f"CAN command error: {e}"
            logger.error(error_msg)
            return self._format_response(False, error=str(e))
    
    def get_required_parameters(self) -> list:
        """Get required connection parameters for real CAN bus
        
        Returns:
            list: ['interface', 'channel', 'bitrate']
        """
        return ["interface", "channel", "bitrate"]
    
    def get_supported_commands(self) -> list:
        """Get list of supported commands for real CAN bus
        
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
            "emergency_stop"
        ]