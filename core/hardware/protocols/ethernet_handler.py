"""
Ethernet Protocol Handler - Real hardware communication via Ethernet/TCP/IP

Provides real hardware control for robots with Ethernet interfaces.
"""

import socket
import json
import logging
from typing import Dict, Any
import time

from .base_protocol_handler import BaseProtocolHandler

logger = logging.getLogger("EthernetHandler")

class EthernetHandler(BaseProtocolHandler):
    """Ethernet/TCP/IP protocol handler for robot communication"""
    
    def __init__(self):
        """Initialize Ethernet handler"""
        super().__init__()
        self.socket = None
        self.host = None
        self.port = None
        self.timeout = 5.0
        self.buffer_size = 4096
        
        # Robot command set
        self.commands = {
            "initialize": self._handle_initialize,
            "get_joint_position": self._handle_get_joint_position,
            "set_joint_position": self._handle_set_joint_position,
            "set_joint_positions": self._handle_set_joint_positions,
            "get_sensor_value": self._handle_get_sensor_value,
            "get_battery_level": self._handle_get_battery_level,
            "get_temperature": self._handle_get_temperature,
            "emergency_stop": self._handle_emergency_stop,
            "clear_emergency_stop": self._handle_clear_emergency_stop,
            "calibrate_sensors": self._handle_calibrate_sensors,
            "calibrate_joints": self._handle_calibrate_joints,
        }
    
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to robot via Ethernet
        
        Args:
            connection_params: Must contain 'host' and 'port'
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Validate parameters
            if not self.validate_connection_params(connection_params):
                return False
            
            self.host = connection_params["host"]
            self.port = int(connection_params["port"])
            self.timeout = connection_params.get("timeout", 5.0)
            self.buffer_size = connection_params.get("buffer_size", 4096)
            
            logger.info(f"Connecting to robot at {self.host}:{self.port}...")
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # Connect to robot
            self.socket.connect((self.host, self.port))
            
            # Test connection
            test_result = self._send_raw_command({"command": "ping", "data": {}})
            
            if test_result and test_result.get("success"):
                self.connected = True
                logger.info(f"Successfully connected to robot at {self.host}:{self.port}")
                return True
            else:
                logger.error(f"Connection test failed: {test_result}")
                self.disconnect()
                return False
                
        except socket.timeout:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            return False
        except ConnectionRefusedError:
            logger.error(f"Connection refused by {self.host}:{self.port}")
            return False
        except Exception as e:
            logger.error(f"Ethernet connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from robot
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.socket:
                # Send disconnect command
                try:
                    self._send_raw_command({"command": "disconnect", "data": {}})
                except Exception as e:
                    logger.debug(f"Failed to send disconnect command: {e}")
                
                # Close socket
                self.socket.close()
                self.socket = None
            
            self.connected = False
            logger.info("Disconnected from robot")
            return True
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
            return False
    
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to robot
        
        Args:
            command: Command name
            data: Command data
            
        Returns:
            Dict[str, Any]: Response from robot
        """
        if not self.connected or not self.socket:
            return self._format_response(False, error="Not connected to robot")
        
        try:
            # Check if command is supported
            if command not in self.commands:
                return self._format_response(False, error=f"Unsupported command: {command}")
            
            # Send command to robot
            response = self._send_raw_command({
                "command": command,
                "data": data,
                "timestamp": time.time()
            })
            
            if not response:
                return self._format_response(False, error="No response from robot")
            
            # Convert to standard format
            return self._format_response(
                response.get("success", False),
                data=response.get("data", {}),
                error=response.get("error")
            )
            
        except socket.timeout:
            logger.error(f"Command timeout: {command}")
            return self._format_response(False, error=f"Command timeout: {command}")
        except Exception as e:
            logger.error(f"Command error: {e}")
            return self._format_response(False, error=str(e))
    
    def get_required_parameters(self) -> list:
        """Get required connection parameters
        
        Returns:
            list: ['host', 'port']
        """
        return ["host", "port"]
    
    def _send_raw_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send raw command to robot and get response
        
        Args:
            command_data: Command data to send
            
        Returns:
            Dict[str, Any]: Response from robot
        """
        try:
            # Serialize command
            command_json = json.dumps(command_data)
            
            # Send command
            self.socket.sendall(command_json.encode('utf-8') + b'\n')
            
            # Receive response
            response_data = b""
            while True:
                chunk = self.socket.recv(self.buffer_size)
                if not chunk:
                    break
                
                response_data += chunk
                if b'\n' in chunk:
                    break
            
            # Parse response
            response_str = response_data.decode('utf-8').strip()
            response = json.loads(response_str)
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"Communication error: {e}")
            return {"success": False, "error": str(e)}
    
    # ===== Command Handlers =====
    # These methods format commands for the robot and parse responses
    
    def _handle_initialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize command"""
        return self._send_raw_command({"command": "initialize", "data": data})
    
    def _handle_get_joint_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_joint_position command"""
        return self._send_raw_command({"command": "get_joint_position", "data": data})
    
    def _handle_set_joint_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set_joint_position command"""
        return self._send_raw_command({"command": "set_joint_position", "data": data})
    
    def _handle_set_joint_positions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set_joint_positions command"""
        return self._send_raw_command({"command": "set_joint_positions", "data": data})
    
    def _handle_get_sensor_value(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_sensor_value command"""
        return self._send_raw_command({"command": "get_sensor_value", "data": data})
    
    def _handle_get_battery_level(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_battery_level command"""
        return self._send_raw_command({"command": "get_battery_level", "data": data})
    
    def _handle_get_temperature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_temperature command"""
        return self._send_raw_command({"command": "get_temperature", "data": data})
    
    def _handle_emergency_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency_stop command"""
        return self._send_raw_command({"command": "emergency_stop", "data": data})
    
    def _handle_clear_emergency_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear_emergency_stop command"""
        return self._send_raw_command({"command": "clear_emergency_stop", "data": data})
    
    def _handle_calibrate_sensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calibrate_sensors command"""
        return self._send_raw_command({"command": "calibrate_sensors", "data": data})
    
    def _handle_calibrate_joints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calibrate_joints command"""
        return self._send_raw_command({"command": "calibrate_joints", "data": data})