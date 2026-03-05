"""
Base Protocol Handler - Abstract base class for robot communication protocols

Provides common interface for different communication protocols.
Concrete implementations must override connect(), disconnect(), and send_command() methods.
"""

import abc
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("BaseProtocolHandler")

class BaseProtocolHandler(abc.ABC):
    """Abstract base class for protocol handlers"""
    
    def __init__(self):
        """Initialize protocol handler"""
        self.connected = False
        self.config = {}
        self.connection = None
    
    @abc.abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect using this protocol
        
        Args:
            connection_params: Protocol-specific connection parameters
            
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from protocol
        
        Returns:
            bool: True if disconnection successful
        """
        pass
    
    @abc.abstractmethod
    def send_command(self, command: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send command via protocol
        
        Args:
            command: Command name
            data: Command data
            
        Returns:
            Dict[str, Any]: Response with at least 'success' key
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if protocol is connected
        
        Returns:
            bool: True if connected
        """
        return self.connected
    
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get protocol information
        
        Returns:
            Dict[str, Any]: Protocol information
        """
        return {
            "protocol_type": self.__class__.__name__,
            "connected": self.connected,
            "config": self.config,
        }
    
    def validate_connection_params(self, params: Dict[str, Any]) -> bool:
        """Validate connection parameters
        
        Args:
            params: Connection parameters to validate
            
        Returns:
            bool: True if parameters are valid
        """
        # Base implementation checks for required parameters
        required_params = self.get_required_parameters()
        
        for param in required_params:
            if param not in params:
                logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    @abc.abstractmethod
    def get_required_parameters(self) -> list:
        """Get list of required connection parameters
        
        Returns:
            list: List of required parameter names
        """
        pass
    
    def _format_response(self, success: bool, data: Optional[Dict[str, Any]] = None, 
                        error: Optional[str] = None) -> Dict[str, Any]:
        """Format standard response
        
        Args:
            success: Whether command was successful
            data: Response data
            error: Error message if unsuccessful
            
        Returns:
            Dict[str, Any]: Formatted response
        """
        response = {
            "success": success,
            "timestamp": self._get_timestamp(),
        }
        
        if data:
            response.update(data)
        
        if error and not success:
            response["error"] = error
        
        return response
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string
        
        Returns:
            str: ISO format timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()