"""
External Device Interface - Communication with External Devices

Provides functionality for communication with external devices, sensors, and actuators
using various protocols like serial, TCP/IP, UDP, and more.
"""

import serial
import socket
import threading
import time
import logging
import json
import struct
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import os
import asyncio
import subprocess
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExternalDeviceInterface")

class ExternalDeviceInterface:
    """Interface for communication with external devices"""
    
    def __init__(self):
        """Initialize the external device interface"""
        self.devices = {}
        self.connections = {}
        self.device_threads = {}
        self.callbacks = {}
        self.lock = threading.Lock()
        self.config = {
            "default_timeout": 5.0,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "buffer_size": 4096
        }
        
        # Supported protocols
        self.supported_protocols = ["serial", "tcp", "udp", "websocket"]
        
    def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize the external device interface with configuration"""
        try:
            if config:
                self.config.update(config)
            
            logger.info("External device interface initialized successfully")
            return {"success": True, "message": "External device interface initialized"}
        except Exception as e:
            logger.error(f"External device interface initialization failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def connect_device(self, device_id: str, protocol: str, 
                      connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to an external device using the specified protocol"""
        try:
            with self.lock:
                if device_id in self.devices:
                    return {"success": False, "error": f"Device {device_id} already connected"}
                
                if protocol not in self.supported_protocols:
                    return {"success": False, "error": f"Unsupported protocol: {protocol}"}
                
                # Create a connection based on protocol
                connection = None
                
                if protocol == "serial":
                    connection = self._connect_serial(device_id, connection_params)
                elif protocol == "tcp":
                    connection = self._connect_tcp(device_id, connection_params)
                elif protocol == "udp":
                    connection = self._connect_udp(device_id, connection_params)
                elif protocol == "websocket":
                    # WebSocket implementation would be more complex and might need additional libraries
                    return {"success": False, "error": "WebSocket protocol is not fully implemented yet"}
                
                if not connection:
                    return {"success": False, "error": f"Failed to create connection for device {device_id}"}
                
                # Store device information
                self.devices[device_id] = {
                    "id": device_id,
                    "protocol": protocol,
                    "params": connection_params,
                    "connection": connection,
                    "is_connected": True,
                    "connected_at": datetime.now().isoformat(),
                    "last_data_received": None
                }
                
                # Add to connections dictionary for direct access
                self.connections[device_id] = connection
                
                logger.info(f"Device {device_id} connected successfully via {protocol} protocol")
                return {"success": True, "device_id": device_id, "protocol": protocol}
        except Exception as e:
            logger.error(f"Failed to connect device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def disconnect_device(self, device_id: str) -> Dict[str, Any]:
        """Disconnect from an external device"""
        try:
            with self.lock:
                if device_id not in self.devices:
                    return {"success": False, "error": f"Device {device_id} not found"}
                
                device = self.devices[device_id]
                connection = device["connection"]
                
                # Close the connection based on protocol
                if device["protocol"] == "serial":
                    connection.close()
                elif device["protocol"] == "tcp":
                    connection.close()
                elif device["protocol"] == "udp":
                    connection.close()
                
                # Stop any active threads for this device
                if device_id in self.device_threads:
                    self.device_threads[device_id].stop()
                    del self.device_threads[device_id]
                
                # Remove device from dictionaries
                del self.devices[device_id]
                if device_id in self.connections:
                    del self.connections[device_id]
                if device_id in self.callbacks:
                    del self.callbacks[device_id]
                
                logger.info(f"Device {device_id} disconnected successfully")
                return {"success": True, "device_id": device_id}
        except Exception as e:
            logger.error(f"Failed to disconnect device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def send_data(self, device_id: str, data: Any, 
                 data_type: str = "raw") -> Dict[str, Any]:
        """Send data to an external device"""
        try:
            with self.lock:
                if device_id not in self.devices:
                    return {"success": False, "error": f"Device {device_id} not found"}
                
                device = self.devices[device_id]
                if not device["is_connected"]:
                    return {"success": False, "error": f"Device {device_id} is not connected"}
                
                connection = device["connection"]
                protocol = device["protocol"]
                
                # Process data based on type
                processed_data = self._process_data_for_sending(data, data_type, protocol)
                
                # Send data based on protocol
                bytes_sent = 0
                
                if protocol == "serial":
                    bytes_sent = connection.write(processed_data)
                    connection.flush()
                elif protocol == "tcp":
                    bytes_sent = connection.send(processed_data)
                elif protocol == "udp":
                    # For UDP, connection is a tuple (socket, address)
                    udp_socket, udp_address = connection
                    bytes_sent = udp_socket.sendto(processed_data, udp_address)
                
                logger.debug(f"Sent {bytes_sent} bytes to device {device_id}")
                return {"success": True, "device_id": device_id, "bytes_sent": bytes_sent}
        except Exception as e:
            logger.error(f"Failed to send data to device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def receive_data(self, device_id: str, 
                    data_type: str = "raw", 
                    max_bytes: int = None) -> Dict[str, Any]:
        """Receive data from an external device"""
        try:
            with self.lock:
                if device_id not in self.devices:
                    return {"success": False, "error": f"Device {device_id} not found"}
                
                device = self.devices[device_id]
                if not device["is_connected"]:
                    return {"success": False, "error": f"Device {device_id} is not connected"}
                
                connection = device["connection"]
                protocol = device["protocol"]
                
                # Use default buffer size if not specified
                buffer_size = max_bytes or self.config["buffer_size"]
                
                # Receive data based on protocol
                received_data = b""
                
                if protocol == "serial":
                    # Check if data is available
                    if connection.in_waiting > 0:
                        received_data = connection.read(connection.in_waiting)
                elif protocol == "tcp":
                    # Set non-blocking to avoid hanging
                    connection.setblocking(False)
                    try:
                        received_data = connection.recv(buffer_size)
                    except BlockingIOError:
                        # No data available
                        pass
                    # Reset to blocking mode
                    connection.setblocking(True)
                elif protocol == "udp":
                    # For UDP, connection is a tuple (socket, address)
                    udp_socket, _ = connection
                    udp_socket.setblocking(False)
                    try:
                        received_data, _ = udp_socket.recvfrom(buffer_size)
                    except BlockingIOError:
                        # No data available
                        pass
                    # Reset to blocking mode
                    udp_socket.setblocking(True)
                
                # Process received data
                if received_data:
                    processed_data = self._process_received_data(received_data, data_type)
                    device["last_data_received"] = datetime.now().isoformat()
                    
                    logger.debug(f"Received {len(received_data)} bytes from device {device_id}")
                    return {
                        "success": True,
                        "device_id": device_id,
                        "data": processed_data,
                        "bytes_received": len(received_data)
                    }
                else:
                    return {
                        "success": True,
                        "device_id": device_id,
                        "data": None,
                        "bytes_received": 0,
                        "message": "No data available"
                    }
        except Exception as e:
            logger.error(f"Failed to receive data from device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def start_data_listener(self, device_id: str, callback: Callable, 
                           data_type: str = "raw", 
                           interval: float = 0.1) -> Dict[str, Any]:
        """Start a background thread to listen for data from a device"""
        try:
            with self.lock:
                if device_id not in self.devices:
                    return {"success": False, "error": f"Device {device_id} not found"}
                
                if device_id in self.device_threads:
                    return {"success": False, "error": f"Listener already running for device {device_id}"}
                
                # Store callback
                self.callbacks[device_id] = callback
                
                # Create and start listener thread
                listener_thread = DeviceListenerThread(
                    device_id, 
                    self, 
                    data_type, 
                    interval
                )
                self.device_threads[device_id] = listener_thread
                listener_thread.start()
                
                logger.info(f"Started data listener for device {device_id}")
                return {"success": True, "device_id": device_id}
        except Exception as e:
            logger.error(f"Failed to start data listener for device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def stop_data_listener(self, device_id: str) -> Dict[str, Any]:
        """Stop the background data listener thread for a device"""
        try:
            with self.lock:
                if device_id not in self.device_threads:
                    return {"success": False, "error": f"No listener running for device {device_id}"}
                
                # Stop the thread
                self.device_threads[device_id].stop()
                self.device_threads[device_id].join(timeout=2.0)
                
                # Remove thread and callback
                del self.device_threads[device_id]
                if device_id in self.callbacks:
                    del self.callbacks[device_id]
                
                logger.info(f"Stopped data listener for device {device_id}")
                return {"success": True, "device_id": device_id}
        except Exception as e:
            logger.error(f"Failed to stop data listener for device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connected device"""
        with self.lock:
            if device_id not in self.devices:
                logger.error(f"Device {device_id} not found")
                return None
            
            device = self.devices[device_id]
            return {
                "id": device["id"],
                "protocol": device["protocol"],
                "params": device["params"],
                "is_connected": device["is_connected"],
                "connected_at": device["connected_at"],
                "last_data_received": device["last_data_received"],
                "has_listener": device_id in self.device_threads
            }
    
    def get_all_devices_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all connected devices"""
        result = {}
        with self.lock:
            for device_id in self.devices:
                result[device_id] = self.get_device_info(device_id)
        return result
    
    def _connect_serial(self, device_id: str, params: Dict[str, Any]) -> Optional[serial.Serial]:
        """Connect to a device using serial port"""
        try:
            # Required parameters
            port = params.get("port")
            baudrate = params.get("baudrate", 9600)
            
            if not port:
                logger.error(f"Serial port not specified for device {device_id}")
                return None
            
            # Optional parameters with defaults
            timeout = params.get("timeout", self.config["default_timeout"])
            bytesize = params.get("bytesize", serial.EIGHTBITS)
            parity = params.get("parity", serial.PARITY_NONE)
            stopbits = params.get("stopbits", serial.STOPBITS_ONE)
            
            # Create serial connection
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout
            )
            
            # Check if connection is open
            if ser.is_open:
                logger.info(f"Serial connection to {port} established for device {device_id}")
                return ser
            else:
                logger.error(f"Failed to open serial port {port} for device {device_id}")
                return None
        except Exception as e:
            logger.error(f"Serial connection error for device {device_id}: {str(e)}")
            return None
    
    def _connect_tcp(self, device_id: str, params: Dict[str, Any]) -> Optional[socket.socket]:
        """Connect to a device using TCP/IP"""
        try:
            # Required parameters
            host = params.get("host")
            port = params.get("port")
            
            if not host or not port:
                logger.error(f"TCP host or port not specified for device {device_id}")
                return None
            
            # Optional parameters with defaults
            timeout = params.get("timeout", self.config["default_timeout"])
            
            # Create TCP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Connect to the device
            sock.connect((host, port))
            
            logger.info(f"TCP connection to {host}:{port} established for device {device_id}")
            return sock
        except Exception as e:
            logger.error(f"TCP connection error for device {device_id}: {str(e)}")
            # Clean up socket
            try:
                sock.close()
            except:
                pass
            return None
    
    def _connect_udp(self, device_id: str, params: Dict[str, Any]) -> Optional[Tuple[socket.socket, Tuple[str, int]]]:
        """Connect to a device using UDP"""
        try:
            # Required parameters
            host = params.get("host")
            port = params.get("port")
            
            if not host or not port:
                logger.error(f"UDP host or port not specified for device {device_id}")
                return None
            
            # Optional parameters with defaults
            timeout = params.get("timeout", self.config["default_timeout"])
            
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            
            # For UDP, we just need to create the socket, not connect
            # Return the socket and address tuple
            udp_address = (host, port)
            
            logger.info(f"UDP socket for device {device_id} created for {host}:{port}")
            return (sock, udp_address)
        except Exception as e:
            logger.error(f"UDP socket error for device {device_id}: {str(e)}")
            # Clean up socket
            try:
                sock.close()
            except:
                pass
            return None
    
    def _process_data_for_sending(self, data: Any, data_type: str, protocol: str) -> bytes:
        """Process data into bytes suitable for sending over the protocol"""
        try:
            if isinstance(data, bytes):
                return data
            
            if data_type == "json":
                # Convert JSON data to bytes
                json_str = json.dumps(data) if not isinstance(data, str) else data
                return json_str.encode('utf-8')
            elif data_type == "text":
                # Convert text to bytes
                return str(data).encode('utf-8')
            elif data_type == "binary":
                # Assume data is already in a binary-friendly format
                if isinstance(data, (list, tuple)):
                    return bytes(data)
                else:
                    return str(data).encode('utf-8')
            else:
                # Default: convert to string and encode
                return str(data).encode('utf-8')
        except Exception as e:
            logger.error(f"Error processing data for sending: {str(e)}")
            # Fallback: convert to string and encode
            return str(data).encode('utf-8')
    
    def _process_received_data(self, data: bytes, data_type: str) -> Any:
        """Process received bytes into the specified data type"""
        try:
            if data_type == "json":
                # Try to parse as JSON
                return json.loads(data.decode('utf-8'))
            elif data_type == "text":
                # Decode as text
                return data.decode('utf-8')
            elif data_type == "binary":
                # Return as bytes
                return data
            elif data_type == "hex":
                # Return as hex string
                return data.hex()
            else:
                # Default: try to decode as text, fallback to hex
                try:
                    return data.decode('utf-8')
                except:
                    return data.hex()
        except Exception as e:
            logger.error(f"Error processing received data: {str(e)}")
            # Fallback: return as hex string
            return data.hex()
    
    def scan_serial_ports(self) -> List[Dict[str, Any]]:
        """Scan for available serial ports"""
        available_ports = []
        
        try:
            if platform.system() == 'Windows':
                # For Windows
                import winreg
                
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                        "HARDWARE\DEVICEMAP\SERIALCOMM")
                    i = 0
                    while True:
                        try:
                            value_name, value, _ = winreg.EnumValue(key, i)
                            port_info = {
                                "port": value,
                                "name": value_name,
                                "system": "Windows"
                            }
                            available_ports.append(port_info)
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                except Exception as e:
                    logger.warning(f"Windows registry scan failed: {str(e)}")
                    # Fallback: try COM ports 1-20
                    for i in range(1, 21):
                        port = f"COM{i}"
                        try:
                            ser = serial.Serial(port, timeout=0.1)
                            ser.close()
                            port_info = {
                                "port": port,
                                "name": f"Serial Port {i}",
                                "system": "Windows"
                            }
                            available_ports.append(port_info)
                        except (serial.SerialException, FileNotFoundError):
                            continue
            else:
                # For Linux/Mac
                import glob
                
                # Try to find tty devices
                pattern = '/dev/ttyUSB*' if platform.system() == 'Linux' else '/dev/tty.*'
                for port in glob.glob(pattern):
                    try:
                        ser = serial.Serial(port, timeout=0.1)
                        ser.close()
                        port_info = {
                            "port": port,
                            "name": port,
                            "system": platform.system()
                        }
                        available_ports.append(port_info)
                    except (serial.SerialException, FileNotFoundError):
                        continue
        except Exception as e:
            logger.error(f"Serial port scan failed: {str(e)}")
        
        # Deduplicate ports
        unique_ports = []
        seen_ports = set()
        for port in available_ports:
            if port["port"] not in seen_ports:
                unique_ports.append(port)
                seen_ports.add(port["port"])
        
        logger.info(f"Found {len(unique_ports)} available serial ports")
        return unique_ports
    
    def ping_device(self, device_id: str) -> Dict[str, Any]:
        """Ping a device to check connectivity"""
        try:
            with self.lock:
                if device_id not in self.devices:
                    return {"success": False, "error": f"Device {device_id} not found"}
                
                device = self.devices[device_id]
                protocol = device["protocol"]
                
                # Simple ping implementation based on protocol
                start_time = time.time()
                
                if protocol == "serial":
                    # For serial, we can try to send a simple command and wait for response
                    # This depends on the device's protocol
                    result = self.send_data(device_id, "PING\r\n", "text")
                    if result["success"]:
                        # Wait a short time for response
                        time.sleep(0.1)
                        response = self.receive_data(device_id, "text")
                        if response["success"] and response["data"]:
                            latency = time.time() - start_time
                            return {
                                "success": True,
                                "device_id": device_id,
                                "latency": latency,
                                "response": response["data"]
                            }
                elif protocol == "tcp":
                    # For TCP, we can check if the socket is still connected
                    try:
                        # Send a small packet to check connectivity
                        dummy_data = b"\x00"
                        device["connection"].send(dummy_data)
                        latency = time.time() - start_time
                        return {
                            "success": True,
                            "device_id": device_id,
                            "latency": latency,
                            "status": "connected"
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "device_id": device_id,
                            "status": "disconnected"
                        }
                elif protocol == "udp":
                    # UDP is connectionless, so ping is more complex
                    # We would need a specific protocol implementation
                    return {
                        "success": True,
                        "device_id": device_id,
                        "status": "UDP is connectionless, ping not reliable",
                        "protocol": "udp"
                    }
                
                return {
                    "success": True,
                    "device_id": device_id,
                    "status": "ping completed",
                    "protocol": protocol
                }
        except Exception as e:
            logger.error(f"Failed to ping device {device_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        # Disconnect all devices
        for device_id in list(self.devices.keys()):
            self.disconnect_device(device_id)

class DeviceListenerThread(threading.Thread):
    """Thread for listening to data from an external device"""
    
    def __init__(self, device_id: str, interface: ExternalDeviceInterface, 
                 data_type: str = "raw", interval: float = 0.1):
        """Initialize the listener thread"""
        super().__init__(daemon=True)
        self.device_id = device_id
        self.interface = interface
        self.data_type = data_type
        self.interval = interval
        self.running = False
    
    def run(self):
        """Thread run method"""
        self.running = True
        
        while self.running:
            try:
                # Receive data from the device
                result = self.interface.receive_data(self.device_id, self.data_type)
                
                # If data was received and a callback exists, call it
                if result["success"] and result["data"] is not None:
                    with self.interface.lock:
                        if self.device_id in self.interface.callbacks:
                            try:
                                self.interface.callbacks[self.device_id](
                                    self.device_id, 
                                    result["data"]
                                )
                            except Exception as e:
                                logger.error(f"Error in device {self.device_id} callback: {str(e)}")
            except Exception as e:
                logger.error(f"Error in device {self.device_id} listener thread: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(self.interval)
    
    def stop(self):
        """Stop the listener thread"""
        self.running = False

# Create a global instance of ExternalDeviceInterface for easy access
external_device_interface = ExternalDeviceInterface()