#!/usr/bin/env python
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import random
import time
import logging
import threading
from urllib.parse import urlparse
import base64
import hashlib
import struct
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('device-control-server')

# Active connections dictionary
connected_clients = {}
clients_lock = threading.Lock()

# Mock devices and their states
devices = {
    'camera1': {
        'id': 'camera1',
        'name': 'Left Camera',
        'type': 'camera',
        'status': 'available',
        'active': False,
        'stream_id': None,
        'resolution': '1920x1080',
        'fps': 30,
        'connected_at': None
    },
    'camera2': {
        'id': 'camera2',
        'name': 'Right Camera',
        'type': 'camera',
        'status': 'available',
        'active': False,
        'stream_id': None,
        'resolution': '1920x1080',
        'fps': 30,
        'connected_at': None
    },
    'camera3': {
        'id': 'camera3',
        'name': 'Depth Camera',
        'type': 'camera',
        'status': 'available',
        'active': False,
        'stream_id': None,
        'resolution': '1280x720',
        'fps': 15,
        'connected_at': None
    },
    'sensor1': {
        'id': 'sensor1',
        'name': 'Temperature Sensor',
        'type': 'sensor',
        'status': 'available',
        'connected': False,
        'value': None,
        'unit': '°C',
        'min_value': 0,
        'max_value': 100,
        'connected_at': None
    },
    'sensor2': {
        'id': 'sensor2',
        'name': 'Motion Sensor',
        'type': 'sensor',
        'status': 'available',
        'connected': False,
        'value': False,
        'unit': '',
        'connected_at': None
    },
    'device1': {
        'id': 'device1',
        'name': 'Robotic Arm',
        'type': 'actuator',
        'status': 'available',
        'connected': False,
        'position': [0, 0, 0],
        'power': 0,
        'connected_at': None
    },
    'device2': {
        'id': 'device2',
        'name': 'LED Controller',
        'type': 'actuator',
        'status': 'available',
        'connected': False,
        'brightness': 0,
        'color': '#000000',
        'connected_at': None
    }
}

devices_lock = threading.Lock()

# Generate mock sensor data
def generate_sensor_data(sensor_id):
    with devices_lock:
        device = devices.get(sensor_id)
        if not device or not device['connected']:
            return None
        
        if device['id'] == 'sensor1':  # Temperature
            return round(random.uniform(20.0, 30.0), 1)
        elif device['id'] == 'sensor2':  # Motion
            return random.random() > 0.8
        return None

# Handle device commands
def handle_device_command(command):
    device_id = command.get('device_id')
    action = command.get('action')
    params = command.get('params', {})
    
    with devices_lock:
        if not device_id or device_id not in devices:
            return {
                'success': False,
                'error': f'Device {device_id} not found',
                'device_id': device_id
            }
        
        device = devices[device_id]
        response = {'success': True, 'device_id': device_id}
        
        if action == 'toggle':
            if device['type'] == 'camera':
                device['active'] = not device['active']
                if device['active']:
                    device['status'] = 'active'
                    device['stream_id'] = f'stream_{device_id}_{int(time.time())}'
                    response['message'] = f'{device["name"]} started'
                else:
                    device['status'] = 'available'
                    device['stream_id'] = None
                    response['message'] = f'{device["name"]} stopped'
            else:
                device['connected'] = not device['connected']
                if device['connected']:
                    device['status'] = 'connected'
                    device['connected_at'] = datetime.now().isoformat()
                    response['message'] = f'{device["name"]} connected'
                else:
                    device['status'] = 'available'
                    device['connected_at'] = None
                    response['message'] = f'{device["name"]} disconnected'
        
        elif action == 'configure':
            if 'resolution' in params and device['type'] == 'camera':
                device['resolution'] = params['resolution']
            if 'fps' in params and device['type'] == 'camera':
                device['fps'] = params['fps']
            if 'parameters' in params:
                device['config'] = params['parameters']
            response['message'] = f'{device["name"]} configured'
        
        elif action == 'get_status':
            response['status'] = device.copy()
        
        elif action == 'get_data':
            if device['type'] == 'sensor' and device['connected']:
                data_value = generate_sensor_data(device_id)
                device['value'] = data_value
                response['data'] = {
                    'value': data_value,
                    'unit': device.get('unit', ''),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                response['success'] = False
                response['error'] = 'Device is not a connected sensor'
        
        elif action == 'control' and device['type'] == 'actuator' and device['connected']:
            if device['id'] == 'device1':  # Robotic Arm
                if 'position' in params:
                    device['position'] = params['position']
                if 'power' in params:
                    device['power'] = params['power']
                response['message'] = 'Robotic arm moved'
            elif device['id'] == 'device2':  # LED Controller
                if 'brightness' in params:
                    device['brightness'] = params['brightness']
                if 'color' in params:
                    device['color'] = params['color']
                response['message'] = 'LEDs updated'
        
        else:
            response['success'] = False
            response['error'] = f'Invalid action {action} for device {device_id}'
        
        logger.info(f'Handled command {action} for device {device_id}')
        return response

class WebSocketConnection:
    def __init__(self, rfile, wfile, client_address):
        self.rfile = rfile
        self.wfile = wfile
        self.client_address = client_address
        self.client_id = f'client_{int(time.time())}_{random.randint(1000, 9999)}'
    
    def receive_message(self):
        # Read WebSocket frame header
        header = self.rfile.read(2)
        if len(header) != 2:
            return None
        
        # Parse header
        fin = (header[0] & 0x80) != 0
        opcode = header[0] & 0x0F
        masked = (header[1] & 0x80) != 0
        payload_length = header[1] & 0x7F
        
        # Get payload length
        if payload_length == 126:
            payload_length = struct.unpack('!H', self.rfile.read(2))[0]
        elif payload_length == 127:
            payload_length = struct.unpack('!Q', self.rfile.read(8))[0]
        
        # Read masking key if present
        masking_key = self.rfile.read(4) if masked else None
        
        # Read payload data
        payload = self.rfile.read(payload_length)
        
        # Unmask payload if necessary
        if masked:
            payload = bytearray(payload)
            for i in range(len(payload)):
                payload[i] ^= masking_key[i % 4]
            payload = bytes(payload)
        
        # Handle different opcodes
        if opcode == 0x08:  # Connection close
            return None
        elif opcode == 0x01:  # Text frame
            return payload.decode('utf-8')
        elif opcode == 0x02:  # Binary frame
            return payload
        else:
            # Ignore other opcodes
            return None
    
    def send_message(self, message):
        # Convert message to JSON string
        data = json.dumps(message).encode('utf-8')
        
        # Prepare WebSocket frame
        frame = bytearray()
        
        # Write FIN and opcode
        frame.append(0x81)  # FIN = 1, opcode = 0x01 (text)
        
        # Write payload length
        length = len(data)
        if length <= 125:
            frame.append(length)
        elif length <= 65535:
            frame.append(126)
            frame.extend(struct.pack('!H', length))
        else:
            frame.append(127)
            frame.extend(struct.pack('!Q', length))
        
        # Write payload data
        frame.extend(data)
        
        # Send frame
        try:
            self.wfile.write(frame)
            self.wfile.flush()
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False

class WebSocketHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Upgrade', '').lower() != 'websocket':
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Expected WebSocket upgrade')
            return
        
        # Perform WebSocket handshake
        key = self.headers.get('Sec-WebSocket-Key')
        if not key:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Missing Sec-WebSocket-Key')
            return
        
        # Generate accept key
        magic_string = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        accept = base64.b64encode(hashlib.sha1((key + magic_string).encode()).digest()).decode()
        
        # Send handshake response
        self.send_response(101)
        self.send_header('Upgrade', 'websocket')
        self.send_header('Connection', 'Upgrade')
        self.send_header('Sec-WebSocket-Accept', accept)
        self.end_headers()
        
        # Now handle WebSocket connection
        self.handle_websocket()
    
    def handle_websocket(self):
        # Create a WebSocketConnection object to manage this connection
        ws_connection = WebSocketConnection(self.rfile, self.wfile, self.client_address)
        
        # Add to active connections
        with clients_lock:
            connected_clients[ws_connection.client_id] = ws_connection
        
        logger.info(f'Client connected: {ws_connection.client_id}, total clients: {len(connected_clients)}')
        
        try:
            # Send initial device status to client
            with devices_lock:
                initial_status = {
                    'type': 'initial_status',
                    'devices': devices,
                    'timestamp': datetime.now().isoformat()
                }
            ws_connection.send_message(initial_status)
            
            # Handle incoming messages
            while True:
                try:
                    message = ws_connection.receive_message()
                    if not message:
                        break
                    
                    data = json.loads(message)
                    logger.debug(f'Received message from {ws_connection.client_id}: {data}')
                    
                    if data.get('type') == 'device_command':
                        # Process device command
                        response = handle_device_command(data)
                        response['type'] = 'command_response'
                        ws_connection.send_message(response)
                    
                    elif data.get('type') == 'ping':
                        # Respond to ping to keep connection alive
                        ws_connection.send_message({'type': 'pong'})
                    
                except json.JSONDecodeError:
                    logger.error(f'Invalid JSON message from {ws_connection.client_id}')
                except Exception as e:
                    logger.error(f'Error handling message from {ws_connection.client_id}: {str(e)}')
                    break
                    
        except Exception as e:
            logger.error(f'WebSocket connection error: {e}')
        finally:
            # Unregister client connection
            with clients_lock:
                if ws_connection.client_id in connected_clients:
                    del connected_clients[ws_connection.client_id]
                    logger.info(f'Client disconnected: {ws_connection.client_id}, total clients: {len(connected_clients)}')

    # Disable logging for successful requests
    def log_message(self, format, *args):
        return

# Periodically send sensor data to clients
def send_periodic_sensor_updates():
    while True:
        time.sleep(2)  # Send updates every 2 seconds
        
        # Generate data for all connected sensors
        with devices_lock:
            for device_id, device in devices.items():
                if device['type'] == 'sensor' and device['connected']:
                    data_value = generate_sensor_data(device_id)
                    device['value'] = data_value
                    
                    # Create sensor update message
                    update_message = {
                        'type': 'sensor_update',
                        'device_id': device_id,
                        'data': {
                            'value': data_value,
                            'unit': device.get('unit', ''),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    # Send to all connected clients
                    with clients_lock:
                        for client in list(connected_clients.values()):
                            try:
                                if not client.send_message(update_message):
                                    # Remove clients that couldn't receive messages
                                    del connected_clients[client.client_id]
                            except Exception:
                                pass

# Start the server
def run_server():
    # Start periodic sensor data updates in background thread
    sensor_thread = threading.Thread(target=send_periodic_sensor_updates, daemon=True)
    sensor_thread.start()
    
    # Start HTTP server with WebSocket handler
    port = 8766
    with socketserver.ThreadingTCPServer(('', port), WebSocketHandler) as httpd:
        logger.info(f'Device Control WebSocket Server started on ws://localhost:{port}')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info('Device Control Server stopped by user')
            httpd.server_close()

if __name__ == "__main__":
    run_server()