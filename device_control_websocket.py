import asyncio
import websockets
import json
import random
import time
import threading

# Store active connections
connected_clients = set()
# Mock device states
device_states = {
    'camera1': {'name': 'Left Camera', 'status': 'available', 'active': False},
    'camera2': {'name': 'Right Camera', 'status': 'available', 'active': False},
    'camera3': {'name': 'Depth Camera', 'status': 'available', 'active': False},
    'sensor1': {'name': 'Temperature Sensor', 'status': 'available', 'connected': False, 'value': 25.0},
    'sensor2': {'name': 'Motion Sensor', 'status': 'available', 'connected': False, 'value': False},
    'device1': {'name': 'Robotic Arm', 'status': 'available', 'connected': False},
    'device2': {'name': 'LED Controller', 'status': 'available', 'connected': False}
}
# Sensor data generation parameters
temp_base = 25.0
motion_probability = 0.05

async def handle_client(websocket, path):
    # Register client connection
    client_id = id(websocket)
    print(f"Client connected: {client_id}")
    connected_clients.add(websocket)
    
    try:
        # Send initial status to new client
        initial_status = {
            'type': 'initial_status',
            'devices': device_states
        }
        await websocket.send(json.dumps(initial_status))
        
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received message from client {client_id}: {data}")
                
                # Handle ping messages
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                
                # Handle device status requests
                elif data.get('type') == 'request_status':
                    # Send current device status
                    status_response = {
                        'type': 'initial_status',
                        'devices': device_states
                    }
                    await websocket.send(json.dumps(status_response))
                
                # Handle device control commands
                elif data.get('type') == 'command':
                    device_id = data.get('device_id')
                    command_type = data.get('command')
                    
                    if device_id in device_states:
                        # Process command based on device type
                        if device_id.startswith('camera'):
                            device_states[device_id]['active'] = command_type == 'start'
                        elif device_id.startswith('sensor') or device_id.startswith('device'):
                            device_states[device_id]['connected'] = command_type == 'connect'
                        
                        # Send command response
                        response = {
                            'type': 'command_response',
                            'device_id': device_id,
                            'success': True,
                            'status': device_states[device_id]
                        }
                        await websocket.send(json.dumps(response))
                        
                        # Also broadcast state change to all clients
                        broadcast_message = {
                            'type': 'device_update',
                            'device_id': device_id,
                            'status': device_states[device_id]
                        }
                        await broadcast_message_to_all(json.dumps(broadcast_message))
                    else:
                        # Device not found
                        response = {
                            'type': 'command_response',
                            'device_id': device_id,
                            'success': False,
                            'error': 'Device not found'
                        }
                        await websocket.send(json.dumps(response))
                
                else:
                    print(f"Unknown message type: {data.get('type')}")
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON message from client {client_id}")
            except Exception as e:
                print(f"Error handling message from client {client_id}: {e}")
                
    except websockets.ConnectionClosed:
        print(f"Client disconnected: {client_id}")
    finally:
        # Unregister client connection
        connected_clients.remove(websocket)

async def broadcast_message_to_all(message):
    # Send message to all connected clients
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )

async def generate_sensor_updates():
    # Periodically generate mock sensor data updates
    while True:
        try:
            # Update temperature sensor (random variation around base value)
            if device_states['sensor1']['connected']:
                device_states['sensor1']['value'] = temp_base + random.uniform(-0.5, 0.5)
                temp_update = {
                    'type': 'sensor_update',
                    'device_id': 'sensor1',
                    'data': {
                        'value': device_states['sensor1']['value'],
                        'timestamp': time.strftime('%H:%M:%S')
                    }
                }
                await broadcast_message_to_all(json.dumps(temp_update))
            
            # Update motion sensor (randomly trigger motion)
            if device_states['sensor2']['connected']:
                new_motion = random.random() < motion_probability
                if new_motion != device_states['sensor2']['value']:
                    device_states['sensor2']['value'] = new_motion
                    motion_update = {
                        'type': 'sensor_update',
                        'device_id': 'sensor2',
                        'data': {
                            'value': device_states['sensor2']['value'],
                            'timestamp': time.strftime('%H:%M:%S')
                        }
                    }
                    await broadcast_message_to_all(json.dumps(motion_update))
            
            # Wait for next update cycle
            await asyncio.sleep(2)  # Update every 2 seconds
        except Exception as e:
            print(f"Error generating sensor updates: {e}")
            await asyncio.sleep(1)

async def main():
    # Start WebSocket server
    server = await websockets.serve(
        handle_client,
        "localhost",
        8766  # Same port as used in HomeView.vue
    )
    
    print(f"Device control WebSocket server started on ws://localhost:8766")
    print(f"WebSocket server ready to handle device control connections")
    print(f"Features:")
    print(f"- Supports multiple client connections")
    print(f"- Sends initial device status on connection")
    print(f"- Handles device control commands")
    print(f"- Generates mock sensor data updates")
    print(f"- Supports ping/pong keepalive")
    print(f"- Broadcasts device state changes to all clients")
    
    # Start sensor data generation task
    sensor_task = asyncio.create_task(generate_sensor_updates())
    
    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Device control WebSocket server stopped by keyboard interrupt")
    except Exception as e:
        print(f"Device control WebSocket server error: {e}")