# Mock WebSocket server to simulate real-time communication for auto learning
import asyncio
import json
import random
import time
import uuid
from datetime import datetime
import websockets
import threading

# Global variables to track active learning sessions
learning_sessions = {}

# Helper function to generate realistic learning progress
async def generate_learning_progress(session_id, websocket, is_active):
    progress = 0
    log_count = 0
    
    # Simulate different phases of learning
    phases = [
        {"name": "Initializing", "duration": 2, "progress_increment": 5},
        {"name": "Scanning knowledge base", "duration": 4, "progress_increment": 15},
        {"name": "Analyzing content", "duration": 6, "progress_increment": 30},
        {"name": "Extracting patterns", "duration": 5, "progress_increment": 25},
        {"name": "Updating model", "duration": 3, "progress_increment": 15},
        {"name": "Finalizing", "duration": 2, "progress_increment": 10}
    ]
    
    # Pre-defined log messages for each phase
    phase_logs = {
        "Initializing": [
            "Starting auto learning session",
            "Loading required modules",
            "Setting up learning environment"
        ],
        "Scanning knowledge base": [
            "Scanning available documents",
            "Indexing files in the knowledge base",
            "Identifying relevant content"
        ],
        "Analyzing content": [
            "Extracting key concepts",
            "Identifying relationships between topics",
            "Classifying information by domain"
        ],
        "Extracting patterns": [
            "Finding common patterns",
            "Learning from examples",
            "Refining knowledge representation"
        ],
        "Updating model": [
            "Updating knowledge parameters",
            "Optimizing learning weights",
            "Validating new knowledge"
        ],
        "Finalizing": [
            "Saving learned knowledge",
            "Cleaning up temporary data",
            "Preparing for next session"
        ]
    }
    
    for phase in phases:
        if not is_active[0]:
            break
        
        # Send phase start message
        await send_message(websocket, {
            "type": "log",
            "message": f"{phase['name']}..."
        })
        
        # Simulate phase duration
        phase_steps = int(phase['duration'] * 2)  # 500ms steps
        for _ in range(phase_steps):
            if not is_active[0]:
                break
            
            # Increment progress
            progress_increment = phase['progress_increment'] / phase_steps
            progress = min(progress + progress_increment, 100)
            
            # Create progress update
            progress_data = {
                "type": "progress",
                "progress": min(round(progress), 100),
                "status": "running",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Periodically add logs
            log_count += 1
            if log_count % 4 == 0 and phase["name"] in phase_logs and phase_logs[phase["name"]]:
                log_message = random.choice(phase_logs[phase["name"]])
                progress_data["logs"] = [f"{phase['name']}: {log_message}"]
                
            # Send progress update
            await send_message(websocket, progress_data)
            
            # Small delay between updates
            await asyncio.sleep(0.5)
        
        if not is_active[0]:
            break
    
    # If completed naturally
    if is_active[0] and progress >= 100:
        # Send completion message
        await send_message(websocket, {
            "type": "completed",
            "progress": 100,
            "status": "completed",
            "session_id": session_id,
            "message": "Auto learning completed successfully",
            "timestamp": datetime.now().isoformat()
        })
    elif is_active[0]:
        # In case we didn't reach 100% but the loop completed
        await send_message(websocket, {
            "type": "progress",
            "progress": 100,
            "status": "completed",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

# Helper function to send WebSocket messages
async def send_message(websocket, data):
    try:
        await websocket.send(json.dumps(data))
    except websockets.exceptions.ConnectionClosed:
        # Connection already closed, ignore
        pass
    except Exception as e:
        print(f"Error sending message: {e}")

# Handle WebSocket connection for auto learning
async def handle_auto_learning(websocket, path):
    # Extract session ID from path
    session_id = path.split('/')[-1] if path.split('/')[-1] else f"learn_{uuid.uuid4().hex[:8]}"
    
    # Track if this session is active
    is_active = [True]
    learning_sessions[session_id] = is_active
    
    try:
        print(f"Auto learning WebSocket connection established: {session_id}")
        
        # Send welcome message
        await send_message(websocket, {
            "type": "status",
            "message": "Connected to auto learning service",
            "session_id": session_id
        })
        
        # Start generating progress updates in a separate task
        progress_task = asyncio.create_task(generate_learning_progress(session_id, websocket, is_active))
        
        # Listen for messages from client
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received message from client {session_id}: {data}")
                
                # Handle different message types from client
                if data.get('type') == 'stop':
                    is_active[0] = False
                    await send_message(websocket, {
                        "type": "log",
                        "message": "Auto learning stopped by user"
                    })
                elif data.get('type') == 'ping':
                    await send_message(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await send_message(websocket, {
                        "type": "status",
                        "message": "Message received"
                    })
            except json.JSONDecodeError:
                print(f"Received invalid JSON from client {session_id}")
                await send_message(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"Error processing message from client {session_id}: {e}")
                await send_message(websocket, {
                    "type": "error",
                    "message": str(e)
                })
        
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {session_id}, code: {e.code}, reason: {e.reason}")
    except Exception as e:
        print(f"Error in WebSocket handler: {session_id}, {e}")
    finally:
        # Mark session as inactive
        is_active[0] = False
        # Remove from active sessions
        if session_id in learning_sessions:
            del learning_sessions[session_id]
        # Cancel progress task if it's still running
        if 'progress_task' in locals() and not progress_task.done():
            progress_task.cancel()
        print(f"Auto learning WebSocket session terminated: {session_id}")

# Main function to start the WebSocket server
async def main():
    # Create server for auto learning WebSocket endpoint
    async with websockets.serve(handle_auto_learning, "localhost", 8765):
        print("Mock WebSocket server started on ws://localhost:8765")
        print("Available endpoints:")
        print("- ws://localhost:8765/ws/auto-learning/{sessionId}")
        print("Press Ctrl+C to stop the server")
        # Keep server running indefinitely
        await asyncio.Future()

# Start the server
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Mock WebSocket server stopped")