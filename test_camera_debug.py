import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_camera_stream_start():
    """Test camera stream start endpoint with detailed error reporting"""
    print("Testing camera stream start...")
    
    # First, connect camera
    connect_url = f"{BASE_URL}/api/devices/cameras/0/connect"
    connect_data = {"camera_index": 0}
    
    print(f"Connecting camera: POST {connect_url}")
    connect_response = requests.post(connect_url, json=connect_data, timeout=10)
    print(f"Connect response status: {connect_response.status_code}")
    print(f"Connect response body: {connect_response.text}")
    
    if connect_response.status_code != 200:
        print("Camera connection failed")
        return
    
    # Now try to start stream
    start_url = f"{BASE_URL}/api/cameras/0/stream/start"
    
    print(f"\nStarting camera stream: POST {start_url}")
    start_response = requests.post(start_url, timeout=10)
    print(f"Start response status: {start_response.status_code}")
    print(f"Start response body: {start_response.text}")
    
    if start_response.status_code == 200:
        print("Camera stream started successfully")
    else:
        print("Camera stream start failed")
        try:
            error_data = start_response.json()
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Raw error: {start_response.text}")
    
    # Try to stop stream (even if start failed)
    stop_url = f"{BASE_URL}/api/cameras/0/stream/stop"
    print(f"\nStopping camera stream: POST {stop_url}")
    stop_response = requests.post(stop_url, timeout=10)
    print(f"Stop response status: {stop_response.status_code}")
    print(f"Stop response body: {stop_response.text}")
    
    # Disconnect camera
    disconnect_url = f"{BASE_URL}/api/devices/cameras/0/disconnect"
    print(f"\nDisconnecting camera: POST {disconnect_url}")
    disconnect_response = requests.post(disconnect_url, timeout=10)
    print(f"Disconnect response status: {disconnect_response.status_code}")
    print(f"Disconnect response body: {disconnect_response.text}")

if __name__ == "__main__":
    try:
        test_camera_stream_start()
    except Exception as e:
        print(f"Test failed with exception: {e}")