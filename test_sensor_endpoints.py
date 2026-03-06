import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_sensor_connection(sensor_id="temperature"):
    """Test sensor connection endpoint"""
    print(f"\n=== Testing sensor connection: {sensor_id} ===")
    
    # Test connection
    connect_url = f"{BASE_URL}/api/devices/sensors/{sensor_id}/connect"
    print(f"POST {connect_url}")
    
    try:
        response = requests.post(connect_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('status')}")
            if data.get('data', {}).get('simulated'):
                print("Note: This is a simulated sensor connection (no real hardware required)")
            return True
        else:
            print(f"Connection failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Connection test failed with exception: {e}")
        return False

def test_sensor_data(sensor_id="temperature"):
    """Test sensor data endpoint"""
    print(f"\n=== Testing sensor data: {sensor_id} ===")
    
    # Test getting data
    data_url = f"{BASE_URL}/api/devices/sensors/{sensor_id}/data"
    print(f"GET {data_url}")
    
    try:
        response = requests.get(data_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('status')}")
            sensor_data = data.get('data', {})
            print(f"Sensor data: {json.dumps(sensor_data, indent=2)}")
            return True
        else:
            print(f"Data retrieval failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Data test failed with exception: {e}")
        return False

def test_sensor_disconnection(sensor_id="temperature"):
    """Test sensor disconnection endpoint"""
    print(f"\n=== Testing sensor disconnection: {sensor_id} ===")
    
    # Test disconnection
    disconnect_url = f"{BASE_URL}/api/devices/sensors/{sensor_id}/disconnect"
    print(f"POST {disconnect_url}")
    
    try:
        response = requests.post(disconnect_url, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('status')}")
            return True
        else:
            print(f"Disconnection failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Disconnection test failed with exception: {e}")
        return False

def test_all_sensor_endpoints():
    """Test all sensor endpoints for different sensor types"""
    test_sensors = ["temperature", "humidity", "pressure", "imu", "proximity"]
    
    results = {}
    
    for sensor_id in test_sensors:
        print(f"\n{'='*60}")
        print(f"Testing sensor: {sensor_id}")
        print(f"{'='*60}")
        
        # Test connection
        connect_success = test_sensor_connection(sensor_id)
        
        # Test data (if connected successfully)
        data_success = False
        if connect_success:
            # Small delay to simulate connection time
            time.sleep(0.5)
            data_success = test_sensor_data(sensor_id)
        
        # Test disconnection (always try)
        disconnect_success = test_sensor_disconnection(sensor_id)
        
        results[sensor_id] = {
            "connection": connect_success,
            "data": data_success,
            "disconnection": disconnect_success
        }
        
        # Small delay between sensors
        time.sleep(0.5)
    
    return results

def test_sensor_list_endpoints():
    """Test sensor listing endpoints"""
    print(f"\n{'='*60}")
    print("Testing sensor listing endpoints")
    print(f"{'='*60}")
    
    endpoints = [
        ("GET", "/api/devices/sensors", "List all available sensors"),
        ("GET", "/api/robot/sensors", "List robot sensors"),
        ("GET", "/api/robot/sensors/data", "Get robot sensor data")
    ]
    
    results = {}
    
    for method, endpoint, description in endpoints:
        print(f"\n{description}: {method} {endpoint}")
        url = f"{BASE_URL}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, timeout=10)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success: {data.get('status', 'unknown')}")
                
                # Print sample data if available
                if 'data' in data:
                    data_content = data['data']
                    if isinstance(data_content, list) and len(data_content) > 0:
                        print(f"Found {len(data_content)} items")
                        if len(data_content) <= 3:
                            print(f"Data sample: {json.dumps(data_content[:3], indent=2)}")
                        else:
                            print(f"First 3 items: {json.dumps(data_content[:3], indent=2)}")
                    elif isinstance(data_content, dict):
                        print(f"Data: {json.dumps(data_content, indent=2)}")
                    else:
                        print(f"Data type: {type(data_content)}")
            else:
                print(f"Failed with status {response.status_code}")
                print(f"Response: {response.text[:200]}")
            
            results[endpoint] = response.status_code == 200
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results[endpoint] = False
    
    return results

if __name__ == "__main__":
    print("Sensor Endpoint Test Suite")
    print("=" * 60)
    
    # First, test if server is running
    try:
        health_check = requests.get(f"{BASE_URL}/api/system/status", timeout=5)
        if health_check.status_code != 200:
            print(f"Warning: Server health check failed ({health_check.status_code})")
        else:
            print("Server is running and responding")
    except:
        print("Error: Server may not be running or is not accessible")
        print("Please ensure the server is running on http://localhost:8000")
        exit(1)
    
    # Test sensor listing endpoints
    listing_results = test_sensor_list_endpoints()
    
    # Test individual sensor endpoints
    sensor_results = test_all_sensor_endpoints()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    print("\nSensor Listing Endpoints:")
    for endpoint, success in listing_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {endpoint}")
    
    print("\nIndividual Sensor Endpoints:")
    for sensor_id, results in sensor_results.items():
        connection_status = "✓" if results['connection'] else "✗"
        data_status = "✓" if results['data'] else "✗"
        disconnection_status = "✓" if results['disconnection'] else "✗"
        print(f"  {sensor_id}: Connect={connection_status}, Data={data_status}, Disconnect={disconnection_status}")
    
    # Calculate success rates
    total_tests = len(listing_results) + len(sensor_results) * 3
    passed_tests = sum(listing_results.values()) + sum([
        sum(r.values()) for r in sensor_results.values()
    ])
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("\n✅ Sensor API endpoints are working well!")
    elif success_rate >= 70:
        print("\n⚠️ Sensor API endpoints have some issues")
    else:
        print("\n❌ Sensor API endpoints need attention")