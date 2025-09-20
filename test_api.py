import requests
import sys

try:
    # Test root endpoint
    response_root = requests.get('http://localhost:8000/')
    print('Root endpoint status code:', response_root.status_code)
    print('Root endpoint response:', response_root.json())
    
    # Test health check endpoint
    response_health = requests.get('http://localhost:8000/api/health')
    print('Health endpoint status code:', response_health.status_code)
    print('Health endpoint response:', response_health.json())
    
    # Test knowledge model endpoint
    response_knowledge = requests.get('http://localhost:8000/api/knowledge/stats')
    print('Knowledge stats endpoint status code:', response_knowledge.status_code)
    print('Knowledge stats response:', response_knowledge.json())
    
    # Test knowledge file list endpoint
    response_knowledge_files = requests.get('http://localhost:8000/api/knowledge/files')
    print('Knowledge files endpoint status code:', response_knowledge_files.status_code)
    print('Knowledge files response:', response_knowledge_files.json())
    
    # Test knowledge connection status endpoint
    response_knowledge_connection = requests.get('http://localhost:8000/api/knowledge/connection')
    print('Knowledge connection endpoint status code:', response_knowledge_connection.status_code)
    print('Knowledge connection response:', response_knowledge_connection.json())
    
    # Test knowledge model list endpoint
    response_knowledge_models = requests.get('http://localhost:8000/api/knowledge/models')
    print('Knowledge models endpoint status code:', response_knowledge_models.status_code)
    print('Knowledge models response:', response_knowledge_models.json())
    
    # Test knowledge model status endpoint
    response_knowledge_model_status = requests.get('http://localhost:8000/api/knowledge/model_status')
    print('Knowledge model status endpoint status code:', response_knowledge_model_status.status_code)
    print('Knowledge model status response:', response_knowledge_model_status.json())
    
    print('All API tests completed successfully!')
    sys.exit(0)
except requests.exceptions.ConnectionError as e:
    print(f'Connection error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error: {e}')
    sys.exit(1)