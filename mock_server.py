# Simple mock server to simulate the backend API
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import traceback

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        # Always return a success response for any GET request
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if self.path == '/health':
                response = {
                    'status': 'ok',
                    'message': 'Mock server running',
                    'timestamp': self.date_time_string()
                }
            elif self.path.startswith('/api/models'):
                response = {
                    'status': 'ok',
                    'data': [],
                    'models': [],
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/training/history':
                response = {
                    'status': 'ok',
                    'history': [],
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/knowledge/stats':
                # Return mock knowledge stats
                response = {
                    'status': 'ok',
                    'stats': {
                        'total_files': 25,
                        'total_size': 12500000,  # 12.5 MB
                        'file_types': {
                            'pdf': 10,
                            'doc': 5,
                            'docx': 3,
                            'txt': 7
                        },
                        'last_updated': self.date_time_string()
                    },
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/knowledge/files':
                # Return mock file list
                files = []
                file_names = ['Sample Document', 'Research Paper', 'Technical Manual', 'User Guide', 'Report']
                file_types = ['pdf', 'doc', 'docx', 'txt']
                
                for i in range(10):
                    file_type = random.choice(file_types)
                    files.append({
                        'id': f'mock_{i}_{int(time.time())}',
                        'name': f'{random.choice(file_names)} {i+1}.{file_type}',
                        'size': random.randint(100000, 5000000),  # 100KB to 5MB
                        'type': file_type,
                        'upload_time': self.date_time_string(),
                        'domain': random.choice(['general', 'technical', 'scientific', 'medical', 'legal'])
                    })
                
                response = {
                    'status': 'ok',
                    'files': files,
                    'total': len(files),
                    'timestamp': self.date_time_string()
                }
            else:
                response = {
                    'status': 'ok',
                    'message': 'Mock server response',
                    'path': self.path,
                    'timestamp': self.date_time_string()
                }
            
            # Safely write response
            try:
                self.wfile.write(json.dumps(response).encode())
                print(f"Handled GET request to {self.path} from {self.client_address[0]}")
            except ConnectionAbortedError:
                print(f"Connection aborted by client while responding to GET {self.path}")
            except Exception as e:
                print(f"Error writing response to GET {self.path}: {e}")
        except Exception as e:
            print(f"Error handling GET {self.path}: {e}")
            traceback.print_exc()
                
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        try:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            print(f"Handled OPTIONS request to {self.path}")
        except Exception as e:
            print(f"Error handling OPTIONS {self.path}: {e}")
                
    def do_POST(self):
        # Always return a success response for any POST request
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Default response
            response = {
                'status': 'ok',
                'message': 'Mock server POST response',
                'received': json.loads(post_data.decode()) if post_data else {},
                'timestamp': self.date_time_string()
            }
            
            # Special handling for auto learning endpoints
            if self.path == '/api/knowledge/auto-learning/start':
                # Generate a unique session ID
                session_id = 'learn_' + str(int(time.time())) + '_' + str(random.randint(1000, 9999))
                response = {
                    'success': True,
                    'session_id': session_id,
                    'message': 'Auto learning started successfully',
                    'status': 'running',
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/knowledge/auto-learning/stop':
                response = {
                    'success': True,
                    'message': 'Auto learning stopped successfully',
                    'status': 'stopped',
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/knowledge/auto-learning/progress':
                # Generate random progress (fallback for polling mode)
                progress = random.randint(0, 100)
                status = 'completed' if progress >= 100 else 'running'
                
                # Generate some sample logs
                logs = []
                log_messages = [
                    'Processing knowledge documents',
                    'Extracting key concepts',
                    'Updating knowledge representation',
                    'Validating learned information'
                ]
                
                if random.random() > 0.7:
                    logs.append(random.choice(log_messages))
                
                response = {
                    'success': True,
                    'progress': progress,
                    'status': status,
                    'logs': logs,
                    'timestamp': self.date_time_string()
                }
            elif self.path.startswith('/api/knowledge/files/') and self.path.endswith('/download'):
                # Handle file download requests
                file_id = self.path.split('/')[4]
                response = {
                    'success': True,
                    'file_id': file_id,
                    'message': 'File download started',
                    'timestamp': self.date_time_string()
                }
            elif self.path == '/api/knowledge/files':
                # Handle file upload
                response = {
                    'success': True,
                    'message': 'File uploaded successfully',
                    'file_id': 'file_' + str(int(time.time())) + '_' + str(random.randint(1000, 9999)),
                    'timestamp': self.date_time_string()
                }
            
            # Safely write response
            try:
                self.wfile.write(json.dumps(response).encode())
                print(f"Handled POST request to {self.path}")
            except ConnectionAbortedError:
                print(f"Connection aborted by client while responding to POST {self.path}")
            except Exception as e:
                print(f"Error writing response to POST {self.path}: {e}")
        except Exception as e:
            print(f"Error handling POST {self.path}: {e}")
                
    def do_PUT(self):
        # Always return a success response for any PUT request
        try:
            content_length = int(self.headers['Content-Length'])
            put_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'ok',
                'message': 'Mock server PUT response',
                'received': json.loads(put_data.decode()) if put_data else {},
                'timestamp': self.date_time_string()
            }
            
            # Safely write response
            try:
                self.wfile.write(json.dumps(response).encode())
                print(f"Handled PUT request to {self.path}")
            except ConnectionAbortedError:
                print(f"Connection aborted by client while responding to PUT {self.path}")
            except Exception as e:
                print(f"Error writing response to PUT {self.path}: {e}")
        except Exception as e:
            print(f"Error handling PUT {self.path}: {e}")
                
    def do_DELETE(self):
        # Always return a success response for any DELETE request
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'ok',
                'message': 'Mock server DELETE response',
                'timestamp': self.date_time_string()
            }
            
            # Safely write response
            try:
                self.wfile.write(json.dumps(response).encode())
                print(f"Handled DELETE request to {self.path}")
            except ConnectionAbortedError:
                print(f"Connection aborted by client while responding to DELETE {self.path}")
            except Exception as e:
                print(f"Error writing response to DELETE {self.path}: {e}")
        except Exception as e:
            print(f"Error handling DELETE {self.path}: {e}")

import random
import time

# Run the server
port = 8000
print(f'Starting mock server on port {port}...')
print('Available API endpoints:')
print('- /health - Health check')
print('- /api/knowledge/auto-learning/start - Start auto learning')
print('- /api/knowledge/auto-learning/stop - Stop auto learning')
print('- /api/knowledge/auto-learning/progress - Get learning progress (fallback for polling)')
print('- /api/knowledge/files - Get file list')
print('- /api/knowledge/stats - Get knowledge base statistics')
print('Note: WebSocket server runs separately on port 8766')
server = HTTPServer(('', port), SimpleHTTPRequestHandler)
server.serve_forever()