import os
import sys
import os
from datetime import datetime

# Add the root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any

# Create a simple FastAPI application
app = FastAPI(title="AGI Brain Simple Server", description="A simplified API server for testing")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Root endpoint
@app.get("/")
async def root():
    return {"status": "success", "message": "AGI Brain Simple Server is running"}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

# Knowledge management endpoints
@app.get("/api/knowledge/files")
async def get_knowledge_files():
    return {"status": "success", "files": [], "total": 0, "page": 1, "page_size": 100}

@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    return {"status": "success", "stats": {"total_files": 0, "total_size": 0, "last_updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}}

@app.get("/api/knowledge/connection")
async def get_knowledge_connection():
    return {"status": "connected", "model": "Knowledge Model", "last_heartbeat": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

@app.get("/api/knowledge/models")
async def get_knowledge_models():
    return {"status": "success", "models": [{"id": "knowledge_model", "name": "Knowledge Model", "status": "active", "type": "knowledge"}]}

@app.get("/api/knowledge/model_status")
async def get_knowledge_model_status():
    return {"status": "success", "model": "knowledge", "connection_status": "connected", "performance": {"response_time": 100, "error_rate": 0}}

# Models endpoints
@app.get("/api/models")
async def get_models():
    # 返回模拟的模型数据，以解决前端"Failed to load models"错误
    return {"status": "success", "models": [
        {"id": "manager", "name": "Manager Model", "status": "active", "type": "manager", "port": 8001},
        {"id": "language", "name": "Language Model", "status": "active", "type": "language", "port": 8002},
        {"id": "knowledge", "name": "Knowledge Model", "status": "active", "type": "knowledge", "port": 8003},
        {"id": "vision", "name": "Vision Model", "status": "active", "type": "vision", "port": 8004},
        {"id": "audio", "name": "Audio Model", "status": "active", "type": "audio", "port": 8005}
    ], "total": 5}

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "detail": str(e)}
        )

# Main function with error handling
def run_server():
    try:
        print("Attempting to import uvicorn...")
        import uvicorn
        print("uvicorn imported successfully.")
        
        print(f"Starting AGI Brain Simple Server on http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Attempting to install uvicorn...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'uvicorn', 'fastapi'])
            print("Dependencies installed successfully.")
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=8000)
        except Exception as inner_e:
            print(f"Failed to install dependencies: {inner_e}")
            print("Falling back to basic HTTP server...")
            # 使用Python内置的HTTP服务器作为最后的后备方案
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            import socketserver
            
            class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = json.dumps({"status": "success", "message": "Basic HTTP Server is running"}).encode()
                        self.wfile.write(response)
                    elif self.path == '/api/health':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = json.dumps({"status": "healthy", "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}).encode()
                        self.wfile.write(response)
                    else:
                        self.send_response(404)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = json.dumps({"status": "error", "message": "Not found"}).encode()
                        self.wfile.write(response)
                
                def log_message(self, format, *args):
                    # 禁止默认日志输出
                    return
            
            try:
                httpd = HTTPServer(('127.0.0.1', 8000), SimpleHTTPRequestHandler)
                print("Basic HTTP server started on http://127.0.0.1:8000")
                httpd.serve_forever()
            except Exception as http_e:
                print(f"Failed to start basic HTTP server: {http_e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_server()

# 也直接调用run_server()以确保执行，即使__name__ != "__main__"
run_server()