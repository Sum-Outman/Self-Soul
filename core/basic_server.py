import http.server
import socketserver
import json
from datetime import datetime, timedelta
import sys
import os

# 设置服务器端口
PORT = 8000

# 模拟数据
mock_models = [
    {"id": "manager", "name": "Manager Model", "status": "active", "type": "manager", "port": 8001},
    {"id": "language", "name": "Language Model", "status": "active", "type": "language", "port": 8002},
    {"id": "knowledge", "name": "Knowledge Model", "status": "active", "type": "knowledge", "port": 8003},
    {"id": "vision", "name": "Vision Model", "status": "active", "type": "vision", "port": 8004},
    {"id": "audio", "name": "Audio Model", "status": "active", "type": "audio", "port": 8005}
]

# 自定义HTTP处理器
class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    # 设置CORS头
    def set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
    
    # 处理OPTIONS请求
    def do_OPTIONS(self):
        self.send_response(200)
        self.set_cors_headers()
        self.end_headers()
    
    # 处理GET请求
    def do_GET(self):
        # 健康检查端点 - 处理各种形式的健康检查请求
        if self.path == '/health' or self.path == '/api/health':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "healthy",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": "AGI Brain System is running",
                "version": "1.0.0"
            }).encode()
            self.wfile.write(response)
        # API根端点
        elif self.path == '/api':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "ok",
                "version": "1.0.0",
                "message": "API server is running"
            }).encode()
            self.wfile.write(response)
        # 模型列表端点 - 同时支持/api/models和/models
        elif self.path == '/api/models' or self.path == '/models':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "models": mock_models,
                "total": len(mock_models)
            }).encode()
            self.wfile.write(response)
        # 知识库文件端点
        elif self.path == '/api/knowledge/files':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "files": [],
                "total": 0,
                "page": 1,
                "page_size": 100
            }).encode()
            self.wfile.write(response)
        # 知识库状态端点
        elif self.path == '/api/knowledge/model_status':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "model": "knowledge",
                "connection_status": "connected",
                "performance": {"response_time": 100, "error_rate": 0}
            }).encode()
            self.wfile.write(response)
        # 学习历史端点 - 确保返回完整的数据结构
        elif self.path == '/api/knowledge/learning_history' or self.path == '/api/auto_learning/history':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "history": [
                    {
                        "id": "1",
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "domains": ["general", "language"],
                        "status": "completed",
                        "progress": 100,
                        "row": {}
                    },
                    {
                        "id": "2",
                        "timestamp": (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "domains": ["knowledge"],
                        "status": "completed",
                        "progress": 100,
                        "row": {}
                    }
                ],
                "total": 2,
                "page": 1,
                "page_size": 10,
                "autoLearningHistory": [
                    {
                        "id": "1",
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "domains": ["general", "language"],
                        "status": "completed",
                        "progress": 100,
                        "row": {}
                    },
                    {
                        "id": "2",
                        "timestamp": (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "domains": ["knowledge"],
                        "status": "completed",
                        "progress": 100,
                        "row": {}
                    }
                ]
            }).encode()
            self.wfile.write(response)
        # 训练历史端点 - 支持各种训练历史相关请求
        elif '/training/history' in self.path or self.path == '/training/history':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "history": [],
                "total": 0,
                "page": 1,
                "page_size": 10
            }).encode()
            self.wfile.write(response)
        # 根端点
        elif self.path == '/':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "message": "AGI Brain Basic Server is running",
                "available_endpoints": [
                    "/health", "/api", "/api/models", "/api/knowledge/files", "/api/knowledge/model_status", "/training/history"
                ]
            }).encode()
            self.wfile.write(response)
        # 系统重启端点 - 模拟系统重启
        elif self.path == '/api/system/restart':
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "message": "System restart initiated"
            }).encode()
            self.wfile.write(response)
        # 处理未知端点 - 返回更友好的响应，以便前端可以回退到演示模式
        else:
            self.send_response(200)
            self.set_cors_headers()
            self.end_headers()
            # 对于所有未知端点，返回模拟的成功响应，以便前端可以继续运行
            response = json.dumps({
                "status": "success",
                "message": "Endpoint not fully implemented in basic server",
                "demo_mode": True,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }).encode()
            self.wfile.write(response)

# 启动服务器
if __name__ == "__main__":
    with socketserver.TCPServer(('', PORT), SimpleHTTPRequestHandler) as httpd:
        print(f"Starting AGI Brain Basic Server on port {PORT}")
        print("Available endpoints:")
        print(f"- http://localhost:{PORT}/")
        print(f"- http://localhost:{PORT}/health")
        print(f"- http://localhost:{PORT}/api")
        print(f"- http://localhost:{PORT}/api/models")
        print(f"- http://localhost:{PORT}/api/knowledge/files")
        print(f"- http://localhost:{PORT}/api/knowledge/model_status")
        print("Server running. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")