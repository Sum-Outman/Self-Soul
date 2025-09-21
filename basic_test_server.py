import http.server
import socketserver
import json
from datetime import datetime

# 设置服务器端口
PORT = 8000

class BasicHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    # 重写日志方法，禁止默认日志输出
    def log_message(self, format, *args):
        return
        
    def do_GET(self):
        # 处理根路径请求
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "message": "Basic Test Server is running",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }).encode()
            self.wfile.write(response)
        
        # 处理健康检查端点
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "healthy",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }).encode()
            self.wfile.write(response)
        
        # 处理测试训练端点
        elif self.path == '/api/training/start':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "job_id": "test_job_123",
                "message": "Training started successfully"
            }).encode()
            self.wfile.write(response)
        
        # 处理模型状态端点
        elif self.path == '/api/models/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "success",
                "models": [
                    {"id": "test_model", "status": "active", "training_mode": False}
                ]
            }).encode()
            self.wfile.write(response)
        
        # 处理未找到的路径
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "status": "error",
                "message": "Not found",
                "path": self.path
            }).encode()
            self.wfile.write(response)

# 创建并启动服务器
def run_server():
    try:
        # 使用with语句确保服务器正确关闭
        with socketserver.TCPServer(("127.0.0.1", PORT), BasicHTTPRequestHandler) as httpd:
            print(f"Starting Basic Test Server on http://127.0.0.1:{PORT}")
            print("Available endpoints:")
            print("  /                - Root endpoint")
            print("  /health          - Health check")
            print("  /api/training/start - Start training (simulated)")
            print("  /api/models/status - Model status (simulated)")
            print("Press Ctrl+C to stop the server")
            # 启动服务器
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    run_server()