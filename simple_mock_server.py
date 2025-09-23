import socketserver
import http.server
import json
from datetime import datetime
import os
import socketserver

PORT = 8000

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # 开始设置响应头
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        
        # 根据路径返回不同的模拟数据
        if self.path == '/':
            response = {"status": "success", "message": "Mock API Server is running"}
        elif self.path == '/health':
            response = {"status": "healthy", "version": "1.0.0"}
        elif self.path == '/api/health':
            response = {"status": "healthy", "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
        elif self.path == '/api/knowledge/connection':
            response = {"status": "connected", "model": "Knowledge Model", "last_heartbeat": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
        elif self.path == '/api/knowledge/models':
            response = {"status": "success", "models": [{"id": "knowledge_model", "name": "Knowledge Model", "status": "active", "type": "knowledge"}]}
        elif self.path == '/api/knowledge/model_status':
            response = {"status": "success", "model": "knowledge", "connection_status": "connected", "performance": {"response_time": 100, "error_rate": 0}}
        elif self.path == '/api/models':
            # 修复datetime引用问题
            from datetime import datetime
            
            # 返回完整的本地模型列表（端口8001-8019）
            response = {
                "status": "success", 
                "models": [
                    {"id": "manager", "name": "Manager Model", "status": "running", "type": "Manager Model", "port": 8001},
                    {"id": "language", "name": "Language Model", "status": "running", "type": "Language Model", "port": 8002},
                    {"id": "knowledge", "name": "Knowledge Model", "status": "running", "type": "Knowledge Model", "port": 8003},
                    {"id": "vision", "name": "Vision Model", "status": "stopped", "type": "Vision Model", "port": 8004},
                    {"id": "audio", "name": "Audio Model", "status": "stopped", "type": "Audio Model", "port": 8005},
                    {"id": "autonomous", "name": "Autonomous Model", "status": "stopped", "type": "Autonomous Model", "port": 8006},
                    {"id": "programming", "name": "Programming Model", "status": "stopped", "type": "Programming Model", "port": 8007},
                    {"id": "planning", "name": "Planning Model", "status": "stopped", "type": "Planning Model", "port": 8008},
                    {"id": "emotion", "name": "Emotion Model", "status": "stopped", "type": "Emotion Model", "port": 8009},
                    {"id": "spatial", "name": "Spatial Model", "status": "stopped", "type": "Spatial Model", "port": 8010},
                    {"id": "computer_vision", "name": "Computer Vision Model", "status": "stopped", "type": "Computer Vision Model", "port": 8011},
                    {"id": "sensor", "name": "Sensor Model", "status": "stopped", "type": "Sensor Model", "port": 8012},
                    {"id": "motion", "name": "Motion Model", "status": "stopped", "type": "Motion Model", "port": 8013},
                    {"id": "prediction", "name": "Prediction Model", "status": "stopped", "type": "Prediction Model", "port": 8014},
                    {"id": "advanced_reasoning", "name": "Advanced Reasoning Model", "status": "stopped", "type": "Advanced Reasoning Model", "port": 8015},
                    {"id": "data_fusion", "name": "Data Fusion Model", "status": "stopped", "type": "Data Fusion Model", "port": 8016},
                    {"id": "creative_solving", "name": "Creative Problem Solving Model", "status": "stopped", "type": "Creative Problem Solving Model", "port": 8017},
                    {"id": "meta_cognition", "name": "Meta Cognition Model", "status": "stopped", "type": "Meta Cognition Model", "port": 8018},
                    {"id": "value_alignment", "name": "Value Alignment Model", "status": "stopped", "type": "Value Alignment Model", "port": 8019}
                ],
                "total": 19
            }
        elif self.path == '/api/training/history':
            # 返回模拟的训练历史数据
            from datetime import datetime, timedelta
            import random
            
            # 生成模拟训练历史数据
            history = []
            model_combinations = [
                ['B', 'J'],  # 语言模型 + 知识库
                ['D', 'E', 'F'],  # 视觉相关模型
                ['A', 'B', 'C', 'J'],  # 基础交互组合
                ['A', 'G', 'F'],  # 传感器分析
                ['B', 'K']  # 语言 + 编程
            ]
            datasets = ["General Knowledge Base", "Vision Dataset", "Audio Dataset", "Multimodal Corpus", "Sensor Data"]
            strategies = ["Transfer Learning", "Fine-tuning", "Reinforcement Learning", "Supervised Learning", "Unsupervised Learning"]
            
            for i in range(5):
                date = datetime.utcnow() - timedelta(days=i)
                models = model_combinations[i % len(model_combinations)]
                
                history_item = {
                    "id": f"history_{date.strftime('%Y%m%d')}",
                    "timestamp": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "models": models,
                    "dataset_name": datasets[i % len(datasets)],
                    "duration": 3600 + random.randint(0, 10800),  # 1-4 hours
                    "metrics": {
                        "accuracy": 0.75 + random.random() * 0.2,
                        "loss": 0.1 + random.random() * 0.3
                    },
                    "parameters": {
                        "epochs": 10 + random.randint(0, 20),
                        "batch_size": 32 + random.randint(0, 64),
                        "learning_rate": 0.0001 + random.random() * 0.001
                    },
                    "strategy": strategies[i % len(strategies)]
                }
                
                history.append(history_item)
            
            response = {
                "status": "success",
                "history": history,
                "total": len(history)
            }
        # 新增: 获取训练状态
        elif self.path.startswith('/api/models/') and self.path.endswith('/train/status'):
            # 提取模型ID
            model_id = self.path.split('/')[-3]
            # 导入随机库来模拟训练进度
            import random
            
            # 模拟训练状态 - 随机生成进度，让前端看到变化
            progress = random.randint(0, 100)
            status = 'training'
            message = f'Training in progress: {progress}% complete'
            
            if progress == 100:
                status = 'completed'
                message = 'Training completed successfully'
            elif random.random() < 0.05:  # 5%概率模拟错误
                status = 'error'
                message = 'Training encountered an error'
            
            response = {
                "status": status,
                "progress": progress,
                "message": message,
                "model_id": model_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        # 新增: 获取可用数据集
        elif self.path == '/api/datasets':
            datasets = [
                {"id": "general_knowledge", "name": "General Knowledge Base", "size": 1000000, "type": "text"},
                {"id": "vision_dataset", "name": "Vision Dataset", "size": 500000, "type": "image"},
                {"id": "audio_dataset", "name": "Audio Dataset", "size": 200000, "type": "audio"},
                {"id": "multimodal_corpus", "name": "Multimodal Corpus", "size": 300000, "type": "multimodal"},
                {"id": "sensor_data", "name": "Sensor Data", "size": 750000, "type": "sensor"}
            ]
            response = {
                "status": "success",
                "datasets": datasets,
                "total": len(datasets)
            }
        # 新增: 获取所有模型的训练状态
        elif self.path == '/api/models/training/status':
            # 模拟所有模型的训练状态
            import random
            training_statuses = []
            
            # 为所有19个模型生成训练状态
            all_model_ids = ['manager', 'language', 'knowledge', 'vision', 'audio', 'autonomous', 'programming', 'planning', 'emotion', 'spatial', 'computer_vision', 'sensor', 'motion', 'prediction', 'advanced_reasoning', 'data_fusion', 'creative_solving', 'meta_cognition', 'value_alignment']
            
            for model_id in all_model_ids:
                if random.random() < 0.3:  # 30%概率有训练状态
                    status = 'training'
                    progress = random.randint(0, 100)
                    message = f'Training in progress: {progress}% complete'
                    
                    if progress == 100:
                        status = 'completed'
                        message = 'Training completed successfully'
                    elif random.random() < 0.1:
                        status = 'error'
                        message = 'Training encountered an error'
                    
                    training_statuses.append({
                        "model_id": model_id,
                        "status": status,
                        "progress": progress,
                        "message": message
                    })
            
            response = {
                "status": "success",
                "training_statuses": training_statuses,
                "total": len(training_statuses)
            }
        else:
            # 对于其他所有路径，返回成功状态
            response = {"status": "success", "message": "Mock endpoint"}
        
        # 发送JSON响应
        try:
            # 确保响应不为空
            if not response:
                response = {"status": "success", "message": "Empty response"}
                
            # 序列化JSON数据
            json_data = json.dumps(response, ensure_ascii=False, default=str)
            json_bytes = json_data.encode('utf-8')
            
            # 确保数据不为空
            if len(json_bytes) > 0:
                # 发送Content-Length头
                self.send_header('Content-Length', str(len(json_bytes)))
                self.end_headers()
                
                # 分块发送数据，避免大数据问题
                chunk_size = 8192
                for i in range(0, len(json_bytes), chunk_size):
                    self.wfile.write(json_bytes[i:i+chunk_size])
                self.wfile.flush()
                print(f"Successfully sent response of size: {len(json_bytes)} bytes")
            else:
                self.end_headers()
                print("Warning: Empty response data")
        except BrokenPipeError:
            print("Client disconnected before response could be sent")
        except Exception as e:
            print(f"Error sending response: {type(e).__name__}: {str(e)}")
    
    def do_POST(self):
        # 设置响应头
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        
        # 简单读取请求体（如果需要）
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # 解析JSON数据
        try:
            data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            data = {}
        
        # 处理特定的POST端点
        if self.path.startswith('/api/models/') and self.path.endswith('/train'):
            # 提取模型ID
            model_id = self.path.split('/')[-2]
            # 返回训练开始成功的响应
            response = {
                "status": "success",
                "message": f"Training started for model {model_id}",
                "training_id": f"train_{model_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            }
        elif self.path.startswith('/api/models/') and self.path.endswith('/train/stop'):
            # 提取模型ID
            model_id = self.path.split('/')[-3]
            # 返回停止训练成功的响应
            response = {
                "status": "success",
                "message": f"Training stopped for model {model_id}"
            }
        else:
            # 对于其他所有POST请求，返回成功状态
            response = {"status": "success", "message": "Mock POST response"}
        
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_OPTIONS(self):
        # 处理CORS预检请求
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def log_message(self, format, *args):
        # 禁止默认日志输出
        return

def run_server():
    try:
        # 创建服务器
        with socketserver.TCPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler) as httpd:
            print(f"Mock API Server started on http://127.0.0.1:{PORT}")
            print("This server provides mock responses for frontend development.")
            print("Available endpoints:")
            print("- /                 : Server status")
            print("- /health           : Health check")
            print("- /api/health       : API health check")
            print("- /api/models       : Mock models list")
            print("- /api/knowledge/*  : Mock knowledge endpoints")
            print("- All other paths   : Return mock success response")
            
            # 启动服务器
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    run_server()