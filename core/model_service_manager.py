"""
Model Service Manager
Responsible for creating and managing independent FastAPI applications for each model on specified ports
"""
import os
import sys
import time
import asyncio
import uvicorn
import threading
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

# 添加根目录到sys.path以便绝对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ports_config import get_model_port, MAIN_API_PORT, MODEL_TYPE_ALIASES
from core.error_handling import error_handler
from core.model_registry import ModelRegistry
from core.system_settings_manager import SystemSettingsManager

# 模型服务管理器类
class ModelServiceManager:
    """模型服务管理器，负责创建和管理每个模型的独立服务"""
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.system_settings_manager = SystemSettingsManager()
        self.model_services: Dict[str, Dict[str, Any]] = {}
        self.main_api_service = None
        self.lock = threading.RLock()
    
    def _check_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        import socket
        try:
            # 尝试绑定到端口，如果成功则端口可用
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.settimeout(1)
            test_socket.bind(("0.0.0.0", port))
            test_socket.close()
            return True
        except socket.error as e:
            error_handler.log_warning(f"端口 {port} 不可用: {str(e)}", "ModelServiceManager")
            return False
        except Exception as e:
            error_handler.log_error(f"检查端口 {port} 时发生错误: {str(e)}", "ModelServiceManager")
            return False
    
    def _graceful_shutdown(self, server, thread, timeout=5):
        """优雅关闭服务器线程"""
        try:
            if server and hasattr(server, 'should_exit'):
                server.should_exit = True
            if server and hasattr(server, 'shutdown'):
                server.shutdown()
            
            # 等待线程结束
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    error_handler.log_warning("服务线程未在超时时间内结束，强制终止", "ModelServiceManager")
        except Exception as e:
            error_handler.log_error(f"优雅关闭失败: {str(e)}", "ModelServiceManager")
    
    def _verify_service_running(self, port: int, timeout: float = 5.0, retries: int = 3) -> bool:
        """验证服务是否在指定端口上运行，支持重试机制"""
        import socket
        import time
        
        for attempt in range(retries):
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(timeout)
                result = test_socket.connect_ex(("127.0.0.1", port))
                test_socket.close()
                
                if result == 0:  # 0表示连接成功
                    error_handler.log_info(f"端口 {port} 上的服务验证成功（第{attempt+1}次尝试）", "ModelServiceManager")
                    return True
                else:
                    if attempt < retries - 1:  # 不是最后一次尝试
                        error_handler.log_info(f"端口 {port} 连接失败（第{attempt+1}次尝试），等待重试...", "ModelServiceManager")
                        time.sleep(1)  # 等待1秒后重试
                    else:
                        error_handler.log_warning(f"端口 {port} 连接失败（{retries}次尝试后）", "ModelServiceManager")
                        return False
            except Exception as e:
                if attempt < retries - 1:
                    error_handler.log_info(f"验证端口 {port} 时出错: {str(e)}，等待重试...", "ModelServiceManager")
                    time.sleep(1)
                else:
                    error_handler.log_warning(f"验证服务端口 {port} 时出错（{retries}次尝试后）: {str(e)}", "ModelServiceManager")
                    return False
        
        return False
    
    def create_model_service(self, model_id: str) -> Dict[str, Any]:
        """为指定模型创建FastAPI服务"""
        with self.lock:
            # 检查模型ID是否有效
            if model_id not in self.model_registry.model_types:
                error_handler.log_error(f"无效的模型ID: {model_id}", "ModelServiceManager")
                raise ValueError(f"无效的模型ID: {model_id}")
            
            # 检查是否已经为该模型创建了服务
            if model_id in self.model_services:
                error_handler.log_info(f"模型 {model_id} 的服务已存在", "ModelServiceManager")
                return self.model_services[model_id]
            
            # 获取模型端口
            port = get_model_port(model_id)
            if not port:
                error_handler.log_error(f"模型 {model_id} 未配置端口", "ModelServiceManager")
                raise ValueError(f"模型 {model_id} 未配置端口")
            
            # 创建FastAPI应用
            app = FastAPI(
                title=f"Self Soul - {model_id} Model",
                description=f"{model_id} 模型的独立服务",
                version="1.0.0",
                docs_url=f"/{model_id}/docs",
                redoc_url=None,
                swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js",
                swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css"
            )
            
            # 配置CORS - 限制为可信来源
            # 从环境变量获取允许的来源，默认为本地开发环境
            cors_origins_str = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
            cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
            
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["Authorization", "Content-Type", "X-API-Key"],
            )
            
            # 添加API密钥验证中间件（可选）
            @app.middleware("http")
            async def authenticate_request(request: Request, call_next):
                # 跳过健康检查端点的认证
                if request.url.path.endswith("/health") or request.url.path.endswith("/docs") or request.url.path.endswith("/openapi.json"):
                    return await call_next(request)
                
                # 从环境变量获取API密钥
                api_key = os.environ.get("MODEL_SERVICE_API_KEY")
                # 如果未设置API密钥，则跳过认证（仅用于开发环境）
                if not api_key:
                    return await call_next(request)
                
                # 检查请求头中的API密钥
                request_api_key = request.headers.get("X-API-Key")
                if request_api_key and request_api_key == api_key:
                    return await call_next(request)
                else:
                    return JSONResponse(
                        status_code=401,
                        content={"status": "error", "message": "无效或缺失API密钥"}
                    )
            
            # 加载模型 - 如果失败则创建无模型服务（符合README中所有端口运行的要求）
            model = None
            model_loaded = False
            try:
                model = self.model_registry.load_model(model_id, timeout=120)
                if model:
                    model_loaded = True
                    error_handler.log_info(f"模型 {model_id} 加载成功", "ModelServiceManager")
                    
                    # 初始化模型
                    if hasattr(model, 'initialize'):
                        try:
                            result = model.initialize()
                            if not (result and isinstance(result, dict) and result.get("success", True)):
                                error_handler.log_warning(f"模型 {model_id} 初始化可能失败", "ModelServiceManager")
                        except Exception as e:
                            error_handler.handle_error(e, "ModelServiceManager", f"模型 {model_id} 初始化失败")
                else:
                    error_handler.log_warning(f"模型 {model_id} 加载失败，将创建无模型服务（端口仍会监听）", "ModelServiceManager")
            except Exception as e:
                error_handler.log_warning(f"加载模型 {model_id} 时发生异常: {e}，将创建无模型服务", "ModelServiceManager")
            
            # 定义模型健康检查端点
            @app.get(f"/{model_id}/health")
            async def model_health_check():
                try:
                    status = self.model_registry.get_model_status(model_id)
                    return {
                        "status": "ok", 
                        "model_id": model_id,
                        "details": status
                    }
                except Exception as e:
                    return {
                        "status": "error", 
                        "model_id": model_id,
                        "error": str(e)
                    }
            
            # 定义模型处理端点
            @app.post(f"/{model_id}/process")
            async def model_process(data: Dict[str, Any]):
                try:
                    # 检查模型是否已加载
                    if not model_loaded:
                        raise HTTPException(status_code=503, detail=f"模型 {model_id} 未加载或不可用（服务已启动但模型未初始化）")
                    
                    # 获取模型
                    model = self.model_registry.get_model(model_id)
                    if not model or not hasattr(model, 'process'):
                        raise HTTPException(status_code=500, detail=f"模型 {model_id} 不可用或不支持处理操作")
                    
                    # 处理数据
                    result = model.process(data)
                    return {
                        "status": "success",
                        "model_id": model_id,
                        "result": result
                    }
                except HTTPException:
                    # 重新抛出HTTP异常
                    raise
                except Exception as e:
                    error_handler.handle_error(e, "ModelService", f"模型 {model_id} 处理请求失败")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # 定义获取模型状态端点
            @app.get(f"/{model_id}/status")
            async def get_model_status():
                try:
                    status = self.model_registry.get_model_status(model_id)
                    return {
                        "status": "success",
                        "model_id": model_id,
                        "data": status
                    }
                except Exception as e:
                    error_handler.handle_error(e, "ModelService", f"获取模型 {model_id} 状态失败")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # 定义模型配置端点
            @app.get(f"/{model_id}/config")
            async def get_model_config():
                try:
                    config = self.system_settings_manager.get_model_config(model_id)
                    return {
                        "status": "success",
                        "model_id": model_id,
                        "config": config
                    }
                except Exception as e:
                    error_handler.handle_error(e, "ModelService", f"获取模型 {model_id} 配置失败")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # 为管理器模型添加WebSocket端点（符合README中端口8001用于WebSocket通信的描述）
            if model_id == "manager":
                @app.websocket("/ws")
                async def websocket_endpoint(websocket: WebSocket):
                    """
                    WebSocket端点用于实时通信
                    符合README中端口8001用于WebSocket通信的描述
                    """
                    await websocket.accept()
                    try:
                        # 发送连接确认
                        await websocket.send_json({
                            "status": "connected",
                            "model_id": model_id,
                            "message": "WebSocket连接已建立，端口8001用于实时通信"
                        })
                        
                        # 保持连接并处理消息
                        while True:
                            try:
                                # 接收消息
                                data = await websocket.receive_json()
                                
                                # 处理不同类型的消息
                                if data.get("type") == "ping":
                                    await websocket.send_json({
                                        "type": "pong",
                                        "timestamp": time.time()
                                    })
                                elif data.get("type") == "echo":
                                    await websocket.send_json({
                                        "type": "echo_response",
                                        "original_message": data.get("message", ""),
                                        "timestamp": time.time()
                                    })
                                else:
                                    # 默认回显
                                    await websocket.send_json({
                                        "type": "message_received",
                                        "data": data,
                                        "timestamp": time.time()
                                    })
                                    
                            except WebSocketDisconnect:
                                error_handler.log_info(f"WebSocket客户端断开连接", "ModelServiceManager")
                                break
                            except Exception as e:
                                error_handler.handle_error(e, "ModelServiceManager", "WebSocket消息处理错误")
                                await websocket.send_json({
                                    "type": "error",
                                    "error": str(e)
                                })
                                break
                    except Exception as e:
                        error_handler.handle_error(e, "ModelServiceManager", "WebSocket连接错误")
                    finally:
                        await websocket.close()
            
            # 保存服务信息
            service_info = {
                "app": app,
                "port": port,
                "model_id": model_id,
                "thread": None,
                "server": None,  # 存储uvicorn.Server实例以便优雅关闭
                "is_running": False
            }
            
            self.model_services[model_id] = service_info
            error_handler.log_info(f"成功创建模型 {model_id} 的独立服务，端口: {port}", "ModelServiceManager")
            
            return service_info
    
    def start_model_service(self, model_id: str) -> bool:
        """启动指定模型的服务"""
        with self.lock:
            # 确保服务已创建
            if model_id not in self.model_services:
                try:
                    self.create_model_service(model_id)
                except Exception as e:
                    error_handler.handle_error(e, "ModelServiceManager", f"创建模型 {model_id} 服务失败")
                    return False
            
            service_info = self.model_services[model_id]
            
            # 检查服务是否已经在运行
            if service_info["is_running"]:
                error_handler.log_info(f"模型 {model_id} 的服务已经在运行", "ModelServiceManager")
                return True
            
            # 检查端口可用性
            port = service_info["port"]
            if not self._check_port_available(port):
                error_handler.log_error(f"端口 {port} 已被占用，无法启动模型 {model_id} 的服务", "ModelServiceManager")
                return False
            
            try:
                # 创建并启动线程运行uvicorn服务器
                def run_server():
                    error_handler.log_info(f"开始在后台线程中启动模型 {model_id} 的服务，端口: {service_info['port']}", "ModelService")
                    try:
                        # 为新线程创建并设置事件循环，解决"no current event loop in thread"错误
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # 导入uvicorn配置以确保所有依赖都已加载
                        import uvicorn.config
                        error_handler.log_info(f"Uvicorn配置已加载，准备启动服务器", "ModelService")
                        
                        # 创建uvicorn配置对象
                        config = uvicorn.Config(
                            app=service_info["app"],
                            host="0.0.0.0",
                            port=service_info["port"],
                            log_level="info",
                            use_colors=False,
                            loop="auto",
                            reload=False,
                            workers=1,
                            access_log=True,
                            timeout_keep_alive=5
                        )
                        
                        # 创建服务器实例并运行
                        server = uvicorn.Server(config)
                        # 存储server实例以便优雅关闭
                        service_info["server"] = server
                        error_handler.log_info(f"模型 {model_id} 的服务配置完成，开始运行服务器...", "ModelService")
                        server.run()
                        error_handler.log_info(f"模型 {model_id} 的服务已正常停止", "ModelService")
                    except Exception as e:
                        error_handler.log_error(f"模型 {model_id} 服务运行失败: {str(e)}", "ModelService")
                        import traceback
                        error_handler.log_error(f"完整错误堆栈: {traceback.format_exc()}", "ModelService")
                        service_info["is_running"] = False
                        service_info["server"] = None
                    finally:
                        # 确保server引用被清除
                        service_info["server"] = None
                
                # 启动服务线程
                thread = threading.Thread(target=run_server, daemon=True)
                thread.start()
                error_handler.log_info(f"已启动模型 {model_id} 的服务线程，等待服务器初始化...", "ModelServiceManager")
                
                # 等待足够时间，确保服务器有机会启动（uvicorn启动可能需要几秒钟）
                time.sleep(5)
                
                # 检查线程是否仍然在运行
                if not thread.is_alive():
                    error_handler.log_error(f"模型 {model_id} 的服务线程已退出", "ModelServiceManager")
                    service_info["is_running"] = False
                    service_info["server"] = None
                    return False
                
                # 尝试连接服务器以验证是否真正启动
                if not self._verify_service_running(port):
                    error_handler.log_error(f"模型 {model_id} 的服务在端口 {port} 上未响应", "ModelServiceManager")
                    # 尝试优雅关闭
                    self._graceful_shutdown(service_info.get("server"), thread)
                    service_info["is_running"] = False
                    service_info["server"] = None
                    return False
                
                # 更新服务状态
                service_info["thread"] = thread
                service_info["is_running"] = True
                
                error_handler.log_info(f"成功启动模型 {model_id} 的服务，端口: {service_info['port']}", "ModelServiceManager")
                return True
            except Exception as e:
                error_handler.log_error(f"启动模型 {model_id} 服务失败: {str(e)}", "ModelServiceManager")
                import traceback
                error_handler.log_error(f"完整错误堆栈: {traceback.format_exc()}", "ModelServiceManager")
                # 清理状态
                if "server" in service_info:
                    service_info["server"] = None
                service_info["is_running"] = False
                return False
    
    def stop_model_service(self, model_id: str) -> bool:
        """停止指定模型的服务（优雅关闭）"""
        with self.lock:
            if model_id not in self.model_services:
                error_handler.log_warning(f"模型 {model_id} 的服务不存在", "ModelServiceManager")
                return False
            
            service_info = self.model_services[model_id]
            
            if not service_info["is_running"]:
                error_handler.log_info(f"模型 {model_id} 的服务未运行", "ModelServiceManager")
                return True
            
            # 获取服务器实例和线程
            server = service_info.get("server")
            thread = service_info.get("thread")
            
            # 执行优雅关闭
            self._graceful_shutdown(server, thread)
            
            # 更新服务状态
            service_info["is_running"] = False
            service_info["server"] = None
            service_info["thread"] = None
            
            error_handler.log_info(f"已停止模型 {model_id} 的服务", "ModelServiceManager")
            return True
    
    def start_all_model_services(self) -> Dict[str, bool]:
        """启动所有模型的服务，跳过别名模型"""
        results = {}
        
        for model_id in self.model_registry.model_types.keys():
            # 检查是否为别名模型
            if model_id in MODEL_TYPE_ALIASES:
                error_handler.log_info(f"跳过别名模型 {model_id} 的服务启动（映射到 {MODEL_TYPE_ALIASES[model_id]}）", "ModelServiceManager")
                results[model_id] = True  # 标记为已处理，别名模型使用主模型的服务
                continue
                
            results[model_id] = self.start_model_service(model_id)
        
        return results
    
    def get_service_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取指定模型服务的状态"""
        with self.lock:
            if model_id not in self.model_services:
                return None
            
            service_info = self.model_services[model_id]
            return {
                "model_id": model_id,
                "port": service_info["port"],
                "is_running": service_info["is_running"]
            }
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型服务的状态"""
        results = {}
        
        for model_id in self.model_services.keys():
            status = self.get_service_status(model_id)
            if status:
                results[model_id] = status
        
        return results

# 创建全局模型服务管理器实例
model_service_manager = ModelServiceManager()
