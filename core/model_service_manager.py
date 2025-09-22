"""
Model Service Manager
Responsible for creating and managing independent FastAPI applications for each model on specified ports
"""
import os
import sys
import asyncio
import uvicorn
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

# 添加根目录到sys.path以便绝对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ports_config import get_model_port, MAIN_API_PORT
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
                redoc_url=None
            )
            
            # 配置CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # 加载模型
            model = self.model_registry.load_model(model_id)
            if not model:
                error_handler.log_error(f"加载模型 {model_id} 失败", "ModelServiceManager")
                raise Exception(f"加载模型 {model_id} 失败")
            
            # 初始化模型
            if hasattr(model, 'initialize'):
                try:
                    result = model.initialize()
                    if not (result and isinstance(result, dict) and result.get("success", True)):
                        error_handler.log_warning(f"模型 {model_id} 初始化可能失败", "ModelServiceManager")
                except Exception as e:
                    error_handler.handle_error(e, "ModelServiceManager", f"模型 {model_id} 初始化失败")
            
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
            
            # 保存服务信息
            service_info = {
                "app": app,
                "port": port,
                "model_id": model_id,
                "thread": None,
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
            
            try:
                # 创建并启动线程运行uvicorn服务器
                def run_server():
                    try:
                        uvicorn.run(
                            service_info["app"],
                            host="0.0.0.0",
                            port=service_info["port"],
                            log_level="info"
                        )
                    except Exception as e:
                        error_handler.handle_error(e, "ModelService", f"模型 {model_id} 服务运行失败")
                        service_info["is_running"] = False
                
                # 启动服务线程
                thread = threading.Thread(target=run_server, daemon=True)
                thread.start()
                
                # 更新服务状态
                service_info["thread"] = thread
                service_info["is_running"] = True
                
                error_handler.log_info(f"成功启动模型 {model_id} 的服务，端口: {service_info['port']}", "ModelServiceManager")
                return True
            except Exception as e:
                error_handler.handle_error(e, "ModelServiceManager", f"启动模型 {model_id} 服务失败")
                return False
    
    def stop_model_service(self, model_id: str) -> bool:
        """停止指定模型的服务"""
        with self.lock:
            if model_id not in self.model_services:
                error_handler.log_warning(f"模型 {model_id} 的服务不存在", "ModelServiceManager")
                return False
            
            service_info = self.model_services[model_id]
            
            if not service_info["is_running"]:
                error_handler.log_info(f"模型 {model_id} 的服务未运行", "ModelServiceManager")
                return True
            
            # 目前uvicorn不提供优雅关闭方式，这里只标记状态
            service_info["is_running"] = False
            error_handler.log_info(f"已停止模型 {model_id} 的服务", "ModelServiceManager")
            
            return True
    
    def start_all_model_services(self) -> Dict[str, bool]:
        """启动所有模型的服务"""
        results = {}
        
        for model_id in self.model_registry.model_types.keys():
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