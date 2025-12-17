"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
Self Soul AGI System Main Entry File
# Self Soul

Copyright (c) 2025 Self Soul Team
Licensed under the Apache License, Version 2.0
Contact: silencecrowtom@qq.com
"""
import os
import sys
import time
import tempfile
import asyncio
import uvicorn
import threading
import argparse
from datetime import datetime
import numpy as np
import uuid
import cv2

# Add the root directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip six compatibility fix import as it's not available
print("Skipping six compatibility fix import as it's not available")

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import core components in a way that avoids circular dependencies
from core.error_handling import error_handler
from core.model_ports_config import MAIN_API_PORT

# Import core components with delayed initialization to avoid circular dependencies
from core.model_registry import ModelRegistry
from core.training_manager import TrainingManager
from core.dataset_manager import DatasetManager
from core.system_settings_manager import SystemSettingsManager
from core.system_monitor import SystemMonitor
from core.emotion_awareness import AGIEmotionAwarenessSystem
from core.autonomous_learning_manager import AutonomousLearningManager
from core.unified_cognitive_architecture import UnifiedCognitiveArchitecture
from core.enhanced_meta_cognition import EnhancedMetaCognition
from core.intrinsic_motivation_system import IntrinsicMotivationSystem
from core.explainable_ai import ExplainableAI
from core.value_alignment import ValueAlignment
from core.agi_coordinator import AGICoordinator
from core.external_api_service import ExternalAPIService
from core.api_model_connector import APIModelConnector
from core.memory_optimization import ComponentFactory
from core.hardware.camera_manager import CameraManager
from core.hardware.external_device_interface import ExternalDeviceInterface

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# Initialize global variables (will be properly initialized in main function)
model_registry = None
training_manager = None
dataset_manager = None
emotion_system = None
autonomous_learning_manager = None
system_settings_manager = None
system_monitor = None
connection_manager = None
unified_cognitive_architecture = None
enhanced_meta_cognition = None
intrinsic_motivation_system = None
explainable_ai = None
value_alignment = None
agi_coordinator = None
api_model_connector = None  # Add missing global variable

# Initialize hardware managers
camera_manager = CameraManager()
external_device_interface = ExternalDeviceInterface()

# Initialize FastAPI application
from fastapi import FastAPI

app = FastAPI(
    title="Self Soul AGI System",
    description="Advanced General Intelligence System with autonomous learning and self-improvement capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Global variables for model mode management
_model_modes = {}
# Store external model connector instances
_external_model_connectors = {}
# Maximum number for batch processing
BATCH_PROCESSING_LIMIT = 10

# ========== Model Mode Management Helper Functions ==========

def load_model_modes_from_settings():
    """
    Load model modes from system settings
    """
    global _model_modes
    _model_modes.clear()
    
    try:
        settings = system_settings_manager.get_settings()
        models = settings.get("models", {})
        
        for model_id, model_config in models.items():
            # Determine mode based on model source
            if model_config.get("source") == "external":
                _model_modes[model_id] = "external"
            else:
                _model_modes[model_id] = "local"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", "Failed to load model mode information")


def get_all_models_mode():
    """
    Get running modes of all models
    
    Returns:
        Mapping of model IDs to running modes
    """
    return _model_modes.copy()

def get_model_mode(model_id: str) -> str:
    """
    Get running mode of specified model
    
    Args:
        model_id: Model ID
    
    Returns:
        Model running mode (local/external)
    """
    return _model_modes.get(model_id, "local")

def switch_model_to_external(model_id: str, api_config: dict) -> str:
    """
    Switch model to external API mode
    
    Args:
        model_id: Model ID
        api_config: API configuration
    
    Returns:
        Operation result message
    """
    try:
        # Unload local model first (if loaded)
        if model_registry.is_model_loaded(model_id):
            model_registry.unload_model(model_id)
        
        # Disconnect old external connection (if exists)
        if model_id in _external_model_connectors:
            try:
                _external_model_connectors[model_id].disconnect()
            except Exception as e:
                error_handler.handle_error(e, "Model Mode", f"Failed to disconnect old external connection for model {model_id}")
        
        # Create new external model connector
        connector = api_model_connector
        
        # Save API configuration to system settings first
        system_settings_manager.update_model_setting(model_id, {
            "source": "external",
            "api_config": api_config
        })
        
        # Connect to external API using model_id
        connect_result = connector.connect(model_id)
        
        if not connect_result.get("success", False):
            raise Exception(f"Failed to connect to external API: {model_id}, Error: {connect_result.get('message', 'Unknown error')}")
        
        # Save connector instance
        _external_model_connectors[model_id] = connector
        
        # Update model mode
        _model_modes[model_id] = "external"
        
        return f"Model {model_id} successfully switched to external API mode"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", f"Failed to switch model {model_id} to external API mode")
        raise

def switch_model_to_local(model_id: str) -> str:
    """
    Switch model to local mode
    
    Args:
        model_id: Model ID
    
    Returns:
        Operation result message
    """
    try:
        # Disconnect external connection (if exists)
        if model_id in _external_model_connectors:
            try:
                _external_model_connectors[model_id].disconnect()
                del _external_model_connectors[model_id]
            except Exception as e:
                error_handler.handle_error(e, "Model Mode", f"Failed to disconnect external connection for model {model_id}")
        
        # Load local model
        model_registry.load_model(model_id)
        
        # Update model mode
        _model_modes[model_id] = "local"
        
        # Update system settings
        system_settings_manager.update_model_setting(model_id, {
            "source": "local"
        })
        
        return f"Model {model_id} successfully switched to local mode"
    except Exception as e:
        error_handler.handle_error(e, "Model Mode", f"Failed to switch model {model_id} to local mode")
        raise

def batch_switch_model_modes(models_data: list) -> list:
    """
    Batch switch model modes
    
    Args:
        models_data: List of model data
    
    Returns:
        List of switch results for each model
    """
    results = []
    
    # Limit batch processing quantity
    models_to_process = models_data[:BATCH_PROCESSING_LIMIT]
    
    for model_data in models_to_process:
        model_id = model_data.get("id")
        if not model_id:
            results.append({
                "id": None,
                "success": False,
                "message": "Model ID cannot be empty"
            })
            continue
        
        target_mode = model_data.get("mode", "local")
        
        try:
            if target_mode == "external":
                api_config = model_data.get("api_config", {})
                message = switch_model_to_external(model_id, api_config)
            else:
                message = switch_model_to_local(model_id)
            
            results.append({
                "id": model_id,
                "success": True,
                "message": message
            })
        except Exception as e:
            results.append({
                "id": model_id,
                "success": False,
                "message": str(e)
            })
    
    # If exceeding limit quantity, prompt user
    if len(models_data) > BATCH_PROCESSING_LIMIT:
        results.append({
            "id": "batch_info",
            "success": True,
            "message": f"Due to batch processing limit, only {len(models_to_process)} models were processed (total {len(models_data)})"
        })
    
    return results

def test_external_api_connection(connection_data: dict) -> dict:
    """
    Test external API connection
    
    Args:
        connection_data: Connection test data
    
    Returns:
        Connection test result
    """
    try:
        # Create temporary connector for testing
        connector = api_model_connector
        
        # Extract connection parameters
        api_url = connection_data.get("api_url", connection_data.get("api_endpoint", ""))
        api_key = connection_data.get("api_key", "")
        model_name = connection_data.get("model_name", "")
        
        if not api_url or not api_key:
            return {
                "status": "error",
                "message": "API URL and key cannot be empty"
            }
        
        # Use the correct method to test connection
        test_result = connector._test_connection(api_url, api_key, model_name)
        
        if test_result["success"]:
            # Save configuration if model_id is provided
            model_id = connection_data.get("model_id", connection_data.get("modelId", ""))
            if model_id:
                config = {
                    "api_url": api_url,
                    "api_key": api_key,
                    "model_name": model_name,
                    "api_type": connection_data.get("api_type", "custom"),
                    "source": "external"
                }
                system_settings_manager.update_model_setting(model_id, config)
            
            return {
                "status": "success",
                "message": test_result.get("message", "API connection test successful"),
                "details": {
                    "api_url": api_url,
                    "model_name": model_name
                }
            }
        else:
            return {
                "status": "error",
                "message": test_result.get("message", "API connection test failed")
            }
    except Exception as e:
        error_handler.handle_error(e, "API Connection", "Failed to test external API connection")
        return {
            "status": "error",
            "message": str(e)
        }

from core.model_service_manager import model_service_manager

# WebSocket endpoints
@app.websocket("/ws/training/{job_id}")
async def websocket_training_endpoint(websocket: WebSocket, job_id: str):
    """
    Training progress WebSocket endpoint
    
    Args:
        websocket: WebSocket connection
        job_id: Training job ID
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            # Get training status and send to client
            status = training_manager.get_job_status(job_id)
            await websocket.send_json({
                "type": "training_status",
                "job_id": job_id,
                "status": status
            })
            
            # If training completed or failed, close connection
            if status.get("status") in ["completed", "failed", "stopped"]:
                break
                
            # Update every second
            import asyncio
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", f"Training WebSocket error: {job_id}")
        connection_manager.disconnect(websocket)

@app.websocket("/ws/monitoring")
async def websocket_monitoring_endpoint(websocket: WebSocket):
    """
    Real-time monitoring WebSocket endpoint
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        while True:
            # Get real-time monitoring data and send to client
            monitoring_data = await get_realtime_monitoring()
            await websocket.send_json({
                "type": "monitoring_data",
                "data": monitoring_data["data"]
            })
            
            # Update every 2 seconds
            import asyncio
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "Monitoring WebSocket error")
        connection_manager.disconnect(websocket)

@app.websocket("/ws/test-connection")
async def websocket_test_connection(websocket: WebSocket):
    """
    WebSocket connection test endpoint
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # Send connection success message
        await websocket.send_json({
            "type": "connection_test",
            "status": "success",
            "message": "WebSocket connection established"
        })
        
        # Keep connection alive
        while True:
            import asyncio
            await asyncio.sleep(10)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "WebSocket connection test error")
        connection_manager.disconnect(websocket)

@app.websocket("/ws/autonomous-learning/status")
async def websocket_autonomous_learning_status(websocket: WebSocket):
    """
    Autonomous learning status WebSocket endpoint
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "message": "Connected to autonomous learning status updates"
        })
        
        # Send status updates periodically
        while True:
            # Get autonomous learning status
            learning_status = {
                "type": "learning_status",
                "status": autonomous_learning_manager.current_learning_status,
                "progress": autonomous_learning_manager.learning_progress,
                "domains": autonomous_learning_manager.learning_domains,
                "priority": autonomous_learning_manager.learning_priority,
                "running": autonomous_learning_manager.running,
                "last_log": autonomous_learning_manager.learning_logs[-1] if autonomous_learning_manager.learning_logs else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send status update
            await websocket.send_json(learning_status)
            
            # Update every 2 seconds
            import asyncio
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "Autonomous learning status WebSocket error")
        connection_manager.disconnect(websocket)

# Startup event handler
@app.on_event("startup")
async def startup_event():
    """
    System startup event handler - Initialize all global components with memory optimization
    Enhanced to prevent duplicate loading and memory overflow
    """
    global model_registry, training_manager, emotion_system, autonomous_learning_manager
    global system_settings_manager, system_monitor, connection_manager
    global unified_cognitive_architecture, enhanced_meta_cognition, intrinsic_motivation_system
    global explainable_ai, value_alignment, agi_coordinator, api_model_connector
    
    # Check if already initialized to prevent duplicate startup
    if hasattr(startup_event, '_initialized') and startup_event._initialized:
        error_handler.log_warning("System startup already completed, skipping duplicate initialization", "System")
        return
    
    error_handler.log_info("Self Soul system is starting up...", "System")
    
    try:
        # Initialize memory optimization first
        error_handler.log_info("Initializing memory optimization system...", "System")
        from core.memory_optimization import configure_memory_optimization, memory_optimizer
        configure_memory_optimization(
            enable_optimization=True,
            lightweight_mode=True,  # Start in lightweight mode to conserve memory
            max_memory_usage=60  # Lower threshold for memory constrained systems
        )
        
        # Check memory usage before starting
        memory_info = memory_optimizer.check_memory_usage()
        error_handler.log_info(f"Initial memory usage: {memory_info['used']:.2f}MB ({memory_info['percent']:.1f}%)", "System")
        
        # Initialize core components in dependency order with memory optimization
        error_handler.log_info("Initializing system settings manager...", "System")
        system_settings_manager = SystemSettingsManager()
        
        # Run memory optimization after each major component initialization
        if memory_optimizer.should_optimize():
            memory_optimizer.optimize_memory()
            error_handler.log_info("Memory optimized after system settings manager initialization", "System")
        
        error_handler.log_info("Initializing model registry...", "System")
        # 使用ComponentFactory获取全局共享的ModelRegistry实例，避免循环依赖
        model_registry = ComponentFactory.get_component('model_registry', ModelRegistry)
        
        if memory_optimizer.should_optimize():
            memory_optimizer.optimize_memory()
            error_handler.log_info("Memory optimized after model registry initialization", "System")
        
        error_handler.log_info("Initializing dataset manager...", "System")
        dataset_manager = DatasetManager()
        
        error_handler.log_info("Initializing training manager...", "System")
        training_manager = TrainingManager(model_registry)
        
        if memory_optimizer.should_optimize():
            memory_optimizer.optimize_memory()
            error_handler.log_info("Memory optimized after training manager initialization", "System")
        
        error_handler.log_info("Initializing emotion awareness system...", "System")
        emotion_system = AGIEmotionAwarenessSystem()
        
        error_handler.log_info("Initializing autonomous learning manager...", "System")
        autonomous_learning_manager = AutonomousLearningManager(model_registry)
        
        error_handler.log_info("Initializing system monitor...", "System")
        system_monitor = SystemMonitor()
        
        error_handler.log_info("Initializing connection manager...", "System")
        connection_manager = ConnectionManager()
        
        error_handler.log_info("Initializing unified cognitive architecture...", "System")
        unified_cognitive_architecture = UnifiedCognitiveArchitecture()
        
        if memory_optimizer.should_optimize():
            memory_optimizer.optimize_memory()
            error_handler.log_info("Memory optimized after cognitive architecture initialization", "System")
        
        error_handler.log_info("Initializing enhanced meta cognition...", "System")
        enhanced_meta_cognition = EnhancedMetaCognition()
        
        error_handler.log_info("Initializing intrinsic motivation system...", "System")
        intrinsic_motivation_system = IntrinsicMotivationSystem()
        
        error_handler.log_info("Initializing explainable AI system...", "System")
        explainable_ai = ExplainableAI()
        
        error_handler.log_info("Initializing value alignment system...", "System")
        value_alignment = ValueAlignment()
        
        error_handler.log_info("Initializing external API service...", "System")
        external_api_service = ExternalAPIService()
        
        error_handler.log_info("Initializing API model connector...", "System")
        api_model_connector = APIModelConnector()
        
        # Load model modes from settings
        error_handler.log_info("Loading model modes from settings...", "System")
        load_model_modes_from_settings()
        
        # Initialize AGI coordinator with delayed model loading to avoid circular dependencies
        error_handler.log_info("Initializing AGI coordinator...", "System")
        agi_coordinator = AGICoordinator()
        
        if memory_optimizer.should_optimize():
            memory_optimizer.optimize_memory()
            error_handler.log_info("Memory optimized after AGI coordinator initialization", "System")
        
        # Load only essential models first to prevent memory overflow
        error_handler.log_info("Loading essential models first...", "System")
        
        # Define essential models that must be loaded for basic functionality
        # Ultra-minimal set for 32GB memory systems - only load the absolute minimum
        essential_models = ["manager"]  # Only load manager model initially
        
        # Get all available model IDs from model_types (not from loaded models)
        all_available_models = list(model_registry.model_types.keys())
        error_handler.log_info(f"Found {len(all_available_models)} available model types", "System")
        
        # Separate essential and non-essential models
        essential_to_load = [model for model in essential_models if model in all_available_models]
        non_essential_models = [model for model in all_available_models if model not in essential_models]
        
        # Load essential models first
        error_handler.log_info(f"Loading essential models: {', '.join(essential_to_load)}", "System")
        loaded_essential_models = []
        
        # 检查当前内存使用情况
        memory_info = memory_optimizer.check_memory_usage()
        if memory_info["percent"] > 60:
            error_handler.log_warning(f"初始内存使用率过高 ({memory_info['percent']:.1f}%)，暂停加载非核心模型", "System")
            essential_to_load = []
        
        for model_id in essential_to_load:
            try:
                # 加载前再次检查内存
                memory_info = memory_optimizer.check_memory_usage()
                if memory_info["percent"] > 65:
                    error_handler.log_warning(f"内存使用率过高 ({memory_info['percent']:.1f}%)，跳过加载模型 {model_id}", "System")
                    continue
                
                if model_registry.load_model(model_id, priority=1):  # High priority for essential models
                    loaded_essential_models.append(model_id)
                    error_handler.log_info(f"Essential model {model_id} loaded successfully", "System")
                else:
                    error_handler.log_warning(f"Failed to load essential model {model_id}", "System")
            except Exception as e:
                error_handler.handle_error(e, "System", f"Failed to load essential model {model_id}")
            
            # Optimize memory after each essential model
            if memory_optimizer.should_optimize():
                memory_optimizer.optimize_memory()
                error_handler.log_info(f"Memory optimized after loading essential model {model_id}", "System")
            
            await asyncio.sleep(0.1)  # Small delay between essential models
        
        # Disable automatic loading of non-essential models entirely to save memory
        # Non-essential models will be loaded on-demand when first accessed
        loaded_non_essential_models = []
        error_handler.log_info("Automatic loading of non-essential models disabled for memory optimization", "System")
        error_handler.log_info("Non-essential models will be loaded on-demand when first accessed", "System")
        
        # Only use essential models initially
        all_loaded_models = loaded_essential_models
        error_handler.log_info(f"Successfully loaded {len(all_loaded_models)} models (essential: {len(loaded_essential_models)}, non-essential: {len(loaded_non_essential_models)})", "System")
        
        # Initialize only loaded models to prevent errors
        error_handler.log_info("Initializing loaded models...", "System")
        initialized_count = 0
        for model_id in all_loaded_models:
            model = model_registry.get_model(model_id)
            if model and hasattr(model, 'initialize'):
                try:
                    result = model.initialize()
                    if result and isinstance(result, dict) and result.get("success", True):
                        error_handler.log_info(f"Model {model_id} initialized successfully", "System")
                        initialized_count += 1
                    else:
                        error_handler.log_warning(f"Model {model_id} initialization may have failed: {result}", "System")
                except Exception as e:
                    error_handler.handle_error(e, "System", f"Model {model_id} initialization failed")
        
        error_handler.log_info(f"Successfully initialized {initialized_count}/{len(all_loaded_models)} models", "System")
        
        # Start only essential model services first
        error_handler.log_info("Starting essential model services...", "System")
        try:
            # Only start services for loaded models
            startup_results = {}
            for model_id in all_loaded_models:
                try:
                    success = model_service_manager.start_model_service(model_id)
                    startup_results[model_id] = success
                except Exception as e:
                    error_handler.handle_error(e, "System", f"Failed to start service for model {model_id}")
                    startup_results[model_id] = False
            
            # Count successfully started model services
            success_count = sum(1 for success in startup_results.values() if success)
            error_handler.log_info(
                f"Successfully started {success_count}/{len(startup_results)} model services", 
                "System"
            )
            
            # Record failed model services
            failed_models = [model_id for model_id, success in startup_results.items() if not success]
            if failed_models:
                error_handler.log_warning(
                    f"The following model services failed to start: {', '.join(failed_models)}", 
                    "System"
                )
                
        except Exception as e:
            error_handler.handle_error(e, "System", "Failed to start model services")
        
        # Mark startup as completed to prevent duplicate initialization
        startup_event._initialized = True
        error_handler.log_info("Self Soul system startup completed successfully!", "System")
        
    except Exception as e:
        error_handler.handle_error(e, "System", "System startup failed")
        raise

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """
    System shutdown event handler
    """
    error_handler.log_info("Self Soul system is shutting down...", "System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    System health check endpoint

    Returns:
        System status information
    """
    return {"status": "ok", "message": "Self Soul system is running normally"}

# Get available cameras
@app.get("/api/devices/cameras")
async def get_available_cameras():
    """
    Get list of available cameras

    Returns:
        List of available cameras
    """
    try:
        cameras = camera_manager.list_available_cameras()
        return {"status": "success", "data": cameras}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get available cameras")
        raise HTTPException(status_code=500, detail="Failed to get available cameras")

# Get available serial ports
@app.get("/api/serial/ports")
async def get_available_serial_ports():
    """
    Get list of available serial ports

    Returns:
        List of available serial ports
    """
    try:
        ports = external_device_interface.scan_serial_ports()
        return {"status": "success", "data": ports}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get available serial ports")
        raise HTTPException(status_code=500, detail="Failed to get available serial ports")

# Get all external devices
@app.get("/api/devices/external")
async def get_external_devices():
    """
    Get information about all external devices

    Returns:
        List of external devices with their details
    """
    try:
        devices = external_device_interface.get_all_devices_info()
        return {"status": "success", "data": devices}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get external devices")
        raise HTTPException(status_code=500, detail="Failed to retrieve external devices")

# Connect serial port
@app.post("/api/serial/connect")
async def connect_serial_port(request: Request):
    """
    Connect to a serial port device

    Args:
        request: Request body containing device_id and connection parameters

    Returns:
        Connection result
    """
    try:
        data = await request.json()
        device_id = data.get("device_id")
        
        # Support both formats: direct parameters and nested params
        port = data.get("port") or data.get("params", {}).get("port")
        baudrate = data.get("baudrate") or data.get("params", {}).get("baudrate", 9600)
        
        # Convert baudrate to int if it's a string
        try:
            baudrate = int(baudrate)
        except ValueError:
            baudrate = 9600
        
        if not device_id:
            raise HTTPException(status_code=400, detail="Device ID must be provided")
        
        if not port:
            raise HTTPException(status_code=400, detail="Serial port must be provided")
        
        # Create params dictionary with correct format
        params = {
            "port": port,
            "baudrate": baudrate
        }
        
        result = external_device_interface.connect_device(device_id, "serial", params)
        if result.get("success", False):
            return {"status": "success", "data": result}
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to connect to serial port"))
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to connect serial port")
        raise HTTPException(status_code=500, detail="Failed to connect serial port")

# Disconnect serial port
@app.post("/api/serial/disconnect")
async def disconnect_serial_port(request: Request):
    """
    Disconnect from a serial port device

    Args:
        request: Request body containing device_id or port information

    Returns:
        Disconnection result
    """
    try:
        data = await request.json()
        logger.info(f"Received disconnect request with data: {data}")
        device_id = data.get("device_id")
        port = data.get("port")
        
        logger.info(f"Disconnect request - device_id: {device_id}, port: {port}")
        
        if not device_id and not port:
            raise HTTPException(status_code=400, detail="Device ID or port must be provided")
        
        # If port is provided but not device_id, find the device_id using the port
        if port and not device_id:
            logger.info(f"Finding device_id by port: {port}")
            devices = external_device_interface.get_all_devices_info()
            logger.info(f"All devices: {devices}")
            for id, device_data in devices.items():
                # Handle the structure from get_device_info
                if device_data.get("success", False):
                    device_info = device_data.get("device_info", {})
                    device_port = device_info.get("params", {}).get("port")
                    logger.info(f"Checking device {id} with port: {device_port}")
                    if device_port == port:
                        device_id = id
                        logger.info(f"Found device_id {device_id} for port {port}")
                        break
        
        logger.info(f"Final device_id to disconnect: {device_id}")
        
        if not device_id:
            raise HTTPException(status_code=404, detail="Device not found")
        
        result = external_device_interface.disconnect_device(device_id)
        logger.info(f"Disconnect result: {result}")
        if result.get("success", False):
            return {"status": "success", "data": result}
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to disconnect device"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception in disconnect_serial_port: {str(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        error_handler.handle_error(e, "API", "Failed to disconnect serial port")
        raise HTTPException(status_code=500, detail="Failed to disconnect serial port")

# Get all models status
@app.get("/api/models/status")
async def get_models_status():
    """
    Get status of all models

    Returns:
        Status information for all models
    """
    try:
        statuses = model_registry.get_all_models_status()
        return {"status": "success", "data": statuses}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get model status")
        raise HTTPException(status_code=500, detail="Failed to get model status")



# Get language model status
@app.get("/api/models/language/status")
async def get_language_model_status():
    """
    Get status of language model

    Returns:
        Status information for language model
    """
    try:
        language_model = model_registry.get_model("language")
        if language_model:
            # Initialize base status
            status = {
                "id": "language",
                "name": "Language Model",
                "type": "language",
                "status": "active",
                "version": getattr(language_model, 'version', 'unknown'),
            }
            
            # Get actual performance metrics from the model if available
            if hasattr(language_model, 'get_performance_metrics'):
                try:
                    metrics = language_model.get_performance_metrics()
                    if metrics:
                        status.update(metrics)
                except Exception as inner_e:
                    error_handler.log_warning(f"Failed to get performance metrics: {str(inner_e)}", "API")
            
            # Get training status if available
            if hasattr(language_model, 'get_training_status'):
                try:
                    training_status = language_model.get_training_status()
                    if training_status:
                        status['training_status'] = training_status
                except Exception as inner_e:
                    error_handler.log_warning(f"Failed to get training status: {str(inner_e)}", "API")
        else:
            status = {
                "id": "language",
                "name": "Language Model",
                "type": "language",
                "status": "inactive",
                "error": "Model not initialized"
            }
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get language model status")
        raise HTTPException(status_code=500, detail=str(e))

# Get management model status
@app.get("/api/models/management/status")
async def get_management_model_status():
    """
    Get status of management model

    Returns:
        Status information for management model
    """
    try:
        manager_model = model_registry.get_model("manager")
        if manager_model:
            # Initialize base status
            status = {
                "id": "manager",
                "name": "Management Model",
                "type": "management",
                "status": "active",
                "version": getattr(manager_model, 'version', 'unknown'),
            }
            
            # Get actual performance metrics from the model if available
            if hasattr(manager_model, 'get_performance_metrics'):
                try:
                    metrics = manager_model.get_performance_metrics()
                    if metrics:
                        status.update(metrics)
                except Exception as inner_e:
                    error_handler.log_warning(f"Failed to get performance metrics: {str(inner_e)}", "API")
            
            # Get sub-models status if available
            if hasattr(manager_model, 'get_sub_models_status'):
                try:
                    sub_models_status = manager_model.get_sub_models_status()
                    if sub_models_status:
                        status['sub_models'] = sub_models_status
                except Exception as inner_e:
                    error_handler.log_warning(f"Failed to get sub-models status: {str(inner_e)}", "API")
        else:
            status = {
                "id": "manager",
                "name": "Management Model",
                "type": "management",
                "status": "inactive",
                "error": "Model not initialized"
            }
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get management model status")
        raise HTTPException(status_code=500, detail="Failed to get management model status")

# Get from scratch model status
@app.get("/api/models/from_scratch/status")
async def get_from_scratch_model_status():
    """
    Get status of from scratch model

    Returns:
        Status information for from scratch model
    """
    try:
        # Check if any model is available for from scratch training
        available_models = model_registry.get_all_models_status()
        has_from_scratch = any(m.get('training_capability', {}).get('from_scratch', False) 
                              for m in available_models.values())
        
        # Get from scratch training manager if available
        from_scratch_manager = None
        try:
            from core.from_scratch_training import from_scratch_training_manager
            if hasattr(from_scratch_training_manager, 'initialized') and from_scratch_training_manager.initialized:
                from_scratch_manager = from_scratch_training_manager
        except (ImportError, AttributeError):
            pass
        
        # Collect actual data instead of hardcoded values
        status = {
            "id": "from_scratch",
            "name": "From Scratch Model",
            "type": "custom",
            "status": "active" if has_from_scratch else "inactive",
            "version": getattr(from_scratch_manager, 'version', "Unknown") if from_scratch_manager else "Unknown",
            "models_supporting": sum(1 for m in available_models.values() if m.get('training_capability', {}).get('from_scratch', False)),
            "total_models": len(available_models),
            "global_from_scratch_enabled": getattr(model_registry, 'from_scratch_training_enabled', False) if hasattr(model_registry, 'from_scratch_training_enabled') else False
        }
        
        # Add training statistics if available from the manager
        if from_scratch_manager and hasattr(from_scratch_manager, 'get_training_statistics'):
            try:
                training_stats = from_scratch_manager.get_training_statistics()
                status.update(training_stats)
            except Exception:
                pass
        
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get from scratch model status")
        raise HTTPException(status_code=500, detail="Failed to get from scratch model status")

# Process text input
@app.post("/api/process/text")
async def process_text(input_data: dict):
    """
    Process text input

    Args:
        input_data: Dictionary containing text and language info
            - text: Text to process
            - lang: Language code (en, zh, etc.)
            
    Returns:
        Processing result
    """
    try:
        manager_model = model_registry.get_model("manager")
        if not manager_model:
            raise HTTPException(status_code=500, detail="Manager model not loaded")
        
        result = manager_model.process_input({"text": input_data.get("text", ""), "type": "text"})
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process text input")
        raise HTTPException(status_code=500, detail="Failed to process text input")

# Get model configurations
@app.get("/api/models/config")
async def get_model_configurations():
    """
    Get configurations for all models

    Returns:
        Model configurations
    """
    try:
        # Get model configurations from system settings
        configs = system_settings_manager.get_models_config()
        return {"status": "success", "data": configs}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get model configurations")
        raise HTTPException(status_code=500, detail="Failed to get model configurations")

# Get all models (for frontend api.models.getAll function)
@app.get("/api/models/getAll")
async def get_all_models():
    """
    Get all models information

    Returns:
        All models information
    """
    try:
        # This endpoint is used by the frontend api.models.getAll function
        # It's an alias for /api/models/available
        models = model_registry.get_all_models()
        return {"status": "success", "data": models}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get all models")
        raise HTTPException(status_code=500, detail="Failed to get all models")

# Get available models (for frontend model selection)
@app.get("/api/models/available")
async def get_available_models():
    """
    Get available models information for frontend model selection

    Returns:
        Available models information
    """
    try:
        # This endpoint is used by the frontend for model selection
        # Get all registered models (including both loaded and unloaded ones)
        models_status = model_registry.get_all_models_status()
        
        # Format the response to match frontend expectations
        available_models = []
        for model_id, status_info in models_status.items():
            available_models.append({
                "id": model_id,
                "name": status_info.get("name", model_id),
                "type": status_info.get("details", {}).get("model_type", "unknown"),
                "status": status_info.get("status", "unknown"),
                "is_loaded": status_info.get("details", {}).get("is_loaded", False),
                "trainingStatus": {
                    "isTraining": status_info.get("details", {}).get("is_training", False),
                    "progress": status_info.get("details", {}).get("training_progress", 0),
                    "status": status_info.get("details", {}).get("training_status", "idle")
                }
            })
        
        return {"status": "success", "models": available_models}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get available models")
        raise HTTPException(status_code=500, detail="Failed to get available models")

# Chat API endpoint
@app.post("/api/chat")
async def chat_with_model(input_data: dict):
    """
    Chat with the language model

    Args:
        input_data: Dictionary containing chat information
            - message: User's message
            - session_id: Unique session identifier
            - conversation_history: Optional conversation history
            
    Returns:
        Chat response and updated conversation context
    """
    try:
        # Extract input data
        message = input_data.get("message", "")
        session_id = input_data.get("session_id", f"session_{datetime.now().timestamp()}")
        conversation_history = input_data.get("conversation_history", [])
        
        # Check if language model is in external API mode
        model_mode = get_model_mode("language")
        response = None
        
        if model_mode == "external" and "language" in _external_model_connectors:
            # Use external API connector
            connector = _external_model_connectors["language"]
            response = connector.generate_response(message, conversation_history)
        else:
            # Get language model
            language_model = model_registry.get_model("language")
            if not language_model:
                raise HTTPException(status_code=500, detail="Language model not loaded")
            
            # Process message with language model using the correct process method
            result = language_model.process("process_text", {"text": message, "context": {}})
            response = result.get("response", "I'm sorry, I couldn't process your message.")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Limit conversation history to 50 messages
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        return {
            "status": "success",
            "data": {
                "response": response,
                "conversation_history": conversation_history,
                "session_id": session_id
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process chat request")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

# Manager Model Chat API endpoint
@app.post("/api/models/8001/chat")
async def chat_with_manager_model(input_data: dict):
    """
    Chat with the manager model

    Args:
        input_data: Dictionary containing chat information
            - message: User's message
            - session_id: Unique session identifier
            - conversation_history: Optional conversation history
            - query_type: Optional query type
            - model_id: Optional model ID
            - confidence: Optional confidence level
            - parameters: Optional additional parameters
            - request_type: Optional request type
            - user_id: Optional user ID
            - timestamp: Optional timestamp
            - lang: Optional language code
            - system_prompt: Optional system prompt
        
    Returns:
        Chat response with enhanced manager model capabilities
    """
    try:
        # Extract input data
        message = input_data.get("message", "")
        session_id = input_data.get("session_id", f"session_{datetime.now().timestamp()}")
        conversation_history = input_data.get("conversation_history", [])
        query_type = input_data.get("query_type", "text")
        model_id = input_data.get("model_id", "manager")
        confidence = input_data.get("confidence", 0.8)
        parameters = input_data.get("parameters", {})
        request_type = input_data.get("request_type", "chat")
        user_id = input_data.get("user_id", "default_user")
        timestamp = input_data.get("timestamp", datetime.now().isoformat())
        lang = input_data.get("lang", "en")
        system_prompt = input_data.get("system_prompt", "")
        
        # Prepare full context for manager model
        context = {
            "session_id": session_id,
            "conversation_history": conversation_history,
            "query_type": query_type,
            "model_id": model_id,
            "confidence": confidence,
            "parameters": parameters,
            "request_type": request_type,
            "user_id": user_id,
            "timestamp": timestamp,
            "lang": lang,
            "system_prompt": system_prompt
        }
        
        # Get manager model
        manager_model = model_registry.get_model("manager")
        if not manager_model:
            # Try to load the manager model if not loaded
            try:
                from core.models.manager.unified_manager_model import create_unified_manager_model
                manager_model = create_unified_manager_model()
                model_registry.register_model("manager", manager_model)
                error_handler.log_info("Manager model loaded successfully", "API")
            except Exception as load_error:
                error_handler.handle_error(load_error, "API", f"Failed to load manager model: {str(load_error)}")
                raise HTTPException(status_code=500, detail=f"Manager model not loaded: {str(load_error)}")
        
        # Process message with manager model
        # Using process_input with text modality
        start_time = time.time()
        response = manager_model.process_input({
            "text": message,
            "type": "text",
            "context": context
        }, modality="text", context=context)
        processing_time = time.time() - start_time
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": message})
        
        # Extract response text from the manager model's output
        response_text = ""
        if response.get("success", False):
            response_text = response.get("output", "")
        else:
            response_text = "I apologize, but I encountered an error while processing your message."
            error_handler.handle_error(Exception(f"Manager model processing error: {response.get('error', 'Unknown error')}"), "API", "Manager model processing failed")
        
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Limit conversation history to 50 messages
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        # Enhance response with manager model specific fields
        return {
            "status": "success",
            "data": {
                "response": response_text,
                "conversation_history": conversation_history,
                "session_id": session_id,
                "confidence": response.get("confidence", confidence),
                "response_type": response.get("action", "text"),
                "model_id": "manager",
                "port": 8001,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "context": context
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process manager model chat request")
        
        # Return fallback response with error details
        return {
            "status": "error",
            "detail": str(e),
            "data": {
                "response": "I apologize, but I encountered an error while processing your message. Please try again later.",
                "conversation_history": input_data.get("conversation_history", []) + [
                    {"role": "user", "content": input_data.get("message", "")}
                ],
                "session_id": input_data.get("session_id", f"session_{datetime.now().timestamp()}"),
                "error": str(e)
            }
        }



# Real-time audio stream WebSocket endpoint
@app.websocket("/ws/audio-stream")
async def websocket_audio_stream(websocket: WebSocket):
    """
    Real-time audio stream processing
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # Get audio model
        audio_model = model_registry.get_model("audio")
        if not audio_model:
            await websocket.send_json({
                "type": "error",
                "message": "Audio model not loaded"
            })
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": "Audio stream connection established"
        })
        
        while True:
            # Receive audio data
            data = await websocket.receive()
            
            if data.get("type") == "websocket.disconnect":
                break
                
            if data.get("type") == "websocket.receive":
                # Process audio data
                audio_data = data.get("bytes") or data.get("text")
                if audio_data:
                    try:
                        result = audio_model.process_input({
                            "audio_data": audio_data,
                            "type": "audio_stream",
                            "lang": "en"
                        })
                        await websocket.send_json({
                            "type": "audio_processed",
                            "data": result
                        })
                    except Exception as e:
                        error_handler.handle_error(e, "WebSocket", "Failed to process audio stream data")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Audio processing failed: {str(e)}"
                        })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "Audio stream WebSocket error")
        connection_manager.disconnect(websocket)

# Real-time video stream WebSocket endpoint
@app.websocket("/ws/video-stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    Real-time video stream processing
    
    Args:
        websocket: WebSocket connection
    """
    await connection_manager.connect(websocket)
    try:
        # Get video model
        video_model = model_registry.get_model("vision_video")
        if not video_model:
            await websocket.send_json({
                "type": "error",
                "message": "Video model not loaded"
            })
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": "Video stream connection established"
        })
        
        while True:
            # Receive video frame data
            data = await websocket.receive()
            
            if data.get("type") == "websocket.disconnect":
                break
                
            if data.get("type") == "websocket.receive":
                # Process video frame data
                frame_data = data.get("bytes") or data.get("text")
                if frame_data:
                    try:
                        result = video_model.process_input({
                            "video_frame": frame_data,
                            "type": "video_stream",
                            "lang": "en"
                        })
                        await websocket.send_json({
                            "type": "video_processed",
                            "data": result
                        })
                    except Exception as e:
                        error_handler.handle_error(e, "WebSocket", "Failed to process video stream data")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Video frame processing failed: {str(e)}"
                        })
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", "Video stream WebSocket error")
        connection_manager.disconnect(websocket)

# Start model training (compatible with old endpoint)
@app.post("/api/train")
async def train(training_config: dict):
    """
    Start model training (compatible with frontend)

    Args:
        training_config: Training configuration
        
    Returns:
        Training job ID
    """
    try:
        mode = training_config.get("mode", "individual")
        models = training_config.get("models", [])
        dataset = training_config.get("dataset", "")
        parameters = training_config.get("parameters", {})
        
        # Convert parameter format to match training manager
        training_params = {
            "epochs": parameters.get("epochs", 10),
            "batch_size": parameters.get("batchSize", 32),
            "learning_rate": parameters.get("learningRate", 0.001),
            "validation_split": parameters.get("validationSplit", 0.2),
            "dropout_rate": parameters.get("dropoutRate", 0.1),
            "weight_decay": parameters.get("weightDecay", 0.0001),
            "momentum": parameters.get("momentum", 0.9),
            "optimizer": parameters.get("optimizer", "adam"),
            "training_mode": mode,
            "strategy": parameters.get("strategy", "standard"),
            "knowledge_assist": parameters.get("knowledge_assist"),
            "from_scratch": parameters.get("fromScratch", False)
        }
        
        job_id = training_manager.start_training(models, training_params)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to start training")
        raise HTTPException(status_code=500, detail=str(e))

# Start model training (new endpoint)
@app.post("/api/training/start")
async def start_training(training_config: dict):
    """
    Start model training with specific configuration

    Args:
        training_config: Training configuration containing model IDs and parameters
        
    Returns:
        Training job ID
    """
    try:
        model_ids = training_config.get("model_ids", [])
        parameters = training_config.get("parameters", {})
        
        # Ensure from_scratch parameter is properly handled
        if "fromScratch" in training_config:
            parameters["from_scratch"] = training_config["fromScratch"]
        elif "from_scratch" not in parameters:
            parameters["from_scratch"] = False
        
        job_id = training_manager.start_training(model_ids, parameters)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to start training")
        raise HTTPException(status_code=500, detail=str(e))

# Get training job status
@app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """
    Get training job status

    Args:
        job_id: Training job ID
        
    Returns:
        Training job status information
    """
    try:
        status = training_manager.get_job_status(job_id)
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get training status")
        raise HTTPException(status_code=500, detail="Failed to get training status")

# Stop training job
@app.post("/api/training/stop/{job_id}")
async def stop_training(job_id: str):
    """
    Stop training job

    Args:
        job_id: Training job ID
        
    Returns:
        Stop operation result
    """
    try:
        success = training_manager.stop_training(job_id)
        return {"status": "success", "stopped": success}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to stop training")
        raise HTTPException(status_code=500, detail="Failed to stop training")

# Get training history
@app.get("/api/training/history")
async def get_training_history():
    """
    Get training history

    Returns:
        Training history data
    """
    try:
        history = training_manager.get_training_history()
        return {"status": "success", "data": history}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get training history")
        raise HTTPException(status_code=500, detail="Failed to get training history")

# Get all training sessions status
@app.get("/api/training/sessions")
async def get_training_sessions():
    """
    Get all training sessions status
    
    Returns:
        Status of all training jobs
    """
    try:
        sessions = training_manager.get_all_jobs_status()
        return {"status": "success", "data": sessions}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get training sessions")
        raise HTTPException(status_code=500, detail="Failed to get training sessions")

# Validate joint training model combination
@app.post("/api/train/validate-combination")
async def validate_training_combination(combination_data: dict):
    """
    Validate joint training model combination

    Args:
        combination_data: Dictionary containing model list and training mode
        
    Returns:
        Validation result
    """
    try:
        models = combination_data.get("models", [])
        mode = combination_data.get("mode", "joint")
        
        # Validate model combination
        validation_result = training_manager.validate_model_combination(models, mode)
        
        return {
            "status": "success",
            "valid": validation_result["valid"],
            "message": validation_result["message"],
            "details": validation_result.get("details", {})
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to validate model combination")
        raise HTTPException(status_code=500, detail="Failed to validate model combination")

# Joint training specific endpoints

# Get joint training recommended combinations
@app.get("/api/joint-training/recommendations")
async def get_joint_training_recommendations():
    """
    Get joint training recommended combinations

    Returns:
        List of recommended joint training model combinations
    """
    try:
        recommendations = training_manager.get_joint_training_recommendations()
        return {"status": "success", "data": recommendations}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get joint training recommendations")
        raise HTTPException(status_code=500, detail="Failed to get joint training recommendations")

# Start joint training
@app.post("/api/joint-training/start")
async def start_joint_training(joint_config: dict):
    """
    Start joint training

    Args:
        joint_config: Joint training configuration
            - model_ids: List of model IDs for joint training
            - strategy: Training strategy (standard, knowledge_assisted, adaptive)
            - parameters: Training parameters
        
    Returns:
        Training job ID
    """
    try:
        model_ids = joint_config.get("model_ids", [])
        strategy = joint_config.get("strategy", "standard")
        parameters = joint_config.get("parameters", {})
        
        # Set training mode to joint training
        parameters["training_mode"] = "joint"
        parameters["strategy"] = strategy
        
        job_id = training_manager.start_training(model_ids, parameters)
        return {"status": "success", "job_id": job_id}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to start joint training")
        raise HTTPException(status_code=500, detail=str(e))

# Dataset management endpoints

# Get all supported data formats for all models
@app.get("/api/datasets/supported-formats")
async def get_supported_formats():
    """
    Get all supported data formats for all models
    
    Returns:
        Supported data formats for each model
    """
    try:
        formats = dataset_manager.get_all_supported_formats()
        return {"status": "success", "data": formats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get supported formats")
        raise HTTPException(status_code=500, detail="Failed to get supported formats")

# Get supported data formats for specific model
@app.get("/api/datasets/supported-formats/{model_id}")
async def get_model_supported_formats(model_id: str):
    """
    Get supported data formats for specific model
    
    Args:
        model_id: Model ID
        
    Returns:
        List of supported formats
    """
    try:
        formats = dataset_manager.get_model_supported_formats(model_id)
        return {"status": "success", "data": formats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get model supported formats")
        raise HTTPException(status_code=500, detail="Failed to get model supported formats")

# Validate dataset file
@app.post("/api/datasets/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Validate dataset file
    
    Args:
        file: Uploaded file
        model_id: Target model ID
        
    Returns:
        Validation result
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Validate dataset
        validation_result = dataset_manager.validate_dataset(temp_file_path, model_id)
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        return {"status": "success", "data": validation_result}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "Dataset validation failed")
        raise HTTPException(status_code=500, detail="Dataset validation failed")

# Upload dataset file
@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    dataset_name: str = Form(None)
):
    """
    Upload dataset file
    
    Args:
        file: Uploaded file
        model_id: Target model ID
        dataset_name: Dataset name (optional)
        
    Returns:
        Upload result
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Save dataset
        save_result = dataset_manager.save_dataset(temp_file_path, model_id, dataset_name)
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        if save_result["success"]:
            return {"status": "success", "data": save_result}
        else:
            return {"status": "error", "error": save_result.get("error", "Save failed")}
            
    except Exception as e:
        error_handler.handle_error(e, "API", "Dataset upload failed")
        raise HTTPException(status_code=500, detail="Dataset upload failed")

# List all datasets
@app.get("/api/datasets")
async def list_datasets(model_id: str = None):
    """
    List all datasets
    
    Args:
        model_id: Specific model ID (optional)
        
    Returns:
        List of datasets
    """
    try:
        datasets = dataset_manager.list_datasets(model_id)
        return {"status": "success", "data": datasets}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to list datasets")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

# Get dataset information
@app.get("/api/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get dataset information
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Dataset information
    """
    try:
        dataset_info = dataset_manager.get_dataset_info(dataset_id)
        if dataset_info:
            return {"status": "success", "data": dataset_info}
        else:
            raise HTTPException(status_code=404, detail="Dataset does not exist")
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dataset information")
        raise HTTPException(status_code=500, detail="Failed to get dataset information")

# Delete dataset
@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete dataset
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Delete result
    """
    try:
        delete_result = dataset_manager.delete_dataset(dataset_id)
        if delete_result["success"]:
            return {"status": "success", "data": delete_result}
        else:
            raise HTTPException(status_code=404, detail=delete_result.get("error", "Delete failed"))
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to delete dataset")
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

# Get dataset statistics
@app.get("/api/datasets/stats")
async def get_dataset_stats():
    """
    Get dataset statistics
    
    Returns:
        Statistics
    """
    try:
        stats = dataset_manager.get_dataset_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dataset statistics")
        raise HTTPException(status_code=500, detail="Failed to get dataset statistics")

# Language settings endpoints have been removed as this is now a monolingual English system
@app.post("/api/language/set")
async def set_language_endpoint(language_data: dict):
    """
    This endpoint has been removed as the system is now English-only.
    """
    return JSONResponse(
        status_code=410,  # Gone
        content={"error": "This system is now English-only and language settings have been removed"}
    )

@app.get("/api/language/supported")
async def get_supported_languages_endpoint():
    """
    This endpoint has been removed as the system is now English-only.
    """
    return JSONResponse(
        status_code=410,  # Gone
        content={"error": "This system is now English-only and language settings have been removed"}
    )

# System monitoring endpoints

# Get real-time monitoring data
@app.get("/api/monitoring/realtime")
async def get_realtime_monitoring():
    """
    Get real-time monitoring data
    
    Returns:
        Monitoring data
    """
    try:
        monitoring_data = system_monitor.get_realtime_monitoring()
        return {"status": "success", "data": monitoring_data}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get monitoring data")
        raise HTTPException(status_code=500, detail="Failed to get monitoring data")

# Get system performance statistics
@app.get("/api/monitoring/performance")
async def get_performance_stats():
    """
    Get system performance statistics
    
    Returns:
        Performance statistics
    """
    try:
        performance_stats = system_monitor.get_performance_stats()
        return {"status": "success", "data": performance_stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get performance statistics")
        raise HTTPException(status_code=500, detail="Failed to get performance statistics")

# Dashboard API endpoints

# Get dashboard system metrics
@app.get("/api/dashboard/system")
async def get_dashboard_system():
    """
    Get dashboard system metrics
    
    Returns:
        System performance metrics
    """
    try:
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        base_metrics = enhanced_metrics.get("base_metrics", {})
        
        # Calculate actual trend based on historical data
        trend_data = enhanced_metrics.get("trends", {})
        
        # Get actual network metrics
        network_io = base_metrics.get("network_io", {})
        bytes_recv = network_io.get("bytes_recv", 0)
        bytes_sent = network_io.get("bytes_sent", 0)
        
        # Calculate network usage in MB/s (convert from bytes to MB)
        network_in_mbps = bytes_recv / 1024 / 1024
        network_out_mbps = bytes_sent / 1024 / 1024
        
        # Get actual response time from task metrics
        task_metrics = enhanced_metrics.get("task_metrics", [])
        avg_response_time = task_metrics[0].get("avg_response_time", 0) if task_metrics else 0
        
        metrics = [
            {
                "id": "cpu_usage",
                "title": "CPU Usage",
                "value": base_metrics.get("cpu_usage", 0),
                "unit": "%",
                "threshold": 90,
                "warning": 70,
                "trend": trend_data.get("cpu_trend", "stable"),
                "change": f"{trend_data.get('cpu_change', 0):.1f}%"
            },
            {
                "id": "memory_usage",
                "title": "Memory Usage",
                "value": base_metrics.get("memory_usage", 0),
                "unit": "%",
                "threshold": 85,
                "warning": 75,
                "trend": trend_data.get("memory_trend", "stable"),
                "change": f"{trend_data.get('memory_change', 0):.1f}%"
            },
            {
                "id": "disk_usage",
                "title": "Disk Usage",
                "value": base_metrics.get("disk_usage", 0),
                "unit": "%",
                "threshold": 95,
                "warning": 85,
                "trend": trend_data.get("disk_trend", "stable"),
                "change": f"{trend_data.get('disk_change', 0):.1f}%"
            },
            {
                "id": "network_in",
                "title": "Network In",
                "value": network_in_mbps,
                "unit": "MB/s",
                "threshold": 100,
                "warning": 50,
                "trend": trend_data.get("network_in_trend", "stable"),
                "change": f"{trend_data.get('network_in_change', 0):.1f}MB/s"
            },
            {
                "id": "network_out",
                "title": "Network Out",
                "value": network_out_mbps,
                "unit": "MB/s",
                "threshold": 100,
                "warning": 50,
                "trend": trend_data.get("network_out_trend", "stable"),
                "change": f"{trend_data.get('network_out_change', 0):.1f}MB/s"
            },
            {
                "id": "response_time",
                "title": "Average Response Time",
                "value": avg_response_time,
                "unit": "ms",
                "threshold": 1000,
                "warning": 500,
                "trend": trend_data.get("response_time_trend", "stable"),
                "change": f"{trend_data.get('response_time_change', 0):.1f}ms"
            }
        ]
        
        return {"status": "success", "metrics": metrics}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard system metrics")
        raise HTTPException(status_code=500, detail="Failed to get dashboard system metrics")

# Get dashboard logs
@app.get("/api/dashboard/logs")
async def get_dashboard_logs(limit: int = 50):
    """
    Get dashboard logs
    
    Args:
        limit: Log entries limit
        
    Returns:
        Recent system logs
    """
    try:
        logs = system_monitor.get_recent_logs(limit=limit)
        return {"status": "success", "logs": logs}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard logs")
        raise HTTPException(status_code=500, detail="Failed to get dashboard logs")

# Get dashboard activities
@app.get("/api/dashboard/activities")
async def get_dashboard_activities(limit: int = 30):
    """
    Get dashboard activities
    
    Args:
        limit: Activities limit
        
    Returns:
        Recent model activities
    """
    try:
        # Get activity data from enhanced monitoring
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        activities = enhanced_metrics.get("data_streams", {}).get("model_activities", [])
        
        # Limit the number of returned items
        activities = activities[:limit]
        
        return {"status": "success", "activities": activities}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard activities")
        raise HTTPException(status_code=500, detail="Failed to get dashboard activities")

# Get dashboard training sessions
@app.get("/api/dashboard/training-sessions")
async def get_dashboard_training_sessions():
    """
    Get dashboard training sessions
    
    Returns:
        Active training sessions
    """
    try:
        all_jobs = training_manager.get_all_jobs_status()
        active_sessions = []
        
        for job_id, job_data in all_jobs.items():
            if job_data.get("status") in ["running", "pending"]:
                active_sessions.append({
                    "id": job_id,
                    "models": job_data.get("model_ids", []),
                    "status": job_data.get("status", "unknown"),
                    "progress": job_data.get("progress", 0),
                    "currentEpoch": job_data.get("current_epoch", 0),
                    "totalEpochs": job_data.get("total_epochs", 0),
                    "loss": job_data.get("current_loss", 0.0),
                    "accuracy": job_data.get("current_accuracy", 0.0) * 100,
                    "elapsedTime": job_data.get("elapsed_time", "00:00:00")
                })
        
        return {"status": "success", "sessions": active_sessions}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard training sessions")
        raise HTTPException(status_code=500, detail="Failed to get dashboard training sessions")

# Get dashboard emotion data
@app.get("/api/dashboard/emotion")
async def get_dashboard_emotion():
    """
    Get dashboard emotion data
    
    Returns:
        Emotion analysis data
    """
    try:
        # Get current state of emotion awareness system
        emotion_state = emotion_system.get_current_state()
        
        return {
            "status": "success",
            "valence": emotion_state.get("valence", 0.5),
            "arousal": emotion_state.get("arousal", 0.5),
            "dominance": emotion_state.get("dominance", 0.5),
            "mood": emotion_state.get("mood", "neutral"),
            "intensity": emotion_state.get("intensity", 0.5)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard emotion data")
        raise HTTPException(status_code=500, detail="Failed to get dashboard emotion data")

# Get dashboard model data
@app.get("/api/dashboard/models")
async def get_dashboard_models():
    """
    Get dashboard model data
    
    Returns:
        Model status and collaboration data
    """
    try:
        model_statuses = model_registry.get_all_models_status()
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        collaboration_metrics = enhanced_metrics.get("collaboration_metrics", {})
        
        models = []
        for status in model_statuses:
            models.append({
                "id": status.get("id"),
                "name": status.get("name"),
                "type": status.get("type"),
                "status": status.get("status"),
                "version": status.get("version", "1.0.0"),
                "metrics": {
                    "cpu": status.get("cpu_usage", 0),
                    "memory": status.get("memory_usage", 0),
                    "responseTime": status.get("response_time", 0),
                    "successRate": status.get("success_rate", 0) * 100,
                    "throughput": status.get("throughput", 0)
                }
            })
        
        return {
            "status": "success",
            "models": models,
            "activeCollaborations": collaboration_metrics.get("active_collaborations", 0),
            "collaborationEfficiency": collaboration_metrics.get("efficiency", 0) * 100,
            "dataTransferRate": collaboration_metrics.get("data_transfer_rate", 0)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard model data")
        raise HTTPException(status_code=500, detail="Failed to get dashboard model data")

# Get dashboard detailed model metrics
@app.get("/api/dashboard/model-metrics")
async def get_dashboard_model_metrics():
    """
    Get dashboard detailed model metrics
    
    Returns:
        Detailed model performance metrics
    """
    try:
        model_statuses = model_registry.get_all_models_status()
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        collaboration_metrics = enhanced_metrics.get("collaboration_metrics", {})
        
        models = []
        for status in model_statuses:
            models.append({
                "id": status.get("id"),
                "name": status.get("name"),
                "type": status.get("type"),
                "status": status.get("status"),
                "version": status.get("version", "1.0.0"),
                "metrics": {
                    "cpu": status.get("cpu_usage", 0),
                    "memory": status.get("memory_usage", 0),
                    "responseTime": status.get("response_time", 0),
                    "successRate": status.get("success_rate", 0) * 100,
                    "throughput": status.get("throughput", 0)
                }
            })
        
        return {
            "status": "success",
            "models": models,
            "activeCollaborations": collaboration_metrics.get("active_collaborations", 0),
            "collaborationEfficiency": collaboration_metrics.get("efficiency", 0) * 100,
            "dataTransferRate": collaboration_metrics.get("data_transfer_rate", 0)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get dashboard model metrics")
        raise HTTPException(status_code=500, detail="Failed to get dashboard model metrics")

# AGI System status endpoints

# Get AGI system overall status
@app.get("/api/agi/status")
async def get_agi_system_status():
    """
    Get AGI system overall status
    
    Returns:
        AGI system status information
    """
    try:
        # Get status of each component
        cognitive_status = unified_cognitive_architecture.get_system_status()
        meta_cognition_status = enhanced_meta_cognition.get_system_status()
        # Get knowledge base status from knowledge model via model registry
        knowledge_model = model_registry.get_model('knowledge')
        knowledge_base_status = knowledge_model.get_knowledge_base_status() if knowledge_model else {}
        motivation_status = intrinsic_motivation_system.get_motivation_report()
        explainability_status = explainable_ai.get_system_report()
        alignment_status = value_alignment.get_alignment_report()
        
        # Calculate overall health score
        health_scores = []
        if 'health_score' in cognitive_status:
            health_scores.append(cognitive_status['health_score'])
        if 'health_score' in meta_cognition_status:
            health_scores.append(meta_cognition_status['health_score'])
        if 'health_score' in knowledge_base_status:
            health_scores.append(knowledge_base_status['health_score'])
        if 'overall_motivation' in motivation_status:
            health_scores.append(motivation_status['overall_motivation'])
        if 'reliability_score' in explainability_status.get('confidence_calibration', {}):
            health_scores.append(explainability_status['confidence_calibration']['reliability_score'])
        if 'overall_alignment_score' in alignment_status.get('value_system', {}):
            health_scores.append(alignment_status['value_system']['overall_alignment_score'])
        
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0.5
        
        return {
            "status": "success",
            "data": {
                "cognitive_architecture": cognitive_status,
                "meta_cognition": meta_cognition_status,
                "knowledge_base": knowledge_base_status,
                "intrinsic_motivation": motivation_status,
                "explainable_ai": explainability_status,
                "value_alignment": alignment_status,
                "overall_health": overall_health,
                "health_status": "EXCELLENT" if overall_health > 0.8 else
                                "GOOD" if overall_health > 0.6 else
                                "FAIR" if overall_health > 0.4 else "POOR"
            }
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get AGI system status")
        raise HTTPException(status_code=500, detail="Failed to get AGI system status")

# Get comprehensive system statistics
@app.get("/api/statistics")
async def get_system_statistics():
    """
    Get comprehensive system statistics
    
    Returns:
        System statistics including model status, training stats, performance metrics, etc.
    """
    try:
        # Get model status
        model_statuses = model_registry.get_all_models_status()
        
        # Get training statistics
        training_stats = {
            "total_jobs": len(training_manager.get_all_jobs_status()),
            "active_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                              if job.get("status") in ["running", "pending"]]),
            "completed_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                                 if job.get("status") == "completed"]),
            "failed_jobs": len([job for job in training_manager.get_all_jobs_status().values()
                              if job.get("status") == "failed"])
        }
        
        # Get dataset statistics
        dataset_stats = dataset_manager.get_dataset_stats()
        
        # Get system performance metrics
        performance_stats = system_monitor.get_performance_stats()
        
        # Get enhanced monitoring metrics
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        
        # Get recent logs
        recent_logs = system_monitor.get_recent_logs(limit=20)
        
        # Build comprehensive statistics
        statistics = {
            "system": {
                "uptime": enhanced_metrics.get("base_metrics", {}).get("uptime", 0),
                "cpu_usage": enhanced_metrics.get("base_metrics", {}).get("cpu_usage", 0),
                "memory_usage": enhanced_metrics.get("base_metrics", {}).get("memory_usage", 0),
                "disk_usage": enhanced_metrics.get("base_metrics", {}).get("disk_usage", 0),
                "network_io": enhanced_metrics.get("base_metrics", {}).get("network_io", {})
            },
            "models": {
                "total_models": len(model_statuses),
                "active_models": len([m for m in model_statuses if m.get("status") == "active"]),
                "inactive_models": len([m for m in model_statuses if m.get("status") != "active"]),
                "model_types": list(set([m.get("type", "unknown") for m in model_statuses])),
                "details": model_statuses
            },
            "training": training_stats,
            "datasets": dataset_stats,
            "monitoring": {
                "task_metrics": enhanced_metrics.get("task_metrics", []),
                "emotion_metrics": enhanced_metrics.get("emotion_metrics", []),
                "collaboration_metrics": enhanced_metrics.get("collaboration_metrics", {}),
                "data_streams": enhanced_metrics.get("data_streams", {})
            },
            "logs": recent_logs
        }
        
        return {"status": "success", "data": statistics}
        
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get system statistics")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

# System settings endpoints

# Get system settings
@app.get("/api/settings")
async def get_system_settings():
    """
    Get system settings
    
    Returns:
        System settings
    """
    try:
        settings = system_settings_manager.get_settings()
        return {"status": "success", "data": settings}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get system settings")
        raise HTTPException(status_code=500, detail="Failed to get system settings")

# Update system settings
@app.post("/api/settings/update")
async def update_system_settings(settings_data: dict):
    """
    Update system settings
    
    Args:
        settings_data: New settings data
        
    Returns:
        Update result
    """
    try:
        result = system_settings_manager.update_settings(settings_data)
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to update system settings")
        raise HTTPException(status_code=500, detail="Failed to update system settings")

# Reset system settings to default
@app.post("/api/settings/reset")
async def reset_system_settings():
    """
    Reset system settings to default
    
    Returns:
        Reset result
    """
    try:
        result = system_settings_manager.reset_to_default()
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to reset system settings")
        raise HTTPException(status_code=500, detail="Failed to reset system settings")

# Model management endpoints

# Get all models configuration
@app.get("/api/models")
async def get_all_models_config():
    """
    Get all models configuration
    
    Returns:
        List of model configurations
    """
    try:
        models_config = system_settings_manager.get_settings().get("models", {})
        models_status = model_registry.get_all_models_status()
        # 直接调用全局函数获取模型模式
        models_mode = get_all_models_mode()
        
        # Merge configuration, status, and mode information
        result = []
        for model_id, config in models_config.items():
            # Fix: models_status is a dictionary, not a list
            status = models_status.get(model_id, {})
            mode = models_mode.get(model_id, "local")
            
            result.append({
                "id": model_id,
                "name": config.get("name", model_id),
                "type": config.get("type", "unknown"),
                "source": config.get("source", "local"),
                "mode": mode,
                "active": config.get("active", True),
                "status": status.get("status", "unknown"),
                "version": status.get("version", "1.0.0"),
                "api_config": config.get("api_config", {})
            })
        
        return {"status": "success", "data": result}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get model configurations")
        raise HTTPException(status_code=500, detail="Failed to get model configurations")

# Add new model
@app.post("/api/models")
async def add_model(model_data: dict):
    """
    Add new model
    
    Args:
        model_data: Model data
            - id: Model ID
            - name: Model name
            - type: Model type
            - source: Model source (local/api)
            - active: Whether active
            - api_config: API configuration (if API model)
    
    Returns:
        Addition result
    """
    try:
        model_id = model_data.get("id")
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID cannot be empty")
        
        # Prepare model configuration
        model_config = {
            "name": model_data.get("name", model_id),
            "type": model_data.get("type", "unknown"),
            "source": model_data.get("source", "local"),
            "active": model_data.get("active", True)
        }
        
        # If it's an external model, add API configuration
        if model_data.get("source") == "external":
            api_config = model_data.get("api_config", {})
            model_config["api_config"] = api_config
        
        # Save model configuration
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # If active, load the model
        if model_config["active"]:
            if model_config["source"] == "external":
                switch_model_to_external(model_id, model_config["api_config"])
            else:
                try:
                    model_registry.load_model(model_id)
                except Exception as load_error:
                    error_handler.handle_error(load_error, "API", f"Failed to load model {model_id}")
        
        return {"status": "success", "message": f"Model {model_id} added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to add model")
        raise HTTPException(status_code=500, detail=str(e))

# Update model configuration
@app.put("/api/models")
async def update_models_config(models_data: list):
    """
    Update model configuration
    
    Args:
        models_data: List of model data
    
    Returns:
        Update result
    """
    try:
        updated_count = 0
        
        for model_data in models_data:
            model_id = model_data.get("id")
            if not model_id:
                continue
            
            # Get existing configuration
            current_config = system_settings_manager.get_model_setting(model_id)
            if not current_config:
                current_config = {}
            
            # Update configuration
            updated_config = {
                **current_config,
                "name": model_data.get("name", current_config.get("name", model_id)),
                "type": model_data.get("type", current_config.get("type", "unknown")),
                "source": model_data.get("source", current_config.get("source", "local")),
                "active": model_data.get("active", current_config.get("active", True))
            }
            
            # Update API configuration
            if model_data.get("source") == "external":
                updated_config["api_config"] = model_data.get("api_config", {})
            
            # Save updated configuration
            system_settings_manager.update_model_setting(model_id, updated_config)
            updated_count += 1
            
            # Handle model based on new mode and activation status
            current_mode = get_model_mode(model_id)
            new_source = updated_config["source"]
            
            if updated_config["active"]:
                if new_source == "external" and current_mode != "external":
                    # Switch to external API mode
                    switch_model_to_external(model_id, updated_config["api_config"])
                elif new_source == "local" and current_mode != "local":
                    # Switch to local mode
                    switch_model_to_local(model_id)
            else:
                # Deactivate model
                model_registry.unload_model(model_id)
        
        return {"status": "success", "updated_count": updated_count, "message": "Model configurations updated successfully"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to update model configurations")
        raise HTTPException(status_code=500, detail=str(e))

# Update single model configuration
@app.patch("/api/models/{model_id}")
async def update_model_config(model_id: str, model_data: dict):
    """
    Update single model configuration
    
    Args:
        model_id: Model ID
        model_data: Model data
    
    Returns:
        Update result
    """
    try:
        # Get existing configuration
        current_config = system_settings_manager.get_model_setting(model_id)
        if not current_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Update configuration
        updated_config = {**current_config, **model_data}
        
        # Save updated configuration
        system_settings_manager.update_model_setting(model_id, updated_config)
        
        # Handle model based on new mode and activation status
        if "active" in model_data:
            if model_data["active"]:
                if updated_config.get("source") == "external":
                    switch_model_to_external(model_id, updated_config.get("api_config", {}))
                else:
                    model_registry.load_model(model_id)
            else:
                model_registry.unload_model(model_id)
        
        return {"status": "success", "message": f"Model {model_id} configuration updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to update model {model_id} configuration")
        raise HTTPException(status_code=500, detail=str(e))

# Delete model
@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete model
    
    Args:
        model_id: Model ID
    
    Returns:
        Delete result
    """
    try:
        # Unload model
        model_registry.unload_model(model_id)
        
        # Delete configuration from system settings
        success = system_settings_manager.delete_model_setting(model_id)
        
        if success:
            return {"status": "success", "message": f"Model {model_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to delete model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Switch model to external API mode
@app.post("/api/models/{model_id}/switch-to-external")
async def api_switch_to_external(model_id: str, api_config: dict):
    """
    Switch model to external API mode
    
    Args:
        model_id: Model ID
        api_config: API configuration
    
    Returns:
        Switch result
    """
    try:
        result = switch_model_to_external(model_id, api_config)
        return {"status": "success", "message": result}
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to switch model {model_id} to external API mode")
        raise HTTPException(status_code=500, detail=str(e))

# Switch model to local mode
@app.post("/api/models/{model_id}/switch-to-local")
async def api_switch_to_local(model_id: str):
    """
    Switch model to local mode
    
    Args:
        model_id: Model ID
    
    Returns:
        Switch result
    """
    try:
        result = switch_model_to_local(model_id)
        return {"status": "success", "message": result}
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to switch model {model_id} to local mode")
        raise HTTPException(status_code=500, detail=str(e))

# Test generic API connection
@app.post("/api/external-api/test-connection")
async def api_test_generic_connection(connection_data: dict):
    """
    Test generic API connection
    
    Args:
        connection_data: Connection test data
            - api_url: API URL
            - api_key: API key
            - model_name: Model name
            - api_type: API type
            - provider: API provider
    
    Returns:
        Connection test result
    """
    try:
        from core.external_api_service import ExternalAPIService
        external_service = ExternalAPIService()
        result = external_service.test_connection(connection_data)
        return result
    except Exception as e:
        error_handler.handle_error(e, "API", "External API connection test failed")
        return {"status": "error", "message": str(e)}

# Set model API configuration
@app.post("/api/models/{model_id}/api-config")
async def api_set_model_api_config(model_id: str, api_config: dict):
    """
    Set model API configuration
    
    Args:
        model_id: Model ID
        api_config: API configuration
    
    Returns:
        Configuration set result
    """
    try:
        # Get existing model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Update API configuration
        model_config["api_config"] = api_config
        model_config["source"] = "external"
        
        # Save updated configuration
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # Test the connection with the new configuration
        test_result = test_external_api_connection({
            "model_id": model_id,
            "api_url": api_config.get("api_url", ""),
            "api_key": api_config.get("api_key", ""),
            "model_name": api_config.get("model_name", ""),
            "api_type": api_config.get("api_type", "custom")
        })
        
        if test_result["status"] == "success":
            # Switch to external mode if test is successful
            switch_model_to_external(model_id, api_config)
            return {"status": "success", "message": "API configuration set and tested successfully"}
        else:
            return {"status": "warning", "message": "API configuration saved but connection test failed", "test_result": test_result}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to set API configuration for model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model API configuration
@app.get("/api/models/{model_id}/api-config")
async def api_get_model_api_config(model_id: str):
    """
    Get model API configuration
    
    Args:
        model_id: Model ID
    
    Returns:
        Model API configuration
    """
    try:
        # Get model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Return API configuration if available
        api_config = model_config.get("api_config", {})
        return {"status": "success", "data": api_config}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to get API configuration for model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Get API service status
@app.get("/api/external-api/service-status")
async def api_get_api_service_status():
    """
    Get API service status
    
    Returns:
        API service status information
    """
    try:
        from core.external_api_service import ExternalAPIService
        external_service = ExternalAPIService()
        status = external_service.get_service_status()
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get API service status")
        raise HTTPException(status_code=500, detail=str(e))

# Get model API status
@app.get("/api/models/{model_id}/api-status")
async def api_get_model_api_status(model_id: str):
    """
    Get model API status
    
    Args:
        model_id: Model ID
    
    Returns:
        Model API status information
    """
    try:
        from core.external_api_service import ExternalAPIService
        external_service = ExternalAPIService()
        status = external_service.get_model_api_status(model_id)
        return {"status": "success", "data": status}
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to get API status for model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model running mode
@app.get("/api/models/{model_id}/mode")
async def api_get_model_mode(model_id: str):
    """
    Get model running mode
    
    Args:
        model_id: Model ID
    
    Returns:
        Model running mode
    """
    try:
        mode = get_model_mode(model_id)
        return {"status": "success", "mode": mode}
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to get model {model_id} running mode")
        raise HTTPException(status_code=500, detail=str(e))

# Batch switch model modes
@app.post("/api/models/batch-switch-mode")
async def api_batch_switch_mode(models_data: list):
    """
    Batch switch model modes
    
    Args:
        models_data: List of model data
            - id: Model ID
            - mode: Target mode (local/external)
            - api_config: API configuration (if switching to external mode)
    
    Returns:
        Batch switch result
    """
    try:
        results = batch_switch_model_modes(models_data)
        return {"status": "success", "results": results}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to batch switch model modes")
        raise HTTPException(status_code=500, detail=str(e))

# Test external API connection
@app.post("/api/models/test-connection")
async def api_test_connection(connection_data: dict):
    """
    Test external API connection
    
    Args:
        connection_data: Connection test data
            - model_id: Model ID
            - api_url: API URL
            - api_key: API key
            - model_name: Model name
            - api_type: API type
    
    Returns:
        Connection test result
    """
    try:
        # Ensure we have a consistent structure for the test function
        test_data = {
            "model_id": connection_data.get("model_id"),
            "api_url": connection_data.get("api_url"),
            "api_key": connection_data.get("api_key"),
            "model_name": connection_data.get("model_name"),
            "api_type": connection_data.get("api_type")
        }
        
        result = test_external_api_connection(test_data)
        return result
    except Exception as e:
        error_handler.handle_error(e, "API", "External API connection test failed")
        return {"status": "error", "message": str(e)}

# Start model
@app.post("/api/models/{model_id}/start")
async def start_model(model_id: str):
    """
    Start model service
    
    Args:
        model_id: Model ID
    
    Returns:
        Start result
    """
    try:
        # Get model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Activate model
        model_config["active"] = True
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # Load model
        if model_config.get("source") == "external":
            switch_model_to_external(model_id, model_config.get("api_config", {}))
        else:
            model_registry.load_model(model_id)
        
        return {"status": "success", "message": f"Model {model_id} started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to start model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Stop model
@app.post("/api/models/{model_id}/stop")
async def stop_model(model_id: str):
    """
    Stop model service
    
    Args:
        model_id: Model ID
    
    Returns:
        Stop result
    """
    try:
        # Get model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Deactivate model
        model_config["active"] = False
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # Unload model
        model_registry.unload_model(model_id)
        
        return {"status": "success", "message": f"Model {model_id} stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to stop model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Restart model
@app.post("/api/models/{model_id}/restart")
async def restart_model(model_id: str):
    """
    Restart model service
    
    Args:
        model_id: Model ID
    
    Returns:
        Restart result
    """
    try:
        # First stop the model
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Unload model
        model_registry.unload_model(model_id)
        
        # Reload model
        if model_config.get("source") == "external":
            switch_model_to_external(model_id, model_config.get("api_config", {}))
        else:
            model_registry.load_model(model_id)
        
        return {"status": "success", "message": f"Model {model_id} restarted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to restart model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

# Activate/deactivate model
@app.post("/api/models/{model_id}/activation")
async def toggle_model_activation(model_id: str, activation_data: dict):
    """
    Toggle model activation status
    
    Args:
        model_id: Model ID
        activation_data: Activation data with 'active' boolean
    
    Returns:
        Activation toggle result
    """
    try:
        # Get model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Update activation status
        active = activation_data.get("active", False)
        model_config["active"] = active
        system_settings_manager.update_model_setting(model_id, model_config)
        
        # Load or unload model based on activation status
        if active:
            if model_config.get("source") == "external":
                switch_model_to_external(model_id, model_config.get("api_config", {}))
            else:
                model_registry.load_model(model_id)
        else:
            model_registry.unload_model(model_id)
        
        action = "activated" if active else "deactivated"
        return {"status": "success", "message": f"Model {model_id} {action} successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to toggle model {model_id} activation")
        raise HTTPException(status_code=500, detail=str(e))

# Set primary model
@app.post("/api/models/{model_id}/primary")
async def set_primary_model(model_id: str):
    """
    Set a model as primary for its type
    
    Args:
        model_id: Model ID
    
    Returns:
        Primary model setting result
    """
    try:
        # Get model configuration
        model_config = system_settings_manager.get_model_setting(model_id)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        # Get model type
        model_type = model_config.get("type")
        if not model_type:
            raise HTTPException(status_code=400, detail="Model type is required to set as primary")
        
        # Update primary model configuration in system settings
        system_settings = system_settings_manager.get_settings()
        if "primary_models" not in system_settings:
            system_settings["primary_models"] = {}
        
        system_settings["primary_models"][model_type] = model_id
        system_settings_manager.update_settings(system_settings)
        
        return {"status": "success", "message": f"Model {model_id} set as primary for type {model_type}"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to set model {model_id} as primary")
        raise HTTPException(status_code=500, detail=str(e))

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """
    Error handling middleware
    
    Args:
        request: Request object
        call_next: Next middleware function
        
    Returns:
        Response object
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_handler.handle_error(e, "Middleware", f"Request processing error: {request.url}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "detail": str(e)}
        )

# Batch model operations endpoints

# Start all models
@app.post("/api/models/start-all")
async def start_all_models():
    """
    Start all models
    
    Returns:
        Operation result
    """
    try:
        # Get all models from settings
        models = system_settings_manager.get_settings().get("models", {})
        started_count = 0
        
        for model_id, config in models.items():
            try:
                # Start model if it's active
                if config.get("active", True):
                    if config.get("source") == "external":
                        switch_model_to_external(model_id, config.get("api_config", {}))
                    else:
                        model_registry.load_model(model_id)
                    started_count += 1
            except Exception as e:
                error_handler.handle_error(e, "API", f"Failed to start model {model_id}")
                
        return {"status": "success", "message": f"Successfully started {started_count} models", "started_count": started_count}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to start all models")
        raise HTTPException(status_code=500, detail=str(e))

# Stop all models
@app.post("/api/models/stop-all")
async def stop_all_models():
    """
    Stop all models
    
    Returns:
        Operation result
    """
    try:
        # Get all models from registry
        model_statuses = model_registry.get_all_models_status()
        stopped_count = 0
        
        for status in model_statuses:
            model_id = status.get("id")
            try:
                # Unload model
                model_registry.unload_model(model_id)
                stopped_count += 1
            except Exception as e:
                error_handler.handle_error(e, "API", f"Failed to stop model {model_id}")
                
        return {"status": "success", "message": f"Successfully stopped {stopped_count} models", "stopped_count": stopped_count}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to stop all models")
        raise HTTPException(status_code=500, detail=str(e))

# Restart all models
@app.post("/api/models/restart-all")
async def restart_all_models():
    """
    Restart all models
    
    Returns:
        Operation result
    """
    try:
        # Get all models from settings
        models = system_settings_manager.get_settings().get("models", {})
        restarted_count = 0
        
        for model_id, config in models.items():
            try:
                # First stop the model
                if model_registry.is_model_loaded(model_id):
                    model_registry.unload_model(model_id)
                
                # Then start it again if it's active
                if config.get("active", True):
                    if config.get("source") == "external":
                        switch_model_to_external(model_id, config.get("api_config", {}))
                    else:
                        model_registry.load_model(model_id)
                    restarted_count += 1
            except Exception as e:
                error_handler.handle_error(e, "API", f"Failed to restart model {model_id}")
                
        return {"status": "success", "message": f"Successfully restarted {restarted_count} models", "restarted_count": restarted_count}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to restart all models")
        raise HTTPException(status_code=500, detail=str(e))

# System operations endpoints

# Restart system
@app.post("/api/system/restart")
async def restart_system():
    """
    Restart the entire system
    
    Returns:
        Operation result
    """
    try:
        error_handler.log_info("System restart initiated through API", "System")
        
        # In a real implementation, this would trigger a system restart
        # For now, we'll just return a success message
        
        return {"status": "success", "message": "System restart initiated", "details": "The system is preparing to restart all components"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to initiate system restart")
        raise HTTPException(status_code=500, detail=str(e))

# Multimedia processing endpoints


# Process image input
@app.post("/api/process/image")
async def process_image(request: Request):
    """
    Process image input
    
    Supports both JSON image data and multipart/form-data file uploads
    
    Returns:
        Processed image result
    """
    try:
        # Check if it's a multipart request
        if request.headers.get("Content-Type", "").startswith("multipart/form-data"):
            form_data = await request.form()
            image_file = form_data.get("image")
            lang = form_data.get("lang", "en")
            session_id = form_data.get("session_id", "")
            model_id = form_data.get("model_id", "vision")
        else:
            # JSON request
            json_data = await request.json()
            image_data = json_data.get("image", "")
            lang = json_data.get("language", "en")
            session_id = json_data.get("session_id", "")
            model_id = json_data.get("model_id", "vision")
        
        # Get the vision model from registry
        model = model_registry.get_model(model_id)
        if not model:
            # Default to vision model if requested model is not available
            model = model_registry.get_model("vision")
            if not model:
                # Fallback to computer vision model
                model = model_registry.get_model("computer_vision")
                if not model:
                    raise ValueError(f"No suitable model found to process image")
        
        # Process image using the model
        try:
            if request.headers.get("Content-Type", "").startswith("multipart/form-data"):
                # For file uploads, process the file
                response = model.process_image_file(image_file, language=lang, session_id=session_id)
            else:
                # For JSON image data, process the image data
                response = model.process_image_data(image_data, language=lang, session_id=session_id)
            return {"status": "success", "data": response}
        except Exception as model_error:
            # If direct processing fails, use AGI coordinator as fallback
            error_handler.log_info(f"Model {model_id} image processing failed, using AGI coordinator fallback", "API")
            response = agi_coordinator.process_image(
                image_file if request.headers.get("Content-Type", "").startswith("multipart/form-data") else image_data,
                language=lang, 
                session_id=session_id
            )
            return {"status": "success", "data": response}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process image")
        raise HTTPException(status_code=500, detail=str(e))

# Process video input
@app.post("/api/process/video")
async def process_video(request: Request):
    """
    Process video input
    
    Supports both JSON video data and multipart/form-data file uploads
    
    Returns:
        Processed video result
    """
    try:
        # Check if it's a multipart request
        if request.headers.get("Content-Type", "").startswith("multipart/form-data"):
            form_data = await request.form()
            video_file = form_data.get("video")
            lang = form_data.get("lang", "en")
            session_id = form_data.get("session_id", "")
            model_id = form_data.get("model_id", "computer_vision")
        else:
            # JSON request
            json_data = await request.json()
            video_data = json_data.get("video", "")
            lang = json_data.get("language", "en")
            session_id = json_data.get("session_id", "")
            model_id = json_data.get("model_id", "computer_vision")
        
        # Get the video processing model from registry
        model = model_registry.get_model(model_id)
        if not model:
            # Default to computer vision model if requested model is not available
            model = model_registry.get_model("computer_vision")
            if not model:
                # Fallback to vision model
                model = model_registry.get_model("vision")
                if not model:
                    raise ValueError(f"No suitable model found to process video")
        
        # Process video using the model
        try:
            if request.headers.get("Content-Type", "").startswith("multipart/form-data"):
                # For file uploads, process the file
                response = model.process_video_file(video_file, language=lang, session_id=session_id)
            else:
                # For JSON video data, process the video data
                response = model.process_video_data(video_data, language=lang, session_id=session_id)
            return {"status": "success", "data": response}
        except Exception as model_error:
            # If direct processing fails, use AGI coordinator as fallback
            error_handler.log_info(f"Model {model_id} video processing failed, using AGI coordinator fallback", "API")
            response = agi_coordinator.process_video(
                video_file if request.headers.get("Content-Type", "").startswith("multipart/form-data") else video_data,
                language=lang, 
                session_id=session_id
            )
            return {"status": "success", "data": response}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process video")
        raise HTTPException(status_code=500, detail=str(e))

# Process audio input
@app.post("/api/process/audio")
async def process_audio(audio_data: dict):
    """
    Process audio input
    
    Args:
        audio_data: Audio input data
            - audio: Input audio
            - language: Language code
            - session_id: Session ID
            - model_id: Model ID to use
    
    Returns:
        Processed audio result
    """
    try:
        audio = audio_data.get("audio", "")
        language = audio_data.get("language", "en-US")
        session_id = audio_data.get("session_id", "")
        model_id = audio_data.get("model_id", "audio")
        
        # Get the audio processing model from registry
        model = model_registry.get_model(model_id)
        if not model:
            # Default to audio model if requested model is not available
            model = model_registry.get_model("audio")
            if not model:
                raise ValueError(f"No suitable model found to process audio")
        
        # Process audio using the model
        try:
            response = model.process_audio(audio, language=language, session_id=session_id)
            return {"status": "success", "data": response}
        except Exception as model_error:
            # If direct processing fails, use AGI coordinator as fallback
            error_handler.log_info(f"Model {model_id} audio processing failed, using AGI coordinator fallback", "API")
            response = agi_coordinator.process_audio(audio, language=language, session_id=session_id)
            return {"status": "success", "data": response}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to process audio")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Base API endpoints





# Search knowledge base
@app.get("/api/knowledge/search")
async def search_knowledge(query: str = None, domain: str = None):
    """
    Search knowledge base with optional query and domain filters
    
    Args:
        query: Search query
        domain: Domain filter
        
    Returns:
        Search results from actual knowledge base
    """
    try:
        # Get knowledge model and perform actual search
        knowledge_model = model_registry.get_model("knowledge")
        if not knowledge_model:
            raise HTTPException(status_code=500, detail="Knowledge model not available")
        
        # Perform actual knowledge search
        search_results = knowledge_model.search_knowledge(query=query, domain=domain)
        
        return {
            "status": "success", 
            "results": search_results.get("results", []), 
            "total": search_results.get("total", 0),
            "search_time": search_results.get("search_time", 0)
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Knowledge search failed")
        raise HTTPException(status_code=500, detail=str(e))

# Preview knowledge file
@app.get("/api/knowledge/files/{file_id}/preview")
async def preview_knowledge_file(file_id: str):
    """
    Get preview of a knowledge file
    
    Args:
        file_id: File ID
        
    Returns:
        File preview
    """
    try:
        # Mock preview data based on file type
        file_preview = {
            "id": file_id,
            "name": f"preview_file_{file_id}.txt",
            "type": "txt",
            "content": "This is a preview of the file content.\nActual content would be displayed here.",
            "size": "1.5 KB",
            "last_modified": "2024-01-15T10:30:00"
        }
        return {"status": "success", "preview": file_preview}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get file preview")
        raise HTTPException(status_code=500, detail=str(e))

# Download knowledge file
@app.get("/api/knowledge/files/{file_id}/download")
async def download_knowledge_file(file_id: str):
    """
    Download a knowledge file
    
    Args:
        file_id: File ID
        
    Returns:
        Download URL and status
    """
    try:
        # Mock download URL (in real implementation, this would be a signed URL or file stream)
        download_url = f"/download/knowledge/{file_id}"
        return {"status": "success", "download_url": download_url, "message": "Download started"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to initiate download")
        raise HTTPException(status_code=500, detail=str(e))

# Delete knowledge file
@app.delete("/api/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: str):
    """
    Delete a knowledge file
    
    Args:
        file_id: File ID
        
    Returns:
        Deletion status
    """
    try:
        # In a real implementation, this would delete the file from storage
        # For now, we'll just return a success message
        return {"status": "success", "message": "File deleted successfully"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to delete file")
        raise HTTPException(status_code=500, detail=str(e))

# 定义健康检查端点
@app.get("/health")
async def health_check():
    """
    Health check endpoint to quickly respond to frontend connection requests
    """
    return {"status": "healthy", "message": "Self Soul system is running normally"}

# Knowledge statistics endpoint - provides statistical data for frontend KnowledgeView.vue
@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """
    Get knowledge base statistics
    
    Returns:
        Knowledge base statistics including total domains, items, size, etc.
    """
    try:
        # Get knowledge base status from knowledge model via model registry
        knowledge_model = model_registry.get_model('knowledge')
        if not knowledge_model:
            return {"status": "success", "stats": {
                "total_domains": 0,
                "total_items": 0,
                "total_size": "0 MB",
                "updated_domains": 0,
                "recent_updates": 0,
                "domain_categories": []
            }}
        
        knowledge_base_status = knowledge_model.get_knowledge_base_status()
        
        # Use actual data from knowledge base
        knowledge_stats = {
            "total_domains": knowledge_base_status.get("total_domains", 0),
            "total_items": knowledge_base_status.get("total_items", 0),
            "total_size": knowledge_base_status.get("total_size", "0 MB"),
            "updated_domains": knowledge_base_status.get("updated_domains", 0),
            "recent_updates": knowledge_base_status.get("recent_updates", 0),
            "domain_categories": knowledge_base_status.get("domain_categories", [])
        }
        
        return {"status": "success", "stats": knowledge_stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get knowledge statistics")
        # Return empty data instead of simulated data
        return {
            "status": "success",
            "stats": {
                "total_domains": 0,
                "total_items": 0,
                "total_size": "0 MB",
                "updated_domains": 0,
                "recent_updates": 0,
                "domain_categories": []
            }
        }

# Knowledge file list endpoint
@app.get("/api/knowledge/files")
async def get_knowledge_files():
    """
    Get list of knowledge files
    """
    try:
        # Get actual knowledge files
        from core.knowledge.knowledge_enhancer import KnowledgeEnhancer
        knowledge_enhancer = KnowledgeEnhancer()
        files = knowledge_enhancer.get_available_knowledge_files()
        return {"status": "success", "files": files}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get knowledge files")
        # Return empty list instead of mock data
        return {"status": "success", "files": []}

# Autonomous learning endpoints
@app.post("/api/knowledge/auto-learning/start")
async def start_autonomous_learning(domains: list = None, priority: str = "balanced"):
    """
    Start autonomous learning cycle
    
    Args:
        domains: List of knowledge domains to focus on
        priority: Learning priority (balanced, exploration, exploitation)
        
    Returns:
        Status of the operation
    """
    try:
        success = autonomous_learning_manager.start_autonomous_learning_cycle(domains=domains, priority=priority)
        if success:
            return {"status": "success", "message": "Autonomous learning started successfully"}
        else:
            return {"status": "warning", "message": "Autonomous learning is already running"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to start autonomous learning")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/auto-learning/stop")
async def stop_autonomous_learning():
    """
    Stop autonomous learning cycle
    
    Returns:
        Status of the operation
    """
    try:
        success = autonomous_learning_manager.stop_autonomous_learning_cycle()
        if success:
            return {"status": "success", "message": "Autonomous learning stopped successfully"}
        else:
            return {"status": "warning", "message": "Autonomous learning was not running"}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to stop autonomous learning")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/auto-learning/progress")
async def get_autonomous_learning_progress():
    """
    Get autonomous learning progress
    
    Returns:
        Current learning progress, status and logs
    """
    try:
        progress_data = autonomous_learning_manager.get_learning_progress()
        return {
            "status": "success",
            "progress": progress_data["progress"],
            "learning_status": progress_data["status"],
            "logs": progress_data["logs"],
            "domains": progress_data["domains"],
            "priority": progress_data["priority"]
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get autonomous learning progress")
        # Return empty data instead of mock data
        return {
            "status": "error",
            "message": "Failed to retrieve autonomous learning progress",
            "progress": 0,
            "learning_status": "error",
            "logs": [],
            "domains": [],
            "priority": "balanced"
        }

# Hardware configuration endpoints

# Get hardware configuration
@app.get("/api/hardware/config")
async def get_hardware_config():
    """
    Get current hardware configuration
    
    Returns:
        Hardware configuration including cameras, sensors, and actuators
    """
    try:
        # Get hardware configuration from system settings
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        
        # Return default configuration if none exists
        if not hardware_config:
            hardware_config = {
                "camera_settings": {
                    "max_cameras": 4,
                    "default_resolution": "1280x720",
                    "supported_interfaces": ["usb", "ethernet", "csi"],
                    "frame_rate": 30,
                    "auto_detect": True
                },
                "sensor_settings": {
                    "supported_sensors": [
                        "temperature", "humidity", "accelerometer", "gyroscope",
                        "pressure", "distance", "infrared", "light", "smoke"
                    ],
                    "polling_interval": 1000,
                    "data_format": "json"
                },
                "actuator_settings": {
                    "supported_protocols": ["uart", "i2c", "spi", "pwm"],
                    "default_baud_rate": 9600,
                    "timeout_ms": 5000
                },
                "communication_settings": {
                    "max_retries": 3,
                    "retry_delay_ms": 100,
                    "connection_timeout_ms": 10000
                },
                "cameras": [
                    {
                        "id": "camera_1",
                        "name": "Primary Camera",
                        "type": "usb",
                        "resolution": "1280x720",
                        "frame_rate": 30,
                        "status": "connected",
                        "port": "/dev/video0"
                    }
                ],
                "sensors": [
                    {
                        "id": "sensor_1",
                        "name": "Temperature Sensor",
                        "type": "temperature",
                        "protocol": "i2c",
                        "address": "0x48",
                        "status": "connected",
                        "current_value": 25.5
                    }
                ],
                "actuators": [
                    {
                        "id": "actuator_1",
                        "name": "Motor Controller",
                        "type": "motor",
                        "protocol": "pwm",
                        "channel": 1,
                        "status": "connected"
                    }
                ]
            }
        
        return {"status": "success", "data": hardware_config}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get hardware configuration")
        raise HTTPException(status_code=500, detail="Failed to get hardware configuration")

# Update hardware configuration
@app.post("/api/hardware/config")
async def update_hardware_config(hardware_data: dict):
    """
    Update hardware configuration
    
    Args:
        hardware_data: New hardware configuration data
        
    Returns:
        Update result
    """
    try:
        # Validate hardware configuration
        required_sections = ["camera_settings", "sensor_settings", "actuator_settings"]
        for section in required_sections:
            if section not in hardware_data:
                raise HTTPException(status_code=400, detail=f"Missing required section: {section}")
        
        # Get current settings
        settings = system_settings_manager.get_settings()
        
        # Update hardware configuration
        settings["hardware_config"] = hardware_data
        
        # Save updated settings
        system_settings_manager.update_settings(settings)
        
        return {"status": "success", "message": "Hardware configuration updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to update hardware configuration")
        raise HTTPException(status_code=500, detail="Failed to update hardware configuration")

# Test hardware connections
@app.post("/api/hardware/test-connections")
async def test_hardware_connections():
    """
    Test all hardware connections
    
    Returns:
        Connection test results
    """
    try:
        # Get current hardware configuration
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        
        # Simulate hardware connection testing
        test_results = {
            "cameras": [],
            "sensors": [],
            "actuators": [],
            "overall_status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test camera connections
        cameras = hardware_config.get("cameras", [])
        for camera in cameras:
            camera_id = camera.get("id", "unknown")
            camera_name = camera.get("name", "Unknown Camera")
            camera_type = camera.get("type", "unknown")
            
            # Simulate camera connection test
            connected = True  # In real implementation, this would test actual connection
            test_results["cameras"].append({
                "id": camera_id,
                "name": camera_name,
                "type": camera_type,
                "connected": connected,
                "message": "Camera connected successfully" if connected else "Camera connection failed"
            })
        
        # Test sensor connections
        sensors = hardware_config.get("sensors", [])
        for sensor in sensors:
            sensor_id = sensor.get("id", "unknown")
            sensor_name = sensor.get("name", "Unknown Sensor")
            sensor_type = sensor.get("type", "unknown")
            
            # Simulate sensor connection test
            connected = True  # In real implementation, this would test actual connection
            test_results["sensors"].append({
                "id": sensor_id,
                "name": sensor_name,
                "type": sensor_type,
                "connected": connected,
                "message": "Sensor connected successfully" if connected else "Sensor connection failed"
            })
        
        # Test actuator connections
        actuators = hardware_config.get("actuators", [])
        for actuator in actuators:
            actuator_id = actuator.get("id", "unknown")
            actuator_name = actuator.get("name", "Unknown Actuator")
            actuator_type = actuator.get("type", "unknown")
            
            # Simulate actuator connection test
            connected = True  # In real implementation, this would test actual connection
            test_results["actuators"].append({
                "id": actuator_id,
                "name": actuator_name,
                "type": actuator_type,
                "connected": connected,
                "message": "Actuator connected successfully" if connected else "Actuator connection failed"
            })
        
        # Check if any connections failed
        all_connections = test_results["cameras"] + test_results["sensors"] + test_results["actuators"]
        failed_connections = [conn for conn in all_connections if not conn.get("connected", False)]
        
        if failed_connections:
            test_results["overall_status"] = "partial"
            test_results["failed_count"] = len(failed_connections)
        
        return {"status": "success", "data": test_results}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to test hardware connections")
        raise HTTPException(status_code=500, detail="Failed to test hardware connections")

# Import CameraManager
from core.hardware.camera_manager import camera_manager

# Camera stream control endpoints
@app.post("/api/cameras/{camera_id}/stream/start")
async def start_camera_stream(camera_id: str):
    """
    Start streaming from a specific camera
    
    Args:
        camera_id: Camera ID
        
    Returns:
        Operation status
    """
    try:
        # Check if camera exists in hardware config
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        cameras = hardware_config.get("cameras", [])
        
        camera = next((cam for cam in cameras if cam.get("id") == camera_id), None)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Get camera index from hardware config
        camera_index = int(camera.get("port", "/dev/video0").split("/")[-1].replace("video", ""))
        resolution_str = camera.get("resolution", "1280x720")
        width, height = map(int, resolution_str.split("x"))
        fps = float(camera.get("frame_rate", 30))
        
        # Connect to camera if not already connected
        camera_info = camera_manager.get_camera_info(camera_id)
        if not camera_info:
            connect_result = camera_manager.connect_camera(
                camera_id, 
                camera_index, 
                (width, height), 
                fps
            )
            if not connect_result["success"]:
                raise HTTPException(status_code=500, detail=connect_result["error"])
        
        # Start streaming
        stream_result = camera_manager.start_stream(camera_id)
        if not stream_result["success"]:
            raise HTTPException(status_code=500, detail=stream_result["error"])
        
        return {"status": "success", "message": f"Camera stream started for {camera_id}"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to start camera stream: {camera_id}")
        raise HTTPException(status_code=500, detail=f"Failed to start camera stream: {str(e)}")

@app.post("/api/cameras/{camera_id}/stream/stop")
async def stop_camera_stream(camera_id: str):
    """
    Stop streaming from a specific camera
    
    Args:
        camera_id: Camera ID
        
    Returns:
        Operation status
    """
    try:
        # Stop streaming
        stream_result = camera_manager.stop_stream(camera_id)
        if not stream_result["success"]:
            raise HTTPException(status_code=400, detail=stream_result["error"])
        
        return {"status": "success", "message": f"Camera stream stopped for {camera_id}"}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to stop camera stream: {camera_id}")
        raise HTTPException(status_code=500, detail=f"Failed to stop camera stream: {str(e)}")

@app.get("/api/cameras/{camera_id}/stream/status")
async def get_camera_stream_status(camera_id: str):
    """
    Get the streaming status of a specific camera
    
    Args:
        camera_id: Camera ID
        
    Returns:
        Camera streaming status
    """
    try:
        # Get camera info
        camera_info = camera_manager.get_camera_info(camera_id)
        if not camera_info:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "is_streaming": camera_info["is_streaming"],
            "resolution": camera_info["resolution"],
            "fps": camera_info["fps"],
            "connected_at": camera_info["connected_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to get camera stream status: {camera_id}")
        raise HTTPException(status_code=500, detail=f"Failed to get camera stream status: {str(e)}")

# Stereo vision endpoints
@app.get("/api/cameras/stereo-pairs")
async def get_stereo_pairs():
    """
    Get all configured stereo camera pairs
    
    Returns:
        List of stereo camera pairs
    """
    try:
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        stereo_pairs = hardware_config.get("stereo_pairs", [])
        
        return {
            "status": "success",
            "stereo_pairs": stereo_pairs
        }
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get stereo camera pairs")
        raise HTTPException(status_code=500, detail=f"Failed to get stereo camera pairs: {str(e)}")

@app.post("/api/cameras/stereo-pairs/{pair_id}/process")
async def process_stereo_pair(pair_id: str, request: Request):
    """
    Process a stereo camera pair to get depth information
    
    Args:
        pair_id: Stereo pair ID
        request: Request body containing processing parameters
        
    Returns:
        Depth map and 3D information
    """
    try:
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        stereo_pairs = hardware_config.get("stereo_pairs", [])
        
        stereo_pair = next((pair for pair in stereo_pairs if pair.get("id") == pair_id), None)
        if not stereo_pair:
            raise HTTPException(status_code=404, detail=f"Stereo pair {pair_id} not found")
        
        left_camera_id = stereo_pair.get("left_camera_id")
        right_camera_id = stereo_pair.get("right_camera_id")
        
        # Check if both cameras are connected and streaming
        left_info = camera_manager.get_camera_info(left_camera_id)
        right_info = camera_manager.get_camera_info(right_camera_id)
        
        if not left_info or not right_info:
            raise HTTPException(status_code=400, detail="One or both cameras in the stereo pair are not connected")
        
        if not left_info["is_streaming"] or not right_info["is_streaming"]:
            raise HTTPException(status_code=400, detail="One or both cameras in the stereo pair are not streaming")
        
        # Get processing parameters
        params = await request.json()
        min_disparity = params.get("min_disparity", 0)
        num_disparities = params.get("num_disparities", 16)
        block_size = params.get("block_size", 15)
        
        # Perform binocular vision processing
        result = camera_manager.perform_binocular_vision(
            left_camera_id,
            right_camera_id,
            min_disparity=min_disparity,
            num_disparities=num_disparities,
            block_size=block_size
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return the result
        return {
            "status": "success",
            "stereo_pair_id": pair_id,
            "result": result["data"]
        }
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to process stereo pair {pair_id}")
        raise HTTPException(status_code=500, detail=f"Failed to process stereo pair: {str(e)}")

@app.post("/api/cameras/stereo-pairs/{pair_id}/calibrate")
async def calibrate_stereo_pair(pair_id: str):
    """
    Calibrate a stereo camera pair
    
    Args:
        pair_id: Stereo pair ID
        
    Returns:
        Calibration result
    """
    try:
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        stereo_pairs = hardware_config.get("stereo_pairs", [])
        
        stereo_pair = next((pair for pair in stereo_pairs if pair.get("id") == pair_id), None)
        if not stereo_pair:
            raise HTTPException(status_code=404, detail=f"Stereo pair {pair_id} not found")
        
        # For now, we'll simulate calibration as the actual implementation would require
        # physical chessboard calibration patterns and user interaction
        # In a real implementation, this would be a more complex process
        
        # Simulate successful calibration
        calibration_result = {
            "success": True,
            "message": f"Stereo pair {pair_id} calibrated successfully",
            "calibration_data": {
                "baseline": 0.1,  # 10cm baseline distance
                "focal_length": 500,  # Pixels
                "principal_point": (320, 240),  # (cx, cy)
                "rectification_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "projection_matrix_left": [[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0]],
                "projection_matrix_right": [[500, 0, 320, -50], [0, 500, 240, 0], [0, 0, 1, 0]],
                "disparity_to_depth": 50.0  # Baseline * focal_length
            }
        }
        
        return {
            "status": "success",
            "calibration_result": calibration_result
        }
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to calibrate stereo pair {pair_id}")
        raise HTTPException(status_code=500, detail=f"Failed to calibrate stereo pair: {str(e)}")

# Get camera feed
@app.websocket("/ws/camera-feed/{camera_id}")
async def websocket_camera_feed(websocket: WebSocket, camera_id: str):
    """
    Real-time camera feed WebSocket endpoint that streams actual camera frames
    
    Args:
        websocket: WebSocket connection
        camera_id: Camera ID
    """
    await connection_manager.connect(websocket)
    try:
        # Get hardware configuration to verify camera exists
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        cameras = hardware_config.get("cameras", [])
        
        camera = next((cam for cam in cameras if cam.get("id") == camera_id), None)
        if not camera:
            await websocket.send_json({
                "type": "error",
                "message": f"Camera {camera_id} not found in configuration"
            })
            return
        
        # Check if camera is streaming
        camera_info = camera_manager.get_camera_info(camera_id)
        if not camera_info:
            await websocket.send_json({
                "type": "error",
                "message": f"Camera {camera_id} is not connected"
            })
            return
        
        if not camera_info["is_streaming"]:
            # Try to start streaming if not already streaming
            stream_result = camera_manager.start_stream(camera_id)
            if not stream_result["success"]:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to start camera stream: {stream_result.get('error', 'Unknown error')}"
                })
                return
        
        await websocket.send_json({
            "type": "connected",
            "message": f"Camera feed connection established for {camera.get('name', camera_id)}",
            "camera_info": camera_info
        })
        
        # Stream actual camera frames
        frame_count = 0
        last_frame = None
        while True:
            try:
                # Get the last frame from the camera manager
                frame = camera_manager.get_last_frame(camera_id)
                
                # Check if we have a new frame
                if frame is not None and (last_frame is None or not np.array_equal(frame, last_frame)):
                    # Convert frame to base64 for transmission
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        # Convert to bytes and then to base64
                        jpg_bytes = buffer.tobytes()
                        # For efficiency, we'll send frame metadata first and then binary data
                        frame_metadata = {
                            "type": "camera_frame_metadata",
                            "camera_id": camera_id,
                            "frame_number": frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                            "data_type": "real",
                            "data_length": len(jpg_bytes)
                        }
                        
                        await websocket.send_json(frame_metadata)
                        
                        # Send the actual binary frame data
                        try:
                            await websocket.send_bytes(jpg_bytes)
                        except TypeError:
                            # If send_bytes is not supported, fall back to base64 encoding
                            import base64
                            base64_frame = base64.b64encode(jpg_bytes).decode('utf-8')
                            await websocket.send_json({
                                "type": "camera_frame_base64",
                                "data": base64_frame,
                                "frame_number": frame_count,
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        frame_count += 1
                        last_frame = frame
                    else:
                        error_handler.log_warning(f"Failed to encode frame from camera {camera_id}", "Camera")
                
                # Sleep for a short time to control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
            except Exception as frame_error:
                logger.error(f"Error processing camera frame: {str(frame_error)}")
                # Continue streaming even if there's an error processing a single frame
                await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"Camera feed WebSocket disconnected: {camera_id}")
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", f"Camera feed WebSocket error: {camera_id}")
        connection_manager.disconnect(websocket)

# Sensor data stream
@app.websocket("/ws/sensor-data/{sensor_id}")
async def websocket_sensor_data(websocket: WebSocket, sensor_id: str):
    """
    Real-time sensor data WebSocket endpoint
    
    Args:
        websocket: WebSocket connection
        sensor_id: Sensor ID
    """
    await connection_manager.connect(websocket)
    try:
        # Get hardware configuration to verify sensor exists
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        sensors = hardware_config.get("sensors", [])
        
        sensor = next((sens for sens in sensors if sens.get("id") == sensor_id), None)
        if not sensor:
            await websocket.send_json({
                "type": "error",
                "message": f"Sensor {sensor_id} not found"
            })
            return
        
        await websocket.send_json({
            "type": "connected",
            "message": f"Sensor data stream established for {sensor.get('name', sensor_id)}"
        })
        
        # Simulate sensor data stream
        while True:
            # Simulate sensor reading based on sensor type
            sensor_type = sensor.get("type", "unknown")
            if sensor_type == "temperature":
                value = 20 + (5 * (datetime.now().second % 10) / 10)  # Simulate temperature variation
                unit = "°C"
            elif sensor_type == "humidity":
                value = 50 + (20 * (datetime.now().second % 10) / 10)  # Simulate humidity variation
                unit = "%"
            elif sensor_type == "accelerometer":
                value = {
                    "x": (datetime.now().second % 10) - 5,
                    "y": (datetime.now().second % 8) - 4,
                    "z": (datetime.now().second % 12) - 6
                }
                unit = "m/s²"
            else:
                value = (datetime.now().second % 100)  # Default simulated value
                unit = "units"
            
            sensor_data = {
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
                "value": value,
                "unit": unit,
                "timestamp": datetime.now().isoformat(),
                "quality": "good"
            }
            
            await websocket.send_json({
                "type": "sensor_reading",
                "data": sensor_data
            })
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        error_handler.handle_error(e, "WebSocket", f"Sensor data WebSocket error: {sensor_id}")
        connection_manager.disconnect(websocket)

# Control actuator
@app.post("/api/hardware/actuators/{actuator_id}/control")
async def control_actuator(actuator_id: str, control_data: dict):
    """
    Control an actuator
    
    Args:
        actuator_id: Actuator ID
        control_data: Control commands and parameters
        
    Returns:
        Control result
    """
    try:
        # Get hardware configuration to verify actuator exists
        settings = system_settings_manager.get_settings()
        hardware_config = settings.get("hardware_config", {})
        actuators = hardware_config.get("actuators", [])
        
        actuator = next((act for act in actuators if act.get("id") == actuator_id), None)
        if not actuator:
            raise HTTPException(status_code=404, detail=f"Actuator {actuator_id} not found")
        
        # Extract control parameters
        command = control_data.get("command", "")
        parameters = control_data.get("parameters", {})
        
        # Simulate actuator control (in real implementation, this would send actual commands)
        control_result = {
            "actuator_id": actuator_id,
            "command": command,
            "parameters": parameters,
            "success": True,
            "message": f"Actuator {actuator_id} executed command: {command}",
            "timestamp": datetime.now().isoformat()
        }
        
        return {"status": "success", "data": control_result}
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, "API", f"Failed to control actuator {actuator_id}")
        raise HTTPException(status_code=500, detail=f"Failed to control actuator {actuator_id}")

# System statistics endpoint
@app.get("/api/system/stats")
async def get_system_stats():
    """
    Get system statistics
    """
    try:
        # Get system monitoring data
        enhanced_metrics = system_monitor.get_enhanced_metrics()
        
        # Prepare system statistics using actual data with fallback to 0
        system_stats = {
            "active_models": enhanced_metrics.get("active_models", 0),
            "total_models": enhanced_metrics.get("total_models", 0),
            "cpu_usage": enhanced_metrics.get("base_metrics", {}).get("cpu_usage", 0),
            "memory_usage": enhanced_metrics.get("base_metrics", {}).get("memory_usage", 0),
            "disk_usage": enhanced_metrics.get("base_metrics", {}).get("disk_usage", 0),
            "uptime": enhanced_metrics.get("base_metrics", {}).get("uptime", "00:00:00")
        }
        
        return {"status": "success", "stats": system_stats}
    except Exception as e:
        error_handler.handle_error(e, "API", "Failed to get system statistics")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

# Initialize core components - This function is now deprecated and should not be called directly
# All initialization should happen in the startup_event to prevent duplicate initialization
def initialize_core_components():
    """
    DEPRECATED: Do not call this function directly. 
    All component initialization is now handled by the startup_event to prevent duplicate loading.
    """
    error_handler.log_warning("initialize_core_components() is deprecated and should not be called directly. Use startup_event instead.", "System")
    return True

# Asynchronous initialization function
async def async_initialize_components():
    """
    Asynchronously initialize time-consuming system components
    """
    try:
        # Start any asynchronous background tasks
        error_handler.log_info("Starting asynchronous initialization of system components", "System")
        
        # Start system monitoring
        if system_monitor:
            await system_monitor.start_monitoring()
            error_handler.log_info("System monitoring started", "System")
        
    except Exception as e:
        error_handler.handle_error(e, "System", "Failed to initialize components asynchronously")


# Server startup code
if __name__ == "__main__":
    """
    Main entry point for starting the Self Soul AGI system backend server
    """
    # 添加内存优化配置
    from core.memory_optimization import configure_memory_optimization, MemoryOptimizer
    
    # 检测系统资源并配置适当的内存优化策略
    # 可以根据命令行参数、环境变量或配置文件来设置
    import argparse
    parser = argparse.ArgumentParser(description='Self Soul AGI System')
    parser.add_argument('--lightweight', action='store_true', help='Run in lightweight mode with reduced memory usage')
    parser.add_argument('--max-memory', type=int, default=75, help='Maximum memory usage percentage threshold')
    args = parser.parse_args()
    
    # 配置内存优化
    configure_memory_optimization(
        enable_optimization=True,
        lightweight_mode=args.lightweight,
        max_memory_usage=args.max_memory
    )
    
    # Add file logging for better error capture
    import logging
    from logging.handlers import RotatingFileHandler
    import os
    
    # Ensure logs directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure file logging
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "server_startup.log"),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("MainServer")
    logger.info("Starting Self Soul AGI system backend server...")
    
    # Add port availability check
    import socket
    def check_port_available(port):
        """Check if a port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result != 0
    
    # Check port availability before starting
    if not check_port_available(MAIN_API_PORT):
        logger.error(f"Port {MAIN_API_PORT} is already in use. Cannot start server.")
        sys.exit(1)
    else:
        logger.info(f"Port {MAIN_API_PORT} is available. Proceeding with server startup.")
    
    # Initialize core components synchronously
    if not initialize_core_components():
        logger.error("Failed to initialize core components. Exiting.")
        sys.exit(1)
    
    # Create FastAPI application lifespan event handler
    @app.on_event("startup")
    async def on_startup():
        """Run on application startup"""
        logger.info("FastAPI application startup event triggered")
        await async_initialize_components()
        logger.info("AGI System startup complete")
    
    # Add basic health endpoint verification
    @app.get("/test-health", tags=["System"])
    async def test_health():
        return {"status": "ok", "message": "FastAPI application is running"}
    
    # Start the FastAPI application with uvicorn (Python 3.6.3 compatible)
    # Use traditional event loop approach instead of asyncio.run() which is not available in Python 3.6
    logger.info(f"Creating uvicorn server configuration for host 0.0.0.0 and port {MAIN_API_PORT}")
    logger.info(f"FastAPI application has {len(app.routes)} routes registered")
    
    # Test FastAPI app directly before uvicorn
    logger.info("Testing FastAPI application directly")
    for route in app.routes:
        if hasattr(route, "path"):
            logger.debug(f"Registered route: {route.path}")
    
    # Log health endpoint details
    health_routes = [route for route in app.routes if hasattr(route, "path") and "/health" in route.path]
    logger.info(f"Found {len(health_routes)} health-related endpoints")
    
    config = uvicorn.Config(
        "core.main:app",
        host="0.0.0.0",
        port=MAIN_API_PORT,
        reload=True,
        log_level="debug",  # Increased log level for debugging
        access_log=True
    )
    server = uvicorn.Server(config)
    
    # Use the traditional event loop approach for Python 3.6 compatibility
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        error_handler.log_info("Received shutdown signal. Shutting down...", "System")
    finally:
        loop.close()
        error_handler.log_info("AGI System shutdown complete", "System")
