#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# Minimal FastAPI server for Self Soul AGI System
This version includes mock API endpoints to support the frontend application
"""

import os
import sys
import uuid
import time
from datetime import datetime

# Add the root directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

# Create FastAPI application instance
app = FastAPI(
    title="Self Soul AGI System",
    description="Minimal API server with mock endpoints for Self Soul AGI System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for models
mock_models = [
    {"id": "manager", "name": "Manager Model", "port": 8001, "status": "running", "last_active": datetime.now().isoformat()},
    {"id": "language", "name": "Language Model", "port": 8002, "status": "running", "last_active": datetime.now().isoformat()},
    {"id": "knowledge", "name": "Knowledge Model", "port": 8003, "status": "running", "last_active": datetime.now().isoformat()},
    {"id": "vision_image", "name": "Vision Model", "port": 8004, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "audio", "name": "Audio Model", "port": 8005, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "autonomous", "name": "Autonomous Model", "port": 8006, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "programming", "name": "Programming Model", "port": 8007, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "planning", "name": "Planning Model", "port": 8008, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "emotion", "name": "Emotion Model", "port": 8009, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "spatial", "name": "Spatial Model", "port": 8010, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "computer_vision", "name": "Computer Vision Model", "port": 8011, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "sensor", "name": "Sensor Model", "port": 8012, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "motion", "name": "Motion Model", "port": 8013, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "prediction", "name": "Prediction Model", "port": 8014, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "advanced_reasoning", "name": "Advanced Reasoning Model", "port": 8015, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "data_fusion", "name": "Data Fusion Model", "port": 8016, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "creative_problem_solving", "name": "Creative Problem Solving Model", "port": 8017, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "meta_cognition", "name": "Meta Cognition Model", "port": 8018, "status": "stopped", "last_active": datetime.now().isoformat()},
    {"id": "value_alignment", "name": "Value Alignment Model", "port": 8019, "status": "stopped", "last_active": datetime.now().isoformat()}
]

# Mock data for knowledge files
mock_files = [
    {"id": "1", "name": "system_architecture.pdf", "type": "pdf", "size": "2.5 MB", "last_modified": "2024-01-15T10:30:00"},
    {"id": "2", "name": "model_documentation.md", "type": "md", "size": "1.2 MB", "last_modified": "2024-01-14T15:45:00"},
    {"id": "3", "name": "training_dataset.csv", "type": "csv", "size": "15.8 MB", "last_modified": "2024-01-13T09:12:00"},
    {"id": "4", "name": "knowledge_graph.json", "type": "json", "size": "3.7 MB", "last_modified": "2024-01-12T14:20:00"},
    {"id": "5", "name": "user_manual.docx", "type": "docx", "size": "4.1 MB", "last_modified": "2024-01-11T11:05:00"}
]

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to quickly respond to frontend connection requests
    """
    return {"status": "healthy", "message": "FastAPI server is running"}

# Simple API endpoint
@app.get("/api")
async def api_root():
    """
    Root API endpoint
    """
    return {"status": "ok", "version": "1.0.0", "message": "API server is running"}

# Get models list endpoint
@app.get("/api/models")
async def get_models():
    """
    Get list of available models
    """
    return {"status": "success", "models": mock_models}

# Knowledge files endpoint
@app.get("/api/knowledge/files")
async def get_knowledge_files():
    """
    Get list of knowledge files
    """
    try:
        return {"status": "success", "files": mock_files}
    except Exception as e:
        return {"status": "error", "message": "Failed to get knowledge files"}

# Train endpoint
@app.post("/api/train")
async def train_model(request: Request):
    """
    Start model training
    """
    job_id = str(uuid.uuid4())[:8]
    return {"status": "success", "job_id": job_id, "message": "Training started successfully"}

# Joint training endpoint
@app.post("/api/joint-training/start")
async def start_joint_training(request: Request):
    """
    Start joint model training
    """
    job_id = str(uuid.uuid4())[:8]
    return {"status": "success", "job_id": job_id, "message": "Joint training started successfully"}

# Training status endpoint
@app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """
    Get training status
    """
    # Return mock status with progress
    progress = 30 + (int(time.time()) % 70)  # Random progress between 30-100%
    status = "completed" if progress == 100 else "running"
    
    return {
        "status": "success",
        "job_id": job_id,
        "training_status": status,
        "progress": progress,
        "metrics": {
            "accuracy": 0.85 + (int(time.time()) % 15) / 100,
            "loss": 0.15 - (int(time.time()) % 10) / 100
        }
    }

# Knowledge statistics endpoint - Fix for frontend [Failed to load statistics] error
@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """
    Get knowledge base statistics
    
    Returns:
        Knowledge base statistics including total domains, items, size, etc.
    """
    try:
        # Mock knowledge statistics data for frontend display
        knowledge_stats = {
            "total_domains": 5,
            "total_items": 125,
            "total_size": "15.8 MB",
            "updated_domains": 3,
            "recent_updates": 8,
            "domain_categories": [
                {"name": "System Architecture", "count": 25},
                {"name": "Model Documentation", "count": 30},
                {"name": "Training Data", "count": 40},
                {"name": "Knowledge Graph", "count": 15},
                {"name": "User Manual", "count": 15}
            ]
        }
        
        return {"status": "success", "stats": knowledge_stats}
    except Exception as e:
        # Return mock data even if there's an error to ensure frontend displays correctly
        return {
            "status": "success",
            "stats": {
                "total_domains": 5,
                "total_items": 125,
                "total_size": "15.8 MB",
                "updated_domains": 3,
                "recent_updates": 8,
                "domain_categories": [
                    {"name": "System Architecture", "count": 25},
                    {"name": "Model Documentation", "count": 30},
                    {"name": "Training Data", "count": 40},
                    {"name": "Knowledge Graph", "count": 15},
                    {"name": "User Manual", "count": 15}
                ]
            }
        }

# Knowledge search endpoint
@app.get("/api/knowledge/search")
async def search_knowledge(query: str = None, domain: str = None):
    """
    Search knowledge base with optional query and domain filters
    """
    try:
        # Mock search results
        search_results = [
            {"id": "1", "name": "system_architecture.pdf", "type": "pdf", "size": "2.5 MB", "last_modified": "2024-01-15T10:30:00", "domain": "System Architecture"},
            {"id": "2", "name": "model_documentation.md", "type": "md", "size": "1.2 MB", "last_modified": "2024-01-14T15:45:00", "domain": "Model Documentation"}
        ]
        return {"status": "success", "results": search_results, "total": len(search_results)}
    except Exception as e:
        return {"status": "error", "message": "Search failed"}

# Knowledge file preview endpoint
@app.get("/api/knowledge/files/{file_id}/preview")
async def preview_knowledge_file(file_id: str):
    """
    Get preview of a knowledge file
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
        return {"status": "error", "message": "Failed to get file preview"}

# Knowledge file download endpoint
@app.get("/api/knowledge/files/{file_id}/download")
async def download_knowledge_file(file_id: str):
    """
    Download a knowledge file
    """
    try:
        # Mock download URL (in real implementation, this would be a signed URL or file stream)
        download_url = f"/download/knowledge/{file_id}"
        return {"status": "success", "download_url": download_url, "message": "Download started"}
    except Exception as e:
        return {"status": "error", "message": "Failed to initiate download"}

# Knowledge file delete endpoint
@app.delete("/api/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: str):
    """
    Delete a knowledge file
    """
    try:
        return {"status": "success", "message": "File deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": "Failed to delete file"}

# System stats endpoint
@app.get("/api/system/stats")
async def get_system_stats():
    """
    Get system statistics
    """
    return {
        "status": "success",
        "stats": {
            "active_models": 3,
            "total_models": 19,
            "cpu_usage": 25.3,
            "memory_usage": 42.7,
            "disk_usage": 68.9,
            "uptime": "02:45:18"
        }
    }

# Model operation endpoints
# Start model
@app.post("/api/models/{model_id}/start")
async def start_model(model_id: str):
    """
    Start model service
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        model["status"] = "running"
        model["last_active"] = datetime.now().isoformat()
        return {"status": "success", "message": f"Model {model_id} started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Stop model
@app.post("/api/models/{model_id}/stop")
async def stop_model(model_id: str):
    """
    Stop model service
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        model["status"] = "stopped"
        return {"status": "success", "message": f"Model {model_id} stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Restart model
@app.post("/api/models/{model_id}/restart")
async def restart_model(model_id: str):
    """
    Restart model service
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        model["status"] = "stopped"
        time.sleep(0.5)  # Simulate restart delay
        model["status"] = "running"
        model["last_active"] = datetime.now().isoformat()
        return {"status": "success", "message": f"Model {model_id} restarted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Start all models
@app.post("/api/models/start-all")
async def start_all_models():
    """
    Start all models
    """
    try:
        for model in mock_models:
            model["status"] = "running"
            model["last_active"] = datetime.now().isoformat()
        return {"status": "success", "message": "All models started successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Stop all models
@app.post("/api/models/stop-all")
async def stop_all_models():
    """
    Stop all models
    """
    try:
        # Keep management model running
        for model in mock_models:
            if model["id"] != "manager":
                model["status"] = "stopped"
        return {"status": "success", "message": "All models stopped successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Restart all models
@app.post("/api/models/restart-all")
async def restart_all_models():
    """
    Restart all models
    """
    try:
        for model in mock_models:
            model["status"] = "stopped"
        time.sleep(0.5)  # Simulate restart delay
        for model in mock_models:
            model["status"] = "running"
            model["last_active"] = datetime.now().isoformat()
        return {"status": "success", "message": "All models restarted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Update model
@app.put("/api/models/{model_id}")
async def update_model(model_id: str, request: Request):
    """
    Update model configuration
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        model_data = await request.json()
        # Update model properties
        if "name" in model_data:
            model["name"] = model_data["name"]
        if "type" in model_data:
            model["type"] = model_data["type"]
        
        return {"status": "success", "message": f"Model {model_id} configuration updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Delete model
@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete model
    """
    try:
        # Cannot delete built-in models
        if model_id in ["manager", "language", "knowledge", "vision_image", "audio", "autonomous", "programming", "planning"]:
            return {"status": "error", "message": "Cannot delete built-in models"}
        
        global mock_models
        mock_models = [m for m in mock_models if m["id"] != model_id]
        return {"status": "success", "message": f"Model {model_id} deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Toggle model activation
@app.post("/api/models/{model_id}/activation")
async def toggle_model_activation(model_id: str, request: Request):
    """
    Toggle model activation status
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        data = await request.json()
        active = data.get("active", False)
        
        model["status"] = "running" if active else "stopped"
        if active:
            model["last_active"] = datetime.now().isoformat()
            
        action = "activated" if active else "deactivated"
        return {"status": "success", "message": f"Model {model_id} {action} successfully"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Set primary model
@app.post("/api/models/{model_id}/primary")
async def set_primary_model(model_id: str):
    """
    Set a model as primary for its type
    """
    try:
        model = next((m for m in mock_models if m["id"] == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
        
        model_type = model.get("type", "general")
        return {"status": "success", "message": f"Model {model_id} set as primary for type {model_type}"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# API mode switching endpoints
@app.post("/api/models/{model_id}/switch-to-external")
async def switch_to_external(model_id: str, request: Request):
    """
    Switch model to external API mode
    """
    try:
        return {"status": "success", "message": f"Model {model_id} switched to external API mode successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/{model_id}/switch-to-local")
async def switch_to_local(model_id: str):
    """
    Switch model to local mode
    """
    try:
        return {"status": "success", "message": f"Model {model_id} switched to local mode successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models/{model_id}/mode")
async def get_model_mode(model_id: str):
    """
    Get model running mode
    """
    try:
        return {"status": "success", "mode": "local"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/batch-switch-mode")
async def batch_switch_mode(request: Request):
    """
    Batch switch model modes
    """
    try:
        models_data = await request.json()
        return {"status": "success", "results": "Batch mode switch completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/test-connection")
async def test_connection(request: Request):
    """
    Test external API connection
    """
    try:
        # Simulate successful connection
        return {"status": "success", "message": "Connection test successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# System restart endpoint
@app.post("/api/system/restart")
async def restart_system():
    """
    Restart the entire system
    """
    try:
        return {"status": "success", "message": "System restart initiated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """
    Error handling middleware
    """
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "detail": str(e)}
        )

if __name__ == "__main__":
    print("Starting Self Soul API server with mock endpoints on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)