"""
Model Registry Client for CMCP
=============================

Service discovery and model registration for cross-model communication.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import aiohttp
import threading


@dataclass
class ModelRegistration:
    """Model registration data"""
    model_id: str
    model_type: str
    version: str
    capabilities: List[str] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)
    performance_characteristics: Dict[str, Any] = field(default_factory=dict)
    status: str = "ready"  # ready, busy, offline, maintenance
    last_heartbeat: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "capabilities": self.capabilities,
            "endpoints": self.endpoints,
            "performance_characteristics": self.performance_characteristics,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRegistration':
        """Create from dictionary"""
        return cls(**data)
    
    def is_available(self) -> bool:
        """Check if model is available for requests"""
        return self.status in ["ready", "busy"]
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        from datetime import datetime
        self.last_heartbeat = datetime.utcnow().isoformat() + "Z"
    
    def get_endpoint(self, protocol: str) -> Optional[str]:
        """Get endpoint for specific protocol"""
        return self.endpoints.get(protocol)


class ModelRegistryClient:
    """Client for model registry service discovery"""
    
    def __init__(self, registry_url: Optional[str] = None):
        self.registry_url = registry_url or "http://localhost:8080"
        self.logger = logging.getLogger(__name__)
        
        # Local cache
        self.model_cache: Dict[str, ModelRegistration] = {}
        self.cache_lock = threading.RLock()
        
        # Heartbeat tracking
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_task = None
        self.registered_models: Set[str] = set()
        
        # Health check
        self.health_check_interval = 60  # seconds
        self.health_check_task = None
        
        self.logger.info(f"ModelRegistryClient initialized with registry: {self.registry_url}")
    
    async def start(self):
        """Start registry client background tasks"""
        # Start heartbeat for registered models
        if self.registered_models:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start health check
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("ModelRegistryClient started")
    
    async def stop(self):
        """Stop registry client"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("ModelRegistryClient stopped")
    
    async def register_model(self, registration_data: Dict[str, Any]) -> bool:
        """Register a model with the registry"""
        try:
            registration = ModelRegistration.from_dict(registration_data)
            model_id = registration.model_id
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/api/v1/models/register",
                    json=registration.to_dict()
                ) as response:
                    if response.status == 200:
                        # Update cache
                        with self.cache_lock:
                            self.model_cache[model_id] = registration
                            self.registered_models.add(model_id)
                        
                        # Start heartbeat if not already running
                        if self.heartbeat_task is None or self.heartbeat_task.done():
                            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                        
                        self.logger.info(f"Registered model: {model_id}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to register model {model_id}: {error_text}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            return False
    
    async def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from the registry"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.registry_url}/api/v1/models/{model_id}"
                ) as response:
                    if response.status == 200:
                        # Remove from cache
                        with self.cache_lock:
                            if model_id in self.model_cache:
                                del self.model_cache[model_id]
                            self.registered_models.discard(model_id)
                        
                        self.logger.info(f"Unregistered model: {model_id}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to unregister model {model_id}: {error_text}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error unregistering model: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[ModelRegistration]:
        """Get model information from registry"""
        # Check cache first
        with self.cache_lock:
            if model_id in self.model_cache:
                return self.model_cache[model_id]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.registry_url}/api/v1/models/{model_id}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        registration = ModelRegistration.from_dict(data)
                        
                        # Update cache
                        with self.cache_lock:
                            self.model_cache[model_id] = registration
                        
                        return registration
                    else:
                        self.logger.warning(f"Model {model_id} not found in registry")
                        return None
        
        except Exception as e:
            self.logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    async def list_models(self, model_type: Optional[str] = None, 
                         status: Optional[str] = None) -> List[ModelRegistration]:
        """List all models with optional filtering"""
        try:
            params = {}
            if model_type:
                params['type'] = model_type
            if status:
                params['status'] = status
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.registry_url}/api/v1/models",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [ModelRegistration.from_dict(item) for item in data]
                        
                        # Update cache
                        with self.cache_lock:
                            for model in models:
                                self.model_cache[model.model_id] = model
                        
                        return models
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to list models: {error_text}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    async def discover_models_by_capability(self, capability: str) -> List[ModelRegistration]:
        """Discover models that have a specific capability"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.registry_url}/api/v1/models/capabilities/{capability}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [ModelRegistration.from_dict(item) for item in data]
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to discover models by capability: {error_text}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Error discovering models by capability: {e}")
            return []
    
    async def update_model_status(self, model_id: str, status: str, 
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update model status"""
        try:
            update_data = {"status": status}
            if metadata:
                update_data["metadata"] = metadata
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.registry_url}/api/v1/models/{model_id}/status",
                    json=update_data
                ) as response:
                    if response.status == 200:
                        # Update cache
                        with self.cache_lock:
                            if model_id in self.model_cache:
                                self.model_cache[model_id].status = status
                                if metadata:
                                    self.model_cache[model_id].metadata.update(metadata)
                        
                        self.logger.info(f"Updated status for model {model_id}: {status}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to update model status: {error_text}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Error updating model status: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """Send heartbeats for registered models"""
        while self.registered_models:
            try:
                for model_id in list(self.registered_models):
                    await self._send_heartbeat(model_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self, model_id: str):
        """Send heartbeat for a specific model"""
        try:
            with self.cache_lock:
                if model_id not in self.model_cache:
                    return
                
                registration = self.model_cache[model_id]
                registration.update_heartbeat()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.registry_url}/api/v1/models/{model_id}/heartbeat",
                    json={"timestamp": registration.last_heartbeat}
                ) as response:
                    if response.status == 200:
                        self.logger.debug(f"Heartbeat sent for model {model_id}")
                    else:
                        self.logger.warning(f"Heartbeat failed for model {model_id}")
        
        except Exception as e:
            self.logger.error(f"Error sending heartbeat for model {model_id}: {e}")
    
    async def _health_check_loop(self):
        """Periodic health check of cached models"""
        while True:
            try:
                await self._refresh_cache()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _refresh_cache(self):
        """Refresh model cache from registry"""
        try:
            models = await self.list_models()
            self.logger.debug(f"Refreshed cache with {len(models)} models")
        
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {e}")
    
    def get_cached_model(self, model_id: str) -> Optional[ModelRegistration]:
        """Get model from cache (no network call)"""
        with self.cache_lock:
            return self.model_cache.get(model_id)
    
    def get_all_cached_models(self) -> List[ModelRegistration]:
        """Get all cached models"""
        with self.cache_lock:
            return list(self.model_cache.values())
    
    def clear_cache(self):
        """Clear model cache"""
        with self.cache_lock:
            self.model_cache.clear()
            self.registered_models.clear()
        self.logger.info("Model cache cleared")


# Local registry for testing/development
class LocalModelRegistry:
    """In-memory model registry for testing"""
    
    def __init__(self):
        self.models: Dict[str, ModelRegistration] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register(self, registration: ModelRegistration) -> bool:
        """Register a model"""
        with self.lock:
            self.models[registration.model_id] = registration
            self.logger.info(f"Locally registered model: {registration.model_id}")
            return True
    
    def unregister(self, model_id: str) -> bool:
        """Unregister a model"""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                self.logger.info(f"Locally unregistered model: {model_id}")
                return True
            return False
    
    def get(self, model_id: str) -> Optional[ModelRegistration]:
        """Get a model"""
        with self.lock:
            return self.models.get(model_id)
    
    def list_all(self, model_type: Optional[str] = None, 
                status: Optional[str] = None) -> List[ModelRegistration]:
        """List all models with filtering"""
        with self.lock:
            result = list(self.models.values())
            
            if model_type:
                result = [m for m in result if m.model_type == model_type]
            
            if status:
                result = [m for m in result if m.status == status]
            
            return result
    
    def update_status(self, model_id: str, status: str) -> bool:
        """Update model status"""
        with self.lock:
            if model_id in self.models:
                self.models[model_id].status = status
                return True
            return False


# Global registry instance for simple usage
_local_registry = LocalModelRegistry()

def get_local_registry() -> LocalModelRegistry:
    """Get global local registry instance"""
    return _local_registry