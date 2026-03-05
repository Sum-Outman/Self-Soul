import threading
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json

@dataclass
class ServiceInstance:
    service_id: str
    service_name: str
    host: str
    port: int
    capabilities: List[str]
    last_heartbeat: datetime
    health_status: str
    metadata: Dict[str, str]
    load: float = 0.0
    response_time_ms: float = 0.0

class ServiceRegistry:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, heartbeat_timeout: int = 30):
        if self._initialized:
            return
        self._initialized = True
        self.services: Dict[str, ServiceInstance] = {}
        self.lock = threading.RLock()
        self.heartbeat_timeout = heartbeat_timeout
        self.logger = logging.getLogger(__name__)
        self._start_cleanup_thread()
    
    def register(
        self,
        service_name: str,
        host: str,
        port: int,
        capabilities: List[str] = None,
        metadata: Dict[str, str] = None
    ) -> str:
        service_id = self._generate_service_id(service_name, host, port)
        
        service = ServiceInstance(
            service_id=service_id,
            service_name=service_name,
            host=host,
            port=port,
            capabilities=capabilities or [],
            last_heartbeat=datetime.now(),
            health_status="healthy",
            metadata=metadata or {}
        )
        
        with self.lock:
            self.services[service_id] = service
            self.logger.info(f"服务注册成功: {service_name}@{host}:{port} (ID: {service_id})")
        
        return service_id
    
    def deregister(self, service_id: str) -> bool:
        with self.lock:
            if service_id in self.services:
                service = self.services[service_id]
                del self.services[service_id]
                self.logger.info(f"服务注销: {service.service_name} (ID: {service_id})")
                return True
            return False
    
    def discover(
        self,
        service_name: str,
        capability: Optional[str] = None,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        with self.lock:
            results = []
            for service in self.services.values():
                if service.service_name != service_name:
                    continue
                if capability and capability not in service.capabilities:
                    continue
                if healthy_only and not self._is_healthy(service):
                    continue
                results.append(service)
            return results
    
    def discover_one(
        self,
        service_name: str,
        capability: Optional[str] = None,
        strategy: str = "round_robin"
    ) -> Optional[ServiceInstance]:
        services = self.discover(service_name, capability)
        
        if not services:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_select(services)
        elif strategy == "least_load":
            return self._least_load_select(services)
        elif strategy == "fastest_response":
            return self._fastest_response_select(services)
        else:
            return services[0]
    
    def heartbeat(self, service_id: str, load: float = None, response_time_ms: float = None) -> bool:
        with self.lock:
            if service_id in self.services:
                service = self.services[service_id]
                service.last_heartbeat = datetime.now()
                if load is not None:
                    service.load = load
                if response_time_ms is not None:
                    service.response_time_ms = response_time_ms
                return True
            return False
    
    def update_health_status(self, service_id: str, status: str) -> bool:
        with self.lock:
            if service_id in self.services:
                self.services[service_id].health_status = status
                return True
            return False
    
    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        with self.lock:
            return self.services.get(service_id)
    
    def get_all_services(self) -> List[ServiceInstance]:
        with self.lock:
            return list(self.services.values())
    
    def get_service_count(self, service_name: str = None) -> int:
        with self.lock:
            if service_name:
                return sum(1 for s in self.services.values() if s.service_name == service_name)
            return len(self.services)
    
    def _is_healthy(self, service: ServiceInstance) -> bool:
        timeout = datetime.now() - timedelta(seconds=self.heartbeat_timeout)
        return (
            service.last_heartbeat > timeout and 
            service.health_status == "healthy"
        )
    
    def _generate_service_id(self, service_name: str, host: str, port: int) -> str:
        unique_str = f"{service_name}:{host}:{port}:{time.time()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def _round_robin_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        return services[hash(str(time.time())) % len(services)]
    
    def _least_load_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        return min(services, key=lambda s: s.load)
    
    def _fastest_response_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        return min(services, key=lambda s: s.response_time_ms)
    
    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                try:
                    self.cleanup_stale_services()
                    time.sleep(self.heartbeat_timeout // 2)
                except Exception as e:
                    self.logger.error(f"清理线程错误: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def cleanup_stale_services(self):
        with self.lock:
            stale_ids = [
                sid for sid, service in self.services.items()
                if not self._is_healthy(service)
            ]
            for sid in stale_ids:
                service = self.services[sid]
                del self.services[sid]
                self.logger.warning(f"清理过期服务: {service.service_name} (ID: {sid})")
    
    def get_registry_status(self) -> Dict:
        with self.lock:
            healthy_count = sum(1 for s in self.services.values() if self._is_healthy(s))
            return {
                "total_services": len(self.services),
                "healthy_services": healthy_count,
                "unhealthy_services": len(self.services) - healthy_count,
                "services_by_name": self._count_by_name(),
                "heartbeat_timeout": self.heartbeat_timeout
            }
    
    def _count_by_name(self) -> Dict[str, int]:
        counts = {}
        for service in self.services.values():
            counts[service.service_name] = counts.get(service.service_name, 0) + 1
        return counts

class ServiceClient:
    def __init__(self, registry: ServiceRegistry = None):
        self.registry = registry or ServiceRegistry()
        self.logger = logging.getLogger(__name__)
    
    def call_service(
        self,
        service_name: str,
        endpoint: str,
        data: Dict = None,
        capability: str = None,
        timeout: float = 30.0
    ) -> Dict:
        import requests
        
        service = self.registry.discover_one(service_name, capability)
        
        if not service:
            raise Exception(f"未找到可用服务: {service_name}")
        
        url = f"http://{service.host}:{service.port}/{endpoint}"
        
        start_time = time.time()
        try:
            response = requests.post(url, json=data, timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            
            self.registry.heartbeat(service.service_id, response_time_ms=response_time)
            
            return response.json()
        except Exception as e:
            self.registry.update_health_status(service.service_id, "unhealthy")
            raise

service_registry = ServiceRegistry()