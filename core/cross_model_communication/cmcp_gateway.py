"""
CMCP Gateway - Central Routing Component for Cross-Model Communication
=====================================================================

The CMCP Gateway is the central component that routes messages between models,
manages communication protocols, and provides monitoring and security features.
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import concurrent.futures
import uuid

from .message_format import (
    Message, RequestMessage, ResponseMessage, ErrorMessage,
    MessagePriority, MessageStatus, create_error_response
)
from .protocols import (
    CommunicationProtocol, ProtocolHandler,
    DirectCallHandler, RESTHandler, WebSocketHandler,
    MessageQueueHandler, SharedMemoryHandler
)
from .registry import ModelRegistryClient


@dataclass
class GatewayConfig:
    """Gateway configuration"""
    default_protocol: CommunicationProtocol = CommunicationProtocol.DIRECT_CALL
    protocol_configs: Dict[str, Dict[str, Any]] = None
    load_balancing_strategy: str = "round_robin"  # round_robin, weighted, least_connections
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # failures before opening circuit
    circuit_breaker_reset_timeout: int = 60  # seconds
    max_concurrent_requests: int = 100
    request_timeout_ms: int = 10000
    monitoring_enabled: bool = True
    security_enabled: bool = True
    
    def __post_init__(self):
        if self.protocol_configs is None:
            self.protocol_configs = {}


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, threshold: int = 5, reset_timeout: int = 60):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
    
    def record_success(self):
        """Record a successful operation"""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.last_failure_time = 0
            elif self.state == "CLOSED":
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed operation"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold and self.state == "CLOSED":
                self.state = "OPEN"
                logging.info(f"Circuit breaker opened after {self.failure_count} failures")
    
    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                # Check if reset timeout has passed
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF_OPEN"
                    logging.info("Circuit breaker moved to HALF_OPEN state")
                    return True
                return False
            elif self.state == "HALF_OPEN":
                return True
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
                "threshold": self.threshold,
                "reset_timeout": self.reset_timeout
            }


class CMCPGateway:
    """CMCP Gateway - Central routing component"""
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.protocol_handlers: Dict[CommunicationProtocol, ProtocolHandler] = {}
        self.model_registry = ModelRegistryClient()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Routing and state
        self.routing_table: Dict[str, Dict[str, Any]] = {}
        self.message_history: Dict[str, Dict[str, Any]] = {}
        self.active_requests: Dict[str, asyncio.Task] = {}
        
        # Performance monitoring
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "concurrent_requests": 0,
            "max_concurrent_requests": 0
        }
        
        # Security
        self.security_context = {}
        
        # Thread pool for blocking operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests
        )
        
        # Initialize
        self._initialize_protocols()
        self._initialize_monitoring()
        
        self.logger.info(f"CMCP Gateway initialized with default protocol: {self.config.default_protocol}")
    
    def _initialize_protocols(self):
        """Initialize all protocol handlers"""
        protocol_configs = self.config.protocol_configs
        
        # Direct Call Protocol
        direct_config = protocol_configs.get(CommunicationProtocol.DIRECT_CALL.value, {})
        self.protocol_handlers[CommunicationProtocol.DIRECT_CALL] = DirectCallHandler(direct_config)
        
        # REST Protocol
        rest_config = protocol_configs.get(CommunicationProtocol.REST.value, {})
        self.protocol_handlers[CommunicationProtocol.REST] = RESTHandler(rest_config)
        
        # WebSocket Protocol
        websocket_config = protocol_configs.get(CommunicationProtocol.WEBSOCKET.value, {})
        self.protocol_handlers[CommunicationProtocol.WEBSOCKET] = WebSocketHandler(websocket_config)
        
        # Message Queue Protocol
        mq_config = protocol_configs.get(CommunicationProtocol.MESSAGE_QUEUE.value, {})
        self.protocol_handlers[CommunicationProtocol.MESSAGE_QUEUE] = MessageQueueHandler(mq_config)
        
        # Shared Memory Protocol
        sm_config = protocol_configs.get(CommunicationProtocol.SHARED_MEMORY.value, {})
        self.protocol_handlers[CommunicationProtocol.SHARED_MEMORY] = SharedMemoryHandler(sm_config)
        
        self.logger.info(f"Initialized {len(self.protocol_handlers)} protocol handlers")
    
    def _initialize_monitoring(self):
        """Initialize monitoring system"""
        if self.config.monitoring_enabled:
            # Start background monitoring task
            asyncio.create_task(self._monitoring_loop())
            self.logger.info("Monitoring system initialized")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await self._collect_metrics()
                await self._check_system_health()
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        # Update protocol metrics
        for protocol, handler in self.protocol_handlers.items():
            if hasattr(handler, 'get_stats'):
                stats = handler.get_stats()
                self.logger.debug(f"Protocol {protocol} stats: {stats}")
        
        # Log gateway metrics periodically
        self.logger.info(
            f"Gateway metrics: "
            f"requests={self.metrics['requests_total']} "
            f"success={self.metrics['requests_successful']} "
            f"fail={self.metrics['requests_failed']} "
            f"avg_latency={self.metrics['average_latency_ms']:.2f}ms "
            f"concurrent={self.metrics['concurrent_requests']}"
        )
    
    async def _check_system_health(self):
        """Check system health status"""
        health_issues = []
        
        # Check protocol connections
        for protocol, handler in self.protocol_handlers.items():
            if not handler.is_connected():
                try:
                    await handler.connect()
                    if not handler.is_connected():
                        health_issues.append(f"Protocol {protocol} not connected")
                except Exception as e:
                    health_issues.append(f"Protocol {protocol} connection failed: {e}")
        
        # Check circuit breakers
        for target, cb in self.circuit_breakers.items():
            state = cb.get_state()
            if state['state'] == 'OPEN':
                health_issues.append(f"Circuit breaker OPEN for {target}")
        
        if health_issues:
            self.logger.warning(f"Health check issues: {health_issues}")
        else:
            self.logger.debug("System health check passed")
    
    async def start(self):
        """Start the gateway and all protocol handlers"""
        self.logger.info("Starting CMCP Gateway...")
        
        # Connect all protocol handlers
        for protocol, handler in self.protocol_handlers.items():
            try:
                await handler.connect()
                self.logger.info(f"Protocol {protocol} started")
            except Exception as e:
                self.logger.error(f"Failed to start protocol {protocol}: {e}")
        
        # Start receiving messages
        for protocol, handler in self.protocol_handlers.items():
            if protocol != CommunicationProtocol.DIRECT_CALL:
                try:
                    await handler.receive(self._handle_incoming_message)
                except Exception as e:
                    self.logger.error(f"Failed to start receive for protocol {protocol}: {e}")
        
        self.logger.info("CMCP Gateway started successfully")
    
    async def stop(self):
        """Stop the gateway and all protocol handlers"""
        self.logger.info("Stopping CMCP Gateway...")
        
        # Cancel active requests
        for message_id, task in self.active_requests.items():
            if not task.done():
                task.cancel()
        
        # Disconnect all protocol handlers
        for protocol, handler in self.protocol_handlers.items():
            try:
                await handler.disconnect()
                self.logger.info(f"Protocol {protocol} stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop protocol {protocol}: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("CMCP Gateway stopped")
    
    async def send_message(self, message: Message) -> Optional[ResponseMessage]:
        """Send a message through the gateway"""
        start_time = time.time()
        message_id = message.message_id
        
        # Update metrics
        self.metrics["requests_total"] += 1
        self.metrics["concurrent_requests"] += 1
        self.metrics["max_concurrent_requests"] = max(
            self.metrics["max_concurrent_requests"],
            self.metrics["concurrent_requests"]
        )
        
        try:
            # Validate message
            errors = message.validate()
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Check security if enabled
            if self.config.security_enabled:
                security_check = await self._check_security(message)
                if not security_check["allowed"]:
                    raise PermissionError(security_check.get("reason", "Security check failed"))
            
            # Store in history
            self.message_history[message_id] = {
                "message": message.to_dict(),
                "start_time": start_time,
                "status": "processing"
            }
            
            # Route message
            response = await self._route_message(message)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            self.metrics["average_latency_ms"] = (
                self.metrics["total_latency_ms"] / self.metrics["requests_total"]
            )
            
            # Update history
            if response:
                self.message_history[message_id].update({
                    "end_time": time.time(),
                    "latency_ms": latency_ms,
                    "status": "completed",
                    "response": response.to_dict() if response else None
                })
                
                if isinstance(response, ResponseMessage) and response.is_success():
                    self.metrics["requests_successful"] += 1
                else:
                    self.metrics["requests_failed"] += 1
            else:
                self.metrics["requests_failed"] += 1
            
            self.logger.debug(f"Message {message_id} processed in {latency_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            # Handle errors
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["requests_failed"] += 1
            
            # Update history
            if message_id in self.message_history:
                self.message_history[message_id].update({
                    "end_time": time.time(),
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": str(e)
                })
            
            self.logger.error(f"Failed to send message {message_id}: {e}")
            
            # Create error response
            if isinstance(message, RequestMessage) and message.metadata.requires_response:
                return create_error_response(
                    message,
                    "cmcp_gateway",
                    "INTERNAL_ERROR",
                    str(e),
                    retryable=True
                )
            
            return None
            
        finally:
            # Clean up
            self.metrics["concurrent_requests"] -= 1
            if message_id in self.active_requests:
                del self.active_requests[message_id]
    
    async def _route_message(self, message: Message) -> Optional[ResponseMessage]:
        """Route message to appropriate target(s)"""
        if not message.target_models:
            raise ValueError("No target models specified in message")
        
        # For now, handle single target (will extend to multiple later)
        target_model = message.target_models[0]
        
        # Get target model information
        model_info = await self.model_registry.get_model(target_model)
        if not model_info:
            raise ValueError(f"Model '{target_model}' not found in registry")
        
        # Check circuit breaker
        if self.config.circuit_breaker_enabled:
            cb = self._get_circuit_breaker(target_model)
            if not cb.allow_request():
                raise ConnectionError(f"Circuit breaker OPEN for model '{target_model}'")
        
        # Select protocol
        protocol = self._select_protocol(message, model_info)
        handler = self.protocol_handlers.get(protocol)
        
        if not handler:
            raise ValueError(f"Protocol handler not found for {protocol}")
        
        # Get endpoint
        endpoint = self._get_endpoint(protocol, model_info, message.operation)
        
        # Send message via protocol handler
        response = await handler.send(message, endpoint)
        
        # Update circuit breaker
        if self.config.circuit_breaker_enabled:
            cb = self._get_circuit_breaker(target_model)
            if response and isinstance(response, ResponseMessage) and response.is_success():
                cb.record_success()
            else:
                cb.record_failure()
        
        return response
    
    def _select_protocol(self, message: Message, model_info: Dict[str, Any]) -> CommunicationProtocol:
        """Select appropriate communication protocol"""
        # Check if model has preferred protocol
        if 'preferred_protocol' in model_info:
            preferred = model_info['preferred_protocol']
            try:
                protocol = CommunicationProtocol(preferred)
                if protocol in self.protocol_handlers:
                    return protocol
            except ValueError:
                pass
        
        # Check message priority for protocol selection
        if message.priority in [MessagePriority.HIGH.value, MessagePriority.CRITICAL.value]:
            # High priority messages use direct call if available
            if CommunicationProtocol.DIRECT_CALL in self.protocol_handlers:
                return CommunicationProtocol.DIRECT_CALL
        
        # Use default protocol
        return self.config.default_protocol
    
    def _get_endpoint(self, protocol: CommunicationProtocol, model_info: Dict[str, Any], 
                     operation: Optional[str] = None) -> str:
        """Get endpoint for specific protocol and operation"""
        # Check for protocol-specific endpoint
        endpoints = model_info.get('endpoints', {})
        
        if protocol.value in endpoints:
            endpoint = endpoints[protocol.value]
        else:
            # Fallback to model_id
            endpoint = model_info.get('model_id', 'unknown')
        
        # For direct call, include operation
        if protocol == CommunicationProtocol.DIRECT_CALL and operation:
            return f"{endpoint}::{operation}"
        
        return endpoint
    
    def _get_circuit_breaker(self, target: str) -> CircuitBreaker:
        """Get or create circuit breaker for target"""
        if target not in self.circuit_breakers:
            self.circuit_breakers[target] = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_reset_timeout
            )
        return self.circuit_breakers[target]
    
    async def _check_security(self, message: Message) -> Dict[str, Any]:
        """Check message security"""
        # Basic security checks
        checks = {
            "allowed": True,
            "reason": "",
            "checks_passed": []
        }
        
        # Check source model (if specified)
        if message.source_model:
            # Verify source model is registered
            source_info = await self.model_registry.get_model(message.source_model)
            if not source_info:
                checks["allowed"] = False
                checks["reason"] = f"Source model '{message.source_model}' not registered"
                return checks
            
            checks["checks_passed"].append("source_verified")
        
        # Check target models
        for target in message.target_models:
            target_info = await self.model_registry.get_model(target)
            if not target_info:
                checks["allowed"] = False
                checks["reason"] = f"Target model '{target}' not registered"
                return checks
            
            # Check if operation is allowed for this model
            # (In production, would check capabilities or permissions)
            
            checks["checks_passed"].append(f"target_verified_{target}")
        
        # Check message integrity
        # (In production, would verify signatures, etc.)
        
        checks["checks_passed"].append("integrity_check")
        
        return checks
    
    async def _handle_incoming_message(self, message: Message):
        """Handle incoming message from protocol handlers"""
        try:
            self.logger.info(f"Received incoming message: {message.message_id}")
            
            # Store in history
            self.message_history[message.message_id] = {
                "message": message.to_dict(),
                "received_time": time.time(),
                "status": "received"
            }
            
            # For now, just log incoming messages
            # In production, would route to appropriate handler
            self.logger.debug(f"Incoming message: {message.to_json(indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle incoming message: {e}")
    
    def register_direct_model(self, model_id: str, model_instance: Any):
        """Register a model for direct calls"""
        direct_handler = self.protocol_handlers.get(CommunicationProtocol.DIRECT_CALL)
        if direct_handler and isinstance(direct_handler, DirectCallHandler):
            direct_handler.register_model(model_id, model_instance)
            
            # Also register in model registry
            asyncio.create_task(self.model_registry.register_model({
                "model_id": model_id,
                "model_type": "direct",
                "endpoints": {
                    CommunicationProtocol.DIRECT_CALL.value: model_id
                }
            }))
            
            self.logger.info(f"Registered direct model: {model_id}")
        else:
            self.logger.warning("Direct call protocol not available")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current gateway metrics"""
        # Calculate additional metrics
        success_rate = 0.0
        if self.metrics["requests_total"] > 0:
            success_rate = self.metrics["requests_successful"] / self.metrics["requests_total"] * 100
        
        # Protocol metrics
        protocol_metrics = {}
        for protocol, handler in self.protocol_handlers.items():
            if hasattr(handler, 'get_stats'):
                protocol_metrics[protocol.value] = handler.get_stats()
        
        # Circuit breaker states
        cb_states = {}
        for target, cb in self.circuit_breakers.items():
            cb_states[target] = cb.get_state()
        
        return {
            "gateway_metrics": {
                **self.metrics,
                "success_rate_percent": success_rate,
                "message_history_size": len(self.message_history),
                "active_requests": len(self.active_requests),
                "circuit_breakers": len(self.circuit_breakers)
            },
            "protocol_metrics": protocol_metrics,
            "circuit_breaker_states": cb_states,
            "config": {
                "default_protocol": self.config.default_protocol.value,
                "load_balancing_strategy": self.config.load_balancing_strategy,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
                "max_concurrent_requests": self.config.max_concurrent_requests
            }
        }
    
    def get_message_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific message"""
        return self.message_history.get(message_id)
    
    def clear_message_history(self, older_than_seconds: Optional[int] = None):
        """Clear message history"""
        if older_than_seconds:
            cutoff_time = time.time() - older_than_seconds
            to_delete = []
            
            for msg_id, record in self.message_history.items():
                if record.get("start_time", 0) < cutoff_time:
                    to_delete.append(msg_id)
            
            for msg_id in to_delete:
                del self.message_history[msg_id]
            
            self.logger.info(f"Cleared {len(to_delete)} old messages from history")
        else:
            self.message_history.clear()
            self.logger.info("Cleared all message history")


# Singleton gateway instance
_gateway_instance: Optional[CMCPGateway] = None

def get_gateway(config: Optional[GatewayConfig] = None) -> CMCPGateway:
    """Get singleton CMCP gateway instance"""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = CMCPGateway(config)
    return _gateway_instance

async def start_gateway(config: Optional[GatewayConfig] = None):
    """Start the singleton gateway"""
    gateway = get_gateway(config)
    await gateway.start()
    return gateway

async def stop_gateway():
    """Stop the singleton gateway"""
    global _gateway_instance
    if _gateway_instance:
        await _gateway_instance.stop()
        _gateway_instance = None