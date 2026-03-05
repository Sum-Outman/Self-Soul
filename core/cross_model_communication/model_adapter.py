"""
Model Communication Adapter for CMCP
===================================

Adapter pattern for integrating models with the cross-model communication protocol.
Provides a standardized interface for models to send and receive messages.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type
from abc import ABC, abstractmethod
import inspect
import functools

from .message_format import (
    Message, RequestMessage, ResponseMessage, ErrorMessage,
    MessagePriority, create_request, create_response, create_error_response
)
from .cmcp_gateway import get_gateway
from .registry import ModelRegistration, get_local_registry


class ModelCommunicationMixin:
    """
    Mixin class to add CMCP communication capabilities to models.
    
    Usage:
    ```python
    class MyModel(UnifiedModelTemplate, ModelCommunicationMixin):
        def __init__(self, config):
            super().__init__(config)
            self.init_communication()
        
        @communication_handler(operation="process_data")
        async def handle_process_data(self, message: RequestMessage) -> ResponseMessage:
            # Process the request
            result = self.process_data(message.data)
            return create_response(message, self.model_id, result)
    ```
    """
    
    def init_communication(self, adapter_config: Optional[Dict[str, Any]] = None):
        """Initialize communication capabilities"""
        self._cmcp_adapter = ModelCommunicationAdapter(self, adapter_config or {})
        self._cmcp_adapter.start()
    
    async def send_to_model(self, target_model: str, operation: str, data: Dict[str, Any],
                           priority: str = MessagePriority.MEDIUM.value,
                           timeout_ms: int = 5000,
                           requires_response: bool = True) -> Optional[ResponseMessage]:
        """Send a message to another model"""
        return await self._cmcp_adapter.send_message(
            target_model=target_model,
            operation=operation,
            data=data,
            priority=priority,
            timeout_ms=timeout_ms,
            requires_response=requires_response
        )
    
    def register_handler(self, operation: str, handler_func: Callable):
        """Register a handler for a specific operation"""
        self._cmcp_adapter.register_handler(operation, handler_func)
    
    def get_adapter(self):
        """Get the communication adapter"""
        return self._cmcp_adapter


def communication_handler(operation: Optional[str] = None, 
                         priority: str = MessagePriority.MEDIUM.value):
    """
    Decorator for marking model methods as communication handlers.
    
    Usage:
    ```python
    @communication_handler(operation="process_text")
    async def handle_text_processing(self, message: RequestMessage) -> ResponseMessage:
        # Handle the request
        pass
    ```
    """
    def decorator(func):
        # Store metadata on the function
        func._cmcp_handler = True
        func._cmcp_operation = operation or func.__name__
        func._cmcp_priority = priority
        
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check if first argument is a Message
            if args and isinstance(args[0], (Message, dict)):
                # It's being called as a handler
                return await func(self, *args, **kwargs)
            else:
                # It's being called as a normal method
                return await func(self, *args, **kwargs)
        
        return wrapper
    
    return decorator


class ModelCommunicationAdapter:
    """
    Adapter for integrating a model with CMCP.
    
    This adapter handles:
    1. Message serialization/deserialization
    2. Protocol selection and routing
    3. Error handling and retries
    4. Performance monitoring
    5. Security enforcement
    """
    
    def __init__(self, model_instance: Any, config: Dict[str, Any]):
        self.model = model_instance
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model identification
        self.model_id = getattr(model_instance, 'model_id', 
                               config.get('model_id', f"model_{id(model_instance)}"))
        self.model_type = getattr(model_instance, 'model_type', 
                                 config.get('model_type', 'unknown'))
        
        # Communication state
        self.gateway = None
        self.handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.processing_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Performance metrics
        self.metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "average_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        }
        
        # Auto-discover handlers
        self._discover_handlers()
        
        self.logger.info(f"ModelCommunicationAdapter initialized for {self.model_id}")
    
    def _discover_handlers(self):
        """Auto-discover communication handlers on the model"""
        for name, method in inspect.getmembers(self.model, callable):
            # Check if method has CMCP handler metadata
            if hasattr(method, '_cmcp_handler'):
                operation = getattr(method, '_cmcp_operation', name)
                self.register_handler(operation, method)
                self.logger.debug(f"Discovered handler for operation '{operation}'")
    
    async def start(self):
        """Start the adapter"""
        if self.running:
            return
        
        self.running = True
        
        # Get gateway
        self.gateway = get_gateway()
        
        # Register model for direct calls
        self.gateway.register_direct_model(self.model_id, self.model)
        
        # Register with local registry
        await self._register_with_registry()
        
        # Start message processing
        self.processing_tasks = [
            asyncio.create_task(self._process_messages())
            for _ in range(self.config.get('concurrent_workers', 3))
        ]
        
        self.logger.info(f"ModelCommunicationAdapter started for {self.model_id}")
    
    async def stop(self):
        """Stop the adapter"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Unregister from registry
        await self._unregister_from_registry()
        
        self.logger.info(f"ModelCommunicationAdapter stopped for {self.model_id}")
    
    async def _register_with_registry(self):
        """Register model with service registry"""
        try:
            registration = ModelRegistration(
                model_id=self.model_id,
                model_type=self.model_type,
                version=getattr(self.model, 'version', '1.0.0'),
                capabilities=self._get_capabilities(),
                endpoints={
                    "direct_call": self.model_id,
                    # Add other endpoints as configured
                },
                performance_characteristics=self._get_performance_characteristics(),
                status="ready",
                metadata={
                    "adapter_started": time.time(),
                    "handler_count": len(self.handlers)
                }
            )
            
            # Register with local registry
            local_registry = get_local_registry()
            local_registry.register(registration)
            
            # TODO: Register with remote registry if configured
            
            self.logger.info(f"Registered model {self.model_id} with registry")
            
        except Exception as e:
            self.logger.error(f"Failed to register with registry: {e}")
    
    async def _unregister_from_registry(self):
        """Unregister model from service registry"""
        try:
            local_registry = get_local_registry()
            local_registry.unregister(self.model_id)
            self.logger.info(f"Unregistered model {self.model_id} from registry")
        except Exception as e:
            self.logger.error(f"Failed to unregister from registry: {e}")
    
    def _get_capabilities(self) -> List[str]:
        """Get model capabilities from handlers"""
        capabilities = []
        
        # Add handler operations as capabilities
        capabilities.extend(list(self.handlers.keys()))
        
        # Add model-specific capabilities
        if hasattr(self.model, 'get_capabilities'):
            try:
                model_caps = self.model.get_capabilities()
                if isinstance(model_caps, list):
                    capabilities.extend(model_caps)
            except:
                pass
        
        return list(set(capabilities))  # Remove duplicates
    
    def _get_performance_characteristics(self) -> Dict[str, Any]:
        """Get model performance characteristics"""
        characteristics = {
            "max_concurrent_requests": self.config.get('max_concurrent_requests', 10),
            "estimated_processing_time_ms": self.config.get('estimated_processing_time_ms', 100),
            "supported_formats": ["json"],
            "resource_requirements": {
                "memory_mb": self.config.get('memory_requirement_mb', 512),
                "cpu_cores": self.config.get('cpu_cores', 1)
            }
        }
        
        # Add model-specific characteristics
        if hasattr(self.model, 'get_performance_characteristics'):
            try:
                model_chars = self.model.get_performance_characteristics()
                if isinstance(model_chars, dict):
                    characteristics.update(model_chars)
            except:
                pass
        
        return characteristics
    
    def register_handler(self, operation: str, handler_func: Callable):
        """Register a handler for a specific operation"""
        # Validate handler signature
        sig = inspect.signature(handler_func)
        params = list(sig.parameters.values())
        
        if len(params) < 2 or params[1].name != 'message':
            self.logger.warning(
                f"Handler '{operation}' should accept 'message' as second parameter"
            )
        
        self.handlers[operation] = handler_func
        self.logger.info(f"Registered handler for operation '{operation}'")
    
    async def send_message(self, target_model: str, operation: str, data: Dict[str, Any],
                          priority: str = MessagePriority.MEDIUM.value,
                          timeout_ms: int = 5000,
                          requires_response: bool = True) -> Optional[ResponseMessage]:
        """Send a message to another model"""
        try:
            if not self.gateway:
                raise RuntimeError("Adapter not started")
            
            # Create request message
            request = create_request(
                source_model=self.model_id,
                target_models=[target_model],
                operation=operation,
                data=data,
                priority=priority,
                timeout_ms=timeout_ms,
                requires_response=requires_response
            )
            
            # Send via gateway
            response = await self.gateway.send_message(request)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_model}: {e}")
            
            if requires_response:
                # Create error response
                return create_error_response(
                    request,
                    self.model_id,
                    "COMMUNICATION_ERROR",
                    str(e),
                    retryable=True
                )
            
            return None
    
    async def receive_message(self, message: Message):
        """Receive and process an incoming message"""
        try:
            # Put message in queue for processing
            await self.message_queue.put(message)
            self.metrics["messages_received"] += 1
            
            self.logger.debug(f"Received message {message.message_id} for operation '{message.operation}'")
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            self.metrics["messages_failed"] += 1
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get next message
                message = await self.message_queue.get()
                
                # Process message
                await self._process_single_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                self.metrics["messages_failed"] += 1
    
    async def _process_single_message(self, message: Message):
        """Process a single message"""
        start_time = time.time()
        
        try:
            # Validate message
            if not isinstance(message, RequestMessage):
                self.logger.warning(f"Received non-request message: {type(message)}")
                return
            
            # Check if operation is supported
            operation = message.operation
            if operation not in self.handlers:
                error_msg = f"Operation '{operation}' not supported"
                self.logger.warning(error_msg)
                
                if message.metadata.requires_response:
                    error_response = create_error_response(
                        message,
                        self.model_id,
                        "OPERATION_NOT_SUPPORTED",
                        error_msg,
                        retryable=False
                    )
                    # Send error response
                    await self._send_response(error_response)
                
                return
            
            # Get handler
            handler = self.handlers[operation]
            
            # Prepare context
            context = {
                "message_id": message.message_id,
                "source_model": message.source_model,
                "priority": message.priority,
                "timestamp": message.timestamp,
                "adapter": self
            }
            
            # Call handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(self.model, message, context)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: handler(self.model, message, context)
                )
            
            # Send response if required
            if message.metadata.requires_response:
                if isinstance(result, ResponseMessage):
                    response = result
                elif isinstance(result, dict):
                    response = create_response(message, self.model_id, result)
                else:
                    response = create_response(
                        message, 
                        self.model_id, 
                        {"result": result}
                    )
                
                await self._send_response(response)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics["messages_processed"] += 1
            self.metrics["total_processing_time_ms"] += processing_time_ms
            self.metrics["average_processing_time_ms"] = (
                self.metrics["total_processing_time_ms"] / self.metrics["messages_processed"]
            )
            
            self.logger.debug(
                f"Processed message {message.message_id} in {processing_time_ms:.2f}ms"
            )
            
        except Exception as e:
            # Handle processing error
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics["messages_failed"] += 1
            self.logger.error(f"Failed to process message {message.message_id}: {e}")
            
            # Send error response if required
            if isinstance(message, RequestMessage) and message.metadata.requires_response:
                error_response = create_error_response(
                    message,
                    self.model_id,
                    "PROCESSING_ERROR",
                    str(e),
                    retryable=False
                )
                await self._send_response(error_response)
    
    async def _send_response(self, response: ResponseMessage):
        """Send a response message"""
        try:
            # For now, use the gateway to send response back
            # In production, might use different routing logic
            if self.gateway and response.original_source:
                # Create a message targeting the original source
                response.target_models = [response.original_source]
                await self.gateway.send_message(response)
                
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics"""
        queue_size = self.message_queue.qsize() if hasattr(self.message_queue, 'qsize') else 0
        
        return {
            **self.metrics,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "handler_count": len(self.handlers),
            "queue_size": queue_size,
            "running": self.running,
            "processing_tasks": len(self.processing_tasks)
        }
    
    def get_handler_operations(self) -> List[str]:
        """Get list of supported operations"""
        return list(self.handlers.keys())
    
    def has_handler(self, operation: str) -> bool:
        """Check if operation is supported"""
        return operation in self.handlers


class SimpleModelAdapter(ModelCommunicationAdapter):
    """
    Simplified adapter for models that don't need full CMCP integration.
    
    This adapter provides basic communication capabilities without
    requiring complex setup or external dependencies.
    """
    
    def __init__(self, model_instance: Any, config: Dict[str, Any] = None):
        super().__init__(model_instance, config or {})
        
        # Simplified configuration
        self.simple_mode = config.get('simple_mode', True)
        self.auto_register = config.get('auto_register', False)
    
    async def start(self):
        """Start simplified adapter"""
        if self.simple_mode:
            # Simple mode: just register handlers
            self.running = True
            self.logger.info(f"SimpleModelAdapter started for {self.model_id}")
        else:
            # Full mode: use parent implementation
            await super().start()
    
    async def send_message(self, target_model: str, operation: str, data: Dict[str, Any],
                          priority: str = MessagePriority.MEDIUM.value,
                          timeout_ms: int = 5000,
                          requires_response: bool = True) -> Optional[ResponseMessage]:
        """Send message in simple mode"""
        if self.simple_mode:
            # In simple mode, use direct gateway if available
            try:
                gateway = get_gateway()
                if gateway:
                    return await super().send_message(
                        target_model, operation, data, priority, timeout_ms, requires_response
                    )
            except:
                pass
            
            # Fallback: log and return mock response
            self.logger.info(f"Simple mode: would send to {target_model}.{operation}")
            
            if requires_response:
                # Create mock response
                from .message_format import ResponseMessage, MessageStatus
                return ResponseMessage(
                    source_model=target_model,
                    original_source=self.model_id,
                    operation=operation,
                    data={"mock_response": True, **data},
                    status=MessageStatus.SUCCESS.value
                )
            
            return None
        else:
            # Use parent implementation
            return await super().send_message(
                target_model, operation, data, priority, timeout_ms, requires_response
            )


def create_adapter_for_model(model_instance: Any, 
                           config: Optional[Dict[str, Any]] = None) -> ModelCommunicationAdapter:
    """
    Factory function to create appropriate adapter for a model.
    
    Automatically detects model type and creates appropriate adapter.
    """
    # Check if model has simple mode flag
    if hasattr(model_instance, '_cmcp_simple_mode') and model_instance._cmcp_simple_mode:
        return SimpleModelAdapter(model_instance, config or {})
    
    # Check model capabilities
    if hasattr(model_instance, 'get_capabilities'):
        try:
            capabilities = model_instance.get_capabilities()
            if isinstance(capabilities, list) and 'simple_communication' in capabilities:
                return SimpleModelAdapter(model_instance, config or {})
        except:
            pass
    
    # Default to full adapter
    return ModelCommunicationAdapter(model_instance, config or {})