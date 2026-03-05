"""
Communication Protocol Handlers for CMCP
=======================================

Implements various communication protocols for cross-model communication.
"""

import asyncio
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import threading
import concurrent.futures

from .message_format import Message, RequestMessage, ResponseMessage


class CommunicationProtocol(Enum):
    """Supported communication protocols"""
    DIRECT_CALL = "direct_call"
    REST = "rest"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    SHARED_MEMORY = "shared_memory"


class ProtocolHandler(ABC):
    """Base class for all protocol handlers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.protocol_name = self.__class__.__name__.replace('Handler', '').upper()
        self.connected = False
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'total_latency_ms': 0.0
        }
    
    @abstractmethod
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send a message using this protocol"""
        pass
    
    @abstractmethod
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """Start receiving messages"""
        pass
    
    @abstractmethod
    async def connect(self):
        """Establish connection for this protocol"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection for this protocol"""
        pass
    
    def is_connected(self) -> bool:
        """Check if protocol is connected"""
        return self.connected
    
    def update_stats(self, sent: bool = False, received: bool = False, 
                    error: bool = False, latency_ms: float = 0.0):
        """Update protocol statistics"""
        if sent:
            self.stats['messages_sent'] += 1
            self.stats['total_latency_ms'] += latency_ms
        if received:
            self.stats['messages_received'] += 1
        if error:
            self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        avg_latency = 0.0
        if self.stats['messages_sent'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['messages_sent']
        
        return {
            **self.stats,
            'average_latency_ms': avg_latency,
            'protocol': self.protocol_name,
            'connected': self.connected
        }
    
    def _validate_message(self, message: Message) -> List[str]:
        """Validate message before sending"""
        errors = message.validate()
        if errors:
            self.logger.warning(f"Message validation errors: {errors}")
        return errors


class DirectCallHandler(ProtocolHandler):
    """Direct call protocol handler - Python method calls within same process"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protocol_name = "DIRECT_CALL"
        self.model_registry = {}  # model_id -> model_instance
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get('max_workers', 10)
        )
    
    async def connect(self):
        """Direct call protocol is always connected"""
        self.connected = True
        self.logger.info("Direct call protocol connected")
    
    async def disconnect(self):
        """Disconnect direct call protocol"""
        self.connected = False
        self.executor.shutdown(wait=True)
        self.logger.info("Direct call protocol disconnected")
    
    def register_model(self, model_id: str, model_instance: Any):
        """Register a model for direct calls"""
        self.model_registry[model_id] = model_instance
        self.logger.info(f"Registered model '{model_id}' for direct calls")
    
    def unregister_model(self, model_id: str):
        """Unregister a model"""
        if model_id in self.model_registry:
            del self.model_registry[model_id]
            self.logger.info(f"Unregistered model '{model_id}' from direct calls")
    
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send message via direct method call"""
        start_time = time.time()
        
        try:
            # Validate message
            errors = self._validate_message(message)
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Parse target endpoint (format: model_id::operation)
            if '::' in target_endpoint:
                target_model, operation = target_endpoint.split('::', 1)
            else:
                target_model = target_endpoint
                operation = message.operation
            
            # Check if model is registered
            if target_model not in self.model_registry:
                raise ValueError(f"Model '{target_model}' not registered for direct calls")
            
            model_instance = self.model_registry[target_model]
            
            # Check if model has the operation method
            if not hasattr(model_instance, operation):
                raise AttributeError(f"Model '{target_model}' has no method '{operation}'")
            
            # Prepare method call
            method = getattr(model_instance, operation)
            data = message.data or {}
            
            # Execute method (async or sync)
            if asyncio.iscoroutinefunction(method):
                # Async method
                result = await method(**data)
            else:
                # Sync method - run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: method(**data)
                )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = ResponseMessage(
                response_to=message.message_id,
                source_model=target_model,
                original_source=message.source_model,
                operation=message.operation,
                data=result if isinstance(result, dict) else {'result': result},
                status='success',
                metadata={'requires_response': False}
            )
            
            # Update statistics
            self.update_stats(sent=True, latency_ms=latency_ms)
            
            self.logger.debug(f"Direct call to '{target_model}.{operation}' completed in {latency_ms:.2f}ms")
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            self.logger.error(f"Direct call failed: {e}")
            
            # Create error response
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "MODEL_ERROR",
                str(e),
                retryable=False
            )
    
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """Direct call protocol doesn't need explicit receive setup"""
        self.logger.info("Direct call receive handler ready")
        # In direct call protocol, messages are handled synchronously in send()


class RESTHandler(ProtocolHandler):
    """REST API protocol handler"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protocol_name = "REST"
        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.timeout = config.get('timeout', 10.0)
        self.session = None
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # Try to import aiohttp
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            self.aiohttp = None
            self.logger.warning("aiohttp not installed, REST protocol will use fallback")
    
    async def connect(self):
        """Establish REST connection (create session)"""
        try:
            if self.aiohttp:
                self.session = self.aiohttp.ClientSession(
                    timeout=self.aiohttp.ClientTimeout(total=self.timeout)
                )
                self.connected = True
                self.logger.info(f"REST protocol connected to {self.base_url}")
            else:
                # Fallback to requests (sync)
                self.connected = True
                self.logger.warning("REST protocol using fallback mode (no aiohttp)")
        except Exception as e:
            self.logger.error(f"Failed to connect REST protocol: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Close REST session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        self.logger.info("REST protocol disconnected")
    
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send message via REST API"""
        start_time = time.time()
        
        try:
            # Validate message
            errors = self._validate_message(message)
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Build URL
            if target_endpoint.startswith('http'):
                url = target_endpoint
            else:
                url = f"{self.base_url}/{target_endpoint.lstrip('/')}"
            
            # Prepare request
            headers = {
                'Content-Type': 'application/json',
                'X-CMCP-Version': message.protocol_version,
                'X-Message-ID': message.message_id,
                'X-Source-Model': message.source_model or 'unknown'
            }
            
            payload = message.to_json()
            
            # Send request with retries
            last_error = None
            for attempt in range(self.retry_attempts):
                try:
                    if self.aiohttp and self.session:
                        # Async request with aiohttp
                        async with self.session.post(url, data=payload, headers=headers) as response:
                            response_text = await response.text()
                            status_code = response.status
                    else:
                        # Fallback sync request
                        import requests
                        response = requests.post(url, data=payload, headers=headers, timeout=self.timeout)
                        response_text = response.text
                        status_code = response.status_code
                    
                    # Parse response
                    if status_code == 200:
                        response_data = json.loads(response_text)
                        
                        # Calculate latency
                        latency_ms = (time.time() - start_time) * 1000
                        
                        # Update statistics
                        self.update_stats(sent=True, latency_ms=latency_ms)
                        
                        self.logger.debug(f"REST request to {url} completed in {latency_ms:.2f}ms")
                        
                        # Convert to ResponseMessage
                        from .message_format import ResponseMessage
                        return ResponseMessage.from_dict(response_data)
                    
                    elif status_code >= 500:
                        # Server error - retryable
                        last_error = f"Server error {status_code}: {response_text}"
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                    
                    else:
                        # Client error - not retryable
                        last_error = f"Client error {status_code}: {response_text}"
                        break
                        
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        break
            
            # All retries failed or non-retryable error
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            
            self.logger.error(f"REST request failed after {self.retry_attempts} attempts: {last_error}")
            
            # Create error response
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "CONNECTION_ERROR",
                last_error or "Unknown REST error",
                retryable=True
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            self.logger.error(f"REST send failed: {e}")
            
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "INTERNAL_ERROR",
                str(e),
                retryable=False
            )
    
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """REST protocol is request-response, no persistent receive needed"""
        self.logger.info("REST receive handler ready (request-response mode)")


class WebSocketHandler(ProtocolHandler):
    """WebSocket protocol handler"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protocol_name = "WEBSOCKET"
        self.server_url = config.get('server_url', 'ws://localhost:8000/ws')
        self.reconnect_delay = config.get('reconnect_delay', 1.0)
        self.websocket = None
        self.receive_callback = None
        self.receive_task = None
        self.running = False
        
        # Try to import websockets
        try:
            import websockets
            self.websockets = websockets
        except ImportError:
            self.websockets = None
            self.logger.warning("websockets not installed, WebSocket protocol disabled")
    
    async def connect(self):
        """Establish WebSocket connection"""
        if not self.websockets:
            self.logger.error("WebSocket protocol requires 'websockets' package")
            self.connected = False
            return
        
        try:
            self.websocket = await self.websockets.connect(self.server_url)
            self.connected = True
            self.logger.info(f"WebSocket connected to {self.server_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        self.logger.info("WebSocket disconnected")
    
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send message via WebSocket"""
        start_time = time.time()
        
        try:
            # Validate message
            errors = self._validate_message(message)
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Ensure connection
            if not self.connected or not self.websocket:
                await self.connect()
                if not self.connected:
                    raise ConnectionError("WebSocket not connected")
            
            # Prepare message with target endpoint
            message_dict = message.to_dict()
            message_dict['_target_endpoint'] = target_endpoint
            
            # Send message
            await self.websocket.send(json.dumps(message_dict))
            
            # Wait for response if needed
            if message.metadata.requires_response:
                # For WebSocket, we need to match response to request
                # This simple implementation waits for next message
                # In production, would use correlation IDs
                response_text = await self.websocket.recv()
                response_data = json.loads(response_text)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Update statistics
                self.update_stats(sent=True, latency_ms=latency_ms)
                
                # Convert to ResponseMessage
                from .message_format import ResponseMessage
                return ResponseMessage.from_dict(response_data)
            else:
                # Fire-and-forget
                latency_ms = (time.time() - start_time) * 1000
                self.update_stats(sent=True, latency_ms=latency_ms)
                return None
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            self.logger.error(f"WebSocket send failed: {e}")
            
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "CONNECTION_ERROR",
                str(e),
                retryable=True
            )
    
    async def _receive_loop(self):
        """WebSocket receive loop"""
        while self.running and self.connected and self.websocket:
            try:
                message_text = await self.websocket.recv()
                message_data = json.loads(message_text)
                
                # Convert to Message
                from .message_format import Message
                message = Message.from_dict(message_data)
                
                # Update statistics
                self.update_stats(received=True)
                
                # Call callback
                if self.receive_callback:
                    self.receive_callback(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"WebSocket receive error: {e}")
                self.update_stats(error=True)
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """Start WebSocket receive loop"""
        self.receive_callback = callback
        
        # Ensure connection
        if not self.connected:
            await self.connect()
        
        if self.connected:
            self.running = True
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.logger.info(f"WebSocket receive handler started for {endpoint or 'all'}")
        else:
            self.logger.error("Cannot start WebSocket receive - not connected")


class MessageQueueHandler(ProtocolHandler):
    """Message queue protocol handler (Redis/RabbitMQ)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protocol_name = "MESSAGE_QUEUE"
        self.broker_url = config.get('broker_url', 'redis://localhost:6379')
        self.queue_prefix = config.get('queue_prefix', 'cmcp_')
        self.consumer_tag = None
        self.channel = None
        
        # Try to import Redis
        try:
            import redis
            self.redis = redis
            self.redis_client = None
        except ImportError:
            self.redis = None
            self.logger.warning("Redis not installed, MessageQueue protocol disabled")
    
    async def connect(self):
        """Connect to message queue broker"""
        if not self.redis:
            self.logger.error("MessageQueue protocol requires 'redis' package")
            self.connected = False
            return
        
        try:
            self.redis_client = self.redis.from_url(self.broker_url)
            # Test connection
            self.redis_client.ping()
            self.connected = True
            self.logger.info(f"MessageQueue connected to {self.broker_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect MessageQueue: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from message queue"""
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None
        
        self.connected = False
        self.logger.info("MessageQueue disconnected")
    
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send message to message queue"""
        start_time = time.time()
        
        try:
            # Validate message
            errors = self._validate_message(message)
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Ensure connection
            if not self.connected or not self.redis_client:
                await self.connect()
                if not self.connected:
                    raise ConnectionError("MessageQueue not connected")
            
            # Prepare queue name
            queue_name = f"{self.queue_prefix}{target_endpoint}"
            
            # Publish message
            message_dict = message.to_dict()
            message_dict['_target_endpoint'] = target_endpoint
            
            self.redis_client.rpush(queue_name, json.dumps(message_dict))
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(sent=True, latency_ms=latency_ms)
            
            self.logger.debug(f"Message sent to queue '{queue_name}' in {latency_ms:.2f}ms")
            
            # For message queues, response is handled asynchronously
            # Return None or a placeholder response
            return None
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            self.logger.error(f"MessageQueue send failed: {e}")
            
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "CONNECTION_ERROR",
                str(e),
                retryable=True
            )
    
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """Start consuming messages from queue"""
        if not self.redis:
            self.logger.error("Cannot start MessageQueue receive - Redis not available")
            return
        
        # Ensure connection
        if not self.connected:
            await self.connect()
        
        if not self.connected:
            self.logger.error("Cannot start MessageQueue receive - not connected")
            return
        
        # Start consumer in background
        queue_name = f"{self.queue_prefix}{endpoint}" if endpoint else f"{self.queue_prefix}general"
        
        async def consume_messages():
            while True:
                try:
                    # Blocking pop with timeout
                    message_data = self.redis_client.blpop(queue_name, timeout=1)
                    
                    if message_data:
                        _, message_json = message_data
                        message_dict = json.loads(message_json)
                        
                        # Convert to Message
                        from .message_format import Message
                        message = Message.from_dict(message_dict)
                        
                        # Update statistics
                        self.update_stats(received=True)
                        
                        # Call callback
                        callback(message)
                    
                    # Brief pause to prevent tight loop
                    await asyncio.sleep(0.1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"MessageQueue consume error: {e}")
                    self.update_stats(error=True)
                    await asyncio.sleep(1)
        
        # Start consumer task
        asyncio.create_task(consume_messages())
        self.logger.info(f"MessageQueue consumer started for queue '{queue_name}'")


class SharedMemoryHandler(ProtocolHandler):
    """Shared memory protocol handler (high-performance local communication)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.protocol_name = "SHARED_MEMORY"
        self.shared_memory_blocks = {}
        self.locks = {}
        self.memory_size = config.get('memory_size', 10 * 1024 * 1024)  # 10MB default
        
        # Import multiprocessing for shared memory
        try:
            import multiprocessing
            self.multiprocessing = multiprocessing
        except ImportError:
            self.multiprocessing = None
            self.logger.warning("multiprocessing not available, SharedMemory protocol disabled")
    
    async def connect(self):
        """Shared memory is always connected locally"""
        if self.multiprocessing:
            self.connected = True
            self.logger.info("SharedMemory protocol ready")
        else:
            self.connected = False
            self.logger.error("SharedMemory protocol requires multiprocessing")
    
    async def disconnect(self):
        """Clean up shared memory blocks"""
        for key, block in self.shared_memory_blocks.items():
            if hasattr(block, 'close'):
                block.close()
            if hasattr(block, 'unlink'):
                block.unlink()
        
        self.shared_memory_blocks.clear()
        self.locks.clear()
        self.connected = False
        self.logger.info("SharedMemory protocol disconnected")
    
    async def send(self, message: Message, target_endpoint: str) -> Optional[ResponseMessage]:
        """Send message via shared memory"""
        start_time = time.time()
        
        try:
            # Validate message
            errors = self._validate_message(message)
            if errors:
                raise ValueError(f"Message validation failed: {errors}")
            
            # Prepare memory key
            memory_key = f"cmcp_{target_endpoint}"
            
            # Create or get shared memory block
            if memory_key not in self.shared_memory_blocks:
                # Create new shared memory
                import multiprocessing
                self.shared_memory_blocks[memory_key] = multiprocessing.Array('c', self.memory_size)
                self.locks[memory_key] = multiprocessing.Lock()
            
            memory_block = self.shared_memory_blocks[memory_key]
            lock = self.locks[memory_key]
            
            # Serialize message
            message_json = message.to_json()
            message_bytes = message_json.encode('utf-8')
            
            # Check size
            if len(message_bytes) > self.memory_size:
                raise ValueError(f"Message too large: {len(message_bytes)} > {self.memory_size}")
            
            # Write to shared memory with lock
            with lock:
                # Clear memory
                memory_block[:len(message_bytes)] = message_bytes
                # Null terminate
                if len(message_bytes) < self.memory_size:
                    memory_block[len(message_bytes)] = b'\0'
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(sent=True, latency_ms=latency_ms)
            
            self.logger.debug(f"Message written to shared memory '{memory_key}' in {latency_ms:.2f}ms")
            
            # Shared memory is one-way, no immediate response
            return None
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.update_stats(sent=True, error=True, latency_ms=latency_ms)
            self.logger.error(f"SharedMemory send failed: {e}")
            
            from .message_format import create_error_response
            return create_error_response(
                message,
                target_endpoint,
                "INTERNAL_ERROR",
                str(e),
                retryable=False
            )
    
    async def receive(self, callback: Callable[[Message], None], endpoint: Optional[str] = None):
        """Monitor shared memory for incoming messages"""
        if not self.multiprocessing:
            self.logger.error("Cannot start SharedMemory receive - multiprocessing not available")
            return
        
        memory_key = f"cmcp_{endpoint}" if endpoint else "cmcp_general"
        
        async def monitor_memory():
            last_content = b''
            
            while True:
                try:
                    # Check if memory block exists
                    if memory_key in self.shared_memory_blocks:
                        memory_block = self.shared_memory_blocks[memory_key]
                        lock = self.locks[memory_key]
                        
                        with lock:
                            # Read content until null terminator
                            content_bytes = bytearray()
                            for i in range(self.memory_size):
                                byte = memory_block[i]
                                if byte == b'\0' or byte == 0:
                                    break
                                content_bytes.append(ord(byte) if isinstance(byte, str) else byte)
                        
                        # Check if content changed
                        if content_bytes != last_content and content_bytes:
                            try:
                                # Parse message
                                message_json = content_bytes.decode('utf-8')
                                message_dict = json.loads(message_json)
                                
                                # Convert to Message
                                from .message_format import Message
                                message = Message.from_dict(message_dict)
                                
                                # Update statistics
                                self.update_stats(received=True)
                                
                                # Call callback
                                callback(message)
                                
                                # Update last content
                                last_content = content_bytes
                                
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                self.logger.warning(f"Failed to parse shared memory content: {e}")
                    
                    # Wait before checking again
                    await asyncio.sleep(0.1)  # 100ms polling interval
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"SharedMemory monitor error: {e}")
                    self.update_stats(error=True)
                    await asyncio.sleep(1)
        
        # Start monitoring task
        asyncio.create_task(monitor_memory())
        self.logger.info(f"SharedMemory monitor started for '{memory_key}'")