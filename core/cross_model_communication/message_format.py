"""
Standard Message Format for Cross-Model Communication
===================================================

Defines the standardized message structure for all inter-model communications.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import hashlib


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageStatus(Enum):
    """Message status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ErrorCode(Enum):
    """Standard error codes"""
    # Validation errors (1000-1999)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_FORMAT = "INVALID_FORMAT"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_VALUE = "INVALID_VALUE"
    
    # Communication errors (2000-2999)
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    
    # Model errors (3000-3999)
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"
    MODEL_ERROR = "MODEL_ERROR"
    
    # Resource errors (4000-4999)
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    
    # Security errors (5000-5999)
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    
    # System errors (6000-6999)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class MessageMetadata:
    """Message metadata"""
    requires_response: bool = True
    response_format: str = "standard"
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    security_level: str = "standard"  # standard, sensitive, critical
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    ttl_seconds: Optional[int] = None  # Time-to-live in seconds
    retry_count: int = 0
    max_retries: int = 3
    created_by: Optional[str] = None
    processing_deadline: Optional[float] = None  # Unix timestamp


@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_received: Optional[int] = None
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None


@dataclass
class ErrorDetails:
    """Detailed error information"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    stack_trace: Optional[str] = None
    retryable: bool = False
    original_error: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Base message class for all CMCP communications"""
    protocol_version: str = "1.0"
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time())}_{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_model: Optional[str] = None
    target_models: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    priority: str = MessagePriority.MEDIUM.value
    timeout_ms: int = 5000
    data: Optional[Dict[str, Any]] = None
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.data is None:
            self.data = {}
        
        # Convert metadata dict to MessageMetadata if needed
        if isinstance(self.metadata, dict):
            self.metadata = MessageMetadata(**self.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        result = asdict(self)
        
        # Handle nested dataclasses
        result['metadata'] = asdict(self.metadata)
        
        return result
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        # Handle metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = MessageMetadata(**data['metadata'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_hash(self) -> str:
        """Get unique hash for message content"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """Validate message structure, return list of errors"""
        errors = []
        
        # Check required fields
        if not self.message_id:
            errors.append("message_id is required")
        
        if not self.timestamp:
            errors.append("timestamp is required")
        
        if not self.protocol_version:
            errors.append("protocol_version is required")
        
        if not self.operation:
            errors.append("operation is required")
        
        # Validate priority
        try:
            MessagePriority(self.priority)
        except ValueError:
            errors.append(f"Invalid priority: {self.priority}")
        
        # Validate timeout
        if self.timeout_ms <= 0:
            errors.append("timeout_ms must be positive")
        
        return errors


@dataclass
class RequestMessage(Message):
    """Request message for initiating operations"""
    def __post_init__(self):
        super().__post_init__()
        
        # Request messages should always require response
        if not self.metadata.requires_response:
            self.metadata.requires_response = True


@dataclass
class ResponseMessage(Message):
    """Response message for operation results"""
    response_to: Optional[str] = None
    status: str = MessageStatus.SUCCESS.value
    original_source: Optional[str] = None
    error: Optional[ErrorDetails] = None
    resource_usage: Optional[ResourceUsage] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Response messages don't require further response
        self.metadata.requires_response = False
        
        # Convert error dict to ErrorDetails if needed
        if isinstance(self.error, dict):
            self.error = ErrorDetails(**self.error)
        
        # Convert resource_usage dict to ResourceUsage if needed
        if isinstance(self.resource_usage, dict):
            self.resource_usage = ResourceUsage(**self.resource_usage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response message to dictionary"""
        result = super().to_dict()
        
        # Handle nested dataclasses
        if self.error:
            result['error'] = asdict(self.error)
        
        if self.resource_usage:
            result['resource_usage'] = asdict(self.resource_usage)
        
        return result
    
    def is_success(self) -> bool:
        """Check if response indicates success"""
        return self.status == MessageStatus.SUCCESS.value
    
    def is_error(self) -> bool:
        """Check if response indicates error"""
        return self.status == MessageStatus.ERROR.value
    
    def get_error_message(self) -> Optional[str]:
        """Get error message if present"""
        if self.error:
            return self.error.message
        return None


@dataclass
class ErrorMessage(ResponseMessage):
    """Specialized error response message"""
    def __post_init__(self):
        # Force status to error
        self.status = MessageStatus.ERROR.value
        super().__post_init__()


@dataclass
class ModelRegistration:
    """Model registration information for service discovery"""
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
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRegistration':
        """Create from dictionary"""
        return cls(**data)
    
    def is_available(self) -> bool:
        """Check if model is available for requests"""
        return self.status in ["ready", "busy"]
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow().isoformat() + "Z"
    
    def get_endpoint(self, protocol: str) -> Optional[str]:
        """Get endpoint for specific protocol"""
        return self.endpoints.get(protocol)


def create_request(
    source_model: str,
    target_models: List[str],
    operation: str,
    data: Dict[str, Any],
    priority: str = MessagePriority.MEDIUM.value,
    timeout_ms: int = 5000,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    requires_response: bool = True
) -> RequestMessage:
    """Helper function to create a request message"""
    metadata = MessageMetadata(
        requires_response=requires_response,
        session_id=session_id,
        correlation_id=correlation_id
    )
    
    return RequestMessage(
        source_model=source_model,
        target_models=target_models,
        operation=operation,
        priority=priority,
        timeout_ms=timeout_ms,
        data=data,
        metadata=metadata
    )


def create_response(
    request: Message,
    source_model: str,
    data: Dict[str, Any],
    status: str = MessageStatus.SUCCESS.value,
    error: Optional[ErrorDetails] = None,
    resource_usage: Optional[ResourceUsage] = None
) -> ResponseMessage:
    """Helper function to create a response message"""
    return ResponseMessage(
        response_to=request.message_id,
        source_model=source_model,
        original_source=request.source_model,
        target_models=[request.source_model] if request.source_model else [],
        operation=request.operation,
        priority=request.priority,
        data=data,
        status=status,
        error=error,
        resource_usage=resource_usage,
        metadata=MessageMetadata(requires_response=False)
    )


def create_error_response(
    request: Message,
    source_model: str,
    error_code: str,
    error_message: str,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None,
    retryable: bool = False
) -> ErrorMessage:
    """Helper function to create an error response"""
    error_details = ErrorDetails(
        code=error_code,
        message=error_message,
        details=details,
        suggestion=suggestion,
        retryable=retryable
    )
    
    return ErrorMessage(
        response_to=request.message_id,
        source_model=source_model,
        original_source=request.source_model,
        target_models=[request.source_model] if request.source_model else [],
        operation=request.operation,
        priority=request.priority,
        data=None,
        error=error_details,
        metadata=MessageMetadata(requires_response=False)
    )