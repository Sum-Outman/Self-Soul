# Cross-Model Communication Protocol (CMCP) v1.0

## Overview
The Cross-Model Communication Protocol (CMCP) is a standardized framework for enabling seamless communication and collaboration between different AI models in the Self-Soul AGI system. This protocol builds upon existing collaboration infrastructure while providing a unified, consistent interface for all inter-model communications.

## Design Principles
1. **Unified Interface**: Single API for all model communication regardless of underlying transport
2. **Protocol Agnostic**: Support multiple communication methods (REST, WebSocket, message queues, direct calls)
3. **Standardized Format**: Consistent message structure across all model interactions
4. **Extensible**: Easy to add new model types, communication methods, and collaboration patterns
5. **Observable**: Built-in monitoring, logging, and performance tracking

## Core Components

### 1. Standard Message Format

#### Request Message
```json
{
  "protocol_version": "1.0",
  "message_id": "msg_1772683275_abc123",
  "timestamp": "2026-03-05T11:47:55.123456Z",
  "source_model": "translation_model",
  "target_models": ["language_model", "knowledge_model"],
  "operation": "translate_with_context",
  "priority": "medium",  // low, medium, high, critical
  "timeout_ms": 5000,
  "data": {
    "text": "Hello world",
    "source_lang": "en",
    "target_lang": "zh",
    "context": "technical documentation"
  },
  "metadata": {
    "requires_response": true,
    "response_format": "standard",
    "session_id": "session_123",
    "correlation_id": "corr_456",
    "security_level": "standard"  // standard, sensitive, critical
  }
}
```

#### Response Message
```json
{
  "protocol_version": "1.0",
  "message_id": "msg_1772683275_abc123",
  "response_to": "msg_1772683275_abc123",
  "timestamp": "2026-03-05T11:47:55.234567Z",
  "source_model": "language_model",
  "original_source": "translation_model",
  "status": "success",  // success, partial_success, error, timeout
  "operation": "translate_with_context",
  "data": {
    "translated_text": "你好世界",
    "confidence": 0.95,
    "context_used": true
  },
  "metadata": {
    "processing_time_ms": 120,
    "model_version": "1.2.3",
    "resource_usage": {
      "cpu_percent": 15.2,
      "memory_mb": 256.5
    }
  },
  "error": null
}
```

#### Error Message
```json
{
  "protocol_version": "1.0",
  "message_id": "msg_1772683275_abc123",
  "response_to": "msg_1772683275_abc123",
  "timestamp": "2026-03-05T11:47:55.345678Z",
  "source_model": "language_model",
  "original_source": "translation_model",
  "status": "error",
  "operation": "translate_with_context",
  "data": null,
  "metadata": {
    "processing_time_ms": 5,
    "error_phase": "validation"
  },
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input text: text cannot be empty",
    "details": {
      "field": "text",
      "issue": "must be non-empty string",
      "value_received": ""
    },
    "suggestion": "Provide a non-empty string for the 'text' parameter"
  }
}
```

### 2. Communication Protocols

CMCP supports multiple communication methods:

#### 2.1 Direct Call Protocol (DCP)
- **Description**: Direct Python method calls within same process
- **Use Case**: High-performance, low-latency communication between co-located models
- **Implementation**: Python function calls with standardized parameters
- **Advantages**: Lowest latency, no serialization overhead
- **Limitations**: Models must be in same Python process

#### 2.2 REST Protocol (REST)
- **Description**: HTTP/REST API calls between models
- **Use Case**: Cross-process or cross-machine communication
- **Implementation**: Standard HTTP POST requests with JSON payloads
- **Advantages**: Standard, well-understood, firewall-friendly
- **Limitations**: Higher latency, connection overhead

#### 2.3 Message Queue Protocol (MQP)
- **Description**: Asynchronous communication via message queues
- **Use Case**: Decoupled, reliable communication with guaranteed delivery
- **Implementation**: Redis, RabbitMQ, or Kafka-based messaging
- **Advantages**: Asynchronous, reliable, scalable
- **Limitations**: More complex setup, eventual consistency

#### 2.4 WebSocket Protocol (WSP)
- **Description**: Real-time bidirectional communication
- **Use Case**: Streaming data, real-time collaboration
- **Implementation**: WebSocket connections with JSON message framing
- **Advantages**: Real-time, bidirectional, efficient for streaming
- **Limitations**: Connection management overhead

#### 2.5 Shared Memory Protocol (SMP)
- **Description**: High-speed communication via shared memory
- **Use Case**: Extreme performance requirements within same machine
- **Implementation**: Memory-mapped files or shared memory segments
- **Advantages**: Highest performance for large data transfers
- **Limitations**: Complex, limited to same machine

### 3. Model Registry & Discovery

#### Model Registration
```json
{
  "model_id": "language_model_v1",
  "model_type": "language",
  "version": "1.2.3",
  "capabilities": ["text_generation", "translation", "summarization"],
  "endpoints": {
    "direct": "core.models.language.unified_language_model.UnifiedLanguageModel",
    "rest": "http://localhost:8001/api/v1/language",
    "websocket": "ws://localhost:8001/ws/language",
    "message_queue": "language_model_queue"
  },
  "performance_characteristics": {
    "max_input_size": 4096,
    "typical_response_time_ms": 150,
    "supported_formats": ["text", "json"],
    "resource_requirements": {
      "gpu_memory_mb": 2048,
      "cpu_cores": 2
    }
  },
  "status": "ready",  // ready, busy, offline, maintenance
  "last_heartbeat": "2026-03-05T11:47:55.123456Z"
}
```

#### Service Discovery
- **Central Registry**: Single source of truth for all model endpoints
- **Health Checks**: Regular heartbeat monitoring
- **Load Balancing**: Automatic routing to available instances
- **Failover**: Automatic failover to backup instances
- **Version Management**: Support for multiple model versions

### 4. Communication Patterns

#### 4.1 Request-Response Pattern
```
Source Model → CMCP Gateway → Target Model → Response
```
- **Use Case**: Most common pattern for querying other models
- **Implementation**: Synchronous or asynchronous request with timeout

#### 4.2 Publish-Subscribe Pattern
```
Publisher → Topic → Multiple Subscribers
```
- **Use Case**: Event-driven architectures, notifications
- **Implementation**: Message queue with topic-based routing

#### 4.3 Workflow Pattern
```
Model A → (Result) → Model B → (Result) → Model C
```
- **Use Case**: Complex multi-step processing pipelines
- **Implementation**: Workflow engine with dependency tracking

#### 4.4 Broadcast Pattern
```
Source Model → All Models
```
- **Use Case**: System-wide notifications, configuration updates
- **Implementation**: Fan-out messaging to all registered models

#### 4.5 Aggregation Pattern
```
Multiple Models → Aggregator → Combined Result
```
- **Use Case**: Ensemble methods, result fusion
- **Implementation**: Collect and merge results from multiple models

### 5. Security Model

#### Authentication
- **Model Identity**: Each model has unique identity and credentials
- **API Keys**: For REST and WebSocket communications
- **JWT Tokens**: For session-based authentication
- **Mutual TLS**: For production-grade security

#### Authorization
- **Capability-Based**: Models can only perform operations they're authorized for
- **Role-Based**: Different privilege levels for different model types
- **Resource Limits**: Rate limiting and quota enforcement

#### Data Protection
- **Encryption**: TLS for network traffic, encryption at rest
- **Data Minimization**: Only transmit necessary data
- **Audit Logging**: Complete audit trail of all communications

### 6. Monitoring & Observability

#### Metrics Collection
- **Latency**: End-to-end communication latency
- **Throughput**: Messages per second
- **Error Rates**: Success/failure ratios
- **Resource Usage**: CPU, memory, network usage

#### Health Monitoring
- **Heartbeat**: Regular status updates from all models
- **Dependency Health**: Monitor health of dependent models
- **Circuit Breakers**: Automatic failover when models are unhealthy

#### Distributed Tracing
- **Trace IDs**: Unique identifier for each request chain
- **Span Collection**: Timing information for each processing step
- **Correlation**: Link related operations across models

### 7. Implementation Architecture

#### 7.1 CMCP Gateway
Central component that routes messages between models:
```
┌─────────────────────────────────────────────┐
│              CMCP Gateway                    │
├─────────────────────────────────────────────┤
│ • Message Routing                           │
│ • Protocol Translation                       │
│ • Load Balancing                            │
│ • Security Enforcement                      │
│ • Monitoring & Logging                      │
└─────────────────────────────────────────────┘
```

#### 7.2 Model Adapters
Lightweight adapters for each model type:
```python
class ModelCommunicationAdapter:
    """Adapter for model communication"""
    
    def __init__(self, model_instance, config):
        self.model = model_instance
        self.config = config
        self.protocols = self._initialize_protocols()
    
    def send_message(self, target_model, message):
        """Send message to another model"""
        # Protocol selection logic
        # Message serialization
        # Error handling
        pass
    
    def receive_message(self, message):
        """Receive and process incoming message"""
        # Message validation
        # Operation dispatch
        # Response generation
        pass
```

#### 7.3 Protocol Handlers
Specialized handlers for each communication protocol:
```python
class ProtocolHandler:
    """Base class for protocol handlers"""
    
    def send(self, message):
        raise NotImplementedError
    
    def receive(self, callback):
        raise NotImplementedError

class DirectCallHandler(ProtocolHandler):
    """Direct call protocol handler"""
    
    def send(self, message):
        # Direct Python method call
        pass

class RESTHandler(ProtocolHandler):
    """REST protocol handler"""
    
    def send(self, message):
        # HTTP POST request
        pass
```

### 8. Configuration

#### Global Configuration
```yaml
# config/cross_model_communication.yml
cmcp:
  protocol_version: "1.0"
  default_protocol: "direct"
  
  protocols:
    direct:
      enabled: true
      timeout_ms: 5000
    
    rest:
      enabled: true
      base_url: "http://localhost:8000"
      timeout_ms: 10000
      retry_attempts: 3
    
    websocket:
      enabled: true
      server_url: "ws://localhost:8000/ws"
      reconnect_delay_ms: 1000
    
    message_queue:
      enabled: false
      broker_url: "redis://localhost:6379"
      queue_prefix: "cmcp_"
  
  security:
    authentication_required: true
    encryption_enabled: true
  
  monitoring:
    enabled: true
    metrics_port: 9090
    log_level: "INFO"
```

#### Model-Specific Configuration
```yaml
# config/models/language_model_communication.yml
language_model:
  communication:
    protocols:
      direct:
        priority: 1
        timeout_ms: 3000
      
      rest:
        endpoint: "/api/v1/language"
        priority: 2
      
      websocket:
        channel: "language_updates"
        priority: 3
    
    capabilities:
      - "text_generation"
      - "translation"
      - "summarization"
    
    dependencies:
      - "knowledge_model"
      - "translation_model"
    
    rate_limits:
      requests_per_minute: 1000
      concurrent_connections: 10
```

### 9. Error Handling & Recovery

#### Error Classification
- **Transient Errors**: Network timeouts, temporary unavailability
- **Permanent Errors**: Invalid operations, unsupported formats
- **Resource Errors**: Memory limits, rate limits exceeded

#### Recovery Strategies
- **Retry Logic**: Exponential backoff for transient errors
- **Circuit Breakers**: Prevent cascading failures
- **Fallback Mechanisms**: Alternative models or operations
- **Graceful Degradation**: Reduced functionality instead of complete failure

#### Error Propagation
- **Localized Errors**: Handle at lowest possible level
- **Context Preservation**: Include original context in error messages
- **Actionable Errors**: Provide clear recovery instructions

### 10. Performance Optimization

#### Caching Strategies
- **Result Caching**: Cache frequent query results
- **Connection Pooling**: Reuse connections for HTTP/WebSocket
- **Batch Processing**: Group multiple operations into batches

#### Compression
- **Message Compression**: GZIP compression for large messages
- **Binary Protocols**: Protocol buffers for high-performance scenarios
- **Selective Serialization**: Only serialize changed data

#### Load Distribution
- **Round Robin**: Simple distribution across available instances
- **Weighted Routing**: Based on model performance characteristics
- **Location-Aware**: Prefer geographically closer instances

### 11. Migration Path

#### Phase 1: Core Infrastructure
1. Implement CMCP Gateway
2. Create standard message format
3. Implement Direct Call Protocol
4. Add basic monitoring

#### Phase 2: Protocol Support
1. Add REST Protocol support
2. Implement WebSocket Protocol
3. Add message queue support
4. Implement security features

#### Phase 3: Advanced Features
1. Add workflow orchestration
2. Implement advanced monitoring
3. Add performance optimization
4. Implement comprehensive security

#### Phase 4: Integration
1. Migrate existing models to CMCP
2. Update collaboration systems
3. Implement backward compatibility
4. Performance benchmarking

### 12. API Reference

#### Core CMCP Client
```python
# Basic usage example
from core.cross_model_communication import CMCPClient

# Initialize client
client = CMCPClient(config_path="config/cross_model_communication.yml")

# Send message to another model
response = client.send_message(
    target_model="language_model",
    operation="generate_text",
    data={"prompt": "Explain quantum computing"},
    timeout_ms=5000
)

# Receive messages
client.register_callback(
    operation="process_audio",
    callback=audio_processing_handler
)
```

#### Model Integration
```python
# Model implementation example
from core.cross_model_communication import ModelCommunicationMixin

class UnifiedLanguageModel(..., ModelCommunicationMixin):
    """Language model with built-in communication support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.init_communication()
    
    @communication_handler(operation="generate_text")
    def handle_text_generation(self, message):
        """Handle text generation requests"""
        result = self.generate_text(message.data["prompt"])
        return self.format_response(result)
```

### 13. Testing & Validation

#### Unit Tests
- Message format validation
- Protocol handler correctness
- Error handling scenarios

#### Integration Tests
- End-to-end communication flows
- Cross-protocol compatibility
- Performance benchmarking

#### Load Testing
- High-volume message processing
- Concurrent connection handling
- Resource usage under load

### 14. Compliance & Standards

#### Data Privacy
- GDPR compliance for personal data
- Data retention policies
- Right to erasure support

#### Security Standards
- OWASP security guidelines
- Encryption standards compliance
- Audit trail requirements

#### Industry Standards
- OpenAPI specification for REST endpoints
- AsyncAPI for message queue interfaces
- OpenTelemetry for distributed tracing

---

## Appendix

### A. Message Schema Definitions
Complete JSON schema definitions for all message types.

### B. Protocol Implementation Details
Detailed implementation guides for each communication protocol.

### C. Security Implementation Guide
Step-by-step security configuration and hardening guide.

### D. Performance Tuning Guide
Optimization techniques for different deployment scenarios.

### E. Migration Guide
Step-by-step guide for migrating existing systems to CMCP.

---

**Version**: 1.0  
**Last Updated**: 2026-03-05  
**Status**: Draft  
**Author**: Self-Soul AGI Team