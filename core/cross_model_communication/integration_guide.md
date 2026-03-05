# CMCP Integration Guide for Self-Soul AGI Models

## Overview
This guide explains how to integrate existing Self-Soul AGI models with the Cross-Model Communication Protocol (CMCP). The integration enables models to communicate seamlessly, collaborate on complex tasks, and participate in multi-model workflows.

## Prerequisites
- Python 3.7+
- Existing Self-Soul AGI models
- Basic understanding of async/await patterns

## Integration Levels

### Level 1: Basic Integration (Minimal Changes)
Add communication capabilities without modifying core model logic.

### Level 2: Enhanced Integration (Moderate Changes)
Implement communication handlers and leverage CMCP features.

### Level 3: Full Integration (Comprehensive Changes)
Redesign model architecture around CMCP principles.

## Step-by-Step Integration

### Step 1: Add CMCP Dependencies
Add CMCP to your model's requirements or import it:

```python
from core.cross_model_communication import (
    ModelCommunicationMixin, 
    communication_handler,
    get_gateway
)
```

### Step 2: Update Model Class Inheritance
Add `ModelCommunicationMixin` to your model's base classes:

```python
# Before:
class UnifiedLanguageModel(UnifiedModelTemplate):
    ...

# After:
class UnifiedLanguageModel(UnifiedModelTemplate, ModelCommunicationMixin):
    ...
```

### Step 3: Initialize Communication
Add communication initialization in `__init__`:

```python
def __init__(self, config=None):
    super().__init__(config)
    
    # Initialize communication
    self.init_communication({
        'model_id': self.model_id,  # Use existing model_id if available
        'model_type': 'language',   # Set appropriate model type
        'max_concurrent_requests': 10,
        'estimated_processing_time_ms': 150
    })
```

### Step 4: Add Communication Handlers
Mark existing methods as communication handlers or create new ones:

```python
# Option A: Decorate existing method
@communication_handler(operation="generate_text")
async def generate_text(self, prompt, max_length=100):
    # Existing implementation
    result = await self._generate_text_impl(prompt, max_length)
    return {
        'generated_text': result,
        'tokens_generated': len(result.split()),
        'success': True
    }

# Option B: Create dedicated handler
@communication_handler(operation="process_request")
async def handle_process_request(self, message, context):
    """Handle CMCP requests"""
    operation = message.data.get('operation')
    data = message.data.get('data', {})
    
    if operation == 'generate_text':
        result = await self.generate_text(
            data.get('prompt'), 
            data.get('max_length', 100)
        )
    elif operation == 'translate':
        result = await self.translate_text(
            data.get('text'),
            data.get('source_lang'),
            data.get('target_lang')
        )
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    # Return CMCP response
    from core.cross_model_communication.message_format import create_response
    return create_response(message, self.model_id, result)
```

### Step 5: Update Method Signatures for CMCP
If using dedicated handlers, ensure they accept CMCP messages:

```python
@communication_handler(operation="your_operation")
async def handle_your_operation(self, message, context):
    """
    Args:
        message: RequestMessage from CMCP
        context: Additional context including trace_id, source_model, etc.
    
    Returns:
        ResponseMessage or dict that can be converted to ResponseMessage
    """
    # Extract data from message
    input_data = message.data
    
    # Call existing model logic
    result = await self.your_existing_method(**input_data)
    
    # Return response
    return {
        'result': result,
        'processing_time_ms': context.get('processing_time', 0),
        'model_version': self.version
    }
```

### Step 6: Implement Model Capabilities (Optional but Recommended)
Add a method to declare model capabilities:

```python
def get_capabilities(self):
    """Return list of supported operations"""
    return [
        "generate_text",
        "translate",
        "summarize",
        "analyze_sentiment",
        "answer_questions"
    ]
```

## Integration Examples for Specific Models

### Example 1: Language Model Integration

```python
from core.cross_model_communication import ModelCommunicationMixin, communication_handler
from core.models.language.unified_language_model import UnifiedLanguageModel

class CMCPIntegratedLanguageModel(UnifiedLanguageModel, ModelCommunicationMixin):
    """Language model with CMCP integration"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize communication with language-specific config
        self.init_communication({
            'model_id': f"language_model_{self.model_id}",
            'model_type': 'language',
            'max_concurrent_requests': 20,
            'estimated_processing_time_ms': 200,
            'memory_requirement_mb': 2048
        })
    
    @communication_handler(operation="generate_text")
    async def handle_generate_text(self, message, context):
        """Handle text generation requests"""
        prompt = message.data.get("prompt", "")
        max_length = message.data.get("max_length", 100)
        temperature = message.data.get("temperature", 0.7)
        
        # Call existing generate_text method
        result = await self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # Format as CMCP response
        return {
            'generated_text': result,
            'prompt': prompt,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature
            },
            'model_info': {
                'version': self.version,
                'model_id': self.model_id
            }
        }
    
    @communication_handler(operation="translate")
    async def handle_translate(self, message, context):
        """Handle translation requests"""
        text = message.data.get("text", "")
        source_lang = message.data.get("source_lang", "auto")
        target_lang = message.data.get("target_lang", "en")
        
        # Use existing translation capability
        translated = await self.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        return {
            'original_text': text,
            'translated_text': translated,
            'languages': {
                'source': source_lang,
                'target': target_lang
            },
            'translation_quality': 'high'  # Could be based on confidence
        }
    
    def get_capabilities(self):
        """Return language model capabilities"""
        return [
            "generate_text",
            "translate",
            "summarize",
            "answer_questions",
            "sentiment_analysis",
            "text_classification"
        ]
```

### Example 2: Vision Model Integration

```python
from core.cross_model_communication import ModelCommunicationMixin, communication_handler
from core.models.vision.unified_vision_model import UnifiedVisionModel

class CMCPIntegratedVisionModel(UnifiedVisionModel, ModelCommunicationMixin):
    """Vision model with CMCP integration"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.init_communication({
            'model_id': f"vision_model_{self.model_id}",
            'model_type': 'vision',
            'max_concurrent_requests': 10,  # Vision models are more resource-intensive
            'estimated_processing_time_ms': 500,
            'gpu_required': True,
            'memory_requirement_mb': 4096
        })
    
    @communication_handler(operation="analyze_image")
    async def handle_analyze_image(self, message, context):
        """Handle image analysis requests"""
        image_data = message.data.get("image_data")
        image_format = message.data.get("format", "rgb")
        analysis_types = message.data.get("analysis_types", ["objects", "faces"])
        
        # Process image
        image = self._decode_image(image_data, image_format)
        
        results = {}
        
        if "objects" in analysis_types:
            objects = await self.detect_objects(image)
            results['objects'] = objects
        
        if "faces" in analysis_types:
            faces = await self.detect_faces(image)
            results['faces'] = faces
        
        if "text" in analysis_types:
            text = await self.extract_text(image)
            results['text'] = text
        
        return {
            'analysis_results': results,
            'image_info': {
                'format': image_format,
                'size': len(image_data),
                'dimensions': image.shape if hasattr(image, 'shape') else 'unknown'
            },
            'processing_time_ms': context.get('processing_time', 0)
        }
    
    @communication_handler(operation="generate_caption")
    async def handle_generate_caption(self, message, context):
        """Generate image caption"""
        image_data = message.data.get("image_data")
        style = message.data.get("style", "descriptive")
        
        image = self._decode_image(image_data)
        caption = await self.generate_image_caption(image, style=style)
        
        return {
            'caption': caption,
            'style': style,
            'confidence': 0.85  # Example confidence score
        }
```

### Example 3: Multi-Model Collaboration

```python
class CollaborativeTaskOrchestrator:
    """Orchestrator for multi-model tasks using CMCP"""
    
    def __init__(self):
        self.gateway = get_gateway()
        
    async def process_complex_request(self, request_data):
        """Process request using multiple models"""
        # Step 1: Analyze with vision model
        vision_response = await self.gateway.send_message(
            RequestMessage(
                source_model="orchestrator",
                target_models=["vision_model"],
                operation="analyze_image",
                data={
                    "image_data": request_data["image"],
                    "analysis_types": ["objects", "text"]
                }
            )
        )
        
        if not vision_response.is_success():
            return self._handle_error(vision_response)
        
        vision_results = vision_response.data
        
        # Step 2: Process text with language model
        if "text" in vision_results.get("analysis_results", {}):
            text = vision_results["analysis_results"]["text"]
            
            language_response = await self.gateway.send_message(
                RequestMessage(
                    source_model="orchestrator",
                    target_models=["language_model"],
                    operation="process_text",
                    data={
                        "text": text,
                        "operations": ["summarize", "extract_keywords"]
                    }
                )
            )
            
            text_results = language_response.data if language_response.is_success() else {}
        else:
            text_results = {}
        
        # Step 3: Retrieve knowledge about detected objects
        objects = vision_results.get("analysis_results", {}).get("objects", [])
        knowledge_responses = []
        
        for obj in objects[:3]:  # Limit to 3 objects
            knowledge_response = await self.gateway.send_message(
                RequestMessage(
                    source_model="orchestrator",
                    target_models=["knowledge_model"],
                    operation="retrieve_information",
                    data={"topic": obj, "max_results": 2}
                )
            )
            
            if knowledge_response.is_success():
                knowledge_responses.append(knowledge_response.data)
        
        # Step 4: Synthesize final response
        final_response = await self.gateway.send_message(
            RequestMessage(
                source_model="orchestrator",
                target_models=["language_model"],
                operation="generate_text",
                data={
                    "prompt": self._create_synthesis_prompt(
                        vision_results, text_results, knowledge_responses
                    ),
                    "max_length": 500
                }
            )
        )
        
        return {
            "success": True,
            "components": {
                "vision_analysis": vision_results,
                "text_processing": text_results,
                "knowledge_retrieval": knowledge_responses
            },
            "synthesis": final_response.data if final_response.is_success() else None
        }
```

## Migration Strategies

### Strategy 1: Wrapper Pattern (Least Invasive)
Create a wrapper class that delegates to the existing model:

```python
class CMCPWrapper:
    """Wrapper for existing models without modifying them"""
    
    def __init__(self, original_model):
        self.model = original_model
        self.adapter = create_adapter_for_model(self)
        
    @communication_handler(operation="process")
    async def handle_request(self, message, context):
        # Extract parameters from message
        # Call original model
        # Convert response to CMCP format
        pass
```

### Strategy 2: Adapter Pattern (Moderate Changes)
Use the built-in adapter system:

```python
# Existing model remains unchanged
existing_model = UnifiedLanguageModel(config)

# Create adapter
adapter = create_adapter_for_model(existing_model, {
    'model_id': 'language_v1',
    'model_type': 'language'
})

# Register specific methods as handlers
adapter.register_handler('generate_text', existing_model.generate_text)
adapter.register_handler('translate', existing_model.translate)

# Start adapter
await adapter.start()
```

### Strategy 3: Full Integration (Most Powerful)
Inherit from both original model and CMCP mixin:

```python
class FullyIntegratedModel(UnifiedLanguageModel, ModelCommunicationMixin):
    # As shown in examples above
    pass
```

## Configuration Reference

### Gateway Configuration
```yaml
# config/cmcp_gateway.yml
gateway:
  default_protocol: "direct_call"  # direct_call, rest, websocket, message_queue, shared_memory
  protocol_configs:
    direct_call:
      max_workers: 20
      timeout_ms: 10000
    
    rest:
      base_url: "http://localhost:8000"
      timeout_ms: 15000
      retry_attempts: 3
    
    websocket:
      server_url: "ws://localhost:8000/ws"
      reconnect_delay_ms: 1000
  
  load_balancing_strategy: "round_robin"  # round_robin, weighted, least_connections
  circuit_breaker_enabled: true
  circuit_breaker_threshold: 5
  circuit_breaker_reset_timeout: 60
  max_concurrent_requests: 100
  monitoring_enabled: true
  security_enabled: true
```

### Model Adapter Configuration
```yaml
# config/model_adapters/language_model.yml
language_model:
  adapter:
    model_id: "unified_language_model_v1"
    model_type: "language"
    max_concurrent_requests: 15
    estimated_processing_time_ms: 200
    memory_requirement_mb: 2048
    gpu_required: false
    simple_mode: false  # Use full CMCP features
    
  handlers:
    - operation: "generate_text"
      method: "generate_text"
      timeout_ms: 5000
      priority: "medium"
    
    - operation: "translate"
      method: "translate_text"
      timeout_ms: 3000
      priority: "high"
    
    - operation: "summarize"
      method: "summarize_text"
      timeout_ms: 4000
      priority: "medium"
  
  capabilities:
    - "text_generation"
    - "translation"
    - "summarization"
    - "sentiment_analysis"
    - "question_answering"
  
  dependencies:
    - "knowledge_model"  # For enhanced responses
    - "translation_model"  # For multilingual support
```

## Error Handling and Recovery

### Handling Communication Errors
```python
async def safe_model_call(self, target_model, operation, data, max_retries=3):
    """Make resilient model calls with retries"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = await self.send_to_model(
                target_model=target_model,
                operation=operation,
                data=data,
                timeout_ms=5000 * (attempt + 1)  # Exponential backoff
            )
            
            if response and response.is_success():
                return response.data
            
            elif response and response.error and response.error.retryable:
                last_error = response.error.message
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            else:
                # Non-retryable error
                raise ValueError(f"Model error: {response.get_error_message() if response else 'Unknown'}")
                
        except (TimeoutError, ConnectionError) as e:
            last_error = str(e)
            await asyncio.sleep(2 ** attempt)
            continue
    
    # All retries failed
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
```

### Circuit Breaker Integration
```python
from core.cross_model_communication.cmcp_gateway import CircuitBreaker

class ResilientModelClient:
    def __init__(self):
        self.circuit_breakers = {}
    
    def get_circuit_breaker(self, model_id):
        if model_id not in self.circuit_breakers:
            self.circuit_breakers[model_id] = CircuitBreaker(
                threshold=5,
                reset_timeout=60
            )
        return self.circuit_breakers[model_id]
    
    async def call_with_circuit_breaker(self, model_id, operation, data):
        cb = self.get_circuit_breaker(model_id)
        
        if not cb.allow_request():
            raise CircuitBreakerOpenError(f"Circuit breaker open for {model_id}")
        
        try:
            response = await self.send_to_model(model_id, operation, data)
            
            if response and response.is_success():
                cb.record_success()
                return response
            else:
                cb.record_failure()
                raise ModelError(response.get_error_message() if response else "Unknown error")
                
        except Exception as e:
            cb.record_failure()
            raise
```

## Performance Optimization

### Connection Pooling
```python
class OptimizedCMCPClient:
    def __init__(self):
        self.connection_pools = {}
    
    async def get_connection(self, model_id, protocol):
        """Get or create connection from pool"""
        pool_key = f"{model_id}_{protocol}"
        
        if pool_key not in self.connection_pools:
            self.connection_pools[pool_key] = ConnectionPool(
                max_size=10,
                idle_timeout=300
            )
        
        return await self.connection_pools[pool_key].acquire()
```

### Message Compression
```python
import zlib
import json

class CompressedMessageHandler:
    @staticmethod
    def compress_message(message):
        """Compress large messages"""
        message_json = message.to_json()
        
        if len(message_json) > 1024:  # Compress if > 1KB
            compressed = zlib.compress(message_json.encode())
            return {
                'compressed': True,
                'algorithm': 'zlib',
                'data': compressed.hex()
            }
        
        return {'compressed': False, 'data': message_json}
    
    @staticmethod
    def decompress_message(compressed_data):
        """Decompress message"""
        if compressed_data.get('compressed'):
            data_bytes = bytes.fromhex(compressed_data['data'])
            decompressed = zlib.decompress(data_bytes).decode()
            return Message.from_json(decompressed)
        else:
            return Message.from_json(compressed_data['data'])
```

## Testing Integration

### Unit Tests for CMCP Handlers
```python
import pytest
from core.cross_model_communication.message_format import create_request

class TestLanguageModelCMCP:
    async def test_generate_text_handler(self):
        model = CMCPIntegratedLanguageModel()
        await model.start()
        
        # Create test message
        message = create_request(
            source_model="test_client",
            target_models=[model.model_id],
            operation="generate_text",
            data={
                "prompt": "Test prompt",
                "max_length": 50
            }
        )
        
        # Call handler directly
        response = await model.handle_generate_text(message, {})
        
        # Verify response
        assert response.is_success()
        assert "generated_text" in response.data
        assert len(response.data["generated_text"]) > 0
        
        await model.stop()
```

### Integration Tests
```python
class TestMultiModelCollaboration:
    async def test_vision_language_collaboration(self):
        # Start models
        vision_model = CMCPIntegratedVisionModel()
        language_model = CMCPIntegratedLanguageModel()
        
        await vision_model.start()
        await language_model.start()
        
        # Test collaboration
        image_data = self._load_test_image()
        
        # Vision analysis
        vision_response = await vision_model.send_to_model(
            target_model=vision_model.model_id,
            operation="analyze_image",
            data={"image_data": image_data}
        )
        
        assert vision_response.is_success()
        
        # Language processing of vision results
        objects = vision_response.data.get("objects", [])
        
        if objects:
            language_response = await language_model.send_to_model(
                target_model=language_model.model_id,
                operation="generate_text",
                data={
                    "prompt": f"Describe a scene with {', '.join(objects[:3])}",
                    "max_length": 100
                }
            )
            
            assert language_response.is_success()
        
        await vision_model.stop()
        await language_model.stop()
```

## Deployment Considerations

### Scaling with CMCP
```python
class ScalableModelDeployment:
    """Deployment strategies for CMCP-integrated models"""
    
    @staticmethod
    def horizontal_scaling(model_class, instance_count=3):
        """Deploy multiple instances of a model"""
        instances = []
        
        for i in range(instance_count):
            instance = model_class(config={
                'model_id': f"{model_class.__name__.lower()}_{i}",
                'instance_number': i
            })
            
            instances.append(instance)
        
        # Register all instances with load balancer
        load_balancer = LoadBalancer(strategy="round_robin")
        for instance in instances:
            load_balancer.register_instance(instance.model_id, instance)
        
        return instances, load_balancer
    
    @staticmethod
    def vertical_scaling(model_class, resource_multiplier=2):
        """Increase resources for a single instance"""
        return model_class(config={
            'max_concurrent_requests': 50 * resource_multiplier,
            'memory_allocation_mb': 4096 * resource_multiplier,
            'gpu_count': 2 if resource_multiplier > 1 else 1
        })
```

## Monitoring and Observability

### Custom Metrics
```python
from core.cross_model_communication.monitoring import get_monitor

class MonitoredModel:
    def __init__(self):
        self.monitor = get_monitor()
        
    async def process_with_metrics(self, operation, data):
        start_time = time.time()
        
        try:
            result = await self._process(operation, data)
            
            # Record success metrics
            processing_time = time.time() - start_time
            self.monitor.record_message_processing(
                message=None,  # Would be actual message in real scenario
                processing_time=processing_time,
                success=True,
                error_type=None
            )
            
            self.monitor.increment_counter(
                "custom_model_operations",
                labels={"operation": operation, "status": "success"}
            )
            
            return result
            
        except Exception as e:
            # Record error metrics
            self.monitor.increment_counter(
                "custom_model_operations",
                labels={"operation": operation, "status": "error"}
            )
            
            self.monitor.log_message(
                "error",
                f"Operation {operation} failed",
                {"error": str(e), "data": data}
            )
            
            raise
```

## Migration Checklist

### Phase 1: Preparation
- [ ] Review existing model architecture
- [ ] Identify communication patterns
- [ ] Choose integration level (1, 2, or 3)
- [ ] Set up CMCP dependencies
- [ ] Create test environment

### Phase 2: Basic Integration
- [ ] Add ModelCommunicationMixin to model classes
- [ ] Initialize communication in __init__
- [ ] Test basic model instantiation
- [ ] Verify CMCP components load correctly

### Phase 3: Handler Implementation
- [ ] Identify key operations to expose
- [ ] Create @communication_handler decorated methods
- [ ] Implement proper request/response handling
- [ ] Add error handling and validation

### Phase 4: Testing
- [ ] Write unit tests for handlers
- [ ] Test inter-model communication
- [ ] Verify error scenarios
- [ ] Performance testing

### Phase 5: Deployment
- [ ] Configure gateway settings
- [ ] Set up monitoring
- [ ] Deploy to staging environment
- [ ] Validate in integrated context

### Phase 6: Optimization
- [ ] Implement connection pooling
- [ ] Add message compression for large payloads
- [ ] Configure circuit breakers
- [ ] Set up alerting rules

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Models Can't Find Each Other
**Solution**: Ensure models are registered with the registry:
```python
# Manually register if auto-registration fails
from core.cross_model_communication.registry import get_local_registry
registry = get_local_registry()
registry.register(model_registration)
```

#### Issue 2: Timeout Errors
**Solution**: Increase timeout or implement retry logic:
```python
# In adapter config
config = {
    'timeout_ms': 10000,  # Increase from default 5000
    'retry_attempts': 3
}
```

#### Issue 3: Memory Issues with Large Messages
**Solution**: Implement message compression or streaming:
```python
# Use compression for large messages
if len(message_json) > 1024 * 1024:  # 1MB
    message.metadata['compression'] = 'gzip'
    message.data = compress_data(message.data)
```

#### Issue 4: Circuit Breaker Tripping Frequently
**Solution**: Adjust thresholds or investigate underlying issues:
```python
config = {
    'circuit_breaker_threshold': 10,  # Increase from 5
    'circuit_breaker_reset_timeout': 120  # Increase from 60
}
```

## Support and Resources

- **Documentation**: `cross_model_communication_protocol.md`
- **Examples**: `examples/cmcp_demo.py`
- **API Reference**: Module docstrings
- **Troubleshooting**: Check gateway logs and metrics

## Conclusion

CMCP integration transforms standalone AI models into collaborative agents. Start with Level 1 integration for basic communication, then progressively adopt more advanced features as needed. The protocol is designed to be incrementally adoptable while providing powerful capabilities for complex multi-model systems.