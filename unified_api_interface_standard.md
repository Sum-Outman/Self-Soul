# Unified API Interface Standard

## Overview
This document defines a unified API interface standard for all Self-Soul models to ensure consistency, interoperability, and maintainability across the system.

## Core Principles
1. **Consistency**: All models should follow the same interface patterns
2. **Predictability**: Developers should know what to expect from any model
3. **Extensibility**: The interface should support future enhancements
4. **Backward Compatibility**: Changes should not break existing integrations

## Standard Response Format

### Success Response
```python
{
    "success": True,                    # Boolean: Operation success status
    "status": "success",                # String: Detailed status code
    "result": { ... },                  # Dict/Any: Primary operation result
    "message": "Operation completed",   # String: Human-readable message
    "timestamp": "2026-03-05T11:10:00Z", # String: ISO format timestamp
    "model_id": "translation_model",    # String: Identifier of the model
    "operation": "translate",           # String: Name of the operation performed
    "metadata": {                       # Dict: Additional metadata
        "processing_time_ms": 120,
        "confidence": 0.95,
        "version": "1.0.0"
    }
}
```

### Error Response
```python
{
    "success": False,                   # Boolean: Operation success status
    "status": "error",                  # String: Detailed status code
    "error": {                          # Dict: Error details
        "code": "INVALID_INPUT",
        "message": "Invalid input format",
        "details": "The 'text' parameter must be a non-empty string"
    },
    "timestamp": "2026-03-05T11:10:00Z",
    "model_id": "translation_model",
    "operation": "translate",
    "metadata": {
        "processing_time_ms": 5,
        "suggestion": "Check input format"
    }
}
```

## Standard Status Codes
| Status Code | Description | HTTP Equivalent |
|-------------|-------------|-----------------|
| `success` | Operation completed successfully | 200 OK |
| `created` | Resource created successfully | 201 Created |
| `accepted` | Request accepted for processing | 202 Accepted |
| `partial` | Partial success | 206 Partial Content |
| `error` | General error | 400 Bad Request |
| `invalid_input` | Invalid input parameters | 400 Bad Request |
| `not_found` | Resource not found | 404 Not Found |
| `unauthorized` | Authentication required | 401 Unauthorized |
| `forbidden` | Insufficient permissions | 403 Forbidden |
| `conflict` | Resource conflict | 409 Conflict |
| `server_error` | Internal server error | 500 Internal Server Error |
| `not_implemented` | Feature not implemented | 501 Not Implemented |
| `unavailable` | Service temporarily unavailable | 503 Service Unavailable |

## Standard Operation Patterns

### 1. Core Operations
All models should support these core operations:

```python
# Health check
def health_check(self) -> Dict[str, Any]:
    """Check model health and readiness"""
    return {
        "success": True,
        "status": "success",
        "result": {"healthy": True, "ready": True},
        "message": "Model is healthy and ready",
        "timestamp": current_iso_timestamp(),
        "model_id": self.model_id,
        "operation": "health_check",
        "metadata": {"version": self.version, "uptime_seconds": self.uptime}
    }

# Get capabilities
def get_capabilities(self) -> Dict[str, Any]:
    """Get model capabilities and supported operations"""
    return {
        "success": True,
        "status": "success",
        "result": {
            "model_type": self.model_type,
            "supported_operations": list(self.supported_operations),
            "input_formats": self.supported_input_formats,
            "output_formats": self.supported_output_formats,
            "version": self.version
        },
        "message": "Capabilities retrieved",
        "timestamp": current_iso_timestamp(),
        "model_id": self.model_id,
        "operation": "get_capabilities"
    }
```

### 2. Domain-Specific Operations
Each model implements domain-specific operations following the standard pattern:

```python
# Translation Model
def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> Dict[str, Any]:
    """Translate text between languages"""
    try:
        # Process translation
        translated_text = self._process_translation(text, source_lang, target_lang)
        
        return {
            "success": True,
            "status": "success",
            "result": {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang
            },
            "message": "Translation completed successfully",
            "timestamp": current_iso_timestamp(),
            "model_id": self.model_id,
            "operation": "translate",
            "metadata": {
                "processing_time_ms": processing_time,
                "confidence": confidence_score,
                "character_count": len(text)
            }
        }
    except Exception as e:
        return self._format_error(e, "translate")

# Math Model  
def evaluate_expression(self, expression: str, variables: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """Evaluate mathematical expression"""
    try:
        result = self._evaluate_math(expression, variables)
        
        return {
            "success": True,
            "status": "success",
            "result": {
                "expression": expression,
                "value": result,
                "variables": variables or {}
            },
            "message": "Expression evaluated successfully",
            "timestamp": current_iso_timestamp(),
            "model_id": self.model_id,
            "operation": "evaluate_expression",
            "metadata": {
                "processing_time_ms": processing_time,
                "complexity": expression_complexity
            }
        }
    except Exception as e:
        return self._format_error(e, "evaluate_expression")
```

## Error Handling Standard

### Error Format
```python
def _format_error(self, error: Exception, operation: str) -> Dict[str, Any]:
    """Format error response according to standard"""
    error_code = self._get_error_code(error)
    error_message = str(error)
    
    return {
        "success": False,
        "status": "error",
        "error": {
            "code": error_code,
            "message": error_message,
            "details": self._get_error_details(error),
            "operation": operation
        },
        "timestamp": current_iso_timestamp(),
        "model_id": self.model_id,
        "operation": operation,
        "metadata": {
            "suggestion": self._get_error_suggestion(error),
            "recovery_action": self._get_recovery_action(error)
        }
    }
```

### Standard Error Codes
- `INVALID_INPUT`: Invalid or malformed input
- `VALIDATION_ERROR`: Input validation failed
- `PROCESSING_ERROR`: Error during processing
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `UNAUTHORIZED`: Authentication or authorization failed
- `RATE_LIMITED`: Rate limit exceeded
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable
- `NOT_IMPLEMENTED`: Operation not implemented
- `CONFIGURATION_ERROR`: Configuration error
- `DEPENDENCY_ERROR`: External dependency error

## Input Validation Standard

### Common Validation Rules
1. **Required Parameters**: Clearly document required vs optional parameters
2. **Type Checking**: Validate parameter types
3. **Range Validation**: Validate numerical ranges
4. **Format Validation**: Validate string formats (email, URL, etc.)
5. **Size Limits**: Enforce size limits on inputs

### Validation Response Format
```python
{
    "success": False,
    "status": "invalid_input",
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Input validation failed",
        "details": {
            "field": "text",
            "issue": "must be non-empty string",
            "value_received": None
        }
    },
    "timestamp": current_iso_timestamp(),
    "model_id": self.model_id,
    "operation": operation_name,
    "metadata": {
        "suggestion": "Provide a non-empty string for the 'text' parameter"
    }
}
```

## Implementation Guidelines

### 1. Base Model Class
All models should inherit from a base class that provides:
- Standard response formatting methods
- Error handling utilities
- Input validation helpers
- Logging and monitoring integration

### 2. Response Helper Methods
```python
class UnifiedModelBase:
    """Base class for all unified models"""
    
    def _format_success(self, result: Any, operation: str, message: str = None, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format success response"""
        return {
            "success": True,
            "status": "success",
            "result": result,
            "message": message or f"{operation} completed successfully",
            "timestamp": self._current_timestamp(),
            "model_id": self.model_id,
            "operation": operation,
            "metadata": metadata or {}
        }
    
    def _format_error(self, error: Exception, operation: str, 
                     error_code: str = None) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "status": "error",
            "error": {
                "code": error_code or self._get_error_code(error),
                "message": str(error),
                "operation": operation
            },
            "timestamp": self._current_timestamp(),
            "model_id": self.model_id,
            "operation": operation,
            "metadata": {}
        }
    
    def _validate_input(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate input against rules"""
        # Implementation of validation logic
        pass
```

### 3. Migration Strategy
For existing models, implement a phased migration:
1. **Phase 1**: Add new standardized methods alongside existing ones
2. **Phase 2**: Update validation scripts to accept both old and new formats
3. **Phase 3**: Gradually migrate clients to use new standardized methods
4. **Phase 4**: Deprecate old methods (with proper warnings)
5. **Phase 5**: Remove old methods after sufficient migration period

## Testing Requirements

### Validation Test Cases
All models must pass validation with the following test patterns:
1. **Success Cases**: Test with valid inputs, verify standard response format
2. **Error Cases**: Test with invalid inputs, verify error response format
3. **Boundary Cases**: Test edge cases and boundary conditions
4. **Performance Cases**: Test response time and resource usage

### Example Test Case
```python
{
    "name": "English to Chinese translation",
    "operation": "translate",
    "data": {
        "text": "hello world",
        "source_lang": "en",
        "target_lang": "zh"
    },
    "expected_keys": ["success", "status", "result", "message", "timestamp", "model_id", "operation"],
    "expected_status": "success",
    "success_required": True
}
```

## Compliance Checklist
- [ ] All public methods return standardized response format
- [ ] Error handling follows standard error format
- [ ] Input validation provides clear error messages
- [ ] Health check endpoint implemented
- [ ] Capabilities endpoint implemented
- [ ] All responses include timestamp and model_id
- [ ] Metadata includes relevant processing information
- [ ] Backward compatibility maintained for existing clients
- [ ] Documentation updated to reflect new interface

## Versioning
- **Version 1.0**: Initial standard (current)
- **Future versions**: Will maintain backward compatibility
- **Deprecation**: Old versions will be deprecated with 6-month notice

## References
- [REST API Design Guidelines](https://restfulapi.net/)
- [Google API Design Guide](https://cloud.google.com/apis/design)
- [Microsoft REST API Guidelines](https://github.com/microsoft/api-guidelines)