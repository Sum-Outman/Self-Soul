"""
Audio Processing Exceptions

This module defines specialized exceptions for audio processing operations.
These exceptions provide more granular error handling and recovery strategies
for the unified audio model.
"""

from typing import Optional, Dict, Any
import traceback


class AudioProcessingError(Exception):
    """Base exception for all audio processing errors."""
    
    def __init__(self, message: str, error_code: Optional[int] = None, 
                 component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 6000  # Audio processing base error code
        self.component = component or "audio_processor"
        self.details = details or {}
        self.stack_trace = traceback.format_exc()
        
    def __str__(self):
        return f"[{self.component}] {self.message} (Error Code: {self.error_code})"


class AudioValidationError(AudioProcessingError):
    """Exception raised for audio data validation errors."""
    
    def __init__(self, message: str, validation_type: str = "general", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6101, component="audio_validator", details=details)
        self.validation_type = validation_type


class AudioInputError(AudioProcessingError):
    """Exception raised for audio input related errors."""
    
    def __init__(self, message: str, input_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6102, component="audio_input", details=details)
        self.input_type = input_type


class AudioOutputError(AudioProcessingError):
    """Exception raised for audio output related errors."""
    
    def __init__(self, message: str, output_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6103, component="audio_output", details=details)
        self.output_type = output_type


class AudioFeatureExtractionError(AudioProcessingError):
    """Exception raised for audio feature extraction errors."""
    
    def __init__(self, message: str, feature_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6201, component="feature_extractor", details=details)
        self.feature_type = feature_type


class AudioModelError(AudioProcessingError):
    """Exception raised for audio model related errors."""
    
    def __init__(self, message: str, model_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6301, component="audio_model", details=details)
        self.model_type = model_type


class SpeechRecognitionError(AudioModelError):
    """Exception raised for speech recognition errors."""
    
    def __init__(self, message: str, language: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_type="speech_recognition", details=details)
        self.language = language
        self.error_code = 6311


class SpeechSynthesisError(AudioModelError):
    """Exception raised for speech synthesis errors."""
    
    def __init__(self, message: str, voice: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_type="speech_synthesis", details=details)
        self.voice = voice
        self.error_code = 6321


class MusicRecognitionError(AudioModelError):
    """Exception raised for music recognition errors."""
    
    def __init__(self, message: str, genre: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, model_type="music_recognition", details=details)
        self.genre = genre
        self.error_code = 6331


class AudioStreamError(AudioProcessingError):
    """Exception raised for audio streaming errors."""
    
    def __init__(self, message: str, stream_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6401, component="audio_stream", details=details)
        self.stream_type = stream_type


class AudioDeviceError(AudioProcessingError):
    """Exception raised for audio device errors."""
    
    def __init__(self, message: str, device_name: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6501, component="audio_device", details=details)
        self.device_name = device_name


class AudioLibraryError(AudioProcessingError):
    """Exception raised for audio library dependency errors."""
    
    def __init__(self, message: str, library_name: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6601, component="audio_library", details=details)
        self.library_name = library_name


class AudioResourceError(AudioProcessingError):
    """Exception raised for audio resource management errors."""
    
    def __init__(self, message: str, resource_type: str = "unknown", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=6701, component="audio_resource", details=details)
        self.resource_type = resource_type


# Context managers for audio resource management
class AudioResourceManager:
    """Context manager for managing audio resources."""
    
    def __init__(self, resource_name: str, logger=None):
        self.resource_name = resource_name
        self.logger = logger
        self.resource = None
        
    def __enter__(self):
        if self.logger:
            self.logger.info(f"Acquiring audio resource: {self.resource_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            try:
                self._release_resource()
                if self.logger:
                    self.logger.info(f"Released audio resource: {self.resource_name}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error releasing audio resource {self.resource_name}: {e}")
        
        if exc_type is not None:
            # Log the exception but don't suppress it
            if self.logger:
                self.logger.error(f"Exception in audio resource context {self.resource_name}: {exc_val}")
            return False  # Re-raise the exception
        
        return True
    
    def _release_resource(self):
        """Release the managed resource. To be implemented by subclasses."""
        pass


class PyAudioResourceManager(AudioResourceManager):
    """Context manager for PyAudio resources."""
    
    def __enter__(self):
        try:
            import pyaudio  # type: ignore
        except ImportError as e:
            # Raise AudioLibraryError with appropriate details
            raise AudioLibraryError(
                message=f"PyAudio library not available: {e}",
                library_name="pyaudio",
                details={"error": str(e), "suggestion": "Install pyaudio using 'pip install pyaudio'"}
            ) from e
        
        super().__enter__()
        self.resource = pyaudio.PyAudio()
        return self.resource
    
    def _release_resource(self):
        if self.resource:
            self.resource.terminate()


class AudioStreamContext(AudioResourceManager):
    """Context manager for audio streams."""
    
    def __init__(self, stream, logger=None):
        super().__init__("audio_stream", logger)
        self.stream = stream
        
    def __enter__(self):
        super().__enter__()
        return self.stream
    
    def _release_resource(self):
        if self.stream:
            if hasattr(self.stream, 'is_active') and self.stream.is_active():
                self.stream.stop_stream()
            if hasattr(self.stream, 'close'):
                self.stream.close()


# Helper function for creating error responses
def create_audio_error_response(error: Exception, operation: str, 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a structured error response for audio processing errors."""
    
    if isinstance(error, AudioProcessingError):
        # Use the specialized audio exception information
        response = {
            "success": False,
            "error_type": error.__class__.__name__,
            "error_code": error.error_code,
            "component": error.component,
            "message": str(error),
            "operation": operation,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Add details from the exception
        if error.details:
            response["details"] = error.details
            
        # Add context if provided
        if context:
            response["context"] = context
            
        # Add recovery suggestions based on error type
        if isinstance(error, AudioValidationError):
            response["recovery_suggestion"] = "Validate input audio data format, sample rate, and duration."
        elif isinstance(error, AudioDeviceError):
            response["recovery_suggestion"] = "Check audio device connection, permissions, and drivers."
        elif isinstance(error, AudioLibraryError):
            response["recovery_suggestion"] = f"Install required audio library: {error.library_name}"
        elif isinstance(error, AudioResourceError):
            response["recovery_suggestion"] = "Check system resources (memory, disk space) and close unused applications."
        elif isinstance(error, SpeechRecognitionError):
            response["recovery_suggestion"] = "Check audio clarity, reduce background noise, or try different language settings."
        elif isinstance(error, SpeechSynthesisError):
            response["recovery_suggestion"] = "Try different voice settings or reduce text length."
        elif isinstance(error, AudioStreamError):
            response["recovery_suggestion"] = "Check audio stream configuration and network connectivity."
        else:
            response["recovery_suggestion"] = "Check audio processing parameters and system configuration."
            
    else:
        # Generic exception handling
        response = {
            "success": False,
            "error_type": error.__class__.__name__,
            "error_code": 6000,  # Generic audio error
            "component": "audio_processor",
            "message": str(error),
            "operation": operation,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "recovery_suggestion": "Check system logs for detailed error information."
        }
        
        # Add stack trace for debugging (in non-production environments)
        import os
        if os.getenv('ENVIRONMENT') != 'production':
            response["stack_trace"] = traceback.format_exc()
    
    return response


# Decorator for audio operation error handling
def handle_audio_errors(operation_name: str):
    """Decorator to handle errors in audio processing operations."""
    
    def decorator(func):
        import functools
        import logging
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get logger from self if available
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(__name__)
            
            try:
                return func(*args, **kwargs)
                
            except AudioProcessingError as e:
                # Re-raise specialized audio errors
                logger.error(f"Audio processing error in {operation_name}: {e}")
                raise
                
            except (ImportError, ModuleNotFoundError) as e:
                # Audio library dependency error
                error_msg = f"Missing audio library dependency for {operation_name}: {e}"
                logger.error(error_msg)
                raise AudioLibraryError(
                    message=error_msg,
                    library_name=str(e).split("'")[1] if "'" in str(e) else "unknown",
                    details={"operation": operation_name, "error": str(e)}
                ) from e
                
            except (MemoryError, OSError) as e:
                # Resource-related errors
                error_msg = f"Resource error in {operation_name}: {e}"
                logger.error(error_msg)
                raise AudioResourceError(
                    message=error_msg,
                    resource_type="system_resource",
                    details={"operation": operation_name, "error": str(e)}
                ) from e
                
            except (ValueError, TypeError) as e:
                # Validation errors
                error_msg = f"Validation error in {operation_name}: {e}"
                logger.error(error_msg)
                raise AudioValidationError(
                    message=error_msg,
                    validation_type="input_validation",
                    details={"operation": operation_name, "error": str(e)}
                ) from e
                
            except Exception as e:
                # Catch-all for other exceptions
                error_msg = f"Unexpected error in {operation_name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise AudioProcessingError(
                    message=error_msg,
                    details={"operation": operation_name, "error_type": type(e).__name__, "error": str(e)}
                ) from e
        
        return wrapper
    
    return decorator
