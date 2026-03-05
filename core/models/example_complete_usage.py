"""
Complete Usage Example for CompositeBaseModel Framework

This example demonstrates the complete workflow for using the CompositeBaseModel framework:
1. Model initialization and configuration
2. Parameter loading and saving
3. Inference execution with optimization
4. Training lifecycle management
5. System monitoring and optimization

The example shows how to create a specialized model that inherits from CompositeBaseModel
and uses all the Mixin capabilities for a production-ready AGI system.
"""

import os
import tempfile
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.models.base.composite_base_model import CompositeBaseModel
from core.models.base.unified_mixins import (
    UnifiedPerformanceCacheMixin,
    UnifiedErrorHandlingResourceMixin,
    TrainingLifecycleMixin,
    UnifiedAGICoreMixin,
    UnifiedExternalAPIMixin
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExampleSpecializedModel(
    CompositeBaseModel,
    UnifiedPerformanceCacheMixin,
    UnifiedErrorHandlingResourceMixin,
    TrainingLifecycleMixin,
    UnifiedAGICoreMixin,
    UnifiedExternalAPIMixin
):
    """
    Example specialized model demonstrating complete framework usage.
    
    This model inherits from CompositeBaseModel and all Mixins to showcase
    the full capabilities of the framework.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None, **kwargs):
        """
        Initialize the example model with configuration.
        
        Args:
            model_config: Model configuration dictionary
            **kwargs: Additional parameters passed to parent classes
        """
        # Initialize with configuration
        super().__init__(model_config=model_config, **kwargs)
        
        # Set model-specific attributes
        self.model_name = "ExampleSpecializedModel"
        self.model_version = "1.0.0"
        
        # Example-specific state
        self.custom_weights = None
        self.feature_extractor = None
        self.training_data_stats = {}
        
        # Enable enhanced capabilities
        self.enable_advanced_error_handling(True)
        self.enable_resource_monitoring(True)
        self.enable_from_scratch_training(True)
        
        logger.info(f"Example model initialized: {self.model_name} v{self.model_version}")
    
    def _custom_initialization(self, initialization_data: Any = None):
        """
        Custom initialization for the example model.
        
        Args:
            initialization_data: Optional initialization data
        """
        logger.info("Performing custom initialization...")
        
        # Initialize custom weights if provided
        if initialization_data and isinstance(initialization_data, dict):
            self.custom_weights = initialization_data.get('weights', None)
            self.feature_extractor = initialization_data.get('feature_extractor', 'default')
            logger.info(f"Custom weights loaded: {self.custom_weights is not None}")
            logger.info(f"Feature extractor: {self.feature_extractor}")
        
        # Initialize example-specific capabilities
        self._initialize_example_capabilities()
    
    def _initialize_example_capabilities(self):
        """Initialize example-specific capabilities."""
        # Register example-specific callbacks
        self.register_training_callback('epoch_start', self._on_epoch_start)
        self.register_training_callback('epoch_end', self._on_epoch_end)
        
        # Configure example-specific caching
        self.enable_caching(True)
        self.cache_ttl = 600  # 10 minutes
        
        logger.info("Example capabilities initialized")
    
    def _on_epoch_start(self, event_data: Dict[str, Any]):
        """Callback for epoch start event."""
        logger.info(f"Epoch {event_data.get('epoch', 'unknown')} started")
    
    def _on_epoch_end(self, event_data: Dict[str, Any]):
        """Callback for epoch end event."""
        epoch = event_data.get('epoch', 'unknown')
        loss = event_data.get('loss', 'unknown')
        logger.info(f"Epoch {epoch} ended with loss: {loss}")
    
    def _perform_inference(self, processed_input: Any, **kwargs) -> Any:
        """
        Implement the actual inference logic for this specialized model.
        
        This is the core inference method that must be implemented by all
        specialized models.
        
        Args:
            processed_input: Pre-processed input data
            **kwargs: Additional inference parameters
            
        Returns:
            Inference result
        """
        logger.info("Performing example inference...")
        
        # Example inference logic
        if isinstance(processed_input, str):
            # Text processing example
            result = self._process_text_input(processed_input, **kwargs)
        elif isinstance(processed_input, (list, tuple)):
            # Sequence processing example
            result = self._process_sequence_input(processed_input, **kwargs)
        elif isinstance(processed_input, dict):
            # Dictionary processing example
            result = self._process_dict_input(processed_input, **kwargs)
        else:
            # Default processing
            result = {
                'input_type': type(processed_input).__name__,
                'input_value': str(processed_input),
                'processed_at': datetime.now().isoformat(),
                'model_confidence': 0.85
            }
        
        # Apply AGI reasoning if available
        if hasattr(self, 'reason_about_problem'):
            try:
                reasoning_context = {
                    'input_type': type(processed_input).__name__,
                    'result_summary': str(result)[:200]
                }
                reasoning_result = self.reason_about_problem(
                    f"Analyzing inference result for: {str(processed_input)[:100]}...",
                    context=reasoning_context
                )
                result['agi_reasoning'] = reasoning_result
                result['enhanced_with_agi'] = True
            except Exception as e:
                logger.warning(f"AGI reasoning failed: {e}")
                result['agi_reasoning'] = {'error': str(e)}
                result['enhanced_with_agi'] = False
        
        logger.info(f"Inference completed successfully")
        return result
    
    def _process_text_input(self, text: str, **kwargs) -> Dict[str, Any]:
        """Example text processing logic."""
        # Simple text analysis
        word_count = len(text.split())
        char_count = len(text)
        sentiment = "neutral"
        
        # Simple sentiment analysis (example)
        positive_words = ['good', 'great', 'excellent', 'happy', 'positive']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'negative']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        
        return {
            'text_length': len(text),
            'word_count': word_count,
            'character_count': char_count,
            'sentiment': sentiment,
            'sentiment_scores': {
                'positive': positive_score,
                'negative': negative_score
            },
            'processing_method': 'text_analysis'
        }
    
    def _process_sequence_input(self, sequence: List, **kwargs) -> Dict[str, Any]:
        """Example sequence processing logic."""
        return {
            'sequence_length': len(sequence),
            'sequence_type': type(sequence).__name__,
            'first_element': str(sequence[0]) if sequence else None,
            'last_element': str(sequence[-1]) if sequence else None,
            'average_length': sum(len(str(item)) for item in sequence) / max(len(sequence), 1),
            'processing_method': 'sequence_analysis'
        }
    
    def _process_dict_input(self, data_dict: Dict, **kwargs) -> Dict[str, Any]:
        """Example dictionary processing logic."""
        return {
            'dict_keys': list(data_dict.keys()),
            'dict_size': len(data_dict),
            'key_types': {k: type(v).__name__ for k, v in data_dict.items()},
            'processing_method': 'dict_analysis'
        }
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """
        Get example-specific model state for serialization.
        
        Overrides the base method to include example-specific data.
        """
        return {
            'custom_weights': self.custom_weights,
            'feature_extractor': self.feature_extractor,
            'training_data_stats': self.training_data_stats,
            'example_specific_field': 'example_value'
        }
    
    def set_model_specific_state(self, state: Dict[str, Any]):
        """
        Set example-specific model state from serialized data.
        
        Overrides the base method to restore example-specific data.
        """
        self.custom_weights = state.get('custom_weights')
        self.feature_extractor = state.get('feature_extractor', 'default')
        self.training_data_stats = state.get('training_data_stats', {})
        logger.info(f"Model-specific state restored: feature_extractor={self.feature_extractor}")


def demonstrate_complete_workflow():
    """
    Demonstrate the complete workflow for using the CompositeBaseModel framework.
    
    This function shows:
    1. Model creation and initialization
    2. Configuration and setup
    3. Inference execution with different input types
    4. Performance monitoring and optimization
    5. Model saving and loading (serialization)
    6. Training lifecycle management
    7. System status monitoring
    """
    print("=" * 80)
    print("COMPOSITE BASE MODEL FRAMEWORK - COMPLETE USAGE EXAMPLE")
    print("=" * 80)
    
    # 1. Create temporary directory for model files
    temp_dir = tempfile.mkdtemp(prefix="model_example_")
    print(f"\n1. Created temporary directory: {temp_dir}")
    
    # 2. Model configuration
    model_config = {
        'cache_enabled': True,
        'cache_ttl': 300,
        'max_retries': 3,
        'enable_performance_monitoring': True,
        'enable_resource_monitoring': True,
        'enable_from_scratch_training': True,
        'model_type': 'example'
    }
    
    # 3. Create and initialize the model
    print("\n2. Creating and initializing the model...")
    model = ExampleSpecializedModel(model_config=model_config)
    
    # 4. Initialize with custom data
    initialization_data = {
        'weights': [0.1, 0.2, 0.3, 0.4],
        'feature_extractor': 'advanced_cnn'
    }
    
    print("\n3. Initializing model with custom data...")
    success = model.initialize(initialization_data)
    print(f"   Initialization {'successful' if success else 'failed'}")
    
    # 5. Perform inference with different input types
    print("\n4. Performing inference with different input types...")
    
    # Text input
    print("\n   a) Text input inference:")
    text_result = model.process_input("This is a great example of the framework's capabilities!")
    print(f"   Result: {text_result}")
    
    # List input
    print("\n   b) List input inference:")
    list_result = model.process_input(["item1", "item2", "item3", "item4"])
    print(f"   Result: {list_result}")
    
    # Dictionary input
    print("\n   c) Dictionary input inference:")
    dict_result = model.process_input({"name": "example", "value": 42, "active": True})
    print(f"   Result: {dict_result}")
    
    # 6. Perform optimized inference
    print("\n5. Performing optimized inference with enhanced features...")
    optimized_result = model.perform_optimized_inference(
        "Optimized inference example",
        use_cache=True,
        cache_ttl=600,
        enable_performance_monitoring=True,
        enable_agi_enhancement=True,
        max_retries=2
    )
    print(f"   Optimized inference completed with metadata")
    if '_optimization_metadata' in optimized_result:
        metadata = optimized_result['_optimization_metadata']
        print(f"   Execution time: {metadata.get('execution_time', 0):.4f}s")
        print(f"   Cache used: {metadata.get('cache_used', False)}")
        print(f"   AGI enhancement: {metadata.get('agi_enhancement_applied', False)}")
    
    # 7. Save model state
    print("\n6. Saving model state...")
    
    # Save to JSON format
    json_path = os.path.join(temp_dir, "model_state.json")
    success = model.save_model_state(json_path, format='json')
    print(f"   JSON save {'successful' if success else 'failed'}: {json_path}")
    
    # Save to H5 format (if h5py available)
    try:
        h5_path = os.path.join(temp_dir, "model_state.h5")
        success = model.save_model_state(h5_path, format='h5')
        print(f"   H5 save {'successful' if success else 'failed'}: {h5_path}")
    except ImportError:
        print(f"   H5 save skipped (h5py not installed)")
    
    # 8. Load model state
    print("\n7. Loading model state...")
    
    # Create a new model instance
    model2 = ExampleSpecializedModel(model_config=model_config)
    
    # Load from JSON
    success = model2.load_model_state(json_path, format='json')
    print(f"   JSON load {'successful' if success else 'failed'}")
    
    # Verify loaded model works
    if success:
        test_result = model2.process_input("Test after loading")
        print(f"   Loaded model inference: {test_result}")
    
    # 9. Get system status
    print("\n8. Getting comprehensive system status...")
    system_status = model.get_system_status()
    
    print(f"   System health: {system_status.get('system_health', 'unknown')}")
    print(f"   Model info: {system_status.get('model_info', {}).get('model_name', 'unknown')}")
    
    # Display key performance metrics
    if 'performance_metrics' in system_status:
        metrics = system_status['performance_metrics']
        print(f"   Performance metrics:")
        print(f"     Total requests: {metrics.get('total_requests', 0)}")
        print(f"     Successful: {metrics.get('successful_requests', 0)}")
        print(f"     Failed: {metrics.get('failed_requests', 0)}")
        print(f"     Avg response time: {metrics.get('average_response_time', 0):.4f}s")
    
    # 10. Perform auto-optimization
    print("\n9. Performing auto-optimization...")
    optimization_results = model.auto_optimize()
    print(f"   Optimization completed:")
    
    for key, result in optimization_results.items():
        if 'error' not in key:
            print(f"     {key}: {result}")
    
    # 11. Training lifecycle demonstration
    print("\n10. Demonstrating training lifecycle...")
    
    # Start training session
    model.start_training_session(
        training_data="example_training_data",
        session_name="example_session",
        training_type="from_scratch"
    )
    
    # Simulate training epochs
    for epoch in range(3):
        model.update_training_progress(epoch=epoch, loss=1.0/(epoch+1))
        print(f"   Epoch {epoch}: loss={1.0/(epoch+1):.4f}")
    
    # Complete training session
    model.complete_training_session(success=True)
    
    # Get training history
    training_history = model.get_training_history()
    print(f"   Total training sessions: {len(training_history)}")
    
    # 12. Demonstrate error handling
    print("\n11. Demonstrating error handling...")
    
    # Simulate an error (will be handled by error handling mixin)
    try:
        # This should trigger error handling
        model.process_input(None)
    except Exception as e:
        print(f"   Error handled: {type(e).__name__}: {e}")
    
    # Get error history
    if hasattr(model, 'error_history') and model.error_history:
        print(f"   Recent errors: {len(model.error_history)}")
        for i, error in enumerate(model.error_history[-3:]):
            print(f"     Error {i+1}: {error.get('error_type', 'unknown')}")
    
    # 13. Demonstrate caching
    print("\n12. Demonstrating caching...")
    
    # Enable caching
    model.enable_caching(True)
    
    # Process the same input twice - second should be faster due to caching
    input_data = "This input should be cached"
    
    # First inference (cache miss)
    result1 = model.process_input(input_data)
    print(f"   First inference (cache miss)")
    
    # Second inference (cache hit)
    result2 = model.process_input(input_data)
    print(f"   Second inference (cache hit)")
    
    # Get cache statistics
    if hasattr(model, 'get_cache_stats'):
        cache_stats = model.get_cache_stats()
        print(f"   Cache stats: hits={cache_stats.get('hits', 0)}, misses={cache_stats.get('misses', 0)}")
    
    # 14. Cleanup
    print("\n13. Cleaning up...")
    
    # Shutdown model
    shutdown_success = model.shutdown()
    print(f"   Model shutdown {'successful' if shutdown_success else 'failed'}")
    
    # Cleanup temporary files
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"   Temporary directory cleaned: {temp_dir}")
    except Exception as e:
        print(f"   Failed to cleanup temporary directory: {e}")
    
    print("\n" + "=" * 80)
    print("COMPLETE WORKFLOW DEMONSTRATION FINISHED")
    print("=" * 80)
    
    return True


def run_example_as_main():
    """Run the example if this file is executed as main."""
    try:
        success = demonstrate_complete_workflow()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Example failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_example_as_main())