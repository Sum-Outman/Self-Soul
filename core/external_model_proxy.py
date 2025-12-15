"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
Internal Model Proxy - For connecting and managing internal AGI models
"""

import logging
import json
from typing import Dict, Any, Optional
from .error_handling import error_handler


class InternalModelProxy:
    """Internal Model Proxy Class
    
    Function: Proxy internal AGI models, providing unified interface for system calls
    Uses local models instead of external APIs for complete AGI compliance
    """
    
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """Initialize internal model proxy
        
        Args:
            model_id: Identifier for the model type (e.g., 'language', 'audio', 'vision')
            model_config: Configuration parameters for the model
        """
        self.logger = logging.getLogger(__name__)
        self.model_id = model_id
        self.model_config = model_config
        self.model_name = model_config.get('model_name', model_id)
        
        # Internal model instance
        self.internal_model = None
        
        self.logger.info(f"Internal model proxy initialized: {model_id}")
    
    def _load_internal_model(self) -> bool:
        """Load the appropriate internal model based on model_id
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # 使用绝对导入以解决Pylance的静态分析问题
            # Use absolute imports to fix Pylance static analysis issues
            import sys
            import os
            
            # 添加项目根目录到Python路径
            # Add project root directory to Python path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            if self.model_id.startswith('language'):
                from core.models.language import UnifiedLanguageModel  # type: ignore
                self.internal_model = UnifiedLanguageModel(self.model_config)
            elif self.model_id.startswith('audio'):
                from core.models.audio import UnifiedAudioModel  # type: ignore
                self.internal_model = UnifiedAudioModel(self.model_config)
            elif self.model_id.startswith('image') or self.model_id.startswith('vision'):
                from core.models.vision import UnifiedVisionModel  # type: ignore
                self.internal_model = UnifiedVisionModel(self.model_config)
            elif self.model_id.startswith('video'):
                from core.models.video import UnifiedVideoModel  # type: ignore
                self.internal_model = UnifiedVideoModel(self.model_config)
            elif self.model_id.startswith('knowledge'):
                from core.models.knowledge import UnifiedKnowledgeModel  # type: ignore
                self.internal_model = UnifiedKnowledgeModel(self.model_config)
            else:
                # Default to composite base model for unknown types
                from core.models.base import CompositeBaseModel  # type: ignore
                self.internal_model = CompositeBaseModel(self.model_config)
            
            # Initialize the model
            init_result = self.internal_model.initialize()
            if not init_result.get('success', False):
                error_handler.log_warning(f"Failed to initialize internal model: {init_result.get('error', 'Unknown error')}", "InternalModelProxy")
                return False
            
            return True
            
        except ImportError as e:
            error_handler.handle_error(e, "InternalModelProxy", f"Failed to import internal model for {self.model_id}")
            return False
        except Exception as e:
            error_handler.handle_error(e, "InternalModelProxy", f"Error loading internal model: {str(e)}")
            return False
    
    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process input data using internal model
        
        Args:
            input_data: Input data to process
            
        Returns:
            Optional[Dict[str, Any]]: Processing result
        """
        try:
            # Load model if not already loaded
            if self.internal_model is None:
                if not self._load_internal_model():
                    return None
            
            # Process using internal model
            return self.internal_model.process(input_data)
            
        except Exception as e:
            error_handler.handle_error(e, "InternalModelProxy", f"Error processing data with internal model: {str(e)}")
            return None
    
    def train(self, training_data: Dict[str, Any] = None, callback=None) -> Dict[str, Any]:
        """Train the internal model
        
        Args:
            training_data: Training data
            callback: Progress callback function
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            # Load model if not already loaded
            if self.internal_model is None:
                if not self._load_internal_model():
                    return {
                        'status': 'error',
                        'message': 'Failed to load internal model for training',
                        'model_id': self.model_id
                    }
            
            # Train the model
            train_result = self.internal_model.train(training_data)
            
            if callback:
                callback(100, {'status': 'completed', 'message': 'Training completed successfully'})
            
            return {
                'status': 'completed',
                'message': 'Training completed successfully',
                'model_id': self.model_id,
                'results': train_result
            }
            
        except Exception as e:
            error_handler.handle_error(e, "InternalModelProxy", f"Error training internal model: {str(e)}")
            
            if callback:
                callback(100, {'status': 'error', 'message': f'Training failed: {str(e)}'})
            
            return {
                'status': 'error',
                'message': f'Training failed: {str(e)}',
                'model_id': self.model_id
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status
        
        Returns:
            Dict[str, Any]: Status information
        """
        try:
            # Load model if not already loaded
            if self.internal_model is None:
                if not self._load_internal_model():
                    return {
                        'model_id': self.model_id,
                        'status': 'error',
                        'model_type': 'internal',
                        'model_name': self.model_name,
                        'error': 'Failed to load internal model'
                    }
            
            # Get status from internal model
            model_status = self.internal_model.get_status()
            
            return {
                'model_id': self.model_id,
                'status': 'ready',
                'model_type': 'internal',
                'model_name': self.model_name,
                'internal_status': model_status
            }
            
        except Exception as e:
            return {
                'model_id': self.model_id,
                'status': 'error',
                'model_type': 'internal',
                'model_name': self.model_name,
                'error': f'Status check failed: {str(e)}'
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.internal_model:
            try:
                # Call cleanup on internal model if available
                if hasattr(self.internal_model, 'cleanup'):
                    self.internal_model.cleanup()
            except Exception as e:
                error_handler.log_warning(f"Error during model cleanup: {str(e)}", "InternalModelProxy")
        
        self.logger.info(f"Internal model proxy cleanup completed: {self.model_id}")

# Export class
InternalModelProxy = InternalModelProxy
