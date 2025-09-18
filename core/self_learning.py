"""
Self Learning Module - Core component for AGI self-learning capabilities
This module provides the interface for self-learning functionalities
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfLearningModule:
    """
    Self Learning Module for AGI system
    Handles basic self-learning operations and serves as an interface
    to the advanced self-learning system
    """
    
    def __init__(self):
        self.initialized = False
        self.learning_enabled = True
        self.last_learning_time = None
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0
        }
    
    def initialize(self) -> bool:
        """
        Initialize the self-learning module
        Returns: True if successful, False otherwise
        """
        try:
            # Try to import advanced self-learning if available
            try:
                from core.advanced_self_learning import AdvancedSelfLearningSystem
                self.advanced_system = AdvancedSelfLearningSystem()
                logger.info("Advanced self-learning system linked")
            except ImportError:
                self.advanced_system = None
                logger.warning("Advanced self-learning system not available, using basic mode")
            
            self.initialized = True
            self.last_learning_time = datetime.now()
            logger.info("Self Learning Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Self Learning Module: {e}")
            return False
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Learn from a single interaction
        Args:
            interaction_data: Dictionary containing interaction details
        Returns: True if learning was successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Self Learning Module not initialized")
            return False
        
        try:
            self.learning_stats['total_learning_sessions'] += 1
            
            # Use advanced system if available
            if self.advanced_system:
                success = self.advanced_system.process_interaction(interaction_data)
            else:
                # Basic learning logic
                success = self._basic_learning(interaction_data)
            
            if success:
                self.learning_stats['successful_learnings'] += 1
            else:
                self.learning_stats['failed_learnings'] += 1
            
            self.last_learning_time = datetime.now()
            return success
            
        except Exception as e:
            logger.error(f"Error in learning from interaction: {e}")
            self.learning_stats['failed_learnings'] += 1
            return False
    
    def _basic_learning(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Basic learning implementation
        Args:
            interaction_data: Interaction data to learn from
        Returns: True if successful
        """
        # Simple learning logic - can be enhanced later
        logger.info(f"Basic learning from interaction: {interaction_data.get('type', 'unknown')}")
        return True
    
    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get current learning status and statistics
        Returns: Dictionary with learning status
        """
        return {
            'initialized': self.initialized,
            'learning_enabled': self.learning_enabled,
            'last_learning_time': self.last_learning_time.isoformat() if self.last_learning_time else None,
            'statistics': self.learning_stats.copy(),
            'advanced_system_available': self.advanced_system is not None
        }
    
    def enable_learning(self, enable: bool = True) -> None:
        """
        Enable or disable learning
        Args:
            enable: True to enable, False to disable
        """
        self.learning_enabled = enable
        logger.info(f"Learning {'enabled' if enable else 'disabled'}")
    
    def reset_learning_stats(self) -> None:
        """Reset learning statistics"""
        self.learning_stats = {
            'total_learning_sessions': 0,
            'successful_learnings': 0,
            'failed_learnings': 0
        }
        logger.info("Learning statistics reset")

# Global instance for easy access
self_learning_module = SelfLearningModule()

def initialize_self_learning() -> bool:
    """Initialize the global self learning module"""
    return self_learning_module.initialize()

def learn_from_interaction(interaction_data: Dict[str, Any]) -> bool:
    """Learn from interaction using global module"""
    return self_learning_module.learn_from_interaction(interaction_data)

def get_learning_status() -> Dict[str, Any]:
    """Get learning status from global module"""
    return self_learning_module.get_learning_status()
