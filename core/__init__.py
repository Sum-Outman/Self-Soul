"""
# Self Soul AGI System Core Package
# This package contains the core components of the Self Soul AGI system.
"""

__version__ = "1.0.0"
__author__ = "Self Soul Team"
__email__ = "silencecrowtom@qq.com"

# Import error_handling module to make it available
from . import error_handling
from .error_handling import error_handler

# Import new training modules for easy access
from .training_scheduler import TrainingScheduler, TrainingJob
from .resource_manager import ResourceManager
from .training_monitor import TrainingMonitor, AlertSeverity
from .data_preprocessor import DataPreprocessor, DataPreprocessorConfig, DataType, AugmentationType, DatasetSplit
from .training_manager import TrainingManager, get_training_manager
