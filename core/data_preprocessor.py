import zlib
"""
Data Preprocessor Module - Specialized data preprocessing for training workflows

Provides data loading, augmentation, normalization, batching, and dataset splitting
capabilities specifically for model training. Works in conjunction with existing
DataProcessor for multimodal data handling.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import time
from dataclasses import dataclass
from enum import Enum

# Import existing data processor for multimodal support
from core.data_processor import DataProcessor, DataType

logger = logging.getLogger(__name__)

class AugmentationType(Enum):
    """Types of data augmentation"""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class DatasetSplit(Enum):
    """Dataset split types"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"

@dataclass
class DataPreprocessorConfig:
    """Configuration for data preprocessing"""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Augmentation settings
    augmentation_level: AugmentationType = AugmentationType.BASIC
    enable_random_crop: bool = True
    enable_random_flip: bool = True
    enable_color_jitter: bool = False
    enable_random_rotation: bool = False
    
    # Normalization settings
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None
    normalize_enabled: bool = True
    
    # Dataset split ratios
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cache settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_dir: str = "data_cache"

class DataPreprocessor:
    """Specialized data preprocessor for training workflows"""
    
    def __init__(self, config: Optional[DataPreprocessorConfig] = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Preprocessor configuration
        """
        self.config = config or DataPreprocessorConfig()
        self.data_processor = DataProcessor()
        self.cache = {}
        self.datasets = {}
        
        # Create cache directory if needed
        if self.config.enable_caching:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info("Data Preprocessor initialized")
    
    def load_dataset(self, 
                    dataset_path: Union[str, Path, List],
                    data_type: DataType,
                    labels: Optional[List] = None,
                    dataset_name: str = "default") -> Dict[str, Any]:
        """
        Load dataset from various sources
        
        Args:
            dataset_path: Path to dataset or list of data items
            data_type: Type of data in dataset
            labels: Optional labels for supervised learning
            dataset_name: Name for referencing this dataset
            
        Returns:
            Dictionary containing loaded dataset metadata
        """
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{dataset_name}_{data_type.value}"
            if self.config.enable_caching and cache_key in self.cache:
                logger.info(f"Loading dataset '{dataset_name}' from cache")
                return self.cache[cache_key]
            
            dataset_info = {
                'name': dataset_name,
                'data_type': data_type,
                'source': dataset_path,
                'load_time': None,
                'samples': 0,
                'labels_available': labels is not None,
                'processed_data': None
            }
            
            # Load data based on type and path
            if isinstance(dataset_path, (str, Path)):
                # Load from file or directory
                loaded_data = self._load_from_path(dataset_path, data_type)
            elif isinstance(dataset_path, list):
                # Already a list of data items
                loaded_data = dataset_path
            else:
                raise ValueError(f"Unsupported dataset_path type: {type(dataset_path)}")
            
            # Process data using DataProcessor
            processed_data = []
            for item in loaded_data:
                try:
                    processed_item = self.data_processor.process(
                        item, data_type, target_type=data_type
                    )
                    processed_data.append(processed_item)
                except Exception as e:
                    logger.warning(f"Failed to process data item: {e}")
                    continue
            
            dataset_info['processed_data'] = processed_data
            dataset_info['samples'] = len(processed_data)
            dataset_info['load_time'] = time.time() - start_time
            
            # Store labels if provided
            if labels is not None:
                if len(labels) != len(processed_data):
                    logger.warning(f"Label count ({len(labels)}) doesn't match data count ({len(processed_data)})")
                else:
                    dataset_info['labels'] = labels
            
            # Cache the dataset
            if self.config.enable_caching:
                self.cache[cache_key] = dataset_info
                self._save_to_cache(cache_key, dataset_info)
            
            logger.info(f"Loaded dataset '{dataset_name}' with {len(processed_data)} samples "
                       f"in {dataset_info['load_time']:.2f}s")
            
            self.datasets[dataset_name] = dataset_info
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            return {
                'success': False,
                'error': str(e),
                'name': dataset_name
            }
    
    def _load_from_path(self, path: Union[str, Path], data_type: DataType) -> List[Any]:
        """
        Load data from file system path
        
        Args:
            path: Path to data file or directory
            data_type: Type of data to load
            
        Returns:
            List of loaded data items
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        if path.is_file():
            # Single file
            return [str(path)]
        elif path.is_dir():
            # Directory with multiple files
            data_files = []
            
            # Define file extensions based on data type
            extensions = {
                DataType.TEXT: ['.txt', '.json', '.csv'],
                DataType.IMAGE: ['.jpg', '.jpeg', '.png', '.bmp'],
                DataType.AUDIO: ['.wav', '.mp3', '.flac'],
                DataType.VIDEO: ['.mp4', '.avi', '.mov'],
                DataType.JSON: ['.json']
            }
            
            # Get relevant extensions
            relevant_ext = extensions.get(data_type, [])
            
            # Recursively search for files
            for ext in relevant_ext:
                data_files.extend(path.rglob(f'*{ext}'))
            
            if not data_files:
                # If no specific extensions found, try to find all files
                data_files = list(path.rglob('*'))
                data_files = [f for f in data_files if f.is_file()]
            
            return [str(f) for f in data_files]
        else:
            raise ValueError(f"Unsupported path type: {path}")
    
    def apply_augmentation(self, data: Any, data_type: DataType, 
                          augmentation_level: AugmentationType = None) -> Any:
        """
        Apply data augmentation based on data type
        
        Args:
            data: Input data to augment
            data_type: Type of data
            augmentation_level: Level of augmentation to apply
            
        Returns:
            Augmented data
        """
        if augmentation_level is None:
            augmentation_level = self.config.augmentation_level
        
        if augmentation_level == AugmentationType.NONE:
            return data
        
        try:
            if data_type == DataType.IMAGE:
                return self._augment_image(data, augmentation_level)
            elif data_type == DataType.AUDIO:
                return self._augment_audio(data, augmentation_level)
            elif data_type == DataType.TEXT:
                return self._augment_text(data, augmentation_level)
            else:
                # No augmentation for other data types
                return data
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return data
    
    def _augment_image(self, image, augmentation_level: AugmentationType):
        """
        Augment image data
        """
        # Import here to avoid dependency if not used
        from PIL import Image, ImageOps, ImageEnhance
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        augmented = image
        
        # Basic augmentation
        if augmentation_level.value >= AugmentationType.BASIC.value:
            if self.config.enable_random_flip and ((zlib.adler32(str(str(image.tobytes().encode('utf-8')) & 0xffffffff)) + "flip") % 100) < 50:
                augmented = ImageOps.mirror(augmented)
        
        # Moderate augmentation
        if augmentation_level.value >= AugmentationType.MODERATE.value:
            if self.config.enable_random_rotation and ((zlib.adler32(str(str(image.tobytes().encode('utf-8')) & 0xffffffff)) + "rotation") % 100) < 50:
                # 确定性旋转角度基于图像宽度
                angle = (image.width % 30) - 15  # -15到15度范围内
                augmented = augmented.rotate(angle, expand=True)
        
        # Aggressive augmentation
        if augmentation_level.value >= AugmentationType.AGGRESSIVE.value:
            if self.config.enable_color_jitter and ((zlib.adler32(str(str(image.tobytes().encode('utf-8')) & 0xffffffff)) + "color") % 100) < 50:
                enhancer = ImageEnhance.Color(augmented)
                # 确定性颜色增强因子基于图像高度
                enhancement_factor = 0.8 + (image.height % 40) * 0.01  # 0.8到1.2范围内
                augmented = enhancer.enhance(enhancement_factor)
        
        return augmented
    
    def _augment_audio(self, audio, augmentation_level: AugmentationType):
        """
        Augment audio data
        """
        
        # In practice, would apply time stretching, pitch shifting, etc.
        return audio
    
    def _augment_text(self, text, augmentation_level: AugmentationType):
        """
        Augment text data
        """
        
        # In practice, would apply synonym replacement, backtranslation, etc.
        return text
    
    def normalize_data(self, data: Any, data_type: DataType) -> Any:
        """
        Normalize data based on type and configuration
        
        Args:
            data: Input data
            data_type: Type of data
            
        Returns:
            Normalized data
        """
        if not self.config.normalize_enabled:
            return data
        
        try:
            if data_type == DataType.IMAGE:
                return self._normalize_image(data)
            elif data_type in [DataType.AUDIO, DataType.SENSOR, DataType.SPATIAL]:
                return self._normalize_numeric(data)
            else:
                return data
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return data
    
    def _normalize_image(self, image):
        """
        Normalize image data
        """
        # Convert to tensor if not already
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
        else:
            # Assume PIL Image
            import torchvision.transforms as transforms
            transform = transforms.ToTensor()
            tensor = transform(image)
        
        # Apply normalization if means and stds are provided
        if self.config.normalize_mean and self.config.normalize_std:
            normalize = transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
            tensor = normalize(tensor)
        
        return tensor
    
    def _normalize_numeric(self, data):
        """
        Normalize numeric data to [0, 1] range
        """
        if isinstance(data, np.ndarray):
            data_min = data.min()
            data_max = data.max()
            
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = np.zeros_like(data)
            
            return normalized
        elif isinstance(data, torch.Tensor):
            data_min = data.min()
            data_max = data.max()
            
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = torch.zeros_like(data)
            
            return normalized
        else:
            # For scalar or list
            data_array = np.array(data)
            return self._normalize_numeric(data_array)
    
    def split_dataset(self, 
                     dataset_info: Dict[str, Any],
                     split_ratios: Optional[Dict[DatasetSplit, float]] = None) -> Dict[DatasetSplit, Dict[str, Any]]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            dataset_info: Dataset information from load_dataset
            split_ratios: Custom split ratios
            
        Returns:
            Dictionary of split datasets
        """
        try:
            if split_ratios is None:
                split_ratios = {
                    DatasetSplit.TRAIN: self.config.train_ratio,
                    DatasetSplit.VALIDATION: self.config.validation_ratio,
                    DatasetSplit.TEST: self.config.test_ratio
                }
            
            # Validate split ratios
            total_ratio = sum(split_ratios.values())
            if abs(total_ratio - 1.0) > 0.001:
                raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
            
            processed_data = dataset_info.get('processed_data', [])
            labels = dataset_info.get('labels', None)
            
            if not processed_data:
                raise ValueError("No data to split")
            
            # Shuffle indices deterministically
            indices = np.arange(len(processed_data))
            indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(processed_data).encode('utf-8')) & 0xffffffff) + str(x) + "shuffle"))
            
            # Calculate split points
            n_total = len(indices)
            n_train = int(n_total * split_ratios[DatasetSplit.TRAIN])
            n_val = int(n_total * split_ratios[DatasetSplit.VALIDATION])
            
            # Split indices
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            # Create split datasets
            splits = {}
            for split_name, split_indices in [
                (DatasetSplit.TRAIN, train_indices),
                (DatasetSplit.VALIDATION, val_indices),
                (DatasetSplit.TEST, test_indices)
            ]:
                if len(split_indices) == 0:
                    logger.warning(f"No samples in {split_name} split")
                    continue
                
                split_data = [processed_data[i] for i in split_indices]
                
                split_info = {
                    'name': f"{dataset_info['name']}_{split_name.value}",
                    'data_type': dataset_info['data_type'],
                    'samples': len(split_data),
                    'indices': split_indices,
                    'processed_data': split_data
                }
                
                if labels is not None:
                    split_labels = [labels[i] for i in split_indices]
                    split_info['labels'] = split_labels
                
                splits[split_name] = split_info
            
            logger.info(f"Dataset split: {len(train_indices)} train, "
                       f"{len(val_indices)} validation, {len(test_indices)} test")
            
            return splits
            
        except Exception as e:
            logger.error(f"Dataset split failed: {e}")
            return {}
    
    def create_data_loader(self, 
                          split_info: Dict[str, Any],
                          batch_size: Optional[int] = None,
                          shuffle: Optional[bool] = None) -> DataLoader:
        """
        Create PyTorch DataLoader for a dataset split
        
        Args:
            split_info: Dataset split information
            batch_size: Batch size (uses config if None)
            shuffle: Whether to shuffle data (uses config if None)
            
        Returns:
            PyTorch DataLoader
        """
        try:
            batch_size = batch_size or self.config.batch_size
            shuffle = shuffle if shuffle is not None else self.config.shuffle
            
            # Create custom dataset
            dataset = self._create_torch_dataset(split_info)
            
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor,
                persistent_workers=self.config.persistent_workers
            )
            
            logger.info(f"Created DataLoader with {len(dataset)} samples, "
                       f"batch size {batch_size}, shuffle={shuffle}")
            
            return data_loader
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise
    
    def _create_torch_dataset(self, split_info: Dict[str, Any]) -> Dataset:
        """
        Create PyTorch Dataset from split information
        
        Args:
            split_info: Dataset split information
            
        Returns:
            PyTorch Dataset
        """
        class CustomDataset(Dataset):
            def __init__(self, split_info, preprocessor):
                self.data = split_info.get('processed_data', [])
                self.labels = split_info.get('labels', None)
                self.data_type = split_info.get('data_type')
                self.preprocessor = preprocessor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                data_item = self.data[idx]
                
                # Apply augmentation (only for training)
                if 'train' in split_info.get('name', '').lower():
                    data_item = self.preprocessor.apply_augmentation(
                        data_item, self.data_type
                    )
                
                # Apply normalization
                data_item = self.preprocessor.normalize_data(data_item, self.data_type)
                
                if self.labels is not None:
                    label = self.labels[idx]
                    return data_item, label
                else:
                    return data_item
        
        return CustomDataset(split_info, self)
    
    def preprocess_training_data(self,
                                dataset_path: Union[str, Path, List],
                                data_type: DataType,
                                labels: Optional[List] = None,
                                dataset_name: str = "training_data") -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for training data
        
        Args:
            dataset_path: Path to dataset or list of data items
            data_type: Type of data in dataset
            labels: Optional labels for supervised learning
            dataset_name: Name for this dataset
            
        Returns:
            Dictionary containing preprocessed datasets and loaders
        """
        try:
            start_time = time.time()
            
            # 1. Load dataset
            dataset_info = self.load_dataset(
                dataset_path, data_type, labels, dataset_name
            )
            
            if dataset_info.get('samples', 0) == 0:
                raise ValueError(f"No data loaded for dataset '{dataset_name}'")
            
            # 2. Split dataset
            splits = self.split_dataset(dataset_info)
            
            # 3. Create data loaders
            loaders = {}
            for split_name, split_info in splits.items():
                loader = self.create_data_loader(
                    split_info,
                    shuffle=(split_name == DatasetSplit.TRAIN)
                )
                loaders[split_name] = loader
            
            # 4. Collect statistics
            stats = {
                'total_samples': dataset_info['samples'],
                'split_samples': {k.value: v['samples'] for k, v in splits.items()},
                'data_type': data_type.value,
                'processing_time': time.time() - start_time
            }
            
            result = {
                'success': True,
                'dataset_info': dataset_info,
                'splits': splits,
                'data_loaders': loaders,
                'statistics': stats,
                'preprocessing_time': stats['processing_time']
            }
            
            logger.info(f"Complete preprocessing finished in {stats['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Training data preprocessing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'preprocessing_time': time.time() - start_time
            }
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """
        Save data to cache directory
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        try:
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.npy"
            
            # Convert data to numpy-serializable format
            if isinstance(data, dict):
                # For simplicity, we'll cache processed_data separately
                if 'processed_data' in data:
                    # We can't cache arbitrary objects easily
                    # In practice, you might want to cache to disk differently
                    pass
            
            logger.debug(f"Cached data with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self.cache.clear()
        logger.info("In-memory cache cleared")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing activities
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'datasets_loaded': len(self.datasets),
            'dataset_names': list(self.datasets.keys()),
            'cache_enabled': self.config.enable_caching,
            'cache_size': len(self.cache),
            'config': {
                'batch_size': self.config.batch_size,
                'augmentation_level': self.config.augmentation_level.value,
                'normalize_enabled': self.config.normalize_enabled
            }
        }
        
        # Add dataset statistics
        for name, info in self.datasets.items():
            summary[f'dataset_{name}'] = {
                'samples': info.get('samples', 0),
                'data_type': info.get('data_type', 'unknown').value,
                'labels_available': info.get('labels_available', False)
            }
        
        return summary

# Factory function for easy instantiation
def create_data_preprocessor(config: Optional[DataPreprocessorConfig] = None) -> DataPreprocessor:
    """
    Create and initialize a DataPreprocessor instance
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialized DataPreprocessor
    """
    try:
        preprocessor = DataPreprocessor(config)
        logger.info("DataPreprocessor instance created successfully")
        return preprocessor
    except Exception as e:
        logger.error(f"Failed to create DataPreprocessor: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = DataPreprocessorConfig(
        batch_size=16,
        augmentation_level=AugmentationType.BASIC,
        enable_random_flip=True,
        normalize_enabled=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    )
    
    # Create preprocessor
    preprocessor = create_data_preprocessor(config)
    
    # Example: Preprocess a dataset
    try:
        # Example usage - provide real dataset paths for actual processing
        result = preprocessor.preprocess_training_data(
            dataset_path="path/to/real/dataset",
            data_type=DataType.IMAGE,
            dataset_name="example_dataset"
        )
        
        if result['success']:
            print(f"Preprocessing successful:")
            print(f"  Total samples: {result['statistics']['total_samples']}")
            print(f"  Processing time: {result['preprocessing_time']:.2f}s")
            
            # Access data loaders
            train_loader = result['data_loaders'].get('train')
            if train_loader:
                print(f"  Train batches: {len(train_loader)}")
    except Exception as e:
        print(f"Example preprocessing failed: {e}")
