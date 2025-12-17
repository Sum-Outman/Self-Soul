"""
Unified Spatial Model - Based on UnifiedModelTemplate

This model provides spatial perception, visual spatial modeling, spatial positioning,
distance perception, volume recognition, and moving object tracking capabilities
using the unified template architecture.
"""

import logging
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
import sys
import os

# Add the root directory to Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.models.unified_model_template import UnifiedModelTemplate
from core.data_processor import preprocess_stereo_images
from core.unified_stream_processor import StreamProcessor
from core.error_handling import error_handler
from core.agi_tools import AGITools


class SpatialNeuralNetwork(nn.Module):
    """
    Neural network for spatial perception tasks including stereo vision, depth estimation,
    object detection, and motion tracking.
    """
    
    def __init__(self, input_channels=6, hidden_size=256, num_layers=3, output_size=128):
        super(SpatialNeuralNetwork, self).__init__()
        
        # Convolutional layers for feature extraction from stereo images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # LSTM for temporal/spatial sequence processing
        self.lstm = nn.LSTM(256 * 16 * 16, hidden_size, num_layers, dropout=0.2)
        
        # Attention mechanism for spatial reasoning
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1)
        
        # Output layers for different spatial tasks
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Depth estimation
            nn.Sigmoid()
        )
        
        self.object_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)  # Object detection (position + size + type)
        )
        
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)   # Motion vector (x, y, z)
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)  # Flatten
        
        # LSTM processing (treat as sequence of 1)
        lstm_input = features.unsqueeze(0)  # Add sequence dimension (seq_len=1, batch_size, features)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        lstm_out = lstm_out.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, features)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use last hidden state for predictions
        context = attn_out[:, -1, :]
        context = self.dropout(context)
        
        # Multi-task outputs
        depth_pred = self.depth_head(context)
        object_pred = self.object_head(context)
        motion_pred = self.motion_head(context)
        
        return {
            'depth': depth_pred,
            'objects': object_pred,
            'motion': motion_pred,
            'attention_weights': attn_weights
        }


class SpatialDataset(Dataset):
    """
    Dataset for spatial perception training data.
    Handles stereo image pairs, depth maps, object annotations, and motion data.
    """
    
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        data = self.data_pairs[idx]
        
        # Extract stereo images
        left_img = data.get('left_image', np.zeros((480, 640, 3), dtype=np.float32))
        right_img = data.get('right_image', np.zeros((480, 640, 3), dtype=np.float32))
        
        # Convert to tensor and combine stereo pair
        if isinstance(left_img, np.ndarray):
            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float() / 255.0
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float() / 255.0
        else:
            left_tensor = left_img
            right_tensor = right_img
        
        # Combine stereo pair along channel dimension
        stereo_pair = torch.cat([left_tensor, right_tensor], dim=0)
        
        # Prepare targets
        depth_target = torch.tensor(data.get('depth_map', 0.0), dtype=torch.float32)
        object_target = torch.tensor(data.get('object_data', [0]*10), dtype=torch.float32)
        motion_target = torch.tensor(data.get('motion_data', [0, 0, 0]), dtype=torch.float32)
        
        return {
            'stereo_pair': stereo_pair,
            'depth_target': depth_target,
            'object_target': object_target,
            'motion_target': motion_target
        }


class UnifiedSpatialModel(UnifiedModelTemplate):
    """
    Unified spatial perception model providing stereo vision, spatial mapping,
    object tracking, and distance perception capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Spatial perception configuration
        self.camera_baseline = config.get("camera_baseline", 0.12) if config else 0.12
        self.focal_length = config.get("focal_length", 800) if config else 800
        self.min_depth = config.get("min_depth", 0.1) if config else 0.1
        self.max_depth = config.get("max_depth", 20.0) if config else 20.0
        
        # Spatial modeling configuration
        self.grid_resolution = config.get("grid_resolution", 0.01) if config else 0.01
        self.map_size = config.get("map_size", (10, 10, 3)) if config else (10, 10, 3)
        
        # Stereo matcher
        self.stereo = None
        
        # Spatial data
        self.spatial_map = np.zeros(self.map_size, dtype=np.float32)
        self.object_tracking = {}
        
        # Self position and velocity
        self.self_position = np.array([0, 0, 0])
        self.self_velocity = np.array([0, 0, 0])
        
        # Previous frame data for motion detection
        self._prev_gray = None
        self._prev_depth = None
        
        self.logger.info("Unified spatial model initialized")

    # ===== ABSTRACT METHOD IMPLEMENTATIONS =====
    
    def _get_model_id(self) -> str:
        """Return the model identifier"""
        return "spatial"
    
    def _get_model_type(self) -> str:
        """Return the model type"""
        return "spatial"
    
    def _get_supported_operations(self) -> List[str]:
        """Return list of operations this model supports"""
        return [
            "build_spatial_map", 
            "locate_objects", 
            "track_moving_objects",
            "analyze_spatial_data",
            "export_spatial_data",
            "import_spatial_data",
            "get_spatial_status"
        ]
    
    def _initialize_model_specific_components(self, config: Dict[str, Any]):
        """Initialize model-specific components"""
        # Initialize stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=15,
            P1=8*3*15**2,
            P2=32*3*15**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Initialize spatial data structures
        self.spatial_map = np.zeros(self.map_size, dtype=np.float32)
        self.object_tracking = {}
        self.self_position = np.array([0, 0, 0])
        self.self_velocity = np.array([0, 0, 0])
        
        # Initialize neural network
        self._initialize_neural_networks()
        
        # Initialize AGI components for spatial reasoning
        self._initialize_agi_spatial_components()
        
        self.logger.info("Spatial model specific components initialized")
        
    def _initialize_neural_networks(self):
        """Initialize the neural networks used by the spatial model"""
        try:
            self.logger.info("Initializing spatial neural network")
            
            # Initialize neural network with configuration
            nn_config = self.config.get("neural_network", {})
            
            # Get network parameters from config
            input_channels = nn_config.get("input_channels", 6)
            hidden_size = nn_config.get("hidden_size", 256)
            num_layers = nn_config.get("num_layers", 3)
            output_size = nn_config.get("output_size", 128)
            
            # Create the neural network
            self.neural_network = SpatialNeuralNetwork(
                input_channels=input_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size
            )
            
            # Initialize loss functions
            self.depth_criterion = nn.MSELoss()
            self.object_criterion = nn.MSELoss()
            self.motion_criterion = nn.MSELoss()
            
            # Initialize optimizer
            learning_rate = nn_config.get("learning_rate", 0.001)
            self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
            
            # Set training state flags
            self.is_trained = False
            self.training_completed = False
            self.training_epochs_completed = 0
            
            self.logger.info("Spatial neural network initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize spatial neural network: {str(e)}")
            # Create a default network even if initialization fails
            self.neural_network = SpatialNeuralNetwork()
    
    def _process_operation(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process spatial operations with model-specific logic"""
        try:
            if operation == "build_spatial_map":
                return self._build_spatial_map_operation(input_data)
            elif operation == "locate_objects":
                return self._locate_objects_operation(input_data)
            elif operation == "track_moving_objects":
                return self._track_moving_objects_operation(input_data)
            elif operation == "analyze_spatial_data":
                return self._analyze_spatial_data_operation(input_data)
            elif operation == "export_spatial_data":
                return self._export_spatial_data_operation(input_data)
            elif operation == "import_spatial_data":
                return self._import_spatial_data_operation(input_data)
            elif operation == "get_spatial_status":
                return self._get_spatial_status_operation(input_data)
            else:
                return {"success": False, "error": f"Unknown spatial operation: {operation}"}
                
        except Exception as e:
            self.logger.error(f"Spatial operation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_stream_processor(self) -> StreamProcessor:
        """Create spatial-specific stream processor"""
        return SpatialStreamProcessor(self)
    
    def _perform_inference(self, processed_input: Any, **kwargs) -> Any:
        """Execute inference - implement abstract method required by CompositeBaseModel"""
        try:
            error_handler.log_info("Starting spatial inference", "SpatialModel")
            
            # Determine operation type
            operation = kwargs.get('operation', 'analyze_spatial_data')
            
            # Format input data
            if isinstance(processed_input, dict) and 'data' in processed_input:
                data = processed_input['data']
            else:
                data = processed_input
            
            # Use existing process method to handle operation
            result = self._process_operation(operation, data, **kwargs)
            
            # Return core inference result based on operation type
            if operation == "build_spatial_map":
                return {
                    "inference_type": "spatial_mapping",
                    "map_dimensions": result.get("spatial_map_shape", (0, 0, 0)),
                    "self_position": result.get("self_position", [0, 0, 0]),
                    "success": result.get("success", False)
                }
            elif operation == "locate_objects":
                return {
                    "inference_type": "object_detection",
                    "objects_found": result.get("object_count", 0),
                    "objects": result.get("objects", []),
                    "tracking_count": result.get("tracked_objects", 0),
                    "success": result.get("success", False)
                }
            elif operation == "track_moving_objects":
                return {
                    "inference_type": "motion_tracking",
                    "moving_objects": result.get("moving_objects", []),
                    "moving_count": result.get("moving_count", 0),
                    "success": result.get("success", False)
                }
            elif operation == "analyze_spatial_data":
                return {
                    "inference_type": "comprehensive_analysis",
                    "objects": result.get("objects", []),
                    "moving_objects": result.get("moving_objects", []),
                    "map_dimensions": result.get("spatial_map_shape", (0, 0, 0)),
                    "self_position": result.get("self_position", [0, 0, 0]),
                    "tracking_count": result.get("tracked_objects", 0),
                    "success": result.get("success", False)
                }
            elif operation in ["export_spatial_data", "import_spatial_data"]:
                return {
                    "inference_type": "data_management",
                    "success": result.get("success", False),
                    "message": result.get("message", "")
                }
            elif operation == "get_spatial_status":
                return {
                    "inference_type": "status_report",
                    "status": result.get("spatial_status", {}),
                    "success": result.get("success", False)
                }
            else:
                return {
                    "inference_type": "unknown_operation",
                    "success": False,
                    "error": f"Unsupported spatial operation: {operation}"
                }
                
        except Exception as e:
            error_handler.handle_error(e, "SpatialModel", "Spatial inference failed")
            return {"error": str(e)}

    # ===== SPATIAL OPERATION IMPLEMENTATIONS =====
    
    def _build_spatial_map_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build spatial map from stereo images"""
        left_image = input_data.get("left_image")
        right_image = input_data.get("right_image")
        context = input_data.get("context", {})
        
        if left_image is None or right_image is None:
            return {"success": False, "error": "Missing left_image or right_image"}
        
        # Preprocess stereo images
        left_img, right_img = preprocess_stereo_images(left_image, right_image)
        
        # Compute disparity and depth maps
        disparity_map = self._compute_disparity(left_img, right_img)
        depth_map = self._compute_depth(disparity_map)
        
        # Update spatial map and self position
        self._update_spatial_map(depth_map)
        self._update_self_position(context)
        
        return {
            "success": True,
            "spatial_map_shape": self.spatial_map.shape,
            "self_position": self.self_position.tolist(),
            "depth_map_available": depth_map is not None
        }
    
    def _locate_objects_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Locate objects in spatial environment"""
        left_image = input_data.get("left_image")
        right_image = input_data.get("right_image")
        
        if left_image is None or right_image is None:
            return {"success": False, "error": "Missing left_image or right_image"}
        
        # Preprocess stereo images
        left_img, right_img = preprocess_stereo_images(left_image, right_image)
        
        # Compute disparity and depth maps
        disparity_map = self._compute_disparity(left_img, right_img)
        depth_map = self._compute_depth(disparity_map)
        
        # Detect objects
        objects = self._detect_objects(left_img, depth_map)
        
        # Calculate object volumes and update tracking
        for obj in objects:
            obj["volume"] = self._calculate_volume(obj, depth_map)
        
        self._update_object_tracking(objects)
        
        return {
            "success": True,
            "objects": objects,
            "object_count": len(objects),
            "tracked_objects": len(self.object_tracking)
        }
    
    def _track_moving_objects_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track moving objects in spatial environment"""
        left_image = input_data.get("left_image")
        right_image = input_data.get("right_image")
        
        if left_image is None or right_image is None:
            return {"success": False, "error": "Missing left_image or right_image"}
        
        # Preprocess stereo images
        left_img, right_img = preprocess_stereo_images(left_image, right_image)
        
        # Compute disparity and depth maps
        disparity_map = self._compute_disparity(left_img, right_img)
        depth_map = self._compute_depth(disparity_map)
        
        # Detect moving objects
        moving_objects = self._detect_moving_objects(left_img, depth_map)
        
        # Predict movement direction
        for obj in moving_objects:
            obj["predicted_direction"] = self._predict_movement(obj)
        
        return {
            "success": True,
            "moving_objects": moving_objects,
            "moving_count": len(moving_objects)
        }
    
    def _analyze_spatial_data_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive spatial data analysis"""
        left_image = input_data.get("left_image")
        right_image = input_data.get("right_image")
        context = input_data.get("context", {})
        
        if left_image is None or right_image is None:
            return {"success": False, "error": "Missing left_image or right_image"}
        
        # Preprocess stereo images
        left_img, right_img = preprocess_stereo_images(left_image, right_image)
        
        # Compute disparity and depth maps
        disparity_map = self._compute_disparity(left_img, right_img)
        depth_map = self._compute_depth(disparity_map)
        
        # Detect objects and moving objects
        objects = self._detect_objects(left_img, depth_map)
        moving_objects = self._detect_moving_objects(left_img, depth_map)
        
        # Update spatial data
        self._update_spatial_map(depth_map)
        self._update_self_position(context)
        self._update_object_tracking(objects)
        
        return {
            "success": True,
            "objects": objects,
            "moving_objects": moving_objects,
            "spatial_map_shape": self.spatial_map.shape,
            "self_position": self.self_position.tolist(),
            "tracked_objects": len(self.object_tracking),
            "depth_map_available": depth_map is not None
        }
    
    def _export_spatial_data_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export spatial data for persistence"""
        return {
            "success": True,
            "spatial_data": {
                "spatial_map": self.spatial_map.tolist() if self.spatial_map is not None else None,
                "object_tracking": self.object_tracking,
                "self_position": self.self_position.tolist(),
                "self_velocity": self.self_velocity.tolist(),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _import_spatial_data_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import spatial data from previous session"""
        spatial_data = input_data.get("spatial_data")
        if not spatial_data:
            return {"success": False, "error": "Missing spatial_data"}
        
        try:
            if "spatial_map" in spatial_data and spatial_data["spatial_map"] is not None:
                self.spatial_map = np.array(spatial_data["spatial_map"])
            if "object_tracking" in spatial_data:
                self.object_tracking = spatial_data["object_tracking"]
            if "self_position" in spatial_data:
                self.self_position = np.array(spatial_data["self_position"])
            if "self_velocity" in spatial_data:
                self.self_velocity = np.array(spatial_data["self_velocity"])
            
            return {"success": True, "message": "Spatial data imported successfully"}
            
        except Exception as e:
            return {"success": False, "error": f"Import failed: {str(e)}"}
    
    def _get_spatial_status_operation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed spatial model status"""
        return {
            "success": True,
            "spatial_status": {
                "model_id": self.model_id,
                "is_initialized": self.is_initialized,
                "self_position": self.self_position.tolist(),
                "self_velocity": self.self_velocity.tolist(),
                "tracked_objects": len(self.object_tracking),
                "spatial_map_size": self.spatial_map.shape,
                "camera_baseline": self.camera_baseline,
                "focal_length": self.focal_length,
                "object_tracking_ids": list(self.object_tracking.keys())
            }
        }

    # ===== SPATIAL PROCESSING METHODS =====
    
    def _compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Compute disparity map from stereo images"""
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity
    
    def _compute_depth(self, disparity_map: np.ndarray) -> np.ndarray:
        """Compute depth map from disparity"""
        if disparity_map is None or np.all(disparity_map == 0):
            return np.zeros_like(disparity_map)
        
        disparity_map[disparity_map == 0] = 0.1
        depth_map = (self.camera_baseline * self.focal_length) / disparity_map
        depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
        return depth_map
    
    def _update_spatial_map(self, depth_map: np.ndarray):
        """Update spatial map with new depth data"""
        try:
            height, width = depth_map.shape
            for y in range(height):
                for x in range(width):
                    depth = depth_map[y, x]
                    if self.min_depth <= depth <= self.max_depth:
                        camera_x = (x - width/2) * depth / self.focal_length
                        camera_y = (y - height/2) * depth / self.focal_length
                        camera_z = depth
                        
                        world_x = self.self_position[0] + camera_x
                        world_y = self.self_position[1] + camera_y
                        world_z = self.self_position[2] + camera_z
                        
                        grid_x = int(world_x / self.grid_resolution)
                        grid_y = int(world_y / self.grid_resolution)
                        grid_z = int(world_z / self.grid_resolution)
                        
                        if (0 <= grid_x < self.map_size[0] and 
                            0 <= grid_y < self.map_size[1] and 
                            0 <= grid_z < self.map_size[2]):
                            self.spatial_map[grid_x, grid_y, grid_z] = depth
            
        except Exception as e:
            self.logger.error(f"Spatial map update failed: {str(e)}")
    
    def _update_self_position(self, context: Dict):
        """Update self position from sensor data"""
        if "sensor_data" in context:
            sensor_data = context["sensor_data"]
            self.self_position = np.array(sensor_data.get("position", [0, 0, 0]))
            self.self_velocity = np.array(sensor_data.get("velocity", [0, 0, 0]))
    
    def _detect_objects(self, image: np.ndarray, depth_map: np.ndarray) -> list:
        """Detect objects in the scene"""
        try:
            objects = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                        depth = depth_map[center_y, center_x]
                        
                        world_x = self.self_position[0] + (center_x - image.shape[1]/2) * depth / self.focal_length
                        world_y = self.self_position[1] + (center_y - image.shape[0]/2) * depth / self.focal_length
                        world_z = self.self_position[2] + depth
                        
                        size_x = w * depth / self.focal_length
                        size_y = h * depth / self.focal_length
                        size_z = min(size_x, size_y)
                        
                        aspect_ratio = w / h
                        if 0.8 <= aspect_ratio <= 1.2:
                            obj_type = "cube"
                        elif aspect_ratio > 1.5:
                            obj_type = "horizontal_rectangle"
                        else:
                            obj_type = "vertical_rectangle"
                        
                        objects.append({
                            "id": f"obj_{i+1}",
                            "position": [float(world_x), float(world_y), float(world_z)],
                            "size": [float(size_x), float(size_y), float(size_z)],
                            "type": obj_type,
                            "confidence": 0.85
                        })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {str(e)}")
            return []
    
    def _calculate_volume(self, obj: Dict, depth_map: np.ndarray) -> float:
        """Calculate object volume"""
        try:
            size = obj.get("size", [0, 0, 0])
            return float(size[0] * size[1] * size[2])
        except Exception as e:
            self.logger.error(f"Volume calculation failed: {str(e)}")
            return 0.0
    
    def _update_object_tracking(self, objects: List[Dict]):
        """Update object tracking data"""
        try:
            current_time = time.time()
            
            for obj_id in list(self.object_tracking.keys()):
                if obj_id not in [obj["id"] for obj in objects]:
                    del self.object_tracking[obj_id]
            
            for obj in objects:
                obj_id = obj["id"]
                if obj_id in self.object_tracking:
                    track_data = self.object_tracking[obj_id]
                    old_pos = np.array(track_data["position"])
                    new_pos = np.array(obj["position"])
                    
                    time_diff = current_time - track_data["last_seen"]
                    if time_diff > 0:
                        velocity = (new_pos - old_pos) / time_diff
                        track_data["velocity"] = velocity.tolist()
                    
                    track_data["position"] = obj["position"]
                    track_data["size"] = obj["size"]
                    track_data["last_seen"] = current_time
                    track_data["seen_count"] += 1
                else:
                    self.object_tracking[obj_id] = {
                        "position": obj["position"],
                        "size": obj["size"],
                        "type": obj["type"],
                        "velocity": [0, 0, 0],
                        "first_seen": current_time,
                        "last_seen": current_time,
                        "seen_count": 1
                    }
                    
        except Exception as e:
            self.logger.error(f"Object tracking update failed: {str(e)}")
    
    def _detect_moving_objects(self, image: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        """Detect moving objects using optical flow"""
        try:
            moving_objects = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if self._prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mask = mag > 2.0
                
                if np.any(motion_mask):
                    motion_mask_uint8 = (motion_mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(motion_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 50:
                            x, y, w, h = cv2.boundingRect(contour)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                                depth = depth_map[center_y, center_x]
                                
                                world_x = self.self_position[0] + (center_x - image.shape[1]/2) * depth / self.focal_length
                                world_y = self.self_position[1] + (center_y - image.shape[0]/2) * depth / self.focal_length
                                world_z = self.self_position[2] + depth
                                
                                flow_x = flow[center_y, center_x, 0]
                                flow_y = flow[center_y, center_x, 1]
                                velocity_x = flow_x * depth / self.focal_length
                                velocity_y = flow_y * depth / self.focal_length
                                
                                moving_objects.append({
                                    "id": f"moving_{len(moving_objects)+1}",
                                    "position": [float(world_x), float(world_y), float(world_z)],
                                    "velocity": [float(velocity_x), float(velocity_y), 0],
                                    "motion_magnitude": float(mag[center_y, center_x]),
                                    "confidence": 0.8
                                })
            
            self._prev_gray = gray.copy()
            self._prev_depth = depth_map.copy() if depth_map is not None else None
            
            return moving_objects
            
        except Exception as e:
            self.logger.error(f"Moving object detection failed: {str(e)}")
            return []
    
    def _predict_movement(self, obj: Dict) -> List[float]:
        """Predict movement direction"""
        try:
            velocity = obj.get("velocity", [0, 0, 0])
            return [velocity[0], velocity[1], velocity[2]]
        except Exception as e:
            self.logger.error(f"Movement prediction failed: {str(e)}")
            return [0, 0, 0]

    # ===== TRAINING IMPLEMENTATION =====
    
    def _train_model_specific(self, training_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train spatial model from scratch using neural network"""
        try:
            self.logger.info("Starting neural network training for spatial model")
            
            # Ensure neural network is initialized
            if not hasattr(self, 'neural_network') or self.neural_network is None:
                self._initialize_neural_networks()
            
            # Prepare training data - use real training data instead of synthetic
            if isinstance(training_data, list) and len(training_data) > 0:
                dataset = SpatialDataset(training_data)
            else:
                # Load real training data from external sources
                real_data = self._load_real_training_data()
                if real_data:
                    dataset = SpatialDataset(real_data)
                else:
                    # Fallback to sample data only if no real data available
                    self.logger.warning("No real training data available, using synthetic data")
                    sample_data = self._create_sample_training_data()
                    dataset = SpatialDataset(sample_data)
            
            # Training configuration
            epochs = config.get('epochs', 50)
            batch_size = config.get('batch_size', 8)
            learning_rate = config.get('learning_rate', 0.001)
            
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Update optimizer with learning rate from config
            if learning_rate != self.config.get('neural_network', {}).get('learning_rate', 0.001):
                self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
            
            # Training metrics
            metrics = {
                "total_loss": [],
                "depth_loss": [],
                "object_loss": [],
                "motion_loss": [],
                "learning_rate": learning_rate
            }
            
            # Early stopping configuration
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            self.logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
            
            for epoch in range(epochs):
                epoch_depth_loss = 0.0
                epoch_object_loss = 0.0
                epoch_motion_loss = 0.0
                epoch_total_loss = 0.0
                batch_count = 0
                
                # Training loop
                self.neural_network.train()
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.neural_network(batch['stereo_pair'])
                    
                    # Calculate losses for each task
                    depth_loss = self.depth_criterion(
                        outputs['depth'].squeeze(), 
                        batch['depth_target']
                    )
                    
                    object_loss = self.object_criterion(
                        outputs['objects'], 
                        batch['object_target']
                    )
                    
                    motion_loss = self.motion_criterion(
                        outputs['motion'], 
                        batch['motion_target']
                    )
                    
                    # Combined loss (weighted sum)
                    total_loss = 0.4 * depth_loss + 0.4 * object_loss + 0.2 * motion_loss
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    # Accumulate losses
                    epoch_depth_loss += depth_loss.item()
                    epoch_object_loss += object_loss.item()
                    epoch_motion_loss += motion_loss.item()
                    epoch_total_loss += total_loss.item()
                    batch_count += 1
                
                # Calculate average losses for the epoch
                if batch_count > 0:
                    avg_depth_loss = epoch_depth_loss / batch_count
                    avg_object_loss = epoch_object_loss / batch_count
                    avg_motion_loss = epoch_motion_loss / batch_count
                    avg_total_loss = epoch_total_loss / batch_count
                    
                    metrics["total_loss"].append(avg_total_loss)
                    metrics["depth_loss"].append(avg_depth_loss)
                    metrics["object_loss"].append(avg_object_loss)
                    metrics["motion_loss"].append(avg_motion_loss)
                    
                    # Early stopping check
                    if avg_total_loss < best_loss:
                        best_loss = avg_total_loss
                        patience_counter = 0
                        # Save best model
                        self._save_model_checkpoint(epoch, avg_total_loss)
                    else:
                        patience_counter += 1
                    
                    # Log progress
                    if (epoch + 1) % 5 == 0:
                        self.logger.info(
                            f"Epoch {epoch+1}/{epochs} - "
                            f"Total Loss: {avg_total_loss:.4f}, "
                            f"Depth Loss: {avg_depth_loss:.4f}, "
                            f"Object Loss: {avg_object_loss:.4f}, "
                            f"Motion Loss: {avg_motion_loss:.4f}"
                        )
                    
                    # Check for early stopping
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Final model evaluation
            final_metrics = self._evaluate_model(dataloader)
            metrics.update(final_metrics)
            
            # Save training history
            training_history = {
                "timestamp": datetime.now().isoformat(),
                "parameters": config,
                "metrics": metrics,
                "final_total_loss": metrics["total_loss"][-1] if metrics["total_loss"] else float('inf'),
                "best_loss": best_loss,
                "epochs_completed": min(epoch + 1, epochs)
            }
            
            self._save_training_history(training_history)
            
            # Update training status flags
            self.is_trained = True
            self.training_completed = True
            self.training_epochs_completed = epoch + 1
            
            self.logger.info("Spatial model training completed successfully")
            
            return {
                "status": "training_completed",
                "epochs": epoch + 1,
                "from_scratch": True,
                "training_history": training_history,
                "final_metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {str(e)}")
            return {"status": "training_failed", "error": str(e)}
    
    def _create_sample_training_data(self) -> List[Dict]:
        """Create sample training data for spatial model"""
        sample_data = []
        
        # Generate synthetic stereo image pairs with depth information
        for i in range(100):  # Create 100 sample pairs
            # Create synthetic stereo images (simple patterns for demonstration)
            height, width = 240, 320  # Reduced resolution for training efficiency
            
            # Left image with random patterns
            left_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Right image with horizontal shift based on depth
            depth_map = np.random.uniform(0.5, 10.0, (height, width))
            disparity_map = (self.camera_baseline * self.focal_length) / depth_map
            disparity_map = np.clip(disparity_map, 0, 64).astype(np.int32)
            
            right_img = np.zeros_like(left_img)
            for y in range(height):
                for x in range(width):
                    new_x = max(0, min(width-1, x - disparity_map[y, x]))
                    right_img[y, new_x] = left_img[y, x]
            
            # Object data (simplified for training)
            object_data = [
                np.random.uniform(0, 1),  # x position
                np.random.uniform(0, 1),  # y position  
                np.random.uniform(0.1, 0.5),  # width
                np.random.uniform(0.1, 0.5),  # height
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # padding
            ]
            
            # Motion data
            motion_data = [
                np.random.uniform(-0.1, 0.1),  # dx
                np.random.uniform(-0.1, 0.1),  # dy
                np.random.uniform(-0.05, 0.05)  # dz
            ]
            
            sample_data.append({
                'left_image': left_img,
                'right_image': right_img,
                'depth_map': np.mean(depth_map),
                'object_data': object_data,
                'motion_data': motion_data
            })
        
        return sample_data
    
    def _evaluate_model(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.neural_network.eval()
        total_depth_error = 0.0
        total_object_error = 0.0
        total_motion_error = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.neural_network(batch['stereo_pair'])
                
                # Calculate errors
                depth_error = torch.abs(outputs['depth'].squeeze() - batch['depth_target']).mean().item()
                object_error = torch.abs(outputs['objects'] - batch['object_target']).mean().item()
                motion_error = torch.abs(outputs['motion'] - batch['motion_target']).mean().item()
                
                total_depth_error += depth_error
                total_object_error += object_error
                total_motion_error += motion_error
                sample_count += batch['stereo_pair'].size(0)
        
        if sample_count > 0:
            return {
                "depth_estimation_error": total_depth_error / sample_count,
                "object_detection_error": total_object_error / sample_count,
                "motion_prediction_error": total_motion_error / sample_count
            }
        else:
            return {
                "depth_estimation_error": float('inf'),
                "object_detection_error": float('inf'),
                "motion_prediction_error": float('inf')
            }
    
    def _save_model_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        try:
            import os
            checkpoint_dir = "data/model_checkpoints/spatial"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"spatial_model_epoch_{epoch+1}_loss_{loss:.4f}.pth"
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.neural_network.state_dict(),
                'loss': loss,
                'camera_baseline': self.camera_baseline,
                'focal_length': self.focal_length
            }, checkpoint_path)
            
            self.logger.info(f"Model checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save model checkpoint: {str(e)}")
    
    def _update_model_parameters_from_training(self, metrics: Dict[str, List[float]]):
        """Optimize model parameters based on training results"""
        try:
            avg_accuracy = np.mean(metrics["accuracy"])
            if avg_accuracy > 0.8:
                self.focal_length *= (1.0 + (avg_accuracy - 0.8) * 0.05)
                self.logger.info(f"Optimized focal length: {self.focal_length:.2f}")
            
            avg_depth_error = np.mean(metrics["depth_estimation_error"])
            if avg_depth_error < 0.1:
                self.camera_baseline *= (1.0 + (0.1 - avg_depth_error) * 0.1)
                self.logger.info(f"Optimized camera baseline: {self.camera_baseline:.3f}m")
            
        except Exception as e:
            self.logger.warning(f"Parameter optimization failed: {str(e)}")
    
    def _save_training_history(self, history: Dict[str, Any]):
        """Save training history to file"""
        try:
            import json
            import os
            
            history_dir = "data/training_history"
            os.makedirs(history_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spatial_training_{timestamp}.json"
            filepath = os.path.join(history_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Training history saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training history: {str(e)}")

    # ===== AGI COMPONENT INITIALIZATION =====
    
    def _initialize_agi_spatial_components(self) -> None:
        """Initialize AGI components for advanced spatial reasoning and cognitive capabilities"""
        try:
            # Use unified AGI tools to initialize all components
            agi_components = AGITools.initialize_agi_components()
            
            # Assign components
            self.agi_spatial_reasoning = agi_components["reasoning_engine"]
            self.agi_meta_learning = agi_components["meta_learning_system"]
            self.agi_self_reflection = agi_components["self_reflection_module"]
            self.agi_cognitive_engine = agi_components["cognitive_engine"]
            self.agi_problem_solver = agi_components["problem_solver"]
            self.agi_creative_generator = agi_components["creative_generator"]
            
            self.logger.info("AGI spatial components initialized successfully using unified tools")
            
        except Exception as e:
            self.logger.error(f"AGI component initialization failed: {str(e)}")
            # Fallback to basic components if AGI initialization fails
            self._initialize_basic_agi_components()
    
    def _create_agi_spatial_reasoning_engine(self) -> Dict[str, Any]:
        """Create AGI spatial reasoning engine for advanced spatial intelligence"""
        return {
            "engine_type": "agi_spatial_reasoning",
            "capabilities": [
                "3d_spatial_reasoning",
                "multi_perspective_analysis", 
                "dynamic_environment_modeling",
                "spatial_temporal_integration",
                "causal_spatial_inference",
                "predictive_spatial_analysis"
            ],
            "reasoning_depth": "deep_cognitive",
            "spatial_resolution": "high_precision",
            "temporal_horizon": "long_term",
            "integration_level": "multi_modal_fusion"
        }
    
    def _create_agi_meta_learning_system(self) -> Dict[str, Any]:
        """Create AGI meta-learning system for spatial pattern recognition"""
        return {
            "system_type": "agi_meta_learning_spatial",
            "learning_mechanisms": [
                "spatial_pattern_abstraction",
                "cross_domain_transfer",
                "experience_compression", 
                "adaptive_parameter_optimization",
                "hierarchical_feature_learning",
                "context_aware_adaptation"
            ],
            "pattern_recognition": "multi_scale",
            "adaptation_speed": "rapid",
            "generalization_capability": "strong",
            "knowledge_retention": "persistent"
        }
    
    def _create_agi_self_reflection_module(self) -> Dict[str, Any]:
        """Create AGI self-reflection module for spatial performance optimization"""
        return {
            "module_type": "agi_self_reflection_spatial",
            "reflection_capabilities": [
                "performance_analysis",
                "error_diagnosis", 
                "strategy_evaluation",
                "improvement_planning",
                "goal_alignment_check",
                "capability_assessment"
            ],
            "analysis_depth": "comprehensive",
            "feedback_loop": "continuous",
            "improvement_focus": "proactive",
            "adaptation_strategy": "multi_objective"
        }
    
    def _create_agi_cognitive_engine(self) -> Dict[str, Any]:
        """Create AGI cognitive engine for spatial understanding"""
        return {
            "engine_type": "agi_cognitive_spatial",
            "cognitive_processes": [
                "spatial_attention",
                "working_memory_management",
                "long_term_integration",
                "executive_control",
                "meta_cognitive_monitoring",
                "conscious_processing"
            ],
            "attention_mechanism": "selective_focus",
            "memory_capacity": "expansive",
            "processing_depth": "deep_understanding",
            "integration_level": "holistic"
        }
    
    def _create_agi_spatial_problem_solver(self) -> Dict[str, Any]:
        """Create AGI spatial problem solver for complex spatial challenges"""
        return {
            "solver_type": "agi_spatial_problem_solver",
            "problem_solving_approaches": [
                "problem_decomposition",
                "solution_synthesis",
                "constraint_satisfaction",
                "optimization_techniques",
                "creative_abstraction",
                "adaptive_strategies"
            ],
            "solution_quality": "optimal",
            "reasoning_depth": "thorough",
            "creativity_level": "innovative",
            "adaptability": "high"
        }
    
    def _create_agi_creative_generator(self) -> Dict[str, Any]:
        """Create AGI creative generator for spatial innovation"""
        return {
            "generator_type": "agi_creative_spatial",
            "creative_processes": [
                "novel_strategy_generation",
                "alternative_scenario_exploration",
                "emergent_behavior_utilization",
                "cross_domain_insight_transfer",
                "conceptual_blending",
                "pattern_completion_creativity"
            ],
            "novelty_level": "high",
            "diversity": "broad",
            "usefulness": "practical",
            "innovation_potential": "significant"
        }
    
    def _initialize_basic_agi_components(self) -> None:
        """Initialize basic AGI components as fallback"""
        self.agi_spatial_reasoning = {"engine_type": "basic_spatial_reasoning"}
        self.agi_meta_learning = {"system_type": "basic_meta_learning"} 
        self.agi_self_reflection = {"module_type": "basic_self_reflection"}
        self.agi_cognitive_engine = {"engine_type": "basic_cognitive"}
        self.agi_problem_solver = {"solver_type": "basic_problem_solver"}
        self.agi_creative_generator = {"generator_type": "basic_creative"}
        
        self.logger.warning("Using basic AGI components as fallback")


class SpatialStreamProcessor(StreamProcessor):
    """Spatial-specific stream processor for real-time processing"""
    
    def __init__(self, spatial_model: UnifiedSpatialModel):
        super().__init__()
        self.spatial_model = spatial_model
        self.logger = logging.getLogger(__name__)
    
    def process_stream_data(self, stream_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process spatial stream data"""
        try:
            left_image = stream_data.get("left_frame")
            right_image = stream_data.get("right_frame")
            operation = stream_data.get("operation", "analyze_spatial_data")
            
            if left_image is None or right_image is None:
                return {"success": False, "error": "Missing stereo frames"}
            
            # Process using the spatial model
            result = self.spatial_model.process({
                "operation": operation,
                "left_image": left_image,
                "right_image": right_image,
                "context": stream_data.get("context", {})
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stream processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "processor_type": "spatial",
            "supported_operations": self.spatial_model.supported_operations,
            "model_id": self.spatial_model.model_id
        }


# Factory function for creating unified spatial models
def create_unified_spatial_model(config: Dict[str, Any] = None) -> UnifiedSpatialModel:
    """Create a unified spatial model instance"""
    return UnifiedSpatialModel(config)
