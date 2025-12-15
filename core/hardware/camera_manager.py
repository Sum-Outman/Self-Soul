"""
Camera Manager - Multi-camera Support System

Provides functionality for managing multiple cameras, capturing images and video streams,
and processing visual data for binocular vision capabilities.
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraManager")

class CameraManager:
    """Multi-camera management system for binocular vision and multi-camera support"""
    
    def __init__(self):
        """Initialize the camera manager"""
        self.cameras = {}
        self.active_streams = {}
        self.camera_threads = {}
        self.callbacks = {}
        self.lock = threading.Lock()
        self.config = {
            "default_resolution": (640, 480),
            "default_fps": 30,
            "buffer_size": 5,
            "image_format": "bgr",
            "video_codec": "XVID"
        }
        
    def initialize(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize camera manager with configuration"""
        try:
            if config:
                self.config.update(config)
            
            logger.info("Camera manager initialized successfully")
            return {"success": True, "message": "Camera manager initialized"}
        except Exception as e:
            logger.error(f"Camera manager initialization failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def list_available_cameras(self, max_devices: int = 10) -> List[Dict[str, Any]]:
        """List all available camera devices"""
        available_cameras = []
        
        # Try to open each camera index up to max_devices
        for i in range(max_devices):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW backend for Windows
            if cap.isOpened():
                # Get basic camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                camera_info = {
                    "id": i,
                    "index": i,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "status": "available"
                }
                available_cameras.append(camera_info)
                cap.release()
            
        logger.info(f"Found {len(available_cameras)} available cameras")
        return available_cameras
    
    def connect_camera(self, camera_id: str, camera_index: int, 
                      resolution: Tuple[int, int] = None, 
                      fps: float = None) -> Dict[str, Any]:
        """Connect to a specific camera device"""
        try:
            with self.lock:
                if camera_id in self.cameras:
                    return {"success": False, "error": f"Camera {camera_id} already connected"}
                
                # Use default resolution and fps if not provided
                resolution = resolution or self.config["default_resolution"]
                fps = fps or self.config["default_fps"]
                
                # Try to connect to the camera
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    return {"success": False, "error": f"Failed to open camera at index {camera_index}"}
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # Create camera info dictionary
                self.cameras[camera_id] = {
                    "id": camera_id,
                    "index": camera_index,
                    "capture": cap,
                    "resolution": resolution,
                    "fps": fps,
                    "is_streaming": False,
                    "buffer": [],
                    "last_frame": None,
                    "connected_at": datetime.now().isoformat()
                }
                
                # Initialize stream and buffer
                self.active_streams[camera_id] = False
                self.cameras[camera_id]["buffer"] = []
                
                logger.info(f"Camera {camera_id} connected successfully")
                return {"success": True, "camera_id": camera_id, "resolution": resolution, "fps": fps}
        except Exception as e:
            logger.error(f"Failed to connect camera {camera_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def start_stream(self, camera_id: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start streaming from a connected camera"""
        try:
            with self.lock:
                if camera_id not in self.cameras:
                    return {"success": False, "error": f"Camera {camera_id} not found"}
                
                if self.active_streams.get(camera_id, False):
                    return {"success": False, "error": f"Camera {camera_id} is already streaming"}
                
                self.active_streams[camera_id] = True
                
                # Store callback if provided
                if callback:
                    self.callbacks[camera_id] = callback
                
                # Start streaming thread
                self.camera_threads[camera_id] = threading.Thread(
                    target=self._streaming_thread, 
                    args=(camera_id,),
                    daemon=True
                )
                self.camera_threads[camera_id].start()
                
                logger.info(f"Started streaming from camera {camera_id}")
                return {"success": True, "camera_id": camera_id}
        except Exception as e:
            logger.error(f"Failed to start stream from camera {camera_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def stop_stream(self, camera_id: str) -> Dict[str, Any]:
        """Stop streaming from a camera"""
        try:
            with self.lock:
                if camera_id not in self.cameras or not self.active_streams.get(camera_id, False):
                    return {"success": False, "error": f"Camera {camera_id} is not streaming"}
                
                self.active_streams[camera_id] = False
                
                # Wait for thread to stop
                if camera_id in self.camera_threads:
                    self.camera_threads[camera_id].join(timeout=2.0)
                    del self.camera_threads[camera_id]
                
                # Remove callback if exists
                if camera_id in self.callbacks:
                    del self.callbacks[camera_id]
                
                logger.info(f"Stopped streaming from camera {camera_id}")
                return {"success": True, "camera_id": camera_id}
        except Exception as e:
            logger.error(f"Failed to stop stream from camera {camera_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def capture_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Capture a single frame from a camera"""
        try:
            with self.lock:
                if camera_id not in self.cameras:
                    logger.error(f"Camera {camera_id} not found")
                    return None
                
                cap = self.cameras[camera_id]["capture"]
                ret, frame = cap.read()
                
                if not ret:
                    logger.error(f"Failed to capture frame from camera {camera_id}")
                    return None
                
                # Update last frame
                self.cameras[camera_id]["last_frame"] = frame
                return frame
        except Exception as e:
            logger.error(f"Error capturing frame from camera {camera_id}: {str(e)}")
            return None
    
    def get_last_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the last captured frame from a camera"""
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return None
            
            return self.cameras[camera_id]["last_frame"]
    
    def disconnect_camera(self, camera_id: str) -> Dict[str, Any]:
        """Disconnect a camera device"""
        try:
            with self.lock:
                if camera_id not in self.cameras:
                    return {"success": False, "error": f"Camera {camera_id} not found"}
                
                # Stop streaming if active
                if self.active_streams.get(camera_id, False):
                    self.stop_stream(camera_id)
                
                # Release the camera capture object
                self.cameras[camera_id]["capture"].release()
                
                # Remove camera from dictionaries
                del self.cameras[camera_id]
                if camera_id in self.active_streams:
                    del self.active_streams[camera_id]
                
                logger.info(f"Camera {camera_id} disconnected successfully")
                return {"success": True, "camera_id": camera_id}
        except Exception as e:
            logger.error(f"Failed to disconnect camera {camera_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_camera_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connected camera"""
        with self.lock:
            if camera_id not in self.cameras:
                logger.error(f"Camera {camera_id} not found")
                return None
            
            camera = self.cameras[camera_id]
            return {
                "id": camera["id"],
                "index": camera["index"],
                "resolution": camera["resolution"],
                "fps": camera["fps"],
                "is_streaming": self.active_streams.get(camera_id, False),
                "connected_at": camera["connected_at"]
            }
    
    def get_all_cameras_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all connected cameras"""
        result = {}
        with self.lock:
            for camera_id in self.cameras:
                result[camera_id] = self.get_camera_info(camera_id)
        return result
    
    def perform_binocular_vision(self, left_camera_id: str, right_camera_id: str, 
                                min_disparity: int = 0, num_disparities: int = 16, 
                                block_size: int = 15) -> Dict[str, Any]:
        """Perform binocular vision processing for depth perception"""
        try:
            # Get frames from both cameras
            left_frame = self.get_last_frame(left_camera_id)
            right_frame = self.get_last_frame(right_camera_id)
            
            if left_frame is None or right_frame is None:
                return {"success": False, "error": "Failed to get frames from both cameras"}
            
            # Convert to grayscale for stereo matching
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Create StereoBM object for disparity map calculation with configurable parameters
            stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
            stereo.setMinDisparity(min_disparity)
            
            # Compute disparity map
            disparity = stereo.compute(left_gray, right_gray)
            
            # Normalize the disparity map for display
            disparity_normalized = cv2.normalize(
                disparity, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # In a real application, we would use actual calibration data
            # For now, we'll simulate depth map calculation with default parameters
            baseline = 0.1  # 10cm baseline distance
            focal_length = 500  # Pixels
            
            # Create point cloud data (simplified)
            h, w = left_gray.shape
            point_cloud = []
            colors = []
            
            for y in range(0, h, 5):  # Sample every 5th pixel for performance
                for x in range(0, w, 5):
                    disp = disparity[y, x]
                    if disp > 0:
                        # Calculate 3D coordinates using triangulation formula
                        z = (baseline * focal_length) / disp if disp > 0 else 0
                        x3d = (x - w/2) * z / focal_length
                        y3d = (y - h/2) * z / focal_length
                        
                        point_cloud.append([x3d, y3d, z])
                        # Get color from original image
                        b, g, r = left_frame[y, x]
                        colors.append([r/255, g/255, b/255])
            
            # Prepare result data
            result_data = {
                "disparity_map": disparity_normalized,
                "depth_map": {
                    "data": disparity.tolist() if isinstance(disparity, np.ndarray) else disparity,
                    "width": w,
                    "height": h,
                    "min_disparity": min_disparity,
                    "num_disparities": num_disparities,
                    "block_size": block_size
                },
                "point_cloud": {
                    "points": point_cloud,
                    "colors": colors,
                    "count": len(point_cloud)
                },
                "left_frame_shape": left_frame.shape,
                "right_frame_shape": right_frame.shape,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "data": result_data
            }
        except Exception as e:
            logger.error(f"Binocular vision processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def save_frame(self, camera_id: str, filepath: str) -> Dict[str, Any]:
        """Save the last captured frame to a file"""
        try:
            frame = self.get_last_frame(camera_id)
            if frame is None:
                return {"success": False, "error": "No frame available to save"}
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save frame
            success = cv2.imwrite(filepath, frame)
            if not success:
                return {"success": False, "error": "Failed to save frame"}
            
            logger.info(f"Frame from camera {camera_id} saved to {filepath}")
            return {"success": True, "filepath": filepath}
        except Exception as e:
            logger.error(f"Failed to save frame from camera {camera_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _streaming_thread(self, camera_id: str):
        """Thread function for streaming camera data"""
        try:
            camera = self.cameras[camera_id]
            cap = camera["capture"]
            fps = camera["fps"]
            
            # Calculate frame interval
            frame_interval = 1.0 / fps
            
            while self.active_streams.get(camera_id, False):
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Update last frame
                with self.lock:
                    camera["last_frame"] = frame
                    
                    # Add to buffer and maintain buffer size
                    camera["buffer"].append(frame)
                    if len(camera["buffer"]) > self.config["buffer_size"]:
                        camera["buffer"].pop(0)
                
                # Call callback if exists
                if camera_id in self.callbacks:
                    try:
                        self.callbacks[camera_id](camera_id, frame)
                    except Exception as e:
                        logger.error(f"Error in camera {camera_id} callback: {str(e)}")
                
                # Sleep to maintain desired FPS
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed_time)
                time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Error in camera {camera_id} streaming thread: {str(e)}")
            # Stop streaming in case of error
            with self.lock:
                self.active_streams[camera_id] = False
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        # Disconnect all cameras
        for camera_id in list(self.cameras.keys()):
            self.disconnect_camera(camera_id)

# Create a global instance of CameraManager for easy access
camera_manager = CameraManager()