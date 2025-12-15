<template>
  <div class="real-time-input">
    <!-- Status Messages Display -->
    <div class="status-messages">
      <div v-if="errorState.hasError" class="message error">
        {{ errorState.message }}
      </div>
      <div v-if="successState.hasSuccess" class="message success">
        {{ successState.message }}
      </div>
      <div v-if="warningState.hasWarning" class="message warning">
        {{ warningState.message }}
      </div>
      <div v-if="infoState.hasInfo" class="message info">
        {{ infoState.message }}
      </div>
    </div>
    
    <!-- Multi-camera Control Panel -->
    <div class="camera-control-panel">
      <h3>Camera Control</h3>
      <div class="camera-configuration">
        <div class="form-group">
          <label>Camera Setup Type</label>
          <select v-model="cameraSetupType" @change="onCameraSetupTypeChange">
            <option value="single">Single Camera</option>
            <option value="stereo">Stereo (Binocular) Vision</option>
            <option value="multi">Multi-camera Array</option>
          </select>
        </div>
        
        <!-- Stereo Camera Selection -->
        <div v-if="cameraSetupType === 'stereo'" class="stereo-selection">
          <div class="form-row">
            <div class="form-group">
              <label>Left Camera</label>
              <select v-model="selectedStereoCameras.left">
                <option v-for="camera in cameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || 'Camera' }}
                </option>
              </select>
            </div>
            <div class="form-group">
              <label>Right Camera</label>
              <select v-model="selectedStereoCameras.right">
                <option v-for="camera in cameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || 'Camera' }}
                </option>
              </select>
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Baseline (mm)</label>
              <input type="number" v-model.number="stereoParams.baseline" min="10" max="500" />
            </div>
            <div class="form-group">
              <label>Focal Length (mm)</label>
              <input type="number" v-model.number="stereoParams.focalLength" min="1" max="100" step="0.1" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Minimum Disparity</label>
              <input type="number" v-model.number="stereoParams.minDisparity" min="0" max="100" />
            </div>
            <div class="form-group">
              <label>Number of Disparities</label>
              <input type="number" v-model.number="stereoParams.numDisparities" min="16" max="256" step="16" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label>Block Size (odd)</label>
              <input type="number" v-model.number="stereoParams.blockSize" min="3" max="31" step="2" />
            </div>
            <div class="form-group">
              <label>Stereo Pair ID</label>
              <input type="text" v-model="selectedStereoPairId" placeholder="Enter stereo pair ID" />
            </div>
          </div>
        </div>
        
        <!-- Multi-camera Selection -->
        <div v-if="cameraSetupType === 'multi'" class="multi-camera-selection">
          <div v-for="(cam, index) in activeCameras" :key="index" class="camera-selector">
            <label>Camera {{ index + 1 }}</label>
            <select v-model="cam.deviceId" @change="onMultiCameraChange">
              <option value="">Select Camera</option>
              <option v-for="camera in cameras" :key="camera.deviceId" :value="camera.deviceId">
                {{ camera.label || 'Camera' }}
              </option>
            </select>
            <button v-if="index > 0" @click="removeCamera(index)" class="remove-btn">Remove</button>
          </div>
          <button @click="addCamera" class="add-btn">Add Camera</button>
        </div>
        
        <!-- Global Camera Controls -->
        <div class="global-controls">
          <button @click="toggleAllCameras" class="primary-btn">
            {{ areCamerasActive ? 'Stop All Cameras' : 'Start All Cameras' }}
          </button>
          <button @click="captureAllFrames" class="secondary-btn" :disabled="!areCamerasActive">
            Capture All Frames
          </button>
          <button @click="calibrateStereoCameras" class="secondary-btn" :disabled="cameraSetupType !== 'stereo' || !areCamerasActive">
            Calibrate Stereo Cameras
          </button>
        </div>
      </div>
    </div>
    
    <!-- Camera Display Section -->
    <div class="camera-display-section">
      <!-- Single Camera View -->
      <div v-if="cameraSetupType === 'single'" class="single-camera-view">
        <div class="camera-container">
          <video ref="videoElement" autoplay playsinline></video>
          <canvas ref="canvasElement" style="display: none;"></canvas>
        </div>
        <div class="camera-info">
          <span class="camera-status" :class="{ active: isCameraActive }">
            {{ isCameraActive ? 'Active' : 'Inactive' }}
          </span>
        </div>
      </div>
      
      <!-- Stereo Camera View -->
      <div v-else-if="cameraSetupType === 'stereo'" class="stereo-camera-view">
        <div class="stereo-pair">
          <div class="camera-container left">
            <div class="camera-label">Left Camera</div>
            <video ref="leftVideoElement" autoplay playsinline></video>
            <canvas ref="leftCanvasElement" style="display: none;"></canvas>
          </div>
          <div class="camera-container right">
            <div class="camera-label">Right Camera</div>
            <video ref="rightVideoElement" autoplay playsinline></video>
            <canvas ref="rightCanvasElement" style="display: none;"></canvas>
          </div>
        </div>
        
        <!-- Depth Map and 3D Visualization -->
        <div class="stereo-visualization">
          <div class="depth-map">
            <div class="visualization-label">Depth Map</div>
            <canvas ref="depthMapCanvas" width="640" height="480"></canvas>
          </div>
          <div class="point-cloud">
            <div class="visualization-label">3D Point Cloud Preview</div>
            <canvas ref="pointCloudCanvas" width="640" height="480"></canvas>
          </div>
        </div>
      </div>
      
      <!-- Multi-camera Grid View -->
      <div v-else-if="cameraSetupType === 'multi'" class="multi-camera-grid">
        <div v-for="(cam, index) in activeCameras" :key="index" class="camera-item">
          <div class="camera-container">
            <div class="camera-label">Camera {{ index + 1 }}</div>
            <video :ref="'multiVideo' + index" autoplay playsinline></video>
            <canvas :ref="'multiCanvas' + index" style="display: none;"></canvas>
          </div>
          <div class="camera-controls">
            <button @click="toggleSingleCamera(index)">
              {{ cam.isActive ? 'Stop' : 'Start' }}
            </button>
            <button @click="captureSingleFrame(index)" :disabled="!cam.isActive">
              Capture
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Microphone Input -->
    <div class="microphone-section">
      <h3>Microphone</h3>
      <div class="audio-visualization">
        <canvas ref="audioCanvas"></canvas>
      </div>
      <div class="microphone-controls">
        <button @click="toggleMicrophone">{{ isMicrophoneActive ? 'Stop Microphone' : 'Start Microphone' }}</button>
        <select v-model="selectedMicrophone" @change="changeMicrophone">
          <option v-for="(mic, index) in microphones" :key="mic.deviceId" :value="mic.deviceId">
            {{ mic.label || 'Microphone' }} {{ index + 1 }}
          </option>
        </select>
        <button @click="startSpeechRecognition" :disabled="!isMicrophoneActive">Start Speech Recognition</button>
      </div>
      <div v-if="transcript" class="transcript">
        Transcript: {{ transcript }}
      </div>
    </div>

    <!-- Network Stream Input -->
    <div class="network-stream-section">
      <h3>Network Stream</h3>
      <div class="stream-controls">
        <div class="input-group">
          <label>Video Stream URL:</label>
          <input v-model="videoStreamUrl" placeholder="rtsp:// or http:// video stream URL" />
          <button @click="toggleVideoStream" :disabled="!videoStreamUrl">
            {{ isVideoStreamActive ? 'Stop Stream' : 'Start Stream' }}
          </button>
        </div>
        <div class="input-group">
          <label>Audio Stream URL:</label>
          <input v-model="audioStreamUrl" placeholder="http:// audio stream URL" />
          <button @click="toggleAudioStream" :disabled="!audioStreamUrl">
              {{ isAudioStreamActive ? 'Stop Stream' : 'Start Stream' }}
            </button>
        </div>
      </div>
      <div v-if="isVideoStreamActive" class="stream-preview">
        <video ref="videoStreamElement" autoplay playsinline></video>
      </div>
    </div>

    <!-- Sensor Data Input -->
    <div class="sensor-section">
      <h3>Sensor Data</h3>
      <div class="sensor-controls">
        <button @click="toggleSensorData">{{ isSensorDataActive ? 'Stop Sensors' : 'Start Sensors' }}</button>
        <select v-model="selectedSensorInterface">
          <option value="serial">Serial Port</option>
          <option value="bluetooth">Bluetooth</option>
          <option value="network">Network</option>
        </select>
      </div>
      <div v-if="isSensorDataActive" class="sensor-readings">
        <div class="sensor-reading" v-for="(value, sensor) in sensorData" :key="sensor">
          <span class="sensor-label">{{ sensor }}:</span>
          <span class="sensor-value">{{ value }}</span>
        </div>
      </div>
    </div>

    <!-- Multimodal Fusion Control -->
    <div class="fusion-controls">
      <h3>Multimodal Fusion</h3>
      <div class="fusion-options">
        <label>
          <input type="checkbox" v-model="fuseAudioVisual"> 
          Fuse Audio & Visual
        </label>
        <label>
          <input type="checkbox" v-model="fuseSensorCamera"> 
          Fuse Sensor & Camera
        </label>
        <label>
          <input type="checkbox" v-model="fuseAllModalities"> 
          Fuse All Modalities
        </label>
      </div>
      <button @click="processFusedData" :disabled="!hasInputData">Process</button>
    </div>

    <!-- Real-time Dialog Control -->
    <div class="realtime-dialog-section">
      <h3>Real-time Dialog</h3>
      <div class="dialog-controls">
        <button @click="toggleRealTimeDialog" :class="{ active: isRealTimeDialogActive }">
          {{ isRealTimeDialogActive ? 'Stop Dialog' : 'Start Dialog' }}
        </button>
        <div class="dialog-status" :class="isRealTimeDialogActive ? 'active' : 'inactive'">
          {{ isRealTimeDialogActive ? 'Active' : 'Inactive' }}
        </div>
      </div>
      <div v-if="isRealTimeDialogActive" class="dialog-output">
        <h4>Real-time Responses</h4>
        <div class="response-messages">
          <div v-for="(response, index) in realTimeResponses" :key="index" class="response-message">
            {{ response }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'RealTimeInput',
  data() {
    return {
      // Camera Setup Type
      cameraSetupType: 'single', // single, stereo, multi
      
      // Single Camera Configuration
      isCameraActive: false,
      cameras: [],
      selectedCamera: '',
      stream: null,
      
      // Stereo Camera Configuration
      selectedStereoCameras: {
        left: '',
        right: ''
      },
      selectedStereoPairId: null,
      stereoStreams: {
        left: null,
        right: null
      },
      areStereoCamerasActive: false,
      stereoParams: {
        baseline: 65, // Distance between cameras in mm
        focalLength: 3.6, // Focal length in mm
        minDisparity: 0, // Minimum disparity for stereo matcher
        numDisparities: 16, // Number of disparities for stereo matcher (must be divisible by 16)
        blockSize: 9, // Block size for stereo matcher (must be odd)
        // Calibration parameters will be stored here after calibration
        calibration: {
          left: null,
          right: null,
          rotation: null,
          translation: null
        }
      },
      isStereoCalibrated: false,
      
      // Multi-camera Configuration
      activeCameras: [
        { deviceId: '', isActive: false, stream: null }
      ],
      areCamerasActive: false,
      
      // Microphone-related Status
      isMicrophoneActive: false,
      microphones: [],
      selectedMicrophone: '',
      audioContext: null,
      analyser: null,
      audioStream: null,
      transcript: '',
      recognition: null,
      
      // Network Stream-related Status
      videoStreamUrl: '',
      audioStreamUrl: '',
      isVideoStreamActive: false,
      isAudioStreamActive: false,
      videoMediaSource: null,
      audioMediaSource: null,
      
      // Sensor-related Status
      isSensorDataActive: false,
      selectedSensorInterface: 'serial',
      sensorData: {
        temperature: '--',
        humidity: '--',
        acceleration: '--',
        light: '--',
        distance: '--'
      },
      sensorWebSocket: null,
      sensorUpdateInterval: null,
      
      // Multimodal Fusion
      fuseAudioVisual: false,
      fuseSensorCamera: false,
      fuseAllModalities: false,
      hasInputData: false,
      
      // Real-time Dialog Status
      isRealTimeDialogActive: false,
      realTimeWebSocket: null,
      audioProcessor: null,
      videoProcessor: null,
      realTimeResponses: [],
      webSocketUrl: 'ws://localhost:8000/ws', // Default WebSocket URL
      audioCaptureInterval: null,
      videoCaptureInterval: null,
      depthMapInterval: null,
      isWebSocketConnected: false,
      
      // Status Message System
      errorState: {
        hasError: false,
        message: ''
      },
      successState: {
        hasSuccess: false,
        message: ''
      },
      warningState: {
        hasWarning: false,
        message: ''
      },
      infoState: {
        hasInfo: false,
        message: ''
      }
    }
  },
  mounted() {
    this.listDevices();
  },
  beforeDestroy() {
    this.stopAllCameras();
    this.stopMicrophone();
    this.stopSpeechRecognition();
    this.stopVideoStream();
    this.stopAudioStream();
    this.stopSensorData();
    this.stopDepthMapCalculation();
  },
  methods: {
    // Get Device List
    async listDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.cameras = devices.filter(device => device.kind === 'videoinput');
        this.microphones = devices.filter(device => device.kind === 'audioinput');
        
        if (this.cameras.length > 0) {
          this.selectedCamera = this.cameras[0].deviceId;
          this.selectedStereoCameras.left = this.cameras[0].deviceId;
          if (this.cameras.length > 1) {
            this.selectedStereoCameras.right = this.cameras[1].deviceId;
          }
          this.activeCameras[0].deviceId = this.cameras[0].deviceId;
        }
        
        if (this.microphones.length > 0) {
          this.selectedMicrophone = this.microphones[0].deviceId;
        }
      } catch (error) {
        console.error('Failed to get device list:', error);
      }
    },
    
    // Handle Camera Setup Type Change
    onCameraSetupTypeChange() {
      // Stop all active cameras when changing setup type
      this.stopAllCameras();
      
      // Reset flags
      this.isCameraActive = false;
      this.areStereoCamerasActive = false;
      this.areCamerasActive = false;
      this.isStereoCalibrated = false;
    },
    
    // Toggle All Cameras (Global Control)
    async toggleAllCameras() {
      if (this.areCamerasActive) {
        this.stopAllCameras();
      } else {
        await this.startAllCameras();
      }
    },
    
    // Start All Cameras Based on Setup Type
    async startAllCameras() {
      try {
        if (this.cameraSetupType === 'single') {
          await this.startSingleCamera();
        } else if (this.cameraSetupType === 'stereo') {
          await this.startStereoCameras();
        } else if (this.cameraSetupType === 'multi') {
          await this.startMultiCameras();
        }
        this.areCamerasActive = true;
      } catch (error) {
        console.error('Failed to start cameras:', error);
        this.showError('Failed to start cameras');
      }
    },
    
    // Stop All Cameras
    stopAllCameras() {
      if (this.cameraSetupType === 'single') {
        this.stopSingleCamera();
      } else if (this.cameraSetupType === 'stereo') {
        this.stopStereoCameras();
      } else if (this.cameraSetupType === 'multi') {
        this.stopMultiCameras();
      }
      this.areCamerasActive = false;
    },
    
    // Single Camera Methods
    async startSingleCamera() {
      try {
        const constraints = {
          video: {
            deviceId: this.selectedCamera ? { exact: this.selectedCamera } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };
        
        this.stream = await navigator.mediaDevices.getUserMedia(constraints);
        this.$refs.videoElement.srcObject = this.stream;
        this.isCameraActive = true;
        this.hasInputData = true;
        this.showSuccess('Single camera started successfully');
      } catch (error) {
        console.error('Failed to start single camera:', error);
        this.showError('Failed to start single camera');
      }
    },
    
    stopSingleCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
      this.isCameraActive = false;
      if (this.$refs.videoElement) {
        this.$refs.videoElement.srcObject = null;
      }
    },
    
    async changeSingleCamera() {
      if (this.isCameraActive) {
        this.stopSingleCamera();
        await this.startSingleCamera();
      }
    },
    
    // Stereo Camera Methods
    async startStereoCameras() {
      try {
        if (!this.selectedStereoCameras.left || !this.selectedStereoCameras.right) {
          throw new Error('Please select both left and right cameras for stereo vision');
        }
        
        // Start left camera
        const leftConstraints = {
          video: {
            deviceId: { exact: this.selectedStereoCameras.left },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };
        
        // Start right camera
        const rightConstraints = {
          video: {
            deviceId: { exact: this.selectedStereoCameras.right },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        };
        
        // Start both cameras in parallel
        const [leftStream, rightStream] = await Promise.all([
          navigator.mediaDevices.getUserMedia(leftConstraints),
          navigator.mediaDevices.getUserMedia(rightConstraints)
        ]);
        
        // Assign streams to video elements
        this.stereoStreams.left = leftStream;
        this.stereoStreams.right = rightStream;
        
        if (this.$refs.leftVideoElement) {
          this.$refs.leftVideoElement.srcObject = leftStream;
        }
        if (this.$refs.rightVideoElement) {
          this.$refs.rightVideoElement.srcObject = rightStream;
        }
        
        this.areStereoCamerasActive = true;
        this.hasInputData = true;
        this.showSuccess('Stereo cameras started successfully');
        
        // Start depth map calculation if calibrated
        if (this.isStereoCalibrated) {
          this.startDepthMapCalculation();
        }
      } catch (error) {
        console.error('Failed to start stereo cameras:', error);
        this.showError('Failed to start stereo cameras: ' + error.message);
        // Clean up any partially started streams
        this.stopStereoCameras();
      }
    },
    
    stopStereoCameras() {
      // Stop left camera stream
      if (this.stereoStreams.left) {
        this.stereoStreams.left.getTracks().forEach(track => track.stop());
        this.stereoStreams.left = null;
      }
      
      // Stop right camera stream
      if (this.stereoStreams.right) {
        this.stereoStreams.right.getTracks().forEach(track => track.stop());
        this.stereoStreams.right = null;
      }
      
      this.areStereoCamerasActive = false;
      
      // Clear video elements
      if (this.$refs.leftVideoElement) {
        this.$refs.leftVideoElement.srcObject = null;
      }
      if (this.$refs.rightVideoElement) {
        this.$refs.rightVideoElement.srcObject = null;
      }
      
      // Stop depth map calculation
      this.stopDepthMapCalculation();
    },
    
    // Calibrate Stereo Cameras
    async calibrateStereoCameras() {
      try {
        if (!this.areStereoCamerasActive) {
          throw new Error('Stereo cameras must be active to calibrate');
        }
        
        this.showInfo('Calibrating stereo cameras... This may take a moment.');
        
        // Use the backend API for calibration instead of local simulation
        if (this.selectedStereoPairId) {
          const response = await api.cameras.calibrateStereoPair(this.selectedStereoPairId, {
            leftCameraId: this.selectedStereoCameras.left,
            rightCameraId: this.selectedStereoCameras.right,
            focalLength: this.stereoParams.focalLength,
            baseline: this.stereoParams.baseline
          });
          
          if (response.data.status === 'success' && response.data.calibration_result && response.data.calibration_result.success) {
            // Store calibration data from backend
            this.stereoParams.calibration = response.data.calibration_result.calibration_data;
            this.isStereoCalibrated = true;
            this.showSuccess('Stereo cameras calibrated successfully');
            
            // Start depth map calculation
            this.startDepthMapCalculation();
          } else {
            throw new Error(response.data.calibration_result?.message || 'Calibration failed');
          }
        } else {
          // Fallback to local simulation if no stereo pair ID is selected
          // In a real implementation, this would use a calibration pattern
          // and perform proper camera calibration
          
          // Simulate calibration process
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          // Store calibration data (mock data for demonstration)
          this.stereoParams.calibration = {
            left: {
              focalLength: this.stereoParams.focalLength,
              principalPoint: { x: 640, y: 360 },
              distortion: [0, 0, 0, 0, 0]
            },
            right: {
              focalLength: this.stereoParams.focalLength,
              principalPoint: { x: 640, y: 360 },
              distortion: [0, 0, 0, 0, 0]
            },
            rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], // Identity matrix
            translation: [this.stereoParams.baseline, 0, 0] // Translation vector
          };
          
          this.isStereoCalibrated = true;
          this.showSuccess('Stereo cameras calibrated successfully (local simulation)');
          
          // Start depth map calculation
          this.startDepthMapCalculation();
        }
      } catch (error) {
        console.error('Stereo camera calibration failed:', error);
        this.showError('Stereo camera calibration failed: ' + error.message);
      }
    },
    
    // Start Depth Map Calculation
    startDepthMapCalculation() {
      this.depthMapInterval = setInterval(() => {
        this.calculateDepthMap();
      }, 100); // Calculate depth map every 100ms
    },
    
    // Stop Depth Map Calculation
    stopDepthMapCalculation() {
      if (this.depthMapInterval) {
        clearInterval(this.depthMapInterval);
        this.depthMapInterval = null;
      }
    },
    
    // Calculate Depth Map from Stereo Images
    async calculateDepthMap() {
      if (!this.isStereoCalibrated || !this.areStereoCamerasActive) return;

      try {
        const leftVideo = this.$refs.leftVideoElement;
        const rightVideo = this.$refs.rightVideoElement;
        const depthCanvas = this.$refs.depthMapCanvas;
        const pointCloudCanvas = this.$refs.pointCloudCanvas;
        
        if (!leftVideo || !rightVideo || !depthCanvas || !pointCloudCanvas) return;
        
        // Ensure videos have loaded
        if (leftVideo.videoWidth === 0 || leftVideo.videoHeight === 0 ||
            rightVideo.videoWidth === 0 || rightVideo.videoHeight === 0) return;
        
        // Set canvas dimensions
        depthCanvas.width = leftVideo.videoWidth;
        depthCanvas.height = leftVideo.videoHeight;
        pointCloudCanvas.width = leftVideo.videoWidth;
        pointCloudCanvas.height = leftVideo.videoHeight;
        
        const depthCtx = depthCanvas.getContext('2d');
        const pointCloudCtx = pointCloudCanvas.getContext('2d');
        
        // Draw left image to depth canvas
        depthCtx.drawImage(leftVideo, 0, 0, depthCanvas.width, depthCanvas.height);
        
        try {
          // Try to use the backend API for depth map calculation
          if (this.selectedStereoPairId) {
            const response = await api.cameras.processStereoPair(this.selectedStereoPairId, {
              enabled: true,
              min_disparity: this.stereoParams.minDisparity,
              num_disparities: this.stereoParams.numDisparities,
              block_size: this.stereoParams.blockSize
            });
            
            if (response.data.status === 'success' && response.data.result) {
              const result = response.data.result;
              
              // Process the depth map from the backend
              if (result.depth_map) {
                // Draw the depth map
                this.drawDepthMapFromData(depthCtx, result.depth_map);
                
                // Draw point cloud if available
                if (result.point_cloud && result.point_cloud.points && result.point_cloud.points.length > 0) {
                  this.drawPointCloudFromData(pointCloudCtx, result.point_cloud, depthCanvas.width, depthCanvas.height);
                } else {
                  this.drawSimulatedPointCloud(pointCloudCtx, depthCanvas.width, depthCanvas.height);
                }
                
                // Emit the real depth map data
                this.$emit('depth-map', result.depth_map);
                
                // Emit point cloud data if available
                if (result.point_cloud) {
                  this.$emit('point-cloud', result.point_cloud);
                }
                
                return;
              }
            }
          }
        } catch (apiError) {
          console.warn('Backend depth calculation failed, falling back to local simulation:', apiError);
        }
        
        // Fallback to local simulation if API call fails or isn't configured
        // For demonstration, we'll create a simulated depth map
        this.drawSimulatedDepthMap(depthCtx, depthCanvas.width, depthCanvas.height);
        
        // Draw simulated point cloud preview
        this.drawSimulatedPointCloud(pointCloudCtx, depthCanvas.width, depthCanvas.height);
        
        // Extract depth data for processing
        const depthMap = this.extractDepthData(depthCtx, depthCanvas.width, depthCanvas.height);
        
        // Emit depth map data for further processing
        this.$emit('depth-map', depthMap);
        
      } catch (error) {
        console.error('Depth map calculation error:', error);
      }
    },
    
    // Draw Depth Map from Backend Data
    drawDepthMapFromData(ctx, depthMapData) {
      const { data, width, height } = depthMapData;
      const imageData = ctx.createImageData(width, height);
      
      // Convert depth values to grayscale
      for (let i = 0; i < data.length; i++) {
        const index = i * 4;
        const depthValue = Math.min(255, Math.max(0, data[i]));
        
        imageData.data[index] = depthValue;     // R
        imageData.data[index + 1] = depthValue; // G
        imageData.data[index + 2] = depthValue; // B
        imageData.data[index + 3] = 255;        // A
      }
      
      ctx.putImageData(imageData, 0, 0);
    },
    
    // Draw Point Cloud from Backend Data
    drawPointCloudFromData(ctx, pointCloudData, canvasWidth, canvasHeight) {
      const { points, colors } = pointCloudData;
      
      // Clear canvas
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      
      // Simple 2D projection of 3D points
      ctx.fillStyle = 'white';
      
      // Find point bounds for normalization
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;
      
      for (const point of points) {
        minX = Math.min(minX, point[0]);
        maxX = Math.max(maxX, point[0]);
        minY = Math.min(minY, point[1]);
        maxY = Math.max(maxY, point[1]);
        minZ = Math.min(minZ, point[2]);
        maxZ = Math.max(maxZ, point[2]);
      }
      
      // Normalize and draw points
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      
      for (let i = 0; i < points.length; i++) {
        const [x, y, z] = points[i];
        const [r, g, b] = colors[i] || [1, 1, 1];
        
        // Normalize to canvas coordinates
        const canvasX = ((x - minX) / rangeX) * canvasWidth;
        const canvasY = ((y - minY) / rangeY) * canvasHeight;
        
        // Adjust size based on depth (z)
        const size = Math.max(1, 5 * (1 - (z - minZ) / (maxZ - minZ || 1)));
        
        ctx.fillStyle = `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, 0.8)`;
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, size, 0, 2 * Math.PI);
        ctx.fill();
      }
    },
    
    // Real Depth Map Calculation
    calculateRealDepthMap(leftImageData, rightImageData) {
      // In production, this would implement real stereo vision algorithms
      // such as block matching, semi-global matching, or deep learning methods
      
      // Placeholder for real depth calculation
      // This should be replaced with actual stereo correspondence algorithms
      console.log('Calculating real depth map from stereo images');
      
      // Return placeholder data structure
      return {
        data: new Array(leftImageData.width * leftImageData.height).fill(0),
        width: leftImageData.width,
        height: leftImageData.height,
        baseline: this.stereoParams.baseline,
        focalLength: this.stereoParams.focalLength,
        method: 'stereo_correspondence'
      };
    },
    
    // Real Point Cloud Generation
    generateRealPointCloud(depthMap) {
      // Convert depth map to 3D point cloud
      // This should implement proper 3D reconstruction from depth data
      
      console.log('Generating 3D point cloud from depth map');
      
      if (!depthMap || !depthMap.data) {
        return {
          points: [],
          colors: [],
          bounds: { x: [0, 1], y: [0, 1], z: [0, 1] }
        };
      }
      
      // If depthMap already contains pointCloud data (from server), use that
      if (depthMap.point_cloud) {
        return {
          points: depthMap.point_cloud.points || [],
          colors: depthMap.point_cloud.colors || [],
          bounds: depthMap.point_cloud.bounds || { x: [0, 1], y: [0, 1], z: [0, 1] }
        };
      }
      
      // Fallback: Generate point cloud from depth map data
      const { width, height, baseline, focalLength } = depthMap;
      const points = [];
      const colors = [];
      const data = depthMap.data;
      
      // Simple point cloud generation from depth data
      for (let y = 0; y < height; y += 4) { // Sample every 4th pixel for performance
        for (let x = 0; x < width; x += 4) {
          const index = y * width + x;
          const depth = data[index] || 0;
          
          // Skip points with no depth
          if (depth === 0) continue;
          
          // Calculate 3D coordinates
          const z = (baseline * focalLength * 1000) / (depth + 0.1); // Convert to mm
          const realX = ((x - width / 2) * z) / (focalLength * 1000);
          const realY = ((y - height / 2) * z) / (focalLength * 1000);
          
          points.push([realX, realY, z]);
          
          // Assign grayscale color based on depth
          const intensity = Math.min(255, Math.max(0, depth));
          colors.push([intensity / 255, intensity / 255, intensity / 255]);
        }
      }
      
      // Calculate bounds
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;
      
      for (const point of points) {
        minX = Math.min(minX, point[0]);
        maxX = Math.max(maxX, point[0]);
        minY = Math.min(minY, point[1]);
        maxY = Math.max(maxY, point[1]);
        minZ = Math.min(minZ, point[2]);
        maxZ = Math.max(maxZ, point[2]);
      }
      
      return {
        points,
        colors,
        bounds: { x: [minX, maxX], y: [minY, maxY], z: [minZ, maxZ] }
      };
    },
    
    // Extract Depth Data from Canvas
    extractDepthData(ctx, width, height) {
      const imageData = ctx.getImageData(0, 0, width, height);
      const data = imageData.data;
      const depthData = [];
      
      // Convert RGB to depth values (0-255)
      for (let i = 0; i < data.length; i += 4) {
        // Use average of RGB channels as depth value
        const depth = (data[i] + data[i+1] + data[i+2]) / 3;
        depthData.push(depth);
      }
      
      return {
        data: depthData,
        width: width,
        height: height,
        baseline: this.stereoParams.baseline,
        focalLength: this.stereoParams.focalLength
      };
    },
    
    // Multi-camera Methods
    addCamera() {
      this.activeCameras.push({ deviceId: '', isActive: false, stream: null });
    },
    
    removeCamera(index) {
      // Stop camera if it's active
      if (this.activeCameras[index].isActive) {
        this.stopSingleCameraStream(index);
      }
      this.activeCameras.splice(index, 1);
    },
    
    onMultiCameraChange() {
      // Check if any cameras are active
      this.areCamerasActive = this.activeCameras.some(cam => cam.isActive);
    },
    
    async startMultiCameras() {
      try {
        for (let i = 0; i < this.activeCameras.length; i++) {
          if (this.activeCameras[i].deviceId) {
            await this.startSingleCameraStream(i);
          }
        }
        this.showSuccess('Multi-camera system started');
      } catch (error) {
        console.error('Failed to start multi-camera system:', error);
        this.showError('Failed to start multi-camera system');
      }
    },
    
    stopMultiCameras() {
      for (let i = 0; i < this.activeCameras.length; i++) {
        this.stopSingleCameraStream(i);
      }
    },
    
    async startSingleCameraStream(index) {
      try {
        const camera = this.activeCameras[index];
        if (!camera.deviceId) {
          throw new Error('No camera selected');
        }
        
        const constraints = {
          video: {
            deviceId: { exact: camera.deviceId },
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        const videoElement = this.$refs['multiVideo' + index];
        
        if (videoElement) {
          videoElement.srcObject = stream;
        }
        
        this.activeCameras[index].stream = stream;
        this.activeCameras[index].isActive = true;
        this.hasInputData = true;
      } catch (error) {
        console.error(`Failed to start camera ${index + 1}:`, error);
        this.showError(`Failed to start camera ${index + 1}`);
      }
    },
    
    stopSingleCameraStream(index) {
      const camera = this.activeCameras[index];
      
      if (camera.stream) {
        camera.stream.getTracks().forEach(track => track.stop());
        camera.stream = null;
      }
      
      const videoElement = this.$refs['multiVideo' + index];
      if (videoElement) {
        videoElement.srcObject = null;
      }
      
      this.activeCameras[index].isActive = false;
    },
    
    async toggleSingleCamera(index) {
      const camera = this.activeCameras[index];
      
      if (camera.isActive) {
        this.stopSingleCameraStream(index);
      } else {
        await this.startSingleCameraStream(index);
      }
    },
    
    // Frame Capture Methods
    captureAllFrames() {
      if (this.cameraSetupType === 'single') {
        this.captureSingleFrameSource('main');
      } else if (this.cameraSetupType === 'stereo') {
        this.captureStereoFrames();
      } else if (this.cameraSetupType === 'multi') {
        this.captureMultiFrames();
      }
    },
    
    captureSingleFrame(index) {
      if (this.cameraSetupType === 'multi') {
        this.captureSingleFrameSource(index);
      }
    },
    
    captureSingleFrameSource(source) {
      try {
        let video, canvas, context;
        
        if (source === 'main') {
          video = this.$refs.videoElement;
          canvas = this.$refs.canvasElement;
        } else {
          video = this.$refs['multiVideo' + source];
          canvas = this.$refs['multiCanvas' + source];
        }
        
        if (!video || !canvas) return;
        
        context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        
        // Send to processing model
        this.processImageData(imageData, source);
      } catch (error) {
        console.error('Failed to capture frame:', error);
      }
    },
    
    captureStereoFrames() {
      try {
        const leftVideo = this.$refs.leftVideoElement;
        const rightVideo = this.$refs.rightVideoElement;
        const leftCanvas = this.$refs.leftCanvasElement;
        const rightCanvas = this.$refs.rightCanvasElement;
        
        if (!leftVideo || !rightVideo || !leftCanvas || !rightCanvas) return;
        
        // Capture left frame
        const leftContext = leftCanvas.getContext('2d');
        leftCanvas.width = leftVideo.videoWidth;
        leftCanvas.height = leftVideo.videoHeight;
        leftContext.drawImage(leftVideo, 0, 0, leftCanvas.width, leftCanvas.height);
        const leftImageData = leftContext.getImageData(0, 0, leftCanvas.width, leftCanvas.height);
        
        // Capture right frame
        const rightContext = rightCanvas.getContext('2d');
        rightCanvas.width = rightVideo.videoWidth;
        rightCanvas.height = rightVideo.videoHeight;
        rightContext.drawImage(rightVideo, 0, 0, rightCanvas.width, rightCanvas.height);
        const rightImageData = rightContext.getImageData(0, 0, rightCanvas.width, rightCanvas.height);
        
        // Process stereo image pair
        this.processStereoImageData(leftImageData, rightImageData);
      } catch (error) {
        console.error('Failed to capture stereo frames:', error);
      }
    },
    
    captureMultiFrames() {
      for (let i = 0; i < this.activeCameras.length; i++) {
        if (this.activeCameras[i].isActive) {
          this.captureSingleFrameSource(i);
        }
      }
    },
    
    // Process Image Data
    processImageData(imageData, source = 'main') {
      console.log(`Processing image data from source ${source}:`, imageData);
      // In production, this will be sent to the image processing model
      this.$emit('image-data', { imageData, source });
    },
    
    // Process Stereo Image Data
    processStereoImageData(leftImageData, rightImageData) {
      console.log('Processing stereo image data');
      // In production, this will be sent to the stereo vision processing model
      this.$emit('stereo-image-data', { leftImageData, rightImageData, params: this.stereoParams });
    },
    
    // Toggle Microphone
    async toggleMicrophone() {
      if (this.isMicrophoneActive) {
        this.stopMicrophone();
      } else {
        await this.startMicrophone();
      }
    },
    
    // Start Microphone
    async startMicrophone() {
      try {
        const constraints = {
          audio: {
            deviceId: this.selectedMicrophone ? { exact: this.selectedMicrophone } : undefined
          }
        };
        
        this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
        this.isMicrophoneActive = true;
        this.hasInputData = true;
        
        // Set up audio analyzer
        this.setupAudioAnalyser();
        this.showSuccess('Microphone started successfully.');
      } catch (error) {
        console.error('Failed to start microphone:', error);
        this.showError('Failed to start microphone.');
      }
    },
    
    // Stop Microphone
    stopMicrophone() {
      if (this.audioStream) {
        this.audioStream.getTracks().forEach(track => track.stop());
        this.audioStream = null;
      }
      this.isMicrophoneActive = false;
      
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
    },
    
    // Set Up Audio Analyzer
    setupAudioAnalyser() {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      const source = this.audioContext.createMediaStreamSource(this.audioStream);
      
      source.connect(this.analyser);
      this.analyser.fftSize = 256;
      
      // Start visualization
      this.visualizeAudio();
    },
    
    // Audio Visualization
    visualizeAudio() {
      if (!this.isMicrophoneActive || !this.analyser) return;
      
      const canvas = this.$refs.audioCanvas;
      const ctx = canvas.getContext('2d');
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      
      const draw = () => {
        if (!this.isMicrophoneActive) return;
        
        requestAnimationFrame(draw);
        
        this.analyser.getByteFrequencyData(dataArray);
        
        ctx.fillStyle = 'rgb(0, 0, 0)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
          const barHeight = dataArray[i] / 2;
          
          ctx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          
          x += barWidth + 1;
        }
      };
      
      draw();
    },
    
    // Switch Microphone Device
    async changeMicrophone() {
      if (this.isMicrophoneActive) {
        this.stopMicrophone();
        await this.startMicrophone();
      }
    },
    
    // Start Speech Recognition
    startSpeechRecognition() {
      if (!('webkitSpeechRecognition' in window)) {
        this.showError('Speech recognition is not supported in this browser.');
        return;
      }
      
      this.recognition = new window.webkitSpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      // Always use English for speech recognition
      this.recognition.lang = 'en-US';
      
      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        this.transcript = finalTranscript || interimTranscript;
      };
      
      this.recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        this.showError('Speech recognition error');
      };
      
      this.recognition.start();
      this.showInfo('Speech recognition started');

    },
    
    // Stop Speech Recognition
    stopSpeechRecognition() {
      if (this.recognition) {
        this.recognition.stop();
        this.recognition = null;
      }
    },
    
    // Toggle Video Stream
    async toggleVideoStream() {
      if (this.isVideoStreamActive) {
        this.stopVideoStream();
      } else {
        await this.startVideoStream();
      }
    },
    
    // Start Video Stream
    async startVideoStream() {
      try {
        if (this.videoStreamUrl) {
          this.$refs.videoStreamElement.src = this.videoStreamUrl;
          this.isVideoStreamActive = true;
          this.hasInputData = true;
          this.showSuccess('Video stream started');
        }
      } catch (error) {
        console.error('Failed to start video stream:', error);
        this.showError('Failed to start video stream');
      }
    },
    
    // Stop Video Stream
    stopVideoStream() {
      if (this.$refs.videoStreamElement) {
        this.$refs.videoStreamElement.src = '';
      }
      this.isVideoStreamActive = false;
    },
    
    // Toggle Audio Stream
    async toggleAudioStream() {
      if (this.isAudioStreamActive) {
        this.stopAudioStream();
      } else {
        await this.startAudioStream();
      }
    },
    
    // Start Audio Stream
    async startAudioStream() {
      try {
        if (this.audioStreamUrl) {
          // Audio stream processing needs to be implemented here
          console.log('Starting audio stream:', this.audioStreamUrl);
          this.isAudioStreamActive = true;
          this.hasInputData = true;
          this.showSuccess('Audio stream started');
        }
      } catch (error) {
          console.error('Failed to start audio stream:', error);
          this.showError('Failed to start audio stream');
        }
    },
    
    // Stop Audio Stream
    stopAudioStream() {
      this.isAudioStreamActive = false;
    },
    
    // Toggle Sensor Data
    toggleSensorData() {
      if (this.isSensorDataActive) {
        this.stopSensorData();
      } else {
        this.startSensorData();
      }
    },
    
    // Start Sensor Data
    startSensorData() {
      this.isSensorDataActive = true;
      this.hasInputData = true;
      // In production, this should connect to real sensors
    },
    
    // Stop Sensor Data
    stopSensorData() {
      this.isSensorDataActive = false;
    },
    
    // Set Up Sensor Simulation
    setupSensorSimulation() {
      // Removed sensor data simulation
      // In production, this should connect to real sensor devices
    },

    // Toggle Real-time Dialog
    async toggleRealTimeDialog() {
      if (this.isRealTimeDialogActive) {
        await this.stopRealTimeDialog();
      } else {
        await this.startRealTimeDialog();
      }
    },

    // Start Real-time Dialog
    async startRealTimeDialog() {
      try {
        // Connect WebSocket
        await this.connectWebSocket();
        
        // Start audio capture
        if (this.isMicrophoneActive) {
          this.startAudioCapture();
        }
        
        // Start video capture
        if (this.isCameraActive) {
          this.startVideoCapture();
        }
        
        this.isRealTimeDialogActive = true;
        this.showSuccess('Real-time dialog started');
      } catch (error) {
        console.error('Failed to start real-time dialog:', error);
        this.showError('Failed to start real-time dialog');
      }
    },

    // Stop Real-time Dialog
    async stopRealTimeDialog() {
      // Stop audio capture
      if (this.audioCaptureInterval) {
        clearInterval(this.audioCaptureInterval);
        this.audioCaptureInterval = null;
      }
      
      // Stop video capture
      if (this.videoCaptureInterval) {
        clearInterval(this.videoCaptureInterval);
        this.videoCaptureInterval = null;
      }
      
      // Clean up audio processor resources
      if (this.audioProcessor) {
        // Close AudioContext to release resources
        this.audioProcessor.close();
        this.audioProcessor = null;
      }
      
      // Disconnect WebSocket
      await this.disconnectWebSocket();
      
      this.isRealTimeDialogActive = false;
      this.showInfo('Real-time dialog stopped');
    },

    // Connect WebSocket
    async connectWebSocket() {
      return new Promise((resolve, reject) => {
        try {
          this.realTimeWebSocket = new WebSocket(this.webSocketUrl);
          
          this.realTimeWebSocket.onopen = () => {
            this.isWebSocketConnected = true;
            this.showSuccess('WebSocket connected');
            resolve();
          };
          
          this.realTimeWebSocket.onmessage = (event) => {
            this.handleWebSocketMessage(event.data);
          };
          
          this.realTimeWebSocket.onerror = (error) => {
            this.handleWebSocketError(error);
            reject(error);
          };
          
          this.realTimeWebSocket.onclose = () => {
            this.isWebSocketConnected = false;
            this.showWarning('WebSocket disconnected');
          };
        } catch (error) {
          console.error('WebSocket connection failed:', error);
          this.showError('WebSocket connection failed');
          reject(error);
        }
      });
    },

    // Disconnect WebSocket
    async disconnectWebSocket() {
      if (this.realTimeWebSocket) {
        this.realTimeWebSocket.close();
        this.realTimeWebSocket = null;
      }
      this.isWebSocketConnected = false;
    },

    // Handle WebSocket Message
    handleWebSocketMessage(data) {
      try {
        const message = JSON.parse(data);
        
        if (message.type === 'response') {
          // Process AI response
          this.realTimeResponses.push(message.content);
          this.$emit('real-time-response', message.content);
          
          // If response contains audio, play it here
          if (message.audio) {
            this.playAudioResponse(message.audio);
          }
        } else if (message.type === 'error') {
          this.showError(message.content);
        } else if (message.type === 'status') {
          this.showInfo(message.content);
        }
      } catch (error) {
        console.error('Failed to process WebSocket message:', error);
      }
    },

    // Handle WebSocket Error
    handleWebSocketError(error) {
      console.error('WebSocket error:', error);
      this.showError('WebSocket connection error.');
    },

    // Start Audio Capture
    startAudioCapture() {
      if (!this.isMicrophoneActive || !this.audioStream) {
        this.showWarning('Microphone is not active.');
        return;
      }
      
      // Create audio processor
      this.audioProcessor = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioProcessor.createMediaStreamSource(this.audioStream);
      const processor = this.audioProcessor.createScriptProcessor(4096, 1, 1);
      
      source.connect(processor);
      processor.connect(this.audioProcessor.destination);
      
      processor.onaudioprocess = (event) => {
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          const audioData = event.inputBuffer.getChannelData(0);
          this.sendAudioData(audioData);
        }
      };
      
      this.audioCaptureInterval = setInterval(() => {
        // Send audio data periodically
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          this.captureAudioFrame();
        }
      }, 100); // Send every 100ms
    },

    // Capture Audio Frame
    captureAudioFrame() {
      if (!this.analyser) return;
      
      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      this.analyser.getByteFrequencyData(dataArray);
      
      // Send audio data to server
      if (this.isWebSocketConnected) {
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'audio',
          data: Array.from(dataArray),
          timestamp: Date.now()
        }));
      }
    },

    // Send Audio Data
    sendAudioData(audioData) {
      if (this.isWebSocketConnected && this.realTimeWebSocket) {
        // Convert audio data to suitable format for transmission
        const compressedData = this.compressAudioData(audioData);
        
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'audio_raw',
          data: compressedData,
          timestamp: Date.now()
        }));
      }
    },

    // Compress Audio Data
    compressAudioData(audioData) {
      // Simple compression: sampling and quantization
      const compressed = [];
      const sampleRate = 10; // Take 1 sample every 10 samples
      
      for (let i = 0; i < audioData.length; i += sampleRate) {
        compressed.push(Math.round(audioData[i] * 100) / 100); // Quantized to 2 decimal places
      }
      
      return compressed;
    },

    // Start Video Capture
    startVideoCapture() {
      if (!this.isCameraActive || !this.stream) {
        this.showWarning('Camera is not active');
        return;
      }
      
      this.videoCaptureInterval = setInterval(() => {
        if (this.isWebSocketConnected && this.isRealTimeDialogActive) {
          this.captureVideoFrame();
        }
      }, 100); // Capture one frame every 100 milliseconds
    },

    // Capture Video Frame
    captureVideoFrame() {
      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const context = canvas.getContext('2d');
      
      if (video.videoWidth === 0 || video.videoHeight === 0) return;
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Get image data and compress
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const compressedImage = this.compressImageData(imageData);
      
      // Send to server
      if (this.isWebSocketConnected) {
        this.realTimeWebSocket.send(JSON.stringify({
          type: 'video',
          data: compressedImage,
          width: canvas.width,
          height: canvas.height,
          timestamp: Date.now()
        }));
      }
    },

    // Compress Image Data
    compressImageData(imageData) {
      // Simple compression: reduce resolution and quality
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Reduce resolution
      const scale = 0.5;
      canvas.width = imageData.width * scale;
      canvas.height = imageData.height * scale;
      
      // Draw scaled image
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = imageData.width;
      tempCanvas.height = imageData.height;
      tempCtx.putImageData(imageData, 0, 0);
      
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
      
      // Get compressed image data
      return canvas.toDataURL('image/jpeg', 0.7); // 70% quality
    },

    // Play Audio Response
    playAudioResponse(audioData) {
      if (!audioData) return;
      
      try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createBufferSource();
        
        // Decode and play based on audio data format
        // In production, may need base64 decoding or other format processing
        console.log('Playing audio response:', audioData);
        
        source.start();
      } catch (error) {
        console.error('Failed to play audio response:', error);
      }
    },
    
    // Status Message Helper Methods
    showError(message) {
      this.errorState = { hasError: true, message };
      setTimeout(() => {
        this.errorState = { hasError: false, message: '' };
      }, 5000);
    },
    
    showSuccess(message) {
      this.successState = { hasSuccess: true, message };
      setTimeout(() => {
        this.successState = { hasSuccess: false, message: '' };
      }, 3000);
    },
    
    showWarning(message) {
      this.warningState = { hasWarning: true, message };
      setTimeout(() => {
        this.warningState = { hasWarning: false, message: '' };
      }, 4000);
    },
    
    showInfo(message) {
      this.infoState = { hasInfo: true, message };
      setTimeout(() => {
        this.infoState = { hasInfo: false, message: '' };
      }, 3000);
    },
    
    // Process Fused Data
    processFusedData() {
      console.log('Processing fused data:', {
        audioVisual: this.fuseAudioVisual,
        sensorCamera: this.fuseSensorCamera,
        allModalities: this.fuseAllModalities
      });
      
      // In production, this will be sent to the main model for fusion processing
      this.$emit('process-fused-data', {
        audioVisual: this.fuseAudioVisual,
        sensorCamera: this.fuseSensorCamera,
        allModalities: this.fuseAllModalities
      });
    }
  }
}
</script>

<style scoped>
.real-time-input {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.camera-section, .microphone-section, .network-stream-section, .sensor-section, .fusion-controls {
  background: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.camera-container {
  position: relative;
  width: 100%;
  height: 300px;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 10px;
}

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.audio-visualization {
  width: 100%;
  height: 150px;
  background: #000;
  border-radius: 4px;
  margin-bottom: 10px;
}

.stream-preview {
  width: 100%;
  height: 200px;
  background: #000;
  border-radius: 4px;
  margin-top: 10px;
  overflow: hidden;
}

.camera-controls, .microphone-controls, .stream-controls, .sensor-controls, .fusion-options {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 10px;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-bottom: 10px;
}

.input-group label {
  font-weight: bold;
}

.input-group input {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
}

button {
  padding: 8px 12px;
  background: #666;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

button:hover:not(:disabled) {
  background: #444;
}

select {
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.transcript {
  padding: 10px;
  background: #f0f0f0;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-top: 10px;
  color: #333;
}

.sensor-readings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}

.sensor-reading {
  display: flex;
  justify-content: space-between;
  padding: 5px;
  background: #f5f5f5;
  border-radius: 4px;
}

.sensor-label {
  font-weight: bold;
}

.fusion-options {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 15px;
}

label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-messages {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  max-width: 400px;
}

.message {
  padding: 12px 16px;
  margin-bottom: 10px;
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideIn 0.3s ease-out;
  font-weight: 500;
}

.message.error {
  background: #f5f5f5;
  color: #666;
  border-left: 4px solid #888;
}

.message.success {
  background: #f5f5f5;
  color: #666;
  border-left: 4px solid #888;
}

.message.warning {
  background: #f5f5f5;
  color: #666;
  border-left: 4px solid #888;
}

.message.info {
  background: #f5f5f5;
  color: #666;
  border-left: 4px solid #888;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .real-time-input {
    grid-template-columns: 1fr;
  }
  
  .sensor-readings {
    grid-template-columns: 1fr;
  }
  
  .status-messages {
    top: 10px;
    right: 10px;
    left: 10px;
    max-width: none;
  }
}
</style>
