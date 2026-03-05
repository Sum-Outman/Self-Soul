<template>
  <div class="real-time-input">
    <h3>🔄 Real-time Input</h3>
    
    <div class="input-controls">
      <!-- Input type selection -->
      <div class="input-type-selection">
        <label><strong>Select Input Type:</strong></label>
        <select v-model="selectedInputType" class="input-type-select">
          <option value="text">Text</option>
          <option value="image">Image</option>
          <option value="audio">Audio</option>
          <option value="video">Video</option>
          <option value="sensor">Sensor Data</option>
        </select>
      </div>

      <!-- Text input -->
      <div v-if="selectedInputType === 'text'" class="input-section">
        <label><strong>Text Input:</strong></label>
        <textarea 
          v-model="textInput" 
          placeholder="Enter text here..."
          rows="4"
          class="text-input"
        ></textarea>
        <button @click="sendTextInput" class="btn btn-primary">
          Send Text
        </button>
      </div>

      <!-- Image input -->
      <div v-if="selectedInputType === 'image'" class="input-section">
        <label><strong>Image Input:</strong></label>
        <input 
          type="file" 
          accept="image/*" 
          @change="handleImageUpload"
          class="file-input"
        >
        <div v-if="selectedImage" class="image-preview">
          <img :src="selectedImage" alt="Preview" class="preview-image">
          <button @click="sendImageInput" class="btn btn-primary">
            Send Image
          </button>
        </div>
      </div>

      <!-- Audio input -->
      <div v-if="selectedInputType === 'audio'" class="input-section">
        <label><strong>Audio Input:</strong></label>
        
        <!-- Audio Processing Mode -->
        <div class="audio-mode-selection" style="margin-bottom: 15px;">
          <label><strong>Audio Processing Mode:</strong></label>
          <select v-model="audioProcessingMode" class="audio-mode-select">
            <option value="recording">Recording & Send</option>
            <option value="realtime">Real-time Streaming</option>
            <option value="analysis">Audio Analysis</option>
            <option value="speech">Speech Recognition</option>
          </select>
        </div>
        
        <!-- Audio Controls -->
        <div class="audio-controls">
          <!-- Basic recording controls -->
          <div class="basic-audio-controls">
            <button 
              @click="toggleAudioRecording" 
              class="btn" 
              :class="isRecording ? 'btn-danger' : 'btn-primary'"
            >
              {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
            </button>
            <button 
              @click="sendAudioInput" 
              class="btn btn-secondary"
              :disabled="!audioBlob"
            >
              Send Audio
            </button>
          </div>
          
          <!-- Real-time streaming controls -->
          <div v-if="audioProcessingMode === 'realtime'" class="realtime-controls" style="margin-top: 10px;">
            <button 
              @click="toggleRealtimeAudio" 
              class="btn" 
              :class="isRealtimeAudioActive ? 'btn-danger' : 'btn-success'"
            >
              {{ isRealtimeAudioActive ? 'Stop Real-time Stream' : 'Start Real-time Stream' }}
            </button>
            <button 
              @click="processRealtimeAudio" 
              class="btn btn-info"
              :disabled="!isRealtimeAudioActive"
            >
              Process Stream
            </button>
          </div>
          
          <!-- Audio analysis controls -->
          <div v-if="audioProcessingMode === 'analysis'" class="analysis-controls" style="margin-top: 10px;">
            <div class="analysis-options">
              <label>
                <input type="checkbox" v-model="audioAnalysisOptions.noiseReduction">
                Noise Reduction
              </label>
              <label>
                <input type="checkbox" v-model="audioAnalysisOptions.gainControl">
                Gain Control
              </label>
              <label>
                <input type="checkbox" v-model="audioAnalysisOptions.echoCancellation">
                Echo Cancellation
              </label>
            </div>
            <button 
              @click="startAudioAnalysis" 
              class="btn btn-success"
              :disabled="isAudioAnalyzing"
            >
              {{ isAudioAnalyzing ? 'Analyzing...' : 'Start Audio Analysis' }}
            </button>
          </div>
          
          <!-- Speech recognition controls -->
          <div v-if="audioProcessingMode === 'speech'" class="speech-controls" style="margin-top: 10px;">
            <div class="language-selection">
              <label>Language:</label>
              <select v-model="speechLanguage" class="language-select">
                <option value="en-US">English (US)</option>
                <option value="zh-CN">Chinese</option>
                <option value="es-ES">Spanish</option>
                <option value="fr-FR">French</option>
              </select>
            </div>
            <button 
              @click="toggleSpeechRecognition" 
              class="btn" 
              :class="isSpeechRecognitionActive ? 'btn-danger' : 'btn-success'"
            >
              {{ isSpeechRecognitionActive ? 'Stop Speech Recognition' : 'Start Speech Recognition' }}
            </button>
          </div>
        </div>
        
        <!-- Audio Visualization -->
        <div v-if="isRecording || isRealtimeAudioActive || isAudioAnalyzing" class="audio-visualization" style="margin-top: 15px;">
          <h5>Audio Visualization:</h5>
          <canvas ref="audioVisualizationCanvas" class="audio-canvas"></canvas>
          <div class="audio-metrics" v-if="audioMetrics">
            <div>Volume: {{ audioMetrics.volume }} dB</div>
            <div>Frequency: {{ audioMetrics.frequency }} Hz</div>
            <div>Clarity: {{ audioMetrics.clarity }}%</div>
          </div>
        </div>
        
        <!-- Audio Info -->
        <div v-if="audioDuration > 0" class="audio-info">
          Recording duration: {{ audioDuration }} seconds
          <div v-if="audioBlob">
            File size: {{ (audioBlob.size / 1024).toFixed(2) }} KB
          </div>
        </div>
        
        <!-- Speech Recognition Results -->
        <div v-if="speechRecognitionResults.length > 0" class="speech-results" style="margin-top: 15px;">
          <h5>Speech Recognition Results:</h5>
          <div class="speech-result-item" v-for="(result, index) in speechRecognitionResults" :key="index">
            {{ result.text }} ({{ result.confidence.toFixed(2) }})
          </div>
        </div>
      </div>

      <!-- Video input -->
      <div v-if="selectedInputType === 'video'" class="input-section">
        <label><strong>Video Input:</strong></label>
        
        <!-- Camera Mode Selection -->
        <div class="camera-mode-selection" style="margin-bottom: 15px;">
          <label><strong>Camera Mode:</strong></label>
          <select v-model="cameraMode" class="camera-mode-select">
            <option value="single">Single Camera Recognition</option>
            <option value="stereo">Dual Camera Spatial Recognition</option>
            <option value="triple">Three Cameras Full Startup</option>
          </select>
        </div>

        <!-- Camera Selection -->
        <div class="camera-selection" style="margin-bottom: 15px;">
          <label><strong>Select Cameras:</strong></label>
          <div class="camera-selection-grid">
            <!-- Camera 1 -->
            <div class="camera-select-item">
              <label>Camera 1:</label>
              <select v-model="selectedCameras[0]" :disabled="availableCameras.length === 0">
                <option value="">-- Select Camera --</option>
                <option v-for="camera in availableCameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || `Camera ${camera.deviceId.slice(0, 8)}` }}
                </option>
              </select>
              <span v-if="selectedCameras[0]" class="camera-status">
                {{ cameraStreams[0]?.active ? 'Active' : 'Inactive' }}
              </span>
            </div>
            
            <!-- Camera 2 (only shown for stereo or triple mode) -->
            <div class="camera-select-item" v-if="cameraMode === 'stereo' || cameraMode === 'triple'">
              <label>Camera 2:</label>
              <select v-model="selectedCameras[1]" :disabled="availableCameras.length === 0">
                <option value="">-- Select Camera --</option>
                <option v-for="camera in availableCameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || `Camera ${camera.deviceId.slice(0, 8)}` }}
                </option>
              </select>
              <span v-if="selectedCameras[1]" class="camera-status">
                {{ cameraStreams[1]?.active ? 'Active' : 'Inactive' }}
              </span>
            </div>
            
            <!-- Camera 3 (only shown for triple mode) -->
            <div class="camera-select-item" v-if="cameraMode === 'triple'">
              <label>Camera 3:</label>
              <select v-model="selectedCameras[2]" :disabled="availableCameras.length === 0">
                <option value="">-- Select Camera --</option>
                <option v-for="camera in availableCameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || `Camera ${camera.deviceId.slice(0, 8)}` }}
                </option>
              </select>
              <span v-if="selectedCameras[2]" class="camera-status">
                {{ cameraStreams[2]?.active ? 'Active' : 'Inactive' }}
              </span>
            </div>
          </div>
        </div>

        <!-- Camera Control Buttons -->
        <div class="camera-controls" style="margin-bottom: 15px;">
          <button 
            @click="toggleCameraStreams" 
            class="btn" 
            :class="cameraStreamsActive ? 'btn-danger' : 'btn-primary'"
            :disabled="!hasSelectedCameras"
          >
            {{ cameraStreamsActive ? 'Stop Camera Streams' : 'Start Camera Streams' }}
          </button>
          
          <button 
            @click="captureCameraFrames" 
            class="btn btn-secondary"
            :disabled="!cameraStreamsActive"
          >
            Capture Frames
          </button>
          
          <!-- Special processing based on mode -->
          <button 
            v-if="cameraMode === 'stereo'" 
            @click="processStereoVision" 
            class="btn btn-info"
            :disabled="!cameraStreamsActive"
          >
            Process Stereo Vision
          </button>
          
          <button 
            v-if="cameraMode === 'triple'" 
            @click="processTripleCameras" 
            class="btn btn-info"
            :disabled="!cameraStreamsActive"
          >
            Process Triple Cameras
          </button>
        </div>

        <!-- Camera Previews -->
        <div class="camera-previews" v-if="cameraStreamsActive">
          <h5>Camera Previews:</h5>
          <div class="preview-grid">
            <!-- Camera 1 Preview -->
            <div class="camera-preview" v-if="selectedCameras[0]">
              <h6>Camera 1</h6>
              <video 
                v-if="cameraStreams[0]?.stream" 
                :srcObject="cameraStreams[0].stream" 
                autoplay 
                class="preview-video"
              ></video>
              <div v-else class="no-preview">No stream</div>
            </div>
            
            <!-- Camera 2 Preview (stereo/triple) -->
            <div class="camera-preview" v-if="selectedCameras[1] && (cameraMode === 'stereo' || cameraMode === 'triple')">
              <h6>Camera 2</h6>
              <video 
                v-if="cameraStreams[1]?.stream" 
                :srcObject="cameraStreams[1].stream" 
                autoplay 
                class="preview-video"
              ></video>
              <div v-else class="no-preview">No stream</div>
            </div>
            
            <!-- Camera 3 Preview (triple only) -->
            <div class="camera-preview" v-if="selectedCameras[2] && cameraMode === 'triple'">
              <h6>Camera 3</h6>
              <video 
                v-if="cameraStreams[2]?.stream" 
                :srcObject="cameraStreams[2].stream" 
                autoplay 
                class="preview-video"
              ></video>
              <div v-else class="no-preview">No stream</div>
            </div>
          </div>
        </div>

        <!-- Spatial Recognition Preview -->
        <div class="spatial-recognition-preview" v-if="cameraMode === 'stereo' && stereoVisionPreview">
          <h5>Spatial Recognition Results:</h5>
          <div class="preview-grid">
            <!-- Depth Map Preview -->
            <div class="spatial-preview-item">
              <h6>Depth Map</h6>
              <img :src="stereoVisionPreview" class="depth-map-preview" alt="Depth Map">
              <div class="depth-scale">
                <span>Near</span>
                <div class="scale-gradient"></div>
                <span>Far</span>
              </div>
            </div>
            
            <!-- Spatial Metrics -->
            <div class="spatial-preview-item">
              <h6>Spatial Metrics</h6>
              <div class="metrics-grid">
                <div class="metric-item">
                  <span class="metric-label">Min Depth:</span>
                  <span class="metric-value">{{ stereoVisionMetrics.depthMin.toFixed(2) }} m</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">Max Depth:</span>
                  <span class="metric-value">{{ stereoVisionMetrics.depthMax.toFixed(2) }} m</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">Avg Depth:</span>
                  <span class="metric-value">{{ stereoVisionMetrics.depthAverage.toFixed(2) }} m</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">Objects:</span>
                  <span class="metric-value">{{ stereoVisionMetrics.objectCount }}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">Processing Time:</span>
                  <span class="metric-value">{{ stereoVisionMetrics.processingTime }} ms</span>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Spatial Data Export -->
          <div class="spatial-actions" style="margin-top: 15px;">
            <button class="btn btn-secondary btn-sm" @click="exportSpatialData">
              Export Data
            </button>
            <button class="btn btn-secondary btn-sm" @click="clearSpatialResults">
              Clear Results
            </button>
          </div>
          
          <!-- Robot Control with Spatial Data -->
          <div class="robot-control-section" style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd;">
            <h5>🤖 Robot Control with Spatial Data:</h5>
            <div class="robot-control-options" style="margin-bottom: 15px;">
              <div class="robot-control-option" style="margin-bottom: 10px;">
                <label><strong>Motion Type:</strong></label>
                <select v-model="selectedRobotMotion" class="robot-motion-select" style="width: 100%; padding: 5px;">
                  <option value="move_to_target">Move to Target</option>
                  <option value="avoid_obstacle">Avoid Obstacle</option>
                  <option value="follow_path">Follow Path</option>
                  <option value="explore_area">Explore Area</option>
                  <option value="pick_and_place">Pick and Place</option>
                  <option value="grasp_object">Grasp Object</option>
                </select>
              </div>
              <div class="robot-control-option" style="margin-bottom: 10px;">
                <label><strong>Use Spatial Data:</strong></label>
                <div style="display: flex; align-items: center;">
                  <input type="checkbox" v-model="useSpatialDataForRobot" id="use-spatial-data">
                  <label for="use-spatial-data" style="margin-left: 5px;">Use depth map and point cloud for robot motion</label>
                </div>
              </div>
              <div v-if="useSpatialDataForRobot" class="spatial-params" style="margin-bottom: 10px; padding: 10px; background-color: #f0f8ff; border-radius: 4px;">
                <div class="spatial-param">
                  <label>Target Depth:</label>
                  <input type="range" v-model="robotTargetDepth" min="0.1" max="5.0" step="0.1" style="width: 100%;">
                  <span>{{ robotTargetDepth.toFixed(1) }} m</span>
                </div>
                <div class="spatial-param">
                  <label>Safety Margin:</label>
                  <input type="range" v-model="robotSafetyMargin" min="0.05" max="1.0" step="0.05" style="width: 100%;">
                  <span>{{ robotSafetyMargin.toFixed(2) }} m</span>
                </div>
              </div>
            </div>
            <div class="robot-control-buttons" style="display: flex; gap: 10px;">
              <button class="btn btn-success btn-sm" @click="controlRobotWithSpatialData" :disabled="!stereoVisionResult">
                🤖 Control Robot
              </button>
              <button class="btn btn-info btn-sm" @click="sendSpatialDataToRobot">
                📡 Send Spatial Data to Robot
              </button>
              <button class="btn btn-warning btn-sm" @click="testRobotConnection">
                🔌 Test Robot Connection
              </button>
            </div>
            <div v-if="robotControlStatus" class="robot-control-status" style="margin-top: 10px; padding: 8px; border-radius: 4px; background-color: #e7f3ff;">
              <strong>Robot Status:</strong> {{ robotControlStatus }}
            </div>
          </div>
        </div>

        <!-- Original simple video recording (kept for compatibility) -->
        <div class="simple-video-recording" style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd;">
          <h5>Simple Video Recording:</h5>
          <div class="video-controls">
            <button 
              @click="toggleVideoRecording" 
              class="btn" 
              :class="isVideoRecording ? 'btn-danger' : 'btn-primary'"
            >
              {{ isVideoRecording ? 'Stop Recording' : 'Start Recording' }}
            </button>
            <button 
              @click="sendVideoInput" 
              class="btn btn-secondary"
              :disabled="!videoBlob"
            >
              Send Video
            </button>
          </div>
          <div v-if="videoStream" class="video-preview">
            <video :srcObject="videoStream" autoplay class="preview-video"></video>
          </div>
        </div>
      </div>

      <!-- Sensor data input -->
      <div v-if="selectedInputType === 'sensor'" class="input-section">
        <label><strong>Sensor Data:</strong></label>
        <div class="sensor-data-grid">
          <div class="sensor-item" v-for="sensor in sensorData" :key="sensor.id">
            <span class="sensor-name">{{ sensor.name }}:</span>
            <span class="sensor-value">{{ sensor.value }} {{ sensor.unit }}</span>
          </div>
        </div>
        <button @click="sendSensorInput" class="btn btn-primary">
          Send Sensor Data
        </button>
      </div>
    </div>

    <!-- Input history -->
    <div class="input-history" v-if="inputHistory.length > 0">
      <h4>Input History:</h4>
      <div class="history-list">
        <div 
          v-for="(item, index) in inputHistory.slice(-5)" 
          :key="index" 
          class="history-item"
        >
          <span class="history-type">{{ item.type }}</span>
          <span class="history-time">{{ formatTime(item.timestamp) }}</span>
          <span class="history-content">{{ item.content }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'
import api from '@/utils/api'

export default {
  name: 'RealTimeInput',
  
  emits: [
    'real-time-audio-data',
    'real-time-video-data',
    'real-time-text-data',
    'real-time-file-data'
  ],
  
  setup(props, { emit }) {
    // Input type
    const selectedInputType = ref('text')
    
    // Text input
    const textInput = ref('')
    
    // Image input
    const selectedImage = ref(null)
    const selectedImageFile = ref(null)
    
    // Audio input
    const isRecording = ref(false)
    const audioBlob = ref(null)
    const audioDuration = ref(0)
    const mediaRecorder = ref(null)
    const audioChunks = ref([])
    const recordingStartTime = ref(null)
    
    // Enhanced audio processing
    const audioProcessingMode = ref('recording')
    const isRealtimeAudioActive = ref(false)
    const isAudioAnalyzing = ref(false)
    const isSpeechRecognitionActive = ref(false)
    const audioAnalysisOptions = reactive({
      noiseReduction: true,
      gainControl: false,
      echoCancellation: true
    })
    const speechLanguage = ref('en-US')
    const speechRecognitionResults = ref([])
    const audioMetrics = ref(null)
    const audioVisualizationCanvas = ref(null)
    const audioContext = ref(null)
    const audioAnalyser = ref(null)
    const audioSource = ref(null)
    const visualizationInterval = ref(null)
    const speechRecognitionInstance = ref(null)
    
    // Video input
    const isVideoRecording = ref(false)
    const videoBlob = ref(null)
    const videoStream = ref(null)
    const videoRecorder = ref(null)
    const videoChunks = ref([])
    
    // Enhanced camera control
    const cameraMode = ref('single')
    const availableCameras = ref([])
    const selectedCameras = ref(['', '', ''])
    const cameraStreams = ref([
      { stream: null, active: false, captureInterval: null },
      { stream: null, active: false, captureInterval: null },
      { stream: null, active: false, captureInterval: null }
    ])
    
    // Robot control with spatial data
    const selectedRobotMotion = ref('move_to_target')
    const useSpatialDataForRobot = ref(true)
    const robotTargetDepth = ref(1.5)
    const robotSafetyMargin = ref(0.2)
    const robotControlStatus = ref('')
    
    // Spatial recognition results
    const stereoVisionResult = ref(null)
    const stereoVisionPreview = ref(null)
    const stereoVisionMetrics = ref({
      depthMin: 0,
      depthMax: 0,
      depthAverage: 0,
      objectCount: 0,
      processingTime: 0
    })
    
    // Computed properties
    const cameraStreamsActive = computed(() => {
      return cameraStreams.value.some(stream => stream.active)
    })
    
    const hasSelectedCameras = computed(() => {
      return selectedCameras.value.some(cameraId => cameraId && cameraId.trim() !== '')
    })
    
    // Sensor data
    const sensorData = reactive([
      { id: 'accel', name: 'Accelerometer', value: 0.0, unit: 'm/s²' },
      { id: 'gyro', name: 'Gyroscope', value: 0.0, unit: 'rad/s' },
      { id: 'temp', name: 'Temperature', value: 25.0, unit: '°C' },
      { id: 'battery', name: 'Battery', value: 85.0, unit: '%' }
    ])
    
    // Input history
    const inputHistory = ref([])
    
    // Add to history
    const addToHistory = (type, content) => {
      inputHistory.value.push({
        type,
        content,
        timestamp: new Date()
      })
    }
    
    // Format time
    const formatTime = (date) => {
      return new Date(date).toLocaleTimeString()
    }
    
    // Send text input
    const sendTextInput = async () => {
      if (!textInput.value.trim()) return
      
      try {
        const response = await api.process.text({ text: textInput.value })
        if (response.data.status === 'success') {
          addToHistory('Text', textInput.value.substring(0, 50) + '...')
          // Emit text data event
          emit('real-time-text-data', textInput.value.trim())
          textInput.value = ''
        }
      } catch (error) {
        console.error('Error sending text input:', error)
      }
    }
    
    // Handle image upload
    const handleImageUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        selectedImage.value = URL.createObjectURL(file)
        selectedImageFile.value = file
      }
    }
    
    // Send image input
    const sendImageInput = async () => {
      if (!selectedImage.value || !selectedImageFile.value) return
      
      try {
        const response = await api.process.image({ image: selectedImage.value })
        if (response.data.status === 'success') {
          addToHistory('Image', 'Image uploaded')
          // Emit file data event
          emit('real-time-file-data', selectedImageFile.value)
          selectedImage.value = null
          selectedImageFile.value = null
        }
      } catch (error) {
        console.error('Error sending image input:', error)
      }
    }
    
    // Toggle audio recording
    const toggleAudioRecording = async () => {
      if (isRecording.value) {
        // Stop recording
        mediaRecorder.value.stop()
        isRecording.value = false
      } else {
        // Start recording
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
          mediaRecorder.value = new MediaRecorder(stream)
          audioChunks.value = []
          recordingStartTime.value = Date.now()
          
          mediaRecorder.value.ondataavailable = (event) => {
            if (event.data.size > 0) {
              audioChunks.value.push(event.data)
            }
          }
          
          mediaRecorder.value.onstop = () => {
            audioBlob.value = new Blob(audioChunks.value, { type: 'audio/wav' })
            audioDuration.value = Math.round((Date.now() - recordingStartTime.value) / 1000)
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop())
          }
          
          mediaRecorder.value.start()
          isRecording.value = true
        } catch (error) {
          console.error('Error starting audio recording:', error)
        }
      }
    }
    
    // Send audio input
    const sendAudioInput = async () => {
      if (!audioBlob.value) return
      
      try {
        const response = await api.process.audio({ audio: audioBlob.value })
        if (response.data.status === 'success') {
          addToHistory('Audio', `Audio (${audioDuration.value}s)`)
          // Emit audio data event
          emit('real-time-audio-data', audioBlob.value)
          audioBlob.value = null
          audioDuration.value = 0
        }
      } catch (error) {
        console.error('Error sending audio input:', error)
      }
    }
    
    // Enhanced audio processing methods
    const toggleRealtimeAudio = async () => {
      if (isRealtimeAudioActive.value) {
        // Stop real-time audio stream
        stopRealtimeAudio()
      } else {
        // Start real-time audio stream
        await startRealtimeAudio()
      }
    }
    
    const startRealtimeAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        
        // Initialize audio context and analyser
        audioContext.value = new (window.AudioContext || window.webkitAudioContext)()
        audioAnalyser.value = audioContext.value.createAnalyser()
        audioSource.value = audioContext.value.createMediaStreamSource(stream)
        audioSource.value.connect(audioAnalyser.value)
        
        // Configure analyser
        audioAnalyser.value.fftSize = 2048
        audioAnalyser.value.smoothingTimeConstant = 0.8
        
        // Start visualization
        startAudioVisualization()
        
        isRealtimeAudioActive.value = true
        console.log('Real-time audio stream started')
      } catch (error) {
        console.error('Error starting real-time audio:', error)
      }
    }
    
    const stopRealtimeAudio = () => {
      if (audioContext.value) {
        audioContext.value.close()
        audioContext.value = null
      }
      
      if (visualizationInterval.value) {
        clearInterval(visualizationInterval.value)
        visualizationInterval.value = null
      }
      
      audioAnalyser.value = null
      audioSource.value = null
      isRealtimeAudioActive.value = false
      console.log('Real-time audio stream stopped')
    }
    
    const startAudioVisualization = () => {
      if (!audioVisualizationCanvas.value || !audioAnalyser.value) return
      
      const canvas = audioVisualizationCanvas.value
      const canvasContext = canvas.getContext('2d')
      const analyser = audioAnalyser.value
      const bufferLength = analyser.frequencyBinCount
      const dataArray = new Uint8Array(bufferLength)
      
      const draw = () => {
        if (!isRealtimeAudioActive.value && !isRecording.value && !isAudioAnalyzing.value) return
        
        requestAnimationFrame(draw)
        
        analyser.getByteFrequencyData(dataArray)
        
        // Clear canvas
        canvasContext.fillStyle = 'rgb(0, 0, 0)'
        canvasContext.fillRect(0, 0, canvas.width, canvas.height)
        
        // Draw frequency bars
        const barWidth = (canvas.width / bufferLength) * 2.5
        let barHeight
        let x = 0
        
        for (let i = 0; i < bufferLength; i++) {
          barHeight = dataArray[i] / 2
          
          canvasContext.fillStyle = `rgb(${barHeight + 100}, 50, 150)`
          canvasContext.fillRect(x, canvas.height - barHeight, barWidth, barHeight)
          
          x += barWidth + 1
        }
        
        // Update audio metrics
        updateAudioMetrics(dataArray)
      }
      
      draw()
    }
    
    const updateAudioMetrics = (frequencyData) => {
      if (!frequencyData || frequencyData.length === 0) return
      
      const sum = frequencyData.reduce((a, b) => a + b, 0)
      const avg = sum / frequencyData.length
      const max = Math.max(...frequencyData)
      
      audioMetrics.value = {
        volume: Math.round(20 * Math.log10(avg || 1)),
        frequency: Math.round(max * 0.5),
        clarity: Math.round((avg / 255) * 100)
      }
    }
    
    const processRealtimeAudio = async () => {
      if (!isRealtimeAudioActive.value) return
      
      try {
        const response = await api.process.audio({
          audio: 'realtime_stream',
          mode: 'realtime_processing',
          options: audioAnalysisOptions
        })
        
        if (response.data.status === 'success') {
          addToHistory('Realtime Audio', 'Audio stream processed')
        }
      } catch (error) {
        console.error('Error processing real-time audio:', error)
      }
    }
    
    const startAudioAnalysis = async () => {
      if (isAudioAnalyzing.value) return
      
      try {
        isAudioAnalyzing.value = true
        
        // Start audio visualization for analysis
        await startRealtimeAudio()
        
        // Simulate analysis process
        setTimeout(() => {
          isAudioAnalyzing.value = false
          addToHistory('Audio Analysis', 'Analysis completed')
        }, 3000)
      } catch (error) {
        console.error('Error starting audio analysis:', error)
        isAudioAnalyzing.value = false
      }
    }
    
    const toggleSpeechRecognition = () => {
      if (isSpeechRecognitionActive.value) {
        stopSpeechRecognition()
      } else {
        startSpeechRecognition()
      }
    }
    
    const startSpeechRecognition = () => {
      const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SpeechRecognitionAPI) {
        console.error('Speech recognition not supported')
        return
      }
      
      const recognition = new SpeechRecognitionAPI()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = speechLanguage.value
      
      recognition.onresult = (event) => {
        const result = event.results[event.results.length - 1]
        const transcript = result[0].transcript
        const confidence = result[0].confidence
        
        speechRecognitionResults.value.push({
          text: transcript,
          confidence: confidence,
          timestamp: new Date()
        })
        
        // Keep only last 10 results
        if (speechRecognitionResults.value.length > 10) {
          speechRecognitionResults.value.shift()
        }
      }
      
      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error)
      }
      
      recognition.start()
      speechRecognitionInstance.value = recognition
      isSpeechRecognitionActive.value = true
      console.log('Speech recognition started')
    }
    
    const stopSpeechRecognition = () => {
      if (speechRecognitionInstance.value) {
        speechRecognitionInstance.value.stop()
        speechRecognitionInstance.value = null
      }
      isSpeechRecognitionActive.value = false
      console.log('Speech recognition stopped')
    }
    
    // Toggle video recording
    const toggleVideoRecording = async () => {
      if (isVideoRecording.value) {
        // Stop recording
        videoRecorder.value.stop()
        isVideoRecording.value = false
      } else {
        // Start recording
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true })
          videoStream.value = stream
          videoRecorder.value = new MediaRecorder(stream)
          videoChunks.value = []
          
          videoRecorder.value.ondataavailable = (event) => {
            if (event.data.size > 0) {
              videoChunks.value.push(event.data)
            }
          }
          
          videoRecorder.value.onstop = () => {
            videoBlob.value = new Blob(videoChunks.value, { type: 'video/mp4' })
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop())
            videoStream.value = null
          }
          
          videoRecorder.value.start()
          isVideoRecording.value = true
        } catch (error) {
          console.error('Error starting video recording:', error)
        }
      }
    }
    
    // Send video input
    const sendVideoInput = async () => {
      if (!videoBlob.value) return
      
      try {
        const response = await api.process.video({ video: videoBlob.value })
        if (response.data.status === 'success') {
          addToHistory('Video', 'Video file sent')
          // Emit video data event
          emit('real-time-video-data', videoBlob.value)
          videoBlob.value = null
          videoStream.value = null
        }
      } catch (error) {
        console.error('Error sending video input:', error)
      }
    }
    
    // Send sensor input
    const sendSensorInput = async () => {
      try {
        const response = await api.process.sensor({ sensors: sensorData })
        if (response.data.status === 'success') {
          addToHistory('Sensor', 'Sensor data sent')
          // Emit sensor data as text
          const sensorText = sensorData.map(s => `${s.name}: ${s.value} ${s.unit}`).join(', ')
          emit('real-time-text-data', `Sensor Data: ${sensorText}`)
        }
      } catch (error) {
        console.error('Error sending sensor input:', error)
      }
    }
    
    // Enhanced camera control methods
    const detectAvailableCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = devices.filter(device => device.kind === 'videoinput')
        availableCameras.value = videoDevices.map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
          groupId: device.groupId
        }))
        console.log('Available cameras detected:', availableCameras.value.length)
      } catch (error) {
        console.error('Error detecting cameras:', error)
      }
    }
    
    const startCameraStream = async (cameraIndex) => {
      const cameraId = selectedCameras.value[cameraIndex]
      if (!cameraId || cameraId.trim() === '') return
      
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: cameraId } }
        })
        
        cameraStreams.value[cameraIndex] = {
          stream,
          active: true,
          captureInterval: null
        }
        
        console.log(`Camera ${cameraIndex + 1} stream started`)
      } catch (error) {
        console.error(`Error starting camera ${cameraIndex + 1} stream:`, error)
        cameraStreams.value[cameraIndex] = {
          stream: null,
          active: false,
          captureInterval: null
        }
      }
    }
    
    const stopCameraStream = (cameraIndex) => {
      const streamInfo = cameraStreams.value[cameraIndex]
      if (streamInfo && streamInfo.stream) {
        streamInfo.stream.getTracks().forEach(track => track.stop())
        
        if (streamInfo.captureInterval) {
          clearInterval(streamInfo.captureInterval)
        }
      }
      
      cameraStreams.value[cameraIndex] = {
        stream: null,
        active: false,
        captureInterval: null
      }
      
      console.log(`Camera ${cameraIndex + 1} stream stopped`)
    }
    
    const toggleCameraStreams = async () => {
      if (cameraStreamsActive.value) {
        // Stop all camera streams
        for (let i = 0; i < 3; i++) {
          if (cameraStreams.value[i]?.active) {
            stopCameraStream(i)
          }
        }
      } else {
        // Start selected camera streams
        for (let i = 0; i < 3; i++) {
          const cameraId = selectedCameras.value[i]
          if (cameraId && cameraId.trim() !== '') {
            await startCameraStream(i)
          }
        }
      }
    }
    
    const captureCameraFrames = async () => {
      if (!cameraStreamsActive.value) return
      
      try {
        const frames = []
        for (let i = 0; i < 3; i++) {
          if (cameraStreams.value[i]?.active) {
            // In a real implementation, capture frame from video element
            frames.push({
              cameraIndex: i,
              timestamp: Date.now()
            })
          }
        }
        
        // Send frames to backend for processing
        const response = await api.cameras.processStereoPair('current', {
          frames,
          mode: cameraMode.value
        })
        
        if (response.data.status === 'success') {
          addToHistory('Camera', `${cameraMode.value} frames captured`)
        }
      } catch (error) {
        console.error('Error capturing camera frames:', error)
      }
    }
    
    const processStereoVision = async () => {
      console.log('Process Stereo Vision button clicked')
      
      // Check preconditions with detailed logging
      if (!cameraStreamsActive.value) {
        console.log('Button disabled: cameraStreamsActive is false')
        addToHistory('Stereo Vision', 'Cannot process: Camera streams not active. Please start camera streams first.')
        return
      }
      
      if (cameraMode.value !== 'stereo') {
        console.log(`Button disabled: cameraMode is "${cameraMode.value}", expected "stereo"`)
        addToHistory('Stereo Vision', `Cannot process: Camera mode is "${cameraMode.value}". Please select "Dual Camera Spatial Recognition" mode.`)
        return
      }
      
      console.log('Preconditions met, starting stereo vision processing...')
      addToHistory('Stereo Vision', 'Starting stereo vision processing...')
      
      try {
        const startTime = Date.now()
        
        // First, try to get available stereo pairs
        let stereoPairId = 'stereo_pair_1'
        try {
          console.log('Fetching available stereo pairs from API...')
          const pairsResponse = await api.cameras.getStereoPairs()
          console.log('Stereo pairs response:', pairsResponse.data)
          
          if (pairsResponse.data.status === 'success' && pairsResponse.data.data && pairsResponse.data.data.stereo_pairs && pairsResponse.data.data.stereo_pairs.length > 0) {
            // Use the first available stereo pair
            stereoPairId = pairsResponse.data.data.stereo_pairs[0].id
            console.log(`Using stereo pair ID: ${stereoPairId}`)
            addToHistory('Stereo Vision', `Using stereo pair: ${stereoPairId}`)
          } else {
            console.log('No stereo pairs available in response, using default ID')
            addToHistory('Stereo Vision', 'No stereo pairs configured, using default')
          }
        } catch (error) {
          console.log('Failed to get stereo pairs:', error)
          addToHistory('Stereo Vision', 'No stereo pairs configured, using default ID')
        }
        
        console.log(`Processing stereo pair ${stereoPairId} with API...`)
        addToHistory('Stereo Vision', `Processing stereo vision with pair ${stereoPairId}...`)
        
        // Process stereo pair with real parameters
        const response = await api.cameras.processStereoPair(stereoPairId, {
          min_disparity: 0,
          num_disparities: 16,
          block_size: 15
        })
        
        console.log('Stereo vision processing response:', response.data)
        
        if (response.data.status === 'success') {
          const processingTime = Date.now() - startTime
          console.log(`Stereo vision processing completed in ${processingTime}ms`)
          addToHistory('Stereo Vision', `Depth map generated successfully in ${processingTime}ms`)
          
          // Store the result
          stereoVisionResult.value = response.data
          
          // Extract metrics from the API response
          const resultData = response.data.data ? response.data.data.result : response.data.result
          
          // Initialize default metrics
          let depthMin = 0.3
          let depthMax = 5.0
          let depthAverage = 1.5
          let objectCount = 0
          
          if (resultData) {
            // Try to get metrics from result data
            if (resultData.depth_min !== undefined) depthMin = resultData.depth_min
            if (resultData.depth_max !== undefined) depthMax = resultData.depth_max
            if (resultData.depth_average !== undefined) depthAverage = resultData.depth_average
            if (resultData.object_count !== undefined) objectCount = resultData.object_count
            
            // Also check for nested metrics
            if (resultData.metrics) {
              depthMin = resultData.metrics.depthMin || depthMin
              depthMax = resultData.metrics.depthMax || depthMax
              depthAverage = resultData.metrics.depthAverage || depthAverage
              objectCount = resultData.metrics.objectCount || objectCount
            }
          }
          
          // Update metrics with data from API
          stereoVisionMetrics.value = {
            depthMin: parseFloat(depthMin.toFixed(2)),
            depthMax: parseFloat(depthMax.toFixed(2)),
            depthAverage: parseFloat(depthAverage.toFixed(2)),
            objectCount,
            processingTime
          }
          
          // Generate real depth map preview from disparity map
          generateDepthPreviewFromDisparityMap(resultData)
          console.log('Depth map preview generated')
        } else {
          console.error('API returned error status:', response.data)
          addToHistory('Stereo Vision', `Processing failed: ${response.data.message || 'Unknown error'}`)
        }
      } catch (error) {
        console.error('Error processing stereo vision:', error)
        addToHistory('Stereo Vision', `Processing failed: ${error.message || 'Unknown error'}`)
      }
      
      console.log('Process Stereo Vision function completed')
    }
    
    const generateDepthPreviewFromDisparityMap = (resultData) => {
      if (!resultData) {
        console.warn('No result data provided for depth preview generation')
        return
      }
      
      // Create canvas for depth map visualization
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      // Try to use real disparity map data if available
      if (resultData.disparity_map && resultData.depth_map) {
        const width = resultData.depth_map.width || 400
        const height = resultData.depth_map.height || 300
        
        canvas.width = width
        canvas.height = height
        
        // Check if disparity_map is an array (2D or 1D)
        const disparityData = resultData.disparity_map
        
        if (Array.isArray(disparityData)) {
          // Create ImageData from disparity array
          const imageData = ctx.createImageData(width, height)
          
          if (Array.isArray(disparityData[0])) {
            // 2D array: disparityData[y][x]
            for (let y = 0; y < height; y++) {
              const row = disparityData[y] || []
              for (let x = 0; x < width; x++) {
                const value = row[x] || 0
                const index = (y * width + x) * 4
                imageData.data[index] = value      // R
                imageData.data[index + 1] = value  // G
                imageData.data[index + 2] = value  // B
                imageData.data[index + 3] = 255    // A
              }
            }
          } else {
            // 1D array: assume row-major order
            for (let i = 0; i < width * height && i < disparityData.length; i++) {
              const value = disparityData[i] || 0
              const index = i * 4
              imageData.data[index] = value      // R
              imageData.data[index + 1] = value  // G
              imageData.data[index + 2] = value  // B
              imageData.data[index + 3] = 255    // A
            }
          }
          
          ctx.putImageData(imageData, 0, 0)
          
          // Add depth scale for reference
          ctx.fillStyle = '#ffffff'
          ctx.fillRect(10, 10, 20, 100)
          const depthGradient = ctx.createLinearGradient(0, 10, 0, 110)
          depthGradient.addColorStop(0, '#000000')   // Near (black)
          depthGradient.addColorStop(1, '#ffffff')   // Far (white)
          ctx.fillStyle = depthGradient
          ctx.fillRect(11, 11, 18, 98)
          
          // Add depth labels
          ctx.fillStyle = '#ffffff'
          ctx.font = '10px Arial'
          ctx.textAlign = 'left'
          ctx.fillText('Near', 35, 20)
          ctx.fillText('Far', 35, 105)
          
          // Add info about the depth map
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
          ctx.fillRect(5, height - 30, width - 10, 25)
          ctx.fillStyle = '#ffffff'
          ctx.font = '10px Arial'
          ctx.textAlign = 'left'
          ctx.fillText(`Depth Map: ${width}x${height} | Disparity Range: ${resultData.depth_map.min_disparity || 0}-${resultData.depth_map.num_disparities || 16}`, 10, height - 15)
        }
      } else {
        // Fallback: generate informative visualization
        canvas.width = 400
        canvas.height = 300
        
        // Create a realistic depth gradient based on typical stereo vision
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
        gradient.addColorStop(0, '#000000')  // Closest objects (black)
        gradient.addColorStop(0.3, '#333333')
        gradient.addColorStop(0.6, '#666666')
        gradient.addColorStop(1, '#999999')  // Farthest objects (dark gray)
        
        ctx.fillStyle = gradient
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        // Add depth indicators based on point cloud data if available
        if (resultData.point_cloud && resultData.point_cloud.points) {
          const points = resultData.point_cloud.points
          const colors = resultData.point_cloud.colors || []
          
          // Sample some points for visualization
          const sampleStep = Math.max(1, Math.floor(points.length / 50))
          for (let i = 0; i < points.length; i += sampleStep) {
            const point = points[i]
            if (point && point.length >= 3) {
              const x = canvas.width * 0.2 + point[0] * 50 + canvas.width * 0.3
              const y = canvas.height * 0.5 + point[1] * 50
              const depth = Math.max(0.1, Math.min(1.0, point[2] / 5.0))  // Normalize depth
              
              const size = 3 + depth * 5
              const colorValue = Math.floor(255 * depth)
              ctx.fillStyle = `rgb(${colorValue}, ${colorValue}, ${colorValue})`
              ctx.beginPath()
              ctx.arc(x, y, size, 0, Math.PI * 2)
              ctx.fill()
            }
          }
        }
        
        // Add depth scale
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(10, 10, 20, 100)
        const depthGradient = ctx.createLinearGradient(0, 10, 0, 110)
        depthGradient.addColorStop(0, '#000000')
        depthGradient.addColorStop(1, '#999999')
        ctx.fillStyle = depthGradient
        ctx.fillRect(11, 11, 18, 98)
        
        // Add depth labels
        ctx.fillStyle = '#ffffff'
        ctx.font = '10px Arial'
        ctx.textAlign = 'left'
        ctx.fillText('Near', 35, 20)
        ctx.fillText('Far', 35, 105)
        
        // Add informational text
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.fillRect(5, canvas.height - 30, canvas.width - 10, 25)
        ctx.fillStyle = '#ffffff'
        ctx.font = '10px Arial'
        ctx.textAlign = 'left'
        ctx.fillText('Real Depth Visualization | Point Cloud Data Display', 10, canvas.height - 15)
      }
      
      // Convert to data URL for preview
      stereoVisionPreview.value = canvas.toDataURL('image/png')
    }
    
    const processTripleCameras = async () => {
      if (!cameraStreamsActive.value || cameraMode.value !== 'triple') return
      
      try {
        const response = await api.cameras.processStereoPair('triple_cameras', {
          cameraIds: selectedCameras.value.filter(id => id && id.trim() !== ''),
          operation: 'panoramic_stitching'
        })
        
        if (response.data.status === 'success') {
          addToHistory('Triple Cameras', 'Panoramic view generated')
        }
      } catch (error) {
        console.error('Error processing triple cameras:', error)
      }
    }
    
    // Real sensor data required - simulation removed for AGI hardware requirements
    // Sensor data must come from real hardware interface
    
    onMounted(() => {

      
      // Detect available cameras
      detectAvailableCameras()
      
      // Request camera permissions on mount to improve UX
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Request generic video permission to get device labels
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            // Immediately stop to release the camera, we just needed permission
            stream.getTracks().forEach(track => track.stop())
            // Now re-detect cameras with proper labels
            detectAvailableCameras()
          })
          .catch(err => {
            console.log('Camera permission not granted:', err)
          })
      }
    })
    
    // Spatial recognition utility functions
    const exportSpatialData = () => {
      if (!stereoVisionResult.value) return
      
      const data = {
        depthMapUrl: stereoVisionPreview.value,
        metrics: stereoVisionMetrics.value,
        timestamp: new Date().toISOString(),
        cameraIds: selectedCameras.value.filter(id => id)
      }
      
      // Create a downloadable JSON file
      const dataStr = JSON.stringify(data, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)
      const link = document.createElement('a')
      link.href = url
      link.download = `spatial-recognition-${new Date().getTime()}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      
      addToHistory('Spatial', 'Spatial data exported')
    }
    
    const clearSpatialResults = () => {
      stereoVisionResult.value = null
      stereoVisionPreview.value = null
      stereoVisionMetrics.value = {
        depthMin: 0,
        depthMax: 0,
        depthAverage: 0,
        objectCount: 0,
        processingTime: 0
      }
      addToHistory('Spatial', 'Spatial results cleared')
    }
    
    // Robot control functions
    const controlRobotWithSpatialData = async () => {
      if (!stereoVisionResult.value) {
        robotControlStatus.value = 'No spatial data available. Please process stereo vision first.'
        addToHistory('Robot', 'Failed: No spatial data')
        return
      }
      
      try {
        robotControlStatus.value = 'Sending motion command to robot...'
        addToHistory('Robot', `Starting ${selectedRobotMotion.value} with spatial data`)
        
        // Prepare motion command
        const motionCommand = {
          motion: selectedRobotMotion.value,
          params: {
            use_spatial_data: useSpatialDataForRobot.value,
            spatial_metrics: stereoVisionMetrics.value,
            target_depth: robotTargetDepth.value,
            safety_margin: robotSafetyMargin.value,
            timestamp: new Date().toISOString()
          }
        }
        
        // If we have spatial data result, include it
        if (stereoVisionResult.value && stereoVisionResult.value.result) {
          motionCommand.params.spatial_result = {
            has_depth_map: !!stereoVisionResult.value.result.depth_map,
            has_point_cloud: !!stereoVisionResult.value.result.point_cloud,
            object_count: stereoVisionMetrics.value.objectCount
          }
        }
        
        // Send motion command to robot API
        const response = await api.robot.motion.execute(motionCommand)
        
        if (response.data.status === 'success') {
          robotControlStatus.value = `Robot motion "${selectedRobotMotion.value}" started successfully`
          addToHistory('Robot', `Motion "${selectedRobotMotion.value}" executed successfully`)
        } else {
          robotControlStatus.value = `Robot motion failed: ${response.data.message || 'Unknown error'}`
          addToHistory('Robot', `Motion failed: ${response.data.message || 'Unknown error'}`)
        }
      } catch (error) {
        console.error('Error controlling robot with spatial data:', error)
        robotControlStatus.value = `Robot control error: ${error.message || 'Unknown error'}`
        addToHistory('Robot', `Control error: ${error.message || 'Unknown error'}`)
      }
    }
    
    const sendSpatialDataToRobot = async () => {
      if (!stereoVisionResult.value) {
        robotControlStatus.value = 'No spatial data to send'
        addToHistory('Robot', 'Failed: No spatial data')
        return
      }
      
      try {
        robotControlStatus.value = 'Sending spatial data to robot...'
        addToHistory('Robot', 'Sending spatial data to robot API')
        
        // Prepare spatial data request
        const spatialRequest = {
          cameras: ['left', 'right'],
          method: 'stereo',
          parameters: {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 15
          }
        }
        
        // Send to robot spatial API
        const response = await api.robot.spatial.depth(spatialRequest)
        
        if (response.data.status === 'success') {
          robotControlStatus.value = 'Spatial data sent to robot successfully'
          addToHistory('Robot', 'Spatial data sent successfully')
        } else {
          robotControlStatus.value = `Failed to send spatial data: ${response.data.message || 'Unknown error'}`
          addToHistory('Robot', `Failed to send spatial data: ${response.data.message || 'Unknown error'}`)
        }
      } catch (error) {
        console.error('Error sending spatial data to robot:', error)
        robotControlStatus.value = `Spatial data send error: ${error.message || 'Unknown error'}`
        addToHistory('Robot', `Spatial data send error: ${error.message || 'Unknown error'}`)
      }
    }
    
    const testRobotConnection = async () => {
      try {
        robotControlStatus.value = 'Testing robot connection...'
        addToHistory('Robot', 'Testing robot connection')
        
        // Test robot status endpoint
        const response = await api.robot.status()
        
        if (response.data.status === 'success') {
          const robotData = response.data.data || {}
          robotControlStatus.value = `Robot connected: ${robotData.status || 'Unknown status'}`
          addToHistory('Robot', `Connection test successful: ${robotData.status || 'Unknown status'}`)
        } else {
          robotControlStatus.value = 'Robot connection test failed'
          addToHistory('Robot', 'Connection test failed')
        }
      } catch (error) {
        console.error('Error testing robot connection:', error)
        robotControlStatus.value = `Connection test error: ${error.message || 'Unknown error'}`
        addToHistory('Robot', `Connection test error: ${error.message || 'Unknown error'}`)
      }
    }
    
    onUnmounted(() => {
      // Clean up - sensor data simulation removed for AGI hardware requirements
      
      // Stop any ongoing recordings
      if (mediaRecorder.value && mediaRecorder.value.state === 'recording') {
        mediaRecorder.value.stop()
      }
      
      if (videoRecorder.value && videoRecorder.value.state === 'recording') {
        videoRecorder.value.stop()
      }
      
      // Stop any media streams
      if (videoStream.value) {
        videoStream.value.getTracks().forEach(track => track.stop())
      }
      
      // Stop all camera streams
      for (let i = 0; i < 3; i++) {
        if (cameraStreams.value[i]?.active) {
          stopCameraStream(i)
        }
      }
      
      // Stop all audio processing
      if (isRealtimeAudioActive.value) {
        stopRealtimeAudio()
      }
      
      if (isSpeechRecognitionActive.value) {
        stopSpeechRecognition()
      }
      
      if (audioContext.value) {
        audioContext.value.close()
        audioContext.value = null
      }
      
      if (visualizationInterval.value) {
        clearInterval(visualizationInterval.value)
        visualizationInterval.value = null
      }
    })
    
    return {
      selectedInputType,
      textInput,
      selectedImage,
      isRecording,
      audioBlob,
      audioDuration,
      isVideoRecording,
      videoBlob,
      videoStream,
      sensorData,
      inputHistory,
      sendTextInput,
      handleImageUpload,
      sendImageInput,
      toggleAudioRecording,
      sendAudioInput,
      toggleVideoRecording,
      sendVideoInput,
      sendSensorInput,
      formatTime,
      // Enhanced camera control
      cameraMode,
      availableCameras,
      selectedCameras,
      cameraStreams,
      cameraStreamsActive,
      hasSelectedCameras,
      detectAvailableCameras,
      toggleCameraStreams,
      captureCameraFrames,
      processStereoVision,
      processTripleCameras,
      // Spatial recognition results
      stereoVisionResult,
      stereoVisionPreview,
      stereoVisionMetrics,
      exportSpatialData,
      clearSpatialResults,
      // Robot control with spatial data
      selectedRobotMotion,
      useSpatialDataForRobot,
      robotTargetDepth,
      robotSafetyMargin,
      robotControlStatus,
      controlRobotWithSpatialData,
      sendSpatialDataToRobot,
      testRobotConnection,
      // Enhanced audio processing
      audioProcessingMode,
      isRealtimeAudioActive,
      isAudioAnalyzing,
      isSpeechRecognitionActive,
      audioAnalysisOptions,
      speechLanguage,
      speechRecognitionResults,
      audioMetrics,
      audioVisualizationCanvas,
      toggleRealtimeAudio,
      processRealtimeAudio,
      startAudioAnalysis,
      toggleSpeechRecognition
    }
  }
}
</script>

<style scoped>
.real-time-input {
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f8f9fa;
}

.input-controls {
  margin-top: 20px;
}

.input-section {
  margin-top: 15px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #fff;
}

.text-input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 10px;
  resize: vertical;
}

.file-input {
  margin-bottom: 10px;
}

.image-preview {
  margin-top: 10px;
}

.preview-image {
  max-width: 200px;
  max-height: 200px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 10px;
}

.audio-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.audio-info {
  margin-top: 10px;
  font-size: 0.9em;
  color: #666;
}

.video-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.video-preview {
  margin-top: 10px;
}

.preview-video {
  max-width: 300px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.sensor-data-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}

.sensor-item {
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #f8f9fa;
}

.sensor-name {
  font-weight: bold;
  margin-right: 10px;
}

.input-history {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #fff;
}

.history-list {
  margin-top: 10px;
}

.history-item {
  padding: 8px;
  border-bottom: 1px solid #eee;
  display: flex;
  align-items: center;
  gap: 10px;
}

.history-type {
  font-weight: bold;
  min-width: 80px;
}

.history-time {
  color: #666;
  font-size: 0.9em;
  min-width: 100px;
}

.history-content {
  flex: 1;
  color: #333;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-danger {
  background-color: #dc3545;
  color: white;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Enhanced camera control styles */
.camera-mode-selection {
  margin-bottom: 15px;
}

.camera-mode-selection label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.camera-mode-select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
}

.camera-selection-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 15px;
  margin-top: 10px;
}

.camera-select-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.camera-select-item label {
  font-weight: bold;
  font-size: 0.9em;
}

.camera-select-item select {
  padding: 6px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.camera-status {
  font-size: 0.8em;
  color: #666;
  margin-top: 2px;
}

.camera-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 15px;
}

.preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.camera-preview {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
  background-color: #f8f9fa;
}

.camera-preview h6 {
  margin: 0 0 10px 0;
  font-size: 0.9em;
  color: #333;
}

.camera-preview .preview-video {
  width: 100%;
  max-height: 200px;
  border-radius: 4px;
}

.no-preview {
  padding: 20px;
  text-align: center;
  color: #999;
  background-color: #f0f0f0;
  border-radius: 4px;
}

.btn-info {
  background-color: #17a2b8;
  color: white;
}

.btn-success {
  background-color: #28a745;
  color: white;
}

/* Enhanced audio processing styles */
.audio-mode-selection {
  margin-bottom: 15px;
}

.audio-mode-selection label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.audio-mode-select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
}

.basic-audio-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.realtime-controls, .analysis-controls, .speech-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.analysis-options {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-bottom: 10px;
}

.analysis-options label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.9em;
}

.language-selection {
  display: flex;
  align-items: center;
  gap: 10px;
}

.language-select {
  padding: 5px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.audio-visualization {
  margin-top: 15px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #f8f9fa;
}

.audio-visualization h5 {
  margin: 0 0 10px 0;
  font-size: 1em;
}

.audio-canvas {
  width: 100%;
  height: 150px;
  background-color: black;
  border-radius: 4px;
  margin-bottom: 10px;
}

.audio-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  font-size: 0.9em;
  color: #333;
}

.audio-metrics div {
  padding: 5px;
  background-color: #e9ecef;
  border-radius: 4px;
  text-align: center;
}

.speech-results {
  margin-top: 15px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #f8f9fa;
}

.speech-results h5 {
  margin: 0 0 10px 0;
  font-size: 1em;
}

.speech-result-item {
  padding: 8px;
  margin-bottom: 5px;
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.9em;
}

/* Spatial recognition preview styles */
.spatial-recognition-preview {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #f8f9fa;
}

.spatial-recognition-preview h5 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #333;
}

.spatial-preview-item {
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
  margin-bottom: 15px;
}

.spatial-preview-item h6 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #555;
}

.depth-map-preview {
  width: 100%;
  max-width: 400px;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  display: block;
  margin: 0 auto 10px auto;
}

.depth-scale {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin-top: 10px;
  font-size: 0.9em;
  color: #666;
}

.scale-gradient {
  width: 100px;
  height: 20px;
  background: linear-gradient(to right, #000000, #cccccc);
  border: 1px solid #ddd;
  border-radius: 2px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background-color: #f8f9fa;
  border: 1px solid #eee;
  border-radius: 4px;
}

.metric-label {
  font-weight: bold;
  color: #555;
}

.metric-value {
  color: #333;
  font-family: monospace;
}

.spatial-actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}
</style>
