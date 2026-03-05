<template>
  <div class="home-view">
    <!-- User Guide Component -->

    
    <!-- Page Content Area -->
    
    <div class="input-area">
      <div class="conversation-header">
        <h2>Conversation</h2>
        <div class="main-model-status inline-status">
          <span class="model-name">Manager Model</span>
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ modelConnectionStatus === 'connected' ? 'Connected' : modelConnectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected' }}</span>
        </div>
        <button @click="clearAllMessages" class="clear-btn" :disabled="messages.length === 0">
          Clear All Messages
        </button>
      </div>
      <div class="chat-container">
        <div v-for="message in messages" :key="message.id" class="message-wrapper" :class="message.type">
          <div v-if="message.type === 'system'" class="system-message-indicator">
            <span class="system-icon">ℹ️</span>
            <span class="system-label">System Message</span>
          </div>
          <div v-if="message.type === 'loading'" class="loading-message-indicator">
            <span class="loading-spinner"></span>
            <span class="loading-text">AI is thinking...</span>
          </div>
          <div v-if="message.type === 'user' || message.type === 'bot'" class="message" :class="message.type">
            <div class="message-sender">
              <span v-if="message.type === 'user'" class="sender-avatar user-avatar">👤</span>
              <span v-else-if="message.type === 'bot'" class="sender-avatar bot-avatar">🤖</span>
              <span class="sender-name">{{ message.type === 'user' ? 'You' : 'AI Assistant' }}</span>
              <span class="message-timestamp">{{ formatTime(message.time) }}</span>
            </div>
            <div class="message-content">{{ message.content }}</div>
            <div v-if="message.confidence && message.type === 'bot'" class="message-confidence">
              <span class="confidence-label">Confidence</span>
              <div class="confidence-bar">
                <div class="confidence-fill" :style="{ width: (message.confidence * 100) + '%' }"></div>
              </div>
              <span class="confidence-value">{{ Math.round(message.confidence * 100) }}%</span>
            </div>
          </div>
          <div v-else class="message" :class="message.type">
            <div class="message-content">{{ message.content }}</div>
            <div class="message-time">{{ formatTime(message.time) }}</div>
          </div>
        </div>
      </div>

      <div class="input-controls">
        <div class="input-with-feedback">
          <input type="text"
                 v-model="inputText"
                 @keyup.enter="sendMessage"
                 placeholder="Type your message..."
                 :disabled="isSendingMessage">
          <span v-if="isSendingMessage" class="sending-indicator">
            <span class="sending-dot"></span>
            <span class="sending-dot"></span>
            <span class="sending-dot"></span>
          </span>
        </div>
        <button @click="sendMessage" :disabled="isSendingMessage || !inputText.trim()" class="send-btn">
          <span v-if="isSendingMessage">Sending...</span>
          <span v-else>Send</span>
        </button>
        <div class="input-options">
          <button @click="startVoiceRecognition" :disabled="isSendingMessage" class="input-option-btn voice-btn" :class="{ 'voice-active': isVoiceInputActive }">
            <span class="btn-icon">
              <span v-if="isVoiceInputActive" class="recording-indicator">
                <span class="recording-dot"></span>
              </span>
              <span v-else>🎤</span>
            </span>
            {{ isVoiceInputActive ? 'Recording...' : 'Voice Input' }}
          </button>
          <button @click="selectImage" :disabled="isSendingMessage" class="input-option-btn">
            <span class="btn-icon">🖼️</span>
            Image
          </button>
          <input type="file" ref="imageInput" style="display: none"
                 accept="image/*" @change="handleImageUpload">
          <button @click="selectVideo" :disabled="isSendingMessage" class="input-option-btn">
            <span class="btn-icon">🎥</span>
            Video
          </button>
          <input type="file" ref="videoInput" style="display: none"
                 accept="video/*" @change="handleVideoUpload">
          <button @click="toggleRealTimeInput" :disabled="isSendingMessage" class="input-option-btn">
            <span class="btn-icon">⚡</span>
            {{ showRealTimeInput ? 'Hide real-time' : 'Real-time input' }}
          </button>
        </div>
      </div>
      
      <!-- Video Dialog Interface -->
      <div class="video-dialog-section">
        <div class="video-dialog-header">
          <h3>🎥 Video Dialogue</h3>
          <div class="video-dialog-status">
            <span class="status-indicator" :class="videoDialogStatus"></span>
            <span class="status-text">{{ getVideoDialogStatusText() }}</span>
          </div>
        </div>
        
        <div class="video-dialog-content">
          <div class="video-preview-container">
            <div class="video-preview">
              <video ref="videoDialogPreview" autoplay playsinline></video>
              <canvas ref="videoOverlayCanvas" class="video-overlay-canvas"></canvas>
              <div v-if="!isVideoDialogActive" class="video-placeholder">
                <span class="placeholder-icon">📹</span>
                <span class="placeholder-text">Click "Start Video Dialogue" to enable camera</span>
              </div>
            </div>
            <div class="video-controls">
              <button @click="toggleVideoDialog" class="video-dialog-btn" :class="{ 'active': isVideoDialogActive }">
                {{ isVideoDialogActive ? 'Stop Video Dialogue' : 'Start Video Dialogue' }}
              </button>
              <select v-model="selectedCamera" v-if="availableCameras.length > 0" class="camera-select">
                <option v-for="camera in availableCameras" :key="camera.deviceId" :value="camera.deviceId">
                  {{ camera.label || `Camera ${camera.deviceId.slice(0, 8)}` }}
                </option>
              </select>
            </div>
          </div>
          
          <div class="ai-response-container">
            <div class="ai-response-header">
              <h4>🤖 AI Response</h4>
              <button @click="clearVideoResponses" class="clear-btn" :disabled="videoDialogResponses.length === 0">
                Clear
              </button>
            </div>
            <div class="ai-response-messages">
              <div v-if="videoDialogResponses.length === 0" class="empty-responses">
                No AI response yet. AI will analyze video content and respond after starting video dialogue.
              </div>
              <div v-else class="response-list">
                <div v-for="(response, index) in videoDialogResponses" :key="index" class="response-item">
                  <div class="response-content">{{ response }}</div>
                  <div class="response-time">{{ formatResponseTime(index) }}</div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Object Detection Results -->
          <div class="object-detection-results" v-if="objectDetectionResults.length > 0">
            <div class="object-detection-header">
              <h5>🎯 Detected Objects</h5>
              <span class="object-detection-count">{{ objectDetectionResults.length }} objects</span>
            </div>
            <div class="object-list">
              <div v-for="(obj, index) in objectDetectionResults" :key="index" class="object-item">
                <span class="object-label">{{ obj.label }}</span>
                <span class="object-confidence">{{ (obj.confidence * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="video-dialog-info">
          <div class="info-item">
            <span class="info-label">Camera Status:</span>
            <span class="info-value">{{ isCameraAvailable ? 'Available' : 'Unavailable' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">AI Connection:</span>
            <span class="info-value">{{ isWebSocketConnected ? 'Connected' : 'Disconnected' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Frame Count:</span>
            <span class="info-value">{{ frameCount }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Response Count:</span>
            <span class="info-value">{{ videoDialogResponses.length }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Average Response Time:</span>
            <span class="info-value">{{ averageResponseTime > 0 ? averageResponseTime.toFixed(1) + 'ms' : 'N/A' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">Frame Rate:</span>
            <span class="info-value">{{ Math.min(4, 1000 / 250) }} FPS</span>
          </div>
        </div>
      </div>
      
      <!-- Real-time Input Interface -->
      <div v-if="showRealTimeInput" class="real-time-section">
        <RealTimeInput 
          @real-time-audio-data="handleRealTimeAudioData"
          @real-time-video-data="handleRealTimeVideoData"
          @real-time-text-data="handleRealTimeTextData"
          @real-time-file-data="handleRealTimeFileData"
        />
      </div>
    </div>


  </div>
</template>

<script>
import RealTimeInput from '@/components/RealTimeInput.vue';
import UserGuide from '@/components/UserGuide.vue';
import api from '@/utils/api';
import { 
  handleEnhancedError,
  handleApiError,
  handleNetworkError,
  handleConfigError,
  handleValidationError,
  logInfo,
  logWarning,
  logSuccess 
} from '@/utils/enhancedErrorHandler';

export default {
  name: 'HomeView',
  components: {
    RealTimeInput,
    UserGuide
  },
  data() {
    return {
      isMounted: false,
      inputText: '',
      messages: [],
      isVoiceInputActive: false,
      isSendingMessage: false,
      showRealTimeInput: false,
      showUserGuide: false,
      backendConnected: false,
      backendStatus: 'disconnected',
      recognition: null,
      videoInput: null,
      // Audio recording for real voice recognition
      mediaStream: null,
      mediaRecorder: null,
      audioChunks: [],
      isRecording: false,
      audioBlob: null,
      modelPerformanceData: [],
      models: [],
      modelConnectionStatus: 'unknown',
      managementModel: {
        name: 'Management Model',
        status: 'inactive',
        lastActive: null
      },
      connectedText: '',
      activeModels: 0,
      // Enhanced multi-camera support data for stereo vision
      cameras: [],
      // External devices with enhanced communication protocols
      externalDevices: [],
      // Real sensor data storage - initialized as empty object
      sensorData: {},
      // Device control WebSocket connection with enhanced protocols
      deviceControlWebSocket: null,
      deviceControlConnected: false,
      deviceControlReconnectInterval: null,
      deviceControlPingInterval: null,
      webSocketErrorNotified: false,
      // Stereo vision calibration data
      stereoCalibrationData: {
        baseline: null,
        focalLength: null,
        principalPoint: {x: null, y: null},
        rotationMatrix: [],
        translationVector: []
      },
      // Serial Communication Data
      availableSerialPorts: [],
      selectedSerialPort: '',
      serialBaudRate: '9600',
      serialConnected: false,
      serialSendData: '',
      serialReceivedData: '',
      sendAsHex: false,
      appendCR: false,
      appendLF: false,
      autoScroll: true,
      serialListenerInterval: null,
      // Active stereo vision pairs
      stereoVisionPairs: [],
      // Device communication protocols
      communicationProtocols: ['WebSocket', 'MQTT', 'HTTP', 'Serial', 'Bluetooth', 'Custom'],
      // Connection status for each device
      deviceConnectionStatus: {},
      // Video Dialog Data
      isVideoDialogActive: false,
      videoDialogStatus: 'inactive', // 'inactive', 'connecting', 'active', 'error'
      availableCameras: [],
      selectedCamera: '',
      isCameraAvailable: false,
      videoDialogResponses: [],
      isWebSocketConnected: false,
      videoStream: null,
      videoWebSocket: null,
      videoCaptureInterval: null,
      isFrameProcessing: false,
      lastFrameLog: 0,
      heartbeatInterval: null,
      frameCount: 0,
      videoCanvas: null,
      videoContext: null,
      compressionCanvas: null,
      compressionContext: null,
      frameTimestamps: new Map(),
      responseTimes: [],
      averageResponseTime: 0,
      // Object detection data
      objectDetectionResults: [],
      videoOverlayCanvas: null,
      videoOverlayContext: null
    };
  },
  mounted() {
    this.isMounted = true;
    // Load history messages
    this.loadHistoryMessages();
    // Connect to backend service first
    this.connectToBackend();
    // Initialize system after attempting connection
    this.initializeSystem();
    // Initialize speech recognition
    this.initSpeechRecognition();
    
    // Listen to global voice input events from App.vue
    window.addEventListener('voice-input', this.handleVoiceInputEvent);
    
    // Listen to real-time input events from RealTimeInput component
    this.setupRealTimeInputListeners();
    
    // Initialize camera list for video dialog
    this.listCameras();
    
    // Initialize device control WebSocket connection
    this.initializeDeviceControl();
    
    // Load real device data from backend
    this.loadDeviceData();
    
    // Initialize serial communication
    this.refreshSerialPorts();
  },
  beforeUnmount() {
    this.isMounted = false;
    // Remove event listeners when component is unmounted
    window.removeEventListener('voice-input', this.handleVoiceInputEvent);
    
    // Clean up device control WebSocket connection
    this.disconnectDeviceControlWebSocket();
    
    // Clean up serial communication
    this.disconnectSerialPort();
    
    // Clean up camera WebSocket connections
    this.cleanupCameraWebSockets();
    
    // Clean up voice recognition
    this.stopVoiceRecognition();
    
    // Clean up video dialog resources
    if (this.isVideoDialogActive) {
      this.stopVideoDialog();
    }
  },
  computed: {
      // Calculate active model count
      activeModelsCount() {
        const count = this.modelPerformanceData.filter(model => model.status === 'active').length;
        // Synchronously update activeModels property
        this.activeModels = count;
        return count;
      },
      // Management model status text
      managementModelStatusText() {
        return this.managementModel.status === 'active' ? 
          'Connected' : 
          'Disconnected';
      },
      // Overall model connection status
      overallModelConnectionStatus() {
        if (this.backendConnected && this.managementModel.status === 'active') {
          return 'Connected';
        } else if (this.backendStatus === 'connecting') {
          return 'Connecting...';
        } else {
          return 'Disconnected';
        }
      },
      // Video dialog WebSocket URL
      videoDialogWebSocketUrl() {
        const baseUrl = this.getWebSocketBaseUrl();
        return `${baseUrl}/ws/video-stream`;
      }
    },
    methods: {
      // Format time for display
      formatTime(timeString) {
        if (!timeString) return '';
        try {
          const time = new Date(timeString);
          if (isNaN(time.getTime())) {
            // If not a valid date string, try to parse from locale time
            return timeString;
          }
          return time.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false
          });
        } catch (error) {
          return timeString;
        }
      },
      
      // Initialize device control system
      initializeDeviceControl() {
        // Start real WebSocket connection
        this.connectDeviceControlWebSocket();
      },
      
      // Generic async operation function for HomeView
      async performAsyncOperation(options) {
        const {
          apiCall,
          loadingProperty,
          errorProperty,
          successMessage,
          errorMessage,
          errorContext,
          showSuccess = true,
          showError = true,
          onBeforeStart,
          onSuccess,
          onError,
          onFinally
        } = options

        if (loadingProperty) this[loadingProperty] = true
        if (errorProperty) this[errorProperty] = null

        // Execute before start callback if provided
        if (onBeforeStart && typeof onBeforeStart === 'function') {
          await onBeforeStart.call(this)
        }

        try {
          const response = await apiCall()
          
          if (onSuccess && typeof onSuccess === 'function') {
            await onSuccess.call(this, response.data)
          }
          
          return response.data
        } catch (error) {
          if (import.meta.env.DEV) {
            console.error(errorContext || 'Async operation error:', error)
          }
          
          if (errorProperty) this[errorProperty] = error
          
          if (onError && typeof onError === 'function') {
            await onError.call(this, error)
          }
          
          throw error
        } finally {
          if (loadingProperty) this[loadingProperty] = false
          if (onFinally && typeof onFinally === 'function') {
            await onFinally.call(this)
          }
        }
      },
      
      // Add system message
      addSystemMessage(content) {
        // Generate more reliable unique ID: timestamp + high-performance timer + message count
        const uniqueId = `msg_${Date.now()}_${performance?.now?.()?.toFixed(3) || '0'}_${this.messages.length}`
        const systemMessage = {
          id: uniqueId,
          type: 'system',
          content: content,
          time: new Date().toLocaleTimeString()
        };
        this.messages.push(systemMessage);
        this.saveMessages();
      },
      
      // Connect to backend service
      async connectToBackend() {
        try {
          await this.performAsyncOperation({
            apiCall: () => api.health.get(),
            loadingProperty: null, // No specific loading property
            errorProperty: null, // Error handled via errorHandler
            errorContext: 'Connect to Backend',
            onBeforeStart: () => {
              logInfo('Connecting to backend service...');
              this.backendStatus = 'connecting';
            },
            onSuccess: (data) => {
              if (data.status === 'healthy') {
                this.backendConnected = true;
                this.backendStatus = 'connected';
                logInfo('Successfully connected to backend service');
                
                // Update management model status
                this.managementModel.status = 'active';
                this.managementModel.lastActive = new Date().toISOString();
                this.modelConnectionStatus = 'connected';
              } else {
                throw new Error('Backend health check failed');
              }
            },
            onError: (error) => {
              handleNetworkError(error, 'backend service');
              this.backendConnected = false;
              this.backendStatus = 'disconnected';
              this.managementModel.status = 'inactive';
              this.modelConnectionStatus = 'disconnected';
            }
          });
        } catch (error) {
          // Error is already handled in onError callback
          if (import.meta.env.DEV) {
            console.log('Connect to backend operation completed with error:', error);
          }
        }
      },
      
      // Save messages to local storage
      saveMessages() {
        try {
          localStorage.setItem('chat_messages', JSON.stringify(this.messages));
        } catch (error) {
          handleEnhancedError(error, 'Saving chat message to local storage');
        }
      },

      // Load history messages from local storage
      loadHistoryMessages() {
        try {
          const savedMessages = localStorage.getItem('chat_messages');
          if (savedMessages) {
            this.messages = JSON.parse(savedMessages);
          }
        } catch (error) {
          handleEnhancedError(error, 'Loading history messages from local storage');
          this.messages = [];
        }
      },

        // Initialize system
        initializeSystem() {
          logInfo('Self Soul System initializing...');
          // Show welcome message
          this.addSystemMessage('Welcome to the Self Soul System!');
        
        // Always try to connect to real backend, never use mock data automatically
        this.connectToBackend();
      },
    
    // Get session ID for tracking conversation context
    getSessionId() {
      let sessionId = localStorage.getItem('agi_session_id');
      if (!sessionId) {
        // Use standard UUID generation method
        if (typeof crypto !== 'undefined' && crypto.randomUUID) {
          sessionId = crypto.randomUUID();
        } else {
          // Fallback solution: timestamp + high-performance timer + random number
          const timestamp = Date.now();
          const perf = performance?.now?.() || Math.random() * 10000;
          const random = Math.random().toString(36).substr(2, 9);
          sessionId = `session_${timestamp}_${perf.toFixed(3)}_${random}`;
        }
        localStorage.setItem('agi_session_id', sessionId);
      }
      return sessionId;
    },
    
    // Process image input
    async processImageInput(imageData) {
      try {
        const response = await api.post('/api/process/image', {
          image: imageData,
          language: 'en-US',
          session_id: this.getSessionId()
        }, {
          timeout: 60000
        });
        
        if (response.data.status === 'success') {
          return response.data.data;
        } else {
          throw new Error(response.data.detail || 'Failed to process image');
        }
      } catch (error) {
        handleApiError(error, 'Processing image');
        throw error; // Propagate error instead of returning placeholder message
      }
    },
    
    // Process video input
    async processVideoInput(videoData) {
      try {
        const response = await api.post('/api/process/video', {
          video: videoData,
          language: 'en-US',
          session_id: this.getSessionId()
        }, {
          timeout: 120000
        });
        
        if (response.data.status === 'success') {
          return response.data.data;
        } else {
          throw new Error(response.data.detail || 'Failed to process video');
        }
      } catch (error) {
        handleApiError(error, 'Processing video');
        throw error; // Propagate error instead of returning placeholder message
      }
    },
    
    // Send message to management model
    async sendMessage() {
      const messageText = this.inputText.trim();
      if (!messageText) return;
      
      // Add user message
      const userMessage = {
        id: Date.now(),
        type: 'user',
        content: messageText,
        time: new Date().toLocaleTimeString()
      };
      this.messages.push(userMessage);
      this.inputText = '';
      this.saveMessages();
      
      // Add loading status message
      const loadingMessageId = Date.now() + 0.5;
      const loadingMessage = {
        id: loadingMessageId,
        type: 'loading',
        content: 'AI is thinking...',
        time: new Date().toLocaleTimeString()
      };
      this.messages.push(loadingMessage);
      this.saveMessages();
      
      try {
        // Send message to management model API
        // Prepare conversation history in proper format
        const conversationHistory = this.messages
          .filter(msg => msg.type === 'user' || msg.type === 'bot')
          .map(msg => ({
            role: msg.type === 'user' ? 'user' : 'assistant',
            content: msg.content
          }));
        
        const response = await api.post('/api/models/8001/chat', {
          message: messageText,
          session_id: this.getSessionId(),
          timestamp: new Date().toISOString(),
          conversation_history: conversationHistory,
          lang: 'en',
          user_id: 'default_user',
          model_id: 'manager'
        });
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        // Add model response with additional information from enhanced API
        // Validate API response has required fields
        if (!response.data.data?.response && !response.data.response) {
          throw new Error('API response missing required content field');
        }
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: response.data.data?.response || response.data.response,
          time: new Date().toLocaleTimeString(),
          modelId: response.data.data?.model_id || response.data.model_id || '8001',
          modelName: 'Management Model',
          confidence: response.data.data?.confidence || response.data.confidence || 0.97,
          processingTime: response.data.data?.processing_time || 0
        };
        
        this.messages.push(botMessage);
        this.saveMessages();
        
        // Update model connection status
        this.modelConnectionStatus = 'connected';
        this.managementModel.status = 'active';
        
      } catch (error) {
        console.error('Error sending message to management model:', error);
        handleApiError(error, 'Sending message to management model');
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        // Add error message
        const errorMessage = {
          id: Date.now() + 1,
          type: 'system',
          content: `Failed to connect to management model: ${error.message || 'Connection error'}`,
          time: new Date().toLocaleTimeString()
        };
        
        this.messages.push(errorMessage);
        this.saveMessages();
        
        // Update model connection status
        this.modelConnectionStatus = 'disconnected';
        this.managementModel.status = 'inactive';
      }
    },
    
    // Clear all messages
    clearAllMessages() {
      if (confirm('Are you sure you want to clear all messages?')) {
        this.messages = [];
        this.saveMessages();
      }
    },
    
    // Process audio input
    async processAudioInput(audioData) {
      try {
        const response = await api.post('/api/process/audio', {
          audio: audioData,
          language: 'en-US',
          session_id: this.getSessionId()
        }, {
          timeout: 45000
        });
        
        if (response.data.status === 'success') {
          return response.data.data;
        } else {
          throw new Error(response.data.detail || 'Failed to process audio');
        }
      } catch (error) {
        handleApiError(error, 'Processing audio');
        throw error; // Propagate error instead of returning placeholder message
      }
    },
    
    // Initialize voice recognition (disabled, using global voice recognition)
    initSpeechRecognition() {
      try {
        // Check if browser supports speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
          this.recognition = new SpeechRecognition();
        }
      } catch (error) {
        handleConfigError(error, 'Speech recognition initialization');
      }
    },
    
    // Handle global voice input events from App.vue
    handleVoiceInputEvent(event) {
      const text = event.detail;
      if (text && text.trim()) {
        this.inputText = text;
        this.addSystemMessage(`Voice input received: ${text}`);
        // Auto send message with optimized delay for better UX
        setTimeout(() => {
          if (this.isMounted) {
            this.sendMessage();
          }
        }, 300);
      }
    },

    // Handle real-time audio data
    async handleRealTimeAudioData(audioData) {
      try {
        logInfo('Received real-time audio data');
        this.addSystemMessage('Real-time audio data received');
        
        // Add loading status message
        const loadingMessageId = Date.now() + 0.5;
        const loadingMessage = {
          id: loadingMessageId,
          type: 'loading',
          content: 'Processing audio...',
          time: new Date().toLocaleTimeString()
        };
        this.messages.push(loadingMessage);
        this.saveMessages();
        
        // Process audio input
        const responseText = await this.processUserInput(audioData, 'audio');
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: responseText,
          time: new Date().toLocaleTimeString()
        };
        
        this.messages.push(botMessage);
        this.saveMessages();
      } catch (error) {
        handleApiError(error, 'Process real-time audio data');
        this.addSystemMessage('Failed to process audio');
      }
    },

    // Get model display name
    getModelDisplayName(modelId) {
      const modelNames = {
        '8001': 'Manager Model',
        '8002': 'Language Model',
        '8003': 'Knowledge Model',
        '8004': 'Vision Model',
        '8005': 'Audio Model',
        '8006': 'Autonomous Model',
        '8007': 'Programming Model',
        '8008': 'Planning Model',
        '8009': 'Emotion Model',
        '8010': 'Spatial Model',
        '8011': 'Computer Vision Model',
        '8012': 'Sensor Model',
        '8013': 'Motion Model',
        '8014': 'Prediction Model',
        '8015': 'Advanced Reasoning Model',
        '8016': 'Data Fusion Model',
        '8017': 'Creative Problem Solving Model',
        '8018': 'Meta Cognition Model',
        '8019': 'Value Alignment Model',
        '8020': 'Image Vision Model',
        '8021': 'Video Vision Model',
        '8022': 'Finance Model',
        '8023': 'Medical Model',
        '8024': 'Collaboration Model',
        '8025': 'Optimization Model',
        '8026': 'Computer Model',
        '8027': 'Mathematics Model'
      };
      return modelNames[modelId] || `Model ${modelId}`;
    },

    // Handle real-time video data
    async handleRealTimeVideoData(videoData) {
      try {
        logInfo('Received real-time video data');
        this.addSystemMessage('Real-time video data received');
        
        // Add loading status message
        const loadingMessageId = Date.now() + 0.5;
        const loadingMessage = {
          id: loadingMessageId,
          type: 'loading',
          content: 'Processing video...',
          time: new Date().toLocaleTimeString()
        };
        this.messages.push(loadingMessage);
        this.saveMessages();
        
        // Process video input
        const responseText = await this.processUserInput(videoData, 'video');
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: responseText,
          time: new Date().toLocaleTimeString()
        };
        
        this.messages.push(botMessage);
        this.saveMessages();
      } catch (error) {
        handleApiError(error, 'Process real-time video data');
        this.addSystemMessage('Failed to process video');
      }
    },

    // Handle real-time text data
    async handleRealTimeTextData(textData) {
      try {
        logInfo('Received real-time text data');
        this.addSystemMessage(`Real-time text received: ${textData}`);
        
        // Directly send message using text data
        this.inputText = textData;
        await this.sendMessage();
      } catch (error) {
        handleApiError(error, 'Process real-time text data');
        this.addSystemMessage('Failed to process text');
      }
    },

    // Handle real-time file data
    async handleRealTimeFileData(fileData) {
      try {
        logInfo('Received real-time file data');
        this.addSystemMessage(`Real-time file received: ${fileData.name || 'File'}`);
        
        // Process according to file type
        if (fileData.type.includes('image')) {
          await this.handleImageUpload({ target: { files: [fileData] } });
        } else if (fileData.type.includes('video')) {
          await this.handleVideoUpload({ target: { files: [fileData] } });
        } else if (fileData.type.includes('audio')) {
          await this.handleRealTimeAudioData(fileData);
        } else {
          // Other file types
          const fileMessage = {
            id: Date.now(),
            type: 'user',
            content: `[File: ${fileData.name}]`,
            time: new Date().toLocaleTimeString()
          };
          
          const botMessage = {
            id: Date.now() + 1,
            type: 'bot',
            content: 'File received. Processing...',
            time: new Date().toLocaleTimeString()
          };
          
          this.messages.push(fileMessage, botMessage);
          this.saveMessages();
        }
      } catch (error) {
        handleApiError(error, 'Process real-time file data');
        this.addSystemMessage('Failed to process file');
      }
    },

    // Device control WebSocket management
    connectDeviceControlWebSocket() {
      // Clear any existing intervals
      this.disconnectDeviceControlWebSocket();
      
      try {
        // Create real WebSocket connection - device control runs on the main server
        const baseUrl = this.getWebSocketBaseUrl();
        const wsUrl = `${baseUrl}/ws/device-control`;
        this.deviceControlWebSocket = new WebSocket(wsUrl);
        
        this.deviceControlWebSocket.onopen = () => {
          if (import.meta.env.DEV) {
            console.log('Device control WebSocket connection established');
          }
          this.deviceControlConnected = true;
          this.addSystemMessage('Device control system: Connected');
          
          // Start ping to keep connection alive
          this.startDeviceControlPing();
          
          // Request initial device status
          this.requestDeviceStatus();
          
          // Cancel any pending reconnection attempts
          if (this.deviceControlReconnectInterval) {
            clearInterval(this.deviceControlReconnectInterval);
            this.deviceControlReconnectInterval = null;
          }
        };
        
        this.deviceControlWebSocket.onmessage = this.handleDeviceControlMessage.bind(this);
        
        this.deviceControlWebSocket.onclose = () => {
          if (import.meta.env.DEV) {
            console.log('Device control WebSocket connection closed');
          }
          this.deviceControlConnected = false;
          this.deviceControlWebSocket = null;
          this.addSystemMessage('Device control system: Disconnected');
          
          // Stop ping interval
          this.stopDeviceControlPing();
          
          // Set up reconnection
          this.setupReconnection();
        };
        
        this.deviceControlWebSocket.onerror = (error) => {
          console.error('Device control WebSocket error:', error);
          this.addSystemMessage?.('Device control system: Connection error');
        };
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        this.addSystemMessage?.('Device control system: Failed to connect');
        this.setupReconnection();
      }
    },
    
    // Request initial device status from server
    requestDeviceStatus() {
      if (this.deviceControlWebSocket && this.deviceControlWebSocket.readyState === WebSocket.OPEN) {
        this.deviceControlWebSocket.send(JSON.stringify({ type: 'request_status' }));
      }
    },

    disconnectDeviceControlWebSocket() {
      // Stop ping interval
      this.stopDeviceControlPing();

      // Clear reconnection interval
      if (this.deviceControlReconnectInterval) {
        clearInterval(this.deviceControlReconnectInterval);
        this.deviceControlReconnectInterval = null;
      }

      // Clear device update interval

      // Close WebSocket connection if it exists
      if (this.deviceControlWebSocket) {
        this.deviceControlWebSocket.close();
        this.deviceControlWebSocket = null;
      }

      this.deviceControlConnected = false;
    },

    setupReconnection() {
      // Try to reconnect every 5 seconds if not already trying
      if (!this.deviceControlReconnectInterval) {
        this.deviceControlReconnectInterval = setInterval(() => {
          if (import.meta.env.DEV) {
            console.log('Attempting to reconnect to device control WebSocket...');
          }
          this.connectDeviceControlWebSocket();
        }, 5000);
      }
    },

    startDeviceControlPing() {
      // Send ping every 30 seconds to keep connection alive
      this.deviceControlPingInterval = setInterval(() => {
        if (this.deviceControlWebSocket && this.deviceControlWebSocket.readyState === WebSocket.OPEN) {
          this.deviceControlWebSocket.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    },

    stopDeviceControlPing() {
      if (this.deviceControlPingInterval) {
        clearInterval(this.deviceControlPingInterval);
        this.deviceControlPingInterval = null;
      }
    },

    handleDeviceControlMessage(event) {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'initial_status') {
          // Update device states from initial status
          this.updateDeviceStates(message.devices);
        } else if (message.type === 'command_response') {
          // Handle command response
          this.handleCommandResponse(message);
        } else if (message.type === 'sensor_update') {
          // Update sensor data
          this.updateSensorDataFromServer(message.device_id, message.data);
        } else if (message.type === 'pong') {
          // Just ignore pong responses
        }
      } catch (error) {
        console.error('Failed to parse device control message:', error);
      }
    },

    updateDeviceStates(devicesData) {
      // Update cameras
      for (const deviceId in devicesData) {
        if (devicesData[deviceId].type === 'camera') {
          const localCamera = this.cameras.find(cam => cam.id === deviceId);
          if (localCamera) {
            localCamera.status = devicesData[deviceId].status;
            localCamera.active = devicesData[deviceId].active;
            localCamera.stream = devicesData[deviceId].stream_id;
          }
        } else {
          // Update other devices
          const localDevice = this.externalDevices.find(dev => dev.id === deviceId);
          if (localDevice) {
            localDevice.status = devicesData[deviceId].status;
            localDevice.connected = devicesData[deviceId].connected;
          }
        }
      }
    },

    handleCommandResponse(response) {
      if (response.success) {
        if (response.message) {
          this.addSystemMessage(response.message);
        }
        // Update device state if status is included in response
        if (response.status) {
          if (response.status.type === 'camera') {
            const localCamera = this.cameras.find(cam => cam.id === response.device_id);
            if (localCamera) {
              Object.assign(localCamera, response.status);
            }
          } else {
            const localDevice = this.externalDevices.find(dev => dev.id === response.device_id);
            if (localDevice) {
              Object.assign(localDevice, response.status);
            }
          }
        }
      } else {
        this.addSystemMessage(`Error: ${response.error || 'Command failed'}`);
      }
    },



    sendDeviceControlCommand(deviceId, action, params = {}) {
      if (this.deviceControlWebSocket && this.deviceControlWebSocket.readyState === WebSocket.OPEN) {
        const command = {
          type: 'device_command',
          device_id: deviceId,
          action: action,
          params: params
        };
        this.deviceControlWebSocket.send(JSON.stringify(command));
        return true;
      } else {
        console.warn('Device control WebSocket not connected');
        this.addSystemMessage('Device control system: Not connected, command not sent');
        return false;
      }
    },

    // Enhanced multi-camera control methods with stereo vision support
    toggleCamera(cameraId) {
      // For stereo cameras, toggle both cameras in the pair
      const camera = this.cameras.find(cam => cam.id === cameraId);
      if (camera && camera.isStereo && camera.stereoPairId) {
        const stereoPair = this.stereoVisionPairs.find(pair => pair.id === camera.stereoPairId);
        if (stereoPair) {
          // Toggle both cameras in the pair simultaneously
          this.sendDeviceControlCommand(stereoPair.leftCameraId, 'toggle');
          this.sendDeviceControlCommand(stereoPair.rightCameraId, 'toggle');
          
          // Update stereo pair status
          stereoPair.isActive = !stereoPair.isActive;
          this.addSystemMessage(`Toggled stereo pair: ${stereoPair.name}`);
          return;
        }
      }
      
      // For non-stereo cameras, toggle individually
      this.sendDeviceControlCommand(cameraId, 'toggle');
    },

    // Enhanced camera configuration with more parameters
    configureCamera(cameraId) {
      const camera = this.cameras.find(cam => cam.id === cameraId);
      if (camera) {
        // For stereo cameras, offer stereo-specific configuration
        if (camera.isStereo) {
          const configOptions = `1. Resolution: ${camera.resolution}\n` +
                              `2. FPS: ${camera.fps}\n` +
                              `3. Exposure: ${camera.exposure}\n` +
                              `4. Stereo Calibration`;
          
          const choice = prompt(`Configure ${camera.name}:\n${configOptions}\nEnter option number (1-4):`, '1');
          
          switch(choice) {
            case '1':
              const resolution = prompt(`Enter resolution (e.g., 1920x1080):`, camera.resolution);
              if (resolution) {
                this.sendDeviceControlCommand(cameraId, 'configure', { resolution });
              }
              break;
            case '2':
              const fps = prompt(`Enter FPS (frames per second):`, camera.fps);
              if (fps && !isNaN(fps)) {
                this.sendDeviceControlCommand(cameraId, 'configure', { fps: parseInt(fps) });
              }
              break;
            case '3':
              const exposure = prompt(`Enter exposure (auto or value):`, camera.exposure);
              if (exposure) {
                this.sendDeviceControlCommand(cameraId, 'configure', { exposure });
              }
              break;
            case '4':
              this.calibrateStereoPair(camera.stereoPairId);
              break;
          }
        } else {
          // Standard camera configuration
          const resolution = prompt(`Configure ${camera.name}:\nEnter resolution (e.g., 1920x1080):`, camera.resolution);
          if (resolution) {
            this.sendDeviceControlCommand(cameraId, 'configure', { resolution });
          }
        }
      }
    },

    // Enhanced external device control with protocol-specific options
    toggleDevice(deviceId) {
      const device = this.externalDevices.find(dev => dev.id === deviceId);
      if (device) {
        if (device.status === 'connected') {
          // If connected, disconnect with protocol-specific handling
          this.sendDeviceControlCommand(deviceId, 'disconnect', { protocol: device.protocol });
        } else {
          // If disconnected, connect with protocol-specific parameters
          const connectParams = {
            protocol: device.protocol
          };
          
          // Add protocol-specific parameters
          if (device.protocol === 'Serial') {
            connectParams.port = device.port || prompt(`Enter serial port (e.g., COM3):`);
            connectParams.baudRate = device.baudRate || parseInt(prompt(`Enter baud rate (e.g., 9600):`));
          } else if (device.protocol === 'MQTT' || device.protocol === 'HTTP' || device.protocol === 'Custom') {
            connectParams.address = device.address || prompt(`Enter ${device.protocol} address:`);
          }
          
          this.sendDeviceControlCommand(deviceId, 'connect', connectParams);
        }
      }
    },

    // Enhanced device configuration with protocol-specific settings
    configureDevice(deviceId) {
      const device = this.externalDevices.find(dev => dev.id === deviceId);
      if (device) {
        // Create protocol-specific configuration UI
        let configParams = {};
        
        switch(device.protocol) {
          case 'Serial':
            const port = prompt(`Serial Port:`, device.port || 'COM3');
            const baudRate = prompt(`Baud Rate:`, device.baudRate || '9600');
            if (port && baudRate) {
              configParams = { port, baudRate: parseInt(baudRate) };
            }
            break;
          case 'MQTT':
            const address = prompt(`MQTT Broker Address:`, device.address || 'mqtt://localhost:1883');
            const topic = prompt(`MQTT Topic:`, device.topic || 'devices/' + deviceId);
            if (address && topic) {
              configParams = { address, topic };
            }
            break;
          case 'HTTP':
            const httpUrl = prompt(`HTTP Endpoint URL:`, device.address || 'http://localhost:3000/devices/' + deviceId);
            const method = prompt(`HTTP Method (GET/POST):`, device.httpMethod || 'POST');
            if (httpUrl && method) {
              configParams = { address: httpUrl, method };
            }
            break;
          default:
            const params = prompt(`Configure ${device.name}:\nEnter configuration parameters:`, device.config?.parameters || 'default');
            if (params) {
              configParams = { parameters: params };
            }
        }
        
        if (Object.keys(configParams).length > 0) {
          this.sendDeviceControlCommand(deviceId, 'configure', configParams);
        }
      }
    },

    // Control robotic arm
    controlRoboticArm(position, power) {
      this.sendDeviceControlCommand('device1', 'control', { position, power });
    },

    // Control LED
    controlLED(brightness, color) {
      this.sendDeviceControlCommand('device2', 'control', { brightness, color });
    },

    // Enhanced message handling from WebSocket
    handleDeviceControlMessage(event) {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'initial_status') {
          // Update device states from initial status
          this.updateDeviceStates(message.devices);
          this.addSystemMessage('Device control: Initial status received');
        } else if (message.type === 'command_response') {
          // Handle command response
          this.handleCommandResponse(message);
        } else if (message.type === 'sensor_update') {
          // Update sensor data
          this.updateSensorDataFromServer(message.device_id, message.data);
        } else if (message.type === 'device_update') {
          // Update single device status
          const deviceData = {[message.device_id]: message.data};
          this.updateDeviceStates(deviceData);
        } else if (message.type === 'system_message') {
          // Add system message
          this.addSystemMessage(message.content);
        } else if (message.type === 'pong') {
          // Just acknowledge pong responses
        } else {
          if (import.meta.env.DEV) {
            console.log('Received unknown WebSocket message type:', message.type);
          }
        }
      } catch (error) {
        console.error('Failed to parse device control message:', error);
      }
    },
    
    // Enhanced sensor data update with support for multiple sensor types
    updateSensorDataFromServer(deviceId, data) {
      // Update sensor data based on device ID and data received
      try {
        if (!deviceId || !data) return;
        
        // Group sensors by type for easier management
        const sensorTypeMap = {
          'temperature': ['sensor_temp', 'temp_sensor', 'temperature'],
          'humidity': ['sensor_humidity', 'hum_sensor', 'humidity'],
          'pressure': ['sensor_pressure', 'pres_sensor', 'pressure'],
          'light': ['sensor_light', 'light_sensor', 'light'],
          'sound': ['sensor_sound', 'audio_sensor', 'sound'],
          'motion': ['sensor_motion', 'movement_sensor', 'motion'],
          'accelerometer': ['sensor_accel', 'accelerometer', 'accel'],
          'gyroscope': ['sensor_gyro', 'gyroscope', 'gyro'],
          'magnetometer': ['sensor_mag', 'magnetometer', 'mag'],
          'distance': ['sensor_distance', 'range_sensor', 'distance'],
          'proximity': ['sensor_proximity', 'prox_sensor', 'proximity'],
          'depth': ['sensor_depth', 'depth_camera', 'depth'],
          'custom': ['custom_sensor']
        };
        
        // Get timestamp from server data or use current time
        const timestamp = data.timestamp || new Date().toISOString();
        
        // Find matching sensor type
        for (const [sensorType, ids] of Object.entries(sensorTypeMap)) {
          if (ids.some(id => deviceId.includes(id))) {
            switch(sensorType) {
              case 'temperature':
              case 'humidity':
              case 'pressure':
              case 'light':
              case 'sound':
              case 'distance':
              case 'proximity':
                // Store value with timestamp
                this.sensorData[sensorType] = {
                  value: data.value,
                  timestamp: timestamp
                };
                break;
              case 'motion':
                // Store value with timestamp
                this.sensorData.motion = {
                  value: data.value,
                  timestamp: timestamp
                };
                break;
              case 'accelerometer':
              case 'gyroscope':
              case 'magnetometer':
                // Handle 3-axis sensors with timestamp
                this.sensorData[sensorType] = {
                  x: data.x || 0,
                  y: data.y || 0,
                  z: data.z || 0,
                  timestamp: timestamp
                };
                break;
              case 'depth':
                // Handle depth maps for stereo vision with timestamp
                if (data.depthMap) {
                  this.sensorData.depthMap = {
                    data: data.depthMap,
                    timestamp: timestamp
                  };
                }
                if (data.pointCloud) {
                  this.sensorData.pointCloud = {
                    data: data.pointCloud,
                    timestamp: timestamp
                  };
                }
                break;
              case 'custom':
                // Handle custom sensor data with timestamp
                if (!this.sensorData.custom) {
                  this.sensorData.custom = {};
                }
                this.sensorData.custom[deviceId] = {
                  ...data,
                  timestamp: timestamp
                };
                break;
            }
            break;
          }
        }
        
        // Log sensor data update
        logInfo(`Updated sensor data from ${deviceId}`, data);
      } catch (error) {
        handleApiError(error, 'Update sensor data');
      }
    },
    
    // Stereo vision specific methods
    async calibrateStereoPair(pairId) {
      const stereoPair = this.stereoVisionPairs.find(pair => pair.id === pairId);
      if (stereoPair) {
        this.addSystemMessage(`Starting calibration for stereo pair: ${stereoPair.name}`);
        try {
          const response = await api.cameras.calibrateStereoPair(pairId, {
            leftCameraId: stereoPair.leftCameraId,
            rightCameraId: stereoPair.rightCameraId
          });
          
          if (response.data.success) {
            this.addSystemMessage(`Calibration for stereo pair ${stereoPair.name} completed successfully`);
            // Update calibration data if available
            if (response.data.calibrationData) {
              this.stereoCalibrationData[pairId] = response.data.calibrationData;
            }
          } else {
            throw new Error(response.data.detail || 'Calibration failed');
          }
        } catch (error) {
          handleApiError(error, 'Calibrate stereo pair');
          this.addSystemMessage(`Failed to calibrate stereo pair: ${error.message || error}`);
        }
      }
    },
    
    // Enable/disable stereo vision processing
    async toggleStereoVision(pairId) {
      const stereoPair = this.stereoVisionPairs.find(pair => pair.id === pairId);
      if (stereoPair) {
        const isEnabled = !stereoPair.isActive;
        try {
          const response = await api.cameras.processStereoPair(pairId, {
            enabled: isEnabled
          });
          
          if (response.data.success) {
            stereoPair.isActive = isEnabled;
            this.addSystemMessage(`${isEnabled ? 'Enabled' : 'Disabled'} stereo vision for pair: ${stereoPair.name}`);
          } else {
            throw new Error(response.data.detail || 'Failed to toggle stereo vision');
          }
        } catch (error) {
          handleApiError(error, 'Switch stereo vision');
          this.addSystemMessage(`Failed to ${isEnabled ? 'enable' : 'disable'} stereo vision: ${error.message || error}`);
        }
      }
    },
    
    // Get 3D position from stereo vision
    get3DPosition(leftPixel, rightPixel, pairId) {
      // Simple implementation - in a real system this would use proper stereo triangulation
      // based on calibration data
      try {
        if (!pairId || !this.stereoCalibrationData[pairId]) {
          throw new Error('Stereo vision not calibrated for this pair');
        }
        
        const calibrationData = this.stereoCalibrationData[pairId];
        if (!calibrationData.baseline || !calibrationData.focal_length) {
          throw new Error('Stereo vision not properly calibrated');
        }
        
        const disparity = Math.abs(leftPixel.x - rightPixel.x);
        if (disparity === 0) return null;
        
        // Calculate depth using triangulation formula
        const depth = (calibrationData.baseline * calibrationData.focal_length) / disparity;
        
        // Calculate 3D coordinates
        const principalPoint = calibrationData.principal_point || { x: 320, y: 240 };
        const x = (leftPixel.x - principalPoint.x) * depth / calibrationData.focal_length;
        const y = (leftPixel.y - principalPoint.y) * depth / calibrationData.focal_length;
        
        return { x, y, z: depth };
      } catch (error) {
        handleApiError(error, 'Calculate 3D position');
        return null;
      }
    },

    // Set up real-time input listeners
    setupRealTimeInputListeners() {
      logInfo('Setting up real-time input event listeners');
      // Event listeners are already bound via @ in the template, additional initialization logic can be added here
    },
    
    // Video Dialog Methods
    // Get video dialog status text
    getVideoDialogStatusText() {
      switch (this.videoDialogStatus) {
        case 'inactive':
          return 'Inactive';
        case 'connecting':
          return 'Connecting...';
        case 'active':
          return 'In conversation';
        case 'error':
          return 'Error';
        default:
          return 'Unknown';
      }
    },
    
    // Format response time for display
    formatResponseTime(index) {
      const now = new Date();
      return now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit',
        hour12: false 
      });
    },
    
    // Clear video responses
    clearVideoResponses() {
      if (this.videoDialogResponses.length > 0) {
        this.videoDialogResponses = [];
        logInfo('Video dialog response cleared');
      }
    },
    
    // List available cameras
    async listCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.availableCameras = devices.filter(device => device.kind === 'videoinput');
        this.isCameraAvailable = this.availableCameras.length > 0;
        
        if (this.availableCameras.length > 0) {
          this.selectedCamera = this.availableCameras[0].deviceId;
          logSuccess(`Found ${this.availableCameras.length} cameras`);
        } else {
          logWarning('No cameras available');
        }
      } catch (error) {
        handleApiError(error, 'Fetch camera list');
        this.isCameraAvailable = false;
      }
    },
    
    // Toggle video dialog
    async toggleVideoDialog() {
      if (this.isVideoDialogActive) {
        await this.stopVideoDialog();
      } else {
        await this.startVideoDialog();
      }
    },
    
    // Start video dialog
    async startVideoDialog() {
      try {
        this.videoDialogStatus = 'connecting';
        
        // List cameras if not already listed
        if (this.availableCameras.length === 0) {
          await this.listCameras();
        }
        
        if (!this.isCameraAvailable) {
          throw new Error('No cameras available');
        }
        
        // Connect to WebSocket
        await this.connectVideoWebSocket();
        
        // Start camera
        await this.startCamera();
        
        // Start video capture
        this.startVideoCapture();
        
        this.isVideoDialogActive = true;
        this.videoDialogStatus = 'active';
        logSuccess('Video dialog started');
      } catch (error) {
        handleApiError(error, 'Start video dialog');
        this.videoDialogStatus = 'error';
        this.isVideoDialogActive = false;
        this.addSystemMessage(`Failed to start video dialog: ${error.message || error}`);
      }
    },
    
    // Stop video dialog
    async stopVideoDialog() {
      // Stop video capture
      if (this.videoCaptureInterval) {
        clearInterval(this.videoCaptureInterval);
        this.videoCaptureInterval = null;
      }
      
      // Stop heartbeat
      this.stopWebSocketHeartbeat();
      
      // Stop camera
      await this.stopCamera();
      
      // Disconnect WebSocket
      await this.disconnectVideoWebSocket();
      
      // Reset performance tracking
      this.frameCount = 0;
      this.isFrameProcessing = false;
      
      this.isVideoDialogActive = false;
      this.videoDialogStatus = 'inactive';
      logInfo('Video dialog stopped');
    },
    
    // Connect to video WebSocket
    async connectVideoWebSocket() {
      return new Promise((resolve, reject) => {
        try {
          this.videoWebSocket = new WebSocket(this.videoDialogWebSocketUrl);
          
          this.videoWebSocket.onopen = () => {
            this.isWebSocketConnected = true;
            logSuccess('Video WebSocket connection established');
            
            // Start heartbeat mechanism
            this.startWebSocketHeartbeat();
            
            resolve();
          };
          
          this.videoWebSocket.onmessage = (event) => {
            this.handleVideoWebSocketMessage(event.data);
          };
          
          this.videoWebSocket.onerror = (error) => {
            console.error('Video WebSocket error:', error);
            this.isWebSocketConnected = false;
            this.stopWebSocketHeartbeat();
            reject(error);
          };
          
          this.videoWebSocket.onclose = () => {
            this.isWebSocketConnected = false;
            this.stopWebSocketHeartbeat();
            logWarning('Video WebSocket connection closed');
            
            // Attempt to reconnect if video dialog is still active
            if (this.isVideoDialogActive) {
              logInfo('Attempting to reconnect WebSocket...');
              setTimeout(() => {
                this.reconnectVideoWebSocket();
              }, 3000);
            }
          };
        } catch (error) {
          console.error('Failed to create video WebSocket:', error);
          reject(error);
        }
      });
    },
    
    // Disconnect video WebSocket
    async disconnectVideoWebSocket() {
      if (this.videoWebSocket) {
        this.videoWebSocket.close();
        this.videoWebSocket = null;
      }
      this.isWebSocketConnected = false;
    },
    
    // Handle video WebSocket message
    handleVideoWebSocketMessage(data) {
      try {
        const message = JSON.parse(data);
        
        if (message.type === 'video_processed') {
          // Calculate response time
          const frameNumber = message.data?.frame_number || this.frameCount;
          const receiveTime = performance.now();
          const sendTime = this.frameTimestamps.get(frameNumber);
          
          let responseTime = null;
          if (sendTime) {
            responseTime = receiveTime - sendTime;
            this.responseTimes.push(responseTime);
            
            // Keep only last 50 response times
            if (this.responseTimes.length > 50) {
              this.responseTimes = this.responseTimes.slice(-50);
            }
            
            // Calculate average response time
            const sum = this.responseTimes.reduce((a, b) => a + b, 0);
            this.averageResponseTime = sum / this.responseTimes.length;
            
            // Remove stored timestamp
            this.frameTimestamps.delete(frameNumber);
          }
          
          // Add AI response from video processing
          const responseText = message.data?.response || message.data?.result || 'Video processed';
          const responseInfo = responseTime ? 
            `${responseText} (Response time: ${responseTime.toFixed(1)}ms)` : 
            responseText;
          
          this.videoDialogResponses.push(responseInfo);
          logInfo(`Received video processing response: ${responseText.substring(0, 50)}... ${responseTime ? `(Response time: ${responseTime.toFixed(1)}ms)` : ''}`);
          
          // Limit responses to last 10
          if (this.videoDialogResponses.length > 10) {
            this.videoDialogResponses = this.videoDialogResponses.slice(-10);
          }
          
          // Process object detection results if available
          if (message.data && message.data.object_detection) {
            const detectionData = message.data.object_detection;
            const objects = detectionData.objects || [];
            
            // Update object detection results
            this.objectDetectionResults = objects;
            
            // Draw bounding boxes on video overlay
            this.drawObjectDetections(objects);
            
            logInfo(`Object detection: found ${objects.length} objects`);
          } else {
            // Clear object detection results if no data
            this.objectDetectionResults = [];
            this.clearObjectDetections();
          }
          
          // Emit event if needed
          this.$emit('video-dialog-response', responseText);
        } else if (message.type === 'connected') {
          logSuccess(`Video stream connection established: ${message.message || ''}`);
          // Clear any previous object detection results
          this.objectDetectionResults = [];
          this.clearObjectDetections();
        } else if (message.type === 'error') {
          logWarning(`Video dialog error: ${message.message || message.content || 'Unknown error'}`);
        } else if (message.type === 'response') {
          // Legacy support for response type
          this.videoDialogResponses.push(message.content);
          logInfo(`Received AI response: ${message.content.substring(0, 50)}...`);
          
          if (this.videoDialogResponses.length > 10) {
            this.videoDialogResponses = this.videoDialogResponses.slice(-10);
          }
          
          this.$emit('video-dialog-response', message.content);
        } else if (message.type === 'status') {
          logInfo(`Video dialog status: ${message.content || message.message}`);
        }
      } catch (error) {
        console.error('Failed to process WebSocket message:', error);
      }
    },
    
    // Draw object detection bounding boxes on video overlay
    drawObjectDetections(objects) {
      try {
        // Get or create overlay canvas context
        if (!this.videoOverlayCanvas) {
          this.videoOverlayCanvas = this.$refs.videoOverlayCanvas;
          if (!this.videoOverlayCanvas) return;
          
          this.videoOverlayContext = this.videoOverlayCanvas.getContext('2d');
          if (!this.videoOverlayContext) return;
        }
        
        const canvas = this.videoOverlayCanvas;
        const ctx = this.videoOverlayContext;
        const video = this.$refs.videoDialogPreview;
        
        // Ensure canvas matches video dimensions
        if (video && video.videoWidth > 0 && video.videoHeight > 0) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
        } else {
          // Default dimensions if video not ready
          canvas.width = canvas.offsetWidth || 640;
          canvas.height = canvas.offsetHeight || 480;
        }
        
        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!objects || objects.length === 0) return;
        
        // Draw each bounding box
        objects.forEach(obj => {
          const bbox = obj.bbox || [0, 0, 0, 0]; // [x1, y1, x2, y2]
          const label = obj.label || 'object';
          const confidence = obj.confidence || 0;
          
          // Scale coordinates if needed (assuming bbox coordinates are normalized 0-1)
          let x1, y1, x2, y2;
          if (bbox.length === 4) {
            // Check if coordinates are normalized (0-1) or absolute
            if (bbox[0] <= 1 && bbox[1] <= 1 && bbox[2] <= 1 && bbox[3] <= 1) {
              // Normalized coordinates, scale to canvas size
              x1 = bbox[0] * canvas.width;
              y1 = bbox[1] * canvas.height;
              x2 = bbox[2] * canvas.width;
              y2 = bbox[3] * canvas.height;
            } else {
              // Absolute coordinates, use as is
              x1 = bbox[0];
              y1 = bbox[1];
              x2 = bbox[2];
              y2 = bbox[3];
            }
            
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Determine color based on confidence
            const hue = confidence * 120; // 0 (red) to 120 (green)
            const color = `hsl(${hue}, 100%, 50%)`;
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label background
            const labelText = `${label} (${(confidence * 100).toFixed(1)}%)`;
            ctx.font = '14px Arial';
            const textWidth = ctx.measureText(labelText).width;
            const textHeight = 16;
            
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - textHeight, textWidth + 8, textHeight);
            
            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.textBaseline = 'top';
            ctx.fillText(labelText, x1 + 4, y1 - textHeight + 2);
          }
        });
        
        logInfo(`Drawn ${objects.length} object detection bounding boxes`);
      } catch (error) {
        console.error('Failed to draw object detections:', error);
      }
    },
    
    // Clear object detection drawings
    clearObjectDetections() {
      try {
        if (this.videoOverlayContext && this.videoOverlayCanvas) {
          this.videoOverlayContext.clearRect(0, 0, this.videoOverlayCanvas.width, this.videoOverlayCanvas.height);
        }
        this.objectDetectionResults = [];
      } catch (error) {
        console.error('Failed to clear object detections:', error);
      }
    },
    
    // WebSocket heartbeat methods
    startWebSocketHeartbeat() {
      // Clear any existing heartbeat
      this.stopWebSocketHeartbeat();
      
      // Send heartbeat every 10 seconds
      this.heartbeatInterval = setInterval(() => {
        if (this.videoWebSocket && this.videoWebSocket.readyState === WebSocket.OPEN) {
          try {
            this.videoWebSocket.send(JSON.stringify({
              type: 'heartbeat',
              timestamp: Date.now(),
              status: 'alive'
            }));
          } catch (error) {
            console.error('Failed to send WebSocket heartbeat:', error);
          }
        }
      }, 10000); // 10 seconds
    },
    
    stopWebSocketHeartbeat() {
      if (this.heartbeatInterval) {
        clearInterval(this.heartbeatInterval);
        this.heartbeatInterval = null;
      }
    },
    
    // Reconnect video WebSocket
    async reconnectVideoWebSocket() {
      if (!this.isVideoDialogActive) return;
      
      try {
        logInfo('Reconnecting video WebSocket...');
        await this.connectVideoWebSocket();
        logSuccess('Video WebSocket reconnection successful');
      } catch (error) {
        console.error('Video WebSocket reconnection failed:', error);
        // Try again after 5 seconds if still active
        if (this.isVideoDialogActive) {
          setTimeout(() => {
            this.reconnectVideoWebSocket();
          }, 5000);
        }
      }
    },
    
    // Start camera
    async startCamera() {
      try {
        const constraints = {
          video: {
            deviceId: this.selectedCamera ? { exact: this.selectedCamera } : undefined,
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        };
        
        this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        this.$refs.videoDialogPreview.srcObject = this.videoStream;
        logSuccess('Camera started');
      } catch (error) {
        handleApiError(error, 'Start camera');
        throw error;
      }
    },
    
    // Stop camera
    async stopCamera() {
      if (this.videoStream) {
        const tracks = this.videoStream.getTracks();
        tracks.forEach(track => track.stop());
        this.videoStream = null;
        
        if (this.$refs.videoDialogPreview) {
          this.$refs.videoDialogPreview.srcObject = null;
        }
        
        logInfo('Camera stopped');
      }
    },
    
    // Start video capture
    startVideoCapture() {
      // Performance optimization for video frame capture
      this.videoCaptureInterval = setInterval(() => {
        if (this.isWebSocketConnected && this.isVideoDialogActive) {
          // Skip frame if previous frame is still being processed
          if (!this.isFrameProcessing) {
            this.captureVideoFrame();
          } else {
            // Optional: log skipped frame for debugging
            if (this.videoDialogStatus === 'active' && performance.now() - this.lastFrameLog > 5000) {
              console.log('Video frame skipped - previous frame still processing');
              this.lastFrameLog = performance.now();
            }
          }
        }
      }, 250); // Capture frame every 250 milliseconds for smoother real-time video (4 FPS)
    },
    
    // Capture video frame
    captureVideoFrame() {
      // Set frame processing flag
      this.isFrameProcessing = true;
      const startTime = performance.now();
      
      try {
        const video = this.$refs.videoDialogPreview;
        if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
          this.isFrameProcessing = false;
          return;
        }
        
        // Use reusable canvas objects for performance
        if (!this.videoCanvas) {
          this.videoCanvas = document.createElement('canvas');
          this.videoContext = this.videoCanvas.getContext('2d');
          this.compressionCanvas = document.createElement('canvas');
          this.compressionContext = this.compressionCanvas.getContext('2d');
        }
        
        // Update canvas size if needed
        if (this.videoCanvas.width !== video.videoWidth || this.videoCanvas.height !== video.videoHeight) {
          this.videoCanvas.width = video.videoWidth;
          this.videoCanvas.height = video.videoHeight;
        }
        
        // Capture frame
        this.videoContext.drawImage(video, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
        
        // Optimized compression: reduce resolution and quality based on performance
        const scale = 0.25; // 25% size for better performance (was 0.3)
        const quality = 0.6; // Lower quality for faster transmission (was 0.7)
        
        if (this.compressionCanvas.width !== this.videoCanvas.width * scale || 
            this.compressionCanvas.height !== this.videoCanvas.height * scale) {
          this.compressionCanvas.width = this.videoCanvas.width * scale;
          this.compressionCanvas.height = this.videoCanvas.height * scale;
        }
        
        // Compress image
        this.compressionContext.drawImage(
          this.videoCanvas, 
          0, 0, this.videoCanvas.width, this.videoCanvas.height,
          0, 0, this.compressionCanvas.width, this.compressionCanvas.height
        );
        
        // Convert to base64 with optimized quality
        const compressedImage = this.compressionCanvas.toDataURL('image/jpeg', quality);
        
        // Send to server
          if (this.videoWebSocket && this.isWebSocketConnected) {
            this.frameCount++;
            const timestamp = Date.now();
            const sendTime = performance.now();
            
            // Store frame timestamp for response time calculation
            this.frameTimestamps.set(this.frameCount, sendTime);
            
            this.videoWebSocket.send(JSON.stringify({
              type: 'video_stream',
              video_frame: compressedImage,
              width: this.compressionCanvas.width,
              height: this.compressionCanvas.height,
              timestamp: timestamp,
              lang: 'en',
              frame_number: this.frameCount
            }));
            
            // Log performance every 10 frames
            if (this.frameCount % 10 === 0) {
              const processTime = performance.now() - startTime;
              console.log(`Video frame ${this.frameCount} processed - dimensions: ${this.compressionCanvas.width}x${this.compressionCanvas.height}, processing time: ${processTime.toFixed(1)}ms`);
            }
          }
      } catch (error) {
        console.error('Failed to capture video frame:', error);
      } finally {
        // Reset frame processing flag
        this.isFrameProcessing = false;
      }
    },
    
    async startVoiceRecognition() {
      if (this.isVoiceInputActive) {
        // If already active, stop recognition
        this.stopVoiceRecognition();
        return;
      }
      
      try {
        // First, try to use real audio model for speech recognition if backend is connected
        if (this.backendConnected) {
          try {
            await this.startRealVoiceRecognition();
            return; // If successful, return early
          } catch (realError) {
            console.warn('Real voice recognition failed, falling back to browser SpeechRecognition:', realError);
            this.addSystemMessage('Real voice recognition failed, using browser speech recognition instead.');
            // Continue with browser SpeechRecognition as fallback
          }
        } else {
          this.addSystemMessage('Backend not connected, using browser speech recognition.');
        }
        
        // Check internet connectivity for speech recognition
        if (!navigator.onLine) {
          this.addSystemMessage('No internet connection detected. Speech recognition requires internet access. Please check your connection and try again.');
          return;
        }
        
        // Check browser support for speech recognition
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognitionAPI) {
          this.addSystemMessage('Your browser does not support speech recognition. Please use Chrome, Edge, or another supported browser.');
          return;
        }
        
        // Check page protocol (some browsers restrict microphone access over HTTP)
        if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
          this.addSystemMessage('Speech recognition may be restricted over HTTP. Please use HTTPS or local environment (localhost/127.0.0.1).');
          return;
        }
        
        // Check if speech recognition is initialized
        if (!this.recognition) {
          this.initSpeechRecognition();
        }
        
        if (this.recognition) {
          // Request microphone permission before starting recognition
          try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
          } catch (err) {
            console.error('Microphone permission error:', err);
            let errorMessage = 'Microphone access denied. Please enable microphone access in your browser settings and try again.';
            this.addSystemMessage(errorMessage);
            this.isVoiceInputActive = false;
            return;
          }
          
          // Configure speech recognition for English support
          this.recognition.continuous = true; // Enable continuous listening
          this.recognition.interimResults = true; // Enable interim results
          this.recognition.lang = 'en-US'; // English language support
          this.recognition.maxAlternatives = 3; // Get multiple alternatives
          
          this.recognition.onstart = () => {
            logSuccess('Voice recognition started');
            this.addSystemMessage('Voice recognition started, please start speaking...');
            // Add visual feedback for active recording
            this.isVoiceInputActive = true;
          };
          
          this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
              const transcript = event.results[i][0].transcript;
              if (event.results[i].isFinal) {
                finalTranscript += transcript;
              } else {
                interimTranscript += transcript;
              }
            }
            
            // Update input field with interim results
            if (interimTranscript) {
              this.inputText = interimTranscript;
              // Show visual feedback for interim results
              this.addSystemMessage(`Recognizing: ${interimTranscript}`);
            }
            
            // When final result is ready, send message
            if (finalTranscript.trim()) {
              this.inputText = finalTranscript;
              logSuccess(`Voice recognition result: ${finalTranscript}`);
              this.addSystemMessage(`Voice input received: ${finalTranscript}`);
              // Auto send message with slight delay for better UX
              setTimeout(() => {
                if (this.isMounted) {
                  this.sendMessage();
                  // Reset voice input state after sending
                  setTimeout(() => {
                    this.isVoiceInputActive = false;
                  }, 500);
                }
              }, 800);
            }
          };
          
          this.recognition.onerror = (event) => {
            handleApiError(new Error(event.error), 'Voice recognition error');
            let errorMessage = 'Voice recognition error';
            switch(event.error) {
              case 'not-allowed':
                errorMessage = 'Microphone access denied, please check browser permissions';
                break;
              case 'no-speech':
                errorMessage = 'No speech detected, please ensure microphone is working';
                break;
              case 'audio-capture':
                errorMessage = 'Audio capture failed, please check microphone connection';
                break;
              case 'network':
                errorMessage = 'Network error: Browser speech recognition requires internet connection to Google services. Please check:\n1. Your internet connection\n2. Firewall settings (Google speech recognition services may be blocked)\n3. Try using text input or switch to a different browser\nIf the issue persists, you may need to enable access to Google services.';
                break;
              default:
                errorMessage = `Voice recognition error: ${event.error}`;
            }
            this.addSystemMessage(errorMessage);
            this.isVoiceInputActive = false;
          };
          
          this.recognition.onend = () => {
            this.isVoiceInputActive = false;
            this.addSystemMessage('Voice recognition ended');
          };
          
          // Start speech recognition
          try {
            this.recognition.start();
            logInfo('Starting English voice recognition...');
          } catch (error) {
            handleApiError(error, 'Start voice recognition');
            this.isVoiceInputActive = false;
            this.addSystemMessage('Voice recognition startup failed, please try again later');
          }
        } else {
          // If speech recognition initialization failed, notify user
          logWarning('Voice recognition initialization failed');
          this.addSystemMessage('Speech recognition initialization failed. Please check browser permissions and try again.');
          this.isVoiceInputActive = false;
        }
      } catch (error) {
        handleApiError(error, 'Start voice recognition');
        this.isVoiceInputActive = false;
        this.addSystemMessage(`Voice recognition failed: ${error.message || error}`);
      }
    },
    
    stopVoiceRecognition() {
      if (this.recognition && this.recognition.stop) {
        try {
          this.recognition.stop();
          logInfo('Voice recognition manually stopped');
          this.addSystemMessage('Voice recognition stopped');
        } catch (error) {
          handleApiError(error, 'Stop voice recognition');
        }
      }
      this.isVoiceInputActive = false;
      // Also stop real voice recording if active
      if (this.isRecording) {
        this.stopRealVoiceRecognition();
      }
    },
    
    async startRealVoiceRecognition() {
      if (this.isRecording) {
        // If already recording, stop
        await this.stopRealVoiceRecognition();
        return;
      }
      
      try {
        this.addSystemMessage('Starting real voice recognition using local audio model...');
        this.isRecording = true;
        this.isVoiceInputActive = true;
        
        // Request microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            sampleRate: 16000, // Match audio model expected sample rate
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true
          }
        });
        
        // Create MediaRecorder
        this.mediaRecorder = new MediaRecorder(this.mediaStream, {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 128000
        });
        
        this.audioChunks = [];
        
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data);
          }
        };
        
        this.mediaRecorder.onstop = async () => {
          try {
            this.audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
            await this.processAudioWithBackend(this.audioBlob);
          } catch (error) {
            console.error('Error processing audio:', error);
            this.addSystemMessage('Error processing audio with local model.');
            throw error;
          } finally {
            this.isRecording = false;
            this.isVoiceInputActive = false;
            this.cleanupAudioRecording();
          }
        };
        
        // Start recording
        this.mediaRecorder.start();
        this.addSystemMessage('Recording started. Speak now...');
        
        // Auto-stop after 10 seconds of silence or 30 seconds max
        setTimeout(() => {
          if (this.isRecording) {
            this.stopRealVoiceRecognition();
          }
        }, 30000);
        
      } catch (error) {
        console.error('Real voice recognition failed:', error);
        this.addSystemMessage(`Real voice recognition failed: ${error.message || error}`);
        this.isRecording = false;
        this.isVoiceInputActive = false;
        this.cleanupAudioRecording();
        throw error;
      }
    },
    
    async stopRealVoiceRecognition() {
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
        this.addSystemMessage('Recording stopped, processing...');
      }
      this.cleanupAudioStream();
    },
    
    cleanupAudioRecording() {
      this.cleanupAudioStream();
      this.audioChunks = [];
      this.audioBlob = null;
    },
    
    cleanupAudioStream() {
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(track => track.stop());
        this.mediaStream = null;
      }
      this.mediaRecorder = null;
    },
    
    async processAudioWithBackend(audioBlob) {
      try {
        this.addSystemMessage('Processing audio with local audio model...');
        
        // Convert blob to base64 for API
        const reader = new FileReader();
        const audioBase64 = await new Promise((resolve, reject) => {
          reader.onload = () => {
            const base64 = reader.result.split(',')[1]; // Remove data URL prefix
            resolve(base64);
          };
          reader.onerror = reject;
          reader.readAsDataURL(audioBlob);
        });
        
        // Send to backend audio processing API
        const response = await api.process.audio({
          audio: audioBase64,
          language: 'en-US',
          session_id: `session_${Date.now()}`,
          model_id: 'audio'
        });
        
        if (response.data.status === 'success') {
          const text = response.data.data.text || response.data.data;
          if (text && text.trim()) {
            this.inputText = text;
            this.addSystemMessage(`Voice input received from local model: ${text}`);
            // Auto send message
            setTimeout(() => {
              if (this.isMounted) {
                this.sendMessage();
              }
            }, 500);
          } else {
            throw new Error('No text returned from audio model');
          }
        } else {
          throw new Error(response.data.message || 'Audio processing failed');
        }
      } catch (error) {
        console.error('Audio processing error:', error);
        this.addSystemMessage(`Audio processing failed: ${error.message || error}`);
        throw error;
      }
    },
    
    selectImage() {
      const imageInput = this.$refs.imageInput;
      if (imageInput) {
        imageInput.click();
      } else {
          handleApiError(new Error('Image input element not found'), 'UI element error');
          this.addSystemMessage('Image input element not found');
        }
    },

    selectVideo() {
      const videoInput = this.$refs.videoInput;
      if (videoInput) {
        videoInput.click();
      } else {
          handleApiError(new Error('Video input element not found'), 'UI element error');
          this.addSystemMessage('Video input element not found');
        }
    },
    
    // Toggle real-time input display
    toggleRealTimeInput() {
      this.showRealTimeInput = !this.showRealTimeInput;
    },

    async handleVideoUpload(event) {
      const file = event.target.files[0];
      if (file) {
        try {
          // Actually upload video to backend
          logInfo(`Uploading video: ${file.name}`);
          // Add system message for upload start
          this.addSystemMessage(`Uploading video: ${file.name}`);
          
          // Check file size (video files are usually larger)
          const maxSize = 50 * 1024 * 1024; // 50MB
          if (file.size > maxSize) {
            throw new Error('File is too large');
          }
          
          // Check file type
          const validVideoTypes = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime'];
          if (!validVideoTypes.includes(file.type)) {
            throw new Error('Invalid video format');
          }
          
          // Create FormData object
          const formData = new FormData();
          formData.append('video', file);
          formData.append('lang', document.documentElement.lang || 'en');
          
          // Add loading status message
          const loadingMessageId = Date.now() + 0.5;
          const loadingMessage = {
            id: loadingMessageId,
            type: 'loading',
            content: 'Processing video...',
            time: new Date().toLocaleTimeString()
          };
          this.messages.push(loadingMessage);
          this.saveMessages();
          
          // Send to backend API for video processing
          const response = await api.post('/api/process/video', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            timeout: 300000 // 5 minutes timeout, video processing may take longer
          });
          
          // Remove loading status message
          this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
          
          if (response.data.status === 'success') {
            const videoMessage = {
              id: Date.now(),
              type: 'user',
              content: `[Video: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: response.data.data,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(videoMessage, botMessage);
            // Save to local storage
            this.saveMessages();
          } else {
            throw new Error(response.data.detail || 'Failed to process video');
          }
          
          // Clear file input
          event.target.value = '';
        } catch (error) {
          handleApiError(error, 'Upload video');
          // Add error message
          this.addSystemMessage(`Failed to upload video: ${error.message || error}`);
          // Clear file input
          event.target.value = '';
        }
      }
    },
    
    async handleImageUpload(event) {
      const file = event.target.files[0];
      if (file) {
        try {
          // Actually upload image to backend
          logInfo(`Uploading image: ${file.name}`);
          // Add system message for upload start
          this.addSystemMessage(`Uploading image: ${file.name}`);
          
          // Check file size
          const maxSize = 5 * 1024 * 1024; // 5MB
          if (file.size > maxSize) {
            throw new Error('File is too large');
          }
          
          // Check file type
          const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];
          if (!validImageTypes.includes(file.type)) {
            throw new Error('Invalid image format');
          }
          
          // Create FormData object
          const formData = new FormData();
          formData.append('image', file);
          formData.append('lang', document.documentElement.lang || 'en');
          
          // Add loading status message
          const loadingMessageId = Date.now() + 0.5;
          const loadingMessage = {
            id: loadingMessageId,
            type: 'loading',
            content: 'Processing image...',
            time: new Date().toLocaleTimeString()
          };
          this.messages.push(loadingMessage);
          this.saveMessages();
          
          // Send to backend API for image processing
          const response = await api.post('/api/process/image', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            timeout: 60000 // 1 minute timeout
          });
          
          // Remove loading status message
          this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
          
          if (response.data.status === 'success') {
            const imageMessage = {
              id: Date.now(),
              type: 'user',
              content: `[Image: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: response.data.data,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(imageMessage, botMessage);
            // Save to local storage
            this.saveMessages();
          } else {
            throw new Error(response.data.detail || 'Failed to process image');
          }
          
          // Clear file input
          event.target.value = '';
        } catch (error) {
          handleApiError(error, 'Upload image');
          // Add error message
          this.addSystemMessage(`Failed to upload image: ${error.message || error}`);
          // Clear file input
          event.target.value = '';
        }
      }
    },
    
    // Initialize WebSocket and device control systems
    initializeDeviceControl() {
      // Connect to device control WebSocket
      this.connectDeviceControlWebSocket();
      
      // Vue 3 compatible - cleanup will be handled by onBeforeUnmount hook
      // Removed this.$once which is not supported in Vue 3
    },
    
    // Enhanced device data loading with stereo vision pairs and calibration
    async loadDeviceData() {
      try {
        logInfo('Loading device data from backend...');
        
        // Load cameras data with enhanced properties
        const camerasResponse = await api.cameras.getList();
        if (camerasResponse.data.status === 'success') {
          this.cameras = (camerasResponse.data.cameras || []).map(camera => ({
            ...camera,
            resolution: camera.resolution || '1280x720',
            fps: camera.fps || 30,
            exposure: camera.exposure || 'auto',
            gain: camera.gain || 0,
            isStereo: camera.isStereo || false,
            stereoRole: camera.stereoRole || null, // 'left', 'right', or null
            stereoPairId: camera.stereoPairId || null,
            isStreaming: false, // Initialize streaming status
            websocket: null // Initialize WebSocket reference
          }));
          logInfo(`Loaded ${this.cameras.length} cameras`);
        } else {
          throw new Error('Failed to load cameras data');
        }
        
        // Load external devices data
        const externalDevicesResponse = await api.devices.getExternalDevices();
        if (externalDevicesResponse.data && externalDevicesResponse.data.status === 'success') {
          this.externalDevices = externalDevicesResponse.data.devices || [];
        } else {
          this.externalDevices = [];
        }
        logInfo(`Loaded ${this.externalDevices.length} external devices`);
        
        // Load stereo vision pairs
        const stereoPairsResponse = await api.cameras.getStereoPairs();
        if (stereoPairsResponse.data && stereoPairsResponse.data.status === 'success') {
          this.stereoVisionPairs = stereoPairsResponse.data.pairs || [];
        } else {
          this.stereoVisionPairs = [];
        }
        logInfo(`Loaded ${this.stereoVisionPairs.length} stereo vision pairs`);
        
        // Load stereo calibration data
        try {
          const calibrationResponse = await api.cameras.getStereoCalibration();
          if (calibrationResponse.data.status === 'success') {
            this.stereoCalibrationData = calibrationResponse.data.data || {};
            logInfo('Loaded stereo vision calibration data');
          }
        } catch (error) {
          logWarning('Failed to load stereo calibration data');
        }
        
        this.addSystemMessage(`Device data loaded: ${this.cameras.length} cameras, ${this.externalDevices.length} external devices, ${this.stereoVisionPairs.length} stereo pairs`);
        
      } catch (error) {
        handleApiError(error, 'Load device data');
        this.addSystemMessage('Failed to load device data from backend. Please ensure the backend service is running.');
        
        // Initialize with empty arrays
        this.cameras = [];
        this.externalDevices = [];
        this.stereoVisionPairs = [];
        this.stereoCalibrationData = {};
      }
    },
    
    // Get WebSocket base URL from API configuration
    getWebSocketBaseUrl() {
      // Always connect to backend server on port 8000 for WebSocket connections
      // Vite proxy doesn't handle WebSocket connections, so we need direct connection
      let baseUrl;
      if (typeof window !== 'undefined' && window.location) {
        // Use backend port 8000 regardless of frontend port
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const hostname = window.location.hostname;
        // Backend WebSocket runs on port 8000 (main server port)
        baseUrl = `${protocol}//${hostname}:8000`;
      } else {
        // Fallback for development or SSR
        baseUrl = 'ws://localhost:8000';
      }
      // Convert HTTP/HTTPS to WebSocket protocol is already handled above
      return baseUrl;
    },
    
    // Process user input based on type
    async processUserInput(inputData, inputType) {
      try {
        logInfo(`Processing ${inputType} input`);
        
        const payload = {
          data: inputData,
          type: inputType,
          session_id: this.getSessionId(),
          timestamp: new Date().toISOString()
        };
        
        const response = await api.post('/api/process/input', payload, {
          timeout: 60000
        });
        
        if (response.data.status === 'success') {
          return response.data.data;
        } else {
          throw new Error(response.data.detail || `Failed to process ${inputType} input`);
        }
      } catch (error) {
        handleApiError(error, `Process ${inputType} input`);
        throw error; // Propagate error instead of returning placeholder message
      }
    },
    
    // Format sensor type for display
    formatSensorType(sensorType) {
      return sensorType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    },
    
    // Get sensor data timestamp from real data
    getSensorTimestamp(sensorType, deviceId) {
      // Get actual timestamp from sensor data if available
      if (deviceId) {
        // Handle custom sensors by device ID
        if (this.sensorData.custom && this.sensorData.custom[deviceId] && this.sensorData.custom[deviceId].timestamp) {
          return new Date(this.sensorData.custom[deviceId].timestamp).toLocaleTimeString();
        }
      } else if (sensorType) {
        // Handle standard sensors by type
        if (this.sensorData[sensorType] && this.sensorData[sensorType].timestamp) {
          return new Date(this.sensorData[sensorType].timestamp).toLocaleTimeString();
        }
      }
      // Fallback to current time if no timestamp available
      return new Date().toLocaleTimeString();
    },
    
    // Toggle camera stream
    toggleCameraStream(cameraId) {
      const camera = this.cameras.find(c => c.id === cameraId);
      if (camera) {
        const isStreaming = camera.isStreaming || false;
        
        if (isStreaming) {
          // Stop streaming
          this.stopCameraStream(cameraId);
          camera.isStreaming = false;
          this.addSystemMessage(`Stopped stream for camera: ${camera.name}`);
        } else {
          // Start streaming
          this.startCameraStream(cameraId);
          camera.isStreaming = true;
          this.addSystemMessage(`Started stream for camera: ${camera.name}`);
        }
        
        logInfo(`${camera.name} stream switch, new status: ${camera.isStreaming ? 'Streaming' : 'Stopped'}`);
      }
    },
    
    // Start camera stream
    async startCameraStream(cameraId) {
      try {
        const response = await api.cameras.startStream(cameraId);
        
        if (response.data.success) {
          // Create WebSocket URL for camera feed
          const camera = this.cameras.find(c => c.id === cameraId);
          if (camera) {
            // Store the WebSocket URL
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.host;
            camera.streamUrl = `${wsProtocol}//${wsHost}/ws/camera-feed/${cameraId}`;
            
            // Create a video element for preview
            this.createCameraPreview(cameraId, camera.streamUrl);
          }
        } else {
          throw new Error(response.data.detail || 'Failed to start camera stream');
        }
      } catch (error) {
        handleApiError(error, 'Start camera stream');
        this.addSystemMessage(`Failed to start camera stream: ${error.message || error}`);
        
        // Reset streaming state
        const camera = this.cameras.find(c => c.id === cameraId);
        if (camera) {
          camera.isStreaming = false;
        }
      }
    },
    
    // Stop camera stream
    async stopCameraStream(cameraId) {
      try {
        const response = await api.cameras.stopStream(cameraId);
        
        if (!response.data.success) {
          throw new Error(response.data.detail || 'Failed to stop camera stream');
        }
        
        // Clean up preview
        this.removeCameraPreview(cameraId);
        
        // Clear stream URL
        const camera = this.cameras.find(c => c.id === cameraId);
        if (camera) {
          camera.streamUrl = null;
        }
      } catch (error) {
        handleApiError(error, 'Stop camera stream');
        this.addSystemMessage(`Failed to stop camera stream: ${error.message || error}`);
      }
    },
    
    // Create camera preview element
    createCameraPreview(cameraId, streamUrl) {
      logInfo(`Creating preview for camera ${cameraId}, URL: ${streamUrl}`);
      
      // Find or create a container for the preview
      let previewContainer = document.getElementById(`camera-preview-${cameraId}`);
      if (!previewContainer) {
        // Find the camera card
        const cameraCard = document.querySelector(`[data-camera-id="${cameraId}"]`);
        if (cameraCard) {
          // Create preview container
          previewContainer = document.createElement('div');
          previewContainer.id = `camera-preview-${cameraId}`;
          previewContainer.className = 'camera-preview-container';
          
          // Create video element
          const videoElement = document.createElement('video');
          videoElement.id = `camera-video-${cameraId}`;
          videoElement.className = 'camera-video';
          videoElement.autoplay = true;
          videoElement.playsInline = true;
          
          // Add loading indicator
          const loadingIndicator = document.createElement('div');
          loadingIndicator.className = 'loading-indicator';
          loadingIndicator.textContent = 'Loading stream...';
          
          // Add elements to container
          previewContainer.appendChild(loadingIndicator);
          previewContainer.appendChild(videoElement);
          
          // Insert after device info but before controls
          const deviceInfo = cameraCard.querySelector('.device-info');
          if (deviceInfo) {
            deviceInfo.after(previewContainer);
          } else {
            cameraCard.appendChild(previewContainer);
          }
          
          // Set up WebSocket connection for video stream
          this.setupCameraWebSocket(cameraId, streamUrl, videoElement, loadingIndicator);
        }
      }
    },
    
    // Set up WebSocket connection for camera stream
    setupCameraWebSocket(cameraId, streamUrl, videoElement, loadingIndicator) {
      try {
        // Create WebSocket
        const ws = new WebSocket(streamUrl);
        
        // Store WebSocket reference
        const camera = this.cameras.find(c => c.id === cameraId);
        if (camera) {
          camera.websocket = ws;
        }
        
        // Handle WebSocket events
        ws.onopen = () => {
          logInfo(`Camera ${cameraId} WebSocket connection established`);
          if (loadingIndicator) {
            loadingIndicator.textContent = 'Streaming...';
          }
        };
        
        ws.onmessage = (event) => {
          // Process received frame data
          if (event.data instanceof Blob) {
            const blobUrl = URL.createObjectURL(event.data);
            if (videoElement.src !== blobUrl) {
              videoElement.src = blobUrl;
            }
          } else if (typeof event.data === 'string') {
            try {
              const message = JSON.parse(event.data);
              if (message.type === 'frame' && message.data) {
                // For base64 encoded frames
                const blob = this.base64ToBlob(message.data, 'image/jpeg');
                const blobUrl = URL.createObjectURL(blob);
                if (videoElement.src !== blobUrl) {
                  videoElement.src = blobUrl;
                }
              }
            } catch (e) {
              console.error('Failed to parse WebSocket message:', e);
            }
          }
          
          // Hide loading indicator once we receive the first frame
          if (loadingIndicator && loadingIndicator.parentNode) {
            loadingIndicator.style.display = 'none';
          }
        };
        
        ws.onerror = (error) => {
          handleApiError(error, `Camera ${cameraId} WebSocket error`);
          this.addSystemMessage(`Camera ${cameraId} stream error`);
          if (loadingIndicator) {
            loadingIndicator.textContent = 'Stream error';
            loadingIndicator.className = 'loading-indicator error';
          }
        };
        
        ws.onclose = () => {
          logInfo(`Camera ${cameraId} WebSocket connection closed`);
          if (camera) {
            camera.websocket = null;
          }
          if (loadingIndicator && loadingIndicator.parentNode) {
            loadingIndicator.textContent = 'Stream closed';
          }
        };
        
      } catch (error) {
        handleApiError(error, `Setting up WebSocket for camera ${cameraId} failed`);
        this.addSystemMessage(`Failed to connect to camera ${cameraId} stream`);
        if (loadingIndicator) {
          loadingIndicator.textContent = 'Connection failed';
          loadingIndicator.className = 'loading-indicator error';
        }
      }
    },
    
    // Helper function to convert base64 to Blob
    base64ToBlob(base64Data, contentType) {
      contentType = contentType || '';
      const sliceSize = 1024;
      const byteCharacters = atob(base64Data);
      const byteArrays = [];
      
      for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
          byteNumbers[i] = slice.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
      }
      
      return new Blob(byteArrays, {type: contentType});
    },
    
    // Remove camera preview element
    removeCameraPreview(cameraId) {
      logInfo(`Removing preview for camera ${cameraId}`);
      
      // Close WebSocket if exists
      const camera = this.cameras.find(c => c.id === cameraId);
      if (camera && camera.websocket) {
        try {
          camera.websocket.close();
        } catch (error) {
          console.warn('Failed to close WebSocket:', error);
        }
        camera.websocket = null;
      }
      
      // Remove preview container
      const previewContainer = document.getElementById(`camera-preview-${cameraId}`);
      if (previewContainer && previewContainer.parentNode) {
        previewContainer.parentNode.removeChild(previewContainer);
      }
    },
    
    // View 3D model from stereo pair
    view3DModel(pairId) {
      const pair = this.stereoVisionPairs.find(p => p.id === pairId);
      if (pair && pair.calibrated) {
        // Implement 3D view logic
        logInfo(`Viewing 3D model for ${pair.name}`);
        this.addSystemMessage(`Displaying 3D model from stereo pair: ${pair.name}`);
      }
    },

    // Serial Communication Methods
    async refreshSerialPorts() {
      try {
        this.addSystemMessage('Refreshing serial ports...');
        const response = await api.serial.getPorts();
        this.availableSerialPorts = response.data.ports || [];
        this.addSystemMessage(`Found ${this.availableSerialPorts.length} serial ports`);
      } catch (error) {
        handleApiError(error, 'Refresh serial ports');
        this.addSystemMessage('Failed to refresh serial ports. Please ensure the backend service is running.');
        this.availableSerialPorts = [];
      }
    },

    handlePortChange() {
      this.addSystemMessage(`Selected port: ${this.selectedSerialPort}`);
    },

    async connectSerialPort() {
      if (!this.selectedSerialPort) {
        this.addSystemMessage('Please select a serial port first');
        return;
      }

      try {
        this.addSystemMessage(`Connecting to ${this.selectedSerialPort} at ${this.serialBaudRate} baud...`);
        const response = await api.serial.connect({
          port: this.selectedSerialPort,
          baud_rate: parseInt(this.serialBaudRate)
        });

        if (response.data.success) {
          this.serialConnected = true;
          this.addSystemMessage(`Successfully connected to ${this.selectedSerialPort}`);
          this.startSerialListener();
        } else {
          throw new Error(response.data.message || 'Failed to connect');
        }
      } catch (error) {
        handleApiError(error, `Connect to ${this.selectedSerialPort}`);
        this.addSystemMessage(`Failed to connect to ${this.selectedSerialPort}: ${error.message}`);
        this.serialConnected = false;
      }
    },

    async disconnectSerialPort() {
      try {
        this.stopSerialListener();
        // Only make API call if actually connected and have a selected port
        if (this.serialConnected && this.selectedSerialPort) {
          await api.serial.disconnect({ port: this.selectedSerialPort });
          this.serialConnected = false;
          this.addSystemMessage(`Disconnected from ${this.selectedSerialPort}`);
        } else {
          // Just update UI state if not actually connected
          this.serialConnected = false;
        }
      } catch (error) {
        handleApiError(error, 'Disconnect serial port');
        this.addSystemMessage('Failed to disconnect serial port');
        // Force disconnect state
        this.serialConnected = false;
        this.stopSerialListener();
      }
    },

    async sendSerialData() {
      if (!this.serialConnected || !this.serialSendData.trim()) {
        return;
      }

      try {
        let dataToSend = this.serialSendData;
        
        // Apply formatting options
        if (this.appendCR) {
          dataToSend += '\r';
        }
        if (this.appendLF) {
          dataToSend += '\n';
        }

        const response = await api.serial.send(dataToSend);
        
        if (response.data.success) {
          this.addSystemMessage(`Sent data to ${this.selectedSerialPort}`);
          // Clear send data after successful send
          this.serialSendData = '';
        } else {
          throw new Error(response.data.message || 'Failed to send data');
        }
      } catch (error) {
        handleApiError(error, 'Send serial data');
        this.addSystemMessage('Failed to send data');
      }
    },

    clearReceivedData() {
      this.serialReceivedData = '';
      this.addSystemMessage('Cleared received data');
    },

    startSerialListener() {
      // Stop any existing listener
      this.stopSerialListener();
      
      // Start new listener
      this.serialListenerInterval = setInterval(async () => {
        await this.readSerialData();
      }, 500); // Check for new data every 500ms
    },

    stopSerialListener() {
      if (this.serialListenerInterval) {
        clearInterval(this.serialListenerInterval);
        this.serialListenerInterval = null;
      }
    },

    async readSerialData() {
      try {
        const response = await api.serial.read();
        
        if (response.data.success && response.data.data) {
          this.serialReceivedData += response.data.data;
          this.scrollToBottom();
        }
      } catch (error) {
        // Only log errors if we're still supposed to be connected
        if (this.serialConnected) {
          // In a real app, we might want to handle reconnection here
          // For now, just silently handle it
        }
      }
    },

    cleanupCameraWebSockets() {
      // Clean up all camera WebSocket connections to prevent memory leaks
      if (this.cameras && Array.isArray(this.cameras)) {
        this.cameras.forEach(camera => {
          if (camera.websocket) {
            try {
              camera.websocket.close();
            } catch (error) {
              console.warn('Failed to close camera WebSocket:', error);
            }
            camera.websocket = null;
          }
        });
      }
    },

    scrollToBottom() {
      if (this.autoScroll && this.$refs.serialReceiveArea) {
        const area = this.$refs.serialReceiveArea;
        area.scrollTop = area.scrollHeight;
      }
    }
  }
};
</script>

<style scoped>
/* Model connection status area style - black, white, gray style */
.model-status-area {
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background-color: var(--bg-secondary);
}

.model-status-area h3 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.connected-models {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.main-model-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-name {
  font-weight: 500;
  color: var(--text-primary);
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--text-tertiary);
}

.status-indicator.connected {
    background-color: #4caf50; /* Green - Connected */
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
  }

  .status-indicator.connecting {
    background-color: #ff9800; /* Orange - Connecting */
  }

  .status-indicator.disconnected {
    background-color: #f44336; /* Red - Disconnected */
  }

.status-text {
  font-size: 14px;
  color: var(--text-secondary);
}

.active-models {
  font-size: 14px;
  color: var(--text-secondary);
}

/* Conversation title and clear button area */
.conversation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.conversation-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.clear-btn {
  padding: 6px 12px;
  font-size: 14px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background-color: var(--bg-secondary);
  color: var(--text-secondary);
  cursor: pointer;
  transition: var(--transition);
}

.clear-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>

<style scoped>
/* Clean black, white, and gray style CSS variables definition */
:root {
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
  --border-radius: 8px;
  --border-radius-lg: 12px;
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.08);
  --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.12);
  --transition: all 0.2s ease;
  --bg-primary: #ffffff;
  --bg-secondary: #f5f5f5;
  --bg-tertiary: #e9e9e9;
  --text-primary: #222222;
  --text-secondary: #555555;
  --text-tertiary: #888888;
  --border-color: #dddddd;
  --border-light: #eeeeee;
  --border-dark: #cccccc;
}

.home-view {
  padding: var(--spacing-lg);
  max-width: 1400px;
  margin: 0 auto;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--border-light);
}

.header h1 {
  color: var(--text-primary);
  font-size: 28px;
  font-weight: 600;
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  flex-wrap: wrap;
}

.server-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-color);
  transition: var(--transition);
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.status-dot.connected {
  background-color: var(--text-primary);
  box-shadow: 0 0 0 2px rgba(100, 100, 100, 0.3);
}

.status-dot.disconnected {
  background-color: var(--text-tertiary);
  box-shadow: 0 0 0 2px rgba(200, 200, 200, 0.3);
}

.header-buttons {
  display: flex;
}

/* Device Status Section Styles */
.device-status-section {
  margin-bottom: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
}

.device-status-section h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-lg);
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
}

.device-category {
  margin-bottom: var(--spacing-xl);
}

.device-category:last-child {
  margin-bottom: 0;
}

.device-category h4 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
  border-bottom: 1px solid var(--border-light);
  padding-bottom: var(--spacing-sm);
}

.device-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
}

.device-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.device-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.device-card.stereo-pair {
  border-left: 4px solid var(--text-primary);
}

.device-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-sm);
}

.device-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 14px;
}

.device-status {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: var(--text-tertiary);
  transition: var(--transition);
}

.device-status.active,
.device-status.connected {
  background-color: var(--text-primary);
  box-shadow: 0 0 0 2px rgba(34, 34, 34, 0.1);
}

.device-status.available {
  background-color: var(--text-secondary);
}

.device-status.error {
  background-color: var(--text-secondary);
}

.device-info {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: var(--spacing-md);
  line-height: 1.4;
}

.device-type,
.device-resolution,
.device-protocol,
.stereo-cameras {
  display: block;
  margin-bottom: var(--spacing-xs);
}

.calibration-status {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  margin-top: var(--spacing-xs);
}

.calibration-status.calibrated {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.calibration-status.uncalibrated {
  background-color: var(--bg-secondary);
  color: var(--text-secondary);
  border: 1px dashed var(--border-color);
}

.device-controls {
  display: flex;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-md);
}

.device-btn {
  flex: 1;
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 11px;
  transition: var(--transition);
  white-space: nowrap;
}

.device-btn:hover {
  background: var(--bg-tertiary);
  border-color: var(--border-dark);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.device-btn:disabled {
  background: var(--bg-secondary);
  color: var(--text-tertiary);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* Sensor Data Display */
.sensor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: var(--spacing-md);
}

.sensor-device-header {
  margin-bottom: var(--spacing-md);
}

.sensor-device-header h5 {
  margin: 0;
  font-size: 14px;
  color: var(--text-primary);
  font-weight: 600;
}

.sensor-data {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
}

.sensor-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm) 0;
  border-bottom: 1px solid var(--border-light);
  font-size: 13px;
}

.sensor-item:last-child {
  border-bottom: none;
}

.sensor-label {
  font-weight: 500;
  color: var(--text-primary);
}

.sensor-value {
  color: var(--text-secondary);
  font-weight: 600;
  margin-left: var(--spacing-md);
}

.sensor-timestamp {
  font-size: 11px;
  color: var(--text-tertiary);
  margin-left: auto;
  white-space: nowrap;
}

.model-performance {
  font-size: 14px;
  color: var(--text-secondary);
}

.input-area {
  margin-bottom: var(--spacing-lg);
}

.input-area h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: var(--spacing-md);
}

.chat-container {
  height: 500px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 24px;
  margin-bottom: 24px;
  overflow-y: auto;
  background: var(--bg-secondary);
  scrollbar-width: thin;
  scrollbar-color: var(--border-color) var(--bg-primary);
}

.chat-container::-webkit-scrollbar {
  width: 8px;
}

.chat-container::-webkit-scrollbar-track {
  background: var(--bg-primary);
  border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb {
  background: var(--border-color);
  border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
  background: var(--border-dark);
}

.message {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid var(--border-color);
  animation: fadeIn 0.3s ease-in-out;
  max-width: 70%;
  word-wrap: break-word;
  line-height: 1.6;
  position: relative;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  background: var(--bg-primary);
  margin-left: auto;
  border-color: var(--border-color);
  border-bottom-right-radius: 6px;
}

.message.bot {
  background: var(--bg-tertiary);
  margin-right: auto;
  border-color: var(--border-color);
  border-bottom-left-radius: 6px;
}

.message.system {
  background: var(--bg-secondary);
  text-align: center;
  margin: 16px auto;
  border-color: var(--border-color);
  font-style: italic;
  color: var(--text-secondary);
  max-width: 90%;
}

.message.loading {
  background: var(--bg-tertiary);
  margin-right: auto;
  border-color: var(--border-color);
  position: relative;
  overflow: hidden;
  border-bottom-left-radius: 6px;
}

.message.loading::after {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(200, 200, 200, 0.1), transparent);
  animation: loadingShimmer 1.5s infinite;
}

@keyframes loadingShimmer {
  100% {
    left: 100%;
  }
}

/* Empty chat state */
.chat-container:empty::before {
  content: 'No messages yet. Start a conversation!';
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-tertiary);
  font-style: italic;
  text-align: center;
  padding: 24px;
}

/* Enhanced message styles for improved user experience */
.message-wrapper {
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease-in-out;
}

.message-sender {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 13px;
}

.sender-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}

.user-avatar {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.bot-avatar {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
}

.sender-name {
  font-weight: 600;
  color: var(--text-primary);
  flex: 1;
}

.message-timestamp {
  color: var(--text-tertiary);
  font-size: 11px;
  opacity: 0.8;
}

.message-content {
  margin-bottom: 8px;
  color: var(--text-primary);
  font-size: 15px;
  line-height: 1.6;
  padding: 12px 16px;
  border-radius: 12px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
}

.message.user .message-content {
  background: linear-gradient(135deg, #e3f2fd, #bbdefb);
  border-color: #90caf9;
  margin-left: auto;
  max-width: 85%;
}

.message.bot .message-content {
  background: linear-gradient(135deg, #f3e5f5, #e1bee7);
  border-color: #ce93d8;
  margin-right: auto;
  max-width: 85%;
}

.message-confidence {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}

.confidence-label {
  font-weight: 500;
}

.confidence-bar {
  flex: 1;
  height: 6px;
  background: var(--bg-tertiary);
  border-radius: 3px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #4caf50, #8bc34a);
  border-radius: 3px;
  transition: width 0.5s ease;
}

.confidence-value {
  font-weight: 600;
  color: var(--text-primary);
  min-width: 40px;
  text-align: right;
}

.system-message-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--bg-secondary);
  border-radius: 20px;
  margin: 12px auto;
  max-width: 80%;
  font-size: 13px;
  color: var(--text-secondary);
}

.system-icon {
  font-size: 14px;
}

.system-label {
  font-weight: 500;
}

.loading-message-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 16px;
  background: var(--bg-tertiary);
  border-radius: 12px;
  margin: 16px auto;
  max-width: 70%;
  border: 1px solid var(--border-color);
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border-color);
  border-top-color: var(--text-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  color: var(--text-primary);
  font-weight: 500;
}

.message-time {
  font-size: 11px;
  color: var(--text-tertiary);
  text-align: right;
  opacity: 0.7;
  display: block;
}

.input-controls {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
  align-items: center;
}

.input-controls input {
  flex: 1;
  min-width: 200px;
  padding: 12px 20px;
  border: 2px solid var(--border-color);
  border-radius: 24px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 15px;
  transition: all 0.2s ease;
}

.input-controls input:focus {
  outline: none;
  border-color: #666;
  box-shadow: 0 0 0 3px rgba(102, 102, 102, 0.1);
}

.input-controls button {
  padding: 12px 24px;
  background: #333;
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
  white-space: nowrap;
  font-size: 15px;
}

.input-controls button:hover {
  background: #555;
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.input-controls button:disabled {
  background: var(--bg-tertiary);
  color: var(--text-tertiary);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.input-options {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 12px;
  border: 1px solid var(--border-color);
}

.input-options button {
  padding: 10px 20px;
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
  white-space: nowrap;
  font-weight: 500;
}

.input-options button:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
  border-color: #666;
}

.input-options button:disabled {
  background: var(--bg-tertiary);
  color: var(--text-tertiary);
  border-color: var(--border-color);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* Voice recording indicator styles */
.voice-btn.voice-active {
  border-color: #ff4757;
  background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), rgba(255, 71, 87, 0.05));
  color: #ff4757;
  box-shadow: 0 0 0 3px rgba(255, 71, 87, 0.1);
  animation: pulse 1.5s infinite ease-in-out;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(255, 71, 87, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(255, 71, 87, 0);
  }
}

.recording-indicator {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  width: 20px;
  height: 20px;
  margin-right: 8px;
}

.recording-dot {
  width: 12px;
  height: 12px;
  background: #ff4757;
  border-radius: 50%;
  animation: blink 1.2s infinite;
  box-shadow: 0 0 10px #ff4757;
}

@keyframes blink {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(0.9);
  }
}

.voice-btn .btn-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 6px;
}

.quick-actions {
  margin-bottom: var(--spacing-lg);
}

.quick-actions h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: var(--spacing-md);
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-md);
}

.actions-grid .nav-link {
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 16px;
  text-align: center;
  text-decoration: none;
  display: block;
  transition: var(--transition);
  font-weight: 500;
}

.actions-grid .nav-link:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.real-time-section {
  margin-top: 24px;
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 24px;
  background: var(--bg-primary);
  box-shadow: var(--shadow-sm);
  transition: all 0.3s ease;
}

.real-time-section:hover {
  box-shadow: var(--shadow-md);
}

.real-time-section h3 {
  color: var(--text-primary);
  font-size: 18px;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color);
}

.guide-button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  transition: var(--transition);
}

.guide-button:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

/* Management model status display style */
.conversation-header {
  display: flex;
  align-items: center;
  gap: 16px;
  justify-content: space-between;
  flex-wrap: wrap;
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
}

.conversation-header > div:first-child {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.conversation-header h2 {
  color: var(--text-primary);
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 0;
  margin-top: 0;
}

.main-model-status.inline-status {
  background-color: var(--bg-secondary);
  border-radius: var(--border-radius);
  padding: 8px 16px;
  font-size: 0.9em;
  white-space: nowrap;
  font-family: inherit;
  font-weight: 500;
  color: var(--text-primary);
  line-height: 1.4;
  letter-spacing: normal;
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--text-tertiary);
}

.status-indicator.connected {
  background-color: #666;
  box-shadow: 0 0 0 3px rgba(102, 102, 102, 0.2);
}

.status-indicator.connecting {
  background-color: #999;
  animation: pulse 1.5s infinite;
}

.status-text {
  font-size: 12px;
  font-weight: 400;
}

/* Multi-camera and Device Status Section Styles */
.device-status-section {
  margin-bottom: var(--spacing-lg);
  padding: var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-secondary);
}

.device-status-section h3 {
  color: var(--text-primary);
  font-size: 18px;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid var(--border-color);
}

.device-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: var(--spacing-md);
}

.device-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  background: var(--bg-primary);
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.device-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-dark);
  transform: translateY(-1px);
}

.device-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-sm);
}

.device-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 14px;
}

.device-status {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.device-status.available {
  background-color: #4caf50; /* Green */
  box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
}

.device-status.active {
  background-color: #2196f3; /* Blue */
  box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.3);
  animation: pulse 2s infinite;
}

.device-status.connected {
  background-color: #4caf50; /* Green */
  box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
}

.device-status.error {
      background-color: #f44336; /* Red */
      box-shadow: 0 0 0 2px rgba(244, 67, 54, 0.3);
    }
    
    /* Camera Preview Styles */
    .camera-preview-container {
      position: relative;
      margin: var(--spacing-md) 0;
      border: 1px solid var(--border-color);
      border-radius: var(--border-radius);
      overflow: hidden;
      background-color: var(--bg-secondary);
    }
    
    .camera-video {
      width: 100%;
      height: auto;
      max-height: 300px;
      object-fit: cover;
    }
    
    .loading-indicator {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background-color: rgba(0, 0, 0, 0.7);
      color: white;
      padding: var(--spacing-sm) var(--spacing-md);
      border-radius: var(--border-radius);
      font-size: 12px;
      font-weight: 500;
      z-index: 10;
    }
    
    .loading-indicator.error {
      background-color: rgba(244, 67, 54, 0.7);
    }

.device-controls {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: space-between;
}

.device-btn {
  flex: 1;
  padding: 6px 12px;
  font-size: 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-secondary);
  color: var(--text-primary);
  cursor: pointer;
  transition: var(--transition);
  font-weight: 500;
}

.device-btn:hover {
  background: var(--bg-tertiary);
  border-color: var(--border-dark);
  transform: translateY(-1px);
}

.device-btn:active {
  transform: translateY(0);
}

/* Sensor data display styles */
.sensor-data-section {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
}

/* Serial Communication Section Styles */
.serial-communication-section {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
}

.serial-communication-section h4 {
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
}

.serial-connection-controls {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.serial-connection-controls > * {
  flex: 1;
  min-width: 150px;
}

.serial-data-section {
  margin-top: var(--spacing-md);
}

.serial-data-section h5 {
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 600;
  margin-top: var(--spacing-sm);
  margin-bottom: var(--spacing-xs);
}

.serial-send-controls {
  display: flex;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.serial-send-controls input[type="text"] {
  flex: 1;
}

.serial-options {
  display: flex;
  gap: var(--spacing-md);
  align-items: center;
  margin-bottom: var(--spacing-sm);
  font-size: 13px;
}

.serial-options label {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  cursor: pointer;
}

.serial-receive-area {
  width: 100%;
  height: 200px;
  padding: var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-secondary);
  font-family: monospace;
  font-size: 13px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.serial-buttons {
  display: flex;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-sm);
}

.serial-status {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius);
  font-size: 12px;
  font-weight: 500;
}

.serial-status.connected {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.serial-status.disconnected {
  background: var(--bg-secondary);
  color: var(--text-secondary);
}

.sensor-data-section h4 {
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
}

.sensor-data-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: var(--spacing-sm);
}

.sensor-data-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--spacing-sm);
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-color);
}

.sensor-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 4px;
}

.sensor-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.sensor-value.true {
  color: #4caf50;
}

.sensor-value.false {
  color: #f44336;
}

/* Responsive Design */
@media (max-width: 768px) {
  .home-view {
    padding: 16px;
  }
  
  .header {
    flex-direction: column;
    gap: 16px;
    align-items: flex-start;
  }
  
  .header-right {
    width: 100%;
    justify-content: space-between;
  }
  
  .header-buttons {
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .status-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  
  .conversation-header {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }
  
  .conversation-header > div:first-child {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }
  
  .conversation-header h2 {
    font-size: 20px;
  }
  
  .chat-container {
    height: 400px;
    padding: 16px;
  }
  
  .message {
    max-width: 85%;
    padding: 12px 16px;
  }
  
  .message.user {
    margin-left: auto;
    margin-right: 10px;
  }
  
  .message.bot {
    margin-right: auto;
    margin-left: 10px;
  }
  
  .message-content {
    font-size: 14px;
  }
  
  .input-controls {
    flex-direction: column;
    gap: 10px;
  }
  
  .input-controls input {
    min-width: unset;
    width: 100%;
    font-size: 14px;
  }
  
  .input-controls button {
    width: 100%;
    font-size: 14px;
  }
  
  .input-options {
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    padding: 12px;
  }
  
  .input-options button {
    flex: 1;
    min-width: 120px;
    padding: 8px 16px;
    font-size: 13px;
  }
  
  .actions-grid {
    grid-template-columns: 1fr;
  }
  
  .real-time-section {
    padding: 16px;
  }
  
  .device-grid {
    grid-template-columns: 1fr;
  }
  
  .device-controls {
    flex-direction: column;
    gap: 8px;
  }
  
  .device-btn {
    width: 100%;
  }
  
  .sensor-data-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Video Dialog Styles */
.video-dialog-section {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-sm);
}

.video-dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  padding-bottom: var(--spacing-md);
  border-bottom: 1px solid var(--border-color);
}

.video-dialog-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
}

.video-dialog-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.video-dialog-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

@media (max-width: 1024px) {
  .video-dialog-content {
    grid-template-columns: 1fr;
  }
}

.video-preview-container {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.video-preview {
  position: relative;
  width: 100%;
  height: 300px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  overflow: hidden;
  background: var(--bg-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  color: var(--text-tertiary);
}

.placeholder-icon {
  font-size: 48px;
}

.placeholder-text {
  font-size: 14px;
  text-align: center;
  max-width: 80%;
}

.video-controls {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
}

.video-dialog-btn {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-lg);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
  font-size: 14px;
}

.video-dialog-btn:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}

.video-dialog-btn.active {
  background: linear-gradient(135deg, #ff4757, #ff6b81);
  color: white;
  border-color: #ff4757;
}

.camera-select {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
}

.ai-response-container {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-secondary);
  overflow: hidden;
}

.ai-response-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md);
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
}

.ai-response-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.ai-response-messages {
  flex: 1;
  height: 300px;
  overflow-y: auto;
  padding: var(--spacing-md);
}

.empty-responses {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-tertiary);
  text-align: center;
  font-style: italic;
  padding: var(--spacing-lg);
}

.response-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.response-item {
  padding: var(--spacing-md);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  transition: var(--transition);
}

.response-item:hover {
  box-shadow: var(--shadow-sm);
  border-color: var(--border-dark);
}

.response-content {
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
  margin-bottom: var(--spacing-xs);
}

.response-time {
  color: var(--text-tertiary);
  font-size: 11px;
  text-align: right;
}

.video-dialog-info {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
}

@media (max-width: 768px) {
  .video-dialog-info {
    grid-template-columns: 1fr;
  }
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm);
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-color);
}

.info-label {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
}

.info-value {
  color: var(--text-primary);
  font-size: 12px;
  font-weight: 600;
}

/* Video overlay canvas for object detection bounding boxes */
.video-overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10;
}

.video-preview {
  position: relative;
  width: 100%;
  height: 300px;
}

.video-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: var(--border-radius);
}

/* Object detection results display */
.object-detection-results {
  margin-top: 15px;
  padding: 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  max-height: 150px;
  overflow-y: auto;
}

.object-detection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.object-detection-header h5 {
  margin: 0;
  font-size: 14px;
  color: var(--text-primary);
}

.object-detection-count {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-primary);
  background: var(--bg-tertiary);
  padding: 2px 8px;
  border-radius: 10px;
}

.object-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.object-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: var(--bg-primary);
  border: 1px solid var(--border-light);
  border-radius: 4px;
  font-size: 11px;
}

.object-label {
  font-weight: 600;
  color: var(--text-primary);
}

.object-confidence {
  color: var(--text-secondary);
  font-size: 10px;
}
  </style>
