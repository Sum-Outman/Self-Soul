<template>
  <div class="home-view">
    <!-- User Guide Component -->
    <UserGuide v-if="showUserGuide" @close="showUserGuide = false" />
    
    <!-- Multi-camera and Device Status Section -->
    <div class="device-status-section">
      <h3>Multi-camera & Device Status</h3>
      <div class="device-grid">
        <div class="device-card" v-for="camera in cameras" :key="camera.id">
          <div class="device-header">
            <span class="device-name">{{ camera.name }}</span>
            <span class="device-status" :class="camera.status"></span>
          </div>
          <div class="device-controls">
            <button @click="toggleCamera(camera.id)" class="device-btn">
              {{ camera.active ? 'Stop' : 'Start' }}
            </button>
            <button @click="configureCamera(camera.id)" class="device-btn">
              Settings
            </button>
          </div>
        </div>
        
        <div class="device-card" v-for="device in externalDevices" :key="device.id">
          <div class="device-header">
            <span class="device-name">{{ device.name }}</span>
            <span class="device-status" :class="device.status"></span>
          </div>
          <div class="device-controls">
            <button @click="toggleDevice(device.id)" class="device-btn">
              {{ device.connected ? 'Disconnect' : 'Connect' }}
            </button>
            <button @click="configureDevice(device.id)" class="device-btn">
              Config
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Page Content Area -->
    
    <div class="input-area">
      <div class="conversation-header">
        <h2>Conversation</h2>
        <div class="main-model-status inline-status">
          <span class="model-name">Management Model</span>
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ modelConnectionStatus === 'connected' ? 'Connected' : modelConnectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected' }}</span>
        </div>
        <button @click="clearAllMessages" class="clear-btn" :disabled="messages.length === 0">
          Clear All Messages
        </button>
      </div>
      <div class="chat-container">
        <div v-for="message in messages" :key="message.id" class="message" :class="message.type">
          <div class="message-content">{{ message.content }}</div>
          <div class="message-time">{{ message.time }}</div>
        </div>
      </div>

      <div class="input-controls">
        <input type="text"
               v-model="inputText"
               @keyup.enter="sendMessage"
               placeholder="Type your message...">
        <button @click="sendMessage">Send</button>
        <div class="input-options">
          <button @click="startVoiceRecognition" :disabled="isVoiceInputActive">
            {{ isVoiceInputActive ? 'Stop Voice Input' : 'Voice Input' }}
          </button>
          <button @click="selectImage">Image Input</button>
          <input type="file" ref="imageInput" style="display: none"
                 accept="image/*" @change="handleImageUpload">
          <button @click="selectVideo">Video Input</button>
          <input type="file" ref="videoInput" style="display: none"
                 accept="video/*" @change="handleVideoUpload">
          <button @click="toggleRealTimeInput">
            {{ showRealTimeInput ? 'Hide Realtime' : 'Show Realtime' }}
          </button>
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
import api from '@/utils/api.js';
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'HomeView',
  components: {
    RealTimeInput,
    UserGuide
  },
  data() {
    return {
      inputText: '',
      messages: [],
      isVoiceInputActive: false,
      showRealTimeInput: false,
      showUserGuide: false,
      backendConnected: false,
      backendStatus: 'disconnected',
      recognition: null,
      videoInput: null,
      modelPerformanceData: [
        { id: 'manager', status: 'active', performance: 95 },
        { id: 'language', status: 'active', performance: 92 },
        { id: 'audio', status: 'active', performance: 88 },
        { id: 'vision_image', status: 'active', performance: 90 },
        { id: 'vision_video', status: 'active', performance: 85 },
        { id: 'spatial', status: 'active', performance: 82 },
        { id: 'sensor', status: 'active', performance: 87 },
        { id: 'computer', status: 'active', performance: 93 },
        { id: 'motion', status: 'active', performance: 80 },
        { id: 'knowledge', status: 'active', performance: 89 },
        { id: 'programming', status: 'active', performance: 91 }
      ],
      // models array synchronized with modelPerformanceData
      models: [
        { id: 'manager', status: 'active', performance: 95 },
        { id: 'language', status: 'active', performance: 92 },
        { id: 'audio', status: 'active', performance: 88 },
        { id: 'vision_image', status: 'active', performance: 90 },
        { id: 'vision_video', status: 'active', performance: 85 },
        { id: 'spatial', status: 'active', performance: 82 },
        { id: 'sensor', status: 'active', performance: 87 },
        { id: 'computer', status: 'active', performance: 93 },
        { id: 'motion', status: 'active', performance: 80 },
        { id: 'knowledge', status: 'active', performance: 89 },
        { id: 'programming', status: 'active', performance: 91 }
      ],
      modelConnectionStatus: 'unknown',
      // Add missing status
      managementModel: {
        name: 'A Management Model',
        status: 'inactive',
        lastActive: null
      },
      connectedText: '',
      activeModels: 0,
      // Multi-camera support data
      cameras: [
        { id: 'camera1', name: 'Left Camera', status: 'available', active: false, stream: null },
        { id: 'camera2', name: 'Right Camera', status: 'available', active: false, stream: null },
        { id: 'camera3', name: 'Depth Camera', status: 'available', active: false, stream: null }
      ],
      // External devices and sensors data
      externalDevices: [
        { id: 'sensor1', name: 'Temperature Sensor', status: 'available', connected: false, type: 'sensor' },
        { id: 'sensor2', name: 'Motion Sensor', status: 'available', connected: false, type: 'sensor' },
        { id: 'device1', name: 'Robotic Arm', status: 'available', connected: false, type: 'actuator' },
        { id: 'device2', name: 'LED Controller', status: 'available', connected: false, type: 'actuator' }
      ],
      // Sensor data storage
      sensorData: {
        temperature: null,
        motion: false,
        humidity: null,
        pressure: null
      }
    };
  },
  mounted() {
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
  },
  beforeUnmount() {
    // Remove event listeners when component is unmounted
    window.removeEventListener('voice-input', this.handleVoiceInputEvent);
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
      }
    },
    methods: {
      // Add system message
      addSystemMessage(content) {
        const systemMessage = {
          id: Date.now() + Math.random(),
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
          errorHandler.logInfo('Connecting to backend service...');
          this.backendStatus = 'connecting';
          
          // Try to connect to real backend
          const response = await api.get('/health');
          
          if (response.data.status === 'ok') {
            this.backendConnected = true;
            this.backendStatus = 'connected';
            errorHandler.logInfo('Successfully connected to backend service');
            
            // Update management model status
            this.managementModel.status = 'active';
            this.managementModel.lastActive = new Date().toISOString();
            this.modelConnectionStatus = 'connected';
          } else {
            throw new Error('Backend health check failed');
          }
        } catch (error) {
          errorHandler.handleError(error, 'Failed to connect to backend');
          this.backendConnected = false;
          this.backendStatus = 'disconnected';
          this.managementModel.status = 'inactive';
          this.modelConnectionStatus = 'disconnected';
        }
      },
      
      // Save messages to local storage
      saveMessages() {
        try {
          localStorage.setItem('chat_messages', JSON.stringify(this.messages));
        } catch (error) {
          errorHandler.handleError(error, 'Failed to save messages to local storage');
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
          errorHandler.handleError(error, 'Failed to load history messages');
          this.messages = [];
        }
      },

      // Initialize system
      initializeSystem() {
        errorHandler.logInfo('Self Soul System initializing...');
        // Show welcome message
        this.addSystemMessage('Welcome to the Self Soul System!');
        
        // Always try to connect to real backend, never use mock data automatically
        this.connectToBackend();
      },
    
    // Get session ID for tracking conversation context
    getSessionId() {
      let sessionId = localStorage.getItem('agi_session_id');
      if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
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
        errorHandler.handleError(error, 'Failed to process image');
        return 'Image analysis failed due to connection issues';
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
        errorHandler.handleError(error, 'Failed to process video');
        return 'Video analysis failed due to connection issues';
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
        const response = await api.post('/api/models/8001/chat', {
          message: messageText,
          session_id: this.getSessionId(),
          timestamp: new Date().toISOString()
        });
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        // Add model response with additional information from enhanced API
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: response.data.response || 'I don\'t have a response yet. Please try again.',
          time: new Date().toLocaleTimeString(),
          modelId: response.data.model_id || '8001',
          modelName: response.data.model_name || 'Management Model',
          confidence: response.data.confidence || 0.97
        };
        
        this.messages.push(botMessage);
        this.saveMessages();
        
        // Update model connection status
        this.modelConnectionStatus = 'connected';
        this.managementModel.status = 'active';
        
      } catch (error) {
        errorHandler.handleError(error, 'Failed to send message to management model');
        
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        // Add error message
        const errorMessage = {
          id: Date.now() + 1,
          type: 'system',
          content: 'Failed to connect to the management model. Using fallback response.',
          time: new Date().toLocaleTimeString()
        };
        
        const fallbackMessage = {
          id: Date.now() + 2,
          type: 'bot',
          content: 'I apologize, but I encountered an error while processing your message. Please try again later.',
          time: new Date().toLocaleTimeString()
        };
        
        this.messages.push(errorMessage, fallbackMessage);
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
        errorHandler.handleError(error, 'Failed to process audio');
        return 'Audio processing failed due to connection issues';
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
        errorHandler.handleError(error, 'Failed to initialize voice recognition');
      }
    },
    
    // Handle global voice input events from App.vue
    handleVoiceInputEvent(event) {
      const text = event.detail;
      if (text && text.trim()) {
        this.inputText = text;
        this.addSystemMessage(`Voice input received: ${text}`);
        // Auto send message
        setTimeout(() => {
          this.sendMessage();
        }, 500);
      }
    },

    // Handle real-time audio data
    async handleRealTimeAudioData(audioData) {
      try {
        errorHandler.logInfo('Received real-time audio data');
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
        errorHandler.handleError(error, 'Failed to process real-time audio data');
        this.addSystemMessage('Failed to process audio');
      }
    },

    // Get model display name
    getModelDisplayName(modelId) {
      const modelNames = {
        '8001': 'Management Model',
        '8002': 'Language Model',
        '8003': 'Knowledge Model',
        '8004': 'Vision Model',
        '8005': 'Audio Model',
        '8006': 'Autonomous Model',
        '8007': 'Programming Model',
        '8008': 'Planning Model',
        '8009': 'Emotion Model',
        '8010': 'Spatial Model',
        '8011': 'Computer Vision Model'
      };
      return modelNames[modelId] || `Model ${modelId}`;
    },

    // Handle real-time video data
    async handleRealTimeVideoData(videoData) {
      try {
        errorHandler.logInfo('Received real-time video data');
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
        errorHandler.handleError(error, 'Failed to process real-time video data');
        this.addSystemMessage('Failed to process video');
      }
    },

    // Handle real-time text data
    async handleRealTimeTextData(textData) {
      try {
        errorHandler.logInfo('Received real-time text data');
        this.addSystemMessage(`Real-time text received: ${textData}`);
        
        // Directly send message using text data
        this.inputText = textData;
        await this.sendMessage();
      } catch (error) {
        errorHandler.handleError(error, 'Failed to process real-time text data');
        this.addSystemMessage('Failed to process text');
      }
    },

    // Handle real-time file data
    async handleRealTimeFileData(fileData) {
      try {
        errorHandler.logInfo('Received real-time file data');
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
        errorHandler.handleError(error, 'Failed to process real-time file data');
        this.addSystemMessage('Failed to process file');
      }
    },

    // Multi-camera control methods
    toggleCamera(cameraId) {
      const camera = this.cameras.find(cam => cam.id === cameraId);
      if (camera) {
        camera.active = !camera.active;
        if (camera.active) {
          this.startCameraStream(cameraId);
          this.addSystemMessage(`${camera.name} started`);
        } else {
          this.stopCameraStream(cameraId);
          this.addSystemMessage(`${camera.name} stopped`);
        }
      }
    },

    configureCamera(cameraId) {
      const camera = this.cameras.find(cam => cam.id === cameraId);
      if (camera) {
        // Show camera configuration dialog
        const resolution = prompt(`Configure ${camera.name}:\nEnter resolution (e.g., 1920x1080):`, '1920x1080');
        if (resolution) {
          camera.config = { resolution };
          this.addSystemMessage(`${camera.name} configured with ${resolution}`);
        }
      }
    },

    // External device control methods
    toggleDevice(deviceId) {
      const device = this.externalDevices.find(dev => dev.id === deviceId);
      if (device) {
        device.connected = !device.connected;
        if (device.connected) {
          this.connectDevice(deviceId);
          this.addSystemMessage(`${device.name} connected`);
        } else {
          this.disconnectDevice(deviceId);
          this.addSystemMessage(`${device.name} disconnected`);
        }
      }
    },

    configureDevice(deviceId) {
      const device = this.externalDevices.find(dev => dev.id === deviceId);
      if (device) {
        // Show device configuration dialog
        const config = prompt(`Configure ${device.name}:\nEnter configuration parameters:`, 'default');
        if (config) {
          device.config = { parameters: config };
          this.addSystemMessage(`${device.name} configured`);
        }
      }
    },

    // Camera stream management
    async startCameraStream(cameraId) {
      try {
        const camera = this.cameras.find(cam => cam.id === cameraId);
        if (camera) {
          // Simulate camera stream initialization
          camera.stream = `stream_${cameraId}_${Date.now()}`;
          camera.status = 'active';
          
          // Start sensor data updates for this camera
          this.startSensorDataUpdates();
        }
      } catch (error) {
        errorHandler.handleError(error, `Failed to start camera stream: ${cameraId}`);
      }
    },

    stopCameraStream(cameraId) {
      const camera = this.cameras.find(cam => cam.id === cameraId);
      if (camera) {
        camera.stream = null;
        camera.status = 'available';
        
        // Stop sensor data updates if all cameras are off
        if (this.cameras.every(cam => !cam.active)) {
          this.stopSensorDataUpdates();
        }
      }
    },

    // Device connection management
    async connectDevice(deviceId) {
      try {
        const device = this.externalDevices.find(dev => dev.id === deviceId);
        if (device) {
          // Simulate device connection
          device.status = 'connected';
          
          // Start sensor data collection for sensors
          if (device.type === 'sensor') {
            this.startSensorDataUpdates();
          }
        }
      } catch (error) {
        errorHandler.handleError(error, `Failed to connect device: ${deviceId}`);
      }
    },

    disconnectDevice(deviceId) {
      const device = this.externalDevices.find(dev => dev.id === deviceId);
      if (device) {
        device.status = 'available';
        
        // Stop sensor data collection if no sensors are connected
        if (this.externalDevices.every(dev => dev.type !== 'sensor' || !dev.connected)) {
          this.stopSensorDataUpdates();
        }
      }
    },

    // Sensor data management
    startSensorDataUpdates() {
      // Start periodic sensor data updates
      if (!this.sensorUpdateInterval) {
        this.sensorUpdateInterval = setInterval(() => {
          this.updateSensorData();
        }, 2000); // Update every 2 seconds
      }
    },

    stopSensorDataUpdates() {
      // Stop sensor data updates
      if (this.sensorUpdateInterval) {
        clearInterval(this.sensorUpdateInterval);
        this.sensorUpdateInterval = null;
      }
    },

    updateSensorData() {
      // Simulate sensor data updates
      this.sensorData = {
        temperature: this.generateRandomValue(20, 30, 1),
        motion: Math.random() > 0.8,
        humidity: this.generateRandomValue(40, 80, 1),
        pressure: this.generateRandomValue(980, 1020, 1)
      };
    },

    generateRandomValue(min, max, precision = 0) {
      const value = Math.random() * (max - min) + min;
      return precision === 0 ? Math.round(value) : Number(value.toFixed(precision));
    },

    // Set up real-time input listeners
    setupRealTimeInputListeners() {
      errorHandler.logInfo('Setting up real-time input event listeners');
      // Event listeners are already bound via @ in the template, additional initialization logic can be added here
    },
    
    async startVoiceInput() {
      this.isVoiceInputActive = !this.isVoiceInputActive;
      if (this.isVoiceInputActive) {
        try {
          // Check if speech recognition is initialized
          if (this.recognition) {
            // Start speech recognition
            this.recognition.start();
            errorHandler.logInfo('Starting speech recognition...');
            
            // Add system message for speech recognition start
            this.addSystemMessage('Speech recognition started');
          } else {
            // If speech recognition is not available, notify user
            errorHandler.logWarning('Speech recognition feature is not available');
            this.addSystemMessage('Speech recognition is not available');
            this.isVoiceInputActive = false;
          }
        } catch (error) {
          errorHandler.handleError(error, 'Failed to start speech recognition');
          this.isVoiceInputActive = false;
          // Add error message
          this.addSystemMessage(`Speech recognition failed: ${error.message || error}`);
        }
      } else {
        // If recognizing, stop speech recognition
        if (this.recognition && this.recognition.stop) {
          this.recognition.stop();
        }
      }
    },
    
    selectImage() {
      const imageInput = this.$refs.imageInput;
      if (imageInput) {
        imageInput.click();
      } else {
          errorHandler.handleError(new Error('Image input element not found'), 'UI Element Error');
          this.addSystemMessage('Image input element not found');
        }
    },

    selectVideo() {
      const videoInput = this.$refs.videoInput;
      if (videoInput) {
        videoInput.click();
      } else {
          errorHandler.handleError(new Error('Video input element not found'), 'UI Element Error');
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
          errorHandler.logInfo(`Uploading video: ${file.name}`);
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
          errorHandler.handleError(error, 'Failed to upload video');
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
          errorHandler.logInfo(`Uploading image: ${file.name}`);
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
          errorHandler.handleError(error, 'Failed to upload image');
          // Add error message
          this.addSystemMessage(`Failed to upload image: ${error.message || error}`);
          // Clear file input
          event.target.value = '';
        }
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
  max-width: 1200px;
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
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.header-button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
}

.header-button:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.model-status {
    position: relative;
  }

  .management-model-status {
    margin-top: 20px;
  }

  .model-card.main {
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
    border-color: var(--border-dark);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
  }

  .model-card.main:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
  }

  .model-status-indicator.connected {
    background-color: var(--text-primary);
    animation: pulse 2s infinite;
  }

  .model-status-indicator.connecting {
    background-color: var(--text-secondary);
    animation: pulse 1s infinite;
  }

  .model-status-indicator.disconnected {
    background-color: var(--text-tertiary);
  }

  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 rgba(100, 100, 100, 0.7);
    }
    70% {
      box-shadow: 0 0 0 10px rgba(100, 100, 100, 0);
    }
    100% {
      box-shadow: 0 0 0 0 rgba(100, 100, 100, 0);
    }
  }

  .model-card {
    margin-bottom: var(--spacing-lg);
  }

.model-status h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: var(--spacing-md);
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  background: var(--bg-primary);
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.model-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-dark);
  transform: translateY(-2px);
}

.model-name {
  font-weight: 600;
  margin-bottom: var(--spacing-sm);
  color: var(--text-primary);
}

.model-status-indicator {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  margin-bottom: var(--spacing-sm);
}

.model-status-indicator.active {
  background-color: var(--text-primary);
}

.model-status-indicator.inactive {
  background-color: var(--text-tertiary);
}

.model-status-indicator.error {
  background-color: var(--text-secondary);
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

.message-content {
  margin-bottom: 8px;
  color: var(--text-primary);
  font-size: 15px;
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
  </style>
