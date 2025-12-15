<template>
  <div class="home-view">
    <!-- User Guide Component -->
    <UserGuide v-if="showUserGuide" @close="showUserGuide = false" />
    
    <!-- Multi-camera and Device Status Section -->
    <div class="device-status-section">
      <h3>Multi-camera & Device Management</h3>
      
      <!-- Regular Cameras -->
      <div class="device-category">
        <h4>Cameras</h4>
        <div class="device-grid">
          <div class="device-card" v-for="camera in cameras" :key="camera.id" :data-camera-id="camera.id">
            <div class="device-header">
              <span class="device-name">{{ camera.name }}</span>
              <span class="device-status" :class="camera.status"></span>
            </div>
            <div class="device-info">
              <span class="device-type">{{ camera.type || 'Standard' }}</span>
              <span class="device-resolution">{{ camera.resolution || 'Unknown' }}</span>
            </div>
            <div class="device-controls">
              <button @click="toggleCamera(camera.id)" class="device-btn">
                {{ camera.active ? 'Stop' : 'Start' }}
              </button>
              <button @click="configureCamera(camera.id)" class="device-btn">
                Settings
              </button>
              <button v-if="camera.calibrated" @click="toggleCameraStream(camera.id)" class="device-btn">
                Stream
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Stereo Vision Pairs -->
      <div class="device-category" v-if="stereoVisionPairs.length > 0">
        <h4>Stereo Vision Pairs</h4>
        <div class="device-grid">
          <div class="device-card stereo-pair" v-for="pair in stereoVisionPairs" :key="pair.id">
            <div class="device-header">
              <span class="device-name">{{ pair.name }}</span>
              <span class="device-status" :class="pair.status"></span>
            </div>
            <div class="device-info">
              <span class="stereo-cameras">{{ pair.leftCameraName }} & {{ pair.rightCameraName }}</span>
              <span class="calibration-status" :class="pair.calibrated ? 'calibrated' : 'uncalibrated'">
                {{ pair.calibrated ? 'Calibrated' : 'Not Calibrated' }}
              </span>
            </div>
            <div class="device-controls">
              <button @click="toggleStereoVision(pair.id)" class="device-btn">
                {{ pair.active ? 'Disable' : 'Enable' }}
              </button>
              <button @click="calibrateStereoPair(pair.id)" class="device-btn">
                Calibrate
              </button>
              <button @click="view3DModel(pair.id)" class="device-btn">
                3D View
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- External Devices -->
      <div class="device-category">
        <h4>External Devices</h4>
        <div class="device-grid">
          <div class="device-card" v-for="device in externalDevices" :key="device.id">
            <div class="device-header">
              <span class="device-name">{{ device.name }}</span>
              <span class="device-status" :class="device.status"></span>
            </div>
            <div class="device-info">
              <span class="device-type">{{ device.type || 'Unknown' }}</span>
              <span class="device-protocol">{{ device.protocol || 'Unknown' }}</span>
            </div>
            <div class="device-controls">
              <button @click="toggleDevice(device.id)" class="device-btn">
                {{ device.connected ? 'Disconnect' : 'Connect' }}
              </button>
              <button @click="configureDevice(device.id)" class="device-btn">
                Config
              </button>
              <button v-if="device.type === 'robotic_arm'" @click="controlRoboticArm(device.id)" class="device-btn">
                Control Arm
              </button>
              <button v-if="device.type === 'led_controller'" @click="controlLED(device.id)" class="device-btn">
                Control LED
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Sensor Data Display -->
      <div class="device-category" v-if="Object.keys(sensorData).length > 0">
        <h4>Sensor Data</h4>
        <div class="sensor-grid">
          <div v-for="(sensors, deviceId) in sensorData" :key="deviceId">
            <div class="sensor-device-header">
              <h5>Device ID: {{ deviceId }}</h5>
            </div>
            <div class="sensor-data">
              <div class="sensor-item" v-for="(value, sensorType) in sensors" :key="sensorType">
                <span class="sensor-label">{{ formatSensorType(sensorType) }}:</span>
                <span class="sensor-value">{{ value }}</span>
                <span class="sensor-timestamp">{{ getSensorTimestamp(sensorType, deviceId) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Serial Communication -->
      <div class="device-category">
        <h4>Serial Communication</h4>
        <div class="serial-communication">
          <!-- Serial Port Connection -->
          <div class="serial-connection">
            <div class="connection-controls">
              <select v-model="selectedSerialPort" @change="handlePortChange">
                <option value="">Select Port</option>
                <option v-for="port in availableSerialPorts" :key="port.port" :value="port.port">
                  {{ port.port }} - {{ port.name }}
                </option>
              </select>
              <select v-model="serialBaudRate">
                <option value="9600">9600</option>
                <option value="19200">19200</option>
                <option value="38400">38400</option>
                <option value="57600">57600</option>
                <option value="115200">115200</option>
              </select>
              <button @click="refreshSerialPorts" class="device-btn">
                Refresh Ports
              </button>
              <button @click="connectSerialPort" :disabled="!selectedSerialPort || serialConnected" class="device-btn">
                Connect
              </button>
              <button @click="disconnectSerialPort" :disabled="!serialConnected" class="device-btn">
                Disconnect
              </button>
              <span class="connection-status" :class="serialConnected ? 'connected' : 'disconnected'">{{ serialConnected ? 'Connected' : 'Disconnected' }}</span>
            </div>
          </div>

          <!-- Data Send/Receive -->
          <div class="serial-data-section">
            <!-- Send Data -->
            <div class="serial-send">
              <h5>Send Data</h5>
              <textarea v-model="serialSendData" placeholder="Enter data to send..."></textarea>
              <div class="send-controls">
                <label>
                  <input type="checkbox" v-model="sendAsHex"> Send as HEX
                </label>
                <label>
                  <input type="checkbox" v-model="appendCR"> Append CR
                </label>
                <label>
                  <input type="checkbox" v-model="appendLF"> Append LF
                </label>
                <button @click="sendSerialData" :disabled="!serialConnected" class="device-btn">
                  Send
                </button>
              </div>
            </div>

            <!-- Receive Data -->
            <div class="serial-receive">
              <h5>Received Data</h5>
              <div class="receive-controls">
                <button @click="clearReceivedData" class="device-btn">
                  Clear
                </button>
                <label>
                  <input type="checkbox" v-model="autoScroll"> Auto-scroll
                </label>
              </div>
              <div ref="serialReceiveArea" class="receive-data-area">
                <pre>{{ serialReceivedData }}</pre>
              </div>
            </div>
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
      deviceConnectionStatus: {}
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
    
    // Initialize device control WebSocket connection
    this.initializeDeviceControl();
    
    // Load real device data from backend
    this.loadDeviceData();
    
    // Initialize serial communication
    this.refreshSerialPorts();
  },
  beforeUnmount() {
    // Remove event listeners when component is unmounted
    window.removeEventListener('voice-input', this.handleVoiceInputEvent);
    
    // Clean up device control WebSocket connection
    this.disconnectDeviceControlWebSocket();
    
    // Clean up serial communication
    this.disconnectSerialPort();
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
      // Initialize device control system
      initializeDeviceControl() {
        // Start real WebSocket connection
        this.connectDeviceControlWebSocket();
      },
      
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
          const response = await api.health.get();
          
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
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: response.data.data?.response || response.data.response || 'I don\'t have a response yet. Please try again.',
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

    // Device control WebSocket management
    connectDeviceControlWebSocket() {
      // Clear any existing intervals
      this.disconnectDeviceControlWebSocket();
      
      try {
        // Create real WebSocket connection
        const wsUrl = `ws://localhost:8766`;
        this.deviceControlWebSocket = new WebSocket(wsUrl);
        
        this.deviceControlWebSocket.onopen = () => {
          console.log('Device control WebSocket connection established');
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
          console.log('Device control WebSocket connection closed');
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
          this.addSystemMessage('Device control system: Connection error');
        };
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        this.addSystemMessage('Device control system: Failed to connect');
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
          console.log('Attempting to reconnect to device control WebSocket...');
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

    updateSensorDataFromServer(deviceId, data) {
      if (deviceId === 'sensor1') {
        this.sensorData.temperature = data.value;
      } else if (deviceId === 'sensor2') {
        this.sensorData.motion = data.value;
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
          console.log('Received unknown WebSocket message type:', message.type);
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
                this.sensorData[sensorType] = data.value;
                break;
              case 'motion':
                this.sensorData.motion = data.value;
                break;
              case 'accelerometer':
              case 'gyroscope':
              case 'magnetometer':
                // Handle 3-axis sensors
                this.sensorData[sensorType] = {
                  x: data.x || 0,
                  y: data.y || 0,
                  z: data.z || 0
                };
                break;
              case 'depth':
                // Handle depth maps for stereo vision
                if (data.depthMap) {
                  this.sensorData.depthMap = data.depthMap;
                }
                if (data.pointCloud) {
                  this.sensorData.pointCloud = data.pointCloud;
                }
                break;
              case 'custom':
                // Handle custom sensor data
                this.sensorData.custom[deviceId] = data;
                break;
            }
            break;
          }
        }
        
        // Log sensor data update
        errorHandler.logInfo(`Updated sensor data from ${deviceId}`, data);
      } catch (error) {
        errorHandler.handleError(error, 'Failed to update sensor data');
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
          errorHandler.handleError(error, 'Failed to calibrate stereo pair');
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
          errorHandler.handleError(error, 'Failed to toggle stereo vision');
          this.addSystemMessage(`Failed to ${isEnabled ? 'enable' : 'disable'} stereo vision: ${error.message || error}`);
        }
      }
    },
    
    // Get 3D position from stereo vision
    get3DPosition(leftPixel, rightPixel) {
      // Simple implementation - in a real system this would use proper stereo triangulation
      // based on calibration data
      try {
        if (!this.stereoCalibrationData.baseline || !this.stereoCalibrationData.focalLength) {
          throw new Error('Stereo vision not calibrated');
        }
        
        const disparity = Math.abs(leftPixel.x - rightPixel.x);
        if (disparity === 0) return null;
        
        // Calculate depth using triangulation formula
        const depth = (this.stereoCalibrationData.baseline * this.stereoCalibrationData.focalLength) / disparity;
        
        // Calculate 3D coordinates
        const x = (leftPixel.x - this.stereoCalibrationData.principalPoint.x) * depth / this.stereoCalibrationData.focalLength;
        const y = (leftPixel.y - this.stereoCalibrationData.principalPoint.y) * depth / this.stereoCalibrationData.focalLength;
        
        return { x, y, z: depth };
      } catch (error) {
        errorHandler.handleError(error, 'Failed to calculate 3D position');
        return null;
      }
    },

    // Set up real-time input listeners
    setupRealTimeInputListeners() {
      errorHandler.logInfo('Setting up real-time input event listeners');
      // Event listeners are already bound via @ in the template, additional initialization logic can be added here
    },
    
    async startVoiceRecognition() {
      this.isVoiceInputActive = !this.isVoiceInputActive;
      if (this.isVoiceInputActive) {
        try {
          // Check if speech recognition is initialized
          if (this.recognition) {
            // Configure speech recognition
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';
            
            this.recognition.onresult = (event) => {
              const transcript = event.results[0][0].transcript;
              if (transcript.trim()) {
                this.inputText = transcript;
                this.addSystemMessage(`Voice input received: ${transcript}`);
                // Auto send message after a short delay
                setTimeout(() => {
                  this.sendMessage();
                }, 500);
              }
            };
            
            this.recognition.onerror = (event) => {
              errorHandler.handleError(new Error(event.error), 'Speech recognition error');
              this.addSystemMessage(`Speech recognition error: ${event.error}`);
              this.isVoiceInputActive = false;
            };
            
            this.recognition.onend = () => {
              this.isVoiceInputActive = false;
            };
            
            // Start speech recognition
            this.recognition.start();
            errorHandler.logInfo('Starting speech recognition...');
            
            // Add system message for speech recognition start
            this.addSystemMessage('Speech recognition started - speak now');
          } else {
            // If speech recognition is not available, notify user
            errorHandler.handleWarning('Speech recognition feature is not available');
            this.addSystemMessage('Speech recognition is not available in this browser');
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
        this.addSystemMessage('Speech recognition stopped');
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
        errorHandler.logInfo('Loading device data from backend...');
        
        // Load cameras data with enhanced properties
        const camerasResponse = await api.cameras.getList();
        if (camerasResponse.data.status === 'success') {
          this.cameras = camerasResponse.data.data.map(camera => ({
            ...camera,
            resolution: camera.resolution || '1280x720',
            fps: camera.fps || 30,
            exposure: camera.exposure || 'auto',
            gain: camera.gain || 0,
            isStereo: camera.isStereo || false,
            stereoRole: camera.stereoRole || null, // 'left', 'right', or null
            stereoPairId: camera.stereoPairId || null
          })) || [];
          errorHandler.logInfo(`Loaded ${this.cameras.length} cameras`);
        } else {
          throw new Error('Failed to load cameras data');
        }
        
        // Load external devices data with enhanced protocol support
        const devicesResponse = await api.get('/api/devices/external');
        if (devicesResponse.data.status === 'success') {
          const devicesData = devicesResponse.data.data;
          // Convert object to array if needed (backend returns dict, frontend expects array)
          const devicesArray = Array.isArray(devicesData) ? devicesData : Object.values(devicesData);
          // Extract device_info from each device entry (backend returns nested structure)
          const flattenedDevices = devicesArray.map(device => 
            device.device_info ? device.device_info : device
          );
          this.externalDevices = flattenedDevices.map(device => ({
            ...device,
            protocol: device.protocol || 'WebSocket',
            baudRate: device.baudRate || null,
            port: device.port || null,
            address: device.address || null,
            timeout: device.timeout || 5000,
            maxRetries: device.maxRetries || 3,
            lastCommandTime: null,
            lastResponseTime: null
          })) || [];
          errorHandler.logInfo(`Loaded ${this.externalDevices.length} external devices`);
        } else {
          throw new Error('Failed to load external devices data');
        }
        
        // Load stereo vision pairs
        try {
          const stereoPairsResponse = await api.cameras.getStereoPairs();
          if (stereoPairsResponse.data.status === 'success') {
            this.stereoVisionPairs = stereoPairsResponse.data.data || [];
            errorHandler.logInfo(`Loaded ${this.stereoVisionPairs.length} stereo vision pairs`);
          }
        } catch (error) {
          errorHandler.handleWarning('Failed to load stereo pairs from primary API, falling back to secondary endpoint');
          // Fallback to original endpoint for backward compatibility
          try {
            const stereoPairsResponse = await api.get('/api/devices/stereo-pairs');
            if (stereoPairsResponse.data.status === 'success') {
              this.stereoVisionPairs = stereoPairsResponse.data.data || [];
              errorHandler.logInfo(`Loaded ${this.stereoVisionPairs.length} stereo vision pairs`);
            }
          } catch (fallbackError) {
            errorHandler.handleWarning('Failed to load stereo pairs from fallback endpoint');
          }
        }
        
        // Load stereo calibration data - endpoint not implemented in backend
        // try {
        //   const calibrationResponse = await api.get('/api/devices/stereo-calibration');
        //   if (calibrationResponse.data.status === 'success') {
        //     this.stereoCalibrationData = calibrationResponse.data.data || {};
        //     errorHandler.logInfo('Loaded stereo vision calibration data');
        //   }
        // } catch (error) {
        //   errorHandler.handleWarning('Failed to load stereo calibration data');
        // }
        errorHandler.logInfo('Skipping stereo calibration data load - endpoint not implemented');
        
        this.addSystemMessage(`Device data loaded: ${this.cameras.length} cameras, ${this.externalDevices.length} external devices, ${this.stereoVisionPairs.length} stereo pairs`);
        
      } catch (error) {
        errorHandler.handleError(error, 'Failed to load device data');
        this.addSystemMessage('Failed to load device data from backend. Please ensure the backend service is running.');
        
        // Initialize with empty arrays
        this.cameras = [];
        this.externalDevices = [];
        this.stereoVisionPairs = [];
        this.stereoCalibrationData = {};
      }
    },
    
    // Process user input based on type
    async processUserInput(inputData, inputType) {
      try {
        errorHandler.logInfo(`Processing ${inputType} input`);
        
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
        errorHandler.handleError(error, `Failed to process ${inputType} input`);
        return `${inputType.charAt(0).toUpperCase() + inputType.slice(1)} processing failed due to connection issues`;
      }
    },
    
    // Format sensor type for display
    formatSensorType(sensorType) {
      return sensorType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    },
    
    // Get sensor data timestamp from real data
    getSensorTimestamp(sensorType, deviceId) {
      // This should be replaced with actual timestamp from sensor data
      const now = new Date();
      return now.toLocaleTimeString();
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
        
        errorHandler.logInfo(`${camera.name} stream toggled, new state: ${camera.isStreaming ? 'streaming' : 'stopped'}`);
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
        errorHandler.handleError(error, 'Failed to start camera stream');
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
        errorHandler.handleError(error, 'Failed to stop camera stream');
        this.addSystemMessage(`Failed to stop camera stream: ${error.message || error}`);
      }
    },
    
    // Create camera preview element
    createCameraPreview(cameraId, streamUrl) {
      errorHandler.logInfo(`Creating preview for camera ${cameraId} with URL: ${streamUrl}`);
      
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
          errorHandler.logInfo(`WebSocket connection established for camera ${cameraId}`);
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
          errorHandler.handleError(error, `Camera ${cameraId} WebSocket error`);
          this.addSystemMessage(`Camera ${cameraId} stream error`);
          if (loadingIndicator) {
            loadingIndicator.textContent = 'Stream error';
            loadingIndicator.className = 'loading-indicator error';
          }
        };
        
        ws.onclose = () => {
          errorHandler.logInfo(`WebSocket connection closed for camera ${cameraId}`);
          if (camera) {
            camera.websocket = null;
          }
          if (loadingIndicator && loadingIndicator.parentNode) {
            loadingIndicator.textContent = 'Stream closed';
          }
        };
        
      } catch (error) {
        errorHandler.handleError(error, `Failed to setup WebSocket for camera ${cameraId}`);
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
      errorHandler.logInfo(`Removing preview for camera ${cameraId}`);
      
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
        errorHandler.logInfo(`Viewing 3D model for ${pair.name}`);
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
        errorHandler.handleError(error, 'Failed to refresh serial ports');
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
        errorHandler.handleError(error, `Failed to connect to ${this.selectedSerialPort}`);
        this.addSystemMessage(`Failed to connect to ${this.selectedSerialPort}: ${error.message}`);
        this.serialConnected = false;
      }
    },

    async disconnectSerialPort() {
      try {
        this.stopSerialListener();
        await api.serial.disconnect();
        this.serialConnected = false;
        this.addSystemMessage(`Disconnected from ${this.selectedSerialPort}`);
      } catch (error) {
        errorHandler.handleError(error, 'Failed to disconnect serial port');
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
        errorHandler.handleError(error, 'Failed to send serial data');
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
  </style>
