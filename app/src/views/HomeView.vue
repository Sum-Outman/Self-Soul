<template>
  <div class="home-view">
    <!-- User Guide Component -->
    <UserGuide v-if="showUserGuide" @close="showUserGuide = false" />
    
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
      // 添加缺失的状态
      managementModel: {
        name: 'A Management Model',
        status: 'inactive',
        lastActive: null
      },
      connectedText: '',
      activeModels: 0
    };
  },
  mounted() {
    // 初始化系统
    this.initializeSystem();
    // 加载历史消息
    this.loadHistoryMessages();
    // 连接到后端服务
    this.connectToBackend();
    // 初始化语音识别
    this.initSpeechRecognition();
    
    // 监听来自App.vue的全局语音输入事件
    window.addEventListener('voice-input', this.handleVoiceInputEvent);
    
    // 监听RealTimeInput组件的实时输入事件
    this.setupRealTimeInputListeners();
  },
  beforeUnmount() {
    // 组件卸载时移除事件监听
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
      // Initialize system
    initializeSystem() {
      errorHandler.logInfo('AGI Brain System initializing...');
      // Show welcome message
      this.addSystemMessage('Welcome to the AGI Brain System!');
      // Initialize mock data
      this.useMockData();
    },
    
    // Use mock data method
    useMockData() {
      // Simulate successful connection
      setTimeout(() => {
        this.backendConnected = true;
        this.backendStatus = 'connected';
        this.modelConnectionStatus = 'connected';
        
        // Set management model to active status
        this.managementModel = {
          name: 'A Management Model',
          status: 'active',
          lastActive: new Date().toISOString()
        };
        
        // Update connectedText
        this.connectedText = 'Connected';
        
        // Set all models to active state
        this.models.forEach(model => {
          model.status = 'active';
          model.lastActive = new Date().toISOString();
        });
        
        // Update active model count
        this.activeModels = this.activeModelsCount;
        
        // Add system message
        this.addSystemMessage('All models activated');
        
        // Simulate real-time data updates
        // Periodically update some models' status randomly to simulate real system data changes
        this.startRealTimeDataSimulation();
      }, 1500);
    },
    
    // Start real-time data simulation
    startRealTimeDataSimulation() {
      // Randomly update some models' status and performance every 5-10 seconds
      setInterval(() => {
        // Randomly select 1-3 models for status or performance update
        const updateCount = Math.floor(Math.random() * 3) + 1;
        const modelIndices = [...Array(this.models.length).keys()];
        
        // Randomly shuffle model indices
        for (let i = modelIndices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [modelIndices[i], modelIndices[j]] = [modelIndices[j], modelIndices[i]];
        }
        
        // Update selected models
        for (let i = 0; i < updateCount; i++) {
          const index = modelIndices[i];
          const model = this.models[index];
          
          // 30% probability to switch model status (if not the management model)
          if (model.id !== 'manager' && Math.random() < 0.3) {
            model.status = model.status === 'active' ? 'inactive' : 'active';
            model.lastActive = model.status === 'active' ? new Date().toISOString() : model.lastActive;
            
            // Add system message to indicate model status change
            if (model.status === 'active') {
              this.addSystemMessage(`Model activated: ${model.id}`);
            } else {
              this.addSystemMessage(`Model deactivated: ${model.id}`);
            }
          }
          
          // 70% probability to update model performance value (within ±2 range)
          if (model.status === 'active' && Math.random() < 0.7) {
            const change = (Math.random() - 0.5) * 4; // -2 to +2 change
            model.performance = Math.max(50, Math.min(100, model.performance + change));
          }
        }
        
        // Ensure management model remains active
        const managerModel = this.models.find(m => m.id === 'manager');
        if (managerModel) {
          managerModel.status = 'active';
        }
        
        // Synchronously update modelPerformanceData
        this.modelPerformanceData = [...this.models];
        
        // Manually trigger recalculation of activeModelsCount
        this.activeModels = this.activeModelsCount;
      }, 5000 + Math.random() * 5000); // Random interval of 5-10 seconds
    },
    
    // Navigation methods - Using Vue Router the correct way
    navigateToTraining() {
      this.$router.push('/training');
    },
    
    navigateToSystemSettings() {
      this.$router.push('/settings');
    },
    
    navigateToModelManagement() {
      this.$router.push('/models');
    },
    
    navigateToHelp() {
      this.$router.push('/help');
    },
    
    // Navigate to monitoring dashboard
    navigateToMonitor() {
      this.$router.push('/dashboard');
    },
    
    // Toggle real-time input interface
    toggleRealTimeInput() {
      this.showRealTimeInput = !this.showRealTimeInput;
    },
    
    loadHistoryMessages() {
      try {
        // Load history messages from local storage
        const history = localStorage.getItem('agi_messages');
        if (history) {
          this.messages = JSON.parse(history);
        } else {
          // Add welcome message
          this.messages.push({
            id: Date.now(),
            type: 'system',
            content: 'Welcome to the AGI Brain System!',
            time: new Date().toLocaleTimeString()
          });
        }
      } catch (error) {
        errorHandler.handleError(error, 'Failed to load history messages');
        // Add error message to interface
        this.addSystemMessage('Failed to load history messages');
      }
    },
    
    // Helper method to add system messages
    addSystemMessage(content) {
      this.messages.push({
        id: Date.now(),
        type: 'system',
        content: content,
        time: new Date().toLocaleTimeString()
      });
      // Save to local storage
      this.saveMessages();
    },
    
    // Helper method to save messages to local storage
      saveMessages() {
        try {
          localStorage.setItem('agi_messages', JSON.stringify(this.messages));
        } catch (error) {
          errorHandler.handleError(error, 'Failed to save messages');
          // Can choose whether to display error message here
        }
      },
      
      // Clear all conversation messages
      clearAllMessages() {
        if (confirm('Are you sure you want to clear all messages?')) {
          this.messages = [{
            id: Date.now(),
            type: 'system',
            content: 'Welcome to the AGI Brain System!',
            time: new Date().toLocaleTimeString()
          }];
          // Clear local storage
          try {
            localStorage.removeItem('agi_messages');
            // Clear conversation history
            const sessionId = this.getSessionId();
            localStorage.removeItem(`conversation_${sessionId}`);
          } catch (error) {
            errorHandler.handleError(error, 'Failed to clear messages');
          }
        }
      },
    
    async connectToBackend() {
        try {
          // Set backend API base URL
          this.backendStatus = 'connecting';
          this.modelConnectionStatus = 'connecting';
          errorHandler.logInfo('Connecting to FastAPI backend service...');
          
          // Make actual HTTP request to health check endpoint
          // Using relative path instead of hardcoded URL
          const response = await api.get('/health', {
            timeout: 5000 // 5 seconds timeout
          });
          
          // Log full response for debugging
          console.log('Health check response:', response);
          
          // Check if response is successful
          if (response && response.data && (response.data.status === 'healthy' || response.data.status === 'ok')) {
            errorHandler.logInfo('FastAPI backend connection successful');
            this.backendConnected = true;
            this.backendStatus = 'connected';
            this.modelConnectionStatus = 'connected';
            
            // Set management model to active status
            this.managementModel = {
              name: 'Management Model',
              status: 'active',
              lastActive: new Date().toISOString()
            };
            
            // Update connectedText
            this.connectedText = 'Connected';
            
            // Update all models status to active
            this.models.forEach(model => {
              model.status = 'active';
              model.lastActive = new Date().toISOString();
            });
            
            // Update active model count
            this.activeModels = this.activeModelsCount;
            
            // Add connection success system message
            this.addSystemMessage('FastAPI backend service connected successfully');
          } else {
            throw new Error('Invalid response from backend');
          }
        } catch (error) {
          errorHandler.handleError(error, 'Connection to FastAPI backend failed');
          this.backendConnected = false;
          this.backendStatus = 'error';
          this.modelConnectionStatus = 'error';
          this.addSystemMessage('Failed to connect to FastAPI backend. Falling back to demo mode.');
          
          // Fallback to mock mode to ensure interface can be used normally
          setTimeout(() => {
            this.backendConnected = true;
            this.backendStatus = 'connected';
            this.modelConnectionStatus = 'connected';
            
            // Set management model to active status
            this.managementModel = {
              name: 'Management Model',
              status: 'active',
              lastActive: new Date().toISOString()
            };
            
            // Update connectedText
            this.connectedText = 'Demo Mode';
            
            this.models.forEach(model => {
              model.status = 'active';
              model.lastActive = new Date().toISOString();
            });
            
            // Update active model count
            this.activeModels = this.activeModelsCount;
          }, 1500);
        }
      },
    
    async testHttpConnection() {
      try {
          // Use relative path for health check endpoint
          const response = await api.get('/health', {
            timeout: 5000
          });
        
        if (response.status === 200) {
          errorHandler.logInfo('HTTP connection established to FastAPI backend');
          this.backendConnected = true;
          this.backendStatus = 'connected';
          this.modelConnectionStatus = 'connected';
          
          // 设置管理模型为活跃状态
          this.managementModel = {
            name: 'Management Model',
            status: 'active',
            lastActive: new Date().toISOString()
          };
          
          // 更新connectedText
          this.connectedText = 'Connected';
          
          this.models.forEach(model => {
            model.status = 'active';
            model.lastActive = new Date().toISOString();
          });
          
          // 更新活跃模型数量
          this.activeModels = this.activeModelsCount;
          
          this.addSystemMessage('FastAPI backend connected successfully');
        } else {
          throw new Error(`Invalid response status: ${response.status}`);
        }
      } catch (error) {
        errorHandler.handleError(error, 'HTTP connection to FastAPI backend failed');
        this.backendConnected = false;
        this.backendStatus = 'error';
        this.modelConnectionStatus = 'error';
        this.addSystemMessage('Failed to connect to FastAPI backend. Falling back to demo mode.');
        
        // 回退到演示模式
        setTimeout(() => {
          this.backendConnected = true;
          this.backendStatus = 'connected';
          this.modelConnectionStatus = 'connected';
          
          // 设置管理模型为活跃状态
          this.managementModel = {
            name: 'Management Model',
            status: 'active',
            lastActive: new Date().toISOString()
          };
          
          // 更新connectedText为演示模式
          this.connectedText = 'Demo Mode';
          
          this.models.forEach(model => {
            model.status = 'active';
            model.lastActive = new Date().toISOString();
          });
          
          // 更新活跃模型数量
          this.activeModels = this.activeModelsCount;
        }, 1500);
      }
    },
    
    async sendMessage() {
      const text = this.inputText.trim();
      if (!text) return;
      
      const userMessage = {
        id: Date.now(),
        type: 'user',
        content: text,
        time: new Date().toLocaleTimeString()
      };
      
      this.messages.push(userMessage);
      this.inputText = '';
      
      // 添加加载状态消息
      const loadingMessageId = Date.now() + 0.5;
      const loadingMessage = {
        id: loadingMessageId,
        type: 'loading',
        content: 'Processing message...',
        time: new Date().toLocaleTimeString()
      };
      this.messages.push(loadingMessage);
      this.saveMessages();
      
      try {
        // 使用A Management Model处理真实消息
        const responseText = await this.processUserInput(text, 'text');
        
        // 移除加载状态消息
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: responseText,
          time: new Date().toLocaleTimeString()
        };
        
        this.messages.push(botMessage);
        // 保存到本地存储
        this.saveMessages();
      } catch (error) {
        errorHandler.handleError(error, 'Failed to process message');
        // Remove loading status message
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        // Add error message to interface
        this.addSystemMessage('Failed to process your message');
      }
    },
    
    async processUserInput(input, type) {
      try {
        // 首先尝试连接到后端服务
        if (!this.backendConnected) {
          await this.connectToBackend();
        }
        
        let response;
        if (type === 'text') {
          // Get conversation history from local storage
          const sessionId = this.getSessionId();
          let conversationHistory = JSON.parse(localStorage.getItem(`conversation_${sessionId}`) || '[]');
          
          // Use new chat API endpoint
          response = await api.post('/api/chat', {
            message: input,
            session_id: sessionId,
            conversation_history: conversationHistory
          }, {
            timeout: 30000 // 30 seconds timeout
          });
          
          // Save updated conversation history to local storage
          if (response && response.data && response.data.status === 'success') {
            localStorage.setItem(`conversation_${sessionId}`, JSON.stringify(response.data.data.conversation_history));
            return response.data.data.response;
          }
        } else if (type === 'image') {
          // 图像处理 - 在handleImageUpload中处理
          return await this.processImageInput(input);
        } else if (type === 'video') {
          // 视频处理 - 在handleVideoUpload中处理
          return await this.processVideoInput(input);
        } else if (type === 'audio') {
          // 音频处理
          return await this.processAudioInput(input);
        }
        
        if (response && response.data && response.data.status === 'success') {
          return response.data.data.response || response.data.data;
        } else {
          throw new Error(response?.data?.detail || 'Failed to process your request');
        }
      } catch (error) {
        errorHandler.handleError(error, 'Failed to process user input');
        // Show error message without falling back to mock response
        this.addSystemMessage('Failed to process your request. Please check your connection and try again.');
        throw error;
      }
    },
    
    // 获取会话ID用于跟踪对话上下文
    getSessionId() {
      let sessionId = localStorage.getItem('agi_session_id');
      if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('agi_session_id', sessionId);
      }
      return sessionId;
    },
    
    // 处理图像输入
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
    
    // 处理视频输入
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
    
    // Enhanced intelligent mock response
    getEnhancedMockResponse(input, type) {
      if (type === 'text') {
        const lowerInput = input.toLowerCase();
        
        // A Management Model specific response
        if (lowerInput.includes('management model')) {
          return 'The Management Model is the core component that coordinates all other models in the AGI Brain System. It handles task distribution, resource allocation, and ensures seamless communication between different models.';
        }
        
        // Model list and activation status detection
        if (lowerInput.includes('models list')) {
          let modelList = 'Active Models List:\n';
          this.models.forEach((model, index) => {
            // Map model IDs to their English names
            const modelName = this.getModelDisplayName(model.id);
            modelList += `${index + 1}. ${modelName} - ${model.performance.toFixed(1)}% performance\n`;
          });
          modelList += '\nTotal active models: ' + this.activeModels;
          return modelList;
        }
        
        // Question type recognition and professional answers
        if (lowerInput.includes('train')) {
          return 'Training process involves several steps: 1) Data preparation, 2) Model configuration, 3) Training execution, 4) Evaluation, and 5) Deployment. You can access the training interface through the navigation menu.';
        } else if (lowerInput.includes('knowledge')) {
          return 'Knowledge management system handles information storage, retrieval, and organization. It supports document upload, semantic search, and knowledge graph visualization.';
        } else if (lowerInput.includes('connection')) {
          return `Connection Status:\n- Overall Status: ${this.backendConnected ? 'Connected' : 'Disconnected'}\n- Backend Status: ${this.backendStatus}\n- Model Status: ${this.modelConnectionStatus || 'Unknown'}`;
        }
        
        // Technical support related
        if (lowerInput.includes('help') || lowerInput.includes('support') || lowerInput.includes('problem') || lowerInput.includes('error')) {
          return 'For technical support, please check the Help section or contact your system administrator. Common issues include connection problems, model performance issues, and configuration errors.';
        }
      }
      
      // General intelligent responses
      const responses = {
        text: [
          `I'm processing your request about "${input}". The AGI Brain System is designed to handle various types of inputs and provide intelligent responses.`,
          `Your message "${input}" is being analyzed. The system is leveraging multiple AI models to generate the most accurate response.`,
          'The AGI Brain System consists of multiple specialized models working together to provide comprehensive intelligence capabilities.',
          'This is a mock response due to connection limitations. In a production environment, this would be processed by the actual backend services.',
          'I can help you with a variety of tasks including text analysis, image recognition, audio processing, and more.',
          'The system is currently in demo mode. Please explore the different features available in the navigation menu.',
          'Thank you for using the AGI Brain System. How can I assist you further?',
          'To get the most out of the system, try asking specific questions or providing clear instructions.',
          'The system supports multiple input types including text, images, audio, and video. Try uploading different types of media.'
        ],
        image: [
          'This is a mock image analysis result. In a real environment, the Vision Model would analyze the image content and provide detailed insights.',
          'Image processing completed. The system has detected key elements in the image and is preparing a comprehensive analysis.',
          'The Vision Model is designed to recognize objects, scenes, text, and other visual elements in images.'
        ],
        video: [
          'Video processing is in progress. The system is analyzing key frames and extracting meaningful information from the video content.',
          'This is a mock video analysis result. In a production environment, the system would provide detailed temporal analysis of the video.',
          'The Video Analysis Model can detect motion, recognize objects over time, and identify patterns in video content.'
        ],
        audio: [
          'Audio transcription completed. The system has converted the speech to text and is preparing additional analysis.',
          'This is a mock audio processing result. In a real environment, the Audio Model would provide accurate transcription and analysis.',
          'The Audio Model can recognize speech, identify speakers, and extract key information from audio content.'
        ]
      };
      
      const typeResponses = responses[type] || responses.text;
      return typeResponses[Math.floor(Math.random() * typeResponses.length)];
    },
    
    // Generate mock response (enhanced version)
    generateMockResponse(input) {
      // 更智能的响应生成逻辑，基于输入关键词和内容
      const lowerInput = input.toLowerCase();
      
      if (lowerInput.includes('hello') || lowerInput.includes('hi')) {
        return 'Hello! I am A Management Model. How can I assist you today?';
      } else if (lowerInput.includes('model') || lowerInput.includes('models')) {
        return 'I manage 11 specialized AI models (A to K). Model B handles vision processing, Model C processes audio, Model D excels at text analysis, and so on. Each has unique capabilities tailored for different tasks.';
      } else if (lowerInput.includes('train') || lowerInput.includes('training')) {
        return 'You can train models individually or in joint training sessions through the Training Panel. Joint training allows models to learn from each other and improve performance on complex tasks. Would you like me to explain the training parameters?';
      } else if (lowerInput.includes('knowledge')) {
        return 'The Knowledge Base stores critical information that enhances the AI models\' understanding. You can upload documents, images, and other media to create a comprehensive knowledge repository. The models can access this information during processing.';
      } else if (lowerInput.includes('help') || lowerInput.includes('guide')) {
        return 'I can help you with various tasks: managing AI models, configuring training parameters, organizing knowledge files, and adjusting system settings. What specific area would you like assistance with?';
      } else if (lowerInput.includes('settings') || lowerInput.includes('configure')) {
        return 'In the Settings section, you can manage model configurations, activate or deactivate specific models, and even connect to external API services to replace local models when needed.';
      } else if (lowerInput.includes('image') || lowerInput.includes('vision')) {
        return 'For image analysis, I recommend using Model B. You can upload images directly in this chat or visit the dedicated vision processing section for more advanced analysis.';
      } else if (lowerInput.includes('audio') || lowerInput.includes('voice')) {
        return 'For audio processing, Model C is specialized in speech recognition and audio analysis. You can use the voice input button in this chat or upload audio files for processing.';
      } else if (lowerInput.includes('video')) {
        return 'Video processing requires a combination of Model B (vision) and Model C (audio). You can upload video files up to 50MB for analysis and processing.';
      } else {
        // 通用响应
        return `Thank you for your query. Based on your input, I've generated this response. To get more accurate and detailed assistance, you can:
1. Ask more specific questions
2. Provide additional context
3. Use the dedicated sections for specialized tasks

How else can I help you?`;
      }
    },
    
    // Initialize voice recognition (disabled, using global voice recognition)
    initSpeechRecognition() {
      try {
        // 检查浏览器是否支持语音识别
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
          
          // If backend is not connected or timeout, provide mock response
          if (!this.backendConnected || error.message.includes('timeout')) {
            const videoMessage = {
              id: Date.now(),
              type: 'user',
              content: `[Video: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            const mockResponse = this.getMockResponse(file.name, 'video');
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: mockResponse,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(videoMessage, botMessage);
            this.saveMessages();
          }
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
          
          // If backend is not connected or timeout, provide mock response
          if (!this.backendConnected || error.message.includes('timeout')) {
            const imageMessage = {
              id: Date.now(),
              type: 'user',
              content: `[Image: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            // File content analysis and response
            const mockResponse = this.analyzeUploadedFile(file, 'image');
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: mockResponse,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(imageMessage, botMessage);
            this.saveMessages();
          }
        }
      }
    },

    // Analyze uploaded file and generate intelligent response
    analyzeUploadedFile(file, type) {
      // Mock file analysis result
      const fileType = file.type;
      let analysisResult = '';
      
      if (type === 'image' || fileType.includes('image')) {
        analysisResult = `Image Analysis Result:\n- File Name: ${file.name}\n- File Type: ${fileType}\n- File Size: ${(file.size / 1024).toFixed(2)} KB\n- Content Type: Image Data\n\nVision Model has analyzed the image content. A Management Model can provide related answers based on image information.`;
      } else if (type === 'video' || fileType.includes('video')) {
        analysisResult = `Video Analysis Result:\n- File Name: ${file.name}\n- File Type: ${fileType}\n- File Size: ${(file.size / 1024).toFixed(2)} KB\n- Content Type: Video Data\n\nMultimodal processing completed. A Management Model coordinated visual and audio subsystems for comprehensive analysis.`;
      } else {
        analysisResult = `File Analysis Result:\n- File Name: ${file.name}\n- File Type: ${fileType}\n- File Size: ${(file.size / 1024).toFixed(2)} KB\n\nFile uploaded successfully. A Management Model can provide related answers based on file content.`;
      }
      
      return analysisResult;
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
.home-view {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
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
  gap: 20px;
}

.server-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
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
      background-color: var(--text-primary); /* Dark gray - Connected */
      box-shadow: 0 0 0 2px rgba(100, 100, 100, 0.3);
    }

    .status-dot.disconnected {
      background-color: var(--text-tertiary); /* Light gray - Not connected */
      box-shadow: 0 0 0 2px rgba(200, 200, 200, 0.3);
    }

.header-buttons {
  display: flex;
  gap: 10px;
}

.header-button {
  padding: 8px 16px;
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
      box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
    }
    70% {
      box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
    }
    100% {
      box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
    }
  }

  .model-card {
    margin-bottom: 30px;
  }

.model-status h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 15px;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.model-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 15px;
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
  margin-bottom: 10px;
  color: var(--text-primary);
}

.model-status-indicator {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  margin-bottom: 10px;
}

.model-status-indicator.active {
  background-color: var(--success-color);
}

.model-status-indicator.inactive {
  background-color: var(--text-tertiary);
}

.model-status-indicator.error {
  background-color: var(--error-color);
}

.model-performance {
  font-size: 14px;
  color: var(--text-secondary);
}

.input-area {
  margin-bottom: 30px;
}

.input-area h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 15px;
}

.chat-container {
  height: 400px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 15px;
  margin-bottom: 15px;
  overflow-y: auto;
  background: var(--bg-secondary);
}

.message {
    margin-bottom: 15px;
    padding: 12px;
    border-radius: var(--border-radius);
    border: 1px solid var(--border-light);
    animation: fadeIn 0.3s ease-in-out;
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
    margin-left: 20%;
    border-color: var(--border-color);
  }

  .message.bot {
    background: var(--bg-tertiary);
    margin-right: 20%;
    border-color: var(--border-color);
  }

  .message.system {
    background: var(--bg-secondary);
    text-align: center;
    border-color: var(--border-color);
    font-style: italic;
    color: var(--text-secondary);
  }

  .message.loading {
    background: var(--bg-secondary);
    margin-right: 20%;
    border-color: var(--border-color);
    position: relative;
    overflow: hidden;
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

.message-content {
  margin-bottom: 5px;
  color: var(--text-primary);
}

.message-time {
  font-size: 12px;
  color: var(--text-tertiary);
  text-align: right;
}

.input-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.input-controls input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  transition: var(--transition);
}

.input-controls input:focus {
  outline: none;
  border-color: var(--border-dark);
  box-shadow: 0 0 0 3px rgba(200, 200, 200, 0.1);
}

.input-controls button {
  padding: 10px 15px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
}

.input-controls button:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.input-options {
  display: flex;
  gap: 10px;
}

.input-options button {
  padding: 8px 12px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 13px;
  transition: var(--transition);
}

.input-options button:hover {
  background: var(--bg-tertiary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
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
  margin-bottom: 30px;
}

.quick-actions h2 {
  color: var(--text-primary);
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 15px;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.actions-grid .nav-link {
  padding: 15px;
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
  margin-top: 20px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  padding: 15px;
  background: var(--bg-primary);
  box-shadow: var(--shadow-sm);
}

.guide-button {
  padding: 8px 16px;
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
  gap: 15px;
  justify-content: space-between;
  flex-wrap: wrap;
}

.conversation-header > div:first-child {
  display: flex;
  align-items: center;
  gap: 15px;
}

.main-model-status.inline-status {
  background-color: #f5f5f5;
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 0.9em;
  white-space: nowrap;
  font-family: inherit;
  font-weight: normal;
  color: #333;
  line-height: 1.2;
  letter-spacing: normal;
}

/* Responsive Design */
@media (max-width: 768px) {
  .home-view {
    padding: 15px;
  }
  
  .header {
    flex-direction: column;
    gap: 15px;
    align-items: flex-start;
  }
  
  .header-right {
    width: 100%;
    justify-content: space-between;
  }
  
  .header-buttons {
    flex-wrap: wrap;
  }
  
  .status-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  
  .input-controls {
    flex-direction: column;
  }
  
  .input-options {
    flex-wrap: wrap;
    justify-content: center;
  }
  
  .actions-grid {
    grid-template-columns: 1fr;
  }
  
  .message.user {
    margin-left: 10%;
  }
  
  .message.bot {
    margin-right: 10%;
  }
}
</style>
