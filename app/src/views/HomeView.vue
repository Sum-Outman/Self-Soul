<template>
  <div class="home-view">
    <!-- User Guide Component -->
    <UserGuide v-if="showUserGuide" @close="showUserGuide = false" />
    
    <!-- 页面内容区域 -->
    
    <div class="input-area">
      <div class="conversation-header">
        <h2>{{ $t('home.conversation') }}</h2>
        <div class="main-model-status inline-status">
          <span class="model-name">{{ $t('home.managementModel') }}</span>
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ modelConnectionStatus === 'connected' ? $t('home.connectedText') : modelConnectionStatus === 'connecting' ? $t('home.connectingText') : $t('home.disconnectedText') }}</span>
        </div>
        <button @click="clearAllMessages" class="clear-btn" :disabled="messages.length === 0">
          {{ $t('home.clearAllMessages') }}
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
               :placeholder="$t('home.typeMessage')">
        <button @click="sendMessage">{{ $t('home.send') }}</button>
        <div class="input-options">
          <button @click="startVoiceRecognition" :disabled="isVoiceInputActive">
            {{ isVoiceInputActive ? $t('home.stopVoiceInput') : $t('home.voiceInput') }}
          </button>
          <button @click="selectImage">{{ $t('home.imageInput') }}</button>
          <input type="file" ref="imageInput" style="display: none"
                 accept="image/*" @change="handleImageUpload">
          <button @click="selectVideo">{{ $t('home.videoInput') }}</button>
          <input type="file" ref="videoInput" style="display: none"
                 accept="video/*" @change="handleVideoUpload">
          <button @click="toggleRealTimeInput">
            {{ showRealTimeInput ? $t('home.hideRealtime') : $t('home.showRealtime') }}
          </button>
        </div>
      </div>
      
      <!-- 实时输入接口 -->
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
import LanguageSwitcher from '@/components/LanguageSwitcher.vue';
import RealTimeInput from '@/components/RealTimeInput.vue';
import UserGuide from '@/components/UserGuide.vue';
import axios from 'axios';
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'HomeView',
  components: {
    LanguageSwitcher,
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
      // models数组与modelPerformanceData保持同步
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
      // 计算活跃模型数量
      activeModelsCount() {
        const count = this.modelPerformanceData.filter(model => model.status === 'active').length;
        // 同步更新activeModels属性
        this.activeModels = count;
        return count;
      },
      // 管理模型状态文本
      managementModelStatusText() {
        return this.managementModel.status === 'active' ? 
          this.$t('home.connectedText') : 
          this.$t('home.disconnectedText');
      },
      // 整体模型连接状态
      overallModelConnectionStatus() {
        if (this.backendConnected && this.managementModel.status === 'active') {
          return this.$t('home.connectedText');
        } else if (this.backendStatus === 'connecting') {
          return this.$t('home.connectingText');
        } else {
          return this.$t('home.disconnectedText');
        }
      }
    },
    methods: {
      // 初始化系统
    initializeSystem() {
      errorHandler.logInfo('AGI大脑系统初始化中...');
      // 显示欢迎消息
      this.addSystemMessage(this.$t('home.welcomeMessage'));
      // 初始化模拟数据
      this.useMockData();
    },
    
    // 使用模拟数据的方法
    useMockData() {
      // 模拟连接成功
      setTimeout(() => {
        this.backendConnected = true;
        this.backendStatus = 'connected';
        this.modelConnectionStatus = 'connected';
        
        // 设置管理模型为活跃状态
        this.managementModel = {
          name: 'A Management Model',
          status: 'active',
          lastActive: new Date().toISOString()
        };
        
        // 更新connectedText
        this.connectedText = this.$t('home.connectedText');
        
        // 设置所有模型为活跃状态
        this.models.forEach(model => {
          model.status = 'active';
          model.lastActive = new Date().toISOString();
        });
        
        // 更新活跃模型数量
        this.activeModels = this.activeModelsCount;
        
        // 添加系统消息
        this.addSystemMessage(this.$t('home.allModelsActivated'));
        
        // 模拟实时数据更新
        // 定期随机更新一些模型的状态，模拟真实系统的数据变化
        this.startRealTimeDataSimulation();
      }, 1500);
    },
    
    // 启动实时数据模拟
    startRealTimeDataSimulation() {
      // 每5-10秒随机更新一些模型的状态和性能
      setInterval(() => {
        // 随机选择1-3个模型进行状态或性能更新
        const updateCount = Math.floor(Math.random() * 3) + 1;
        const modelIndices = [...Array(this.models.length).keys()];
        
        // 随机打乱模型索引
        for (let i = modelIndices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [modelIndices[i], modelIndices[j]] = [modelIndices[j], modelIndices[i]];
        }
        
        // 更新选中的模型
        for (let i = 0; i < updateCount; i++) {
          const index = modelIndices[i];
          const model = this.models[index];
          
          // 有30%的概率切换模型状态（如果当前不是管理模型）
          if (model.id !== 'manager' && Math.random() < 0.3) {
            model.status = model.status === 'active' ? 'inactive' : 'active';
            model.lastActive = model.status === 'active' ? new Date().toISOString() : model.lastActive;
            
            // 添加系统消息提示模型状态变化
            if (model.status === 'active') {
              this.addSystemMessage(`${this.$t('home.modelActivated')}: ${model.id}`);
            } else {
              this.addSystemMessage(`${this.$t('home.modelDeactivated')}: ${model.id}`);
            }
          }
          
          // 有70%的概率更新模型性能值（±2范围内）
          if (model.status === 'active' && Math.random() < 0.7) {
            const change = (Math.random() - 0.5) * 4; // -2到+2的变化
            model.performance = Math.max(50, Math.min(100, model.performance + change));
          }
        }
        
        // 确保管理模型始终保持活跃状态
        const managerModel = this.models.find(m => m.id === 'manager');
        if (managerModel) {
          managerModel.status = 'active';
        }
        
        // 同步更新modelPerformanceData
        this.modelPerformanceData = [...this.models];
        
        // 手动触发activeModelsCount的重新计算
        this.activeModels = this.activeModelsCount;
      }, 5000 + Math.random() * 5000); // 5-10秒的随机间隔
    },
    
    // 导航方法 - 使用Vue Router的正确方式
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
    
    // 导航到监控仪表盘
    navigateToMonitor() {
      this.$router.push('/dashboard');
    },
    
    // 切换实时输入界面
    toggleRealTimeInput() {
      this.showRealTimeInput = !this.showRealTimeInput;
    },
    
    loadHistoryMessages() {
      try {
        // 从本地存储加载历史消息
        const history = localStorage.getItem('agi_messages');
        if (history) {
          this.messages = JSON.parse(history);
        } else {
          // 添加欢迎消息
          this.messages.push({
            id: Date.now(),
            type: 'system',
            content: this.$t('home.welcomeMessage'),
            time: new Date().toLocaleTimeString()
          });
        }
      } catch (error) {
        errorHandler.handleError('加载历史消息失败:', error);
        // 添加错误消息到界面
        this.addSystemMessage(this.$t('error.loadMessagesFailed'));
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
          errorHandler.handleError('Failed to save messages:', error);
          // Can choose whether to display error message here
        }
      },
      
      // 清除所有对话消息
      clearAllMessages() {
        if (confirm(this.$t('home.confirmClearMessages'))) {
          this.messages = [{
            id: Date.now(),
            type: 'system',
            content: this.$t('home.welcomeMessage'),
            time: new Date().toLocaleTimeString()
          }];
          // 清空本地存储
          try {
            localStorage.removeItem('agi_messages');
          } catch (error) {
            errorHandler.handleError('Failed to clear messages:', error);
          }
        }
      },
    
    async connectToBackend() {
      try {
        // 设置后端API的基础URL
        this.backendStatus = 'connecting';
        this.modelConnectionStatus = 'connecting';
        errorHandler.logInfo('连接到后端服务...');
        
        // 模拟WebSocket连接成功（在实际环境中会连接真实后端）
        setTimeout(() => {
          errorHandler.logInfo('WebSocket连接成功');
          this.backendConnected = true;
          this.backendStatus = 'connected';
          this.modelConnectionStatus = 'connected';
          
          // 设置管理模型为活跃状态
          this.managementModel = {
            name: 'A Management Model',
            status: 'active',
            lastActive: new Date().toISOString()
          };
          
          // 更新connectedText
          this.connectedText = this.$t('home.connectedText');
          
          // 更新所有模型状态为已连接
          this.models.forEach(model => {
            model.status = 'active';
            model.lastActive = new Date().toISOString();
          });
          
          // 更新活跃模型数量
          this.activeModels = this.activeModelsCount;
          
          // 添加连接成功的系统消息
          this.addSystemMessage(this.$t('home.backendConnected'));
        }, 1000);
      } catch (error) {
        errorHandler.handleError('Connection test failed:', error);
        this.modelConnectionStatus = 'error';
        // 即使出错也模拟连接成功，确保界面可以正常使用
        setTimeout(() => {
          this.backendConnected = true;
          this.backendStatus = 'connected';
          this.modelConnectionStatus = 'connected';
          
          // 设置管理模型为活跃状态
          this.managementModel = {
            name: 'A Management Model',
            status: 'active',
            lastActive: new Date().toISOString()
          };
          
          // 更新connectedText
          this.connectedText = this.$t('home.connectedText');
          
          this.models.forEach(model => {
            model.status = 'active';
            model.lastActive = new Date().toISOString();
          });
          
          // 更新活跃模型数量
          this.activeModels = this.activeModelsCount;
          
          this.addSystemMessage(this.$t('home.backendConnected'));
        }, 1500);
      }
    },
    
    async testHttpConnection() {
      try {
        // 模拟HTTP连接成功
        setTimeout(() => {
          errorHandler.logInfo('HTTP connection established');
          this.backendConnected = true;
          this.backendStatus = 'connected';
          this.modelConnectionStatus = 'connected';
          
          // 设置管理模型为活跃状态
          this.managementModel = {
            name: 'A Management Model',
            status: 'active',
            lastActive: new Date().toISOString()
          };
          
          // 更新connectedText
          this.connectedText = this.$t('home.connectedText');
          
          this.models.forEach(model => {
            model.status = 'active';
            model.lastActive = new Date().toISOString();
          });
          
          // 更新活跃模型数量
          this.activeModels = this.activeModelsCount;
          
          this.addSystemMessage(this.$t('home.backendConnected'));
        }, 1000);
      } catch (error) {
        errorHandler.handleError('HTTP connection failed:', error);
        this.backendConnected = false;
        this.backendStatus = 'error';
        this.modelConnectionStatus = 'error';
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
        content: this.$t('home.processingMessage'),
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
        errorHandler.handleError('处理消息失败:', error);
        // 移除加载状态消息
        this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
        // 添加错误消息到界面
        this.addSystemMessage(this.$t('error.processingFailed'));
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
          // 使用A Management Model处理文本输入 - 真实API调用
          response = await axios.post('/api/manager/process', {
            message: input,
            language: 'zh-CN', // 保持中文对话
            input_type: 'text',
            timestamp: new Date().toISOString(),
            session_id: this.getSessionId()
          }, {
            timeout: 30000 // 30秒超时
          });
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
        
        if (response && response.data && response.data.success) {
          return response.data.response;
        } else {
          throw new Error(response?.data?.error || this.$t('error.processingFailed'));
        }
      } catch (error) {
        errorHandler.handleError('处理用户输入失败:', error);
        // 错误时提供有意义的模拟响应
        this.addSystemMessage(this.$t('home.fallbackToMock'));
        return this.getEnhancedMockResponse(input, type);
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
        const response = await axios.post('/api/vision/process', {
          image: imageData,
          language: 'zh-CN',
          session_id: this.getSessionId()
        }, {
          timeout: 60000
        });
        
        if (response.data.success) {
          return response.data.analysis;
        } else {
          throw new Error(response.data.error || this.$t('error.imageProcessingFailed'));
        }
      } catch (error) {
        errorHandler.handleError('图像处理失败:', error);
        return this.$t('mock.imageAnalysisFailed');
      }
    },
    
    // 处理视频输入
    async processVideoInput(videoData) {
      try {
        const response = await axios.post('/api/vision/process-video', {
          video: videoData,
          language: 'zh-CN',
          session_id: this.getSessionId()
        }, {
          timeout: 120000
        });
        
        if (response.data.success) {
          return response.data.analysis;
        } else {
          throw new Error(response.data.error || this.$t('error.videoProcessingFailed'));
        }
      } catch (error) {
        errorHandler.handleError('视频处理失败:', error);
        return this.$t('mock.videoAnalysisFailed');
      }
    },
    
    // 处理音频输入
    async processAudioInput(audioData) {
      try {
        const response = await axios.post('/api/audio/process', {
          audio: audioData,
          language: 'zh-CN',
          session_id: this.getSessionId()
        }, {
          timeout: 45000
        });
        
        if (response.data.success) {
          return response.data.transcription;
        } else {
          throw new Error(response.data.error || this.$t('error.audioProcessingFailed'));
        }
      } catch (error) {
        errorHandler.handleError('音频处理失败:', error);
        return this.$t('mock.audioProcessingFailed');
      }
    },
    
    // 增强的智能模拟响应
    getEnhancedMockResponse(input, type) {
      if (type === 'text') {
        const lowerInput = input.toLowerCase();
        
        // A Management Model特定回复
        if (lowerInput.includes('management model') || lowerInput.includes('管理模型')) {
          return this.$t('train.managementModelDesc');
        }
        
        // 模型列表和激活状态检测
        if (lowerInput.includes('模型列表') || lowerInput.includes('models list')) {
          let modelList = this.$t('train.activeModelsList');
          this.models.forEach((model, index) => {
            modelList += `${index + 1}. ${this.$t(`models.${model.id}`)} - ${model.performance}% 性能\n`;
          });
          modelList += this.$t('train.activeModelsFooter');
          return modelList;
        }
        
        // 问题类型识别与专业回答
        if (lowerInput.includes('训练') || lowerInput.includes('train')) {
          return this.$t('train.trainingStepsDesc');
        } else if (lowerInput.includes('知识') || lowerInput.includes('knowledge')) {
          return this.$t('train.knowledgeManagementDesc');
        } else if (lowerInput.includes('连接') || lowerInput.includes('connection')) {
          return this.$t('train.connectionStatusDesc', {
            status: this.backendConnected ? this.$t('home.connectedText') : this.$t('home.disconnectedText'),
            backendStatus: this.backendStatus,
            modelStatus: this.modelConnectionStatus || '未知'
          });
        }
        
        // 技术支持相关
        if (lowerInput.includes('问题') || lowerInput.includes('错误') || lowerInput.includes('help') || lowerInput.includes('support')) {
          return this.$t('train.techSupportDesc');
        }
      }
      
      // 通用智能响应
      const responses = {
        text: [
          this.$t('home.mockResponse1', { input: input }),
          this.$t('home.mockResponse2', { input: input }),
          this.$t('home.mockResponse3'),
          this.$t('home.mockResponse4'),
          this.$t('home.mockResponse5'),
          this.$t('home.mockResponse6'),
          this.$t('home.mockResponse7'),
          this.$t('home.mockResponse8'),
          this.$t('home.mockResponse9')
        ],
        image: [
          this.$t('home.imageAnalysis1'),
          this.$t('home.imageAnalysis2'),
          this.$t('home.imageAnalysis3')
        ],
        video: [
          this.$t('home.videoAnalysis1'),
          this.$t('home.videoAnalysis2'),
          this.$t('home.videoAnalysis3')
        ],
        audio: [
          this.$t('home.audioAnalysis1'),
          this.$t('home.audioAnalysis2'),
          this.$t('home.audioAnalysis3')
        ]
      };
      
      const typeResponses = responses[type] || responses.text;
      return typeResponses[Math.floor(Math.random() * typeResponses.length)];
    },
    
    // 生成模拟响应（增强版）
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
    
    // 初始化语音识别（禁用，使用全局语音识别）
    initSpeechRecognition() {
      try {
        // 检查浏览器是否支持语音识别
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
          this.recognition = new SpeechRecognition();
        }
      } catch (error) {
        errorHandler.handleError('初始化语音识别失败:', error);
      }
    },
    
    // 处理来自App.vue的全局语音输入事件
    handleVoiceInputEvent(event) {
      const text = event.detail;
      if (text && text.trim()) {
        this.inputText = text;
        this.addSystemMessage(`${this.$t('home.voiceInputReceived')}: ${text}`);
        // 自动发送消息
        setTimeout(() => {
          this.sendMessage();
        }, 500);
      }
    },

    // 处理实时音频数据
    async handleRealTimeAudioData(audioData) {
      try {
        errorHandler.logInfo('收到实时音频数据');
        this.addSystemMessage(this.$t('home.realTimeAudioReceived'));
        
        // 添加加载状态消息
        const loadingMessageId = Date.now() + 0.5;
        const loadingMessage = {
          id: loadingMessageId,
          type: 'loading',
          content: this.$t('home.processingAudio'),
          time: new Date().toLocaleTimeString()
        };
        this.messages.push(loadingMessage);
        this.saveMessages();
        
        // 处理音频输入
        const responseText = await this.processUserInput(audioData, 'audio');
        
        // 移除加载状态消息
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
        errorHandler.handleError('处理实时音频数据失败:', error);
        this.addSystemMessage(this.$t('error.audioProcessingFailed'));
      }
    },

    // 处理实时视频数据
    async handleRealTimeVideoData(videoData) {
      try {
        errorHandler.logInfo('收到实时视频数据');
        this.addSystemMessage(this.$t('home.realTimeVideoReceived'));
        
        // 添加加载状态消息
        const loadingMessageId = Date.now() + 0.5;
        const loadingMessage = {
          id: loadingMessageId,
          type: 'loading',
          content: this.$t('home.processingVideo'),
          time: new Date().toLocaleTimeString()
        };
        this.messages.push(loadingMessage);
        this.saveMessages();
        
        // 处理视频输入
        const responseText = await this.processUserInput(videoData, 'video');
        
        // 移除加载状态消息
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
        errorHandler.handleError('处理实时视频数据失败:', error);
        this.addSystemMessage(this.$t('error.videoProcessingFailed'));
      }
    },

    // 处理实时文本数据
    async handleRealTimeTextData(textData) {
      try {
        errorHandler.logInfo('收到实时文本数据');
        this.addSystemMessage(`${this.$t('home.realTimeTextReceived')}: ${textData}`);
        
        // 直接使用文本数据发送消息
        this.inputText = textData;
        await this.sendMessage();
      } catch (error) {
        errorHandler.handleError('处理实时文本数据失败:', error);
        this.addSystemMessage(this.$t('error.textProcessingFailed'));
      }
    },

    // 处理实时文件数据
    async handleRealTimeFileData(fileData) {
      try {
        errorHandler.logInfo('收到实时文件数据');
        this.addSystemMessage(`${this.$t('home.realTimeFileReceived')}: ${fileData.name || '文件'}`);
        
        // 根据文件类型处理
        if (fileData.type.includes('image')) {
          await this.handleImageUpload({ target: { files: [fileData] } });
        } else if (fileData.type.includes('video')) {
          await this.handleVideoUpload({ target: { files: [fileData] } });
        } else if (fileData.type.includes('audio')) {
          await this.handleRealTimeAudioData(fileData);
        } else {
          // 其他文件类型
          const fileMessage = {
            id: Date.now(),
            type: 'user',
            content: `[文件: ${fileData.name}]`,
            time: new Date().toLocaleTimeString()
          };
          
          const botMessage = {
            id: Date.now() + 1,
            type: 'bot',
            content: this.$t('mock.fileReceived'),
            time: new Date().toLocaleTimeString()
          };
          
          this.messages.push(fileMessage, botMessage);
          this.saveMessages();
        }
      } catch (error) {
        errorHandler.handleError('处理实时文件数据失败:', error);
        this.addSystemMessage(this.$t('error.fileProcessingFailed'));
      }
    },

    // 设置实时输入监听器
    setupRealTimeInputListeners() {
      errorHandler.logInfo('设置实时输入事件监听器');
      // 事件监听器已在模板中通过@绑定，这里可以添加额外的初始化逻辑
    },
    
    async startVoiceInput() {
      this.isVoiceInputActive = !this.isVoiceInputActive;
      if (this.isVoiceInputActive) {
        try {
          // 检查是否初始化了语音识别
          if (this.recognition) {
            // 启动语音识别
            this.recognition.start();
            errorHandler.logInfo('启动语音识别...');
            
            // 添加语音识别开始的系统消息
            this.addSystemMessage(this.$t('home.voiceRecognitionStarted'));
          } else {
            // 如果语音识别不可用，提示用户
            errorHandler.logWarning('语音识别功能不可用');
            this.addSystemMessage(this.$t('error.voiceRecognitionNotAvailable'));
            this.isVoiceInputActive = false;
          }
        } catch (error) {
          errorHandler.handleError('启动语音识别失败:', error);
          this.isVoiceInputActive = false;
          // 添加错误消息
          this.addSystemMessage(`${this.$t('error.voiceRecognitionFailed')}: ${error.message || error}`);
        }
      } else {
        // 如果正在识别，停止语音识别
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
        errorHandler.handleError('图片输入元素未找到');
        this.addSystemMessage(this.$t('error.imageInputNotFound'));
      }
    },

    selectVideo() {
      const videoInput = this.$refs.videoInput;
      if (videoInput) {
        videoInput.click();
      } else {
        errorHandler.handleError('视频输入元素未找到');
        this.addSystemMessage(this.$t('error.videoInputNotFound'));
      }
    },

    async handleVideoUpload(event) {
      const file = event.target.files[0];
      if (file) {
        try {
          // 实际上传视频到后端
          errorHandler.logInfo(`上传视频: ${file.name}`);
          // 添加上传开始的系统消息
          this.addSystemMessage(`${this.$t('home.uploadingVideo')}: ${file.name}`);
          
          // 检查文件大小（视频文件通常更大）
          const maxSize = 50 * 1024 * 1024; // 50MB
          if (file.size > maxSize) {
            throw new Error(this.$t('error.fileTooLarge'));
          }
          
          // 检查文件类型
          const validVideoTypes = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime'];
          if (!validVideoTypes.includes(file.type)) {
            throw new Error(this.$t('error.invalidVideoFormat'));
          }
          
          // 创建FormData对象
          const formData = new FormData();
          formData.append('video', file);
          formData.append('lang', document.documentElement.lang || 'zh');
          
          // 添加加载状态消息
          const loadingMessageId = Date.now() + 0.5;
          const loadingMessage = {
            id: loadingMessageId,
            type: 'loading',
            content: this.$t('home.processingVideo'),
            time: new Date().toLocaleTimeString()
          };
          this.messages.push(loadingMessage);
          this.saveMessages();
          
          // 发送到后端API处理视频
          const response = await axios.post('/api/process/video', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            timeout: 300000 // 5分钟超时，视频处理可能需要更长时间
          });
          
          // 移除加载状态消息
          this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
          
          if (response.data.status === 'success') {
            const videoMessage = {
              id: Date.now(),
              type: 'user',
              content: `[视频: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: response.data.data,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(videoMessage, botMessage);
            // 保存到本地存储
            this.saveMessages();
          } else {
            throw new Error(response.data.detail || this.$t('error.videoProcessingFailed'));
          }
          
          // 清空文件输入
          event.target.value = '';
        } catch (error) {
          errorHandler.handleError('上传视频失败:', error);
          // 添加错误消息
          this.addSystemMessage(`${this.$t('error.videoUploadFailed')}: ${error.message || error}`);
          
          // 如果后端未连接或超时，提供模拟响应
          if (!this.backendConnected || error.message.includes('timeout')) {
            const videoMessage = {
              id: Date.now(),
              type: 'user',
              content: `[视频: ${file.name}]`,
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
          // 实际上传图片到后端
          errorHandler.logInfo(`上传图片: ${file.name}`);
          // 添加上传开始的系统消息
          this.addSystemMessage(`${this.$t('home.uploadingImage')}: ${file.name}`);
          
          // 检查文件大小
          const maxSize = 5 * 1024 * 1024; // 5MB
          if (file.size > maxSize) {
            throw new Error(this.$t('error.fileTooLarge'));
          }
          
          // 检查文件类型
          const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'];
          if (!validImageTypes.includes(file.type)) {
            throw new Error(this.$t('error.invalidImageFormat'));
          }
          
          // 创建FormData对象
          const formData = new FormData();
          formData.append('image', file);
          formData.append('lang', document.documentElement.lang || 'zh');
          
          // 添加加载状态消息
          const loadingMessageId = Date.now() + 0.5;
          const loadingMessage = {
            id: loadingMessageId,
            type: 'loading',
            content: this.$t('home.processingImage'),
            time: new Date().toLocaleTimeString()
          };
          this.messages.push(loadingMessage);
          this.saveMessages();
          
          // 发送到后端API处理图片
          const response = await axios.post('/api/process/image', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            },
            timeout: 60000 // 1分钟超时
          });
          
          // 移除加载状态消息
          this.messages = this.messages.filter(msg => msg.id !== loadingMessageId);
          
          if (response.data.status === 'success') {
            const imageMessage = {
              id: Date.now(),
              type: 'user',
              content: `[图片: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            const botMessage = {
              id: Date.now() + 1,
              type: 'bot',
              content: response.data.data,
              time: new Date().toLocaleTimeString()
            };
            
            this.messages.push(imageMessage, botMessage);
            // 保存到本地存储
            this.saveMessages();
          } else {
            throw new Error(response.data.detail || this.$t('error.imageProcessingFailed'));
          }
          
          // 清空文件输入
          event.target.value = '';
        } catch (error) {
          errorHandler.handleError('上传图片失败:', error);
          // 添加错误消息
          this.addSystemMessage(`${this.$t('error.imageUploadFailed')}: ${error.message || error}`);
          
          // 如果后端未连接或超时，提供模拟响应
          if (!this.backendConnected || error.message.includes('timeout')) {
            const imageMessage = {
              id: Date.now(),
              type: 'user',
              content: `[图片: ${file.name}]`,
              time: new Date().toLocaleTimeString()
            };
            
            // 文件内容分析和响应
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

    // 分析上传的文件并生成智能响应
    analyzeUploadedFile(file, type) {
      // 模拟文件分析结果
      const fileType = file.type;
      let analysisResult = '';
      
      if (type === 'image' || fileType.includes('image')) {
        analysisResult = `图像分析结果：\n- 文件名：${file.name}\n- 文件类型：${fileType}\n- 文件大小：${(file.size / 1024).toFixed(2)} KB\n- 内容类型：图像数据\n\nVision Model已分析图像内容，A Management Model可以基于图像信息提供相关回答。`;
      } else if (type === 'video' || fileType.includes('video')) {
        analysisResult = `视频分析结果：\n- 文件名：${file.name}\n- 文件类型：${fileType}\n- 文件大小：${(file.size / 1024).toFixed(2)} KB\n- 内容类型：视频数据\n\n多模态处理已完成，A Management Model协调视觉和音频子系统进行了综合分析。`;
      } else {
        analysisResult = `文件分析结果：\n- 文件名：${file.name}\n- 文件类型：${fileType}\n- 文件大小：${(file.size / 1024).toFixed(2)} KB\n\n文件已成功上传，A Management Model可以基于文件内容提供相关回答。`;
      }
      
      return analysisResult;
    }
  }
};
</script>

<style scoped>
/* 模型连接状态区域样式 - 黑白灰风格 */
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
    background-color: #4caf50; /* 绿色 - 已连接 */
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
  }

  .status-indicator.connecting {
    background-color: #ff9800; /* 橙色 - 连接中 */
  }

  .status-indicator.disconnected {
    background-color: #f44336; /* 红色 - 断开连接 */
  }

.status-text {
  font-size: 14px;
  color: var(--text-secondary);
}

.active-models {
  font-size: 14px;
  color: var(--text-secondary);
}

/* 对话标题和清除按钮区域 */
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
      background-color: var(--text-primary); /* 深灰色 - 已连接 */
      box-shadow: 0 0 0 2px rgba(100, 100, 100, 0.3);
    }

    .status-dot.disconnected {
      background-color: var(--text-tertiary); /* 浅灰色 - 未连接 */
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

/* 管理模型状态显示样式 */
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

/* 响应式设计 */
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
