<template>
  <div class="conversation-container">
    <div class="conversation-header">
      <h1>Self Soul - Intelligent Conversation</h1>
      <div class="header-controls">
        <div class="model-status">
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ connectionStatusText }}</span>
        </div>
        <button @click="clearConversation" class="clear-btn">Clear Conversation</button>
      </div>
    </div>
    
    <div class="chat-container" ref="chatContainer">
      <div v-if="messages.length === 0" class="empty-chat">
        <div class="welcome-message">
          <h3>Welcome to Self Soul</h3>
          <p>Have a real conversation with the main management model, supporting emotional analysis and multimodal interaction</p>
          <div class="connection-info">
            <p>Management Model Port: 8001</p>
            <p>Current Status: {{ connectionStatusText }}</p>
          </div>
        </div>
      </div>
      
      <div v-for="message in messages" :key="message.id" class="message-wrapper">
        <div :class="['message', message.sender, { 'error': message.isError }]">
          <div class="message-header">
            <span class="sender-name">{{ message.sender === 'user' ? 'You' : 'Self Soul' }}</span>
            <span class="timestamp">{{ formatTimestamp(message.timestamp) }}</span>
          </div>
          <div class="message-content">
            <div v-if="message.type === 'text'">{{ message.content }}</div>
            <div v-else-if="message.type === 'image'" class="media-content">
              <img :src="message.content" alt="Image message" class="media-preview">
            </div>
            <div v-else-if="message.type === 'audio'" class="media-content">
              <audio controls :src="message.content" class="audio-player"></audio>
            </div>
          </div>
          <div v-if="message.emotion && message.sender === 'model'" class="emotion-indicator">
            <span class="emotion-label">Emotion: {{ message.emotion }}</span>
            <span v-if="message.confidence" class="confidence">Confidence: {{ (message.confidence * 100).toFixed(1) }}%</span>
          </div>
          <div v-if="message.isError" class="error-indicator">
            <span class="error-icon">⚠️</span>
            <span class="error-text">Connection Error</span>
          </div>
        </div>
      </div>
      
      <div v-if="isLoading" class="loading-indicator">
        <div class="typing-animation">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span class="loading-text">Self Soul is thinking...</span>
      </div>
    </div>
    
    <div class="input-area">
      <div class="input-controls">
        <input 
          type="text" 
          v-model="newMessage"
          @keyup.enter="sendMessage"
          placeholder="Type a message... (supports text, image URL, audio URL)"
          :disabled="isLoading"
        >
        <div class="input-buttons">
        <button @click="sendMessage" :disabled="!newMessage.trim() || isLoading" class="send-btn">
          {{ isLoading ? 'Sending...' : 'Send' }}
        </button>
        <button @click="toggleSpeechRecognition" :disabled="isLoading" class="voice-input-btn" :class="{ 'active': isSpeechRecognitionActive }">
          {{ isSpeechRecognitionActive ? 'Stop' : 'Voice Input' }}
        </button>
      </div>
      </div>
        <div class="multimodal-hint">
          <small>Hint: You can enter image URLs or audio URLs for multimodal interaction</small>
        </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import api from '../utils/api.js';

// Refs
const chatContainer = ref(null);

// Data
const messages = ref([]);
const newMessage = ref('');
const modelConnectionStatus = ref('connecting');
const isLoading = ref(false);
const error = ref(null);

// Speech recognition
const recognition = ref(null);
const transcript = ref('');
const isSpeechRecognitionActive = ref(false);

// Computed
const connectionStatusText = computed(() => {
  const statusMap = {
    'connected': 'Connected',
    'connecting': 'Connecting',
    'disconnected': 'Disconnected'
  };
  return statusMap[modelConnectionStatus.value] || 'Unknown Status';
});

// Methods
function detectMessageType(content) {
  if (content.match(/\.(jpg|jpeg|png|gif|bmp|webp)$/i)) {
    return 'image';
  } else if (content.match(/\.(mp3|wav|ogg|m4a)$/i)) {
    return 'audio';
  }
  return 'text';
}

async function sendMessage() {
  if (!newMessage.value.trim() || isLoading.value) return;
  
  const userMessage = newMessage.value.trim();
  const messageType = detectMessageType(userMessage);
  
  // Add user message
  messages.value.push({
    id: Date.now(),
    sender: 'user',
    content: userMessage,
    type: messageType,
    timestamp: Date.now()
  });
  
  // Clear input
  newMessage.value = '';
  isLoading.value = true;
  error.value = null;
  
  try {
    // Send message to management model (model ID 8001)
    const response = await api.managementChat({
      message: userMessage,
      message_type: messageType,
      history: messages.value.slice(-10).map(msg => ({
        content: msg.content,
        type: msg.type,
        sender: msg.sender
      }))
    });
    
    // Add model response
    const modelResponse = {
      id: Date.now() + 1,
      sender: 'model',
      content: response.data.response,
      type: response.data.type || 'text',
      timestamp: Date.now(),
      emotion: response.data.emotion,
      confidence: response.data.confidence
    };
    
    messages.value.push(modelResponse);
    modelConnectionStatus.value = 'connected';
    
  } catch (err) {
    console.error('Failed to send message:', err);
    error.value = 'Cannot connect to management model. Please ensure the backend server is running.';
    
    // Add error message
    messages.value.push({
      id: Date.now() + 1,
      sender: 'model',
      content: 'Sorry, I cannot process your request at the moment. Please ensure the backend server is running and try again.',
      type: 'text',
      timestamp: Date.now(),
      isError: true
    });
    
    modelConnectionStatus.value = 'disconnected';
  } finally {
    isLoading.value = false;
    scrollToBottom();
  }
}

function clearConversation() {
  messages.value = [];
  error.value = null;
}

function scrollToBottom() {
  if (chatContainer.value) {
    setTimeout(() => {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
    }, 100);
  }
}

function formatTimestamp(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now - date;
  
  if (diff < 60000) { // within 1 minute
    return 'Just now';
  } else if (diff < 3600000) { // within 1 hour
    return `${Math.floor(diff / 60000)}m ago`;
  } else if (diff < 86400000) { // within 1 day
    return `${Math.floor(diff / 3600000)}h ago`;
  } else {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      month: 'short',
      day: 'numeric'
    });
  }
}

async function checkModelConnection() {
  try {
    await api.health.get();
    modelConnectionStatus.value = 'connected';
  } catch (err) {
    console.error('Model connection check failed:', err);
    modelConnectionStatus.value = 'disconnected';
  }
}

// Speech recognition methods
async function startSpeechRecognition() {
  // 重置之前的错误
  error.value = null;
  
  // 检查浏览器支持
  const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognitionAPI) {
    error.value = '语音识别功能在当前浏览器中不被支持，请使用 Chrome 或 Edge 浏览器。';
    return;
  }
  
  // 检查页面协议（某些浏览器限制HTTP下的麦克风访问）
  if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    error.value = '语音识别功能在 HTTP 协议下可能受到限制，请使用 HTTPS 访问或在本地环境（localhost/127.0.0.1）下使用。';
    return;
  }
  
  // 确保之前的识别实例已停止
  if (recognition.value) {
    try {
      recognition.value.stop();
    } catch (e) {
      console.log('之前的语音识别实例已停止或不存在');
    }
    recognition.value = null;
  }
  
  // 请求麦克风权限
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error('麦克风权限错误:', err);
    let errorMessage;
    
    switch (err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        errorMessage = '麦克风权限被拒绝，请在浏览器设置中启用麦克风访问权限后重试。\n\n操作方法：\n1. 点击地址栏左侧的锁定图标\n2. 选择"网站设置"\n3. 在麦克风选项中选择"允许"';
        break;
      case 'NotFoundError':
        errorMessage = '未检测到麦克风设备，请连接麦克风后重试。';
        break;
      case 'NotReadableError':
        errorMessage = '麦克风正在被其他应用使用，请关闭该应用后重试。';
        break;
      case 'AbortError':
        errorMessage = '权限请求被中断，请重试。';
        break;
      case 'TypeError':
        errorMessage = '无法获取麦克风设备，请检查设备连接和浏览器设置。';
        break;
      default:
        errorMessage = `无法访问麦克风：${err.message}。请检查设备设置并重试。`;
    }
    
    error.value = errorMessage;
    return;
  }
  
  // 初始化语音识别
  try {
    recognition.value = new SpeechRecognitionAPI();
    recognition.value.continuous = true;
    recognition.value.interimResults = true;
    recognition.value.lang = 'zh-CN'; // 设置为中文识别
    recognition.value.maxAlternatives = 1;
    
    recognition.value.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcriptText = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcriptText;
          // 添加最终转录到输入字段
          newMessage.value = finalTranscript;
          transcript.value = '';
        } else {
          interimTranscript += transcriptText;
          transcript.value = interimTranscript;
        }
      }
    };
    
    recognition.value.onerror = (event) => {
      console.error('语音识别错误:', event.error);
      let errorMessage;
      
      switch (event.error) {
        case 'not-allowed':
          errorMessage = '麦克风权限被拒绝，请在浏览器设置中启用麦克风访问权限后重试。\n\n操作方法：\n1. 点击地址栏左侧的锁定图标\n2. 选择"网站设置"\n3. 在麦克风选项中选择"允许"';
          break;
        case 'no-speech':
          errorMessage = '未检测到语音，请对着麦克风清晰说话。';
          break;
        case 'aborted':
          errorMessage = '语音识别被中断，请重试。';
          break;
        case 'audio-capture':
          errorMessage = '音频捕获失败，请检查麦克风设置。';
          break;
        case 'network':
          errorMessage = '网络错误，语音识别需要网络连接。请检查您的互联网连接。';
          break;
        case 'service-not-allowed':
          errorMessage = '语音识别服务不可用，可能是由于浏览器设置或网络限制。';
          break;
        case 'bad-grammar':
          errorMessage = '语音识别语法错误，请重试。';
          break;
        case 'language-not-supported':
          errorMessage = '不支持当前语言，请尝试其他语言设置。';
          break;
        default:
          errorMessage = `语音识别错误：${event.error}`;
      }
      
      error.value = errorMessage;
      isSpeechRecognitionActive.value = false;
      
      // 确保识别服务已停止
      if (recognition.value) {
        try {
          recognition.value.stop();
        } catch (e) {
          console.log('停止语音识别实例时出错:', e);
        }
        recognition.value = null;
      }
    };
    
    recognition.value.onend = () => {
      console.log('语音识别已结束');
      isSpeechRecognitionActive.value = false;
      recognition.value = null;
    };
    
    recognition.value.onstart = () => {
      console.log('语音识别已成功启动');
      isSpeechRecognitionActive.value = true;
    };
    
    recognition.value.start();
  } catch (err) {
    console.error('初始化语音识别失败:', err);
    error.value = `启动语音识别失败：${err.message}。请重试。`;
    isSpeechRecognitionActive.value = false;
    
    // 确保识别服务已停止
    if (recognition.value) {
      try {
        recognition.value.stop();
      } catch (e) {
        console.log('清理语音识别实例时出错:', e);
      }
      recognition.value = null;
    }
  }
}

function stopSpeechRecognition() {
  if (recognition.value) {
    recognition.value.stop();
    recognition.value = null;
    isSpeechRecognitionActive.value = false;
  }
}

function toggleSpeechRecognition() {
  if (isSpeechRecognitionActive.value) {
    stopSpeechRecognition();
  } else {
    startSpeechRecognition();
  }
}

// Lifecycle
onMounted(() => {
  checkModelConnection();
});
</script>

<style scoped>
.conversation-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 70px);
  background: #f8f9fa;
  color: #2c3e50;
  margin-top: 70px;
}

.conversation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background-color: rgba(255, 255, 255, 0.95);
  border-bottom: 1px solid #e0e0e0;
  backdrop-filter: blur(10px);
}

.conversation-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: #2c3e50;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background-color: #f8f9fa;
  border-radius: 20px;
  border: 1px solid #e0e0e0;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #888888;
  animation: pulse 2s infinite;
}

.status-indicator.connected {
  background-color: #555555;
}

.status-indicator.connecting {
  background-color: #999999;
}

.status-text {
  font-size: 0.85rem;
  font-weight: 500;
  color: #495057;
}

.clear-btn {
  padding: 0.5rem 1rem;
  background-color: #e9ecef;
  color: #495057;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.clear-btn:hover {
  background-color: #dee2e6;
  color: #212529;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  background-color: rgba(255, 255, 255, 0.1);
}

.empty-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.welcome-message {
  text-align: center;
  background: rgba(255, 255, 255, 0.95);
  padding: 2rem;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  max-width: 500px;
}

.welcome-message h3 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
  font-size: 1.5rem;
}

.welcome-message p {
  margin: 0 0 1.5rem 0;
  color: #6c757d;
  line-height: 1.5;
}

.connection-info {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.connection-info p {
  margin: 0.25rem 0;
  font-size: 0.9rem;
  color: #495057;
}

.message-wrapper {
  display: flex;
}

.message-wrapper.user {
  justify-content: flex-end;
}

.message-wrapper.model {
  justify-content: flex-start;
}

.message {
  max-width: 70%;
  padding: 1rem;
  border-radius: 16px;
  word-wrap: break-word;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  position: relative;
}

.message.user {
  background: #007bff;
  color: white;
}

.message.model {
  background: rgba(255, 255, 255, 0.95);
  color: #2c3e50;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.message.error {
  background: rgba(255, 235, 238, 0.95);
  border: 1px solid rgba(244, 67, 54, 0.3);
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.message.user .message-header {
  border-bottom-color: rgba(255, 255, 255, 0.3);
}

.sender-name {
  font-weight: 600;
  font-size: 0.9rem;
}

.timestamp {
  font-size: 0.75rem;
  opacity: 0.7;
}

.message-content {
  line-height: 1.5;
}

.media-content {
  margin: 0.5rem 0;
}

.media-preview {
  max-width: 300px;
  max-height: 200px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.audio-player {
  width: 250px;
  height: 40px;
  border-radius: 20px;
}

.emotion-indicator {
  margin-top: 0.75rem;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
}

.emotion-label {
  color: #495057;
  font-weight: 500;
}

.confidence {
  color: #6c757d;
  font-style: italic;
}

.error-indicator {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #dc3545;
  font-size: 0.8rem;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  margin: 0.5rem 0;
}

.typing-animation {
  display: flex;
  gap: 0.25rem;
}

.typing-animation span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--text-tertiary);
  animation: typing 1.4s infinite ease-in-out;
}

.typing-animation span:nth-child(1) { animation-delay: -0.32s; }
.typing-animation span:nth-child(2) { animation-delay: -0.16s; }

.loading-text {
  color: #495057;
  font-size: 0.9rem;
  font-weight: 500;
}

.input-area {
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.95);
  border-top: 1px solid #e0e0e0;
}

.input-controls {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.5rem;
}

.input-area input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e9ecef;
  border-radius: 12px;
  font-size: 1rem;
  transition: all 0.2s ease;
  background: white;
}

.input-area input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.input-area input:disabled {
  background-color: #f8f9fa;
  color: #6c757d;
}

.input-buttons {
  display: flex;
  gap: 0.5rem;
}

.send-btn {
  padding: 0.75rem 1.5rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 80px;
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
}

.send-btn:disabled {
  background: #adb5bd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.voice-input-btn {
  padding: 0.75rem 1.5rem;
  background: #28a745;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 120px;
}

.voice-input-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
  background: #218838;
}

.voice-input-btn:disabled {
  background: #adb5bd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.voice-input-btn.active {
  background: #dc3545;
  animation: pulse 1.5s infinite;
}

.voice-input-btn.active:hover:not(:disabled) {
  box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
  background: #c82333;
}

.multimodal-hint {
  text-align: center;
  margin-bottom: 0.5rem;
}

.multimodal-hint small {
  color: #6c757d;
  font-style: italic;
}

.error-message {
  color: #dc3545;
  background: rgba(220, 53, 69, 0.1);
  padding: 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(220, 53, 69, 0.2);
  font-size: 0.9rem;
  text-align: center;
}

/* Animations */
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

@keyframes typing {
  0%, 60%, 100% { transform: scale(0.8); opacity: 0.5; }
  30% { transform: scale(1); opacity: 1; }
}

/* Scrollbar styling */
.chat-container::-webkit-scrollbar {
  width: 6px;
}

.chat-container::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}
</style>
