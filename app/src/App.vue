<template>
  <div id="app">
    <!-- Top Navigation Bar -->
    <nav class="top-menu-bar">
      <div class="menu-left">
        <span class="system-title">Self Soul</span>
      </div>
      <div class="menu-right">
        <!-- Function Buttons -->
        <router-link to="/" class="menu-link">Home</router-link>
        <router-link to="/training" class="menu-link">Training</router-link>
        <router-link to="/knowledge" class="menu-link">Knowledge</router-link>
        <router-link to="/chat-from-scratch" class="menu-link">Chat From Scratch</router-link>
        <router-link to="/settings" class="menu-link">Settings</router-link>
        <router-link to="/help" class="menu-link">Help</router-link>
        
        <!-- Server Connection Status -->
        <div class="connection-status" :style="{ color: connectionColor }">
          {{ connectionStatus }}
        </div>
      </div>
    </nav>

    <router-view/>
    
    <!-- Voice Input Floating Button -->
    <button 
      v-if="showVoiceInput" 
      class="voice-btn"
      @click="toggleVoiceInput"
      :class="{ 'listening': isVoiceInputActive }"
      aria-label="Voice Input"
    >
      <span v-if="!isVoiceInputActive">🎤</span>
      <span v-else class="pulse-animation">🎤</span>
    </button>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import errorHandler from '@/utils/errorHandler'
import api from '@/utils/api.js'

export default {
  name: 'App',
  emits: ['voice-input'],
  components: {},
  setup(props, { emit }) {
    const router = useRouter();
    const showVoiceInput = ref(true);
    const isVoiceInputActive = ref(false);
    const isConnected = ref(false);
    const connectionStatus = ref('Connecting...');
    const connectionColor = ref('#ff9800'); // Orange
    let speechRecognition = null;
    let connectionInterval = null;
    
    // WebSocket connection will be initialized on demand when needed
    // to avoid unnecessary connections
    
    // Check server connection status
    const checkServerConnection = () => {
      api.get('/health') // 使用统一的API实例和相对路径
        .then(response => {
          isConnected.value = true;
          connectionStatus.value = 'Connected to Main API';
          connectionColor.value = '#4CAF50'; // Green
          
          // If there's a new server message, show notification
          if (response.data && response.data.status) {
            console.log('Server connection established');
          }
        })
        .catch(error => {
          isConnected.value = false;
          connectionStatus.value = 'Main API Disconnected';
          connectionColor.value = '#f44336'; // Red
          console.error('Server connection error:', error);
        });
    };
    
    // Initialize speech recognition
    const initSpeechRecognition = () => {
      try {
        // Prefer standard SpeechRecognition API over webkit prefix
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
        if (SpeechRecognition) {
          speechRecognition = new SpeechRecognition()
          
          // Set speech recognition parameters
          speechRecognition.continuous = false
          speechRecognition.interimResults = false
          speechRecognition.lang = 'en-US'
          
          // Handle speech recognition results
          speechRecognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript
            processVoiceCommand(speechResult)
          }
          
          // Handle speech recognition errors
          speechRecognition.onerror = (error) => {
            console.error('Speech recognition error:', error)
            isVoiceInputActive.value = false
          }
          
          // Handle speech recognition end
          speechRecognition.onend = () => {
            isVoiceInputActive.value = false
          }
        } else {
          console.warn('Web Speech API is not supported in this browser.')
          showVoiceInput.value = false
        }
      } catch (error) {
        errorHandler.handleError('Error initializing speech recognition:', error)
      }
    }
    
    // Initialize components
    const initializeComponentsSilently = () => {
      try {
        errorHandler.logInfo('AGI Brain System components are initializing in the background...')
        
        // Simulate background initialization process
        Promise.all([
          delay(300), // Language related initialization (simplified)
          delay(500), // Connect to backend services
          delay(800)  // Preload necessary models
        ]).then(() => {
          errorHandler.logInfo('AGI Brain System components initialization completed')
        }).catch(error => {
          errorHandler.handleError('Error during system components initialization:', error)
        })
      } catch (error) {
        errorHandler.handleError('Error during system components initialization:', error)
      }
    }
    
    // Toggle voice input state
    const toggleVoiceInput = () => {
      if (isVoiceInputActive.value) {
        if (speechRecognition) {
          speechRecognition.stop()
        }
        isVoiceInputActive.value = false
      } else {
        if (speechRecognition) {
          speechRecognition.start()
          isVoiceInputActive.value = true
        } else {
          console.warn('Speech recognition is not available')
        }
      }
    }
    
    // Process voice commands
    const processVoiceCommand = (command) => {
      // Basic navigation commands
      if (command.toLowerCase().includes('home')) {
        router.push('/')
      } else if (command.toLowerCase().includes('training')) {
        router.push('/training')
      } else if (command.toLowerCase().includes('knowledge')) {
        router.push('/knowledge')
      } else if (command.toLowerCase().includes('settings')) {
        router.push('/settings')
      } else if (command.toLowerCase().includes('help')) {
        router.push('/help')
      }
      // System commands
      else if (command.toLowerCase().includes('connect')) {
        checkServerConnection()
      } else if (command.toLowerCase().includes('refresh')) {
        location.reload()
      }
      // Send to current view for processing
      else {
        // Emit global event for the active view component to handle
        window.dispatchEvent(new CustomEvent('voice-command', { detail: command }))
      }
    }
    
    // Helper function
    const delay = (ms) => {
      return new Promise(resolve => setTimeout(resolve, ms))
    }
    
    // Life cycle hooks
    onMounted(() => {
      // Register error handler
      window.addEventListener('error', errorHandler.handleError)
      window.addEventListener('unhandledrejection', errorHandler.handlePromiseRejection)
      
      // Periodically check server connection
      connectionInterval = setInterval(() => {
        checkServerConnection()
      }, 5000); // Check every 5 seconds
      
      // Initialize speech recognition
      initSpeechRecognition()
      
      // Check connection immediately
      checkServerConnection()
    })
    
    onUnmounted(() => {
      // Clear interval
      clearInterval(connectionInterval)
      
      // Remove event listeners
      window.removeEventListener('error', errorHandler.handleError)
      window.removeEventListener('unhandledrejection', errorHandler.handlePromiseRejection)
    })
    
    return {
      showVoiceInput,
      isVoiceInputActive,
      isConnected,
      connectionStatus,
      connectionColor,
      toggleVoiceInput
    }
  }
}
</script>

<style scoped>
/* 使用main.css中定义的黑白灰浅色主题变量 */

/* 全局样式重置 */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  line-height: 1.6;
  color: var(--text-primary);
  background-color: var(--bg-primary);
}

/* App容器样式 */
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 顶部菜单栏样式 - 使用黑白灰浅色主题 */
.top-menu-bar {
  background: var(--bg-secondary);
  color: var(--text-primary);
  padding: 15px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--shadow-sm);
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  height: 70px;
  border-bottom: 1px solid var(--border-color);
}

.menu-left {
  flex: 1;
}

.system-title {
  font-size: 1.2rem;
  font-weight: bold;
  color: var(--text-primary);
}

.menu-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

/* Removed language selector as it's no longer needed */

/* 菜单项样式 */
.menu-link {
  padding: 8px 16px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  text-decoration: none;
  font-size: 14px;
  transition: var(--transition);
  display: inline-block;
}

/* 连接状态样式 */
.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 14px;
  color: var(--text-primary);
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #666;
  transition: background-color 0.3s ease;
}

.status-indicator.connected {
  background-color: #4CAF50; /* Green indicates connected */
  box-shadow: 0 0 8px rgba(76, 175, 80, 0.6);
}

.status-indicator.disconnected {
  background-color: #F44336; /* Red indicates disconnected */
  box-shadow: 0 0 8px rgba(244, 67, 54, 0.6);
}

.status-text {
  font-size: 14px;
  color: var(--text-primary);
}

.menu-link:hover {
  background: var(--bg-tertiary);
  text-decoration: none;
  color: var(--text-primary);
  border-color: var(--border-dark);
}

/* 为router-view添加顶部边距，避免被菜单栏遮挡 */
#app > :not(.top-menu-bar):not(.voice-input-container) {
  margin-top: 70px;
  min-height: calc(100vh - 70px);
}

/* 语音输入浮动按钮 - 使用黑白灰浅色主题 */
.voice-input-container {
  position: fixed;
  bottom: 30px;
  right: 30px;
  display: flex;
  align-items: center;
  gap: 15px;
  z-index: 999;
}

.voice-btn {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 2px solid var(--border-color);
  font-size: 24px;
  cursor: pointer;
  box-shadow: var(--shadow-md);
  transition: var(--transition);
  display: flex;
  justify-content: center;
  align-items: center;
}

.voice-btn:hover {
  transform: scale(1.05);
  box-shadow: var(--shadow-md);
  background: var(--bg-tertiary);
  border-color: var(--border-dark);
}

.voice-btn.listening {
  animation: pulse 2s infinite;
  background: var(--bg-tertiary);
  border-color: var(--border-dark);
}

.voice-status {
  background: var(--bg-secondary);
  color: var(--text-primary);
  padding: 8px 15px;
  border-radius: 20px;
  font-size: 14px;
  white-space: nowrap;
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
}

/* 脉冲动画 */
@keyframes pulse {
  0% {
    box-shadow: 0 2px 8px rgba(136, 136, 136, 0.4);
  }
  50% {
    box-shadow: 0 4px 16px rgba(136, 136, 136, 0.6);
  }
  100% {
    box-shadow: 0 2px 8px rgba(136, 136, 136, 0.4);
  }
}

.pulse-animation {
  animation: pulse-icon 1s infinite;
}

@keyframes pulse-icon {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

.voice-feedback {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 2000;
  animation: fadeInOut 3s ease-in-out;
}

.voice-command-preview {
  background: var(--bg-secondary);
  color: var(--text-primary);
  padding: 15px 25px;
  border-radius: 10px;
  font-size: 1.1rem;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border-color);
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInOut {
  0% { opacity: 0; transform: translate(-50%, -40%); }
  20% { opacity: 1; transform: translate(-50%, -50%); }
  80% { opacity: 1; transform: translate(-50%, -50%); }
  100% { opacity: 0; transform: translate(-50%, -60%); }
}
</style>
