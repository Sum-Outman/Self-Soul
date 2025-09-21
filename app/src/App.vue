<template>
  <div id="app">
    <!-- System Top Menu Bar -->
    <div class="top-menu-bar">
      <div class="menu-left">
        <span class="system-title">AGI Brain System</span>
      </div>
      <div class="menu-right">
        <!-- Function Buttons -->
        <router-link to="/" class="menu-link">Interaction</router-link>
        <router-link to="/training" class="menu-link">Training</router-link>
        <router-link to="/knowledge" class="menu-link">Knowledge</router-link>
        <router-link to="/settings" class="menu-link">Settings</router-link>
        <router-link to="/help" class="menu-link">Help</router-link>
        
        <!-- Server Connection Status -->
        <div class="connection-status">
          <span class="status-indicator" :class="{ 'connected': isConnected, 'disconnected': !isConnected }"></span>
          <span class="status-text">{{ isConnected ? 'Connected' : 'Disconnected' }}</span>
        </div>
      </div>
    </div>

    <router-view/>
    
    <!-- Voice Input Floating Button -->
    <div class="voice-input-container" v-if="showVoiceInput">
      <button @click="toggleVoiceInput" class="voice-btn" :class="{ 'listening': recognitionInProgress }">
        <span v-if="!recognitionInProgress">🎤</span>
        <span v-else class="pulse-animation">🎤</span>
      </button>
      <div class="voice-status" v-if="recognitionInProgress">Listening...</div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import errorHandler from '@/utils/errorHandler'

export default {
  name: 'App',
  emits: ['voice-input'],
  components: {
    // 按需导入组件
  },
  setup(props, { emit }) {
    const router = useRouter();
    const showVoiceInput = ref(true);
    const showVoiceStatus = ref(false);
    const voiceStatusMessage = ref('');
    const speechRecognition = ref(null);
    const recognitionInProgress = ref(false);
    const isConnected = ref(false); // Server connection status, default disconnected
    
    // WebSocket connection
    let ws = null;
    let wsReconnectTimer = null;
    const RECONNECT_INTERVAL = 5000;
    const MAX_RECONNECT_ATTEMPTS = 5;
    let reconnectAttempts = 0;
    
    // Initialize WebSocket connection
    const initWebSocket = () => {
      try {
        // Close any existing connection
        if (ws && ws.readyState !== WebSocket.CLOSED) {
          ws.close();
        }
        
        // Create new WebSocket connection (using main API gateway port 8000)
        const wsUrl = `ws://${window.location.hostname || 'localhost'}:8000/ws`;
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('WebSocket connection established');
          isConnected.value = true;
          reconnectAttempts = 0;
        };
        
        ws.onmessage = (event) => {
          // Handle incoming messages from server
          try {
            const data = JSON.parse(event.data);
            // Process server messages here
          } catch (error) {
            errorHandler.handleError('Error parsing WebSocket message:', error);
          }
        };
        
        ws.onclose = () => {
          console.log('WebSocket connection closed');
          isConnected.value = false;
          
          // Schedule reconnection
          if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
            wsReconnectTimer = setTimeout(initWebSocket, RECONNECT_INTERVAL);
          } else {
            console.error('Max reconnection attempts reached');
          }
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          errorHandler.handleError('WebSocket error:', error);
          isConnected.value = false;
        };
      } catch (error) {
        errorHandler.handleError('Error initializing WebSocket:', error);
        isConnected.value = false;
        
        // Schedule reconnection
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts++;
          console.log(`Attempting to reconnect (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
          wsReconnectTimer = setTimeout(initWebSocket, RECONNECT_INTERVAL);
        }
      }
    };
    
    // Check server connection status
    const checkServerConnection = () => {
      // Try to connect to the main API gateway
      fetch(`http://${window.location.hostname || 'localhost'}:8000/api/status`, {
        method: 'GET',
        timeout: 3000
      })
        .then(response => {
          if (response.ok) {
            isConnected.value = true;
            // If API connection is successful, try to establish WebSocket
            if (!ws || ws.readyState === WebSocket.CLOSED) {
              initWebSocket();
            }
          } else {
            isConnected.value = false;
          }
        })
        .catch(() => {
          isConnected.value = false;
        })
        .finally(() => {
          // Check connection status every 5 seconds
          setTimeout(checkServerConnection, 5000);
        });
    };
    
    // Initialize speech recognition
    const initSpeechRecognition = () => {
      try {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
          speechRecognition.value = new SpeechRecognition()
          
          // Set speech recognition parameters
          speechRecognition.value.lang = getSpeechLanguage() // Get language from user settings or browser
          speechRecognition.value.interimResults = true
          speechRecognition.value.maxAlternatives = 1
          speechRecognition.value.continuous = false
          
          // Handle speech recognition results
          speechRecognition.value.onresult = (event) => {
            const transcript = event.results[0][0].transcript
            processVoiceCommand(transcript)
          }
          
          // Handle speech recognition errors
          speechRecognition.value.onerror = (event) => {
            errorHandler.handleError('Speech recognition error:', event.error)
            recognitionInProgress.value = false
            showVoiceStatus.value = true
            voiceStatusMessage.value = `Speech recognition error: ${event.error}`
            setTimeout(() => {
              showVoiceStatus.value = false
            }, 3000)
          }
          
          // Handle speech recognition end
          speechRecognition.value.onend = () => {
            recognitionInProgress.value = false
            showVoiceStatus.value = true
            voiceStatusMessage.value = 'Speech recognition ended'
            setTimeout(() => {
              showVoiceStatus.value = false
            }, 2000)
          }
          
          // Handle speech recognition start
          speechRecognition.value.onstart = () => {
            recognitionInProgress.value = true
            showVoiceStatus.value = true
            voiceStatusMessage.value = 'Listening...'
          }
        } else {
          errorHandler.handleWarning('Speech recognition is not supported in this browser')
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
    
    // Speech recognition related methods
    const getSpeechLanguage = () => {
      // Get user language from localStorage or browser settings
      const savedLanguage = localStorage.getItem('speechLanguage');
      
      if (savedLanguage) {
        return savedLanguage;
      }
      
      // Default to browser language or 'en-US'
      return navigator.language || navigator.userLanguage || 'en-US';
    }
    
    const toggleVoiceInput = () => {
      if (!speechRecognition.value) return
      
      if (recognitionInProgress.value) {
        speechRecognition.value.stop()
      } else {
        speechRecognition.value.lang = getSpeechLanguage()
        speechRecognition.value.start()
        recognitionInProgress.value = true
        showVoiceStatus.value = true
        voiceStatusMessage.value = 'Listening...'
      }
    }
    
    const processVoiceCommand = (command) => {
      try {
        // Voice command processing
        errorHandler.logInfo('Voice command:', command)
        showVoiceStatus.value = true
        voiceStatusMessage.value = `Recognized: ${command}`
        
        // Simple command processing logic
        const lowerCommand = command.toLowerCase()
        
        // Navigation commands - only to valid routes
        if (lowerCommand.includes('home')) {
          router.push('/')
        } else if (lowerCommand.includes('train')) {
          router.push('/training')
        } else if (lowerCommand.includes('knowledge')) {
          router.push('/knowledge')
        } else if (lowerCommand.includes('settings')) {
          router.push('/settings')
        } else if (lowerCommand.includes('help')) {
          router.push('/help')
        } else {
          // Always send voice input to child components regardless of current route
          window.dispatchEvent(new CustomEvent('voice-input', { detail: command }));
        }
        
        setTimeout(() => {
          showVoiceStatus.value = false
        }, 3000)
      } catch (error) {
        errorHandler.handleError('Error processing voice command:', error)
      }
    }
    
    // Helper function
    const delay = (ms) => {
      return new Promise(resolve => setTimeout(resolve, ms))
    }
    
    // Life cycle hooks
    onMounted(() => {
      checkServerConnection();
      initSpeechRecognition();
    })
    
    onUnmounted(() => {
      // Cleanup resources
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
      }
      
      if (ws && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
    })
    
    return {
      showVoiceInput,
      recognitionInProgress,
      isConnected,
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
