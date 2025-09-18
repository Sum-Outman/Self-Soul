<template>
  <div id="app">
    <!-- System Top Menu Bar -->
    <div class="top-menu-bar">
      <div class="menu-left">
        <span class="system-title">{{ $t('main.title') }}</span>
      </div>
      <div class="menu-right">
        <!-- Language Switch Dropdown -->
        <select v-model="currentLanguage" @change="changeLanguage" class="language-select">
          <option value="en">{{ $t('language.english') }}</option>
          <option value="zh">{{ $t('language.chinese') }}</option>
          <option value="de">{{ $t('language.german') }}</option>
          <option value="ja">{{ $t('language.japanese') }}</option>
          <option value="ru">{{ $t('language.russian') }}</option>
        </select>
        
        <!-- Function Buttons -->
        <router-link to="/" class="menu-link">
          {{ $t('main.tabs.interaction') || '交互对话' }}
        </router-link>
        <router-link to="/training" class="menu-link">
          {{ $t('main.tabs.training') }}
        </router-link>
        <router-link to="/knowledge" class="menu-link">
          {{ $t('main.tabs.knowledge') || '知识管理' }}
        </router-link>
        <router-link to="/settings" class="menu-link">
          {{ $t('main.tabs.settings') }}
        </router-link>
        <router-link to="/help" class="menu-link">
          {{ $t('main.tabs.help') }}
        </router-link>
        
        <!-- Server Connection Status -->
        <div class="connection-status">
          <span class="status-indicator" :class="{ 'connected': isConnected, 'disconnected': !isConnected }"></span>
          <span class="status-text">{{ $t(isConnected ? 'main.status.connected' : 'main.status.disconnected') }}</span>
        </div>
      </div>
    </div>

    <router-view/>
    
    <!-- 语音输入浮动按钮 | Voice Input Floating Button -->
    <div class="voice-input-container" v-if="showVoiceInput">
      <button @click="toggleVoiceInput" class="voice-btn" :class="{ 'listening': recognitionInProgress }">
        <span v-if="!recognitionInProgress">🎤</span>
        <span v-else class="pulse-animation">🎤</span>
      </button>
      <div class="voice-status" v-if="recognitionInProgress">
        {{ $t('voice.listening') || '正在聆听...' }}
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { switchLanguage, detectAndSetBrowserLanguage } from './i18n.js'
import i18n from './i18n.js'
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
    const currentLanguage = ref('zh');
    const showVoiceInput = ref(true);
    const showVoiceStatus = ref(false);
    const voiceStatusMessage = ref('');
    const speechRecognition = ref(null);
    const recognitionInProgress = ref(false);
    const isConnected = ref(false); // 服务器连接状态，默认为断开
    
    // 模拟检查服务器连接状态
    const checkServerConnection = () => {
      try {
        // 这里应该是实际的WebSocket连接检查逻辑
        // 目前使用随机模拟连接状态
        const randomConnection = Math.random() > 0.3; // 70%的概率模拟连接成功
        isConnected.value = randomConnection;
        
        // 每5秒检查一次连接状态
        setTimeout(checkServerConnection, 5000);
      } catch (error) {
        errorHandler.handleError('检查服务器连接时出错:', error);
        isConnected.value = false;
      }
    };
    // 移除监控标签，保持界面简洁
    
    // 初始化语言设置
    const initializeLanguage = () => {
      try {
        // 优先使用已保存的语言偏好
        const savedLanguage = localStorage.getItem('user-language') || localStorage.getItem('agi_language')
        if (savedLanguage && ['zh', 'en', 'de', 'ja', 'ru'].includes(savedLanguage)) {
          currentLanguage.value = savedLanguage
          // 确保i18n使用正确的语言
          switchLanguage(savedLanguage)
        } else {
          // 自动检测浏览器语言
          detectAndSetBrowserLanguage()
          // 设置当前语言为检测到的语言或默认中文
          currentLanguage.value = i18n.global.locale.value || 'zh'
          switchLanguage(currentLanguage.value)
        }
        
        errorHandler.logInfo('语言设置已初始化:', currentLanguage.value)
      } catch (error) {
        errorHandler.handleWarning('初始化语言设置时出错:', error)
        // 出错时默认使用中文
        currentLanguage.value = 'zh'
        switchLanguage('zh')
        localStorage.setItem('user-language', 'zh')
      }
    }
    
    // 初始化语音识别
    const initSpeechRecognition = () => {
      try {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
          const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
          speechRecognition.value = new SpeechRecognition()
          
          // 设置语音识别参数
          speechRecognition.value.lang = getSpeechLanguage()
          speechRecognition.value.interimResults = true
          speechRecognition.value.maxAlternatives = 1
          speechRecognition.value.continuous = false
          
          // 处理语音识别结果
          speechRecognition.value.onresult = (event) => {
            const transcript = event.results[0][0].transcript
            processVoiceCommand(transcript)
          }
          
          // 处理语音识别错误
          speechRecognition.value.onerror = (event) => {
            errorHandler.handleError('语音识别错误:', event.error)
            recognitionInProgress.value = false
            showVoiceStatus.value = true
            voiceStatusMessage.value = `语音识别错误: ${event.error}`
            setTimeout(() => {
              showVoiceStatus.value = false
            }, 3000)
          }
          
          // 处理语音识别结束
          speechRecognition.value.onend = () => {
            recognitionInProgress.value = false
            showVoiceStatus.value = true
            voiceStatusMessage.value = '语音识别已结束'
            setTimeout(() => {
              showVoiceStatus.value = false
            }, 2000)
          }
          
          // 处理语音识别开始
          speechRecognition.value.onstart = () => {
            recognitionInProgress.value = true
            showVoiceStatus.value = true
            voiceStatusMessage.value = '正在聆听...'
          }
        } else {
          errorHandler.handleWarning('当前浏览器不支持语音识别')
        }
      } catch (error) {
        errorHandler.handleError('初始化语音识别时出错:', error)
      }
    }
    
    // 初始化组件
    const initializeComponentsSilently = () => {
      try {
        errorHandler.logInfo('Self Soul 系统组件正在后台初始化...')
        
        // 模拟后台初始化过程
        Promise.all([
          delay(300), // 语言相关初始化
          delay(500), // 连接后端服务
          delay(800)  // 预加载必要的模型
        ]).then(() => {
          errorHandler.logInfo('Self Soul 系统组件初始化完成')
        }).catch(error => {
          errorHandler.handleError('系统组件初始化过程中出现错误:', error)
        })
      } catch (error) {
        errorHandler.handleError('系统组件初始化过程中出现错误:', error)
      }
    }
    
    // 语言切换
    const changeLanguage = () => {
      try {
        // 使用导入的switchLanguage函数
        const success = switchLanguage(currentLanguage.value)
        if (success) {
          // 使用一致的key 'user-language'，与i18n.js保持一致
          localStorage.setItem('user-language', currentLanguage.value)
          errorHandler.logInfo(`系统语言已切换至: ${currentLanguage.value}`)
          
          // 如果语音识别已初始化，更新语言设置
          if (speechRecognition.value) {
            speechRecognition.value.lang = currentLanguage.value
          }
        }
      } catch (error) {
        errorHandler.handleError('切换语言时出错:', error)
      }
    }
    
    // 语音识别相关方法
    const getSpeechLanguage = () => {
      const langMap = {
        'zh': 'zh-CN',
        'en': 'en-US',
        'de': 'de-DE',
        'ja': 'ja-JP',
        'ru': 'ru-RU'
      }
      return langMap[currentLanguage.value] || 'en-US'
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
        voiceStatusMessage.value = '正在聆听...'
      }
    }
    
    const processVoiceCommand = (command) => {
      try {
        errorHandler.logInfo('语音命令:', command)
        showVoiceStatus.value = true
        voiceStatusMessage.value = `已识别: ${command}`
        
        // 简单的命令处理逻辑
        const lowerCommand = command.toLowerCase()
        
        // 导航命令 - 只指向有效的路由
        if (lowerCommand.includes('主页') || lowerCommand.includes('home')) {
          router.push('/')
        } else if (lowerCommand.includes('训练') || lowerCommand.includes('train')) {
          router.push('/training')
        } else if (lowerCommand.includes('知识库') || lowerCommand.includes('knowledge')) {
          router.push('/knowledge')
        } else if (lowerCommand.includes('设置') || lowerCommand.includes('settings')) {
          router.push('/settings')
        } else if (lowerCommand.includes('帮助') || lowerCommand.includes('help')) {
          router.push('/help')
        } else if (router.currentRoute.value.path === '/') {
          // 如果在对话页面，将语音输入发送给子组件
          window.dispatchEvent(new CustomEvent('voice-input', { detail: command }));
        }
        
        setTimeout(() => {
          showVoiceStatus.value = false
        }, 3000)
      } catch (error) {
        errorHandler.handleError('处理语音命令时出错:', error)
      }
    }
    
    // 辅助函数
    const delay = (ms) => {
      return new Promise(resolve => setTimeout(resolve, ms))
    }
    
    // 生命周期钩子
    onMounted(() => {
      initializeLanguage()
      initSpeechRecognition()
      initializeComponentsSilently()
      checkServerConnection() // 开始检查服务器连接状态
      
      // 监听语言变化事件，确保语言选择器同步更新
      window.addEventListener('language-changed', (event) => {
        currentLanguage.value = event.detail.lang
      })
    })

    // 添加卸载时清理事件监听器
    onUnmounted(() => {
      window.removeEventListener('language-changed', (event) => {
        currentLanguage.value = event.detail.lang
      })
    })
    
    return {      currentLanguage,      showVoiceInput,      showVoiceStatus,      voiceStatusMessage,      recognitionInProgress,      isConnected,      changeLanguage,      toggleVoiceInput,      getSpeechLanguage,      processVoiceCommand    }
  }
}
</script>

<style>
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

/* 语言选择下拉菜单 */
.language-select {
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  cursor: pointer;
}

.language-select option {
  background: var(--bg-primary);
  color: var(--text-primary);
}

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
  background-color: #4CAF50; /* 绿色表示连接 */
  box-shadow: 0 0 8px rgba(76, 175, 80, 0.6);
}

.status-indicator.disconnected {
  background-color: #F44336; /* 红色表示断开 */
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
