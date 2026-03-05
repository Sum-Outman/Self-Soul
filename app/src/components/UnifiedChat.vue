<template>
  <div class="unified-chat">
    <!-- Chat Header -->
    <div class="chat-header">
      <h2>{{ title }}</h2>
      <div class="header-controls">
        <div class="model-status" v-if="showModelStatus">
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ modelConnectionStatusText }}</span>
        </div>
        <button @click="clearMessages" class="clear-btn" :disabled="messages.length === 0">
          Clear
        </button>
        <button @click="toggleSettings" class="settings-btn" :class="{ 'active': showSettings }">
          Settings
        </button>
        <button @click="toggleVideoDialog" v-if="showVideoDialog" class="video-btn" :class="{ 'active': showVideo }">
          {{ showVideo ? 'Hide Video' : 'Video' }}
        </button>
      </div>
    </div>

    <!-- Settings Panel -->
    <div v-if="showSettings" class="settings-panel">
      <div class="settings-section">
        <h4>Model Configuration</h4>
        <div class="setting-item">
          <label for="model-select">Model:</label>
          <select id="model-select" v-model="selectedModel" class="setting-select">
            <option v-for="model in availableModels" :key="model.id" :value="model.id">
              {{ model.name }}
            </option>
          </select>
        </div>
        <div class="setting-item">
          <label for="temperature">Temperature:</label>
          <input type="range" id="temperature" v-model="temperature" min="0" max="2" step="0.1" class="setting-slider">
          <span class="setting-value">{{ temperature.toFixed(1) }}</span>
        </div>
      </div>
    </div>

    <!-- Video Dialog Section -->
    <div v-if="showVideo && showVideoDialog" class="video-dialog-section">
      <div class="video-section-header">
        <h4>Video Dialogue</h4>
        <button @click="toggleVideoStream" class="video-control-btn">
          {{ isVideoActive ? 'Stop Video' : 'Start Video' }}
        </button>
      </div>
      <div class="video-preview" v-if="isVideoActive">
        <video ref="videoElement" autoplay playsinline></video>
      </div>
    </div>

    <!-- Chat Messages -->
    <div class="chat-messages" ref="messagesContainer">
      <div v-if="messages.length === 0" class="empty-messages">
        <p>No messages yet. Start a conversation!</p>
      </div>
      <div v-else>
        <div v-for="message in messages" :key="message.id" class="message-item" :class="message.type">
          <div class="message-header">
            <span class="message-sender">{{ getSenderName(message.type) }}</span>
            <span class="message-time">{{ formatTime(message.timestamp) }}</span>
          </div>
          <div class="message-content">{{ message.content }}</div>
          <div v-if="message.confidence !== undefined" class="message-confidence">
            <div class="confidence-bar">
              <div class="confidence-fill" :style="{ width: message.confidence * 100 + '%' }"></div>
            </div>
            <span class="confidence-value">{{ Math.round(message.confidence * 100) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="chat-input-area">
      <div class="input-row">
        <input
          type="text"
          v-model="inputText"
          @keyup.enter="sendMessage"
          placeholder="Type your message..."
          :disabled="isSending"
          class="message-input"
        />
        <button @click="sendMessage" :disabled="isSending || !inputText.trim()" class="send-button">
          <span v-if="isSending">Sending...</span>
          <span v-else>Send</span>
        </button>
      </div>
      <div class="input-options" v-if="showInputOptions">
        <button @click="toggleVoiceInput" class="input-option" :class="{ 'active': isVoiceActive }">
          {{ isVoiceActive ? 'Stop Voice' : 'Voice' }}
        </button>
        <button @click="selectFile" class="input-option">
          File
        </button>
        <input type="file" ref="fileInput" style="display: none" @change="handleFileUpload">
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import api from '@/utils/api'

export default {
  name: 'UnifiedChat',
  props: {
    title: {
      type: String,
      default: 'Conversation'
    },
    showModelStatus: {
      type: Boolean,
      default: true
    },
    showVideoDialog: {
      type: Boolean,
      default: false
    },
    showInputOptions: {
      type: Boolean,
      default: true
    },
    initialModel: {
      type: String,
      default: 'language'
    },
    mode: {
      type: String,
      default: 'standard' // 'standard', 'training', 'scratch'
    }
  },
  setup(props, { emit }) {
    // State
    const messages = ref([])
    const inputText = ref('')
    const isSending = ref(false)
    const showSettings = ref(false)
    const showVideo = ref(false)
    const isVideoActive = ref(false)
    const isVoiceActive = ref(false)
    const selectedModel = ref(props.initialModel)
    const temperature = ref(0.7)
    const modelConnectionStatus = ref('disconnected')
    const availableModels = ref([])
    const messagesContainer = ref(null)
    const videoElement = ref(null)
    const fileInput = ref(null)
    
    // Computed
    const modelConnectionStatusText = computed(() => {
      switch (modelConnectionStatus.value) {
        case 'connected': return 'Connected'
        case 'connecting': return 'Connecting...'
        default: return 'Disconnected'
      }
    })

    // Methods
    const getSenderName = (type) => {
      switch (type) {
        case 'user': return 'You'
        case 'bot': return 'AI'
        case 'system': return 'System'
        default: return 'Unknown'
      }
    }

    const formatTime = (timestamp) => {
      if (!timestamp) return ''
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    const addMessage = (content, type = 'user', confidence = null) => {
      const message = {
        id: Date.now(),
        content,
        type,
        timestamp: Date.now(),
        confidence
      }
      messages.value.push(message)
      scrollToBottom()
    }

    const clearMessages = () => {
      messages.value = []
      emit('messages-cleared')
    }

    const scrollToBottom = () => {
      nextTick(() => {
        if (messagesContainer.value) {
          messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
        }
      })
    }

    const sendMessage = async () => {
      const text = inputText.value.trim()
      if (!text || isSending.value) return

      isSending.value = true
      const userMessage = text
      inputText.value = ''
      
      // Add user message
      addMessage(userMessage, 'user')
      
      // Emit send-message event for parent to handle API call
      emit('send-message', {
        text: userMessage,
        model: selectedModel.value,
        temperature: temperature.value,
        timestamp: Date.now()
      })
      
      isSending.value = false
    }

    const toggleSettings = () => {
      showSettings.value = !showSettings.value
    }

    const toggleVideoDialog = () => {
      showVideo.value = !showVideo.value
    }

    const toggleVideoStream = async () => {
      if (!isVideoActive.value) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true })
          if (videoElement.value) {
            videoElement.value.srcObject = stream
          }
          isVideoActive.value = true
          emit('video-started')
          
          // Start video dialogue processing
          startVideoDialogue(stream)
          
        } catch (error) {
          addMessage(`Failed to access camera: ${error.message}`, 'system')
          emit('error', error)
        }
      } else {
        if (videoElement.value && videoElement.value.srcObject) {
          const tracks = videoElement.value.srcObject.getTracks()
          tracks.forEach(track => track.stop())
          videoElement.value.srcObject = null
        }
        isVideoActive.value = false
        emit('video-stopped')
        
        // Stop video dialogue processing
        stopVideoDialogue()
      }
    }

    const startVideoDialogue = (stream) => {
      // Set up canvas for capturing video frames
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      // Set canvas dimensions to match video
      if (videoElement.value) {
        canvas.width = videoElement.value.videoWidth || 640
        canvas.height = videoElement.value.videoHeight || 480
      }
      
      // Store the canvas and context
      window.videoCanvas = canvas
      window.videoContext = ctx
      window.videoInterval = null
      
      // Start capturing and processing video frames
      const processVideoFrame = async () => {
        if (!videoElement.value || !isVideoActive.value) return
        
        try {
          // Draw current video frame to canvas
          if (videoElement.value.videoWidth > 0 && videoElement.value.videoHeight > 0) {
            canvas.width = videoElement.value.videoWidth
            canvas.height = videoElement.value.videoHeight
            ctx.drawImage(videoElement.value, 0, 0, canvas.width, canvas.height)
            
            // Convert canvas to data URL (JPEG format, compressed)
            const imageData = canvas.toDataURL('image/jpeg', 0.7)
            
            // Send video frame to backend for processing
            await processVideoFrameWithAI(imageData)
          }
        } catch (error) {
          console.error('Error processing video frame:', error)
          // Don't stop the video on minor errors
        }
      }
      
      // Capture frames every 3 seconds (adjust as needed)
      window.videoInterval = setInterval(processVideoFrame, 3000)
      
      // Process first frame immediately
      setTimeout(() => processVideoFrame(), 500)
    }

    const stopVideoDialogue = () => {
      if (window.videoInterval) {
        clearInterval(window.videoInterval)
        window.videoInterval = null
      }
      if (window.videoCanvas) {
        window.videoCanvas = null
        window.videoContext = null
      }
    }

    const processVideoFrameWithAI = async (imageData) => {
      try {
        // Extract base64 data from data URL
        const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '')
        
        // Create FormData for file upload
        const formData = new FormData()
        const blob = await (await fetch(imageData)).blob()
        formData.append('video', blob, 'video-frame.jpg')
        formData.append('language', 'en')
        formData.append('session_id', `video-session-${Date.now()}`)
        formData.append('model_id', 'vision')  // Use vision model for video analysis
        
        // Send to video processing API
        const response = await fetch('/api/process/video', {
          method: 'POST',
          body: formData
        })
        
        if (!response.ok) {
          throw new Error(`Video processing failed: ${response.status}`)
        }
        
        const result = await response.json()
        
        if (result.status === 'success' && result.data) {
          // Process the AI response
          await handleVideoAIResponse(result.data)
        }
        
      } catch (error) {
        console.error('Error sending video frame to AI:', error)
        // Log error but don't disrupt video stream
        if (error.message.includes('failed') && !error.message.includes('minor')) {
          addMessage(`Video analysis error: ${error.message}`, 'system')
        }
      }
    }

    const handleVideoAIResponse = async (aiData) => {
      // Extract meaningful information from AI response
      let aiMessage = ''
      
      if (aiData.text) {
        // Direct text response
        aiMessage = aiData.text
      } else if (aiData.description) {
        // Image/video description
        aiMessage = `I see: ${aiData.description}`
      } else if (aiData.analysis) {
        // Analysis result
        aiMessage = `Analysis: ${JSON.stringify(aiData.analysis, null, 2)}`
      } else if (aiData.message) {
        // Generic message
        aiMessage = aiData.message
      } else if (typeof aiData === 'string') {
        // String response
        aiMessage = aiData
      } else {
        // Default response
        aiMessage = 'I\'m analyzing the video feed...'
      }
      
      // Limit message length and clean up
      if (aiMessage.length > 500) {
        aiMessage = aiMessage.substring(0, 497) + '...'
      }
      
      // Add AI response to chat
      addMessage(aiMessage, 'bot', 0.85)
      
      // Emit event for parent component
      emit('video-ai-response', {
        response: aiData,
        message: aiMessage,
        timestamp: Date.now()
      })
    }

    const toggleVoiceInput = () => {
      if (!isVoiceActive.value) {
        // Start voice input
        startVoiceRecognition()
      } else {
        // Stop voice input
        stopVoiceRecognition()
      }
    }

    const startVoiceRecognition = () => {
      try {
        // Check if browser supports speech recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
        if (!SpeechRecognition) {
          throw new Error('Speech recognition not supported in this browser')
        }

        // Initialize speech recognition
        const recognition = new SpeechRecognition()
        recognition.continuous = true
        recognition.interimResults = true
        recognition.lang = 'en-US'

        // Setup event handlers
        recognition.onstart = () => {
          isVoiceActive.value = true
          addMessage('Voice recognition started. Speak now...', 'system')
          emit('voice-started')
        }

        recognition.onresult = (event) => {
          let finalTranscript = ''
          let interimTranscript = ''

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript
            if (event.results[i].isFinal) {
              finalTranscript += transcript
            } else {
              interimTranscript += transcript
            }
          }

          // Update input field with interim results
          if (interimTranscript) {
            inputText.value = interimTranscript
          }

          // If we have final results, send them
          if (finalTranscript) {
            inputText.value = finalTranscript
            // Auto-send the message
            sendMessage()
            // Stop recognition after sending
            recognition.stop()
          }
        }

        recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error)
          addMessage(`Speech recognition error: ${event.error}`, 'system')
          emit('error', new Error(`Speech recognition error: ${event.error}`))
          isVoiceActive.value = false
        }

        recognition.onend = () => {
          isVoiceActive.value = false
          emit('voice-stopped')
        }

        // Store recognition instance for stopping
        window.currentRecognition = recognition
        recognition.start()
        
      } catch (error) {
        console.error('Failed to start voice recognition:', error)
        addMessage(`Failed to start voice recognition: ${error.message}`, 'system')
        emit('error', error)
        isVoiceActive.value = false
      }
    }

    const stopVoiceRecognition = () => {
      if (window.currentRecognition) {
        window.currentRecognition.stop()
        window.currentRecognition = null
      }
      isVoiceActive.value = false
      emit('voice-stopped')
    }

    const selectFile = () => {
      if (fileInput.value) {
        fileInput.value.click()
      }
    }

    const handleFileUpload = (event) => {
      const file = event.target.files[0]
      if (file) {
        emit('file-uploaded', file)
        addMessage(`File uploaded: ${file.name}`, 'system')
      }
    }

    const loadAvailableModels = async () => {
      try {
        // Replace with actual API call
        const response = await api.models.available()
        
        if (response.data && response.data.status === 'success' && response.data.data) {
          // Extract available models from API response
          availableModels.value = response.data.data.map(model => ({
            id: model.id || model.model_id,
            name: model.name || model.model_name || model.id || 'Unknown Model',
            description: model.description || '',
            port: model.port,
            status: model.status || 'unknown'
          }))
        } else {
          // Fallback to default models if API response is not as expected
          console.warn('API response format unexpected, using default models:', response.data)
          availableModels.value = [
            { id: 'language', name: 'Language Model' },
            { id: 'vision', name: 'Vision Model' },
            { id: 'audio', name: 'Audio Model' }
          ]
        }
      } catch (error) {
        console.error('Failed to load models:', error)
        emit('error', error)
        // Fallback to simulated models on error
        availableModels.value = [
          { id: 'language', name: 'Language Model' },
          { id: 'vision', name: 'Vision Model' },
          { id: 'audio', name: 'Audio Model' }
        ]
      }
    }

    const updateModelConnectionStatus = async () => {
      try {
        // Implement actual connection check
        const response = await api.models.testConnection()
        
        if (response.data && response.data.status === 'success') {
          modelConnectionStatus.value = 'connected'
        } else {
          modelConnectionStatus.value = 'disconnected'
        }
      } catch (error) {
        console.error('Model connection check failed:', error)
        modelConnectionStatus.value = 'disconnected'
      }
    }

    // Lifecycle
    onMounted(() => {
      loadAvailableModels()
      updateModelConnectionStatus()
      emit('mounted')
    })

    onUnmounted(() => {
      // Clean up video stream
      if (isVideoActive.value && videoElement.value && videoElement.value.srcObject) {
        const tracks = videoElement.value.srcObject.getTracks()
        tracks.forEach(track => track.stop())
      }
      
      // Clean up video dialogue processing
      stopVideoDialogue()
      
      // Clean up voice recognition
      stopVoiceRecognition()
      
      emit('unmounted')
    })

    // Public methods for parent component
    const addBotMessage = (content, confidence = null) => {
      addMessage(content, 'bot', confidence)
    }
    
    const addSystemMessage = (content) => {
      addMessage(content, 'system')
    }
    
    const setConnectionStatus = (status) => {
      modelConnectionStatus.value = status
    }

    return {
      // Refs
      messages,
      inputText,
      isSending,
      showSettings,
      showVideo,
      isVideoActive,
      isVoiceActive,
      selectedModel,
      temperature,
      modelConnectionStatus,
      availableModels,
      messagesContainer,
      videoElement,
      fileInput,
      
      // Computed
      modelConnectionStatusText,
      
      // Methods
      getSenderName,
      formatTime,
      addMessage,
      clearMessages,
      sendMessage,
      toggleSettings,
      toggleVideoDialog,
      toggleVideoStream,
      toggleVoiceInput,
      selectFile,
      handleFileUpload,
      addBotMessage,
      addSystemMessage,
      setConnectionStatus,
      
      // Video and Voice Control Methods
      stopVideoDialogue,
      stopVoiceRecognition,
      startVoiceRecognition
    }
  }
}
</script>

<style scoped>
.unified-chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #f5f5f5;
  border-radius: 8px;
  overflow: hidden;
}

.chat-header {
  background: #fff;
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-controls {
  display: flex;
  gap: 8px;
  align-items: center;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-indicator.connected {
  background: #4caf50;
}

.status-indicator.connecting {
  background: #ff9800;
}

.status-indicator.disconnected {
  background: #f44336;
}

.status-text {
  font-size: 14px;
  color: #666;
}

.clear-btn, .settings-btn, .video-btn {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.settings-btn.active, .video-btn.active {
  background: #2196f3;
  color: white;
  border-color: #2196f3;
}

.settings-panel {
  background: white;
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
}

.settings-section {
  margin-bottom: 16px;
}

.settings-section h4 {
  margin: 0 0 12px 0;
  color: #333;
}

.setting-item {
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.setting-item label {
  min-width: 100px;
  font-size: 14px;
  color: #555;
}

.setting-select, .setting-slider {
  flex: 1;
}

.setting-value {
  min-width: 30px;
  text-align: right;
  font-size: 14px;
  color: #666;
}

.video-dialog-section {
  background: white;
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
}

.video-section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.video-section-header h4 {
  margin: 0;
  color: #333;
}

.video-control-btn {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.video-control-btn:hover {
  background: #f5f5f5;
}

.video-preview {
  width: 100%;
  height: 200px;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
}

.video-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: white;
}

.empty-messages {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: #999;
  font-style: italic;
}

.message-item {
  margin-bottom: 16px;
  padding: 12px;
  border-radius: 8px;
  background: #f8f9fa;
}

.message-item.user {
  background: #e3f2fd;
  margin-left: 20%;
}

.message-item.bot {
  background: #f1f8e9;
  margin-right: 20%;
}

.message-item.system {
  background: #fff3e0;
}

.message-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
}

.message-sender {
  font-weight: 600;
  color: #333;
}

.message-time {
  font-size: 12px;
  color: #999;
}

.message-content {
  color: #333;
  line-height: 1.5;
}

.message-confidence {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.confidence-bar {
  flex: 1;
  height: 6px;
  background: #e0e0e0;
  border-radius: 3px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  background: #4caf50;
  transition: width 0.3s ease;
}

.confidence-value {
  font-size: 12px;
  color: #666;
  min-width: 40px;
}

.chat-input-area {
  background: white;
  padding: 16px;
  border-top: 1px solid #e0e0e0;
}

.input-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.message-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.message-input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.send-button {
  padding: 10px 20px;
  background: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.send-button:disabled {
  background: #bbdefb;
  cursor: not-allowed;
}

.input-options {
  display: flex;
  gap: 8px;
}

.input-option {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.input-option:hover {
  background: #f5f5f5;
}

.input-option.active {
  background: #2196f3;
  color: white;
  border-color: #2196f3;
}
</style>