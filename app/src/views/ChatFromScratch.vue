<template>
  <div class="chat-from-scratch">
    <!-- Header Section -->
    <div class="chat-header">
      <h1>Chat From Scratch</h1>
      <div class="header-controls">
        <button @click="clearChat" class="clear-btn">
          Clear Chat
        </button>
        <button @click="toggleTrainingStatus" class="status-btn">
          {{ showTrainingStatus ? 'Hide Status' : 'Show Status' }}
        </button>
      </div>
    </div>

    <!-- Training Status Panel (collapsible) -->
    <div class="training-status-panel" v-if="showTrainingStatus">
      <h3>Model Status</h3>
      <div class="status-grid">
        <div class="status-item">
          <label>Training Mode:</label>
          <span>{{ modelStatus.trainingMode }}</span>
        </div>
        <div class="status-item">
          <label>Vocabulary Size:</label>
          <span>{{ modelStatus.vocabSize }}</span>
        </div>
        <div class="status-item">
          <label>Training Epochs:</label>
          <span>{{ modelStatus.epochs }}</span>
        </div>
        <div class="status-item">
          <label>Last Activity:</label>
          <span>{{ modelStatus.lastActivity }}</span>
        </div>
        <div class="status-item">
          <label>Confidence Level:</label>
          <div class="confidence-bar">
            <div class="confidence-fill" :style="{ width: modelStatus.confidence + '%' }"></div>
          </div>
          <span class="confidence-text">{{ modelStatus.confidence }}%</span>
        </div>
      </div>
    </div>

    <!-- Chat Messages Container -->
    <div class="chat-container" ref="chatContainer">
      <div v-if="messages.length === 0" class="empty-chat">
        <p>No messages yet. Start a conversation!</p>
      </div>
      <div v-for="message in messages" :key="message.id" class="message-wrapper">
        <div :class="['message', message.sender === 'user' ? 'user' : 'model']">
          <div class="message-header">
            <span class="sender-name">{{ message.sender === 'user' ? 'You' : 'FromScratch AI' }}</span>
            <span class="message-time">{{ formatTime(message.timestamp) }}</span>
          </div>
          <div class="message-content">
            {{ message.content }}
          </div>
          <div v-if="message.sender === 'model'" class="message-meta">
            <span class="confidence-indicator" :title="'Confidence: ' + message.confidence + '%'">
              {{ message.confidence }}%
            </span>
            <span class="response-type" :title="'Response Type: ' + message.responseType">
              {{ message.responseType }}
            </span>
          </div>
        </div>
      </div>
      <!-- Loading indicator -->
      <div v-if="isLoading" class="loading-indicator">
        <div class="loader"></div>
        <span>AI is thinking...</span>
      </div>
    </div>

    <!-- Message Input Area -->
    <div class="message-input-area">
      <div class="input-wrapper">
        <input
          v-model="newMessage"
          @keyup.enter="sendMessage"
          type="text"
          placeholder="Type your message here..."
          class="message-input"
          :disabled="isLoading"
        >
        <button
          @click="sendMessage"
          class="send-btn"
          :disabled="isLoading || !newMessage.trim()"
        >
          Send
        </button>
      </div>
      <div class="input-tips">
        <small>Press Enter to send, or click the Send button</small>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick } from 'vue'
import api from '@/utils/api.js'
import errorHandler from '@/utils/errorHandler'

export default {
  name: 'ChatFromScratch',
  setup() {
    // State variables
    const messages = ref([])
    const newMessage = ref('')
    const isLoading = ref(false)
    const chatContainer = ref(null)
    const showTrainingStatus = ref(true)
    const modelStatus = ref({
      trainingMode: 'From Scratch',
      vocabSize: 0,
      epochs: 0,
      lastActivity: 'Never',
      confidence: 0
    })
    
    // Helper functions
    const formatTime = (timestamp) => {
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const scrollToBottom = async () => {
      await nextTick()
      if (chatContainer.value) {
        chatContainer.value.scrollTop = chatContainer.value.scrollHeight
      }
    }
    
    // Core methods
    const sendMessage = async () => {
      const messageText = newMessage.value.trim()
      if (!messageText || isLoading.value) return
      
      // Add user message
      const userMessage = {
        id: Date.now(),
        sender: 'user',
        content: messageText,
        timestamp: new Date().toISOString()
      }
      messages.value.push(userMessage)
      newMessage.value = ''
      
      scrollToBottom()
      isLoading.value = true
      
      try {
        // Send message to API
        const response = await api.post('/api/chat', {
          message: messageText,
          model_type: 'from_scratch'
        })
        
        // Update model status
        updateModelStatus(response.data.status)
        
        // Add model response with real data from API
        const modelMessage = {
          id: Date.now() + 1,
          sender: 'model',
          content: response.data.response || 'I don\'t have a response yet. I\'m still learning.',
          timestamp: new Date().toISOString(),
          confidence: response.data.confidence || 0, // Use real confidence value or 0 if not provided
          responseType: response.data.response_type || 'Generated'
        }
        messages.value.push(modelMessage)
        
      } catch (error) {
        // Handle error with fallback response
        errorHandler.handleError(error, 'ChatFromScratch', 'Failed to get AI response')
        
        // Add fallback response
        const fallbackMessage = {
          id: Date.now() + 1,
          sender: 'model',
          content: 'I apologize, but I encountered an error while processing your message. I\'m still learning how to respond properly.',
          timestamp: new Date().toISOString(),
          confidence: 30,
          responseType: 'Fallback'
        }
        messages.value.push(fallbackMessage)
      } finally {
        isLoading.value = false
        scrollToBottom()
      }
    }
    
    const updateModelStatus = (statusData) => {
      if (statusData) {
        modelStatus.value = {
          trainingMode: statusData.training_mode || 'From Scratch',
          vocabSize: statusData.vocab_size || 0,
          epochs: statusData.epochs || 0,
          lastActivity: statusData.last_activity || new Date().toLocaleString(),
          confidence: statusData.confidence || 0 // Use real confidence value or 0 if not provided
        }
      } else {
        // Initialize with zero values if no data
        modelStatus.value = {
          trainingMode: 'From Scratch',
          vocabSize: 0,
          epochs: 0,
          lastActivity: 'Never',
          confidence: 0
        }
      }
    }
    
    const clearChat = () => {
      messages.value = []
    }
    
    const toggleTrainingStatus = () => {
      showTrainingStatus.value = !showTrainingStatus.value
    }
    
    const initializeChat = async () => {
      try {
        // Fetch initial model status
        const response = await api.models.fromScratchStatus()
        updateModelStatus(response.data)
        
        // Add welcome message with real confidence data
        const welcomeMessage = {
          id: 1,
          sender: 'model',
          content: 'Hello! I am an AI learning from scratch. I don\'t have much knowledge yet, but I\'m eager to learn from our conversation!',
          timestamp: new Date().toISOString(),
          confidence: response.data.confidence || 0,
          responseType: 'Welcome'
        }
        messages.value.push(welcomeMessage)
        
      } catch (error) {
        errorHandler.handleError(error, 'ChatFromScratch', 'Failed to initialize chat')
        
        // Add fallback welcome message with zero confidence
        const fallbackWelcome = {
          id: 1,
          sender: 'model',
          content: 'Hello! I am an AI learning from scratch. Let\'s have a conversation!',
          timestamp: new Date().toISOString(),
          confidence: 0,
          responseType: 'Fallback Welcome'
        }
        messages.value.push(fallbackWelcome)
      }
    }
    
    // Lifecycle hooks
    onMounted(() => {
      initializeChat()
    })
    
    return {
      messages,
      newMessage,
      isLoading,
      chatContainer,
      showTrainingStatus,
      modelStatus,
      formatTime,
      sendMessage,
      clearChat,
      toggleTrainingStatus
    }
  }
}
</script>

<style scoped>
.chat-from-scratch {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f5f5;
  color: #333;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #333;
  color: white;
  border-bottom: 1px solid #555;
}

.chat-header h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 500;
}

.header-controls {
  display: flex;
  gap: 1rem;
}

.clear-btn, .status-btn {
  padding: 0.5rem 1rem;
  background-color: #555;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s;
}

.clear-btn:hover, .status-btn:hover {
  background-color: #666;
}

.training-status-panel {
  padding: 1rem 2rem;
  background-color: #e0e0e0;
  border-bottom: 1px solid #ccc;
}

.training-status-panel h3 {
  margin-top: 0;
  font-size: 1.1rem;
  color: #333;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.status-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.status-item label {
  font-size: 0.85rem;
  color: #666;
  font-weight: 500;
}

.status-item span {
  font-size: 0.9rem;
  color: #333;
}

.confidence-bar {
  height: 8px;
  background-color: #ccc;
  border-radius: 4px;
  overflow: hidden;
  margin: 0.25rem 0;
}

.confidence-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.confidence-text {
  font-size: 0.85rem;
  color: #4CAF50;
  font-weight: 500;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.empty-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
  font-style: italic;
}

.message-wrapper {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.message {
  max-width: 70%;
  padding: 1rem;
  border-radius: 8px;
  word-wrap: break-word;
}

.message.user {
  align-self: flex-end;
  background-color: #e3f2fd;
  color: #333;
}

.message.model {
  align-self: flex-start;
  background-color: white;
  color: #333;
  border: 1px solid #e0e0e0;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
}

.sender-name {
  font-weight: 600;
  color: #555;
}

.message-time {
  color: #999;
}

.message-content {
  font-size: 1rem;
  line-height: 1.5;
}

.message-meta {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #666;
}

.confidence-indicator {
  background-color: #f0f0f0;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
}

.response-type {
  background-color: #f0f0f0;
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  text-transform: uppercase;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  color: #666;
  font-style: italic;
}

.loader {
  width: 20px;
  height: 20px;
  border: 2px solid #f3f3f3;
  border-top: 2px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.message-input-area {
  padding: 1rem 2rem;
  background-color: white;
  border-top: 1px solid #e0e0e0;
}

.input-wrapper {
  display: flex;
  gap: 0.5rem;
}

.message-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}

.message-input:focus {
  border-color: #90caf9;
}

.message-input:disabled {
  background-color: #f5f5f5;
  color: #999;
}

.send-btn {
  padding: 0.75rem 2rem;
  background-color: #333;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: background-color 0.2s;
}

.send-btn:hover:not(:disabled) {
  background-color: #555;
}

.send-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.input-tips {
  margin-top: 0.5rem;
  text-align: center;
  color: #999;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .chat-header {
    padding: 1rem;
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .header-controls {
    justify-content: center;
  }
  
  .chat-container {
    padding: 1rem;
  }
  
  .message {
    max-width: 90%;
  }
  
  .status-grid {
    grid-template-columns: 1fr;
  }
}
</style>
