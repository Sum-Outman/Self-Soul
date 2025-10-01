<template>
  <div class="conversation-container">
    <!-- Header Section -->
    <div class="conversation-header">
      <h1>Conversation</h1>
      <div class="header-controls">
        <div class="model-selection">
          <label for="model-selector">Select Model:</label>
          <select id="model-selector" v-model="selectedModel" @change="onModelChange">
            <option value="language">Language Model</option>
            <option value="management">Management Model</option>
            <option value="from_scratch">From Scratch Model</option>
          </select>
        </div>
        <div class="model-status inline-status">
          <span class="model-name">{{ getModelDisplayName(selectedModel) }}</span>
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ modelConnectionStatus === 'connected' ? 'Connected' : modelConnectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected' }}</span>
        </div>
        <button @click="clearAllMessages" class="clear-btn" :disabled="messages.length === 0">
          Clear All Messages
        </button>
      </div>
    </div>

    <!-- Model Status Panel (collapsible) -->
    <div class="model-status-panel" v-if="showModelStatus">
      <h3>Model Information</h3>
      <div class="status-grid">
        <div class="status-item">
          <label>Model Type:</label>
          <span>{{ getModelDisplayName(selectedModel) }}</span>
        </div>
        <div class="status-item">
          <label>Training Mode:</label>
          <span>{{ modelInfo.trainingMode }}</span>
        </div>
        <div class="status-item">
          <label>Last Activity:</label>
          <span>{{ modelInfo.lastActivity }}</span>
        </div>
        <div class="status-item">
          <label>Confidence Level:</label>
          <div class="confidence-bar">
            <div class="confidence-fill" :style="{ width: modelInfo.confidence + '%' }"></div>
          </div>
          <span class="confidence-text">{{ modelInfo.confidence }}%</span>
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
            <span class="sender-name">{{ message.sender === 'user' ? 'You' : getModelDisplayName(selectedModel) }}</span>
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

    <!-- Input Area -->
    <div class="input-area">
      <div class="input-wrapper">
        <input 
          type="text" 
          class="message-input"
          v-model="newMessage"
          @keyup.enter="sendMessage"
          :disabled="isLoading || modelConnectionStatus !== 'connected'"
          placeholder="Type your message..."
          ref="messageInput"
        >
        <button 
          class="send-btn"
          @click="sendMessage"
          :disabled="!newMessage.trim() || isLoading || modelConnectionStatus !== 'connected'"
        >
          Send
        </button>
      </div>
      <div class="input-tips" v-if="modelConnectionStatus !== 'connected'">
        {{ modelConnectionStatus === 'disconnected' ? 'Not connected to backend. Please check your connection.' : 'Connecting to backend...' }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import api from '../utils/api.js'

// Reactive data
const messages = ref([])
const newMessage = ref('')
const isLoading = ref(false)
const chatContainer = ref(null)
const messageInput = ref(null)
const showModelStatus = ref(false)
const modelConnectionStatus = ref('connecting')
const selectedModel = ref('language')
const modelInfo = ref({
  trainingMode: 'Inactive',
  vocabSize: 0,
  epochs: 0,
  lastActivity: 'Never',
  confidence: 70
})
const sessionId = ref(`session_${Date.now()}`)
const conversationHistory = ref([])

// Helper functions
const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const getModelDisplayName = (modelType) => {
  const names = {
    language: 'Language Model',
    management: 'Management Model',
    from_scratch: 'From Scratch AI'
  }
  return names[modelType] || 'AI Assistant'
}

// Core functions
const initializeConversation = async () => {
  try {
    // Check backend connection
    await checkBackendConnection()
    
    // Load model info
    await loadModelInfo()
    
    // Add welcome message if no messages exist
    if (messages.value.length === 0) {
      addWelcomeMessage()
    }
  } catch (error) {
    console.error('Failed to initialize conversation:', error)
    modelConnectionStatus.value = 'disconnected'
    
    // Add fallback welcome message
    addWelcomeMessage(true)
  }
}

const checkBackendConnection = async () => {
  try {
    modelConnectionStatus.value = 'connecting'
    const response = await api.health.get()
    if (response.status === 200) {
      modelConnectionStatus.value = 'connected'
      return true
    }
    modelConnectionStatus.value = 'disconnected'
    return false
  } catch (error) {
    console.error('Backend connection check failed:', error)
    modelConnectionStatus.value = 'disconnected'
    return false
  }
}

const loadModelInfo = async () => {
  try {
    let response
    switch (selectedModel.value) {
      case 'from_scratch':
        response = await api.models.fromScratchStatus()
        break
      default:
        // For language and management models, use a generic status check
        response = await api.models.trainingStatus()
    }
    
    if (response.data) {
      modelInfo.value = {
        trainingMode: response.data.trainingMode || 'Inactive',
        vocabSize: response.data.vocabSize || 0,
        epochs: response.data.epochs || 0,
        lastActivity: response.data.lastActivity || 'Never',
        confidence: response.data.confidence || 70
      }
    }
  } catch (error) {
    console.error('Failed to load model info:', error)
    // Keep default values if loading fails
  }
}

const addWelcomeMessage = (isFallback = false) => {
  const welcomeMessage = {
    id: 1,
    sender: 'model',
    content: isFallback 
      ? `Hello! This is the ${getModelDisplayName(selectedModel.value)}. I'm here to assist you with your questions and tasks.`
      : `Hello! I'm the ${getModelDisplayName(selectedModel.value)}. How can I help you today?`,
    timestamp: new Date().toISOString(),
    confidence: 65,
    responseType: isFallback ? 'Fallback Welcome' : 'Welcome'
  }
  messages.value.push(welcomeMessage)
}

const sendMessage = async () => {
  const messageText = newMessage.value.trim()
  if (!messageText || isLoading.value || modelConnectionStatus.value !== 'connected') {
    return
  }
  
  // Add user message to the chat
  const userMessage = {
    id: Date.now(),
    sender: 'user',
    content: messageText,
    timestamp: new Date().toISOString()
  }
  messages.value.push(userMessage)
  
  // Clear input and disable while processing
  newMessage.value = ''
  isLoading.value = true
  
  // Scroll to bottom
  await nextTick()
  scrollToBottom()
  
  try {
    // Prepare request data based on model type
    const requestData = {
      message: messageText,
      session_id: sessionId.value,
      conversation_history: conversationHistory.value
    }
    
    // Determine which API endpoint to use based on selected model
    let response
    if (selectedModel.value === 'management') {
      // For management model, use specific endpoint with enhanced parameters
      const enhancedRequestData = {
        ...requestData,
        model_id: 'manager',
        query_type: 'text',
        confidence: 0.8,
        request_type: 'chat',
        user_id: 'conversation_user',
        timestamp: new Date().toISOString(),
        lang: 'en'
      }
      response = await api.post('/api/models/8001/chat', enhancedRequestData)
    } else if (selectedModel.value === 'from_scratch') {
      // For from scratch model, use language model endpoint with specific parameters
      const scratchRequestData = {
        ...requestData,
        model_type: 'from_scratch'
      }
      response = await api.post('/api/chat', scratchRequestData)
    } else {
      // For language model, use general chat endpoint
      response = await api.post('/api/chat', requestData)
    }
    
    // Process response with enhanced error handling
    if (response.data && response.data.status === 'success') {
      // Extract response data with fallback values
      const responseData = response.data.data || {}
      const aiResponse = responseData.response || `I'm the ${getModelDisplayName(selectedModel.value)}. I received your message: "${messageText}"`
      const confidence = responseData.confidence || Math.floor(Math.random() * 20) + 70
      const responseType = responseData.response_type || 'Text Response'
      
      // Add AI response to chat
      const modelMessage = {
        id: Date.now() + 1,
        sender: 'model',
        content: aiResponse,
        timestamp: new Date().toISOString(),
        confidence: confidence,
        responseType: responseType
      }
      messages.value.push(modelMessage)
      
      // Update conversation history with enhanced handling
      if (responseData.conversation_history && Array.isArray(responseData.conversation_history)) {
        conversationHistory.value = responseData.conversation_history
      } else {
        // Fallback: update conversation history manually
        conversationHistory.value.push({ role: 'user', content: messageText })
        conversationHistory.value.push({ role: 'assistant', content: aiResponse })
        
        // Limit conversation history to 50 messages
        if (conversationHistory.value.length > 50) {
          conversationHistory.value = conversationHistory.value.slice(-50)
        }
      }
      
      // Update model info based on response
      if (responseData.model_id || responseData.port || responseData.processing_time) {
        modelInfo.value.lastActivity = new Date().toLocaleTimeString()
        modelInfo.value.confidence = confidence
      }
    } else {
      throw new Error(response.data?.message || 'Invalid response from server')
    }
  } catch (error) {
    console.error('Error sending message:', error)
    
    // Enhanced error handling with specific error messages
    let errorMessage = 'Sorry, I couldn\'t process your request at the moment. Please try again later.'
    
    if (error.response) {
      // Server responded with error status
      if (error.response.status === 404) {
        errorMessage = 'The requested API endpoint was not found. Please check if the backend service is running.'
      } else if (error.response.status === 500) {
        errorMessage = 'Internal server error occurred. The AI model might be temporarily unavailable.'
      } else if (error.response.status === 503) {
        errorMessage = 'Service temporarily unavailable. Please try again in a few moments.'
      }
    } else if (error.request) {
      // Request was made but no response received
      errorMessage = 'Unable to connect to the AI backend. Please check your network connection and ensure the backend service is running.'
    }
    
    // Add error message to chat
    const errorSystemMessage = {
      id: Date.now() + 2,
      sender: 'system',
      content: errorMessage,
      timestamp: new Date().toISOString()
    }
    messages.value.push(errorSystemMessage)
  } finally {
    // Re-enable input
    isLoading.value = false
    
    // Focus input again
    if (messageInput.value) {
      messageInput.value.focus()
    }
    
    // Scroll to bottom
    await nextTick()
    scrollToBottom()
  }
}

const clearAllMessages = () => {
  messages.value = []
  conversationHistory.value = []
  addWelcomeMessage()
}

const scrollToBottom = () => {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

const onModelChange = () => {
  // Clear current messages and load new model info
  messages.value = []
  conversationHistory.value = []
  sessionId.value = `session_${Date.now()}`
  initializeConversation()
}

// Lifecycle hooks
onMounted(() => {
  initializeConversation()
  
  // Focus input after component is mounted
  if (messageInput.value) {
    messageInput.value.focus()
  }
  
  // Start periodic connection checks
  setInterval(() => {
    if (modelConnectionStatus.value === 'disconnected') {
      checkBackendConnection()
    }
  }, 5000)
})

// Expose to template
const toggleModelStatus = () => {
  showModelStatus.value = !showModelStatus.value
}

// Expose functions and variables to template
return {
  messages,
  newMessage,
  isLoading,
  chatContainer,
  messageInput,
  showModelStatus,
  modelInfo,
  modelConnectionStatus,
  selectedModel,
  formatTime,
  sendMessage,
  clearAllMessages,
  toggleModelStatus,
  getModelDisplayName,
  onModelChange
}
</script>

<style scoped>
/* CSS Variables for consistent styling */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --text-primary: #333333;
  --text-secondary: #666666;
  --text-tertiary: #999999;
  --border-color: #e0e0e0;
  --border-light: #f0f0f0;
  --accent-color: #333333;
  --accent-hover: #555555;
  --user-message-bg: #f0f0f0;
  --model-message-bg: #ffffff;
}

.conversation-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: var(--bg-secondary);
  color: var(--text-primary);
}

.conversation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem;
  background-color: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.conversation-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary);
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-selection {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.model-selection label {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.model-selection select {
  padding: 0.5rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-size: 0.9rem;
  cursor: pointer;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
}

.model-name {
  font-weight: 500;
  color: var(--text-secondary);
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--text-tertiary);
  transition: background-color 0.3s ease;
}

.status-indicator.connected {
  background-color: #4CAF50;
}

.status-indicator.connecting {
  background-color: #FFC107;
  animation: pulse 1.5s infinite;
}

.status-indicator.disconnected {
  background-color: #F44336;
}

.status-text {
  color: var(--text-tertiary);
  font-size: 0.85rem;
}

.clear-btn {
  padding: 0.5rem 1.5rem;
  background-color: var(--bg-primary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s ease;
}

.clear-btn:hover:not(:disabled) {
  background-color: var(--border-light);
  border-color: var(--text-tertiary);
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.model-status-panel {
  padding: 1rem 2rem;
  background-color: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
}

.model-status-panel h3 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 1rem 0;
  color: var(--text-primary);
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
  color: var(--text-tertiary);
  font-weight: 500;
}

.status-item span {
  font-size: 0.95rem;
  color: var(--text-secondary);
}

.confidence-bar {
  width: 100%;
  height: 6px;
  background-color: var(--border-light);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.25rem;
}

.confidence-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}

.confidence-text {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  background-color: var(--bg-secondary);
}

.chat-container::-webkit-scrollbar {
  width: 8px;
}

.chat-container::-webkit-scrollbar-track {
  background: var(--border-light);
  border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb {
  background: var(--text-tertiary);
  border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}

.empty-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-tertiary);
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
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.message:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.message.user {
  align-self: flex-end;
  background-color: var(--user-message-bg);
  color: var(--text-primary);
  border: 1px solid var(--border-light);
}

.message.model {
  align-self: flex-start;
  background-color: var(--model-message-bg);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
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
  color: var(--text-secondary);
}

.message-time {
  color: var(--text-tertiary);
}

.message-content {
  font-size: 1rem;
  line-height: 1.5;
  color: var(--text-primary);
}

.message-meta {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: var(--text-tertiary);
}

.confidence-indicator {
  background-color: var(--border-light);
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
}

.response-type {
  background-color: var(--border-light);
  padding: 0.2rem 0.5rem;
  border-radius: 3px;
  text-transform: uppercase;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  color: var(--text-secondary);
  font-style: italic;
}

.loader {
  width: 20px;
  height: 20px;
  border: 2px solid var(--border-light);
  border-top: 2px solid var(--accent-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.input-area {
  padding: 1.5rem 2rem;
  background-color: var(--bg-primary);
  border-top: 1px solid var(--border-color);
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.05);
}

.input-wrapper {
  display: flex;
  gap: 0.75rem;
  align-items: flex-end;
}

.message-input {
  flex: 1;
  padding: 1rem 1.25rem;
  border: 2px solid var(--border-color);
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  resize: none;
}

.message-input:focus {
  border-color: var(--accent-color);
  box-shadow: 0 0 0 3px rgba(51, 51, 51, 0.05);
}

.message-input:disabled {
  background-color: var(--border-light);
  color: var(--text-tertiary);
  cursor: not-allowed;
}

.send-btn {
  padding: 1rem 2rem;
  background-color: var(--accent-color);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: background-color 0.2s ease;
  white-space: nowrap;
}

.send-btn:hover:not(:disabled) {
  background-color: var(--accent-hover);
}

.send-btn:disabled {
  background-color: var(--text-tertiary);
  cursor: not-allowed;
}

.input-tips {
  margin-top: 0.75rem;
  text-align: center;
  color: var(--text-tertiary);
  font-size: 0.9rem;
}

/* Animations */
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .conversation-header {
    padding: 1rem;
    flex-direction: column;
    gap: 1rem;
    text-align: center;
  }
  
  .header-controls {
    justify-content: center;
    flex-wrap: wrap;
  }
  
  .model-selection {
    width: 100%;
    justify-content: center;
  }
  
  .model-selection select {
    width: 100%;
    max-width: 200px;
  }
  
  .chat-container {
    padding: 1rem;
  }
  
  .message {
    max-width: 90%;
    padding: 0.75rem;
  }
  
  .status-grid {
    grid-template-columns: 1fr;
  }
  
  .input-area {
    padding: 1rem;
  }
  
  .input-wrapper {
    flex-direction: column;
    align-items: stretch;
  }
  
  .send-btn {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .conversation-header h1 {
    font-size: 1.25rem;
  }
  
  .message-content {
    font-size: 0.9rem;
  }
  
  .message-header {
    font-size: 0.8rem;
  }
}
</style>
