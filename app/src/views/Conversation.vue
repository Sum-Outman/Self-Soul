<template>
  <div class="conversation-container">
    <div class="conversation-header">
      <h1>Self Soul - Intelligent Conversation</h1>
      <div class="header-controls">
        <button @click="toggleConversationHistory" class="history-btn" :class="{ 'active': showConversationHistory }">
          {{ showConversationHistory ? 'Hide History' : 'Show History' }}
        </button>
        <div class="model-status">
          <span class="status-indicator" :class="modelConnectionStatus"></span>
          <span class="status-text">{{ connectionStatusText }}</span>
        </div>
        <button @click="toggleEmotionVisualization" class="emotion-viz-btn" :class="{ 'active': showEmotionVisualization }">
          {{ showEmotionVisualization ? 'Hide Emotions' : 'Show Emotions' }}
        </button>
        <button @click="toggleSettingsPanel" class="settings-btn" :class="{ 'active': showSettingsPanel }">
          {{ showSettingsPanel ? 'Hide Settings' : 'Settings' }}
        </button>
        <button @click="clearConversation" class="clear-btn">Clear Conversation</button>
        <button @click="saveCurrentConversation" :disabled="messages.length === 0" class="save-btn">
          Save Conversation
        </button>
      </div>
    </div>
    
    <!-- Settings Panel -->
    <div v-if="showSettingsPanel" class="settings-panel">
      <div class="settings-header">
        <h3>Conversation Settings</h3>
      </div>
      
      <div class="settings-content">
        <div class="setting-section">
          <h4>Model Configuration</h4>
          <div class="setting-item">
            <label for="model-select">Model Selection:</label>
            <select id="model-select" v-model="selectedModel" class="settings-select">
              <option v-for="model in availableModels" :key="model.id" :value="model.id">
                {{ model.name }} (Port: {{ model.port }})
              </option>
            </select>
          </div>
        </div>
        
        <div class="setting-section">
          <h4>Generation Parameters</h4>
          <div class="setting-item">
            <label for="temperature-slider">
              Temperature: <span class="setting-value">{{ temperature.toFixed(1) }}</span>
            </label>
            <input 
              id="temperature-slider"
              type="range" 
              v-model="temperature"
              min="0" 
              max="2" 
              step="0.1"
              class="settings-slider"
            >
            <div class="slider-labels">
              <span>Precise (0)</span>
              <span>Balanced (1)</span>
              <span>Creative (2)</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label for="max-tokens">Max Tokens:</label>
            <input 
              id="max-tokens"
              type="number" 
              v-model.number="maxTokens"
              min="1" 
              max="8192" 
              step="1"
              class="settings-input"
            >
            <span class="setting-hint">Maximum length of response</span>
          </div>
        </div>
        
        <div class="setting-section">
          <h4>System Prompt</h4>
          <div class="setting-item">
            <label for="system-prompt">System Instructions:</label>
            <textarea 
              id="system-prompt"
              v-model="systemPrompt"
              rows="4"
              class="settings-textarea"
              placeholder="Enter system instructions for the model..."
            ></textarea>
            <span class="setting-hint">This defines how the model should behave</span>
          </div>
        </div>
        
        <div class="setting-section">
          <h4>Voice Output</h4>
          <div class="setting-item">
            <label class="checkbox-label">
              <input 
                type="checkbox" 
                v-model="enableVoiceOutput"
                class="settings-checkbox"
              >
              Enable voice output for responses
            </label>
            <span class="setting-hint">Model responses will be spoken aloud when enabled</span>
          </div>
        </div>
        
        <div class="settings-actions">
          <button @click="saveConversationSettings" class="settings-save-btn">
            Save Settings
          </button>
          <button @click="toggleSettingsPanel" class="settings-close-btn">
            Close
          </button>
        </div>
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
    
    <!-- Conversation History Panel -->
    <div v-if="showConversationHistory" class="conversation-history-panel">
      <div class="history-header">
        <h3>Conversation History</h3>
        <button @click="loadConversationHistory" class="refresh-btn">
          Refresh
        </button>
      </div>
      
      <div v-if="conversationHistory.length === 0" class="no-history">
        <p>No saved conversations yet.</p>
        <p>Start a conversation and save it to see it here.</p>
      </div>
      
      <div v-else class="history-list">
        <div 
          v-for="conversation in conversationHistory" 
          :key="conversation.id"
          class="history-item"
          :class="{ 'active': currentConversationId === conversation.id }"
          @click="loadConversation(conversation.id)"
        >
          <div class="history-item-header">
            <span class="history-title">{{ conversation.title || 'Untitled Conversation' }}</span>
            <span class="history-time">{{ formatHistoryTimestamp(conversation.created_at) }}</span>
          </div>
          <div class="history-item-preview">
            <span class="preview-text">{{ conversation.preview || 'No preview available' }}</span>
            <span class="message-count">{{ conversation.message_count }} messages</span>
          </div>
          <div class="history-item-actions">
            <button @click.stop="deleteConversation(conversation.id)" class="delete-btn">
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Emotion Visualization Panel -->
    <div v-if="showEmotionVisualization" class="emotion-visualization-panel">
      <div class="emotion-header">
        <h3>Emotion Analysis</h3>
        <button @click="updateEmotionChart" class="refresh-btn">
          Refresh
        </button>
      </div>
      
      <div class="emotion-stats">
        <div class="stat-item">
          <span class="stat-label">Total Messages:</span>
          <span class="stat-value">{{ messages.length }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">With Emotion:</span>
          <span class="stat-value">{{ messages.filter(msg => msg.emotion && msg.sender === 'model').length }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Most Common:</span>
          <span class="stat-value">{{ getMostCommonEmotion() || 'None' }}</span>
        </div>
      </div>
      
      <div class="emotion-chart-container">
        <div v-if="!emotionChartData" class="no-data-message">
          <p>Emotion analysis will appear here once you start a conversation.</p>
        </div>
        <div class="chart-wrapper" v-else>
          <canvas ref="emotionChartCanvas"></canvas>
        </div>
      </div>
      
      <div class="emotion-details">
        <h4>Emotion Details</h4>
        <div class="emotion-list">
          <div 
            v-for="(emotion, index) in getEmotionList()" 
            :key="emotion.name"
            class="emotion-detail-item"
          >
            <div class="emotion-bar">
              <div class="emotion-bar-fill" :style="{ width: emotion.percentage + '%' }"></div>
            </div>
            <div class="emotion-info">
              <span class="emotion-name">{{ emotion.name }}</span>
              <span class="emotion-count">{{ emotion.count }} ({{ emotion.percentage }}%)</span>
            </div>
          </div>
        </div>
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
        <button @click="toggleCameraCapture" :disabled="isLoading || isRecording" class="camera-capture-btn" :class="{ 'active': isCameraActive }">
          {{ isCameraActive ? 'Stop Camera' : 'Camera' }}
        </button>
        <button @click="toggleAudioRecording" :disabled="isLoading || isCameraActive" class="audio-record-btn" :class="{ 'active': isRecording }">
          {{ isRecording ? 'Stop Recording' : 'Record' }}
        </button>
        <button @click="openFileUpload" :disabled="isLoading || isUploading" class="file-upload-btn">
          {{ isUploading ? 'Uploading...' : 'Upload File' }}
        </button>
      </div>
      
      <!-- Hidden file upload input -->
      <input 
        type="file" 
        ref="fileUploadRef" 
        @change="handleFileUpload" 
        multiple
        accept="image/*,audio/*,video/*,.pdf,.txt,.doc,.docx"
        style="display: none;"
      >
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
import { ref, onMounted, computed, watch, onUnmounted } from 'vue';
import api from '@/utils/api';
import { Chart } from 'chart.js/auto';
import { notify } from '@/plugins/notification';

// Refs
const isMounted = ref(true);
const chatContainer = ref(null);
const emotionChartCanvas = ref(null);
const emotionChart = ref(null);

// Data
const messages = ref([]);
const newMessage = ref('');
const modelConnectionStatus = ref('connecting');
const isLoading = ref(false);
const error = ref(null);

// Conversation history management
const conversationHistory = ref([]);
const currentConversationId = ref(null);
const showConversationHistory = ref(false);

// Local storage key for conversation history
const CONVERSATION_STORAGE_KEY = 'self_soul_conversation_history';

// Helper functions for localStorage
const saveConversationHistoryToStorage = () => {
  try {
    const data = {
      conversations: conversationHistory.value,
      lastSaved: new Date().toISOString()
    };
    localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(data));
  } catch (error) {
    console.error('Failed to save conversation history to localStorage:', error);
  }
};

const loadConversationHistoryFromStorage = () => {
  try {
    const stored = localStorage.getItem(CONVERSATION_STORAGE_KEY);
    if (stored) {
      const data = JSON.parse(stored);
      if (data.conversations && Array.isArray(data.conversations)) {
        conversationHistory.value = data.conversations;
        return true;
      }
    }
  } catch (error) {
    console.error('Failed to load conversation history from localStorage:', error);
  }
  return false;
};

// Emotion visualization
const showEmotionVisualization = ref(true);
const emotionChartData = ref(null);

// Conversation settings
const showSettingsPanel = ref(false);
const selectedModel = ref('management-model');
const temperature = ref(0.7);
const maxTokens = ref(2048);
const availableModels = ref([]);
const systemPrompt = ref('You are Self Soul, a comprehensive AGI system. Respond to users thoughtfully with emotional awareness.');

// File upload
const fileUploadRef = ref(null);
const uploadedFiles = ref([]);
const isUploading = ref(false);

// Speech recognition
const recognition = ref(null);
const transcript = ref('');
const isSpeechRecognitionActive = ref(false);

// Camera and audio recording
const videoStream = ref(null);
const audioStream = ref(null);
const isCameraActive = ref(false);
const isRecording = ref(false);
const recordedChunks = ref([]);
const mediaRecorder = ref(null);
const cameraCanvas = ref(null);
const cameraPreviewUrl = ref(null);
const enableVoiceOutput = ref(true);

// Computed
const connectionStatusText = computed(() => {
  const statusMap = {
    'connected': 'Connected',
    'connecting': 'Connecting',
    'disconnected': 'Disconnected'
  };
  return statusMap[modelConnectionStatus.value] || 'Unknown Status';
});

// Watch for model selection changes
watch(selectedModel, (newModelId, oldModelId) => {
  if (newModelId !== oldModelId) {
    console.log(`Model selection changed from ${oldModelId} to ${newModelId}`);
    updateModelConnection();
  }
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

// Generic conversation operation function
const performConversationOperation = async (options) => {
  const {
    apiCall,
    loadingRef,
    errorRef,
    successMessage,
    errorMessage,
    errorContext,
    showSuccess = true,
    showError = true,
    onBeforeStart,
    onSuccess,
    onError,
    onFinally,
    autoScroll = true
  } = options

  if (loadingRef) loadingRef.value = true
  if (errorRef) errorRef.value = null

  // Execute before start callback if provided
  if (onBeforeStart && typeof onBeforeStart === 'function') {
    await onBeforeStart()
  }

  try {
    const response = await apiCall()
    
    if (onSuccess && typeof onSuccess === 'function') {
      await onSuccess(response.data)
    }
    
    // Auto-scroll to bottom if enabled
    if (autoScroll) {
      scrollToBottom()
    }
    
    return response.data
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error(errorContext || 'Conversation operation error:', error)
    }
    
    if (errorRef) errorRef.value = error
    
    if (onError && typeof onError === 'function') {
      await onError(error)
    }
    
    throw error
  } finally {
    if (loadingRef) loadingRef.value = false
    if (onFinally && typeof onFinally === 'function') {
      await onFinally()
    }
  }
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
  error.value = null;
  
  try {
    await performConversationOperation({
      apiCall: () => api.managementChat({
        message: userMessage,
        message_type: messageType,
        history: messages.value.slice(-10).map(msg => ({
          content: msg.content,
          type: msg.type,
          sender: msg.sender
        }))
      }),
      loadingRef: isLoading,
      errorRef: error,
      errorContext: 'Send Message',
      autoScroll: true,
      onSuccess: (data) => {
        if (import.meta.env.DEV) {
          console.log('API Response data:', data);
        }
        
        // Check response status
        if (data.status === 'success') {
          // Success response
          const modelResponse = {
            id: Date.now() + 1,
            sender: 'model',
            content: data.data.response,
            type: data.data.response_type || 'text',
            timestamp: Date.now(),
            emotion: data.data.emotion,
            confidence: data.data.confidence
          };
          
          messages.value.push(modelResponse);
          modelConnectionStatus.value = 'connected';
          
          // Synthesize and play speech for model response
          if (data.data.response && typeof data.data.response === 'string') {
            synthesizeAndPlaySpeech(data.data.response, data.data.emotion || {});
          }
        } else if (data.status === 'error') {
          // Error response from backend
          const errorResponse = data.data;
          error.value = `Model error: ${errorResponse.response}`;
          
          // Add error message
          messages.value.push({
            id: Date.now() + 1,
            sender: 'model',
            content: errorResponse.response || 'Model returned an error without details.',
            type: 'text',
            timestamp: Date.now(),
            isError: true
          });
          
          modelConnectionStatus.value = 'disconnected';
          throw new Error(`Model returned error: ${errorResponse.response}`);
        } else {
          // Unknown response format
          throw new Error(`Unknown response status: ${data.status}`);
        }
      },
      onError: (error) => {
        console.error('Failed to send message:', error);
        error.value = `Cannot connect to management model: ${error.message}`;
        
        // Add error message
        messages.value.push({
          id: Date.now() + 1,
          sender: 'model',
          content: `Connection failed: ${error.message || 'Unable to reach the model service'}`,
          type: 'text',
          timestamp: Date.now(),
          isError: true
        });
        
        modelConnectionStatus.value = 'disconnected';
      }
    });
  } catch (error) {
    // Error is already handled in onError callback
    if (import.meta.env.DEV) {
      console.log('Send message operation completed with error:', error);
    }
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

// Synthesize and play speech from text
async function synthesizeAndPlaySpeech(text, emotion = {}) {
  try {
    // Check if voice output is enabled
    if (!enableVoiceOutput.value) {
      return;
    }
    
    if (!text || text.trim() === '') {
      console.warn('No text provided for speech synthesis');
      return;
    }
    
    // Call speech synthesis API
    const response = await api.process.speech({
      text: text,
      voice: 'neutral',
      speed: 1.0,
      language: 'en',
      emotion: emotion
    });
    
    if (response.data.status === 'success') {
      // Get audio data from response
      const audioData = response.data.data;
      
      // Check if audio data is base64 encoded
      if (typeof audioData === 'string' && audioData.startsWith('data:audio/')) {
        // Already a data URL
        const audio = new Audio(audioData);
        audio.play().catch(error => {
          console.error('Error playing audio:', error);
        });
      } else if (typeof audioData === 'string') {
        // Assume base64 encoded audio without data URL prefix
        const audioBlob = base64ToBlob(audioData, 'audio/wav');
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play().catch(error => {
          console.error('Error playing audio:', error);
        });
        
        // Clean up URL after playback
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
        };
      } else if (audioData instanceof ArrayBuffer || audioData.buffer instanceof ArrayBuffer) {
        // Handle ArrayBuffer data
        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play().catch(error => {
          console.error('Error playing audio:', error);
        });
        
        // Clean up URL after playback
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
        };
      } else {
        console.warn('Unsupported audio data format:', typeof audioData);
      }
    } else {
      console.error('Speech synthesis API returned error:', response.data);
    }
  } catch (error) {
    console.error('Failed to synthesize speech:', error);
  }
}

// Helper function to convert base64 to Blob
function base64ToBlob(base64Data, contentType = '', sliceSize = 512) {
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
  
  return new Blob(byteArrays, { type: contentType });
}

// Fetch available models for selection
async function fetchAvailableModels() {
  try {
    // Try multiple API endpoints to get models
    let models = [];
    
    // First try the models.get() endpoint
    try {
      const response = await api.models.get();
      if (response.data && Array.isArray(response.data)) {
        models = response.data;
      } else if (response.data && response.data.models && Array.isArray(response.data.models)) {
        models = response.data.models;
      }
    } catch (error) {
      console.log('models.get() endpoint failed, trying models.getAll()...', error);
    }
    
    // If first attempt failed, try getAll endpoint
    if (models.length === 0) {
      try {
        const response = await api.models.getAll();
        if (response.data && Array.isArray(response.data)) {
          models = response.data;
        } else if (response.data && response.data.models && Array.isArray(response.data.models)) {
          models = response.data.models;
        }
      } catch (error) {
        console.log('models.getAll() endpoint failed, trying models.available()...', error);
      }
    }
    
    // If still no models, try available endpoint
    if (models.length === 0) {
      try {
        const response = await api.models.available();
        if (response.data && Array.isArray(response.data)) {
          models = response.data;
        } else if (response.data && response.data.models && Array.isArray(response.data.models)) {
          models = response.data.models;
        }
      } catch (error) {
        console.log('models.available() endpoint failed', error);
      }
    }
    
    // Format models for display
    if (models.length > 0) {
      availableModels.value = models.map(model => ({
        id: model.id || model.modelId || `model-${model.port}`,
        name: model.name || model.modelName || `Model ${model.port}`,
        port: model.port || 8001,
        type: model.type || 'local',
        active: model.active || model.isActive || false
      }));
      
      console.log(`Loaded ${availableModels.value.length} models for selection`);
      
      // If no model is selected yet, select the first one
      if (!selectedModel.value && availableModels.value.length > 0) {
        selectedModel.value = availableModels.value[0].id;
        updateModelConnection();
      }
    } else {
      console.log('No models available from API endpoints');
      // No models available - set error state and clear models
      availableModels.value = [];
      selectedModel.value = '';
      error.value = 'No models available from server';
      console.error('No models available from API endpoints');
    }
  } catch (err) {
    console.error('Error fetching available models:', err);
    // Set error state and clear models instead of using fallback
    availableModels.value = [];
    selectedModel.value = '';
    error.value = `Failed to load models: ${err.message}`;
  }
}

// Speech recognition methods
async function startSpeechRecognition() {
  // Reset previous errors
  error.value = null;
  
  // Check browser support
  const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognitionAPI) {
    error.value = 'Speech recognition is not supported in your current browser. Please use Chrome or Edge.';
    return;
  }
  
  // Check page protocol (some browsers restrict microphone access over HTTP)
  if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    error.value = 'Speech recognition may be restricted over HTTP. Please use HTTPS or local environment (localhost/127.0.0.1).';
    return;
  }
  
  // Ensure previous recognition instance is stopped
  if (recognition.value) {
    try {
      recognition.value.stop();
    } catch (e) {
      if (import.meta.env.DEV) {
        console.log('Previous speech recognition instance stopped or does not exist');
      }
    }
    recognition.value = null;
  }
  
  // Request microphone permission
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error('Microphone permission error:', err);
    let errorMessage;
    
    switch (err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        errorMessage = 'Microphone permission denied. Please enable microphone access in your browser settings and try again.\n\nInstructions:\n1. Click the lock icon on the left side of the address bar\n2. Select "Site settings"\n3. Choose "Allow" for microphone access';
        break;
      case 'NotFoundError':
        errorMessage = 'No microphone device detected. Please connect a microphone and try again.';
        break;
      case 'NotReadableError':
        errorMessage = 'Microphone is being used by another application. Please close that application and try again.';
        break;
      case 'AbortError':
        errorMessage = 'Permission request was interrupted. Please try again.';
        break;
      case 'TypeError':
        errorMessage = 'Failed to access microphone device. Please check device connections and browser settings.';
        break;
      default:
        errorMessage = `Failed to access microphone: ${err.message}. Please check device settings and try again.`;
    }
    
    error.value = errorMessage;
    return;
  }
  
  // Initialize speech recognition
  try {
    recognition.value = new SpeechRecognitionAPI();
    recognition.value.continuous = true;
    recognition.value.interimResults = true;
    recognition.value.lang = 'en-US'; // Set to English recognition
    recognition.value.maxAlternatives = 1;
    
    recognition.value.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcriptText = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcriptText;
          // Add final transcript to input field
          newMessage.value = finalTranscript;
          transcript.value = '';
        } else {
          interimTranscript += transcriptText;
          transcript.value = interimTranscript;
        }
      }
    };
    
    recognition.value.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      let errorMessage;
      
      switch (event.error) {
        case 'not-allowed':
          errorMessage = 'Microphone permission denied. Please enable microphone access in your browser settings and try again.\n\nInstructions:\n1. Click the lock icon on the left side of the address bar\n2. Select "Site settings"\n3. Choose "Allow" for microphone access';
          break;
        case 'no-speech':
          errorMessage = 'No speech detected. Please speak clearly into the microphone.';
          break;
        case 'aborted':
          errorMessage = 'Speech recognition was interrupted. Please try again.';
          break;
        case 'audio-capture':
          errorMessage = 'Audio capture failed. Please check your microphone settings.';
          break;
        case 'network':
          errorMessage = 'Network error. Speech recognition requires internet connection. Please check your internet connection.';
          break;
        case 'service-not-allowed':
          errorMessage = 'Speech recognition service is not available, possibly due to browser settings or network restrictions.';
          break;
        case 'bad-grammar':
          errorMessage = 'Speech recognition grammar error. Please try again.';
          break;
        case 'language-not-supported':
          errorMessage = 'Current language is not supported. Please try a different language setting.';
          break;
        default:
          errorMessage = `Speech recognition error: ${event.error}`;
      }
      
      error.value = errorMessage;
      isSpeechRecognitionActive.value = false;
      
      // Ensure recognition service is stopped
      if (recognition.value) {
        try {
          recognition.value.stop();
        } catch (e) {
          if (import.meta.env.DEV) {
            console.log('Error stopping speech recognition instance:', e);
          }
        }
        recognition.value = null;
      }
    };
    
    recognition.value.onend = () => {
      if (import.meta.env.DEV) {
        console.log('Speech recognition has ended');
      }
      isSpeechRecognitionActive.value = false;
      recognition.value = null;
    };
    
    recognition.value.onstart = () => {
      if (import.meta.env.DEV) {
        console.log('Speech recognition has started successfully');
      }
      isSpeechRecognitionActive.value = true;
    };
    
    recognition.value.start();
  } catch (err) {
    console.error('Failed to initialize speech recognition:', err);
    error.value = `Failed to start speech recognition: ${err.message}. Please try again.`;
    isSpeechRecognitionActive.value = false;
    
    // Ensure recognition service is stopped
    if (recognition.value) {
      try {
        recognition.value.stop();
      } catch (e) {
        if (import.meta.env.DEV) {
          console.log('Error cleaning up speech recognition instance:', e);
        }
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

function toggleConversationHistory() {
  showConversationHistory.value = !showConversationHistory.value;
}

function toggleEmotionVisualization() {
  showEmotionVisualization.value = !showEmotionVisualization.value;
  if (showEmotionVisualization.value) {
    updateEmotionChart();
  }
}

function toggleSettingsPanel() {
  showSettingsPanel.value = !showSettingsPanel.value;
}

function saveConversationSettings() {
  try {
    console.log('Save settings button clicked');
    
    // Save settings to localStorage
    const settings = {
      selectedModel: selectedModel.value,
      temperature: temperature.value,
      maxTokens: maxTokens.value,
      systemPrompt: systemPrompt.value,
      savedAt: new Date().toISOString()
    };
    
    console.log('Saving settings:', settings);
    localStorage.setItem('self-soul-conversation-settings', JSON.stringify(settings));
    console.log('Settings saved to localStorage');
    
    // Show success message
    error.value = null;
    
    // Try to show notification
    try {
      notify.success('Conversation settings saved successfully!');
    } catch (notifyErr) {
      console.warn('Notification system error:', notifyErr);
      // Fallback: show alert if notify fails
      alert('Settings saved successfully!');
    }
    
    // Update model connection based on selected model
    updateModelConnection();
    console.log('Settings save completed');
  } catch (err) {
    console.error('Failed to save settings:', err);
    error.value = 'Failed to save conversation settings';
    alert('Failed to save settings: ' + err.message);
  }
}

function loadConversationSettings() {
  try {
    const savedSettings = localStorage.getItem('self-soul-conversation-settings');
    if (savedSettings) {
      const settings = JSON.parse(savedSettings);
      selectedModel.value = settings.selectedModel || 'management-model';
      temperature.value = settings.temperature || 0.7;
      maxTokens.value = settings.maxTokens || 2048;
      systemPrompt.value = settings.systemPrompt || 'You are Self Soul, a comprehensive AGI system. Respond to users thoughtfully with emotional awareness.';
    }
  } catch (err) {
    console.error('Failed to load settings:', err);
    // Use default values
  }
}

function updateModelConnection() {
  const selectedModelInfo = availableModels.value.find(model => model.id === selectedModel.value);
  if (selectedModelInfo) {
    // Update API base URL based on selected model port
    // Note: This assumes the frontend can dynamically adjust API endpoints
    // In a real implementation, you would update the API client configuration
    if (import.meta.env.DEV) {
      console.log(`Switching to ${selectedModelInfo.name} on port ${selectedModelInfo.port}`);
    }
    // Here you would update the API client configuration
  }
  // Re-check connection
  checkModelConnection();
}

// Conversation history methods
async function loadConversationHistory() {
  try {
    // First try to load from localStorage
    const loaded = loadConversationHistoryFromStorage();
    if (loaded) {
      console.log('Conversation history loaded from localStorage');
      return;
    }
    
    // If no localStorage data, try to load from API
    await performConversationOperation({
      apiCall: () => api.knowledge.files(),
      errorRef: error,
      errorContext: 'Load Conversation History',
      autoScroll: false,
      onSuccess: (data) => {
        if (data.files && Array.isArray(data.files)) {
          // Convert API files to conversation format
          conversationHistory.value = data.files.map(file => ({
            id: file.id,
            title: file.name,
            created_at: file.upload_date,
            preview: file.content?.substring(0, 100) || 'No preview',
            message_count: file.message_count || 1
          }));
        }
      },
      onError: (error) => {
        console.error('Failed to load conversation history from API:', error);
        conversationHistory.value = [];
      }
    });
  } catch (error) {
    console.error('Failed to load conversation history:', error);
    conversationHistory.value = [];
  }
}

async function saveCurrentConversation() {
  if (messages.value.length === 0) return;
  
  try {
    const conversationData = {
      id: Date.now(),
      title: `Conversation ${new Date().toLocaleString()}`,
      messages: JSON.parse(JSON.stringify(messages.value)), // Deep copy
      created_at: new Date().toISOString(),
      preview: messages.value[0]?.content?.substring(0, 100) || 'No preview',
      message_count: messages.value.length
    };
    
    // Add to conversation history
    conversationHistory.value.unshift(conversationData);
    
    // Save to localStorage
    saveConversationHistoryToStorage();
    
    // Show success message
    error.value = null;
    notify.success('Conversation saved successfully!');
  } catch (error) {
    console.error('Failed to save conversation:', error);
    error.value = 'Failed to save conversation';
  }
}

async function loadConversation(conversationId) {
  try {
    const conversation = conversationHistory.value.find(c => c.id === conversationId);
    if (conversation && conversation.messages) {
      // Load actual messages from saved conversation
      messages.value = conversation.messages;
      currentConversationId.value = conversationId;
      error.value = null;
      showSystemMessage(`Loaded conversation: ${conversation.title}`);
    } else {
      error.value = 'Conversation not found or has no messages';
    }
  } catch (error) {
    console.error('Failed to load conversation:', error);
    error.value = 'Failed to load conversation';
  }
}

async function deleteConversation(conversationId) {
  if (!confirm('Are you sure you want to delete this conversation?')) return;
  
  try {
    conversationHistory.value = conversationHistory.value.filter(c => c.id !== conversationId);
    
    // Update localStorage
    saveConversationHistoryToStorage();
    
    if (currentConversationId.value === conversationId) {
      currentConversationId.value = null;
      messages.value = [];
    }
    
    error.value = null;
    showSystemMessage('Conversation deleted successfully');
  } catch (error) {
    console.error('Failed to delete conversation:', error);
    error.value = 'Failed to delete conversation';
  }
}

function showSystemMessage(message) {
  messages.value.push({
    id: Date.now(),
    sender: 'system',
    content: message,
    type: 'text',
    timestamp: Date.now()
  });
}

function formatHistoryTimestamp(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric',
    year: 'numeric'
  });
}

// File upload methods
function openFileUpload() {
  if (fileUploadRef.value) {
    fileUploadRef.value.click();
  }
}

async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;
  
  isUploading.value = true;
  error.value = null;
  
  try {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.knowledge.upload(formData);
      
      uploadedFiles.value.push({
        id: Date.now() + i,
        name: file.name,
        type: file.type,
        size: file.size,
        url: URL.createObjectURL(file)
      });
      
      // If it's an image or audio, add as message
      if (file.type.startsWith('image/')) {
        newMessage.value = URL.createObjectURL(file);
        sendMessage();
      } else if (file.type.startsWith('audio/')) {
        newMessage.value = URL.createObjectURL(file);
        sendMessage();
      }
    }
  } catch (error) {
    console.error('File upload failed:', error);
    error.value = 'Failed to upload file(s)';
  } finally {
    isUploading.value = false;
    if (fileUploadRef.value) {
      fileUploadRef.value.value = '';
    }
  }
}

// Camera and audio recording methods
async function toggleCameraCapture() {
  if (isCameraActive.value) {
    stopCameraCapture();
  } else {
    startCameraCapture();
  }
}

async function startCameraCapture() {
  try {
    error.value = null;
    
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      error.value = 'Camera access is not supported in your current browser.';
      return;
    }
    
    // Request camera permission
    videoStream.value = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'environment'
      }
    });
    
    isCameraActive.value = true;
    error.value = null;
    
    // Create canvas for capturing frames
    if (!cameraCanvas.value) {
      cameraCanvas.value = document.createElement('canvas');
    }
    
    // Create video element for preview (but don't show it)
    const video = document.createElement('video');
    video.srcObject = videoStream.value;
    video.play();
    
    // Capture frame every 3 seconds
    const captureFrame = () => {
      if (!isCameraActive.value || !isMounted.value) return;
      
      const canvas = cameraCanvas.value;
      const video = document.createElement('video');
      video.srcObject = videoStream.value;
      
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      cameraPreviewUrl.value = canvas.toDataURL('image/jpeg', 0.8);
      
      // Automatically send the captured image every 5 seconds
      setTimeout(captureFrame, 5000);
    };
    
    // Start capturing after a short delay
    setTimeout(captureFrame, 1000);
    
  } catch (err) {
    console.error('Failed to start camera:', err);
    handleCameraError(err);
  }
}

function stopCameraCapture() {
  if (videoStream.value) {
    videoStream.value.getTracks().forEach(track => track.stop());
    videoStream.value = null;
  }
  isCameraActive.value = false;
  cameraPreviewUrl.value = null;
}

function handleCameraError(err) {
  let errorMessage;
  
  switch (err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      errorMessage = 'Camera permission denied. Please enable camera access in your browser settings.';
      break;
    case 'NotFoundError':
      errorMessage = 'No camera device detected. Please connect a camera and try again.';
      break;
    case 'NotReadableError':
      errorMessage = 'Camera is being used by another application. Please close that application and try again.';
      break;
    case 'AbortError':
      errorMessage = 'Permission request was interrupted. Please try again.';
      break;
    case 'TypeError':
      errorMessage = 'Failed to access camera device. Please check device connections.';
      break;
    default:
      errorMessage = `Failed to access camera: ${err.message}.`;
  }
  
  error.value = errorMessage;
  isCameraActive.value = false;
}

async function toggleAudioRecording() {
  if (isRecording.value) {
    stopAudioRecording();
  } else {
    startAudioRecording();
  }
}

async function startAudioRecording() {
  try {
    error.value = null;
    
    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      error.value = 'Microphone access is not supported in your current browser.';
      return;
    }
    
    // Request microphone permission
    audioStream.value = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44100
      }
    });
    
    // Setup MediaRecorder
    recordedChunks.value = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm') 
      ? 'audio/webm;codecs=opus'
      : 'audio/mp4';
    
    mediaRecorder.value = new MediaRecorder(audioStream.value, { mimeType });
    
    mediaRecorder.value.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.value.push(event.data);
      }
    };
    
    mediaRecorder.value.onstop = () => {
      const audioBlob = new Blob(recordedChunks.value, { type: mimeType });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Send the recorded audio as message
      newMessage.value = audioUrl;
      sendMessage();
      
      // Clean up
      recordedChunks.value = [];
      audioStream.value.getTracks().forEach(track => track.stop());
      audioStream.value = null;
    };
    
    mediaRecorder.value.start(1000); // Collect data every second
    isRecording.value = true;
    error.value = null;
    
  } catch (err) {
    console.error('Failed to start audio recording:', err);
    handleAudioError(err);
  }
}

function stopAudioRecording() {
  if (mediaRecorder.value && isRecording.value) {
    mediaRecorder.value.stop();
  }
  isRecording.value = false;
}

function handleAudioError(err) {
  let errorMessage;
  
  switch (err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      errorMessage = 'Microphone permission denied. Please enable microphone access in your browser settings.';
      break;
    case 'NotFoundError':
      errorMessage = 'No microphone device detected. Please connect a microphone and try again.';
      break;
    case 'NotReadableError':
      errorMessage = 'Microphone is being used by another application. Please close that application and try again.';
      break;
    case 'AbortError':
      errorMessage = 'Permission request was interrupted. Please try again.';
      break;
    case 'TypeError':
      errorMessage = 'Failed to access microphone device. Please check device connections.';
      break;
    default:
      errorMessage = `Failed to access microphone: ${err.message}.`;
  }
  
  error.value = errorMessage;
  isRecording.value = false;
}

// Camera capture snapshot function
function captureCameraSnapshot() {
  if (!isCameraActive.value || !cameraCanvas.value) return null;
  
  const canvas = cameraCanvas.value;
  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
  
  // Send the snapshot as message
  newMessage.value = dataUrl;
  sendMessage();
  
  return dataUrl;
}

// Emotion visualization methods
function updateEmotionChart() {
  const emotionMessages = messages.value.filter(msg => msg.emotion && msg.sender === 'model');
  if (emotionMessages.length === 0) {
    emotionChartData.value = null;
    // Destroy chart if it exists
    if (emotionChart.value) {
      emotionChart.value.destroy();
      emotionChart.value = null;
    }
    return;
  }
  
  const emotionCounts = {};
  emotionMessages.forEach(msg => {
    emotionCounts[msg.emotion] = (emotionCounts[msg.emotion] || 0) + 1;
  });
  
  emotionChartData.value = {
    labels: Object.keys(emotionCounts),
    datasets: [{
      label: 'Emotion Distribution',
      data: Object.values(emotionCounts),
      backgroundColor: ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8']
    }]
  };
  
  // Initialize or update chart
  initializeEmotionChart();
}

function initializeEmotionChart() {
  if (!emotionChartCanvas.value || !emotionChartData.value) return;
  
  // Destroy existing chart if it exists
  if (emotionChart.value) {
    emotionChart.value.destroy();
  }
  
  // Create new chart
  const ctx = emotionChartCanvas.value.getContext('2d');
  emotionChart.value = new Chart(ctx, {
    type: 'doughnut',
    data: emotionChartData.value,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 20,
            usePointStyle: true,
            pointStyle: 'circle'
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.label || '';
              const value = context.raw || 0;
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
              return `${label}: ${value} (${percentage}%)`;
            }
          }
        }
      },
      animation: {
        animateScale: true,
        animateRotate: true
      }
    }
  });
}

function getMostCommonEmotion() {
  if (!emotionChartData.value || !emotionChartData.value.labels || emotionChartData.value.labels.length === 0) {
    return null;
  }
  
  const labels = emotionChartData.value.labels;
  const data = emotionChartData.value.datasets[0].data;
  let maxIndex = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[maxIndex]) {
      maxIndex = i;
    }
  }
  
  return labels[maxIndex];
}

function getEmotionList() {
  if (!emotionChartData.value || !emotionChartData.value.labels || emotionChartData.value.labels.length === 0) {
    return [];
  }
  
  const labels = emotionChartData.value.labels;
  const data = emotionChartData.value.datasets[0].data;
  const total = data.reduce((sum, count) => sum + count, 0);
  
  return labels.map((label, index) => ({
    name: label,
    count: data[index],
    percentage: total > 0 ? Math.round((data[index] / total) * 100) : 0
  })).sort((a, b) => b.count - a.count);
}

// Lifecycle
onMounted(() => {
  isMounted.value = true;
  loadConversationSettings();
  // Fetch available models first, then check connection
  fetchAvailableModels();
  checkModelConnection();
  loadConversationHistory();
});

onUnmounted(() => {
  isMounted.value = false;
  
  // Clean up emotion chart
  if (emotionChart.value) {
    emotionChart.value.destroy();
  }
  
  // Clean up camera resources
  if (videoStream.value) {
    videoStream.value.getTracks().forEach(track => track.stop());
    videoStream.value = null;
  }
  
  // Clean up audio recording resources
  if (audioStream.value) {
    audioStream.value.getTracks().forEach(track => track.stop());
    audioStream.value = null;
  }
  
  // Clean up media recorder
  if (mediaRecorder.value && mediaRecorder.value.state !== 'inactive') {
    mediaRecorder.value.stop();
    mediaRecorder.value = null;
  }
  
  // Clean up speech recognition
  if (recognition.value && recognition.value.stop) {
    try {
      recognition.value.stop();
    } catch (error) {
      console.warn('Failed to stop speech recognition:', error);
    }
    recognition.value = null;
  }
  
  // Clean up camera canvas
  if (cameraCanvas.value) {
    cameraCanvas.value = null;
  }
  
  // Reset camera active state to stop captureFrame timeouts
  isCameraActive.value = false;
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
  transition: all 0.3s ease;
}

/* Adjust chat container when history panel is visible */
.conversation-container:has(.conversation-history-panel) .chat-container {
  margin-right: 350px;
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

.file-upload-btn {
  padding: 0.75rem 1.5rem;
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 120px;
}

.file-upload-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3);
  background: #5a6268;
}

.file-upload-btn:disabled {
  background: #adb5bd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.camera-capture-btn {
  padding: 0.75rem 1.5rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 100px;
}

.camera-capture-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
  background: #0069d9;
}

.camera-capture-btn:disabled {
  background: #adb5bd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.camera-capture-btn.active {
  background: #dc3545;
  animation: pulse 1.5s infinite;
}

.camera-capture-btn.active:hover:not(:disabled) {
  box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
  background: #c82333;
}

.audio-record-btn {
  padding: 0.75rem 1.5rem;
  background: #17a2b8;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
  min-width: 100px;
}

.audio-record-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(23, 162, 184, 0.3);
  background: #138496;
}

.audio-record-btn:disabled {
  background: #adb5bd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.audio-record-btn.active {
  background: #dc3545;
  animation: pulse 1.5s infinite;
}

.audio-record-btn.active:hover:not(:disabled) {
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

/* Conversation history panel styles */
.conversation-history-panel {
  position: fixed;
  top: 70px;
  right: 0;
  width: 350px;
  height: calc(100vh - 70px);
  background: rgba(255, 255, 255, 0.98);
  border-left: 1px solid #e0e0e0;
  box-shadow: -4px 0 16px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  z-index: 1000;
  overflow-y: auto;
  padding: 1rem;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #e9ecef;
}

.history-header h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #2c3e50;
}

.refresh-btn,
.history-btn,
.save-btn {
  padding: 0.5rem 1rem;
  background: #e9ecef;
  color: #495057;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.refresh-btn:hover,
.history-btn:hover,
.save-btn:hover:not(:disabled) {
  background: #dee2e6;
  color: #212529;
}

.history-btn.active {
  background: #007bff;
  color: white;
  border-color: #007bff;
}

.save-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.no-history {
  text-align: center;
  padding: 2rem 1rem;
  color: #6c757d;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.history-item {
  padding: 1rem;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.history-item:hover {
  background: #e9ecef;
  border-color: #dee2e6;
}

.history-item.active {
  background: #e7f1ff;
  border-color: #007bff;
}

.history-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.history-title {
  font-weight: 600;
  color: #2c3e50;
  font-size: 0.95rem;
}

.history-time {
  font-size: 0.8rem;
  color: #6c757d;
}

.history-item-preview {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.preview-text {
  font-size: 0.85rem;
  color: #495057;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}

.message-count {
  font-size: 0.8rem;
  color: #6c757d;
  background: #e9ecef;
  padding: 0.125rem 0.5rem;
  border-radius: 12px;
}

.history-item-actions {
  display: flex;
  justify-content: flex-end;
}

.delete-btn {
  padding: 0.25rem 0.75rem;
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s ease;
}

.delete-btn:hover {
  background: #f1b0b7;
  border-color: #ed969e;
}

/* Emotion visualization panel styles */
.emotion-visualization-panel {
  position: fixed;
  top: 70px;
  left: 0;
  width: 350px;
  height: calc(100vh - 70px);
  background: rgba(255, 255, 255, 0.98);
  border-right: 1px solid #e0e0e0;
  box-shadow: 4px 0 16px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  z-index: 1000;
  overflow-y: auto;
  padding: 1rem;
}

.emotion-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #e9ecef;
}

.emotion-header h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #2c3e50;
}

.emotion-stats {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-label {
  font-size: 0.9rem;
  color: #495057;
  font-weight: 500;
}

.stat-value {
  font-size: 0.9rem;
  color: #2c3e50;
  font-weight: 600;
}

.emotion-chart-container {
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.chart-placeholder {
  text-align: center;
  padding: 2rem 1rem;
  color: #6c757d;
}

.chart-wrapper {
  height: 200px;
}

.emotion-details {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.emotion-details h4 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
}

.emotion-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.emotion-detail-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.emotion-bar {
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.emotion-bar-fill {
  height: 100%;
  background: #007bff;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.emotion-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.emotion-name {
  font-size: 0.85rem;
  color: #495057;
  font-weight: 500;
  text-transform: capitalize;
}

.emotion-count {
  font-size: 0.8rem;
  color: #6c757d;
}

.emotion-viz-btn {
  padding: 0.5rem 1rem;
  background: #e9ecef;
  color: #495057;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.emotion-viz-btn:hover {
  background: #dee2e6;
  color: #212529;
}

.emotion-viz-btn.active {
  background: #ffc107;
  color: #212529;
  border-color: #ffc107;
}

/* Settings panel styles */
.settings-panel {
  position: fixed;
  top: 70px;
  left: 50%;
  transform: translateX(-50%);
  width: 90%;
  max-width: 800px;
  max-height: 80vh;
  background: rgba(255, 255, 255, 0.98);
  border: 1px solid #e0e0e0;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
  backdrop-filter: blur(20px);
  z-index: 2000;
  overflow-y: auto;
  padding: 0;
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translate(-50%, -20px);
  }
  to {
    opacity: 1;
    transform: translate(-50%, 0);
  }
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #e9ecef;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px 16px 0 0;
  color: white;
}

.settings-header h3 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.settings-content {
  padding: 2rem;
}

.setting-section {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #e9ecef;
}

.setting-section:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.setting-section h4 {
  margin: 0 0 1.5rem 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: #2c3e50;
}

.setting-item {
  margin-bottom: 1.5rem;
}

.setting-item:last-child {
  margin-bottom: 0;
}

.setting-item label {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 1rem;
  font-weight: 500;
  color: #495057;
}

.setting-value {
  display: inline-block;
  margin-left: 0.5rem;
  padding: 0.25rem 0.75rem;
  background: #e9ecef;
  border-radius: 12px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #495057;
}

.settings-select {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #dee2e6;
  border-radius: 12px;
  font-size: 1rem;
  font-family: inherit;
  background: white;
  color: #495057;
  transition: all 0.2s ease;
  cursor: pointer;
}

.settings-select:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}

.settings-slider {
  width: 100%;
  height: 8px;
  margin: 0.5rem 0;
  -webkit-appearance: none;
  appearance: none;
  background: #e9ecef;
  border-radius: 4px;
  outline: none;
}

.settings-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #007bff;
  cursor: pointer;
  border: 3px solid white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  transition: all 0.2s ease;
}

.settings-slider::-webkit-slider-thumb:hover {
  transform: scale(1.1);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
}

.slider-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #6c757d;
}

.settings-input {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #dee2e6;
  border-radius: 12px;
  font-size: 1rem;
  font-family: inherit;
  background: white;
  color: #495057;
  transition: all 0.2s ease;
}

.settings-input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}

.settings-textarea {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #dee2e6;
  border-radius: 12px;
  font-size: 1rem;
  font-family: inherit;
  background: white;
  color: #495057;
  transition: all 0.2s ease;
  resize: vertical;
}

.settings-textarea:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25);
}

.setting-hint {
  display: block;
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #6c757d;
  font-style: italic;
}

.settings-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e9ecef;
}

.settings-save-btn {
  padding: 0.75rem 2rem;
  background: #28a745;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.settings-save-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(40, 167, 69, 0.3);
  background: #218838;
}

.settings-close-btn {
  padding: 0.75rem 2rem;
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.settings-close-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(108, 117, 125, 0.3);
  background: #5a6268;
}

/* Settings button in header */
.settings-btn {
  padding: 0.5rem 1rem;
  background: #17a2b8;
  color: white;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s ease;
}

.settings-btn:hover {
  background: #138496;
  color: white;
}

.settings-btn.active {
  background: #dc3545;
  border-color: #dc3545;
  animation: pulse 1.5s infinite;
}

/* Adjust chat container when emotion panel is visible */
.conversation-container:has(.emotion-visualization-panel) .chat-container {
  margin-left: 350px;
}

/* Adjust when both panels are visible */
.conversation-container:has(.conversation-history-panel):has(.emotion-visualization-panel) .chat-container {
  margin-left: 350px;
  margin-right: 350px;
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
