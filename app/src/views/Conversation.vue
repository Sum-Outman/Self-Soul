<template>
  <div class="conversation-container">
    <div class="conversation-header">
      <h1>Conversation</h1>
      <div class="model-status">
        <span class="status-indicator" :class="modelConnectionStatus"></span>
        <span class="status-text">{{ modelConnectionStatus }}</span>
      </div>
    </div>
    
    <div class="chat-container" ref="chatContainer">
      <div v-if="messages.length === 0" class="empty-chat">
        <p>No messages yet</p>
      </div>
      
      <div v-for="message in messages" :key="message.id" class="message-wrapper">
        <div :class="['message', message.sender]">
          <div class="message-content">{{ message.content }}</div>
        </div>
      </div>
    </div>
    
    <div class="input-area">
      <input 
        type="text" 
        v-model="newMessage"
        @keyup.enter="sendMessage"
        placeholder="Type a message..."
      >
      <button @click="sendMessage" :disabled="!newMessage.trim()">Send</button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

// Refs
const chatContainer = ref(null);

// Data
const messages = ref([]);
const newMessage = ref('');
const modelConnectionStatus = ref('connected');

// Methods
function sendMessage() {
  if (!newMessage.value.trim()) return;
  
  // Add user message
  messages.value.push({
    id: Date.now(),
    sender: 'user',
    content: newMessage.value.trim(),
    timestamp: Date.now()
  });
  
  // Clear input
  newMessage.value = '';
  
  // Simulate model response
  setTimeout(() => {
    messages.value.push({
      id: Date.now() + 1,
      sender: 'model',
      content: "This is a mock response. I'm functioning correctly!",
      timestamp: Date.now()
    });
    scrollToBottom();
  }, 500);
  
  scrollToBottom();
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
}

// Lifecycle
onMounted(() => {
  // Initialize component
});
</script>

<style scoped>
.conversation-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f8f9fa;
  color: #333333;
}

.conversation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #ffffff;
  border-bottom: 1px solid #dddddd;
}

.conversation-header h1 {
  margin: 0;
  font-size: 1.5rem;
}

.model-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #f44336;
}

.status-indicator.connected {
  background-color: #4caf50;
}

.status-text {
  font-size: 0.9rem;
  color: #555555;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.empty-chat {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #888888;
  font-style: italic;
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
  border-radius: 12px;
  word-wrap: break-word;
}

.message.user {
  background-color: #3a7ca5;
  color: white;
}

.message.model {
  background-color: #ffffff;
  color: #333333;
  border: 1px solid #dddddd;
}

.input-area {
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background-color: #ffffff;
  border-top: 1px solid #dddddd;
}

.input-area input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #dddddd;
  border-radius: 4px;
  font-size: 1rem;
}

.input-area button {
  padding: 0.75rem 1.5rem;
  background-color: #3a7ca5;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

.input-area button:hover:not(:disabled) {
  background-color: #2c5e7a;
}

.input-area button:disabled {
  background-color: #888888;
  cursor: not-allowed;
}
</style>
