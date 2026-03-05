<template>
  <div id="app">
    <!-- Top Menu Bar -->
    <nav class="top-menu-bar">
      <div class="menu-left">
        <div class="system-title">Self Soul   (Email: silencecrowtom@qq.com)</div>
      </div>
      <div class="menu-right">
        <router-link to="/" class="menu-link" active-class="router-link-active">Home</router-link>
        <router-link to="/conversation" class="menu-link" active-class="router-link-active">Conversation</router-link>
        <router-link to="/training" class="menu-link" active-class="router-link-active">Training</router-link>
        <router-link to="/knowledge" class="menu-link" active-class="router-link-active">Knowledge</router-link>
        <router-link to="/autonomous-evolution" class="menu-link" active-class="router-link-active">Autonomous Evolution</router-link>
        <router-link to="/agi" class="menu-link" active-class="router-link-active">AGI Dashboard</router-link>
        <router-link to="/settings" class="menu-link" active-class="router-link-active">Settings</router-link>
        <router-link to="/help" class="menu-link" active-class="router-link-active">Help</router-link>
        <router-link to="/robot-settings" class="menu-link" active-class="router-link-active">Robot Settings</router-link>

        
        <!-- Server Connection Status -->
        <div class="connection-status" :style="{ color: connectionColor }">
          <span class="status-indicator" :style="{ backgroundColor: connectionColor }"></span>
          {{ connectionStatus }}
        </div>
      </div>
    </nav>

    <!-- Main Content Area -->
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { handleError } from '@/utils/errorHandler'
import api from '@/utils/api'
import { useSystemStore } from '@/stores/system'
import { useUiStore } from '@/stores/ui'

const router = useRouter()
const systemStore = useSystemStore()
const uiStore = useUiStore()

const connectionInterval = ref<number | null>(null)

// Connection status from system store
const connectionStatus = computed(() => {
  if (!systemStore.isConnected) return 'Disconnected'
  return 'Connected'
})

const connectionColor = computed(() => {
  return systemStore.isConnected ? '#555555' : '#888888'
})

// Check server connection status
const checkServerConnection = async () => {
  try {
    const response = await api.health.get()
    systemStore.setConnectionStatus(true)
    
    // If there's a new server message, show notification
    if (response.data && response.data.status) {
      console.log('Server connection established')
    }
  } catch (error) {
    systemStore.setConnectionStatus(false)
    console.error('Server connection error:', error)
  }
}

// Initialize components
const initializeComponentsSilently = () => {
  try {
    console.log('Self Soul System components are initializing...')
    
    // Directly log initialization completion
    console.log('Self Soul System components initialization completed')
  } catch (error) {
    console.error('Error during system components initialization:', error)
  }
}

// Named error handlers for proper removal in onUnmounted
const globalErrorHandler = (error: ErrorEvent) => {
  handleError(error.error || error, 'Global')
}

const unhandledRejectionHandler = (event: PromiseRejectionEvent) => {
  event.preventDefault()
  handleError(event.reason, 'Unhandled Promise')
}

// Life cycle hooks
onMounted(() => {
  // Register error handlers
  window.addEventListener('error', globalErrorHandler)
  window.addEventListener('unhandledrejection', unhandledRejectionHandler)
  
  // Periodically check server connection
  connectionInterval.value = window.setInterval(() => {
    checkServerConnection()
  }, 5000) // Check every 5 seconds
  
  // Check connection immediately
  checkServerConnection()
  
  // Initialize UI preferences
  uiStore.loadPreferences()
})

onUnmounted(() => {
  // Clear interval
  if (connectionInterval.value !== null) {
    clearInterval(connectionInterval.value)
  }
  
  // Remove event listeners properly
  window.removeEventListener('error', globalErrorHandler)
  window.removeEventListener('unhandledrejection', unhandledRejectionHandler)
  
  // Save UI preferences
  uiStore.savePreferences()
})
</script>

<style scoped>
/* Clean black-white-gray light style variables */
:root {
  --text-primary: #222;
  --text-secondary: #555;
  --text-disabled: #888;
  --bg-primary: #ffffff;
  --bg-secondary: #f8f8f8;
  --bg-tertiary: #f0f0f0;
  --border-color: #e0e0e0;
  --border-dark: #d0d0d0;
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.03);
  --border-radius-sm: 4px;
  --transition: all 0.2s ease;
}

/* Global style reset */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.5;
  color: var(--text-primary);
  background-color: var(--bg-primary);
  font-size: 14px;
}

/* App container styles */
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Top menu bar styles - clean black-white-gray light theme */
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
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.5px;
}

.menu-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Menu item styles */
.menu-link {
  padding: 8px 16px;
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  text-decoration: none;
  font-size: 14px;
  transition: var(--transition);
  display: inline-block;
  font-weight: 400;
}

.menu-link:hover {
  background: var(--bg-tertiary);
  border-color: var(--border-dark);
}

/* Connection status styles */
.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 14px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-text {
  font-size: 14px;
  color: var(--text-primary);
}

/* Add top margin to router-view to avoid being blocked by menu bar */
#app > .main-content {
  margin-top: 70px;
  min-height: calc(100vh - 70px);
}
</style>