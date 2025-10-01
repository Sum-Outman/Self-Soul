<template>
  <div id="app">
    <!-- Top Navigation Bar -->
    <nav class="top-menu-bar">
      <div class="menu-left">
        <span class="system-title">Self Brain</span>
      </div>
      <div class="menu-right">
        <!-- Function Buttons -->
        <router-link to="/" class="menu-link">Home</router-link>
        <router-link to="/training" class="menu-link">Training</router-link>
        <router-link to="/conversation" class="menu-link">Conversation</router-link>
        <router-link to="/knowledge" class="menu-link">Knowledge</router-link>
        <router-link to="/settings" class="menu-link">Settings</router-link>
        
        <!-- Server Connection Status -->
        <div class="connection-status" :style="{ color: connectionColor }">
          <span class="status-indicator" :style="{ backgroundColor: connectionColor }"></span>
          {{ connectionStatus }}
        </div>
      </div>
    </nav>

    <router-view/>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import errorHandler from '@/utils/errorHandler'
import api from '@/utils/api.js'

export default {
  name: 'App',
  components: {},
  setup(props) {
    const router = useRouter();
    const isConnected = ref(false);
    const connectionStatus = ref('Connecting...');
    const connectionColor = ref('#ff9800'); // Orange
    let connectionInterval = null;
    
    // WebSocket connection will be initialized on demand when needed
    // to avoid unnecessary connections
    
    // Check server connection status
    const checkServerConnection = () => {
      api.health.get()
        .then(response => {
          isConnected.value = true;
          connectionStatus.value = 'Connected';
          connectionColor.value = '#4caf50'; // Green for connected
          
          // If there's a new server message, show notification
          if (response.data && response.data.status) {
            console.log('Server connection established');
          }
        })
        .catch(error => {
          isConnected.value = false;
          connectionStatus.value = 'Disconnected';
          connectionColor.value = '#f44336'; // Red for disconnected
          console.error('Server connection error:', error);
        });
    };
    
    // Initialize components
    const initializeComponentsSilently = () => {
      try {
        console.log('Self Brain System components are initializing...')
        
        // Directly log initialization completion
        console.log('Self Brain System components initialization completed')
      } catch (error) {
        console.error('Error during system components initialization:', error)
      }
    }
    
    // Life cycle hooks
    onMounted(() => {
      // Register error handler
      window.addEventListener('error', (error) => errorHandler.handleError(error, 'Global'))
      window.addEventListener('unhandledrejection', (event) => {
        event.preventDefault();
        errorHandler.handleError(event.reason, 'Unhandled Promise');
      })
      
      // Periodically check server connection
      connectionInterval = setInterval(() => {
        checkServerConnection()
      }, 5000); // Check every 5 seconds
      
      // Check connection immediately
      checkServerConnection()
    })
    
    onUnmounted(() => {
      // Clear interval
      clearInterval(connectionInterval)
      
      // Remove event listeners
      window.removeEventListener('error', (error) => errorHandler.handleError(error, 'Global'))
      window.removeEventListener('unhandledrejection', (event) => {
        event.preventDefault();
        errorHandler.handleError(event.reason, 'Unhandled Promise');
      })
    })
    
    return {
      isConnected,
      connectionStatus,
      connectionColor
    }
  }
}
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
  padding-top: 70px;
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
  font-size: 13px;
  padding: 6px 12px;
  border-radius: var(--border-radius-sm);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

/* Adjust router view for fixed header */
.router-view {
  flex: 1;
  padding: 20px;
}
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

.menu-link:hover {
  background: var(--bg-tertiary);
  text-decoration: none;
  color: var(--text-primary);
  border-color: var(--border-dark);
}

/* Add top margin to router-view to avoid being blocked by menu bar */
#app > :not(.top-menu-bar) {
  margin-top: 70px;
  min-height: calc(100vh - 70px);
}
</style>
