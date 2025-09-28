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
        <router-link to="/settings" class="menu-link">Settings</router-link>
        <router-link to="/help" class="menu-link">Help</router-link>
        
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
      api.get('/health') // Use unified API instance and relative path
        .then(response => {
          isConnected.value = true;
          connectionStatus.value = 'Connected to Main API';
          connectionColor.value = '#4caf50'; // Green for connected
          
          // If there's a new server message, show notification
          if (response.data && response.data.status) {
            console.log('Server connection established');
          }
        })
        .catch(error => {
          isConnected.value = false;
          connectionStatus.value = 'Main API Disconnected';
          connectionColor.value = '#f44336'; // Red for disconnected
          console.error('Server connection error:', error);
        });
    };
    
    // Initialize components
    const initializeComponentsSilently = () => {
      try {
        errorHandler.logInfo('AGI Brain System components are initializing...')
        
        // Directly log initialization completion
        errorHandler.logInfo('AGI Brain System components initialization completed')
      } catch (error) {
        errorHandler.handleError('Error during system components initialization:', error)
      }
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
      isConnected,
      connectionStatus,
      connectionColor
    }
  }
}
</script>

<style scoped>
/* Use black, white and gray light theme variables defined in main.css */

/* Global style reset */
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

/* App container styles */
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Top menu bar styles - using black, white and gray light theme */
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

/* Menu item styles */
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
