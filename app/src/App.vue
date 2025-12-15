<template>
  <div id="app">
    <!-- Top Menu Bar -->
    <nav class="top-menu-bar">
      <div class="menu-left">
        <div class="system-title">Self Soul System</div>
      </div>
      <div class="menu-right">
        <router-link to="/" class="menu-link" active-class="router-link-active">Home</router-link>
        <router-link to="/conversation" class="menu-link" active-class="router-link-active">Conversation</router-link>
        <router-link to="/training" class="menu-link" active-class="router-link-active">Training</router-link>
        <router-link to="/knowledge" class="menu-link" active-class="router-link-active">Knowledge</router-link>
        <router-link to="/settings" class="menu-link" active-class="router-link-active">Settings</router-link>
        <router-link to="/help" class="menu-link" active-class="router-link-active">Help</router-link>
        
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

<script>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { handleError } from '@/utils/errorHandler'
import api from '@/utils/api.js'

export default {
  name: 'App',
  components: {},
  setup(props) {
    const router = useRouter();
    const isConnected = ref(false);
    const connectionStatus = ref('Connecting...');
    const connectionColor = ref('#999999'); // Gray
    let connectionInterval = null;
    const lastConnectionCheck = ref(null);
    
    // WebSocket connection will be initialized on demand when needed
    // to avoid unnecessary connections
    
    // Check server connection status
    const checkServerConnection = () => {
      api.health.get()
        .then(response => {
          isConnected.value = true;
          connectionStatus.value = 'Connected';
          connectionColor.value = '#555555'; // Dark gray for connected
          
          // If there's a new server message, show notification
          if (response.data && response.data.status) {
            console.log('Server connection established');
          }
        })
        .catch(error => {
          isConnected.value = false;
          connectionStatus.value = 'Disconnected';
          connectionColor.value = '#888888'; // Medium gray for disconnected
          console.error('Server connection error:', error);
        });
    };
    
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
    const globalErrorHandler = (error) => {
      handleError(error, 'Global');
    };
    
    const unhandledRejectionHandler = (event) => {
      event.preventDefault();
      handleError(event.reason, 'Unhandled Promise');
    };
    
    // Life cycle hooks
    onMounted(() => {
      // Register error handlers
      window.addEventListener('error', globalErrorHandler);
      window.addEventListener('unhandledrejection', unhandledRejectionHandler);
      
      // Periodically check server connection
      connectionInterval = setInterval(() => {
        checkServerConnection()
      }, 5000); // Check every 5 seconds
      
      // Check connection immediately
      checkServerConnection()
    })
    
    onUnmounted(() => {
      // Clear interval
      if (connectionInterval) {
        clearInterval(connectionInterval);
      }
      
      // Remove event listeners properly
      window.removeEventListener('error', globalErrorHandler);
      window.removeEventListener('unhandledrejection', unhandledRejectionHandler);
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
#app > :not(.top-menu-bar) {
  margin-top: 70px;
  min-height: calc(100vh - 70px);
}
</style>
