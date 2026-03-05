<template>
  <div class="system-health-monitor">
    <div class="monitor-header">
      <h3>System Health Monitor</h3>
      <div class="controls">
        <button @click="refreshHealth" :disabled="isRefreshing" class="refresh-btn">
          <span v-if="isRefreshing">Refreshing...</span>
          <span v-else>Refresh</span>
        </button>
        <div class="auto-refresh">
          <label>
            <input type="checkbox" v-model="autoRefresh" />
            Auto-refresh (30s)
          </label>
        </div>
      </div>
    </div>

    <!-- Overall System Status -->
    <div class="status-section">
      <h4>Overall System Status</h4>
      <div class="status-grid">
        <div class="status-item" :class="overallStatus">
          <div class="status-icon">
            <span v-if="overallStatus === 'healthy'">✅</span>
            <span v-else-if="overallStatus === 'warning'">⚠️</span>
            <span v-else>❌</span>
          </div>
          <div class="status-info">
            <div class="status-title">System Health</div>
            <div class="status-value">{{ overallStatus === 'healthy' ? 'Healthy' : overallStatus === 'warning' ? 'Warning' : 'Unhealthy' }}</div>
          </div>
        </div>
        
        <div class="status-item">
          <div class="status-icon">🕒</div>
          <div class="status-info">
            <div class="status-title">Last Check</div>
            <div class="status-value">{{ lastCheckTime }}</div>
          </div>
        </div>
        
        <div class="status-item">
          <div class="status-icon">📊</div>
          <div class="status-info">
            <div class="status-title">Response Time</div>
            <div class="status-value">{{ responseTime }}ms</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Server Status -->
    <div class="server-section">
      <h4>Server Status</h4>
      <div class="server-grid">
        <div class="server-card" :class="frontendStatus">
          <div class="server-header">
            <div class="server-icon">🌐</div>
            <div class="server-name">Frontend Server</div>
          </div>
          <div class="server-status">{{ frontendStatus === 'up' ? 'Online' : 'Offline' }}</div>
          <div class="server-details">
            <div>Port: 5175</div>
            <div>Protocol: HTTP</div>
          </div>
        </div>
        
        <div class="server-card" :class="backendStatus">
          <div class="server-header">
            <div class="server-icon">⚙️</div>
            <div class="server-name">Backend API</div>
          </div>
          <div class="server-status">{{ backendStatus === 'up' ? 'Online' : 'Offline' }}</div>
          <div class="server-details">
            <div>Port: 8000</div>
            <div>Protocol: HTTP</div>
          </div>
        </div>
      </div>
    </div>

    <!-- API Endpoints Health -->
    <div class="endpoints-section">
      <h4>API Endpoints Health</h4>
      <div class="endpoints-grid">
        <div class="endpoint-card" v-for="endpoint in endpoints" :key="endpoint.name" :class="endpoint.status">
          <div class="endpoint-header">
            <div class="endpoint-name">{{ endpoint.name }}</div>
            <div class="endpoint-status">{{ endpoint.status === 'healthy' ? '✓' : endpoint.status === 'warning' ? '⚠' : '✗' }}</div>
          </div>
          <div class="endpoint-url">{{ endpoint.url }}</div>
          <div class="endpoint-response">
            <span class="response-time">{{ endpoint.responseTime }}ms</span>
            <span class="response-code" :class="getStatusCodeClass(endpoint.statusCode)">{{ endpoint.statusCode }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Model Status Summary -->
    <div class="models-section" v-if="modelStats.total > 0">
      <h4>Model Status Summary</h4>
      <div class="models-grid">
        <div class="model-stat">
          <div class="stat-value">{{ modelStats.total }}</div>
          <div class="stat-label">Total Models</div>
        </div>
        <div class="model-stat">
          <div class="stat-value">{{ modelStats.active }}</div>
          <div class="stat-label">Active</div>
        </div>
        <div class="model-stat">
          <div class="stat-value">{{ modelStats.running }}</div>
          <div class="stat-label">Running</div>
        </div>
        <div class="model-stat">
          <div class="stat-value">{{ modelStats.stopped }}</div>
          <div class="stat-label">Stopped</div>
        </div>
      </div>
    </div>

    <!-- Recent Errors -->
    <div class="errors-section" v-if="recentErrors.length > 0">
      <h4>Recent System Errors</h4>
      <div class="errors-list">
        <div class="error-item" v-for="error in recentErrors" :key="error.timestamp">
          <div class="error-time">{{ formatTime(error.timestamp) }}</div>
          <div class="error-message">{{ error.message }}</div>
          <div class="error-context">{{ error.context }}</div>
        </div>
      </div>
    </div>

    <!-- Health Check Log -->
    <div class="log-section">
      <div class="log-header">
        <h4>Health Check Log</h4>
        <button @click="clearLog" class="clear-btn">Clear Log</button>
      </div>
      <div class="log-content">
        <div class="log-entry" v-for="entry in healthLog" :key="entry.timestamp">
          <span class="log-time">{{ formatTime(entry.timestamp) }}</span>
          <span class="log-status" :class="entry.status">{{ entry.status }}</span>
          <span class="log-message">{{ entry.message }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import api from '@/utils/api'
import { handleApiError } from '@/utils/enhancedErrorHandler'

export default {
  name: 'SystemHealthMonitor',
  data() {
    return {
      autoRefresh: false,
      isRefreshing: false,
      refreshTimer: null,
      
      // Health status
      overallStatus: 'unknown',
      lastCheckTime: 'Never',
      responseTime: 0,
      
      // Server status
      frontendStatus: 'unknown',
      backendStatus: 'unknown',
      
      // API endpoints
      endpoints: [
        { name: 'Health Check', url: '/health', status: 'unknown', responseTime: 0, statusCode: 0 },
        { name: 'Model List', url: '/api/models/getAll', status: 'unknown', responseTime: 0, statusCode: 0 },
        { name: 'Knowledge API', url: '/api/knowledge/files', status: 'unknown', responseTime: 0, statusCode: 0 },
        { name: 'Training Status', url: '/api/models/training/status', status: 'unknown', responseTime: 0, statusCode: 0 }
      ],
      
      // Model statistics
      modelStats: {
        total: 0,
        active: 0,
        running: 0,
        stopped: 0
      },
      
      // Error tracking
      recentErrors: [],
      healthLog: []
    }
  },
  mounted() {
    this.refreshHealth()
    if (this.autoRefresh) {
      this.startAutoRefresh()
    }
  },
  beforeUnmount() {
    this.stopAutoRefresh()
  },
  watch: {
    autoRefresh(newVal) {
      if (newVal) {
        this.startAutoRefresh()
      } else {
        this.stopAutoRefresh()
      }
    }
  },
  methods: {
    async refreshHealth() {
      this.isRefreshing = true
      const startTime = Date.now()
      
      try {
        // Clear previous status
        this.overallStatus = 'checking'
        this.lastCheckTime = this.formatTime(new Date())
        
        // Check frontend (always up if we're here)
        this.frontendStatus = 'up'
        
        // Check backend health
        await this.checkBackendHealth()
        
        // Check all API endpoints
        await this.checkAllEndpoints()
        
        // Get model statistics
        await this.getModelStatistics()
        
        // Update overall status
        this.updateOverallStatus()
        
        // Calculate response time
        this.responseTime = Date.now() - startTime
        
        // Log successful check
        this.addLogEntry('success', `Health check completed in ${this.responseTime}ms`)
        
      } catch (error) {
        handleApiError(error, 'System Health Monitor')
        this.addLogEntry('error', `Health check failed: ${error.message}`)
        this.overallStatus = 'unhealthy'
      } finally {
        this.isRefreshing = false
      }
    },
    
    async checkBackendHealth() {
      try {
        const startTime = Date.now()
        const response = await api.get('/health')
        const responseTime = Date.now() - startTime
        
        if (response.status === 200) {
          this.backendStatus = 'up'
          this.addLogEntry('success', `Backend health check: ${responseTime}ms`)
        } else {
          this.backendStatus = 'down'
          this.addLogEntry('warning', `Backend returned status ${response.status}`)
        }
      } catch (error) {
        this.backendStatus = 'down'
        this.addLogEntry('error', `Backend health check failed: ${error.message}`)
        throw error
      }
    },
    
    async checkAllEndpoints() {
      const endpointPromises = this.endpoints.map(async (endpoint, index) => {
        try {
          const startTime = Date.now()
          const response = await api.get(endpoint.url)
          const responseTime = Date.now() - startTime
          
          this.endpoints[index].responseTime = responseTime
          this.endpoints[index].statusCode = response.status
          
          if (response.status === 200) {
            this.endpoints[index].status = 'healthy'
            this.addLogEntry('success', `${endpoint.name}: ${responseTime}ms`)
          } else if (response.status >= 400 && response.status < 500) {
            this.endpoints[index].status = 'warning'
            this.addLogEntry('warning', `${endpoint.name}: ${response.status}`)
          } else {
            this.endpoints[index].status = 'unhealthy'
            this.addLogEntry('error', `${endpoint.name}: ${response.status}`)
          }
          
        } catch (error) {
          this.endpoints[index].status = 'unhealthy'
          this.endpoints[index].statusCode = error.response?.status || 0
          this.endpoints[index].responseTime = 0
          
          this.addLogEntry('error', `${endpoint.name}: ${error.message}`)
        }
      })
      
      await Promise.allSettled(endpointPromises)
    },
    
    async getModelStatistics() {
      try {
        const response = await api.get('/api/models/getAll')
        
        if (response.data && response.data.models) {
          const models = response.data.models
          
          this.modelStats.total = models.length
          this.modelStats.active = models.filter(m => m.isActive).length
          this.modelStats.running = models.filter(m => m.status === 'running').length
          this.modelStats.stopped = models.filter(m => m.status === 'stopped').length
          
          this.addLogEntry('success', `Loaded ${models.length} models`)
        }
      } catch (error) {
        this.addLogEntry('warning', `Failed to load model statistics: ${error.message}`)
      }
    },
    
    updateOverallStatus() {
      // Determine overall status based on endpoint health
      const unhealthyCount = this.endpoints.filter(e => e.status === 'unhealthy').length
      const warningCount = this.endpoints.filter(e => e.status === 'warning').length
      
      if (unhealthyCount > 0 || this.backendStatus === 'down') {
        this.overallStatus = 'unhealthy'
      } else if (warningCount > 0) {
        this.overallStatus = 'warning'
      } else {
        this.overallStatus = 'healthy'
      }
    },
    
    startAutoRefresh() {
      if (this.refreshTimer) {
        clearInterval(this.refreshTimer)
      }
      this.refreshTimer = setInterval(() => {
        this.refreshHealth()
      }, 30000) // 30 seconds
    },
    
    stopAutoRefresh() {
      if (this.refreshTimer) {
        clearInterval(this.refreshTimer)
        this.refreshTimer = null
      }
    },
    
    addLogEntry(status, message) {
      const entry = {
        timestamp: new Date().toISOString(),
        status,
        message
      }
      
      this.healthLog.unshift(entry)
      
      // Keep only last 50 entries
      if (this.healthLog.length > 50) {
        this.healthLog = this.healthLog.slice(0, 50)
      }
      
      // Track errors
      if (status === 'error') {
        this.recentErrors.unshift({
          timestamp: entry.timestamp,
          message,
          context: 'Health Check'
        })
        
        if (this.recentErrors.length > 10) {
          this.recentErrors = this.recentErrors.slice(0, 10)
        }
      }
    },
    
    clearLog() {
      this.healthLog = []
      this.recentErrors = []
    },
    
    formatTime(timestamp) {
      if (!timestamp) return 'Unknown'
      
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    },
    
    getStatusCodeClass(statusCode) {
      if (statusCode >= 200 && statusCode < 300) return 'success'
      if (statusCode >= 400 && statusCode < 500) return 'warning'
      return 'error'
    }
  }
}
</script>

<style scoped>
.system-health-monitor {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.monitor-header h3 {
  margin: 0;
  color: #2c3e50;
}

.controls {
  display: flex;
  align-items: center;
  gap: 15px;
}

.refresh-btn {
  padding: 6px 12px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.refresh-btn:hover:not(:disabled) {
  background: #2980b9;
}

.refresh-btn:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}

.auto-refresh label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  cursor: pointer;
}

.status-section,
.server-section,
.endpoints-section,
.models-section,
.errors-section,
.log-section {
  margin-bottom: 25px;
}

.status-section h4,
.server-section h4,
.endpoints-section h4,
.models-section h4,
.errors-section h4,
.log-section h4 {
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 16px;
}

.status-grid,
.server-grid,
.endpoints-grid,
.models-grid {
  display: grid;
  gap: 15px;
}

.status-grid {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.status-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid #95a5a6;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.status-item.healthy {
  border-left-color: #27ae60;
}

.status-item.warning {
  border-left-color: #f39c12;
}

.status-item.unhealthy {
  border-left-color: #e74c3c;
}

.status-item.checking {
  border-left-color: #3498db;
}

.status-icon {
  font-size: 24px;
}

.status-info {
  flex: 1;
}

.status-title {
  font-size: 12px;
  color: #7f8c8d;
  margin-bottom: 4px;
}

.status-value {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.server-grid {
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.server-card {
  padding: 20px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid #95a5a6;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.server-card.up {
  border-left-color: #27ae60;
}

.server-card.down {
  border-left-color: #e74c3c;
}

.server-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

.server-icon {
  font-size: 24px;
}

.server-name {
  font-weight: 600;
  color: #2c3e50;
}

.server-status {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 10px;
}

.server-details {
  font-size: 12px;
  color: #7f8c8d;
}

.endpoints-grid {
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.endpoint-card {
  padding: 15px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid #95a5a6;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.endpoint-card.healthy {
  border-left-color: #27ae60;
}

.endpoint-card.warning {
  border-left-color: #f39c12;
}

.endpoint-card.unhealthy {
  border-left-color: #e74c3c;
}

.endpoint-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.endpoint-name {
  font-weight: 600;
  color: #2c3e50;
}

.endpoint-status {
  font-size: 18px;
  font-weight: 600;
}

.endpoint-url {
  font-size: 12px;
  color: #7f8c8d;
  margin-bottom: 10px;
  word-break: break-all;
}

.endpoint-response {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
}

.response-time {
  color: #7f8c8d;
}

.response-code.success {
  color: #27ae60;
}

.response-code.warning {
  color: #f39c12;
}

.response-code.error {
  color: #e74c3c;
}

.models-grid {
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
}

.model-stat {
  padding: 15px;
  background: white;
  border-radius: 6px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 12px;
  color: #7f8c8d;
}

.errors-list {
  max-height: 200px;
  overflow-y: auto;
}

.error-item {
  padding: 10px;
  background: white;
  border-radius: 4px;
  margin-bottom: 8px;
  border-left: 4px solid #e74c3c;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.error-time {
  font-size: 11px;
  color: #7f8c8d;
  margin-bottom: 4px;
}

.error-message {
  font-size: 14px;
  color: #2c3e50;
  margin-bottom: 2px;
}

.error-context {
  font-size: 11px;
  color: #95a5a6;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.clear-btn {
  padding: 4px 8px;
  background: #95a5a6;
  color: white;
  border: none;
  border-radius: 3px;
  font-size: 12px;
  cursor: pointer;
}

.clear-btn:hover {
  background: #7f8c8d;
}

.log-content {
  max-height: 200px;
  overflow-y: auto;
  background: white;
  border-radius: 4px;
  padding: 10px;
  font-family: 'Courier New', monospace;
  font-size: 12px;
}

.log-entry {
  display: flex;
  gap: 10px;
  margin-bottom: 5px;
  padding: 3px 0;
  border-bottom: 1px solid #f1f1f1;
}

.log-entry:last-child {
  border-bottom: none;
}

.log-time {
  color: #7f8c8d;
  min-width: 70px;
}

.log-status {
  font-weight: 600;
  min-width: 60px;
}

.log-status.success {
  color: #27ae60;
}

.log-status.warning {
  color: #f39c12;
}

.log-status.error {
  color: #e74c3c;
}

.log-message {
  flex: 1;
  color: #2c3e50;
  word-break: break-all;
}
</style>