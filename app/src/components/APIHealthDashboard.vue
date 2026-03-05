<template>
  <div class="api-health-dashboard">
    <!-- API Health Status Dashboard -->
    <div class="dashboard-header">
      <h2>API Health Status Dashboard</h2>
      <div class="controls">
        <button @click="refreshHealthData" :disabled="isRefreshing" class="refresh-btn">
          <span v-if="isRefreshing">Refreshing...</span>
          <span v-else>Refresh</span>
        </button>
        <div class="auto-refresh">
          <label>
            <input type="checkbox" v-model="autoRefresh" />
            Auto-refresh ({{ refreshInterval }}s)
          </label>
        </div>
        <div class="monitoring-controls">
          <button @click="toggleRealTimeMonitoring" :class="{ active: realTimeMonitoring }" class="monitor-btn">
            {{ realTimeMonitoring ? 'Stop Real-time' : 'Start Real-time' }}
          </button>
          <button @click="exportHealthData" class="export-btn">Export Data</button>
        </div>
      </div>
    </div>

    <!-- Real-time Status Indicators -->
    <div v-if="realTimeMonitoring" class="real-time-indicators">
      <div class="indicator">
        <span class="indicator-label">Live Updates:</span>
        <span class="indicator-value" :class="{ active: isLive }">
          {{ isLive ? 'Active' : 'Inactive' }}
        </span>
      </div>
      <div class="indicator">
        <span class="indicator-label">Last Update:</span>
        <span class="indicator-value">{{ lastUpdateTime }}</span>
      </div>
      <div class="indicator">
        <span class="indicator-label">Updates Count:</span>
        <span class="indicator-value">{{ updateCount }}</span>
      </div>
    </div>

    <!-- API Provider Status Overview -->
    <div class="api-status-overview">
      <div class="status-summary">
        <div class="summary-card" :class="overallStatus">
          <h3>Overall API Status</h3>
          <p class="status-value">{{ overallStatusText }}</p>
          <p class="health-score">Health Score: {{ overallHealthScore }}%</p>
          <p class="uptime">Average Uptime: {{ averageUptime }}%</p>
          <div class="status-breakdown">
            <span class="status-item normal">{{ statusCounts.normal }} Normal</span>
            <span class="status-item warning">{{ statusCounts.warning }} Warning</span>
            <span class="status-item critical">{{ statusCounts.critical }} Critical</span>
          </div>
        </div>
        
        <div class="summary-card">
          <h3>Connected APIs</h3>
          <p class="metric-value">{{ connectedCount }}</p>
          <p class="metric-label">Total APIs: {{ totalCount }}</p>
          <div class="connection-breakdown">
            <span class="connection-item connected">{{ connectedCount }} Connected</span>
            <span class="connection-item disconnected">{{ disconnectedCount }} Disconnected</span>
          </div>
        </div>
        
        <div class="summary-card">
          <h3>Response Time</h3>
          <p class="metric-value">{{ averageResponseTime }}ms</p>
          <p class="metric-label">Average</p>
          <div class="response-breakdown">
            <span class="response-item">Min: {{ minResponseTime }}ms</span>
            <span class="response-item">Max: {{ maxResponseTime }}ms</span>
          </div>
        </div>
        
        <div class="summary-card">
          <h3>Performance Metrics</h3>
          <p class="metric-value">{{ overallSuccessRate }}%</p>
          <p class="metric-label">Success Rate</p>
          <div class="performance-breakdown">
            <span class="performance-item">Total Requests: {{ totalRequests }}</span>
            <span class="performance-item">Failed: {{ totalFailedRequests }}</span>
            <span class="performance-item">Consecutive Failures: {{ maxConsecutiveFailures }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed API Provider Status -->
    <div class="api-details-section">
      <h3>API Provider Details</h3>
      <div class="api-grid">
        <div 
          v-for="api in apiProviders" 
          :key="api.provider" 
          class="api-card"
          :class="api.status.alert_level"
        >
          <div class="api-header">
            <h4>{{ getProviderDisplayName(api.provider) }}</h4>
            <div class="status-indicator" :class="{ connected: api.status.connected, disconnected: !api.status.connected }"></div>
          </div>
          
          <div class="api-metrics">
            <div class="metric-row">
              <span class="metric-label">Status:</span>
              <span class="metric-value" :class="{ connected: api.status.connected, disconnected: !api.status.connected }">
                {{ api.status.connected ? 'Connected' : 'Disconnected' }}
              </span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Health Score:</span>
              <span class="metric-value" :class="getHealthScoreClass(api.status.health_score)">
                {{ api.status.health_score || 0 }}%
              </span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Response Time:</span>
              <span class="metric-value">{{ api.status.average_response_time || 0 }}ms</span>
              <span class="metric-detail">(min: {{ api.status.min_response_time || 0 }}ms, max: {{ api.status.max_response_time || 0 }}ms)</span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Success Rate:</span>
              <span class="metric-value">{{ api.status.success_rate || 0 }}%</span>
              <span class="metric-detail">({{ api.status.total_requests || 0 }} total, {{ api.status.failed_requests || 0 }} failed)</span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Uptime:</span>
              <span class="metric-value">{{ api.status.uptime_percentage || 0 }}%</span>
              <span class="metric-detail">{{ formatDuration(api.status.total_downtime || 0) }} downtime</span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Last Check:</span>
              <span class="metric-value">{{ formatTimestamp(api.status.last_check) }}</span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Consecutive Failures:</span>
              <span class="metric-value" :class="{ critical: api.status.consecutive_failures > 3, warning: api.status.consecutive_failures > 0 }">
                {{ api.status.consecutive_failures || 0 }}
              </span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">Last Success:</span>
              <span class="metric-value">{{ formatTimestamp(api.status.last_success) }}</span>
            </div>
            
            <div class="metric-row">
              <span class="metric-label">API Version:</span>
              <span class="metric-value">{{ api.status.api_version || 'Unknown' }}</span>
            </div>
          </div>
          
          <div v-if="api.status.alert_messages && api.status.alert_messages.length > 0" class="alerts-section">
            <div class="alert-header">
              <span class="alert-icon">⚠️</span>
              <span class="alert-title">Alerts</span>
            </div>
            <div class="alert-messages">
              <div 
                v-for="(alert, index) in api.status.alert_messages" 
                :key="index" 
                class="alert-message"
                :class="api.status.alert_level"
              >
                {{ alert }}
              </div>
            </div>
          </div>
          
          <div class="api-actions">
            <button @click="testConnection(api.provider)" class="test-btn">Test</button>
            <button @click="viewDetails(api.provider)" class="details-btn">Details</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance Charts -->
    <div class="charts-section">
      <h3>Performance Trends</h3>
      <div class="charts-grid">
        <div class="chart-container">
          <h4>Response Time Trends</h4>
          <div class="chart-content">
            <div v-if="performanceData.length > 0" class="trend-chart">
              <!-- Response time chart will be implemented with chart library -->
              <p>Average Response Time: {{ averageResponseTime }}ms</p>
            </div>
            <div v-else class="no-data">
              <p>No performance data available</p>
            </div>
          </div>
        </div>
        
        <div class="chart-container">
          <h4>Success Rate Trends</h4>
          <div class="chart-content">
            <div v-if="performanceData.length > 0" class="trend-chart">
              <!-- Success rate chart will be implemented with chart library -->
              <p>Overall Success Rate: {{ overallSuccessRate }}%</p>
            </div>
            <div v-else class="no-data">
              <p>No performance data available</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Real-time Alerts -->
    <div v-if="criticalAlerts.length > 0" class="alerts-section">
      <h3>Critical Alerts</h3>
      <div class="alerts-list">
        <div 
          v-for="(alert, index) in criticalAlerts" 
          :key="index" 
          class="alert-item critical"
        >
          <span class="alert-icon">🚨</span>
          <span class="alert-message">{{ alert.message }}</span>
          <span class="alert-time">{{ formatTimestamp(alert.timestamp) }}</span>
          <button @click="dismissAlert(alert.id)" class="dismiss-btn">Dismiss</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { notify } from '@/plugins/notification';

export default {
  name: 'APIHealthDashboard',
  data() {
    return {
      apiProviders: [],
      performanceData: [],
      isRefreshing: false,
      autoRefresh: false,
      refreshInterval: 30,
      refreshTimer: null,
      criticalAlerts: [],
      realTimeMonitoring: false,
      realTimeSocket: null,
      isLive: false,
      lastUpdateTime: 'Never',
      updateCount: 0,
      realTimeTimer: null
    }
  },
  computed: {
    overallStatus() {
      const criticalCount = this.apiProviders.filter(api => 
        api.status.alert_level === 'critical'
      ).length
      const warningCount = this.apiProviders.filter(api => 
        api.status.alert_level === 'warning'
      ).length
      
      if (criticalCount > 0) return 'critical'
      if (warningCount > 0) return 'warning'
      return 'normal'
    },
    
    overallStatusText() {
      switch (this.overallStatus) {
        case 'critical': return 'Critical'
        case 'warning': return 'Warning'
        default: return 'Normal'
      }
    },
    
    overallHealthScore() {
      if (this.apiProviders.length === 0) return 0
      const totalScore = this.apiProviders.reduce((sum, api) => 
        sum + (api.status.health_score || 0), 0
      )
      return Math.round(totalScore / this.apiProviders.length)
    },
    
    averageUptime() {
      if (this.apiProviders.length === 0) return 0
      const totalUptime = this.apiProviders.reduce((sum, api) => 
        sum + (api.status.uptime_percentage || 0), 0
      )
      return Math.round(totalUptime / this.apiProviders.length)
    },
    
    connectedCount() {
      return this.apiProviders.filter(api => api.status.connected).length
    },
    
    totalCount() {
      return this.apiProviders.length
    },
    
    averageResponseTime() {
      if (this.apiProviders.length === 0) return 0
      const totalTime = this.apiProviders.reduce((sum, api) => 
        sum + (api.status.average_response_time || 0), 0
      )
      return Math.round(totalTime / this.apiProviders.length)
    },
    
    overallSuccessRate() {
      if (this.apiProviders.length === 0) return 0
      const totalRate = this.apiProviders.reduce((sum, api) => 
        sum + (api.status.success_rate || 0), 0
      )
      return Math.round(totalRate / this.apiProviders.length)
    },
    
    statusCounts() {
      const counts = { normal: 0, warning: 0, critical: 0 }
      this.apiProviders.forEach(api => {
        counts[api.status.alert_level || 'normal']++
      })
      return counts
    },
    
    disconnectedCount() {
      return this.apiProviders.filter(api => !api.status.connected).length
    },
    
    minResponseTime() {
      if (this.apiProviders.length === 0) return 0
      const times = this.apiProviders.map(api => api.status.min_response_time || Infinity)
      return Math.min(...times)
    },
    
    maxResponseTime() {
      if (this.apiProviders.length === 0) return 0
      const times = this.apiProviders.map(api => api.status.max_response_time || 0)
      return Math.max(...times)
    },
    
    totalRequests() {
      return this.apiProviders.reduce((sum, api) => sum + (api.status.total_requests || 0), 0)
    },
    
    totalFailedRequests() {
      return this.apiProviders.reduce((sum, api) => sum + (api.status.failed_requests || 0), 0)
    },
    
    maxConsecutiveFailures() {
      if (this.apiProviders.length === 0) return 0
      const failures = this.apiProviders.map(api => api.status.consecutive_failures || 0)
      return Math.max(...failures)
    }
  },
  methods: {
    async refreshHealthData() {
      this.isRefreshing = true
      try {
        // Fetch API health data from backend
        const response = await fetch('/api/external-api/health-status')
        if (response.ok) {
          const data = await response.json()
          this.apiProviders = data.providers || []
          this.performanceData = data.performance_data || []
          this.criticalAlerts = data.critical_alerts || []
        } else {
          console.error('Failed to fetch API health data')
        }
      } catch (error) {
        console.error('Error fetching API health data:', error)
      } finally {
        this.isRefreshing = false
      }
    },
    
    getProviderDisplayName(provider) {
      const names = {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'google': 'Google AI',
        'aws': 'AWS',
        'azure': 'Azure',
        'huggingface': 'Hugging Face',
        'replicate': 'Replicate',
        'cohere': 'Cohere'
      }
      return names[provider] || provider
    },
    
    getHealthScoreClass(score) {
      if (score >= 80) return 'excellent'
      if (score >= 60) return 'good'
      if (score >= 40) return 'fair'
      return 'poor'
    },
    
    formatTimestamp(timestamp) {
      if (!timestamp) return 'Never'
      return new Date(timestamp).toLocaleString()
    },
    
    async testConnection(provider) {
      try {
        const response = await fetch(`/api/external-api/test-connection/${provider}`)
        if (response.ok) {
          const result = await response.json()
          if (result.connected) {
            notify.success(`Connection test for ${this.getProviderDisplayName(provider)}: Success`);
          } else {
            notify.error(`Connection test for ${this.getProviderDisplayName(provider)}: Failed`);
          }
          // Refresh data after test
          this.refreshHealthData()
        }
      } catch (error) {
        console.error('Error testing connection:', error)
        notify.error('Connection test failed')
      }
    },
    
    viewDetails(provider) {
      // Navigate to API details page or show modal
      console.log('View details for:', provider)
    },
    
    dismissAlert(alertId) {
      this.criticalAlerts = this.criticalAlerts.filter(alert => alert.id !== alertId)
    },
    
    toggleRealTimeMonitoring() {
      this.realTimeMonitoring = !this.realTimeMonitoring
      if (this.realTimeMonitoring) {
        this.startRealTimeMonitoring()
      } else {
        this.stopRealTimeMonitoring()
      }
    },
    
    startRealTimeMonitoring() {
      this.isLive = true
      this.updateCount = 0
      
      // Use WebSocket for real-time updates
      if (window.location.protocol === 'https:') {
        this.realTimeSocket = new WebSocket('wss://' + window.location.host + '/ws/monitoring')
      } else {
        this.realTimeSocket = new WebSocket('ws://' + window.location.host + '/ws/monitoring')
      }
      
      this.realTimeSocket.onopen = () => {
        console.log('Real-time monitoring WebSocket connected')
      }
      
      this.realTimeSocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'monitoring_data') {
            // Update API health data from WebSocket
            if (data.data.providers) {
              this.apiProviders = data.data.providers
            }
            if (data.data.performance_data) {
              this.performanceData = data.data.performance_data
            }
            if (data.data.critical_alerts) {
              this.criticalAlerts = data.data.critical_alerts
            }
            
            this.updateCount++
            this.lastUpdateTime = new Date().toLocaleTimeString()
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      this.realTimeSocket.onerror = (error) => {
        console.error('Real-time monitoring WebSocket error:', error)
        // Fallback to polling if WebSocket fails
        this.fallbackToPolling()
      }
      
      this.realTimeSocket.onclose = () => {
        console.log('Real-time monitoring WebSocket disconnected')
        this.isLive = false
        // Fallback to polling if WebSocket closes
        this.fallbackToPolling()
      }
      
      console.log('Real-time monitoring started')
    },
    
    fallbackToPolling() {
      // Fallback to polling if WebSocket is not available
      if (!this.realTimeTimer) {
        console.log('Falling back to polling for real-time updates')
        this.realTimeTimer = setInterval(() => {
          this.refreshHealthData()
          this.updateCount++
          this.lastUpdateTime = new Date().toLocaleTimeString()
        }, 10000) // Poll every 10 seconds as fallback
      }
    },
    
    stopRealTimeMonitoring() {
      this.isLive = false
      if (this.realTimeTimer) {
        clearInterval(this.realTimeTimer)
        this.realTimeTimer = null
      }
      console.log('Real-time monitoring stopped')
    },
    
    exportHealthData() {
      const exportData = {
        timestamp: new Date().toISOString(),
        apiProviders: this.apiProviders,
        performanceData: this.performanceData,
        criticalAlerts: this.criticalAlerts,
        summary: {
          overallHealthScore: this.overallHealthScore,
          averageUptime: this.averageUptime,
          connectedCount: this.connectedCount,
          totalCount: this.totalCount,
          averageResponseTime: this.averageResponseTime,
          overallSuccessRate: this.overallSuccessRate
        }
      }
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `api-health-data-${new Date().toISOString().split('T')[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
      
      console.log('Health data exported')
    },
    
    formatDuration(seconds) {
      if (!seconds || seconds === 0) return '0s'
      
      const hours = Math.floor(seconds / 3600)
      const minutes = Math.floor((seconds % 3600) / 60)
      const secs = Math.floor(seconds % 60)
      
      const parts = []
      if (hours > 0) parts.push(`${hours}h`)
      if (minutes > 0) parts.push(`${minutes}m`)
      if (secs > 0 || parts.length === 0) parts.push(`${secs}s`)
      
      return parts.join(' ')
    }
  },
  mounted() {
    this.refreshHealthData()
    
    // Set up auto-refresh
    this.$watch('autoRefresh', (newVal) => {
      if (newVal) {
        this.refreshTimer = setInterval(() => {
          this.refreshHealthData()
        }, this.refreshInterval * 1000)
      } else if (this.refreshTimer) {
        clearInterval(this.refreshTimer)
        this.refreshTimer = null
      }
    })
  },
  
  beforeUnmount() {
    // Clean up all timers
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer)
      this.refreshTimer = null
    }
    
    if (this.realTimeTimer) {
      this.stopRealTimeMonitoring()
    }
    
    // Close WebSocket connection if exists
    if (this.realTimeSocket) {
      this.realTimeSocket.close()
      this.realTimeSocket = null
    }
  }
}
</script>

<style scoped>
.api-health-dashboard {
  padding: 20px;
  background: #f5f5f5;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.dashboard-header h2 {
  color: #333;
  margin: 0;
}

.controls {
  display: flex;
  gap: 15px;
  align-items: center;
  flex-wrap: wrap;
}

.refresh-btn {
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.refresh-btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.auto-refresh label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 14px;
}

.monitoring-controls {
  display: flex;
  gap: 10px;
}

.monitor-btn {
  padding: 8px 16px;
  background: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.monitor-btn.active {
  background: #dc3545;
}

.export-btn {
  padding: 8px 16px;
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.real-time-indicators {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 10px;
  background: #e9ecef;
  border-radius: 4px;
}

.indicator {
  display: flex;
  gap: 5px;
  font-size: 14px;
}

.indicator-label {
  color: #666;
  font-weight: bold;
}

.indicator-value {
  color: #6c757d;
}

.indicator-value.active {
  color: #28a745;
  font-weight: bold;
}

.api-status-overview {
  margin-bottom: 30px;
}

.status-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.summary-card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.summary-card.normal {
  border-left: 4px solid #28a745;
}

.summary-card.warning {
  border-left: 4px solid #ffc107;
}

.summary-card.critical {
  border-left: 4px solid #dc3545;
}

.status-value {
  font-size: 24px;
  font-weight: bold;
  margin: 10px 0;
}

.health-score, .uptime {
  font-size: 14px;
  color: #666;
  margin: 5px 0;
}

.status-breakdown, .connection-breakdown, .response-breakdown, .performance-breakdown {
  display: flex;
  gap: 10px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.status-item, .connection-item, .response-item, .performance-item {
  font-size: 12px;
  padding: 2px 6px;
  border-radius: 3px;
  background: #f8f9fa;
}

.status-item.normal {
  background: #d4edda;
  color: #155724;
}

.status-item.warning {
  background: #fff3cd;
  color: #856404;
}

.status-item.critical {
  background: #f8d7da;
  color: #721c24;
}

.connection-item.connected {
  background: #d4edda;
  color: #155724;
}

.connection-item.disconnected {
  background: #f8d7da;
  color: #721c24;
}

.metric-value {
  font-size: 32px;
  font-weight: bold;
  color: #007bff;
  margin: 10px 0;
}

.metric-label {
  font-size: 14px;
  color: #666;
}

.api-details-section {
  margin-bottom: 30px;
}

.api-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.api-card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  border-left: 4px solid #28a745;
}

.api-card.warning {
  border-left-color: #ffc107;
}

.api-card.critical {
  border-left-color: #dc3545;
}

.api-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.api-header h4 {
  margin: 0;
  color: #333;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.status-indicator.connected {
  background: #28a745;
}

.status-indicator.disconnected {
  background: #dc3545;
}

.api-metrics {
  margin-bottom: 15px;
}

.metric-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.metric-label {
  color: #666;
}

.metric-value.connected {
  color: #28a745;
}

.metric-value.disconnected {
  color: #dc3545;
}

.metric-value.excellent {
  color: #28a745;
}

.metric-value.good {
  color: #17a2b8;
}

.metric-value.fair {
  color: #ffc107;
}

.metric-value.poor {
  color: #dc3545;
}

.metric-value.critical {
  color: #dc3545;
  font-weight: bold;
}

.metric-value.warning {
  color: #ffc107;
  font-weight: bold;
}

.metric-detail {
  font-size: 11px;
  color: #999;
  margin-left: 5px;
}

.alerts-section {
  margin-top: 15px;
}

.alert-header {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-bottom: 10px;
  font-weight: bold;
}

.alert-messages {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 4px;
  padding: 10px;
}

.alert-message {
  font-size: 12px;
  margin-bottom: 5px;
  padding: 5px;
  border-radius: 3px;
}

.alert-message.warning {
  background: #fff3cd;
  color: #856404;
}

.alert-message.critical {
  background: #f8d7da;
  color: #721c24;
}

.api-actions {
  display: flex;
  gap: 10px;
}

.test-btn, .details-btn {
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.test-btn {
  background: #007bff;
  color: white;
}

.details-btn {
  background: #6c757d;
  color: white;
}

.charts-section {
  margin-bottom: 30px;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
}

.chart-container {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.chart-container h4 {
  margin: 0 0 15px 0;
  color: #333;
}

.no-data {
  text-align: center;
  color: #666;
  padding: 20px;
}

.alerts-list {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  overflow: hidden;
}

.alert-item {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  border-bottom: 1px solid #eee;
}

.alert-item.critical {
  background: #f8d7da;
  color: #721c24;
}

.alert-icon {
  margin-right: 10px;
  font-size: 16px;
}

.alert-message {
  flex: 1;
  font-weight: bold;
}

.alert-time {
  font-size: 12px;
  color: #666;
  margin-right: 10px;
}

.dismiss-btn {
  background: #dc3545;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
}
</style>