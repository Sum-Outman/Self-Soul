<template>
  <div class="evolution-process-monitor">
    <!-- Page Header -->
    <div class="page-header">
      <h1>Evolution Process Monitoring</h1>
      <p class="subtitle">Real-time monitoring of autonomous evolution stages, resource usage, and system alerts</p>
    </div>

    <!-- Real-time Status Overview -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Real-time Evolution Status</h2>
        <div class="section-actions">
          <button @click="refreshMonitoringData" class="btn btn-secondary" :disabled="refreshing">
            <span v-if="refreshing">Refreshing...</span>
            <span v-else>Refresh Data</span>
          </button>
          <div class="auto-refresh-toggle">
            <label>
              <input type="checkbox" v-model="autoRefresh" />
              Auto Refresh ({{ refreshInterval }}s)
            </label>
          </div>
        </div>
      </div>

      <!-- Evolution Stages Visualization -->
      <div class="evolution-stages">
        <h3>Evolution Stages Progress</h3>
        <div class="stages-timeline">
          <div 
            v-for="stage in evolutionStages" 
            :key="stage.stage_name"
            class="stage-item"
            :class="[stage.status, { active: stage.status === 'running' }]"
          >
            <div class="stage-icon">
              <span v-if="stage.status === 'running'">⚡</span>
              <span v-else-if="stage.status === 'completed'">✓</span>
              <span v-else-if="stage.status === 'failed'">✗</span>
              <span v-else-if="stage.status === 'paused'">⏸️</span>
              <span v-else>⏳</span>
            </div>
            <div class="stage-info">
              <h4>{{ formatStageName(stage.stage_name) }}</h4>
              <p class="stage-phase">{{ stage.current_phase }}</p>
              <div class="stage-progress">
                <div class="progress-bar">
                  <div 
                    class="progress-fill" 
                    :style="{ width: stage.progress_percentage + '%' }"
                  ></div>
                </div>
                <span class="progress-text">{{ stage.progress_percentage.toFixed(1) }}%</span>
              </div>
              <div class="stage-details">
                <div class="detail-item">
                  <span class="label">Status:</span>
                  <span class="value status-badge" :class="stage.status">{{ stage.status }}</span>
                </div>
                <div v-if="stage.estimated_time_remaining" class="detail-item">
                  <span class="label">ETA:</span>
                  <span class="value">{{ formatTimeRemaining(stage.estimated_time_remaining) }}</span>
                </div>
                <div class="detail-item">
                  <span class="label">Resources:</span>
                  <span class="value">
                    CPU: {{ stage.resources_consumed.cpu }}%, 
                    Mem: {{ stage.resources_consumed.memory }}%
                    <span v-if="stage.resources_consumed.gpu > 0">, GPU: {{ stage.resources_consumed.gpu }}%</span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Resource Usage Monitoring -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Resource Usage Monitoring</h2>
        <div class="section-actions">
          <button @click="toggleResourceCharts" class="btn btn-outline">
            {{ showResourceCharts ? 'Hide Charts' : 'Show Charts' }}
          </button>
        </div>
      </div>

      <div class="resource-metrics">
        <div class="metric-cards">
          <div class="metric-card" v-for="(usage, resource) in resourceUsage" :key="resource">
            <h3>{{ formatResourceName(resource) }}</h3>
            <div class="metric-value-container">
              <p class="metric-value">{{ usage.current }}%</p>
              <p class="metric-trend">
                <span :class="getTrendClass(usage.current, usage.average)">
                  {{ getTrendIcon(usage.current, usage.average) }}
                </span>
                {{ getTrendText(usage.current, usage.average) }}
              </p>
            </div>
            <div class="metric-details">
              <div class="detail-item">
                <span class="label">Max:</span>
                <span class="value">{{ usage.max }}%</span>
              </div>
              <div class="detail-item">
                <span class="label">Average:</span>
                <span class="value">{{ usage.average }}%</span>
              </div>
              <div class="detail-item">
                <span class="label">Limit:</span>
                <span class="value">{{ getResourceLimit(resource) }}%</span>
              </div>
            </div>
            <div class="usage-bar">
              <div 
                class="usage-fill" 
                :style="{ width: usage.current + '%' }"
                :class="getUsageLevelClass(usage.current)"
              ></div>
            </div>
          </div>
        </div>

        <!-- Resource Charts (Optional) -->
        <div v-if="showResourceCharts" class="resource-charts">
          <div class="chart-container">
            <h4>CPU & Memory Usage Over Time</h4>
            <div class="chart-placeholder">
              <p>Chart visualization would show here with real-time data</p>
              <div class="mock-chart">
                <div class="chart-line cpu-line"></div>
                <div class="chart-line memory-line"></div>
                <div class="chart-line gpu-line"></div>
              </div>
            </div>
          </div>
          <div class="chart-container">
            <h4>Resource Allocation by Stage</h4>
            <div class="chart-placeholder">
              <p>Stage-wise resource allocation visualization</p>
              <div class="mock-pie-chart"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Alerts Panel -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Active Alerts & Notifications</h2>
        <div class="section-actions">
          <button @click="viewAllAlerts" class="btn btn-outline">
            View All Alerts
          </button>
          <button @click="acknowledgeAllAlerts" class="btn btn-secondary" :disabled="activeAlerts.length === 0">
            Acknowledge All
          </button>
        </div>
      </div>

      <div v-if="activeAlerts.length > 0" class="alerts-panel">
        <div 
          v-for="alert in activeAlerts" 
          :key="alert.alert_id"
          class="alert-card"
          :class="alert.severity"
        >
          <div class="alert-header">
            <div class="alert-severity-indicator" :class="alert.severity"></div>
            <h3>{{ formatAlertType(alert.alert_type) }}</h3>
            <span class="alert-timestamp">{{ formatTimestamp(alert.timestamp) }}</span>
            <span class="alert-resolution" :class="alert.resolution_status">
              {{ alert.resolution_status }}
            </span>
          </div>
          <div class="alert-content">
            <p class="alert-message">{{ alert.message }}</p>
            <div class="alert-details">
              <div class="detail-item">
                <span class="label">Affected Component:</span>
                <span class="value">{{ alert.affected_component }}</span>
              </div>
              <div v-if="alert.details" class="detail-item">
                <span class="label">Details:</span>
                <span class="value">{{ alert.details }}</span>
              </div>
            </div>
          </div>
          <div class="alert-actions">
            <button 
              @click="acknowledgeAlert(alert.alert_id)"
              class="btn btn-sm btn-outline"
              :disabled="alert.resolution_status !== 'open'"
            >
              Acknowledge
            </button>
            <button 
              @click="resolveAlert(alert.alert_id)"
              class="btn btn-sm btn-primary"
              :disabled="alert.resolution_status === 'resolved'"
            >
              Mark Resolved
            </button>
            <button 
              @click="viewAlertDetails(alert.alert_id)"
              class="btn btn-sm btn-outline"
            >
              View Details
            </button>
          </div>
        </div>
      </div>
      <div v-else class="no-alerts">
        <div class="no-alerts-icon">✅</div>
        <h3>No Active Alerts</h3>
        <p>All systems operating normally within expected parameters.</p>
      </div>
    </div>

    <!-- Evolution Metrics Dashboard -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Evolution Performance Metrics</h2>
        <div class="section-actions">
          <select v-model="metricsTimeRange" class="form-control-sm">
            <option value="1">Last 1 hour</option>
            <option value="6">Last 6 hours</option>
            <option value="24">Last 24 hours</option>
            <option value="168">Last 7 days</option>
          </select>
        </div>
      </div>

      <div class="evolution-metrics">
        <div class="metrics-grid">
          <div class="metric-card" v-for="(value, metric) in evolutionMetrics" :key="metric">
            <h3>{{ formatMetricName(metric) }}</h3>
            <p class="metric-value">{{ formatMetricValue(metric, value) }}</p>
            <div class="metric-trend-indicator">
              <span class="trend-icon" :class="getMetricTrend(metric)">📈</span>
              <span class="trend-text">{{ getMetricTrendText(metric) }}</span>
            </div>
            <div class="metric-description">
              {{ getMetricDescription(metric) }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Stage Control Panel -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Stage Control Panel</h2>
        <div class="section-actions">
          <button @click="pauseAllStages" class="btn btn-warning" :disabled="!hasRunningStages">
            Pause All
          </button>
          <button @click="resumeAllStages" class="btn btn-success" :disabled="!hasPausedStages">
            Resume All
          </button>
          <button @click="restartFailedStages" class="btn btn-secondary" :disabled="!hasFailedStages">
            Restart Failed
          </button>
        </div>
      </div>

      <div class="stage-controls">
        <div class="control-grid">
          <div class="control-card" v-for="stage in evolutionStages" :key="stage.stage_name">
            <h4>{{ formatStageName(stage.stage_name) }}</h4>
            <div class="control-status">
              <span class="status-indicator" :class="stage.status"></span>
              {{ stage.status }}
            </div>
            <div class="control-actions">
              <button 
                @click="toggleStagePause(stage.stage_name)"
                class="btn btn-sm"
                :class="stage.status === 'paused' ? 'btn-success' : 'btn-warning'"
                :disabled="stage.status === 'completed' || stage.status === 'failed'"
              >
                {{ stage.status === 'paused' ? 'Resume' : 'Pause' }}
              </button>
              <button 
                @click="restartStage(stage.stage_name)"
                class="btn btn-sm btn-secondary"
                :disabled="stage.status === 'running' || stage.status === 'pending'"
              >
                Restart
              </button>
              <button 
                @click="skipStage(stage.stage_name)"
                class="btn btn-sm btn-outline"
                :disabled="stage.status !== 'pending'"
              >
                Skip
              </button>
            </div>
            <div class="control-details">
              <div class="detail-item">
                <span class="label">Current Phase:</span>
                <span class="value">{{ stage.current_phase }}</span>
              </div>
              <div class="detail-item">
                <span class="label">Progress:</span>
                <span class="value">{{ stage.progress_percentage.toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Loading Overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-content">
        <div class="spinner"></div>
        <p>{{ loadingMessage }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useToast } from 'vue-toastification'

export default {
  name: 'EvolutionProcessMonitor',
  
  setup() {
    const toast = useToast()
    
    // Reactive state
    const evolutionStages = ref([])
    const resourceUsage = ref({})
    const activeAlerts = ref([])
    const evolutionMetrics = ref({})
    const refreshing = ref(false)
    const loading = ref(false)
    const loadingMessage = ref('')
    const autoRefresh = ref(true)
    const refreshInterval = ref(30) // seconds
    const showResourceCharts = ref(false)
    const metricsTimeRange = ref('24') // hours
    let refreshTimer = null
    
    // Computed properties
    const hasRunningStages = computed(() => {
      return evolutionStages.value.some(stage => stage.status === 'running')
    })
    
    const hasPausedStages = computed(() => {
      return evolutionStages.value.some(stage => stage.status === 'paused')
    })
    
    const hasFailedStages = computed(() => {
      return evolutionStages.value.some(stage => stage.status === 'failed')
    })
    
    // Methods
    const refreshMonitoringData = async () => {
      if (refreshing.value) return
      
      refreshing.value = true
      try {
        // Fetch evolution stages
        const stagesResponse = await fetch('/api/evolution/monitoring/stages')
        if (stagesResponse.ok) {
          const data = await stagesResponse.json()
          evolutionStages.value = data.current_stages || []
          resourceUsage.value = data.resource_usage || {}
          evolutionMetrics.value = data.evolution_metrics || {}
        }
        
        // Fetch active alerts
        const alertsResponse = await fetch('/api/evolution/monitoring/alerts?limit=10')
        if (alertsResponse.ok) {
          activeAlerts.value = await alertsResponse.json()
        }
        
        toast.success('Monitoring data refreshed')
      } catch (error) {
        console.error('Error refreshing monitoring data:', error)
        toast.error('Failed to refresh monitoring data')
        
        // Fallback to mock data
        fallbackMockData()
      } finally {
        refreshing.value = false
      }
    }
    
    const fallbackMockData = () => {
      // Mock evolution stages
      evolutionStages.value = [
        {
          stage_name: 'perception',
          current_phase: 'performance_data_collection',
          progress_percentage: 85.5,
          estimated_time_remaining: 300.0,
          resources_consumed: { cpu: 35.2, memory: 42.8, gpu: 15.3 },
          status: 'running'
        },
        {
          stage_name: 'decision',
          current_phase: 'strategy_selection',
          progress_percentage: 45.0,
          estimated_time_remaining: 180.0,
          resources_consumed: { cpu: 12.5, memory: 8.2, gpu: 0.0 },
          status: 'running'
        },
        {
          stage_name: 'execution',
          current_phase: 'knowledge_fusion',
          progress_percentage: 20.0,
          estimated_time_remaining: 600.0,
          resources_consumed: { cpu: 28.7, memory: 35.1, gpu: 25.6 },
          status: 'pending'
        },
        {
          stage_name: 'feedback',
          current_phase: 'results_validation',
          progress_percentage: 0.0,
          estimated_time_remaining: null,
          resources_consumed: { cpu: 0.0, memory: 0.0, gpu: 0.0 },
          status: 'pending'
        }
      ]
      
      // Mock resource usage
      resourceUsage.value = {
        cpu: { current: 45.3, max: 80.0, average: 38.7 },
        memory: { current: 62.8, max: 85.0, average: 58.2 },
        gpu: { current: 28.5, max: 90.0, average: 22.1 },
        network: { current: 12.3, max: 50.0, average: 10.8 }
      }
      
      // Mock alerts
      activeAlerts.value = [
        {
          alert_id: 'alert_001',
          alert_type: 'knowledge_validation_failed',
          severity: 'warning',
          message: '3 knowledge candidates failed validation checks',
          timestamp: Date.now() / 1000 - 1800,
          affected_component: 'knowledge_self_growth_engine',
          resolution_status: 'acknowledged',
          details: 'Validation failed due to insufficient cross-source verification'
        },
        {
          alert_id: 'alert_002',
          alert_type: 'resource_insufficient',
          severity: 'info',
          message: 'GPU memory usage approaching limit (85%)',
          timestamp: Date.now() / 1000 - 900,
          affected_component: 'model_self_iteration_engine',
          resolution_status: 'open',
          details: 'Current GPU usage: 82%, threshold: 85%'
        }
      ]
      
      // Mock evolution metrics
      evolutionMetrics.value = {
        knowledge_growth_rate: 2.5,
        model_improvement_rate: 1.8,
        cross_domain_transfer_success: 0.75,
        evolution_efficiency: 0.68,
        stage_completion_rate: 0.5,
        resource_utilization: 0.72,
        error_rate: 0.02,
        throughput: 45.2
      }
    }
    
    const formatStageName = (stageName) => {
      const mapping = {
        'perception': 'Perception',
        'decision': 'Decision Making',
        'execution': 'Execution',
        'feedback': 'Feedback Loop'
      }
      return mapping[stageName] || stageName.charAt(0).toUpperCase() + stageName.slice(1)
    }
    
    const formatTimeRemaining = (seconds) => {
      if (!seconds) return 'N/A'
      
      const hours = Math.floor(seconds / 3600)
      const minutes = Math.floor((seconds % 3600) / 60)
      
      if (hours > 0) {
        return `${hours}h ${minutes}m`
      } else {
        return `${minutes}m`
      }
    }
    
    const formatResourceName = (resource) => {
      const mapping = {
        'cpu': 'CPU Usage',
        'memory': 'Memory Usage',
        'gpu': 'GPU Usage',
        'network': 'Network I/O'
      }
      return mapping[resource] || resource.toUpperCase()
    }
    
    const getResourceLimit = (resource) => {
      const limits = {
        'cpu': 90,
        'memory': 85,
        'gpu': 90,
        'network': 80
      }
      return limits[resource] || 80
    }
    
    const getTrendClass = (current, average) => {
      if (current > average * 1.1) return 'trend-up'
      if (current < average * 0.9) return 'trend-down'
      return 'trend-stable'
    }
    
    const getTrendIcon = (current, average) => {
      if (current > average * 1.1) return '📈'
      if (current < average * 0.9) return '📉'
      return '➡️'
    }
    
    const getTrendText = (current, average) => {
      const diff = ((current - average) / average * 100).toFixed(1)
      if (current > average * 1.1) return `+${diff}% above average`
      if (current < average * 0.9) return `${diff}% below average`
      return 'Stable'
    }
    
    const getUsageLevelClass = (usage) => {
      if (usage < 50) return 'usage-low'
      if (usage < 75) return 'usage-medium'
      if (usage < 90) return 'usage-high'
      return 'usage-critical'
    }
    
    const formatAlertType = (alertType) => {
      const mapping = {
        'knowledge_validation_failed': 'Knowledge Validation Failed',
        'model_rollback': 'Model Rollback',
        'resource_insufficient': 'Resource Insufficient',
        'cross_domain_failure': 'Cross-Domain Failure',
        'stage_timeout': 'Stage Timeout',
        'system_error': 'System Error'
      }
      return mapping[alertType] || alertType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    const formatTimestamp = (timestamp) => {
      const date = new Date(timestamp * 1000)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const acknowledgeAlert = async (alertId) => {
      try {
        // In a real implementation, this would call an API endpoint
        const alert = activeAlerts.value.find(a => a.alert_id === alertId)
        if (alert) {
          alert.resolution_status = 'acknowledged'
          toast.success(`Alert ${alertId} acknowledged`)
        }
      } catch (error) {
        console.error('Error acknowledging alert:', error)
        toast.error('Failed to acknowledge alert')
      }
    }
    
    const resolveAlert = async (alertId) => {
      try {
        // In a real implementation, this would call an API endpoint
        const alert = activeAlerts.value.find(a => a.alert_id === alertId)
        if (alert) {
          alert.resolution_status = 'resolved'
          toast.success(`Alert ${alertId} marked as resolved`)
        }
      } catch (error) {
        console.error('Error resolving alert:', error)
        toast.error('Failed to resolve alert')
      }
    }
    
    const viewAlertDetails = (alertId) => {
      toast.info(`Viewing details for alert ${alertId}`)
      // In a real implementation, this would open a detailed view
    }
    
    const viewAllAlerts = () => {
      toast.info('Opening all alerts view')
      // In a real implementation, this would navigate to alerts page
    }
    
    const acknowledgeAllAlerts = () => {
      activeAlerts.value.forEach(alert => {
        if (alert.resolution_status === 'open') {
          alert.resolution_status = 'acknowledged'
        }
      })
      toast.success('All open alerts acknowledged')
    }
    
    const formatMetricName = (metric) => {
      const mapping = {
        'knowledge_growth_rate': 'Knowledge Growth Rate',
        'model_improvement_rate': 'Model Improvement Rate',
        'cross_domain_transfer_success': 'Cross-Domain Success Rate',
        'evolution_efficiency': 'Evolution Efficiency',
        'stage_completion_rate': 'Stage Completion Rate',
        'resource_utilization': 'Resource Utilization',
        'error_rate': 'Error Rate',
        'throughput': 'Throughput'
      }
      return mapping[metric] || metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    const formatMetricValue = (metric, value) => {
      if (typeof value === 'number') {
        if (metric.includes('rate') || metric.includes('success') || metric.includes('efficiency')) {
          return `${(value * 100).toFixed(1)}%`
        } else if (metric === 'knowledge_growth_rate' || metric === 'model_improvement_rate') {
          return `${value.toFixed(2)}/day`
        } else if (metric === 'throughput') {
          return `${value.toFixed(1)} ops/sec`
        } else if (metric === 'error_rate') {
          return `${(value * 100).toFixed(2)}%`
        }
      }
      return value
    }
    
    const getMetricTrend = (metric) => {
      // Mock trend calculation
      const trends = {
        'knowledge_growth_rate': 'up',
        'model_improvement_rate': 'up',
        'cross_domain_transfer_success': 'stable',
        'evolution_efficiency': 'up',
        'stage_completion_rate': 'down',
        'resource_utilization': 'stable',
        'error_rate': 'down',
        'throughput': 'up'
      }
      return trends[metric] || 'stable'
    }
    
    const getMetricTrendText = (metric) => {
      const trend = getMetricTrend(metric)
      const texts = {
        'up': 'Improving',
        'down': 'Declining',
        'stable': 'Stable'
      }
      return texts[trend]
    }
    
    const getMetricDescription = (metric) => {
      const descriptions = {
        'knowledge_growth_rate': 'Average new knowledge concepts added per day',
        'model_improvement_rate': 'Average model performance improvement per day',
        'cross_domain_transfer_success': 'Success rate of cross-domain capability transfers',
        'evolution_efficiency': 'Overall efficiency of evolution process',
        'stage_completion_rate': 'Percentage of stages completed on time',
        'resource_utilization': 'Utilization efficiency of allocated resources',
        'error_rate': 'Error frequency in evolution operations',
        'throughput': 'Number of evolution operations per second'
      }
      return descriptions[metric] || 'Performance metric'
    }
    
    const toggleResourceCharts = () => {
      showResourceCharts.value = !showResourceCharts.value
    }
    
    const pauseAllStages = () => {
      evolutionStages.value.forEach(stage => {
        if (stage.status === 'running') {
          stage.status = 'paused'
        }
      })
      toast.warning('All running stages paused')
    }
    
    const resumeAllStages = () => {
      evolutionStages.value.forEach(stage => {
        if (stage.status === 'paused') {
          stage.status = 'running'
        }
      })
      toast.success('All paused stages resumed')
    }
    
    const restartFailedStages = () => {
      evolutionStages.value.forEach(stage => {
        if (stage.status === 'failed') {
          stage.status = 'pending'
          stage.progress_percentage = 0
        }
      })
      toast.info('Failed stages queued for restart')
    }
    
    const toggleStagePause = (stageName) => {
      const stage = evolutionStages.value.find(s => s.stage_name === stageName)
      if (stage) {
        if (stage.status === 'paused') {
          stage.status = 'running'
          toast.success(`${formatStageName(stageName)} resumed`)
        } else if (stage.status === 'running') {
          stage.status = 'paused'
          toast.warning(`${formatStageName(stageName)} paused`)
        }
      }
    }
    
    const restartStage = (stageName) => {
      const stage = evolutionStages.value.find(s => s.stage_name === stageName)
      if (stage && (stage.status === 'completed' || stage.status === 'failed')) {
        stage.status = 'pending'
        stage.progress_percentage = 0
        toast.info(`${formatStageName(stageName)} queued for restart`)
      }
    }
    
    const skipStage = (stageName) => {
      const stage = evolutionStages.value.find(s => s.stage_name === stageName)
      if (stage && stage.status === 'pending') {
        stage.status = 'completed'
        stage.progress_percentage = 100
        toast.info(`${formatStageName(stageName)} skipped`)
      }
    }
    
    const startAutoRefresh = () => {
      if (autoRefresh.value && !refreshTimer) {
        refreshTimer = setInterval(() => {
          if (!refreshing.value) {
            refreshMonitoringData()
          }
        }, refreshInterval.value * 1000)
      }
    }
    
    const stopAutoRefresh = () => {
      if (refreshTimer) {
        clearInterval(refreshTimer)
        refreshTimer = null
      }
    }
    
    // Lifecycle hooks
    onMounted(() => {
      refreshMonitoringData()
      startAutoRefresh()
    })
    
    onUnmounted(() => {
      stopAutoRefresh()
    })
    
    // Watch autoRefresh toggle
    const watchAutoRefresh = (newValue) => {
      if (newValue) {
        startAutoRefresh()
      } else {
        stopAutoRefresh()
      }
    }
    
    // Return everything
    return {
      // State
      evolutionStages,
      resourceUsage,
      activeAlerts,
      evolutionMetrics,
      refreshing,
      loading,
      loadingMessage,
      autoRefresh,
      refreshInterval,
      showResourceCharts,
      metricsTimeRange,
      
      // Computed
      hasRunningStages,
      hasPausedStages,
      hasFailedStages,
      
      // Methods
      refreshMonitoringData,
      formatStageName,
      formatTimeRemaining,
      formatResourceName,
      getResourceLimit,
      getTrendClass,
      getTrendIcon,
      getTrendText,
      getUsageLevelClass,
      formatAlertType,
      formatTimestamp,
      acknowledgeAlert,
      resolveAlert,
      viewAlertDetails,
      viewAllAlerts,
      acknowledgeAllAlerts,
      formatMetricName,
      formatMetricValue,
      getMetricTrend,
      getMetricTrendText,
      getMetricDescription,
      toggleResourceCharts,
      pauseAllStages,
      resumeAllStages,
      restartFailedStages,
      toggleStagePause,
      restartStage,
      skipStage
    }
  }
}
</script>

<style scoped>
.evolution-process-monitor {
  padding: 20px;
  max-width: 1600px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 30px;
}

.page-header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
  color: #222;
}

.page-header .subtitle {
  color: #555;
  font-size: 1.1rem;
}

.dashboard-section {
  margin-bottom: 40px;
  padding: 25px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  font-size: 1.8rem;
  color: #222;
}

.section-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}

.auto-refresh-toggle label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.evolution-stages {
  margin-top: 20px;
}

.evolution-stages h3 {
  margin-bottom: 20px;
  color: #333;
  font-size: 1.4rem;
}

.stages-timeline {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.stage-item {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 2px solid var(--border-color);
  transition: all 0.3s ease;
}

.stage-item.active {
  border-color: #2196F3;
  box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
}

.stage-item.running {
  border-left: 4px solid #2196F3;
}

.stage-item.completed {
  border-left: 4px solid #4CAF50;
}

.stage-item.failed {
  border-left: 4px solid #F44336;
}

.stage-item.paused {
  border-left: 4px solid #FF9800;
}

.stage-item.pending {
  border-left: 4px solid #9E9E9E;
}

.stage-icon {
  font-size: 2rem;
  margin-bottom: 15px;
}

.stage-info h4 {
  margin: 0 0 10px 0;
  color: #222;
  font-size: 1.3rem;
}

.stage-phase {
  color: #666;
  font-size: 0.95rem;
  margin-bottom: 15px;
  font-style: italic;
}

.stage-progress {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 15px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #E0E0E0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #2196F3, #21CBF3);
  border-radius: 4px;
  transition: width 0.5s ease;
}

.progress-text {
  font-weight: 600;
  color: #333;
  min-width: 60px;
}

.stage-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.detail-item {
  display: flex;
}

.detail-item .label {
  font-weight: 600;
  color: #666;
  min-width: 80px;
}

.detail-item .value {
  color: #333;
}

.status-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.status-badge.running {
  background-color: #E3F2FD;
  color: #1565C0;
}

.status-badge.completed {
  background-color: #E8F5E9;
  color: #2E7D32;
}

.status-badge.failed {
  background-color: #FFEBEE;
  color: #C62828;
}

.status-badge.paused {
  background-color: #FFF3E0;
  color: #EF6C00;
}

.status-badge.pending {
  background-color: #F5F5F5;
  color: #616161;
}

.resource-metrics {
  margin-top: 20px;
}

.metric-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.metric-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.metric-card h3 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 1.2rem;
}

.metric-value-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: #2196F3;
  margin: 0;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #666;
  font-size: 0.9rem;
}

.trend-up {
  color: #4CAF50;
}

.trend-down {
  color: #F44336;
}

.trend-stable {
  color: #FF9800;
}

.metric-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 15px;
}

.usage-bar {
  height: 6px;
  background: #E0E0E0;
  border-radius: 3px;
  overflow: hidden;
}

.usage-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease;
}

.usage-fill.usage-low {
  background-color: #4CAF50;
}

.usage-fill.usage-medium {
  background-color: #FFC107;
}

.usage-fill.usage-high {
  background-color: #FF9800;
}

.usage-fill.usage-critical {
  background-color: #F44336;
}

.resource-charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-top: 30px;
}

.chart-container {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.chart-container h4 {
  margin: 0 0 15px 0;
  color: #333;
}

.chart-placeholder {
  height: 200px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  color: #666;
}

.mock-chart {
  width: 100%;
  height: 150px;
  position: relative;
  margin-top: 20px;
}

.chart-line {
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  border-radius: 1px;
}

.cpu-line {
  top: 30%;
  background: #2196F3;
  animation: wave 2s ease-in-out infinite;
}

.memory-line {
  top: 50%;
  background: #4CAF50;
  animation: wave 2.5s ease-in-out infinite 0.2s;
}

.gpu-line {
  top: 70%;
  background: #FF9800;
  animation: wave 3s ease-in-out infinite 0.4s;
}

.mock-pie-chart {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: conic-gradient(
    #2196F3 0deg 120deg,
    #4CAF50 120deg 240deg,
    #FF9800 240deg 360deg
  );
  margin-top: 20px;
}

@keyframes wave {
  0%, 100% { transform: scaleX(0.8); opacity: 0.7; }
  50% { transform: scaleX(1); opacity: 1; }
}

.alerts-panel {
  display: flex;
  flex-direction: column;
  gap: 15px;
  margin-top: 20px;
}

.alert-card {
  padding: 20px;
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  background: var(--bg-primary);
}

.alert-card.warning {
  border-left: 4px solid #FF9800;
}

.alert-card.info {
  border-left: 4px solid #2196F3;
}

.alert-card.error {
  border-left: 4px solid #F44336;
}

.alert-card.critical {
  border-left: 4px solid #D32F2F;
  background: #FFEBEE;
}

.alert-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 15px;
  flex-wrap: wrap;
}

.alert-severity-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.alert-severity-indicator.warning {
  background-color: #FF9800;
}

.alert-severity-indicator.info {
  background-color: #2196F3;
}

.alert-severity-indicator.error {
  background-color: #F44336;
}

.alert-severity-indicator.critical {
  background-color: #D32F2F;
}

.alert-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: #222;
}

.alert-timestamp {
  color: #666;
  font-size: 0.9rem;
  margin-left: auto;
}

.alert-resolution {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.alert-resolution.open {
  background-color: #FFF3E0;
  color: #EF6C00;
}

.alert-resolution.acknowledged {
  background-color: #E3F2FD;
  color: #1565C0;
}

.alert-resolution.resolved {
  background-color: #E8F5E9;
  color: #2E7D32;
}

.alert-content {
  margin-bottom: 15px;
}

.alert-message {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 1rem;
  line-height: 1.5;
}

.alert-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.alert-actions {
  display: flex;
  gap: 10px;
}

.no-alerts {
  padding: 40px;
  text-align: center;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 2px dashed var(--border-color);
}

.no-alerts-icon {
  font-size: 3rem;
  margin-bottom: 20px;
  color: #4CAF50;
}

.no-alerts h3 {
  margin: 0 0 10px 0;
  color: #333;
}

.no-alerts p {
  color: #666;
  margin: 0;
}

.evolution-metrics {
  margin-top: 20px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.metric-card h3 {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 1.1rem;
}

.metric-value {
  font-size: 1.8rem;
  font-weight: 700;
  color: #2196F3;
  margin: 10px 0;
}

.metric-trend-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 10px 0;
}

.trend-icon {
  font-size: 1.2rem;
}

.trend-icon.up {
  color: #4CAF50;
}

.trend-icon.down {
  color: #F44336;
}

.trend-icon.stable {
  color: #FF9800;
}

.trend-text {
  color: #666;
  font-size: 0.9rem;
}

.metric-description {
  color: #666;
  font-size: 0.85rem;
  line-height: 1.4;
  margin-top: 10px;
}

.stage-controls {
  margin-top: 20px;
}

.control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.control-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.control-card h4 {
  margin: 0 0 15px 0;
  color: #222;
  font-size: 1.2rem;
}

.control-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 15px;
  color: #666;
  font-weight: 500;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-indicator.running {
  background-color: #2196F3;
  animation: pulse 1.5s infinite;
}

.status-indicator.completed {
  background-color: #4CAF50;
}

.status-indicator.failed {
  background-color: #F44336;
}

.status-indicator.paused {
  background-color: #FF9800;
}

.status-indicator.pending {
  background-color: #9E9E9E;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.control-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 15px;
  flex-wrap: wrap;
}

.control-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background-color: #2196F3;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #0b7dda;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background-color: #545b62;
}

.btn-success {
  background-color: #4CAF50;
  color: white;
}

.btn-success:hover:not(:disabled) {
  background-color: #3d8b40;
}

.btn-warning {
  background-color: #FF9800;
  color: white;
}

.btn-warning:hover:not(:disabled) {
  background-color: #e68900;
}

.btn-outline {
  background-color: transparent;
  color: #2196F3;
  border: 1px solid #2196F3;
}

.btn-outline:hover:not(:disabled) {
  background-color: #2196F3;
  color: white;
}

.btn-sm {
  padding: 5px 10px;
  font-size: 0.8rem;
}

.form-control-sm {
  padding: 6px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  background: white;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading-content {
  text-align: center;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #2196F3;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .section-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
  
  .section-actions {
    width: 100%;
    justify-content: space-between;
  }
  
  .stages-timeline {
    grid-template-columns: 1fr;
  }
  
  .metric-cards,
  .metrics-grid,
  .control-grid {
    grid-template-columns: 1fr;
  }
  
  .resource-charts {
    grid-template-columns: 1fr;
  }
  
  .alert-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .alert-timestamp {
    margin-left: 0;
  }
}
</style>