<template>
  <div class="monitor-dashboard">
    <!-- Real-time Monitoring Dashboard -->
    <div class="dashboard-header">
      <h2>Real-time Monitoring Dashboard</h2>
      <div class="controls">
        <button @click="refreshData" :disabled="isRefreshing" class="refresh-btn">
          <span v-if="isRefreshing">Refreshing...</span>
          <span v-else>Refresh</span>
        </button>
        <div class="auto-refresh">
          <label>
            <input type="checkbox" v-model="autoRefresh" />
            Auto-refresh ({{ refreshInterval }}s)
          </label>
        </div>
      </div>
    </div>

    <!-- System Status Overview -->
    <div class="status-overview">
      <div class="status-card" :class="systemStatus">
        <h3>System Status</h3>
        <p class="status-value">{{ systemStatus === 'normal' ? 'Normal' : systemStatus === 'warning' ? 'Warning' : 'Error' }}</p>
        <p class="uptime">Uptime: {{ systemUptime }}</p>
      </div>
      
      <div class="status-card">
        <h3>Active Models</h3>
        <p class="metric-value">{{ activeModelsCount }}</p>
        <p class="metric-label">Total Models: {{ totalModelsCount }}</p>
      </div>
      
      <div class="status-card">
        <h3>CPU Usage</h3>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: cpuUsage + '%' }"></div>
          </div>
          <span class="progress-text">{{ cpuUsage }}%</span>
        </div>
      </div>
      
      <div class="status-card">
        <h3>Memory Usage</h3>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: memoryUsage + '%' }"></div>
          </div>
          <span class="progress-text">{{ memoryUsage }}%</span>
        </div>
      </div>
    </div>

    <!-- Model Performance Metrics -->
    <div class="metrics-section">
      <h3>Model Performance Metrics</h3>
        <div class="metrics-grid">
          <div class="metric-card" v-for="metric in modelMetrics" :key="metric.name">
            <h4>{{ metric.name === 'accuracy' ? 'Accuracy' : metric.name === 'latency' ? 'Latency' : metric.name === 'throughput' ? 'Throughput' : 'Error Rate' }}</h4>
            <p class="metric-value">{{ metric.value }}</p>
            <p class="metric-trend" :class="metric.trend">
              <span v-if="metric.trend === 'up'">Improving</span>
              <span v-else-if="metric.trend === 'down'">Worsening</span>
              <span v-else>Stable</span>
              {{ metric.change }}
            </p>
          </div>
        </div>
    </div>

    <!-- Real-time Data Stream -->
    <div class="data-stream-section">
      <h3>Real-time Data Stream</h3>
        <div class="stream-container">
          <div class="stream-item" v-for="(item, index) in dataStream" :key="index">
            <span class="timestamp">{{ item.timestamp }}</span>
            <span class="model-name">{{ item.model }}</span>
            <span class="event-type" :class="item.type">{{ item.type === 'processing' ? 'Processing' : item.type === 'success' ? 'Success' : item.type === 'error' ? 'Error' : item.type === 'training' ? 'Training' : item.type === 'update' ? 'Update' : 'Coordination' }}</span>
            <span class="event-details">{{ item.details }}</span>
          </div>
        </div>
    </div>

    <!-- Chart Visualization -->
    <div class="charts-section">
      <h3>Performance Charts</h3>
        <div class="charts-grid">
          <div class="chart-container">
            <h4>CPU & Memory Usage</h4>
            <div v-if="cpuUsage || memoryUsage" class="chart-content">
              <p>CPU: {{ cpuUsage }}%, Memory: {{ memoryUsage }}%</p>
              <!-- Chart component will be added here once backend provides data -->
            </div>
            <div v-else class="no-data">
              <p>No data available</p>
            </div>
          </div>
          <div class="chart-container">
            <h4>Model Performance</h4>
            <div v-if="modelMetrics.length > 0" class="chart-content">
              <p>Model metrics displayed when available</p>
              <!-- Chart component will be added here once backend provides data -->
            </div>
            <div v-else class="no-data">
              <p>No model performance data available</p>
            </div>
          </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MonitorDashboard',
  data() {
    return {
      autoRefresh: true,
      refreshInterval: 5,
      isRefreshing: false,
      systemStatus: 'normal',
      systemUptime: '00:00:00',
      activeModelsCount: 0,
      totalModelsCount: 0,
      cpuUsage: 0,
      memoryUsage: 0,
      modelMetrics: [],
      dataStream: []
    }
  },
  mounted() {
    this.startAutoRefresh();
    this.loadInitialData();
  },
  beforeUnmount() {
    this.stopAutoRefresh();
  },
  methods: {
    startAutoRefresh() {
      if (this.autoRefresh) {
        this.refreshTimer = setInterval(() => {
          this.refreshData();
        }, this.refreshInterval * 1000);
      }
    },
    stopAutoRefresh() {
      if (this.refreshTimer) {
        clearInterval(this.refreshTimer);
      }
    },
    async refreshData() {
      this.isRefreshing = true;
      try {
        // Fetch real data from backend
        const response = await this.$api.get('/api/monitoring/data');
        
        if (response.data.status === 'success') {
          this.updateWithRealData(response.data.data);
        } else {
          throw new Error(response.data.message || 'Failed to fetch monitoring data');
        }
      } catch (error) {
        console.error('Failed to fetch monitoring data:', error);
        this.$errorHandler.handleError(error, 'MonitorDashboard - refreshData');
        // Show notification to user
        if (this.$emit) {
          this.$emit('show-notification', {
            message: 'Failed to refresh monitoring data. Please check your connection.',
            type: 'error'
          });
        }
      } finally {
        this.isRefreshing = false;
      }
    },
    loadInitialData() {
      // Initialize with placeholder values
      this.systemUptime = '00:00:00';
      this.activeModelsCount = 0;
    },
    updateWithRealData(data) {
      // Update component with real data from backend
      // This would be implemented once backend API is available
      if (data) {
        this.systemStatus = data.systemStatus || 'normal';
        this.systemUptime = data.systemUptime || '00:00:00';
        this.activeModelsCount = data.activeModelsCount || 0;
        this.totalModelsCount = data.totalModelsCount || 0;
        this.cpuUsage = data.cpuUsage || 0;
        this.memoryUsage = data.memoryUsage || 0;
        this.modelMetrics = data.modelMetrics || [];
        this.dataStream = data.dataStream || [];
      }
    }
  },
  watch: {
    autoRefresh(newVal) {
      if (newVal) {
        this.startAutoRefresh();
      } else {
        this.stopAutoRefresh();
      }
    }
  }
}
</script>

<style scoped>
.monitor-dashboard {
  padding: 20px;
  background: #ffffff;
      min-height: 100vh;
    }
    
    .no-data {
      padding: 40px;
      text-align: center;
      color: #7f8c8d;
    }

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dashboard-header h2 {
  color: #2c3e50;
  margin: 0;
}

.controls {
  display: flex;
  align-items: center;
  gap: 15px;
}

.refresh-btn {
  padding: 8px 16px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s ease;
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
  cursor: pointer;
}

.status-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.status-card {
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  text-align: center;
}

.status-card.normal {
  border-left: 4px solid #27ae60;
}

.status-card.warning {
  border-left: 4px solid #f39c12;
}

.status-card.critical {
  border-left: 4px solid #e74c3c;
}

.status-card h3 {
  margin: 0 0 10px 0;
  color: #2c3e50;
  font-size: 14px;
  text-transform: uppercase;
}

.status-value {
  font-size: 24px;
  font-weight: bold;
  color: #2c3e50;
  margin: 10px 0;
}

.uptime {
  font-size: 12px;
  color: #7f8c8d;
  margin: 5px 0 0 0;
}

.metric-value {
  font-size: 32px;
  font-weight: bold;
  color: #3498db;
  margin: 10px 0;
}

.metric-label {
  font-size: 12px;
  color: #7f8c8d;
  margin: 0;
}

.progress-container {
  display: flex;
  align-items: center;
  gap: 10px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #ecf0f1;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #3498db;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 14px;
  font-weight: bold;
  color: #2c3e50;
  min-width: 40px;
}

.metrics-section {
  margin-bottom: 20px;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metrics-section h3 {
  margin: 0 0 20px 0;
  color: #2c3e50;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.metric-card {
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
  text-align: center;
}

.metric-card h4 {
  margin: 0 0 10px 0;
  color: #2c3e50;
  font-size: 12px;
  text-transform: uppercase;
}

.metric-trend.up {
  color: #27ae60;
}

.metric-trend.down {
  color: #e74c3c;
}

.metric-trend.stable {
  color: #7f8c8d;
}

.data-stream-section {
  margin-bottom: 20px;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.data-stream-section h3 {
  margin: 0 0 15px 0;
  color: #2c3e50;
}

.stream-container {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #ecf0f1;
  border-radius: 6px;
  padding: 10px;
}

.stream-item {
  display: grid;
  grid-template-columns: 80px 100px 100px 1fr;
  gap: 10px;
  padding: 8px;
  border-bottom: 1px solid #ecf0f1;
  font-size: 12px;
}

.stream-item:last-child {
  border-bottom: none;
}

.timestamp {
  color: #7f8c8d;
  font-family: monospace;
}

.model-name {
  font-weight: bold;
  color: #2c3e50;
}

.event-type {
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 11px;
  text-align: center;
}

.event-type.processing {
  background: #3498db;
  color: white;
}

.event-type.success {
  background: #27ae60;
  color: white;
}

.event-type.error {
  background: #e74c3c;
  color: white;
}

.event-type.training {
  background: #f39c12;
  color: white;
}

.event-type.update {
  background: #9b59b6;
  color: white;
}

.event-type.coordination {
  background: #34495e;
  color: white;
}

.event-details {
  color: #2c3e50;
}

.charts-section {
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.charts-section h3 {
  margin: 0 0 20px 0;
  color: #2c3e50;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
}

.chart-container {
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
}

.chart-container h4 {
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 14px;
}

.chart-placeholder {
  height: 200px;
  background: #ecf0f1;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7f8c8d;
  font-style: italic;
}
</style>
