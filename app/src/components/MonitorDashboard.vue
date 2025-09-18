<template>
  <div class="monitor-dashboard">
    <!-- 实时监视仪表盘 | Real-time Monitoring Dashboard -->
    <div class="dashboard-header">
      <h2>{{ $t('monitor.title') }}</h2>
      <div class="controls">
        <button @click="refreshData" :disabled="isRefreshing" class="refresh-btn">
          <span v-if="isRefreshing">{{ $t('monitor.refreshing') }}</span>
          <span v-else>{{ $t('monitor.refresh') }}</span>
        </button>
        <div class="auto-refresh">
          <label>
            <input type="checkbox" v-model="autoRefresh" />
            {{ $t('monitor.autoRefresh') }} ({{ refreshInterval }}s)
          </label>
        </div>
      </div>
    </div>

    <!-- 系统状态概览 | System Status Overview -->
    <div class="status-overview">
      <div class="status-card" :class="systemStatus">
        <h3>{{ $t('monitor.systemStatus') }}</h3>
        <p class="status-value">{{ $t(`monitor.status.${systemStatus}`) }}</p>
        <p class="uptime">{{ $t('monitor.uptime') }}: {{ systemUptime }}</p>
      </div>
      
      <div class="status-card">
        <h3>{{ $t('monitor.activeModels') }}</h3>
        <p class="metric-value">{{ activeModelsCount }}</p>
        <p class="metric-label">{{ $t('monitor.totalModels') }}: {{ totalModelsCount }}</p>
      </div>
      
      <div class="status-card">
        <h3>{{ $t('monitor.cpuUsage') }}</h3>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: cpuUsage + '%' }"></div>
          </div>
          <span class="progress-text">{{ cpuUsage }}%</span>
        </div>
      </div>
      
      <div class="status-card">
        <h3>{{ $t('monitor.memoryUsage') }}</h3>
        <div class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: memoryUsage + '%' }"></div>
          </div>
          <span class="progress-text">{{ memoryUsage }}%</span>
        </div>
      </div>
    </div>

    <!-- 模型性能指标 | Model Performance Metrics -->
    <div class="metrics-section">
      <h3>{{ $t('monitor.modelMetrics') }}</h3>
      <div class="metrics-grid">
        <div class="metric-card" v-for="metric in modelMetrics" :key="metric.name">
          <h4>{{ $t(`monitor.metrics.${metric.name}`) }}</h4>
          <p class="metric-value">{{ metric.value }}</p>
          <p class="metric-trend" :class="metric.trend">
            <span v-if="metric.trend === 'up'">{{ $t('monitor.trend.up') }}</span>
            <span v-else-if="metric.trend === 'down'">{{ $t('monitor.trend.down') }}</span>
            <span v-else>{{ $t('monitor.trend.stable') }}</span>
            {{ metric.change }}
          </p>
        </div>
      </div>
    </div>

    <!-- 实时数据流 | Real-time Data Stream -->
    <div class="data-stream-section">
      <h3>{{ $t('monitor.realtimeData') }}</h3>
      <div class="stream-container">
        <div class="stream-item" v-for="(item, index) in dataStream" :key="index">
          <span class="timestamp">{{ item.timestamp }}</span>
          <span class="model-name">{{ item.model }}</span>
          <span class="event-type" :class="item.type">{{ $t(`monitor.eventTypes.${item.type}`) }}</span>
          <span class="event-details">{{ item.details }}</span>
        </div>
      </div>
    </div>

    <!-- 图表可视化 | Chart Visualization -->
    <div class="charts-section">
      <h3>{{ $t('monitor.performanceCharts') }}</h3>
      <div class="charts-grid">
        <div class="chart-container">
          <h4>{{ $t('monitor.cpuMemoryChart') }}</h4>
          <div class="chart-placeholder">
            <p>{{ $t('monitor.chartPlaceholder') }}</p>
          </div>
        </div>
        <div class="chart-container">
          <h4>{{ $t('monitor.modelPerformanceChart') }}</h4>
          <div class="chart-placeholder">
            <p>{{ $t('monitor.chartPlaceholder') }}</p>
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
      totalModelsCount: 11,
      cpuUsage: 45,
      memoryUsage: 62,
      modelMetrics: [
        { name: 'accuracy', value: '95.2%', trend: 'up', change: '+0.3%' },
        { name: 'latency', value: '128ms', trend: 'down', change: '-12ms' },
        { name: 'throughput', value: '256 req/s', trend: 'up', change: '+8' },
        { name: 'errorRate', value: '1.2%', trend: 'down', change: '-0.4%' }
      ],
      dataStream: [
        { timestamp: '15:23:45', model: 'Language', type: 'processing', details: 'Processing user query' },
        { timestamp: '15:23:42', model: 'Audio', type: 'success', details: 'Audio processed successfully' },
        { timestamp: '15:23:40', model: 'Vision', type: 'training', details: 'Training epoch 25 completed' },
        { timestamp: '15:23:38', model: 'Knowledge', type: 'update', details: 'Knowledge base updated' },
        { timestamp: '15:23:35', model: 'Manager', type: 'coordination', details: 'Task assigned to Language model' }
      ]
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
        // 模拟数据刷新 | Simulate data refresh
        await new Promise(resolve => setTimeout(resolve, 1000));
        this.updateMetrics();
        this.addDataStreamItem();
      } catch (error) {
        this.$errorHandler.handleError(error, 'MonitorDashboard - refreshData');
      } finally {
        this.isRefreshing = false;
      }
    },
    loadInitialData() {
      // 加载初始数据 | Load initial data
      this.systemUptime = this.calculateUptime();
      this.activeModelsCount = Math.floor(Math.random() * 5) + 3;
    },
    updateMetrics() {
      // 更新指标数据 | Update metric data
      this.cpuUsage = Math.floor(Math.random() * 30) + 30;
      this.memoryUsage = Math.floor(Math.random() * 40) + 40;
      this.activeModelsCount = Math.floor(Math.random() * 5) + 3;
      
      // 更新模型指标 | Update model metrics
      this.modelMetrics.forEach(metric => {
        const change = Math.random() * 10 - 5;
        metric.change = change > 0 ? `+${change.toFixed(1)}` : change.toFixed(1);
        metric.trend = change > 0 ? 'up' : change < 0 ? 'down' : 'stable';
      });
    },
    addDataStreamItem() {
      // 添加新的数据流项目 | Add new data stream item
      const models = ['Language', 'Audio', 'Vision', 'Knowledge', 'Manager', 'Sensor', 'Spatial'];
      const types = ['processing', 'success', 'error', 'training', 'update', 'coordination'];
      const details = [
        'Processing user query',
        'Task completed successfully',
        'Training in progress',
        'Model updated',
        'Coordinating tasks',
        'Sensor data received',
        'Spatial analysis complete'
      ];
      
      const now = new Date();
      const timestamp = now.toTimeString().split(' ')[0];
      
      this.dataStream.unshift({
        timestamp,
        model: models[Math.floor(Math.random() * models.length)],
        type: types[Math.floor(Math.random() * types.length)],
        details: details[Math.floor(Math.random() * details.length)]
      });
      
      // 保持数据流长度 | Keep data stream length
      if (this.dataStream.length > 20) {
        this.dataStream.pop();
      }
    },
    calculateUptime() {
      // 计算系统运行时间 | Calculate system uptime
      const hours = Math.floor(Math.random() * 24);
      const minutes = Math.floor(Math.random() * 60);
      const seconds = Math.floor(Math.random() * 60);
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
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
  background: #f5f7fa;
  min-height: 100vh;
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
