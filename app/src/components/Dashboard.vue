/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


<template>
  <div class="dashboard">
    <div class="resource-monitor">
      <h3>Resource Usage</h3>
      <div class="resource-grid">
        <div class="resource-card">
          <h4>CPU</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: cpuUsage + '%' }"></div>
          </div>
          <span>{{ cpuUsage }}%</span>
        </div>
        <div class="resource-card">
          <h4>Memory</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: memoryUsage + '%' }"></div>
          </div>
          <span>{{ memoryUsage }}%</span>
        </div>
        <div class="resource-card">
          <h4>GPU</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: gpuUsage + '%' }"></div>
          </div>
          <span>{{ gpuUsage }}%</span>
        </div>
      </div>
    </div>
    
    <div class="model-performance">
      <h3>Model Performance</h3>
      <div class="performance-grid">
        <div v-for="model in models" :key="model.name" class="model-card">
          <h4>{{ model.name }}</h4>
          <div class="metrics">
            <div>Accuracy: {{ model.accuracy }}%</div>
            <div>Response Time: {{ model.latency }}ms</div>
            <div>Status: 
              <span :class="model.status">{{ this.getStatusText(model.status) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="real-time-data">
      <h3>Real-time Data</h3>
      <div class="sensor-grid">
        <div v-for="sensor in sensors" :key="sensor.id" class="sensor-card">
          <h4>{{ sensor.name }}</h4>
          <div class="value">{{ sensor.value }} {{ sensor.unit }}</div>
          <div class="status" :class="sensor.status">{{ this.getStatusText(sensor.status) }}</div>
        </div>
      </div>
      
      <div class="emotion-monitor">
        <h4>Emotion State</h4>
        <div class="emotion-indicator">
          <div class="emotion-bar" :style="{ width: emotionValue + '%' }"></div>
        </div>
        <div class="emotion-labels">
          <span>Calm</span>
          <span>Neutral</span>
          <span>Excited</span>
        </div>
        <div class="emotion-value">{{ emotionState }} ({{ emotionValue }}%)</div>
      </div>
    </div>
  </div>
</template>

<script>
import api from '../utils/api.js';

export default {
  name: 'Dashboard',
  data() {
    return {
      cpuUsage: 0,
      memoryUsage: 0,
      gpuUsage: 0,
      models: [],
      sensors: [],
      emotionValue: 50,
      emotionState: 'Neutral',
      loading: true,
      error: null,
      updateInterval: null
    }
  },
  methods: {
    getStatusText(status) {
      const statusMap = {
        active: 'Active',
        warning: 'Warning',
        error: 'Error',
        normal: 'Normal',
        idle: 'Idle',
        training: 'Training',
        connected: 'Connected',
        disconnected: 'Disconnected'
      };
      return statusMap[status] || status;
    },
    
    async loadSystemStats() {
      try {
        const response = await api.system.stats();
        const stats = response.data;
        this.cpuUsage = stats.cpu_usage || 0;
        this.memoryUsage = stats.memory_usage || 0;
        this.gpuUsage = stats.gpu_usage || 0;
      } catch (error) {
        console.error('Failed to load system stats:', error);
        // 设置默认值，如果后端不可用
        this.cpuUsage = 25;
        this.memoryUsage = 45;
        this.gpuUsage = 15;
      }
    },
    
    async loadModels() {
      try {
        const response = await api.models.get();
        this.models = response.data.models || [];
      } catch (error) {
        console.error('Failed to load models:', error);
        // 如果后端不可用，显示空模型列表
        this.models = [];
      }
    },
    
    async loadSensors() {
      try {
        const response = await api.devices.getSensors();
        this.sensors = response.data.sensors || [];
      } catch (error) {
        console.error('Failed to load sensors:', error);
        // 如果后端不可用，显示空传感器列表
        this.sensors = [];
      }
    },
    
    async loadAllData() {
      this.loading = true;
      this.error = null;
      
      try {
        await Promise.all([
          this.loadSystemStats(),
          this.loadModels(),
          this.loadSensors()
        ]);
      } catch (error) {
        this.error = 'Failed to load dashboard data. Please check if the backend server is running.';
        console.error('Dashboard data loading error:', error);
      } finally {
        this.loading = false;
      }
    },
    
    startRealTimeUpdates() {
      // 每5秒更新一次数据
      this.updateInterval = setInterval(() => {
        this.loadAllData();
      }, 5000);
    },
    
    stopRealTimeUpdates() {
      if (this.updateInterval) {
        clearInterval(this.updateInterval);
        this.updateInterval = null;
      }
    }
  },
  async mounted() {
    await this.loadAllData();
    this.startRealTimeUpdates();
  },
  beforeUnmount() {
    this.stopRealTimeUpdates();
  }
}
</script>

<style scoped>
.dashboard {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  padding: 20px;
}

.resource-grid, .performance-grid, .sensor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 15px;
}

.resource-card, .model-card, .sensor-card {
  background: #f9f9f9;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.progress-bar {
  height: 10px;
  background: #e0e0e0;
  border-radius: 5px;
  margin: 10px 0;
}

.progress {
  height: 100%;
  background: #42b983;
  border-radius: 5px;
}

.model-card .metrics {
  margin-top: 10px;
}

.status {
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 0.8em;
}

.status.active {
  background: #d4edda;
  color: #155724;
}

.status.warning {
  background: #fff3cd;
  color: #856404;
}

.status.error {
  background: #f8d7da;
  color: #721c24;
}

.emotion-monitor {
  grid-column: span 2;
  background: #f9f9f9;
  border-radius: 8px;
  padding: 15px;
  margin-top: 15px;
}

.emotion-indicator {
  height: 20px;
  background: #e0e0e0;
  border-radius: 10px;
  margin: 10px 0;
  position: relative;
}

.emotion-bar {
  height: 100%;
  background: var(--primary-color);
  border-radius: 10px;
  transition: width 0.5s ease;
}

.emotion-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 0.9em;
}

.emotion-value {
  text-align: center;
  margin-top: 10px;
  font-weight: bold;
}
</style>
