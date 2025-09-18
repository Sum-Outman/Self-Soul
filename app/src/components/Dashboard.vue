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
      <h3>{{ $t('resource_usage') }}</h3>
      <div class="resource-grid">
        <div class="resource-card">
          <h4>CPU</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: cpuUsage + '%' }"></div>
          </div>
          <span>{{ cpuUsage }}%</span>
        </div>
        <div class="resource-card">
          <h4>{{ $t('memory') }}</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: memoryUsage + '%' }"></div>
          </div>
          <span>{{ memoryUsage }}%</span>
        </div>
        <div class="resource-card">
          <h4>{{ $t('gpu') }}</h4>
          <div class="progress-bar">
            <div class="progress" :style="{ width: gpuUsage + '%' }"></div>
          </div>
          <span>{{ gpuUsage }}%</span>
        </div>
      </div>
    </div>
    
    <div class="model-performance">
      <h3>{{ $t('model_performance') }}</h3>
      <div class="performance-grid">
        <div v-for="model in models" :key="model.name" class="model-card">
          <h4>{{ model.name }}</h4>
          <div class="metrics">
            <div>{{ $t('accuracy') }}: {{ model.accuracy }}%</div>
            <div>{{ $t('response_time') }}: {{ model.latency }}ms</div>
            <div>{{ $t('status') }}: 
              <span :class="model.status">{{ $t(model.status) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="real-time-data">
      <h3>{{ $t('real_time_data') }}</h3>
      <div class="sensor-grid">
        <div v-for="sensor in sensors" :key="sensor.id" class="sensor-card">
          <h4>{{ sensor.name }}</h4>
          <div class="value">{{ sensor.value }} {{ sensor.unit }}</div>
          <div class="status" :class="sensor.status">{{ $t(sensor.status) }}</div>
        </div>
      </div>
      
      <div class="emotion-monitor">
        <h4>{{ $t('emotion_state') }}</h4>
        <div class="emotion-indicator">
          <div class="emotion-bar" :style="{ width: emotionValue + '%' }"></div>
        </div>
        <div class="emotion-labels">
          <span>{{ $t('calm') }}</span>
          <span>{{ $t('neutral') }}</span>
          <span>{{ $t('excited') }}</span>
        </div>
        <div class="emotion-value">{{ emotionState }} ({{ emotionValue }}%)</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Dashboard',
  data() {
    return {
      cpuUsage: 45,
      memoryUsage: 60,
      gpuUsage: 30,
      models: [
        { name: this.$t('language_model'), accuracy: 92, latency: 120, status: 'active' },
        { name: this.$t('audio_model'), accuracy: 85, latency: 150, status: 'active' },
        { name: this.$t('image_model'), accuracy: 88, latency: 200, status: 'active' },
        { name: this.$t('video_model'), accuracy: 78, latency: 250, status: 'warning' },
        { name: this.$t('spatial_model'), accuracy: 90, latency: 180, status: 'active' },
        { name: this.$t('sensor_model'), accuracy: 95, latency: 100, status: 'active' },
        { name: this.$t('knowledge_model'), accuracy: 96, latency: 300, status: 'active' },
        { name: this.$t('programming_model'), accuracy: 82, latency: 220, status: 'warning' }
      ],
      sensors: [
        { id: 'temp', name: this.$t('temperature'), value: 25.5, unit: '°C', status: 'normal' },
        { id: 'humidity', name: this.$t('humidity'), value: 45, unit: '%', status: 'normal' },
        { id: 'pressure', name: this.$t('pressure'), value: 1013, unit: 'hPa', status: 'normal' },
        { id: 'light', name: this.$t('light'), value: 850, unit: 'lux', status: 'normal' }
      ],
      emotionValue: 50,
      emotionState: this.$t('neutral')
    }
  },
  mounted() {
    // 模拟实时数据更新
    setInterval(() => {
      this.cpuUsage = Math.min(100, Math.max(10, this.cpuUsage + (Math.random() - 0.5) * 5));
      this.memoryUsage = Math.min(100, Math.max(20, this.memoryUsage + (Math.random() - 0.5) * 3));
      this.gpuUsage = Math.min(100, Math.max(15, this.gpuUsage + (Math.random() - 0.5) * 4));
      
      // 更新传感器数据
      this.sensors.forEach(sensor => {
        if (sensor.id === 'temp') {
          sensor.value = (25 + Math.random() * 5).toFixed(1);
        } else if (sensor.id === 'humidity') {
          sensor.value = Math.floor(40 + Math.random() * 20);
        } else if (sensor.id === 'pressure') {
          sensor.value = Math.floor(1000 + Math.random() * 30);
        } else if (sensor.id === 'light') {
          sensor.value = Math.floor(800 + Math.random() * 200);
        }
      });
      
      // 随机更新模型状态
      this.models.forEach(model => {
        if (Math.random() > 0.9) {
          model.status = Math.random() > 0.5 ? 'warning' : 'error';
        } else if (Math.random() > 0.8) {
          model.status = 'active';
        }
      });
      
      // 更新情感状态
      this.emotionValue = Math.max(0, Math.min(100, this.emotionValue + (Math.random() - 0.5) * 10));
      if (this.emotionValue > 70) {
        this.emotionState = this.$t('excited');
      } else if (this.emotionValue > 40) {
        this.emotionState = this.$t('neutral');
      } else {
        this.emotionState = this.$t('calm');
      }
    }, 2000);
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
  background: linear-gradient(90deg, #4a86e8, #42b983, #ff6b6b);
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