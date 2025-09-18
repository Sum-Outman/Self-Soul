<!--
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
-->

<template>
  <div class="model-management">
    <h2>{{ $t('modelManagement.title') }}</h2>
    
    <!-- 模型状态概览 | Model Status Overview -->
    <div class="status-overview">
      <div class="overview-card">
        <h3>{{ $t('modelManagement.totalModels') }}</h3>
        <span class="count">{{ models.length }}</span>
      </div>
      <div class="overview-card">
        <h3>{{ $t('modelManagement.activeModels') }}</h3>
        <span class="count active">{{ activeModelsCount }}</span>
      </div>
      <div class="overview-card">
        <h3>{{ $t('modelManagement.externalModels') }}</h3>
        <span class="count external">{{ externalModelsCount }}</span>
      </div>
    </div>

    <!-- 模型列表 | Model List -->
    <div class="model-list">
      <div v-for="model in models" :key="model.id" class="model-card" :class="model.status">
        <div class="model-header">
          <h3>{{ $t(`models.${model.id}`) }}</h3>
          <span class="model-id">({{ model.id }})</span>
          <span class="status-badge" :class="model.status">
            {{ $t(`modelStatus.${model.status}`) }}
          </span>
        </div>

        <div class="model-details">
          <!-- 模型模式选择 | Model Mode Selection -->
          <div class="mode-selection">
            <label>{{ $t('modelManagement.modelMode') }}:</label>
            <select v-model="model.mode" @change="changeModelMode(model)">
              <option value="local">{{ $t('modelManagement.localMode') }}</option>
              <option value="external">{{ $t('modelManagement.externalMode') }}</option>
            </select>
          </div>

          <!-- 外部API配置 | External API Configuration -->
          <div v-if="model.mode === 'external'" class="external-config">
            <div class="config-input">
              <label>{{ $t('modelManagement.endpoint') }}:</label>
              <input type="text" v-model="model.externalConfig.endpoint" 
                     :placeholder="$t('modelManagement.endpointPlaceholder')">
            </div>
            <div class="config-input">
              <label>{{ $t('modelManagement.apiKey') }}:</label>
              <input type="password" v-model="model.externalConfig.apiKey" 
                     :placeholder="$t('modelManagement.apiKeyPlaceholder')">
            </div>
            <div class="config-input">
              <label>{{ $t('modelManagement.modelName') }}:</label>
              <input type="text" v-model="model.externalConfig.modelName" 
                     :placeholder="$t('modelManagement.modelNamePlaceholder')">
            </div>
            <div class="config-actions">
              <button @click="testConnection(model)" class="test-btn" :disabled="!canTestConnection(model)">
                {{ $t('modelManagement.testConnection') }}
              </button>
              <button @click="saveConfig(model)" class="save-btn">
                {{ $t('modelManagement.saveConfig') }}
              </button>
            </div>
            <div v-if="model.connectionStatus" class="connection-status" :class="model.connectionStatus.type">
              {{ model.connectionStatus.message }}
            </div>
          </div>

          <!-- 模型控制按钮 | Model Control Buttons -->
          <div class="control-buttons">
            <button @click="startModel(model)" :disabled="model.status === 'running'" class="start-btn">
              {{ $t('modelManagement.start') }}
            </button>
            <button @click="stopModel(model)" :disabled="model.status !== 'running'" class="stop-btn">
              {{ $t('modelManagement.stop') }}
            </button>
            <button @click="restartModel(model)" class="restart-btn">
              {{ $t('modelManagement.restart') }}
            </button>
          </div>

          <!-- 性能指标 | Performance Metrics -->
          <div v-if="model.metrics" class="performance-metrics">
            <h4>{{ $t('modelManagement.performance') }}</h4>
            <div class="metrics-grid">
              <div class="metric">
                <span class="label">{{ $t('modelManagement.cpuUsage') }}:</span>
                <span class="value">{{ model.metrics.cpuUsage }}%</span>
              </div>
              <div class="metric">
                <span class="label">{{ $t('modelManagement.memoryUsage') }}:</span>
                <span class="value">{{ model.metrics.memoryUsage }}MB</span>
              </div>
              <div class="metric">
                <span class="label">{{ $t('modelManagement.responseTime') }}:</span>
                <span class="value">{{ model.metrics.responseTime }}ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 批量操作 | Batch Operations -->
    <div class="batch-operations">
      <h3>{{ $t('modelManagement.batchOperations') }}</h3>
      <div class="batch-buttons">
        <button @click="startAllModels" class="batch-start">
          {{ $t('modelManagement.startAll') }}
        </button>
        <button @click="stopAllModels" class="batch-stop">
          {{ $t('modelManagement.stopAll') }}
        </button>
        <button @click="restartAllModels" class="batch-restart">
          {{ $t('modelManagement.restartAll') }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ModelManagement',
  data() {
    return {
      models: [
        {
          id: 'manager',
          name: 'Manager Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 15,
            memoryUsage: 256,
            responseTime: 120
          }
        },
        {
          id: 'language',
          name: 'Language Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 25,
            memoryUsage: 512,
            responseTime: 150
          }
        },
        {
          id: 'audio',
          name: 'Audio Model',
          status: 'stopped',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 0,
            memoryUsage: 128,
            responseTime: 0
          }
        },
        {
          id: 'vision_image',
          name: 'Image Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 35,
            memoryUsage: 1024,
            responseTime: 200
          }
        },
        {
          id: 'vision_video',
          name: 'Video Model',
          status: 'error',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 0,
            memoryUsage: 0,
            responseTime: 0
          }
        },
        {
          id: 'spatial',
          name: 'Spatial Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 20,
            memoryUsage: 384,
            responseTime: 180
          }
        },
        {
          id: 'sensor',
          name: 'Sensor Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 18,
            memoryUsage: 256,
            responseTime: 100
          }
        },
        {
          id: 'computer',
          name: 'Computer Control Model',
          status: 'stopped',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 0,
            memoryUsage: 64,
            responseTime: 0
          }
        },
        {
          id: 'motion',
          name: 'Motion Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 22,
            memoryUsage: 512,
            responseTime: 160
          }
        },
        {
          id: 'knowledge',
          name: 'Knowledge Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 30,
            memoryUsage: 2048,
            responseTime: 300
          }
        },
        {
          id: 'programming',
          name: 'Programming Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 28,
            memoryUsage: 1024,
            responseTime: 220
          }
        },
        {
          id: 'emotion',
          name: 'Emotion Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 18,
            memoryUsage: 256,
            responseTime: 90
          }
        },
        {
          id: 'finance',
          name: 'Finance Model',
          status: 'stopped',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 0,
            memoryUsage: 128,
            responseTime: 0
          }
        },
        {
          id: 'medical',
          name: 'Medical Model',
          status: 'stopped',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 0,
            memoryUsage: 192,
            responseTime: 0
          }
        },
        {
          id: 'planning',
          name: 'Planning Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 24,
            memoryUsage: 512,
            responseTime: 180
          }
        },
        {
          id: 'prediction',
          name: 'Prediction Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 26,
            memoryUsage: 768,
            responseTime: 200
          }
        },
        {
          id: 'collaboration',
          name: 'Collaboration Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 20,
            memoryUsage: 384,
            responseTime: 150
          }
        },
        {
          id: 'optimization',
          name: 'Optimization Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 22,
            memoryUsage: 512,
            responseTime: 170
          }
        },
        {
          id: 'autonomous',
          name: 'Autonomous Model',
          status: 'running',
          mode: 'local',
          externalConfig: {
            endpoint: '',
            apiKey: '',
            modelName: ''
          },
          connectionStatus: null,
          metrics: {
            cpuUsage: 28,
            memoryUsage: 1024,
            responseTime: 220
          }
        }
      ]
    };
  },
  computed: {
    activeModelsCount() {
      return this.models.filter(model => model.status === 'running').length;
    },
    externalModelsCount() {
      return this.models.filter(model => model.mode === 'external').length;
    }
  },
  methods: {
    canTestConnection(model) {
      return model.mode === 'external' && 
             model.externalConfig.endpoint && 
             model.externalConfig.apiKey;
    },
    
    async testConnection(model) {
      if (!this.canTestConnection(model)) {
        return;
      }
      
      model.connectionStatus = {
        type: 'testing',
        message: this.$t('modelManagement.connecting')
      };
      
      try {
        // 模拟API连接测试 | Simulate API connection test
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // 随机成功或失败以演示功能 | Random success/failure for demo
        const success = Math.random() > 0.3;
        
        if (success) {
          model.connectionStatus = {
            type: 'success',
            message: this.$t('modelManagement.connectionSuccess')
          };
        } else {
          model.connectionStatus = {
            type: 'error',
            message: this.$t('modelManagement.connectionFailed')
          };
        }
      } catch (error) {
        model.connectionStatus = {
          type: 'error',
          message: this.$t('modelManagement.connectionError')
        };
      }
    },
    
    saveConfig(model) {
      // 保存配置到本地存储或后端 | Save configuration to local storage or backend
      localStorage.setItem(`model_config_${model.id}`, JSON.stringify({
        mode: model.mode,
        externalConfig: model.externalConfig
      }));
      
      this.$notify({
        title: this.$t('modelManagement.configSaved'),
        message: this.$t('modelManagement.configSavedMessage', { model: this.$t(`models.${model.id}`) }),
        type: 'success'
      });
    },
    
    async startModel(model) {
      model.status = 'starting';
      
      try {
        // 模拟启动过程 | Simulate startup process
        await new Promise(resolve => setTimeout(resolve, 1000));
        model.status = 'running';
        
        this.$notify({
          title: this.$t('modelManagement.modelStarted'),
          message: this.$t('modelManagement.modelStartedMessage', { model: this.$t(`models.${model.id}`) }),
          type: 'success'
        });
      } catch (error) {
        model.status = 'error';
        this.$notify({
          title: this.$t('modelManagement.startFailed'),
          message: this.$t('modelManagement.startFailedMessage', { model: this.$t(`models.${model.id}`) }),
          type: 'error'
        });
      }
    },
    
    async stopModel(model) {
      model.status = 'stopping';
      
      try {
        // 模拟停止过程 | Simulate stop process
        await new Promise(resolve => setTimeout(resolve, 500));
        model.status = 'stopped';
        
        this.$notify({
          title: this.$t('modelManagement.modelStopped'),
          message: this.$t('modelManagement.modelStoppedMessage', { model: this.$t(`models.${model.id}`) }),
          type: 'success'
        });
      } catch (error) {
        model.status = 'error';
        this.$notify({
          title: this.$t('modelManagement.stopFailed'),
          message: this.$t('modelManagement.stopFailedMessage', { model: this.$t(`models.${model.id}`) }),
          type: 'error'
        });
      }
    },
    
    async restartModel(model) {
      await this.stopModel(model);
      await this.startModel(model);
    },
    
    changeModelMode(model) {
      if (model.mode === 'local') {
        model.connectionStatus = null;
      }
    },
    
    async startAllModels() {
      for (const model of this.models) {
        if (model.status !== 'running') {
          await this.startModel(model);
        }
      }
    },
    
    async stopAllModels() {
      for (const model of this.models) {
        if (model.status === 'running') {
          await this.stopModel(model);
        }
      }
    },
    
    async restartAllModels() {
      await this.stopAllModels();
      await this.startAllModels();
    }
  },
  mounted() {
    // 加载保存的配置 | Load saved configurations
    this.models.forEach(model => {
      const savedConfig = localStorage.getItem(`model_config_${model.id}`);
      if (savedConfig) {
        const config = JSON.parse(savedConfig);
        model.mode = config.mode;
        model.externalConfig = config.externalConfig;
      }
    });
  }
};
</script>

<style scoped>
.model-management {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.status-overview {
  display: flex;
  gap: 20px;
  margin-bottom: 30px;
}

.overview-card {
  flex: 1;
  background: #ffffff;
  color: #222222;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  border: 1px solid #e0e0e0;
}

.overview-card h3 {
  margin: 0 0 10px 0;
  font-size: 1rem;
  opacity: 0.9;
}

.count {
  font-size: 2rem;
  font-weight: bold;
}

.count.active {
  color: #444444;
}

.count.external {
  color: #666666;
}

.model-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.model-card {
  background: white;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #666666;
}

.model-card.running {
  border-left-color: #444444;
}

.model-card.stopped {
  border-left-color: #888888;
}

.model-card.error {
  border-left-color: #222222;
}

.model-card.starting,
.model-card.stopping {
  border-left-color: #666666;
}

.model-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
  flex-wrap: wrap;
  gap: 10px;
}

.model-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: #303133;
}

.model-id {
  color: #909399;
  font-size: 0.9rem;
}

.status-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: bold;
}

.status-badge.running {
  background: #f0f0f0;
  color: #444444;
}

.status-badge.stopped {
  background: #f8f8f8;
  color: #888888;
}

.status-badge.error {
  background: #f5f5f5;
  color: #222222;
}

.status-badge.starting,
.status-badge.stopping {
  background: #fafafa;
  color: #666666;
}

.model-details {
  margin-top: 15px;
}

.mode-selection {
  margin-bottom: 15px;
}

.mode-selection label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #606266;
}

.mode-selection select {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  background: white;
}

.external-config {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 6px;
  margin-bottom: 15px;
}

.config-input {
  margin-bottom: 10px;
}

.config-input label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #606266;
  font-size: 0.9rem;
}

.config-input input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  background: white;
}

.config-actions {
  display: flex;
  gap: 10px;
  margin-top: 15px;
}

.test-btn, .save-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
}

.test-btn {
  background: #666666;
  color: white;
}

.test-btn:disabled {
  background: #cccccc;
  cursor: not-allowed;
}

.save-btn {
  background: #888888;
  color: white;
}

.connection-status {
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 0.9rem;
}

.connection-status.testing {
  background: #f5f5f5;
  color: #666666;
}

.connection-status.success {
  background: #f0f0f0;
  color: #444444;
}

.connection-status.error {
  background: #f8f8f8;
  color: #222222;
}

.control-buttons {
  display: flex;
  gap: 10px;
  margin: 15px 0;
}

.start-btn, .stop-btn, .restart-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
}

.start-btn {
  background: #666666;
  color: white;
}

.start-btn:disabled {
  background: #cccccc;
  cursor: not-allowed;
}

.stop-btn {
  background: #888888;
  color: white;
}

.stop-btn:disabled {
  background: #dddddd;
  cursor: not-allowed;
}

.restart-btn {
  background: #777777;
  color: white;
}

.performance-metrics {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #ebeef5;
}

.performance-metrics h4 {
  margin: 0 0 10px 0;
  color: #606266;
  font-size: 1rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 10px;
}

.metric {
  display: flex;
  flex-direction: column;
}

.label {
  font-size: 0.8rem;
  color: #909399;
  margin-bottom: 2px;
}

.value {
  font-weight: bold;
  color: #303133;
}

.batch-operations {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.batch-operations h3 {
  margin: 0 0 15px 0;
  color: #303133;
}

.batch-buttons {
  display: flex;
  gap: 15px;
}

.batch-start, .batch-stop, .batch-restart {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: bold;
}

.batch-start {
  background: #666666;
  color: white;
}

.batch-stop {
  background: #888888;
  color: white;
}

.batch-restart {
  background: #777777;
  color: white;
}

.batch-start:hover, .batch-stop:hover, .batch-restart:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

@media (max-width: 768px) {
  .status-overview {
    flex-direction: column;
  }
  
  .model-list {
    grid-template-columns: 1fr;
  }
  
  .batch-buttons {
    flex-direction: column;
  }
}
</style>
