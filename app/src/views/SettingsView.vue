<template>
  <div class="settings-view">
    <!-- 加载状态指示器 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading"></div>
      <span>{{ $t('common.loading') }}</span>
    </div>
    
    <div class="model-configuration">
      <h3>{{ $t('settings.modelConfig') }}</h3>
      
      <!-- 添加新模型部分 -->
      <div class="add-model-section">
        <h4>{{ $t('settings.addNewModel') }}</h4>
        <div class="add-model-form">
          <div class="form-group">
            <label>{{ $t('settings.modelId') }}</label>
            <input type="text" v-model="newModel.id" :placeholder="$t('settings.modelIdPlaceholder')"
                   @keyup.enter="addNewModel">
            <small class="form-hint">例如: manager, language, audio, vision 等</small>
          </div>
          <div class="form-group">
            <label>{{ $t('settings.modelName') }}</label>
            <input type="text" v-model="newModel.name" :placeholder="$t('settings.modelNamePlaceholder')"
                   @keyup.enter="addNewModel">
            <small class="form-hint">例如: 管理模型, 语言模型, 音频模型等</small>
          </div>
          <button @click="addNewModel" class="add-btn" :disabled="!isValidNewModel || addingModel">
            <span v-if="addingModel" class="loading-small"></span>
            {{ addingModel ? $t('common.loading') : $t('settings.addModel') }}
          </button>
        </div>
      </div>
      
      <!-- 模型列表 -->
      <div v-if="models.length === 0" class="empty-state">
        <p>{{ $t('settings.noModels') }}</p>
        <small>{{ $t('settings.addFirstModelHint') }}</small>
      </div>
      
      <div v-for="(model, index) in models" :key="model.id" class="model-card">
        <div class="model-header">
          <div class="model-info">
            <h4>{{ model.name }} ({{ model.id }})</h4>
            <div class="model-status-container">
              <div class="model-status" :class="model.status">
                {{ $t(`settings.status.${model.status}`) }}
              </div>
              <div class="model-active-indicator" :class="{ active: model.active }">
                {{ model.active ? $t('settings.active') : $t('settings.inactive') }}
              </div>
            </div>
          </div>
          <div class="model-actions">
            <button @click="toggleActivation(model)" :class="['activation-btn', model.active ? 'active' : 'inactive']"
                    :disabled="model.status === 'testing'">
              <span v-if="model.status === 'testing'" class="loading-small"></span>
              {{ model.active ? $t('settings.deactivate') : $t('settings.activate') }}
            </button>
            <button @click="removeModel(index)" class="remove-btn" :disabled="model.status === 'testing'">
              {{ $t('settings.remove') }}
            </button>
          </div>
        </div>
        
        <div class="model-settings">
          <div class="model-type">
            <label>{{ $t('settings.modelType') }}</label>
            <select v-model="model.type" @change="onModelTypeChange(model)" :disabled="model.status === 'testing'">
              <option value="local">{{ $t('settings.localModel') }}</option>
              <option value="api">{{ $t('settings.apiModel') }}</option>
            </select>
          </div>
          
          <div v-if="model.type === 'api'" class="api-settings">
            <div class="form-group">
              <label>{{ $t('settings.apiEndpoint') }}</label>
              <input type="url" v-model="model.apiEndpoint" placeholder="https://api.example.com/v1/models"
                     :disabled="model.status === 'testing'">
              <small class="form-hint" v-if="!isValidUrl(model.apiEndpoint) && model.apiEndpoint">
                {{ $t('settings.invalidUrl') }}
              </small>
            </div>
            
            <div class="form-group">
              <label>{{ $t('settings.apiKey') }}</label>
              <input type="password" v-model="model.apiKey" :placeholder="$t('settings.apiKeyPlaceholder')"
                     :disabled="model.status === 'testing'">
            </div>
            
            <div class="form-group">
              <label>{{ $t('settings.modelName') }}</label>
              <input type="text" v-model="model.modelName" placeholder="gpt-4"
                     :disabled="model.status === 'testing'">
            </div>
            
            <div class="api-actions">
              <button @click="testConnection(model)" class="test-btn" :disabled="!isValidApiConfig(model) || model.status === 'testing'">
                <span v-if="model.status === 'testing'" class="loading-small"></span>
                {{ model.status === 'testing' ? $t('settings.connecting') : $t('settings.testConnection') }}
              </button>
              
              <button v-if="model.status === 'connected'" @click="useAsPrimary(model)" class="primary-btn">
                {{ $t('settings.useAsPrimary') }}
              </button>
            </div>
            
            <div v-if="model.lastTested" class="test-result">
              <small>{{ $t('settings.lastTested') }}: {{ formatDateTime(model.lastTested) }}</small>
            </div>
          </div>
          
          <div v-else class="local-settings">
            <div class="form-group">
              <label>{{ $t('settings.localModelPath') }}</label>
              <input type="text" v-model="model.localPath" :placeholder="$t('settings.localModelPathPlaceholder')">
            </div>
            
            <div class="form-group">
              <label>{{ $t('settings.modelVersion') }}</label>
              <input type="text" v-model="model.version" placeholder="1.0.0">
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="action-buttons">
      <button @click="saveSettings" class="save-btn" :disabled="saving">
        <span v-if="saving" class="loading-small"></span>
        {{ saving ? $t('settings.saving') : $t('settings.save') }}
      </button>
      <button @click="resetSettings" class="reset-btn" :disabled="loading">
        {{ $t('settings.reset') }}
      </button>
    </div>
    
    <!-- 统计信息 -->
    <div class="stats-section">
      <h4>{{ $t('settings.statistics') }}</h4>
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-value">{{ models.length }}</span>
          <span class="stat-label">{{ $t('settings.totalModels') }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">{{ activeModelsCount }}</span>
          <span class="stat-label">{{ $t('settings.activeModels') }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">{{ apiModelsCount }}</span>
          <span class="stat-label">{{ $t('settings.apiModels') }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">{{ connectedModelsCount }}</span>
          <span class="stat-label">{{ $t('settings.connectedModels') }}</span>
        </div>
      </div>
    </div>

    <!-- 模型状态部分 -->
    <div class="model-status-section">
      <h3>{{ $t('home.modelStatus') }}</h3>
      <div class="status-grid">
        <div v-for="modelData in modelPerformanceData" :key="modelData.id" class="model-performance-card">
          <div class="model-name">{{ $t(`models.${modelData.id}`) }}</div>
          <div class="model-status-indicator" :class="modelData.status"></div>
          <div class="model-performance">
            <span>{{ $t('home.performance') }}: {{ modelData.performance }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import errorHandler from '@/utils/errorHandler';

export default {
  name: 'SettingsView',
  data() {
    return {
      models: [],
      newModel: {
        id: '',
        name: '',
        type: 'local',
        status: 'disconnected',
        active: true
      },
      loading: false,
      saving: false,
      addingModel: false,
      modelPerformanceData: [
        { id: 'manager', status: 'active', performance: 95 },
        { id: 'language', status: 'active', performance: 92 },
        { id: 'audio', status: 'active', performance: 88 },
        { id: 'vision_image', status: 'active', performance: 90 },
        { id: 'vision_video', status: 'active', performance: 85 },
        { id: 'spatial', status: 'active', performance: 82 },
        { id: 'sensor', status: 'active', performance: 87 },
        { id: 'computer', status: 'active', performance: 93 },
        { id: 'motion', status: 'active', performance: 80 },
        { id: 'knowledge', status: 'active', performance: 89 },
        { id: 'programming', status: 'active', performance: 91 }
      ],
      modelConnectionStatus: 'connected',
      // 模拟数据用于API不可用时
      mockModels: [
        {
          id: 'manager',
          name: '管理模型',
          type: 'local',
          status: 'connected',
          active: true,
          apiEndpoint: '',
          apiKey: '',
          modelName: '',
          localPath: '/models/manager',
          version: '1.0.0',
          lastTested: new Date()
        },
        {
          id: 'language',
          name: '语言模型',
          type: 'local',
          status: 'connected',
          active: true,
          apiEndpoint: '',
          apiKey: '',
          modelName: '',
          localPath: '/models/language',
          version: '1.2.0',
          lastTested: new Date()
        }
      ]
    }
  },
  methods: {
    async loadModels() {
      this.loading = true;
      try {
        const response = await axios.get('/api/models');
        if (response.data.success) {
          this.models = response.data.models;
        } else {
          errorHandler.handleError('Failed to load model configuration:', response.data.error);
          // API失败时使用模拟数据
          this.models = this.mockModels;
        }
      } catch (error) {
        errorHandler.handleError('Failed to load model configuration:', error);
        // API不可用时使用模拟数据
        this.models = this.mockModels;
      }
      this.loading = false;
    },
    onModelTypeChange(model) {
      // Reset API settings when switching to local
      if (model.type === 'local') {
        model.apiEndpoint = '';
        model.apiKey = '';
        model.modelName = '';
        model.status = 'connected';
      } else {
        model.status = 'disconnected';
      }
    },
    async testConnection(model) {
      model.status = 'testing';
      try {
        const response = await axios.post('/api/models/test-connection', {
          model_id: model.id,
          api_endpoint: model.apiEndpoint,
          api_key: model.apiKey,
          model_name: model.modelName
        });
        
        if (response.data.success) {
          model.status = 'connected';
          model.lastTested = new Date();
          this.$notify({
            title: this.$t('settings.connectionSuccess'),
            message: this.$t('settings.connectionSuccessMsg', { model: this.$t(`models.${model.id}`) }),
            type: 'success'
          });
        } else {
          model.status = 'failed';
          this.$notify({
            title: this.$t('settings.connectionFailed'),
            message: response.data.error || this.$t('settings.connectionFailedMsg'),
            type: 'error'
          });
        }
      } catch (error) {
        model.status = 'failed';
        errorHandler.handleError('Failed to test connection:', error);
        this.$notify({
          title: this.$t('settings.connectionFailed'),
          message: error.response?.data?.error || this.$t('settings.connectionFailedMsg'),
          type: 'error'
        });
      }
    },
    async saveSettings() {
      this.saving = true;
      try {
        const response = await axios.put('/api/models', { models: this.models });
        if (response.data.success) {
          this.$notify({
            title: this.$t('settings.saved'),
            message: this.$t('settings.savedMsg'),
            type: 'success'
          });
        } else {
          errorHandler.handleError('Failed to save settings:', response.data.error);
        }
      } catch (error) {
        errorHandler.handleError('Failed to save settings:', error);
      }
      this.saving = false;
    },
    resetSettings() {
      // Reset to default settings - now loaded from backend
      this.loadModels();
    },
    async addNewModel() {
      // Check if model ID already exists
      if (this.models.some(m => m.id === this.newModel.id)) {
        this.$notify({
          title: this.$t('settings.addModelFailed'),
          message: this.$t('settings.modelIdExists'),
          type: 'error'
        });
        return;
      }

      this.addingModel = true;
      try {
        const response = await axios.post('/api/models', this.newModel);
        if (response.data.success) {
          this.models.push(response.data.model);
          this.newModel = {
            id: '',
            name: '',
            type: 'local',
            status: 'disconnected',
            active: true
          };
          this.$notify({
            title: this.$t('settings.modelAdded'),
            message: this.$t('settings.modelAddedMsg', { model: response.data.model.name }),
            type: 'success'
          });
        } else {
          errorHandler.handleError('Failed to add model:', response.data.error);
        }
      } catch (error) {
        errorHandler.handleError('Failed to add model:', error);
      }
      this.addingModel = false;
    },
    async removeModel(index) {
      const model = this.models[index];
      if (confirm(this.$t('settings.confirmRemove', { model: this.$t(`models.${model.id}`) }))) {
        try {
          const response = await axios.delete(`/api/models/${model.id}`);
          if (response.data.success) {
            this.models.splice(index, 1);
            this.$notify({
              title: this.$t('settings.modelRemoved'),
              message: this.$t('settings.modelRemovedMsg', { model: model.name }),
              type: 'success'
            });
          } else {
            errorHandler.handleError('Failed to delete model:', response.data.error);
          }
        } catch (error) {
          errorHandler.handleError('Failed to delete model:', error);
        }
      }
    },
    async toggleActivation(model) {
      try {
        const response = await axios.patch(`/api/models/${model.id}`, { active: !model.active });
        if (response.data.success) {
          model.active = !model.active;
          const statusKey = model.active ? 'modelActivated' : 'modelDeactivated';
          this.$notify({
            title: this.$t(`settings.${statusKey}`),
            message: this.$t(`settings.${statusKey}Msg`, { model: this.$t(`models.${model.id}`) }),
            type: 'info'
          });
        } else {
          errorHandler.handleError('Failed to toggle model status:', response.data.error);
        }
      } catch (error) {
        errorHandler.handleError('Failed to toggle model status:', error);
      }
    },
    useAsPrimary(model) {
      this.$notify({
        title: this.$t('settings.primaryModelSet'),
        message: this.$t('settings.primaryModelSetMsg', { model: this.$t(`models.${model.id}`) }),
        type: 'success'
      });
    },
    isValidUrl(string) {
      try {
        new URL(string);
        return true;
      } catch (_) {
        return false;
      }
    },
    formatDateTime(date) {
      return new Date(date).toLocaleString('zh-CN');
    }
  },
  computed: {
    // 统计活跃模型数量
    activeModelsCount() {
      return this.models.filter(model => model.active).length;
    },
    // 统计API模型数量
    apiModelsCount() {
      return this.models.filter(model => model.type === 'api').length;
    },
    // 统计已连接模型数量
    connectedModelsCount() {
      return this.models.filter(model => model.status === 'connected').length;
    },
    // 检查新模型是否有效
    isValidNewModel() {
      return this.newModel.id.trim() && this.newModel.name.trim();
    },
    // 检查API配置是否有效
    isValidApiConfig(model) {
      if (model.type !== 'api') return false;
      return model.apiEndpoint && model.modelName && this.isValidUrl(model.apiEndpoint);
    }
  },
  mounted() {
    this.loadModels();
  }
}
</script>

<style scoped>
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f5f5f5;
  --bg-tertiary: #e0e0e0;
  --text-primary: #333333;
  --text-secondary: #666666;
  --border-color: #dddddd;
  --border-dark: #cccccc;
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
  --border-radius-sm: 4px;
  --border-radius-md: 8px;
  --transition-fast: 0.2s ease;
}

.settings-view {
  padding: var(--spacing-lg);
  max-width: 800px;
  margin: 0 auto;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-color);
  border-top: 3px solid var(--text-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-small {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--border-color);
  border-top: 2px solid var(--text-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 8px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.model-configuration {
  margin-top: var(--spacing-xl);
}

.model-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background-color: var(--bg-secondary);
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.model-info h4 {
  margin: 0 0 var(--spacing-xs) 0;
  color: var(--text-primary);
  font-weight: 600;
}

.model-status-container {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
}

.model-status {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: bold;
}

.model-status.connected {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-dark);
}

.model-status.disconnected {
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.model-status.testing {
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.model-status.failed {
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.model-active-indicator {
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 0.8rem;
  background-color: var(--bg-tertiary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.model-active-indicator.active {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--border-dark);
}

.model-actions {
  display: flex;
  gap: var(--spacing-sm);
}

.activation-btn, .remove-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 500;
  transition: all var(--transition-fast);
}

.activation-btn.active {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.activation-btn.inactive {
  background-color: var(--bg-primary);
  color: var(--text-secondary);
}

.activation-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
}

.activation-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.remove-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.remove-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.remove-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.model-settings {
  padding: var(--spacing-sm);
  background-color: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.form-group {
  margin-bottom: var(--spacing-md);
}

label {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 500;
  color: var(--text-primary);
}

input, select {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 1rem;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  transition: border-color var(--transition-fast);
}

input:focus, select:focus {
  outline: none;
  border-color: var(--border-dark);
  box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.05);
}

.form-hint {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin-top: 4px;
  display: block;
}

.api-actions {
  display: flex;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
}

.test-btn, .primary-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color var(--transition-fast);
}

.test-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.test-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.test-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.primary-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.primary-btn:hover {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.test-result {
  margin-top: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.add-model-section {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  background-color: var(--bg-secondary);
}

.add-model-form {
  display: grid;
  grid-template-columns: 1fr 1fr auto;
  gap: var(--spacing-md);
  align-items: end;
}

.add-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color var(--transition-fast);
}

.add-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.add-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.empty-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-secondary);
}

.empty-state p {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 1.1rem;
}

.action-buttons {
  display: flex;
  gap: var(--spacing-md);
  margin-top: var(--spacing-xl);
  justify-content: center;
}

.save-btn, .reset-btn {
  padding: var(--spacing-sm) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  min-width: 120px;
}

.save-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.save-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.save-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.reset-btn {
  background-color: var(--bg-primary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
}

.reset-btn:hover:not(:disabled) {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
}

.reset-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.stats-section {
  margin-top: var(--spacing-xl);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  background-color: var(--bg-secondary);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--spacing-md);
  margin-top: var(--spacing-md);
}

.stat-item {
  text-align: center;
  padding: var(--spacing-md);
  background-color: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.stat-value {
  display: block;
  font-size: 2rem;
  font-weight: bold;
  color: var(--text-primary);
  margin-bottom: var(--spacing-xs);
}

.stat-label {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.model-status-section {
  margin-top: var(--spacing-xl);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  background-color: var(--bg-secondary);
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-md);
  margin-top: var(--spacing-md);
}

.model-performance-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  padding: var(--spacing-sm);
  background-color: var(--bg-primary);
  text-align: center;
}

.model-name {
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: var(--spacing-xs);
}

.model-status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin: 8px auto;
  background-color: var(--text-secondary);
}

.model-status-indicator.active {
  background-color: var(--text-primary);
}

.model-status-indicator.connected {
  background-color: var(--text-primary);
}

.model-performance {
  font-size: 0.9rem;
  color: var(--text-secondary);
}

h3, h4 {
  color: var(--text-primary);
  margin-top: 0;
}

h3 {
  margin-bottom: var(--spacing-lg);
  font-size: 1.5rem;
}

h4 {
  margin-bottom: var(--spacing-md);
  font-size: 1.2rem;
}

@media (max-width: 768px) {
  .add-model-form {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .model-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-md);
  }
  
  .model-actions {
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
