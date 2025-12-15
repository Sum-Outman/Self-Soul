<template>
  <div class="training-control-panel">
      <h2>Model Training</h2>
      
      <div class="model-selection">
        <h3>Select Models</h3>
      <div v-for="model in availableModels" :key="model.id" class="model-item">
            <input type="checkbox" :id="model.id" v-model="selectedModels" :value="model.id">
            <label :for="model.id">{{ model.name }}</label>
            
            <div class="model-config" v-if="selectedModels.includes(model.id)">
              <div class="config-option">
                <label>Model Source:</label>
                <select v-model="modelSources[model.id]">
                  <option value="local">Local Model</option>
                  <option value="external">External API Model</option>
                </select>
              </div>
          
          <div v-if="modelSources[model.id] === 'external'" class="external-config">
                <div class="config-input">
                  <label>Endpoint:</label>
                  <input type="text" v-model="externalConfigs[model.id].endpoint" placeholder="https://api.example.com">
                </div>
                <div class="config-input">
                  <label>API Key:</label>
                  <input type="password" v-model="externalConfigs[model.id].apiKey" placeholder="API Key">
                </div>
                <div class="config-input">
                  <label>Model Name:</label>
                  <input type="text" v-model="externalConfigs[model.id].modelName" placeholder="Model Name">
                </div>
                <button @click="testConnection(model.id)" class="test-btn">
                  Test Connection
                </button>
            <span v-if="connectionStatus[model.id]" :class="['status', connectionStatus[model.id].status]">
              {{ connectionStatus[model.id].message }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <div class="training-options">
        <h3>Training Options</h3>
        <div class="option">
          <label>Training Mode:</label>
          <select v-model="trainingMode" @change="onTrainingModeChange">
            <option value="individual">Individual</option>
            <option value="joint">Joint</option>
          </select>
        </div>

      <!-- Joint training specific options -->
      <div v-if="trainingMode === 'joint'" class="joint-training-options">
        <div class="option">
          <label>Training Strategy:</label>
          <select v-model="trainingStrategy">
            <option value="standard">Standard</option>
            <option value="knowledge_assisted">Knowledge Assisted</option>
            <option value="progressive">Progressive</option>
            <option value="adaptive">Adaptive</option>
          </select>
        </div>

        <div class="option">
          <label>Knowledge Assistance:</label>
          <input type="checkbox" v-model="knowledgeAssist">
        </div>

        <div class="option">
          <button @click="loadRecommendedCombinations" class="recommend-btn">
            Load Recommendations
          </button>
        </div>

        <!-- Recommended combinations display -->
        <div v-if="recommendedCombinations.length > 0" class="recommended-combinations">
          <h4>Recommended Combinations</h4>
          <div v-for="(combo, index) in recommendedCombinations" :key="index" class="combo-item">
            <input type="radio" :id="'combo-' + index" :value="combo.models" v-model="selectedCombination">
            <label :for="'combo-' + index">
              {{ combo.name }} ({{ combo.models.join(', ') }})
            <span class="combo-score">Score: {{ combo.score.toFixed(2) }}</span>
            </label>
          </div>
        </div>
      </div>

      <div class="option">
          <label>Epochs:</label>
          <input type="number" v-model.number="epochs" min="1" max="1000">
        </div>

      <div class="option">
          <label>Learning Rate:</label>
          <input type="number" v-model.number="learningRate" step="0.001" min="0.0001" max="1">
        </div>

      <div class="option">
          <label>Batch Size:</label>
          <input type="number" v-model.number="batchSize" min="1" max="1024">
        </div>

      <div class="option">
          <label>Validation Split:</label>
          <input type="number" v-model.number="validationSplit" step="0.01" min="0" max="0.5">
        </div>
    </div>

    <div class="actions">
      <button @click="prepareModelsForTraining" :disabled="isPreparing || isTraining">
          {{ isPreparing ? 'Preparing Models...' : 'Prepare Models' }}
        </button>
        <button @click="startTraining" :disabled="isTraining || !allModelsPrepared">
          {{ isTraining ? 'Training in Progress' : 'Start Training' }}
        </button>
        <button @click="stopTraining" :disabled="!isTraining">Stop Training</button>
    </div>

    <div class="model-preparation" v-if="isPreparing">
        <h3>Model Preparation</h3>
        <div v-for="model in modelPreparationStatus" :key="model.id" class="preparation-item">
          <div class="model-name">{{ getModelName(model.id) }} - {{ model.status }}</div>
        <div class="progress-bar">
          <div class="progress" :style="{ width: `${model.progress}%` }" :class="getStatusClass(model.status)"></div>
          <span>{{ model.progress }}%</span>
        </div>
      </div>
    </div>

    <div class="training-progress" v-if="isTraining">
        <h3>Training Progress</h3>
        <div v-for="model in trainingProgress" :key="model.id" class="progress-item">
          <div class="model-name">{{ model.name }}</div>
        <div class="progress-bar">
          <div class="progress" :style="{ width: `${model.progress}%` }"></div>
          <span>{{ model.progress }}%</span>
        </div>
      </div>
    </div>

    <div class="performance-metrics" v-if="trainingMetrics.length > 0">
        <h3>Performance Metrics</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Loss</th>
              <th>Accuracy (%)</th>
              <th>Training Time</th>
            </tr>
          </thead>
        <tbody>
          <tr v-for="metric in trainingMetrics" :key="metric.model">
            <td>{{ getModelName(metric.model) }}</td>
            <td>{{ metric.loss.toFixed(4) }}</td>
            <td>{{ metric.accuracy.toFixed(2) }}%</td>
            <td>{{ formatTime(metric.trainingTime) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import api from '@/utils/api.js';
export default {
  name: 'TrainingControlPanel',
  data() {
    return {
      availableModels: [
        { id: 'manager', name: 'Manager Model' },
        { id: 'language', name: 'Language Model' },
        { id: 'audio', name: 'Audio Model' },
        { id: 'vision_image', name: 'Image Model' },
        { id: 'vision_video', name: 'Video Model' },
        { id: 'spatial', name: 'Spatial Model' },
        { id: 'sensor', name: 'Sensor Model' },
        { id: 'computer', name: 'Computer Control Model' },
        { id: 'motion', name: 'Motion Model' },
        { id: 'knowledge', name: 'Knowledge Model' },
        { id: 'programming', name: 'Programming Model' },
        { id: 'emotion', name: 'Emotion Model' },
        { id: 'finance', name: 'Finance Model' },
        { id: 'medical', name: 'Medical Model' },
        { id: 'planning', name: 'Planning Model' },
        { id: 'prediction', name: 'Prediction Model' },
        { id: 'collaboration', name: 'Collaboration Model' },
        { id: 'optimization', name: 'Optimization Model' },
        { id: 'autonomous', name: 'Autonomous Model' }
      ],
      selectedModels: ['manager', 'language', 'audio', 'vision_image', 'vision_video', 'spatial', 'sensor', 'computer', 'motion', 'knowledge', 'programming'],
      modelSources: {}, // Store the source of each model (local/external)
      externalConfigs: {}, // Store external model configurations
      connectionStatus: {}, // Store connection test status
      trainingMode: 'joint',
      // Joint training specific variables
      trainingStrategy: 'standard',
      knowledgeAssist: false,
      recommendedCombinations: [],
      selectedCombination: null,
      epochs: 50,
      learningRate: 0.001,
      batchSize: 32,
      validationSplit: 0.2,
      isTraining: false,
      isPreparing: false,
      trainingProgress: [],
      trainingMetrics: [],
      trainingInterval: null,
      jobId: null,
      pollingInterval: null,
      preparationInterval: null,
      modelPreparationStatus: []
    };
  },
  computed: {
    allModelsPrepared() {
      // Check if all selected models are prepared
      if (!this.modelPreparationStatus || this.modelPreparationStatus.length === 0) {
        return false;
      }
      
      // Ensure every selected model has preparation status
      return this.selectedModels.every(modelId => {
        const status = this.modelPreparationStatus.find(m => m.id === modelId);
        return status && status.status === 'PREPARED';
      });
    }
  },
  methods: {
    // Get model name by ID without i18n dependency
    getModelName(modelId) {
      const model = this.availableModels.find(m => m.id === modelId);
      return model ? model.name : modelId;
    },
    
    async loadSavedConfigs() {
      try {
        // In pure frontend mode, load locally stored configurations
        const savedConfigs = localStorage.getItem('trainingConfigs');
        if (savedConfigs) {
          const configs = JSON.parse(savedConfigs);
          // Apply saved configurations
          if (configs.modelSources) {
            this.modelSources = configs.modelSources;
          }
          if (configs.externalConfigs) {
            this.externalConfigs = configs.externalConfigs;
          }
          if (configs.selectedModels) {
            this.selectedModels = configs.selectedModels;
          }
          if (configs.trainingMode) {
            this.trainingMode = configs.trainingMode;
          }
          if (configs.trainingParams) {
            const { epochs, learningRate, batchSize, validationSplit } = configs.trainingParams;
            this.epochs = epochs || 50;
            this.learningRate = learningRate || 0.001;
            this.batchSize = batchSize || 32;
            this.validationSplit = validationSplit || 0.2;
          }
        }
      } catch (error) {
        console.error('Error loading saved configurations:', error);
      }
    },
    
    onTrainingModeChange() {
      // Reset joint training related options when training mode changes
      if (this.trainingMode !== 'joint') {
        this.trainingStrategy = 'standard';
        this.knowledgeAssist = false;
        this.recommendedCombinations = [];
        this.selectedCombination = null;
      }
    },

    async loadRecommendedCombinations() {
      try {
        // Build query parameters
        const params = new URLSearchParams({
            models: this.selectedModels.join(','),
            strategy: this.trainingStrategy,
            knowledgeAssist: this.knowledgeAssist.toString()
          });
          
          // Call backend API to get recommended joint training combinations (GET method)
          // Currently using apiClient directly since this specific endpoint isn't in the api namespace
          const response = await apiClient.get(`/api/joint-training/recommendations?${params}`);
        
        if (result.status === 'success') {
          this.recommendedCombinations = result.data.recommendations || result.data;
          if (this.recommendedCombinations && this.recommendedCombinations.length > 0) {
            this.selectedCombination = this.recommendedCombinations[0].models;
          }
          this.$notify({
            title: 'Recommendations Loaded',
            message: `Found ${this.recommendedCombinations ? this.recommendedCombinations.length : 0} recommended combinations`,
            type: 'success'
          });
        } else {
          throw new Error(result.detail || 'Failed to load recommendations');
        }
      } catch (error) {
        console.error('Failed to load recommended combinations:', error);
        this.$notify({
          title: 'Error',
          message: error.message || 'Failed to load recommendations',
          type: 'error'
        });
      }
    },
    async prepareModelsForTraining() {
      try {
        this.isPreparing = true;
        this.modelPreparationStatus = this.selectedModels.map(modelId => ({
          id: modelId,
          status: 'PREPARING',
          progress: 0
        }));
        
        // Start preparing each selected model
        for (const modelId of this.selectedModels) {
          await this.prepareModel(modelId);
        }
        
        this.isPreparing = false;
        this.$notify({
          title: 'Model Preparation Complete',
          message: 'All selected models are prepared for training',
          type: 'success'
        });
      } catch (error) {
        console.error('Model preparation failed:', error);
        this.$notify({
          title: 'Error',
          message: error.message || 'Failed to prepare models',
          type: 'error'
        });
        this.isPreparing = false;
      }
    },
    
    async prepareModel(modelId) {
      try {
        // Get model source
        const source = this.modelSources[modelId] || 'local';
        
        // If it's an external model, check connection first
        if (source === 'external') {
          const config = this.externalConfigs[modelId];
          if (!config.endpoint || !config.apiKey) {
            throw new Error(`Missing configuration for external model ${modelId}`);
          }
          
          // Test connection if not already tested
          const status = this.connectionStatus[modelId];
          if (!status || status.status !== 'success') {
            await this.testConnection(modelId);
            
            // Check connection status after testing
            const newStatus = this.connectionStatus[modelId];
            if (!newStatus || newStatus.status !== 'success') {
              throw new Error(`Connection test failed for model ${modelId}`);
            }
          }
        }
        
        // Update preparation status
        this.updatePreparationStatus(modelId, 'PREPARING', 30);
        
        // Send request to prepare model for training
        const response = await api.post('/api/training/prepare', {
          model_id: modelId,
          source: source,
          config: source === 'external' ? this.externalConfigs[modelId] : null
        });
        
        if (response.data.success) {
          this.updatePreparationStatus(modelId, 'PREPARED', 100);
        } else {
          throw new Error(response.data.message || `Failed to prepare model ${modelId}`);
        }
      } catch (error) {
        console.error(`Failed to prepare model ${modelId}:`, error);
        this.updatePreparationStatus(modelId, 'FAILED', 0);
        throw error;
      }
    },
    
    updatePreparationStatus(modelId, status, progress) {
      const modelIndex = this.modelPreparationStatus.findIndex(m => m.id === modelId);
      if (modelIndex !== -1) {
        this.modelPreparationStatus[modelIndex] = {
          ...this.modelPreparationStatus[modelIndex],
          status,
          progress
        };
      }
    },
    
    getStatusClass(status) {
      switch(status) {
        case 'PREPARED':
          return 'success';
        case 'FAILED':
          return 'error';
        default:
          return '';
      }
    },
    
    async startTraining() {
      try {
        // Check if all models are prepared
        if (!this.allModelsPrepared) {
          this.$notify({
            title: 'Error',
            message: 'All models must be prepared before training',
            type: 'error'
          });
          return;
        }
        
        this.isTraining = true;
        
        // Build training configuration
        const trainingConfig = {
          mode: this.trainingMode,
          models: this.selectedModels,
          parameters: {
            epochs: this.epochs,
            learningRate: this.learningRate,
            batchSize: this.batchSize,
            validationSplit: this.validationSplit
          }
        };
        
        // If it's joint training, add joint training specific configuration
        if (this.trainingMode === 'joint') {
          trainingConfig.jointTraining = {
            strategy: this.trainingStrategy,
            knowledgeAssist: this.knowledgeAssist,
            selectedCombination: this.selectedCombination
          };
        }
        
        // Determine API endpoint
        const apiEndpoint = this.trainingMode === 'joint' 
          ? '/api/joint-training/start' 
          : '/api/train';
        
        // Call backend API to start training
        const response = await api.post(apiEndpoint, trainingConfig);
        
        if (response.data.status === 'success') {
            this.jobId = response.data.job_id;
          this.$notify({
            title: 'Training Started',
            message: `Training job ${this.jobId} has started`,
            type: 'success'
          });
          
          // Start polling training status
          this.startPollingTrainingStatus();
        } else {
          throw new Error(result.detail || 'Failed to start training');
        }
      } catch (error) {
        console.error('Training start failed:', error);
        this.$notify({
            title: 'Error',
            message: error.message || 'Failed to start training',
            type: 'error'
          });
        this.isTraining = false;
      }
    },
    
    async startPollingTrainingStatus() {
      if (!this.jobId) return;
      
      this.pollingInterval = setInterval(async () => {
        try {
            const response = await api.training.status(this.jobId);
            
            if (response.data.status === 'success') {
              const status = response.data.data;
            
            // Update training progress
            if (status.progress !== undefined) {
              this.trainingProgress = this.selectedModels.map(modelId => ({
                id: modelId,
                progress: status.progress
              }));
            }
            
            // Check if training is completed
            if (status.status === 'completed' || status.status === 'failed') {
              this.stopTraining();
              
              if (status.status === 'completed') {
                this.$notify({
                  title: 'Training Completed',
                  message: 'Training has been successfully completed',
                  type: 'success'
                });
                
                // Get training results
                await this.loadTrainingResults();
              } else {
                this.$notify({
                  title: 'Training Failed',
                  message: status.error || 'Training process failed',
                  type: 'error'
                });
              }
            }
          }
        } catch (error) {
          console.error('Failed to get training status:', error);
        }
      }, 2000); // Poll every 2 seconds
    },
    
    async loadTrainingResults() {
      try {
        // Use api instance to get training history
        const response = await api.training.history();
        
        if (response.data.status === 'success') {
          // Find training results for current job
          const currentJob = response.data.data.find(job => job.job_id === this.jobId);
          if (currentJob && currentJob.metrics) {
            this.trainingMetrics = Object.entries(currentJob.metrics).map(([modelId, metrics]) => ({
              model: modelId,
              loss: metrics.loss || 0,
              accuracy: metrics.accuracy || 0,
              trainingTime: metrics.training_time || 0
            }));
          }
        }
      } catch (error) {
        console.error('Failed to load training results:', error);
        // If backend API doesn't exist, set an empty training metrics array
        this.trainingMetrics = [];
      }
    },
    
    stopTraining() {
      this.isTraining = false;
      clearInterval(this.trainingInterval);
      clearInterval(this.pollingInterval);
    },
    
    generateTrainingMetrics() {
      this.trainingMetrics = this.selectedModels.map(modelId => ({
        model: modelId,
        loss: Math.random() * 0.5,
        accuracy: 80 + Math.random() * 20,
        trainingTime: 30 + Math.random() * 120 // 30-150 seconds
      }));
    },
    
    formatTime(seconds) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}m ${secs}s`;
    },
    
    async testConnection(modelId) {
      // Set connection status to testing
        this.$set(this.connectionStatus, modelId, {
          status: 'testing',
          message: 'Connecting...'
        });
      
      try {
        // Get external configuration
        const config = this.externalConfigs[modelId];
        if (!config.endpoint || !config.apiKey) {
          throw new Error('Missing configuration parameters');
        }
        
        // Call backend API for real connection test
        const response = await api.post('/api/models/test-connection', {
          model_id: modelId,
          api_url: config.endpoint,
          api_key: config.apiKey,
          model_name: config.modelName
        });
        
        const result = response.data;
        
        if (result.status === 'success') {
          this.$set(this.connectionStatus, modelId, {
                status: 'success',
                message: 'Connection successful'
              });
          
          // Save successful configuration to system settings
          await this.saveModelConfig(modelId, config);
        } else {
          this.$set(this.connectionStatus, modelId, {
                status: 'error',
                message: result.message || 'Connection failed'
              });
        }
      } catch (error) {
        this.$set(this.connectionStatus, modelId, {
              status: 'error',
              message: error.message || 'Connection error'
            });
      }
    },
    
    async saveModelConfig(modelId, config) {
      try {
        // Save model configuration to backend
        await api.post(`/api/models/${modelId}/switch-to-external`, {
          api_url: config.endpoint,
          api_key: config.apiKey,
          model_name: config.modelName
        });
      } catch (error) {
        console.error('Failed to save configuration:', error);
      }
    },
    
    async loadSavedConfigs() {
      try {
        // Load all system settings, including model configurations
        // Currently using apiClient directly since this specific endpoint isn't in the api namespace
        const response = await apiClient.get('/api/settings');
        const result = response.data;
        
        if (result.status === 'success' && result.data && result.data.models) {
          const modelsSettings = result.data.models;
          
          // Set configuration for each model
          for (const model of this.availableModels) {
            const config = modelsSettings[model.id] || {};
            
            // Set model source
            if (config.source === 'api' || config.type === 'api') {
              this.$set(this.modelSources, model.id, 'external');
              
              // Set external configuration
              this.$set(this.externalConfigs, model.id, {
                endpoint: config.api_url || config.endpoint || '',
                apiKey: config.api_key || '',
                modelName: config.model_name || ''
              });
            } else {
              this.$set(this.modelSources, model.id, 'local');
              this.$set(this.externalConfigs, model.id, {
                endpoint: '',
                apiKey: '',
                modelName: ''
              });
            }
          }
        } else {
          // Default settings
          for (const model of this.availableModels) {
            this.$set(this.modelSources, model.id, 'local');
            this.$set(this.externalConfigs, model.id, {
              endpoint: '',
              apiKey: '',
              modelName: ''
            });
          }
        }
      } catch (error) {
        console.error('Failed to load configurations:', error);
        // Initialize default configurations
        for (const model of this.availableModels) {
          this.$set(this.modelSources, model.id, 'local');
          this.$set(this.externalConfigs, model.id, {
            endpoint: '',
            apiKey: '',
            modelName: ''
          });
        }
      }
    }
  },
  beforeUnmount() {
    clearInterval(this.trainingInterval);
    clearInterval(this.pollingInterval);
  }
};
</script>

<style scoped>
.training-control-panel {
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.model-selection, .training-options, .training-progress, .performance-metrics {
  margin-bottom: 20px;
  padding: 15px;
  background-color: white;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.model-item {
  margin: 8px 0;
}

.model-config {
  margin-left: 25px;
  margin-top: 8px;
  padding: 10px;
  background-color: #f9fafc;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.config-option {
  margin: 8px 0;
  display: flex;
  align-items: center;
}

.config-option label {
  width: 120px;
  margin-right: 10px;
  font-weight: 500;
}

.external-config {
  margin-top: 10px;
  padding: 12px;
  background-color: #f0f4f8;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}

.config-input {
  margin: 8px 0;
  display: flex;
  align-items: center;
}

.config-input label {
  width: 100px;
  margin-right: 10px;
  font-weight: 500;
}

.config-input input {
  padding: 6px 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  flex: 1;
}

.test-btn {
  padding: 6px 12px;
  background-color: #67c23a;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 8px;
}

.test-btn:hover {
  background-color: #5daf34;
}

.status {
  margin-left: 10px;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.status.success {
  background-color: #f0f9eb;
  color: #67c23a;
  border: 1px solid #e1f3d8;
}

.status.error {
  background-color: #fef0f0;
  color: #f56c6c;
  border: 1px solid #fde2e2;
}

.status.testing {
  background-color: #f4f4f5;
  color: #909399;
  border: 1px solid #e9e9eb;
}

.option {
  margin: 10px 0;
  display: flex;
  align-items: center;
}

.option label {
  width: 150px;
  margin-right: 10px;
  font-weight: 500;
}

.option input, .option select {
  padding: 6px 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}

.option input[type="checkbox"] {
  width: 16px;
  height: 16px;
}

.joint-training-options {
  margin-top: 15px;
  padding: 15px;
  background-color: #f0f9ff;
  border-radius: 6px;
  border-left: 4px solid #409eff;
}

.recommend-btn {
  padding: 8px 16px;
  background-color: #e6a23c;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
}

.recommend-btn:hover {
  background-color: #d48806;
}

.recommended-combinations {
  margin-top: 15px;
  padding: 15px;
  background-color: #f9fafc;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.recommended-combinations h4 {
  margin: 0 0 12px 0;
  color: #303133;
  font-size: 16px;
}

.combo-item {
  margin: 8px 0;
  padding: 10px;
  background-color: white;
  border-radius: 4px;
  border: 1px solid #ebeef5;
  transition: all 0.3s;
}

.combo-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.1);
}

.combo-item label {
  display: flex;
  align-items: center;
  cursor: pointer;
  font-weight: 500;
}

.combo-score {
  margin-left: 8px;
  padding: 2px 6px;
  background-color: #67c23a;
  color: white;
  border-radius: 10px;
  font-size: 12px;
  font-weight: normal;
}

.actions {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.actions button {
  padding: 10px 20px;
  background-color: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.3s;
}

.actions button:hover:not(:disabled) {
  background-color: #66b1ff;
}

.actions button:disabled {
  background-color: #a0cfff;
  cursor: not-allowed;
}

.actions button:last-child {
  background-color: #f56c6c;
}

.actions button:last-child:hover:not(:disabled) {
  background-color: #f78989;
}

.model-preparation {
  margin-top: 20px;
  padding: 15px;
  background-color: white;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.preparation-item {
  margin: 12px 0;
}

.training-progress {
  margin-top: 20px;
}

.progress-item {
  margin: 12px 0;
}

.model-name {
  font-weight: 500;
  margin-bottom: 5px;
  color: #303133;
}

.progress-bar {
  height: 24px;
  background-color: #ebeef5;
  border-radius: 12px;
  position: relative;
  overflow: hidden;
}

.progress {
  height: 100%;
  background: linear-gradient(90deg, #409eff, #66b1ff);
  border-radius: 12px;
  transition: width 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.progress.success {
  background: linear-gradient(90deg, #67c23a, #85ce61);
}

.progress.error {
  background: linear-gradient(90deg, #f56c6c, #f78989);
}

.progress-bar span {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  color: white;
  font-size: 12px;
  font-weight: 500;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.performance-metrics {
  margin-top: 20px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

th, td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #ebeef5;
}

th {
  background-color: #f5f7fa;
  font-weight: 600;
  color: #303133;
}

tr:hover {
  background-color: #f5f7fa;
}

/* Responsive Design */
@media (max-width: 768px) {
  .training-control-panel {
    padding: 15px;
  }
  
  .option {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .option label {
    width: 100%;
    margin-bottom: 5px;
  }
  
  .config-option, .config-input {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .config-option label, .config-input label {
    width: 100%;
    margin-bottom: 5px;
  }
  
  .actions {
    flex-direction: column;
  }
  
  .actions button {
    width: 100%;
    margin-bottom: 8px;
  }
}
</style>
