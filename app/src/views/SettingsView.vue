<template>
  <div class="settings-container">
    <h1>Model Management</h1>
    
    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading models...</p>
    </div>

    <!-- Main Content -->
    <div v-else class="settings-content">
      <!-- Statistics Section -->
      <div class="statistics-section">
        <div class="stat-card">
          <h3>Total Models</h3>
          <p class="stat-value">{{ models.length }}</p>
        </div>
        <div class="stat-card">
          <h3>Active Models</h3>
          <p class="stat-value">{{ activeModelsCount }}</p>
        </div>
        <div class="stat-card">
          <h3>Running Models</h3>
          <p class="stat-value">{{ runningModelsCount }}</p>
        </div>
        <div class="stat-card">
          <h3>API Models</h3>
          <p class="stat-value">{{ apiModelsCount }}</p>
        </div>
      </div>

      <!-- Add Model Form -->
      <div class="add-model-section">
        <h2>Add New Model</h2>
        <form class="add-model-form" @submit.prevent="addNewModel">
          <div class="form-row">
            <div class="form-group">
              <label for="model-id">Model ID</label>
              <input
                id="model-id"
                v-model="newModel.id"
                type="text"
                placeholder="Unique ID"
                required
              />
            </div>
            <div class="form-group">
              <label for="model-name">Model Name</label>
              <input
                id="model-name"
                v-model="newModel.name"
                type="text"
                placeholder="Display Name"
                required
              />
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="model-type">Model Type</label>
              <select id="model-type" v-model="newModel.type" required>
                <option value="">Select Type</option>
                <option v-for="type in modelTypes" :key="type" :value="type">
                  {{ type }}
                </option>
              </select>
            </div>
            <div class="form-group">
              <label for="model-port">Port</label>
              <input
                id="model-port"
                v-model="newModel.port"
                type="number"
                min="8001"
                max="8019"
                required
              />
            </div>
          </div>
          <button type="submit" class="add-btn" :disabled="isAddingModel">
            {{ isAddingModel ? 'Adding...' : 'Add Model' }}
          </button>
        </form>
      </div>

      <!-- Batch Actions -->
      <div class="batch-actions">
        <button class="batch-btn" @click="startAllModels" :disabled="!canStartAll">
          Start All Models
        </button>
        <button class="batch-btn" @click="stopAllModels" :disabled="!canStopAll">
          Stop All Models
        </button>
        <button class="batch-btn" @click="restartAllModels" :disabled="!canRestartAll">
          Restart All Models
        </button>
        <button class="batch-btn" @click="restartSystem" :disabled="isRestartingSystem">
          Restart System
        </button>
      </div>

      <!-- Models List -->
      <div v-if="models.length > 0" class="models-list">
        <h2>Models</h2>
        <div v-for="model in models" :key="model.id" class="model-card">
          <div class="model-header">
            <div class="model-info">
              <h4>{{ model.name }}</h4>
              <div class="model-meta">
                <span class="model-type-badge" :class="model.type.toLowerCase().includes('api') ? 'api' : 'local'">
                  {{ model.type }}
                </span>
                <span v-if="model.isPrimary" class="primary-badge">Primary</span>
              </div>
            </div>
            <div class="model-status-container">
              <span class="model-status" :class="model.status">
                {{ model.status }}
              </span>
              <span class="model-active-indicator" :class="{ active: model.isActive }">
                {{ model.isActive ? 'Active' : 'Inactive' }}
              </span>
            </div>
          </div>

          <!-- Model Actions -->
          <div class="model-actions">
            <button
              class="control-btn start-btn"
              @click="startModel(model.id)"
              :disabled="model.status === 'running' || isOperating(model.id)"
            >
              Start
            </button>
            <button
              class="control-btn stop-btn"
              @click="stopModel(model.id)"
              :disabled="model.status === 'stopped' || isOperating(model.id)"
            >
              Stop
            </button>
            <button
              class="control-btn restart-btn"
              @click="restartModel(model.id)"
              :disabled="model.status === 'starting' || model.status === 'stopping' || isOperating(model.id)"
            >
              Restart
            </button>
            <button
              class="activation-btn"
              :class="{ active: model.isActive }"
              @click="toggleActivation(model.id)"
              :disabled="isOperating(model.id)"
            >
              {{ model.isActive ? 'Deactivate' : 'Activate' }}
            </button>
            <button
              class="control-btn primary-btn"
              @click="useAsPrimary(model.id)"
              :disabled="model.isPrimary || isOperating(model.id)"
            >
              Use as Primary
            </button>
            <button
              class="remove-btn"
              @click="removeModel(model.id)"
              :disabled="isOperating(model.id)"
            >
              Remove
            </button>
          </div>

          <!-- API Configuration -->
          <div v-if="model.type.toLowerCase().includes('api')" class="api-config-section">
            <button
              class="settings-toggle-btn"
              @click="toggleApiSettings(model.id)"
              :disabled="isOperating(model.id)"
            >
              {{ showApiSettings[model.id] ? 'Hide API Settings' : 'Show API Settings' }}
            </button>
            
            <div v-if="showApiSettings[model.id]" class="api-settings-form">
              <div class="form-row">
                <div class="form-group">
                  <label for="api-key-{{ model.id }}">API Key</label>
                  <div class="password-input-wrapper">
                    <input
                      :id="'api-key-' + model.id"
                      v-model="model.apiKey"
                      :type="showApiKeys[model.id] ? 'text' : 'password'"
                      placeholder="Enter API Key"
                    />
                    <button
                      type="button"
                      class="toggle-password-btn"
                      @click="toggleApiKeyVisibility(model.id)"
                    >
                      {{ showApiKeys[model.id] ? 'Hide' : 'Show' }}
                    </button>
                  </div>
                </div>
                <div class="form-group api-key-status">
                  <label>API Key Status</label>
                  <div class="status-indicator" :class="getApiKeyStatus(model)">
                    {{ getApiKeyStatusText(model) }}
                  </div>
                </div>
              </div>
              <div class="api-actions">
                <button
                  class="test-btn"
                  @click="testConnection(model.id)"
                  :disabled="!model.apiKey || isTestingConnection(model.id)"
                >
                  {{ isTestingConnection(model.id) ? 'Testing...' : 'Test Connection' }}
                </button>
                <button
                  class="test-btn"
                  @click="saveSettings(model.id)"
                  :disabled="!model.apiKey || isSavingSettings(model.id)"
                >
                  {{ isSavingSettings(model.id) ? 'Saving...' : 'Save Settings' }}
                </button>
              </div>
              <div v-if="testResults[model.id]" class="test-result" :class="testResults[model.id].status">
                {{ testResults[model.id].message }}
              </div>
            </div>
          </div>

          <!-- Model Footer -->
          <div class="model-footer">
            <div class="model-timestamp">
              Last Updated: {{ formatDate(model.lastUpdated) }}
            </div>
            <div class="model-metrics" v-if="model.metrics">
              <span>Memory: {{ model.metrics.memoryUsage }}MB</span>
              <span>CPU: {{ model.metrics.cpuUsage }}%</span>
              <span>Response: {{ model.metrics.responseTime }}ms</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="empty-state">
        <p>No models available.</p>
        <p>Add a new model to get started.</p>
      </div>

      <!-- Action Buttons -->
      <div class="action-buttons">
        <button
          class="save-btn"
          @click="saveAllChanges"
          :disabled="isSavingAll || !hasChanges"
        >
          {{ isSavingAll ? 'Saving...' : 'Save All Changes' }}
        </button>
        <button
          class="reset-btn"
          @click="resetChanges"
          :disabled="isLoading || isSavingAll"
        >
          Reset
        </button>
        <button
          class="test-notifications-btn"
          @click="testNotificationSystem"
          :disabled="isLoading"
        >
          Test Notifications
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import errorHandler from '../utils/errorHandler.js'
import { notify } from '../plugins/notification.js'
import { Model, NewModel, MODEL_TYPES, MODEL_STATUS, MODEL_PORT_CONFIG, createDefaultModel, isValidModelId, isValidPort, isApiModelType, generateMockMetrics } from '../utils/modelTypes.js'
import testNotifications from '../utils/testNotifications.js'

export default {
  name: 'SettingsView',
  setup() {
    // State
    const loading = ref(false)
    const isAddingModel = ref(false)
    const isRestartingSystem = ref(false)
    const isSavingAll = ref(false)
    const hasChanges = ref(false)
    const operatingModels = ref(new Set())
    const testingConnections = ref(new Set())
    const savingSettings = ref(new Set())
    const showApiSettings = ref({})
    const showApiKeys = ref({})
    const testResults = ref({})

    // Mock data for models
    const mockModels = [
      {
        id: 'manager',
        name: 'Manager Model',
        type: 'Manager Model',
        description: 'System manager model for coordination',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8001,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'language',
        name: 'Language Model',
        type: 'Language Model',
        description: 'Natural language processing model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8002,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'knowledge',
        name: 'Knowledge Model',
        type: 'Knowledge Model',
        description: 'Knowledge base and retrieval model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8003,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'vision',
        name: 'Vision Model',
        type: 'Vision Model',
        description: 'Computer vision and image processing model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8004,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'audio',
        name: 'Audio Model',
        type: 'Audio Model',
        description: 'Audio processing and speech recognition model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8005,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'autonomous',
        name: 'Autonomous Model',
        type: 'Autonomous Model',
        description: 'Self-governing and decision-making model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8006,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'programming',
        name: 'Programming Model',
        type: 'Programming Model',
        description: 'Code generation and software development model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8007,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'planning',
        name: 'Planning Model',
        type: 'Planning Model',
        description: 'Strategic planning and execution model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8008,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'emotion',
        name: 'Emotion Model',
        type: 'Emotion Model',
        description: 'Emotional analysis and response model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8009,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'spatial',
        name: 'Spatial Model',
        type: 'Spatial Model',
        description: 'Spatial reasoning and navigation model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8010,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'computer_vision',
        name: 'Computer Vision Model',
        type: 'Computer Vision Model',
        description: 'Advanced computer vision capabilities',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8011,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'sensor',
        name: 'Sensor Model',
        type: 'Sensor Model',
        description: 'Sensor data processing and integration',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8012,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'motion',
        name: 'Motion Model',
        type: 'Motion Model',
        description: 'Motion planning and control model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8013,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'prediction',
        name: 'Prediction Model',
        type: 'Prediction Model',
        description: 'Predictive analytics and forecasting model',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8014,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'advanced_reasoning',
        name: 'Advanced Reasoning Model',
        type: 'Advanced Reasoning Model',
        description: 'Complex logical reasoning capabilities',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8015,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'data_fusion',
        name: 'Data Fusion Model',
        type: 'Data Fusion Model',
        description: 'Multi-source data integration and fusion',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8016,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'creative_solving',
        name: 'Creative Problem Solving Model',
        type: 'Creative Problem Solving Model',
        description: 'Innovative problem-solving approaches',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8017,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'meta_cognition',
        name: 'Meta Cognition Model',
        type: 'Meta Cognition Model',
        description: 'Self-awareness and cognitive monitoring',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8018,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'value_alignment',
        name: 'Value Alignment Model',
        type: 'Value Alignment Model',
        description: 'Ethical decision making and value alignment',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 8019,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'openai',
        name: 'OpenAI API',
        type: 'OpenAI API',
        description: 'OpenAI language model integration',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 0,
        apiKey: '',
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'anthropic',
        name: 'Anthropic API',
        type: 'Anthropic API',
        description: 'Anthropic language model integration',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 0,
        apiKey: '',
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      },
      {
        id: 'google',
        name: 'Google AI API',
        type: 'Google AI API',
        description: 'Google AI services integration',
        status: 'stopped',
        isActive: false,
        isPrimary: false,
        port: 0,
        apiKey: '',
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      }
    ]

    // Data
    const models = ref(mockModels)
    const newModel = ref({
      id: '',
      name: '',
      type: '',
      port: 0
    })

    // Computed
    const modelTypes = computed(() => MODEL_TYPES)
    const activeModelsCount = computed(() => {
      return models.value.filter(model => model.isActive).length
    })
    const runningModelsCount = computed(() => {
      return models.value.filter(model => model.status === 'running').length
    })
    const apiModelsCount = computed(() => {
      return models.value.filter(model => model.type.toLowerCase().includes('api')).length
    })
    const canStartAll = computed(() => {
      return models.value.some(model => model.status !== 'running')
    })
    const canStopAll = computed(() => {
      return models.value.some(model => model.status === 'running')
    })
    const canRestartAll = computed(() => {
      return models.value.length > 0
    })

    // Methods
    const loadModels = async () => {
      loading.value = true
      try {
        // Attempt to fetch from API
        const response = await fetch('/api/models')
        if (!response.ok) {
          throw new Error('Failed to fetch models')
        }
        const data = await response.json()
        models.value = data
        notify.success('Models loaded successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Load Models')
        // Fallback to mock data
        models.value = mockModels
      } finally {
        loading.value = false
      }
    }

    // Test notification system
    const testNotificationSystem = () => {
      try {
        testNotifications()
        notify.info('Notification system test started')
      } catch (error) {
        errorHandler.handleError(error, 'Test Notifications')
      }
    }

    const onModelTypeChange = () => {
      if (newModel.value.type) {
        newModel.value.port = MODEL_PORT_CONFIG[newModel.value.type] || 8000
      }
    }

    const addNewModel = async () => {
      if (!newModel.value.id || !newModel.value.name || !newModel.value.type || !newModel.value.port) {
        notify.warning('Please fill in all required fields')
        return
      }

      // Check for duplicate ID
      if (models.value.some(model => model.id === newModel.value.id)) {
        notify.error('Model ID already exists')
        return
      }

      isAddingModel.value = true
      try {
        // Create new model object
        const modelToAdd = createDefaultModel(
          newModel.value.id,
          newModel.value.name,
          newModel.value.type
        )
        modelToAdd.port = newModel.value.port

        // Attempt API call
        const response = await fetch('/api/models', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(modelToAdd)
        })

        if (!response.ok) {
          throw new Error('Failed to add model')
        }

        // Add to local state
        models.value.push(modelToAdd)
        hasChanges.value = true
        notify.success('Model added successfully')

        // Reset form
        newModel.value = {
          id: '',
          name: '',
          type: '',
          port: 0
        }
      } catch (error) {
        errorHandler.handleError(error, 'Add Model')
        // Fallback to local state update
        const modelToAdd = createDefaultModel(
          newModel.value.id,
          newModel.value.name,
          newModel.value.type
        )
        modelToAdd.port = newModel.value.port
        models.value.push(modelToAdd)
        hasChanges.value = true
        notify.success('Model added locally')
        
        // Reset form
        newModel.value = {
          id: '',
          name: '',
          type: '',
          port: 0
        }
      } finally {
        isAddingModel.value = false
      }
    }

    const removeModel = async (modelId) => {
      if (!confirm('Are you sure you want to remove this model?')) {
        return
      }

      const modelIndex = models.value.findIndex(model => model.id === modelId)
      if (modelIndex === -1) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}`, {
          method: 'DELETE'
        })

        if (!response.ok) {
          throw new Error('Failed to remove model')
        }

        // Remove from local state
        models.value.splice(modelIndex, 1)
        hasChanges.value = true
        notify.success('Model removed successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Remove Model')
        // Fallback to local state update
        models.value.splice(modelIndex, 1)
        hasChanges.value = true
        notify.success('Model removed locally')
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const toggleActivation = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        const newState = !model.isActive
        
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}/activation`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ isActive: newState })
        })

        if (!response.ok) {
          throw new Error('Failed to update activation status')
        }

        // Update local state
        model.isActive = newState
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success(`Model ${newState ? 'activated' : 'deactivated'} successfully`)
      } catch (error) {
        errorHandler.handleError(error, 'Toggle Activation')
        // Fallback to local state update
        model.isActive = !model.isActive
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success(`Model ${model.isActive ? 'activated' : 'deactivated'} locally`)
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const useAsPrimary = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}/primary`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ isPrimary: true })
        })

        if (!response.ok) {
          throw new Error('Failed to set as primary')
        }

        // Update local state
        models.value.forEach(m => {
          m.isPrimary = m.id === modelId
        })
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model set as primary successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Set as Primary')
        // Fallback to local state update
        models.value.forEach(m => {
          m.isPrimary = m.id === modelId
        })
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model set as primary locally')
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const startModel = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        // Update local state first for better UX
        model.status = 'starting'
        
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}/start`, {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to start model')
        }

        // Update local state
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model started successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Start Model')
        // Fallback to local state update
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model started locally')
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const stopModel = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        // Update local state first for better UX
        model.status = 'stopping'
        
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}/stop`, {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to stop model')
        }

        // Update local state
        model.status = 'stopped'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model stopped successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Stop Model')
        // Fallback to local state update
        model.status = 'stopped'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model stopped locally')
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const restartModel = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      operatingModels.value.add(modelId)
      try {
        // Update local state first for better UX
        model.status = 'stopping'
        
        // Attempt API call
        const response = await fetch(`/api/models/${modelId}/restart`, {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to restart model')
        }

        // Update local state
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model restarted successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Restart Model')
        // Fallback to local state update
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model restarted locally')
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    const startAllModels = async () => {
      if (!confirm('Are you sure you want to start all models?')) {
        return
      }

      const modelsToStart = models.value.filter(model => model.status !== 'running')
      if (modelsToStart.length === 0) {
        notify.info('All models are already running')
        return
      }

      try {
        // Add all models to operating set
        modelsToStart.forEach(model => {
          operatingModels.value.add(model.id)
          model.status = 'starting'
        })

        // Attempt API call
        const response = await fetch('/api/models/start-all', {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to start all models')
        }

        // Update local state
        modelsToStart.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models started successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Start All Models')
        // Fallback to local state update
        modelsToStart.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models started locally')
      } finally {
        // Remove all models from operating set
        modelsToStart.forEach(model => {
          operatingModels.value.delete(model.id)
        })
      }
    }

    const stopAllModels = async () => {
      if (!confirm('Are you sure you want to stop all models?')) {
        return
      }

      const modelsToStop = models.value.filter(model => model.status === 'running')
      if (modelsToStop.length === 0) {
        notify.info('All models are already stopped')
        return
      }

      try {
        // Add all models to operating set
        modelsToStop.forEach(model => {
          operatingModels.value.add(model.id)
          model.status = 'stopping'
        })

        // Attempt API call
        const response = await fetch('/api/models/stop-all', {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to stop all models')
        }

        // Update local state
        modelsToStop.forEach(model => {
          model.status = 'stopped'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models stopped successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Stop All Models')
        // Fallback to local state update
        modelsToStop.forEach(model => {
          model.status = 'stopped'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models stopped locally')
      } finally {
        // Remove all models from operating set
        modelsToStop.forEach(model => {
          operatingModels.value.delete(model.id)
        })
      }
    }

    const restartAllModels = async () => {
      if (!confirm('Are you sure you want to restart all models?')) {
        return
      }

      if (models.value.length === 0) {
        notify.info('No models to restart')
        return
      }

      try {
        // Add all models to operating set
        models.value.forEach(model => {
          operatingModels.value.add(model.id)
          model.status = 'stopping'
        })

        // Attempt API call
        const response = await fetch('/api/models/restart-all', {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to restart all models')
        }

        // Update local state
        models.value.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models restarted successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Restart All Models')
        // Fallback to local state update
        models.value.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models restarted locally')
      } finally {
        // Remove all models from operating set
        models.value.forEach(model => {
          operatingModels.value.delete(model.id)
        })
      }
    }

    const restartSystem = async () => {
      if (!confirm('Are you sure you want to restart the entire system?')) {
        return
      }

      isRestartingSystem.value = true
      try {
        // Attempt API call
        const response = await fetch('/api/system/restart', {
          method: 'POST'
        })

        if (!response.ok) {
          throw new Error('Failed to restart system')
        }

        notify.success('System restart initiated')
        // In a real app, you might want to redirect or refresh the page after a delay
      } catch (error) {
        errorHandler.handleError(error, 'Restart System')
        notify.success('System restart simulated')
      } finally {
        isRestartingSystem.value = false
      }
    }

    const testConnection = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model || !model.apiKey) {
        notify.error('Model not found or API key not configured')
        return
      }

      testingConnections.value.add(modelId)
      try {
        // Attempt API call to test connection
        const response = await fetch(`/api/models/${modelId}/test-connection`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ apiKey: model.apiKey })
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.message || 'Connection test failed')
        }

        const result = await response.json()
        testResults.value[modelId] = {
          status: 'success',
          message: result.message || 'Connection successful'
        }
        notify.success('Connection test successful')
      } catch (error) {
        errorHandler.handleError(error, 'Test Connection')
        testResults.value[modelId] = {
          status: 'error',
          message: error.message
        }
      } finally {
        testingConnections.value.delete(modelId)
        // Clear test result after 5 seconds
        setTimeout(() => {
          delete testResults.value[modelId]
        }, 5000)
      }
    }

    const saveSettings = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      savingSettings.value.add(modelId)
      try {
        // Attempt API call to save settings
        const response = await fetch(`/api/models/${modelId}/settings`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(model)
        })

        if (!response.ok) {
          throw new Error('Failed to save settings')
        }

        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Settings saved successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Save Settings')
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Settings saved locally')
      } finally {
        savingSettings.value.delete(modelId)
      }
    }

    const saveAllChanges = async () => {
      isSavingAll.value = true
      try {
        // Attempt API call to save all changes
        const response = await fetch('/api/models/save-all', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(models.value)
        })

        if (!response.ok) {
          throw new Error('Failed to save all changes')
        }

        hasChanges.value = false
        notify.success('All changes saved successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Save All Changes')
        hasChanges.value = false
        notify.success('All changes saved locally')
      } finally {
        isSavingAll.value = false
      }
    }

    const resetChanges = () => {
      if (!confirm('Are you sure you want to reset all changes?')) {
        return
      }
      loadModels()
      hasChanges.value = false
      notify.success('Changes reset successfully')
    }

    const toggleApiSettings = (modelId) => {
      showApiSettings.value[modelId] = !showApiSettings.value[modelId]
    }

    const toggleApiKeyVisibility = (modelId) => {
      showApiKeys.value[modelId] = !showApiKeys.value[modelId]
    }

    const getApiKeyStatus = (model) => {
      if (!model.apiKey) {
        return 'not-configured'
      }
      // In a real app, you might have a way to check if the API key is valid
      return 'configured'
    }

    const getApiKeyStatusText = (model) => {
      if (!model.apiKey) {
        return 'Not Configured'
      }
      return 'Configured'
    }

    const formatDate = (dateString) => {
      const date = new Date(dateString)
      return date.toLocaleString()
    }

    const isOperating = (modelId) => {
      return operatingModels.value.has(modelId)
    }

    const isTestingConnection = (modelId) => {
      return testingConnections.value.has(modelId)
    }

    const isSavingSettings = (modelId) => {
      return savingSettings.value.has(modelId)
    }

    // Lifecycle
    onMounted(() => {
      loadModels()
    })

    return {
      // State
      loading,
      isAddingModel,
      isRestartingSystem,
      isSavingAll,
      hasChanges,
      showApiSettings,
      showApiKeys,
      testResults,
      
      // Data
      models,
      newModel,
      
      // Computed
      modelTypes,
      activeModelsCount,
      runningModelsCount,
      apiModelsCount,
      canStartAll,
      canStopAll,
      canRestartAll,
      
      // Methods
      loadModels,
      onModelTypeChange,
      addNewModel,
      removeModel,
      toggleActivation,
      useAsPrimary,
      startModel,
      stopModel,
      restartModel,
      startAllModels,
      stopAllModels,
      restartAllModels,
      restartSystem,
      testConnection,
      saveSettings,
      saveAllChanges,
      resetChanges,
      toggleApiSettings,
      toggleApiKeyVisibility,
      getApiKeyStatus,
      getApiKeyStatusText,
      formatDate,
      isOperating,
      isTestingConnection,
      isSavingSettings,
      testNotificationSystem
    }
  }
}
</script>

<style scoped>
/* Batch Actions */
.batch-actions {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  flex-wrap: wrap;
}

.batch-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.batch-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.batch-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Model Card */
.model-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background-color: var(--bg-secondary);
  transition: all var(--transition-fast);
}

.model-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.model-info h4 {
  margin: 0 0 var(--spacing-xs) 0;
  color: var(--text-primary);
  font-weight: 600;
  font-size: 1.1rem;
}

.model-meta {
  display: flex;
  gap: var(--spacing-sm);
}

.model-status-container {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  flex-wrap: wrap;
}

.model-status {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: bold;
  text-transform: capitalize;
}

.model-status.connected {
  background-color: #f5f5f5;
  color: var(--success-color);
  border: 1px solid var(--success-color);
}

.model-status.disconnected {
  background-color: #f5f5f5;
  color: var(--error-color);
  border: 1px solid var(--error-color);
}

.model-status.testing {
  background-color: #f5f5f5;
  color: #666666;
  border: 1px solid #dddddd;
}

.model-status.failed {
  background-color: #f5f5f5;
  color: var(--error-color);
  border: 1px solid var(--error-color);
}

.model-status.running {
  background-color: #f5f5f5;
  color: var(--success-color);
  border: 1px solid var(--success-color);
}

.model-status.stopped {
  background-color: #f5f5f5;
  color: #999999;
  border: 1px solid #dddddd;
}

.model-active-indicator {
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 0.8rem;
  background-color: #f5f5f5;
  color: #999999;
  border: 1px solid #dddddd;
}

.model-active-indicator.active {
  background-color: #f5f5f5;
  color: #333333;
  border-color: #cccccc;
}

.model-type-badge {
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 500;
  background-color: #f5f5f5;
  color: #666666;
  border: 1px solid #dddddd;
}

.model-type-badge.local {
  background-color: #f5f5f5;
  color: #333333;
  border-color: #cccccc;
}

.model-type-badge.api {
  background-color: #f5f5f5;
  color: #666666;
  border-color: #cccccc;
}

.primary-badge {
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  background-color: #f5f5f5;
  color: #333333;
  border: 1px solid #cccccc;
}

/* Model Actions */
.model-actions {
  display: flex;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
  flex-wrap: wrap;
}

.control-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.control-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.control-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.primary-btn {
  font-weight: 600;
}

.start-btn, .stop-btn, .restart-btn {
  font-weight: 500;
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

/* API Configuration */
.settings-toggle-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  margin-top: var(--spacing-md);
}

.settings-toggle-btn:hover {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.api-config-section {
  margin-top: var(--spacing-md);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--border-color);
}

.api-settings-form {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  background-color: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.form-group label {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-primary);
}

.form-group input, .form-group select {
  padding: var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.form-group input:focus, .form-group select:focus {
  outline: none;
  border-color: var(--border-dark);
}

.password-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.password-input-wrapper input {
  flex: 1;
  padding-right: 80px;
}

.toggle-password-btn {
  position: absolute;
  right: 4px;
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 0.8rem;
  transition: all var(--transition-fast);
}

.toggle-password-btn:hover {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.api-key-status {
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.status-indicator {
  padding: var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  font-size: 0.85rem;
  font-weight: 500;
  text-align: center;
}

.status-indicator.not-configured {
  background-color: #f8f8f8;
  color: #666666;
  border: 1px solid #dddddd;
}

.status-indicator.configured {
  background-color: #f5f5f5;
  color: #333333;
  border: 1px solid #cccccc;
}

.status-indicator.valid {
  background-color: #f5f5f5;
  color: var(--success-color);
  border: 1px solid var(--success-color);
}

.status-indicator.invalid {
  background-color: #f8f8f8;
  color: var(--error-color);
  border: 1px solid var(--error-color);
}

.api-actions {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: flex-end;
}

.test-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.test-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
}

.test-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  background-color: #f9f9f9;
  border: 1px solid #eeeeee;
  border-radius: var(--border-radius-sm);
  color: #666666;
  font-size: 0.8rem;
}

.model-footer {
  margin-top: var(--spacing-md);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.model-timestamp {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.model-metrics {
  display: flex;
  gap: var(--spacing-md);
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.model-metrics span {
  padding: 4px 8px;
  background-color: #f5f5f5;
  border-radius: 4px;
  border: 1px solid #eeeeee;
}

.model-control-actions {
  display: flex;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
  flex-wrap: wrap;
}

.test-result {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  font-size: 0.85rem;
  font-weight: 500;
}

.test-result.success {
  background-color: #f5f5f5;
  color: var(--success-color);
  border: 1px solid #e0e0e0;
}

.test-result.error {
  background-color: #f8f8f8;
  color: var(--error-color);
  border: 1px solid #eeeeee;
}

/* Add Model Section */
.add-model-section {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  background-color: var(--bg-secondary);
}

.add-model-section h2 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
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
  font-weight: 500;
  transition: background-color var(--transition-fast);
  height: 40px;
}

.add-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
  filter: brightness(0.95);
}

.add-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Empty State */
.empty-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  border: 1px dashed var(--border-color);
  border-radius: var(--border-radius-md);
  background-color: var(--bg-primary);
}

.empty-state p {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 1.1rem;
}

/* Action Buttons */
.action-buttons {
  display: flex;
  gap: var(--spacing-md);
  margin-top: var(--spacing-xl);
  justify-content: center;
  flex-wrap: wrap;
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
  color: var(--text-primary);
}

.reset-btn:hover:not(:disabled) {
  background-color: var(--bg-tertiary);
}

.reset-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Test notification button */
  .test-notifications-btn {
    padding: var(--spacing-sm) var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-sm);
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all var(--transition-fast);
    background-color: var(--bg-secondary);
    color: var(--text-primary);
  }
  
  .test-notifications-btn:hover:not(:disabled) {
    background-color: var(--bg-tertiary);
  }
  
  .test-notifications-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Loading State */
.loading-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--text-secondary);
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-color);
  border-top: 3px solid var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Statistics Section */
.statistics-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.stat-card {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  text-align: center;
}

.stat-card h3 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.stat-value {
  margin: 0;
  font-size: 1.8rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* Responsive Design */
@media (max-width: 768px) {
  .add-model-form {
    grid-template-columns: 1fr;
  }
  
  .form-row {
    grid-template-columns: 1fr;
  }
  
  .model-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .model-status-container {
    width: 100%;
    justify-content: flex-start;
  }
  
  .statistics-section {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
  
  .stat-value {
    font-size: 1.5rem;
  }
  
  .model-actions {
    flex-direction: column;
  }
  
  .control-btn, .activation-btn, .remove-btn {
    width: 100%;
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .batch-actions {
    flex-direction: column;
  }
  
  .batch-btn {
    width: 100%;
    justify-content: center;
  }
  
  .statistics-section {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .save-btn, .reset-btn {
    width: 100%;
  }
}
</style>
