<template>
  <div class="autonomous-evolution-view">
    <!-- Status Messages -->
    <div class="status-messages">
      <div v-if="errorState?.hasError" class="message error">
        <span class="icon">⚠️</span>
        {{ errorState?.message }}
      </div>
      <div v-if="successState?.hasSuccess" class="message success">
        <span class="icon">✅</span>
        {{ successState?.message }}
      </div>
      <div v-if="warningState?.hasWarning" class="message warning">
        <span class="icon">⚠️</span>
        {{ warningState?.message }}
      </div>
      <div v-if="infoState?.hasInfo" class="message info">
        <span class="icon">ℹ️</span>
        {{ infoState?.message }}
      </div>
    </div>

    <!-- Page Header -->
    <div class="page-header">
      <h1>Autonomous Evolution Management</h1>
      <p class="subtitle">Manage and control autonomous evolution capabilities for neural network architectures</p>
    </div>

    <!-- Main Control Dashboard -->
    <div class="evolution-dashboard">
      <!-- Evolution Engine Status -->
      <div class="dashboard-section">
        <div class="section-header">
          <h2>Evolution Engine Status</h2>
          <div class="section-actions">
            <button @click="refreshEvolutionStatus" class="btn btn-secondary">
              <span class="btn-icon">🔄</span>
              Refresh
            </button>
          </div>
        </div>
        
        <div class="status-cards">
          <div class="status-card" :class="{ active: evolutionStatus?.is_active }">
            <div class="status-card-header">
              <span class="status-icon">
                <span v-if="evolutionStatus?.is_active" class="active-indicator">⚡</span>
                <span v-else class="inactive-indicator">⏸️</span>
              </span>
              <h3>Evolution Engine</h3>
            </div>
            <div class="status-card-content">
              <div class="status-item">
                <span class="label">Status:</span>
                <span :class="['value', evolutionStatus?.is_active ? 'active' : 'inactive']">
                  {{ evolutionStatus?.is_active ? 'Active' : 'Inactive' }}
                </span>
              </div>
              <div class="status-item">
                <span class="label">Current Generation:</span>
                <span class="value">{{ evolutionStatus?.current_generation || 0 }}</span>
              </div>
              <div class="status-item">
                <span class="label">Population Size:</span>
                <span class="value">{{ evolutionStatus?.population_size || 0 }}</span>
              </div>
              <div class="status-item">
                <span class="label">Best Accuracy:</span>
                <span class="value">{{ evolutionStatus?.best_accuracy ? (evolutionStatus.best_accuracy * 100).toFixed(2) + '%' : 'N/A' }}</span>
              </div>
            </div>
          </div>

          <div class="status-card">
            <div class="status-card-header">
              <span class="status-icon">🧬</span>
              <h3>Algorithms</h3>
            </div>
            <div class="status-card-content">
              <div class="status-item">
                <span class="label">Active Algorithms:</span>
                <span class="value">{{ evolutionStatus?.active_algorithms?.length || 0 }}</span>
              </div>
              <div v-if="evolutionStatus?.active_algorithms" class="algorithm-list">
                <span v-for="algo in evolutionStatus.active_algorithms" :key="algo" class="algorithm-tag">
                  {{ algo }}
                </span>
              </div>
            </div>
          </div>

          <div class="status-card">
            <div class="status-card-header">
              <span class="status-icon">🖥️</span>
              <h3>Hardware</h3>
            </div>
            <div class="status-card-content">
              <div class="status-item">
                <span class="label">Hardware Type:</span>
                <span class="value">{{ evolutionStatus?.hardware_info?.type || 'Unknown' }}</span>
              </div>
              <div class="status-item">
                <span class="label">Memory Usage:</span>
                <span class="value">{{ evolutionStatus?.hardware_info?.memory_usage || 'N/A' }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Evolution Control -->
      <div class="dashboard-section">
        <div class="section-header">
          <h2>Evolution Control</h2>
        </div>
        
        <div class="control-panel">
          <div class="control-group">
            <h3>Evolution Mode</h3>
            <div class="mode-buttons">
              <button 
                @click="setEvolutionMode('basic')" 
                :class="{ active: evolutionConfig?.mode === 'basic' }"
                class="mode-btn"
              >
                Basic
              </button>
              <button 
                @click="setEvolutionMode('enhanced')" 
                :class="{ active: evolutionConfig?.mode === 'enhanced' }"
                class="mode-btn"
              >
                Enhanced
              </button>
              <button 
                @click="setEvolutionMode('federated')" 
                :class="{ active: evolutionConfig?.mode === 'federated' }"
                class="mode-btn"
              >
                Federated
              </button>
              <button 
                @click="setEvolutionMode('hardware_aware')" 
                :class="{ active: evolutionConfig?.mode === 'hardware_aware' }"
                class="mode-btn"
              >
                Hardware Aware
              </button>
            </div>
          </div>

          <div class="control-group">
            <h3>Evolution Actions</h3>
            <div class="action-buttons">
              <button 
                @click="startEvolution" 
                :disabled="evolutionStatus?.is_active"
                class="btn btn-primary"
              >
                <span class="btn-icon">▶️</span>
                Start Evolution
              </button>
              <button 
                @click="stopEvolution" 
                :disabled="!evolutionStatus?.is_active"
                class="btn btn-warning"
              >
                <span class="btn-icon">⏸️</span>
                Stop Evolution
              </button>
              <button 
                @click="resetEvolution" 
                class="btn btn-secondary"
              >
                <span class="btn-icon">🔄</span>
                Reset Evolution
              </button>
              <button 
                @click="exportEvolutionData" 
                class="btn btn-secondary"
              >
                <span class="btn-icon">💾</span>
                Export Data
              </button>
            </div>
          </div>

          <div class="control-group">
            <h3>Configuration</h3>
            <div class="config-params">
              <div class="param-item">
                <label for="population-size">Population Size:</label>
                <input 
                  type="number" 
                  id="population-size" 
                  v-model.number="configParams.population_size"
                  min="10"
                  max="1000"
                />
              </div>
              <div class="param-item">
                <label for="mutation-rate">Mutation Rate:</label>
                <input 
                  type="range" 
                  id="mutation-rate" 
                  v-model.number="configParams.mutation_rate"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                />
                <span class="param-value">{{ (configParams.mutation_rate * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <label for="crossover-rate">Crossover Rate:</label>
                <input 
                  type="range" 
                  id="crossover-rate" 
                  v-model.number="configParams.crossover_rate"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                />
                <span class="param-value">{{ (configParams.crossover_rate * 100).toFixed(1) }}%</span>
              </div>
              <div class="param-item">
                <label for="generations">Max Generations:</label>
                <input 
                  type="number" 
                  id="generations" 
                  v-model.number="configParams.max_generations"
                  min="1"
                  max="1000"
                />
              </div>
              <button @click="updateConfigParams" class="btn btn-secondary">
                Update Configuration
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Evolution History & Performance -->
      <div class="dashboard-section">
        <div class="section-header">
          <h2>Evolution History & Performance</h2>
          <div class="section-actions">
            <button @click="refreshEvolutionHistory" class="btn btn-secondary">
              <span class="btn-icon">🔄</span>
              Refresh History
            </button>
          </div>
        </div>
        
        <div class="history-panel">
          <div class="performance-chart-container">
            <h3>Performance Trend</h3>
            <div v-if="evolutionHistory.length > 0" class="performance-chart">
              <!-- Simple ASCII-style chart for performance visualization -->
              <div class="chart-lines">
                <div v-for="(line, index) in performanceChartLines" :key="index" class="chart-line">
                  <div class="line-label">{{ line.label }}</div>
                  <div class="line-bar">
                    <div class="line-fill" :style="{ width: line.percentage + '%' }"></div>
                  </div>
                  <div class="line-value">{{ line.value }}</div>
                </div>
              </div>
            </div>
            <div v-else class="no-data">
              <p>No evolution history available. Start evolution to see performance data.</p>
            </div>
          </div>

          <div class="history-list-container">
            <h3>Recent Generations</h3>
            <div v-if="evolutionHistory.length > 0" class="history-list">
              <div v-for="item in evolutionHistory.slice(0, 10)" :key="item.generation" class="history-item">
                <div class="history-item-header">
                  <span class="generation">Generation {{ item.generation }}</span>
                  <span class="timestamp">{{ formatTime(item.timestamp) }}</span>
                </div>
                <div class="history-item-content">
                  <div class="metric">
                    <span class="metric-label">Accuracy:</span>
                    <span class="metric-value">{{ (item.accuracy * 100).toFixed(2) }}%</span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Architecture:</span>
                    <span class="metric-value">{{ item.architecture_id }}</span>
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="no-data">
              <p>No evolution history available.</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Advanced Features -->
      <div class="dashboard-section">
        <div class="section-header">
          <h2>Advanced Evolution Features</h2>
        </div>
        
        <div class="advanced-features">
          <div class="feature-card">
            <div class="feature-header">
              <span class="feature-icon">🔍</span>
              <h3>Neural Architecture Search (NAS)</h3>
            </div>
            <div class="feature-content">
              <p>Differentiable Architecture Search (DARTS) for automatic neural network design.</p>
              <div class="feature-controls">
                <button 
                  @click="startNASearch" 
                  :disabled="nasStatus?.is_active"
                  class="btn btn-primary"
                >
                  {{ nasStatus?.is_active ? 'NAS Running...' : 'Start NAS' }}
                </button>
                <button 
                  @click="stopNASearch" 
                  :disabled="!nasStatus?.is_active"
                  class="btn btn-warning"
                >
                  Stop NAS
                </button>
              </div>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-header">
              <span class="feature-icon">🤖</span>
              <h3>Reinforcement Learning Optimization</h3>
            </div>
            <div class="feature-content">
              <p>PPO algorithm for optimizing evolution strategy parameters.</p>
              <div class="feature-controls">
                <button 
                  @click="toggleRLOptimizer" 
                  :class="{ active: rlStatus?.is_active }"
                  class="btn btn-secondary"
                >
                  {{ rlStatus?.is_active ? 'RL Active' : 'Enable RL' }}
                </button>
              </div>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-header">
              <span class="feature-icon">🌐</span>
              <h3>Federated Evolution</h3>
            </div>
            <div class="feature-content">
              <p>Distributed, privacy-preserving collaborative evolution.</p>
              <div class="feature-controls">
                <button 
                  @click="startFederatedEvolution" 
                  :disabled="federatedStatus?.is_active"
                  class="btn btn-primary"
                >
                  {{ federatedStatus?.is_active ? 'Federated Active' : 'Start Federated' }}
                </button>
                <button 
                  @click="showFederatedClients" 
                  class="btn btn-secondary"
                >
                  View Clients
                </button>
              </div>
            </div>
          </div>

          <div class="feature-card">
            <div class="feature-header">
              <span class="feature-icon">⚡</span>
              <h3>Online Evolution</h3>
            </div>
            <div class="feature-content">
              <p>Real-time architecture adaptation without service interruption.</p>
              <div class="feature-controls">
                <button 
                  @click="toggleOnlineEvolution" 
                  :class="{ active: onlineEvolutionStatus?.is_active }"
                  class="btn btn-secondary"
                >
                  {{ onlineEvolutionStatus?.is_active ? 'Online Active' : 'Enable Online' }}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { handleError } from '@/utils/errorHandler'
import api from '@/utils/api'

export default {
  name: 'AutonomousEvolutionView',
  setup() {
    // State management
    const evolutionStatus = ref(null)
    const evolutionConfig = ref(null)
    const evolutionHistory = ref([])
    const configParams = ref({
      population_size: 100,
      mutation_rate: 0.1,
      crossover_rate: 0.7,
      max_generations: 100
    })
    
    const nasStatus = ref({ is_active: false })
    const rlStatus = ref({ is_active: false })
    const federatedStatus = ref({ is_active: false })
    const onlineEvolutionStatus = ref({ is_active: false })
    
    // UI state
    const errorState = ref(null)
    const successState = ref(null)
    const warningState = ref(null)
    const infoState = ref(null)
    
    // Computed properties
    const performanceChartLines = computed(() => {
      if (evolutionHistory.value.length === 0) return []
      
      // Simplified chart - just show accuracy trend
      const last10 = evolutionHistory.value.slice(-10)
      return last10.map((item, index) => ({
        label: `G${item.generation}`,
        value: `${(item.accuracy * 100).toFixed(1)}%`,
        percentage: item.accuracy * 100
      }))
    })
    
    // Methods
    const showMessage = (type, message) => {
      // Clear previous message
      errorState.value = null
      successState.value = null
      warningState.value = null
      infoState.value = null
      
      switch (type) {
        case 'error':
          errorState.value = { hasError: true, message }
          break
        case 'success':
          successState.value = { hasSuccess: true, message }
          break
        case 'warning':
          warningState.value = { hasWarning: true, message }
          break
        case 'info':
          infoState.value = { hasInfo: true, message }
          break
      }
      
      // Auto-clear messages after 5 seconds
      setTimeout(() => {
        errorState.value = null
        successState.value = null
        warningState.value = null
        infoState.value = null
      }, 5000)
    }
    
    const formatTime = (timestamp) => {
      if (!timestamp) return 'N/A'
      const date = new Date(timestamp * 1000)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const refreshEvolutionStatus = async () => {
      try {
        const response = await api.evolution.getStatus()
        evolutionStatus.value = response.data
        showMessage('success', 'Evolution status refreshed')
      } catch (error) {
        console.error('Error refreshing evolution status:', error)
        // 使用模拟状态作为后备
        const mockStatus = {
          is_active: false,
          current_generation: 0,
          population_size: 100,
          best_accuracy: 0.0,
          best_architecture: null,
          active_algorithms: ["genetic_algorithm", "simulated_annealing"],
          hardware_info: {
            type: "cpu",
            memory_gb: 16.0,
            compute_units: 8,
            has_gpu: false,
            simulated: true
          },
          evolution_mode: "basic",
          start_time: null,
          elapsed_time: 0
        }
        
        evolutionStatus.value = mockStatus
        showMessage('info', 'Using simulated evolution status data')
      }
    }
    
    const getEvolutionConfig = async () => {
      try {
        const response = await api.evolution.getConfig()
        evolutionConfig.value = response.data
        
        // Update config params with current values
        if (response.data.config) {
          configParams.value = {
            ...configParams.value,
            ...response.data.config
          }
        }
      } catch (error) {
        console.error('Error getting evolution config:', error)
        // 使用模拟配置作为后备
        const mockConfig = {
          config: {
            population_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            max_generations: 100,
            evolution_module_type: "enhanced",
            enable_hardware_aware: true,
            enable_nas: false,
            enable_rl_optimization: false,
            enable_online_evolution: false,
            performance_targets: {
              accuracy: 0.9,
              efficiency: 0.8,
              robustness: 0.7
            },
            simulated: true
          }
        }
        
        evolutionConfig.value = mockConfig
        configParams.value = {
          ...configParams.value,
          ...mockConfig.config
        }
      }
    }
    
    const refreshEvolutionHistory = async () => {
      try {
        const response = await api.evolution.getHistory()
        evolutionHistory.value = response.data.history || []
        showMessage('success', 'Evolution history refreshed')
      } catch (error) {
        console.error('Error refreshing evolution history:', error)
        // 使用模拟数据作为后备
        const mockHistory = []
        const baseTime = Date.now() / 1000 - 24 * 3600  // 24小时前
        
        for (let i = 0; i < 20; i++) {
          const generation = i + 1
          const timestamp = baseTime + i * 1800  // 每30分钟一个点
          const accuracy = 0.6 + (i * 0.01) + (0.05 * (Math.random() - 0.5))  // 逐渐提高，有波动
          
          mockHistory.push({
            generation: generation,
            timestamp: timestamp,
            accuracy: Math.min(0.95, accuracy),
            architecture_id: `arch_${Math.floor(i / 5) + 1}`,
            architecture_summary: {
              type: 'classification',
              layers: (i % 6) + 3,
              parameters: 50000 + (i * 1000)
            },
            performance_metrics: {
              accuracy: Math.min(0.95, accuracy),
              efficiency: 0.7 + (i * 0.005),
              robustness: 0.6 + (i * 0.008)
            }
          })
        }
        
        evolutionHistory.value = mockHistory
        showMessage('info', 'Using simulated evolution history data')
      }
    }
    
    const startEvolution = async () => {
      try {
        const response = await api.evolution.start({
          config: configParams.value
        })
        evolutionStatus.value = response.data
        showMessage('success', 'Evolution started successfully')
      } catch (error) {
        console.error('Error starting evolution:', error)
        showMessage('error', 'Failed to start evolution')
      }
    }
    
    const stopEvolution = async () => {
      try {
        const response = await api.evolution.stop()
        evolutionStatus.value = response.data
        showMessage('success', 'Evolution stopped successfully')
      } catch (error) {
        console.error('Error stopping evolution:', error)
        showMessage('error', 'Failed to stop evolution')
      }
    }
    
    const resetEvolution = async () => {
      try {
        const response = await api.evolution.reset()
        evolutionStatus.value = response.data
        evolutionHistory.value = []
        showMessage('success', 'Evolution reset successfully')
      } catch (error) {
        console.error('Error resetting evolution:', error)
        showMessage('error', 'Failed to reset evolution')
      }
    }
    
    const setEvolutionMode = async (mode) => {
      try {
        const response = await api.evolution.setMode({ mode })
        evolutionConfig.value = response.data
        showMessage('success', `Evolution mode set to ${mode}`)
      } catch (error) {
        console.error('Error setting evolution mode:', error)
        showMessage('error', 'Failed to set evolution mode')
      }
    }
    
    const updateConfigParams = async () => {
      try {
        const response = await api.evolution.updateConfig(configParams.value)
        evolutionConfig.value = response.data
        showMessage('success', 'Configuration updated successfully')
      } catch (error) {
        console.error('Error updating configuration:', error)
        showMessage('error', 'Failed to update configuration')
      }
    }
    
    const exportEvolutionData = async () => {
      try {
        const response = await api.evolution.exportData()
        // In a real implementation, this would trigger a file download
        showMessage('success', 'Evolution data export initiated')
      } catch (error) {
        console.error('Error exporting evolution data:', error)
        showMessage('error', 'Failed to export evolution data')
      }
    }
    
    const startNASearch = async () => {
      try {
        const response = await api.evolution.nas.start()
        nasStatus.value = response.data
        showMessage('success', 'Neural Architecture Search started')
      } catch (error) {
        console.error('Error starting NAS:', error)
        showMessage('error', 'Failed to start Neural Architecture Search')
      }
    }
    
    const stopNASearch = async () => {
      try {
        const response = await api.evolution.nas.stop()
        nasStatus.value = response.data
        showMessage('success', 'Neural Architecture Search stopped')
      } catch (error) {
        console.error('Error stopping NAS:', error)
        showMessage('error', 'Failed to stop Neural Architecture Search')
      }
    }
    
    const toggleRLOptimizer = async () => {
      try {
        if (rlStatus.value.is_active) {
          const response = await api.evolution.rl.stop()
          rlStatus.value = response.data
          showMessage('success', 'RL optimizer disabled')
        } else {
          const response = await api.evolution.rl.start()
          rlStatus.value = response.data
          showMessage('success', 'RL optimizer enabled')
        }
      } catch (error) {
        console.error('Error toggling RL optimizer:', error)
        showMessage('error', 'Failed to toggle RL optimizer')
      }
    }
    
    const startFederatedEvolution = async () => {
      try {
        const response = await api.evolution.federated.start()
        federatedStatus.value = response.data
        showMessage('success', 'Federated evolution started')
      } catch (error) {
        console.error('Error starting federated evolution:', error)
        showMessage('error', 'Failed to start federated evolution')
      }
    }
    
    const showFederatedClients = () => {
      showMessage('info', 'Federated clients view would open in a real implementation')
    }
    
    const toggleOnlineEvolution = async () => {
      try {
        if (onlineEvolutionStatus.value.is_active) {
          const response = await api.evolution.online.stop()
          onlineEvolutionStatus.value = response.data
          showMessage('success', 'Online evolution disabled')
        } else {
          const response = await api.evolution.online.start()
          onlineEvolutionStatus.value = response.data
          showMessage('success', 'Online evolution enabled')
        }
      } catch (error) {
        console.error('Error toggling online evolution:', error)
        showMessage('error', 'Failed to toggle online evolution')
      }
    }
    
    // Lifecycle
    onMounted(() => {
      refreshEvolutionStatus()
      getEvolutionConfig()
      refreshEvolutionHistory()
    })
    
    return {
      // State
      evolutionStatus,
      evolutionConfig,
      evolutionHistory,
      configParams,
      nasStatus,
      rlStatus,
      federatedStatus,
      onlineEvolutionStatus,
      errorState,
      successState,
      warningState,
      infoState,
      
      // Computed
      performanceChartLines,
      
      // Methods
      formatTime,
      refreshEvolutionStatus,
      refreshEvolutionHistory,
      startEvolution,
      stopEvolution,
      resetEvolution,
      setEvolutionMode,
      updateConfigParams,
      exportEvolutionData,
      startNASearch,
      stopNASearch,
      toggleRLOptimizer,
      startFederatedEvolution,
      showFederatedClients,
      toggleOnlineEvolution,
      showMessage
    }
  }
}
</script>

<style scoped>
.autonomous-evolution-view {
  padding: 20px;
  max-width: 1400px;
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

.status-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.status-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
  transition: var(--transition);
}

.status-card.active {
  border-color: #4CAF50;
  box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
}

.status-card-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.status-icon {
  font-size: 1.5rem;
  margin-right: 10px;
}

.status-card-content {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status-item .label {
  color: #555;
  font-weight: 500;
}

.status-item .value {
  font-weight: 600;
  color: #222;
}

.status-item .value.active {
  color: #4CAF50;
}

.status-item .value.inactive {
  color: #f44336;
}

.algorithm-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.algorithm-tag {
  padding: 4px 8px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  font-size: 0.9rem;
  color: #555;
}

.control-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
}

.control-group h3 {
  margin-bottom: 15px;
  color: #222;
  font-size: 1.3rem;
}

.mode-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.mode-btn {
  padding: 10px 20px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  transition: var(--transition);
}

.mode-btn:hover {
  background: var(--bg-secondary);
}

.mode-btn.active {
  background: #2196F3;
  color: white;
  border-color: #2196F3;
}

.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.btn {
  display: inline-flex;
  align-items: center;
  padding: 10px 20px;
  border: none;
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
}

.btn:hover {
  opacity: 0.9;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #2196F3;
  color: white;
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: #222;
  border: 1px solid var(--border-color);
}

.btn-warning {
  background: #ff9800;
  color: white;
}

.btn-icon {
  margin-right: 8px;
}

.config-params {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.param-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.param-item label {
  min-width: 120px;
  color: #555;
}

.param-item input[type="range"] {
  flex: 1;
}

.param-value {
  min-width: 40px;
  text-align: right;
  color: #222;
  font-weight: 500;
}

.history-panel {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 30px;
}

.performance-chart-container,
.history-list-container {
  background: var(--bg-primary);
  padding: 20px;
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.performance-chart-container h3,
.history-list-container h3 {
  margin-bottom: 15px;
  color: #222;
}

.chart-lines {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.chart-line {
  display: flex;
  align-items: center;
  gap: 10px;
}

.line-label {
  min-width: 40px;
  font-size: 0.9rem;
  color: #555;
}

.line-bar {
  flex: 1;
  height: 20px;
  background: var(--bg-tertiary);
  border-radius: 10px;
  overflow: hidden;
}

.line-fill {
  height: 100%;
  background: #2196F3;
  border-radius: 10px;
  transition: width 0.3s ease;
}

.line-value {
  min-width: 50px;
  text-align: right;
  font-weight: 500;
  color: #222;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 400px;
  overflow-y: auto;
}

.history-item {
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.history-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.generation {
  font-weight: 600;
  color: #222;
}

.timestamp {
  font-size: 0.9rem;
  color: #888;
}

.history-item-content {
  display: flex;
  gap: 20px;
}

.metric {
  display: flex;
  align-items: center;
  gap: 5px;
}

.metric-label {
  color: #555;
  font-size: 0.9rem;
}

.metric-value {
  font-weight: 500;
  color: #222;
}

.advanced-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.feature-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
}

.feature-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.feature-icon {
  font-size: 1.5rem;
  margin-right: 10px;
}

.feature-header h3 {
  font-size: 1.2rem;
  color: #222;
}

.feature-content {
  color: #555;
  line-height: 1.5;
}

.feature-content p {
  margin-bottom: 15px;
}

.feature-controls {
  display: flex;
  gap: 10px;
}

.no-data {
  text-align: center;
  padding: 40px 20px;
  color: #888;
}

.status-messages {
  margin-bottom: 20px;
}

.message {
  padding: 15px;
  border-radius: var(--border-radius-sm);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.message.error {
  background: #ffebee;
  border: 1px solid #ffcdd2;
  color: #d32f2f;
}

.message.success {
  background: #e8f5e8;
  border: 1px solid #c8e6c9;
  color: #388e3c;
}

.message.warning {
  background: #fff8e1;
  border: 1px solid #ffecb3;
  color: #f57c00;
}

.message.info {
  background: #e3f2fd;
  border: 1px solid #bbdefb;
  color: #1976d2;
}

.message .icon {
  font-size: 1.2rem;
}

/* Responsive adjustments */
@media (max-width: 1200px) {
  .history-panel {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .status-cards,
  .control-panel,
  .advanced-features {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .param-item {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .param-item label {
    margin-bottom: 5px;
  }
}
</style>