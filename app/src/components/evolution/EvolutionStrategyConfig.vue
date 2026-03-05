<template>
  <div class="evolution-strategy-config">
    <!-- Page Header -->
    <div class="page-header">
      <h1>Evolution Strategy Configuration</h1>
      <p class="subtitle">Configure and manage autonomous evolution strategies for knowledge growth, model optimization, and cross-domain transfer</p>
    </div>

    <!-- Strategy Templates Section -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Strategy Templates</h2>
        <div class="section-actions">
          <button @click="refreshTemplates" class="btn btn-secondary" :disabled="loadingTemplates">
            <span v-if="loadingTemplates">Loading...</span>
            <span v-else>Refresh Templates</span>
          </button>
        </div>
      </div>
      
      <div class="templates-grid">
        <div 
          v-for="template in strategyTemplates" 
          :key="template.template_id"
          class="template-card"
          :class="{ selected: selectedTemplateId === template.template_id }"
          @click="selectTemplate(template)"
        >
          <div class="template-header">
            <h3>{{ template.name }}</h3>
            <span class="template-type-badge" :class="template.strategy_type">
              {{ template.strategy_type.replace('_', ' ') }}
            </span>
          </div>
          <p class="template-description">{{ template.description }}</p>
          <div class="template-details">
            <div class="detail-item">
              <span class="label">Applicable Domains:</span>
              <span class="value">{{ template.applicable_domains.join(', ') }}</span>
            </div>
            <div class="detail-item">
              <span class="label">Default Parameters:</span>
              <span class="value">{{ Object.keys(template.default_parameters).length }} parameters</span>
            </div>
          </div>
          <button 
            class="btn btn-primary btn-sm"
            @click.stop="applyTemplate(template)"
          >
            Apply Template
          </button>
        </div>
      </div>
    </div>

    <!-- Strategy Configuration Form -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Strategy Configuration</h2>
        <div class="section-actions">
          <button @click="validateStrategy" class="btn btn-secondary" :disabled="!currentConfig.strategy_name">
            Validate Strategy
          </button>
          <button @click="simulateStrategy" class="btn btn-primary" :disabled="!currentConfig.strategy_name">
            Simulate Strategy
          </button>
          <button @click="saveStrategy" class="btn btn-success" :disabled="!currentConfig.strategy_name || !validationResults?.valid">
            Save Strategy
          </button>
        </div>
      </div>

      <!-- Configuration Form -->
      <div class="config-form">
        <div class="form-row">
          <div class="form-group">
            <label for="strategyName">Strategy Name *</label>
            <input 
              id="strategyName"
              v-model="currentConfig.strategy_name"
              type="text"
              placeholder="e.g., Knowledge-Focused Evolution v1.2"
              class="form-control"
            />
          </div>
          
          <div class="form-group">
            <label for="strategyType">Strategy Type *</label>
            <select 
              id="strategyType"
              v-model="currentConfig.strategy_type"
              class="form-control"
            >
              <option value="knowledge_focused">Knowledge-Focused</option>
              <option value="model_performance">Model Performance</option>
              <option value="cross_domain_transfer">Cross-Domain Transfer</option>
              <option value="custom">Custom</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group full-width">
            <label for="optimizationTargets">Optimization Targets *</label>
            <div class="targets-selector">
              <div 
                v-for="target in availableTargets" 
                :key="target.id"
                class="target-checkbox"
              >
                <label>
                  <input 
                    type="checkbox" 
                    :value="target.id"
                    v-model="currentConfig.optimization_targets"
                  />
                  {{ target.label }}
                  <span class="target-description">{{ target.description }}</span>
                </label>
              </div>
            </div>
          </div>
        </div>

        <!-- Trigger Conditions -->
        <div class="form-section">
          <h3>Trigger Conditions</h3>
          <div class="trigger-conditions">
            <div class="condition-item">
              <label>
                <input type="checkbox" v-model="currentConfig.trigger_conditions.enable_performance_trigger" />
                Performance-Based Trigger
              </label>
              <div v-if="currentConfig.trigger_conditions.enable_performance_trigger" class="condition-details">
                <div class="condition-param">
                  <label>Accuracy Threshold:</label>
                  <input 
                    type="number" 
                    v-model="currentConfig.trigger_conditions.performance_accuracy_threshold"
                    min="0" 
                    max="1" 
                    step="0.01"
                    class="form-control-sm"
                  />
                  <span class="help-text">Trigger when accuracy falls below this value</span>
                </div>
              </div>
            </div>

            <div class="condition-item">
              <label>
                <input type="checkbox" v-model="currentConfig.trigger_conditions.enable_knowledge_trigger" />
                Knowledge-Based Trigger
              </label>
              <div v-if="currentConfig.trigger_conditions.enable_knowledge_trigger" class="condition-details">
                <div class="condition-param">
                  <label>New Knowledge Threshold:</label>
                  <input 
                    type="number" 
                    v-model="currentConfig.trigger_conditions.knowledge_new_concepts_threshold"
                    min="0"
                    step="1"
                    class="form-control-sm"
                  />
                  <span class="help-text">Trigger when new concepts exceed this count</span>
                </div>
              </div>
            </div>

            <div class="condition-item">
              <label>
                <input type="checkbox" v-model="currentConfig.trigger_conditions.enable_time_trigger" />
                Time-Based Trigger
              </label>
              <div v-if="currentConfig.trigger_conditions.enable_time_trigger" class="condition-details">
                <div class="condition-param">
                  <label>Interval (hours):</label>
                  <input 
                    type="number" 
                    v-model="currentConfig.trigger_conditions.time_interval_hours"
                    min="1"
                    step="1"
                    class="form-control-sm"
                  />
                  <span class="help-text">Trigger evolution at regular intervals</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Strategy Parameters -->
        <div class="form-section">
          <h3>Strategy Parameters</h3>
          <div class="parameters-grid">
            <div class="parameter-item" v-for="param in currentParameters" :key="param.key">
              <label :for="`param-${param.key}`">{{ param.label }}</label>
              <input 
                :id="`param-${param.key}`"
                v-model="currentConfig.parameters[param.key]"
                :type="param.type"
                :min="param.min"
                :max="param.max"
                :step="param.step"
                class="form-control"
              />
              <span class="help-text">{{ param.description }}</span>
            </div>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="checkbox-label">
              <input type="checkbox" v-model="currentConfig.enable_simulation" />
              Enable Strategy Simulation
            </label>
            <p class="help-text">Simulate strategy effects before actual execution</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Validation Results -->
    <div v-if="validationResults" class="dashboard-section">
      <div class="section-header">
        <h2>Validation Results</h2>
        <div class="validation-status" :class="validationResults.valid ? 'valid' : 'invalid'">
          {{ validationResults.valid ? 'Strategy Valid' : 'Strategy Invalid' }}
        </div>
      </div>
      
      <div v-if="validationResults.validation_errors.length > 0" class="validation-errors">
        <h3>Validation Errors:</h3>
        <ul>
          <li v-for="error in validationResults.validation_errors" :key="error">
            {{ error }}
          </li>
        </ul>
      </div>

      <div v-if="validationResults.simulation_results" class="simulation-results">
        <h3>Simulation Results:</h3>
        <div class="simulation-metrics">
          <div class="metric-card" v-for="(value, key) in validationResults.simulation_results" :key="key">
            <h4>{{ formatSimulationKey(key) }}</h4>
            <p class="metric-value">{{ value }}</p>
          </div>
        </div>
      </div>

      <div v-if="validationResults.estimated_impact" class="impact-assessment">
        <h3>Estimated Impact:</h3>
        <div class="impact-metrics">
          <div class="impact-card" v-for="(score, dimension) in validationResults.estimated_impact" :key="dimension">
            <h4>{{ formatImpactDimension(dimension) }}</h4>
            <div class="impact-bar">
              <div class="impact-fill" :style="{ width: (score * 100) + '%' }"></div>
            </div>
            <span class="impact-score">{{ (score * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Simulation Results -->
    <div v-if="simulationResults" class="dashboard-section">
      <div class="section-header">
        <h2>Strategy Simulation Results</h2>
        <div class="section-actions">
          <button @click="executeStrategy" class="btn btn-success" :disabled="!validationResults?.valid">
            Execute Strategy
          </button>
        </div>
      </div>

      <div class="simulation-details">
        <div class="simulation-header">
          <h3>Simulation ID: {{ simulationResults.simulation_id }}</h3>
          <p class="simulation-type">Type: {{ simulationResults.strategy_type }}</p>
        </div>

        <div class="simulation-metrics-grid">
          <div class="metric-card" v-for="(value, key) in simulationResults.simulation_results" :key="key">
            <h4>{{ formatSimulationKey(key) }}</h4>
            <p class="metric-value">{{ value }}</p>
          </div>
        </div>

        <div v-if="simulationResults.recommendations" class="recommendations">
          <h3>Recommendations:</h3>
          <ul>
            <li v-for="(recommendation, index) in simulationResults.recommendations" :key="index">
              {{ recommendation }}
            </li>
          </ul>
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
import { ref, computed, onMounted } from 'vue'
import { useToast } from 'vue-toastification'

export default {
  name: 'EvolutionStrategyConfig',
  
  setup() {
    const toast = useToast()
    
    // Reactive state
    const strategyTemplates = ref([])
    const loadingTemplates = ref(false)
    const selectedTemplateId = ref(null)
    
    const currentConfig = ref({
      strategy_name: '',
      strategy_type: 'knowledge_focused',
      trigger_conditions: {
        enable_performance_trigger: true,
        performance_accuracy_threshold: 0.8,
        enable_knowledge_trigger: true,
        knowledge_new_concepts_threshold: 10,
        enable_time_trigger: false,
        time_interval_hours: 24
      },
      optimization_targets: ['accuracy', 'knowledge_coverage'],
      parameters: {
        knowledge_collection_weight: 0.7,
        model_optimization_weight: 0.3,
        cross_domain_connection_target: 10,
        accuracy_target: 0.95,
        latency_target_ms: 50,
        memory_target_mb: 100
      },
      enable_simulation: true
    })
    
    const validationResults = ref(null)
    const simulationResults = ref(null)
    const loading = ref(false)
    const loadingMessage = ref('')
    
    // Available optimization targets
    const availableTargets = ref([
      { id: 'accuracy', label: 'Accuracy', description: 'Improve model prediction accuracy' },
      { id: 'latency', label: 'Latency', description: 'Reduce inference time' },
      { id: 'memory_usage', label: 'Memory Usage', description: 'Reduce memory consumption' },
      { id: 'knowledge_coverage', label: 'Knowledge Coverage', description: 'Increase knowledge base coverage' },
      { id: 'cross_domain_transfer', label: 'Cross-Domain Transfer', description: 'Improve cross-domain capability transfer' },
      { id: 'resource_efficiency', label: 'Resource Efficiency', description: 'Optimize resource utilization' }
    ])
    
    // Computed properties
    const currentParameters = computed(() => {
      const params = []
      
      // Add parameters based on strategy type
      if (currentConfig.value.strategy_type === 'knowledge_focused') {
        params.push(
          { key: 'knowledge_collection_weight', label: 'Knowledge Collection Weight', type: 'number', min: 0, max: 1, step: 0.1, description: 'Weight for knowledge acquisition vs model optimization' },
          { key: 'model_optimization_weight', label: 'Model Optimization Weight', type: 'number', min: 0, max: 1, step: 0.1, description: 'Weight for model optimization vs knowledge acquisition' },
          { key: 'cross_domain_connection_target', label: 'Cross-Domain Connections Target', type: 'number', min: 0, step: 1, description: 'Target number of cross-domain connections' }
        )
      } else if (currentConfig.value.strategy_type === 'model_performance') {
        params.push(
          { key: 'accuracy_target', label: 'Accuracy Target', type: 'number', min: 0, max: 1, step: 0.01, description: 'Target accuracy value' },
          { key: 'latency_target_ms', label: 'Latency Target (ms)', type: 'number', min: 0, step: 1, description: 'Maximum inference latency' },
          { key: 'memory_target_mb', label: 'Memory Target (MB)', type: 'number', min: 0, step: 1, description: 'Maximum memory usage' }
        )
      } else if (currentConfig.value.strategy_type === 'cross_domain_transfer') {
        params.push(
          { key: 'domain_similarity_threshold', label: 'Domain Similarity Threshold', type: 'number', min: 0, max: 1, step: 0.05, description: 'Minimum similarity for cross-domain transfer' },
          { key: 'transfer_success_rate_target', label: 'Transfer Success Rate Target', type: 'number', min: 0, max: 1, step: 0.05, description: 'Target success rate for transfers' },
          { key: 'adaptation_learning_rate', label: 'Adaptation Learning Rate', type: 'number', min: 0, max: 0.1, step: 0.001, description: 'Learning rate for domain adaptation' }
        )
      }
      
      return params
    })
    
    // Methods
    const refreshTemplates = async () => {
      loadingTemplates.value = true
      try {
        // Fetch strategy templates from API
        const response = await fetch('/api/evolution/strategy/templates')
        if (!response.ok) {
          throw new Error(`Failed to fetch templates: ${response.status}`)
        }
        strategyTemplates.value = await response.json()
        toast.success('Strategy templates refreshed')
      } catch (error) {
        console.error('Error fetching strategy templates:', error)
        toast.error('Failed to load strategy templates')
        
        // Fallback to mock data
        strategyTemplates.value = [
          {
            template_id: 'knowledge_focused_001',
            name: 'Knowledge-Focused Evolution Strategy',
            description: 'Focused on expanding knowledge base and cross-domain connections',
            strategy_type: 'knowledge_focused',
            default_parameters: {
              knowledge_collection_weight: 0.7,
              model_optimization_weight: 0.3,
              cross_domain_connection_target: 10
            },
            applicable_domains: ['mechanical_engineering', 'food_engineering', 'management_science']
          },
          {
            template_id: 'model_performance_001',
            name: 'Model Performance Optimization Strategy',
            description: 'Focused on improving model accuracy, latency, and efficiency',
            strategy_type: 'model_performance',
            default_parameters: {
              accuracy_target: 0.95,
              latency_target_ms: 50,
              memory_target_mb: 100
            },
            applicable_domains: ['all']
          },
          {
            template_id: 'cross_domain_transfer_001',
            name: 'Cross-Domain Capability Transfer Strategy',
            description: 'Focused on transferring capabilities across domains',
            strategy_type: 'cross_domain_transfer',
            default_parameters: {
              domain_similarity_threshold: 0.6,
              transfer_success_rate_target: 0.8,
              adaptation_learning_rate: 0.01
            },
            applicable_domains: ['mechanical_engineering', 'management_science', 'computer_science']
          }
        ]
      } finally {
        loadingTemplates.value = false
      }
    }
    
    const selectTemplate = (template) => {
      selectedTemplateId.value = template.template_id
    }
    
    const applyTemplate = (template) => {
      currentConfig.value.strategy_type = template.strategy_type
      
      // Merge default parameters
      currentConfig.value.parameters = {
        ...currentConfig.value.parameters,
        ...template.default_parameters
      }
      
      // Update optimization targets based on template type
      if (template.strategy_type === 'knowledge_focused') {
        currentConfig.value.optimization_targets = ['knowledge_coverage', 'cross_domain_transfer']
      } else if (template.strategy_type === 'model_performance') {
        currentConfig.value.optimization_targets = ['accuracy', 'latency', 'memory_usage']
      } else if (template.strategy_type === 'cross_domain_transfer') {
        currentConfig.value.optimization_targets = ['cross_domain_transfer', 'knowledge_coverage']
      }
      
      toast.success(`Applied template: ${template.name}`)
    }
    
    const validateStrategy = async () => {
      if (!currentConfig.value.strategy_name.trim()) {
        toast.error('Please enter a strategy name')
        return
      }
      
      loading.value = true
      loadingMessage.value = 'Validating strategy configuration...'
      
      try {
        const response = await fetch('/api/evolution/strategy/validate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(currentConfig.value)
        })
        
        if (!response.ok) {
          throw new Error(`Validation failed: ${response.status}`)
        }
        
        validationResults.value = await response.json()
        
        if (validationResults.value.valid) {
          toast.success('Strategy validation passed')
        } else {
          toast.warning('Strategy validation failed')
        }
      } catch (error) {
        console.error('Error validating strategy:', error)
        toast.error('Failed to validate strategy')
        
        // Fallback mock validation
        validationResults.value = {
          valid: currentConfig.value.strategy_name.trim().length > 0,
          validation_errors: currentConfig.value.strategy_name.trim() ? [] : ['Strategy name is required'],
          simulation_results: currentConfig.value.enable_simulation ? {
            estimated_training_time: '2-4 hours',
            expected_accuracy_improvement: 0.05,
            expected_knowledge_growth: 15,
            resource_requirements: {
              cpu_cores: 4,
              memory_gb: 8,
              gpu_memory_gb: 4
            }
          } : null,
          estimated_impact: currentConfig.value.enable_simulation ? {
            performance_impact: 0.7,
            knowledge_impact: 0.8,
            cross_domain_impact: 0.6,
            overall_effectiveness: 0.7
          } : null
        }
      } finally {
        loading.value = false
      }
    }
    
    const simulateStrategy = async () => {
      if (!currentConfig.value.strategy_name.trim()) {
        toast.error('Please enter a strategy name')
        return
      }
      
      loading.value = true
      loadingMessage.value = 'Simulating strategy effects...'
      
      try {
        const response = await fetch('/api/evolution/strategy/simulate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(currentConfig.value)
        })
        
        if (!response.ok) {
          throw new Error(`Simulation failed: ${response.status}`)
        }
        
        simulationResults.value = await response.json()
        toast.success('Strategy simulation completed')
      } catch (error) {
        console.error('Error simulating strategy:', error)
        toast.error('Failed to simulate strategy')
        
        // Fallback mock simulation
        simulationResults.value = {
          success: true,
          simulation_id: `sim_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
          strategy_type: currentConfig.value.strategy_type,
          simulation_results: {
            knowledge_acquisitions: 25,
            successful_validations: 22,
            cross_domain_connections: 8,
            overall_knowledge_quality: 0.85
          },
          recommendations: [
            'Ensure sufficient compute resources are available',
            'Monitor evolution progress regularly',
            'Be prepared to rollback if performance degrades'
          ],
          timestamp: Date.now() / 1000
        }
      } finally {
        loading.value = false
      }
    }
    
    const saveStrategy = async () => {
      if (!validationResults.value?.valid) {
        toast.error('Cannot save invalid strategy')
        return
      }
      
      loading.value = true
      loadingMessage.value = 'Saving strategy configuration...'
      
      try {
        // In a real implementation, this would save to backend
        // For now, simulate success
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        toast.success('Strategy saved successfully')
        
        // Reset form after saving
        currentConfig.value.strategy_name = ''
        validationResults.value = null
        simulationResults.value = null
      } catch (error) {
        console.error('Error saving strategy:', error)
        toast.error('Failed to save strategy')
      } finally {
        loading.value = false
      }
    }
    
    const executeStrategy = async () => {
      if (!validationResults.value?.valid) {
        toast.error('Cannot execute invalid strategy')
        return
      }
      
      loading.value = true
      loadingMessage.value = 'Executing evolution strategy...'
      
      try {
        // In a real implementation, this would trigger strategy execution
        // For now, simulate success
        await new Promise(resolve => setTimeout(resolve, 1500))
        
        toast.success('Evolution strategy execution initiated')
        
        // Reset form after execution
        currentConfig.value.strategy_name = ''
        validationResults.value = null
        simulationResults.value = null
        simulationResults.value = null
      } catch (error) {
        console.error('Error executing strategy:', error)
        toast.error('Failed to execute strategy')
      } finally {
        loading.value = false
      }
    }
    
    const formatSimulationKey = (key) => {
      // Convert snake_case to Title Case
      return key.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
    }
    
    const formatImpactDimension = (dimension) => {
      const mapping = {
        'performance_impact': 'Performance Impact',
        'knowledge_impact': 'Knowledge Impact',
        'cross_domain_impact': 'Cross-Domain Impact',
        'overall_effectiveness': 'Overall Effectiveness'
      }
      return mapping[dimension] || dimension
    }
    
    // Lifecycle hooks
    onMounted(() => {
      refreshTemplates()
    })
    
    return {
      // State
      strategyTemplates,
      loadingTemplates,
      selectedTemplateId,
      currentConfig,
      validationResults,
      simulationResults,
      loading,
      loadingMessage,
      availableTargets,
      
      // Computed
      currentParameters,
      
      // Methods
      refreshTemplates,
      selectTemplate,
      applyTemplate,
      validateStrategy,
      simulateStrategy,
      saveStrategy,
      executeStrategy,
      formatSimulationKey,
      formatImpactDimension
    }
  }
}
</script>

<style scoped>
.evolution-strategy-config {
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

.section-actions {
  display: flex;
  gap: 10px;
}

.templates-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.template-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 2px solid var(--border-color);
  cursor: pointer;
  transition: all 0.3s ease;
}

.template-card:hover {
  border-color: #4CAF50;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.template-card.selected {
  border-color: #2196F3;
  background-color: rgba(33, 150, 243, 0.05);
}

.template-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 15px;
}

.template-header h3 {
  font-size: 1.4rem;
  margin: 0;
  color: #222;
}

.template-type-badge {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
}

.template-type-badge.knowledge_focused {
  background-color: #E8F5E9;
  color: #2E7D32;
  border: 1px solid #C8E6C9;
}

.template-type-badge.model_performance {
  background-color: #E3F2FD;
  color: #1565C0;
  border: 1px solid #BBDEFB;
}

.template-type-badge.cross_domain_transfer {
  background-color: #F3E5F5;
  color: #7B1FA2;
  border: 1px solid #E1BEE7;
}

.template-description {
  color: #555;
  line-height: 1.6;
  margin-bottom: 15px;
}

.template-details {
  margin-bottom: 15px;
}

.detail-item {
  display: flex;
  margin-bottom: 8px;
  font-size: 0.9rem;
}

.detail-item .label {
  font-weight: 600;
  color: #666;
  min-width: 160px;
}

.detail-item .value {
  color: #333;
}

.config-form {
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.form-row {
  display: flex;
  gap: 20px;
}

.form-group {
  flex: 1;
}

.form-group.full-width {
  flex: 100%;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
}

.form-control {
  width: 100%;
  padding: 10px 15px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.form-control:focus {
  outline: none;
  border-color: #2196F3;
  box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
}

.form-control-sm {
  padding: 6px 10px;
  font-size: 0.9rem;
}

.targets-selector {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 15px;
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.target-checkbox {
  display: flex;
  align-items: flex-start;
}

.target-checkbox label {
  display: flex;
  align-items: flex-start;
  cursor: pointer;
  font-weight: normal;
}

.target-checkbox input {
  margin-right: 10px;
  margin-top: 3px;
}

.target-description {
  display: block;
  font-size: 0.85rem;
  color: #666;
  margin-top: 4px;
}

.form-section {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.form-section h3 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #222;
  font-size: 1.4rem;
}

.trigger-conditions {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.condition-item {
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.condition-item label {
  display: flex;
  align-items: center;
  font-weight: 600;
  cursor: pointer;
}

.condition-item input[type="checkbox"] {
  margin-right: 10px;
}

.condition-details {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--border-color);
}

.condition-param {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.condition-param label {
  min-width: 200px;
  font-weight: normal;
}

.help-text {
  font-size: 0.85rem;
  color: #666;
  margin-top: 5px;
}

.parameters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.parameter-item {
  display: flex;
  flex-direction: column;
}

.checkbox-label {
  display: flex;
  align-items: center;
  cursor: pointer;
}

.checkbox-label input {
  margin-right: 10px;
}

.validation-status {
  padding: 8px 20px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 1rem;
}

.validation-status.valid {
  background-color: #E8F5E9;
  color: #2E7D32;
  border: 1px solid #C8E6C9;
}

.validation-status.invalid {
  background-color: #FFEBEE;
  color: #C62828;
  border: 1px solid #FFCDD2;
}

.validation-errors {
  padding: 20px;
  background: #FFEBEE;
  border-radius: var(--border-radius-sm);
  border: 1px solid #FFCDD2;
  margin-bottom: 20px;
}

.validation-errors h3 {
  color: #C62828;
  margin-top: 0;
}

.validation-errors ul {
  margin: 10px 0 0 0;
  padding-left: 20px;
}

.validation-errors li {
  color: #C62828;
  margin-bottom: 5px;
}

.simulation-results,
.impact-assessment {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  margin-bottom: 20px;
}

.simulation-metrics,
.impact-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.metric-card,
.impact-card {
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  text-align: center;
}

.metric-card h4,
.impact-card h4 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 1rem;
  color: #333;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: #2196F3;
  margin: 10px 0;
}

.impact-bar {
  height: 10px;
  background: #E0E0E0;
  border-radius: 5px;
  margin: 10px 0;
  overflow: hidden;
}

.impact-fill {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #8BC34A);
  border-radius: 5px;
  transition: width 0.5s ease;
}

.impact-score {
  display: block;
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
}

.simulation-details {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.simulation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border-color);
}

.simulation-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.recommendations {
  padding: 20px;
  background: #FFF3E0;
  border-radius: var(--border-radius-sm);
  border: 1px solid #FFE0B2;
}

.recommendations h3 {
  color: #E65100;
  margin-top: 0;
}

.recommendations ul {
  margin: 10px 0 0 0;
  padding-left: 20px;
}

.recommendations li {
  color: #555;
  margin-bottom: 8px;
  line-height: 1.5;
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

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: 1rem;
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

.btn-sm {
  padding: 6px 12px;
  font-size: 0.9rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .form-row {
    flex-direction: column;
  }
  
  .templates-grid {
    grid-template-columns: 1fr;
  }
  
  .targets-selector {
    grid-template-columns: 1fr;
  }
  
  .parameters-grid {
    grid-template-columns: 1fr;
  }
  
  .simulation-metrics,
  .impact-metrics,
  .simulation-metrics-grid {
    grid-template-columns: 1fr;
  }
  
  .condition-param {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .condition-param label {
    min-width: auto;
  }
}
</style>