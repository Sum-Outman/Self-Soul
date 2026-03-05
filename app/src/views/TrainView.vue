<template>
  <div class="train-view">
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
    
    <!-- Training Control Panel -->
    <div class="control-panel">
      <div class="mode-selection">
        <h2>Training Mode</h2>
        <div class="mode-options">
          <button 
            @click="trainingMode = 'individual'"
            :class="{ active: trainingMode === 'individual' }"
          >
            Individual
          </button>
          <button 
            @click="trainingMode = 'joint'"
            :class="{ active: trainingMode === 'joint' }"
          >
            Joint
          </button>
        </div>
      </div>
      
      <div class="model-selection">
        <h2>Select Models</h2>
        
        <!-- Recommended Combinations -->
            <div class="recommended-combinations" v-if="trainingMode === 'joint'">
              <h3>Recommended Combinations</h3>
              <div class="combination-buttons">
                <button 
                  v-for="(combination, name) in recommendedCombinations" 
                  :key="name"
                  @click="selectRecommendedCombination(combination)"
                  class="combination-btn"
                >
                  {{ name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' ') }}
                </button>
                <!-- Select All Models Button -->
                <button 
                  @click="selectAllModels"
                  class="combination-btn"
                >
                  Select All Models
                </button>
              </div>
            </div>
        
        <div class="model-grid">
              <div 
                v-for="model in (filteredModels || [])" 
                :key="model.id"
                class="model-option"
                :class="{ 
                  selected: selectedModels?.includes(model.id),
                  required: isModelRequired(model.id),
                  disabled: isModelDisabled(model.id)
                }"
                @click="toggleModelSelection(model.id)"
                @dblclick="showModelDetail(model.id)"
                :title="getModelTooltip(model.id)"
              >
                {{ model.name }}
                <span v-if="isModelRequired(model.id)" class="required-indicator">*</span>
                <span v-if="bulkSelectMode" class="bulk-mode-indicator">
                  {{ selectedModels.includes(model.id) ? '✓' : '○' }}
                </span>
              </div>
            </div>
        
        <!-- Combination Validation Feedback -->
            <div class="validation-feedback" :class="{ valid: combinationValid, invalid: !combinationValid }">
              <span v-if="combinationValid">✓ Combination Valid</span>
              <span v-else>✗ {{ validationMessage }}</span>
            </div>
        
        <!-- Model Dependencies -->
            <div class="model-dependencies" v-if="trainingMode === 'joint' && selectedModels?.length > 0">
              <h3>Dependencies</h3>
              <div class="dependency-list">
                <div v-for="dependency in currentDependencies" :key="dependency.model" class="dependency-item">
                  <span class="model-name">{{ dependency.model.charAt(0).toUpperCase() + dependency.model.slice(1) }}</span>
                  <span class="dependency-arrow">→</span>
                  <span class="depends-on">{{ dependency.dependencies.map(d => d.charAt(0).toUpperCase() + d.slice(1)).join(', ') }}</span>
                </div>
              </div>
            </div>
      </div>
      
      <div class="dataset-selection">
        <h2>Dataset</h2>
        <div class="dataset-controls">
          <select v-model="selectedDataset" class="dataset-select">
            <option 
              v-for="dataset in (datasets || [])" 
              :key="dataset.id"
              :value="dataset.id"
            >
              {{ dataset.name }}
            </option>
          </select>
          <button @click="recommendDataset" class="recommend-btn" :disabled="selectedModels?.length === 0">
            Intelligent Recommendation
          </button>
          <button @click="openUploadDialog" class="upload-btn">
            Upload Dataset
          </button>
        </div>
        <input 
          type="file" 
          ref="datasetInput" 
          style="display: none" 
          @change="handleDatasetUpload"
          multiple
        >
      </div>
      
      <!-- Device Selection -->
      <div class="device-selection">
        <h2>Training Device</h2>
        <div class="device-options">
          <div 
            v-for="device in (availableDevices || [])" 
            :key="device.id"
            class="device-option"
            :class="{ 
              selected: selectedDevice === device.id,
              recommended: device.recommended,
              unavailable: !device.available
            }"
            @click="selectDevice(device.id)"
          >
            <div class="device-icon">{{ device.icon }}</div>
            <div class="device-info">
              <div class="device-name">{{ device.name }}</div>
              <div class="device-details">{{ device.details }}</div>
              <div v-if="device.recommended" class="recommended-badge">Recommended</div>
              <div v-if="!device.available" class="unavailable-badge">Unavailable</div>
            </div>
          </div>
        </div>
        
        <!-- Device Switching Controls -->
        <div class="device-switching" v-if="isTraining">
          <h3>Device Switching</h3>
          <div class="switch-controls">
            <button 
              @click="switchTrainingDevice('cpu')" 
              :disabled="selectedDevice === 'cpu' || !availableDevices?.find(d => d.id === 'cpu')?.available"
              class="switch-btn"
            >
              Switch to CPU
            </button>
            <button 
              @click="switchTrainingDevice('cuda')" 
              :disabled="selectedDevice === 'cuda' || !availableDevices?.find(d => d.id === 'cuda')?.available"
              class="switch-btn"
            >
              Switch to GPU (CUDA)
            </button>
            <button 
              @click="switchTrainingDevice('mps')" 
              :disabled="selectedDevice === 'mps' || !availableDevices?.find(d => d.id === 'mps')?.available"
              class="switch-btn"
            >
              Switch to GPU (MPS)
            </button>
          </div>
          <div class="switch-info">
            <p>Device switching will pause training, transfer model state, and resume training on the new device.</p>
          </div>
        </div>
      </div>
      
      <div class="parameter-settings">
        <h2>Parameters</h2>
        <div class="parameter-grid">
          <div class="parameter">
            <label>Epochs:</label>
            <input type="number" v-model.number="parameters.epochs" min="1" max="1000">
          </div>
          <div class="parameter">
            <label>Batch Size:</label>
            <input type="number" v-model.number="parameters.batchSize" min="1" max="1024">
          </div>
          <div class="parameter">
            <label>Learning Rate:</label>
            <input type="number" v-model.number="parameters.learningRate" step="0.001" min="0.0001" max="1">
          </div>
          <div class="parameter">
            <label>Validation Split:</label>
            <input type="number" v-model.number="parameters.validationSplit" step="0.05" min="0.1" max="0.5">
          </div>
          <div class="parameter" style="grid-column: span 2;">
            <label>
              <input type="checkbox" v-model="parameters.fromScratch">
              Train from Scratch (No Pretrained Models)
            </label>
          </div>
          <div class="parameter" style="grid-column: span 2;">
            <label>
              <input type="checkbox" v-model="parameters.useExternalModelAssistance">
              Use External Model Assistance
            </label>
          </div>
          <div class="parameter" v-if="parameters.useExternalModelAssistance">
            <label>External Model:</label>
            <select v-model="parameters.externalModelId">
              <option value="">Select External Model...</option>
              <option v-for="model in availableExternalModels" 
                      :key="model.id" 
                      :value="model.id"
                      :disabled="!model.available"
              >
                {{ model.name }} {{ model.available ? '✓' : '(Unavailable)' }}
              </option>
            </select>
          </div>
          <div class="parameter">
            <label>Dropout Rate:</label>
            <input type="number" v-model.number="parameters.dropoutRate" step="0.05" min="0" max="0.5">
          </div>
          <div class="parameter">
            <label>Weight Decay:</label>
            <input type="number" v-model.number="parameters.weightDecay" step="0.0001" min="0" max="0.01">
          </div>
          <div class="parameter">
            <label>Momentum:</label>
            <input type="number" v-model.number="parameters.momentum" step="0.1" min="0" max="0.99">
          </div>
          <div class="parameter">
            <label>Optimizer:</label>
            <select v-model="parameters.optimizer">
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="rmsprop">RMSProp</option>
              <option value="adagrad">AdaGrad</option>
            </select>
          </div>
        </div>
      </div>
      
      <!-- Training Strategy Selection -->
      <div class="strategy-selection">
        <h2>Training Strategy</h2>
        <div class="strategy-options">
          <div 
            v-for="strategy in (trainingStrategies || [])" 
            :key="strategy.id"
            class="strategy-option"
            :class="{ selected: selectedStrategy === strategy.id }"
            @click="selectedStrategy = strategy.id"
          >
            {{ strategy.name }}
          </div>
        </div>
        
        <!-- Knowledge Assistance Options -->
        <div class="knowledge-assist-options" v-if="selectedStrategy === 'knowledge_assisted'">
          <h3>Knowledge Assistance Options</h3>
          <div class="knowledge-options-grid">
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.domainKnowledge">
                Domain Knowledge
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.commonSense">
                Common Sense
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.proceduralKnowledge">
                Procedural Knowledge
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.contextualLearning">
                Contextual Learning
              </label>
            </div>
            <div class="knowledge-option">
              <label>Knowledge Intensity:</label>
              <input type="range" v-model.number="knowledgeAssistOptions.knowledgeIntensity" min="0.1" max="1" step="0.1">
              <span>{{ knowledgeAssistOptions.knowledgeIntensity }}</span>
            </div>
          </div>
        </div>
        
        <!-- Pretrained Fine-tuning Options -->
        <div class="pretrained-options" v-if="selectedStrategy === 'pretrained'">
          <h3>Pretrained Fine-tuning Options</h3>
          <div class="pretrained-options-grid">
            <div class="pretrained-option">
              <label>Pretrained Model ID:</label>
              <input type="text" v-model="parameters.pretrainedModelId" placeholder="Enter pretrained model ID or URL">
            </div>
            <div class="pretrained-option">
              <label>
                <input type="checkbox" v-model="parameters.freezeLayers">
                Freeze Layers
              </label>
            </div>
            <div class="pretrained-option" v-if="parameters.freezeLayers">
              <label>Freeze Layer Count:</label>
              <input type="number" v-model.number="parameters.freezeLayerCount" min="0" max="100">
            </div>
            <div class="pretrained-option">
              <label>Fine-tuning Mode:</label>
              <select v-model="parameters.fineTuningMode">
                <option value="full">Full Fine-tuning</option>
                <option value="partial">Partial Fine-tuning</option>
                <option value="linear">Linear Probing</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      
      <div class="action-buttons">
        <button 
          @click="startTraining" 
          :disabled="isTraining"
          class="start-btn"
        >
          {{ isTraining ? 'Training in Progress' : 'Start Training' }}
        </button>
        <button 
          @click="stopTraining" 
          :disabled="!isTraining"
          class="stop-btn"
        >
          Stop Training
        </button>
      </div>
    </div>
    
    <!-- Training Progress -->
    <div class="training-progress">
      <h2>Progress</h2>
      <div class="progress-container">
        <div class="progress-bar" :style="{ width: trainingProgress + '%' }">
          {{ trainingProgress }}%
        </div>
      </div>
      <div class="progress-details">
        <div>Epoch: {{ currentEpoch }}/{{ parameters.epochs }}</div>
        <div>Loss: {{ currentLoss.toFixed(4) }}</div>
        <div>Accuracy: {{ currentAccuracy.toFixed(2) }}%</div>
        <div>Time: {{ elapsedTime }}</div>
      </div>
      
      <!-- Command Line Terminal -->
      <div class="terminal-section">
        <h3>Command Line</h3>
        <TerminalWindow
          :logs="trainingLogs"
          :title="'Training Terminal'"
          :show-timestamps="true"
          :show-input="false"
          :auto-scroll="true"
          :max-lines="200"
          class="light"
        />
      </div>
    </div>
    
    <!-- Model Evaluation -->
    <div class="model-evaluation" v-if="evaluationResults">
      <h2>Evaluation</h2>
      <div class="evaluation-grid">
        <div class="metric-card">
          <h3>Accuracy</h3>
          <div class="metric-value">{{ evaluationResults?.accuracy ?? 0 }}%</div>
        </div>
        <div class="metric-card">
          <h3>Loss</h3>
          <div class="metric-value">{{ (evaluationResults?.loss ?? 0).toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>Precision</h3>
          <div class="metric-value">{{ (evaluationResults?.precision ?? 0).toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>Recall</h3>
          <div class="metric-value">{{ (evaluationResults?.recall ?? 0).toFixed(4) }}</div>
        </div>
      </div>
      
      <div class="confusion-matrix">
        <h3>Confusion Matrix</h3>
        <div class="matrix-grid">
          <div class="matrix-header"></div>
          <div 
            v-for="label in (evaluationResults.labels || [])" 
            :key="label" 
            class="matrix-header"
          >
            {{ label }}
          </div>
          <template v-for="(row, rowIndex) in (evaluationResults.confusionMatrix || [])" :key="rowIndex">
            <div class="matrix-header">{{ evaluationResults?.labels?.[rowIndex] }}</div>
            <div 
              v-for="(cell, colIndex) in row" 
              :key="colIndex"
              class="matrix-cell"
              :class="{ highlight: rowIndex === colIndex }"
            >
              {{ cell }}
            </div>
          </template>
        </div>
      </div>
    </div>
    
    <!-- Knowledge Base Learning Control Panel -->
    <div class="knowledge-learning-panel">
      <h2>Knowledge Base Learning Management</h2>
      
      <!-- Status Messages for Knowledge Learning -->
      <div class="knowledge-status-messages">
        <div v-if="knowledgeLearningErrorState?.hasError" class="message error">
          <span class="icon">⚠️</span>
          {{ knowledgeLearningErrorState?.message }}
        </div>
        <div v-if="knowledgeLearningSuccessState?.hasSuccess" class="message success">
          <span class="icon">✅</span>
          {{ knowledgeLearningSuccessState?.message }}
        </div>
      </div>
      
      <div class="knowledge-control-grid">
        <!-- Model Selection for Knowledge Learning -->
        <div class="knowledge-model-selection">
          <h3>Select Learning Model</h3>
          <div class="knowledge-model-options">
            <div 
              v-for="model in (knowledgeLearningModels || [])" 
              :key="model.id"
              class="knowledge-model-option"
              :class="{ 
                selected: knowledgeLearningModel === model.id,
                recommended: model.recommended
              }"
              @click="selectKnowledgeLearningModel(model.id)"
              :title="model.description"
            >
              <div class="knowledge-model-icon">{{ model.icon }}</div>
              <div class="knowledge-model-info">
                <div class="knowledge-model-name">{{ model.name }}</div>
                <div class="knowledge-model-type">{{ model.type }}</div>
              </div>
            </div>
          </div>
          <div class="knowledge-model-description">
            <p>{{ selectedKnowledgeLearningModelDescription }}</p>
          </div>
        </div>
        
        <!-- Learning Parameters -->
        <div class="knowledge-parameters">
          <h3>Learning Parameters</h3>
          <div class="knowledge-parameter-grid">
            <div class="knowledge-parameter">
              <label>Learning Priority:</label>
              <select v-model="knowledgeLearningPriority">
                <option value="balanced">Balanced</option>
                <option value="exploration">Exploration (Discover New Knowledge)</option>
                <option value="exploitation">Exploitation (Deepen Existing Knowledge)</option>
              </select>
            </div>
            <div class="knowledge-parameter">
              <label>Focus Domains:</label>
              <div class="knowledge-domains-select">
                <div 
                  v-for="domain in (availableKnowledgeDomains || [])" 
                  :key="domain.id"
                  class="knowledge-domain-option"
                  :class="{ selected: knowledgeLearningDomains?.includes(domain.id) }"
                  @click="toggleKnowledgeDomain(domain.id)"
                >
                  {{ domain.name }}
                </div>
              </div>
              <div class="knowledge-domain-hint">
                Selected: {{ knowledgeLearningDomains?.length }} domains
              </div>
            </div>
            <div class="knowledge-parameter">
              <label>Learning Intensity:</label>
              <input type="range" v-model.number="knowledgeLearningIntensity" min="0.1" max="1" step="0.1">
              <span>{{ knowledgeLearningIntensity }}</span>
            </div>
            <div class="knowledge-parameter">
              <label>Max Learning Time (minutes):</label>
              <input type="number" v-model.number="knowledgeLearningMaxTime" min="1" max="240">
            </div>
          </div>
        </div>
        
        <!-- Learning Progress -->
        <div class="knowledge-progress-section">
          <h3>Learning Progress</h3>
          <div class="knowledge-progress-container" v-if="knowledgeLearningActive">
            <div class="knowledge-progress-bar" :style="{ width: knowledgeLearningProgress + '%' }">
              {{ knowledgeLearningProgress }}%
            </div>
          </div>
          <div class="knowledge-progress-details" v-if="knowledgeLearningActive">
            <div>Status: {{ knowledgeLearningStatus }}</div>
            <div>Start Time: {{ formatDate(knowledgeLearningStartTime) }}</div>
            <div>Domains: {{ knowledgeLearningDomains?.join(', ') }}</div>
            <div>Priority: {{ knowledgeLearningPriority }}</div>
          </div>
          <div class="knowledge-not-started" v-else>
            <p>Knowledge base learning is not active. Click "Start Knowledge Learning" to begin.</p>
          </div>
        </div>
        
        <!-- Action Buttons -->
        <div class="knowledge-action-buttons">
          <button 
            @click="startKnowledgeLearning" 
            :disabled="knowledgeLearningActive"
            class="knowledge-start-btn"
          >
            {{ knowledgeLearningActive ? 'Learning in Progress' : 'Start Knowledge Learning' }}
          </button>
          <button 
            @click="stopKnowledgeLearning" 
            :disabled="!knowledgeLearningActive"
            class="knowledge-stop-btn"
          >
            Stop Knowledge Learning
          </button>
        </div>
        
        <!-- Learning Logs -->
        <div class="knowledge-logs-section">
          <h3>Learning Logs</h3>
          <TerminalWindow
            :logs="knowledgeLearningLogs"
            :title="'Knowledge Learning Terminal'"
            :show-timestamps="true"
            :show-input="false"
            :auto-scroll="true"
            :max-lines="100"
            class="light"
          />
        </div>
      </div>
    </div>
    
    <!-- Training Management Panel -->
    <div class="training-management">
      <h2>Training Management</h2>
      
      <div class="management-grid">
        <!-- Current Training Status -->
        <div class="management-card">
          <h3>Current Training Status</h3>
          <div class="status-info">
            <div class="status-item">
              <span class="status-label">Active Training:</span>
              <span class="status-value" :class="{ active: isTraining, inactive: !isTraining }">
                {{ isTraining ? 'Running' : 'Idle' }}
              </span>
            </div>
            <div class="status-item">
              <span class="status-label">Selected Models ({{ selectedModels.length }}):</span>
              <span class="status-value">
                <span v-if="selectedModels.length === 0">No models selected</span>
                <span v-else>
                  <span v-for="modelId in selectedModels" :key="modelId" class="model-chip">
                    {{ getModelName(modelId) }}
                    <button v-if="!isTraining" @click.stop="removeModelFromSelection(modelId)" class="chip-remove">×</button>
                  </span>
                </span>
              </span>
            </div>
            <div class="status-item enhanced-model-controls" v-if="!isTraining">
              <span class="status-label">Enhanced Selection:</span>
              <div class="enhanced-model-panel">
                <!-- Search and Filter Row -->
                <div class="model-search-filter-row">
                  <div class="model-search">
                    <input 
                      type="text" 
                      v-model="modelSearchQuery" 
                      placeholder="Search models..." 
                      class="search-input"
                      :disabled="availableModels.length === 0"
                    />
                    <span class="search-icon">🔍</span>
                  </div>
                  <div class="model-category-filter">
                    <select v-model="selectedModelCategory" class="category-select">
                      <option v-for="category in modelCategories" :key="category.id" :value="category.id">
                        {{ category.name }}
                      </option>
                    </select>
                  </div>
                </div>
                
                <!-- Quick Actions Row -->
                <div class="model-quick-actions">
                  <div class="bulk-actions">
                    <button @click="toggleBulkSelectMode" class="btn-small" :class="{ active: bulkSelectMode }">
                      {{ bulkSelectMode ? 'Exit Bulk Mode' : 'Bulk Select' }}
                    </button>
                    <button @click="selectAllModels" class="btn-small" :disabled="availableModels.length === 0 || isAllModelsSelected">
                      Select All
                    </button>
                    <button @click="clearModelSelection" class="btn-small" :disabled="selectedModels.length === 0">
                      Clear All
                    </button>
                  </div>
                  <div class="category-actions">
                    <select @change="selectModelsByCategory($event.target.value)" class="category-action-select" :disabled="availableModels.length === 0">
                      <option value="">Add by Category...</option>
                      <option v-for="category in modelCategories.filter(c => c.id !== 'all')" :key="category.id" :value="category.id">
                        Add {{ category.name }}
                      </option>
                    </select>
                  </div>
                </div>
                
                <!-- Quick Select Dropdown -->
                <div class="quick-select-row">
                  <div class="quick-model-select">
                    <select v-model="quickSelectModel" @change="addModelToSelection" class="quick-select" :disabled="filteredModels.length === 0">
                      <option value="">Quick Add Model...</option>
                      <option v-for="model in filteredModels" :key="model.id" :value="model.id" :disabled="selectedModels.includes(model.id)">
                        {{ model.name }} ({{ model.id }}) - {{ getModelCategory(model.id) }}
                      </option>
                    </select>
                    <button @click="addModelToSelection" class="btn-small" :disabled="!quickSelectModel">Add</button>
                  </div>
                  <div class="model-stats">
                    <span class="model-count">{{ filteredModels.length }} models</span>
                    <span class="selected-count">{{ selectedModels.length }} selected</span>
                  </div>
                </div>
                
                <!-- Filtered Models Preview (if search or filter active) -->
                <div class="filtered-models-preview" v-if="modelSearchQuery.trim() !== '' || selectedModelCategory !== 'all'">
                  <div class="preview-header">
                    <h4>Filtered Models ({{ filteredModels.length }})</h4>
                    <button @click="modelSearchQuery = ''; selectedModelCategory = 'all'" class="btn-small">Clear Filters</button>
                  </div>
                  <div class="preview-chips">
                    <span 
                      v-for="model in filteredModels.slice(0, 8)" 
                      :key="model.id" 
                      class="model-chip preview-chip"
                      :class="{ selected: selectedModels.includes(model.id) }"
                      @click="toggleModelSelection(model.id)"
                      @dblclick="showModelDetail(model.id)"
                      :title="getModelTooltip(model.id)"
                    >
                      {{ model.name }}
                      <span v-if="selectedModels.includes(model.id)" class="chip-check">✓</span>
                    </span>
                    <span v-if="filteredModels.length > 8" class="more-indicator">+{{ filteredModels.length - 8 }} more</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="status-item external-api-selection">
              <span class="status-label">External API:</span>
              <div class="external-api-controls">
                <label class="external-api-toggle">
                  <input 
                    type="checkbox" 
                    v-model="parameters.useExternalModelAssistance"
                    :disabled="isTraining"
                  >
                  <span>Use External Model</span>
                </label>
                <div class="external-api-dropdown" v-if="parameters.useExternalModelAssistance">
                  <select 
                    v-model="parameters.externalModelId" 
                    :disabled="isTraining"
                    class="external-api-select"
                  >
                    <option value="">Select External Model...</option>
                    <option v-for="model in availableExternalModels" 
                            :key="model.id" 
                            :value="model.id"
                            :disabled="!model.available"
                    >
                      {{ model.name }} {{ model.available ? '✓' : '(Unavailable)' }}
                    </option>
                  </select>
                  <div class="external-api-info" v-if="parameters.externalModelId">
                    <small>Selected: {{ parameters.externalModelId }}</small>
                  </div>
                </div>
              </div>
            </div>
            <div class="status-item" v-if="isTraining">
              <span class="status-label">Job ID:</span>
              <span class="status-value">{{ currentJobId || 'N/A' }}</span>
            </div>
            <div class="status-item" v-if="isTraining">
              <span class="status-label">Progress:</span>
              <span class="status-value">{{ trainingProgress }}%</span>
            </div>
            <div class="status-item" v-if="isTraining">
              <span class="status-label">Elapsed Time:</span>
              <span class="status-value">{{ elapsedTime }}</span>
            </div>
            <div class="status-item" v-if="isTraining">
              <span class="status-label">Current Epoch:</span>
              <span class="status-value">{{ currentEpoch }} / {{ parameters.epochs || 100 }}</span>
            </div>
          </div>
          
          <div class="management-actions">
            <button @click="startTraining" :disabled="isTraining || selectedModels.length === 0" class="btn-primary">
              Start Training
            </button>
            <button @click="stopTraining" :disabled="!isTraining" class="btn-danger">
              Stop Training
            </button>
            <button @click="refreshManagement" class="btn-secondary">
              Refresh Status
            </button>
          </div>
        </div>
        
        <!-- External Model Usage -->
        <div class="management-card">
          <h3>External Model Usage</h3>
          <div class="external-model-stats">
            <div class="stats-item" v-if="externalModelStats.length > 0">
              <table class="stats-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Usage Count</th>
                    <th>Last Used</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="stat in externalModelStats" :key="stat.model">
                    <td>{{ stat.model }}</td>
                    <td>{{ stat.count }}</td>
                    <td>{{ stat.lastUsed ? formatDate(stat.lastUsed) : 'Never' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="empty-stats" v-else>
              <p>No external models have been used yet.</p>
              <p>Enable "Use External Model Assistance" in training parameters to use external models.</p>
            </div>
          </div>
          
          <div class="external-model-info">
            <h4>Available External Models</h4>
            <ul class="model-list">
              <li v-for="model in availableExternalModels" :key="model.id">
                <span class="model-name">{{ model.name }}</span>
                <span class="model-id">({{ model.id }})</span>
                <span class="model-status" :class="{ available: model.available, unavailable: !model.available }">
                  {{ model.available ? '✓ Available' : '✗ Unavailable' }}
                </span>
              </li>
            </ul>
          </div>
        </div>
        
        <!-- Resource Monitoring -->
        <div class="management-card">
          <h3>Resource Monitoring</h3>
          <div class="resource-stats">
            <div class="resource-item">
              <span class="resource-label">Training Device:</span>
              <span class="resource-value">{{ selectedDevice || 'auto' }}</span>
            </div>
            <div class="resource-item">
              <span class="resource-label">Available Devices:</span>
              <div class="device-list">
                <span v-for="device in (availableDevices || [])" 
                      :key="device.id"
                      class="device-tag"
                      :class="{ 
                        available: device.available, 
                        selected: selectedDevice === device.id,
                        recommended: device.recommended 
                      }">
                  {{ device.name }}
                  <span v-if="!device.available" class="unavailable-indicator">(Unavailable)</span>
                  <span v-if="device.recommended" class="recommended-indicator">(Recommended)</span>
                </span>
              </div>
            </div>
            <div class="resource-item">
              <span class="resource-label">Memory Usage:</span>
              <span class="resource-value">{{ memoryUsage || 'Monitoring not available' }}</span>
            </div>
            <div class="resource-item">
              <span class="resource-label">CPU Usage:</span>
              <span class="resource-value">{{ cpuUsage || 'Monitoring not available' }}</span>
            </div>
          </div>
          
          <!-- Resource Usage Chart -->
          <div class="resource-chart-container">
            <h4>Resource Usage History</h4>
            <div class="chart-wrapper">
              <canvas id="resourceChartCanvas"></canvas>
            </div>
          </div>
          
          <div class="resource-actions">
            <button @click="switchTrainingDevice('cpu')" 
                    :disabled="selectedDevice === 'cpu' || !availableDevices?.find(d => d.id === 'cpu')?.available"
                    class="btn-small">
              Switch to CPU
            </button>
            <button @click="switchTrainingDevice('cuda')" 
                    :disabled="selectedDevice === 'cuda' || !availableDevices?.find(d => d.id === 'cuda')?.available"
                    class="btn-small">
              Switch to CUDA
            </button>
            <button @click="switchTrainingDevice('mps')" 
                    :disabled="selectedDevice === 'mps' || !availableDevices?.find(d => d.id === 'mps')?.available"
                    class="btn-small">
              Switch to MPS
            </button>
            <button @click="checkResourceAvailability" class="btn-small">
              Check Resources
            </button>
          </div>
        </div>
        
        <!-- Training Task Management -->
        <div class="management-card">
          <h3>Training Task Management</h3>
          <div class="task-list">
            <div class="task-item" v-if="activeTrainingTasks.length > 0">
              <h4>Active Tasks</h4>
              <ul>
                <li v-for="task in activeTrainingTasks" :key="task.id">
                  <span class="task-name">{{ task.name }}</span>
                  <span class="task-status" :class="task.status">{{ task.status }}</span>
                  <button @click="stopTask(task.id)" class="btn-small btn-danger">Stop</button>
                </li>
              </ul>
            </div>
            <div class="no-tasks" v-else>
              <p>No active training tasks.</p>
            </div>
          </div>
          
          <div class="task-history-summary">
            <h4>Training History Summary</h4>
            <div class="summary-stats">
              <div class="summary-item">
                <span class="summary-label">Total Training Sessions:</span>
                <span class="summary-value">{{ trainingHistory.length }}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Successful:</span>
                <span class="summary-value">{{ successfulTrainings }}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Average Accuracy:</span>
                <span class="summary-value">{{ averageAccuracy }}%</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">Total Training Time:</span>
                <span class="summary-value">{{ totalTrainingTime }}</span>
              </div>
            </div>
          </div>
          
          <div class="task-actions">
            <button @click="loadTrainingHistory" class="btn-small">
              Refresh History
            </button>
            <button @click="clearTrainingHistory" class="btn-small btn-danger">
              Clear History
            </button>
            <button @click="exportTrainingData" class="btn-small">
              Export Data
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Training History -->
    <div class="training-history">
      <h2>History</h2>
      <table class="history-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Models</th>
            <th>Dataset</th>
            <th>Duration</th>
            <th>Accuracy</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="session in (trainingHistory || [])" :key="session.id">
            <td>{{ formatDate(session.date) }}</td>
            <td>
              <span v-for="model in (session.models || [])" :key="model">
                {{ model }}
              </span>
            </td>
            <td>{{ session.dataset }}</td>
            <td>{{ formatDuration(session.duration) }}</td>
            <td>{{ session.accuracy }}%</td>
            <td>
              <button @click="viewSession(session.id)">View</button>
              <button @click="compareSession(session.id)">Compare</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    
    <!-- Model Details Modal -->
    <div v-if="showModelDetails" class="model-details-modal" @click.self="closeModelDetails">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Model Details</h3>
          <button @click="closeModelDetails" class="modal-close">×</button>
        </div>
        <div class="modal-body" v-if="selectedModelDetails">
          <div class="model-detail-section">
            <div class="detail-row">
              <span class="detail-label">Model ID:</span>
              <span class="detail-value">{{ selectedModelDetails.id }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Name:</span>
              <span class="detail-value">{{ selectedModelDetails.name }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Backend ID:</span>
              <span class="detail-value">{{ selectedModelDetails.backendId }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Category:</span>
              <span class="detail-value">{{ getModelCategory(selectedModelDetails.id) }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Status:</span>
              <span class="detail-value status-badge" :class="selectedModelDetails.status">
                {{ selectedModelDetails.status }}
              </span>
            </div>
            <div class="detail-row full-width">
              <span class="detail-label">Description:</span>
              <div class="detail-value description-text">
                {{ selectedModelDetails.description || 'No description available' }}
              </div>
            </div>
          </div>
          <div class="modal-actions">
            <button 
              @click="toggleModelSelection(selectedModelDetails.id)" 
              class="btn-primary"
              :disabled="isTraining"
            >
              {{ selectedModels.includes(selectedModelDetails.id) ? 'Deselect Model' : 'Select Model' }}
            </button>
            <button @click="closeModelDetails" class="btn-secondary">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.train-view {
  max-width: 1200px;
  margin: 70px auto 0;
  padding: 20px;
  font-family: var(--font-family);
  color: var(--text-primary);
  background-color: var(--bg-primary);
  min-height: calc(100vh - 70px);
}

.status-messages {
  margin-bottom: 20px;
}

.message {
  padding: 12px 16px;
  margin-bottom: 8px;
  border-radius: 4px;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.message.error {
  background-color: #f8f0f0;
  color: #d32f2f;
  border: 1px solid #f8d7da;
}

.message.success {
  background-color: #f0f8f0;
  color: #2e7d32;
  border: 1px solid #d4edda;
}

.message.warning {
  background-color: #fdf7e6;
  color: #f57c00;
  border: 1px solid #fff3cd;
}

.message.info {
  background-color: #f0f4f8;
  color: #1976d2;
  border: 1px solid #cce7ff;
}

.control-panel {
  background-color: #fff;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.control-panel h2 {
  margin-top: 0;
  margin-bottom: 16px;
  font-size: 18px;
  color: #111;
  font-weight: 600;
}

.control-panel h3 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
  color: #222;
  font-weight: 500;
}

.mode-selection, .model-selection, .dataset-selection, .device-selection, .parameter-settings, .strategy-selection {
  margin-bottom: 24px;
}

.mode-options, .strategy-options {
  display: flex;
  gap: 12px;
}

.mode-options button, .strategy-options .strategy-option {
  padding: 8px 16px;
  border: 1px solid #ddd;
  background-color: #fff;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 14px;
}

.mode-options button:hover, .strategy-options .strategy-option:hover {
  background-color: #f5f5f5;
  border-color: #bbb;
}

.mode-options button.active, .strategy-options .strategy-option.selected {
  background-color: #333;
  color: #fff;
  border-color: #333;
}

.model-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.model-option {
  padding: 12px 16px;
  border: 2px solid #ddd;
  background-color: #fff;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 14px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.model-option:hover {
  border-color: #999;
  background-color: #f9f9f9;
}

.model-option.selected {
  border-color: #333;
  background-color: #f0f0f0;
  font-weight: 500;
}

.model-option.required {
  border-width: 2px;
  border-style: dashed;
}

/* Device Selection Styles */
.device-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.device-option {
  display: flex;
  align-items: center;
  padding: 16px;
  border: 2px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: #fff;
}

.device-option:hover {
  border-color: #999;
  background-color: #f9f9f9;
  transform: translateY(-2px);
}

.device-option.selected {
  border-color: #333;
  background-color: #f0f0f0;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.device-option.recommended {
  border-color: #4CAF50;
  background-color: #f8fff8;
}

.device-option.unavailable {
  opacity: 0.5;
  cursor: not-allowed;
  background-color: #f5f5f5;
}

.device-option.unavailable:hover {
  border-color: #ddd;
  background-color: #f5f5f5;
  transform: none;
}

.device-icon {
  font-size: 32px;
  margin-right: 16px;
  flex-shrink: 0;
}

.device-info {
  flex: 1;
}

.device-name {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
  color: #333;
}

.device-details {
  font-size: 14px;
  color: #666;
  line-height: 1.4;
}

.recommended-badge {
  display: inline-block;
  background-color: #4CAF50;
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  margin-top: 8px;
}

.unavailable-badge {
  display: inline-block;
  background-color: #f44336;
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  margin-top: 8px;
}

/* Device Switching Controls */
.device-switching {
  background-color: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
  margin-top: 16px;
  border: 1px solid #e9ecef;
}

.device-switching h3 {
  margin-top: 0;
  margin-bottom: 12px;
  color: #495057;
}

.switch-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.switch-btn {
  padding: 8px 16px;
  border: 1px solid #007bff;
  background-color: #fff;
  color: #007bff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.switch-btn:hover:not(:disabled) {
  background-color: #007bff;
  color: white;
}

.switch-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  border-color: #ccc;
  color: #ccc;
}

.switch-info {
  font-size: 14px;
  color: #6c757d;
  line-height: 1.5;
}

.switch-info p {
  margin: 0;
}

/* External Model Assistance Styles */
.parameter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.parameter {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.parameter label {
  font-weight: 500;
  font-size: 14px;
  color: #333;
}

.parameter input[type="number"],
.parameter input[type="text"],
.parameter select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.parameter input[type="number"]:focus,
.parameter input[type="text"]:focus,
.parameter select:focus {
  outline: none;
  border-color: #007bff;
}

.parameter input[type="checkbox"] {
  margin-right: 8px;
}

.parameter label[for] {
  display: flex;
  align-items: center;
  cursor: pointer;
}

.model-option.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.dataset-select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  margin-right: 12px;
  background-color: #fff;
  min-width: 200px;
}

.upload-btn, .start-btn, .stop-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
}

.upload-btn {
  background-color: #666;
  color: white;
}

.upload-btn:hover {
  background-color: #555;
}

.start-btn {
  background-color: #333;
  color: white;
  padding: 10px 24px;
  font-size: 16px;
}

.start-btn:hover:not(:disabled) {
  background-color: #222;
}

.stop-btn {
  background-color: #666;
  color: white;
  padding: 10px 24px;
  font-size: 16px;
  margin-left: 12px;
}

.stop-btn:hover:not(:disabled) {
  background-color: #555;
}

.start-btn:disabled, .stop-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.parameter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.parameter {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.parameter label {
  font-size: 14px;
  color: #444;
  font-weight: 500;
}

.parameter input[type="number"], .parameter select {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
  background-color: #fff;
}

.validation-feedback {
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 14px;
  margin-top: 8px;
}

.validation-feedback.valid {
  background-color: #f0f8f0;
  color: #2e7d32;
  border: 1px solid #d4edda;
}

.validation-feedback.invalid {
  background-color: #f8f0f0;
  color: #d32f2f;
  border: 1px solid #f8d7da;
}

.model-dependencies {
  margin-top: 16px;
}

.dependency-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.dependency-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #666;
}

.dependency-arrow {
  color: #999;
}

.training-progress {
  background-color: #fff;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.progress-container {
  height: 30px;
  background-color: #f0f0f0;
  border-radius: 15px;
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-bar {
  height: 100%;
  background-color: #333;
  text-align: center;
  line-height: 30px;
  color: white;
  font-size: 14px;
  font-weight: 500;
  transition: width 0.3s ease;
}

.progress-details {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
  color: #666;
}

.terminal-section {
  margin-top: 24px;
}

.model-evaluation {
  background-color: #fff;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.evaluation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.metric-card {
  background-color: #f5f5f5;
  padding: 16px;
  border-radius: 8px;
  text-align: center;
}

.metric-card h3 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: #333;
}

.confusion-matrix {
  margin-top: 24px;
}

.matrix-grid {
  display: grid;
  grid-template-columns: auto repeat(auto-fit, minmax(60px, 1fr));
  gap: 4px;
}

.matrix-header {
  padding: 8px;
  font-size: 14px;
  font-weight: 500;
  text-align: center;
  background-color: #f0f0f0;
  border-radius: 4px;
}

.matrix-cell {
  padding: 12px 8px;
  text-align: center;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.matrix-cell.highlight {
  background-color: #f0f0f0;
  font-weight: 500;
}

.training-history {
  background-color: #fff;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.history-table {
  width: 100%;
  border-collapse: collapse;
}

.history-table th, .history-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
  font-size: 14px;
}

.history-table th {
  background-color: #f5f5f5;
  font-weight: 600;
  color: #333;
}

.history-table tr:hover {
  background-color: #f9f9f9;
}

.history-table button {
  padding: 4px 8px;
  margin-right: 8px;
  border: 1px solid #ddd;
  background-color: #fff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.history-table button:hover {
  background-color: #f5f5f5;
  border-color: #999;
}

.knowledge-assist-options {
  background-color: #f9f9f9;
  padding: 16px;
  border-radius: 4px;
  margin-top: 12px;
}

.knowledge-options-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.knowledge-option {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.knowledge-option label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.knowledge-option input[type="range"] {
  flex: 1;
}

.recommended-combinations {
  margin-bottom: 16px;
}

.combination-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.combination-btn {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background-color: #fff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.combination-btn:hover {
  background-color: #f5f5f5;
  border-color: #999;
}

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}
</style>

<script>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue';
import api from '@/utils/api';
import { 
  handleApiError,
  logInfo,
  logWarning,
  logSuccess 
} from '@/utils/enhancedErrorHandler';
import { letterToId, idToLetter, letterToIdMap, modelIdToLetterMap, lettersToIds, idsToLetters, letterIds, getModelDisplayName, getModelDescription } from '@/utils/modelIdMapper';
import TerminalWindow from '@/components/TerminalWindow.vue';
import { notify } from '@/plugins/notification';
import { Chart, registerables } from 'chart.js';

export default {
    setup() {
      // Component mounted state
      const isMounted = ref(true)
      
      // Training mode
    const trainingMode = ref('individual');
    
    // Available models - dynamically fetched from API
    const availableModels = ref([]);
    const modelsLoading = ref(true);
    const modelsError = ref(null);
    
    // Get available model list
    // Generic data loading function for TrainView
    const performDataLoad = async (options) => {
      const {
        apiCall,
        loadingRef,
        errorRef,
        successMessage,
        errorMessage,
        errorContext,
        showSuccess = true,
        showError = true,
        onSuccess,
        onError,
        onFinally
      } = options

      if (loadingRef) loadingRef.value = true
      if (errorRef) errorRef.value = null

      try {
        const response = await apiCall()
        
        if (onSuccess && typeof onSuccess === 'function') {
          await onSuccess(response.data)
        }
        
        if (showSuccess && successMessage) {
          showInfo(successMessage)
        }
        
        return response.data
      } catch (error) {
        if (import.meta.env.DEV) {
          console.error(errorContext || 'Data load error:', error)
        }
        
        if (errorRef) errorRef.value = error
        
        if (onError && typeof onError === 'function') {
          await onError(error)
        }
        
        if (showError && errorMessage) {
          showError(errorMessage)
        }
        
        throw error
      } finally {
        if (loadingRef) loadingRef.value = false
        if (onFinally && typeof onFinally === 'function') {
          await onFinally()
        }
      }
    }

    const loadAvailableModels = async () => {
      return await performDataLoad({
        apiCall: () => api.models.available(),
        loadingRef: modelsLoading,
        errorRef: modelsError,
        successMessage: 'Models loaded successfully from backend',
        errorMessage: 'Failed to load models from backend',
        errorContext: 'Failed to load models',
        showSuccess: true,
        showError: true,
        onSuccess: (data) => {
          if (data.status === 'success') {
            availableModels.value = (data.models || []).map(model => {
              const frontendId = idToLetter(model.id);
              return {
                ...model,
                id: frontendId,
                backendId: model.id
              };
            });
          } else {
            throw new Error(data.message || 'Failed to load models');
          }
        },
        onError: (error) => {
          handleApiError(error, 'Failed to load models');
          // Do not set default models when API fails, keep empty array
          availableModels.value = [];
        }
      });
    };
    
    
    // Selected models
    const selectedModels = ref([]);
    
    // Quick model selection for management panel
    const quickSelectModel = ref('');
    
    // Datasets
    const datasets = ref([]);
    const datasetsLoading = ref(true);
    const datasetsError = ref(null);
    
    // Load datasets from API
    const loadDatasets = async () => {
      return await performDataLoad({
        apiCall: () => api.datasets.get(),
        loadingRef: datasetsLoading,
        errorRef: datasetsError,
        successMessage: null, // We'll handle logging manually
        errorMessage: null,
        errorContext: 'Failed to load datasets',
        showSuccess: false,
        showError: false,
        onSuccess: (data) => {
          if (data && data.datasets) {
            datasets.value = data.datasets;
            addLog('Datasets loaded successfully from backend');
            
            // Set default selected dataset if not already set
            if (!selectedDataset.value && datasets.value.length > 0) {
              selectedDataset.value = datasets.value[0].id;
              addLog(`Default dataset selected: ${datasets.value[0].name}`);
            }
          } else {
            addLog('API returned empty dataset list');
            datasets.value = [];
          }
        },
        onError: (error) => {
          datasetsError.value = error;
          addLog('Failed to load datasets from backend');
          datasets.value = [];
        }
      });
    };
    
    // Load recommended combinations from API
    const loadRecommendedCombinations = async () => {
      try {
        const response = await api.training.getJointTrainingRecommendations();
        if (response.data && response.data.success && response.data.recommendations) {
          recommendedCombinations.value = response.data.recommendations;
          addLog('Joint training recommendations loaded successfully');
        } else {
          addLog('Failed to load joint training recommendations: API returned empty data');
          recommendedCombinations.value = {};
        }
      } catch (error) {
        handleApiError(error, 'Load Joint Training Recommendations');
        addLog('Failed to load joint training recommendations from API');
        recommendedCombinations.value = {};
      }
    };
    
    // Recommended combinations - loaded from API
    const recommendedCombinations = ref({});
    
    // Combination validation status
    const combinationValid = ref(true);
    const validationMessage = ref('');
    
    // Model dependencies - will be loaded from API
    const modelDependencies = ref({});
    const modelDependenciesLoading = ref(false);
    const modelDependenciesError = ref(null);
    
    // Load model dependencies from API
    const loadModelDependencies = async () => {
      modelDependenciesLoading.value = true;
      modelDependenciesError.value = null;
      
      try {
        const response = await api.models.getDependencies();
        
        if (response.data && response.data.status === 'success' && response.data.dependencies) {
          // Convert backend model IDs to frontend letter IDs for consistency
          const convertedDependencies = {};
          
          Object.entries(response.data.dependencies).forEach(([backendModelId, deps]) => {
            const frontendModelId = idToLetter(backendModelId);
            
            if (frontendModelId) {
              // Convert each dependency ID to frontend letter ID
              const frontendDeps = deps.map(depId => idToLetter(depId)).filter(depId => depId);
              convertedDependencies[frontendModelId] = frontendDeps;
            }
          });
          
          modelDependencies.value = convertedDependencies;
          addLog('Model dependencies loaded successfully from backend');
        } else {
          throw new Error('Failed to load model dependencies: API returned invalid data');
        }
      } catch (error) {
        modelDependenciesError.value = error;
        handleApiError(error, 'Load Model Dependencies');
        addLog('Failed to load model dependencies from backend');
        // Set empty dependencies on error
        modelDependencies.value = {};
      } finally {
        modelDependenciesLoading.value = false;
      }
    };
    
    // Computed properties
    const currentDependencies = computed(() => {
      const dependencies = [];
      selectedModels.value.forEach(model => {
        if (modelDependencies.value[model]) {
          modelDependencies.value[model].forEach(dep => {
            if (!selectedModels.value.includes(dep)) {
              dependencies.push({
                model: model,
                dependencies: [dep]
              });
            }
          });
        }
      });
      return dependencies;
    });
    
    // Check if model is required (has dependencies)
    const isModelRequired = computed(() => (modelId) => {
      return currentDependencies.value.some(dep => 
        dep.dependencies.includes(modelId)
      );
    });
    
    // Check if model is disabled (missing dependencies or not supported by dataset)
    const isModelDisabled = computed(() => (modelId) => {
      if (trainingMode.value === 'individual') return false;
      
      // Check if the selected dataset supports this model
      const selectedDatasetConfig = datasets.value.find(d => d.id === selectedDataset.value);
      if (selectedDatasetConfig && selectedDatasetConfig.supportedModels) {
        // If dataset has supportedModels, check if this model is in the list
        if (!selectedDatasetConfig.supportedModels.includes(modelId)) {
          return true; // Model not supported by dataset
        }
      }
      
      // Check if there are unsatisfied dependencies
      const dependencies = modelDependencies.value[modelId];
      if (dependencies && dependencies.length > 0) {
        return dependencies.some(dep => !selectedModels.value.includes(dep));
      }
      return false;
    });
    
    // Get model tooltip information
    const getModelTooltip = computed(() => (modelId) => {
      const model = availableModels.value.find(m => m.id === modelId);
      const dependencies = modelDependencies.value[modelId];
      let tooltip = '';
      
      // Add model description if available
      if (model && model.description) {
        tooltip += `${model.description}\n`;
      }
      
      // Add dependencies if any
      if (dependencies && dependencies.length > 0) {
        tooltip += `Requires: ${dependencies.map(d => {
          const depModel = availableModels.value.find(m => m.id === d);
          return depModel ? depModel.name : d.toUpperCase();
        }).join(', ')}`;
      }
      
      return tooltip;
    });
    
    // Enhanced model filtering computed properties
    const filteredModels = computed(() => {
      if (!availableModels.value || availableModels.value.length === 0) {
        return [];
      }
      
      let filtered = [...availableModels.value];
      
      // Apply category filter
      if (selectedModelCategory.value !== 'all') {
        const categoryMap = {
          'language': ['B'],
          'vision': ['D', 'E', 'P'],
          'audio': ['C'],
          'sensor': ['G', 'Q'],
          'knowledge': ['J'],
          'planning': ['L', 'M', 'S'],
          'specialized': ['V', 'W', 'X', 'U', 'T', 'H', 'I', 'F', 'K', 'N']
        };
        
        const categoryModels = categoryMap[selectedModelCategory.value] || [];
        filtered = filtered.filter(model => categoryModels.includes(model.id));
      }
      
      // Apply search filter
      if (modelSearchQuery.value.trim() !== '') {
        const query = modelSearchQuery.value.toLowerCase().trim();
        filtered = filtered.filter(model => 
          model.name.toLowerCase().includes(query) ||
          (model.id && model.id.toLowerCase().includes(query)) ||
          (model.description && model.description.toLowerCase().includes(query))
        );
      }
      
      return filtered;
    });
    
    const isAllModelsSelected = computed(() => {
      if (!availableModels.value || availableModels.value.length === 0) return false;
      return availableModels.value.every(model => selectedModels.value.includes(model.id));
    });
    
    const isSomeModelsSelected = computed(() => {
      return selectedModels.value.length > 0;
    });
    
    // Select recommended combination - ensure all dependencies are satisfied
    const selectRecommendedCombination = (combination) => {
      // Deep copy the incoming combination
      const newSelection = [...combination];
      
      // Collect all required dependency models
      const requiredModels = new Set(newSelection);
      let dependenciesFound = true;
      
      // Loop until no new dependencies are found
      while (dependenciesFound) {
        dependenciesFound = false;
        for (const model of Array.from(requiredModels)) {
          const dependencies = modelDependencies.value[model];
          if (dependencies && dependencies.length > 0) {
            for (const dep of dependencies) {
              if (!requiredModels.has(dep)) {
                requiredModels.add(dep);
                newSelection.push(dep);
                dependenciesFound = true;
              }
            }
          }
        }
      }
      
      // Deduplicate and set selected models
      selectedModels.value = [...new Set(newSelection)];
    };
    

    
    // Recommend dataset based on selected models
    const recommendDataset = () => {
      if (selectedModels.value.length === 0) {
        showWarning('Please select at least one model for dataset recommendation');
        return;
      }
      
      // Find dataset that supports all selected models
      const compatibleDatasets = datasets.value.filter(dataset => {
        if (!dataset.supportedModels) return false;
        return selectedModels.value.every(modelId => 
          dataset.supportedModels.includes(modelId)
        );
      });
      
      if (compatibleDatasets.length > 0) {
        // Sort by number of supported models (descending) to get the most comprehensive dataset
        compatibleDatasets.sort((a, b) => {
          const aCount = a.supportedModels ? a.supportedModels.length : 0;
          const bCount = b.supportedModels ? b.supportedModels.length : 0;
          return bCount - aCount;
        });
        
        const recommendedDataset = compatibleDatasets[0];
        selectedDataset.value = recommendedDataset.id;
        showSuccess(`Recommended dataset: ${recommendedDataset.name}`);
        
        // If multimodal dataset is available and selected models are diverse, suggest it
        const multimodalDataset = datasets.value.find(d => d.name.includes('Multimodal'));
        if (multimodalDataset && selectedModels.value.length >= 3) {
          showInfo('Multimodal dataset recommended for comprehensive training');
        }
      } else {
        showWarning('No compatible dataset found for the selected models. Please upload a suitable dataset.');
      }
    };
    
    // Clear selected models
    const clearSelectedModels = () => {
      selectedModels.value = [];
      showInfo('All models deselected');
    };
    
    // Get model name by ID
    const getModelName = (modelId) => {
      const model = availableModels.value.find(m => m.id === modelId);
      return model ? model.name : modelId;
    };
    
    // Show warning
    const showWarning = (message) => {
      warningState.value = {
        hasWarning: true,
        message
      };
      
      setTimeout(() => {
        if (isMounted.value) {
          warningState.value.hasWarning = false;
        }
      }, 8000);
    };
    
    // Show success message
    const showSuccess = (message) => {
      successState.value = {
        hasSuccess: true,
        message
      };
      
      setTimeout(() => {
        if (isMounted.value) {
          successState.value.hasSuccess = false;
        }
      }, 5000);
    };
    
    // Show info message
    const showInfo = (message) => {
      infoState.value = {
        hasInfo: true,
        message
      };
      
      setTimeout(() => {
        if (isMounted.value) {
          infoState.value.hasInfo = false;
        }
      }, 5000);
    };
    
    // Validate model combination
    const validateModelCombination = () => {
      if (trainingMode.value === 'individual') {
        combinationValid.value = true;
        validationMessage.value = '';
        return;
      }
      
      // Check dependencies
      const missingDependencies = [];
      selectedModels.value.forEach(model => {
        const dependencies = modelDependencies.value[model];
        if (dependencies) {
          dependencies.forEach(dep => {
            if (!selectedModels.value.includes(dep)) {
              missingDependencies.push({
                model: model,
                dependency: dep
              });
            }
          });
        }
      });
      
      if (missingDependencies.length > 0) {
        combinationValid.value = false;
        validationMessage.value = 'Missing dependencies: ' + 
          missingDependencies.map(d => 
            `${d.model.toUpperCase()} → ${d.dependency.toUpperCase()}`
          ).join(', ');
      } else {
        combinationValid.value = true;
        validationMessage.value = 'Model combination is valid';
      }
    };
    
    // Watch for model selection changes
    watch([selectedModels, trainingMode], () => {
      validateModelCombination();
    }, { immediate: true });
    
    // Selected dataset - initialized as empty, will be set after datasets are loaded
    const selectedDataset = ref('');
    
    // Available devices
    const availableDevices = ref([]);
    const devicesLoading = ref(true);
    const devicesError = ref(null);
    
    // Load available devices from API
    const loadAvailableDevices = async () => {
      try {
        devicesLoading.value = true;
        devicesError.value = null;
        
        const response = await api.training.availableDevices();
        
        if (response.data.status === 'success') {
          // Map backend device data to frontend format
          availableDevices.value = (response.data.devices || []).map(device => ({
            id: device.id,
            name: device.name,
            icon: device.icon || '💻',
            details: device.description || '',
            available: device.available,
            recommended: device.recommended
          }));
        } else {
          throw new Error(response.data.message || 'API returned unsuccessful status');
        }
      } catch (error) {
        devicesError.value = error.message || 'Failed to load devices';
        availableDevices.value = [];
      } finally {
        devicesLoading.value = false;
      }
    };
    
    // Selected device
    const selectedDevice = ref('auto');
    
    // Training parameters
    const parameters = ref({
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      validationSplit: 0.2,
      dropoutRate: 0.1,
      weightDecay: 0.0001,
      momentum: 0.9,
      optimizer: 'adam',
      learningRateSchedule: 'constant',
      // External model assistance
      useExternalModelAssistance: false,
      externalModelId: 'gpt-4',
      // Pretraining parameters
      pretrainedModelId: null,
      freezeLayers: false,
      freezeLayerCount: 0,
      fineTuningMode: 'full'
    });
    
    // Training strategy options
    const trainingStrategies = ref([
      { id: 'standard', name: 'Standard Training' },
      { id: 'knowledge_assisted', name: 'Knowledge Assisted Training' },
      { id: 'progressive', name: 'Progressive Training' },
      { id: 'adaptive', name: 'Adaptive Learning' },
      { id: 'pretrained', name: 'Pre-trained Fine-tuning' }
    ]);
    
    // Selected training strategy
    const selectedStrategy = ref('standard');
    
    // Whether to enable knowledge assistance
    const enableKnowledgeAssist = ref(false);
    
    // Knowledge assistance options
    const knowledgeAssistOptions = ref({
      domainKnowledge: true,
      commonSense: true,
      proceduralKnowledge: false,
      contextualLearning: true,
      knowledgeIntensity: 0.7
    });
    
    // Training status
    const isTraining = ref(false);
    const trainingProgress = ref(0);
    const currentEpoch = ref(0);
    const currentLoss = ref(0);
    const currentAccuracy = ref(0);
    const elapsedTime = ref('00:00:00');
    const trainingLogs = ref([]);
    const currentJobId = ref(null);
    const websocketConnection = ref(null);
    const statusPollingInterval = ref(null);
    
    // Evaluation results
    const evaluationResults = ref(null);
    
    // Additional evaluation metrics
    const validationLoss = ref(0);
    const validationAccuracy = ref(0);
    
    // Training history
    const trainingHistory = ref([]);
    
    // For session comparison
    const comparingSessions = ref([]);
    
    // Training management variables
    const externalModelStats = ref([]);
    const availableExternalModels = ref([]);
    const memoryUsage = ref('');
    const cpuUsage = ref('');
    const activeTrainingTasks = ref([]);
    
    // Resource monitoring chart
    const resourceChart = ref(null);
    const resourceHistory = ref({
      timestamps: [],
      cpu: [],
      memory: [],
      gpu: []
    });
    const maxHistoryPoints = 20;
    const resourceUpdateInterval = ref(null);
    
    // Enhanced model selection variables
    const modelSearchQuery = ref('');
    const selectedModelCategory = ref('all');
    const modelCategories = ref([
      { id: 'all', name: 'All Models' },
      { id: 'language', name: 'Language Models' },
      { id: 'vision', name: 'Vision Models' },
      { id: 'audio', name: 'Audio Models' },
      { id: 'sensor', name: 'Sensor Models' },
      { id: 'knowledge', name: 'Knowledge Models' },
      { id: 'planning', name: 'Planning Models' },
      { id: 'specialized', name: 'Specialized Models' }
    ]);
    const showModelDetails = ref(false);
    const selectedModelDetails = ref(null);
    const bulkSelectMode = ref(false);
    
    // Computed training statistics
    const successfulTrainings = computed(() => {
      return trainingHistory.value.filter(session => 
        session.status === 'completed' || session.accuracy > 0
      ).length;
    });
    
    const averageAccuracy = computed(() => {
      const completedSessions = trainingHistory.value.filter(session => 
        session.status === 'completed' || session.accuracy > 0
      );
      if (completedSessions.length === 0) return 0;
      const totalAccuracy = completedSessions.reduce((sum, session) => sum + (session.accuracy || 0), 0);
      return Math.round((totalAccuracy / completedSessions.length) * 100) / 100;
    });
    
    const totalTrainingTime = computed(() => {
      const completedSessions = trainingHistory.value.filter(session => 
        session.status === 'completed'
      );
      if (completedSessions.length === 0) return '0h 0m';
      const totalSeconds = completedSessions.reduce((sum, session) => {
        if (session.duration) {
          // Parse duration string like "1h 30m" or "45m"
          const hoursMatch = session.duration.match(/(\d+)h/);
          const minutesMatch = session.duration.match(/(\d+)m/);
          const hours = hoursMatch ? parseInt(hoursMatch[1]) : 0;
          const minutes = minutesMatch ? parseInt(minutesMatch[1]) : 0;
          return sum + hours * 3600 + minutes * 60;
        }
        return sum;
      }, 0);
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    });
    
    // Knowledge Base Learning Management
    const knowledgeLearningModels = ref([
      { id: 'transformer', name: 'Transformer', type: 'Deep Learning', icon: '🧠', description: 'Deep learning model for knowledge extraction', recommended: true },
      { id: 'graph', name: 'Graph Neural Network', type: 'Graph Learning', icon: '🕸️', description: 'Graph-based knowledge representation learning', recommended: false },
      { id: 'memory', name: 'Memory Network', type: 'Memory-Augmented', icon: '💾', description: 'Memory-enhanced knowledge retention', recommended: false },
      { id: 'hybrid', name: 'Hybrid Model', type: 'Multi-Modal', icon: '🔗', description: 'Combines multiple learning approaches', recommended: true }
    ]);
    
    const knowledgeLearningModel = ref('transformer');
    const knowledgeLearningPriority = ref('balanced');
    const knowledgeLearningDomains = ref([]);
    const knowledgeLearningIntensity = ref(0.5);
    const knowledgeLearningMaxTime = ref(60);
    const knowledgeLearningActive = ref(false);
    const knowledgeLearningPollInterval = ref(null);
    const knowledgeLearningProgress = ref(0);
    const knowledgeLearningStatus = ref('Ready to start');
    const knowledgeLearningStartTime = ref(null);
    const knowledgeLearningLogs = ref([]);
    const knowledgeLearningErrorState = ref({
      hasError: false,
      message: '',
      type: ''
    });
    const knowledgeLearningSuccessState = ref({
      hasSuccess: false,
      message: ''
    });
    
    const availableKnowledgeDomains = ref([
      { id: 'science', name: 'Science' },
      { id: 'technology', name: 'Technology' },
      { id: 'arts', name: 'Arts & Humanities' },
      { id: 'history', name: 'History' },
      { id: 'mathematics', name: 'Mathematics' },
      { id: 'philosophy', name: 'Philosophy' },
      { id: 'linguistics', name: 'Linguistics' },
      { id: 'psychology', name: 'Psychology' }
    ]);
    
    const selectedKnowledgeLearningModelDescription = computed(() => {
      const model = knowledgeLearningModels.value.find(m => m.id === knowledgeLearningModel.value);
      return model ? model.description : 'Select a learning model';
    });
    
    const selectKnowledgeLearningModel = (modelId) => {
      knowledgeLearningModel.value = modelId;
      addLog(`Selected knowledge learning model: ${modelId}`);
    };
    
    const toggleKnowledgeDomain = (domainId) => {
      const index = knowledgeLearningDomains.value.indexOf(domainId);
      if (index > -1) {
        knowledgeLearningDomains.value.splice(index, 1);
      } else {
        knowledgeLearningDomains.value.push(domainId);
      }
      addLog(`Toggled knowledge domain: ${domainId}, Selected: ${knowledgeLearningDomains.value.length} domains`);
    };
    
    const startKnowledgeLearning = async () => {
      if (knowledgeLearningActive.value) {
        addLog('Knowledge learning already in progress', 'warning');
        return;
      }
      
      try {
        knowledgeLearningActive.value = true;
        knowledgeLearningProgress.value = 0;
        knowledgeLearningStatus.value = 'Initializing...';
        knowledgeLearningStartTime.value = new Date();
        knowledgeLearningLogs.value = [];
        knowledgeLearningErrorState.value.hasError = false;
        knowledgeLearningSuccessState.value.hasSuccess = false;
        
        addLog('Starting knowledge base learning...', 'info');
        addLog(`Model: ${knowledgeLearningModel.value}`, 'info');
        addLog(`Domains: ${knowledgeLearningDomains.value.join(', ')}`, 'info');
        addLog(`Priority: ${knowledgeLearningPriority.value}`, 'info');
        addLog(`Intensity: ${knowledgeLearningIntensity.value}`, 'info');
        
        // Call backend API to start knowledge learning
        const response = await api.training.startKnowledgeLearning({
          model: knowledgeLearningModel.value,
          domains: knowledgeLearningDomains.value,
          priority: knowledgeLearningPriority.value,
          intensity: knowledgeLearningIntensity.value,
          max_time: knowledgeLearningMaxTime.value
        });
        
        if (response.data.status === 'success') {
          knowledgeLearningSuccessState.value = {
            hasSuccess: true,
            message: 'Knowledge learning started successfully'
          };
          addLog('Knowledge learning started successfully', 'success');
          
          // Start real progress polling
          startKnowledgeLearningProgress();
        } else {
          throw new Error(response.data.message || 'Failed to start knowledge learning');
        }
      } catch (error) {
        knowledgeLearningErrorState.value = {
          hasError: true,
          message: error.message || 'Failed to start knowledge learning',
          type: 'knowledge_learning'
        };
        addLog(`Failed to start knowledge learning: ${error.message}`, 'error');
        knowledgeLearningActive.value = false;
      }
    };
    
    const stopKnowledgeLearning = async () => {
      if (!knowledgeLearningActive.value) {
        addLog('No active knowledge learning session', 'warning');
        return;
      }
      
      try {
        addLog('Stopping knowledge learning...', 'info');
        
        // Call backend API to stop knowledge learning
        await api.training.stopKnowledgeLearning();
        
        knowledgeLearningActive.value = false;
        knowledgeLearningStatus.value = 'Stopped';
        knowledgeLearningSuccessState.value = {
          hasSuccess: true,
          message: 'Knowledge learning stopped successfully'
        };
        addLog('Knowledge learning stopped successfully', 'success');
        
        // Clear the polling interval
        if (knowledgeLearningPollInterval.value) {
          clearInterval(knowledgeLearningPollInterval.value);
          knowledgeLearningPollInterval.value = null;
        }
      } catch (error) {
        knowledgeLearningErrorState.value = {
          hasError: true,
          message: error.message || 'Failed to stop knowledge learning',
          type: 'knowledge_learning'
        };
        addLog(`Failed to stop knowledge learning: ${error.message}`, 'error');
      }
    };
    
    const startKnowledgeLearningProgress = () => {
      // Clear any existing interval
      if (knowledgeLearningPollInterval.value) {
        clearInterval(knowledgeLearningPollInterval.value);
        knowledgeLearningPollInterval.value = null;
      }
      
      knowledgeLearningPollInterval.value = setInterval(async () => {
        if (!knowledgeLearningActive.value) {
          clearInterval(knowledgeLearningPollInterval.value);
          knowledgeLearningPollInterval.value = null;
          return;
        }
        
        try {
          // Try to get progress from knowledge auto-learning API
          const progressResponse = await api.knowledge.autoLearning.progress();
          
          if (progressResponse.data.status === 'success') {
            const progressData = progressResponse.data;
            
            // Update progress based on real data
            if (progressData.progress !== undefined) {
              knowledgeLearningProgress.value = progressData.progress;
            }
            
            // Update status based on real data
            if (progressData.status) {
              knowledgeLearningStatus.value = progressData.status;
            }
            
            // Update logs with real messages
            if (progressData.logs && Array.isArray(progressData.logs)) {
              progressData.logs.forEach(log => {
                if (!knowledgeLearningLogs.value.some(existingLog => 
                  existingLog.timestamp === log.timestamp && existingLog.message === log.message)) {
                  addLog(log.message, log.level || 'info');
                  knowledgeLearningLogs.value.push({
                    timestamp: log.timestamp || new Date().toISOString(),
                    message: log.message,
                    level: log.level || 'info'
                  });
                }
              });
            }
            
            // Check if learning is completed
            if (progressData.completed) {
              knowledgeLearningActive.value = false;
              knowledgeLearningStatus.value = 'Completed';
              knowledgeLearningSuccessState.value = {
                hasSuccess: true,
                message: 'Knowledge learning completed successfully'
              };
              addLog('Knowledge learning completed!', 'success');
              clearInterval(knowledgeLearningPollInterval.value);
              knowledgeLearningPollInterval.value = null;
            }
          } else {
            // If API returns unsuccessful, show error and stop polling
            console.error('Knowledge learning progress API returned unsuccessful status:', progressResponse.data);
            knowledgeLearningActive.value = false;
            knowledgeLearningStatus.value = 'Failed';
            knowledgeLearningErrorState.value = {
              hasError: true,
              message: 'Knowledge learning progress API returned error',
              type: 'api_error'
            };
            addLog('Knowledge learning failed: API returned error', 'error');
            clearInterval(knowledgeLearningPollInterval.value);
            knowledgeLearningPollInterval.value = null;
          }
        } catch (error) {
          // If progress API fails, show error and stop polling
          console.error('Failed to get knowledge learning progress:', error);
          knowledgeLearningActive.value = false;
          knowledgeLearningStatus.value = 'Failed';
          knowledgeLearningErrorState.value = {
            hasError: true,
            message: `Failed to get knowledge learning progress: ${error.message}`,
            type: 'api_error'
          };
          addLog('Knowledge learning failed: API connection error', 'error');
          clearInterval(knowledgeLearningPollInterval.value);
          knowledgeLearningPollInterval.value = null;
        }
      }, 3000); // Poll every 3 seconds
    };
    
    // File upload reference
    const datasetInput = ref(null);
    
    // Timer reference
    let trainingTimer = null;
    let startTime = null;
    
    // Error state
    const errorState = ref({
      hasError: false,
      message: '',
      type: ''
    });
    
    // Success state
    const successState = ref({
      hasSuccess: false,
      message: ''
    });
    
    // Warning state
    const warningState = ref({
      hasWarning: false,
      message: ''
    });
    
    // Information state
    const infoState = ref({
      hasInfo: false,
      message: ''
    });
    
    // Toggle model selection
    const toggleModelSelection = (modelId) => {
      if (trainingMode.value === 'individual') {
        // Individual training mode can only select one model
        selectedModels.value = [modelId];
      } else {
        // Joint training mode allows multiple selections
        const index = selectedModels.value.indexOf(modelId);
        if (index > -1) {
          selectedModels.value.splice(index, 1);
        } else {
          selectedModels.value.push(modelId);
        }
      }
    };
    
    // Remove model from selection (for management panel)
    const removeModelFromSelection = (modelId) => {
      const index = selectedModels.value.indexOf(modelId);
      if (index > -1) {
        selectedModels.value.splice(index, 1);
        addLog(`Removed model ${getModelName(modelId)} from selection`);
        showInfo(`Model ${getModelName(modelId)} removed from selection`);
      }
    };
    
    // Add model to selection from quick select
    const addModelToSelection = () => {
      if (quickSelectModel.value && !selectedModels.value.includes(quickSelectModel.value)) {
        selectedModels.value.push(quickSelectModel.value);
        addLog(`Added model ${getModelName(quickSelectModel.value)} to selection`);
        showInfo(`Model ${getModelName(quickSelectModel.value)} added to selection`);
        quickSelectModel.value = ''; // Reset select
      }
    };
    
    // Clear all model selection
    const clearModelSelection = () => {
      selectedModels.value = [];
      addLog('Cleared all model selections');
      showInfo('All models deselected');
    };
    
    // Enhanced model selection functions
    const toggleBulkSelectMode = () => {
      bulkSelectMode.value = !bulkSelectMode.value;
      if (!bulkSelectMode.value) {
        addLog('Exited bulk selection mode');
        showInfo('Bulk selection mode deactivated');
      } else {
        addLog('Entered bulk selection mode');
        showInfo('Bulk selection mode activated. Select multiple models quickly.');
      }
    };
    
    const selectAllModels = () => {
      if (!availableModels.value || availableModels.value.length === 0) return;
      
      selectedModels.value = availableModels.value.map(model => model.id);
      addLog(`Selected all ${availableModels.value.length} models`);
      showInfo(`All ${availableModels.value.length} models selected`);
    };
    
    const selectModelsByCategory = (categoryId) => {
      if (!availableModels.value || availableModels.value.length === 0) return;
      
      // Use the same category mapping as filteredModels
      const categoryMap = {
        'language': ['B'],
        'vision': ['D', 'E', 'P'],
        'audio': ['C'],
        'sensor': ['G', 'Q'],
        'knowledge': ['J'],
        'planning': ['L', 'M', 'S'],
        'specialized': ['V', 'W', 'X', 'U', 'T', 'H', 'I', 'F', 'K', 'N']
      };
      
      const categoryModels = categoryMap[categoryId] || [];
      const modelsToSelect = availableModels.value
        .filter(model => categoryModels.includes(model.id))
        .map(model => model.id);
      
      // Add models to selection (avoid duplicates)
      modelsToSelect.forEach(modelId => {
        if (!selectedModels.value.includes(modelId)) {
          selectedModels.value.push(modelId);
        }
      });
      
      const categoryName = modelCategories.value.find(cat => cat.id === categoryId)?.name || categoryId;
      addLog(`Selected ${modelsToSelect.length} models from ${categoryName} category`);
      showInfo(`Added ${modelsToSelect.length} ${categoryName} models to selection`);
    };
    
    const showModelDetail = (modelId) => {
      const model = availableModels.value.find(m => m.id === modelId);
      if (model) {
        selectedModelDetails.value = {
          id: model.id,
          name: model.name,
          description: model.description,
          backendId: model.backendId,
          type: model.type,
          status: model.status
        };
        showModelDetails.value = true;
      }
    };
    
    const closeModelDetails = () => {
      showModelDetails.value = false;
      selectedModelDetails.value = null;
    };
    
    const getModelCategory = (modelId) => {
      const categoryMap = {
        'language': ['B'],
        'vision': ['D', 'E', 'P'],
        'audio': ['C'],
        'sensor': ['G', 'Q'],
        'knowledge': ['J'],
        'planning': ['L', 'M', 'S'],
        'specialized': ['V', 'W', 'X', 'U', 'T', 'H', 'I', 'F', 'K', 'N']
      };
      
      for (const [category, models] of Object.entries(categoryMap)) {
        if (models.includes(modelId)) {
          return modelCategories.value.find(cat => cat.id === category)?.name || category;
        }
      }
      return 'Other';
    };
    
    // Open upload dialog
    const openUploadDialog = () => {
      datasetInput.value.click();
    };
    
    // Handle dataset upload
    const handleDatasetUpload = async (event) => {
      const files = event.target.files;
      if (files.length === 0) return;
      
      try {
        addLog(`Uploading dataset: ${files[0].name}`);
        
        // Create FormData
        const formData = new FormData();
        formData.append('file', files[0]);
        
        // Call FastAPI backend dataset upload API
        const response = await api.datasets.upload(formData, {
          timeout: 30000 // 30 second timeout
        });
        
        // Add newly uploaded dataset
        const newDataset = {
          id: response.data.dataset_id,
          name: response.data.dataset_name
        };
        
        datasets.value.push(newDataset);
        selectedDataset.value = newDataset.id;
        
        // Show success message
        addLog(`Dataset upload successful: ${newDataset.name}`);
        showInfo('Dataset uploaded successfully');
      } catch (error) {
        addLog(`Dataset upload failed: ${error.message || 'Unknown error'}`);
        showError('Failed to upload dataset. Please try again.');
      }
    };
    
    
    
    // Helper function to convert camelCase to snake_case
    const toSnakeCase = (obj) => {
      if (!obj || typeof obj !== 'object') return obj;
      
      const result = {};
      for (const [key, value] of Object.entries(obj)) {
        // Convert key from camelCase to snake_case
        const snakeKey = key.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
        // Recursively process nested objects
        result[snakeKey] = typeof value === 'object' && value !== null ? toSnakeCase(value) : value;
      }
      return result;
    };
    
    // Start training
    const startTraining = async () => {
      if (selectedModels.value.length === 0) {
        addLog('Please select at least one model to train');
        return;
      }
      
      try {
        isTraining.value = true;
        trainingProgress.value = 0;
        currentEpoch.value = 0;
        currentLoss.value = 0;
        currentAccuracy.value = 0;
        trainingLogs.value = [];
        evaluationResults.value = null;
        
        // Reset timer
        startTime = new Date();
        updateElapsedTime();
        trainingTimer = setInterval(updateElapsedTime, 1000);
        
        // Add start log
        addLog(`Training started in ${trainingMode.value} mode with models: ${selectedModels.value.map(m => m.charAt(0).toUpperCase() + m.slice(1)).join(', ')} using dataset: ${datasets.value.find(d => d.id === selectedDataset.value).name}`);
        
        // Prepare training request data according to backend API schema
        // Extract parameters that need to be sent to backend
        const { 
          useExternalModelAssistance, 
          externalModelId, 
          pretrainedModelId,
          freezeLayers,
          freezeLayerCount,
          fineTuningMode,
          ...trainingParams 
        } = parameters.value;
        
        // Convert training parameters to snake_case for backend compatibility
        const backendParameters = toSnakeCase(trainingParams);
        
        // Build training data according to backend TrainingRequest model
        const trainingData = {
          mode: trainingMode.value,
          models: selectedModels.value.map(modelId => {
            const model = availableModels.value.find(m => m.id === modelId);
            return model ? model.backendId : modelId;
          }),
          dataset: selectedDataset.value,
          parameters: backendParameters,
          from_scratch: parameters.value.fromScratch || false,
          device: selectedDevice.value,
          external_model_assistance: parameters.value.useExternalModelAssistance || false,
          external_model_id: parameters.value.externalModelId
        };
        
        // Call FastAPI backend training start API
        try {
          // Always use the same start API endpoint (now supports external_model_assistance)
          const response = await api.training.start(trainingData);
          
          currentJobId.value = response.data.training_id || response.data.job_id;
          addLog(`Training job created with ID: ${currentJobId.value}`);
          
          // Start WebSocket connection for real-time updates
          startWebSocketConnection(currentJobId.value);
        } catch (apiError) {
          // If API call fails, show error and stop training
          addLog(`Failed to connect to server: ${apiError.message || 'Unknown error'}`);
          addLog('Please ensure the backend service is running on the correct port');
          
          // Reset training state
          isTraining.value = false;
          currentJobId.value = null;
          clearInterval(trainingTimer);
          
          showError('Failed to start training: Backend service unavailable');
        }
      } catch (error) {
        addLog(`Failed to start training: ${error.message || 'Unknown error'}`);
        showError('Failed to start training. Please check your connection and try again.');
        
        // Reset training state
        isTraining.value = false;
        currentJobId.value = null;
      }
    };

    // Start WebSocket connection (enhanced version with reconnection logic and status monitoring)
    const startWebSocketConnection = (jobId) => {
      try {
        // Connect to main server WebSocket endpoint for training updates
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Use the same host as the API, default to port 8000 for backend
        const host = window.location.hostname;
        // Use window.location.port for development (Vite proxy) or default to 8000
        const port = window.location.port || '8000';
        const wsUrl = `${wsProtocol}://${host}:${port}/ws/training/${jobId}`;
        
        // Add connection attempt log
        addLog('Connecting to WebSocket: ' + wsUrl.replace(/^(wss?:\/\/[^/]+).*/, '$1/...'));
        
        websocketConnection.value = new WebSocket(wsUrl);
        
        // Connection state
        let connectionAttempts = 0;
        const maxReconnectAttempts = 3;
        let reconnectTimeout = null;
        let heartbeatInterval = null;
        
        websocketConnection.value.onopen = () => {
          connectionAttempts = 0;
          addLog('WebSocket connected successfully', 'success');
          showInfo('Real-time updates enabled');
          
          // Start heartbeat to keep connection alive
          heartbeatInterval = setInterval(() => {
            if (websocketConnection.value && websocketConnection.value.readyState === WebSocket.OPEN) {
              try {
                websocketConnection.value.send(JSON.stringify({type: 'ping', timestamp: Date.now()}));
              } catch (error) {
                console.warn('WebSocket heartbeat failed:', error);
              }
            }
          }, 30000); // 30 seconds
        };
        
        websocketConnection.value.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // Handle different message types
            switch (data.type) {
              case 'progress':
                trainingProgress.value = data.progress;
                currentEpoch.value = data.epoch;
                currentLoss.value = data.loss;
                currentAccuracy.value = data.accuracy;
                
                // If there are additional training metrics, update them too
                if (data.validation_loss !== undefined) {
                  validationLoss.value = data.validation_loss;
                }
                if (data.validation_accuracy !== undefined) {
                  validationAccuracy.value = data.validation_accuracy;
                }
                break;
              
              case 'log':
                const logType = data.level || 'info';
                addLog(data.message, logType);
                break;
              
              case 'error':
                addLog(`ERROR: ${data.message}`, 'error');
                showError(data.message, 'training');
                break;
              
              case 'completed':
                trainingProgress.value = 100;
                evaluationResults.value = data.evaluation;
                completeTraining();
                break;
              
              case 'system':
                // System messages, usually not displayed to users
                if (import.meta.env.DEV) {
                  console.log('System message:', data);
                }
                break;
              
              default:
                addLog('Unknown WebSocket message type: ' + data.type, 'warning');
            }
          } catch (error) {
            handleApiError(error, 'WebSocket message parse error');
            addLog('WebSocket message parse error: ' + error.message, 'error');
          }
        };
        
        websocketConnection.value.onerror = (error) => {
          const errorMessage = error.message || 'Unknown error';
          addLog('WebSocket connection error: ' + errorMessage, 'error');
          handleApiError(error, 'WebSocket connection error');
          
                // Attempt reconnection
                if (connectionAttempts < maxReconnectAttempts && isTraining.value) {
            connectionAttempts++;
            const delay = Math.pow(2, connectionAttempts) * 1000;
            addLog(`Reconnecting to WebSocket (attempt ${connectionAttempts}/${maxReconnectAttempts}, delay ${delay/1000}s)`, 'warning');
            
            reconnectTimeout = setTimeout(() => {
              if (isMounted.value && isTraining.value) {
                startWebSocketConnection(jobId);
              }
            }, delay);
          } else if (connectionAttempts >= maxReconnectAttempts) {
            addLog(`Max WebSocket reconnection attempts (${maxReconnectAttempts}) reached`, 'error');
            showError('Falling back to polling mode');
            
                // If reconnection fails, switch to polling mode
                if (!statusPollingInterval.value) {
              startStatusPolling(jobId);
            }
          }
        };
        
        websocketConnection.value.onclose = (event) => {
          const code = event.code || 'unknown';
          const reason = event.reason || '';
          
          if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
          }
          
          // Normal closure does not need to show error
          if (code === 1000 || code === 1001) {
            addLog('WebSocket connection closed normally', 'info');
          } else {
            addLog(`WebSocket connection closed unexpectedly (code: ${code}, reason: ${reason})`, 'warning');
          }
        };
        
        // Set up timeout detection
        const timeoutId = setTimeout(() => {
          if (isMounted.value && websocketConnection.value && websocketConnection.value.readyState !== WebSocket.OPEN) {
            addLog('WebSocket connection timeout', 'error');
            showError('WebSocket connection timeout');
          }
        }, 5000);
        
        // Clear timeout when connection is successful
        websocketConnection.value.onopen = function() {
          clearTimeout(timeoutId);
          connectionAttempts = 0;
          addLog('WebSocket connected successfully', 'success');
          showInfo('Real-time updates enabled');
        };
      } catch (error) {
        addLog(`WebSocket connection failed: ${error.message}`, 'error');
        
        // Connection failed, directly switch to polling mode
        if (!statusPollingInterval.value) {
          startStatusPolling(jobId);
        }
      }
    };

    // Start status polling (enhanced version with error handling and adaptive polling interval)
    const startStatusPolling = (jobId) => {
      // Avoid duplicate polling startup
      if (statusPollingInterval.value) {
        clearInterval(statusPollingInterval.value);
      }
      
      let pollingInterval = 2000; // Initial 2 second interval
      let consecutiveFailures = 0;
      const maxFailures = 3;
      
      addLog(`Starting polling mode (interval: ${pollingInterval/1000}s)`, 'info');
      
      statusPollingInterval.value = setInterval(async () => {
        try {
          const response = await api.training.status(jobId);
          
          const status = response.data;
          
          // Reset failure counter
          consecutiveFailures = 0;
          
          // Adaptive polling interval based on training progress
          if (status.progress > 90) {
            pollingInterval = 1000; // Poll more frequently when nearing completion
          } else if (status.progress > 50) {
            pollingInterval = 1500;
          } else {
            pollingInterval = 2000;
          }
          
          // Update training status
          if (status.progress !== undefined) {
            trainingProgress.value = status.progress;
          }
          if (status.current_epoch !== undefined) {
            currentEpoch.value = status.current_epoch;
          }
          if (status.loss !== undefined) {
            currentLoss.value = status.loss;
          }
          if (status.accuracy !== undefined) {
            currentAccuracy.value = status.accuracy;
          }
          
          // Update other possible metrics
          if (status.validation_loss !== undefined) {
            validationLoss.value = status.validation_loss;
          }
          if (status.validation_accuracy !== undefined) {
            validationAccuracy.value = status.validation_accuracy;
          }
          
          // If there are log messages, add them to logs too
          if (status.logs && status.logs.length > 0) {
            status.logs.forEach(log => {
              addLog(log.message, log.level || 'info');
            });
          }
          
          // If training is complete, stop polling
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
            clearInterval(statusPollingInterval.value);
            
            if (status.status === 'completed') {
              evaluationResults.value = status.evaluation;
              completeTraining();
            } else if (status.status === 'failed') {
              addLog('Training failed: ' + (status.error || 'Unknown reason'), 'error');
              showError('Training failed: ' + (status.error || 'Unknown reason'));
              stopTraining();
            } else if (status.status === 'stopped') {
              addLog('Training stopped by server', 'info');
              stopTraining();
            }
          }
          
        } catch (error) {
          consecutiveFailures++;
          
          // Log error but continue trying
            if (consecutiveFailures <= maxFailures) {
              addLog('Polling error (attempt ' + consecutiveFailures + '/' + maxFailures + '): ' + error.message, 'warning');
              
              // Increase polling interval on failure
              pollingInterval = Math.min(10000, pollingInterval * 1.5);
            } else {
              // Exceeded maximum failures, show error and stop training
              addLog('Maximum polling failures reached (' + maxFailures + ')', 'error');
              addLog('Failed to communicate with server', 'error');
              
              clearInterval(statusPollingInterval.value);
              
              // If training is still in progress, stop training
              if (isTraining.value) {
                showError('Training stopped due to connection issues');
                stopTraining();
              }
            }
        }
      }, pollingInterval);
    };
    
    
    // Complete training (enhanced version with detailed training summary)
    const completeTraining = () => {
      clearInterval(trainingTimer);
      
      // Add to training history - save frontend letter IDs for display
      const duration = (new Date() - startTime) / 1000;
      const accuracy = evaluationResults.value ? evaluationResults.value.accuracy : 0;
      const loss = evaluationResults.value ? evaluationResults.value.loss : 0;
      
      // Add detailed training summary
      addLog('====================================================');
      addLog('Training complete summary:', {
        models: selectedModels.value.map(m => m.charAt(0).toUpperCase() + m.slice(1)).join(', '),
        dataset: datasets.value.find(d => d.id === selectedDataset.value).name,
        duration: formatDuration(duration)
      });
      addLog('Final metrics: Accuracy: ' + accuracy.toFixed(2) + '%, Loss: ' + loss.toFixed(4));
      
      // If there are detailed evaluation results, show more information
        if (evaluationResults.value) {
          if (evaluationResults.value.precision !== undefined) {
            addLog('Precision: ' + (evaluationResults.value.precision * 100).toFixed(2) + '%');
          }
          if (evaluationResults.value.recall !== undefined) {
            addLog('Recall: ' + (evaluationResults.value.recall * 100).toFixed(2) + '%');
          }
          if (evaluationResults.value.f1Score !== undefined) {
            addLog('F1 Score: ' + (evaluationResults.value.f1Score * 100).toFixed(2) + '%');
          }
        }
        
        // Calculate training efficiency metrics
        const epochs = parameters.value.epochs;
        const efficiency = accuracy / (epochs * duration / 3600); // Accuracy improvement per hour per epoch
        const efficiencyRating = efficiency > 50 ? 'excellent' : efficiency > 30 ? 'good' : efficiency > 15 ? 'satisfactory' : 'room for improvement';
        addLog('Training efficiency: ' + efficiencyRating.charAt(0).toUpperCase() + efficiencyRating.slice(1) + ' (' + efficiency.toFixed(2) + ')');
      
      addLog('====================================================');
      
      // Update training history
      trainingHistory.value.unshift({
        id: Date.now(),
        date: new Date(),
        models: [...selectedModels.value], // Save frontend letter IDs for display
        dataset: datasets.value.find(d => d.id === selectedDataset.value).name,
        duration: duration,
        accuracy: accuracy,
        loss: loss,
        parameters: { ...parameters.value },
        strategy: selectedStrategy.value
      });
      
      // Reload training history to ensure synchronization with backend
      loadTrainingHistory();
      
      // Show success message
      showSuccess('Training completed successfully');
    };
    
    // Stop training
    const stopTraining = async () => {
      if (!isTraining.value) {
        addLog('No active training session');
        return;
      }
      
      // Confirm with user before stopping
      if (!confirm('Are you sure you want to stop the current training? This action cannot be undone.')) {
        addLog('Training stop cancelled by user');
        return;
      }

      try {
        // Try to stop training via API
        if (currentJobId.value) {
          await api.training.stop(currentJobId.value);
          addLog('Training stop request sent to server');
        }
      } catch (error) {
        addLog(`Failed to send stop request to server: ${error.message || 'Unknown error'}`);
        showWarning('Failed to communicate with server, stopping locally');
      } finally {
        // Local cleanup of training state
        isTraining.value = false;
        clearInterval(trainingTimer);
        clearInterval(statusPollingInterval.value);
        
        // Close WebSocket connection
        if (websocketConnection.value) {
          websocketConnection.value.close();
          websocketConnection.value = null;
        }
        
        currentJobId.value = null;
        addLog('Training stopped');
        showInfo('Training has been stopped');
      }
    };
    
    // Update elapsed time
    const updateElapsedTime = () => {
      if (!startTime) return;
      
      const now = new Date();
      const diff = now - startTime;
      const hours = Math.floor(diff / 3600000);
      const minutes = Math.floor((diff % 3600000) / 60000);
      const seconds = Math.floor((diff % 60000) / 1000);
      
      elapsedTime.value = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };
    
    // Show error message
    const showError = (message, type = 'general') => {
      errorState.value = {
        hasError: true,
        message,
        type
      };
      
      // Auto-clear error after 5 seconds
      setTimeout(() => {
        if (isMounted.value) {
          errorState.value.hasError = false;
        }
      }, 5000);
      
      addLog(`ERROR: ${message}`);
    };
    
    // Load training history
    const loadTrainingHistory = async () => {
      try {
        // Call FastAPI backend to get training history
        const response = await api.training.history();
        
        // Strictly check response data structure
        if (response && response.data && Array.isArray(response.data.history)) {
          // Process backend returned history data
          trainingHistory.value = response.data.history.map(item => ({
            id: item.id,
            date: new Date(item.timestamp),
            models: item.models,
            dataset: item.dataset_name,
            duration: item.duration,
            accuracy: item.metrics ? (item.metrics.accuracy * 100 || 0) : 0,
            loss: item.metrics ? item.metrics.loss : 0,
            parameters: item.parameters || {},
            strategy: item.strategy || 'Unknown'
          }));
          
          showInfo('Training history loaded successfully');
        } else {
          // When response format is incorrect, show error
          console.warn('Training history response format incorrect');
          showError('Failed to load training history: Invalid response format from backend');
          trainingHistory.value = [];
        }
      } catch (error) {
        console.error('Failed to load training history:', error);
        // If API doesn't exist or backend is unavailable, show error
        showError('Failed to load training history from backend. Please ensure the backend service is running.');
        trainingHistory.value = [];
      }
    };
    
    // Add log (enhanced version, supports different log types and formats)
    const addLog = (message, type = 'info') => {
      const timestamp = new Date().toLocaleTimeString();
      
      // Add log entry with type information for styling
      trainingLogs.value.push({
        timestamp,
        message,
        type // info, success, warning, error, debug
      });
      
      // Keep logs scrolled to bottom, optimize scrolling performance
      nextTick(() => {
        const logContainer = document.getElementById('training-logs');
        if (logContainer) {
          // Use requestAnimationFrame to ensure smooth scrolling
          requestAnimationFrame(() => {
            logContainer.scrollTop = logContainer.scrollHeight;
          });
        }
      });
    };
    
    // Device selection methods
    const detectAvailableDevices = async () => {
      try {
        // Use real backend API to get available devices
        const response = await api.training.availableDevices();
        
        if (response.data.status === 'success') {
          // Map backend device data to frontend format with priority
          const devicePriorityMap = {
            'cpu': 1,
            'cuda': 2,
            'mps': 1,
            'auto': 0
          };
          
          const mappedDevices = (response.data.devices || []).map(device => ({
            id: device.id,
            name: device.name,
            available: device.available,
            priority: devicePriorityMap[device.id] || 0
          }));
          
          // Update available devices
          availableDevices.value = mappedDevices;
          
          // Auto-select CPU if no device is selected and CPU is available
          if (!selectedDevice.value && mappedDevices.find(d => d.id === 'cpu' && d.available)) {
            selectedDevice.value = 'cpu';
          }
          
          return mappedDevices;
        } else {
          throw new Error('API response unsuccessful');
        }
      } catch (error) {
        console.error('Error detecting available devices:', error);
        // Show error and return empty devices list
        showError(`Failed to detect available devices: ${error.message}`);
        availableDevices.value = [];
        return availableDevices.value;
      }
    };
    
    const selectDevice = async (deviceId) => {
      if (!availableDevices.value.find(d => d.id === deviceId)?.available) {
        showError(`Device ${deviceId} is not available on this system`);
        return;
      }
      
      selectedDevice.value = deviceId;
      addLog(`Selected training device: ${deviceId}`);
      
      // Optimize training parameters for selected device
      await optimizeForDevice(deviceId);
    };
    
    const switchTrainingDevice = async (newDevice) => {
      if (!isTraining.value) {
        showError('Cannot switch device: No active training session');
        return;
      }
      
      if (selectedDevice.value === newDevice) {
        showInfo(`Already using ${newDevice} device`);
        return;
      }
      
      if (!availableDevices.value.find(d => d.id === newDevice)?.available) {
        showError(`Device ${newDevice} is not available on this system`);
        return;
      }
      
      try {
        addLog(`Switching training device from ${selectedDevice.value} to ${newDevice}...`);
        
        // Send device switch request to backend
        const response = await api.training.switchDevice(currentJobId.value, newDevice);
        
        if (response.data.success) {
          selectedDevice.value = newDevice;
          addLog(`Device switched successfully to ${newDevice}`);
          showSuccess(`Training device switched to ${newDevice}`);
          
          // Optimize parameters for new device
          await optimizeForDevice(newDevice);
        } else {
          showError(`Failed to switch device: ${response.data.message}`);
        }
      } catch (error) {
        addLog(`Device switch failed: ${error.message}`);
        showError('Failed to switch training device');
      }
    };
    
    const optimizeForDevice = async (deviceId) => {
      const optimizations = {
        cpu: {
          batchSize: 16,
          mixedPrecision: false,
          gradientAccumulation: 1
        },
        cuda: {
          batchSize: 64,
          mixedPrecision: true,
          gradientAccumulation: 4
        },
        mps: {
          batchSize: 32,
          mixedPrecision: true,
          gradientAccumulation: 2
        },
        auto: {
          batchSize: 32,
          mixedPrecision: true,
          gradientAccumulation: 2
        }
      };
      
      const optimization = optimizations[deviceId] || optimizations.auto;
      
      // Update parameters
      parameters.value.batchSize = optimization.batchSize;
      
      addLog(`Optimized training parameters for ${deviceId} device`);
    };
    
    // Initialize

    
    // Initialize when component is mounted
    onMounted(async () => {
      isMounted.value = true;
      await loadAvailableModels();
      await loadModelDependencies();
      await loadDatasets();
      await loadRecommendedCombinations();
      await loadAvailableDevices();
      await loadTrainingHistory();
      await refreshManagement();
      
      // Set default selected models
      if (availableModels.value.length > 0) {
        // Ensure default selected models are valid
        selectedModels.value = selectedModels.value.filter(modelId => 
          availableModels.value.some(m => m.id === modelId)
        );
        
        // If no valid selected models, select the first one
        if (selectedModels.value.length === 0 && availableModels.value.length > 0) {
          selectedModels.value = [availableModels.value[0].id];
        }
      }
      
      // Initialize and start resource monitoring chart
      nextTick(() => {
        initializeResourceChart();
        startResourceMonitoring();
      });
    });
    
    // Cleanup when component is unmounted
    onUnmounted(() => {
      isMounted.value = false;
      clearInterval(trainingTimer);
      clearInterval(statusPollingInterval.value);
      if (knowledgeLearningPollInterval.value) {
        clearInterval(knowledgeLearningPollInterval.value);
      }
      
      if (websocketConnection.value) {
        websocketConnection.value.close();
      }
      
      // Stop resource monitoring and destroy chart
      stopResourceMonitoring();
      if (resourceChart.value) {
        resourceChart.value.destroy();
        resourceChart.value = null;
      }
    });
    
    // Training management functions
    const refreshManagement = async () => {
      addLog('Refreshing training management data...');
      await loadTrainingHistory();
      await loadAvailableDevices();
      await checkResourceAvailability(); // Add resource monitoring
      // Load external model stats
      try {
        const response = await api.training.externalModelStats();
        if (response.data.status === 'success') {
          externalModelStats.value = response.data.stats || [];
          // Update available external models from stats
          availableExternalModels.value = externalModelStats.value.map(stat => ({
            id: stat.model.toLowerCase().replace(/\s+/g, '-'),
            name: stat.model,
            available: stat.count > 0  // Consider available if there are usage records
          }));
        }
      } catch (error) {
        console.warn('Failed to load external model stats:', error);
        availableExternalModels.value = [];
      }
      showInfo('Management data refreshed');
    };
    
    // Initialize resource monitoring chart
    const initializeResourceChart = () => {
      try {
        // Register Chart.js components
        Chart.register(...registerables);
        
        // Get canvas context
        const ctx = document.getElementById('resourceChartCanvas');
        if (!ctx) {
          console.warn('Resource chart canvas not found');
          return;
        }
        
        // Create chart instance
        resourceChart.value = new Chart(ctx, {
          type: 'line',
          data: {
            labels: resourceHistory.value.timestamps,
            datasets: [
              {
                label: 'CPU Usage (%)',
                data: resourceHistory.value.cpu,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.4,
                fill: true
              },
              {
                label: 'Memory Usage (%)',
                data: resourceHistory.value.memory,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.4,
                fill: true
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: 'Resource Usage Monitoring'
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                title: {
                  display: true,
                  text: 'Usage (%)'
                }
              },
              x: {
                title: {
                  display: true,
                  text: 'Time'
                }
              }
            }
          }
        });
        
        addLog('Resource monitoring chart initialized');
      } catch (error) {
        console.error('Failed to initialize resource chart:', error);
      }
    };
    
    // Update resource chart with new data
    const updateResourceChart = (cpuValue, memoryValue) => {
      if (!resourceChart.value) return;
      
      const now = new Date();
      const timeLabel = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      
      // Add new data point
      resourceHistory.value.timestamps.push(timeLabel);
      resourceHistory.value.cpu.push(cpuValue);
      resourceHistory.value.memory.push(memoryValue);
      
      // Keep only maxHistoryPoints data points
      if (resourceHistory.value.timestamps.length > maxHistoryPoints) {
        resourceHistory.value.timestamps.shift();
        resourceHistory.value.cpu.shift();
        resourceHistory.value.memory.shift();
      }
      
      // Update chart
      resourceChart.value.data.labels = resourceHistory.value.timestamps;
      resourceChart.value.data.datasets[0].data = resourceHistory.value.cpu;
      resourceChart.value.data.datasets[1].data = resourceHistory.value.memory;
      resourceChart.value.update('none');
    };
    
    // Start resource monitoring updates
    const startResourceMonitoring = () => {
      if (resourceUpdateInterval.value) {
        clearInterval(resourceUpdateInterval.value);
      }
      
      // Initial update
      checkResourceAvailability();
      
      // Set up periodic updates every 5 seconds
      resourceUpdateInterval.value = setInterval(() => {
        checkResourceAvailability();
      }, 5000);
    };
    
    // Stop resource monitoring
    const stopResourceMonitoring = () => {
      if (resourceUpdateInterval.value) {
        clearInterval(resourceUpdateInterval.value);
        resourceUpdateInterval.value = null;
      }
    };
    
    const checkResourceAvailability = async () => {
      addLog('Checking resource availability...');
      try {
        const response = await api.system.stats();
        if (response.data.status === 'success') {
          const statsData = response.data.data || {};
          
          // Extract numeric values for chart
          const cpuValue = typeof statsData.cpu_usage === 'number' ? statsData.cpu_usage : 
                          (typeof statsData.cpu_usage === 'string' ? parseFloat(statsData.cpu_usage) : 0);
          const memoryValue = typeof statsData.memory_usage === 'number' ? statsData.memory_usage :
                             (typeof statsData.memory_usage === 'string' ? parseFloat(statsData.memory_usage) : 0);
          
          // Update text display values
          memoryUsage.value = `${isNaN(memoryValue) ? 'Unknown' : memoryValue.toFixed(1)}%`;
          cpuUsage.value = `${isNaN(cpuValue) ? 'Unknown' : cpuValue.toFixed(1)}%`;
          
          // Update chart with valid numeric values
          if (!isNaN(cpuValue) && !isNaN(memoryValue)) {
            updateResourceChart(cpuValue, memoryValue);
          }
        }
      } catch (error) {
        console.warn('Failed to load resource usage:', error);
        memoryUsage.value = 'Monitoring unavailable';
        cpuUsage.value = 'Monitoring unavailable';
      }
      showInfo('Resource availability updated');
    };
    
    const stopTask = async (taskId) => {
      if (!confirm('Are you sure you want to stop this training task?')) return;
      addLog(`Stopping training task: ${taskId}`);
      try {
        await api.training.stop(taskId);
        activeTrainingTasks.value = activeTrainingTasks.value.filter(task => task.id !== taskId);
        showSuccess('Training task stopped successfully');
      } catch (error) {
        showError(`Failed to stop task: ${error.message}`);
      }
    };
    
    const clearTrainingHistory = () => {
      if (!confirm('Are you sure you want to clear all training history? This action cannot be undone.')) return;
      trainingHistory.value = [];
      addLog('Training history cleared');
      showInfo('Training history cleared');
    };
    
    const exportTrainingData = () => {
      addLog('Exporting training data...');
      const data = {
        trainingHistory: trainingHistory.value,
        externalModelStats: externalModelStats.value,
        exportDate: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training-data-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showSuccess('Training data exported successfully');
    };
    
    // Export functionality
return {
      // State
      trainingMode,
      availableModels,
      modelsLoading,
      modelsError,
      selectedModels,
      datasets,
      recommendedCombinations,
      combinationValid,
      validationMessage,
      currentDependencies,
      selectedDataset,
      availableDevices,
      selectedDevice,
      parameters,
      trainingStrategies,
      selectedStrategy,
      enableKnowledgeAssist,
      knowledgeAssistOptions,
      isTraining,
      trainingProgress,
      currentEpoch,
      currentLoss,
      currentAccuracy,
      elapsedTime,
      trainingLogs,
      evaluationResults,
      validationLoss,
      validationAccuracy,
      trainingHistory,
      comparingSessions,
      datasetInput,
      errorState,
      knowledgeLearningErrorState,
      successState,
      warningState,
      infoState,
      externalModelStats,
      availableExternalModels,
      memoryUsage,
      cpuUsage,
      activeTrainingTasks,
      
      // Enhanced model selection state
      modelSearchQuery,
      selectedModelCategory,
      modelCategories,
      showModelDetails,
      selectedModelDetails,
      bulkSelectMode,
      
      // Computed properties
      isModelRequired,
      isModelDisabled,
      getModelTooltip,
      successfulTrainings,
      averageAccuracy,
      totalTrainingTime,
      
      // Enhanced model selection computed properties
      filteredModels,
      isAllModelsSelected,
      isSomeModelsSelected,
      
      // Methods
      toggleModelSelection,
      selectRecommendedCombination,
      openUploadDialog,
      handleDatasetUpload,
      startTraining,
      stopTraining,
      loadAvailableModels,
      loadTrainingHistory,
      selectDevice,
      switchTrainingDevice,
      detectAvailableDevices,
      showError,
      showWarning,
      showSuccess,
      showInfo,
      addLog,
      refreshManagement,
      checkResourceAvailability,
      stopTask,
      clearTrainingHistory,
      exportTrainingData,
      getModelName,
      removeModelFromSelection,
      addModelToSelection,
      clearModelSelection,
      
      // Enhanced model selection methods
      toggleBulkSelectMode,
      selectAllModels,
      selectModelsByCategory,
      showModelDetail,
      closeModelDetails,
      getModelCategory,
      
      // Utility functions
      formatDate(date) {
        return new Date(date).toLocaleDateString();
      },
      formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours}h ${minutes}m ${secs}s`;
      },
      
      // View training session details
      viewSession(id) {
        addLog(`Viewing session details: ${id}`);
        try {
          // Find corresponding session
          const session = trainingHistory.value.find(s => s.id === id);
          if (session) {
            // In actual application, this should open a details modal
            // But for quick fix, we can simply show an alert with session info
            const sessionDetails = `Session ID: ${session.id}\nDate: ${this.formatDate(session.date)}\nModels: ${session.models.join(', ')}\nDataset: ${session.dataset}\nDuration: ${this.formatDuration(session.duration)}\nAccuracy: ${session.accuracy}%`;
            notify.info(sessionDetails);
          }
        } catch (error) {
          console.error('Error viewing session:', error);
          showError('Failed to view session details');
        }
      },
      // Compare training sessions
      compareSession(id) {
        addLog(`Comparing session: ${id}`);
        try {
          // Find corresponding session
          const session = trainingHistory.value.find(s => s.id === id);
          if (session) {
            // Check if session is already selected
            const sessionIndex = comparingSessions.value.findIndex(s => s.id === id);
            
            if (sessionIndex === -1) {
              // Add to comparison list
              comparingSessions.value.push(session);
              addLog(`Added session ${id} to comparison`);
              
              // If two or more sessions are already selected, perform simple comparison
              if (comparingSessions.value.length >= 2) {
                // Find best performing session
                let bestSession = comparingSessions.value[0];
                comparingSessions.value.forEach(s => {
                  if (s.accuracy > bestSession.accuracy) {
                    bestSession = s;
                  } else if (s.accuracy === bestSession.accuracy && s.loss < bestSession.loss) {
                    bestSession = s;
                  }
                });
                
                // Show comparison results
                addLog(`=== Session Comparison Results ===`);
                addLog(`Best performing session: ${bestSession.id} with ${bestSession.accuracy.toFixed(2)}% accuracy`);
                addLog(`Sessions compared: ${comparingSessions.value.length}`);
                
                // Clear comparison list
                comparingSessions.value = [];
                showInfo('Session comparison completed. Check logs for results.');
              } else if (comparingSessions.value.length === 1) {
                showInfo('Select another session to compare.');
              }
            }
          } else {
            showError('Session not found');
          }
        } catch (error) {
          console.error('Error comparing session:', error);
          showError('Failed to compare sessions');
        }
      }
    };
  },
  components: {
    TerminalWindow
  }
};
</script>

<style scoped>
/* CSS Variables for Monochrome Style */
:root {
  --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  --text-primary: #333333;
  --text-secondary: #666666;
  --text-light: #999999;
  --bg-primary: #ffffff;
  --bg-secondary: #f9f9f9;
  --bg-hover: #f0f0f0;
  --border-color: #dddddd;
  --primary-color: #444444;
  --primary-color-light: #f0f0f0;
  --primary-color-dark: #333333;
  --error-bg: #f8f8f8;
  --error-bg-light: #fafafa;
  --error-text: #666666;
  --error-color: #666666;
  --error-color-dark: #555555;
  --error-border: #eeeeee;
  --success-bg: #f8f8f8;
  --success-bg-light: #fafafa;
  --success-text: #666666;
  --success-border: #eeeeee;
  --warning-bg: #f8f8f8;
  --warning-bg-light: #fafafa;
  --warning-text: #666666;
  --warning-color: #666666;
  --info-bg: #f8f8f8;
  --info-text: #666666;
  --info-border: #eeeeee;
  --disabled-bg: #eeeeee;
}



.status-messages {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  max-width: 400px;
}

.message {
  padding: 12px 16px;
  margin-bottom: 10px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  animation: slideIn 0.3s ease-out;
}

.message.error {
  background-color: var(--error-bg);
  color: var(--error-text);
  border: 1px solid var(--error-border);
}

.message.success {
  background-color: var(--success-bg);
  color: var(--success-text);
  border: 1px solid var(--success-border);
}

.message.warning {
  background-color: var(--warning-bg);
  color: var(--warning-text);
  border: 1px solid var(--warning-border);
}

.message.info {
  background-color: var(--info-bg);
  color: var(--info-text);
  border: 1px solid var(--info-border);
}

.icon {
  margin-right: 8px;
  font-size: 18px;
}

.control-panel {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin-bottom: 30px;
}

.mode-selection,
.model-selection,
.dataset-selection,
.parameter-settings,
.strategy-selection {
  background-color: var(--bg-secondary);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

h1 {
  font-size: 28px;
  margin-bottom: 20px;
  color: var(--text-primary);
}

h2 {
  font-size: 20px;
  margin-bottom: 15px;
  color: var(--text-primary);
}

h3 {
  font-size: 16px;
  margin-bottom: 10px;
  color: var(--text-secondary);
}

.mode-options {
  display: flex;
  gap: 10px;
}

.mode-options button,
.strategy-option {
  padding: 10px 15px;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.mode-options button:hover,
.strategy-option:hover {
  background-color: var(--bg-hover);
}

.mode-options button.active,
.strategy-option.selected {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 15px;
}

.model-option {
  padding: 10px;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.model-option:hover {
  background-color: var(--bg-hover);
}

.model-option.selected {
  background-color: var(--primary-color-light);
  border-color: var(--primary-color);
}

.model-option.required {
  border-color: var(--warning-color);
  background-color: var(--warning-bg-light);
}

.model-option.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.bulk-mode-indicator {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 16px;
  height: 16px;
  font-size: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 50%;
}

.model-option.selected .bulk-mode-indicator {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color-dark);
}

.required-indicator {
  position: absolute;
  top: 5px;
  right: 5px;
  color: var(--warning-color);
  font-weight: bold;
}

.validation-feedback {
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 15px;
}

.validation-feedback.valid {
  background-color: var(--success-bg-light);
  color: var(--success-text);
  border: 1px solid var(--success-border);
}

.validation-feedback.invalid {
  background-color: var(--error-bg-light);
  color: var(--error-text);
  border: 1px solid var(--error-border);
}

.model-dependencies {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--border-color);
}

.dependency-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.dependency-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}

.dataset-select,
.parameter input,
.parameter select {
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.upload-btn {
  padding: 8px 15px;
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 10px;
}

.upload-btn:hover {
  background-color: var(--primary-color-dark);
}

.parameter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.parameter {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.parameter label {
  font-size: 14px;
  color: var(--text-secondary);
}

.action-buttons {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin-top: 20px;
}

.start-btn,
.stop-btn {
  padding: 12px 25px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s;
}

.start-btn {
  background-color: var(--primary-color);
  color: white;
}

.start-btn:hover:not(:disabled) {
  background-color: var(--primary-color-dark);
}

.start-btn:disabled {
  background-color: var(--disabled-bg);
  cursor: not-allowed;
}

.stop-btn {
  background-color: var(--error-color);
  color: white;
}

.stop-btn:hover:not(:disabled) {
  background-color: var(--error-color-dark);
}

.stop-btn:disabled {
  background-color: var(--disabled-bg);
  cursor: not-allowed;
}

.training-progress {
  background-color: var(--bg-secondary);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  margin-bottom: 30px;
}

.progress-container {
  height: 20px;
  background-color: var(--bg-primary);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 10px;
  border: 1px solid var(--border-color);
}

.progress-bar {
  height: 100%;
  background-color: var(--primary-color);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: width 0.3s ease;
}

.progress-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
  font-size: 14px;
}

.terminal-section {
  margin-top: 20px;
}

.model-evaluation {
  background-color: var(--bg-secondary);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  margin-bottom: 30px;
}

.evaluation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.metric-card {
  background-color: var(--bg-primary);
  padding: 15px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  text-align: center;
}

.metric-card h3 {
  font-size: 14px;
  margin-bottom: 8px;
  color: var(--text-secondary);
}

.metric-value {
  font-size: 24px;
  font-weight: bold;
  color: var(--primary-color);
}

.confusion-matrix {
  margin-top: 20px;
}

.matrix-grid {
  display: grid;
  grid-template-columns: auto repeat(auto-fit, minmax(60px, 1fr));
  gap: 2px;
  margin-top: 10px;
}

.matrix-header {
  background-color: var(--bg-primary);
  padding: 8px;
  text-align: center;
  font-size: 12px;
  font-weight: bold;
  border: 1px solid var(--border-color);
}

.matrix-cell {
  background-color: var(--bg-primary);
  padding: 8px;
  text-align: center;
  font-size: 12px;
  border: 1px solid var(--border-color);
}

.matrix-cell.highlight {
  background-color: var(--primary-color-light);
  font-weight: bold;
}

.training-history {
  background-color: var(--bg-secondary);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 15px;
}

.history-table th,
.history-table td {
  padding: 10px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.history-table th {
  background-color: var(--bg-primary);
  font-weight: bold;
  color: var(--text-primary);
}

.history-table tr:hover {
  background-color: var(--bg-hover);
}

.history-table button {
  padding: 5px 10px;
  margin-right: 5px;
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
}

.history-table button:hover {
  background-color: var(--primary-color-dark);
}

.knowledge-assist-options {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid var(--border-color);
}

.knowledge-options-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 10px;
}

.knowledge-option {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.combination-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 15px;
}

.combination-btn {
  padding: 8px 12px;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.combination-btn:hover {
  background-color: var(--primary-color-light);
  border-color: var(--primary-color);
}

/* Responsive design */
@media (max-width: 768px) {
  .train-view {
    padding: 0;
  }
  
  .control-panel {
    gap: 15px;
  }
  
  .model-grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
  
  .parameter-grid {
    grid-template-columns: 1fr;
  }
  
  .evaluation-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .history-table {
    font-size: 12px;
  }
  
  .history-table th,
  .history-table td {
    padding: 6px;
  }
}

/* Model selection chips for management panel */
.model-chip {
  display: inline-flex;
  align-items: center;
  background-color: var(--primary-color-light);
  color: var(--primary-color);
  padding: 4px 8px;
  border-radius: 4px;
  margin: 2px;
  font-size: 13px;
  border: 1px solid var(--border-color);
}

.chip-remove {
  background: none;
  border: none;
  color: var(--text-light);
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  margin-left: 4px;
  padding: 0 2px;
}

.chip-remove:hover {
  color: var(--error-color);
}

.quick-model-select {
  display: flex;
  gap: 8px;
  align-items: center;
}

.quick-select {
  flex: 1;
  padding: 6px 8px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
}

/* Status items styling for management panel */
.status-item {
  display: flex;
  margin-bottom: 12px;
}

.status-label {
  width: 140px;
  color: var(--text-secondary);
  font-weight: 500;
  font-size: 14px;
}

.status-value {
  flex: 1;
  color: var(--text-primary);
  font-size: 14px;
}

.status-value.active {
  color: #4CAF50;
  font-weight: 500;
}

.status-value.inactive {
  color: var(--text-light);
}

/* Enhanced Model Selection Styles */
.enhanced-model-controls .status-label {
  width: 140px;
}

.enhanced-model-panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 12px;
  margin-top: 8px;
}

.model-search-filter-row {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.model-search {
  flex: 2;
  position: relative;
}

.search-input {
  width: 100%;
  padding: 8px 12px 8px 36px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 14px;
  background: var(--bg-primary);
}

.search-input:disabled {
  background: var(--disabled-bg);
  cursor: not-allowed;
}

.search-icon {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-light);
}

.model-category-filter {
  flex: 1;
}

.category-select {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 14px;
  background: var(--bg-primary);
}

.model-quick-actions {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.bulk-actions {
  flex: 2;
  display: flex;
  gap: 8px;
}

.category-actions {
  flex: 1;
}

.category-action-select {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 14px;
  background: var(--bg-primary);
}

.btn-small.active {
  background: var(--primary-color-dark);
  color: white;
}

.quick-select-row {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.quick-model-select {
  flex: 3;
  display: flex;
  gap: 8px;
}

.quick-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 14px;
  background: var(--bg-primary);
}

.model-stats {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary);
}

.model-count,
.selected-count {
  padding: 4px 8px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  text-align: center;
}

.filtered-models-preview {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 12px;
  margin-top: 12px;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.preview-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.preview-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.preview-chip {
  padding: 4px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.preview-chip:hover {
  background: var(--bg-hover);
}

.preview-chip.selected {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color-dark);
}

.chip-check {
  margin-left: 4px;
  font-weight: bold;
}

.more-indicator {
  padding: 4px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  font-size: 12px;
  color: var(--text-light);
  font-style: italic;
}

/* Model Details Modal Styles */
.model-details-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
}

.modal-content {
  background: var(--bg-primary);
  border-radius: 8px;
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
}

.modal-header h3 {
  margin: 0;
  font-size: 18px;
}

.modal-close {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
  padding: 0;
  width: 30px;
  height: 30px;
  line-height: 1;
}

.modal-close:hover {
  color: var(--text-primary);
}

.modal-body {
  padding: 20px;
}

.model-detail-section {
  margin-bottom: 20px;
}

.detail-row {
  display: flex;
  margin-bottom: 12px;
  align-items: flex-start;
}

.detail-row.full-width {
  flex-direction: column;
  align-items: stretch;
}

.detail-label {
  width: 120px;
  color: var(--text-secondary);
  font-weight: 500;
  font-size: 14px;
  flex-shrink: 0;
}

.detail-value {
  flex: 1;
  color: var(--text-primary);
  font-size: 14px;
}

.description-text {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 12px;
  margin-top: 8px;
  line-height: 1.5;
}

.status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.available {
  background: #e8f5e9;
  color: #2e7d32;
}

.status-badge.unavailable {
  background: #ffebee;
  color: #c62828;
}

.status-badge.training {
  background: #fff3e0;
  color: #ef6c00;
}

.modal-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  padding-top: 20px;
  border-top: 1px solid var(--border-color);
}

/* External API Selection Styling */
.status-item.external-api-selection {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border-color);
}

.external-api-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
}

.external-api-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
  padding: 4px 0;
}

.external-api-toggle input[type="checkbox"] {
  margin: 0;
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.external-api-dropdown {
  margin-left: 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.external-api-select {
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  max-width: 300px;
  cursor: pointer;
}

.external-api-select:disabled {
  background-color: var(--disabled-bg);
  cursor: not-allowed;
  opacity: 0.6;
}

.external-api-select option {
  padding: 8px;
  background-color: var(--bg-primary);
  color: var(--text-primary);
}

.external-api-select option:disabled {
  color: var(--text-light);
  font-style: italic;
}

.external-api-info {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* Animation */
@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* Resource Monitoring Chart */
.resource-chart-container {
  margin-top: 20px;
  padding: 15px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
}

.resource-chart-container h4 {
  margin: 0 0 15px 0;
  font-size: 16px;
  color: var(--text-primary);
}

.chart-wrapper {
  position: relative;
  height: 250px;
  width: 100%;
}
</style>
