<template>
  <div class="train-view">
    <!-- Status Messages -->
    <div class="status-messages">
      <div v-if="errorState.hasError" class="message error">
        <span class="icon">⚠️</span>
        {{ errorState.message }}
      </div>
      <div v-if="successState.hasSuccess" class="message success">
        <span class="icon">✅</span>
        {{ successState.message }}
      </div>
      <div v-if="warningState.hasWarning" class="message warning">
        <span class="icon">⚠️</span>
        {{ warningState.message }}
      </div>
      <div v-if="infoState.hasInfo" class="message info">
        <span class="icon">ℹ️</span>
        {{ infoState.message }}
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
                v-for="model in availableModels" 
                :key="model.id"
                class="model-option"
                :class="{ 
                  selected: selectedModels.includes(model.id),
                  required: isModelRequired(model.id),
                  disabled: isModelDisabled(model.id)
                }"
                @click="toggleModelSelection(model.id)"
                :title="getModelTooltip(model.id)"
              >
                {{ model.name }}
                <span v-if="isModelRequired(model.id)" class="required-indicator">*</span>
              </div>
            </div>
        
        <!-- Combination Validation Feedback -->
            <div class="validation-feedback" :class="{ valid: combinationValid, invalid: !combinationValid }">
              <span v-if="combinationValid">✓ Combination Valid</span>
              <span v-else>✗ {{ validationMessage }}</span>
            </div>
        
        <!-- Model Dependencies -->
            <div class="model-dependencies" v-if="trainingMode === 'joint' && selectedModels.length > 0">
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
        <select v-model="selectedDataset" class="dataset-select">
          <option 
            v-for="dataset in datasets" 
            :key="dataset.id"
            :value="dataset.id"
          >
            {{ dataset.name }}
          </option>
        </select>
        <button @click="openUploadDialog" class="upload-btn">
          Upload Dataset
        </button>
        <input 
          type="file" 
          ref="datasetInput" 
          style="display: none" 
          @change="handleDatasetUpload"
          multiple
        >
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
          <div class="parameter" style="grid-column: span 2;">
            <label>
              <input type="checkbox" v-model="parameters.fromScratch">
              Train from Scratch (No Pretrained Models)
            </label>
          </div>
          <div class="parameter">
            <label>Learning Rate:</label>
            <input type="number" v-model.number="parameters.learningRate" step="0.001" min="0.0001" max="1">
          </div>
          <div class="parameter">
            <label>Validation Split:</label>
            <input type="number" v-model.number="parameters.validationSplit" step="0.05" min="0.1" max="0.5">
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
            v-for="strategy in trainingStrategies" 
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
          <div class="metric-value">{{ evaluationResults.accuracy }}%</div>
        </div>
        <div class="metric-card">
          <h3>Loss</h3>
          <div class="metric-value">{{ evaluationResults.loss.toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>Precision</h3>
          <div class="metric-value">{{ evaluationResults.precision.toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>Recall</h3>
          <div class="metric-value">{{ evaluationResults.recall.toFixed(4) }}</div>
        </div>
      </div>
      
      <div class="confusion-matrix">
        <h3>Confusion Matrix</h3>
        <div class="matrix-grid">
          <div class="matrix-header"></div>
          <div 
            v-for="label in evaluationResults.labels" 
            :key="label" 
            class="matrix-header"
          >
            {{ label }}
          </div>
          <template v-for="(row, rowIndex) in evaluationResults.confusionMatrix" :key="rowIndex">
            <div class="matrix-header">{{ evaluationResults.labels[rowIndex] }}</div>
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
          <tr v-for="session in trainingHistory" :key="session.id">
            <td>{{ formatDate(session.date) }}</td>
            <td>
              <span v-for="model in session.models" :key="model">
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

.mode-selection, .model-selection, .dataset-selection, .parameter-settings, .strategy-selection {
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
import errorHandler from '@/utils/errorHandler';
import { letterToId, idToLetter, letterToIdMap, idToLetterMap, lettersToIds, idsToLetters } from '@/utils/modelIdMapper';
import TerminalWindow from '@/components/TerminalWindow.vue';

export default {
    setup() {
      
      // Training mode
    const trainingMode = ref('individual');
    
    // Available models - dynamically fetched from API
    const availableModels = ref([]);
    const modelsLoading = ref(true);
    const modelsError = ref(null);
    
    // Get available model list
    const loadAvailableModels = async () => {
      try {
        modelsLoading.value = true;
        modelsError.value = null;
        
        // Use real API to fetch available models
        const response = await api.get('/api/models/available');
        
        if (response.data.status === 'success') {
          availableModels.value = response.data.models || [];
          showInfo('Models loaded successfully from backend');
        } else {
          throw new Error(response.data.message || 'Failed to load models');
        }
      } catch (error) {
        // Log error
        console.error('Error loading models:', error);
        errorHandler.handleError(error, 'Failed to load models');
        showError('Failed to load models from backend. Using default model list.');
        
        // Initialize with default model list based on modelIdMapper
        availableModels.value = Object.keys(letterToIdMap).map(letterId => {
          const stringId = letterToIdMap[letterId];
          return {
            id: letterId,
            backendId: stringId,
            name: getModelDisplayName(letterId),
            description: getModelDescription(letterId),
            type: 'default',
            status: 'available',
            trainingStatus: {
              isTraining: false,
              progress: 0,
              status: 'idle'
            }
          };
        });
      } finally {
        modelsLoading.value = false;
      }
    };
    
    
    // Selected models
    const selectedModels = ref(['B', 'C']);
    
    // Datasets
    const datasets = ref([
      { id: 'multimodal_v1', name: 'Multimodal Dataset v1' },
      { id: 'language_only', name: 'Language Only Dataset' },
      { id: 'vision_only', name: 'Vision Only Dataset' },
      { id: 'sensor_only', name: 'Sensor Only Dataset' }
    ]);
    
    // Recommended combinations - includes all 19 system models (A-S)
    const recommendedCombinations = ref({
      basic_interaction: ['A', 'B', 'C'],
      visual_processing: ['A', 'D', 'J', 'K'],
      sensor_analysis: ['A', 'L'],
      knowledge_intensive: ['A', 'C'],
      specialized_domains: ['A', 'C'],
      emotional_intelligence: ['A', 'B', 'I'],
      complete_system: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
      autonomous_control: ['A', 'F', 'M'],
      cognitive_processing: ['A', 'B', 'O', 'R'],
      multimodal_perception: ['A', 'D', 'E', 'L', 'J'],
      full_system: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    });
    
    // Combination validation status
    const combinationValid = ref(true);
    const validationMessage = ref('');
    
    // Model dependencies - dependencies for all 19 system models (A-S)
    const modelDependencies = ref({
      A: ['B', 'C'], // Manager model depends on language and knowledge models
      B: ['C'],      // Language model depends on knowledge model
      C: [],         // Knowledge model has no dependencies
      D: ['J'],      // Vision model depends on spatial model
      E: ['B'],      // Audio model depends on language model
      F: ['A', 'C'], // Autonomous model depends on manager and knowledge models
      G: ['B', 'C'], // Programming model depends on language and knowledge models
      H: ['B', 'C'], // Planning model depends on language and knowledge models
      I: ['B', 'C'], // Emotion model depends on language and knowledge models
      J: [],         // Spatial model has no dependencies
      K: ['D', 'C'], // Computer Vision model depends on vision and knowledge models
      L: [],         // Sensor model has no dependencies
      M: ['J', 'C'], // Motion model depends on spatial and knowledge models
      N: ['C'],      // Prediction model depends on knowledge model
      O: ['B', 'C'], // Advanced Reasoning model depends on language and knowledge models
      P: ['C'],      // Data Fusion model depends on knowledge model
      Q: ['B', 'C'], // Creative Problem Solving model depends on language and knowledge models
      R: ['B', 'C'], // Meta Cognition model depends on language and knowledge models
      S: ['B', 'C']  // Value Alignment model depends on language and knowledge models
    });
    
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
    
    // Check if model is disabled (missing dependencies)
    const isModelDisabled = computed(() => (modelId) => {
      if (trainingMode.value === 'individual') return false;
      
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
    
    // Select all models
    const selectAllModels = () => {
      // Get all available model IDs
      const allModelIds = availableModels.value.map(model => model.id);
      
      // Use selectRecommendedCombination function to ensure dependency handling
      selectRecommendedCombination(allModelIds);
      
      // Show success message
      showSuccess('All models selected');
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
        warningState.value.hasWarning = false;
      }, 8000);
    };
    
    // Show success message
    const showSuccess = (message) => {
      successState.value = {
        hasSuccess: true,
        message
      };
      
      setTimeout(() => {
        successState.value.hasSuccess = false;
      }, 5000);
    };
    
    // Show info message
    const showInfo = (message) => {
      infoState.value = {
        hasInfo: true,
        message
      };
      
      setTimeout(() => {
        infoState.value.hasInfo = false;
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
    
    // Selected dataset
    const selectedDataset = ref('multimodal_v1');
    
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
      learningRateSchedule: 'constant'
    });
    
    // Training strategy options
    const trainingStrategies = ref([
      { id: 'standard', name: 'Standard Training' },
      { id: 'knowledge_assisted', name: 'Knowledge Assisted Training' },
      { id: 'progressive', name: 'Progressive Training' },
      { id: 'adaptive', name: 'Adaptive Learning' }
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
        const response = await api.post('/api/datasets/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
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
        
        // Prepare training request data
        const trainingData = {
          models: selectedModels.value.map(modelId => {
            const model = availableModels.value.find(m => m.id === modelId);
            return model ? model.backendId : modelId;
          }),
          dataset_id: selectedDataset.value,
          parameters: {
            ...parameters.value,
            strategy: selectedStrategy.value,
            knowledge_assist: selectedStrategy === 'knowledge_assisted' ? knowledgeAssistOptions.value : null
          },
          training_mode: trainingMode.value,
          fromScratch: parameters.value.fromScratch || false
        };
        
        // Call FastAPI backend training start API
        try {
          const response = await api.post('/api/training/start', trainingData);
          
          currentJobId.value = response.data.job_id;
          addLog(`Training job created with ID: ${currentJobId.value}`);
          
          // Start WebSocket connection for real-time updates
          startWebSocketConnection(currentJobId.value);
        } catch (apiError) {
          // If API call fails, show error and stop training
          addLog(`Failed to connect to server: ${apiError.message || 'Unknown error'}`);
          addLog('Please ensure the backend service is running on localhost:8000');
          
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
        // Connect to real-time data stream manager based on configured port
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}://localhost:8765/ws/training/${jobId}`;
        
        // Add connection attempt log
        addLog('Connecting to WebSocket: ' + wsUrl.replace(/^(wss?:\/\/[^/]+).*/, '$1/...'));
        
        websocketConnection.value = new WebSocket(wsUrl);
        
        // Connection state
        let connectionAttempts = 0;
        const maxReconnectAttempts = 3;
        let reconnectTimeout = null;
        
        websocketConnection.value.onopen = () => {
          connectionAttempts = 0;
          addLog('WebSocket connected successfully', 'success');
          showInfo('Real-time updates enabled');
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
                console.log('System message:', data);
                break;
              
              default:
                addLog('Unknown WebSocket message type: ' + data.type, 'warning');
            }
          } catch (error) {
            errorHandler.handleError(error, 'WebSocket message parse error');
            addLog('WebSocket message parse error: ' + error.message, 'error');
          }
        };
        
        websocketConnection.value.onerror = (error) => {
          const errorMessage = error.message || 'Unknown error';
          addLog('WebSocket connection error: ' + errorMessage, 'error');
          errorHandler.handleError(error, 'WebSocket connection error');
          
                // Attempt reconnection
                if (connectionAttempts < maxReconnectAttempts && isTraining.value) {
            connectionAttempts++;
            const delay = Math.pow(2, connectionAttempts) * 1000;
            addLog(`Reconnecting to WebSocket (attempt ${connectionAttempts}/${maxReconnectAttempts}, delay ${delay/1000}s)`, 'warning');
            
            reconnectTimeout = setTimeout(() => {
              startWebSocketConnection(jobId);
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
          if (websocketConnection.value && websocketConnection.value.readyState !== WebSocket.OPEN) {
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

      try {
        // Try to stop training via API
        if (currentJobId.value && !currentJobId.value.startsWith('mock_')) {
          await api.post('/api/training/stop', { job_id: currentJobId.value });
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
        errorState.value.hasError = false;
      }, 5000);
      
      addLog(`ERROR: ${message}`);
    };
    
    // Load training history
    const loadTrainingHistory = async () => {
      try {
        // Call FastAPI backend to get training history
        const response = await api.get('/api/training/history');
        
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
    
    // Initialize
    const init = async () => {
      await loadAvailableModels();
      await loadTrainingHistory();
      
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
    };
    
    // Initialize when component is mounted
    onMounted(() => {
      init();
    });
    
    // Cleanup when component is unmounted
    onUnmounted(() => {
      clearInterval(trainingTimer);
      clearInterval(statusPollingInterval.value);
      
      if (websocketConnection.value) {
        websocketConnection.value.close();
      }
    });
    
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
      successState,
      warningState,
      infoState,
      
      // Computed properties
      isModelRequired,
      isModelDisabled,
      getModelTooltip,
      
      // Methods
      toggleModelSelection,
      selectRecommendedCombination,
      openUploadDialog,
      handleDatasetUpload,
      startTraining,
      stopTraining,
      loadAvailableModels,
      loadTrainingHistory,
      showError,
      showWarning,
      showSuccess,
      showInfo,
      addLog,
      
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
            alert(sessionDetails);
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
</style>
