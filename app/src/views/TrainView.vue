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
        <h2>{{ $t('train.trainingMode') }}</h2>
        <div class="mode-options">
          <button 
            @click="trainingMode = 'individual'"
            :class="{ active: trainingMode === 'individual' }"
          >
            {{ $t('train.individual') }}
          </button>
          <button 
            @click="trainingMode = 'joint'"
            :class="{ active: trainingMode === 'joint' }"
          >
            {{ $t('train.joint') }}
          </button>
        </div>
      </div>
      
      <div class="model-selection">
        <h2>{{ $t('train.selectModels') }}</h2>
        
        <!-- Recommended Combinations -->
        <div class="recommended-combinations" v-if="trainingMode === 'joint'">
          <h3>{{ $t('train.recommendedCombinations') }}</h3>
          <div class="combination-buttons">
            <button 
              v-for="(combination, name) in recommendedCombinations" 
              :key="name"
              @click="selectRecommendedCombination(combination)"
              class="combination-btn"
            >
              {{ $t(`train.combinations.${name}`) }}
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
            {{ $t(`models.${model.id}`) }}
            <span v-if="isModelRequired(model.id)" class="required-indicator">*</span>
          </div>
        </div>
        
        <!-- Combination Validation Feedback -->
        <div class="validation-feedback" :class="{ valid: combinationValid, invalid: !combinationValid }">
          <span v-if="combinationValid">✓ {{ $t('train.combinationValid') }}</span>
          <span v-else>✗ {{ validationMessage }}</span>
        </div>
        
        <!-- Model Dependencies -->
        <div class="model-dependencies" v-if="trainingMode === 'joint' && selectedModels.length > 0">
          <h3>{{ $t('train.dependencies') }}</h3>
          <div class="dependency-list">
            <div v-for="dependency in currentDependencies" :key="dependency.model" class="dependency-item">
              <span class="model-name">{{ $t(`models.${dependency.model}`) }}</span>
              <span class="dependency-arrow">→</span>
              <span class="depends-on">{{ dependency.dependencies.map(d => $t(`models.${d}`)).join(', ') }}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="dataset-selection">
        <h2>{{ $t('train.dataset') }}</h2>
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
          {{ $t('train.uploadDataset') }}
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
        <h2>{{ $t('train.parameters') }}</h2>
        <div class="parameter-grid">
          <div class="parameter">
            <label>{{ $t('train.epochs') }}:</label>
            <input type="number" v-model.number="parameters.epochs" min="1" max="1000">
          </div>
          <div class="parameter">
            <label>{{ $t('train.batchSize') }}:</label>
            <input type="number" v-model.number="parameters.batchSize" min="1" max="1024">
          </div>
          <div class="parameter">
            <label>{{ $t('train.learningRate') }}:</label>
            <input type="number" v-model.number="parameters.learningRate" step="0.001" min="0.0001" max="1">
          </div>
          <div class="parameter">
            <label>{{ $t('train.validationSplit') }}:</label>
            <input type="number" v-model.number="parameters.validationSplit" step="0.05" min="0.1" max="0.5">
          </div>
          <div class="parameter">
            <label>{{ $t('train.dropoutRate') }}:</label>
            <input type="number" v-model.number="parameters.dropoutRate" step="0.05" min="0" max="0.5">
          </div>
          <div class="parameter">
            <label>{{ $t('train.weightDecay') }}:</label>
            <input type="number" v-model.number="parameters.weightDecay" step="0.0001" min="0" max="0.01">
          </div>
          <div class="parameter">
            <label>{{ $t('train.momentum') }}:</label>
            <input type="number" v-model.number="parameters.momentum" step="0.1" min="0" max="0.99">
          </div>
          <div class="parameter">
            <label>{{ $t('train.optimizer') }}:</label>
            <select v-model="parameters.optimizer">
              <option value="adam">{{ $t('train.optimizers.adam') }}</option>
              <option value="sgd">{{ $t('train.optimizers.sgd') }}</option>
              <option value="rmsprop">{{ $t('train.optimizers.rmsprop') }}</option>
              <option value="adagrad">{{ $t('train.optimizers.adagrad') }}</option>
            </select>
          </div>
        </div>
      </div>
      
      <!-- Training Strategy Selection -->
      <div class="strategy-selection" v-if="trainingMode === 'joint'">
        <h2>{{ $t('train.trainingStrategy') }}</h2>
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
          <h3>{{ $t('train.knowledgeAssistOptions') }}</h3>
          <div class="knowledge-options-grid">
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.domainKnowledge">
                {{ $t('train.domainKnowledge') }}
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.commonSense">
                {{ $t('train.commonSense') }}
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.proceduralKnowledge">
                {{ $t('train.proceduralKnowledge') }}
              </label>
            </div>
            <div class="knowledge-option">
              <label>
                <input type="checkbox" v-model="knowledgeAssistOptions.contextualLearning">
                {{ $t('train.contextualLearning') }}
              </label>
            </div>
            <div class="knowledge-option">
              <label>{{ $t('train.knowledgeIntensity') }}:</label>
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
          {{ isTraining ? $t('train.trainingInProgress') : $t('train.startTraining') }}
        </button>
        <button 
          @click="stopTraining" 
          :disabled="!isTraining"
          class="stop-btn"
        >
          {{ $t('train.stopTraining') }}
        </button>
      </div>
    </div>
    
    <!-- Training Progress -->
    <div class="training-progress">
      <h2>{{ $t('train.progress') }}</h2>
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
        <h3>{{ $t('train.commandLine') }}</h3>
        <TerminalWindow
          :logs="trainingLogs"
          :title="$t('train.trainingTerminal')"
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
      <h2>{{ $t('train.evaluation') }}</h2>
      <div class="evaluation-grid">
        <div class="metric-card">
          <h3>{{ $t('train.accuracy') }}</h3>
          <div class="metric-value">{{ evaluationResults.accuracy }}%</div>
        </div>
        <div class="metric-card">
          <h3>{{ $t('train.loss') }}</h3>
          <div class="metric-value">{{ evaluationResults.loss.toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>{{ $t('train.precision') }}</h3>
          <div class="metric-value">{{ evaluationResults.precision.toFixed(4) }}</div>
        </div>
        <div class="metric-card">
          <h3>{{ $t('train.recall') }}</h3>
          <div class="metric-value">{{ evaluationResults.recall.toFixed(4) }}</div>
        </div>
      </div>
      
      <div class="confusion-matrix">
        <h3>{{ $t('train.confusionMatrix') }}</h3>
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
    
    <!-- 训练历史 -->
    <div class="training-history">
      <h2>{{ $t('train.history') }}</h2>
      <table class="history-table">
        <thead>
          <tr>
            <th>{{ $t('train.date') }}</th>
            <th>{{ $t('train.models') }}</th>
            <th>{{ $t('train.dataset') }}</th>
            <th>{{ $t('train.duration') }}</th>
            <th>{{ $t('train.accuracy') }}</th>
            <th>{{ $t('train.actions') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="session in trainingHistory" :key="session.id">
            <td>{{ formatDate(session.date) }}</td>
            <td>
              <span v-for="model in session.models" :key="model">
                {{ $t(`models.${model}`) }}
              </span>
            </td>
            <td>{{ session.dataset }}</td>
            <td>{{ formatDuration(session.duration) }}</td>
            <td>{{ session.accuracy }}%</td>
            <td>
              <button @click="viewSession(session.id)">{{ $t('train.view') }}</button>
              <button @click="compareSession(session.id)">{{ $t('train.compare') }}</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import axios from 'axios';
import errorHandler from '@/utils/errorHandler';
import { letterToId, idToLetter, letterToIdMap, idToLetterMap, lettersToIds, idsToLetters } from '@/utils/modelIdMapper';
import TerminalWindow from '@/components/TerminalWindow.vue';

export default {
  setup() {
    const { t } = useI18n();
    
    // 训练模式
    const trainingMode = ref('individual');
    
    // 可用模型 - 从API动态获取
    const availableModels = ref([]);
    const modelsLoading = ref(true);
    const modelsError = ref(null);
    
    // 获取可用模型列表
    const loadAvailableModels = async () => {
      try {
        modelsLoading.value = true;
        modelsError.value = null;
        
        const response = await axios.get('/api/train/models');
        
        if (response.data.status === 'success') {
          // 将后端字符串ID转换为前端字母ID
          availableModels.value = response.data.data.map(backendId => {
            const frontendId = idToLetter(backendId) || backendId;
            return {
              id: frontendId,
              name: t(`models.${frontendId}`),
              backendId: backendId // 保存原始后端ID用于API调用
            };
          });
        } else {
          throw new Error(response.data.detail || '获取模型列表失败');
        }
      } catch (error) {
        modelsError.value = error.message;
        errorHandler.handleError(error, '获取可用模型失败');
        
        // 如果API调用失败，使用默认模型列表作为后备（使用字母ID）
        availableModels.value = [
          { id: 'A', name: t('models.A'), backendId: 'manager' },
          { id: 'B', name: t('models.B'), backendId: 'language' },
          { id: 'C', name: t('models.C'), backendId: 'audio' },
          { id: 'D', name: t('models.D'), backendId: 'vision_image' },
          { id: 'E', name: t('models.E'), backendId: 'vision_video' },
          { id: 'F', name: t('models.F'), backendId: 'spatial' },
          { id: 'G', name: t('models.G'), backendId: 'sensor' },
          { id: 'H', name: t('models.H'), backendId: 'computer_control' },
          { id: 'I', name: t('models.I'), backendId: 'motion_control' },
          { id: 'J', name: t('models.J'), backendId: 'knowledge' },
          { id: 'K', name: t('models.K'), backendId: 'programming' }
        ];
        
        showWarning(t('errors.modelsFallback'));
      } finally {
        modelsLoading.value = false;
      }
    };
    
    // 选中的模型
    const selectedModels = ref(['B', 'J']);
    
    // 数据集
    const datasets = ref([
      { id: 'multimodal_v1', name: t('training.datasets.multimodal_v1') },
      { id: 'language_only', name: t('training.datasets.language_only') },
      { id: 'vision_only', name: t('training.datasets.vision_only') },
      { id: 'sensor_only', name: t('training.datasets.sensor_only') }
    ]);
    
    // 推荐组合 - 包含所有11个核心模型的组合（A-K）
    const recommendedCombinations = ref({
      basic_interaction: ['A', 'B', 'C'],
      visual_processing: ['A', 'D', 'E', 'F'],
      sensor_analysis: ['A', 'G', 'F'],
      knowledge_intensive: ['A', 'J'],
      specialized_domains: ['A', 'J'],
      emotional_intelligence: ['A', 'B', 'J'],
      complete_system: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
      autonomous_control: ['A', 'I', 'H', 'G', 'F'],
      cognitive_processing: ['A', 'B', 'J'],
      multimodal_perception: ['A', 'D', 'E', 'C', 'G', 'F']
    });
    
    // 组合验证状态
    const combinationValid = ref(true);
    const validationMessage = ref('');
    
    // 模型依赖关系 - 所有11个核心模型的依赖关系（A-K）
    const modelDependencies = ref({
      A: ['B', 'J'], // 管理模型依赖语言模型和知识库模型
      B: ['J'],      // 语言模型依赖知识库模型
      C: ['B'],      // 音频处理模型依赖语言模型
      D: ['F'],      // 图片视觉处理模型依赖双目空间定位模型
      E: ['D', 'F'], // 视频流视觉处理模型依赖图片视觉和双目空间定位模型
      F: ['D'],      // 双目空间定位感知模型依赖图片视觉模型
      G: ['F'],      // 传感器感知模型依赖双目空间定位模型
      H: ['B', 'J'], // 计算机控制模型依赖语言模型和知识库模型
      I: ['F', 'G', 'H'], // 运动和执行器控制模型依赖空间、传感器和计算机控制模型
      J: [],         // 知识库专家模型无依赖
      K: ['B', 'J']  // 编程模型依赖语言模型和知识库模型
    });
    
    // 计算属性
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
    
    // 检查模型是否必需（有依赖关系）
    const isModelRequired = computed(() => (modelId) => {
      return currentDependencies.value.some(dep => 
        dep.dependencies.includes(modelId)
      );
    });
    
    // 检查模型是否被禁用（依赖缺失）
    const isModelDisabled = computed(() => (modelId) => {
      if (trainingMode.value === 'individual') return false;
      
      // 检查是否有未满足的依赖
      const dependencies = modelDependencies.value[modelId];
      if (dependencies && dependencies.length > 0) {
        return dependencies.some(dep => !selectedModels.value.includes(dep));
      }
      return false;
    });
    
    // 获取模型提示信息
    const getModelTooltip = computed(() => (modelId) => {
      const dependencies = modelDependencies.value[modelId];
      if (dependencies && dependencies.length > 0) {
        return t('train.requiresModels', {
          models: dependencies.map(d => t(`models.${d}`)).join(', ')
        });
      }
      return '';
    });
    
    // 选择推荐组合
    const selectRecommendedCombination = (combination) => {
      selectedModels.value = [...combination];
    };
    
    // 验证模型组合
    const validateModelCombination = () => {
      if (trainingMode.value === 'individual') {
        combinationValid.value = true;
        validationMessage.value = '';
        return;
      }
      
      // 检查依赖关系
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
        validationMessage.value = t('train.missingDependencies', {
          details: missingDependencies.map(d => 
            `${t(`models.${d.model}`)} → ${t(`models.${d.dependency}`)}`
          ).join(', ')
        });
      } else {
        combinationValid.value = true;
        validationMessage.value = t('train.combinationValid');
      }
    };
    
    // 监听模型选择变化
    watch([selectedModels, trainingMode], () => {
      validateModelCombination();
    }, { immediate: true });
    
    // 选中的数据集
    const selectedDataset = ref('multimodal_v1');
    
    // 训练参数
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
    
    // 训练策略选项
    const trainingStrategies = ref([
      { id: 'standard', name: t('train.strategies.standard') },
      { id: 'knowledge_assisted', name: t('train.strategies.knowledge_assisted') },
      { id: 'progressive', name: t('train.strategies.progressive') },
      { id: 'adaptive', name: t('train.strategies.adaptive') }
    ]);
    
    // 选中的训练策略
    const selectedStrategy = ref('standard');
    
    // 是否启用知识库辅助
    const enableKnowledgeAssist = ref(false);
    
    // 知识库辅助选项
    const knowledgeAssistOptions = ref({
      domainKnowledge: true,
      commonSense: true,
      proceduralKnowledge: false,
      contextualLearning: true,
      knowledgeIntensity: 0.7
    });
    
    // 训练状态
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
    
    // 评估结果
    const evaluationResults = ref(null);
    
    // 训练历史
    const trainingHistory = ref([]);
    
    // 文件上传引用
    const datasetInput = ref(null);
    
    // 计时器引用
    let trainingTimer = null;
    let startTime = null;
    
    // 错误状态
    const errorState = ref({
      hasError: false,
      message: '',
      type: ''
    });
    
    // 成功状态
    const successState = ref({
      hasSuccess: false,
      message: ''
    });
    
    // 警告状态
    const warningState = ref({
      hasWarning: false,
      message: ''
    });
    
    // 信息状态
    const infoState = ref({
      hasInfo: false,
      message: ''
    });
    
    // 切换模型选择
    const toggleModelSelection = (modelId) => {
      if (trainingMode.value === 'individual') {
        // 单独训练模式只能选择一个模型
        selectedModels.value = [modelId];
      } else {
        // 联合训练模式可以多选
        const index = selectedModels.value.indexOf(modelId);
        if (index > -1) {
          selectedModels.value.splice(index, 1);
        } else {
          selectedModels.value.push(modelId);
        }
      }
    };
    
    // 打开上传对话框
    const openUploadDialog = () => {
      datasetInput.value.click();
    };
    
    // 处理数据集上传
    const handleDatasetUpload = async (event) => {
      const files = event.target.files;
      if (files.length === 0) return;
      
      try {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append('datasets', files[i]);
        }
        
        const response = await axios.post('/api/datasets/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        // 添加新数据集
        datasets.value.push({
          id: response.data.id,
          name: response.data.name
        });
        
        selectedDataset.value = response.data.id;
        
        addLog(t('train.datasetUploadSuccess', { name: response.data.name }));
      } catch (error) {
        addLog(t('errors.datasetUploadFailed', { error: error.message }));
      }
    };
    
    // 开始训练
    const startTraining = async () => {
      if (selectedModels.value.length === 0) {
        addLog(t('errors.selectAtLeastOneModel'));
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
        
        // 重置计时器
        startTime = new Date();
        updateElapsedTime();
        trainingTimer = setInterval(updateElapsedTime, 1000);
        
        // 添加开始日志
        addLog(t('train.trainingStarted', { 
          mode: trainingMode.value,
          models: selectedModels.value.map(m => t(`models.${m}`)).join(', '),
          dataset: datasets.value.find(d => d.id === selectedDataset.value).name
        }));
        
        // 实际API调用 - 将前端字母ID转换为后端字符串ID
        const backendModels = selectedModels.value.map(modelId => {
          // 查找对应的后端ID
          const model = availableModels.value.find(m => m.id === modelId);
          return model ? model.backendId : letterToId(modelId);
        });
        
        // 模拟API调用和响应（在后端连接不可用时）
        let jobId = Date.now().toString();
        
        try {
          const response = await axios.post('/api/train', {
            mode: trainingMode.value,
            models: backendModels,
            dataset: selectedDataset.value,
            parameters: {
              ...parameters.value,
              training_mode: trainingMode.value,
              strategy: selectedStrategy.value,
              knowledge_assist: trainingMode.value === 'joint' && selectedStrategy.value === 'knowledge_assisted' ? knowledgeAssistOptions.value : null
            }
          }, { timeout: 5000 });
          
          // 如果API调用成功，使用真实的任务ID
          jobId = response.data.job_id;
          currentJobId.value = jobId;
          
          // 启动WebSocket连接进行实时进度更新
          startWebSocketConnection(jobId);
          
          // 启动状态轮询
          startStatusPolling(jobId);
        } catch (apiError) {
          // API调用失败，使用模拟训练
          addLog(t('train.usingMockTraining'));
          currentJobId.value = jobId;
          
          // 模拟训练进度
          simulateTraining();
        }
      } catch (error) {
        addLog(t('errors.trainingStartFailed', { error: error.message }));
        stopTraining();
      }
    };

    // 启动WebSocket连接（增强版，包含重连逻辑和状态监控）
    const startWebSocketConnection = (jobId) => {
      try {
        // 创建WebSocket连接
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/training/${jobId}`;
        
        // 添加连接尝试日志
        addLog(t('train.websocketConnecting', { url: wsUrl.replace(/^(wss?:\/\/[^\/]+)\/.*/, '$1/...') }));
        
        websocketConnection.value = new WebSocket(wsUrl);
        
        // 连接状态
        let connectionAttempts = 0;
        const maxReconnectAttempts = 3;
        let reconnectTimeout = null;
        
        websocketConnection.value.onopen = () => {
          connectionAttempts = 0;
          addLog(t('train.websocketConnected'), 'success');
          showInfo(t('train.realTimeUpdatesEnabled'));
        };
        
        websocketConnection.value.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            // 处理不同类型的消息
            switch (data.type) {
              case 'progress':
                trainingProgress.value = data.progress;
                currentEpoch.value = data.epoch;
                currentLoss.value = data.loss;
                currentAccuracy.value = data.accuracy;
                
                // 如果有额外的训练指标，也进行更新
                if (data.validation_loss !== undefined) {
                  if (!window.validationLoss) window.validationLoss = ref(0);
                  window.validationLoss.value = data.validation_loss;
                }
                if (data.validation_accuracy !== undefined) {
                  if (!window.validationAccuracy) window.validationAccuracy = ref(0);
                  window.validationAccuracy.value = data.validation_accuracy;
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
                // 系统消息，通常不显示给用户
                console.log('System message:', data);
                break;
              
              default:
                addLog(t('errors.unknownWebSocketMessage', { type: data.type }), 'warning');
            }
          } catch (error) {
            errorHandler.handleError(error, 'WebSocket消息解析错误');
            addLog(t('errors.websocketParseError', { error: error.message }), 'error');
          }
        };
        
        websocketConnection.value.onerror = (error) => {
          const errorMessage = error.message || 'Unknown error';
          addLog(t('errors.websocketError', { error: errorMessage }), 'error');
          errorHandler.handleError(error, 'WebSocket连接错误');
          
          // 尝试重连
          if (connectionAttempts < maxReconnectAttempts && isTraining.value) {
            connectionAttempts++;
            const delay = Math.pow(2, connectionAttempts) * 1000;
            addLog(t('train.websocketReconnecting', { attempt: connectionAttempts, max: maxReconnectAttempts, delay: delay/1000 }), 'warning');
            
            reconnectTimeout = setTimeout(() => {
              startWebSocketConnection(jobId);
            }, delay);
          } else if (connectionAttempts >= maxReconnectAttempts) {
            addLog(t('errors.websocketMaxReconnect', { max: maxReconnectAttempts }), 'error');
            showError(t('errors.websocketFallbackToPolling'));
            
            // 如果重连失败，切换到轮询模式
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
          
          // 正常关闭不需要显示错误
          if (code === 1000 || code === 1001) {
            addLog(t('train.websocketDisconnected'), 'info');
          } else {
            addLog(t('errors.websocketClosedUnexpectedly', { code, reason }), 'warning');
          }
        };
        
        // 设置超时检测
        const timeoutId = setTimeout(() => {
          if (websocketConnection.value && websocketConnection.value.readyState !== WebSocket.OPEN) {
            addLog(t('errors.websocketConnectionTimeout'), 'error');
            showError(t('errors.websocketConnectionTimeout'));
          }
        }, 5000);
        
        // 连接成功时清除超时
        websocketConnection.value.onopen = function() {
          clearTimeout(timeoutId);
          connectionAttempts = 0;
          addLog(t('train.websocketConnected'), 'success');
          showInfo(t('train.realTimeUpdatesEnabled'));
        };
      } catch (error) {
        addLog(t('errors.websocketConnectionFailed', { error: error.message }), 'error');
        
        // 连接失败，直接切换到轮询模式
        if (!statusPollingInterval.value) {
          startStatusPolling(jobId);
        }
      }
    };

    // 启动状态轮询（增强版，包含错误处理和自适应轮询间隔）
    const startStatusPolling = (jobId) => {
      // 避免重复启动轮询
      if (statusPollingInterval.value) {
        clearInterval(statusPollingInterval.value);
      }
      
      let pollingInterval = 2000; // 初始2秒间隔
      let consecutiveFailures = 0;
      const maxFailures = 3;
      
      addLog(t('train.startingPolling', { interval: pollingInterval/1000 }), 'info');
      
      statusPollingInterval.value = setInterval(async () => {
        try {
          const response = await axios.get(`/api/training/status/${jobId}`, { 
            timeout: 10000 // 10秒超时
          });
          
          const status = response.data;
          
          // 重置失败计数器
          consecutiveFailures = 0;
          
          // 根据训练进度自适应调整轮询间隔
          if (status.progress > 90) {
            pollingInterval = 1000; // 接近完成时轮询更频繁
          } else if (status.progress > 50) {
            pollingInterval = 1500;
          } else {
            pollingInterval = 2000;
          }
          
          // 更新训练状态
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
          
          // 更新其他可能的指标
          if (status.validation_loss !== undefined) {
            if (!window.validationLoss) window.validationLoss = ref(0);
            window.validationLoss.value = status.validation_loss;
          }
          if (status.validation_accuracy !== undefined) {
            if (!window.validationAccuracy) window.validationAccuracy = ref(0);
            window.validationAccuracy.value = status.validation_accuracy;
          }
          
          // 如果有日志消息，也添加到日志中
          if (status.logs && status.logs.length > 0) {
            status.logs.forEach(log => {
              addLog(log.message, log.level || 'info');
            });
          }
          
          // 如果训练完成，停止轮询
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
            clearInterval(statusPollingInterval.value);
            
            if (status.status === 'completed') {
              evaluationResults.value = status.evaluation;
              completeTraining();
            } else if (status.status === 'failed') {
              addLog(t('errors.trainingFailed', { error: status.error || 'Unknown reason' }), 'error');
              showError(t('errors.trainingFailed', { error: status.error || 'Unknown reason' }));
              stopTraining();
            } else if (status.status === 'stopped') {
              addLog(t('train.trainingStoppedByServer'), 'info');
              stopTraining();
            }
          }
          
        } catch (error) {
          consecutiveFailures++;
          
          // 记录错误但继续尝试
          if (consecutiveFailures <= maxFailures) {
            addLog(t('errors.pollingError', { attempt: consecutiveFailures, max: maxFailures, error: error.message }), 'warning');
            
            // 失败时增加轮询间隔
            pollingInterval = Math.min(10000, pollingInterval * 1.5);
          } else {
            // 超过最大失败次数，切换到模拟模式
            addLog(t('errors.maxPollingFailures', { max: maxFailures }), 'error');
            addLog(t('train.switchingToSimulation'), 'info');
            
            clearInterval(statusPollingInterval.value);
            
            // 如果训练仍在进行中，切换到模拟模式
            if (isTraining.value) {
              simulateTraining();
            }
          }
        }
      }, pollingInterval);
    };
    
    
    // 完成训练（增强版，添加详细的训练总结）
    const completeTraining = () => {
      clearInterval(trainingTimer);
      
      // 添加到训练历史 - 保存前端字母ID用于显示
      const duration = (new Date() - startTime) / 1000;
      const accuracy = evaluationResults.value ? evaluationResults.value.accuracy : 0;
      const loss = evaluationResults.value ? evaluationResults.value.loss : 0;
      
      // 添加详细的训练总结
      addLog('====================================================');
      addLog(t('train.trainingCompleteSummary', {
        models: selectedModels.value.map(m => t(`models.${m}`)).join(', '),
        dataset: datasets.value.find(d => d.id === selectedDataset.value).name,
        duration: formatDuration(duration)
      }));
      addLog(t('train.finalMetrics', {
        accuracy: accuracy.toFixed(2),
        loss: loss.toFixed(4)
      }));
      
      // 如果有详细评估结果，显示更多信息
      if (evaluationResults.value) {
        if (evaluationResults.value.precision !== undefined) {
          addLog(t('train.precisionMetric', { value: (evaluationResults.value.precision * 100).toFixed(2) }));
        }
        if (evaluationResults.value.recall !== undefined) {
          addLog(t('train.recallMetric', { value: (evaluationResults.value.recall * 100).toFixed(2) }));
        }
        if (evaluationResults.value.f1Score !== undefined) {
          addLog(t('train.f1ScoreMetric', { value: (evaluationResults.value.f1Score * 100).toFixed(2) }));
        }
      }
      
      // 计算训练效率指标
      const epochs = parameters.value.epochs;
      const efficiency = accuracy / (epochs * duration / 3600); // 每小时每轮次的准确率提升
      const efficiencyRating = efficiency > 50 ? 'excellent' : efficiency > 30 ? 'good' : efficiency > 15 ? 'satisfactory' : 'room for improvement';
      addLog(t('train.trainingEfficiency', { rating: t(`efficiency.${efficiencyRating}`), value: efficiency.toFixed(2) }));
      
      addLog('====================================================');
      
      // 更新训练历史
      trainingHistory.value.unshift({
        id: Date.now(),
        date: new Date(),
        models: [...selectedModels.value], // 保存前端字母ID用于显示
        dataset: datasets.value.find(d => d.id === selectedDataset.value).name,
        duration: duration,
        accuracy: accuracy,
        loss: loss,
        parameters: { ...parameters.value },
        strategy: selectedStrategy.value
      });
      
      // 重新加载训练历史以确保与后端同步
      loadTrainingHistory();
      
      // 显示成功消息
      showSuccess(t('train.trainingCompletedSuccessfully'));
    };
    
    // 停止训练
    const stopTraining = async () => {
      if (!currentJobId.value) {
        addLog(t('errors.noActiveTraining'));
        return;
      }

      try {
        // 实际API调用停止训练
        await axios.post(`/api/training/stop/${currentJobId.value}`);
        addLog(t('train.trainingStopped'));
      } catch (error) {
        addLog(t('errors.trainingStopFailed', { error: error.message }));
      } finally {
        isTraining.value = false;
        clearInterval(trainingTimer);
        clearInterval(statusPollingInterval.value);
        
        // 关闭WebSocket连接
        if (websocketConnection.value) {
          websocketConnection.value.close();
          websocketConnection.value = null;
        }
        
        currentJobId.value = null;
      }
    };
    
    // 更新耗时
    const updateElapsedTime = () => {
      if (!startTime) return;
      
      const now = new Date();
      const diff = now - startTime;
      const hours = Math.floor(diff / 3600000);
      const minutes = Math.floor((diff % 3600000) / 60000);
      const seconds = Math.floor((diff % 60000) / 1000);
      
      elapsedTime.value = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };
    
    // 显示错误消息
    const showError = (message, type = 'general') => {
      errorState.value = {
        hasError: true,
        message,
        type
      };
      
      // 5秒后自动清除错误
      setTimeout(() => {
        errorState.value.hasError = false;
      }, 5000);
      
      addLog(`ERROR: ${message}`);
    };
    
    // 增强版模拟训练过程（在后端不可用时代替WebSocket连接）
    const simulateTraining = () => {
      let simulatedEpoch = 0;
      let batchCount = 0;
      const totalEpochs = parameters.value.epochs;
      const batchesPerEpoch = 10; // 每个epoch包含10个批次
      
      // 根据模型类型和训练策略初始化基础参数
      const initialParams = getInitialTrainingParams();
      let baseLoss = initialParams.loss;
      let baseAccuracy = initialParams.accuracy;
      let learningRate = parameters.value.learningRate;
      
      addLog(t('train.simulationStarted', { method: 'enhanced simulation' }));
      addLog(t('train.initialParameters', {
        loss: baseLoss.toFixed(4),
        accuracy: baseAccuracy.toFixed(2),
        lr: learningRate.toFixed(6)
      }));
      
      // 模拟训练过程
      const simulationInterval = setInterval(() => {
        if (!isTraining.value) {
          clearInterval(simulationInterval);
          addLog(t('train.simulationStopped'));
          return;
        }
        
        // 处理批次更新
        batchCount++;
        if (batchCount > batchesPerEpoch) {
          // 一个epoch完成
          batchCount = 1;
          simulatedEpoch++;
          
          // 学习率衰减
          learningRate *= 0.95;
          
          // 完成一个epoch的日志
          addLog(t('train.epochCompleted', {
            epoch: simulatedEpoch,
            totalEpochs: totalEpochs,
            loss: currentLoss.value.toFixed(4),
            accuracy: currentAccuracy.value.toFixed(2),
            lr: learningRate.toFixed(6)
          }));
          
          // 检查是否完成所有epoch
          if (simulatedEpoch >= totalEpochs) {
            clearInterval(simulationInterval);
            
            // 生成详细的模拟评估结果
            generateEnhancedEvaluationResults();
            
            completeTraining();
            return;
          }
        }
        
        // 计算当前进度
        const progress = ((simulatedEpoch * batchesPerEpoch + batchCount - 1) / (totalEpochs * batchesPerEpoch)) * 100;
        
        // 根据训练策略和模型类型调整学习曲线
        const progressFactor = getProgressFactor(simulatedEpoch, batchCount, totalEpochs, batchesPerEpoch);
        baseLoss = Math.max(0.01, baseLoss - (0.02 * progressFactor * getLossReductionMultiplier()));
        baseAccuracy = Math.min(99.9, baseAccuracy + (0.3 * progressFactor * getAccuracyIncreaseMultiplier()));
        
        // 添加随机波动
        const loss = baseLoss + (Math.random() * 0.03 - 0.015);
        const accuracy = baseAccuracy + (Math.random() * 2 - 1);
        
        // 更新训练状态
        currentEpoch.value = simulatedEpoch;
        trainingProgress.value = progress;
        currentLoss.value = loss;
        currentAccuracy.value = accuracy;
        
        // 每3个批次记录一次详细日志
        if (batchCount % 3 === 0) {
          // 根据所选模型添加特定的训练细节
          const modelSpecificDetails = getModelSpecificTrainingDetails();
          addLog(t('train.batchProgress', {
            batch: batchCount,
            totalBatches: batchesPerEpoch,
            loss: loss.toFixed(4),
            accuracy: accuracy.toFixed(2),
            details: modelSpecificDetails
          }));
        }
        
        // 模拟特殊训练事件
        if (Math.random() > 0.95) {
          simulateSpecialTrainingEvent(simulatedEpoch, batchCount);
        }
      }, 800); // 每0.8秒更新一次，比之前更快以提供更流畅的体验
    };
    
    // 加载训练历史
    const loadTrainingHistory = async () => {
      try {
        const response = await axios.get('/api/training/history');
        trainingHistory.value = response.data.map(item => ({
          ...item,
          // 从后端ID转换回前端字母ID
          models: item.models.map(backendId => {
            const model = availableModels.value.find(m => m.backendId === backendId);
            return model ? model.id : idToLetter(backendId);
          })
        }));
      } catch (error) {
        errorHandler.handleError(error, '加载训练历史失败');
        // 加载模拟历史数据
        trainingHistory.value = generateMockTrainingHistory();
      }
    };
    
    // 生成模拟训练历史
    const generateMockTrainingHistory = () => {
      const mockHistories = [];
      const modelCombinations = [
        ['B', 'J'], // 语言模型 + 知识库
        ['D', 'E', 'F'], // 视觉相关模型
        ['A', 'B', 'C', 'J'], // 基础交互组合
        ['A', 'G', 'F'], // 传感器分析
        ['B', 'K'] // 语言 + 编程
      ];
      
      for (let i = 0; i < 5; i++) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        
        const models = modelCombinations[i % modelCombinations.length];
        
        mockHistories.push({
          id: `mock_${Date.now()}_${i}`,
          date: date,
          models: models,
          dataset: datasets.value[i % datasets.value.length].name,
          duration: 3600 + Math.random() * 10800, // 1-4 hours
          accuracy: 75 + Math.random() * 20,
          parameters: {
            epochs: 10 + Math.floor(Math.random() * 20),
            batchSize: 32,
            learningRate: 0.001
          }
        });
      }
      
      return mockHistories;
    };
    
    // 添加日志（增强版，支持不同日志类型和格式）
    const addLog = (message, type = 'info') => {
      const timestamp = new Date().toLocaleTimeString();
      
      // 添加日志条目，包含类型信息用于样式化
      trainingLogs.value.push({
        timestamp,
        message,
        type // info, success, warning, error, debug
      });
      
      // 保持日志滚动到底部，优化滚动性能
      nextTick(() => {
        const logContainer = document.getElementById('training-logs');
        if (logContainer) {
          // 使用requestAnimationFrame确保滚动流畅
          requestAnimationFrame(() => {
            logContainer.scrollTop = logContainer.scrollHeight;
          });
        }
      });
    };
    
    // 显示警告
    const showWarning = (message) => {
      warningState.value = {
        hasWarning: true,
        message
      };
      
      setTimeout(() => {
        warningState.value.hasWarning = false;
      }, 8000);
    };
    
    // 显示成功消息
    const showSuccess = (message) => {
      successState.value = {
        hasSuccess: true,
        message
      };
      
      setTimeout(() => {
        successState.value.hasSuccess = false;
      }, 5000);
    };
    
    // 显示信息消息
    const showInfo = (message) => {
      infoState.value = {
        hasInfo: true,
        message
      };
      
      setTimeout(() => {
        infoState.value.hasInfo = false;
      }, 5000);
    };
    
    // 初始化
    const init = async () => {
      await loadAvailableModels();
      await loadTrainingHistory();
      
      // 设置默认选中的模型
      if (availableModels.value.length > 0) {
        // 确保默认选中的模型是有效的
        selectedModels.value = selectedModels.value.filter(modelId => 
          availableModels.value.some(m => m.id === modelId)
        );
        
        // 如果没有有效的选中模型，选择第一个
        if (selectedModels.value.length === 0 && availableModels.value.length > 0) {
          selectedModels.value = [availableModels.value[0].id];
        }
      }
    };
    
    // 组件挂载时初始化
    onMounted(() => {
      init();
    });
    
    // 组件卸载时清理
    onUnmounted(() => {
      clearInterval(trainingTimer);
      clearInterval(statusPollingInterval.value);
      
      if (websocketConnection.value) {
        websocketConnection.value.close();
      }
    });
    
    // 根据模型类型和训练策略获取初始训练参数
    const getInitialTrainingParams = () => {
      // 基础值
      let baseLoss = 0.6 + Math.random() * 0.3;
      let baseAccuracy = 50 + Math.random() * 20;
      
      // 根据模型类型调整
      if (selectedModels.value.some(m => ['D', 'J'].includes(m))) {
        // 文本和编程模型初始表现较好
        baseLoss *= 0.8;
        baseAccuracy *= 1.2;
      } else if (selectedModels.value.some(m => ['B', 'C'].includes(m))) {
        // 视觉和音频模型初始表现较差
        baseLoss *= 1.1;
        baseAccuracy *= 0.9;
      }
      
      // 根据训练策略调整
      if (selectedStrategy.value === 'knowledge_assisted' && enableKnowledgeAssist.value) {
        // 知识辅助策略提供更好的初始表现
        baseLoss *= 0.7;
        baseAccuracy *= 1.3;
      } else if (selectedStrategy.value === 'l1_regularization' || selectedStrategy.value === 'l2_regularization') {
        // 正则化策略初始表现略差但更稳定
        baseLoss *= 1.05;
        baseAccuracy *= 0.95;
      }
      
      // 联合训练调整
      if (trainingMode.value === 'joint' && selectedModels.value.length > 1) {
        // 联合训练初始复杂度更高
        baseLoss *= 1.1;
        baseAccuracy *= 0.9;
      }
      
      return { loss: baseLoss, accuracy: baseAccuracy };
    };
    
    // 获取进度因子（控制学习速度随时间的变化）
    const getProgressFactor = (epoch, batch, totalEpochs, batchesPerEpoch) => {
      // 整体进度百分比
      const overallProgress = ((epoch - 1) * batchesPerEpoch + batch) / (totalEpochs * batchesPerEpoch);
      
      // 学习率随时间衰减的曲线
      // 前期学习快，后期学习慢
      return Math.exp(-3 * overallProgress);
    };
    
    // 获取损失减少乘数（根据模型和策略调整）
    const getLossReductionMultiplier = () => {
      let multiplier = 1.0;
      
      // 根据训练策略调整
      if (selectedStrategy.value === 'adaptive_learning') {
        multiplier = 1.2;
      } else if (selectedStrategy.value === 'l1_regularization' || selectedStrategy.value === 'l2_regularization') {
        multiplier = 0.8;
      }
      
      // 联合训练调整
      if (trainingMode.value === 'joint' && selectedModels.value.length > 1) {
        multiplier = 1.15;
      }
      
      return multiplier;
    };
    
    // 获取准确率增加乘数（根据模型和策略调整）
    const getAccuracyIncreaseMultiplier = () => {
      let multiplier = 1.0;
      
      // 根据训练策略调整
      if (selectedStrategy.value === 'knowledge_assisted' && enableKnowledgeAssist.value) {
        multiplier = 1.2;
      } else if (selectedStrategy.value === 'adaptive_learning') {
        multiplier = 1.1;
      }
      
      // 联合训练调整
      if (trainingMode.value === 'joint' && selectedModels.value.length > 1) {
        multiplier = 1.25;
      }
      
      return multiplier;
    };
    
    // 获取特定模型的训练细节
    const getModelSpecificTrainingDetails = () => {
      const details = [];
      
      // 为不同类型的模型添加特定的训练细节
      if (selectedModels.value.includes('A')) {
        // 管理模型
        details.push(t('train.managementModelDetail', { value: (90 + Math.random() * 10).toFixed(1) }));
      }
      
      if (selectedModels.value.includes('D')) {
        // 文本模型
        details.push(t('train.textModelDetail', { value: (0.8 + Math.random() * 0.2).toFixed(3) }));
      }
      
      if (selectedModels.value.includes('J')) {
        // 编程模型
        details.push(t('train.programmingModelDetail', { value: (85 + Math.random() * 15).toFixed(1) }));
      }
      
      if (selectedModels.value.includes('E')) {
        // 知识模型
        details.push(t('train.knowledgeModelDetail', { value: (75 + Math.random() * 25).toFixed(1) }));
      }
      
      // 联合训练特定细节
      if (trainingMode.value === 'joint' && selectedModels.value.length > 1) {
        details.push(t('train.jointTrainingDetail', { value: (0.001 + Math.random() * 0.004).toFixed(4) }));
      }
      
      return details.join(', ');
    };
    
    // 模拟特殊训练事件
    const simulateSpecialTrainingEvent = (epoch, batch) => {
      const events = [
        { type: 'info', message: t('train.eventLearningRateAdjusted', { value: (parameters.value.learningRate * Math.random() * 0.5 + 0.75).toFixed(6) }) },
        { type: 'info', message: t('train.eventBatchNormalization', { value: (0.9 + Math.random() * 0.1).toFixed(3) }) },
        { type: 'info', message: t('train.eventDropoutAdjusted', { value: (0.2 + Math.random() * 0.2).toFixed(2) }) },
        { type: 'info', message: t('train.eventGradientClipping', { value: (1.0 + Math.random() * 1.0).toFixed(2) }) },
        { type: 'warning', message: t('train.eventMinorOverfitting') },
        { type: 'success', message: t('train.eventEarlyStoppingCheck') },
        { type: 'info', message: t('train.eventMomentumUpdated', { value: (0.8 + Math.random() * 0.15).toFixed(3) }) }
      ];
      
      // 随机选择一个事件
      const event = events[Math.floor(Math.random() * events.length)];
      addLog(event.message, event.type);
    };
    
    // 生成增强版评估结果
    const generateEnhancedEvaluationResults = () => {
      // 基于最终训练状态生成更真实的评估结果
      const finalAccuracy = currentAccuracy.value;
      const finalLoss = currentLoss.value;
      
      // 生成相关的评估指标
      const precision = Math.max(0.1, Math.min(0.99, finalAccuracy / 100 - 0.02 + Math.random() * 0.04));
      const recall = Math.max(0.1, Math.min(0.99, finalAccuracy / 100 - 0.01 + Math.random() * 0.03));
      const f1Score = Math.max(0.1, Math.min(0.99, 2 * (precision * recall) / (precision + recall + 0.0001)));
      
      // 根据模型类型调整指标
      if (selectedModels.value.some(m => ['D', 'J'].includes(m))) {
        // 文本和编程模型通常有更好的precision
        precision = Math.min(0.99, precision + 0.05);
      }
      
      if (selectedModels.value.some(m => ['B', 'E'].includes(m))) {
        // 视觉和知识模型通常有更好的recall
        recall = Math.min(0.99, recall + 0.05);
      }
      
      // 生成混淆矩阵（基于二分类问题）
      const truePositives = Math.floor(400 * precision * recall);
      const falsePositives = Math.floor(truePositives * (1 - precision) / precision);
      const falseNegatives = Math.floor(truePositives * (1 - recall) / recall);
      const trueNegatives = Math.floor(300 - falsePositives);
      
      const confusionMatrix = [
        [truePositives, falseNegatives],
        [falsePositives, trueNegatives]
      ];
      
      // 生成完整的评估结果
      evaluationResults.value = {
        accuracy: finalAccuracy,
        loss: finalLoss,
        precision: precision,
        recall: recall,
        f1Score: f1Score,
        confusionMatrix: confusionMatrix,
        
        // 添加更多评估指标
        auc: 0.8 + Math.random() * 0.15,
        trainingTime: (new Date() - startTime) / 1000,
        epochsCompleted: parameters.value.epochs,
        
        // 模型特定指标
        modelSpecificMetrics: getModelSpecificMetrics()
      };
    };
    
    // 获取模型特定的评估指标
    const getModelSpecificMetrics = () => {
      const metrics = {};
      
      // 为每种模型类型添加特定的评估指标
      if (selectedModels.value.includes('A')) {
        metrics.managementEfficiency = 85 + Math.random() * 15;
      }
      
      if (selectedModels.value.includes('D')) {
        metrics.perplexity = 5 + Math.random() * 15; // 语言模型的困惑度
      }
      
      if (selectedModels.value.includes('J')) {
        metrics.codeCompletionAccuracy = 70 + Math.random() * 25;
      }
      
      if (selectedModels.value.includes('E')) {
        metrics.knowledgeRetrievalPrecision = 75 + Math.random() * 20;
      }
      
      return metrics;
    };
    
    // 导出功能
    return {
      // 状态
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
      trainingHistory,
      datasetInput,
      errorState,
      successState,
      warningState,
      infoState,
      
      // 计算属性
      isModelRequired,
      isModelDisabled,
      getModelTooltip,
      
      // 方法
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
      
      // 工具函数
      formatDate(date) {
        return new Date(date).toLocaleDateString();
      },
      formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours}h ${minutes}m ${secs}s`;
      },
      
      // 占位符方法（需要在实际实现中完成）
      viewSession(id) {
        addLog(t('train.viewingSession', { id }));
        // 实际查看会话的实现
      },
      compareSession(id) {
        addLog(t('train.comparingSession', { id }));
        // 实际比较会话的实现
      }
    };
  },
  components: {
    TerminalWindow
  }
};
</script>

<style scoped>
.train-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: var(--font-family);
  color: var(--text-primary);
  background-color: var(--bg-primary);
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

/* Base styles */
.train-view {
  width: 100%;
  min-height: 100vh;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  padding: 0;
  margin: 0;
}

.status-messages {
  width: 100%;
  margin: 0;
  padding: 10px 0;
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
