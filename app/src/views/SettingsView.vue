<template>
  <div class="settings-container">
    
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
          <p class="stat-value">{{ models ? models.length : 0 }}</p>
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
      <!-- API Service Status -->
      <div class="api-service-status-section">
        <div class="api-service-card">
          <h3>Global API Service Status</h3>
          <div class="api-status-info" v-if="globalApiStatus">
            <p><strong>Status:</strong> <span :class="globalApiStatus.available ? 'status-online' : 'status-offline'">
              {{ globalApiStatus.available ? 'Available' : 'Unavailable' }}
            </span></p>
            <p v-if="globalApiStatus.services"><strong>Active Services:</strong> {{ globalApiStatus.services.join(', ') }}</p>
            <p v-if="globalApiStatus.timestamp"><strong>Last Check:</strong> {{ formatDateTime(globalApiStatus.timestamp) }}</p>
          </div>
          <button class="refresh-btn" @click="getApiServiceStatus" :disabled="checkingGlobalStatus">
            {{ checkingGlobalStatus ? 'Checking...' : 'Check Status' }}
          </button>
        </div>
      </div>
      
      <!-- Model Configuration Type Indicator -->
      <div class="model-configuration-type">
        <span class="config-type-badge" :class="model && model.externalConfig ? 'external' : 'local'">
          {{ model && model.externalConfig ? 'External API' : 'Local Model' }}
        </span>
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

      <!-- Model Configuration Section -->
      <div class="model-config-section">
        <h2>Model Configuration</h2>
        <div class="model-filters">
          <div class="filter-group">
            <label>Filter by Type:</label>
            <select v-model="modelFilterType">
              <option value="all">All Models</option>
              <option value="local">Local Models</option>
              <option value="external">External API Models</option>
            </select>
          </div>
          <div class="filter-group">
            <label>Filter by Status:</label>
            <select v-model="modelFilterStatus">
              <option value="all">All Statuses</option>
              <option value="running">Running</option>
              <option value="stopped">Stopped</option>
            </select>
          </div>
        </div>
        
        <!-- Bulk Operations Toolbar -->
        <div class="bulk-operations-toolbar" v-if="filteredModels && filteredModels.length > 0">
          <div class="bulk-selection-info">
            <label class="bulk-select-all">
              <input 
                type="checkbox" 
                :checked="isAllSelected" 
                @change="toggleSelectAll"
                :disabled="bulkActionInProgress"
              />
              <span>Select All ({{ selectedModelsCount }} selected)</span>
            </label>
            
            <div class="bulk-quick-select" v-if="selectedModelsCount === 0">
              <span>Quick select:</span>
              <button @click="selectModelsByStatus('running')" class="quick-select-btn">
                Running Models
              </button>
              <button @click="selectModelsByStatus('stopped')" class="quick-select-btn">
                Stopped Models
              </button>
              <button @click="selectModelsByStatus('error')" class="quick-select-btn">
                Error Models
              </button>
            </div>
          </div>
          
          <div class="bulk-actions" v-if="canPerformBulkAction">
            <span class="bulk-action-label">Perform action on {{ selectedModelsCount }} selected model(s):</span>
            <div class="bulk-action-buttons">
              <button 
                @click="performBulkAction('start')" 
                class="bulk-action-btn start"
                :disabled="bulkActionInProgress"
              >
                <span v-if="bulkActionInProgress">Processing...</span>
                <span v-else>Start Selected</span>
              </button>
              <button 
                @click="performBulkAction('stop')" 
                class="bulk-action-btn stop"
                :disabled="bulkActionInProgress"
              >
                <span v-if="bulkActionInProgress">Processing...</span>
                <span v-else>Stop Selected</span>
              </button>
              <button 
                @click="performBulkAction('restart')" 
                class="bulk-action-btn restart"
                :disabled="bulkActionInProgress"
              >
                <span v-if="bulkActionInProgress">Processing...</span>
                <span v-else>Restart Selected</span>
              </button>
              <button 
                @click="performBulkAction('delete')" 
                class="bulk-action-btn delete"
                :disabled="bulkActionInProgress"
              >
                <span v-if="bulkActionInProgress">Processing...</span>
                <span v-else>Delete Selected</span>
              </button>
            </div>
          </div>
        </div>
        
        <div class="models-grid">
          <div v-for="model in filteredModels" :key="model.id" class="model-card">
            <div class="model-header">
              <div class="model-select-checkbox">
                <input 
                  type="checkbox" 
                  :id="`model-select-${model.id}`"
                  :checked="selectedModels.includes(model.id)"
                  @change="toggleModelSelection(model.id)"
                  :disabled="bulkActionInProgress"
                />
              </div>
              <h3>{{ model.name }}</h3>
              <span class="model-type-badge" :class="model.source === 'local' ? 'local' : 'api'">
                {{ model.source === 'local' ? 'Local' : 'External API' }}
              </span>
            </div>
            <div class="model-info">
              <p><strong>ID:</strong> {{ model.id }}</p>
              <p><strong>Type:</strong> {{ model.type }}</p>
              <p><strong>Status:</strong> <span :class="`status ${model.status}`">{{ model.status }}</span></p>
              <p v-if="model.port"><strong>Port:</strong> {{ model.port }}</p>
            </div>
            
            <!-- Model Type Selection -->
            <div class="model-source-selector">
              <label for="source-{{ model.id }}">Model Source:</label>
              <select id="source-{{ model.id }}" v-model="model.source" @change="onSourceChange(model.id, model.source)">
                <option value="local">Local Model</option>
                <option value="external">External API</option>
              </select>
            </div>
            
            <!-- API Configuration Form -->
            <div v-if="model.source === 'external'" class="api-config-section">
              <h4>API Configuration</h4>
              <div class="api-config-form">
                <div class="form-group">
                  <label :for="`api-url-${model.id}`">API URL</label>
                  <input
                    :id="`api-url-${model.id}`"
                    v-model="model.api_url"
                    type="url"
                    placeholder="Enter API URL"
                  />
                </div>
                <div class="form-group">
                  <label :for="`api-key-${model.id}`">API Key</label>
                  <div class="password-input-wrapper">
                    <input
                      :id="`api-key-${model.id}`"
                      v-model="model.api_key"
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
                <div class="form-group">
                <label :for="`model-name-${model.id}`">Model Name</label>
                <input
                  :id="`model-name-${model.id}`"
                  v-model="model.model_name"
                  type="text"
                  placeholder="Enter Model Name"
                />
              </div>
              
              <!-- External Model Assistance Settings -->
              <div class="form-group">
                <div class="checkbox-wrapper">
                  <input
                    :id="`use-assistance-${model.id}`"
                    type="checkbox"
                    v-model="model.use_external_model_assistance"
                  />
                  <label :for="`use-assistance-${model.id}`">Enable External Model Assistance</label>
                </div>
                <div class="form-help-text">
                  Allow this model to use external API assistance for training improvement
                </div>
              </div>
              
              <div class="form-group" v-if="model.use_external_model_assistance">
                <label :for="`assistance-model-${model.id}`">Assistance Model</label>
                <select :id="`assistance-model-${model.id}`" v-model="model.external_model_id">
                  <option value="">Select external model</option>
                  <option v-for="extModel in models" :key="extModel.id" :value="extModel.id" :disabled="extModel.id === model.id">
                    {{ extModel.name }} ({{ extModel.source === 'external' ? 'External' : 'Local' }})
                  </option>
                </select>
                <div class="form-help-text">
                  Select which external model to use for assistance during training
                </div>
              </div>
              <div class="form-group">
                <label :for="`api-source-${model.id}`">API Provider</label>
                <select :id="`api-source-${model.id}`" v-model="model.sourceProvider" @change="applyApiTemplate(model.id, model.sourceProvider)">
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google AI</option>
                  <option value="aws">AWS</option>
                  <option value="azure">Azure</option>
                  <option value="huggingface">Hugging Face</option>
                  <option value="custom">Custom</option>
                  <option value="deepseek">DeepSeek</option>
                  <option value="siliconflow">SiliconFlow</option>
                  <option value="zhipu">Zhipu AI</option>
                  <option value="baidu">Baidu ERNIE</option>
                  <option value="alibaba">Alibaba Qwen</option>
                  <option value="moonshot">Moonshot</option>
                  <option value="yi">Yi</option>
                  <option value="tencent">Tencent Hunyuan</option>
                  <option value="ollama">Ollama</option>
                </select>
              </div>
                <div class="api-actions">
                  <button 
                    class="test-btn" 
                    @click="testApiConnection(model.id)"
                    :disabled="isTestingConnection(model.id)"
                  >
                    {{ isTestingConnection(model.id) ? 'Testing...' : 'Test Connection' }}
                  </button>
                  <button 
                    class="test-btn" 
                    @click="getModelApiStatus(model.id)"
                    :disabled="model.status === 'running'"
                  >
                    Check API Status
                  </button>
                  <button 
                    class="save-btn"
                    @click="saveModelConfig(model.id)"
                    :disabled="isSavingSettings(model.id)"
                  >
                    {{ isSavingSettings(model.id) ? 'Saving...' : 'Save' }}
                  </button>
                </div>
                <div v-if="testResults[model.id]" class="test-results">
                  <div v-if="testResults[model.id].success" class="success-message">
                    {{ testResults[model.id].message }}
                  </div>
                  <div v-else class="error-message">
                    {{ testResults[model.id].message }}
                  </div>
                </div>
                <!-- API Status Display -->
                <div v-if="model.apiStatus" class="test-results">
                  <div class="info-message">
                    <p><strong>API Status:</strong> {{ model.apiStatus.connected ? 'Connected' : 'Disconnected' }}</p>
                    <p v-if="model.apiStatus.version"><strong>API Version:</strong> {{ model.apiStatus.version }}</p>
                    <p v-if="model.apiStatus.latency"><strong>Latency:</strong> {{ model.apiStatus.latency }}ms</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- System Health Monitor -->
      <div class="health-monitor-section">
        <h2>System Health Monitoring</h2>
        <SystemHealthMonitor />
      </div>

      <!-- Hardware Configuration Section -->
      <div class="hardware-config-section">
        <h2>Hardware Configuration</h2>
        
        <!-- Camera Configuration -->
        <div class="camera-config">
          <h3>Camera Configuration</h3>
          <div class="form-row">
            <div class="form-group">
              <label for="camera-count">Number of Cameras</label>
              <input
                id="camera-count"
                v-model.number="hardwareConfig.cameraCount"
                type="number"
                min="1"
                max="10"
                @change="updateCameraConfig"
              />
            </div>
            <div class="form-group">
              <label for="camera-resolution">Default Resolution</label>
              <select id="camera-resolution" v-model="hardwareConfig.defaultResolution">
                <option value="640x480">640x480</option>
                <option value="1280x720">1280x720</option>
                <option value="1920x1080">1920x1080</option>
                <option value="3840x2160">3840x2160</option>
              </select>
            </div>
          </div>
          
          <!-- Individual Camera Configuration -->
          <div v-for="camera in hardwareConfig.cameras" :key="camera.id" class="camera-item">
            <h4>Camera {{ camera.id }}</h4>
            <div class="form-row">
              <div class="form-group">
                <label :for="`camera-name-${camera.id}`">Camera Name</label>
                <input
                  :id="`camera-name-${camera.id}`"
                  v-model="camera.name"
                  type="text"
                  placeholder="Camera description"
                />
              </div>
              <div class="form-group">
                <label :for="`camera-type-${camera.id}`">Camera Type</label>
                <select :id="`camera-type-${camera.id}`" v-model="camera.type">
                  <option value="mono">Monocular</option>
                  <option value="stereo">Stereo (Binocular)</option>
                  <option value="depth">Depth Camera</option>
                  <option value="thermal">Thermal Camera</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label :for="`camera-device-${camera.id}`">Device ID</label>
                <input
                  :id="`camera-device-${camera.id}`"
                  v-model="camera.deviceId"
                  type="text"
                  placeholder="Enter Device ID"
                />
              </div>
              <div class="form-group">
                <label :for="`camera-fps-${camera.id}`">Frame Rate (FPS)</label>
                <input
                  :id="`camera-fps-${camera.id}`"
                  v-model.number="camera.fps"
                  type="number"
                  min="1"
                  max="60"
                />
              </div>
            </div>
            <div v-if="camera.type === 'stereo'" class="stereo-config">
              <h5>Stereo Camera Configuration</h5>
              <div class="form-row">
                <div class="form-group">
                  <label :for="`camera-baseline-${camera.id}`">Baseline (mm)</label>
                  <input
                    :id="`camera-baseline-${camera.id}`"
                    v-model.number="camera.baseline"
                    type="number"
                    min="50"
                    max="300"
                    step="1"
                  />
                </div>
                <div class="form-group">
                  <label :for="`camera-focal-${camera.id}`">Focal Length (mm)</label>
                  <input
                    :id="`camera-focal-${camera.id}`"
                    v-model.number="camera.focalLength"
                    type="number"
                    min="2"
                    max="50"
                    step="0.1"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- External Device Configuration -->
        <div class="device-config">
          <h3>External Device Configuration</h3>
          <div class="form-row">
            <div class="form-group">
              <label for="device-interface">Default Interface</label>
              <select id="device-interface" v-model="hardwareConfig.defaultInterface">
                <option value="usb">USB</option>
                <option value="serial">Serial</option>
                <option value="bluetooth">Bluetooth</option>
                <option value="wifi">WiFi</option>
                <option value="ethernet">Ethernet</option>
              </select>
            </div>
            <div class="form-group">
              <label for="device-baudrate">Default Baud Rate</label>
              <select id="device-baudrate" v-model="hardwareConfig.defaultBaudRate">
                <option value="9600">9600</option>
                <option value="19200">19200</option>
                <option value="38400">38400</option>
                <option value="57600">57600</option>
                <option value="115200">115200</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Serial Port Configuration -->
        <div class="serial-config">
          <h3>Serial Port Configuration</h3>
          <div class="form-row">
            <div class="form-group">
              <label for="serial-device-id">Device ID</label>
              <input
                id="serial-device-id"
                v-model="hardwareConfig.serialDeviceId"
                type="text"
                placeholder="Enter Device ID"
              />
            </div>
            <div class="form-group">
              <label for="serial-port">Serial Port</label>
              <select 
                id="serial-port" 
                v-model="hardwareConfig.selectedSerialPort"
                :disabled="hardwareConfig.serialConnectionStatus === 'connecting' || hardwareConfig.serialConnectionStatus === 'connected'"
              >
                <option value="">Select a serial port</option>
                <option v-for="port in hardwareConfig.serialPorts" :key="port.port" :value="port.port">
                  {{ port.port }} ({{ port.description || 'Unknown device' }})
                </option>
              </select>
              <button 
                class="refresh-btn" 
                @click="scanSerialPorts"
                :disabled="hardwareConfig.serialConnectionStatus === 'connecting' || hardwareConfig.serialConnectionStatus === 'connected'"
              >
                Refresh Ports
              </button>
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="serial-baudrate">Baud Rate</label>
              <select 
                id="serial-baudrate" 
                v-model="hardwareConfig.serialBaudRate"
                :disabled="hardwareConfig.serialConnectionStatus === 'connecting' || hardwareConfig.serialConnectionStatus === 'connected'"
              >
                <option value="9600">9600</option>
                <option value="19200">19200</option>
                <option value="38400">38400</option>
                <option value="57600">57600</option>
                <option value="115200">115200</option>
                <option value="230400">230400</option>
                <option value="460800">460800</option>
                <option value="921600">921600</option>
              </select>
            </div>
            <div class="form-group">
              <label for="serial-status">Connection Status</label>
              <div class="connection-status">
                <span :class="'status-' + hardwareConfig.serialConnectionStatus">
                  {{ hardwareConfig.serialConnectionStatus === 'disconnected' ? 'Disconnected' : 
                     hardwareConfig.serialConnectionStatus === 'connecting' ? 'Connecting...' : 
                     hardwareConfig.serialConnectionStatus === 'connected' ? 'Connected' : 'Error' }}
                </span>
              </div>
            </div>
          </div>
          <div class="serial-actions">
            <button 
              class="hardware-btn connect-btn" 
              @click="connectSerialPort"
              :disabled="!hardwareConfig.selectedSerialPort || !hardwareConfig.serialDeviceId || 
                       hardwareConfig.serialConnectionStatus === 'connecting' || hardwareConfig.serialConnectionStatus === 'connected'"
            >
              {{ hardwareConfig.serialConnectionStatus === 'connecting' ? 'Connecting...' : 'Connect' }}
            </button>
            <button 
              class="hardware-btn disconnect-btn" 
              @click="disconnectSerialPort"
              :disabled="hardwareConfig.serialConnectionStatus !== 'connected'"
            >
              Disconnect
            </button>
          </div>
          <div v-if="hardwareConfig.serialConnectionError" class="error-message">
            {{ hardwareConfig.serialConnectionError }}
          </div>
          
          <!-- Sensor Devices -->
          <div class="sensor-devices">
            <h4>Sensor Devices</h4>
            <div v-for="sensor in hardwareConfig.sensors" :key="sensor.id" class="sensor-item">
              <h5>Sensor {{ sensor.id }}</h5>
              <div class="form-row">
                <div class="form-group">
                  <label :for="`sensor-type-${sensor.id}`">Sensor Type</label>
                  <select :id="`sensor-type-${sensor.id}`" v-model="sensor.type">
                    <option value="temperature">Temperature</option>
                    <option value="humidity">Humidity</option>
                    <option value="accelerometer">Accelerometer</option>
                    <option value="gyroscope">Gyroscope</option>
                    <option value="pressure">Pressure</option>
                    <option value="distance">Distance</option>
                    <option value="infrared">Infrared</option>
                    <option value="smoke">Smoke</option>
                    <option value="light">Light</option>
                    <option value="taste">Taste</option>
                  </select>
                </div>
                <div class="form-group">
                  <label :for="`sensor-port-${sensor.id}`">Port/Address</label>
                  <input
                    :id="`sensor-port-${sensor.id}`"
                    v-model="sensor.port"
                    type="text"
                    placeholder="Enter Port/Address"
                  />
                </div>
              </div>
            </div>
          </div>

          <!-- Actuator Devices -->
          <div class="actuator-devices">
            <h4>Actuator Devices</h4>
            <div v-for="actuator in hardwareConfig.actuators" :key="actuator.id" class="actuator-item">
              <h5>Actuator {{ actuator.id }}</h5>
              <div class="form-row">
                <div class="form-group">
                  <label :for="`actuator-type-${actuator.id}`">Actuator Type</label>
                  <select :id="`actuator-type-${actuator.id}`" v-model="actuator.type">
                    <option value="motor">Motor</option>
                    <option value="servo">Servo</option>
                    <option value="relay">Relay</option>
                    <option value="solenoid">Solenoid</option>
                    <option value="valve">Valve</option>
                    <option value="pump">Pump</option>
                  </select>
                </div>
                <div class="form-group">
                  <label :for="`actuator-port-${actuator.id}`">Port/Address</label>
                  <input
                    :id="`actuator-port-${actuator.id}`"
                    v-model="actuator.port"
                    type="text"
                    placeholder="Enter Port/Address"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Hardware Actions -->
        <div class="hardware-actions">
          <button class="hardware-btn" @click="testHardwareConnections" :disabled="isTestingHardware">
            {{ isTestingHardware ? 'Testing...' : 'Test Hardware Connections' }}
          </button>
          <button class="hardware-btn" @click="saveHardwareConfig" :disabled="isSavingHardware">
            {{ isSavingHardware ? 'Saving...' : 'Save Hardware Configuration' }}
          </button>
          <button class="hardware-btn" @click="resetHardwareConfig">
            Reset Hardware Configuration
          </button>
        </div>
      </div>

      <!-- Performance Monitoring Section -->
      <div class="performance-monitoring-section">
        <h2>System Performance Monitoring</h2>
        
        <div class="monitoring-controls">
          <button 
            @click="startPerformanceMonitoring" 
            class="monitoring-btn start"
            :disabled="isMonitoringActive"
          >
            Start Monitoring
          </button>
          <button 
            @click="stopPerformanceMonitoring" 
            class="monitoring-btn stop"
            :disabled="!isMonitoringActive"
          >
            Stop Monitoring
          </button>
          <span class="monitoring-status" :class="{ active: isMonitoringActive }">
            {{ isMonitoringActive ? 'Monitoring Active' : 'Monitoring Inactive' }}
          </span>
        </div>
        
        <div class="performance-metrics-grid" v-if="performanceMetrics.lastUpdated">
          <div class="metric-card">
            <div class="metric-value">{{ performanceMetrics.cpuUsage.toFixed(1) }}%</div>
            <div class="metric-label">CPU Usage</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ performanceMetrics.memoryUsage.toFixed(1) }}%</div>
            <div class="metric-label">Memory Usage</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ performanceMetrics.diskUsage.toFixed(1) }}%</div>
            <div class="metric-label">Disk Usage</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ (performanceMetrics.networkIn / 1024).toFixed(2) }} KB/s</div>
            <div class="metric-label">Network In</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">{{ (performanceMetrics.networkOut / 1024).toFixed(2) }} KB/s</div>
            <div class="metric-label">Network Out</div>
          </div>
        </div>
        
        <div class="model-response-times" v-if="Object.keys(performanceMetrics.modelResponseTimes).length > 0">
          <h3>Model Response Times</h3>
          <div class="response-times-grid">
            <div v-for="(time, modelId) in performanceMetrics.modelResponseTimes" :key="modelId" class="response-time-card">
              <div class="model-id">{{ modelId }}</div>
              <div class="response-time">{{ time.toFixed(2) }} ms</div>
            </div>
          </div>
        </div>
        
        <div class="metric-timestamp" v-if="performanceMetrics.lastUpdated">
          Last updated: {{ formatDateTime(performanceMetrics.lastUpdated) }}
        </div>
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
        <button class="batch-btn" @click="checkAllModelsHealth" :disabled="!models || models.length === 0 || isCheckingAllHealth">
          {{ isCheckingAllHealth ? 'Checking...' : 'Check All Models Health' }}
        </button>
        <button class="batch-btn" @click="restartSystem" :disabled="isRestartingSystem">
          Restart System
        </button>
      </div>

      <!-- Models List -->
      <div v-if="models && models.length > 0" class="models-list">
        <h2>Models</h2>
        <div v-for="model in models" :key="model.id" class="model-card">
          <div class="model-header">
            <div class="model-info">
              <h4>{{ model.name }}</h4>
              <div class="model-meta">
                <select v-model="model.source" class="model-config-type-select" @change="onSourceChange(model.id, model.source)">
                  <option value="local">Local</option>
                  <option value="external">External API</option>
                </select>
                <span class="model-type-badge" :class="model.type && typeof model.type === 'string' && model.type.toLowerCase().includes('api') ? 'api' : 'local'">
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
            <!-- Control Actions Group -->
            <div class="model-actions-group">
              <div class="model-actions-group-title">Control Actions</div>
              <div class="model-actions-buttons">
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
              </div>
            </div>
            
            <!-- Status Actions Group -->
            <div class="model-actions-group">
              <div class="model-actions-group-title">Status Actions</div>
              <div class="model-actions-buttons">
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
              </div>
            </div>
            
            <!-- Advanced Actions Group -->
            <div class="model-actions-group">
              <div class="model-actions-group-title">Advanced Actions</div>
              <div class="model-actions-buttons">
                <button
                  class="control-btn train-btn"
                  @click="openTrainModal(model)"
                  :disabled="isOperating(model.id)"
                >
                  Train from Scratch
                </button>
                <button
                  class="remove-btn"
                  @click="removeModel(model.id)"
                  :disabled="isOperating(model.id)"
                >
                  Remove
                </button>
              </div>
            </div>
          </div>

          <!-- API Configuration -->
          <div class="api-config-section">
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
                <div class="form-group">
                  <label for="api-url-{{ model.id }}">API URL</label>
                  <input
                    :id="'api-url-' + model.id"
                    v-model="model.apiUrl"
                    type="text"
                    placeholder="Enter API URL"
                  />
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label for="model-name-{{ model.id }}">Model Name</label>
                  <input
                    :id="'model-name-' + model.id"
                    v-model="model.modelName"
                    type="text"
                    placeholder="Enter Model Name"
                  />
                </div>
                <div class="form-group">
                  <label for="api-provider-{{ model.id }}">API Provider</label>
                <select
                  :id="'api-provider-' + model.id"
                  v-model="model.sourceProvider"
                  @change="applyApiTemplate(model.id, model.sourceProvider)"
                >
                  <option value="custom">Custom</option>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google AI</option>
                  <option value="huggingface">Hugging Face</option>
                  <option value="mistral">Mistral AI</option>
                  <option value="deepseek">DeepSeek</option>
                  <option value="siliconflow">SiliconFlow</option>
                  <option value="zhipu">Zhipu AI</option>
                  <option value="baidu">Baidu ERNIE</option>
                  <option value="alibaba">Alibaba Qwen</option>
                  <option value="moonshot">Moonshot</option>
                  <option value="yi">Yi</option>
                  <option value="tencent">Tencent Hunyuan</option>
                  <option value="ollama">Ollama</option>
                </select>
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label for="rate-limit-{{ model.id }}">Rate Limit (requests per minute)</label>
                  <input
                    :id="'rate-limit-' + model.id"
                    v-model="model.rateLimit"
                    type="number"
                    min="1"
                    max="10000"
                    placeholder="1000"
                  />
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
                  :disabled="!model.apiKey || !model.apiUrl || !model.modelName || isTestingConnection(model.id)"
                >
                  {{ isTestingConnection(model.id) ? 'Testing...' : 'Test Connection' }}
                </button>
                <button
                  class="test-btn"
                  @click="saveSettings(model.id)"
                  :disabled="!model.apiKey || !model.apiUrl || !model.modelName || isSavingSettings(model.id)"
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
  
  <!-- Training Modal -->
  <div v-if="showTrainModal" class="modal-overlay" @click="closeTrainModal">
    <div class="modal-content" @click.stop>
      <div class="modal-header">
        <h3>Train Model from Scratch: {{ selectedModelForTraining?.name }}</h3>
        <button class="close-btn" @click="closeTrainModal">&times;</button>
      </div>
      <div class="modal-body">
        <!-- Dataset Selection -->
        <div class="form-group">
          <label for="dataset-select">Select Dataset</label>
          <select id="dataset-select" v-model="selectedDataset" required>
            <option value="">Select a dataset</option>
            <option v-for="dataset in availableDatasets" :key="dataset.id" :value="dataset.id">
              {{ dataset.name }} ({{ dataset.size }} samples)
            </option>
          </select>
        </div>
        
        <!-- Training Parameters -->
        <div class="training-params">
          <h4>Training Parameters</h4>
          <div class="form-row">
            <div class="form-group">
              <label for="epochs">Epochs</label>
              <input id="epochs" v-model.number="trainingParams.epochs" type="number" min="1" max="100" />
            </div>
            <div class="form-group">
              <label for="batch-size">Batch Size</label>
              <input id="batch-size" v-model.number="trainingParams.batchSize" type="number" min="1" max="1024" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label for="learning-rate">Learning Rate</label>
              <input id="learning-rate" v-model.number="trainingParams.learningRate" type="number" min="0.00001" max="0.1" step="0.00001" />
            </div>
            <div class="form-group">
              <label for="validation-split">Validation Split</label>
              <input id="validation-split" v-model.number="trainingParams.validationSplit" type="number" min="0.01" max="0.5" step="0.01" />
            </div>
          </div>
        </div>
        
        <!-- Training Progress -->
        <div v-if="trainingStatus !== 'idle'" class="training-progress">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: trainingProgress + '%' }"></div>
          </div>
          <div class="progress-info">
            <span>{{ trainingProgress }}%</span>
            <span class="training-status" :class="trainingStatus">{{ trainingStatus.toUpperCase() }}</span>
          </div>
          <div v-if="trainingMessage" class="training-message">{{ trainingMessage }}</div>
        </div>
      </div>
      <div class="modal-footer">
        <button 
          class="btn btn-primary" 
          @click="startTraining" 
          :disabled="!selectedDataset || trainingStatus === 'training'"
        >
          {{ trainingStatus === 'training' ? 'Training...' : 'Start Training' }}
        </button>
        <button 
          class="btn btn-secondary" 
          @click="stopTraining" 
          :disabled="trainingStatus !== 'training'"
        >
          Stop Training
        </button>
        <button 
          class="btn btn-cancel" 
          @click="closeTrainModal" 
          :disabled="trainingStatus === 'training'"
        >
          Cancel
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { 
  handleEnhancedError,
  handleApiError,
  handleNetworkError,
  handleConfigError,
  handleValidationError,
  logInfo,
  logWarning,
  logSuccess 
} from '@/utils/enhancedErrorHandler'
import { notify } from '@/plugins/notification'
import { Model, NewModel, MODEL_TYPES, MODEL_STATUS, MODEL_PORT_CONFIG, createDefaultModel, isValidModelId, isValidPort, isApiModelType } from '@/utils/modelTypes'
import testNotifications from '@/utils/testNotifications'
import api from '@/utils/api'
import { performDataLoad, performDataOperation } from '@/utils/operationHelpers'
import SystemHealthMonitor from '@/components/SystemHealthMonitor.vue'

export default {
  name: 'SettingsView',
  components: {
    SystemHealthMonitor
  },
  setup() {
    // State
    const isMounted = ref(true)
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
    const globalApiStatus = ref(null)
    const checkingGlobalStatus = ref(false)
    const isCheckingAllHealth = ref(false)
    
    // Train from scratch related states
    const showTrainModal = ref(false)
    const selectedModelForTraining = ref(null)
    const trainingProgress = ref(0)
    const trainingStatus = ref('idle') // idle, training, completed, error
    const trainingMessage = ref('')
    const availableDatasets = ref([])
    const selectedDataset = ref('')
    const trainingParams = ref({
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      validationSplit: 0.2
    })

    // Hardware configuration states
    const isTestingHardware = ref(false)
    const isSavingHardware = ref(false)
    const hardwareConfig = ref({
      // Serial port related configuration
      serialPorts: [],
      selectedSerialPort: '',
      serialBaudRate: '9600',
      serialDeviceId: '',
      serialConnectionStatus: 'disconnected', // disconnected, connecting, connected, error
      serialConnectionError: '',
      cameraCount: 0,
      defaultResolution: '1280x720',
      defaultInterface: 'usb',
      defaultBaudRate: '9600',
      cameras: [],
      sensors: [],
      actuators: []
    })

    // Data - will be populated from real API
    const models = ref([])
    const newModel = ref({
      id: '',
      name: '',
      type: '',
      port: 0
    })
    
    // Model configuration filter states
    const modelFilterType = ref('all')
    const modelFilterStatus = ref('all')
    
    // Batch operation states
    const selectedModels = ref(new Set())
    const bulkActionInProgress = ref(false)
    
    // Performance monitoring states
    const performanceMetrics = ref({
      cpuUsage: 0,
      memoryUsage: 0,
      diskUsage: 0,
      networkIn: 0,
      networkOut: 0,
      modelResponseTimes: {},
      lastUpdated: null
    })
    const monitoringInterval = ref(null)
    const isMonitoringActive = ref(false)
    
    // Enhanced error handling helper
    const handleError = (error, context = 'system operation') => {
      // Determine error type based on context and error object
      const errorString = String(error).toLowerCase()
      const errorMessage = error.message ? error.message.toLowerCase() : ''
      
      // Check for network errors
      if (error.code === 'ECONNREFUSED' || 
          error.code === 'ECONNABORTED' || 
          error.code === 'ETIMEDOUT' ||
          errorMessage.includes('network') ||
          errorMessage.includes('connection') ||
          errorMessage.includes('timeout')) {
        return handleNetworkError(error, context)
      }
      
      // Check for API errors
      if (error.response || errorMessage.includes('api') || context.includes('API')) {
        return handleApiError(error, context)
      }
      
      // Check for configuration errors
      if (errorMessage.includes('config') || 
          errorMessage.includes('setting') || 
          errorMessage.includes('port') ||
          context.includes('configuration') ||
          context.includes('settings')) {
        return handleConfigError(error, context)
      }
      
      // Check for validation errors
      if (errorMessage.includes('validation') ||
          errorMessage.includes('invalid') ||
          errorMessage.includes('required')) {
        return handleValidationError(error, context)
      }
      
      // Default to enhanced error handling
      return handleEnhancedError(error, context)
    }

    // Computed
    const modelTypes = computed(() => MODEL_TYPES)
    const activeModelsCount = computed(() => {
      return models.value ? models.value.filter(model => model.isActive).length : 0
    })
    const runningModelsCount = computed(() => {
      return models.value ? models.value.filter(model => model.status === 'running').length : 0
    })
    const apiModelsCount = computed(() => {
      return models.value ? models.value.filter(model => {
        const typeStr = model.type && typeof model.type === 'string' ? model.type.toLowerCase() : ''
        return typeStr.includes('api') || model.source === 'external'
      }).length : 0
    })
    const filteredModels = computed(() => {
      if (!models.value) {
        return []
      }
      
      let filtered = models.value
      
      // Filter by type
      if (modelFilterType.value === 'local') {
        filtered = filtered.filter(model => (model.source || 'local') === 'local')
      } else if (modelFilterType.value === 'external') {
        filtered = filtered.filter(model => (model.source || 'local') === 'external')
      }
      
      // Filter by status
      if (modelFilterStatus.value !== 'all') {
        filtered = filtered.filter(model => model.status === modelFilterStatus.value)
      }
      
      // Debug logging in development
      if (import.meta.env.DEV && filtered.length !== models.value.length) {
        console.log('Models filtered:', {
          total: models.value.length,
          filtered: filtered.length,
          filterType: modelFilterType.value,
          filterStatus: modelFilterStatus.value,
          models: models.value.map(m => ({ id: m.id, source: m.source, status: m.status }))
        })
      }
      
      return filtered
    })
    const canStartAll = computed(() => {
      return models.value ? models.value.some(model => model.status !== 'running') : false
    })
    const canStopAll = computed(() => {
      return models.value ? models.value.some(model => model.status === 'running') : false
    })
    const canRestartAll = computed(() => {
      return models.value ? models.value.length > 0 : false
    })
    
    // Batch operation computed properties
    const selectedModelsCount = computed(() => {
      return selectedModels.value.size
    })
    
    const canPerformBulkAction = computed(() => {
      return selectedModels.value.size > 0 && !bulkActionInProgress.value
    })
    
    const isAllSelected = computed(() => {
      if (filteredModels.value.length === 0) return false
      return filteredModels.value.every(model => selectedModels.value.has(model.id))
    })
    
    const filteredModelIds = computed(() => {
      return filteredModels.value.map(model => model.id)
    })

    // Methods
    
    // Generic model operation function
    const performModelOperation = async (modelId, operation, options = {}) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return false
      }
      
      const {
        apiEndpoint = `/api/models/${modelId}/${operation}`,
        method = 'post',
        requestData = {},
        successStatus = operation === 'start' ? 'running' : 
                       operation === 'stop' ? 'stopped' : 
                       operation === 'restart' ? 'running' : model.status,
        errorStatus = operation === 'start' ? 'stopped' : 
                      operation === 'stop' ? 'running' : model.status,
        successMessage = `Model ${operation}ed successfully`,
        errorMessage = `Failed to ${operation} model`,
        showSuccess = true,
        showError = true
      } = options
      
      operatingModels.value.add(modelId)
      try {
        // Update local state first for better UX
        model.status = `${operation}ing`
        
        // Use api instance for the request
        await api[method](apiEndpoint, requestData)
        
        // Update local state on success
        model.status = successStatus
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        if (showSuccess) {
          notify.success(successMessage)
        }
        return true
      } catch (error) {
        handleError(error, `${operation.charAt(0).toUpperCase() + operation.slice(1)} Model`)
        // Revert status on error
        model.status = errorStatus
        if (showError) {
          notify.error(`${errorMessage}: ${error.message}`)
        }
        return false
      } finally {
        operatingModels.value.delete(modelId)
      }
    }

    // Generic batch model operation function
    const performBatchModelOperation = async (operation, options = {}) => {
      const {
        apiEndpoint = `/api/models/${operation}`,
        method = 'post',
        requestData = {},
        filterCondition = null,
        successStatus = operation === 'start-all' ? 'running' : 
                       operation === 'stop-all' ? 'stopped' : 
                       operation === 'restart-all' ? 'running' : 'unknown',
        errorStatus = operation === 'start-all' ? 'stopped' : 
                      operation === 'stop-all' ? 'running' : 'stopped',
        operationStatus = operation.includes('start') ? 'starting' : 
                         operation.includes('stop') ? 'stopping' : 'restarting',
        confirmMessage = `Are you sure you want to ${operation.replace('-', ' ')} all models?`,
        successMessage = `All models ${operation.replace('-', ' ')}ed successfully`,
        errorMessage = `Failed to ${operation.replace('-', ' ')} all models`,
        noModelsMessage = operation === 'restart-all' ? 'No models to restart' : 
                         `All models are already ${operation.includes('stop') ? 'stopped' : 'running'}`,
        showSuccess = true,
        showError = true
      } = options

      // Confirmation dialog
      if (!confirm(confirmMessage)) {
        return false
      }

      // Filter models based on operation
      let targetModels = []
      if (filterCondition) {
        targetModels = models.value.filter(filterCondition)
      } else if (operation === 'start-all') {
        targetModels = models.value.filter(model => model.status !== 'running')
      } else if (operation === 'stop-all') {
        targetModels = models.value.filter(model => model.status === 'running')
      } else if (operation === 'restart-all') {
        targetModels = models.value
      }

      // Check if there are models to operate on
      if (targetModels.length === 0) {
        notify.info(noModelsMessage)
        return false
      }

      try {
        // Add all target models to operating set
        targetModels.forEach(model => {
          operatingModels.value.add(model.id)
          model.status = operationStatus
        })

        // Use api instance for the request
        await api[method](apiEndpoint, requestData)

        // Update local state on success
        targetModels.forEach(model => {
          model.status = successStatus
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        
        if (showSuccess) {
          notify.success(successMessage)
        }
        return true
      } catch (error) {
        const actionName = operation.split('-').map(word => 
          word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ')
        handleError(error, `${actionName} All Models`)
        
        // Revert status on error
        targetModels.forEach(model => {
          model.status = errorStatus
        })
        
        if (showError) {
          notify.error(`${errorMessage}: ${error.message}`)
        }
        return false
      } finally {
        // Remove all target models from operating set
        targetModels.forEach(model => {
          operatingModels.value.delete(model.id)
        })
      }
    }

    const loadModels = async () => {
      loading.value = true
      try {
        if (import.meta.env.DEV) {
          console.log('loadModels: Starting API call to /api/models/getAll')
        }
        
        const response = await api.models.getAll()
        
        if (import.meta.env.DEV) {
          console.log('loadModels: Full API response structure:', {
            status: response.status,
            data: response.data,
            dataType: typeof response.data,
            isArray: Array.isArray(response.data),
            keys: response.data ? Object.keys(response.data) : []
          })
        }
        
        let apiModels = []
        
        // Handle multiple possible response structures
        if (response.data) {
          // Structure 1: {status: 'success', data: [...]}
          if (response.data.status === 'success' && Array.isArray(response.data.data)) {
            apiModels = response.data.data
            if (import.meta.env.DEV) {
              console.log('loadModels: Using response.data.data structure, count:', apiModels.length)
            }
          }
          // Structure 2: Direct array
          else if (Array.isArray(response.data)) {
            apiModels = response.data
            if (import.meta.env.DEV) {
              console.log('loadModels: Using response.data (direct array) structure, count:', apiModels.length)
            }
          }
          // Structure 3: {data: [...]} without status field
          else if (response.data.data && Array.isArray(response.data.data)) {
            apiModels = response.data.data
            if (import.meta.env.DEV) {
              console.log('loadModels: Using response.data.data (without status) structure, count:', apiModels.length)
            }
          }
          // Structure 4: Other structure, attempt to extract data
          else if (response.data.models && Array.isArray(response.data.models)) {
            apiModels = response.data.models
            if (import.meta.env.DEV) {
              console.log('loadModels: Using response.data.models structure, count:', apiModels.length)
            }
          }
        }
        
        if (import.meta.env.DEV) {
          console.log('loadModels: Processed models from API:', apiModels.length)
          if (apiModels.length > 0) {
            console.log('First model structure:', apiModels[0])
            console.log('First model fields:', Object.keys(apiModels[0]))
            console.log('First model active field:', apiModels[0].active)
            console.log('First model isActive field:', apiModels[0].isActive)
          }
        }
        
        if (apiModels.length === 0) {
          if (import.meta.env.DEV) {
            console.log('loadModels: API returned empty models list')
          }
          models.value = []
          notify.info('No models configured in the system')
        } else {
          // Set models from API response with field mapping
          const mappedModels = apiModels.map(model => ({
            ...model,
            // Map backend active to frontend isActive, ensure default value is true
            isActive: model.active !== undefined ? model.active : (model.isActive !== undefined ? model.isActive : true),
            // Ensure other fields are also correctly mapped (prioritize frontend field names, compatible with backend field names)
            isPrimary: model.isPrimary !== undefined ? model.isPrimary : (model.is_primary !== undefined ? model.is_primary : false),
            lastUpdated: model.lastUpdated !== undefined ? model.lastUpdated : (model.last_updated !== undefined ? model.last_updated : '')
          }))
          
          models.value = mappedModels || []
          
          if (import.meta.env.DEV) {
            console.log('loadModels: Mapped models count:', models.value.length)
            console.log('loadModels: First mapped model:', models.value[0])
            console.log('loadModels: First mapped model isActive:', models.value[0].isActive)
          }
          
          // Initialize trainingStatus for all API models and set default source
          models.value.forEach(model => {
            if (!model.trainingStatus) {
              model.trainingStatus = { isTraining: false, progress: 0, status: 'idle' }
            }
            // Set default source to ensure models are visible before configs load
            if (!model.source || model.source === '') {
              model.source = 'local'
            }
          })
          
          notify.success(`Loaded ${models.value.length} models successfully`)
        }
        
        // Load training status for each model
        await loadTrainingStatus()
        
        // Load model configurations
        await loadModelConfigs()
        
      } catch (error) {
        console.error('loadModels: Failed to load models from API:', error)
        console.error('Error details:', {
          message: error.message,
          code: error.code,
          config: error.config,
          response: error.response?.data
        })
        models.value = []
        notify.error('Failed to load models from API: ' + (error.message || 'Unknown error'))
      } finally {
        loading.value = false
        if (import.meta.env.DEV) {
          console.log('loadModels: Final model count:', models.value.length)
          console.log('loadModels: models.value:', models.value)
        }
      }
    }
    
    // Load model configurations
    const loadModelConfigs = async () => {
      return await performDataLoad('model-configs', {
        apiCall: () => api.modelConfigs.getAll(),
        dataPath: 'data',
        onSuccess: (configs) => {
          // Update models with their configurations
          models.value.forEach(model => {
            // Check if configs is an object (new format) or array (old format)
            let config = null
            if (Array.isArray(configs)) {
              config = configs.find(c => c.model_id === model.id)
            } else if (typeof configs === 'object') {
              // Handle object format where keys are model IDs
              const nameKey = model.name && typeof model.name === 'string' ? model.name.toLowerCase() : ''
              const typeKey = model.type && typeof model.type === 'string' ? model.type.toLowerCase() : ''
              config = configs[model.id] || configs[nameKey] || configs[typeKey]
            }
            
            if (config) {
              model.source = config.source || 'local'
              model.apiUrl = config.api_url || ''
              model.apiKey = config.api_key || ''
              model.modelName = config.model_name || ''
              model.sourceProvider = config.source_provider || 'custom'
              model.use_external_model_assistance = config.use_external_model_assistance || false
              model.external_model_id = config.external_model_id || ''
            } else {
              // Set default values if no configuration found
              model.source = 'local'
              model.apiUrl = ''
              model.apiKey = ''
              model.modelName = model.name || ''
              model.sourceProvider = 'custom'
              model.use_external_model_assistance = false
              model.external_model_id = ''
            }
          })
        },
        onError: (error) => {
          // Log error but don't show notification
          console.error('Failed to load model configurations:', error)
          console.error('Error stack:', error.stack)
          // Continue with existing model configurations - don't block other operations
        },
        successMessage: '', // No success notification
        errorMessage: 'Failed to load model configurations',
        errorContext: 'Load Model Configs',
        showSuccess: false,
        showError: false, // Don't show error notification (silent failure)
        silentError: true // Don't show any error notifications
      })
    }
    
    // Get specific model API configuration
    const getModelApiConfig = async (modelId) => {
      return await performDataLoad('model-api-config', {
        apiCall: () => api.modelConfigs.getApiConfig(modelId),
        onSuccess: (data) => {
          const model = models.value.find(m => m.id === modelId)
          if (model) {
            model.apiConfig = data
          }
        },
        successMessage: 'API configuration loaded successfully',
        errorMessage: 'Failed to get API configuration',
        errorContext: 'Get Model API Config',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError,
        fallbackValue: null
      })
    }
    
    // Get specific model API status
    const getModelApiStatus = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return null
      }
      
      let loadingMessageId = null
      
      return await performDataLoad('model-api-status', {
        apiCall: () => api.modelConfigs.getApiStatus(modelId),
        onBeforeStart: () => {
          loadingMessageId = notify.info('Checking API status...')
        },
        onSuccess: (data) => {
          model.apiStatus = data
        },
        onFinally: () => {
          if (loadingMessageId) {
            notify.remove(loadingMessageId)
          }
        },
        successMessage: 'API status checked successfully',
        errorMessage: 'Failed to get API status',
        errorContext: 'Get Model API Status',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError,
        fallbackValue: null
      })
    }
    
    // Get global API service status
    const getApiServiceStatus = async () => {
      return await performDataLoad('api-service-status', {
        apiCall: () => api.externalApi.getServiceStatus(),
        loadingRef: checkingGlobalStatus,
        dataPath: 'data',
        onSuccess: (data, fullResponse) => {
          // Validate response structure
          if (!data || !fullResponse.data || fullResponse.data.status !== 'success') {
            throw new Error('Invalid API response')
          }
          globalApiStatus.value = data.data
        },
        successMessage: 'Global API service status checked successfully',
        errorMessage: 'Failed to get API service status',
        errorContext: 'Get API Service Status',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError,
        fallbackValue: null
      })
    }
    
    // Check health of all models
    const checkAllModelsHealth = async () => {
      if (models.value.length === 0) {
        notify.info('No models to check')
        return
      }
      
      isCheckingAllHealth.value = true
      const results = []
      
      try {
        for (const model of models.value) {
          try {
            let healthResult = null
            if (model.source === 'external' && model.apiUrl && model.apiKey && model.modelName) {
              // Test API connection for external models
              const testData = {
                apiUrl: model.apiUrl,
                apiKey: model.apiKey,
                modelName: model.modelName,
                sourceProvider: model.sourceProvider
              }
              const response = await api.externalApi.testGenericConnection(testData)
              healthResult = {
                modelId: model.id,
                modelName: model.name,
                status: 'healthy',
                message: 'API connection successful',
                type: 'external'
              }
            } else {
              // For local models, check if they're running
              healthResult = {
                modelId: model.id,
                modelName: model.name,
                status: model.status === 'running' ? 'healthy' : 'unhealthy',
                message: model.status === 'running' ? 'Model is running' : 'Model is not running',
                type: 'local'
              }
            }
            results.push(healthResult)
          } catch (error) {
            results.push({
              modelId: model.id,
              modelName: model.name,
              status: 'unhealthy',
              message: error.message || 'Health check failed',
              type: model.source === 'external' ? 'external' : 'local'
            })
          }
        }
        
        // Summarize results
        const healthyCount = results.filter(r => r.status === 'healthy').length
        const unhealthyCount = results.length - healthyCount
        
        if (unhealthyCount === 0) {
          notify.success(`All ${healthyCount} models are healthy`)
        } else {
          notify.warning(`${healthyCount} models healthy, ${unhealthyCount} models unhealthy`)
        }
        
        // Store results for display (could be shown in a modal or log)
        if (import.meta.env.DEV) {
          console.log('Model health check results:', results)
        }
      } catch (error) {
        console.error('Failed to check model health:', error)
        notify.error('Failed to check model health')
      } finally {
        isCheckingAllHealth.value = false
      }
    }
    
    // Handle source change for existing models
    const onSourceChange = async (modelId, newSource) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) return
      
      model.source = newSource
      model.lastUpdated = new Date().toISOString()
      hasChanges.value = true
      
      // If changing to external, ensure it has necessary API fields
      if (newSource === 'external' && !model.apiUrl) {
        // Set default values based on provider if available
      if (model.sourceProvider === 'openai') {
        model.apiUrl = 'https://api.openai.com/v1/chat/completions'
        model.modelName = 'gpt-4'
      } else if (model.sourceProvider === 'anthropic') {
        model.apiUrl = 'https://api.anthropic.com/v1/messages'
        model.modelName = 'claude-3-opus-20240229'
      } else if (model.sourceProvider === 'google') {
        model.apiUrl = 'https://generativelanguage.googleapis.com/v1beta/models'
        model.modelName = 'gemini-pro'
      } else if (model.sourceProvider === 'huggingface') {
        model.apiUrl = 'https://api-inference.huggingface.co/models'
        model.modelName = 'meta-llama/Llama-2-70b-chat-hf'
      } else if (model.sourceProvider === 'deepseek') {
        model.apiUrl = 'https://api.deepseek.com/v1/chat/completions'
        model.modelName = 'deepseek-chat'
      } else if (model.sourceProvider === 'siliconflow') {
        model.apiUrl = 'https://api.siliconflow.cn/v1/chat/completions'
        model.modelName = 'Qwen2.5-32B-Instruct'
      } else if (model.sourceProvider === 'zhipu') {
        model.apiUrl = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
        model.modelName = 'glm-4'
      } else if (model.sourceProvider === 'baidu') {
        model.apiUrl = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions'
        model.modelName = 'ERNIE-4.0-8K'
      } else if (model.sourceProvider === 'alibaba') {
        model.apiUrl = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        model.modelName = 'qwen-max'
      } else if (model.sourceProvider === 'moonshot') {
        model.apiUrl = 'https://api.moonshot.cn/v1/chat/completions'
        model.modelName = 'moonshot-v1-8k'
      } else if (model.sourceProvider === 'yi') {
        model.apiUrl = 'https://api.01.ai/v1/chat/completions'
        model.modelName = 'yi-34b-chat'
      } else if (model.sourceProvider === 'tencent') {
        model.apiUrl = 'https://hunyuan.tencent.com/api/v1/chat/completions'
        model.modelName = 'hunyuan-standard'
      } else if (model.sourceProvider === 'ollama') {
        model.apiUrl = 'http://localhost:11434/v1/chat/completions'
        model.modelName = 'llama2'
      }
      }
      
      try {
        // Update model type in backend
        await api.modelConfigs.updateType(modelId, { type: newSource })
      } catch (error) {
        console.error('Failed to update model source:', error)
        notify.warning('Failed to update model source in backend')
      }
    }
    
    // Toggle API key visibility
    const toggleApiKeyVisibility = (modelId) => {
      showApiKeys.value[modelId] = !showApiKeys.value[modelId]
    }
    
    // Save model configuration
    const saveModelConfig = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }
      
      const configData = {
        source: model.source,
        apiUrl: model.apiUrl,
        apiKey: model.apiKey,
        modelName: model.modelName,
        sourceProvider: model.sourceProvider,
        use_external_model_assistance: model.use_external_model_assistance || false,
        external_model_id: model.external_model_id || ''
      }
      
      return await performDataOperation('save-model-config', {
        apiCall: () => api.modelConfigs.updateApiConfig(modelId, configData),
        onBeforeStart: () => {
          savingSettings.value.add(modelId)
        },
        onSuccess: () => {
          model.lastUpdated = new Date().toISOString()
        },
        onFinally: () => {
          savingSettings.value.delete(modelId)
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: 'Model configuration saved successfully',
        errorMessage: 'Failed to save model configuration',
        errorContext: 'Save Model Configuration',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
    }
    
    // Test API connection
    const testApiConnection = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model || !model.apiUrl || !model.apiKey || !model.modelName) {
        notify.error('Please complete all API configuration fields first')
        return
      }
      
      const testData = {
        apiUrl: model.apiUrl,
        apiKey: model.apiKey,
        modelName: model.modelName,
        sourceProvider: model.sourceProvider
      }
      
      return await performDataOperation('test-api-connection', {
        apiCall: () => api.externalApi.testGenericConnection(testData),
        onBeforeStart: () => {
          testingConnections.value.add(modelId)
        },
        onSuccess: async (responseData, fullResponse) => {
          // Extract detailed connection information from response
          const connectionDetails = responseData?.data || responseData || {}
          const apiVersion = connectionDetails.api_version || connectionDetails.version || 'Unknown'
          const responseTime = connectionDetails.response_time || connectionDetails.latency || 0
          const provider = connectionDetails.provider || model.sourceProvider || 'Unknown'
          
          testResults.value[modelId] = {
            success: true,
            status: 'success',
            message: connectionDetails.message || 'Connection successful! API is working properly.',
            details: {
              apiVersion,
              responseTime: `${responseTime.toFixed(2)}ms`,
              provider,
              connected: true,
              timestamp: new Date().toISOString()
            }
          }
          
          // Update model status with API information
          model.apiStatus = {
            connected: true,
            version: apiVersion,
            latency: responseTime,
            lastChecked: new Date().toISOString()
          }
          
          // Auto-activate if not already active
          if (!model.isActive) {
            await toggleActivation(modelId)
          }
        },
        onError: (error) => {
          testResults.value[modelId] = {
            success: false,
            status: 'error',
            message: error.response?.data?.message || error.message || 'Connection failed. Please check your settings.',
            details: {
              connected: false,
              errorType: error.response?.data?.error_type || 'unknown',
              timestamp: new Date().toISOString()
            }
          }
        },
        onFinally: () => {
          testingConnections.value.delete(modelId)
          // Clear test results after 5 seconds
          setTimeout(() => {
            if (isMounted.value) {
              delete testResults.value[modelId]
            }
          }, 5000)
        },
        successMessage: 'API connection test passed',
        errorMessage: 'API connection test failed',
        errorContext: 'Test API Connection',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
    }
    
    // Load training status for all models
    const loadTrainingStatus = async () => {
      return await performDataLoad('training-status', {
        apiCall: () => api.models.trainingStatus(),
        dataPath: 'data',
        onSuccess: (data) => {
          // Update models with training status
          // data should be the data field from API response
          // If data is an object with model IDs as keys, convert to array
          let sessions = []
          if (Array.isArray(data)) {
            sessions = data
          } else if (data && typeof data === 'object') {
            // Convert object to array
            sessions = Object.values(data)
          }
          
          models.value.forEach(model => {
            const session = sessions.find(s => s.model_ids?.includes(model.id))
            if (session) {
              model.trainingStatus = {
                isTraining: session.status === 'training',
                progress: session.progress || 0,
                status: session.status || 'idle'
              }
            } else {
              model.trainingStatus = { isTraining: false, progress: 0, status: 'idle' }
            }
          })
        },
        onError: (error) => {
          // Handle any errors gracefully
          // This 404 error doesn't affect serial port functionality
          if (import.meta.env.DEV) {
            console.log('Training status endpoint not available - this is expected if training feature is not implemented')
          }
          
          // Default to not training regardless of error type
          models.value.forEach(model => {
            model.trainingStatus = { isTraining: false, progress: 0, status: 'idle' }
          })
        },
        successMessage: '', // No success notification
        errorMessage: 'Failed to load training status',
        errorContext: 'Load Training Status',
        showSuccess: false,
        showError: false, // Silent failure
        silentError: true
      })
    }
    
    // Load available datasets for training
    const loadDatasets = async () => {
      return await performDataLoad('datasets', {
        apiCall: () => api.datasets.get(),
        dataPath: 'data',
        onSuccess: (data) => {
          availableDatasets.value = data.datasets || []
          if (availableDatasets.value.length > 0) {
            selectedDataset.value = availableDatasets.value[0].id
          }
        },
        onError: (error) => {
          console.error('Failed to load datasets:', error)
          availableDatasets.value = [{ id: 'default', name: 'Default Dataset' }]
          selectedDataset.value = 'default'
        },
        successMessage: '', // No success notification
        errorMessage: 'Failed to load datasets',
        errorContext: 'Load Datasets',
        showSuccess: false,
        showError: false // Silent failure
      })
    }


    // Test notification system
    const testNotificationSystem = () => {
      try {
        testNotifications()
        notify.info('Notification system test started')
      } catch (error) {
        handleError(error, 'Test Notifications')
      }
    }

    const onModelTypeChange = () => {
      if (newModel.value.type) {
        newModel.value.port = MODEL_PORT_CONFIG[newModel.value.type] || 8000
      }
    }

    const addNewModel = async () => {
      // Validation
      if (!newModel.value.id || !newModel.value.name || !newModel.value.type || !newModel.value.port) {
        notify.warning('Please fill in all required fields')
        return
      }

      // Check for duplicate ID
      if (models.value.some(model => model.id === newModel.value.id)) {
        notify.error('Model ID already exists')
        return
      }

      // Create new model object
      const modelToAdd = createDefaultModel(
        newModel.value.id,
        newModel.value.name,
        newModel.value.type
      )
      modelToAdd.port = newModel.value.port

      return await performDataOperation('add-model', {
        apiCall: () => api.post('/api/models', modelToAdd),
        loadingRef: isAddingModel,
        onSuccess: (data) => {
          // Add to local state
          models.value.push(data)
          // Reset form
          newModel.value = {
            id: '',
            name: '',
            type: '',
            port: 0
          }
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: 'Model added successfully',
        errorMessage: 'Failed to add model to backend. Please check backend connectivity.',
        errorContext: 'Add Model',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
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

      return await performDataOperation('remove-model', {
        apiCall: () => api.delete(`/api/models/${modelId}`),
        onBeforeStart: () => {
          operatingModels.value.add(modelId)
        },
        onSuccess: () => {
          models.value.splice(modelIndex, 1)
          hasChanges.value = true
        },
        onError: (error) => {
          handleError(error, 'Remove Model')
          // Fallback to local state update
          models.value.splice(modelIndex, 1)
          hasChanges.value = true
          notify.success('Model removed locally')
          // Return a value to prevent re-throwing
          return { success: true, isFallback: true }
        },
        onFinally: () => {
          operatingModels.value.delete(modelId)
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: 'Model removed successfully',
        errorMessage: 'Failed to remove model',
        errorContext: 'Remove Model',
        showSuccess: true,
        showError: false, // We handle errors manually in onError
        notify: notify,
        handleError: handleError
      })
    }

    const toggleActivation = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      const newState = !model.isActive

      return await performDataOperation('toggle-activation', {
        apiCall: () => api.put(`/api/models/${modelId}/activation`, { isActive: newState }),
        onBeforeStart: () => {
          operatingModels.value.add(modelId)
        },
        onSuccess: () => {
          model.isActive = newState
          model.lastUpdated = new Date().toISOString()
          hasChanges.value = true
        },
        onError: (error) => {
          handleError(error, 'Toggle Activation')
          // Fallback to local state update
          model.isActive = !model.isActive
          model.lastUpdated = new Date().toISOString()
          hasChanges.value = true
          notify.success(`Model ${model.isActive ? 'activated' : 'deactivated'} locally`)
          // Return a value to prevent re-throwing
          return { success: true, isFallback: true }
        },
        onFinally: () => {
          operatingModels.value.delete(modelId)
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: `Model ${newState ? 'activated' : 'deactivated'} successfully`,
        errorMessage: `Failed to ${newState ? 'activate' : 'deactivate'} model`,
        errorContext: 'Toggle Activation',
        showSuccess: true,
        showError: false, // We handle errors manually in onError
        notify: notify,
        handleError: handleError
      })
    }

    const useAsPrimary = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      return await performDataOperation('use-as-primary', {
        apiCall: () => api.put(`/api/models/${modelId}/primary`, { isPrimary: true }),
        onBeforeStart: () => {
          operatingModels.value.add(modelId)
        },
        onSuccess: () => {
          models.value.forEach(m => {
            m.isPrimary = m.id === modelId
          })
          model.lastUpdated = new Date().toISOString()
          hasChanges.value = true
        },
        onError: (error) => {
          handleError(error, 'Set as Primary')
          // Fallback to local state update
          models.value.forEach(m => {
            m.isPrimary = m.id === modelId
          })
          model.lastUpdated = new Date().toISOString()
          hasChanges.value = true
          notify.success('Model set as primary locally')
          // Return a value to prevent re-throwing
          return { success: true, isFallback: true }
        },
        onFinally: () => {
          operatingModels.value.delete(modelId)
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: 'Model set as primary successfully',
        errorMessage: 'Failed to set model as primary',
        errorContext: 'Set as Primary',
        showSuccess: true,
        showError: false, // We handle errors manually in onError
        notify: notify,
        handleError: handleError
      })
    }

    const startModel = async (modelId) => {
      return await performModelOperation(modelId, 'start')
    }

    const stopModel = async (modelId) => {
      return await performModelOperation(modelId, 'stop')
    }

    const restartModel = async (modelId) => {
      return await performModelOperation(modelId, 'restart', {
        successStatus: 'running',
        errorStatus: 'stopped'
      })
    }

    const startAllModels = async () => {
      return await performBatchModelOperation('start-all')
    }

    const stopAllModels = async () => {
      return await performBatchModelOperation('stop-all')
    }

    const restartAllModels = async () => {
      return await performBatchModelOperation('restart-all')
    }

    const restartSystem = async () => {
      if (!confirm('Are you sure you want to restart the entire system?')) {
        return
      }

      return await performDataOperation('restart-system', {
        apiCall: () => api.post('/api/system/restart'),
        loadingRef: isRestartingSystem,
        successMessage: 'System restart initiated',
        errorMessage: 'System restart failed. Please check if the backend is running.',
        errorContext: 'Restart System',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
    }

    // Batch operation methods
    const toggleModelSelection = (modelId) => {
      if (selectedModels.value.has(modelId)) {
        selectedModels.value.delete(modelId)
      } else {
        selectedModels.value.add(modelId)
      }
    }

    const toggleSelectAll = () => {
      if (isAllSelected.value) {
        // Deselect all filtered models
        filteredModelIds.value.forEach(id => {
          selectedModels.value.delete(id)
        })
      } else {
        // Select all filtered models
        filteredModelIds.value.forEach(id => {
          selectedModels.value.add(id)
        })
      }
    }

    const selectModelsByStatus = (status) => {
      const modelsToSelect = filteredModels.value.filter(model => model.status === status)
      modelsToSelect.forEach(model => {
        selectedModels.value.add(model.id)
      })
    }

    const performBulkAction = async (action) => {
      if (selectedModels.value.size === 0) {
        notify.warning('No models selected')
        return
      }

      const selectedIds = Array.from(selectedModels.value)
      const actionName = {
        'start': 'Start',
        'stop': 'Stop',
        'restart': 'Restart',
        'delete': 'Delete'
      }[action]

      if (!confirm(`Are you sure you want to ${action.toLowerCase()} ${selectedIds.length} selected model(s)?`)) {
        return
      }

      const selectedModelList = models.value.filter(model => selectedIds.includes(model.id))
      
      // Determine API endpoint and method
      let endpoint, method
      switch (action) {
        case 'start':
          endpoint = '/api/models/batch-start'
          method = 'post'
          break
        case 'stop':
          endpoint = '/api/models/batch-stop'
          method = 'post'
          break
        case 'restart':
          endpoint = '/api/models/batch-restart'
          method = 'post'
          break
        case 'delete':
          endpoint = '/api/models/batch-delete'
          method = 'delete'
          break
      }

      const requestData = {
        model_ids: selectedIds
      }

      return await performDataOperation('bulk-action', {
        apiCall: () => api[method](endpoint, requestData),
        onBeforeStart: () => {
          bulkActionInProgress.value = true
          // Update UI immediately
          selectedModelList.forEach(model => {
            if (action === 'start') {
              model.status = 'starting'
            } else if (action === 'stop') {
              model.status = 'stopping'
            } else if (action === 'restart') {
              model.status = 'stopping'
            }
          })
        },
        onSuccess: () => {
          // Update model statuses
          selectedModelList.forEach(model => {
            if (action === 'start') {
              model.status = 'running'
            } else if (action === 'stop') {
              model.status = 'stopped'
            } else if (action === 'restart') {
              model.status = 'running'
            } else if (action === 'delete') {
              // Remove deleted models from the list
              const index = models.value.findIndex(m => m.id === model.id)
              if (index !== -1) {
                models.value.splice(index, 1)
              }
            }
            model.lastUpdated = new Date().toISOString()
          })
          // Clear selection
          selectedModels.value.clear()
          hasChanges.value = true
        },
        onError: (error) => {
          handleError(error, `${actionName} Batch Operation`)
          notify.error(`Failed to ${action.toLowerCase()} selected models: ${error.message}`)
        },
        onFinally: () => {
          bulkActionInProgress.value = false
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: `${actionName} action completed successfully for ${selectedIds.length} model(s)`,
        errorMessage: `Failed to ${action.toLowerCase()} selected models`,
        errorContext: `${actionName} Batch Operation`,
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
    }



    const testConnection = async (modelId) => {
      // Unified API connection testing - delegate to testApiConnection
      // which provides better error handling through performDataOperation
      await testApiConnection(modelId)
    }

    // Apply API template based on provider selection
    const applyApiTemplate = (modelId, provider) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) return
      
      // API template configurations for common providers
      const apiTemplates = {
        openai: {
          apiUrl: 'https://api.openai.com/v1',
          modelName: 'gpt-4o',
          rateLimit: 1000,
          apiHeaders: {}
        },
        anthropic: {
          apiUrl: 'https://api.anthropic.com',
          modelName: 'claude-3-5-sonnet-20241022',
          rateLimit: 1000,
          apiHeaders: {}
        },
        google: {
          apiUrl: 'https://generativelanguage.googleapis.com/v1beta',
          modelName: 'gemini-2.0-flash-exp',
          rateLimit: 1000,
          apiHeaders: {}
        },
        azure: {
          apiUrl: 'https://{resource-name}.openai.azure.com/openai/deployments/{deployment-name}',
          modelName: 'gpt-4',
          rateLimit: 1000,
          apiHeaders: {}
        },
        aws: {
          apiUrl: 'https://bedrock-runtime.{region}.amazonaws.com',
          modelName: 'anthropic.claude-3-sonnet-20240229-v1:0',
          rateLimit: 1000,
          apiHeaders: {}
        },
        huggingface: {
          apiUrl: 'https://api-inference.huggingface.co/models',
          modelName: 'meta-llama/Llama-3.2-3B-Instruct',
          rateLimit: 1000,
          apiHeaders: {}
        },
        deepseek: {
          apiUrl: 'https://api.deepseek.com',
          modelName: 'deepseek-chat',
          rateLimit: 1000,
          apiHeaders: {}
        },
        moonshot: {
          apiUrl: 'https://api.moonshot.cn/v1',
          modelName: 'moonshot-v1-8k',
          rateLimit: 1000,
          apiHeaders: {}
        },
        ollama: {
          apiUrl: 'http://localhost:11434/v1',
          modelName: 'llama3.2:latest',
          rateLimit: 1000,
          apiHeaders: {}
        },
        custom: {
          // Custom template - clear defaults
          apiUrl: '',
          modelName: '',
          rateLimit: 1000,
          apiHeaders: {}
        }
      }
      
      const template = apiTemplates[provider] || apiTemplates.custom
      
      // Apply template values, but don't overwrite existing values if they're already set
      if (!model.apiUrl || model.apiUrl === '') {
        model.apiUrl = template.apiUrl
      }
      
      if (!model.modelName || model.modelName === '') {
        model.modelName = template.modelName
      }
      
      if (!model.rateLimit || model.rateLimit === '') {
        model.rateLimit = template.rateLimit
      }
      
      // Update hasChanges flag
      hasChanges.value = true
      
      notify.info(`Applied ${provider} API template. Please update API key and customize settings as needed.`)
    }

    const saveSettings = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model) {
        notify.error('Model not found')
        return
      }

      return await performDataOperation('save-settings', {
        apiCall: () => api.patch(`/api/models/${modelId}`, model),
        onBeforeStart: () => {
          savingSettings.value.add(modelId)
        },
        onSuccess: () => {
          model.lastUpdated = new Date().toISOString()
          hasChanges.value = true
        },
        onFinally: () => {
          savingSettings.value.delete(modelId)
        },
        updateHasChanges: true,
        hasChangesRef: hasChanges,
        successMessage: 'Settings saved successfully',
        errorMessage: 'Failed to save settings to backend',
        errorContext: 'Save Settings',
        showSuccess: true,
        showError: true,
        notify: notify,
        handleError: handleError
      })
    }

    const saveAllChanges = async () => {
      // Debug log: Recording save operation start
      if (import.meta.env.DEV) {
        console.log('🎯 saveAllChanges called, hasChanges:', hasChanges.value)
        console.log('📤 Sending models data to backend:', models.value.length, 'models')
        console.log('📤 First model data:', models.value.length > 0 ? {
          id: models.value[0].id,
          name: models.value[0].name,
          isActive: models.value[0].isActive,
          active: models.value[0].active,
          source: models.value[0].source,
          type: models.value[0].type
        } : 'No models')
        console.log('📤 Full models data (first 3):', models.value.slice(0, 3))
      }
      
      // Convert frontend model data to backend expected format
      const backendModels = models.value.map(model => {
        const backendModel = {
          id: model.id,
          name: model.name,
          type: model.type,
          source: model.source,
          // Map frontend isActive to backend active
          active: model.isActive !== undefined ? model.isActive : (model.active !== undefined ? model.active : true),
          // Map frontend isPrimary to backend is_primary
          is_primary: model.isPrimary !== undefined ? model.isPrimary : (model.is_primary !== undefined ? model.is_primary : false),
          port: model.port || 0,
          description: model.description || '',
          version: model.version || '1.0.0',
          last_updated: model.lastUpdated || new Date().toISOString()
        }
        
        // Add optional API configuration fields
        if (model.apiUrl) backendModel.api_url = model.apiUrl
        if (model.apiKey) backendModel.api_key = model.apiKey
        if (model.modelName) backendModel.model_name = model.modelName
        if (model.config) backendModel.config = model.config
        if (model.api_config) backendModel.api_config = model.api_config
        
        return backendModel
      })
      
      if (import.meta.env.DEV) {
        console.log('🔄 Converted backend models (first 2):', backendModels.slice(0, 2))
        console.log('🔄 Backend model fields:', backendModels.length > 0 ? Object.keys(backendModels[0]) : 'No models')
      }
      
      return await performDataOperation('save-all-changes', {
        loadingRef: isSavingAll,
        apiCall: () => api.put('/api/models', backendModels),
        successMessage: 'All changes saved successfully',
        errorMessage: 'Failed to save all changes to backend',
        errorContext: 'Save All Changes',
        updateHasChanges: true,
        hasChangesRef: hasChanges
      })
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
    
    const formatDateTime = (timestamp) => {
      const date = new Date(timestamp)
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      })
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
    
    // Train from scratch methods
    // Open train modal for a model
    const openTrainModal = (model) => {
      selectedModelForTraining.value = model
      trainingProgress.value = 0
      trainingStatus.value = 'idle'
      trainingMessage.value = ''
      loadDatasets()
      showTrainModal.value = true
    }
    
    // Close train modal
    const closeTrainModal = () => {
      showTrainModal.value = false
      selectedModelForTraining.value = null
      // Clear any existing polling interval
      if (pollInterval.value) {
        clearInterval(pollInterval.value)
        pollInterval.value = null
      }
    }
    
    // Generic training operation function
    const performTrainingOperation = async (operation, options = {}) => {
      const {
        model = selectedModelForTraining.value,
        apiEndpoint = null,
        method = 'post',
        requestData = {},
        validation = null,
        statusBefore = null,
        statusAfter = null,
        messageBefore = null,
        messageAfter = null,
        onSuccess = null,
        onError = null,
        successMessage = `${operation.charAt(0).toUpperCase() + operation.slice(1)} training completed successfully`,
        errorMessage = `Failed to ${operation} training`,
        showSuccess = true,
        showError = true
      } = options

      // Validate model
      if (!model) {
        if (showError) {
          notify.error('No model selected for training')
        }
        return false
      }

      // Execute validation if provided
      if (validation && typeof validation === 'function') {
        const validationResult = validation()
        if (validationResult === false) {
          return false
        }
      }

      // Update training status and message before operation
      if (statusBefore && typeof trainingStatus.value !== 'undefined') {
        trainingStatus.value = statusBefore
      }
      if (messageBefore && typeof trainingMessage.value !== 'undefined') {
        trainingMessage.value = messageBefore
      }

      try {
        // Determine API endpoint if not provided
        const finalApiEndpoint = apiEndpoint || 
          (operation === 'start' ? `/api/models/${model.id}/train` :
           operation === 'stop' ? `/api/models/${model.id}/train/stop` :
           operation === 'status' ? `/api/models/${model.id}/train/status` :
           null)

        if (!finalApiEndpoint) {
          throw new Error(`No API endpoint defined for training operation: ${operation}`)
        }

        // Make API request
        const response = await api[method](finalApiEndpoint, requestData)

        // Check response success for training operations
        if (response.data && response.data.success === false) {
          throw new Error(response.data.message || `Training ${operation} operation failed`)
        }

        // Update training status and message after successful operation
        if (statusAfter && typeof trainingStatus.value !== 'undefined') {
          trainingStatus.value = statusAfter
        }
        if (messageAfter && typeof trainingMessage.value !== 'undefined') {
          trainingMessage.value = messageAfter
        }

        // Execute success callback if provided
        if (onSuccess && typeof onSuccess === 'function') {
          await onSuccess(response.data)
        }

        // Show success message
        if (showSuccess) {
          notify.success(successMessage)
        }

        // Log operation in development
        if (import.meta.env.DEV) {
          console.log(`Training ${operation} operation successful:`, response.data)
        }

        return response.data

      } catch (error) {
        console.error(`Error performing training ${operation} operation:`, error)

        // Update training status on error
        if (typeof trainingStatus.value !== 'undefined') {
          trainingStatus.value = 'error'
        }
        if (typeof trainingMessage.value !== 'undefined') {
          trainingMessage.value = `Error: ${error.message}`
        }

        // Execute error callback if provided
        if (onError && typeof onError === 'function') {
          await onError(error)
        }

        if (showError) {
          notify.error(`${errorMessage}: ${error.message}`)
        }

        throw error
      }
    }

    // Start training a model from scratch
    const startTraining = async () => {
      return await performTrainingOperation('start', {
        validation: () => {
          if (!selectedModelForTraining.value || !selectedDataset.value) {
            return false
          }
          return true
        },
        statusBefore: 'training',
        messageBefore: 'Starting training process...',
        requestData: {
          dataset_id: selectedDataset.value,
          epochs: trainingParams.value.epochs,
          batch_size: trainingParams.value.batchSize,
          learning_rate: trainingParams.value.learningRate,
          validation_split: trainingParams.value.validationSplit,
          from_scratch: true
        },
        onSuccess: () => {
          // Start polling for training status using real API
          pollTrainingStatus()
        },
        successMessage: `Training started for ${selectedModelForTraining.value?.name || 'selected model'}`,
        errorMessage: `Failed to start training for ${selectedModelForTraining.value?.name || 'selected model'}`
      })
    }
    
    // Stop training
    const stopTraining = async () => {
      return await performTrainingOperation('stop', {
        validation: () => {
          if (!selectedModelForTraining.value) {
            return false
          }
          return true
        },
        statusAfter: 'idle',
        messageAfter: 'Training stopped',
        onSuccess: () => {
          // Show info notification instead of success notification
          notify.info(`Training stopped for ${selectedModelForTraining.value?.name || 'selected model'}`)
        },
        showSuccess: false, // Don't show default success notification
        successMessage: '', // Not used since showSuccess is false
        errorMessage: `Failed to stop training for ${selectedModelForTraining.value?.name || 'selected model'}`
      })
    }
    
    // Poll training status using real API
    const pollInterval = ref(null)
    const pollTrainingStatus = async () => {
      if (pollInterval.value) clearInterval(pollInterval.value)
      
      pollInterval.value = setInterval(async () => {
        if (!selectedModelForTraining.value) {
          clearInterval(pollInterval.value)
          return
        }
        
        try {
          // Use real API to get training status
          const response = await api.get(`/api/models/${selectedModelForTraining.value.id}/train/status`)
          const data = response.data
          
          if (data.success) {
            const trainingData = data.data
            trainingProgress.value = trainingData.progress || 0
            trainingMessage.value = trainingData.message || trainingData.status || ''
            trainingStatus.value = trainingData.status || 'training'
            
            // Update model training status in the main list
            const modelIndex = models.value.findIndex(m => m.id === selectedModelForTraining.value.id)
            if (modelIndex !== -1) {
              models.value[modelIndex].trainingStatus = {
                isTraining: trainingData.status === 'training',
                progress: trainingData.progress || 0,
                status: trainingData.status || 'idle'
              }
            }
            
            // Handle completion or error
            if (trainingData.status === 'completed') {
              clearInterval(pollInterval.value)
              pollInterval.value = null
              notify.success(`Training completed for ${selectedModelForTraining.value.name}`)
            } else if (trainingData.status === 'error' || trainingData.status === 'failed') {
              clearInterval(pollInterval.value)
              pollInterval.value = null
              notify.error(`Training failed for ${selectedModelForTraining.value.name}`)
            } else if (trainingData.status === 'stopped') {
              clearInterval(pollInterval.value)
              pollInterval.value = null
              notify.info(`Training stopped for ${selectedModelForTraining.value.name}`)
            }
          } else {
            throw new Error(data.message || 'Failed to get training status')
          }
        } catch (error) {
          console.error('Failed to get training status:', error)
          // Continue polling on error, but log it
        }
      }, 3000) // Poll every 3 seconds
    }

    // Lifecycle
    onMounted(async () => {
      if (import.meta.env.DEV) {
        console.log('onMounted hook started')
      }
      isMounted.value = true
      try {
        // Reset filter states to ensure all models are displayed
        modelFilterType.value = 'all'
        modelFilterStatus.value = 'all'
        
        // Call loadModels method to load models, this method already includes complete error handling and default model loading logic
        await loadModels()
        
        // Load model configurations with error handling
        try {
          await loadModelConfigs()
        } catch (modelConfigError) {
          console.error('Error loading model configurations:', modelConfigError)
          if (import.meta.env.DEV) {
            console.log('Continuing with scanSerialPorts despite model config error')
          }
        }
        
        // Additional logs: Confirm loaded model count
        if (import.meta.env.DEV) {
          console.log('Actual model count displayed after onMounted:', models.value.length)
          console.log('Actual count of local models displayed:', models.value.filter(m => m.source === 'local').length)
          console.log('Actual count of external API models displayed:', models.value.filter(m => m.source === 'external').length)
        }
        
      } catch (error) {
        console.error('Error in onMounted:', error)
        console.error('Error stack:', error.stack)
        notify.error('Failed to load models')
      } finally {
        // Always scan for serial ports, even if previous functions failed
        if (import.meta.env.DEV) {
          console.log('Attempting to scan serial ports...')
        }
        await scanSerialPorts()
        if (import.meta.env.DEV) {
          console.log('onMounted hook completed')
        }
      }
    })

    // Clean up monitoring on component unmount
    onUnmounted(() => {
      isMounted.value = false
      cleanupMonitoring()
      if (import.meta.env.DEV) {
        console.log('SettingsView component unmounted - monitoring cleaned up')
      }
    })

    // Check if we've already auto-started models this session
    const hasAutoStarted = ref(false)

    // Hardware configuration methods
    const updateCameraConfig = () => {
      const currentCount = hardwareConfig.value.cameraCount
      const currentCameras = hardwareConfig.value.cameras
      
      // If count increased, add new cameras
      if (currentCount > currentCameras.length) {
        for (let i = currentCameras.length + 1; i <= currentCount; i++) {
          currentCameras.push({
            id: i,
            name: `Camera ${i}`,
            type: 'mono',
            deviceId: `/dev/video${i - 1}`,
            fps: 30,
            baseline: 65,
            focalLength: 3.6
          })
        }
      }
      // If count decreased, remove extra cameras
      else if (currentCount < currentCameras.length) {
        hardwareConfig.value.cameras = currentCameras.slice(0, currentCount)
      }
      
      hasChanges.value = true
    }

    // Generic hardware test function
    const performHardwareTest = async (options = {}) => {
      const {
        apiEndpoint = '/api/system/hardware-test',
        requestData = {},
        updateTestingState = true,
        showSummary = true,
        showDeviceResults = true,
        successMessage = 'Hardware connections tested successfully',
        errorContext = 'Test Hardware Connections',
        useFallbackErrorHandling = false
      } = options

      // Update testing state if needed
      if (updateTestingState) {
        isTestingHardware.value = true
      }

      try {
        // Prepare request data
        const testRequestData = {
          ...hardwareConfig.value,
          ...requestData
        }

        // Call API for hardware test
        const response = await api.post(apiEndpoint, testRequestData)
        
        const testResults = response.data
        
        if (testResults.success) {
          const { cameras = [], sensors = [], actuators = [], summary = {} } = testResults.data
          
          // Display detailed device test results
          if (showDeviceResults) {
            // Process cameras
            if (cameras && cameras.length > 0) {
              cameras.forEach(camera => {
                if (camera.status === 'connected') {
                  notify.success(`Camera ${camera.id} (${camera.name}) connected successfully`)
                } else {
                  notify.warning(`Camera ${camera.id} (${camera.name}) connection failed: ${camera.message}`)
                }
              })
            }
            
            // Process sensors
            if (sensors && sensors.length > 0) {
              sensors.forEach(sensor => {
                if (sensor.status === 'connected') {
                  notify.success(`${sensor.type} sensor ${sensor.id} connected successfully`)
                } else {
                  notify.warning(`${sensor.type} sensor ${sensor.id} connection failed: ${sensor.message}`)
                }
              })
            }
            
            // Process actuators
            if (actuators && actuators.length > 0) {
              actuators.forEach(actuator => {
                if (actuator.status === 'connected') {
                  notify.success(`${actuator.type} actuator ${actuator.id} connected successfully`)
                } else {
                  notify.warning(`${actuator.type} actuator ${actuator.id} connection failed: ${actuator.message}`)
                }
              })
            }
          }
          
          // Display summary statistics
          if (showSummary && summary) {
            const { total_devices, connected_devices, failed_devices } = summary
            notify.success(`Hardware test completed: ${connected_devices}/${total_devices} devices connected successfully`)
            
            if (failed_devices > 0) {
              notify.warning(`${failed_devices} device(s) failed to connect`)
            }
          } else if (showSummary && !summary) {
            notify.success(successMessage)
          }
          
          return testResults
        } else {
          throw new Error(testResults.message || 'Hardware test failed')
        }
        
      } catch (error) {
        console.error('Hardware connection test failed:', error)
        
        if (useFallbackErrorHandling) {
          notify.error('All hardware connection tests failed. Please check backend connectivity.')
        } else {
          handleError(error, errorContext)
          notify.error(`Hardware connection test failed: ${error.message}`)
        }
        
        throw error
      } finally {
        if (updateTestingState) {
          isTestingHardware.value = false
        }
      }
    }

    const testHardwareConnections = async () => {
      return await performHardwareTest({
        requestData: {
          test_type: 'comprehensive',
          include_detailed_results: true
        }
      })
    }



    // Generic hardware configuration operation function
    const performHardwareConfigOperation = async (operation, options = {}) => {
      const {
        apiEndpoint = operation === 'save' ? '/api/system/hardware-config' : '/api/system/hardware-reset',
        method = 'post',
        requestData = {},
        successMessage = operation === 'save' ? 'Hardware configuration saved successfully' : 'Hardware configuration reset to defaults successfully',
        errorContext = operation === 'save' ? 'Save Hardware Configuration' : 'Reset Hardware Configuration',
        confirmMessage = operation === 'reset' ? 'Are you sure you want to reset hardware configuration to defaults?' : null,
        updateHasChanges = true,
        onBeforeStart = null,
        onSuccess = null,
        onFinally = null,
        showSuccess = true,
        showError = true
      } = options

      // Show confirmation dialog if needed
      if (confirmMessage && !confirm(confirmMessage)) {
        return false
      }

      // Execute before start callback if provided
      if (onBeforeStart && typeof onBeforeStart === 'function') {
        await onBeforeStart()
      }

      try {
        // Prepare request data
        const operationRequestData = {
          ...requestData,
          ...(operation === 'save' ? {
            config_type: 'hardware',
            save_timestamp: new Date().toISOString()
          } : {
            reset_type: 'hardware_config',
            reset_timestamp: new Date().toISOString()
          })
        }

        // Call API
        const response = await api[method](apiEndpoint, operationRequestData)

        if (response.data.success) {
          // Call custom success handler if provided
          if (onSuccess && typeof onSuccess === 'function') {
            await onSuccess(response.data, hardwareConfig)
          }

          // Update hasChanges flag
          if (updateHasChanges) {
            hasChanges.value = true
          }

          // Show success message
          if (showSuccess) {
            notify.success(successMessage)
          }

          // Log operation
          if (import.meta.env.DEV) {
            console.log(`Hardware configuration ${operation} via API:`, hardwareConfig.value)
          }

          return response.data
        } else {
          throw new Error(response.data.message || `Hardware configuration ${operation} failed`)
        }

      } catch (error) {
        console.error(`Failed to ${operation} hardware configuration:`, error)

        if (showError) {
          handleError(error, errorContext)
          notify.error(`Failed to ${operation} hardware configuration: ${error.message}`)
        }

        throw error
      } finally {
        // Execute finally callback if provided
        if (onFinally && typeof onFinally === 'function') {
          await onFinally()
        }
      }
    }

    const saveHardwareConfig = async () => {
      return await performHardwareConfigOperation('save', {
        onBeforeStart: () => {
          isSavingHardware.value = true
        },
        onFinally: () => {
          isSavingHardware.value = false
        }
      })
    }

    const resetHardwareConfig = async () => {
      return await performHardwareConfigOperation('reset', {
        onSuccess: () => {
          // Reset local state to empty defaults
          hardwareConfig.value = {
            serialPorts: [],
            selectedSerialPort: '',
            serialBaudRate: '9600',
            serialDeviceId: '',
            serialConnectionStatus: 'disconnected',
            serialConnectionError: '',
            cameraCount: 0,
            defaultResolution: '1280x720',
            defaultInterface: 'usb',
            defaultBaudRate: '9600',
            cameras: [],
            sensors: [],
            actuators: []
          }
        }
      })
    }

    // Generic serial port operation function
    const performSerialPortOperation = async (operation, options = {}) => {
      const {
        apiMethod = null,
        apiParams = {},
        preCheck = null,
        updateStatus = true,
        statusValue = null,
        clearError = true,
        successMessage = null,
        successStatus = null,
        errorStatus = null,
        errorContext = null,
        updateHasChanges = false,
        onSuccess = null,
        showSuccess = true,
        showError = true
      } = options

      // Execute pre-check if provided
      if (preCheck && typeof preCheck === 'function') {
        const preCheckResult = await preCheck()
        if (preCheckResult === false) {
          return false
        }
      }

      // Clear error if needed
      if (clearError) {
        hardwareConfig.value.serialConnectionError = ''
      }

      try {
        // Update status if needed
        if (updateStatus && statusValue) {
          hardwareConfig.value.serialConnectionStatus = statusValue
        }

        // Call API method
        if (!apiMethod || typeof api.serial[apiMethod] !== 'function') {
          throw new Error(`Invalid serial API method: ${apiMethod}`)
        }

        const response = await api.serial[apiMethod](apiParams)

        // Update status on success if needed
        if (updateStatus && successStatus) {
          hardwareConfig.value.serialConnectionStatus = successStatus
        }

        // Show success message
        if (showSuccess && successMessage) {
          notify.success(successMessage)
        }

        // Update hasChanges flag
        if (updateHasChanges) {
          hasChanges.value = true
        }

        // Call onSuccess callback if provided
        if (onSuccess && typeof onSuccess === 'function') {
          await onSuccess(response.data)
        }

        // Log operation in development
        if (import.meta.env.DEV) {
          console.log(`Serial port ${operation} successful:`, response.data)
        }

        return response.data

      } catch (error) {
        console.error(`Error ${operation} serial port:`, error)

        // Update status on error if needed
        if (updateStatus && errorStatus) {
          hardwareConfig.value.serialConnectionStatus = errorStatus
        }

        // Set error message
        hardwareConfig.value.serialConnectionError = `Failed to ${operation} serial port: ${error.message}`

        if (showError) {
          handleError(error, errorContext || `${operation.charAt(0).toUpperCase() + operation.slice(1)} Serial Port`)
          notify.error(`Failed to ${operation} serial port: ${error.message}`)
        }

        throw error
      }
    }

    // Serial port related methods
    const scanSerialPorts = async () => {
      return await performSerialPortOperation('scan', {
        apiMethod: 'getPorts',
        updateStatus: false,
        onSuccess: (data) => {
          // Update the serial ports list
          hardwareConfig.value.serialPorts = data || []
          
          if (hardwareConfig.value.serialPorts.length === 0) {
            notify.info('No serial ports found. Please check your connections.')
          } else {
            notify.success(`Found ${hardwareConfig.value.serialPorts.length} serial port(s)`)
          }
        }
      })
    }

    const connectSerialPort = async () => {
      return await performSerialPortOperation('connect', {
        apiMethod: 'connect',
        apiParams: {
          port: hardwareConfig.value.selectedSerialPort,
          baudrate: hardwareConfig.value.serialBaudRate,
          device_id: hardwareConfig.value.serialDeviceId
        },
        preCheck: () => {
          if (!hardwareConfig.value.selectedSerialPort || !hardwareConfig.value.serialDeviceId) {
            notify.warning('Please select a serial port and enter a device ID')
            return false
          }
          return true
        },
        statusValue: 'connecting',
        successStatus: 'connected',
        errorStatus: 'error',
        successMessage: 'Serial port connected successfully',
        updateHasChanges: true
      })
    }

    const disconnectSerialPort = async () => {
      return await performSerialPortOperation('disconnect', {
        apiMethod: 'disconnect',
        apiParams: {
          device_id: hardwareConfig.value.serialDeviceId,
          port: hardwareConfig.value.selectedSerialPort
        },
        statusValue: 'connecting',
        successStatus: 'disconnected',
        errorStatus: 'error',
        successMessage: 'Serial port disconnected successfully',
        updateHasChanges: true
      })
    }
    


    
    // Performance monitoring methods
    const startPerformanceMonitoring = () => {
      if (isMonitoringActive.value) {
        notify.info('Performance monitoring is already active')
        return
      }
      
      isMonitoringActive.value = true
      updatePerformanceMetrics()
      
      // Set up interval for updates (every 5 seconds)
      monitoringInterval.value = setInterval(() => {
        updatePerformanceMetrics()
      }, 5000)
      
      notify.success('Performance monitoring started')
    }
    
    const stopPerformanceMonitoring = () => {
      if (!isMonitoringActive.value) {
        notify.info('Performance monitoring is not active')
        return
      }
      
      isMonitoringActive.value = false
      if (monitoringInterval.value) {
        clearInterval(monitoringInterval.value)
        monitoringInterval.value = null
      }
      
      notify.info('Performance monitoring stopped')
    }
    
    const updatePerformanceMetrics = async () => {
      try {
        // Fetch system metrics from API
        const response = await api.system.metrics()
        const metrics = response.data
        
        // Update performance metrics
        performanceMetrics.value = {
          cpuUsage: metrics.cpu_usage || 0,
          memoryUsage: metrics.memory_usage || 0,
          diskUsage: metrics.disk_usage || 0,
          networkIn: metrics.network_in || 0,
          networkOut: metrics.network_out || 0,
          modelResponseTimes: metrics.model_response_times || {},
          lastUpdated: new Date().toISOString()
        }
      } catch (error) {
        console.error('Failed to update performance metrics:', error)
        // Notify user of data unavailability but keep existing metrics
        if (Object.keys(performanceMetrics.value).length === 0) {
          // First time failure, initialize with zeros
          performanceMetrics.value = {
            cpuUsage: 0,
            memoryUsage: 0,
            diskUsage: 0,
            networkIn: 0,
            networkOut: 0,
            modelResponseTimes: {},
            lastUpdated: 'Data unavailable',
            error: true
          }
        } else {
          // Keep existing data but mark as stale
          performanceMetrics.value.lastUpdated = `Update failed: ${new Date().toLocaleTimeString()}`
          performanceMetrics.value.error = true
        }
        notify.error('Failed to retrieve performance metrics, please check backend connection')
      }
    }
    
    // Cleanup function for monitoring
    const cleanupMonitoring = () => {
      if (monitoringInterval.value) {
        clearInterval(monitoringInterval.value)
        monitoringInterval.value = null
      }
      if (pollInterval.value) {
        clearInterval(pollInterval.value)
        pollInterval.value = null
      }
      isMonitoringActive.value = false
    }

    // Watch for changes in models to automatically update hasChanges flag
    watch(() => models.value, (newModels, oldModels) => {
      // Skip initial load or when both are empty
      if (!oldModels || !newModels) return
      
      // Compare the JSON string representations to detect changes
      const newJson = JSON.stringify(newModels)
      const oldJson = JSON.stringify(oldModels)
      
      if (newJson !== oldJson) {
        if (import.meta.env.DEV) {
          console.log('🔄 Models changed, setting hasChanges to true')
          console.log('📊 Change detected in models array, length:', newModels.length)
        }
        hasChanges.value = true
      }
    }, { deep: true, immediate: false })

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
        
        // Train from scratch state
        showTrainModal,
        selectedModelForTraining,
        trainingProgress,
        trainingStatus,
        trainingMessage,
        availableDatasets,
        selectedDataset,
        trainingParams,
        
        // Hardware configuration state
        hardwareConfig,
        isTestingHardware,
        isSavingHardware,
        
        // Data
        models,
        newModel,
        
        // Batch operation state
        selectedModels,
        bulkActionInProgress,
        
        // Performance monitoring state
        performanceMetrics,
        monitoringInterval,
        isMonitoringActive,
        
        // Computed
        modelTypes,
        activeModelsCount,
        runningModelsCount,
        apiModelsCount,
      canStartAll,
      canStopAll,
      canRestartAll,
      
      // Batch operation computed
      selectedModelsCount,
      canPerformBulkAction,
      isAllSelected,
      filteredModelIds,
      
      // Methods
      loadModels,
      loadTrainingStatus,
      loadDatasets,
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
      formatDateTime,
      isOperating,
      isTestingConnection,
        isSavingSettings,
        testNotificationSystem,
        getModelApiConfig,
        getModelApiStatus,
        getApiServiceStatus,
      
      // Train from scratch methods
      openTrainModal,
      closeTrainModal,
      startTraining,
      stopTraining,
      
      // Source handling
      onSourceChange,
      
      // Hardware configuration methods
      updateCameraConfig,
      testHardwareConnections,
      saveHardwareConfig,
      resetHardwareConfig,
      
      // Serial port methods
      scanSerialPorts,
      connectSerialPort,
      disconnectSerialPort,
      
      // Batch operation methods
      toggleModelSelection,
      toggleSelectAll,
      selectModelsByStatus,
      performBulkAction,
      
      // Performance monitoring methods
      startPerformanceMonitoring,
      stopPerformanceMonitoring,
      updatePerformanceMetrics,
      cleanupMonitoring
    }
  }
}
</script>

<style scoped>
/* Settings Container */
.settings-container {
  padding: var(--spacing-lg);
  margin-top: 70px;
  min-height: calc(100vh - 70px);
  font-family: var(--font-family);
  background: var(--bg-primary);
}

/* Main Content Container */
.settings-content {
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
  padding: 0 20px;
}

/* Batch Actions */
.batch-actions {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  flex-wrap: wrap;
}

.batch-btn {
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  outline: none;
}

.batch-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.batch-btn:hover:not(:disabled) {
  background-color: #e6e6e6;
}

.batch-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Bulk Operations Toolbar */
.bulk-operations-toolbar {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.bulk-selection-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.bulk-select-all {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-weight: 500;
  color: var(--text-primary);
  cursor: pointer;
}

.bulk-select-all input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.bulk-quick-select {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.bulk-quick-select span {
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.quick-select-btn {
  padding: var(--spacing-xs) var(--spacing-sm);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.quick-select-btn:hover {
  background-color: var(--bg-hover);
  border-color: var(--primary-color);
}

.bulk-actions {
  border-top: 1px solid var(--border-color);
  padding-top: var(--spacing-md);
}

.bulk-action-label {
  display: block;
  margin-bottom: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.bulk-action-buttons {
  display: flex;
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.bulk-action-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-fast);
  font-size: 0.9rem;
  min-width: 120px;
}

.bulk-action-btn.start {
  background-color: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  border-color: rgba(34, 197, 94, 0.3);
}

.bulk-action-btn.start:hover:not(:disabled) {
  background-color: rgba(34, 197, 94, 0.2);
  border-color: rgba(34, 197, 94, 0.5);
}

.bulk-action-btn.stop {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border-color: rgba(239, 68, 68, 0.3);
}

.bulk-action-btn.stop:hover:not(:disabled) {
  background-color: rgba(239, 68, 68, 0.2);
  border-color: rgba(239, 68, 68, 0.5);
}

.bulk-action-btn.restart {
  background-color: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
  border-color: rgba(59, 130, 246, 0.3);
}

.bulk-action-btn.restart:hover:not(:disabled) {
  background-color: rgba(59, 130, 246, 0.2);
  border-color: rgba(59, 130, 246, 0.5);
}

.bulk-action-btn.delete {
  background-color: rgba(99, 102, 241, 0.1);
  color: #6366f1;
  border-color: rgba(99, 102, 241, 0.3);
}

.bulk-action-btn.delete:hover:not(:disabled) {
  background-color: rgba(99, 102, 241, 0.2);
  border-color: rgba(99, 102, 241, 0.5);
}

.bulk-action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Model select checkbox in model cards */
.model-select-checkbox {
  margin-right: var(--spacing-sm);
}

.model-select-checkbox input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

/* Performance monitoring section */
.performance-monitoring-section {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
}

.performance-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.metric-card {
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md);
  text-align: center;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-xs);
}

.metric-label {
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.metric-timestamp {
  font-size: 0.8rem;
  color: var(--text-tertiary);
  margin-top: var(--spacing-sm);
}

/* Monitoring controls */
.monitoring-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  flex-wrap: wrap;
}

.monitoring-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-fast);
  font-size: 0.9rem;
}

.monitoring-btn.start {
  background-color: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  border-color: rgba(34, 197, 94, 0.3);
}

.monitoring-btn.start:hover:not(:disabled) {
  background-color: rgba(34, 197, 94, 0.2);
  border-color: rgba(34, 197, 94, 0.5);
}

.monitoring-btn.stop {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border-color: rgba(239, 68, 68, 0.3);
}

.monitoring-btn.stop:hover:not(:disabled) {
  background-color: rgba(239, 68, 68, 0.2);
  border-color: rgba(239, 68, 68, 0.5);
}

.monitoring-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.monitoring-status {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--border-radius-sm);
  font-size: 0.85rem;
  font-weight: 500;
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.monitoring-status.active {
  background-color: rgba(34, 197, 94, 0.1);
  color: #22c55e;
}

/* Model response times */
.model-response-times {
  margin-top: var(--spacing-lg);
  padding-top: var(--spacing-lg);
  border-top: 1px solid var(--border-color);
}

.model-response-times h3 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-size: 1.1rem;
}

.response-times-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-sm);
}

.response-time-card {
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-sm);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.model-id {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.response-time {
  font-size: 0.9rem;
  color: var(--text-primary);
  font-weight: 600;
}

/* Model Card */
.model-card {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-xl);
  margin-bottom: var(--spacing-lg);
  background-color: var(--bg-secondary);
  transition: all var(--transition-fast);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.model-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-color: #333333;
}

.model-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
  flex-wrap: wrap;
  gap: var(--spacing-lg);
}

.model-info h4 {
  margin: 0 0 var(--spacing-sm) 0;
  color: var(--text-primary);
  font-weight: 600;
  font-size: 1.2rem;
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
  color: #333333;
  border: 1px solid #cccccc;
}

.model-status.disconnected {
  background-color: #f8f8f8;
  color: #666666;
  border: 1px solid #dddddd;
}

.model-status.testing {
  background-color: #f5f5f5;
  color: #666666;
  border: 1px solid #dddddd;
}

.model-status.failed {
  background-color: #f8f8f8;
  color: #666666;
  border: 1px solid #dddddd;
}

.model-status.running {
  background-color: #f5f5f5;
  color: #333333;
  border: 1px solid #cccccc;
}

.model-status.stopped {
  background-color: #f8f8f8;
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
  background-color: #e6e6e6;
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
  background-color: #e6e6e6;
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
  flex-direction: column;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-md);
  padding: var(--spacing-lg);
  background-color: var(--bg-primary);
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
}

/* Button Groups */
.model-actions-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background-color: #ffffff;
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.model-actions-group-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: #333333;
  margin-bottom: var(--spacing-sm);
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid #f0f0f0;
}

.model-actions-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.control-btn {
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: #ffffff;
  color: #333333;
  min-width: 120px;
  text-align: center;
  outline: none;
}

.control-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.control-btn:hover:not(:disabled) {
  background-color: #f0f0f0;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.control-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.primary-btn {
  font-weight: 600;
  background-color: #f0f0f0;
}

.primary-btn:hover:not(:disabled) {
  background-color: #e0e0e0;
}

.start-btn, .stop-btn, .restart-btn {
  font-weight: 500;
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  transition: all var(--transition-fast);
  background-color: #ffffff;
  color: #333333;
  min-width: 140px;
  text-align: center;
  outline: none;
}

.start-btn:focus, .stop-btn:focus, .restart-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.start-btn:hover:not(:disabled),
.stop-btn:hover:not(:disabled),
.restart-btn:hover:not(:disabled) {
  background-color: #f0f0f0;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.start-btn:disabled,
.stop-btn:disabled,
.restart-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.activation-btn, .remove-btn {
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: #ffffff;
  color: #333333;
  min-width: 140px;
  text-align: center;
  outline: none;
}

.activation-btn:focus, .remove-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.activation-btn.active {
  background-color: #e6e6e6;
  color: #333333;
  font-weight: 600;
}

.activation-btn.inactive {
  background-color: #ffffff;
  color: #666666;
}

.activation-btn:hover:not(:disabled) {
  background-color: #e6e6e6;
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
  background-color: #e6e6e6;
}

.remove-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* API Configuration */
.settings-toggle-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
  margin-top: var(--spacing-lg);
  outline: none;
  min-width: 140px;
  text-align: center;
}

.settings-toggle-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.settings-toggle-btn:hover {
  background-color: #e6e6e6;
}

.api-config-section {
  margin-top: var(--spacing-lg);
  padding-top: var(--spacing-lg);
  border-top: 1px solid var(--border-color);
}

.api-settings-form {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-lg);
  background-color: var(--bg-primary);
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
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
  padding: var(--spacing-md);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 0.95rem;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  transition: border-color 0.2s ease;
}

.form-group input:focus, .form-group select:focus {
  outline: none;
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
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
  background-color: #e6e6e6;
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
  color: #333333;
  border: 1px solid #cccccc;
}

.status-indicator.invalid {
  background-color: #f8f8f8;
  color: #666666;
  border: 1px solid #dddddd;
}

.api-actions {
  display: flex;
  gap: var(--spacing-sm);
  justify-content: flex-end;
}

.test-btn {
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  background-color: var(--bg-primary);
  color: var(--text-primary);
  outline: none;
  min-width: 140px;
  text-align: center;
}

.test-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.test-btn:hover:not(:disabled) {
  background-color: #e6e6e6;
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
  margin-top: var(--spacing-lg);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: var(--spacing-lg);
  padding-top: var(--spacing-lg);
  border-top: 1px solid var(--border-color);
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
  background-color: #e6e6e6;
  color: #333333;
  border: 1px solid #cccccc;
}

.test-result.error {
  background-color: #f8f8f8;
  color: #666666;
  border: 1px solid #dddddd;
}

/* Add Model Section */
.add-model-section {
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-xl);
  margin-bottom: var(--spacing-xl);
  background-color: var(--bg-secondary);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
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
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 0.95rem;
  font-weight: 500;
  transition: background-color var(--transition-fast);
  height: auto;
  outline: none;
  min-width: 140px;
  text-align: center;
}

.add-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.add-btn:hover:not(:disabled) {
  background-color: #e6e6e6;
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
  gap: var(--spacing-lg);
  margin-top: var(--spacing-xl);
  justify-content: center;
  flex-wrap: wrap;
  padding: var(--spacing-xl) 0;
}

.save-btn, .reset-btn {
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all var(--transition-fast);
  min-width: 140px;
  outline: none;
}

.save-btn:focus, .reset-btn:focus {
  border-color: #555555;
  box-shadow: 0 0 0 3px rgba(85, 85, 85, 0.05);
}

.save-btn {
  background-color: var(--bg-tertiary);
  color: var(--text-primary);
}

.save-btn:hover:not(:disabled) {
  background-color: #e6e6e6;
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
  background-color: #e6e6e6;
}

.reset-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  /* Model Configuration Type */
  .model-config-type {
    padding: 4px 8px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .model-config-type.external {
    background-color: #e6e6e6;
    color: #333333;
    border: 1px solid #cccccc;
  }
  
  .model-config-type.local {
    background-color: #f5f5f5;
    color: #666666;
    border: 1px solid #cccccc;
  }
  
  /* Model Configuration Type Select */
  .model-config-type-select {
    padding: 4px 8px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    cursor: pointer;
    margin-right: 8px;
  }
  
  .model-config-type-select:focus {
    outline: none;
    border-color: var(--border-dark);
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
  background-color: #e6e6e6;
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
  border-top: 3px solid #555555;
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
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
}

.stat-card {
  background-color: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-xl);
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
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

/* Training Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  width: 90%;
  max-width: 600px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-md);
  border-bottom: 1px solid var(--border-color);
}

.modal-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: var(--text-primary);
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-secondary);
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--border-radius-sm);
  transition: background-color var(--transition-fast);
}

.close-btn:hover {
  background-color: var(--bg-secondary);
}

.modal-body {
  padding: var(--spacing-md);
}

.modal-footer {
  display: flex;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  border-top: 1px solid var(--border-color);
  justify-content: flex-end;
}

/* Training Parameters Styles */
.training-params {
  margin-top: var(--spacing-lg);
}

.training-params h4 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1rem;
  color: var(--text-primary);
}

/* Training Progress Styles */
.training-progress {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-md);
  background-color: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
}

.progress-bar {
  height: 20px;
  background-color: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: var(--spacing-sm);
}

.progress-fill {
  height: 100%;
  background-color: #555555;
  transition: width var(--transition-fast);
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.training-status {
  font-size: 0.9rem;
  font-weight: 500;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 4px;
}

.training-status.training {
  background-color: #e6e6e6;
  color: #333333;
}

.training-status.completed {
  background-color: #e6e6e6;
  color: #333333;
}

.training-status.error {
  background-color: #f8f8f8;
  color: #666666;
}

.training-message {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-top: var(--spacing-xs);
}

/* Button Styles for Modal */
.btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all var(--transition-fast);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background-color: #333333;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #555555;
}

.btn-secondary {
  background-color: #e0e0e0;
  color: #333333;
}

.btn-secondary:hover:not(:disabled) {
  background-color: #d0d0d0;
}

.btn-cancel {
  background-color: transparent;
  color: #666666;
  border: 1px solid var(--border-color);
}

.btn-cancel:hover:not(:disabled) {
  background-color: var(--bg-secondary);
}

/* Train Button Styles */
.train-btn {
  background-color: #f5f5f5;
  color: #333333;
}

.train-btn:hover:not(:disabled) {
  background-color: #e0e0e0;
}

@media (max-width: 768px) {
  .modal-footer {
    flex-direction: column;
  }
  
  .btn {
    width: 100%;
  }
}

/* Test results */
.success-message {
  background-color: #f0f9f0;
  border: 1px solid #d4edda;
  color: #155724;
  padding: var(--spacing-md);
  border-radius: var(--border-radius-md);
  margin-top: var(--spacing-md);
}

.error-message {
  background-color: #fef0f0;
  border: 1px solid #f8d7da;
  color: #721c24;
  padding: var(--spacing-md);
  border-radius: var(--border-radius-md);
  margin-top: var(--spacing-md);
}

.info-message {
  background-color: #f8f9fa;
  border: 1px solid #d4dae0;
  color: #2c3e50;
  padding: var(--spacing-md);
  border-radius: var(--border-radius-md);
  margin-top: var(--spacing-md);
}

/* API Service Status */
.api-service-status-section {
  margin-bottom: var(--spacing-lg);
}

.api-service-card {
  background: white;
  border-radius: var(--border-radius-md);
  padding: var(--spacing-lg);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.api-service-card h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
}

.api-status-info {
  margin-bottom: var(--spacing-md);
}

.api-status-info p {
  margin: 5px 0;
}

.status-online {
  color: #28a745;
  font-weight: 500;
}

.status-offline {
  color: #dc3545;
  font-weight: 500;
}

/* Health Monitor Section */
.health-monitor-section {
  margin-bottom: 30px;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.health-monitor-section h2 {
  margin: 0 0 20px 0;
  color: #2c3e50;
  font-size: 1.5rem;
  border-bottom: 2px solid #3498db;
  padding-bottom: 10px;
}
</style>

