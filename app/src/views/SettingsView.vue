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
        
        <div class="models-grid">
          <div v-for="model in filteredModels" :key="model.id" class="model-card">
            <div class="model-header">
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
                    placeholder="https://api.example.com/v1/"
                  />
                </div>
                <div class="form-group">
                  <label :for="`api-key-${model.id}`">API Key</label>
                  <div class="password-input-wrapper">
                    <input
                      :id="`api-key-${model.id}`"
                      v-model="model.api_key"
                      :type="showApiKeys[model.id] ? 'text' : 'password'"
                      placeholder="sk-..."
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
                    placeholder="e.g., gpt-4, claude-3-opus"
                  />
                </div>
                <div class="form-group">
                  <label :for="`api-source-${model.id}`">API Provider</label>
                  <select :id="`api-source-${model.id}`" v-model="model.source_provider">
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="google">Google AI</option>
                    <option value="aws">AWS</option>
                    <option value="azure">Azure</option>
                    <option value="huggingface">Hugging Face</option>
                    <option value="custom">Custom</option>
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
                  placeholder="e.g., /dev/video0"
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
                placeholder="Enter device ID (e.g., arduino, sensor)"
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
                    placeholder="e.g., COM3, /dev/ttyUSB0"
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
                    placeholder="e.g., GPIO17, /dev/ttyUSB1"
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
                <select v-model="model.source" class="model-config-type-select" @change="onSourceChange(model.id, model.source)">
                  <option value="local">Local</option>
                  <option value="external">External API</option>
                </select>
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
                  >
                    <option value="custom">Custom</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="google">Google AI</option>
                    <option value="huggingface">Hugging Face</option>
                    <option value="mistral">Mistral AI</option>
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
import { ref, computed, onMounted } from 'vue'
import errorHandler from '../utils/errorHandler.js'
import { notify } from '../plugins/notification.js'
import { Model, NewModel, MODEL_TYPES, MODEL_STATUS, MODEL_PORT_CONFIG, createDefaultModel, isValidModelId, isValidPort, isApiModelType, generateMockMetrics } from '../utils/modelTypes.js'
import testNotifications from '../utils/testNotifications.js'
import api from '../utils/api.js'

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
    const globalApiStatus = ref(null)
    const checkingGlobalStatus = ref(false)
    
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
      cameraCount: 1,
      defaultResolution: '1280x720',
      defaultInterface: 'usb',
      defaultBaudRate: '9600',
      cameras: [
        {
          id: 1,
          name: 'Main Camera',
          type: 'mono',
          deviceId: '/dev/video0',
          fps: 30,
          baseline: 65,
          focalLength: 3.6
        }
      ],
      sensors: [
        {
          id: 1,
          type: 'temperature',
          port: '/dev/ttyUSB0'
        },
        {
          id: 2,
          type: 'humidity',
          port: '/dev/ttyUSB1'
        }
      ],
      actuators: [
        {
          id: 1,
          type: 'motor',
          port: 'GPIO17'
        },
        {
          id: 2,
          type: 'servo',
          port: 'GPIO18'
        }
      ]
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

    // Computed
    const modelTypes = computed(() => MODEL_TYPES)
    const activeModelsCount = computed(() => {
      return models.value.filter(model => model.isActive).length
    })
    const runningModelsCount = computed(() => {
      return models.value.filter(model => model.status === 'running').length
    })
    const apiModelsCount = computed(() => {
      return models.value.filter(model => model.type.toLowerCase().includes('api') || model.source === 'external').length
    })
    const filteredModels = computed(() => {
      let filtered = models.value
      
      // Filter by type
      if (modelFilterType.value === 'local') {
        filtered = filtered.filter(model => model.source === 'local')
      } else if (modelFilterType.value === 'external') {
        filtered = filtered.filter(model => model.source === 'external')
      }
      
      // Filter by status
      if (modelFilterStatus.value !== 'all') {
        filtered = filtered.filter(model => model.status === modelFilterStatus.value)
      }
      
      return filtered
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
        // Use real API to load models
        const response = await api.models.getAll()
        const apiModels = response.data.data || response.data
        
        if (apiModels && apiModels.length > 0) {
          models.value = apiModels
          console.log('Loaded models from API count:', models.value.length)
          notify.success('Models loaded successfully from API')
          
          // Initialize trainingStatus for all API models
          models.value.forEach(model => {
            if (!model.trainingStatus) {
              model.trainingStatus = { isTraining: false, progress: 0, status: 'idle' }
            }
          })
        } else {
          // Fallback to default models if API returns empty
          console.log('API returned empty models list, using default models')
          models.value = getDefaultModels()
          notify.info('Using default model configuration')
        }
        
        // Load training status for each model
        await loadTrainingStatus()
        
        // Load model configurations
        await loadModelConfigs()
      } catch (error) {
        console.error('Error loading models from API:', error)
        errorHandler.handleError(error, 'Load Models')
        
        // Fallback to default models
        console.log('Using default models as fallback')
        models.value = getDefaultModels()
        notify.warning('Failed to load models from API. Using default configuration.')
      } finally {
        console.log('Final total models:', models.value.length)
        console.log('Final local models:', models.value.filter(m => m.source === 'local').length)
        loading.value = false
      }
    }
    
    // Load model configurations
    const loadModelConfigs = async () => {
      try {
        const response = await api.modelConfigs.getAll()
        const configs = response.data || {}
        
        // Update models with their configurations
        models.value.forEach(model => {
          // Check if configs is an object (new format) or array (old format)
          let config = null
          if (Array.isArray(configs)) {
            config = configs.find(c => c.model_id === model.id)
          } else if (typeof configs === 'object') {
            // Handle object format where keys are model IDs
            config = configs[model.id] || configs[model.name.toLowerCase()] || configs[model.type.toLowerCase()]
          }
          
          if (config) {
            model.source = config.source || 'local'
            model.apiUrl = config.api_url || ''
            model.apiKey = config.api_key || ''
            model.modelName = config.model_name || ''
            model.sourceProvider = config.source_provider || 'custom'
          }
        })
      } catch (error) {
        console.error('Failed to load model configurations:', error)
        console.error('Error stack:', error.stack)
        // Continue with existing model configurations - don't block other operations
      }
    }
    
    // Get specific model API configuration
    const getModelApiConfig = async (modelId) => {
      try {
        const response = await api.modelConfigs.getApiConfig(modelId)
        const model = models.value.find(m => m.id === modelId)
        if (model) {
          model.apiConfig = response.data
          notify.success('API configuration loaded successfully')
        }
        return response.data
      } catch (error) {
        console.error('Failed to get model API configuration:', error)
        notify.error('Failed to get API configuration')
        return null
      }
    }
    
    // Get specific model API status
    const getModelApiStatus = async (modelId) => {
      try {
        const model = models.value.find(m => m.id === modelId)
        if (!model) {
          notify.error('Model not found')
          return null
        }
        
        // Add loading state
        const loadingMessageId = notify.info('Checking API status...')
        
        const response = await api.modelConfigs.getApiStatus(modelId)
        
        // Update model API status
        model.apiStatus = response.data
        
        // Show success notification
        notify.remove(loadingMessageId)
        notify.success('API status checked successfully')
        
        return response.data
      } catch (error) {
        console.error('Failed to get model API status:', error)
        notify.error('Failed to get API status')
        return null
      }
    }
    
    // Get global API service status
    const getApiServiceStatus = async () => {
      try {
        checkingGlobalStatus.value = true
        
        const response = await api.externalApi.getServiceStatus()
        
        // Update global API service status
        globalApiStatus.value = {
          ...response.data,
          timestamp: new Date().toISOString()
        }
        
        notify.success('Global API service status checked successfully')
        
        return response.data
      } catch (error) {
        console.error('Failed to get API service status:', error)
        notify.error('Failed to get API service status')
        return null
      } finally {
        checkingGlobalStatus.value = false
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
        }
      }
      
      try {
        // Update model type in backend
        await api.modelConfigs.updateType(modelId, { source: newSource })
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
      
      savingSettings.value.add(modelId)
      try {
        const configData = {
          source: model.source,
          apiUrl: model.apiUrl,
          apiKey: model.apiKey,
          modelName: model.modelName,
          sourceProvider: model.sourceProvider
        }
        
        await api.modelConfigs.updateApiConfig(modelId, configData)
        
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model configuration saved successfully')
      } catch (error) {
        console.error('Failed to save model configuration:', error)
        errorHandler.handleError(error, 'Save Model Configuration')
        notify.error('Failed to save model configuration')
      } finally {
        savingSettings.value.delete(modelId)
      }
    }
    
    // Test API connection
    const testApiConnection = async (modelId) => {
      const model = models.value.find(m => m.id === modelId)
      if (!model || !model.apiUrl || !model.apiKey || !model.modelName) {
        notify.error('Please complete all API configuration fields first')
        return
      }
      
      testingConnections.value.add(modelId)
      try {
        const testData = {
          apiUrl: model.apiUrl,
          apiKey: model.apiKey,
          modelName: model.modelName,
          sourceProvider: model.sourceProvider
        }
        
        // Use the generic test connection endpoint that doesn't require a model ID
        const response = await api.externalApi.testGenericConnection(testData)
        
        testResults.value[modelId] = {
          success: true,
          message: 'Connection successful! API is working properly.'
        }
        notify.success('API connection test passed')
        
        // Auto-activate if not already active
        if (!model.isActive) {
          await toggleActivation(modelId)
        }
      } catch (error) {
        console.error('API connection test failed:', error)
        testResults.value[modelId] = {
          success: false,
          message: error.response?.data?.message || error.message || 'Connection failed. Please check your settings.'
        }
        notify.error('API connection test failed')
      } finally {
        testingConnections.value.delete(modelId)
        
        // Clear test results after 5 seconds
        setTimeout(() => {
          delete testResults.value[modelId]
        }, 5000)
      }
    }
    
    // Load training status for all models
    const loadTrainingStatus = async () => {
      try {
        const response = await api.models.trainingStatus()
        const data = response.data
        // Update models with training status
        models.value.forEach(model => {
          const session = data.data?.find(s => s.model_ids?.includes(model.id))
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
      } catch (error) {
        // Handle any errors gracefully
        // This 404 error doesn't affect serial port functionality
        console.log('Training status endpoint not available - this is expected if training feature is not implemented')
        
        // Default to not training regardless of error type
        models.value.forEach(model => {
          model.trainingStatus = { isTraining: false, progress: 0, status: 'idle' }
        })
      }
    }
    
    // Load available datasets for training
    const loadDatasets = async () => {
      try {
        const response = await api.datasets.get()
        const data = response.data
        availableDatasets.value = data.datasets || []
        if (availableDatasets.value.length > 0) {
          selectedDataset.value = availableDatasets.value[0].id
        }
      } catch (error) {
        console.error('Failed to load datasets:', error)
        availableDatasets.value = [{ id: 'default', name: 'Default Dataset' }]
        selectedDataset.value = 'default'
      }
    }
    
    // Get complete default model configuration - includes all 19 local models (ports 8001-8019)
    const getDefaultModels = () => {
      const defaultModels = [
        // Manager Model
        {
          id: 'manager',
          name: 'Manager Model',
          type: 'Manager Model',
          description: 'System manager model for coordination',
          status: 'running',
          isActive: true,
          isPrimary: true,
          port: 8001,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          metrics: {
            memoryUsage: 128,
            cpuUsage: 5,
            responseTime: 15
          },
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Language Model
        {
          id: 'language',
          name: 'Language Model',
          type: 'Language Model',
          description: 'Natural language processing model',
          status: 'running',
          isActive: true,
          isPrimary: false,
          port: 8002,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          metrics: {
            memoryUsage: 512,
            cpuUsage: 12,
            responseTime: 80
          },
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Knowledge Model
        {
          id: 'knowledge',
          name: 'Knowledge Model',
          type: 'Knowledge Model',
          description: 'Knowledge base and retrieval model',
          status: 'running',
          isActive: true,
          isPrimary: false,
          port: 8003,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          metrics: {
            memoryUsage: 256,
            cpuUsage: 8,
            responseTime: 30
          },
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Vision Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Audio Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Autonomous Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Programming Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Planning Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Emotion Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Spatial Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Computer Vision Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Sensor Model
        {
          id: 'sensor',
          name: 'Sensor Model',
          type: 'Sensor Model',
          description: 'Sensor data processing and analysis',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8012,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Motion Model
        {
          id: 'motion',
          name: 'Motion Model',
          type: 'Motion Model',
          description: 'Motion control and prediction model',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8013,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Prediction Model
        {
          id: 'prediction',
          name: 'Prediction Model',
          type: 'Prediction Model',
          description: 'Forecasting and predictive analytics model',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8014,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Advanced Reasoning Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Data Fusion Model
        {
          id: 'data_fusion',
          name: 'Data Fusion Model',
          type: 'Data Fusion Model',
          description: 'Multi-source data integration and analysis',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8016,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Creative Problem Solving Model
        {
          id: 'creative_solving',
          name: 'Creative Problem Solving Model',
          type: 'Creative Problem Solving Model',
          description: 'Innovative approaches to complex problems',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8017,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Meta Cognition Model
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
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // Value Alignment Model
        {
          id: 'value_alignment',
          name: 'Value Alignment Model',
          type: 'Value Alignment Model',
          description: 'Ethical alignment and value system',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 8019,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          source: 'local',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        // External API Model
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
          apiUrl: 'https://api.openai.com/v1/chat/completions',
          modelName: 'gpt-4',
          sourceProvider: 'openai',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
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
          apiUrl: 'https://api.anthropic.com/v1/messages',
          modelName: 'claude-3-opus-20240229',
          sourceProvider: 'anthropic',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
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
          apiUrl: 'https://generativelanguage.googleapis.com/v1beta/models',
          modelName: 'gemini-pro',
          sourceProvider: 'google',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        {
          id: 'huggingface',
          name: 'Hugging Face API',
          type: 'Hugging Face API',
          description: 'Hugging Face Inference API integration',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 0,
          apiKey: '',
          apiUrl: 'https://api-inference.huggingface.co/models',
          modelName: 'meta-llama/Llama-2-70b-chat-hf',
          sourceProvider: 'huggingface',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        {
          id: 'mistral',
          name: 'Mistral AI API',
          type: 'Mistral AI API',
          description: 'Mistral AI language model integration',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 0,
          apiKey: '',
          apiUrl: 'https://api.mistral.ai/v1/chat/completions',
          modelName: 'mistral-large-latest',
          sourceProvider: 'mistral',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        },
        {
          id: 'custom',
          name: 'Custom API',
          type: 'Custom API',
          description: 'Custom external API model integration',
          status: 'stopped',
          isActive: false,
          isPrimary: false,
          port: 0,
          apiKey: '',
          apiUrl: '',
          modelName: '',
          sourceProvider: 'custom',
          rateLimit: 1000,
          lastUpdated: new Date().toISOString(),
          version: '1.0.0',
          trainingStatus: { isTraining: false, progress: 0, status: 'idle' }
        }
      ]
      
      console.log('Default models count:', defaultModels.length)
      console.log('Default models details:', defaultModels.map(m => ({ id: m.id, name: m.name, port: m.port, source: m.source })))
      return defaultModels
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

        // Use api instance for POST request
        const response = await api.post('/api/models', modelToAdd)

        // Add to local state
        models.value.push(response.data)
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
        notify.error('Failed to add model to backend. Please check backend connectivity.')
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
        // Use api instance for DELETE request
        await api.delete(`/api/models/${modelId}`)

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
        
        // Use api instance for PUT request
        await api.put(`/api/models/${modelId}/activation`, { isActive: newState })

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
        // Use api instance for PUT request
        await api.put(`/api/models/${modelId}/primary`, { isPrimary: true })

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
        
        // Use api instance for POST request
        await api.post(`/api/models/${modelId}/start`)

        // Update local state
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model started successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Start Model')
        // Revert status on error
        model.status = 'stopped'
        notify.error(`Failed to start model: ${error.message}`)
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
        
        // Use api instance for POST request
        await api.post(`/api/models/${modelId}/stop`)

        // Update local state
        model.status = 'stopped'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model stopped successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Stop Model')
        // Revert status on error
        model.status = 'running'
        notify.error(`Failed to stop model: ${error.message}`)
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
        
        // Use api instance for POST request
        await api.post(`/api/models/${modelId}/restart`)

        // Update local state
        model.status = 'running'
        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Model restarted successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Restart Model')
        // Revert status on error
        model.status = 'stopped'
        notify.error(`Failed to restart model: ${error.message}`)
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

        // Use api instance for POST request
        await api.post('/api/models/start-all')

        // Update local state
        modelsToStart.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models started successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Start All Models')
        // Revert status on error
        modelsToStart.forEach(model => {
          model.status = 'stopped'
        })
        notify.error(`Failed to start all models: ${error.message}`)
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

        // Use api instance for POST request
        await api.post('/api/models/stop-all')

        // Update local state
        modelsToStop.forEach(model => {
          model.status = 'stopped'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models stopped successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Stop All Models')
        // Revert status on error
        modelsToStop.forEach(model => {
          model.status = 'running'
        })
        notify.error(`Failed to stop all models: ${error.message}`)
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

        // Use api instance for POST request
        await api.post('/api/models/restart-all')

        // Update local state
        models.value.forEach(model => {
          model.status = 'running'
          model.lastUpdated = new Date().toISOString()
        })
        hasChanges.value = true
        notify.success('All models restarted successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Restart All Models')
        // Revert status on error
        models.value.forEach(model => {
          model.status = 'stopped'
        })
        notify.error(`Failed to restart all models: ${error.message}`)
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
        // Use api instance for POST request
        await api.post('/api/system/restart')

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
      if (!model || !model.apiKey || !model.apiUrl || !model.modelName) {
        notify.error('Model not found or required API configuration missing')
        return
      }

      testingConnections.value.add(modelId)
      try {
        // Update model status to testing
        model.status = 'testing'
        
        // Build complete API configuration parameters for real external API testing
        const connectionData = {
          model_id: modelId,
          api_url: model.apiUrl,
          api_key: model.apiKey,
          model_name: model.modelName,
          api_type: model.sourceProvider || 'custom',
          rate_limit: model.rateLimit || 1000,
          api_headers: model.apiHeaders || {},
          test_prompt: 'Hello, please respond with a short test message to verify API connectivity.'
        }
        
        // Use real API endpoint for external API connection testing
        const response = await api.post('/api/models/test-connection', connectionData)

        if (response.data.success) {
          testResults.value[modelId] = {
            status: 'success',
            message: response.data.message || `Successfully connected to ${model.name}`
          }
          
          // Update model status to connected
          model.status = 'connected'
          // Automatically activate successfully connected external API models
          if (!model.isActive) {
            model.isActive = true
            hasChanges.value = true
          }
          
          // Update API key status to valid
          model.apiKeyStatus = 'valid'
          model.apiKeyStatusText = 'Valid'
          
          notify.success(`Connection to ${model.name} successful`)
        } else {
          throw new Error(response.data.message || 'Connection test failed')
        }
      } catch (error) {
        console.error('External API connection test failed:', error)
        errorHandler.handleError(error, 'Test Connection')
        testResults.value[modelId] = {
          status: 'error',
          message: error.response?.data?.message || error.message || 'Connection to external API failed. Please check your API key, URL, and network connection.'
        }
        // Update model status to failed
        model.status = 'failed'
        // Update API key status to invalid
        model.apiKeyStatus = 'invalid'
        model.apiKeyStatusText = 'Invalid'
        notify.error(`Connection to ${model.name} failed`)
      } finally {
        testingConnections.value.delete(modelId)
        // Use real API mechanism to clear test results after delay
        try {
          // Use API to schedule test result cleanup
          const cleanupResponse = await api.post('/api/external/clear-test-results', {
            model_id: modelId,
            delay_ms: 5000
          })
          
          if (cleanupResponse.data.success) {
            console.log(`Test results cleanup scheduled for model ${modelId}`)
          } else {
            // Fallback to local cleanup if API fails
            setTimeout(() => {
              delete testResults.value[modelId]
              const model = models.value.find(m => m.id === modelId)
              if (model && model.status === 'failed') {
                model.status = 'stopped'
              }
            }, 5000)
          }
        } catch (cleanupError) {
          console.error('Failed to schedule test result cleanup:', cleanupError)
          // Fallback to local cleanup
          setTimeout(() => {
            delete testResults.value[modelId]
            const model = models.value.find(m => m.id === modelId)
            if (model && model.status === 'failed') {
              model.status = 'stopped'
            }
          }, 5000)
        }
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
        // Use api instance for PATCH request
        await api.patch(`/api/models/${modelId}`, model)

        model.lastUpdated = new Date().toISOString()
        hasChanges.value = true
        notify.success('Settings saved successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Save Settings')
        notify.error('Failed to save settings to backend')
      } finally {
        savingSettings.value.delete(modelId)
      }
    }

    const saveAllChanges = async () => {
      isSavingAll.value = true
      try {
        // Use api instance for PUT request
        await api.put('/api/models', models.value)

        hasChanges.value = false
        notify.success('All changes saved successfully')
      } catch (error) {
        errorHandler.handleError(error, 'Save All Changes')
        notify.error('Failed to save all changes to backend')
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
      if (pollInterval) {
        clearInterval(pollInterval)
        pollInterval = null
      }
    }
    
    // Start training a model from scratch
    const startTraining = async () => {
      if (!selectedModelForTraining.value || !selectedDataset.value) return
      
      try {
        trainingStatus.value = 'training'
        trainingMessage.value = 'Starting training process...'
        
        // Use real API to start training
        const response = await api.post(`/api/models/${selectedModelForTraining.value.id}/train-from-scratch`, {
          dataset_id: selectedDataset.value,
          training_params: trainingParams.value,
          model_config: {
            model_id: selectedModelForTraining.value.id,
            model_type: selectedModelForTraining.value.type,
            from_scratch: true
          }
        })
        
        if (response.data.success) {
          notify.success(`Training started for ${selectedModelForTraining.value.name}`)
          
          // Start polling for training status using real API
          pollTrainingStatus()
        } else {
          throw new Error(response.data.message || 'Failed to start training')
        }
      } catch (error) {
        console.error('Failed to start training:', error)
        trainingStatus.value = 'error'
        trainingMessage.value = `Error: ${error.message}`
        notify.error(`Failed to start training for ${selectedModelForTraining.value.name}`)
      }
    }
    
    // Stop training
    const stopTraining = async () => {
      if (!selectedModelForTraining.value) return
      
      try {
        // Use real API to stop training
        const response = await api.post(`/api/models/${selectedModelForTraining.value.id}/train-stop`)
        
        if (response.data.success) {
          trainingStatus.value = 'idle'
          trainingMessage.value = 'Training stopped'
          notify.info(`Training stopped for ${selectedModelForTraining.value.name}`)
        } else {
          throw new Error(response.data.message || 'Failed to stop training')
        }
      } catch (error) {
        console.error('Failed to stop training:', error)
        notify.error(`Failed to stop training for ${selectedModelForTraining.value.name}`)
      }
    }
    
    // Poll training status using real API
    let pollInterval = null
    const pollTrainingStatus = async () => {
      if (pollInterval) clearInterval(pollInterval)
      
      pollInterval = setInterval(async () => {
        if (!selectedModelForTraining.value) {
          clearInterval(pollInterval)
          return
        }
        
        try {
          // Use real API to get training status
          const response = await api.get(`/api/models/${selectedModelForTraining.value.id}/training-status`)
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
              clearInterval(pollInterval)
              pollInterval = null
              notify.success(`Training completed for ${selectedModelForTraining.value.name}`)
            } else if (trainingData.status === 'error' || trainingData.status === 'failed') {
              clearInterval(pollInterval)
              pollInterval = null
              notify.error(`Training failed for ${selectedModelForTraining.value.name}`)
            } else if (trainingData.status === 'stopped') {
              clearInterval(pollInterval)
              pollInterval = null
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
      console.log('onMounted hook started')
      try {
        // 调用loadModels方法加载模型，这个方法已经包含了完整的错误处理和默认模型加载逻辑
        await loadModels()
        
        // Load model configurations with error handling
        try {
          await loadModelConfigs()
        } catch (modelConfigError) {
          console.error('Error loading model configurations:', modelConfigError)
          console.log('Continuing with scanSerialPorts despite model config error')
        }
        
        // Additional logs: Confirm loaded model count
        console.log('Actual model count displayed after onMounted:', models.value.length)
        console.log('Actual count of local models displayed:', models.value.filter(m => m.source === 'local').length)
        console.log('Actual count of external API models displayed:', models.value.filter(m => m.source === 'external').length)
        
      } catch (error) {
        console.error('Error in onMounted:', error)
        console.error('Error stack:', error.stack)
        notify.error('Failed to load models')
      } finally {
        // Always scan for serial ports, even if previous functions failed
        console.log('Attempting to scan serial ports...')
        await scanSerialPorts()
        console.log('onMounted hook completed')
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

    const testHardwareConnections = async () => {
      isTestingHardware.value = true
      try {
        // Use real API to test hardware connections with detailed configuration
        const response = await api.post('/api/system/hardware-test', {
          ...hardwareConfig.value,
          test_type: 'comprehensive',
          include_detailed_results: true
        })
        
        const testResults = response.data
        
        if (testResults.success) {
          const { cameras, sensors, actuators, summary } = testResults.data
          
          // Display detailed test results
          if (cameras && cameras.length > 0) {
            cameras.forEach(camera => {
              if (camera.status === 'connected') {
                notify.success(`Camera ${camera.id} (${camera.name}) connected successfully`)
              } else {
                notify.warning(`Camera ${camera.id} (${camera.name}) connection failed: ${camera.message}`)
              }
            })
          }
          
          if (sensors && sensors.length > 0) {
            sensors.forEach(sensor => {
              if (sensor.status === 'connected') {
                notify.success(`${sensor.type} sensor ${sensor.id} connected successfully`)
              } else {
                notify.warning(`${sensor.type} sensor ${sensor.id} connection failed: ${sensor.message}`)
              }
            })
          }
          
          if (actuators && actuators.length > 0) {
            actuators.forEach(actuator => {
              if (actuator.status === 'connected') {
                notify.success(`${actuator.type} actuator ${actuator.id} connected successfully`)
              } else {
                notify.warning(`${actuator.type} actuator ${actuator.id} connection failed: ${actuator.message}`)
              }
            })
          }
          
          // Display summary statistics
          if (summary) {
            const { total_devices, connected_devices, failed_devices } = summary
            notify.success(`Hardware test completed: ${connected_devices}/${total_devices} devices connected successfully`)
            
            if (failed_devices > 0) {
              notify.warning(`${failed_devices} device(s) failed to connect`)
            }
          } else {
            notify.success('Hardware connections tested successfully')
          }
          
        } else {
          throw new Error(testResults.message || 'Hardware test failed')
        }
        
      } catch (error) {
        console.error('Hardware connection test failed:', error)
        errorHandler.handleError(error, 'Test Hardware Connections')
        notify.error(`Hardware connection test failed: ${error.message}`)
      } finally {
        isTestingHardware.value = false
      }
    }

    // Fallback method for hardware testing - use comprehensive hardware test endpoint
    const testHardwareConnectionsFallback = async () => {
      try {
        // Use the main hardware test endpoint with detailed configuration
        const response = await api.post('/api/system/hardware-test', {
          ...hardwareConfig.value,
          detailed_test: true
        })
        
        const testResults = response.data
        const { cameras, sensors, actuators } = testResults
        
        // Display detailed test results
        if (cameras && cameras.length > 0) {
          cameras.forEach(camera => {
            if (camera.status === 'connected') {
              notify.success(`Camera ${camera.id} (${camera.name}) connected successfully`)
            } else {
              notify.warning(`Camera ${camera.id} (${camera.name}) connection failed: ${camera.message}`)
            }
          })
        }
        
        if (sensors && sensors.length > 0) {
          sensors.forEach(sensor => {
            if (sensor.status === 'connected') {
              notify.success(`${sensor.type} sensor ${sensor.id} connected successfully`)
            } else {
              notify.warning(`${sensor.type} sensor ${sensor.id} connection failed: ${sensor.message}`)
            }
          })
        }
        
        if (actuators && actuators.length > 0) {
          actuators.forEach(actuator => {
            if (actuator.status === 'connected') {
              notify.success(`${actuator.type} actuator ${actuator.id} connected successfully`)
            } else {
              notify.warning(`${actuator.type} actuator ${actuator.id} connection failed: ${actuator.message}`)
            }
          })
        }
        
        notify.success('Hardware connections tested successfully using fallback method')
        
      } catch (fallbackError) {
        console.error('Hardware connection fallback test failed:', fallbackError)
        notify.error('All hardware connection tests failed. Please check backend connectivity.')
      }
    }

    const saveHardwareConfig = async () => {
      isSavingHardware.value = true
      try {
        // Use real API to save hardware configuration to backend
        const response = await api.post('/api/system/hardware-config', {
          ...hardwareConfig.value,
          config_type: 'hardware',
          save_timestamp: new Date().toISOString()
        })
        
        if (response.data.success) {
          notify.success('Hardware configuration saved successfully')
          hasChanges.value = true
          
          // Log the saved configuration
          console.log('Hardware configuration saved to backend:', hardwareConfig.value)
        } else {
          throw new Error(response.data.message || 'Hardware configuration save failed')
        }
        
      } catch (error) {
        console.error('Failed to save hardware configuration:', error)
        errorHandler.handleError(error, 'Save Hardware Configuration')
        notify.error(`Failed to save hardware configuration: ${error.message}`)
      } finally {
        isSavingHardware.value = false
      }
    }

    const resetHardwareConfig = async () => {
      if (!confirm('Are you sure you want to reset hardware configuration to defaults?')) {
        return
      }
      
      try {
        // Use real API to reset hardware configuration
        const response = await api.post('/api/system/hardware-reset', {
          reset_type: 'hardware_config',
          reset_timestamp: new Date().toISOString()
        })
        
        if (response.data.success) {
          // Reset local state to defaults
          hardwareConfig.value = {
            cameraCount: 1,
            defaultResolution: '1280x720',
            defaultInterface: 'usb',
            defaultBaudRate: '9600',
            cameras: [
              {
                id: 1,
                name: 'Main Camera',
                type: 'mono',
                deviceId: '/dev/video0',
                fps: 30,
                baseline: 65,
                focalLength: 3.6
              }
            ],
            sensors: [
              {
                id: 1,
                type: 'temperature',
                port: '/dev/ttyUSB0'
              },
              {
                id: 2,
                type: 'humidity',
                port: '/dev/ttyUSB1'
              }
            ],
            actuators: [
              {
                id: 1,
                type: 'motor',
                port: 'GPIO17'
              },
              {
                id: 2,
                type: 'servo',
                port: 'GPIO18'
              }
            ]
          }
          
          notify.success('Hardware configuration reset to defaults successfully')
          hasChanges.value = true
          
          // Log the reset operation
          console.log('Hardware configuration reset via API:', hardwareConfig.value)
        } else {
          throw new Error(response.data.message || 'Hardware configuration reset failed')
        }
      } catch (error) {
        console.error('Failed to reset hardware configuration:', error)
        errorHandler.handleError(error, 'Reset Hardware Configuration')
        notify.error(`Failed to reset hardware configuration: ${error.message}`)
      }
    }

    // Serial port related methods
    const scanSerialPorts = async () => {
      try {
        // Clear any previous error
        hardwareConfig.value.serialConnectionError = ''
        
        // Call API to get available serial ports
        const response = await api.serial.getPorts()
        
        // Update the serial ports list
        hardwareConfig.value.serialPorts = response.data || []
        
        console.log('Available serial ports:', hardwareConfig.value.serialPorts)
        
        if (hardwareConfig.value.serialPorts.length === 0) {
          notify.info('No serial ports found. Please check your connections.')
        } else {
          notify.success(`Found ${hardwareConfig.value.serialPorts.length} serial port(s)`)          
        }
      } catch (error) {
        console.error('Error scanning serial ports:', error)
        errorHandler.handleError(error, 'Scan Serial Ports')
        hardwareConfig.value.serialConnectionError = `Failed to scan serial ports: ${error.message}`
        notify.error(`Failed to scan serial ports: ${error.message}`)
      }
    }

    const connectSerialPort = async () => {
      if (!hardwareConfig.value.selectedSerialPort || !hardwareConfig.value.serialDeviceId) {
        notify.warning('Please select a serial port and enter a device ID')
        return
      }
      
      try {
        // Update connection status
        hardwareConfig.value.serialConnectionStatus = 'connecting'
        hardwareConfig.value.serialConnectionError = ''
        
        // Call API to connect to serial port
        const response = await api.serial.connect({
          port: hardwareConfig.value.selectedSerialPort,
          baudrate: hardwareConfig.value.serialBaudRate,
          device_id: hardwareConfig.value.serialDeviceId
        })
        
        // Update connection status
        hardwareConfig.value.serialConnectionStatus = 'connected'
        notify.success('Serial port connected successfully')
        
        console.log('Serial port connected:', response.data)
        hasChanges.value = true
      } catch (error) {
        console.error('Error connecting to serial port:', error)
        errorHandler.handleError(error, 'Connect Serial Port')
        hardwareConfig.value.serialConnectionStatus = 'error'
        hardwareConfig.value.serialConnectionError = `Failed to connect to serial port: ${error.message}`
        notify.error(`Failed to connect to serial port: ${error.message}`)
      }
    }

    const disconnectSerialPort = async () => {
      try {
        // Update connection status
        hardwareConfig.value.serialConnectionStatus = 'connecting'
        
        // Call API to disconnect from serial port
        console.log('Disconnecting serial port with params:', {
          device_id: hardwareConfig.value.serialDeviceId,
          port: hardwareConfig.value.selectedSerialPort
        });
        const response = await api.serial.disconnect({
          device_id: hardwareConfig.value.serialDeviceId,
          port: hardwareConfig.value.selectedSerialPort
        })
        
        // Update connection status
        hardwareConfig.value.serialConnectionStatus = 'disconnected'
        hardwareConfig.value.serialConnectionError = ''
        notify.success('Serial port disconnected successfully')
        
        console.log('Serial port disconnected:', response.data)
        hasChanges.value = true
      } catch (error) {
        console.error('Error disconnecting from serial port:', error)
        errorHandler.handleError(error, 'Disconnect Serial Port')
        hardwareConfig.value.serialConnectionStatus = 'error'
        hardwareConfig.value.serialConnectionError = `Failed to disconnect from serial port: ${error.message}`
        notify.error(`Failed to disconnect from serial port: ${error.message}`)
      }
    }

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
      disconnectSerialPort
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
</style>
