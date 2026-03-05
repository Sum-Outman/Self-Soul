<template>
  <div class="user-guide">
    <h3>📖 User Guide</h3>
    
    <div class="guide-sections">
      <!-- Getting Started -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('gettingStarted')">
          <h4>🚀 Getting Started</h4>
          <span class="toggle-icon">{{ expandedSections.gettingStarted ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.gettingStarted" class="section-content">
          <ol>
            <li><strong>Start the Backend Server:</strong> Run `python core/main.py` from the project root</li>
            <li><strong>Start the Frontend:</strong> Run `npm run dev` from the `app` directory</li>
            <li><strong>Access the Interface:</strong> Open {{ frontendUrl }} in your browser</li>
            <li><strong>Explore the Dashboard:</strong> Check system status and model information</li>
          </ol>
        </div>
      </div>

      <!-- Model Management -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('modelManagement')">
          <h4>🤖 Model Management</h4>
          <span class="toggle-icon">{{ expandedSections.modelManagement ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.modelManagement" class="section-content">
          <h5>Available Models:</h5>
          <ul>
            <li><strong>Manager Model (8001):</strong> System coordination and task management</li>
            <li><strong>Language Model (8002):</strong> Natural language processing</li>
            <li><strong>Knowledge Model (8003):</strong> Knowledge storage and retrieval</li>
            <li><strong>Vision Model (8004):</strong> Image processing</li>
            <li><strong>Audio Model (8005):</strong> Audio processing and speech recognition</li>
            <li><strong>Autonomous Model (8006):</strong> Decision making and planning</li>
            <li><strong>And 20 more specialized models...</strong></li>
          </ul>
          <h5>Model Control:</h5>
          <ul>
            <li>Start/Stop individual models</li>
            <li>Monitor model status and performance</li>
            <li>Configure model parameters</li>
          </ul>
        </div>
      </div>

      <!-- Training Guide -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('training')">
          <h4>🎯 Training Guide</h4>
          <span class="toggle-icon">{{ expandedSections.training ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.training" class="section-content">
          <h5>Training Modes:</h5>
          <ul>
            <li><strong>Basic Motion Training:</strong> Fundamental robot motion control</li>
            <li><strong>Perception Training:</strong> Visual and sensor perception</li>
            <li><strong>Collaboration Training:</strong> Multi-model collaboration</li>
            <li><strong>AGI Fusion Training:</strong> Advanced cognitive integration</li>
          </ul>
          <h5>Training Process:</h5>
          <ol>
            <li>Select training mode and models</li>
            <li>Upload or select training dataset</li>
            <li>Configure training parameters</li>
            <li>Start training and monitor progress</li>
            <li>Review training results and logs</li>
          </ol>
        </div>
      </div>

      <!-- Hardware Integration -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('hardware')">
          <h4>🔧 Hardware Integration</h4>
          <span class="toggle-icon">{{ expandedSections.hardware ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.hardware" class="section-content">
          <h5>Supported Hardware:</h5>
          <ul>
            <li><strong>Sensors:</strong> Accelerometer, gyroscope, temperature, etc.</li>
            <li><strong>Cameras:</strong> Mono, stereo, depth cameras</li>
            <li><strong>Actuators:</strong> Servos, motors</li>
            <li><strong>Communication:</strong> USB, serial, Ethernet, Bluetooth</li>
          </ul>
          <h5>Hardware Setup:</h5>
          <ol>
            <li>Connect hardware devices</li>
            <li>Initialize hardware from the settings page</li>
            <li>Test connections and sensor data</li>
            <li>Configure safety limits</li>
          </ol>
        </div>
      </div>

      <!-- Real-time Input -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('realTimeInput')">
          <h4>⚡ Real-time Input</h4>
          <span class="toggle-icon">{{ expandedSections.realTimeInput ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.realTimeInput" class="section-content">
          <h5>Input Types:</h5>
          <ul>
            <li><strong>Text:</strong> Direct text input for language models</li>
            <li><strong>Image:</strong> Upload images for vision processing</li>
            <li><strong>Audio:</strong> Record and process audio input</li>
            <li><strong>Video:</strong> Record and analyze video streams</li>
            <li><strong>Sensor Data:</strong> Real-time sensor readings</li>
          </ul>
          <h5>Usage Tips:</h5>
          <ul>
            <li>Grant necessary permissions for camera and microphone</li>
            <li>Use appropriate file formats for uploads</li>
            <li>Monitor input processing status</li>
          </ul>
        </div>
      </div>

      <!-- API Configuration -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('apiConfig')">
          <h4>🌐 API Configuration</h4>
          <span class="toggle-icon">{{ expandedSections.apiConfig ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.apiConfig" class="section-content">
          <h5>External API Integration:</h5>
          <ul>
            <li>Configure external API endpoints</li>
            <li>Set API keys and authentication</li>
            <li>Test API connections</li>
            <li>Switch between local and API models</li>
          </ul>
          <h5>API Endpoints:</h5>
          <ul>
            <li><strong>Backend API:</strong> {{ backendApiUrl }}</li>
            <li><strong>API Documentation:</strong> {{ backendDocsUrl }}</li>
            <li><strong>Model Services:</strong> {{ modelServicesUrl }}</li>
          </ul>
        </div>
      </div>

      <!-- Troubleshooting -->
      <div class="guide-section">
        <div class="section-header" @click="toggleSection('troubleshooting')">
          <h4>🔍 Troubleshooting</h4>
          <span class="toggle-icon">{{ expandedSections.troubleshooting ? '▼' : '▶' }}</span>
        </div>
        <div v-if="expandedSections.troubleshooting" class="section-content">
          <h5>Common Issues:</h5>
          <ul>
            <li><strong>Backend not running:</strong> Start the server with `python core/main.py`</li>
            <li><strong>Model services not starting:</strong> Check port availability and dependencies</li>
            <li><strong>Hardware not detected:</strong> Ensure devices are connected and drivers are installed</li>
            <li><strong>API errors:</strong> Check network connection and server logs</li>
            <li><strong>Training failures:</strong> Verify dataset format and available resources</li>
          </ul>
          <h5>Debugging Tips:</h5>
          <ul>
            <li>Check browser console for frontend errors</li>
            <li>Review server logs for backend issues</li>
            <li>Test API endpoints directly with tools like Postman</li>
            <li>Verify hardware connections and permissions</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, getCurrentInstance } from 'vue'

export default {
  name: 'UserGuide',
  
  setup() {
    const instance = getCurrentInstance()
    
    // Expanded sections
    const expandedSections = ref({
      gettingStarted: true,
      modelManagement: false,
      training: false,
      hardware: false,
      realTimeInput: false,
      apiConfig: false,
      troubleshooting: false
    })
    
    // Toggle section expansion
    const toggleSection = (section) => {
      expandedSections.value[section] = !expandedSections.value[section]
    }
    
    // Computed URLs
    const frontendUrl = instance?.proxy?.$getFrontendUrl ? instance.proxy.$getFrontendUrl() : 'http://localhost:5175'
    const backendApiUrl = instance?.proxy?.$getBackendUrl ? instance.proxy.$getBackendUrl('/api') : 'http://localhost:8000/api'
    const backendDocsUrl = instance?.proxy?.$getBackendDocsUrl ? instance.proxy.$getBackendDocsUrl() : 'http://localhost:8000/docs'
    const modelServicesRange = '8001-8027' // 模型服务端口范围，可以配置化
    
    // 生成模型服务URL
    const modelServicesUrl = (() => {
      try {
        const url = new URL(frontendUrl)
        return `${url.protocol}//${url.hostname}:${modelServicesRange}`
      } catch (e) {
        return `http://localhost:${modelServicesRange}`
      }
    })()
    
    return {
      expandedSections,
      toggleSection,
      frontendUrl,
      backendApiUrl,
      backendDocsUrl,
      modelServicesRange,
      modelServicesUrl
    }
  }
}
</script>

<style scoped>
.user-guide {
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f8f9fa;
}

.guide-sections {
  margin-top: 20px;
}

.guide-section {
  margin-bottom: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: #fff;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 15px;
  cursor: pointer;
  background-color: #f8f9fa;
  border-bottom: 1px solid #ddd;
}

.section-header:hover {
  background-color: #e9ecef;
}

.section-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.toggle-icon {
  font-size: 12px;
  color: #666;
}

.section-content {
  padding: 15px;
  line-height: 1.6;
}

.section-content h5 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 600;
  color: #495057;
}

.section-content ol,
.section-content ul {
  margin-bottom: 15px;
  padding-left: 20px;
}

.section-content li {
  margin-bottom: 5px;
}

.section-content strong {
  color: #007bff;
}
</style>
