// Real backend API connection implementation
import axios from 'axios'
import type { AxiosInstance, AxiosResponse, AxiosRequestConfig } from 'axios'
import configManager from './config/configManager.js'

// Types
export interface ApiError {
  code: string
  message: string
  config: AxiosRequestConfig
  response?: AxiosResponse
}

export interface ApiResponse<T = any> {
  status: 'success' | 'error'
  data: T
  message?: string
  error?: ApiError
}

export interface TrainingConfig {
  modelIds: string[]
  datasetIds: string[]
  epochs: number
  batchSize: number
  learningRate: number
  device: string
  validationSplit: number
  saveFrequency: number
}

export interface ModelData {
  id: string
  name: string
  type: 'local' | 'api'
  endpoint?: string
  modelName?: string
  active: boolean
  status: string
  cpuUsage: number
  memoryUsage: number
  responseTime: number
  port?: string
  [key: string]: any
}

export interface DatasetInfo {
  id: string
  name: string
  type: string
  size: number
  samples: number
  description?: string
}

export interface TrainingJob {
  id: string
  status: string
  progress: number
  modelIds: string[]
  datasetIds: string[]
  startTime: string
  endTime?: string
  device: string
  metrics?: Record<string, any>
}

export interface RobotHardwareComponent {
  id: string
  name: string
  type: string
  connected: boolean
  status: string
  port?: string
  address?: string
  driver?: string
}

export interface SerialPortConfig {
  port: string
  baudRate: number
  dataBits: 5 | 6 | 7 | 8
  stopBits: 1 | 1.5 | 2
  parity: 'none' | 'odd' | 'even' | 'mark' | 'space'
}

export interface ExternalApiConfig {
  id: string
  name: string
  type: string
  endpoint: string
  apiKey?: string
  active: boolean
  lastTested?: string
}

// Backend API base URL - 使用配置管理器
const API_BASE_URL = configManager.getApiBaseUrl()

// 创建axios实例 - 使用配置管理器中的配置
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: configManager.get('api.timeout', 120000),
  headers: configManager.get('api.headers', {
    'Content-Type': 'application/json; charset=utf-8'
  })
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`)
    return config
  },
  (error) => {
    console.error('[API Request Error]', error)
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.method?.toUpperCase()} ${response.config.baseURL}${response.config.url}`)
    return response
  },
  (error) => {
    // Special handling for training status endpoint 404 error (backend has not implemented this feature)
    const isTrainingStatusError = error.config?.url === '/api/models/training/status' && 
                                 error.response?.status === 404
    
    if (!isTrainingStatusError) {
      console.error('[API Response Error]', error)
      console.error('[Error Details]', {
        code: error.code,
        message: error.message,
        config: error.config ? `${error.config.method?.toUpperCase()} ${error.config.baseURL}${error.config.url}` : 'No config',
        response: error.response ? `${error.response.status} ${error.response.statusText}` : 'No response'
      })
    } else {
      // Only record concise logs, do not show complete error details
      console.log('[Expected API Error] Training status endpoint not implemented yet (404)')
    }
    
    // Handle connection errors
    if (error.code === 'ECONNREFUSED') {
      console.error('[Backend Connection] Server is not running or unreachable')
      return Promise.reject({
        data: {
          status: 'error',
          message: 'Backend server is not available. Please start the server.',
          error: 'Connection refused'
        }
      })
    }
    
    // Handle timeout errors
    if (error.code === 'ECONNABORTED') {
      console.error('[Backend Connection] Request timed out')
      return Promise.reject({
        data: {
          status: 'error',
          message: 'Request timed out. Please check your network connection.',
          error: 'Connection timeout'
        }
      })
    }
    
    // Handle other errors
    if (error.response) {
      // Server returns error status code
      return Promise.reject(error.response)
    } else if (error.request) {
      // Request sent but no response received
      return Promise.reject({
        data: {
          status: 'error',
          message: 'No response from server. Please check if the backend is running.',
          error: 'No response'
        }
      })
    } else {
      // Other errors
      return Promise.reject({
        data: {
          status: 'error',
          message: 'Request failed. Please try again.',
          error: error.message
        }
      })
    }
  }
)

// API method definitions
const api = {
  // Add top-level get and post methods for compatibility with direct calls in HomeView.vue
  get: apiClient.get,
  post: apiClient.post,
  put: apiClient.put,
  delete: apiClient.delete,
  
  // Health check
  health: {
    get: () => apiClient.get('/health')
  },

  // System API
  system: {
    stats: () => apiClient.get('/api/system/stats'),
    restart: () => apiClient.post('/api/system/restart'),
    cleanupDisk: () => apiClient.post('/api/system/cleanup-disk')
  },

  // Model API
  models: {
    get: () => apiClient.get<ApiResponse<ModelData[]>>('/api/models'),
    getAll: () => apiClient.get<ApiResponse<ModelData[]>>('/api/models/getAll'),
    available: () => apiClient.get<ApiResponse<ModelData[]>>('/api/models/available'),
    trainingStatus: () => apiClient.get('/api/models/training/status'),
    fromScratchStatus: () => apiClient.get('/api/models/from_scratch/status'),
    start: (modelId: string) => apiClient.post(`/api/models/${modelId}/start`),
    stop: (modelId: string) => apiClient.post(`/api/models/${modelId}/stop`),
    restart: (modelId: string) => apiClient.post(`/api/models/${modelId}/restart`),
    train: (modelId: string, data: any) => apiClient.post(`/api/models/${modelId}/train`, data),
    stopTraining: (modelId: string) => apiClient.post(`/api/models/${modelId}/train/stop`),
    trainingStatusById: (modelId: string) => apiClient.get(`/api/models/${modelId}/train/status`),
    testConnection: () => apiClient.post('/api/models/test-connection'),
    startAll: () => apiClient.post('/api/models/start-all'),
    stopAll: () => apiClient.post('/api/models/stop-all'),
    restartAll: () => apiClient.post('/api/models/restart-all'),
    add: (modelData: ModelData) => apiClient.post('/api/models', modelData),
    update: (modelId: string, modelData: Partial<ModelData>) => apiClient.put(`/api/models/${modelId}`, modelData),
    delete: (modelId: string) => apiClient.delete(`/api/models/${modelId}`),
    setActivation: (modelId: string, isActive: boolean) => apiClient.put(`/api/models/${modelId}/activation`, { isActive }),
    setPrimary: (modelId: string) => apiClient.put(`/api/models/${modelId}/primary`),
    // Get model performance metrics
    metrics: () => apiClient.get('/api/dashboard/model-metrics'),
    // Get model dependencies
    getDependencies: () => apiClient.get('/api/models/dependencies')
  },

  // Dataset API
  datasets: {
    get: () => apiClient.get<ApiResponse<DatasetInfo[]>>('/api/datasets'),
    list: () => apiClient.get<ApiResponse<DatasetInfo[]>>('/api/training/dataset/list'),
    upload: (formData: FormData) => apiClient.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // Training API
  training: {
    start: (trainingConfig: TrainingConfig) => apiClient.post<ApiResponse<TrainingJob>>('/api/training/start', trainingConfig),
    externalAssistance: (externalTrainingData: any) => apiClient.post('/api/training/external-assistance', externalTrainingData),
    stop: (trainingJobId: string) => apiClient.post(`/api/training/${trainingJobId}/stop`),
    status: (jobId: string) => apiClient.get<ApiResponse<TrainingJob>>(`/api/training/status/${jobId}`),
    history: () => apiClient.get<ApiResponse<TrainingJob[]>>('/api/training/history'),
    getActiveJobs: () => apiClient.get<ApiResponse<TrainingJob[]>>('/api/training/active-jobs'),
    switchDevice: (jobId: string, newDevice: string) => apiClient.post('/api/training/switch-device', { job_id: jobId, new_device: newDevice }),
    availableDevices: () => apiClient.get('/api/training/available-devices'),
    startKnowledgeLearning: (config: any) => apiClient.post('/api/training/knowledge-learning/start', config),
    stopKnowledgeLearning: () => apiClient.post('/api/training/knowledge-learning/stop'),
    externalModelStats: () => apiClient.get('/api/training/external-model-stats'),
    getJointTrainingRecommendations: () => apiClient.get('/api/joint-training/recommendations')
  },

  // Robot Control API (motion, spatial, sensors, etc.)
  robot: {
    // Robot status and hardware
    status: () => apiClient.get<ApiResponse>('/api/robot/status'),
    hardware: {
      detect: () => apiClient.get<ApiResponse<RobotHardwareComponent[]>>('/api/robot/hardware/detect'),
      initialize: () => apiClient.post('/api/robot/hardware/initialize'),
      disconnect: () => apiClient.post('/api/robot/hardware/disconnect'),
      testConnection: () => apiClient.post('/api/robot/hardware/test_connection'),
      scan: () => apiClient.post('/api/robot/hardware/scan'),
      diagnose: () => apiClient.get('/api/robot/hardware/diagnose')
    },
    
    // Joint control
    joints: {
      list: () => apiClient.get('/api/robot/joints'),
      control: (jointCommand: any) => apiClient.post('/api/robot/joint', jointCommand)
    },
    
    // Sensor control
    sensors: {
      list: () => apiClient.get('/api/robot/sensors'),
      toggle: (sensorToggle: any) => apiClient.post('/api/robot/sensor/toggle', sensorToggle),
      calibrate: (sensorId: string) => apiClient.post('/api/robot/sensor/calibrate', { sensor_id: sensorId }),
      fusion: (fusionRequest: any) => apiClient.post('/api/robot/sensor/fusion', fusionRequest),
      analyze: (analysisRequest: any) => apiClient.post('/api/robot/sensor/analyze', analysisRequest)
    },
    
    // Camera control
    cameras: {
      toggle: (cameraCommand: any) => apiClient.post('/api/robot/camera/toggle', cameraCommand),
      calibrate: (cameraId: string) => apiClient.post('/api/robot/camera/calibrate', { camera_id: cameraId })
    },
    
    // Vision and spatial processing
    vision: {
      detect: (detectionParams: any) => apiClient.post('/api/robot/vision/detect', detectionParams),
      segment: (segmentationParams: any) => apiClient.post('/api/robot/vision/segment', segmentationParams)
    },
    
    // Spatial processing
    spatial: {
      depth: (depthRequest: any) => apiClient.post('/api/robot/spatial/depth', depthRequest),
      map: (mappingRequest: any) => apiClient.post('/api/robot/spatial/map', mappingRequest)
    },
    
    // Stereo vision
    stereo: {
      enable: (enableParams: any) => apiClient.post('/api/robot/stereo/enable', enableParams),
      depth: (depthParams: any) => apiClient.post('/api/robot/stereo/depth', depthParams)
    },
    
    // Motion control
    motion: {
      execute: (motionCommand: any) => apiClient.post('/api/robot/motion/execute', motionCommand)
    },
    
    // Task planning and execution
    task: {
      plan: (taskPlan: any) => apiClient.post('/api/robot/task/plan', taskPlan),
      execute: (taskExecute: any) => apiClient.post('/api/robot/task/execute', taskExecute),
      stop: (taskId: string) => apiClient.post('/api/robot/task/stop', { task_id: taskId })
    },
    
    // Emergency and system control
    emergency: {
      stop: () => apiClient.post('/api/robot/emergency/stop')
    },
    
    system: {
      reboot: () => apiClient.post('/api/robot/system/reboot'),
      calibrate: () => apiClient.post('/api/robot/system/calibrate')
    },
    
    // Collaboration patterns
    collaboration: {
      patterns: () => apiClient.get('/api/robot/collaboration/patterns')
    }
  },

  // Knowledge API
  knowledge: {
    files: () => apiClient.get('/api/knowledge/files'),
    filePreview: (fileId: string) => apiClient.get(`/api/knowledge/files/${fileId}/preview`),
    search: (query: string, domain?: string) => apiClient.get('/api/knowledge/search', {
      params: { query, domain }
    }),
    stats: () => apiClient.get('/api/knowledge/stats'),
    upload: (formData: FormData) => apiClient.post('/api/knowledge/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }),
    deleteFile: (fileId: string) => apiClient.delete(`/api/knowledge/files/${fileId}`),
    autoLearning: {
      start: (params = {}) => apiClient.post('/api/knowledge/auto-learning/start', params),
      stop: () => apiClient.post('/api/knowledge/auto-learning/stop'),
      progress: () => apiClient.get('/api/knowledge/auto-learning/progress')
    },
    graph: () => apiClient.get('/api/knowledge/graph')
  },

  // Processing API
  process: {
    image: (imageData: any) => apiClient.post('/api/process/image', imageData),
    video: (videoData: any) => apiClient.post('/api/process/video', videoData),
    audio: (audioData: any) => apiClient.post('/api/process/audio', audioData),
    speech: (speechData: any) => apiClient.post('/api/synthesize/speech', speechData)
  },

  // Chat API
  chat: (messageData: any) => apiClient.post('/api/chat', messageData),
  
  // Manager Model Chat API
  managementChat: (messageData: any) => apiClient.post('/api/models/8001/chat', messageData),

  // External Device API
  devices: {
    getCameras: () => apiClient.get('/api/devices/cameras'),
    connectCamera: (cameraId: string) => apiClient.post(`/api/devices/cameras/${cameraId}/connect`),
    disconnectCamera: (cameraId: string) => apiClient.post(`/api/devices/cameras/${cameraId}/disconnect`),
    getSensors: () => apiClient.get('/api/devices/sensors'),
    connectSensor: (sensorId: string) => apiClient.post(`/api/devices/sensors/${sensorId}/connect`),
    disconnectSensor: (sensorId: string) => apiClient.post(`/api/devices/sensors/${sensorId}/disconnect`),
    getExternalDevices: () => apiClient.get('/api/devices/external')
  },

  // Camera API (supports multi-camera and stream control)
  cameras: {
    // Get camera list
    getList: () => apiClient.get('/api/devices/cameras'),
    
    // Connect camera
    connect: (cameraId: string, params = {}) => apiClient.post(`/api/devices/cameras/${cameraId}/connect`, params),
    
    // Disconnect camera
    disconnect: (cameraId: string) => apiClient.post(`/api/devices/cameras/${cameraId}/disconnect`),
    
    // Start camera stream
    startStream: (cameraId: string) => apiClient.post(`/api/cameras/${cameraId}/stream/start`),
    
    // Stop camera stream
    stopStream: (cameraId: string) => apiClient.post(`/api/cameras/${cameraId}/stream/stop`),
    
    // Get camera stream status
    getStreamStatus: (cameraId: string) => apiClient.get(`/api/cameras/${cameraId}/stream/status`),
    
    // Get hardware configuration
    getHardwareConfig: () => apiClient.get('/api/hardware/config'),
    
    // Update hardware configuration
    updateHardwareConfig: (configData: any) => apiClient.post('/api/hardware/config', configData),
    
    // Test hardware connection
    testConnections: () => apiClient.post('/api/hardware/test-connections'),
    
    // Stereo Vision API
    getStereoPairs: () => apiClient.get('/api/cameras/stereo-pairs'),
    processStereoPair: (pairId: string, params = {}) => apiClient.post(`/api/cameras/stereo-pairs/${pairId}/process`, params),
    calibrateStereoPair: (pairId: string, params = {}) => apiClient.post(`/api/cameras/stereo-pairs/${pairId}/calibrate`, params),
    getStereoCalibration: () => apiClient.get('/api/cameras/stereo-calibration')
  },

  // External API Model Configuration
  externalApi: {
    getConfigs: () => apiClient.get<ApiResponse<ExternalApiConfig[]>>('/api/external-api/configs'),
    addConfig: (configData: Partial<ExternalApiConfig>) => apiClient.post('/api/external-api/configs', configData),
    updateConfig: (configId: string, configData: Partial<ExternalApiConfig>) => apiClient.put(`/api/external-api/configs/${configId}`, configData),
    deleteConfig: (configId: string) => apiClient.delete(`/api/external-api/configs/${configId}`),
    testConnection: (configId: string) => apiClient.post(`/api/external-api/configs/${configId}/test`),
    setActive: (configId: string) => apiClient.post(`/api/external-api/configs/${configId}/activate`),
    
    // Test generic API connection (does not depend on configuration ID)
    testGenericConnection: (configData: Partial<ExternalApiConfig>) => apiClient.post('/api/external-api/test-connection', configData),
    
    // Get API service status
    getServiceStatus: () => apiClient.get('/api/external-api/service-status')
  },

  // Model Configuration API (supports each model independently choosing local or external API)
  modelConfigs: {
    // Get all models' configurations
    getAll: () => apiClient.get('/api/models/config'),
    
    // Get specific model configuration
    getById: (modelId: string) => apiClient.get(`/api/models/${modelId}/config`),
    
    // Update model type (local or api)
    updateType: (modelId: string, modelType: string) => apiClient.post(`/api/models/${modelId}/type`, { type: modelType }),
    
    // Update model API configuration
    updateApiConfig: (modelId: string, configData: any) => apiClient.post(`/api/models/${modelId}/api-config`, configData),
    
    // Test API connection
    testConnection: (modelId: string) => apiClient.post(`/api/models/${modelId}/test-connection`),
    
    // Set model API configuration (supports detailed configuration)
    setApiConfig: (modelId: string, configData: any) => apiClient.post(`/api/models/${modelId}/api/config`, configData),
    
    // Get model API configuration
    getApiConfig: (modelId: string) => apiClient.get(`/api/models/${modelId}/api/config`),
    
    // Get model API connection status
    getApiStatus: (modelId: string) => apiClient.get(`/api/models/${modelId}/api/status`)
  },

  // Autonomous Evolution API
  evolution: {
    // Status and configuration
    getStatus: () => apiClient.get('/api/evolution/status'),
    getConfig: () => apiClient.get('/api/evolution/config'),
    updateConfig: (configData: any) => apiClient.put('/api/evolution/config', configData),
    setMode: (modeData: any) => apiClient.post('/api/evolution/mode', modeData),
    
    // Evolution control
    start: (startData: any) => apiClient.post('/api/evolution/start', startData),
    stop: () => apiClient.post('/api/evolution/stop'),
    reset: () => apiClient.post('/api/evolution/reset'),
    
    // History and data
    getHistory: (params: any) => apiClient.get('/api/evolution/history', { params }),
    exportData: (params: any) => apiClient.get('/api/evolution/export', { params }),
    
    // NAS (Neural Architecture Search)
    nas: {
      getStatus: () => apiClient.get('/api/evolution/nas/status'),
      start: () => apiClient.post('/api/evolution/nas/start'),
      stop: () => apiClient.post('/api/evolution/nas/stop')
    },
    
    // Reinforcement Learning optimization
    rl: {
      getStatus: () => apiClient.get('/api/evolution/rl/status'),
      start: () => apiClient.post('/api/evolution/rl/start'),
      stop: () => apiClient.post('/api/evolution/rl/stop')
    },
    
    // Federated evolution
    federated: {
      getStatus: () => apiClient.get('/api/evolution/federated/status'),
      start: () => apiClient.post('/api/evolution/federated/start')
    },
    
    // Online evolution
    online: {
      getStatus: (modelId: string) => apiClient.get(`/api/evolution/online/${modelId}/status`),
      start: (modelId: string) => apiClient.post(`/api/evolution/online/${modelId}/start`),
      stop: (modelId: string) => apiClient.post(`/api/evolution/online/${modelId}/stop`)
    },
    
    // Health check
    health: () => apiClient.get('/api/evolution/health')
  },

  // Serial Communication API
  serial: {
    // Get available serial port list
    getPorts: () => apiClient.get<ApiResponse<SerialPortConfig[]>>('/api/serial/ports'),
    // Connect to serial port
    connect: (params: any) => apiClient.post('/api/serial/connect', params),
    // Disconnect from serial port
    disconnect: (params: any) => apiClient.post('/api/serial/disconnect', params),
    // Send data
    send: (data: any) => apiClient.post('/api/serial/send', { data }),
    // Get serial port status
    status: () => apiClient.get('/api/serial/status'),
    // Set serial port parameters
    configure: (params: any) => apiClient.post('/api/serial/configure', params),
    // Read serial port data (one-time)
    read: () => apiClient.get('/api/serial/read')
  }
}

// Export API object
export { apiClient, api }
export default api