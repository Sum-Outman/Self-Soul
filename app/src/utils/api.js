// 真实后端API连接实现
import axios from 'axios';

// 后端API基础URL
const API_BASE_URL = 'http://localhost:8000';

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making API request to: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API request error:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API response error:', error);
    
    // 处理连接错误
    if (error.code === 'ECONNREFUSED') {
      console.error('Backend server is not running. Please start the backend server.');
      return Promise.reject({
        data: {
          status: 'error',
          message: 'Backend server is not available. Please start the server.',
          error: 'Connection refused'
        }
      });
    }
    
    // 处理其他错误
    if (error.response) {
      // 服务器返回错误状态码
      return Promise.reject(error.response);
    } else if (error.request) {
      // 请求发送但没有收到响应
      return Promise.reject({
        data: {
          status: 'error',
          message: 'No response from server. Please check if the backend is running.',
          error: 'No response'
        }
      });
    } else {
      // 其他错误
      return Promise.reject({
        data: {
          status: 'error',
          message: 'Request failed. Please try again.',
          error: error.message
        }
      });
    }
  }
);

// API方法定义
const api = {
  // 健康检查
  health: {
    get: () => apiClient.get('/health')
  },

  // 系统API
  system: {
    stats: () => apiClient.get('/api/system/stats'),
    restart: () => apiClient.post('/api/system/restart')
  },

  // 模型API
  models: {
    get: () => apiClient.get('/api/models'),
    trainingStatus: () => apiClient.get('/api/models/training/status'),
    fromScratchStatus: () => apiClient.get('/api/models/from_scratch/status'),
    start: (modelId) => apiClient.post(`/api/models/${modelId}/start`),
    stop: (modelId) => apiClient.post(`/api/models/${modelId}/stop`),
    restart: (modelId) => apiClient.post(`/api/models/${modelId}/restart`),
    train: (modelId, data) => apiClient.post(`/api/models/${modelId}/train`, data),
    stopTraining: (modelId) => apiClient.post(`/api/models/${modelId}/train/stop`),
    trainingStatusById: (modelId) => apiClient.get(`/api/models/${modelId}/train/status`),
    testConnection: () => apiClient.post('/api/models/test-connection'),
    startAll: () => apiClient.post('/api/models/start-all'),
    stopAll: () => apiClient.post('/api/models/stop-all'),
    restartAll: () => apiClient.post('/api/models/restart-all'),
    add: (modelData) => apiClient.post('/api/models', modelData),
    update: (modelId, modelData) => apiClient.put(`/api/models/${modelId}`, modelData),
    delete: (modelId) => apiClient.delete(`/api/models/${modelId}`),
    setActivation: (modelId, isActive) => apiClient.put(`/api/models/${modelId}/activation`, { isActive }),
    setPrimary: (modelId) => apiClient.put(`/api/models/${modelId}/primary`)
  },

  // 数据集API
  datasets: {
    get: () => apiClient.get('/api/datasets'),
    upload: (formData) => apiClient.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // 训练API
  training: {
    start: (trainingConfig) => apiClient.post('/api/training/start', trainingConfig),
    stop: () => apiClient.post('/api/training/stop'),
    status: (jobId) => apiClient.get(`/api/training/status/${jobId}`),
    history: () => apiClient.get('/api/training/history')
  },

  // 知识API
  knowledge: {
    files: () => apiClient.get('/api/knowledge/files'),
    filePreview: (fileId) => apiClient.get(`/api/knowledge/files/${fileId}/preview`),
    search: (query, domain) => apiClient.get('/api/knowledge/search', {
      params: { query, domain }
    }),
    stats: () => apiClient.get('/api/knowledge/stats'),
    upload: (formData) => apiClient.post('/api/knowledge/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }),
    deleteFile: (fileId) => apiClient.delete(`/api/knowledge/files/${fileId}`),
    autoLearning: {
      start: () => apiClient.post('/api/knowledge/auto-learning/start'),
      stop: () => apiClient.post('/api/knowledge/auto-learning/stop'),
      progress: () => apiClient.get('/api/knowledge/auto-learning/progress')
    }
  },

  // 处理API
  process: {
    image: (imageData) => apiClient.post('/api/process/image', imageData),
    video: (videoData) => apiClient.post('/api/process/video', videoData),
    audio: (audioData) => apiClient.post('/api/process/audio', audioData)
  },

  // 聊天API
  chat: (messageData) => apiClient.post('/api/chat', messageData),
  
  // 管理模型聊天API
  managementChat: (messageData) => apiClient.post('/api/models/8001/chat', messageData),

  // 外接设备API
  devices: {
    getCameras: () => apiClient.get('/api/devices/cameras'),
    connectCamera: (cameraId) => apiClient.post(`/api/devices/cameras/${cameraId}/connect`),
    disconnectCamera: (cameraId) => apiClient.post(`/api/devices/cameras/${cameraId}/disconnect`),
    getSensors: () => apiClient.get('/api/devices/sensors'),
    connectSensor: (sensorId) => apiClient.post(`/api/devices/sensors/${sensorId}/connect`),
    disconnectSensor: (sensorId) => apiClient.post(`/api/devices/sensors/${sensorId}/disconnect`)
  },

  // 外接API模型配置
  externalApi: {
    getConfigs: () => apiClient.get('/api/external-api/configs'),
    addConfig: (configData) => apiClient.post('/api/external-api/configs', configData),
    updateConfig: (configId, configData) => apiClient.put(`/api/external-api/configs/${configId}`, configData),
    deleteConfig: (configId) => apiClient.delete(`/api/external-api/configs/${configId}`),
    testConnection: (configId) => apiClient.post(`/api/external-api/configs/${configId}/test`),
    setActive: (configId) => apiClient.post(`/api/external-api/configs/${configId}/activate`)
  }
};

// 导出API对象
export default api;
